"""
Building Facade Material Prediction using Qwen2.5-VL
=====================================================
Merges visibility and obstruction CSVs, filters images where a building is
present and its centerline is unobstructed, then runs VLM inference to
classify the facade material. Supports checkpoint resumption.

Expected directory layout:
    data/csv/{country}/{city}/
        {city}_building_visible.csv     # output of visible_analysis.py
        {city}_building_obstruct.csv    # output of obstruct_analysis.py
        {city}_label.csv                # output of this script (auto-created)

Usage:
    python vlm_predict.py --country China --city "Hong Kong" --model_path /home/nas/yhy/code/building_material/model/material_finetune_v8 --gpu 0 --save_every 500
"""

import os
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor

VALID_LABELS = {"tile", "brick", "concrete", "glass", "metal", "other", "stone", "stucco", "wood"}

PROMPT = (
    "What is the facade material of the central building in this image? "
    "Please choose one of the following: "
    "[tile, brick, concrete, glass, metal, other, stone, stucco, wood] "
    "Only answer with the exact word from the list."
)


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

class Qwen25Adapter:
    """Thin wrapper around Qwen2.5-VL for single-image facade material inference."""

    def __init__(self, model_path: str, device: torch.device, cache_dir: str = None):
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device.index if device.type == "cuda" else "cpu"},
            cache_dir=cache_dir,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        # Left-padding ensures generated tokens appear after the prompt, not before
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    def _parse_label(self, raw: str) -> str:
        """Return the first word of model output if valid, else 'other'."""
        word = raw.strip().lower().split()[0] if raw.strip() else "other"
        return word if word in VALID_LABELS else "other"

    def infer(self, image_path: str) -> str:
        """Run inference on a single image and return a label from VALID_LABELS."""
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": PROMPT},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,  # avoid passing empty video list
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=16)
            # Decode only newly generated tokens (exclude the prompt)
            new_tokens = output_ids[:, inputs.input_ids.shape[-1]:]
            raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return self._parse_label(raw)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_merge(visible_csv: str, obstruct_csv: str) -> pd.DataFrame:
    """Inner-join visibility and obstruction CSVs on (sample_id, img_path)."""
    df_v = pd.read_csv(visible_csv)
    df_o = pd.read_csv(obstruct_csv)
    for col in ("sample_id", "img_path"):
        assert col in df_v.columns and col in df_o.columns, \
            f"Required column '{col}' missing from input CSVs."
    df = pd.merge(df_v, df_o, on=["sample_id", "img_path"],
                  suffixes=("_visible", "_obstruct"), how="inner")
    print(f"[INFO] Merged dataset: {len(df)} rows")
    return df


def load_or_init_result(out_csv: str, df_complete: pd.DataFrame) -> pd.DataFrame:
    """
    Load an existing result file and realign it with the current merged dataset,
    preserving predictions already computed. Initialises pred_label = None for
    new rows. Rows no longer in df_complete are dropped.
    """
    if not os.path.exists(out_csv):
        df = df_complete.copy()
        df["pred_label"] = None
        return df

    df_old = pd.read_csv(out_csv)

    if "pred_label" not in df_old.columns:
        df = df_complete.copy()
        df["pred_label"] = None
        return df

    # Keep only img_path → pred_label; rebuild all other columns from source data
    saved = (
        df_old[["img_path", "pred_label"]]
        .dropna(subset=["img_path"])
        .drop_duplicates("img_path", keep="first")
    )
    df = df_complete.merge(saved, on="img_path", how="left")
    print(f"[INFO] Restored {df['pred_label'].notna().sum()} existing predictions.")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(country: str, city: str, model_path: str,
        gpu_id: int = 0, save_every: int = 500, cache_dir: str = None):
    """
    Full prediction pipeline:
      1. Merge visibility + obstruction CSVs.
      2. Restore previously computed predictions (checkpoint resumption).
      3. Filter: building == 'yes' AND centerline_visible == True AND pred_label is empty.
      4. Run single-image inference, saving checkpoints periodically.
    """
    base         = f"../data/csv/{country}/{city}/{city}"
    visible_csv  = f"{base}_building_visible.csv"
    obstruct_csv = f"{base}_building_obstruct.csv"
    out_csv      = f"{base}_label.csv"

    for path in (visible_csv, obstruct_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file not found: {path}")

    df_complete = load_and_merge(visible_csv, obstruct_csv)
    df_result   = load_or_init_result(out_csv, df_complete)

    # Select rows eligible for inference
    to_predict = df_result[
        (df_result["building"].astype(str).str.lower() == "yes") &
        (df_result["centerline_visible"] == True) &
        (df_result["pred_label"].isna())
    ]
    print(f"[INFO] Rows to predict: {len(to_predict)}")

    if to_predict.empty:
        print("[INFO] Nothing to predict.")
        df_result.to_csv(out_csv, index=False)
        return

    device  = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    adapter = Qwen25Adapter(model_path, device, cache_dir=cache_dir)

    # Pre-build img_path → integer row index map for O(1) in-place writes
    path_to_row = {p: i for i, p in enumerate(df_result["img_path"])}
    label_col   = df_result.columns.get_loc("pred_label")
    count       = 0

    for _, row in tqdm(to_predict.iterrows(), total=len(to_predict), desc="Predicting"):
        img_path = row["img_path"]
        try:
            label = adapter.infer(img_path)
        except Exception as e:
            print(f"[WARNING] Inference failed for {img_path}: {e}")
            label = "other"

        df_result.iloc[path_to_row[img_path], label_col] = label
        count += 1

        # Periodic checkpoint to preserve progress if the run is interrupted
        if count % save_every == 0:
            df_result.to_csv(out_csv, index=False)
            print(f"[INFO] Checkpoint saved ({count} samples processed)")

    df_result.to_csv(out_csv, index=False)
    print(f"[DONE] Saved to {out_csv} | labeled: {df_result['pred_label'].notna().sum()} / {len(df_result)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Building facade material prediction using Qwen2.5-VL.")
    parser.add_argument("--country",     type=str, required=True,
                        help="country name, e.g. 'China'")
    parser.add_argument("--city",       type=str, required=True,
                        help="City name, e.g. 'Hong Kong'")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned Qwen2.5-VL weights (local dir or HuggingFace hub ID)")
    parser.add_argument("--gpu",        type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save a checkpoint every N samples (default: 500)")
    parser.add_argument("--cache_dir",  type=str, default=None,
                        help="Optional cache directory for model weights")
    args = parser.parse_args()

    run(
        country=args.country,
        city=args.city,
        model_path=args.model_path,
        gpu_id=args.gpu,
        save_every=args.save_every,
        cache_dir=args.cache_dir,
    )