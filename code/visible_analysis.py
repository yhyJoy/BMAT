"""
Building Visibility Classification
====================================
Uses a fine-tuned MobileNetV2 to classify whether a building is visible
in each street-view image. Processes panorama files city by city.

Input file (CSV or pickle) must contain: sample_id, year, month, img_path
Output CSV columns: sample_id, year, month, img_path, building, prob

Usage:
    python visible_analysis.py --param "China/Hong Kong" --meta_root ../data/csv --output_root ../data/csv
"""

import os
import argparse

import torch
import pandas as pd
from torch import nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_path: str, num_classes: int, device: torch.device):
    """Load a fine-tuned MobileNetV2 classifier from disk."""
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_pano(path: str) -> pd.DataFrame:
    """Load panorama records from a pickle (.p/.pkl) or CSV file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".p", ".pkl"):
        df = pd.read_pickle(path)
        # Normalise path column name: pickle files use 'path', CSVs use 'img_path'
        if "path" in df.columns and "img_path" not in df.columns:
            df = df.rename(columns={"path": "img_path"})
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .p, .pkl, or .csv.")

    # Normalise slashes and drop rows with missing image paths
    df["img_path"] = df["img_path"].str.replace("\\", "/", regex=False)
    invalid = df["img_path"].isna() | \
              df["img_path"].astype(str).str.strip().str.lower().isin(["none", "nan", ""])
    return df[~invalid].reset_index(drop=True)


def _load_done(output_csv: str) -> set:
    """Return the set of img_paths already present in the output file."""
    if not os.path.exists(output_csv):
        return set()
    try:
        return set(pd.read_csv(output_csv, usecols=["img_path"])["img_path"])
    except Exception:
        return set()


def _append_csv(records: list, path: str, save_cols: list) -> None:
    """Append records to CSV; writes header only if the file does not exist yet."""
    if not records:
        return
    df = pd.DataFrame(records)
    df = df[[c for c in save_cols if c in df.columns]]
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_city(pano_path: str, output_csv: str, model, transform,
               class_names: list, device: torch.device,
               batch_size: int = 64, save_every: int = 1000):
    """
    Run building visibility inference for one city's panorama file.
    Supports checkpoint resumption: img_paths already in output_csv are skipped.
    """
    save_cols = ["sample_id", "year", "month", "img_path", "building", "prob"]

    df   = load_pano(pano_path)
    done = _load_done(output_csv)
    df   = df[~df["img_path"].isin(done)]
    print(f"[INFO] To predict: {len(df)} | Already done: {len(done)}")

    if df.empty:
        print("[INFO] Nothing to predict.")
        return

    buffer = []
    rows   = df.to_dict("records")

    for start in tqdm(range(0, len(rows), batch_size), desc="Predicting"):
        batch = rows[start: start + batch_size]

        # Load images; mark unreadable ones as "error"
        images, valid_idx = [], []
        for i, row in enumerate(batch):
            try:
                images.append(transform(Image.open(row["img_path"]).convert("RGB")))
                valid_idx.append(i)
            except Exception:
                row["building"], row["prob"] = "error", 0.0

        # Run model inference on valid images
        if images:
            with torch.no_grad():
                logits = model(torch.stack(images).to(device))
                probs, indices = torch.softmax(logits, dim=1).max(dim=1)
            for j, idx in enumerate(valid_idx):
                batch[idx]["building"] = class_names[indices[j].item()]
                batch[idx]["prob"]     = round(probs[j].item(), 4)

        buffer.extend(batch)

        # Flush buffer to disk periodically to avoid memory buildup
        if len(buffer) >= save_every:
            _append_csv(buffer, output_csv, save_cols)
            buffer = []

    _append_csv(buffer, output_csv, save_cols)  # flush remaining records
    print(f"[DONE] Saved to {output_csv}")


# ---------------------------------------------------------------------------
# Batch dispatch
# ---------------------------------------------------------------------------

def run_batch(param: str, meta_root: str, output_root: str, model_path: str,
              num_classes: int, class_names: list, device: torch.device,
              batch_size: int, save_every: int):
    """Dispatch inference by 'all' | '{region}' | '{region}/{city}'."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = load_model(model_path, num_classes, device)

    def _find_pano(city_dir):
        # Accept meta_*.p / .pkl / .csv; priority in that order
        for ext in (".p", ".pkl", ".csv"):
            hits = [f for f in os.listdir(city_dir)
                    if f.startswith("meta_") and f.endswith(ext)]
            if hits:
                return os.path.join(city_dir, hits[0])
        return None

    def _run(region, city):
        city_dir   = os.path.join(meta_root, region, city)
        pano_path  = _find_pano(city_dir)
        output_csv = os.path.join(output_root, region, city, f"{city}_building_visible.csv")
        if not pano_path:
            print(f"[SKIP] {region}/{city}: no meta_* file found.")
            return
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        print(f"\n[INFO] Processing {region}/{city}")
        try:
            infer_city(pano_path, output_csv, model, transform,
                       class_names, device, batch_size, save_every)
        except Exception as e:
            print(f"[ERROR] {region}/{city}: {e}")

    parts = param.strip().split("/")
    if param.lower() == "all":
        for region in sorted(os.listdir(meta_root)):
            if not os.path.isdir(os.path.join(meta_root, region)):
                continue
            for city in sorted(os.listdir(os.path.join(meta_root, region))):
                _run(region, city)
    elif len(parts) == 1:
        for city in sorted(os.listdir(os.path.join(meta_root, parts[0]))):
            _run(parts[0], city)
    elif len(parts) == 2:
        _run(parts[0], parts[1])
    else:
        print(f"[ERROR] Unsupported --param format: '{param}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Building visibility classification from street-view images.")
    parser.add_argument("--param",       type=str, required=True,
                        help="'all' | '{region}' | '{region}/{city}'")
    parser.add_argument("--meta_root",   type=str, required=True,
                        help="Root dir containing meta_* panorama files")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root dir for output CSVs (mirrors meta_root structure)")
    parser.add_argument("--model_path",  type=str,
                        default="../model/building_visible_infer.pth",
                        help="Path to MobileNetV2 weights (default: ../model/building_visible_infer.pth)")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--class_names", type=str, nargs="+", default=["no", "yes"])
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--save_every",  type=int, default=1000,
                        help="Flush buffer to disk every N records (default: 1000)")
    parser.add_argument("--device",      type=str, default="cuda:0")
    args = parser.parse_args()

    # Fall back to CPU if CUDA is unavailable
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    run_batch(args.param, args.meta_root, args.output_root, args.model_path,
              args.num_classes, args.class_names, device, args.batch_size, args.save_every)