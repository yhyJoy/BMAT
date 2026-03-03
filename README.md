# BMAT: Building Material Attributes and Temporal Records

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![Paper](https://img.shields.io/badge/Paper-TODO-blue.svg)](TODO)  -->
<!-- [![Dataset](https://img.shields.io/badge/Dataset-TODO-green.svg)](TODO)-->
[![Model](https://img.shields.io/badge/HuggingFace-yinjoy30/BMAT-orange.svg)](https://huggingface.co/yinjoy30/BMAT)

## Overview

BMAT is the first global, longitudinal dataset of building facade materials mapped to individual footprints. It covers **22.09 million buildings** across **73 cities** (2007–2025), produced by combining 147 million street-view images with a fine-tuned Vision-Language Model (F1 = 0.91).

Material categories: `brick` · `concrete` · `glass` · `metal` · `stone` · `stucco` · `tile` · `wood` · `other`

📦 **Dataset**: [TODO] &nbsp;·&nbsp; 🤗 **Model**: [yinjoy30/BMAT](https://huggingface.co/yinjoy30/BMAT)

## Pipeline

<p align="center">
<img width="8122" height="5749" alt="流程图" src="https://github.com/user-attachments/assets/27a712bb-d7fd-4349-b614-5dc7f96c3235" />
<br>
<em>Workflow of the BMAT dataset production pipeline.</em>
</p>

```
[Stage 1]  Data Collection
           Collect street-view imagery, road networks, and building footprints
                             │
                             ▼
[Stage 2]  Per-City Processing
     [2.1] visible_analysis.py    — filter images with no visible building (MobileNetV2)
                             │
     [2.2] obstruct_analysis.py   — cast geometric sight lines from panorama point to
                             │      building centroid/vertices; discard obstructed views
                             ▼
     [2.3] vlm_predict.py         — predict facade material using fine-tuned
                                    Qwen2.5-VL-7B-Instruct on images passing 2.1 & 2.2
                             │
                             ▼
[Stage 3]  Batch Processing
           Run Stages 1 & 2 across all cities in a region or the full dataset
```

## Installation
```bash
git clone https://github.com/yhyJoy/BMAT.git
cd BMAT
pip install -r requirements.txt
```

## Usage

### Stage 2.1 — Visibility Classification
```bash
python code/visible_analysis.py \
    --param "China/Hong Kong" \
    --meta_root data/csv --output_root data/csv \
    --model_path model/building_visible_infer.pth \
    --device cuda:0
```

Input: `meta_{city}.csv` — `sample_id, year, month, img_path`  
Output: `{city}_building_visible.csv` — adds `building` (yes/no), `prob`

### Stage 2.2 — Obstruction Analysis
```bash
python code/obstruct_analysis.py \
    --param "China/Hong Kong" \
    --meta_root data/csv --shp_root data/shp
```

Input: `meta_{city}.csv` + building footprint shapefile (`.shp` / `.gpkg`)  
Output: `{city}_building_obstruct.csv` — adds `centerline_visible` (True/False)

### Stage 2.3 — Material Prediction
```bash
python code/vlm_predict.py \
    --region China --city "Hong Kong" \
    --model_path yinjoy30/BMAT --gpu 0
```

Input: `{city}_building_visible.csv` + `{city}_building_obstruct.csv`  
Output: `{city}_label.csv` — adds `pred_label` ∈ {brick, concrete, glass, metal, stone, stucco, tile, wood, other}

#### Fine-tuning Data

Labelled training data is located in `data/label/`:

- `image_label.csv` — manually annotated by our team
- `image_label_otherSource.csv` — partially sourced from [urban-resource-cadastre-repository](https://github.com/raghudeepika/urban-resource-cadastre-repository) and [Building Facade Image Dataset](https://figshare.com/articles/dataset/Dataset_of_building_characteristics_from_building_fa_ade_images/25931941?file=46673209)

### Stage 3 — Batch Processing

 Each city is processed sequentially through the full Stage 1 → 2 pipeline.



<!-- ## Citation
```bibtex
@article{yin2025bmat,
  title   = {BMAT: A global dataset of building material attributes and temporal records at the footprint level},
  author  = {Yin, Hanyu and Zhang, Fan and Wang, Yuqing and Wu, Lun and Liu, Yu},
  journal = {TODO},
  year    = {2025},
  doi     = {TODO}
}
```
-->
