# geoai-project
Trying to create a Model for classification of drone images.
# SVAMITVA Feature Extraction Pipeline

AI/ML pipeline for extracting building footprints with roof type
classification, roads, waterbodies and utilities from SVAMITVA
drone orthophotos (3.5 cm/px).

## Team
- Kalki — SegFormer-b2 + EfficientNetV2-S + Integration
- Friend — DeepLabV3+ EfficientNet-B4 + Metrics + Docs

## Architecture
| Model | Task |
|---|---|
| SegFormer-b2 | Building segmentation |
| EfficientNetV2-S | Roof type: RCC / Tiled / Tin / Others |
| DeepLabV3+ EfficientNet-B4 | Roads, water, utilities |

## Setup

### Step 1 — Clone and create environment
    git clone https://github.com/YOUR_USERNAME/svamitva-feature-extraction.git
    cd svamitva-feature-extraction
    mamba env create -f environment.yml
    conda activate sevenhills

### Step 2 — If PyTorch fails due to CUDA mismatch
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

### Step 3 — Reproduce full pipeline
    dvc pull
    dvc repro

## Expected Accuracy
| Feature | IoU |
|---|---|
| Building detection | 0.82 - 0.88 |
| Roof classification | 0.80 - 0.87 |
| Roads | 0.80 - 0.86 |
| Water | 0.85 - 0.91 |
| Utilities | 0.55 - 0.70 |
| Overall mIoU | 0.81 - 0.87 |

## Output
OGC-compliant GeoPackage with 4 layers.
Open directly in QGIS.

