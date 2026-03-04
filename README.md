# Auto Georef

Automated system to detect misaligned/georeferenced orthoimages by comparing them against basemap tiles using deep learning-based feature matching.

## Overview

Given an ortho image and basemap image, determine if the ortho is properly georeferenced.

 at the same viewport**Method**: SuperPoint + LightGlue for feature matching, then homography fitting with RANSAC to quantify alignment quality.

**Output**: `{"good_probability": 0.85}` - probability that ortho is correctly aligned.

## Quickstart

```powershell
# 1. Setup env (Windows)
python -m venv .venv-windows
.venv-windows\Scripts\activate
pip install -r georef_check/requirements.txt
playwright install chromium

# 2. Scrape data
python georef_check/scrape_custom.py --count 50

# 3. Label data (manual)
# Edit georef_check/data/raw/dataset_custom/labels.csv with 1 (good) or 0 (bad)

# 4. Build train/test split
python georef_check/build_train_test_split.py

# 5. Extract features (config-driven)
python georef_check/main.py features --config georef_check/configs/georef_check.yaml

# 6. Train (threshold-only baseline)
python georef_check/main.py train --config georef_check/configs/georef_check.yaml --threshold-only

# 7. Check georeferencing
python georef_check/main.py check --config georef_check/configs/georef_check.yaml --input georef_check/data/raw/dataset_custom
```

## How It Works

The system compares **4 images** for each ortho:

1. **Ortho + Streets**: Drone imagery visible over streets basemap
2. **Ortho + Satellite**: Drone imagery visible over satellite imagery
3. **Streets only**: Pure streets basemap (no ortho)
4. **Satellite only**: Pure satellite imagery (no ortho)

### Feature Extraction Pipeline

1. Extract keypoints and descriptors from both images using SuperPoint
2. Match features using LightGlue (Transformer-based matcher)
3. Fit homography with RANSAC to find geometric transformation
4. Compute quality metrics:
   - `inlier_ratio`: fraction of matches consistent with the homography
   - `median_reprojection_error`: median geometric error after transform
   - `num_matches`, `num_inliers`
5. Map metrics to `good_probability`

If ortho is misaligned, the matching will produce fewer inliers and higher reprojection error.

## Project Structure

```
georef_check/
├── main.py                      # CLI entry point
├── config.py                    # Static configuration
├── config_loader.py             # YAML config loading + CLI override
├── configs/
│   ├── georef_check.yaml        # Default runtime config
│   └── README.md                # Config documentation
├── scrape_custom.py             # Web scraper
├── requirements.txt              # Dependencies
├── outputs/                     # Per-run inference outputs
├── data/
│   ├── raw/dataset_custom/      # Training data (4 images per ortho)
│   ├── processed/               # Features CSVs, eval metrics
│   └── models/                  # Trained classifiers, manifests
└── src/
    ├── features/matching.py     # SuperPoint+LightGlue matching
    ├── training/                # XGBoost classifier training
    └── inference/               # End-to-end inference pipeline
```

## Configuration

### YAML Config (Recommended)

Edit `configs/georef_check.yaml`:

```yaml
# Data
dataset_dir: data/raw/dataset_custom
labels_csv: data/raw/dataset_custom/labels.csv
split_csv: data/processed/train_test_split.csv

# Processing
use_streets: true
use_satellite: true
viewport_size: 1024
zoom_level: 17

# Features
feature_version: "v1"
output_features_csv: data/processed/features.csv

# Threshold baseline
threshold: 0.10

# Classifier
classifier_enabled: false
classifier_output_path: data/models/classifier.pkl

# Inference
predict_output_dir: outputs/check_results
```

See `configs/README.md` for full documentation.

### CLI Override

CLI flags override config values:

```bash
# Use config but override threshold
python main.py check --config configs/georef_check.yaml --threshold 0.15

# Use default config (auto-loaded)
python main.py check --input data/raw/dataset_custom --threshold 0.15
```

## Usage

### Extract Features

```bash
python main.py features \
  --input-dir data/raw/dataset_custom \
  --split-file data/processed/train_test_split.csv \
  --output data/processed/features.csv
```

### Train (Threshold-Only Mode)

```bash
python main.py train \
  --data data/processed/features.csv \
  --threshold-only \
  --eval-output data/processed/eval_metrics.json
```

### Train (Classifier Mode)

```bash
python main.py train \
  --data data/processed/features.csv \
  --model xgboost \
  --output data/models/classifier.pkl
```

### Check Georeferencing

```bash
# Threshold mode (no model)
python main.py check \
  --input data/raw/dataset_custom \
  --threshold 0.10

# Classifier mode
python main.py check \
  --input data/raw/dataset_custom \
  --model data/models/classifier.pkl
```

### Dry Run

Validate file presence without processing:

```bash
python main.py features --dry-run
python main.py train --dry-run
python main.py check --dry-run
```

## Output Artifacts

| Artifact | Description |
|----------|-------------|
| `data/processed/features.csv` | Extracted features with labels |
| `data/processed/eval_metrics.json` | Evaluation metrics |
| `data/models/classifier.pkl` | Trained model |
| `data/models/model_manifest.json` | Model metadata |
| `outputs/check_results/` | Per-run inference outputs |

## Data Source

Images scraped from [deadtrees.earth](https://deadtrees.earth) - public orthoimagery datasets.

## Advanced

### Feature Schema

Features extracted per sample:
- `good_probability`: Combined probability score
- `inlier_ratio_satellite`, `median_error_satellite`, `num_matches_satellite`, `num_inliers_satellite`, `good_probability_satellite`
- `inlier_ratio_streets`, `median_error_streets`, `num_matches_streets`, `num_inliers_streets`, `good_probability_streets`
- `inlier_ratio_min`, `median_error_max`, `num_matches_min` (combined robustness)

### Class Imbalance

System warns if minority class < 10 samples:
```
⚠️  WARNING: Class imbalance detected!
   Train: 54 good, 6 bad (ratio: 9.0:1)
```

Use balanced accuracy and confusion matrix for evaluation.
