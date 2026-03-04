# Georef Check Configuration

This directory contains the default runtime configuration for the georeferencing quality check system.

## Usage

```bash
# Use default config
python main.py check --input data/raw/dataset_custom

# Override specific config values via CLI
python main.py check --input data/raw/dataset_custom --threshold 0.15

# Use custom config file
python main.py check --config configs/my_config.yaml --input data/raw/dataset_custom

# CLI flags override config values
python main.py check --config configs/georef_check.yaml --threshold 0.20
```

## Configuration Fields

### Data Paths

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `dataset_dir` | string | Directory containing the 4-state ortho images | `data/raw/dataset_custom` |
| `labels_csv` | string | Path to labels.csv file | `data/raw/dataset_custom/labels.csv` |
| `split_csv` | string | Path to train_test_split.csv | `data/processed/train_test_split.csv` |

### Processing Options

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `use_streets` | boolean | Include streets basemap in feature extraction | `true` |
| `use_satellite` | boolean | Include satellite basemap in feature extraction | `true` |
| `viewport_size` | int | Screenshot size in pixels | `1024` |
| `zoom_level` | int | Tile zoom level for basemap | `17` |

### Feature Extraction

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `feature_version` | string | Version identifier for feature schema | `"v1"` |
| `output_features_csv` | string | Output path for extracted features | `data/processed/features.csv` |

### Threshold Baseline

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `threshold` | float | Classification threshold for good/bad orthos | `0.10` |

### Classifier Training

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `classifier_enabled` | boolean | Enable classifier-based prediction | `false` |
| `classifier_output_path` | string | Path to save trained model | `data/models/classifier.pkl` |
| `model_type` | string | Model type: xgboost, rf, gbm, lr | `xgboost` |

### Inference

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `predict_output_dir` | string | Output directory for prediction results | `outputs/check_results` |

### Evaluation

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `eval_output` | string | Path to save evaluation metrics JSON | `data/processed/eval_metrics.json` |

### Other

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `seed` | int | Random seed for reproducibility | `42` |
| `dry_run` | boolean | Validate file presence without processing | `false` |

## Examples

### Extract features with default config

```bash
python main.py features \
  --input-dir data/raw/dataset_custom \
  --split-file data/processed/train_test_split.csv \
  --output data/processed/features.csv
```

### Train with threshold-only mode

```bash
python main.py train \
  --data data/processed/features.csv \
  --threshold-only \
  --eval-output data/processed/eval_metrics.json
```

### Check georeferencing using threshold mode

```bash
python main.py check \
  --input data/raw/dataset_custom \
  --threshold 0.10
```

### Check with trained classifier

```bash
python main.py check \
  --input data/raw/dataset_custom \
  --model data/models/classifier.pkl \
  --mode classifier
```

## Notes

- If `--config` is not provided, the system falls back to default values from `config.py`
- CLI flags always override config file values
- The YAML config serves as the runtime "source of truth" for reproducible runs
- Static constants (e.g., API URLs, tile server) remain in `config.py`
