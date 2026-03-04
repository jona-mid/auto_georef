# Instructions: Improvements to adopt for the Georeferencing Quality Check repo (inspired by GeoSense-Freiburg/OrthoAOI)

This document describes concrete, actionable changes to improve the **georeferencing quality check** repository by adopting useful structural patterns from `GeoSense-Freiburg/OrthoAOI` (while *not* copying its AOI-segmentation approach, which solves a different problem).

---

## 1) What to adopt vs. what not to adopt

### Adopt (high value)
1. **Config-driven runs (YAML + CLI overrides)**  
   OrthoAOI uses a YAML config file as the primary interface, with CLI flags overriding config values. This is excellent for reproducibility and for running the pipeline in different environments/datasets.

2. **Clear split between “train” and “predict/check” entrypoints**  
   OrthoAOI has a training entrypoint and a standalone prediction entrypoint. For georef-check this maps well to:
   - training a classifier (optional)
   - running `check` in production to output `{"good_probability": ...}`

3. **Artifact management (saving outputs + recording paths/metadata)**  
   OrthoAOI writes back checkpoint paths to config after training. For georef-check, adopt the *idea* (not necessarily “write back to config”) to generate an artifact manifest:
   - model path
   - threshold used
   - feature schema version
   - dataset hash / counts
   - evaluation metrics

4. **Documentation style for configs**  
   OrthoAOI’s `configs/README.md` that explains each config field is a strong pattern to emulate.

### Do NOT adopt (mismatched)
1. **DINOv3 segmentation + LightningModule/DataModule design**  
   OrthoAOI’s model is a segmentation head over DINO patch tokens for AOI masks. Your georef repo’s core task is **pairwise alignment scoring** using **SuperPoint + LightGlue + RANSAC homography** and then mapping to probability.

2. **COCO mask training pipeline**  
   Your labels are per-ortho (good/bad), and your inputs are screenshot pairs (4-state). COCO polygon masks are irrelevant for this repo.

---

## 2) Recommended repo structure changes

Your current structure is already good; the goal is to *standardize interfaces* and make runs reproducible.

### 2.1 Add a `configs/` directory with a default config
Create:
- `configs/georef_check.yaml` (or multiple configs)
- `configs/README.md` describing all fields and examples

**Why:** Makes it easy to run training/inference consistently across machines, especially Windows.

---

## 3) Adopt OrthoAOI-style “config + CLI override” pattern

### 3.1 Desired behavior
- All commands accept `--config configs/georef_check.yaml`
- CLI flags override config values
- Running a command without `--config` still works (defaults)
- Config should capture:
  - dataset paths
  - which basemap modes to use (streets/satellite/both)
  - thresholds and model paths
  - output directories

### 3.2 Suggested config schema (example)
Add `configs/georef_check.yaml` with fields like:

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
threshold: 0.10  # current best from F1 optimization

# Classifier
classifier_enabled: false
classifier_output_path: data/models/classifier.pkl

# Inference
predict_output_dir: outputs/check_results
```

**Note:** You can keep `config.py` for constants, but the YAML config should be the runtime “source of truth” for reproducible runs.

---

## 4) Make “train” vs “check” modes explicit (two-stage maturity model)

Your repo is currently in a great place to support both:
1) **Threshold-only baseline** (already working, easier to deploy)
2) **Classifier-based** (XGBoost planned)

### 4.1 Explicit CLI support
In `main.py`, ensure:
- `features` command always works (independent)
- `train` command produces a model artifact
- `check` command can run in either mode:

Example:

```bash
# Baseline mode
python main.py check --config configs/georef_check.yaml --mode threshold

# Classifier mode
python main.py check --config configs/georef_check.yaml --mode classifier --model data/models/classifier.pkl
```

If you don’t want a `--mode`, use automatic logic:
- if `--model` is provided => classifier
- else => threshold

---

## 5) Standardize output artifacts (adopt OrthoAOI’s “write artifacts” discipline)

### 5.1 Add a run output folder convention
For any run, produce:

- `data/processed/features*.csv` (features extraction)
- `data/processed/eval_metrics*.json` (evaluation)
- `data/models/` (trained models)
- `outputs/` (per-run inference outputs, optional)

### 5.2 Add an “artifact manifest” JSON
After `train`, write a machine-readable manifest:

`data/models/model_manifest.json`:

```json
{
  "created_at": "2026-03-04T12:34:56Z",
  "feature_version": "v1",
  "threshold_baseline": 0.10,
  "classifier_path": "data/models/classifier.pkl",
  "train_samples": 60,
  "test_samples": 17,
  "metrics_path": "data/processed/eval_metrics.json",
  "notes": "Trained on dataset_custom, streets+satellite."
}
```

This replaces OrthoAOI’s “write checkpoint paths back into YAML” pattern with a safer approach for your repo.

---

## 6) Dataset + feature engineering improvements (important for georeferencing QC)

These are project-specific improvements (not from OrthoAOI), but they align with the “clean pipeline” mentality.

### 6.1 Keep streets and satellite metrics separate
Ensure your features include *separate* fields for streets and satellite comparisons, e.g.:

- `inlier_ratio_streets`
- `median_reprojection_error_streets`
- `num_matches_streets`
- `inlier_ratio_satellite`
- `median_reprojection_error_satellite`
- `num_matches_satellite`

Then add combined robustness features:

- `inlier_ratio_min`
- `median_reprojection_error_max`
- `num_matches_min`

This makes your model/threshold more stable.

### 6.2 Handle missing-tile / invalid-sample flags as features
You already discard samples if >50% identical pixels; also consider saving:
- `satellite_missing_tiles_flag` (0/1)
- `streets_missing_tiles_flag` (0/1)
so you can later analyze failure modes.

### 6.3 Address class imbalance (current gap)
You noted only ~6 bad vs ~74 good.

Actions:
- Add a warning in CLI output if minority class count < a threshold
- Prefer metrics like **balanced accuracy, precision/recall**, and report confusion matrix
- When training XGBoost, use:
  - `scale_pos_weight` (or sample weights)
  - stratified splitting
- Collect more “bad” samples as a priority

---

## 7) Documentation changes to implement (copy OrthoAOI’s clarity)

### 7.1 Add a “Quickstart” section that is config-driven
Update your README to show the primary workflow as:

1. Setup env
2. Scrape
3. Label
4. Build split
5. Extract features
6. Evaluate threshold
7. Train classifier (optional)
8. Check

and always show a config-driven invocation.

### 7.2 Add a `configs/README.md`
Like OrthoAOI’s `configs/README.md`, document every config field and its default.

### 7.3 Add a “Production” section
Include:
- how to select threshold
- what files need to be present
- recommended `--headless` scraping mode
- expected output format: `{"good_probability": ...}`

---

## 8) Engineering hygiene (small, high impact)

1. **Add `pyproject.toml` or keep `requirements.txt` but pin key versions**  
   (LightGlue / torch / opencv can be version-sensitive on Windows.)

2. **Add a `--seed` option** (for split building and any stochastic steps)

3. **Add a `--dry-run` mode** for scraping and feature extraction (validate file presence, list planned work)

4. **Add basic unit/integration tests** for:
   - dataset file discovery (4 images per id)
   - feature extraction returns expected columns
   - `check` returns valid JSON

---

## 9) Summary checklist (implementation order)

**Phase 1 (fast wins)**
- [ ] Add `configs/` directory + `configs/georef_check.yaml`
- [ ] Make all CLI commands accept `--config` and support CLI override
- [ ] Add `configs/README.md` documenting fields
- [ ] Standardize output paths and add `outputs/` convention

**Phase 2 (quality + reproducibility)**
- [ ] Add model artifact manifest JSON
- [ ] Improve feature schema: separate streets/satellite + combined min/max
- [ ] Add imbalance warnings + better evaluation reporting

**Phase 3 (production readiness)**
- [ ] Classifier training pipeline stabilized (weights / stratification)
- [ ] Tests for dataset/feature/check pipeline
- [ ] Pin dependencies for Windows stability

---

## 10) Explicit note on OrthoAOI mapping

OrthoAOI is a good reference for:
- clean CLI interfaces
- config patterns
- artifact handling discipline

But the georef-check repo should remain centered on:
- SuperPoint + LightGlue matching
- RANSAC homography consistency metrics
- threshold baseline + optional classifier
- 4-state screenshot dataset design

That combination is well-suited to georeferencing QA.