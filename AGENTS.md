# Georeferencing Quality Check System

## Project Overview

Automated system to detect misaligned/georeferenced orthoimages by comparing them against basemap tiles using deep learning-based feature matching.

**Goal**: Given an ortho image and basemap image at the same viewport, determine if the ortho is properly georeferenced.

**Method**: SuperPoint + LightGlue for feature matching, then homography fitting with RANSAC to quantify alignment quality.

**Output**: `{"good_probability": 0.85}` - probability that ortho is correctly aligned.

## Current Status

### Working Components

| Component | Status | Notes |
|-----------|--------|-------|
| Project structure | ✅ Complete | `georef_check/` directory with modules in `src/` |
| Custom scraper | ✅ Working | `src/data_collection/scraper.py` (main scraper) captures 4 states per ortho |
| Windows Python env | ✅ Complete | `.venv` configured with all deps (including `transformers`) |
| Training data | ✅ 60 samples | `data/raw/dataset_manual/` (approx. 42 good, 18 bad) |
| Labels | ✅ Complete | `labels.csv` populated |
| Train/test split | ✅ Complete | `train_test_split.csv` (51 train, 9 test) |
| Feature extraction | ✅ Complete | SuperPoint+LightGlue in `matching.py` with 1000x1000 cropping |
| Threshold eval | ✅ Complete | Evaluated on manual dataset (production threshold chosen: 0.05) |
| Classifier (XGBoost) | ⚠️ Tried | Trained but showed overfitting; threshold-only baseline outperformed on test set |
| Check command | ✅ Working | Returns `good_probability` |

### Current Outputs (already run)

| Output | Description |
|--------|-------------|
| `data/processed/features.csv` | 60 samples from manual dataset, center-cropped |
| `data/processed/eval_metrics.json` | Threshold-only eval on manual dataset |
| `data/processed/threshold_eval.json` | Full threshold sweep results (baseline vs XGBoost) |

### Remaining Gaps

- **Production Threshold**: Set to `0.05` (updated in `configs/georef_check.yaml`) — consider re-evaluating after more data
- **More bad examples**: Current minority class (~18) is small; collect more misaligned orthos to reduce overfitting risk
- **Investigate failures**: Several samples have 0 inliers or high reprojection error; inspect these ortho IDs for labeling or matching issues

### Files Structure

```
georef_check/
├── archive/                         # Deprecated code (not imported)
│   ├── scraper.py                   # Selenium scraper
│   ├── deadtrees_api.py             # API client
│   ├── playwright_scraper.py        # Playwright scraper
│   ├── viewport.py                  # Render paired viewports
│   ├── tile_fetcher.py              # Fetch basemap tiles
│   ├── augment.py                   # Synthetic negatives
│   ├── test_playwright.py           # UI testing script
│   ├── test_matching.py             # Matching test script
│   └── current_state.md             # Previous working document
├── config.py                        # Configuration (viewport size, zoom, etc.)
├── requirements.txt                 # Python dependencies
├── main.py                          # CLI entry point
├── scrape_custom.py                 # ✅ MAIN SCRAPER - captures 4 states per ortho
├── build_train_test_split.py        # Build train/test split from labeled data
├── gen_labels.py                    # Helper to regenerate labels.csv
├── data/
│   ├── raw/dataset_manual/          # ✅ Current dataset
│   │   ├── {id}_ortho_streets.png   # Drone ON + Streets
│   │   ├── {id}_ortho_satellite.png # Drone ON + Satellite
│   │   ├── {id}_streets_only.png    # Drone OFF + Streets
│   │   ├── {id}_satellite_only.png  # Drone OFF + Satellite
│   │   ├── metadata.json            # Scraped dataset info
│   │   └── labels.csv               # ortho_id, label (0/1)
│   ├── processed/                   # Feature CSVs & eval metrics
│   └── models/                      # Trained classifiers
└── src/
    ├── data_collection/
    │   └── fetcher.py               # Load orthoimages from GeoTIFFs
    ├── features/
    │   ├── extractor.py             # Phase correlation + edge features (legacy)
    │   └── matching.py               # ✅ SuperPoint+LightGlue matching
    ├── training/
    │   ├── dataset.py                # Dataset builder
    │   └── train.py                  # XGBoost training
    └── inference/
        └── pipeline.py               # End-to-end inference
```

## How Georeferencing Check Works

The system compares **4 images** for each ortho:

1. **Ortho + Streets**: Drone imagery visible over streets basemap
2. **Ortho + Satellite**: Drone imagery visible over satellite imagery
3. **Streets only**: Pure streets basemap (no ortho)
4. **Satellite only**: Pure satellite imagery (no ortho)

**Feature Extraction Pipeline:**
- Extract features comparing ortho_streets vs streets_only
- Extract features comparing ortho_satellite vs satellite_only
- Combine features for final prediction

**SuperPoint+LightGlue matching pipeline:**
1. Extract keypoints and descriptors from both images using SuperPoint
2. Match features using LightGlue (Transformer-based matcher)
3. Fit homography with RANSAC to find geometric transformation
4. Compute quality metrics:
   - `inlier_ratio`: fraction of matches consistent with the homography
   - `median_reprojection_error`: median geometric error after transform
5. Map metrics to `good_probability` using a simple formula

If ortho is misaligned, the matching will produce fewer inliers and higher reprojection error.

### Phase Correlation Pre-filter (new)

A fast, CPU‑friendly phase‑correlation prefilter has been added under `src/features/phase_correlation.py`.
- Purpose: quickly estimate a translation (dx, dy) and a PSR (peak‑to‑sidelobe ratio) on Canny edge maps to provide a cheap `good_probability` proxy.
- Usage: run the evaluation script `scripts/pc_check.py eval` to compute PC metrics across the labeled dataset and optionally fit a calibrator with `scripts/pc_check.py calibrate`.
- Best practice: treat PC as a conservative gate—accept only when PSR is high and shift is small; otherwise fallback to the existing SuperPoint+LightGlue matcher for a robust decision.

Files added:
- `src/features/phase_correlation.py` — FFT phase correlation and edge variant
- `scripts/pc_check.py` — evaluate and calibrate PC on `data/raw/dataset_manual`
- `data/models/pc_calibrator.pkl` (when you run `calibrate`) — logistic calibrator mapping (psr,disp) → probability

Note: Images are center-cropped to 1000x1000 during loading (see `src/features/matching.py`).

## Data Collection Workflow

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Scrape (automated)                                 │
│  ─────────────────────────────────────────────────────────  │
│  • Random sample IDs from range 1-8000                      │
│  • For each valid ID:                                       │
│    - Navigate to https://deadtrees.earth/dataset/{id}       │
│    - Turn OFF: Forest Cover, Deadwood, Area of Interest     │
│    - Hide UI elements (header, footer, etc.)                │
│    - Capture ortho_streets.png (Drone ON + Streets)         │
│    - Capture ortho_satellite.png (Drone ON + Imagery)       │
│    - Check for missing tiles in ortho_satellite; if >50%    │
│      identical pixels → discard dataset and skip to next    │
│    - Capture streets_only.png (Drone OFF + Streets)         │
│    - Capture satellite_only.png (Drone OFF + Imagery)       │
│    - Check for missing tiles in satellite_only; if >50%     │
│      identical pixels → discard dataset and skip to next    │
│  • Skip invalid IDs automatically                           │
│  • Wait 15s for first satellite imagery to load             │
│  • Wait 5s for second satellite (tiles already loaded)      │
│  • Output: data/raw/dataset_manual/*.png + metadata.json    │
└─────────────────────────────────────────────────────────────┘

### Important Clarifications

1. **Each ortho has its own ID** (e.g., 565, 2117, 3202, etc.)
2. **Orthos may be good OR bad georeferencing** - user must label
3. **Site access**: deadtrees.earth is publicly accessible (no login required)
4. **Platform**: Run on Windows native Python (not WSL)
5. **Satellite tiles load slowly**: Scraper waits 15 seconds for satellite imagery

### Labels Format (CSV)

Edit `data/raw/dataset_manual/labels.csv`:

```csv
ortho_id,label
565,1
2117,0
3202,1
...
```

   - `label=1`: Ortho is correctly aligned on the map
   - `label=0`: Ortho is misaligned/shifted

## Running on Windows

### Prerequisites

1. **Python dependencies**: playwright, torch, lightglue, opencv, etc.
2. Python 3.10+ installed on Windows

### Dependencies

See `requirements.txt`:
```
rasterio
opencv-python
scikit-image
numpy
pandas
xgboost
scikit-learn
requests
Pillow
tqdm
pyyaml
playwright
torch
torchvision
git+https://github.com/cvg/LightGlue.git
```

### Setup

```powershell
# In Windows PowerShell
cd C:\Users\jonathan\Documents\HiWi\georeferencing\georef_check

# Create Windows virtual environment
python -m venv .venv-windows
.\.venv-windows\Scripts\activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

### Step 1: Scrape

```powershell
# Scrape orthos (collect GeoTIFFs -> viewports)
python georef_check/main.py collect --input-dir <path-to-geotiffs> --output-dir georef_check/data/raw/dataset_manual --n-images 50

# Options:
# --input-dir <path-to-geotiffs>  # Directory containing GeoTIFFs
# --output-dir <output dir>       # Where to write viewports (default: data/raw)
# --n-images 50                   # Number of images to process
```

**What the scraper does:**
1. Randomly sample IDs from range 1-8000
2. Navigate to each dataset page (`https://deadtrees.earth/dataset/{id}`)
3. Turn OFF: Forest Cover, Deadwood, and Area of Interest checkboxes
4. Hide UI elements (header, footer, etc.)
5. Capture 4 screenshots per dataset:
   - `{id}_ortho_streets.png` - Drone ON + Streets basemap
   - `{id}_ortho_satellite.png` - Drone ON + Satellite imagery (waits 15s)
   - `{id}_streets_only.png` - Drone OFF + Streets basemap
   - `{id}_satellite_only.png` - Drone OFF + Satellite imagery (waits 15s)
6. Skip invalid IDs automatically
7. Save to `data/raw/dataset_manual/`

**UI Elements (Ant Design components):**
```python
# Layer checkboxes (need to UNCHECK these):
- "Forest Cover" → label:has-text('Forest Cover') >> input[type='checkbox']
- "Deadwood" → label:has-text('Deadwood') >> input[type='checkbox']
- "Area of Interest" → label:has-text('Area of Interest') >> input[type='checkbox']
- "Drone Imagery" → label:has-text('Drone Imagery') >> input[type='checkbox']

# Basemap radio buttons:
- "Streets" → .ant-segmented-item:has-text('Streets')
- "Imagery" (Satellite) → .ant-segmented-item:has-text('Imagery')

# Map container:
- OpenLayers viewport → .ol-viewport
```

### Step 2: Label (Manual)

1. Open `data/raw/dataset_manual/labels.csv`
2. For each ortho_id in the file:
   - Open `{id}_ortho_streets.png` and `{id}_streets_only.png`
   - Check if drone imagery aligns correctly with streets basemap
   - Optionally check `{id}_ortho_satellite.png` vs `{id}_satellite_only.png`
   - Add `1` (good) or `0` (bad) to the label column
3. Save the file

### Step 3: Extract Features

```powershell
# Extract features from 4-state images using split file
python georef_check/main.py features --input-dir georef_check/data/raw/dataset_manual --split-file georef_check/data/processed/train_test_split.csv --output georef_check/data/processed/features.csv
```

### Step 4: Train Classifier

```powershell
# Train XGBoost on extracted features
python georef_check/main.py train --data georef_check/data/processed/features.csv --output georef_check/data/models/classifier.pkl
```

## Feature Extraction Details (Legacy - needs update)

Old pipeline extracted 8 features from ortho-basemap pairs:

| Feature | Description | Good alignment | Bad alignment |
|---------|-------------|----------------|---------------|
| `dx` | Horizontal shift (pixels) | ~0 | Large |
| `dy` | Vertical shift (pixels) | ~0 | Large |
| `peak_strength` | Phase correlation confidence | High (~1.0) | Low |
| `edge_correlation` | Canny edge matching | High | Low |
| `edge_density_diff` | Edge density difference | Low | High |
| `ssim_approx` | Structural similarity | High (~1.0) | Low |
| `mse` | Mean squared error | Low | High |
| `hist_correlation` | Histogram matching | High (~1.0) | Low |

**New pipeline should extract features comparing:**
- `{id}_ortho_streets.png` vs `{id}_streets_only.png`
- `{id}_ortho_satellite.png` vs `{id}_satellite_only.png`
- Combine both comparisons for final prediction

## Configuration

Edit `config.py`:
```python
VIEWPORT_SIZE = 1024      # Screenshot size
ZOOM_LEVEL = 17           # Tile zoom level
BASEMAP_STYLES = ["satellite", "streets"]
THRESHOLD = 0.5          # Classification threshold
```

## CLI Commands (Reference)

```bash
# Scrape orthos (collect GeoTIFFs -> viewports)
python georef_check/main.py collect --input-dir <path-to-geotiffs> --output-dir georef_check/data/raw/dataset_manual --n-images 50

# Build train/test split from labeled data
python georef_check/src/training/build_split.py --labels georef_check/data/raw/dataset_manual/labels.csv --output georef_check/data/processed/train_test_split.csv

# Extract features (with split file - uses train/test split)
python georef_check/main.py features --input-dir georef_check/data/raw/dataset_manual --split-file georef_check/data/processed/train_test_split.csv --output georef_check/data/processed/features.csv

# Train classifier
python georef_check/main.py train --data georef_check/data/processed/features.csv --output georef_check/data/models/classifier.pkl

# Check georeferencing (returns good_probability)
python georef_check/main.py check --input georef_check/data/raw/dataset_manual
```

## Deadtrees.earth UI Reference

### Layer Controls

**Data Layers Section:**
- Forest Cover (checkbox) - TURN OFF before capture
- Deadwood (checkbox) - TURN OFF before capture  
- Drone Imagery (checkbox) - Toggle ON/OFF for captures
- Area of Interest (checkbox) - TURN OFF before capture

**Basemap Section:**
- Streets (radio button) - Select for streets captures
- Imagery (radio button) - Select for satellite captures

### UI Elements to Hide

```javascript
// Selectors to hide before screenshots
['header', 'footer', '.ant-layout-header', '.ant-layout-sider', '.ant-float-btn-group']
```

### Page Structure

- **URL**: `https://deadtrees.earth/dataset/{id}`
- **Map Library**: OpenLayers (`.ol-viewport`)
- **UI Framework**: Ant Design
- **Public access**: No login required
- **Ortho IDs**: Range approximately 1-8000
- **Invalid IDs**: Some IDs don't have maps, scraper skips automatically

## Files Status

| File | Status | Description |
|------|--------|-------------|
| src/data_collection/scraper.py | ✅ Active | Main scraper - 4 states per ortho, includes missing tile detection |
| build_train_test_split.py | ✅ Helper | Build train/test split from labeled data |
| archive/ | 📁 Archived | Deprecated code (scrapers, test scripts, old docs) |
| gen_labels.py | ✅ Helper | Regenerate labels.csv |
| src/features/matching.py | ✅ Active | SuperPoint+LightGlue matching with homography+RANSAC |

## Notes

- Each ortho is a single drone image with its own ID
- Scraping takes ~25-45 seconds per dataset (15s wait for first satellite, 5s for second, plus failures discarded quickly)
- Run on Windows native Python (not WSL) for browser automation
- Non-headless mode recommended for debugging scraper issues
- **Missing tile detection**: Satellites images are checked for >50% uniform pixels; failed datasets are discarded immediately
- **Fail-fast**: If satellite tiles fail to load, the dataset is skipped without retries, saving time
-- Current dataset: 60 samples (51 train, 9 test) in data/raw/dataset_manual/

## Next Steps

1. **Deploy threshold baseline** (recommended): use `threshold: 0.05` in `configs/georef_check.yaml` for production checks.
2. **Inspect failing ortho IDs**: review samples with 0 inliers or high reprojection error (see `data/processed/features.csv`) and confirm labels.
3. **Collect more negatives**: add more misaligned examples to improve classifier training (aim for 100+ samples before retraining XGBoost).
4. **Re-run threshold sweep** after additional data to re-evaluate whether a classifier outperforms the baseline.
5. **Integrate PC prefilter**: wire PC into `src/inference/pipeline.py` as a conservative gate and add PSR/disp as features to `data/processed/features.csv` for classifier retraining.
