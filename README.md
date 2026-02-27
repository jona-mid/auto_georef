# Auto Georef

Automated system to detect misaligned/georeferenced orthoimages by comparing them against basemap tiles using deep learning-based feature matching.

## Overview

Given an ortho image and basemap image at the same viewport, determine if the ortho is properly georeferenced.

**Method**: SuperPoint + LightGlue for feature matching, then homography fitting with RANSAC to quantify alignment quality.

**Output**: `{"good_probability": 0.85}` - probability that ortho is correctly aligned.

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
5. Map metrics to `good_probability`

If ortho is misaligned, the matching will produce fewer inliers and higher reprojection error.

## Project Structure

```
georef_check/
├── scrape_custom.py          # Web scraper to capture ortho images
├── main.py                   # CLI entry point
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── data/
│   └── raw/dataset_custom/   # Training data (4 images per ortho)
└── src/
    ├── features/matching.py  # SuperPoint+LightGlue matching
    ├── training/             # XGBoost classifier training
    └── inference/            # End-to-end inference pipeline
```

## Installation

```powershell
# Create virtual environment
python -m venv .venv-windows
.venv-windows\Scripts\activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

## Usage

### Scrape Data

```powershell
python scrape_custom.py --count 50
```

### Extract Features

```powershell
python main.py features --input-dir data/raw/dataset_custom --output data/processed/features.csv
```

### Train Classifier

```powershell
python main.py train --data data/processed/features.csv --output data/models/classifier.pkl
```

### Check Georeferencing

```powershell
python main.py check --input data/raw/dataset_custom
```

Output:
```json
{"good_probability": 0.85}
```

## Configuration

Edit `config.py`:
- `VIEWPORT_SIZE`: Screenshot size (default: 1024)
- `ZOOM_LEVEL`: Tile zoom level (default: 17)
- `THRESHOLD`: Classification threshold (default: 0.5)

## Data Source

Images scraped from [deadtrees.earth](https://deadtrees.earth) - public orthoimagery datasets.
