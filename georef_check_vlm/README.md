# Georeferencing Classifier (VLM)

Classify orthoimagery georeferencing quality using OpenRouter vision models.

## Setup

```bash
cd georef_check_vlm
pip install -r requirements.txt
```

Set API key:
```bash
export OPENROUTER_API_KEY=your_key_here
```

## Usage

```bash
python georef_classifier_concurrent.py --data-dir ../georef_check/data/raw/dataset_custom --workers 4
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `../georef_check/data/raw/dataset_custom` | Path to ortho images |
| `--output-csv` | `georef_classification_results.csv` | Output CSV |
| `--api-key` | env `OPENROUTER_API_KEY` | OpenRouter API key |
| `--model` | `google/gemini-3-flash-preview` | Vision model |
| `--workers` | 4 | Concurrent workers (concurrent only) |
| `--limit` | None | Process N orthos for testing |
| `--rps` | 0 | Requests per second cap (0=unlimited) |

## Output

CSV with columns:
- `ortho_id`: Ortho identifier
- `classification`: CORRECT, INCORRECT, or UNCERTAIN
- `explanation`: Model's reasoning

## How It Works

1. Loads 4 images per ortho:
   - `{id}_ortho_streets.png` - Drone over streets
   - `{id}_streets_only.png` - Streets alone
   - `{id}_ortho_satellite.png` - Drone over satellite
   - `{id}_satellite_only.png` - Satellite alone

2. Sends all 4 images to VLM with prompt asking if drone aligns with basemap

3. Parses response: CORRECT/INCORRECT/UNCERTAIN

4. Saves progress every 10 orthos (resume supported)

## Benchmark

Compare VLM predictions against ground truth labels:

```bash
python benchmark.py
```

Output includes:
- Confusion matrix (TP/FP/TN/FN)
- Accuracy, Precision, Recall, F1 Score
- Prediction distribution
- Ground truth distribution
