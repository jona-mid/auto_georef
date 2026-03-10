"""
Benchmark script for FFT Phase Correlation georeferencing checker.

Runs phase correlation on the labeled dataset and evaluates performance
using the train/test split.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FFT Phase Correlation on labeled dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/dataset_manual",
        help="Directory with 4-state PNGs",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="data/processed/train_test_split.csv",
        help="Path to train_test_split.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/eval_metrics_phase.json",
        help="Output JSON file for eval metrics",
    )
    parser.add_argument(
        "--basemap",
        type=str,
        default="both",
        choices=["both", "streets", "satellite"],
        help="Which basemap to use",
    )
    parser.add_argument(
        "--use-edges",
        action="store_true",
        help="Run phase correlation on Canny edge images",
    )
    parser.add_argument(
        "--psr-min",
        type=float,
        default=3.0,
        help="PSR threshold below which result is unreliable",
    )
    parser.add_argument(
        "--psr-good",
        type=float,
        default=8.0,
        help="PSR threshold above which full PSR score is given",
    )
    parser.add_argument(
        "--shift-good",
        type=float,
        default=5.0,
        help="Shift threshold in pixels below which full shift score is given",
    )
    parser.add_argument(
        "--shift-max",
        type=float,
        default=50.0,
        help="Shift threshold in pixels above which shift score is 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    split_file = Path(args.split_file)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        sys.exit(1)

    split_df = pd.read_csv(split_file)
    for col in ["ortho_id", "label", "split"]:
        if col not in split_df.columns:
            print(
                f"Split file must have columns: ortho_id, label, split. Missing: {col}"
            )
            sys.exit(1)

    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    print(f"Dataset: {len(split_df)} total, {len(train_df)} train, {len(test_df)} test")
    print(f"Basemap: {args.basemap}, Use edges: {args.use_edges}")

    from src.features.phase_correlation import check_georeferencing_phase

    results = []
    skipped = 0
    errors = []

    for _, row in split_df.iterrows():
        oid = row["ortho_id"]
        ortho_streets = input_dir / f"{oid}_ortho_streets.png"
        streets_only = input_dir / f"{oid}_streets_only.png"
        ortho_satellite = input_dir / f"{oid}_ortho_satellite.png"
        satellite_only = input_dir / f"{oid}_satellite_only.png"

        required = []
        if args.basemap in ("both", "streets"):
            required.extend([ortho_streets, streets_only])
        if args.basemap in ("both", "satellite"):
            required.extend([ortho_satellite, satellite_only])

        if not all(p.exists() for p in required):
            print(f"Skipping {oid}: missing required PNGs")
            skipped += 1
            continue

        try:
            res = check_georeferencing_phase(
                ortho_streets_path=str(ortho_streets)
                if args.basemap in ("both", "streets")
                else None,
                streets_only_path=str(streets_only)
                if args.basemap in ("both", "streets")
                else None,
                ortho_satellite_path=str(ortho_satellite)
                if args.basemap in ("both", "satellite")
                else None,
                satellite_only_path=str(satellite_only)
                if args.basemap in ("both", "satellite")
                else None,
                basemap=args.basemap,
                use_edges=args.use_edges,
                psr_min=args.psr_min,
                psr_good=args.psr_good,
                shift_good=args.shift_good,
                shift_max=args.shift_max,
            )

            results.append(
                {
                    "ortho_id": oid,
                    "label": int(row["label"]),
                    "split": row["split"],
                    "good_probability": res["combined_good_probability"],
                }
            )
            if "satellite" in res:
                results[-1]["psr_satellite"] = res["satellite"][
                    "peak_to_sidelobe_ratio"
                ]
                results[-1]["shift_satellite"] = res["satellite"]["shift_magnitude_px"]
            if "streets" in res:
                results[-1]["psr_streets"] = res["streets"]["peak_to_sidelobe_ratio"]
                results[-1]["shift_streets"] = res["streets"]["shift_magnitude_px"]
        except Exception as e:
            print(f"Error processing {oid}: {e}")
            errors.append({"ortho_id": oid, "error": str(e)})
            skipped += 1

    if not results:
        print("No results generated!")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    print(
        f"\nProcessed: {len(results_df)} samples, Skipped: {skipped}, Errors: {len(errors)}"
    )

    train_results = results_df[results_df["split"] == "train"]
    test_results = results_df[results_df["split"] == "test"]

    y_train = train_results["label"].values
    y_test = test_results["label"].values
    prob_train = train_results["good_probability"].values
    prob_test = test_results["good_probability"].values

    def find_best_threshold(probs, labels):
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh, best_f1

    best_thresh, train_f1 = find_best_threshold(prob_train, y_train)

    def eval_at(probs, labels, thresh):
        preds = (probs >= thresh).astype(int)
        return {
            "accuracy": accuracy_score(labels, preds),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }

    test_metrics_best = eval_at(prob_test, y_test, best_thresh)
    try:
        test_metrics_best["roc_auc"] = roc_auc_score(y_test, prob_test)
    except ValueError:
        test_metrics_best["roc_auc"] = None

    print(f"\n=== Phase Correlation Benchmark Results ===")
    print(f"Method: FFT Phase Correlation")
    print(f"Basemap: {args.basemap}, Use edges: {args.use_edges}")
    print(
        f"Parameters: psr_min={args.psr_min}, psr_good={args.psr_good}, shift_good={args.shift_good}, shift_max={args.shift_max}"
    )
    print(f"\nTrain: {len(train_results)} samples")
    print(f"Test: {len(test_results)} samples")
    print(f"\nBest threshold (on train, by F1): {best_thresh:.2f} (F1={train_f1:.4f})")
    print(f"\nTest set metrics at best threshold:")
    for k, v in test_metrics_best.items():
        if k != "confusion_matrix" and isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix: {test_metrics_best['confusion_matrix']}")

    print("\nTest set metrics at fixed thresholds:")
    for th in [0.3, 0.5, 0.7]:
        m = eval_at(prob_test, y_test, th)
        print(
            f"  threshold={th}: acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
            f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}"
        )

    output_data = {
        "method": "fft_phase_correlation",
        "basemap": args.basemap,
        "use_edges": args.use_edges,
        "parameters": {
            "psr_min": args.psr_min,
            "psr_good": args.psr_good,
            "shift_good": args.shift_good,
            "shift_max": args.shift_max,
        },
        "dataset": {
            "total": len(split_df),
            "train": len(train_results),
            "test": len(test_results),
            "skipped": skipped,
        },
        "best_threshold": float(best_thresh),
        "train_f1": float(train_f1),
        "test_metrics": test_metrics_best,
        "per_sample_results": results,
        "errors": errors,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
