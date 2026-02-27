#!/usr/bin/env python3
"""
Benchmark VLM georeferencing classifications against ground truth labels.
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def load_labels(labels_csv):
    """Load ground truth labels."""
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    return dict(zip(df["ortho_id"].astype(str), df["label"]))


def load_predictions(predictions_csv):
    """Load VLM predictions."""
    df = pd.read_csv(predictions_csv, encoding="utf-8-sig")
    return dict(zip(df["ortho_id"].astype(str), df["classification"]))


def map_classification(cls):
    """Map VLM classification to binary (1=correct, 0=incorrect/uncertain)."""
    if cls == "CORRECT":
        return 1
    else:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLM classifications against ground truth"
    )
    parser.add_argument(
        "--labels",
        default="../georef_check/data/raw/dataset_custom/labels.csv",
        help="Ground truth labels CSV",
    )
    parser.add_argument(
        "--predictions",
        default="georef_classification_results.csv",
        help="VLM predictions CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"Labels file not found: {args.labels}")
        return 1
    if not os.path.exists(args.predictions):
        print(f"Predictions file not found: {args.predictions}")
        return 1

    labels = load_labels(args.labels)
    predictions = load_predictions(args.predictions)

    print(f"Ground truth: {len(labels)} orthos")
    print(f"Predictions: {len(predictions)} orthos")

    common = set(labels.keys()) & set(predictions.keys())
    print(f"Common: {len(common)} orthos")

    if len(common) == 0:
        print("No common orthos between labels and predictions!")
        return 1

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    details = []

    for oid in sorted(common):
        true_label = labels[oid]
        pred_label = map_classification(predictions[oid])

        if true_label == 1 and pred_label == 1:
            true_pos += 1
            status = "TP"
        elif true_label == 0 and pred_label == 1:
            false_pos += 1
            status = "FP"
        elif true_label == 0 and pred_label == 0:
            true_neg += 1
            status = "TN"
        else:
            false_neg += 1
            status = "FN"

        details.append(
            {
                "ortho_id": oid,
                "true": true_label,
                "pred": predictions[oid],
                "pred_binary": pred_label,
                "status": status,
            }
        )

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"                  Predicted")
    print(f"                  Correct  Incorrect")
    print(f"Actual Correct    {true_pos:6d}  {false_neg:6d}")
    print(f"Actual Incorrect {false_pos:6d}  {true_neg:6d}")

    accuracy = (true_pos + true_neg) / len(common) if len(common) > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.3f} ({true_pos + true_neg}/{len(common)})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    pred_counts = pd.Series([predictions[oid] for oid in common]).value_counts()
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION")
    print("=" * 60)
    for cls, cnt in pred_counts.items():
        print(f"  {cls}: {cnt}")

    true_counts = pd.Series([labels[oid] for oid in common]).value_counts()
    print("\n" + "=" * 60)
    print("GROUND TRUTH DISTRIBUTION")
    print("=" * 60)
    for lbl, cnt in true_counts.items():
        label_name = "Correct" if lbl == 1 else "Incorrect"
        print(f"  {label_name}: {cnt}")

    return 0


if __name__ == "__main__":
    exit(main())
