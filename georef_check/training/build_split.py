#!/usr/bin/env python3
"""
Build stratified train/test split from labels.csv.
Reads data/raw/dataset_custom/labels.csv and writes data/processed/train_test_split.csv
with columns: ortho_id, label, split (train | test).
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Build stratified train/test split from labels.csv")
    parser.add_argument(
        "--labels",
        type=str,
        default="data/raw/dataset_custom/labels.csv",
        help="Path to labels CSV (ortho_id, label)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train_test_split.csv",
        help="Output path for train_test_split.csv",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)
    if "ortho_id" not in df.columns or "label" not in df.columns:
        raise ValueError("labels.csv must have columns: ortho_id, label")

    # Drop rows with missing labels (if any)
    df = df.dropna(subset=["label"]).astype({"ortho_id": "int", "label": "int"})

    if len(df) == 0:
        raise ValueError("No labeled rows in labels.csv")

    # Stratified split when both classes present; otherwise random split
    stratify_arg = df["label"] if df["label"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        train_size=args.train_ratio,
        stratify=stratify_arg,
        random_state=args.seed,
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    out_df = pd.concat([train_df, test_df], ignore_index=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Split: {len(train_df)} train, {len(test_df)} test")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
