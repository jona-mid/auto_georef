"""
Main CLI for georeferencing quality check.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Georeferencing Quality Check System")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    collect_parser = subparsers.add_parser("collect", help="Collect training data")
    collect_parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing GeoTIFF files",
    )
    collect_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for viewports",
    )
    collect_parser.add_argument(
        "--n-images", type=int, default=100, help="Number of images to process"
    )

    features_parser = subparsers.add_parser("features", help="Extract features")
    features_parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing viewports"
    )
    features_parser.add_argument("--labels", type=str, help="Path to labels.csv file")
    features_parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to train_test_split.csv; if set, use 4-state SuperPoint+LightGlue extraction",
    )
    features_parser.add_argument(
        "--satellite-only",
        action="store_true",
        help="Use only satellite images (ortho_satellite vs satellite_only); ignored unless --split-file is set",
    )
    features_parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features.csv",
        help="Output CSV file",
    )
    features_parser.add_argument(
        "--n-negatives",
        type=int,
        default=5,
        help="Negatives per positive (only if no labels)",
    )

    train_parser = subparsers.add_parser("train", help="Train classifier")
    train_parser.add_argument(
        "--data", type=str, default="data/processed/features.csv", help="Features CSV"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "rf", "gbm", "lr"],
        help="Model type",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="data/models/classifier.pkl",
        help="Output model path",
    )
    train_parser.add_argument(
        "--threshold-only",
        action="store_true",
        help="Use good_probability as score; find best threshold on train, report test metrics (no classifier)",
    )
    train_parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Optional path to save eval metrics JSON (e.g. data/processed/eval_metrics.json)",
    )

    check_parser = subparsers.add_parser("check", help="Check georeferencing")
    check_parser.add_argument(
        "--input", type=str, required=True, help="GeoTIFF file, or directory with 4-state PNGs, or path to ortho_streets image"
    )
    check_parser.add_argument(
        "--streets-only", type=str, default=None, help="Path to streets_only image (4-state mode)"
    )
    check_parser.add_argument(
        "--ortho-satellite", type=str, default=None, help="Path to ortho_satellite image (4-state mode)"
    )
    check_parser.add_argument(
        "--satellite-only", type=str, default=None, help="Path to satellite_only image (4-state mode)"
    )
    check_parser.add_argument(
        "--model", type=str, default="data/models/classifier.pkl", help="Model path (GeoTIFF mode)"
    )
    check_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold (GeoTIFF mode)"
    )
    check_parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    if args.command == "collect":
        from src.data_collection.fetcher import DataCollector, OrthoImage

        collector = DataCollector(output_dir=args.output_dir)
        images = collector.load_from_directory(args.input_dir)

        print(f"Found {len(images)} images")

        images = images[: args.n_images]

        renderer = collector
        for i, img in enumerate(images):
            viewport = collector.extract_viewport(img)
            if viewport is not None:
                print(f"Processed {i + 1}/{len(images)}: {img.id}")

    elif args.command == "features":
        import pandas as pd
        import numpy as np
        from PIL import Image
        from src.features import extract_features

        input_dir = Path(args.input_dir)

        # 4-state path: split file + SuperPoint+LightGlue
        if getattr(args, "split_file", None):
            split_path = Path(args.split_file)
            if not split_path.exists():
                print(f"Split file not found: {split_path}")
                sys.exit(1)
            split_df = pd.read_csv(split_path)
            for col in ["ortho_id", "label", "split"]:
                if col not in split_df.columns:
                    print(f"Split file must have columns: ortho_id, label, split. Missing: {col}")
                    sys.exit(1)
            from src.features.matching import check_georeferencing

            satellite_only_mode = getattr(args, "satellite_only", False)
            if satellite_only_mode:
                print("Using satellite images only (ortho_satellite vs satellite_only)")

            rows = []
            for _, row in split_df.iterrows():
                oid = row["ortho_id"]
                ortho_streets = input_dir / f"{oid}_ortho_streets.png"
                streets_only = input_dir / f"{oid}_streets_only.png"
                ortho_satellite = input_dir / f"{oid}_ortho_satellite.png"
                satellite_only = input_dir / f"{oid}_satellite_only.png"
                if satellite_only_mode:
                    required = [ortho_satellite, satellite_only]
                else:
                    required = [ortho_streets, streets_only, ortho_satellite, satellite_only]
                if not all(p.exists() for p in required):
                    print(f"Skipping {oid}: missing required PNGs")
                    continue
                try:
                    res = check_georeferencing(
                        ortho_streets_path=str(ortho_streets) if not satellite_only_mode else None,
                        streets_only_path=str(streets_only) if not satellite_only_mode else None,
                        ortho_satellite_path=str(ortho_satellite),
                        satellite_only_path=str(satellite_only),
                        basemap="satellite" if satellite_only_mode else "both",
                    )
                    row_data = {
                        "ortho_id": oid,
                        "label": int(row["label"]),
                        "split": row["split"],
                        "good_probability": res["combined_good_probability"],
                        "inlier_ratio_satellite": res["satellite"]["inlier_ratio"],
                        "median_error_satellite": res["satellite"]["median_reprojection_error"],
                    }
                    if not satellite_only_mode:
                        row_data["inlier_ratio_streets"] = res["streets"]["inlier_ratio"]
                        row_data["median_error_streets"] = res["streets"]["median_reprojection_error"]
                    rows.append(row_data)
                    print(f"Extracted features for {oid}")
                except Exception as e:
                    print(f"Error processing {oid}: {e}")
            if rows:
                out_df = pd.DataFrame(rows)
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(args.output, index=False)
                n_good = sum(r["label"] for r in rows)
                print(f"\nExtracted 4-state features for {len(rows)} samples")
                print(f"  Label 1 (good): {n_good}, Label 0 (bad): {len(rows) - n_good}")
                print(f"  Saved to: {args.output}")
            else:
                print("No features extracted (all rows skipped or errors)")
        else:
            labels_df = None
            if args.labels:
                labels_path = Path(args.labels)
                if labels_path.exists():
                    labels_df = pd.read_csv(labels_path)
                    print(f"Loaded labels from {labels_path}: {len(labels_df)} entries")

            if labels_df is not None:
                features_list = []
                ortho_ids = []
                labels = []

                for _, row in labels_df.iterrows():
                    ortho_id = str(row["ortho_id"])
                    label = row["label"]

                    if pd.isna(label):
                        continue

                    ortho_path = input_dir / f"{ortho_id}_ortho.png"
                    basemap_path = input_dir / f"{ortho_id}_basemap.png"

                    if not ortho_path.exists() or not basemap_path.exists():
                        print(f"Skipping {ortho_id}: missing files")
                        continue

                    try:
                        ortho_img = np.array(Image.open(ortho_path))
                        basemap_img = np.array(Image.open(basemap_path))

                        features = extract_features(ortho_img, basemap_img)
                        features_list.append(features)
                        ortho_ids.append(ortho_id)
                        labels.append(int(label))

                        print(f"Extracted features for {ortho_id}")
                    except Exception as e:
                        print(f"Error processing {ortho_id}: {e}")

                if features_list:
                    feature_names = list(features_list[0].keys())
                    data = {k: [f[k] for f in features_list] for k in feature_names}
                    data["ortho_id"] = ortho_ids
                    data["label"] = labels

                    df = pd.DataFrame(data)

                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(args.output, index=False)

                    print(f"\nExtracted features for {len(df)} samples")
                    print(f"  Label 1 (good): {sum(labels)}")
                    print(f"  Label 0 (bad): {len(labels) - sum(labels)}")
                    print(f"  Saved to: {args.output}")
                else:
                    print("No features extracted")

            else:
                from src.training.dataset import DatasetBuilder

                builder = DatasetBuilder()
                dataset = builder.build_from_files(
                    args.input_dir,
                    n_negatives_per_sample=args.n_negatives,
                    output_file=args.output,
                )
                print(f"Built dataset with {len(dataset)} samples")

    elif args.command == "train":
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        df = pd.read_csv(args.data)
        if "label" not in df.columns:
            print("Features CSV must have a 'label' column")
            sys.exit(1)

        # Use split column if present; otherwise random 0.8
        if "split" in df.columns:
            train_df = df[df["split"] == "train"].copy()
            test_df = df[df["split"] == "test"].copy()
            if len(train_df) == 0 or len(test_df) == 0:
                print("Split column present but train or test set is empty")
                sys.exit(1)
            print(f"Using split column: {len(train_df)} train, {len(test_df)} test")
        else:
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            split = int(0.8 * len(df))
            train_df = df.iloc[indices[:split]].copy()
            test_df = df.iloc[indices[split:]].copy()
            print(f"Random 0.8 split: {len(train_df)} train, {len(test_df)} test")

        # Threshold-only mode: use good_probability as score, no classifier
        if getattr(args, "threshold_only", False):
            if "good_probability" not in df.columns:
                print("Threshold-only mode requires 'good_probability' column in features CSV")
                sys.exit(1)
            y_train = train_df["label"].values
            y_test = test_df["label"].values
            prob_train = train_df["good_probability"].values
            prob_test = test_df["good_probability"].values

            # Find best threshold on train (maximize F1)
            best_f1 = 0.0
            best_thresh = 0.5
            for thresh in np.arange(0.1, 0.9, 0.05):
                preds = (prob_train >= thresh).astype(int)
                f1 = f1_score(y_train, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

            # Test metrics at best threshold and at 0.3, 0.5, 0.7
            def eval_at(probs, labels, thresh):
                preds = (probs >= thresh).astype(int)
                return {
                    "accuracy": accuracy_score(labels, preds),
                    "precision": precision_score(labels, preds, zero_division=0),
                    "recall": recall_score(labels, preds, zero_division=0),
                    "f1": f1_score(labels, preds, zero_division=0),
                }

            test_metrics_best = eval_at(prob_test, y_test, best_thresh)
            test_metrics_best["roc_auc"] = roc_auc_score(y_test, prob_test)
            test_metrics_best["threshold"] = float(best_thresh)

            print(f"\nThreshold-only evaluation")
            print(f"  Best threshold (on train, by F1): {best_thresh:.2f}")
            print("  Test set metrics at best threshold:")
            for k, v in test_metrics_best.items():
                if k != "threshold" and isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
            print("  Test set metrics at fixed thresholds:")
            for th in [0.3, 0.5, 0.7]:
                m = eval_at(prob_test, y_test, th)
                print(f"    threshold={th}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")

            eval_out = getattr(args, "eval_output", None)
            if eval_out:
                import json
                Path(eval_out).parent.mkdir(parents=True, exist_ok=True)
                with open(eval_out, "w") as f:
                    json.dump(
                        {
                            "best_threshold": best_thresh,
                            "test_metrics": test_metrics_best,
                            "threshold_only": True,
                        },
                        f,
                        indent=2,
                    )
                print(f"\nEval metrics saved to: {eval_out}")
        else:
            from src.training.train import train_model

            feature_cols = [
                c for c in df.columns
                if c not in ["label", "ortho_id", "split"]
            ]
            if not feature_cols:
                print("No feature columns found (need at least one of good_probability or other metrics)")
                sys.exit(1)
            X_train = train_df[feature_cols].values.astype(np.float64)
            y_train = train_df["label"].values
            X_val = test_df[feature_cols].values.astype(np.float64)
            y_val = test_df["label"].values

            # Replace inf/nan so StandardScaler and classifier accept the data
            FINITE_CAP = 1e6
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=FINITE_CAP, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=FINITE_CAP, neginf=0.0)

            clf, metrics = train_model(
                X_train, y_train, X_val, y_val, model_type=args.model
            )

            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            clf.save(args.output)

            print(f"\nModel saved to: {args.output}")
            if "accuracy" in metrics.get("val", {}):
                print(f"Test accuracy: {metrics['val']['accuracy']:.2%}")
            eval_out = getattr(args, "eval_output", None)
            if eval_out:
                import json
                Path(eval_out).parent.mkdir(parents=True, exist_ok=True)
                to_save = {"train": metrics.get("train"), "val": metrics.get("val")}
                if "optimal_threshold" in metrics:
                    to_save["optimal_threshold"] = float(metrics["optimal_threshold"])
                with open(eval_out, "w") as f:
                    json.dump(to_save, f, indent=2)
                print(f"Eval metrics saved to: {eval_out}")

    elif args.command == "check":
        import json

        input_path = Path(args.input)
        streets_only = getattr(args, "streets_only", None)
        ortho_satellite = getattr(args, "ortho_satellite", None)
        satellite_only = getattr(args, "satellite_only", None)

        # 4-state mode: four explicit paths
        if streets_only and ortho_satellite and satellite_only:
            from src.features.matching import check_georeferencing
            res = check_georeferencing(
                str(input_path),
                streets_only,
                ortho_satellite,
                satellite_only,
            )
            results = [{"good_probability": res["combined_good_probability"]}]
        # 4-state mode: directory with 4-state PNGs (one or more orthos)
        elif input_path.is_dir():
            # Check for 4-state naming: *_ortho_streets.png etc.
            ortho_streets_files = list(input_path.glob("*_ortho_streets.png"))
            ortho_ids = []
            for f in ortho_streets_files:
                oid = f.stem.replace("_ortho_streets", "")
                streets_only_p = input_path / f"{oid}_streets_only.png"
                ortho_sat_p = input_path / f"{oid}_ortho_satellite.png"
                satellite_only_p = input_path / f"{oid}_satellite_only.png"
                if streets_only_p.exists() and ortho_sat_p.exists() and satellite_only_p.exists():
                    ortho_ids.append((oid, str(f), str(streets_only_p), str(ortho_sat_p), str(satellite_only_p)))
            if ortho_ids:
                from src.features.matching import check_georeferencing
                results = []
                for oid, p1, p2, p3, p4 in ortho_ids:
                    res = check_georeferencing(p1, p2, p3, p4)
                    results.append({"ortho_id": oid, "good_probability": res["combined_good_probability"]})
            else:
                # Fall back to GeoTIFF pipeline (directory of GeoTIFFs)
                from src.inference.pipeline import GeorefPipeline
                from src.data_collection.fetcher import DataCollector
                pipeline = GeorefPipeline(model_path=args.model, threshold=args.threshold)
                collector = DataCollector()
                images = collector.load_from_directory(str(input_path))
                results = pipeline.check_batch(images)
        elif input_path.is_file():
            # Single file: treat as GeoTIFF (existing behavior)
            from src.inference.pipeline import GeorefPipeline
            pipeline = GeorefPipeline(model_path=args.model, threshold=args.threshold)
            result = pipeline.check_from_file(str(input_path))
            results = [result]
        else:
            print(f"Input not found: {input_path}")
            sys.exit(1)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
        else:
            for r in results:
                print(json.dumps(r))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
