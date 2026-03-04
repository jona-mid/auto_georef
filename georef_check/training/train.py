"""
Training script for georeferencing quality check classifier.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import argparse
from typing import Tuple, Dict, Optional

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

# Try to import xgboost
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using sklearn alternatives")


class GeorefClassifier:
    """Classifier for georeferencing quality check."""

    def __init__(self, model_type: str = "xgboost", **model_kwargs):
        """
        Initialize classifier.

        Args:
            model_type: Type of model ('xgboost', 'rf', 'gbm', 'lr')
            **model_kwargs: Additional arguments for model
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.scaler = StandardScaler()
        self.model = None

    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == "xgboost" and HAS_XGBOOST:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                **self.model_kwargs,
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, **self.model_kwargs
            )
        elif self.model_type == "gbm":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                **self.model_kwargs,
            )
        else:
            # Default to logistic regression
            return LogisticRegression(
                random_state=42, max_iter=1000, **self.model_kwargs
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and train model
        self.model = self._create_model()

        # Handle class imbalance
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

        if self.model_type == "xgboost" and HAS_XGBOOST:
            self.model.set_params(scale_pos_weight=scale_pos_weight)

        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Evaluate the model.

        Returns:
            Dict of metrics
        """
        probs = self.predict_proba(X)
        preds = self.predict(X, threshold)

        metrics = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "roc_auc": roc_auc_score(y, probs),
        }

        return metrics

    def find_optimal_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal threshold based on F1 score."""
        probs = self.predict_proba(X)

        # Find threshold that maximizes F1
        best_f1 = 0
        best_thresh = 0.5

        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return best_thresh

    def save(self, path: str):
        """Save model to file."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "model_type": self.model_type,
                },
                f,
            )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data["model_type"]


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    model_type: str = "xgboost",
    calibrate: bool = True,
) -> Tuple[GeorefClassifier, Dict]:
    """
    Train the georeferencing classifier.

    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features (optional)
        val_labels: Validation labels (optional)
        model_type: Model type
        calibrate: Whether to calibrate probabilities

    Returns:
        Tuple of (trained_model, metrics)
    """
    print(f"Training {model_type} classifier...")
    print(
        f"Training samples: {len(train_labels)} (pos: {np.sum(train_labels)}, neg: {np.sum(train_labels == 0)})"
    )

    # Train
    clf = GeorefClassifier(model_type=model_type)
    clf.fit(train_features, train_labels)

    metrics = {}

    # Training metrics
    train_metrics = clf.evaluate(train_features, train_labels)
    print("\nTraining metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    metrics["train"] = train_metrics

    # Validation metrics
    if val_features is not None:
        val_metrics = clf.evaluate(val_features, val_labels)
        print("\nValidation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Find optimal threshold
        optimal_thresh = clf.find_optimal_threshold(val_features, val_labels)
        print(f"\nOptimal threshold: {optimal_thresh:.2f}")
        metrics["optimal_threshold"] = optimal_thresh

        # Evaluate at optimal threshold
        val_preds = clf.predict(val_features, optimal_thresh)
        print("\nClassification report at optimal threshold:")
        print(
            classification_report(val_labels, val_preds, target_names=["bad", "good"])
        )

        metrics["val"] = val_metrics

    # Cross-validation if no validation set
    if val_features is None:
        print("\nRunning 5-fold cross-validation...")
        X_all = np.vstack([train_features])
        y_all = np.concatenate([train_labels])

        cv_clf = GeorefClassifier(model_type=model_type)
        cv_scores = cross_val_score(cv_clf, X_all, y_all, cv=5, scoring="f1")
        print(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        metrics["cv_f1"] = cv_scores.mean()

    return clf, metrics


def load_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from CSV."""
    df = pd.read_csv(csv_path)

    # Get feature columns (all except label and id)
    feature_cols = [c for c in df.columns if c not in ["label", "ortho_id"]]

    X = df[feature_cols].values
    y = df["label"].values

    return X, y


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train georeferencing classifier")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/features.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "rf", "gbm", "lr"],
        help="Model type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/classifier.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Calibrate probabilities"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    X, y = load_data(args.data)
    print(f"Loaded {len(y)} samples")

    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(y))
    split = int(0.8 * len(y))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Train
    clf, metrics = train_model(
        X_train, y_train, X_val, y_val, model_type=args.model, calibrate=args.calibrate
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.output)

    # Print final threshold
    if "optimal_threshold" in metrics:
        print(f"\nUse threshold: {metrics['optimal_threshold']:.2f}")


if __name__ == "__main__":
    main()
