"""
Dataset class for georeferencing quality check training.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import random

from ..features import extract_features
from .augment import SyntheticNegativeGenerator


class GeorefDataset:
    """
    Dataset for georeferencing quality check.

    Handles loading pre-computed features or extracting from image pairs.
    """

    def __init__(
        self, data_dir: str = "data/processed", features_file: Optional[str] = None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing processed data
            features_file: Optional pre-computed features CSV
        """
        self.data_dir = Path(data_dir)
        self.features_file = features_file

        self.features = []
        self.labels = []
        self.metadata = []

        if features_file and Path(features_file).exists():
            self.load_from_csv(features_file)

    def load_from_csv(self, csv_path: str):
        """Load features from CSV file."""
        df = pd.read_csv(csv_path)
        self.features = df.drop(columns=["label", "ortho_id"], errors="ignore").values
        self.labels = df["label"].values
        self.metadata = df["ortho_id"].tolist() if "ortho_id" in df.columns else []

    def add_sample(self, feature_dict: Dict, label: int, metadata: str = ""):
        """Add a single sample to the dataset."""
        # Flatten feature dict to array
        feature_values = list(feature_dict.values())
        self.features.append(feature_values)
        self.labels.append(label)
        self.metadata.append(metadata)

    def save_to_csv(self, output_path: str):
        """Save dataset to CSV."""
        feature_names = (
            list(self.features[0].keys())
            if isinstance(self.features[0], dict)
            else None
        )

        if feature_names:
            df = pd.DataFrame(
                [list(f.values()) if isinstance(f, dict) else f for f in self.features],
                columns=feature_names,
            )
        else:
            df = pd.DataFrame(self.features)

        df["label"] = self.labels
        if self.metadata:
            df["ortho_id"] = self.metadata

        df.to_csv(output_path, index=False)
        print(f"Saved {len(self)} samples to {output_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        if isinstance(self.features[idx], dict):
            return np.array(list(self.features[idx].values())), self.labels[idx]
        return self.features[idx], self.labels[idx]

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if isinstance(self.features[0], dict):
            return list(self.features[0].keys())
        return [f"feature_{i}" for i in range(len(self.features[0]))]


class DatasetBuilder:
    """
    Builds dataset from image pairs.

    Extracts features from ortho-basemap pairs and creates training data.
    """

    def __init__(self, viewport_size: int = 1024, zoom_level: int = 17):
        self.viewport_size = viewport_size
        self.zoom_level = zoom_level
        self.neg_generator = SyntheticNegativeGenerator()

    def build_from_viewports(
        self,
        viewports: List[Dict],
        n_negatives_per_sample: int = 5,
        output_file: Optional[str] = None,
    ) -> GeorefDataset:
        """
        Build dataset from viewport pairs.

        Args:
            viewports: List of dicts with 'ortho' and 'basemap' arrays
            n_negatives_per_sample: Number of synthetic negatives per positive
            output_file: Optional path to save CSV

        Returns:
            GeorefDataset
        """
        dataset = GeorefDataset()

        for i, vp in enumerate(viewports):
            ortho = vp["ortho"]
            basemap = vp["basemap"]
            ortho_id = vp.get("ortho_id", f"sample_{i}")

            # Positive sample (aligned)
            pos_features = extract_features(ortho, basemap)
            dataset.add_sample(pos_features, label=1, metadata=ortho_id)

            # Negative samples (synthetic misalignment)
            negatives = self.neg_generator.generate_batch(
                basemap, n_negatives_per_sample
            )
            for j, neg_basemap in enumerate(negatives):
                neg_features = extract_features(ortho, neg_basemap)
                dataset.add_sample(
                    neg_features, label=0, metadata=f"{ortho_id}_neg_{j}"
                )

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(viewports)} samples")

        if output_file:
            dataset.save_to_csv(output_file)

        return dataset

    def build_from_files(
        self,
        image_dir: str,
        n_negatives_per_sample: int = 5,
        output_file: Optional[str] = None,
    ) -> GeorefDataset:
        """
        Build dataset from image files.

        Args:
            image_dir: Directory containing ortho and basemap PNG files
            n_negatives_per_sample: Number of negatives per positive
            output_file: Optional path to save CSV

        Returns:
            GeorefDataset
        """
        from PIL import Image

        image_dir = Path(image_dir)

        # Find all ortho files
        ortho_files = sorted(image_dir.glob("*_ortho.png"))

        viewports = []
        for ortho_file in ortho_files:
            ortho_id = ortho_file.stem.replace("_ortho", "")
            basemap_file = image_dir / f"{ortho_id}_basemap.png"

            if basemap_file.exists():
                ortho = np.array(Image.open(ortho_file))
                basemap = np.array(Image.open(basemap_file))

                viewports.append(
                    {"ortho": ortho, "basemap": basemap, "ortho_id": ortho_id}
                )

        return self.build_from_viewports(viewports, n_negatives_per_sample, output_file)


def split_dataset(
    dataset: GeorefDataset, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[GeorefDataset, GeorefDataset]:
    """
    Split dataset into train and validation.

    Args:
        dataset: Source dataset
        train_ratio: Fraction for training
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split = int(len(dataset) * train_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = GeorefDataset()
    val_ds = GeorefDataset()

    for idx in train_indices:
        train_ds.add_sample(
            dataset.features[idx], dataset.labels[idx], dataset.metadata[idx]
        )

    for idx in val_indices:
        val_ds.add_sample(
            dataset.features[idx], dataset.labels[idx], dataset.metadata[idx]
        )

    return train_ds, val_ds


def main():
    """Test dataset building."""
    # This would require actual images to test
    builder = DatasetBuilder()
    print(
        f"DatasetBuilder initialized: {builder.viewport_size}x{builder.viewport_size} at z={builder.zoom_level}"
    )


if __name__ == "__main__":
    main()
