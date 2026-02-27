"""
End-to-end inference pipeline for georeferencing quality check.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import json

from ..features import extract_features
from ..data_collection.viewport import ViewportRenderer
from ..data_collection.fetcher import OrthoImage
from ..training.train import GeorefClassifier


class GeorefPipeline:
    """
    End-to-end pipeline for georeferencing quality check.

    Processes orthoimages and returns georeferencing quality scores.
    """

    def __init__(
        self,
        model_path: str = "data/models/classifier.pkl",
        viewport_size: int = 1024,
        zoom_level: int = 17,
        threshold: float = 0.5,
    ):
        """
        Initialize pipeline.

        Args:
            model_path: Path to trained model
            viewport_size: Viewport size in pixels
            zoom_level: Zoom level for basemap
            threshold: Classification threshold
        """
        self.viewport_size = viewport_size
        self.zoom_level = zoom_level
        self.threshold = threshold

        # Load model
        self.classifier = GeorefClassifier()
        self.classifier.load(model_path)

        # Initialize renderer
        self.renderer = ViewportRenderer(
            viewport_size=viewport_size, zoom_level=zoom_level
        )

    def check_single(self, ortho: OrthoImage) -> Dict:
        """
        Check georeferencing quality for a single orthoimage.

        Args:
            ortho: OrthoImage object

        Returns:
            Dict with results
        """
        # Render viewports
        result = self.renderer.render_paired_viewports(ortho)

        if result is None:
            return {
                "ortho_id": ortho.id,
                "success": False,
                "error": "Failed to render viewports",
            }

        ortho_viewport, basemap_viewport = result

        # Extract features
        features = extract_features(ortho_viewport, basemap_viewport)

        # Predict
        prob = self.classifier.predict_proba(np.array([list(features.values())]))[0]

        return {
            "ortho_id": ortho.id,
            "success": True,
            "geo_ok_probability": float(prob),
            "features": {k: float(v) for k, v in features.items()},
            "recommended_action": "accept" if prob >= self.threshold else "review",
        }

    def check_from_file(self, tiff_path: str) -> Dict:
        """
        Check georeferencing quality from a GeoTIFF file.

        Args:
            tiff_path: Path to GeoTIFF file

        Returns:
            Dict with results
        """
        # Create OrthoImage from file
        ortho = OrthoImage(id=Path(tiff_path).stem, file_path=tiff_path)

        return self.check_single(ortho)

    def check_batch(self, orthos: List[OrthoImage]) -> List[Dict]:
        """
        Check georeferencing quality for multiple orthoimages.

        Args:
            orthos: List of OrthoImage objects

        Returns:
            List of result dicts
        """
        results = []

        for ortho in orthos:
            result = self.check_single(ortho)
            results.append(result)

        return results

    def get_queue_ranking(self, results: List[Dict]) -> List[Dict]:
        """
        Get results sorted by priority for audit queue.

        Items with lower probability (more likely to be geo-bad) come first.

        Args:
            results: List of result dicts

        Returns:
            Sorted list (low confidence first for review)
        """
        # Filter successful checks
        successful = [r for r in results if r.get("success", False)]

        # Sort by probability (low first = more likely to be bad)
        sorted_results = sorted(
            successful, key=lambda x: x.get("geo_ok_probability", 1.0)
        )

        return sorted_results


def check_georeferencing(
    tiff_path: str,
    model_path: str = "data/models/classifier.pkl",
    threshold: float = 0.5,
) -> Dict:
    """
    Quick function to check a single GeoTIFF.

    Args:
        tiff_path: Path to GeoTIFF file
        model_path: Path to trained model
        threshold: Classification threshold

    Returns:
        Dict with results
    """
    pipeline = GeorefPipeline(model_path=model_path, threshold=threshold)

    return pipeline.check_from_file(tiff_path)


def main():
    """Test the pipeline."""
    print("GeorefPipeline initialized")
    print("Usage: pipeline.check_from_file('/path/to/image.tif')")


if __name__ == "__main__":
    main()
