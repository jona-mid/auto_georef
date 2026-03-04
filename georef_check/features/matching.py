"""
SuperPoint + LightGlue feature matching for georeferencing quality check.

Uses Hugging Face transformers for easy model loading.

Note: On Windows, the HF cache may warn about symlinks; we disable that warning.
For faster Hub downloads you can install: pip install hf_xet
"""

import os

# Disable symlinks warning on Windows (cache still works; uses copies instead)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchingResult:
    """Result of feature matching between two images."""

    inlier_ratio: float
    median_reprojection_error: float
    num_matches: int
    num_inliers: int
    good_probability: float


class SuperPointLightGlueMatcher:
    """SuperPoint feature extraction + LightGlue matching."""

    def __init__(
        self,
        max_num_keypoints: int = 2048,
        device: Optional[str] = None,
    ):
        """
        Initialize matcher.

        Args:
            max_num_keypoints: Maximum number of keypoints to detect
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        from transformers import AutoImageProcessor, AutoModel

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                "ETH-CVG/lightglue_superpoint",
                use_fast=True,
            )
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(
                "ETH-CVG/lightglue_superpoint"
            )
        self.model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
        self.model.to(self.device)
        self.model.eval()

    def match_images(self, img1: np.ndarray, img2: np.ndarray) -> MatchingResult:
        """
        Match features between two images and compute quality metrics.

        Args:
            img1: First image (RGB, uint8)
            img2: Second image (RGB, uint8)

        Returns:
            MatchingResult with quality metrics
        """
        # Convert to PIL
        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)

        # Extract features and match
        inputs = self.processor(images=[pil_img1, pil_img2], return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get matches and keypoints
        # Format: matches [1, 2, M], keypoints [1, 2, N, 2]
        matches = (
            outputs["matches"][0].cpu().numpy()
        )  # (2, M) - indices for img0 and img1
        keypoints = (
            outputs["keypoints"][0].cpu().numpy()
        )  # (2, N, 2) - keypoints for img0 and img1

        if matches.shape[1] < 4:
            return MatchingResult(
                inlier_ratio=0.0,
                median_reprojection_error=float("inf"),
                num_matches=matches.shape[1],
                num_inliers=0,
                good_probability=0.0,
            )

        # Get matched keypoint pairs
        kpts1 = keypoints[0, matches[0], :]  # source points from img0
        kpts2 = keypoints[1, matches[1], :]  # destination points from img1

        # Fit homography with RANSAC
        if len(kpts1) >= 4:
            H, mask = cv2.findHomography(
                kpts1, kpts2, cv2.RANSAC, ransacReprojThreshold=3.0
            )
            inliers = (
                mask.ravel().astype(bool)
                if mask is not None
                else np.zeros(len(kpts1), dtype=bool)
            )
        else:
            H = None
            inliers = np.zeros(len(kpts1), dtype=bool)

        num_inliers = np.sum(inliers)
        num_matches = matches.shape[1]
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0

        # Compute reprojection error for inliers
        if num_inliers >= 4 and H is not None:
            # Project points using homography
            kpts1_h = np.hstack([kpts1, np.ones((len(kpts1), 1))])
            projected = (H @ kpts1_h.T).T
            projected = projected[:, :2] / projected[:, 2:3]

            errors = np.linalg.norm(projected - kpts2, axis=1)
            inlier_errors = errors[inliers]
            median_error = (
                float(np.median(inlier_errors))
                if len(inlier_errors) > 0
                else float("inf")
            )
        else:
            median_error = float("inf")

        # Compute good_probability
        good_prob = self._compute_good_probability(inlier_ratio, median_error)

        return MatchingResult(
            inlier_ratio=float(inlier_ratio),
            median_reprojection_error=median_error,
            num_matches=num_matches,
            num_inliers=num_inliers,
            good_probability=good_prob,
        )

    def _compute_good_probability(
        self,
        inlier_ratio: float,
        median_error: float,
    ) -> float:
        """
        Compute probability that images are well-aligned.

        For correctly georeferenced orthos:
        - High inlier ratio (0.3-0.7 typical)
        - Low reprojection error (< 5 pixels)

        For misaligned:
        - Low inlier ratio
        - High reprojection error

        Args:
            inlier_ratio: Fraction of matches consistent with homography
            median_error: Median reprojection error in pixels

        Returns:
            Probability (0-1) that alignment is good
        """
        # Score based on inlier ratio (0-1)
        inlier_score = min(inlier_ratio / 0.4, 1.0)  # 40% inliers = full score

        # Score based on reprojection error
        if median_error < 1.0:
            error_score = 1.0
        elif median_error > 20.0:
            error_score = 0.0
        else:
            error_score = 1.0 - (median_error - 1.0) / 19.0

        # Combine scores (weighted)
        good_prob = 0.6 * inlier_score + 0.4 * error_score

        return float(np.clip(good_prob, 0.0, 1.0))


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def check_georeferencing(
    ortho_streets_path: Optional[str] = None,
    streets_only_path: Optional[str] = None,
    ortho_satellite_path: Optional[str] = None,
    satellite_only_path: Optional[str] = None,
    device: Optional[str] = None,
    basemap: str = "both",
) -> Dict:
    """
    Check georeferencing quality for an ortho by comparing ortho vs basemap.

    Args:
        ortho_streets_path: Path to ortho + streets image (required if basemap in ('both', 'streets'))
        streets_only_path: Path to streets only image (required if basemap in ('both', 'streets'))
        ortho_satellite_path: Path to ortho + satellite image (required if basemap in ('both', 'satellite'))
        satellite_only_path: Path to satellite only image (required if basemap in ('both', 'satellite'))
        device: Device for inference
        basemap: 'both' (streets + satellite, average), 'streets', or 'satellite'

    Returns:
        Dict with matching results and combined_good_probability
    """
    matcher = SuperPointLightGlueMatcher(device=device)
    streets_result = None
    satellite_result = None

    if basemap in ("both", "streets") and ortho_streets_path and streets_only_path:
        ortho_streets = load_image(ortho_streets_path)
        streets_only = load_image(streets_only_path)
        streets_result = matcher.match_images(ortho_streets, streets_only)

    if basemap in ("both", "satellite") and ortho_satellite_path and satellite_only_path:
        ortho_satellite = load_image(ortho_satellite_path)
        satellite_only = load_image(satellite_only_path)
        satellite_result = matcher.match_images(ortho_satellite, satellite_only)

    if basemap == "satellite":
        combined_prob = satellite_result.good_probability if satellite_result else 0.0
    elif basemap == "streets":
        combined_prob = streets_result.good_probability if streets_result else 0.0
    else:
        # both
        probs = [r.good_probability for r in (streets_result, satellite_result) if r is not None]
        combined_prob = sum(probs) / len(probs) if probs else 0.0

    out = {
        "combined_good_probability": combined_prob,
    }
    if streets_result is not None:
        out["streets"] = {
            "inlier_ratio": streets_result.inlier_ratio,
            "median_reprojection_error": streets_result.median_reprojection_error,
            "num_matches": streets_result.num_matches,
            "num_inliers": streets_result.num_inliers,
            "good_probability": streets_result.good_probability,
        }
    if satellite_result is not None:
        out["satellite"] = {
            "inlier_ratio": satellite_result.inlier_ratio,
            "median_reprojection_error": satellite_result.median_reprojection_error,
            "num_matches": satellite_result.num_matches,
            "num_inliers": satellite_result.num_inliers,
            "good_probability": satellite_result.good_probability,
        }
    return out


def main():
    """Test matching on sample images."""
    import os
    from pathlib import Path

    # Find test images
    data_dir = Path("georef_check/data/raw/dataset_custom")

    # Get first ortho ID
    ortho_ids = sorted(
        set(f.stem.rsplit("_", 1)[0] for f in data_dir.glob("*_ortho_streets.png"))
    )

    if not ortho_ids:
        print("No ortho images found!")
        return

    test_id = ortho_ids[0]
    print(f"Testing on ortho {test_id}...")

    ortho_streets = data_dir / f"{test_id}_ortho_streets.png"
    streets_only = data_dir / f"{test_id}_streets_only.png"
    ortho_satellite = data_dir / f"{test_id}_ortho_satellite.png"
    satellite_only = data_dir / f"{test_id}_satellite_only.png"

    if not all(
        p.exists()
        for p in [ortho_streets, streets_only, ortho_satellite, satellite_only]
    ):
        print(f"Missing images for {test_id}")
        return

    # Run matching
    results = check_georeferencing(
        str(ortho_streets),
        str(streets_only),
        str(ortho_satellite),
        str(satellite_only),
    )

    print(f"\nResults for ortho {test_id}:")
    print(
        f"  Streets:  inliers={results['streets']['num_inliers']}/{results['streets']['num_matches']} "
        f"ratio={results['streets']['inlier_ratio']:.3f}, "
        f"error={results['streets']['median_reprojection_error']:.2f}px, "
        f"prob={results['streets']['good_probability']:.3f}"
    )
    print(
        f"  Satellite: inliers={results['satellite']['num_inliers']}/{results['satellite']['num_matches']} "
        f"ratio={results['satellite']['inlier_ratio']:.3f}, "
        f"error={results['satellite']['median_reprojection_error']:.2f}px, "
        f"prob={results['satellite']['good_probability']:.3f}"
    )
    print(f"\n  Combined good_probability: {results['combined_good_probability']:.3f}")


if __name__ == "__main__":
    main()
