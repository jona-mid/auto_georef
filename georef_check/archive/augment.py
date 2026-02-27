"""
Data augmentation for generating synthetic negative examples.

Creates misaligned ortho-basemap pairs for training.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import random


class SyntheticNegativeGenerator:
    """
    Generates synthetic negative examples by applying transformations.

    Applies random offsets, rotations, and scaling to create
    misaligned image pairs.
    """

    def __init__(
        self,
        offset_range: Tuple[float, float] = (-50, 50),
        rotation_range: Tuple[float, float] = (-5, 5),
        scale_range: Tuple[float, float] = (0.95, 1.05),
    ):
        """
        Initialize generator.

        Args:
            offset_range: Range for random offset in pixels
            rotation_range: Range for random rotation in degrees
            scale_range: Range for random scale factor
        """
        self.offset_range = offset_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range

    def apply_translation(self, img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Apply translation to image."""
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    def apply_rotation(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Apply rotation around center."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    def apply_scale(self, img: np.ndarray, scale: float) -> np.ndarray:
        """Apply uniform scaling."""
        h, w = img.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h))

        # Crop or pad back to original size
        if scale > 1.0:
            # Crop center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            return scaled[start_y : start_y + h, start_x : start_x + w]
        else:
            # Pad to original size
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            result = np.zeros_like(img)
            result[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = scaled
            return result

    def generate_negative(
        self,
        img: np.ndarray,
        apply_offset: bool = True,
        apply_rotation: bool = True,
        apply_scale: bool = True,
    ) -> np.ndarray:
        """
        Generate a negative (misaligned) version of the image.

        Args:
            img: Original image
            apply_offset: Whether to apply translation
            apply_rotation: Whether to apply rotation
            apply_scale: Whether to apply scaling

        Returns:
            Transformed image
        """
        result = img.copy()

        if apply_offset:
            dx = random.randint(int(self.offset_range[0]), int(self.offset_range[1]))
            dy = random.randint(int(self.offset_range[0]), int(self.offset_range[1]))
            result = self.apply_translation(result, dx, dy)

        if apply_rotation:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            result = self.apply_rotation(result, angle)

        if apply_scale:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            result = self.apply_scale(result, scale)

        return result

    def generate_batch(self, img: np.ndarray, n_negatives: int = 5) -> List[np.ndarray]:
        """
        Generate multiple negative versions.

        Args:
            img: Original image
            n_negatives: Number of negatives to generate

        Returns:
            List of transformed images
        """
        negatives = []
        for _ in range(n_negatives):
            apply_offset = random.random() > 0.3
            apply_rotation = random.random() > 0.3
            apply_scale = random.random() > 0.5

            neg = self.generate_negative(
                img,
                apply_offset=apply_offset,
                apply_rotation=apply_rotation,
                apply_scale=apply_scale,
            )
            negatives.append(neg)

        return negatives


class HardNegativeGenerator:
    """
    Generates hard negatives by pairing ortho with wrong location.

    Instead of transforming the image, uses a different geographic location.
    """

    def __init__(self, offset_meters: float = 500):
        """
        Initialize.

        Args:
            offset_meters: Approximate offset in meters for hard negatives
        """
        self.offset_meters = offset_meters
        # Rough conversion: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
        self.deg_per_meter = 1.0 / 111000

    def get_offset_coords(self, lat: float, lon: float) -> Tuple[float, float]:
        """Get randomly offset coordinates."""
        # Random direction
        angle = random.uniform(0, 2 * np.pi)

        # Calculate offset in degrees
        offset_deg = self.offset_meters * self.deg_per_meter
        dlat = offset_deg * np.sin(angle)
        dlon = offset_deg * np.cos(angle) / np.cos(np.radians(lat))

        return lat + dlat, lon + dlon


def augment_for_training(
    ortho: np.ndarray,
    basemap: np.ndarray,
    n_positives: int = 1,
    n_negatives: int = 5,
    offset_range: Tuple[float, float] = (-100, 100),
    rotation_range: Tuple[float, float] = (-10, 10),
    scale_range: Tuple[float, float] = (0.95, 1.05),
) -> Tuple[List, List]:
    """
    Create training data by augmenting aligned pairs.

    Args:
        ortho: Ortho viewport
        basemap: Basemap viewport
        n_positives: Number of positive (aligned) samples per image
        n_negatives: Number of negative (misaligned) samples per image
        offset_range: Pixel offset range for negatives
        rotation_range: Rotation range for negatives
        scale_range: Scale range for negatives

    Returns:
        Tuple of (positive_pairs, negative_pairs)
        Each pair is ((ortho, basemap), label)
    """
    generator = SyntheticNegativeGenerator(
        offset_range=offset_range,
        rotation_range=rotation_range,
        scale_range=scale_range,
    )

    positives = []
    negatives = []

    # Positive examples (original aligned pair)
    for _ in range(n_positives):
        positives.append(((ortho.copy(), basemap.copy()), 1))

    # Negative examples (synthetic misalignment)
    for _ in range(n_negatives):
        # Transform basemap (keep ortho fixed)
        neg_basemap = generator.generate_negative(basemap)
        negatives.append(((ortho.copy(), neg_basemap), 0))

    return positives, negatives


def main():
    """Test synthetic negative generation."""
    # Create test image
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    generator = SyntheticNegativeGenerator()

    print("Generating negatives...")
    negatives = generator.generate_batch(img, 5)
    print(f"Generated {len(negatives)} negatives")
    print(f"Shape: {negatives[0].shape}")


if __name__ == "__main__":
    main()
