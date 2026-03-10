#!/usr/bin/env python3
"""
Crop ortho images to 1000x1000 square (center crop, full height).
Creates cropped versions in a new directory.
"""

import os
from pathlib import Path
from PIL import Image


def crop_images_to_square(data_dir, output_dir, size=1000):
    """Crop all PNG images to square, center crop."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = list(data_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")

    for img_path in png_files:
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            width, height = img.size

            # Calculate crop box for center crop
            if width > size:
                left = (width - size) // 2
                right = left + size
            else:
                left = 0
                right = width

            if height > size:
                top = (height - size) // 2
                bottom = top + size
            else:
                top = 0
                bottom = height

            # Crop and resize if needed
            cropped = img.crop((left, top, right, bottom))

            # Save
            output_path = output_dir / img_path.name
            cropped.save(output_path, "PNG")

    print(f"Cropped images saved to {output_dir}")


if __name__ == "__main__":
    # Create cropped versions
    crop_images_to_square(
        "../georef_check/data/raw/dataset_manual",
        "../georef_check/data/raw/dataset_manual_cropped",
        size=1000,
    )
