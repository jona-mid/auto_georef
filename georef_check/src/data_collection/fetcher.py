"""
Data collection module for georeferencing check system.

Handles fetching orthoimages from deadtrees.earth API and local storage.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import rasterio
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
import numpy as np
from PIL import Image
import io


@dataclass
class OrthoImage:
    """Represents an orthoimage with metadata."""

    id: str
    file_path: Optional[str] = None
    url: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = (
        None  # (west, south, east, north)
    )
    center: Optional[Tuple[float, float]] = None  # (lon, lat)
    crs: Optional[str] = None

    @property
    def center_lon(self) -> Optional[float]:
        return self.center[0] if self.center else None

    @property
    def center_lat(self) -> Optional[float]:
        return self.center[1] if self.center else None


class DataCollector:
    """Collects orthoimage metadata and downloads data."""

    def __init__(
        self,
        api_url: str = "https://deadtrees.earth/api/v1",
        dataset_id: int = 1,
        output_dir: str = "data/raw",
    ):
        self.api_url = api_url
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_images(self, limit: Optional[int] = None) -> List[OrthoImage]:
        """
        Fetch list of orthoimages from dataset.

        TODO: Implement actual API call when we know the endpoint structure.
        For now, returns empty list - will work with local files.
        """
        # Placeholder for API call
        # response = requests.get(f"{self.api_url}/datasets/{self.dataset_id}/images")
        # data = response.json()
        # return [OrthoImage(id=item['id'], url=item['url'], ...) for item in data]
        return []

    def load_from_directory(
        self, directory: str, pattern: str = "*.tif*"
    ) -> List[OrthoImage]:
        """
        Load orthoimages from a local directory.

        Args:
            directory: Path to directory containing GeoTIFF files
            pattern: Glob pattern to match files

        Returns:
            List of OrthoImage objects
        """
        dir_path = Path(directory)
        images = []

        for file_path in sorted(dir_path.glob(pattern)):
            try:
                with rasterio.open(file_path) as src:
                    bounds = src.bounds
                    center_lon = (bounds.left + bounds.right) / 2
                    center_lat = (bounds.bottom + bounds.top) / 2

                    ortho = OrthoImage(
                        id=file_path.stem,
                        file_path=str(file_path),
                        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                        center=(center_lon, center_lat),
                        crs=str(src.crs) if src.crs else None,
                    )
                    images.append(ortho)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return images

    def extract_viewport(
        self, ortho: OrthoImage, size: int = 1024, output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Extract a viewport from the orthoimage at its center.

        Args:
            ortho: OrthoImage object
            size: Output viewport size (square)
            output_path: Optional path to save the viewport image

        Returns:
            RGB numpy array (size x size x 3) or None on failure
        """
        if not ortho.file_path:
            return None

        try:
            with rasterio.open(ortho.file_path) as src:
                # Get center coordinates
                if ortho.center:
                    lon, lat = ortho.center
                else:
                    bounds = src.bounds
                    lon = (bounds.left + bounds.right) / 2
                    lat = (bounds.bottom + bounds.top) / 2

                # Convert to pixel coordinates
                row, col = src.index(lon, lat)

                # Calculate window
                half_size = size // 2
                window = (
                    (row - half_size, row + half_size),
                    (col - half_size, col + half_size),
                )

                # Read the data
                data = src.read(window=window)

                # Handle different band counts
                if data.shape[0] == 1:
                    # Grayscale - convert to RGB
                    rgb = np.stack([data[0]] * 3, axis=-1)
                elif data.shape[0] >= 3:
                    # RGB - take first 3 bands
                    rgb = np.transpose(data[:3], (1, 2, 0))
                else:
                    # Use first band for RGB
                    rgb = np.stack([data[0]] * 3, axis=-1)

                # Convert to 8-bit if needed
                if rgb.dtype != np.uint8:
                    rgb = (
                        (rgb / rgb.max() * 255).astype(np.uint8)
                        if rgb.max() > 0
                        else rgb.astype(np.uint8)
                    )

                # Handle window boundaries (may be smaller at edges)
                if rgb.shape[0] != size or rgb.shape[1] != size:
                    # Pad or crop to exact size
                    from PIL import Image

                    pil_img = Image.fromarray(rgb)
                    pil_img = pil_img.resize((size, size), Image.LANCZOS)
                    rgb = np.array(pil_img)

                if output_path:
                    Image.fromarray(rgb).save(output_path)

                return rgb

        except Exception as e:
            print(f"Error extracting viewport from {ortho.file_path}: {e}")
            return None


def main():
    """Test data collection with sample directory."""
    collector = DataCollector()

    # Example: load from a directory
    # images = collector.load_from_directory("/path/to/orthos")
    # print(f"Loaded {len(images)} images")

    print("DataCollector initialized. Use load_from_directory() to load local files.")


if __name__ == "__main__":
    main()
