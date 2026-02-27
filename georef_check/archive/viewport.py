"""
Viewport rendering module.

Combines orthoimage extraction and basemap tile fetching.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from .fetcher import OrthoImage, DataCollector
from .tile_fetcher import TileFetcher


class ViewportRenderer:
    """Renders paired viewports of orthoimage and basemap."""

    def __init__(
        self,
        viewport_size: int = 1024,
        zoom_level: int = 17,
        basemap_style: str = "esri",
    ):
        self.viewport_size = viewport_size
        self.zoom_level = zoom_level
        self.basemap_style = basemap_style

        self.tile_fetcher = TileFetcher(tile_server=basemap_style)
        self.data_collector = DataCollector()

    def render_ortho_viewport(self, ortho: OrthoImage) -> Optional[np.ndarray]:
        """Render viewport from orthoimage."""
        return self.data_collector.extract_viewport(ortho, size=self.viewport_size)

    def render_basemap_viewport(self, lat: float, lon: float) -> Optional[np.ndarray]:
        """Render viewport from basemap tiles."""
        return self.tile_fetcher.fetch_viewport(
            lat, lon, self.zoom_level, self.viewport_size
        )

    def render_paired_viewports(
        self, ortho: OrthoImage
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Render paired ortho and basemap viewports.

        Returns:
            Tuple of (ortho_viewport, basemap_viewport) or None on failure
        """
        if not ortho.center:
            print(f"No center coordinates for ortho {ortho.id}")
            return None

        lat, lon = ortho.center_lat, ortho.center_lon

        ortho_viewport = self.render_ortho_viewport(ortho)
        if ortho_viewport is None:
            return None

        basemap_viewport = self.render_basemap_viewport(lat, lon)
        if basemap_viewport is None:
            return None

        return ortho_viewport, basemap_viewport

    def render_batch(
        self, orthos: List[OrthoImage], output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Render viewports for a batch of orthoimages.

        Args:
            orthos: List of OrthoImage objects
            output_dir: Optional directory to save images

        Returns:
            List of dicts with 'ortho_id', 'ortho', 'basemap' arrays
        """
        results = []

        for ortho in orthos:
            result = {"ortho_id": ortho.id}

            paired = self.render_paired_viewports(ortho)
            if paired is not None:
                result["ortho"] = paired[0]
                result["basemap"] = paired[1]

                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    from PIL import Image

                    Image.fromarray(paired[0]).save(
                        output_path / f"{ortho.id}_ortho.png"
                    )
                    Image.fromarray(paired[1]).save(
                        output_path / f"{ortho.id}_basemap.png"
                    )

                results.append(result)

        return results


def main():
    """Test viewport rendering."""
    # This would require actual ortho files to test
    renderer = ViewportRenderer()
    print(
        f"ViewportRenderer initialized: {renderer.viewport_size}x{renderer.viewport_size} at z={renderer.zoom_level}"
    )


if __name__ == "__main__":
    main()
