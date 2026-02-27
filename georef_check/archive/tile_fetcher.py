"""
Tile fetcher for getting basemap tiles.

Fetches tiles from various tile servers (Mapbox, Google, etc.)
"""

import math
import io
from typing import Tuple, Optional
import requests
import numpy as np
from PIL import Image


class TileFetcher:
    """Fetches map tiles from tile servers."""

    def __init__(self, tile_server: str = "osm"):
        """
        Initialize tile fetcher.

        Args:
            tile_server: Tile server type ('osm', 'google', 'mapbox', 'esri')
        """
        self.tile_server = tile_server

        # Tile server URLs (these are examples - may need API keys)
        self.tile_urls = {
            "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        }

    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile x/y."""
        lat_rad = math.radians(lat_deg)
        n = 2.0**zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def num2deg(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile x/y to lat/lon of top-left corner."""
        n = 2.0**zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def get_tile_url(self, x: int, y: int, z: int) -> str:
        """Get URL for a specific tile."""
        if self.tile_server not in self.tile_urls:
            raise ValueError(f"Unknown tile server: {self.tile_server}")
        return self.tile_urls[self.tile_server].format(x=x, y=y, z=z)

    def fetch_tile(self, x: int, y: int, z: int) -> Optional[Image.Image]:
        """Fetch a single tile."""
        url = self.get_tile_url(x, y, z)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"Error fetching tile {x},{y},{z}: {e}")
        return None

    def fetch_viewport(
        self, lat: float, lon: float, zoom: int, size: int
    ) -> Optional[np.ndarray]:
        """
        Fetch a viewport (square region) at the given coordinates.

        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
            size: Viewport size in pixels

        Returns:
            RGB numpy array (size x size x 3) or None on failure
        """
        # Get center tile
        center_x, center_y = self.deg2num(lat, lon, zoom)

        # Calculate how many tiles we need
        tile_size = 256
        tiles_needed = int(math.ceil(size / tile_size))

        # Create blank canvas
        canvas = Image.new("RGB", (tiles_needed * tile_size, tiles_needed * tile_size))

        # Fetch tiles
        for dy in range(tiles_needed):
            for dx in range(tiles_needed):
                x = center_x + dx - tiles_needed // 2
                y = center_y + dy - tiles_needed // 2

                tile = self.fetch_tile(x, y, zoom)
                if tile:
                    canvas.paste(tile, (dx * tile_size, dy * tile_size))

        # Crop/resize to exact size
        if canvas.size != (size, size):
            canvas = canvas.resize((size, size), Image.LANCZOS)

        return np.array(canvas)

    def fetch_multiple_styles(
        self, lat: float, lon: float, zoom: int, size: int, styles: list = None
    ) -> dict:
        """
        Fetch viewports in multiple basemap styles.

        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
            size: Viewport size
            styles: List of style names (default: ['esri', 'osm'])

        Returns:
            Dict mapping style name to RGB array
        """
        if styles is None:
            styles = ["esri", "osm"]

        result = {}
        for style in styles:
            old_server = self.tile_server
            self.tile_server = style
            viewport = self.fetch_viewport(lat, lon, zoom, size)
            if viewport is not None:
                result[style] = viewport
            self.tile_server = old_server

        return result


def main():
    """Test tile fetching."""
    fetcher = TileFetcher(tile_server="esri")

    # Test with sample coordinates (San Francisco)
    lat, lon = 37.7749, -122.4194
    zoom = 17
    size = 512

    print(f"Fetching viewport at {lat}, {lon} (z={zoom}, {size}x{size})")
    viewport = fetcher.fetch_viewport(lat, lon, zoom, size)

    if viewport is not None:
        print(f"Got viewport: {viewport.shape}")
        # Save for inspection
        Image.fromarray(viewport).save("test_viewport.png")
        print("Saved to test_viewport.png")
    else:
        print("Failed to fetch viewport")


if __name__ == "__main__":
    main()
