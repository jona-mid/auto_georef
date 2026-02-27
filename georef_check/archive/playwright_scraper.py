"""
Playwright-based scraper for deadtrees.earth.

Captures ortho and basemap viewports using browser automation.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from playwright.sync_api import sync_playwright, Browser, Page, ElementHandle


@dataclass
class CapturedViewport:
    """Captured viewport data."""

    ortho_id: str
    ortho_path: str
    basemap_path: str
    lat: float
    lon: float
    zoom: int


class DeadtreesScraper:
    """Scraper for deadtrees.earth using Playwright."""

    def __init__(
        self,
        dataset_id: int = 1,
        output_dir: str = "data/raw",
        viewport_size: int = 1024,
        headless: bool = True,
    ):
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.viewport_size = viewport_size
        self.headless = headless

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.output_dir / f"dataset_{dataset_id}"
        self.dataset_dir.mkdir(exist_ok=True)

        self.playwright = None
        self.browser = None
        self.page = None
        self.captured = []

    def start(self):
        """Start the browser."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless, args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        self.page = self.browser.new_page(
            viewport={"width": self.viewport_size, "height": self.viewport_size}
        )
        print(f"Browser started (headless={self.headless})")

    def close(self):
        """Close the browser."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print("Browser closed")

    def save_metadata(self):
        """Save captured metadata to JSON."""
        metadata_path = self.dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump([asdict(vp) for vp in self.captured], f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    def navigate_to_dataset(self):
        """Navigate to dataset page."""
        url = f"https://deadtrees.earth/dataset/{self.dataset_id}"
        print(f"Navigating to {url}")
        self.page.goto(url, wait_until="networkidle", timeout=60000)
        time.sleep(3)  # Wait for React to render
        print("Dataset page loaded")

    def wait_for_login(self, timeout: int = 30):
        """Wait for user to log in, or check if already logged in."""
        # Check if logged in by looking for user menu
        try:
            # Look for login button vs user avatar
            login_btn = self.page.query_selector("text=Log in")
            if login_btn:
                print("Not logged in. Please log in in the browser window...")
                print("I'll wait for you to log in...")

                # Wait for login by checking for user menu periodically
                for i in range(timeout):
                    time.sleep(1)
                    try:
                        user_menu = self.page.query_selector(
                            "[data-testid='user-menu'], .user-menu, [aria-label='User menu']"
                        )
                        if user_menu:
                            print("Logged in!")
                            return True
                    except:
                        pass
                print("Timeout waiting for login")
                return False
            else:
                print("Already logged in")
                return True
        except Exception as e:
            print(f"Error checking login: {e}")
            return False

    def get_ortho_list_from_grid(self) -> List[Dict]:
        """Extract ortho list from the grid view."""
        orthos = []

        # Try to find ortho cards/elements
        # The exact selector depends on the website structure

        selectors_to_try = [
            ".ortho-card",
            "[data-ortho-id]",
            ".dataset-ortho",
            "article.ortho",
            ".MuiCard-root",
        ]

        for selector in selectors_to_try:
            try:
                elements = self.page.query_selector_all(selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector: {selector}")
                    for i, elem in enumerate(elements[:50]):  # Limit to 50
                        ortho_id = None
                        lat, lon = 0, 0

                        # Try to get ID from various attributes
                        for attr in ["data-ortho-id", "data-id", "id"]:
                            try:
                                val = elem.get_attribute(attr)
                                if val and not val.startswith("Mui"):
                                    ortho_id = val
                                    break
                            except:
                                pass

                        # Try to get coordinates
                        try:
                            # Look for lat/lon in data attributes or text
                            text = elem.text_content()
                            import re

                            coords = re.findall(r"[-]?\d+\.\d+", text)
                            if len(coords) >= 2:
                                lat, lon = float(coords[0]), float(coords[1])
                        except:
                            pass

                        if ortho_id:
                            orthos.append(
                                {
                                    "id": ortho_id,
                                    "lat": lat,
                                    "lon": lon,
                                    "element": elem,
                                }
                            )
                    break
            except Exception as e:
                print(f"Selector {selector} error: {e}")
                continue

        return orthos

    def capture_ortho_viewport(
        self, ortho_id: str, lat: float = 0, lon: float = 0
    ) -> Optional[CapturedViewport]:
        """Capture ortho and basemap viewports for a single ortho."""
        try:
            # Navigate to ortho detail view
            # URL pattern is typically: /dataset/{id}/ortho/{ortho_id}
            ortho_url = (
                f"https://deadtrees.earth/dataset/{self.dataset_id}/ortho/{ortho_id}"
            )
            self.page.goto(ortho_url, wait_until="networkidle", timeout=60000)
            time.sleep(3)  # Wait for map to load

            # Wait for map container
            map_selectors = [
                "[data-testid='map']",
                "#map",
                ".leaflet-container",
                ".map-container",
            ]
            map_elem = None
            for sel in map_selectors:
                map_elem = self.page.query_selector(sel)
                if map_elem:
                    break

            if not map_elem:
                print(f"  Could not find map element for {ortho_id}")
                return None

            # Try to get actual coordinates from the map view
            # The center of the map should give us the coordinates

            # Capture ortho view (with ortho layer visible)
            ortho_path = self.dataset_dir / f"{ortho_id}_ortho.png"
            self.page.screenshot(path=str(ortho_path), full_page=False)
            print(f"  Captured ortho: {ortho_path.name}")

            # Try to toggle to basemap (hide ortho layer)
            # This depends on the UI - try different approaches
            toggle_selectors = [
                "[data-testid='toggle-ortho']",
                "[data-testid='layer-toggle']",
                ".layer-control input",
                "text=Ortho",
                "text=Toggle layers",
            ]

            basemap_captured = False
            for sel in toggle_selectors:
                try:
                    toggle = self.page.query_selector(sel)
                    if toggle:
                        toggle.click()
                        time.sleep(2)

                        # Try to turn off ortho layer if it's a checkbox
                        try:
                            checkbox = self.page.query_selector(
                                "input[type='checkbox']"
                            )
                            if checkbox and checkbox.is_checked():
                                checkbox.uncheck()
                                time.sleep(1)
                        except:
                            pass

                        # Capture basemap
                        basemap_path = self.dataset_dir / f"{ortho_id}_basemap.png"
                        self.page.screenshot(path=str(basemap_path), full_page=False)
                        print(f"  Captured basemap: {basemap_path.name}")
                        basemap_captured = True
                        break
                except:
                    continue

            if not basemap_captured:
                # If we can't toggle, just save the same image as basemap
                # This is a fallback - the model will still work but won't have variety
                basemap_path = self.dataset_dir / f"{ortho_id}_basemap.png"
                self.page.screenshot(path=str(basemap_path), full_page=False)
                print(f"  Captured (fallback): {basemap_path.name}")

            # Try to extract coordinates from the page
            try:
                # Look for coordinate display in the UI
                coord_text = self.page.evaluate("""
                    () => {
                        // Try to find coordinate display
                        const elements = document.querySelectorAll('[data-coord], .coords, .coordinates');
                        for (const el of elements) {
                            return el.textContent;
                        }
                        // Try leaflet
                        const map = document.querySelector('.leaflet-container');
                        if (map && map._latlng) {
                            return map._latlng.lat + ',' + map._latlng.lng;
                        }
                        return null;
                    }
                """)
                if coord_text:
                    import re

                    coords = re.findall(r"[-]?\d+\.\d+", coord_text)
                    if len(coords) >= 2:
                        lat, lon = float(coords[0]), float(coords[1])
            except:
                pass

            return CapturedViewport(
                ortho_id=ortho_id,
                ortho_path=str(ortho_path),
                basemap_path=str(basemap_path),
                lat=lat,
                lon=lon,
                zoom=17,
            )

        except Exception as e:
            print(f"  Error capturing {ortho_id}: {e}")
            return None

    def scrape(self, max_capture: int = 20) -> List[CapturedViewport]:
        """Run the full scraping workflow."""
        self.start()

        try:
            # Navigate to dataset
            self.navigate_to_dataset()

            # Wait for login if needed
            self.wait_for_login(timeout=60)

            # Get ortho list from grid
            print("\nScanning for orthos in dataset...")
            orthos = self.get_ortho_list_from_grid()
            print(f"Found {len(orthos)} orthos")

            # Capture each ortho
            captured = 0
            for i, ortho in enumerate(orthos[:max_capture]):
                print(
                    f"\n[{i + 1}/{min(len(orthos), max_capture)}] Processing {ortho['id']}..."
                )

                viewport = self.capture_ortho_viewport(
                    ortho["id"], ortho.get("lat", 0), ortho.get("lon", 0)
                )

                if viewport:
                    self.captured.append(viewport)
                    captured += 1
                    print(f"  ✓ Captured {ortho['id']}")
                else:
                    print(f"  ✗ Failed to capture {ortho['id']}")

                # Rate limiting
                time.sleep(1)

            # Save metadata
            self.save_metadata()

            print(f"\n✓ Scraping complete! Captured {captured} viewports")
            return self.captured

        finally:
            self.close()


def scrape_deadtrees(
    dataset_id: int = 1,
    output_dir: str = "data/raw",
    max_capture: int = 20,
    headless: bool = False,
):
    """Main function to scrape ortho viewports."""
    scraper = DeadtreesScraper(
        dataset_id=dataset_id, output_dir=output_dir, headless=headless
    )

    return scraper.scrape(max_capture=max_capture)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape ortho viewports from")
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--max-capture", type=int, default=20)
    parser.add_argument("--headless", action="store_true", default=False)

    args = parser.parse_args()

    scrape_deadtrees(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        max_capture=args.max_capture,
        headless=args.headless,
    )
