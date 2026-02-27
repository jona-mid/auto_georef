"""
Web scraper to capture ortho viewports from deadtrees.earth.

Uses Selenium to navigate the website and capture screenshots.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


@dataclass
class OrthoViewport:
    """Captured viewport data."""

    ortho_id: str
    ortho_image_path: str
    basemap_image_path: str
    center_lat: float
    center_lon: float
    zoom: int


class DeadtreesScraper:
    """
    Scrapes ortho viewports from deadtrees.earth.

    Requires Chrome/Chromium installed.
    """

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

        self.driver = None
        self.captured_count = 0

    def setup_driver(self):
        """Set up Selenium WebDriver."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not installed. Run: uv pip install selenium")

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--window-size={self.viewport_size},{self.viewport_size}")

        # Set viewport size for screenshots
        options.add_argument(f"--screenshot={self.viewport_size}x{self.viewport_size}")

        self.driver = webdriver.Chrome(options=options)
        return self.driver

    def close(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()

    def login(self, email: str, password: str):
        """Log in to deadtrees.earth."""
        self.driver.get("https://deadtrees.earth/login")
        time.sleep(2)

        # Find login form and fill credentials
        email_input = self.driver.find_element(By.NAME, "email")
        password_input = self.driver.find_element(By.NAME, "password")

        email_input.send_keys(email)
        password_input.send_keys(password)

        # Click login button
        login_button = self.driver.find_element(
            By.CSS_SELECTOR, "button[type='submit']"
        )
        login_button.click()

        # Wait for redirect
        time.sleep(3)
        print(f"Logged in as {email}")

    def navigate_to_dataset(self):
        """Navigate to the dataset page."""
        url = f"https://deadtrees.earth/dataset/{self.dataset_id}"
        self.driver.get(url)
        time.sleep(5)  # Wait for page to load
        print(f"Navigated to dataset {self.dataset_id}")

    def get_ortho_list(self) -> List[Dict]:
        """
        Get list of orthos in the dataset.

        This needs to be adapted based on the actual page structure.
        """
        # This is a placeholder - the actual implementation depends on the page structure
        orthos = []

        # Try to find ortho elements
        try:
            # Look for ortho cards/elements
            elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-ortho-id]")
            for elem in elements:
                ortho_id = elem.get_attribute("data-ortho-id")
                if ortho_id:
                    orthos.append({"id": ortho_id})
        except Exception as e:
            print(f"Error finding orthos: {e}")

        return orthos

    def capture_ortho_viewport(
        self, ortho_id: str, lat: float, lon: float
    ) -> Optional[OrthoViewport]:
        """
        Capture viewport for a single ortho.

        This navigates to the ortho view and captures screenshots.
        """
        try:
            # Navigate to ortho view
            # URL pattern: https://deadtrees.earth/dataset/{id}/ortho/{ortho_id}
            url = f"https://deadtrees.earth/dataset/{self.dataset_id}/ortho/{ortho_id}"
            self.driver.get(url)
            time.sleep(3)  # Wait for map to load

            # Take screenshot of ortho
            ortho_path = self.dataset_dir / f"{ortho_id}_ortho.png"
            self.driver.save_screenshot(str(ortho_path))

            # Switch to basemap view (toggle ortho layer off)
            # This depends on the actual UI - may need adjustment
            # Try clicking a layer toggle button
            try:
                layer_toggle = self.driver.find_element(
                    By.CSS_SELECTOR, "[data-testid='layer-toggle']"
                )
                layer_toggle.click()
                time.sleep(1)
            except:
                pass  # May not exist

            # Take basemap screenshot
            basemap_path = self.dataset_dir / f"{ortho_id}_basemap.png"
            self.driver.save_screenshot(str(basemap_path))

            self.captured_count += 1

            return OrthoViewport(
                ortho_id=ortho_id,
                ortho_image_path=str(ortho_path),
                basemap_image_path=str(basemap_path),
                center_lat=lat,
                center_lon=lon,
                zoom=17,
            )

        except Exception as e:
            print(f"Error capturing {ortho_id}: {e}")
            return None

    def capture_from_grid(self, max_capture: int = 100) -> List[OrthoViewport]:
        """
        Capture orthos from the dataset grid view.

        Navigates to dataset page and captures visible orthos.
        """
        self.navigate_to_dataset()

        viewports = []
        captured = 0

        # Scroll and find ortho elements
        while captured < max_capture:
            # Find ortho cards in the grid
            try:
                cards = self.driver.find_elements(
                    By.CSS_SELECTOR, ".ortho-card, [data-ortho-id]"
                )
            except:
                cards = []

            if not cards:
                break

            for card in cards:
                if captured >= max_capture:
                    break

                try:
                    ortho_id = card.get_attribute(
                        "data-ortho-id"
                    ) or card.get_attribute("data-id")
                    if not ortho_id:
                        continue

                    # Get coordinates from the card
                    lat = float(card.get_attribute("data-lat") or 0)
                    lon = float(card.get_attribute("data-lon") or 0)

                    # Click to open detail view
                    card.click()
                    time.sleep(3)

                    # Capture viewport
                    viewport = self.capture_ortho_viewport(ortho_id, lat, lon)
                    if viewport:
                        viewports.append(viewport)
                        captured += 1
                        print(f"Captured {captured}/{max_capture}: {ortho_id}")

                    # Go back
                    self.driver.back()
                    time.sleep(2)

                except Exception as e:
                    print(f"Error processing card: {e}")
                    continue

            # Try to load more (scroll down)
            try:
                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight)"
                )
                time.sleep(2)
            except:
                break

        return viewports

    def save_metadata(self, viewports: List[OrthoViewport]):
        """Save metadata about captured viewports."""
        metadata = []
        for vp in viewports:
            metadata.append(
                {
                    "ortho_id": vp.ortho_id,
                    "ortho_image": vp.ortho_image_path,
                    "basemap_image": vp.basemap_image_path,
                    "lat": vp.center_lat,
                    "lon": vp.center_lon,
                    "zoom": vp.zoom,
                }
            )

        meta_path = self.dataset_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {meta_path}")


def scrape_deadtrees(
    dataset_id: int = 1,
    output_dir: str = "data/raw",
    max_capture: int = 100,
    email: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Main function to scrape ortho viewports from deadtrees.earth.
    """
    scraper = DeadtreesScraper(
        dataset_id=dataset_id, output_dir=output_dir, headless=True
    )

    try:
        scraper.setup_driver()

        # Login if credentials provided
        if email and password:
            scraper.login(email, password)

        # Capture orthos
        viewports = scraper.capture_from_grid(max_capture=max_capture)

        # Save metadata
        scraper.save_metadata(viewports)

        print(f"\nCaptured {len(viewports)} ortho viewports")
        return viewports

    finally:
        scraper.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape ortho viewports from deadtrees.earth"
    )
    parser.add_argument("--dataset-id", type=int, default=1, help="Dataset ID")
    parser.add_argument(
        "--output-dir", type=str, default="data/raw", help="Output directory"
    )
    parser.add_argument(
        "--max-capture", type=int, default=100, help="Max orthos to capture"
    )
    parser.add_argument("--email", type=str, help="Email for login")
    parser.add_argument("--password", type=str, help="Password for login")

    args = parser.parse_args()

    scrape_deadtrees(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        max_capture=args.max_capture,
        email=args.email,
        password=args.password,
    )
