"""
Deadtrees API client for downloading orthoimages.

Handles authentication and data download from deadtrees.earth.
"""

import os
import requests
from typing import List, Dict, Optional
from pathlib import Path
import time


class DeadtreesClient:
    """Client for deadtrees.earth API."""

    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        self.email = email
        self.password = password
        self.session = requests.Session()
        self.base_url = "https://deadtrees.earth"
        self.token = None

    def login(self) -> bool:
        """Log in to deadtrees.earth."""
        if not self.email or not self.password:
            print("No credentials provided. Some endpoints may be inaccessible.")
            return False

        # Get login page to extract CSRF token
        resp = self.session.get(f"{self.base_url}/login")

        # Try to login (simplified - may need adjustment based on actual form)
        login_data = {"email": self.email, "password": self.password}

        try:
            resp = self.session.post(
                f"{self.base_url}/api/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                self.token = resp.json().get("token")
                print(f"Logged in as {self.email}")
                return True
        except:
            pass

        print("Login failed. Will try unauthenticated requests.")
        return False

    def get_dataset_info(self, dataset_id: int) -> Optional[Dict]:
        """Get dataset information."""
        # Try public endpoint first
        resp = self.session.get(f"{self.base_url}/api/v1/datasets/{dataset_id}")
        if resp.status_code == 200:
            try:
                return resp.json()
            except:
                pass
        return None

    def get_ortho_list(self, dataset_id: int) -> List[Dict]:
        """Get list of orthos in dataset."""
        # This may need adjustment based on actual API structure
        orthos = []

        # Try paginated endpoint
        page = 1
        while True:
            try:
                resp = self.session.get(
                    f"{self.base_url}/api/v1/datasets/{dataset_id}/images",
                    params={"page": page, "limit": 100},
                )
                if resp.status_code != 200:
                    break

                data = resp.json()
                items = data.get("items", []) or data.get("data", []) or []

                if not items:
                    break

                orthos.extend(items)
                page += 1

                if page > 10:  # Limit pages
                    break

            except Exception as e:
                break

        return orthos

    def download_ortho(self, ortho_id: str, output_path: str) -> bool:
        """Download a single ortho image."""
        # Try different download endpoints
        endpoints = [
            f"/cogs/v1/datasets/*/images/{ortho_id}",
            f"/downloads/v1/images/{ortho_id}",
            f"/api/v1/images/{ortho_id}/download",
        ]

        for ep in endpoints:
            try:
                resp = self.session.get(f"{self.base_url}{ep}", timeout=60)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    return True
            except:
                continue

        return False

    def get_thumbnail(self, ortho_id: str, output_path: str, size: int = 512) -> bool:
        """Download thumbnail."""
        url = f"{self.base_url}/thumbnails/v1/images/{ortho_id}?size={size}"
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return True
        except:
            pass
        return False


def download_dataset(
    dataset_id: int = 1,
    output_dir: str = "data/raw",
    email: Optional[str] = None,
    password: Optional[str] = None,
    max_images: int = 100,
):
    """
    Download orthoimages from a dataset.

    Args:
        dataset_id: Dataset ID to download
        output_dir: Output directory
        email: Email for authentication
        password: Password for authentication
        max_images: Maximum images to download
    """
    client = DeadtreesClient(email, password)
    client.login()

    output_path = Path(output_dir) / f"dataset_{dataset_id}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Get ortho list
    print(f"Fetching ortho list for dataset {dataset_id}...")
    orthos = client.get_ortho_list(dataset_id)
    print(f"Found {len(orthos)} orthos")

    # Download orthos
    downloaded = 0
    for ortho in orthos[:max_images]:
        ortho_id = ortho.get("id") or ortho.get("ortho_id")
        if not ortho_id:
            continue

        print(f"Downloading {ortho_id}...")

        # Try thumbnail first (smaller, faster)
        thumb_path = output_path / f"{ortho_id}_thumb.jpg"
        if client.get_thumbnail(ortho_id, str(thumb_path)):
            downloaded += 1
            print(f"  ✓ Downloaded thumbnail: {thumb_path}")
            continue

        # Try full ortho
        ortho_path = output_path / f"{ortho_id}.tif"
        if client.download_ortho(ortho_id, str(ortho_path)):
            downloaded += 1
            print(f"  ✓ Downloaded ortho: {ortho_path}")
            continue

        print(f"  ✗ Failed to download {ortho_id}")

        if downloaded >= max_images:
            break

    print(f"\nDownloaded {downloaded} images to {output_path}")
    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download orthos from deadtrees.earth")
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--email", type=str, help="Email for login")
    parser.add_argument("--password", type=str, help="Password for login")
    parser.add_argument("--max-images", type=int, default=100)

    args = parser.parse_args()

    download_dataset(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        email=args.email,
        password=args.password,
        max_images=args.max_images,
    )
