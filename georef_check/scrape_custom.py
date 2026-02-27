#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, Page

# Configuration for missing tile detection
MISSING_TILE_THRESHOLD = 0.50  # 50% identical pixels indicates missing tiles
MAX_RETRIES = 2
RETRY_WAIT = 10  # seconds between retries


def hide_ui_overlays(page: Page):
    """Hide UI elements like headers and footers."""
    js_code = """
    () => {
        const selectors = ['header', 'footer', '.ant-layout-header', '.ant-layout-sider', '.ant-float-btn-group'];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                if(el) el.style.display = 'none';
            });
        });
    }
    """
    page.evaluate(js_code)


def get_map_bounds(page: Page):
    """Get the bounding box of the map to crop screenshots."""
    js_code = """
    () => {
        const viewport = document.querySelector('.ol-viewport');
        if (viewport) {
            const rect = viewport.getBoundingClientRect();
            return { found: true, x: rect.x, y: rect.y, width: rect.width, height: rect.height };
        }
        return { found: false };
    }
    """
    return page.evaluate(js_code)


def setup_base_layers(page: Page):
    """Turn off Forest Cover, Deadwood, and Area of Interest."""
    for label_text in ["Forest Cover", "Deadwood", "Area of Interest"]:
        try:
            page.locator(
                f"label:has-text('{label_text}') >> input[type='checkbox']"
            ).uncheck(force=True)
        except:
            pass


def toggle_drone(page: Page, state: bool):
    """Turn drone imagery on or off."""
    try:
        locator = page.locator(
            "label:has-text('Drone Imagery') >> input[type='checkbox']"
        )
        if state:
            locator.check(force=True)
        else:
            locator.uncheck(force=True)
    except:
        pass


def set_basemap(page: Page, basemap_type: str):
    """Set basemap to 'Streets' or 'Imagery'."""
    try:
        page.locator(".ant-segmented-item").filter(has_text=basemap_type).click()
    except:
        pass


def trigger_map_tile_load_js(page: Page, zoom_steps: int = 4) -> None:
    """
    Trigger tile loading via JavaScript by forcing a view change.

    Some imagery layers won't request tiles until a zoom or pan occurs. This function attempts:
    - If OpenLayers map is accessible: zoom out several steps and back in, preserving center.
    - Otherwise: dispatch wheel events (zoom out/in) on the map viewport.
    """
    js = """
    () => {
        const viewport = document.querySelector('.ol-viewport');
        if (!viewport) return false;
        const map = viewport.closest('[class*="map"]')?.__olMap || viewport.__olMap
            || (typeof window.getMap === 'function' && window.getMap())
            || window.map || window.olMap;
        if (!map || typeof map.getView !== 'function') return "wheel";
        const view = map.getView();
        const center = view.getCenter();
        if (!center) return false;
        const z = (typeof view.getZoom === 'function') ? view.getZoom() : null;
        if (z === null || typeof view.animate !== 'function') return "wheel";
        const target = Math.max(2, z - 4);
        // zoom out then back in
        view.animate({ zoom: target, duration: 200 }, { zoom: z, duration: 200 });
        return "ol";
    }
    """
    try:
        mode = page.evaluate(js)
        # Wheel fallback: do multiple steps out and back.
        if mode == "wheel":
            # scroll down => zoom out; scroll up => zoom in
            for _ in range(max(1, int(zoom_steps))):
                page.mouse.wheel(0, 800)
                time.sleep(0.2)
            for _ in range(max(1, int(zoom_steps))):
                page.mouse.wheel(0, -800)
                time.sleep(0.2)
        time.sleep(0.8)
    except Exception:
        pass


def trigger_map_tile_load(page: Page, bounds: dict, pan_pixels: int = 40) -> None:
    """
    Trigger a small pan on the map so the imagery layer loads tiles.
    Uses (1) JS view nudge if possible, (2) mouse drag as fallback.
    """
    if not bounds.get("found"):
        return
    trigger_map_tile_load_js(page)
    cx = bounds["x"] + bounds["width"] / 2
    cy = bounds["y"] + bounds["height"] / 2
    try:
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + pan_pixels, cy)
        page.mouse.up()
        time.sleep(0.3)
        page.mouse.move(cx + pan_pixels, cy)
        page.mouse.down()
        page.mouse.move(cx, cy)
        page.mouse.up()
        time.sleep(0.5)
    except Exception:
        pass


def has_missing_tiles(image_path: Path, threshold=MISSING_TILE_THRESHOLD) -> bool:
    """Check if image contains missing tiles by detecting uniform color regions."""
    img = Image.open(image_path)
    arr = np.array(img)
    pixels = arr.reshape(-1, arr.shape[-1])
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    max_count = counts.max()
    total = pixels.shape[0]
    return (max_count / total) > threshold


def wait_for_satellite_ready(
    page: Page,
    bounds: dict,
    output_path: Path,
    max_wait_seconds: float = 20.0,
    check_interval: float = 3.0,
    initial_wait: float = 2.0,
) -> bool:
    """
    Take satellite screenshot, retrying until tiles are loaded or max_wait is reached.
    Returns True if a good screenshot was saved, False if still missing tiles after max_wait.
    """
    time.sleep(initial_wait)
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        # Keep nudging the view so tiles continue to be requested.
        trigger_map_tile_load(page, bounds)
        take_cropped_screenshot(page, bounds, output_path)
        if not has_missing_tiles(output_path):
            return True
        time.sleep(check_interval)
    return not has_missing_tiles(output_path)


def take_cropped_screenshot(page: Page, map_bounds: dict, output_path: Path):
    """Takes a full screenshot and crops it to the map bounds."""
    temp_path = output_path.with_name(f"temp_{output_path.name}")
    page.screenshot(path=str(temp_path), full_page=False)

    with Image.open(temp_path) as img:
        x = int(map_bounds.get("x", 0))
        y = int(map_bounds.get("y", 0))
        w = int(map_bounds.get("width", 1200))
        h = int(map_bounds.get("height", 900))
        cropped = img.crop((x, y, x + w, y + h))
        cropped.save(str(output_path))

    temp_path.unlink()


SIGN_IN_URL = "https://deadtrees.earth/sign-in"


def login(page: Page, email: str, password: str, timeout: int = 45000) -> bool:
    """Log in at sign-in page. Returns True if login appears successful."""
    try:
        page.goto(SIGN_IN_URL, wait_until="domcontentloaded", timeout=timeout)
        page.wait_for_load_state("networkidle", timeout=15000)
        time.sleep(3)
        email_selectors = [
            'input[name="email"]',
            'input[type="email"]',
            'input[id="email"]',
            'input[placeholder*="mail" i]',
            'input[placeholder*="Email"]',
        ]
        password_selectors = [
            'input[name="password"]',
            'input[type="password"]',
            'input[id="password"]',
            'input[placeholder*="assword" i]',
        ]
        email_loc = None
        for sel in email_selectors:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="visible", timeout=5000)
                email_loc = loc
                break
            except Exception:
                continue
        if not email_loc:
            raise RuntimeError("Could not find email input on sign-in page")
        email_loc.fill(email)
        password_loc = None
        for sel in password_selectors:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="visible", timeout=3000)
                password_loc = loc
                break
            except Exception:
                continue
        if not password_loc:
            raise RuntimeError("Could not find password input on sign-in page")
        password_loc.fill(password)
        try:
            page.locator('button[type="submit"]').first.click(timeout=5000)
        except Exception:
            page.get_by_role(
                "button", name=re.compile(r"log in|sign in|login|anmelden", re.I)
            ).first.click(timeout=5000)
        # Wait for redirect (e.g. to /profile) instead of fixed sleep
        try:
            page.wait_for_url(
                lambda url: "sign-in" not in url and "/login" not in url,
                timeout=15000,
            )
        except Exception:
            pass
        if "/sign-in" in page.url or "/login" in page.url:
            return False
        print("Logged in successfully.")
        return True
    except Exception as e:
        print(f"Login failed: {e}")
        return False


def wait_for_manual_login(page: Page, wait_seconds: int = 60) -> None:
    """Open sign-in page and wait for the user to log in manually in the browser."""
    page.goto(SIGN_IN_URL, wait_until="domcontentloaded", timeout=30000)
    print(
        f"\n>>> Please log in manually in the browser window. Waiting {wait_seconds} seconds... <<<\n"
    )
    time.sleep(wait_seconds)


def capture_one_ortho(
    page: Page,
    ortho_id: int,
    url: str,
    output_dir: Path,
) -> Tuple[bool, Optional[str]]:
    """
    Navigate to url, run capture sequence for one ortho. Returns (success, url_for_metadata).
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
    except Exception:
        return False, None

    time.sleep(3)

    try:
        map_el = page.locator(".ol-viewport")
        if not map_el.is_visible(timeout=2000):
            return False, None
    except Exception:
        return False, None

    bounds = get_map_bounds(page)
    if not bounds.get("found"):
        return False, None

    try:
        page.locator("button:has-text('Accept')").first.click(timeout=1000)
    except Exception:
        pass

    hide_ui_overlays(page)
    setup_base_layers(page)
    time.sleep(1)

    path_ortho_streets = output_dir / f"{ortho_id}_ortho_streets.png"
    path_ortho_satellite = output_dir / f"{ortho_id}_ortho_satellite.png"
    path_streets = output_dir / f"{ortho_id}_streets_only.png"
    path_satellite = output_dir / f"{ortho_id}_satellite_only.png"

    try:
        toggle_drone(page, True)
        set_basemap(page, "Streets")
        time.sleep(3)
        take_cropped_screenshot(page, bounds, path_ortho_streets)

        set_basemap(page, "Imagery")
        trigger_map_tile_load(page, bounds)
        if not wait_for_satellite_ready(
            page, bounds, path_ortho_satellite, initial_wait=5.0
        ):
            for p in [path_ortho_streets, path_ortho_satellite]:
                if p.exists():
                    p.unlink()
            return False, None

        toggle_drone(page, False)
        set_basemap(page, "Streets")
        time.sleep(3)
        take_cropped_screenshot(page, bounds, path_streets)

        set_basemap(page, "Imagery")
        trigger_map_tile_load(page, bounds)
        if not wait_for_satellite_ready(
            page,
            bounds,
            path_satellite,
            max_wait_seconds=15.0,
            initial_wait=5.0,
        ):
            take_cropped_screenshot(page, bounds, path_satellite)
        if has_missing_tiles(path_satellite):
            for p in [
                path_ortho_streets,
                path_ortho_satellite,
                path_streets,
                path_satellite,
            ]:
                if p.exists():
                    p.unlink()
            return False, None

        return True, url
    except Exception:
        return False, None


def scrape_labeled(
    output_dir: Path,
    bad_ids: Optional[List[int]],
    good_count: Optional[int],
    min_id: int,
    max_id: int,
    email: Optional[str] = None,
    password: Optional[str] = None,
    login_wait: int = 60,
    headless: bool = False,
) -> None:
    """Two-phase scrape: bad from dataset-audit (login first), good from dataset."""
    BASE_AUDIT = "https://deadtrees.earth/dataset-audit"
    BASE_DATASET = "https://deadtrees.earth/dataset"

    captured: List[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless, args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.evaluate(
            "() => { window.moveTo(0, 0); window.resizeTo(screen.width, screen.height); }"
        )

        # Phase bad: dataset-audit (login first: programmatic or manual)
        if bad_ids:
            if email and password:
                if not login(page, email, password):
                    print("Programmatic login failed. Falling back to manual login...")
                    wait_for_manual_login(page, wait_seconds=login_wait)
            else:
                wait_for_manual_login(page, wait_seconds=login_wait)

            for i, ortho_id in enumerate(bad_ids):
                url = f"{BASE_AUDIT}/{ortho_id}"
                print(f"[Bad {i + 1}/{len(bad_ids)}] Attempting ID {ortho_id}...")
                ok, meta_url = capture_one_ortho(page, ortho_id, url, output_dir)
                if ok:
                    captured.append(
                        {"ortho_id": ortho_id, "url": meta_url or url, "label": 0}
                    )
                    print(f"  Captured ID {ortho_id} (label=0)")
                else:
                    print(f"  Skipped ID {ortho_id}")

        # Phase good: dataset (public)
        if good_count is not None and good_count > 0:
            bad_set = set(bad_ids) if bad_ids else set()
            candidates = [x for x in range(min_id, max_id + 1) if x not in bad_set]
            random.shuffle(candidates)
            attempts = 0
            for ortho_id in candidates:
                if len([c for c in captured if c.get("label") == 1]) >= good_count:
                    break
                url = f"{BASE_DATASET}/{ortho_id}"
                attempts += 1
                print(f"[Good] Attempting ID {ortho_id} (attempt {attempts})...")
                ok, meta_url = capture_one_ortho(page, ortho_id, url, output_dir)
                if ok:
                    captured.append(
                        {"ortho_id": ortho_id, "url": meta_url or url, "label": 1}
                    )
                    print(f"  Captured ID {ortho_id} (label=1)")

        browser.close()

    # Write labels.csv and metadata.json
    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w") as f:
        f.write("ortho_id,label\n")
        for item in captured:
            f.write(f"{item['ortho_id']},{item['label']}\n")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(captured, f, indent=2)

    print(f"\nDone! Saved {len(captured)} sets of images to {output_dir}")


def scrape_custom(min_id, max_id, count, headless=False):
    output_dir = Path("data/raw/dataset_custom")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = list(range(min_id, max_id + 1))
    random.shuffle(all_ids)

    captured = []

    print(f"Starting custom scrape for {count} records...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless, args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # Maximize
        page.evaluate(
            "() => { window.moveTo(0, 0); window.resizeTo(screen.width, screen.height); }"
        )

        attempts = 0
        for ortho_id in all_ids:
            if len(captured) >= count:
                break

            attempts += 1
            print(
                f"[{len(captured)}/{count}] Attempting ID {ortho_id} (Attempt {attempts})..."
            )

            url = f"https://deadtrees.earth/dataset/{ortho_id}"
            try:
                page.goto(url, wait_until="networkidle", timeout=20000)
            except:
                print(f"  Failed to load page. Skipping.")
                continue

            time.sleep(3)  # Wait for map to settle

            try:
                map_el = page.locator(".ol-viewport")
                if not map_el.is_visible(timeout=2000):
                    print(f"  No map found. Skipping.")
                    continue
            except:
                print(f"  No map found. Skipping.")
                continue

            bounds = get_map_bounds(page)
            if not bounds.get("found"):
                print(f"  Could not get map bounds. Skipping.")
                continue

            try:
                page.locator("button:has-text('Accept')").first.click(timeout=1000)
            except:
                pass

            hide_ui_overlays(page)
            setup_base_layers(page)
            time.sleep(1)

            path_ortho_streets = output_dir / f"{ortho_id}_ortho_streets.png"
            path_ortho_satellite = output_dir / f"{ortho_id}_ortho_satellite.png"
            path_streets = output_dir / f"{ortho_id}_streets_only.png"
            path_satellite = output_dir / f"{ortho_id}_satellite_only.png"

            try:
                # State 1: Ortho + Streets
                toggle_drone(page, True)
                set_basemap(page, "Streets")
                time.sleep(3)
                take_cropped_screenshot(page, bounds, path_ortho_streets)

                # State 2: Ortho + Satellite
                set_basemap(page, "Imagery")
                trigger_map_tile_load(page, bounds)
                if not wait_for_satellite_ready(
                    page, bounds, path_ortho_satellite, initial_wait=5.0
                ):
                    for p in [path_ortho_streets, path_ortho_satellite]:
                        if p.exists():
                            p.unlink()
                    print(
                        f"  Discarding dataset {ortho_id} due to missing tiles in ortho_satellite"
                    )
                    continue

                # State 3: Streets Only
                toggle_drone(page, False)
                set_basemap(page, "Streets")
                time.sleep(3)
                take_cropped_screenshot(page, bounds, path_streets)

                # State 4: Satellite Only (retry until tiles load, same as State 2)
                set_basemap(page, "Imagery")
                trigger_map_tile_load(page, bounds)
                if not wait_for_satellite_ready(
                    page,
                    bounds,
                    path_satellite,
                    max_wait_seconds=15.0,
                    initial_wait=5.0,
                ):
                    take_cropped_screenshot(page, bounds, path_satellite)
                if has_missing_tiles(path_satellite):
                    # Cleanup all 4 images
                    for p in [
                        path_ortho_streets,
                        path_ortho_satellite,
                        path_streets,
                        path_satellite,
                    ]:
                        if p.exists():
                            p.unlink()
                    print(
                        f"  Discarding dataset {ortho_id} due to missing tiles in satellite_only"
                    )
                    continue

                print(f"  Successfully captured 4 states for ID {ortho_id}")
                captured.append({"ortho_id": ortho_id, "url": url})
            except Exception as e:
                print(f"  Error during capture sequence: {e}")
                continue

        browser.close()

    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w") as f:
        f.write("ortho_id,label\n")
        for item in captured:
            f.write(f"{item['ortho_id']},\n")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(captured, f, indent=2)

    print(f"\nDone! Saved {len(captured)} sets of images to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-id", type=int, default=1)
    parser.add_argument("--max-id", type=int, default=8000)
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Legacy: number of random IDs to scrape (used when --bad-ids/--good-count not set)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--bad-ids",
        type=str,
        default=None,
        help="Comma-separated list of bad ortho IDs; scraped from dataset-audit (label=0)",
    )
    parser.add_argument(
        "--good-count",
        type=int,
        default=None,
        help="Number of good orthos to scrape from dataset (label=1), excluding bad-ids",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email for programmatic login (or set DEADTREES_EMAIL)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password for programmatic login (or set DEADTREES_PASSWORD)",
    )
    parser.add_argument(
        "--login-wait",
        type=int,
        default=60,
        help="Seconds to wait for manual login if programmatic login not used or fails (default 60)",
    )
    args = parser.parse_args()

    email = args.email or os.environ.get("DEADTREES_EMAIL")
    password = args.password or os.environ.get("DEADTREES_PASSWORD")

    if args.bad_ids is not None or args.good_count is not None:
        bad_ids = None
        if args.bad_ids:
            bad_ids = [int(x.strip()) for x in args.bad_ids.split(",") if x.strip()]
        output_dir = Path("data/raw/dataset_custom")
        output_dir.mkdir(parents=True, exist_ok=True)
        scrape_labeled(
            output_dir=output_dir,
            bad_ids=bad_ids,
            good_count=args.good_count,
            min_id=args.min_id,
            max_id=args.max_id,
            email=email,
            password=password,
            login_wait=args.login_wait,
            headless=args.headless,
        )
    else:
        scrape_custom(args.min_id, args.max_id, args.count, args.headless)
