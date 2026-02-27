#!/usr/bin/env python3
"""
Georeferencing Classifier using VLM
Classifies orthoimagery as CORRECT, INCORRECT, or UNCERTAIN based on alignment with basemap.

Usage:
    # Sequential (default)
    python georef_classifier.py --data-dir ./data --api-key KEY

    # Concurrent (4 workers)
    python georef_classifier.py --data-dir ./data --api-key KEY --workers 4
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

import config

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("georef_classifier.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an expert in geospatial imagery analysis.

I will show you 4 images from the same geographic location:
1. ortho_streets.png - Drone imagery overlaid on streets basemap
2. streets_only.png - Streets basemap without drone imagery
3. ortho_satellite.png - Drone imagery overlaid on satellite imagery
4. satellite_only.png - Satellite imagery without drone imagery

Task: Determine if the drone imagery is correctly georeferenced (aligned with the basemap).

Look for:
- Roads, buildings, and features in the drone image should align with the underlying basemap
- If misaligned, the drone image will appear shifted or not match basemap features

Respond with ONLY one of:
CORRECT: <brief 1-sentence explanation>
INCORRECT: <brief 1-sentence explanation>
UNCERTAIN: <brief 1-sentence explanation>"""


class GlobalRateLimiter:
    """Global rate limiter for request start times."""

    def __init__(self, rps: float | None):
        self._min_interval = (1.0 / float(rps)) if rps and rps > 0 else None
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_interval is None:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = (
                max(self._next_allowed, time.monotonic()) + self._min_interval
            )


_thread_local = threading.local()


def _get_session() -> requests.Session:
    """One keep-alive Session per worker thread."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


def load_labels(csv_path):
    """Load labels CSV and return set of ortho_ids with labels."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        logger.info(f"Loaded {len(df)} labeled orthos")
        return df
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        raise


def get_ortho_ids(data_dir):
    """Get list of ortho IDs that have all 4 required images."""
    files = os.listdir(data_dir)
    ortho_ids = set()
    for f in files:
        if f.endswith("_ortho_streets.png"):
            ortho_id = f.replace("_ortho_streets.png", "")
            ortho_ids.add(ortho_id)
    return sorted(ortho_ids)


def load_images_to_base64(data_dir, ortho_id):
    """Load all 4 images for an ortho and return as base64 list."""
    suffixes = [
        "_ortho_streets.png",
        "_streets_only.png",
        "_ortho_satellite.png",
        "_satellite_only.png",
    ]
    images_b64 = []

    for suffix in suffixes:
        path = os.path.join(data_dir, ortho_id + suffix)
        if not os.path.exists(path):
            return None
        try:
            with Image.open(path) as img:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                images_b64.append(img_base64)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    return images_b64


def parse_response(response_text):
    """Parse LLM response to extract classification and explanation."""
    if not response_text:
        return ("UNCERTAIN", "No response received", response_text)

    response_upper = response_text.upper()

    classification = "UNCERTAIN"
    if "CORRECT:" in response_upper:
        classification = "CORRECT"
    elif "INCORRECT:" in response_upper:
        classification = "INCORRECT"
    elif "UNCERTAIN:" in response_upper:
        classification = "UNCERTAIN"

    explanation = response_text.strip()
    if len(explanation) > 500:
        explanation = explanation[:497] + "..."

    return (classification, explanation, response_text)


def classify_image(
    images_b64: list[str],
    api_key: str,
    model: str = config.DEFAULT_MODEL,
    max_retries: int = 3,
    timeout_s: float = 120.0,
    rate_limiter: GlobalRateLimiter | None = None,
    session: requests.Session | None = None,
) -> tuple[str, str, str]:
    """Send 4 images to OpenRouter API for classification."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/georef-check-vlm",
        "X-Title": "Georeferencing Classifier",
    }

    content = [{"type": "text", "text": PROMPT_TEMPLATE}]
    for img_b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 500,
        "max_completion_tokens": 500,
    }

    http_session = session or requests.Session()

    for attempt in range(max_retries):
        try:
            if rate_limiter is not None:
                rate_limiter.wait()

            response = http_session.post(
                config.OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )

            if response.status_code == 200:
                result = response.json()
                content_text = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                return parse_response(content_text)
            elif response.status_code == 429:
                wait_time = (2**attempt) * 2
                logger.warning(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return ("ERROR", f"API error: {response.status_code}", response.text)

        except Exception as e:
            logger.error(f"Request error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return ("ERROR", f"Request failed: {str(e)}", "")

    return ("ERROR", "Max retries exceeded", "")


def process_one(
    *,
    ortho_id: str,
    data_dir: str,
    api_key: str,
    model: str,
    timeout_s: float,
    max_retries: int,
    rate_limiter: GlobalRateLimiter | None = None,
) -> dict:
    """Worker task: load images, call LLM, return result."""
    try:
        images_b64 = load_images_to_base64(data_dir, ortho_id)
        if images_b64 is None:
            return {
                "ortho_id": ortho_id,
                "classification": "ERROR",
                "explanation": "Missing images",
            }

        session = _get_session()
        classification, explanation, _raw = classify_image(
            images_b64=images_b64,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            timeout_s=timeout_s,
            rate_limiter=rate_limiter,
            session=session,
        )

        return {
            "ortho_id": ortho_id,
            "classification": classification,
            "explanation": explanation,
        }
    except Exception as e:
        return {"ortho_id": ortho_id, "classification": "ERROR", "explanation": str(e)}


def save_progress(results, output_csv):
    """Save current results to CSV file."""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Progress saved to {output_csv} ({len(results)} entries)")
    except Exception as e:
        logger.error(f"Error saving progress: {e}")


def load_existing_results(output_csv):
    """Load existing results from CSV if it exists."""
    if os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv, encoding="utf-8-sig")
            existing = set(df["ortho_id"].astype(str).tolist())
            logger.info(f"Loaded {len(existing)} existing results from {output_csv}")
            return existing, df.to_dict("records")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
    return set(), []


def run_sequential(args, remaining, results, api_key):
    """Run classification sequentially."""
    total = len(remaining)
    processed = 0

    for i, ortho_id in enumerate(remaining, 1):
        logger.info(f"[{i}/{total}] Processing ortho {ortho_id}...")

        images_b64 = load_images_to_base64(args.data_dir, ortho_id)
        if images_b64 is None:
            logger.warning(f"Skipping {ortho_id} - missing images")
            continue

        classification, explanation, _raw = classify_image(
            images_b64, api_key, args.model, max_retries=args.max_retries
        )

        result = {
            "ortho_id": ortho_id,
            "classification": classification,
            "explanation": explanation,
        }
        results.append(result)
        processed += 1

        logger.info(f"  -> {classification}: {explanation[:100]}")

        if processed % args.save_interval == 0:
            save_progress(results, args.output_csv)

        if i < total:
            time.sleep(args.delay)

    return processed


def run_concurrent(args, remaining, results, api_key):
    """Run classification concurrently with thread pool."""
    total = len(remaining)
    rate_limiter = GlobalRateLimiter(args.rps if args.rps and args.rps > 0 else None)
    save_lock = threading.Lock()

    completed = 0
    start = time.time()

    def _maybe_save() -> None:
        with save_lock:
            save_progress(results, args.output_csv)

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = {
            ex.submit(
                process_one,
                ortho_id=ortho_id,
                data_dir=args.data_dir,
                api_key=api_key,
                model=args.model,
                timeout_s=float(args.timeout),
                max_retries=int(args.max_retries),
                rate_limiter=rate_limiter,
            ): ortho_id
            for ortho_id in remaining
        }

        for fut in as_completed(futures):
            ortho_id = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "ortho_id": ortho_id,
                    "classification": "ERROR",
                    "explanation": str(e),
                }

            results.append(row)
            completed += 1

            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start
                rps_eff = completed / elapsed if elapsed > 0 else 0.0
                logger.info(
                    f"[{completed}/{total}] done (throughput={rps_eff:.2f} imgs/s)"
                )

            if completed % int(args.save_interval) == 0:
                _maybe_save()

    _maybe_save()
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Classify ortho georeferencing using OpenRouter VLM"
    )
    parser.add_argument(
        "--data-dir", default=config.DATA_DIR, help="Directory containing ortho images"
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="Path to labels CSV (optional, for reference)",
    )
    parser.add_argument(
        "--output-csv", default=config.DEFAULT_OUTPUT_CSV, help="Output CSV file path"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        help=f"Model to use (default: {config.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0, sequential only)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N images (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of orthos to process (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers (default: 1, sequential; use >1 for concurrent)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=0.0,
        help="Optional global requests-per-second cap (0 disables, concurrent only)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per image (default: 3)"
    )
    parser.add_argument(
        "--rerun-status",
        type=str,
        default=None,
        help="Rerun orthos with specific status (comma-separated: ERROR,UNCERTAIN)",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("No API key provided. Use --api-key or set OPENROUTER_API_KEY")
        return 1

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1

    ortho_ids = get_ortho_ids(args.data_dir)
    logger.info(f"Found {len(ortho_ids)} orthos with complete image sets")

    if args.limit:
        ortho_ids = ortho_ids[: args.limit]
        logger.info(f"Limited to {args.limit} orthos for testing")

    processed_ids, results = load_existing_results(args.output_csv)

    rerun_statuses = None
    if args.rerun_status:
        rerun_statuses = set(s.strip().upper() for s in args.rerun_status.split(","))
        logger.info(f"Rerun requested for statuses: {rerun_statuses}")

        if processed_ids:
            status_filtered = {
                str(r["ortho_id"]): r
                for r in results
                if r.get("classification", "").upper() in rerun_statuses
            }
            rerun_ids = set(status_filtered.keys())
            processed_ids = processed_ids - rerun_ids
            results = [r for r in results if str(r["ortho_id"]) not in rerun_ids]
            logger.info(f"Re-adding {len(status_filtered)} orthos for processing")

    remaining = [oid for oid in ortho_ids if str(oid) not in processed_ids]
    total = len(remaining)
    logger.info(f"Remaining to process: {total}")

    if total == 0:
        logger.info("Nothing to do.")
        return 0

    logger.info(f"Running with {args.workers} worker(s)")

    if args.workers > 1:
        processed = run_concurrent(args, remaining, results, api_key)
    else:
        processed = run_sequential(args, remaining, results, api_key)

    save_progress(results, args.output_csv)

    logger.info(f"\nClassification complete!")
    logger.info(f"Processed: {processed} orthos")
    logger.info(f"Results saved to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    exit(main())
