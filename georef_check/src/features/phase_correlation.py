"""
FFT-based Phase Correlation for Georeferencing Quality Check.

This module implements robust FFT-based shift detection for checking whether
orthoimages are properly georeferenced against basemap tiles.

Dependencies: numpy, scipy, cv2 (no torch, no heavy ML dependencies)
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path


@dataclass
class PhaseCorrelationResult:
    """Result of FFT-based phase correlation between two images."""

    dx: float
    dy: float
    shift_magnitude_px: float
    peak_value: float
    peak_to_sidelobe_ratio: float
    good_probability: float
    is_reliable: bool


def apply_hann_window(img: np.ndarray) -> np.ndarray:
    """
    Apply a 2D Hann window to reduce spectral leakage at image borders.

    Args:
        img: Input image (2D grayscale)

    Returns:
        Windowed image
    """
    H, W = img.shape
    hann_1d = np.hanning(H)
    hann_2d = np.outer(hann_1d, np.hanning(W))
    return img * hann_2d


def _subpixel_peak(correlation_surface: np.ndarray, peak_y: int, peak_x: int) -> tuple:
    """
    Refine peak location using parabolic interpolation for sub-pixel accuracy.

    Args:
        correlation_surface: 2D correlation surface
        peak_y: Integer y-coordinate of peak
        peak_x: Integer x-coordinate of peak

    Returns:
        (dy, dx) refined shift estimates
    """
    H, W = correlation_surface.shape

    if 1 <= peak_y < H - 1 and 1 <= peak_x < W - 1:
        dy_numer = (
            correlation_surface[peak_y + 1, peak_x]
            - correlation_surface[peak_y - 1, peak_x]
        )
        dy_denom = 2 * (
            correlation_surface[peak_y + 1, peak_x]
            + correlation_surface[peak_y - 1, peak_x]
            - 2 * correlation_surface[peak_y, peak_x]
        )
        dy = peak_y + dy_numer / dy_denom if dy_denom != 0 else peak_y

        dx_numer = (
            correlation_surface[peak_y, peak_x + 1]
            - correlation_surface[peak_y, peak_x - 1]
        )
        dx_denom = 2 * (
            correlation_surface[peak_y, peak_x + 1]
            + correlation_surface[peak_y, peak_x - 1]
            - 2 * correlation_surface[peak_y, peak_x]
        )
        dx = peak_x + dx_numer / dx_denom if dx_denom != 0 else peak_x
    else:
        dy, dx = float(peak_y), float(peak_x)

    return dy, dx


def fft_phase_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    window: bool = True,
    eps: float = 1e-10,
    **kwargs,
) -> PhaseCorrelationResult:
    """
    Compute FFT-based phase correlation between two images.

    Estimates translation (dx, dy) with sub-pixel accuracy and computes
    confidence metrics (peak value, PSR) for determining alignment quality.

    Args:
        img1: First image (RGB or grayscale)
        img2: Second image (RGB or grayscale)
        window: Whether to apply Hann window (reduces spectral leakage)
        eps: Small constant to avoid division by zero

    Returns:
        PhaseCorrelationResult with shift estimates and confidence metrics
    """
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1.copy()

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2.copy()

    gray1 = gray1.astype(np.float64)
    gray2 = gray2.astype(np.float64)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    H, W = gray1.shape

    if window:
        gray1 = apply_hann_window(gray1)
        gray2 = apply_hann_window(gray2)

    F1 = np.fft.fft2(gray1)
    F2 = np.fft.fft2(gray2)

    cross_power = F1 * np.conj(F2)
    magnitude = np.abs(cross_power) + eps
    G = cross_power / magnitude

    correlation_surface = np.real(np.fft.ifft2(G))

    correlation_surface = np.fft.fftshift(correlation_surface)

    cy, cx = H // 2, W // 2
    peak_y, peak_x = np.unravel_index(
        np.argmax(correlation_surface), correlation_surface.shape
    )

    dy_float, dx_float = _subpixel_peak(correlation_surface, peak_y, peak_x)
    dy = dy_float - cy
    dx = dx_float - cx

    if dx > W / 2:
        dx -= W
    elif dx < -W / 2:
        dx += W

    if dy > H / 2:
        dy -= H
    elif dy < -H / 2:
        dy += H

    dx = float(dx)
    dy = float(dy)

    peak_value = float(correlation_surface[peak_y, peak_x])
    peak_value_normalized = (peak_value - correlation_surface.min()) / (
        correlation_surface.max() - correlation_surface.min() + eps
    )
    peak_value_normalized = min(peak_value_normalized, 1.0)

    mask = np.ones_like(correlation_surface, dtype=bool)
    mask[peak_y, peak_x] = False
    if peak_y > 0 and peak_y < H - 1 and peak_x > 0 and peak_x < W - 1:
        mask[peak_y - 1 : peak_y + 2, peak_x - 1 : peak_x + 2] = False

    sidelobe_mean = correlation_surface[mask].mean() if mask.any() else 0.0
    psr = (
        peak_value / (abs(sidelobe_mean) + eps) if sidelobe_mean != 0 else float("inf")
    )

    shift_magnitude = np.sqrt(dx**2 + dy**2)

    return PhaseCorrelationResult(
        dx=dx,
        dy=dy,
        shift_magnitude_px=shift_magnitude,
        peak_value=peak_value_normalized,
        peak_to_sidelobe_ratio=psr,
        good_probability=0.0,
        is_reliable=False,
    )



def _compute_good_probability(
    shift_magnitude_px: float,
    psr: float,
    psr_min: float = 3.0,
    psr_good: float = 8.0,
    shift_good: float = 5.0,
    shift_max: float = 50.0,
) -> tuple:
    """
    Compute good_probability and is_reliable based on shift and PSR.

    Args:
        shift_magnitude_px: Computed shift magnitude in pixels
        psr: Peak-to-sidelobe ratio
        psr_min: Below this, result is unreliable
        psr_good: Above this, full PSR score
        shift_good: Below this, full shift score
        shift_max: Above this, shift score is 0

    Returns:
        (good_probability, is_reliable)
    """
    shift_score = max(0.0, 1.0 - shift_magnitude_px / shift_max)
    psr_score = min(max((psr - psr_min) / (psr_good - psr_min), 0.0), 1.0)

    good_probability = 0.5 * shift_score + 0.5 * psr_score
    good_probability = float(np.clip(good_probability, 0.0, 1.0))

    is_reliable = psr > psr_min and shift_magnitude_px < shift_max

    return good_probability, is_reliable


def fft_phase_correlation_edges(
    img1: np.ndarray,
    img2: np.ndarray,
    window: bool = True,
    canny_low: int = 50,
    canny_high: int = 150,
    **kwargs,
) -> PhaseCorrelationResult:
    """
    Run phase correlation on Canny edge images.

    This often works better for ortho-vs-basemap comparison because the
    color/brightness difference between drone imagery and map tiles can be
    large, but edges (roads, building outlines) are shared.

    Args:
        img1: First image (RGB or grayscale)
        img2: Second image (RGB or grayscale)
        window: Whether to apply Hann window
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        **kwargs: Additional arguments passed to fft_phase_correlation

    Returns:
        PhaseCorrelationResult
    """
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1.copy()

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2.copy()

    edges1 = cv2.Canny(gray1, canny_low, canny_high)
    edges2 = cv2.Canny(gray2, canny_low, canny_high)

    result = fft_phase_correlation(edges1, edges2, window=window, **kwargs)

    return result


def check_georeferencing_phase(
    ortho_streets_path: Optional[str] = None,
    streets_only_path: Optional[str] = None,
    ortho_satellite_path: Optional[str] = None,
    satellite_only_path: Optional[str] = None,
    basemap: str = "both",
    use_edges: bool = False,
    psr_min: float = 3.0,
    psr_good: float = 8.0,
    shift_good: float = 5.0,
    shift_max: float = 50.0,
) -> Dict:
    """
    Check georeferencing quality using FFT phase correlation.

    Mirrors the signature of check_georeferencing in matching.py.

    Args:
        ortho_streets_path: Path to ortho + streets image
        streets_only_path: Path to streets only image
        ortho_satellite_path: Path to ortho + satellite image
        satellite_only_path: Path to satellite only image
        basemap: 'both', 'streets', or 'satellite'
        use_edges: If True, run on Canny edge images
        psr_min: Below this, result is unreliable
        psr_good: Above this, full PSR score
        shift_good: Below this, full shift score
        shift_max: Above this, shift score is 0

    Returns:
        Dict with combined_good_probability and per-basemap results
    """
    kwargs = {
        "psr_min": psr_min,
        "psr_good": psr_good,
        "shift_good": shift_good,
        "shift_max": shift_max,
    }

    phase_func = fft_phase_correlation_edges if use_edges else fft_phase_correlation

    streets_result = None
    satellite_result = None

    if basemap in ("both", "streets") and ortho_streets_path and streets_only_path:
        try:
            ortho_streets = cv2.imread(ortho_streets_path)
            streets_only = cv2.imread(streets_only_path)
            if ortho_streets is None or streets_only is None:
                raise ValueError("Failed to load images")
            ortho_streets = cv2.cvtColor(ortho_streets, cv2.COLOR_BGR2RGB)
            streets_only = cv2.cvtColor(streets_only, cv2.COLOR_BGR2RGB)

            result = phase_func(ortho_streets, streets_only, **kwargs)
            prob, reliable = _compute_good_probability(
                result.shift_magnitude_px,
                result.peak_to_sidelobe_ratio,
                psr_min=psr_min,
                psr_good=psr_good,
                shift_good=shift_good,
                shift_max=shift_max,
            )
            result.good_probability = prob
            result.is_reliable = reliable

            streets_result = {
                "dx": result.dx,
                "dy": result.dy,
                "shift_magnitude_px": result.shift_magnitude_px,
                "peak_value": result.peak_value,
                "peak_to_sidelobe_ratio": result.peak_to_sidelobe_ratio,
                "good_probability": result.good_probability,
                "is_reliable": result.is_reliable,
            }
        except Exception as e:
            print(f"Warning: Error processing streets pair: {e}")

    if (
        basemap in ("both", "satellite")
        and ortho_satellite_path
        and satellite_only_path
    ):
        try:
            ortho_satellite = cv2.imread(ortho_satellite_path)
            satellite_only = cv2.imread(satellite_only_path)
            if ortho_satellite is None or satellite_only is None:
                raise ValueError("Failed to load images")
            ortho_satellite = cv2.cvtColor(ortho_satellite, cv2.COLOR_BGR2RGB)
            satellite_only = cv2.cvtColor(satellite_only, cv2.COLOR_BGR2RGB)

            result = phase_func(ortho_satellite, satellite_only, **kwargs)
            prob, reliable = _compute_good_probability(
                result.shift_magnitude_px,
                result.peak_to_sidelobe_ratio,
                psr_min=psr_min,
                psr_good=psr_good,
                shift_good=shift_good,
                shift_max=shift_max,
            )
            result.good_probability = prob
            result.is_reliable = reliable

            satellite_result = {
                "dx": result.dx,
                "dy": result.dy,
                "shift_magnitude_px": result.shift_magnitude_px,
                "peak_value": result.peak_value,
                "peak_to_sidelobe_ratio": result.peak_to_sidelobe_ratio,
                "good_probability": result.good_probability,
                "is_reliable": result.is_reliable,
            }
        except Exception as e:
            print(f"Warning: Error processing satellite pair: {e}")

    if basemap == "satellite":
        combined_prob = (
            satellite_result["good_probability"] if satellite_result else 0.0
        )
    elif basemap == "streets":
        combined_prob = streets_result["good_probability"] if streets_result else 0.0
    else:
        probs = [
            r["good_probability"]
            for r in (streets_result, satellite_result)
            if r is not None
        ]
        combined_prob = sum(probs) / len(probs) if probs else 0.0

    out = {
        "combined_good_probability": combined_prob,
    }
    if streets_result is not None:
        out["streets"] = streets_result
    if satellite_result is not None:
        out["satellite"] = satellite_result

    return out


def main():
    """Test phase correlation on sample images."""
    from pathlib import Path

    data_dir = Path("georef_check/data/raw/dataset_custom")

    ortho_ids = sorted(
        set(f.stem.rsplit("_", 1)[0] for f in data_dir.glob("*_ortho_streets.png"))
    )

    if not ortho_ids:
        print("No ortho images found!")
        return

    test_id = ortho_ids[0]
    print(f"Testing on ortho {test_id}...")

    ortho_streets = data_dir / f"{test_id}_ortho_streets.png"
    streets_only = data_dir / f"{test_id}_streets_only.png"
    ortho_satellite = data_dir / f"{test_id}_ortho_satellite.png"
    satellite_only = data_dir / f"{test_id}_satellite_only.png"

    if not all(
        p.exists()
        for p in [ortho_streets, streets_only, ortho_satellite, satellite_only]
    ):
        print(f"Missing images for {test_id}")
        return

    print("\n=== Testing fft_phase_correlation (grayscale) ===")
    result = fft_phase_correlation(
        cv2.cvtColor(cv2.imread(str(ortho_streets)), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread(str(streets_only)), cv2.COLOR_BGR2RGB),
    )
    print(f"  dx: {result.dx:.2f} px")
    print(f"  dy: {result.dy:.2f} px")
    print(f"  shift_magnitude: {result.shift_magnitude_px:.2f} px")
    print(f"  peak_value: {result.peak_value:.4f}")
    print(f"  PSR: {result.peak_to_sidelobe_ratio:.2f}")

    print("\n=== Testing fft_phase_correlation_edges ===")
    result_edges = fft_phase_correlation_edges(
        cv2.cvtColor(cv2.imread(str(ortho_streets)), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread(str(streets_only)), cv2.COLOR_BGR2RGB),
    )
    print(f"  dx: {result_edges.dx:.2f} px")
    print(f"  dy: {result_edges.dy:.2f} px")
    print(f"  shift_magnitude: {result_edges.shift_magnitude_px:.2f} px")
    print(f"  peak_value: {result_edges.peak_value:.4f}")
    print(f"  PSR: {result_edges.peak_to_sidelobe_ratio:.2f}")

    print("\n=== Testing check_georeferencing_phase ===")
    results = check_georeferencing_phase(
        str(ortho_streets),
        str(streets_only),
        str(ortho_satellite),
        str(satellite_only),
    )

    print(f"\nResults for ortho {test_id}:")
    if "streets" in results:
        r = results["streets"]
        print(
            f"  Streets:  dx={r['dx']:.2f}, dy={r['dy']:.2f}, "
            f"shift={r['shift_magnitude_px']:.2f}px, psr={r['peak_to_sidelobe_ratio']:.2f}, "
            f"prob={r['good_probability']:.3f}, reliable={r['is_reliable']}"
        )
    if "satellite" in results:
        r = results["satellite"]
        print(
            f"  Satellite: dx={r['dx']:.2f}, dy={r['dy']:.2f}, "
            f"shift={r['shift_magnitude_px']:.2f}px, psr={r['peak_to_sidelobe_ratio']:.2f}, "
            f"prob={r['good_probability']:.3f}, reliable={r['is_reliable']}"
        )
    print(f"\n  Combined good_probability: {results['combined_good_probability']:.3f}")


if __name__ == "__main__":
    main()
