"""
Feature extraction for georeferencing quality check.

Extracts alignment features from ortho-basemap viewport pairs.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional


def compute_phase_correlation(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """
    Compute phase correlation between two images.

    Estimates translation (dx, dy) and peak correlation strength.

    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)

    Returns:
        Dict with 'dx', 'dy', 'peak_strength'
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute phase correlation
    result = cv2.phaseCorrelate(img1.astype(np.float64), img2.astype(np.float64))

    # Handle different OpenCV versions
    if isinstance(result[1], tuple):
        peak_val = result[1][0]
    else:
        peak_val = result[1]
    dx, dy = result[0]

    # Normalize peak to 0-1 range (it can be > 1 in some implementations)
    peak_normalized = min(float(peak_val), 1.0)

    return {"dx": float(dx), "dy": float(dy), "peak_strength": float(peak_normalized)}


def compute_edge_correlation(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """
    Compute edge-based correlation between two images.

    Uses Canny edge detection and normalized cross-correlation.

    Args:
        img1: First image (RGB or grayscale)
        img2: Second image (RGB or grayscale)

    Returns:
        Dict with 'edge_correlation', 'edge_density_diff'
    """
    # Convert to grayscale
    gray1 = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Ensure same size
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Compute Canny edges
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)

    # Normalize to 0-1
    edges1 = edges1.astype(np.float32) / 255.0
    edges2 = edges2.astype(np.float32) / 255.0

    # Compute normalized cross-correlation
    correlation = np.sum(edges1 * edges2) / (
        np.sqrt(np.sum(edges1**2)) * np.sqrt(np.sum(edges2**2)) + 1e-8
    )

    # Edge density difference
    density1 = np.mean(edges1)
    density2 = np.mean(edges2)
    density_diff = abs(density1 - density2)

    return {
        "edge_correlation": float(correlation),
        "edge_density_diff": float(density_diff),
    }


def compute_ssim_approx(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """
    Compute approximate SSIM (Structural Similarity Index).

    Simplified version using basic statistics.

    Args:
        img1: First image (RGB or grayscale)
        img2: Second image (RGB or grayscale)

    Returns:
        Dict with 'ssim_approx', 'mse'
    """
    # Convert to grayscale
    gray1 = (
        img1
        if len(img1.shape) == 2
        else cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    )
    gray2 = (
        img2
        if len(img2.shape) == 2
        else cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    )

    # Ensure same size
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Mean and variance
    mu1, mu2 = np.mean(gray1), np.mean(gray2)
    var1, var2 = np.var(gray1), np.var(gray2)

    # Covariance
    cov = np.mean((gray1 - mu1) * (gray2 - mu2))

    # SSIM components
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / (
        (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
    )

    # MSE
    mse = np.mean((gray1 - gray2) ** 2)

    return {"ssim_approx": float(ssim), "mse": float(mse)}


def compute_histogram_correlation(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """
    Compute histogram-based similarity.

    Args:
        img1: First image (RGB)
        img2: Second image (RGB)

    Returns:
        Dict with 'hist_correlation'
    """
    # Convert to grayscale
    gray1 = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Ensure same size
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Compute histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256]).flatten()
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256]).flatten()

    # Normalize histograms
    hist1 = hist1 / (np.sum(hist1) + 1e-8)
    hist2 = hist2 / (np.sum(hist2) + 1e-8)

    # Compute correlation
    correlation = np.corrcoef(hist1, hist2)[0, 1]

    return {
        "hist_correlation": float(correlation) if not np.isnan(correlation) else 0.0
    }


def extract_features(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """
    Extract all alignment features from an image pair.

    Args:
        img1: First image (ortho viewport)
        img2: Second image (basemap viewport)

    Returns:
        Dict of all extracted features
    """
    features = {}

    # Phase correlation
    pc = compute_phase_correlation(img1, img2)
    features.update(pc)

    # Edge correlation
    ec = compute_edge_correlation(img1, img2)
    features.update(ec)

    # SSIM approximation
    ssim = compute_ssim_approx(img1, img2)
    features.update(ssim)

    # Histogram correlation
    hc = compute_histogram_correlation(img1, img2)
    features.update(hc)

    return features


def compute_shift_magnitude(dx: float, dy: float, pixel_size: float = 1.0) -> float:
    """
    Compute total shift magnitude.

    Args:
        dx: Horizontal shift in pixels
        dy: Vertical shift in pixels
        pixel_size: Size of pixel in meters (for ground distance)

    Returns:
        Shift magnitude in pixels (or meters if pixel_size provided)
    """
    return np.sqrt(dx**2 + dy**2) * pixel_size


def main():
    """Test feature extraction."""
    # Create dummy test images
    img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img2 = img1.copy()  # Perfect alignment

    features = extract_features(img1, img2)
    print("Features from aligned images:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

    # Shifted image
    img2_shifted = np.roll(np.roll(img2, 10, axis=0), 5, axis=1)

    features_shifted = extract_features(img1, img2_shifted)
    print("\nFeatures from shifted images:")
    for k, v in features_shifted.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
