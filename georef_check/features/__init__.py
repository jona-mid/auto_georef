"""
Feature extraction module.

Wrapper for all feature extraction methods.
"""

from .extractor import (
    extract_features,
    compute_phase_correlation,
    compute_edge_correlation,
    compute_ssim_approx,
    compute_histogram_correlation,
    compute_shift_magnitude,
)

__all__ = [
    "extract_features",
    "compute_phase_correlation",
    "compute_edge_correlation",
    "compute_ssim_approx",
    "compute_histogram_correlation",
    "compute_shift_magnitude",
]
