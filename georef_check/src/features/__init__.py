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

from .phase_correlation import (
    fft_phase_correlation,
    fft_phase_correlation_edges,
    check_georeferencing_phase,
    apply_hann_window,
    PhaseCorrelationResult,
)

__all__ = [
    "extract_features",
    "compute_phase_correlation",
    "compute_edge_correlation",
    "compute_ssim_approx",
    "compute_histogram_correlation",
    "compute_shift_magnitude",
    "fft_phase_correlation",
    "fft_phase_correlation_edges",
    "check_georeferencing_phase",
    "apply_hann_window",
    "PhaseCorrelationResult",
]
