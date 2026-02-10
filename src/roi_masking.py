"""
Early ROI Masking (WS4).

Intensity + variance-based masking with purely geometric gates (C6).
NO FFT crystallinity -- avoids circularity.

Gate G3: coverage in [10%, 95%], connected_components <= max_fragments.
Failure: FALLBACK (I2) -- use full-image mask.
"""

import logging
import numpy as np
from scipy import ndimage

from src.pipeline_config import ROIMaskResult, ROIConfig
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


def _compute_gradient_magnitude(image, sigma=1.0):  # sigma from config.gradient_smooth_sigma
    """Compute gradient magnitude using Sobel operators on smoothed image."""
    from scipy.ndimage import sobel, gaussian_filter
    smoothed = gaussian_filter(image.astype(np.float64), sigma=sigma)
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    return np.sqrt(gx**2 + gy**2)


def _compute_lcc_fraction(mask):
    """Compute fraction of mask area occupied by largest connected component.

    Returns 1.0 if mask is all zero (no components).
    """
    total_area = int(np.sum(mask > 0))
    if total_area == 0:
        return 1.0
    labeled, n = ndimage.label(mask)
    if n == 0:
        return 1.0
    component_sizes = ndimage.sum(mask > 0, labeled, range(1, n + 1))
    largest = float(np.max(component_sizes))
    return largest / total_area


def _compute_local_variance(image: np.ndarray, window_size: int = 32) -> np.ndarray:
    """Compute local variance as texture indicator."""
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = ndimage.convolve(image.astype(np.float64), kernel, mode='reflect')
    local_sq_mean = ndimage.convolve(image.astype(np.float64) ** 2, kernel, mode='reflect')
    local_var = local_sq_mean - local_mean ** 2
    return np.maximum(local_var, 0)


def compute_roi_mask(image_seg: np.ndarray,
                     config: ROIConfig = None) -> ROIMaskResult:
    """Compute ROI mask from segmentation-preprocessed image.

    Uses intensity and variance masks only -- NO FFT crystallinity (C6).

    Parameters
    ----------
    image_seg : np.ndarray
        Branch B (segmentation) preprocessed image, [0, 1].
    config : ROIConfig, optional

    Returns
    -------
    ROIMaskResult with full-res mask, tile-grid mask, and diagnostics.
    """
    if config is None:
        config = ROIConfig()

    h, w = image_seg.shape
    diagnostics = {"method": "intensity+variance"}

    # Intensity mask: exclude very dark regions
    intensity_thresh = np.percentile(image_seg, config.intensity_threshold_pct)
    intensity_mask = image_seg > intensity_thresh

    # Variance mask: exclude smooth/uniform regions
    local_var = _compute_local_variance(image_seg, window_size=config.variance_window_size)
    var_thresh = np.percentile(local_var, config.variance_threshold_pct)
    variance_mask = local_var > var_thresh

    # Combine: both must be True
    mask = intensity_mask & variance_mask

    # Morphological cleanup
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=np.ones((config.morph_kernel_size, config.morph_kernel_size)))
    # Slight smoothing for clean edges
    mask_f = ndimage.gaussian_filter(mask.astype(np.float64), sigma=config.smooth_sigma)
    mask = mask_f > 0.5

    mask_uint8 = mask.astype(np.uint8)

    # Compute metrics
    coverage_pct = float(np.sum(mask) / mask.size * 100)
    labeled, n_components = ndimage.label(mask)
    lcc_fraction = _compute_lcc_fraction(mask_uint8)

    diagnostics["coverage_pct"] = coverage_pct
    diagnostics["n_components"] = int(n_components)
    diagnostics["intensity_threshold"] = float(intensity_thresh)
    diagnostics["variance_threshold"] = float(var_thresh)
    diagnostics["lcc_fraction"] = lcc_fraction
    diagnostics["coverage_before_gradient"] = coverage_pct
    diagnostics["gradient_used"] = False
    diagnostics["roi_confidence"] = "normal"

    # Check whether primary mask is acceptable
    coverage_ok = (config.min_coverage_pct <= coverage_pct <= config.max_coverage_pct)
    lcc_ok = (lcc_fraction >= config.min_lcc_fraction)

    if not coverage_ok or not lcc_ok:
        # Gradient-magnitude fallback
        logger.info("Primary mask coverage=%.1f%%, lcc_fraction=%.2f -- "
                     "applying gradient-magnitude fallback.",
                     coverage_pct, lcc_fraction)
        grad_mag = _compute_gradient_magnitude(image_seg, sigma=config.gradient_smooth_sigma)
        grad_thresh = np.percentile(grad_mag, config.gradient_threshold_pct)
        grad_mask = grad_mag > grad_thresh

        # Union gradient mask with primary mask
        combined = (mask_uint8 > 0) | grad_mask

        # Morphological cleanup on combined mask
        combined = ndimage.binary_fill_holes(combined)
        combined = ndimage.binary_opening(combined, structure=np.ones((config.morph_kernel_size, config.morph_kernel_size)))
        combined_f = ndimage.gaussian_filter(combined.astype(np.float64), sigma=config.smooth_sigma)
        combined = combined_f > 0.5

        mask_uint8 = combined.astype(np.uint8)
        coverage_pct = float(np.sum(mask_uint8) / mask_uint8.size * 100)
        labeled, n_components = ndimage.label(mask_uint8)
        lcc_fraction = _compute_lcc_fraction(mask_uint8)

        diagnostics["gradient_used"] = True
        diagnostics["coverage_pct"] = coverage_pct
        diagnostics["n_components"] = int(n_components)
        diagnostics["lcc_fraction"] = lcc_fraction

    # If lcc_fraction is STILL too low after gradient fallback, force full-image ROI
    if lcc_fraction < config.min_lcc_fraction:
        logger.warning("lcc_fraction=%.2f still below %.2f after gradient fallback. "
                       "Forcing full-image ROI with low confidence.",
                       lcc_fraction, config.min_lcc_fraction)
        mask_uint8 = np.ones((h, w), dtype=np.uint8)
        coverage_pct = 100.0
        n_components = 1
        lcc_fraction = 1.0
        diagnostics["roi_confidence"] = "low"
        diagnostics["coverage_pct"] = 100.0
        diagnostics["n_components"] = 1
        diagnostics["lcc_fraction"] = 1.0

    # Gate G3 evaluation
    g3_result = evaluate_gate("G3", {
        "coverage_pct": coverage_pct,
        "n_components": n_components,
        "lcc_fraction": lcc_fraction,
    })

    if not g3_result.passed:
        # FALLBACK (I2): use full-image mask
        logger.warning("G3 failed (%s). Using full-frame mask as fallback.",
                       g3_result.reason)
        mask_uint8 = np.ones((h, w), dtype=np.uint8)
        coverage_pct = 100.0
        n_components = 1
        lcc_fraction = 1.0
        diagnostics["fallback"] = True
        diagnostics["coverage_pct"] = 100.0
        diagnostics["n_components"] = 1
        diagnostics["lcc_fraction"] = 1.0
        diagnostics["roi_confidence"] = "low"
    else:
        diagnostics["fallback"] = False

    diagnostics["g3_passed"] = g3_result.passed
    diagnostics["g3_reason"] = g3_result.reason

    return ROIMaskResult(
        mask_full=mask_uint8,
        mask_grid=np.array([], dtype=bool),  # populated by downsample_to_tile_grid
        coverage_pct=coverage_pct,
        n_components=int(n_components),
        diagnostics=diagnostics,
        lcc_fraction=lcc_fraction,
    )


def downsample_to_tile_grid(mask_full: np.ndarray,
                            tile_size: int,
                            stride: int,
                            min_coverage: float = 0.5) -> np.ndarray:
    """Downsample full-resolution mask to tile grid.

    A tile is considered "in ROI" if at least min_coverage fraction
    of its pixels are masked.

    Returns
    -------
    mask_grid : np.ndarray[bool] of shape (n_rows, n_cols)
    """
    h, w = mask_full.shape
    n_rows = (h - tile_size) // stride + 1
    n_cols = (w - tile_size) // stride + 1

    mask_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * stride
            x = col * stride
            tile_mask = mask_full[y:y + tile_size, x:x + tile_size]
            coverage = np.mean(tile_mask)
            mask_grid[row, col] = coverage >= min_coverage

    return mask_grid
