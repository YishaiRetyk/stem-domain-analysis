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
    local_var = _compute_local_variance(image_seg, window_size=32)
    var_thresh = np.percentile(local_var, config.variance_threshold_pct)
    variance_mask = local_var > var_thresh

    # Combine: both must be True
    mask = intensity_mask & variance_mask

    # Morphological cleanup
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
    # Slight smoothing for clean edges
    mask_f = ndimage.gaussian_filter(mask.astype(np.float64), sigma=2.0)
    mask = mask_f > 0.5

    mask_uint8 = mask.astype(np.uint8)

    # Compute metrics
    coverage_pct = float(np.sum(mask) / mask.size * 100)
    labeled, n_components = ndimage.label(mask)
    diagnostics["coverage_pct"] = coverage_pct
    diagnostics["n_components"] = int(n_components)
    diagnostics["intensity_threshold"] = float(intensity_thresh)
    diagnostics["variance_threshold"] = float(var_thresh)

    # Gate G3 evaluation
    g3_result = evaluate_gate("G3", {
        "coverage_pct": coverage_pct,
        "n_components": n_components,
    })

    if not g3_result.passed:
        # FALLBACK (I2): use full-image mask
        logger.warning("G3 failed (%s). Using full-frame mask as fallback.",
                       g3_result.reason)
        mask_uint8 = np.ones((h, w), dtype=np.uint8)
        coverage_pct = 100.0
        n_components = 1
        diagnostics["fallback"] = True
        diagnostics["coverage_pct"] = 100.0
        diagnostics["n_components"] = 1
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
