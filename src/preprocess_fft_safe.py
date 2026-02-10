"""
FFT-Safe Preprocessing (Branch A).

Steps:
1. float64 conversion
2. Optional hot-pixel removal (I1): median 3x3 on statistical outliers only
3. Outlier clip at configured percentile
4. Robust normalisation: (img - median) / (1.4826 * MAD), clip to [-5,5], map to [0,1]
5. NO Gaussian blur

Gate G2: clipped_fraction < 0.5%, intensity_range_ratio > 10.
Failure: DEGRADE_CONFIDENCE (I3) -- fall back to min-max normalisation.
"""

import logging
import numpy as np
from scipy import ndimage

from src.pipeline_config import FFTPreprocRecord, PreprocConfig
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


def _remove_hot_pixels(image: np.ndarray, sigma: float = 5.0,
                       median_kernel: int = 3) -> tuple:
    """Replace statistical outlier pixels with local median (I1).

    Only replaces pixels where |pixel - local_median| > sigma * local_MAD.
    This is NOT Gaussian blur -- it preserves periodicity for non-outlier pixels.

    Returns (cleaned_image, n_replaced).
    """
    local_median = ndimage.median_filter(image, size=median_kernel)
    deviation = np.abs(image - local_median)
    # Compute local MAD using the same window
    local_mad = ndimage.median_filter(deviation, size=median_kernel)
    # Threshold: pixels deviating by more than sigma * local_MAD
    threshold = sigma * local_mad * 1.4826  # convert MAD to Gaussian sigma
    threshold = np.maximum(threshold, 1e-10)  # avoid zero threshold
    outlier_mask = deviation > threshold
    n_replaced = int(np.sum(outlier_mask))

    cleaned = image.copy()
    cleaned[outlier_mask] = local_median[outlier_mask]
    return cleaned, n_replaced


def _robust_normalize(image: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
    """Robust normalisation: (img - median) / (1.4826 * MAD), clip to [-clip_sigma, clip_sigma], map to [0,1]."""
    med = np.median(image)
    mad = np.median(np.abs(image - med))
    sigma = mad * 1.4826
    if sigma < 1e-10:
        # Degenerate case: constant image
        return np.zeros_like(image)
    normalised = (image - med) / sigma
    normalised = np.clip(normalised, -clip_sigma, clip_sigma)
    # Map [-clip_sigma, clip_sigma] to [0, 1]
    normalised = (normalised + clip_sigma) / (2.0 * clip_sigma)
    return normalised


def _minmax_normalize(image: np.ndarray) -> np.ndarray:
    """Fallback min-max normalisation."""
    imin, imax = image.min(), image.max()
    if imax - imin < 1e-10:
        return np.zeros_like(image)
    return (image - imin) / (imax - imin)


def compute_spectral_entropy(image: np.ndarray) -> float:
    """Compute spectral entropy as a diagnostic (NOT gated)."""
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2
    # Mask DC
    h, w = power.shape
    cy, cx = h // 2, w // 2
    dc_mask = np.ones_like(power, dtype=bool)
    dc_mask[cy - 1:cy + 2, cx - 1:cx + 2] = False
    p = power[dc_mask]
    p = p / (np.sum(p) + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def preprocess_fft_safe(image: np.ndarray,
                        config: PreprocConfig = None) -> FFTPreprocRecord:
    """FFT-safe preprocessing (Branch A).

    Parameters
    ----------
    image : np.ndarray
        Raw input image.
    config : PreprocConfig, optional
        Preprocessing parameters. Uses defaults if None.

    Returns
    -------
    FFTPreprocRecord
    """
    if config is None:
        config = PreprocConfig()

    diagnostics = {
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "input_min": float(np.min(image)),
        "input_max": float(np.max(image)),
    }

    # Step 1: float64
    img = image.astype(np.float64)

    # Step 2: Optional hot-pixel removal (I1)
    n_hot_replaced = 0
    if config.hot_pixel_removal:
        img, n_hot_replaced = _remove_hot_pixels(
            img, sigma=config.hot_pixel_sigma,
            median_kernel=config.hot_pixel_median_kernel)
        logger.info("Hot-pixel removal: replaced %d pixels (sigma=%.1f)",
                     n_hot_replaced, config.hot_pixel_sigma)
    diagnostics["hot_pixels_replaced"] = n_hot_replaced

    # Step 3: Outlier clip
    low_pct = config.clip_percentile
    high_pct = 100 - config.clip_percentile
    p_low = np.percentile(img, low_pct)
    p_high = np.percentile(img, high_pct)
    n_clipped_low = int(np.sum(img < p_low))
    n_clipped_high = int(np.sum(img > p_high))
    n_total = img.size
    clipped_fraction = (n_clipped_low + n_clipped_high) / n_total

    img = np.clip(img, p_low, p_high)
    diagnostics["clipped_fraction"] = clipped_fraction
    diagnostics["n_clipped_low"] = n_clipped_low
    diagnostics["n_clipped_high"] = n_clipped_high

    # Compute intensity range ratio for G2
    p01 = np.percentile(img, 0.1)
    p999 = np.percentile(img, 99.9)
    med = np.median(img)
    intensity_range_ratio = (p999 - p01) / (med + 1e-10) if med > 0 else 0
    diagnostics["intensity_range_ratio"] = float(intensity_range_ratio)

    # Gate G2 evaluation
    g2_result = evaluate_gate("G2", {
        "clipped_fraction": clipped_fraction,
        "intensity_range_ratio": intensity_range_ratio,
    })

    confidence = "normal"
    if not g2_result.passed:
        # DEGRADE_CONFIDENCE (I3): fall back to min-max
        logger.warning("G2 failed (%s). Falling back to min-max normalisation.",
                       g2_result.reason)
        img = _minmax_normalize(img)
        confidence = "degraded"
        diagnostics["normalize_method"] = "minmax"
    else:
        # Step 4: Robust normalisation
        if config.normalize_method == "robust":
            img = _robust_normalize(img, clip_sigma=config.robust_norm_clip_sigma)
            diagnostics["normalize_method"] = "robust"
        else:
            img = _minmax_normalize(img)
            diagnostics["normalize_method"] = "minmax"

    diagnostics["output_min"] = float(np.min(img))
    diagnostics["output_max"] = float(np.max(img))
    diagnostics["output_mean"] = float(np.mean(img))

    # Diagnostic: spectral entropy (NOT gated, C1)
    spectral_entropy = compute_spectral_entropy(img)

    qc_metrics = {
        "spectral_entropy": spectral_entropy,
        "clipped_fraction": clipped_fraction,
        "intensity_range_ratio": float(intensity_range_ratio),
        "g2_passed": g2_result.passed,
        "g2_reason": g2_result.reason,
        "frequency_unit": "cycles/nm",
    }

    return FFTPreprocRecord(
        image_fft=img,
        diagnostics=diagnostics,
        qc_metrics=qc_metrics,
        confidence=confidence,
    )
