"""
FFT Peak Detection and Two-Tier Classification (WS3).

Per-peak SNR (B3), FWHM (B2), symmetry, non-collinearity.
Two-tier classification: Tier A (high-confidence) and Tier B (weak evidence).
"""

import logging
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Optional, Tuple

from src.fft_coords import FFTGrid
from src.pipeline_config import (
    PeakSNR, PeakFWHM, TilePeak, TilePeakSet, TileClassification,
    TierConfig, PeakGateConfig, FWHMConfig,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Peak-height SNR (B3, C3)
# ======================================================================

def _build_exclusion_mask(x_grid: np.ndarray, y_grid: np.ndarray,
                          all_peaks: List[TilePeak],
                          fft_grid: FFTGrid,
                          safety_margin_px: int = 2) -> np.ndarray:
    """Pre-compute combined exclusion mask for all peaks and their antipodals.

    Called once per tile to avoid O(P² × HW) recomputation.
    """
    H, W = x_grid.shape
    exclusion_mask = np.zeros((H, W), dtype=bool)
    for p in all_peaks:
        exclusion_r = safety_margin_px + max(2, p.fwhm / fft_grid.qx_scale / 2 if p.fwhm > 0 else 2)
        exclusion_r_sq = exclusion_r ** 2
        for sign in [1, -1]:
            ppx, ppy = fft_grid.q_to_px(sign * p.qx, sign * p.qy)
            dist_sq = (x_grid - ppx) ** 2 + (y_grid - ppy) ** 2
            exclusion_mask |= (dist_sq <= exclusion_r_sq)
    return exclusion_mask


def compute_peak_snr(power: np.ndarray,
                     target_peak: TilePeak,
                     all_peaks: List[TilePeak],
                     fft_grid: FFTGrid,
                     _cached: Optional[dict] = None,
                     effective_q_min: float = 0.0) -> PeakSNR:
    """Compute peak-height SNR with peak-excluding annular background.

    Signal = max(power[disk_r3]) around peak centre.
    Background = annulus excluding ALL detected peaks ± safety margin.
    sigma via MAD × 1.4826.

    Parameters
    ----------
    _cached : dict, optional
        Pre-computed arrays from classify_tile to avoid per-peak reallocation.
        Keys: 'x_grid', 'y_grid', 'exclusion_mask'.
    """
    H, W = power.shape

    if _cached is not None:
        x_grid = _cached['x_grid']
        y_grid = _cached['y_grid']
        exclusion_mask = _cached['exclusion_mask']
    else:
        y_grid, x_grid = np.mgrid[:H, :W]
        exclusion_mask = _build_exclusion_mask(x_grid, y_grid, all_peaks,
                                               fft_grid)

    # Peak pixel position
    peak_px_x, peak_px_y = fft_grid.q_to_px(target_peak.qx, target_peak.qy)

    # Signal: max in 3-pixel-radius disk
    dist_sq = (x_grid - peak_px_x) ** 2 + (y_grid - peak_px_y) ** 2
    signal_disk = dist_sq <= 9  # r=3
    if not np.any(signal_disk):
        return PeakSNR(0, 0, 1, 0, 0, "no signal pixels")
    signal_peak = float(np.max(power[signal_disk]))

    # Annular band at peak's |q|
    q_mag_grid = fft_grid.q_mag_grid()
    peak_q = target_peak.q_mag
    annular_width = max(0.15, target_peak.fwhm * 1.5 if target_peak.fwhm > 0 else 0.15)
    annulus_mask = np.abs(q_mag_grid - peak_q) <= annular_width
    if effective_q_min > 0:
        annulus_mask &= (q_mag_grid >= effective_q_min)

    background_mask = annulus_mask & ~exclusion_mask & ~signal_disk
    n_bg = int(np.sum(background_mask))

    note = None
    if n_bg < 20:
        background_mask = annulus_mask & ~signal_disk
        n_bg = int(np.sum(background_mask))
        note = "full-annulus fallback (insufficient background after peak exclusion)"

    if n_bg == 0:
        return PeakSNR(signal_peak, 0, 1, signal_peak, 0, "no background pixels")

    bg_values = power[background_mask]
    bg_median = float(np.median(bg_values))
    bg_mad = float(np.median(np.abs(bg_values - bg_median)))
    bg_sigma = bg_mad * 1.4826

    snr = (signal_peak - bg_median) / (bg_sigma + 1e-10)

    if effective_q_min > 0 and peak_q < 2 * effective_q_min:
        low_q_note = f"peak q={peak_q:.3f} < 2×q_min={2*effective_q_min:.3f}"
        note = f"{note}; {low_q_note}" if note else low_q_note

    return PeakSNR(
        signal_peak=signal_peak,
        background_median=bg_median,
        background_mad_sigma=bg_sigma,
        snr=float(snr),
        n_background_px=n_bg,
        note=note,
    )


# ======================================================================
# FWHM measurement (B2): cached grids, proxy width, gated curve_fit
# ======================================================================

# --- Cached 11×11 patch grids (R=5) computed once at import time ---
_PATCH_R = 5
_PATCH_SIZE = 2 * _PATCH_R + 1
_Y11, _X11 = np.mgrid[:_PATCH_SIZE, :_PATCH_SIZE]
_DIST11 = np.sqrt((_X11 - _PATCH_R) ** 2 + (_Y11 - _PATCH_R) ** 2)
_OUTER_RING11 = _DIST11 >= (_PATCH_R - 1)
# Pre-computed radial bin masks for fallback
_RADIAL_MASKS11 = [(_DIST11 >= dr) & (_DIST11 < dr + 1)
                    for dr in range(_PATCH_R + 1)]


def _gaussian_2d(coords, amplitude, cx, cy, sigma_x, sigma_y, theta):
    """2D Gaussian model for curve_fit."""
    x, y = coords
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = cos_t * (x - cx) + sin_t * (y - cy)
    yr = -sin_t * (x - cx) + cos_t * (y - cy)
    g = amplitude * np.exp(-0.5 * (xr ** 2 / (sigma_x ** 2 + 1e-10) +
                                    yr ** 2 / (sigma_y ** 2 + 1e-10)))
    return g.ravel()


def _extract_patch(power, peak, fft_grid):
    """Extract background-subtracted patch around a peak.

    Returns (patch_bg, px, py) or (None, None, None) if out of bounds.
    """
    H, W = power.shape
    peak_px_x, peak_px_y = fft_grid.q_to_px(peak.qx, peak.qy)
    px = int(round(peak_px_x))
    py = int(round(peak_px_y))
    R = _PATCH_R

    if py - R < 0 or py + R + 1 > H or px - R < 0 or px + R + 1 > W:
        return None, None, None

    patch = power[py - R:py + R + 1, px - R:px + R + 1].copy()
    if patch.size < 9:
        return None, None, None

    bg_level = np.median(patch[_OUTER_RING11])
    return patch - bg_level, px, py


def measure_peak_fwhm_proxy(power: np.ndarray,
                            peak: TilePeak,
                            fft_grid: FFTGrid) -> PeakFWHM:
    """Fast moment-based FWHM proxy — no optimizer, O(1) per peak.

    Uses intensity-weighted second moment on the 11×11 patch to estimate
    sigma, then converts to FWHM.  Falls back to radial half-max if the
    moment estimate is degenerate.
    """
    patch_bg, px, py = _extract_patch(power, peak, fft_grid)
    if patch_bg is None:
        return PeakFWHM(fwhm_valid=False, method="failed")

    R = _PATCH_R

    # --- Moment-based sigma estimate ---
    weights = np.maximum(patch_bg, 0)
    total_w = weights.sum()
    if total_w > 0:
        cx = np.sum(_X11 * weights) / total_w
        cy = np.sum(_Y11 * weights) / total_w
        var_x = np.sum(weights * (_X11 - cx) ** 2) / total_w
        var_y = np.sum(weights * (_Y11 - cy) ** 2) / total_w
        sigma_px = np.sqrt(max(var_x, 0.01) * max(var_y, 0.01))
        if 0.3 <= sigma_px <= R:
            sigma_q = sigma_px * fft_grid.qx_scale
            fwhm_q = 2.355 * sigma_q
            return PeakFWHM(
                fwhm_q=float(fwhm_q),
                sigma_x=float(np.sqrt(max(var_x, 0.01)) * fft_grid.qx_scale),
                sigma_y=float(np.sqrt(max(var_y, 0.01)) * fft_grid.qy_scale),
                theta=0.0,
                fwhm_valid=True,
                method="moment_proxy",
            )

    # --- Radial half-max fallback ---
    return _radial_halfmax(patch_bg, fft_grid)


def _radial_halfmax(patch_bg, fft_grid):
    """Radial half-max FWHM using pre-computed bin masks."""
    R = _PATCH_R
    radial_profile = np.zeros(R + 1)
    for dr in range(R + 1):
        m = _RADIAL_MASKS11[dr]
        if np.any(m):
            radial_profile[dr] = np.mean(patch_bg[m])

    peak_val = radial_profile[0]
    if peak_val > 0:
        half_max = peak_val / 2
        fwhm_bins = int(np.sum(radial_profile >= half_max))
        fwhm_q = float(fwhm_bins * fft_grid.qx_scale)
        return PeakFWHM(
            fwhm_q=fwhm_q,
            sigma_x=fwhm_q / 2.355,
            sigma_y=fwhm_q / 2.355,
            theta=0.0,
            fwhm_valid=True,
            method="wedge_fallback",
        )
    return PeakFWHM(fwhm_valid=False, method="failed")


def measure_peak_fwhm(power: np.ndarray,
                      peak: TilePeak,
                      fft_grid: FFTGrid,
                      maxfev: int = 500) -> PeakFWHM:
    """Measure peak FWHM via 2D Gaussian fit with fallback (B2).

    Uses cached 11×11 grids.  ``maxfev`` controls curve_fit iteration
    budget (default 500, was 2000).

    Returns PeakFWHM with fwhm_valid=False if all methods fail.
    """
    patch_bg, px, py = _extract_patch(power, peak, fft_grid)
    if patch_bg is None:
        return PeakFWHM(fwhm_valid=False, method="failed")

    R = _PATCH_R

    # PRIMARY: 2D Gaussian fit (with cached grid arrays)
    try:
        p0 = [float(patch_bg[R, R]), float(R), float(R), 2.0, 2.0, 0.0]
        bounds = ([0, R - 3, R - 3, 0.3, 0.3, -np.pi],
                  [np.inf, R + 3, R + 3, R, R, np.pi])
        popt, pcov = curve_fit(
            _gaussian_2d, (_X11, _Y11), patch_bg.ravel(),
            p0=p0, bounds=bounds, maxfev=maxfev,
        )
        amp, cx, cy, sigma_x, sigma_y, theta = popt

        if sigma_x <= 0 or sigma_y <= 0 or amp < 0:
            raise ValueError("negative sigma or amplitude")
        cond = np.linalg.cond(pcov)
        if cond > 100:
            raise ValueError(f"ill-conditioned fit: cond={cond:.0f}")

        sigma_x_q = abs(sigma_x) * fft_grid.qx_scale
        sigma_y_q = abs(sigma_y) * fft_grid.qy_scale
        fwhm_q = 2.355 * np.sqrt(sigma_x_q * sigma_y_q)

        return PeakFWHM(
            fwhm_q=float(fwhm_q),
            sigma_x=float(sigma_x_q),
            sigma_y=float(sigma_y_q),
            theta=float(np.degrees(theta)),
            fwhm_valid=True,
            method="gaussian_2d",
        )
    except Exception:
        pass

    # FALLBACK: radial half-max (using cached masks)
    return _radial_halfmax(patch_bg, fft_grid)


# ======================================================================
# Symmetry check
# ======================================================================

def check_symmetry(peaks: List[TilePeak],
                   fft_grid: FFTGrid,
                   tolerance_px: float = 2.0) -> Tuple[float, int]:
    """Check ±g symmetry among peaks.

    Returns (symmetry_score, n_paired).
    symmetry_score = n_paired / n_peaks.
    """
    n = len(peaks)
    if n == 0:
        return 0.0, 0

    tol_q = tolerance_px * fft_grid.qx_scale
    paired = [False] * n

    for i in range(n):
        if paired[i]:
            continue
        for j in range(i + 1, n):
            if paired[j]:
                continue
            # Check if g_i + g_j ≈ 0
            residual = np.sqrt((peaks[i].qx + peaks[j].qx) ** 2 +
                               (peaks[i].qy + peaks[j].qy) ** 2)
            if residual < tol_q:
                paired[i] = True
                paired[j] = True
                break

    n_paired = sum(paired)
    return n_paired / n if n > 0 else 0.0, n_paired


# ======================================================================
# Non-collinearity check
# ======================================================================

def count_non_collinear(peaks: List[TilePeak], min_angle_deg: float = 15.0) -> int:
    """Count maximum number of non-collinear g-vectors."""
    if not peaks:
        return 0
    angles = [p.angle_deg for p in peaks]
    # Greedy selection: add peaks that are far enough from all selected
    selected = [angles[0]]
    for a in angles[1:]:
        if all(abs(((a - s + 90) % 180) - 90) > min_angle_deg for s in selected):
            selected.append(a)
    return len(selected)


# ======================================================================
# Two-tier classification
# ======================================================================

def classify_tile(peak_set: TilePeakSet,
                  fft_grid: FFTGrid,
                  tier_config: TierConfig = None,
                  peak_gate_config: PeakGateConfig = None,
                  fwhm_config: FWHMConfig = None,
                  effective_q_min: float = 0.0) -> TileClassification:
    """Classify a tile as Tier A, Tier B, or REJECTED.

    Uses power_spectrum from peak_set for SNR and FWHM computation.

    FWHM policy (controlled by *fwhm_config*):
    - ``auto``: proxy width for all peaks, curve_fit only for top-K high-SNR
    - ``proxy_only``: never run curve_fit
    - ``curve_fit``: always run curve_fit (legacy behaviour, slow)
    """
    if tier_config is None:
        tier_config = TierConfig()
    if peak_gate_config is None:
        peak_gate_config = PeakGateConfig()
    if fwhm_config is None:
        fwhm_config = FWHMConfig()

    peaks = peak_set.peaks
    power = peak_set.power_spectrum

    if not peaks or power is None:
        return TileClassification(
            tier="REJECTED", peaks=[], symmetry_score=0,
            n_non_collinear=0, best_snr=0, best_orientation_deg=0,
        )

    # Pre-compute shared grids once (avoids O(P² × HW) recomputation)
    H, W = power.shape
    y_grid, x_grid = np.mgrid[:H, :W]
    exclusion_mask = _build_exclusion_mask(x_grid, y_grid, peaks, fft_grid)
    _cached = {'x_grid': x_grid, 'y_grid': y_grid,
               'exclusion_mask': exclusion_mask}

    # --- Pass 1: compute SNR for all peaks (cheap) ---
    snr_results = []
    for p in peaks:
        snr_results.append(compute_peak_snr(power, p, peaks, fft_grid,
                                            _cached=_cached,
                                            effective_q_min=effective_q_min))

    # --- Pass 2: FWHM with policy gating ---
    fwhm_method = fwhm_config.method if fwhm_config.enabled else "proxy_only"
    curve_fits_remaining = fwhm_config.max_per_tile

    # For "auto" mode, sort peaks by SNR descending to prioritise high-SNR
    # peaks for the limited curve_fit budget.
    if fwhm_method == "auto":
        ranked_indices = sorted(range(len(peaks)),
                                key=lambda i: snr_results[i].snr,
                                reverse=True)
    else:
        ranked_indices = list(range(len(peaks)))

    fwhm_results = [None] * len(peaks)
    for idx in ranked_indices:
        p = peaks[idx]
        snr_val = snr_results[idx].snr
        use_curve_fit = False

        if fwhm_method == "curve_fit":
            use_curve_fit = True
        elif fwhm_method == "auto":
            use_curve_fit = (snr_val >= fwhm_config.min_snr_for_fit
                             and curve_fits_remaining > 0)

        if use_curve_fit:
            fwhm_results[idx] = measure_peak_fwhm(
                power, p, fft_grid, maxfev=fwhm_config.maxfev)
            curve_fits_remaining -= 1
        else:
            fwhm_results[idx] = measure_peak_fwhm_proxy(power, p, fft_grid)

    # --- Build per-peak metrics ---
    peak_metrics = []
    best_snr = 0.0
    best_orientation = 0.0

    for i, p in enumerate(peaks):
        snr_result = snr_results[i]
        fwhm_result = fwhm_results[i]

        # Update peak's fwhm for downstream
        p.fwhm = fwhm_result.fwhm_q if fwhm_result.fwhm_valid else 0.0

        # FWHM gate check
        fwhm_ok = True
        if fwhm_result.fwhm_valid and p.q_mag > 0:
            fwhm_ok = fwhm_result.fwhm_q <= peak_gate_config.max_fwhm_ratio * p.q_mag

        peak_metrics.append({
            "qx": p.qx, "qy": p.qy, "q_mag": p.q_mag,
            "snr": snr_result.snr,
            "fwhm": fwhm_result.fwhm_q,
            "fwhm_valid": fwhm_result.fwhm_valid,
            "fwhm_method": fwhm_result.method,
            "fwhm_ok": fwhm_ok,
        })

        if snr_result.snr > best_snr:
            best_snr = snr_result.snr
            best_orientation = p.angle_deg

    # Symmetry
    symmetry_score, n_paired = check_symmetry(peaks, fft_grid)

    # Non-collinearity
    n_non_collinear = count_non_collinear(peaks)

    # Classification
    tier = "REJECTED"

    if best_snr >= tier_config.tier_a_snr:
        if (n_non_collinear >= peak_gate_config.min_non_collinear and
                symmetry_score >= peak_gate_config.min_symmetry):
            tier = "A"
        elif best_snr >= tier_config.tier_b_snr:
            tier = "B"
    elif best_snr >= tier_config.tier_b_snr:
        tier = "B"

    return TileClassification(
        tier=tier,
        peaks=peak_metrics,
        symmetry_score=symmetry_score,
        n_non_collinear=n_non_collinear,
        best_snr=best_snr,
        best_orientation_deg=best_orientation,
        gate_details={
            "n_paired": n_paired,
            "tier_a_snr_threshold": tier_config.tier_a_snr,
            "tier_b_snr_threshold": tier_config.tier_b_snr,
        },
    )
