"""
Global FFT + Multi-Ring G-Vector Extraction (WS2, B4).

Full-image FFT, radial profile, background fit, multi-ring g-vector extraction
with polar resampling, antipodal pairing, harmonic de-duplication.

Gate G4: >= 1 peak with SNR >= 3.0.
"""

import logging
import numpy as np
from scipy.signal import windows, find_peaks
from scipy.ndimage import map_coordinates
from typing import List, Optional, TYPE_CHECKING

from src.fft_coords import FFTGrid
from src.pipeline_config import (
    GVector, GlobalPeak, GlobalFFTResult, GlobalFFTConfig,
)
from src.gates import evaluate_gate

if TYPE_CHECKING:
    from src.gpu_backend import DeviceContext

logger = logging.getLogger(__name__)


def _next_power_of_2(n: int) -> int:
    """Return the largest power of 2 <= n."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _angular_distance_deg(a1: float, a2: float) -> float:
    """Smallest angular distance between two angles in degrees."""
    d = abs(a1 - a2) % 360
    return min(d, 360 - d)


def compute_global_fft(image_fft: np.ndarray,
                       fft_grid: FFTGrid,
                       config: GlobalFFTConfig = None,
                       ctx: Optional["DeviceContext"] = None,
                       effective_q_min: float = 0.0) -> GlobalFFTResult:
    """Run full-image FFT and extract g-vectors.

    Parameters
    ----------
    image_fft : np.ndarray
        Branch A (FFT-safe) preprocessed image.
    fft_grid : FFTGrid
        Canonical coordinate system for the full image.
    config : GlobalFFTConfig, optional

    Returns
    -------
    GlobalFFTResult
    """
    if config is None:
        config = GlobalFFTConfig()

    H, W = image_fft.shape
    diagnostics = {"frequency_unit": "cycles/nm"}

    # Cap image size (centre crop to power-of-2)
    max_size = config.max_image_size
    crop_h = min(_next_power_of_2(H), max_size)
    crop_w = min(_next_power_of_2(W), max_size)
    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    cropped = image_fft[y0:y0 + crop_h, x0:x0 + crop_w]

    diagnostics["crop_size"] = [crop_h, crop_w]

    # Hann window
    window = np.outer(windows.hann(crop_h), windows.hann(crop_w))

    # FFT (GPU or CPU)
    if ctx is not None and ctx.using_gpu:
        cropped_d = ctx.to_device(cropped.astype(np.float64))
        window_d = ctx.to_device(window)
        windowed_d = cropped_d * window_d
        del cropped_d, window_d
        fft_result = ctx.fft2(windowed_d)
        del windowed_d
        fft_shifted = ctx.fftshift(fft_result)
        del fft_result
        power_d = ctx.xp.abs(fft_shifted) ** 2
        del fft_shifted
        power = ctx.to_host(power_d)
        del power_d
    else:
        windowed = cropped.astype(np.float64) * window
        fft_result = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft_result)
        power = np.abs(fft_shifted) ** 2

    # Build FFTGrid for the cropped image
    crop_grid = FFTGrid(crop_h, crop_w, fft_grid.pixel_size_nm)

    # Radial profile
    q_mag = crop_grid.q_mag_grid()
    r_px = np.sqrt(
        (np.arange(crop_w)[np.newaxis, :] - crop_grid.dc_x) ** 2 +
        (np.arange(crop_h)[:, np.newaxis] - crop_grid.dc_y) ** 2
    )
    r_int = r_px.astype(int)
    max_r = min(crop_grid.dc_x, crop_grid.dc_y, r_int.max())

    r_flat = r_int.ravel()
    valid = r_flat <= max_r
    radial_sum = np.bincount(r_flat[valid], weights=power.ravel()[valid],
                             minlength=max_r + 1)
    radial_count = np.bincount(r_flat[valid], minlength=max_r + 1).astype(float)
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count

    q_scale = crop_grid.qx_scale  # same as qy_scale for square crops
    q_values = np.arange(max_r + 1) * q_scale

    # Background fit (reuse logic from peak_discovery)
    bg_degree = min(config.background_default_degree, config.background_max_degree)
    background, baseline_model = _fit_background(
        q_values, radial_profile,
        degree=bg_degree,
        effective_q_min=effective_q_min,
        q_fit_min=config.q_fit_min,
        bg_reweight_iterations=config.bg_reweight_iterations,
        bg_reweight_downweight=config.bg_reweight_downweight,
    )
    corrected = radial_profile - background
    diagnostics["baseline_model"] = baseline_model

    # Noise floor
    neg_vals = corrected[corrected < 0]
    if len(neg_vals) > 10:
        noise_floor = float(np.std(neg_vals) * 1.4826)
    else:
        noise_floor = float(np.std(corrected[corrected < np.median(corrected)]))

    # Find radial peaks
    radial_peaks = _find_radial_peaks(q_values, corrected, noise_floor,
                                       min_snr=config.min_peak_snr,
                                       effective_q_min=effective_q_min,
                                       savgol_window_max=config.savgol_window_max,
                                       savgol_window_min=config.savgol_window_min,
                                       savgol_polyorder=config.savgol_polyorder,
                                       radial_peak_distance=config.radial_peak_distance,
                                       radial_peak_width=config.radial_peak_width)
    diagnostics["n_radial_peaks"] = len(radial_peaks)
    diagnostics["effective_q_min"] = effective_q_min

    # Background residual diagnostics
    peak_indices = [rp["index"] for rp in radial_peaks]
    bg_diag = _compute_background_diagnostics(
        q_values, radial_profile, background, corrected,
        config.q_fit_min, peak_indices,
    )
    diagnostics["background_diagnostics"] = bg_diag

    # Convert to GlobalPeak list
    peaks = []
    for p in radial_peaks:
        peaks.append(GlobalPeak(
            q_center=p["q_center"],
            q_fwhm=p["q_width"],
            d_spacing=p["d_spacing"],
            intensity=p["intensity"],
            prominence=p["prominence"],
            snr=p["snr"],
            index=p["index"],
        ))

    # Dominant d-spacing
    d_dom = None
    if peaks:
        best = max(peaks, key=lambda pk: pk.snr)
        d_dom = best.d_spacing

    # Multi-ring g-vector extraction (B4)
    g_vectors = extract_all_g_vectors(
        power, radial_peaks, crop_grid,
        top_k=config.max_g_vectors,
        config=config,
    )

    # Information limit (highest q with detectable signal)
    info_limit_q = None
    if peaks:
        info_limit_q = max(p.q_center + p.q_fwhm for p in peaks)

    # Gate G4
    best_peak_snr = max((p.snr for p in peaks), default=0)
    g4_result = evaluate_gate("G4", best_peak_snr)
    diagnostics["g4_passed"] = g4_result.passed
    diagnostics["g4_reason"] = g4_result.reason

    # FFT guidance strength classification
    n_peaks = len(peaks)
    if best_peak_snr >= config.strong_guidance_snr and n_peaks >= 2:
        fft_guidance = "strong"
    elif best_peak_snr >= config.min_peak_snr:
        fft_guidance = "weak"
    else:
        fft_guidance = "none"

    return GlobalFFTResult(
        power_spectrum=power,
        radial_profile=radial_profile,
        q_values=q_values,
        background=background,
        corrected_profile=corrected,
        noise_floor=noise_floor,
        peaks=peaks,
        g_vectors=g_vectors,
        d_dom=d_dom,
        information_limit_q=info_limit_q,
        diagnostics=diagnostics,
        fft_guidance_strength=fft_guidance,
    )


# ======================================================================
# Background fitting (adapted from peak_discovery.fit_background)
# ======================================================================

def _fit_background(q_values: np.ndarray, profile: np.ndarray,
                    degree: int = 6, exclude_dc: int = 5,
                    effective_q_min: float = 0.0,
                    q_fit_min: float = 0.0,
                    bg_reweight_iterations: int = 4,
                    bg_reweight_downweight: float = 0.1) -> tuple:
    """Iterative reweighted polynomial background fit in log space.

    Returns
    -------
    background : np.ndarray
        Background estimate, same length as *profile*.
    baseline_model : dict
        Serialisable description of the fitted model.
    """
    # Cap polynomial degree at 4
    degree = min(degree, 4)

    # Build validity mask: positive values, above effective_q_min, above q_fit_min
    if effective_q_min > 0 or q_fit_min > 0:
        actual_min = max(effective_q_min, q_fit_min)
        valid = (profile > 0) & (q_values >= actual_min)
    else:
        valid = (profile > 0) & (np.arange(len(profile)) >= exclude_dc)
    if np.sum(valid) < degree + 1:
        baseline_model = {
            "type": "poly", "degree": degree,
            "q_fit_min": float(q_fit_min), "coeffs": [],
        }
        return np.zeros_like(profile), baseline_model

    q_fit = q_values[valid]
    log_profile = np.log10(profile[valid] + 1e-10)
    weights = np.ones_like(log_profile)

    _bg_reweight_iterations = bg_reweight_iterations
    _bg_reweight_downweight = bg_reweight_downweight
    coeffs = None
    for _ in range(_bg_reweight_iterations):
        coeffs = np.polyfit(q_fit, log_profile, degree, w=weights)
        fit = np.polyval(coeffs, q_fit)
        residual = log_profile - fit
        mad = np.median(np.abs(residual - np.median(residual)))
        threshold = 2.0 * mad * 1.4826
        weights = np.where(residual > threshold, _bg_reweight_downweight, 1.0)

    background = np.zeros_like(profile)
    if coeffs is not None:
        background[valid] = 10 ** np.polyval(coeffs, q_values[valid])
    # Fill excluded region with raw profile values
    actual_excl = max(effective_q_min, q_fit_min)
    if actual_excl > 0:
        excluded = q_values < actual_excl
        background[excluded] = profile[excluded]
    else:
        background[:exclude_dc] = profile[:exclude_dc]

    baseline_model = {
        "type": "poly",
        "degree": int(degree),
        "q_fit_min": float(q_fit_min),
        "coeffs": coeffs.tolist() if coeffs is not None else [],
    }
    return background, baseline_model


def _compute_background_diagnostics(q_values, profile, background, corrected,
                                     q_fit_min, peak_positions):
    """Compute residual diagnostics for the background fit.

    Parameters
    ----------
    q_values : np.ndarray
    profile, background, corrected : np.ndarray
    q_fit_min : float
        Lower bound of the fit region (cycles/nm).
    peak_positions : list of int
        Indices into *q_values* of detected radial peaks.

    Returns
    -------
    dict with keys: median_abs_residual, mad_residual,
    neg_excursion_fraction_near_peaks.
    """
    # Residual in the fit region (linear space)
    fit_mask = q_values >= q_fit_min if q_fit_min > 0 else np.ones(len(q_values), dtype=bool)
    residual = (profile - background)[fit_mask]

    if len(residual) == 0:
        return {
            "median_abs_residual": 0.0,
            "mad_residual": 0.0,
            "neg_excursion_fraction_near_peaks": 0.0,
        }

    abs_res = np.abs(residual)
    median_abs = float(np.median(abs_res))
    mad_res = float(np.median(np.abs(abs_res - np.median(abs_res))) * 1.4826)

    # Negative excursion fraction near peaks (within +/-2 bins)
    near_peak_mask = np.zeros(len(q_values), dtype=bool)
    for idx in peak_positions:
        lo = max(0, idx - 2)
        hi = min(len(q_values), idx + 3)
        near_peak_mask[lo:hi] = True

    near_peak_residual = corrected[near_peak_mask]
    if len(near_peak_residual) > 0:
        neg_frac = float(np.sum(near_peak_residual < 0) / len(near_peak_residual))
    else:
        neg_frac = 0.0

    return {
        "median_abs_residual": median_abs,
        "mad_residual": mad_res,
        "neg_excursion_fraction_near_peaks": neg_frac,
    }


# ======================================================================
# Radial peak finding (adapted from peak_discovery.find_diffraction_peaks)
# ======================================================================

def _find_radial_peaks(q_values, corrected, noise_floor,
                       min_q=0.5, max_q=15.0, min_snr=2.0,
                       effective_q_min: float = 0.0,
                       savgol_window_max: int = 11,
                       savgol_window_min: int = 5,
                       savgol_polyorder: int = 2,
                       radial_peak_distance: int = 5,
                       radial_peak_width: int = 2):
    """Find peaks in the background-corrected radial profile."""
    from scipy.signal import savgol_filter

    actual_min_q = max(min_q, effective_q_min)
    valid_mask = (q_values >= actual_min_q) & (q_values <= max_q)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 10:
        return []

    wl = min(savgol_window_max, len(valid_idx) // 2 * 2 + 1)
    if wl < savgol_window_min:
        wl = savgol_window_min
    smooth = savgol_filter(corrected[valid_mask], window_length=wl, polyorder=savgol_polyorder)

    min_prominence = noise_floor * min_snr
    peaks_idx, props = find_peaks(smooth, prominence=max(min_prominence, 1e-10),
                                  width=radial_peak_width, distance=radial_peak_distance)

    results = []
    for i, li in enumerate(peaks_idx):
        gi = valid_idx[li]
        pq = q_values[gi]
        pi = corrected[gi]
        prom = props["prominences"][i]
        w = props["widths"][i] if "widths" in props else 3
        q_step = q_values[1] - q_values[0] if len(q_values) > 1 else 0.01
        results.append({
            "q_center": float(pq),
            "q_width": float(w * q_step),
            "d_spacing": float(1.0 / pq) if pq > 0 else 0,
            "intensity": float(pi),
            "prominence": float(prom),
            "snr": float(prom / noise_floor) if noise_floor > 0 else 0,
            "index": int(gi),
        })

    results.sort(key=lambda x: x["prominence"], reverse=True)
    return results


# ======================================================================
# Multi-ring g-vector extraction (B4)
# ======================================================================

def extract_all_g_vectors(power_spectrum: np.ndarray,
                          radial_peaks: list,
                          fft_grid: FFTGrid,
                          top_k: int = 6,
                          config: GlobalFFTConfig = None) -> List[GVector]:
    """Multi-ring g-vector extraction with harmonic de-duplication.

    For each radial peak ring:
    1. Polar resampling -> angular profile
    2. Angular peak detection
    3. Cartesian antipodal pairing
    Then cross-ring harmonic de-duplication and top-K selection.
    """
    if config is None:
        config = GlobalFFTConfig()

    q_width_expansion = config.q_width_expansion_frac
    harmonic_ratio_tol = config.harmonic_ratio_tol
    harmonic_angle_tol = config.harmonic_angle_tol_deg
    harmonic_snr_ratio = config.harmonic_snr_ratio
    nc_min_angle = config.non_collinear_min_angle_deg
    angular_prom_frac = config.angular_prominence_frac
    angular_peak_dist = config.angular_peak_distance

    all_candidates: List[GVector] = []

    for ring_idx, rp in enumerate(radial_peaks[:8]):  # cap at 8 rings
        q_center = rp["q_center"]
        q_fwhm = rp["q_width"]
        q_width = max(q_fwhm * 2, q_center * q_width_expansion)

        ring_gvecs = _extract_g_vectors_single_ring(
            power_spectrum, q_center, q_width, fft_grid, ring_idx,
            ring_snr=rp.get("snr", 0),
            ring_fwhm=rp.get("q_width", 0.1),
            angular_prominence_frac=angular_prom_frac,
            angular_peak_distance=angular_peak_dist,
        )
        all_candidates.extend(ring_gvecs)

    if not all_candidates:
        return []

    # Cross-ring harmonic de-duplication
    all_candidates.sort(key=lambda v: v.magnitude)
    deduplicated: List[GVector] = []

    for v in all_candidates:
        is_harmonic = False
        for i, existing in enumerate(deduplicated):
            ratio = v.magnitude / existing.magnitude
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < harmonic_ratio_tol:
                if _angular_distance_deg(v.angle_deg, existing.angle_deg) < harmonic_angle_tol:
                    if v.snr > existing.snr * harmonic_snr_ratio:
                        deduplicated[i] = v
                    is_harmonic = True
                    break
        if not is_harmonic:
            deduplicated.append(v)

    # Top-K non-collinear selection
    deduplicated.sort(key=lambda v: v.snr, reverse=True)
    selected: List[GVector] = []
    for v in deduplicated:
        if len(selected) >= top_k:
            break
        if all(_angular_distance_deg(v.angle_deg, s.angle_deg) > nc_min_angle for s in selected):
            selected.append(v)

    return selected


def _extract_g_vectors_single_ring(power_spectrum, q_center, q_width,
                                    fft_grid: FFTGrid, ring_idx: int,
                                    ring_snr: float = 0,
                                    ring_fwhm: float = 0.1,
                                    angular_prominence_frac: float = 0.5,
                                    angular_peak_distance: int = 10) -> List[GVector]:
    """Extract g-vectors from a single q-ring via polar resampling."""
    H, W = power_spectrum.shape
    n_angle = 360

    # Radial range in pixels
    r_min_px = max(1, (q_center - q_width) / fft_grid.qx_scale)
    r_max_px = (q_center + q_width) / fft_grid.qx_scale
    n_radial = max(3, int(round(r_max_px - r_min_px)))

    angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    radii = np.linspace(r_min_px, r_max_px, n_radial)

    # Build angular profile by summing over radial bins
    angular_profile = np.zeros(n_angle)
    for i, angle in enumerate(angles):
        for r in radii:
            x = fft_grid.dc_x + r * np.cos(angle)
            y = fft_grid.dc_y + r * np.sin(angle)
            # Bilinear interpolation
            if 0 <= y < H - 1 and 0 <= x < W - 1:
                val = map_coordinates(power_spectrum, [[y], [x]],
                                      order=1, mode='nearest')[0]
                angular_profile[i] += val

    if np.max(angular_profile) < 1e-10:
        return []

    # Detect angular peaks
    med_profile = np.median(angular_profile)
    peaks_idx, props = find_peaks(
        angular_profile,
        prominence=max(med_profile * angular_prominence_frac, 1e-10),
        distance=angular_peak_distance,
    )
    if len(peaks_idx) == 0:
        return []

    # Cartesian antipodal pairing (B4)
    paired_vectors: List[GVector] = []
    used = set()
    tolerance_q = max(ring_fwhm * 2.0 / 2.355, q_center * 0.03)

    for i_idx in peaks_idx:
        if i_idx in used:
            continue
        angle_i = angles[i_idx]
        gx_i = q_center * np.cos(angle_i)
        gy_i = q_center * np.sin(angle_i)

        best_partner = None
        best_residual = float('inf')

        for j_idx in peaks_idx:
            if j_idx == i_idx or j_idx in used:
                continue
            gx_j = q_center * np.cos(angles[j_idx])
            gy_j = q_center * np.sin(angles[j_idx])
            residual = np.sqrt((gx_i + gx_j) ** 2 + (gy_i + gy_j) ** 2)
            if residual < tolerance_q and residual < best_residual:
                best_partner = j_idx
                best_residual = residual

        if best_partner is not None:
            used.add(i_idx)
            used.add(best_partner)

            # SNR from angular profile peak
            peak_val = angular_profile[i_idx]
            bg_vals = angular_profile[angular_profile < np.percentile(angular_profile, 50)]
            bg_med = np.median(bg_vals) if len(bg_vals) > 0 else 0
            bg_mad = np.median(np.abs(bg_vals - bg_med)) * 1.4826 if len(bg_vals) > 0 else 1
            snr = float((peak_val - bg_med) / (bg_mad + 1e-10))

            angle_deg = float(np.degrees(angle_i))
            paired_vectors.append(GVector(
                gx=float(gx_i),
                gy=float(gy_i),
                magnitude=float(q_center),
                angle_deg=angle_deg,
                d_spacing=float(1.0 / q_center) if q_center > 0 else 0,
                snr=snr,
                fwhm=ring_fwhm,
                ring_index=ring_idx,
            ))

    return paired_vectors
