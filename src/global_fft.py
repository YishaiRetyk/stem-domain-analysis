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
from typing import List, Optional

from src.fft_coords import FFTGrid
from src.pipeline_config import (
    GVector, GlobalPeak, GlobalFFTResult, GlobalFFTConfig,
)
from src.gates import evaluate_gate

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
                       config: GlobalFFTConfig = None) -> GlobalFFTResult:
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
    windowed = cropped.astype(np.float64) * window

    # FFT
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

    radial_sum = np.zeros(max_r + 1)
    radial_count = np.zeros(max_r + 1)
    for ri in range(max_r + 1):
        m = r_int == ri
        radial_sum[ri] = np.sum(power[m])
        radial_count[ri] = np.sum(m)
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count

    q_scale = crop_grid.qx_scale  # same as qy_scale for square crops
    q_values = np.arange(max_r + 1) * q_scale

    # Background fit (reuse logic from peak_discovery)
    background = _fit_background(q_values, radial_profile,
                                 degree=config.background_poly_degree)
    corrected = radial_profile - background

    # Noise floor
    neg_vals = corrected[corrected < 0]
    if len(neg_vals) > 10:
        noise_floor = float(np.std(neg_vals) * 1.4826)
    else:
        noise_floor = float(np.std(corrected[corrected < np.median(corrected)]))

    # Find radial peaks
    radial_peaks = _find_radial_peaks(q_values, corrected, noise_floor,
                                       min_snr=config.min_peak_snr)
    diagnostics["n_radial_peaks"] = len(radial_peaks)

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
    )


# ======================================================================
# Background fitting (adapted from peak_discovery.fit_background)
# ======================================================================

def _fit_background(q_values: np.ndarray, profile: np.ndarray,
                    degree: int = 6, exclude_dc: int = 5) -> np.ndarray:
    """Iterative reweighted polynomial background fit in log space."""
    valid = (profile > 0) & (np.arange(len(profile)) >= exclude_dc)
    if np.sum(valid) < degree + 1:
        return np.zeros_like(profile)

    q_fit = q_values[valid]
    log_profile = np.log10(profile[valid] + 1e-10)
    weights = np.ones_like(log_profile)

    coeffs = None
    for _ in range(3):
        coeffs = np.polyfit(q_fit, log_profile, degree, w=weights)
        fit = np.polyval(coeffs, q_fit)
        residual = log_profile - fit
        mad = np.median(np.abs(residual - np.median(residual)))
        threshold = 2.0 * mad * 1.4826
        weights = np.where(residual > threshold, 0.1, 1.0)

    background = np.zeros_like(profile)
    if coeffs is not None:
        background[valid] = 10 ** np.polyval(coeffs, q_values[valid])
    background[:exclude_dc] = profile[:exclude_dc]
    return background


# ======================================================================
# Radial peak finding (adapted from peak_discovery.find_diffraction_peaks)
# ======================================================================

def _find_radial_peaks(q_values, corrected, noise_floor,
                       min_q=0.5, max_q=15.0, min_snr=2.0):
    """Find peaks in the background-corrected radial profile."""
    from scipy.signal import savgol_filter

    valid_mask = (q_values >= min_q) & (q_values <= max_q)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 10:
        return []

    wl = min(11, len(valid_idx) // 2 * 2 + 1)
    if wl < 5:
        wl = 5
    smooth = savgol_filter(corrected[valid_mask], window_length=wl, polyorder=2)

    min_prominence = noise_floor * min_snr
    peaks_idx, props = find_peaks(smooth, prominence=max(min_prominence, 1e-10),
                                  width=2, distance=5)

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
                          top_k: int = 6) -> List[GVector]:
    """Multi-ring g-vector extraction with harmonic de-duplication.

    For each radial peak ring:
    1. Polar resampling -> angular profile
    2. Angular peak detection
    3. Cartesian antipodal pairing
    Then cross-ring harmonic de-duplication and top-K selection.
    """
    all_candidates: List[GVector] = []

    for ring_idx, rp in enumerate(radial_peaks[:8]):  # cap at 8 rings
        q_center = rp["q_center"]
        q_fwhm = rp["q_width"]
        q_width = max(q_fwhm * 2, q_center * 0.03)

        ring_gvecs = _extract_g_vectors_single_ring(
            power_spectrum, q_center, q_width, fft_grid, ring_idx,
            ring_snr=rp.get("snr", 0),
            ring_fwhm=rp.get("q_width", 0.1),
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
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < 0.05:
                if _angular_distance_deg(v.angle_deg, existing.angle_deg) < 5:
                    if v.snr > existing.snr * 2:
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
        if all(_angular_distance_deg(v.angle_deg, s.angle_deg) > 15 for s in selected):
            selected.append(v)

    return selected


def _extract_g_vectors_single_ring(power_spectrum, q_center, q_width,
                                    fft_grid: FFTGrid, ring_idx: int,
                                    ring_snr: float = 0,
                                    ring_fwhm: float = 0.1) -> List[GVector]:
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
        prominence=max(med_profile * 0.5, 1e-10),
        distance=10,
    )
    if len(peaks_idx) == 0:
        return []

    # Cartesian antipodal pairing (B4)
    paired_vectors: List[GVector] = []
    used = set()
    tolerance_q = q_center * 0.05

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
