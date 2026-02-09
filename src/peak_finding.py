"""
Peak-Finding + Lattice Validation (WS6).

Subpixel peak localisation on bandpass-filtered image (I5).
D-spacing-adaptive separation (C9).
NN-distance lattice validation.

Gate G12: >= 50% of peaks with NN distance within 20% of expected d.
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.signal import windows
from typing import List, Optional, Tuple

from src.fft_coords import FFTGrid
from src.pipeline_config import SubpixelPeak, LatticeValidation
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


# ======================================================================
# Bandpass-filtered peak image (I5)
# ======================================================================

def build_bandpass_image(image_fft: np.ndarray,
                         g_dom_magnitude: Optional[float],
                         fft_grid: FFTGrid,
                         bandwidth_fraction: float = 0.3,
                         effective_q_min: float = 0.0,
                         g_vectors=None,
                         angular_width_deg: float = 30.0,
                         use_directional_mask: bool = True) -> np.ndarray:
    """Build bandpass-filtered image for peak finding (I5).

    Ring mask at |g_dom| with cosine-tapered edges.
    When g_vectors are provided and use_directional_mask is True,
    restricts the ring to angular wedges around each g-vector direction
    (and its antipodal direction).
    Falls back to raw image if no g_dom available.
    """
    if g_dom_magnitude is None or g_dom_magnitude <= 0:
        logger.info("No dominant g-vector, using raw image for peak finding")
        return image_fft.copy()

    H, W = image_fft.shape
    FT = np.fft.fft2(image_fft)
    FT_shifted = np.fft.fftshift(FT)

    q_mag_grid = fft_grid.q_mag_grid()
    q_center = g_dom_magnitude
    q_bandwidth = bandwidth_fraction * q_center
    q_inner = q_center - q_bandwidth
    q_outer = q_center + q_bandwidth
    taper_width = q_bandwidth * 0.3

    # Build ring mask with cosine taper
    ring_mask = np.zeros((H, W))
    flat_inner = q_inner + taper_width
    flat_outer = q_outer - taper_width

    # Flat passband
    flat = (q_mag_grid >= flat_inner) & (q_mag_grid <= flat_outer)
    ring_mask[flat] = 1.0

    # Inner taper
    inner_taper = (q_mag_grid >= q_inner) & (q_mag_grid < flat_inner)
    if taper_width > 0:
        ring_mask[inner_taper] = 0.5 * (1 + np.cos(
            np.pi * (flat_inner - q_mag_grid[inner_taper]) / taper_width))

    # Outer taper
    outer_taper = (q_mag_grid > flat_outer) & (q_mag_grid <= q_outer)
    if taper_width > 0:
        ring_mask[outer_taper] = 0.5 * (1 + np.cos(
            np.pi * (q_mag_grid[outer_taper] - flat_outer) / taper_width))

    # Zero below q_min to prevent DC leakage through cosine taper
    if effective_q_min > 0:
        ring_mask[q_mag_grid < effective_q_min] = 0.0

    # Directional angular wedge mask
    if g_vectors is not None and use_directional_mask and len(g_vectors) > 0:
        angle_grid = fft_grid.angle_grid_deg()  # (-180, 180]
        angular_mask = np.zeros((H, W), dtype=bool)
        half_width = angular_width_deg

        for gv in g_vectors:
            for sign in [1, -1]:
                g_angle = np.degrees(np.arctan2(sign * gv.gy, sign * gv.gx))
                # Angular distance (handle wraparound)
                delta = np.abs(angle_grid - g_angle)
                delta = np.minimum(delta, 360 - delta)
                angular_mask |= (delta <= half_width)

        # Combine: ring AND angular
        ring_mask = ring_mask * angular_mask.astype(float)

    filtered = FT_shifted * ring_mask
    peak_image = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))

    return peak_image


# ======================================================================
# Subpixel peak detection
# ======================================================================

def find_subpixel_peaks(peak_image: np.ndarray,
                         expected_d_nm: float,
                         pixel_size_nm: float,
                         min_prominence: float = 0.1,
                         tile_size: int = 256) -> List[SubpixelPeak]:
    """Find subpixel peaks with d-spacing-adaptive separation (C9).

    min_separation_px = clamp(0.6 * expected_d_nm / pixel_size_nm, min=2, max=tile_size/4)
    """
    d_px = expected_d_nm / pixel_size_nm
    min_sep = max(2, min(tile_size / 4, 0.6 * d_px))
    min_sep_int = max(2, int(round(min_sep)))

    logger.info("Peak finding: min_separation=%.1f px (d_expected=%.3f nm)",
                min_sep, expected_d_nm)

    # Local maximum detection
    footprint_size = 2 * min_sep_int + 1
    footprint = np.ones((footprint_size, footprint_size))
    local_max = ndimage.maximum_filter(peak_image, footprint=footprint)

    # Threshold: min_prominence * local background
    bg_estimate = ndimage.percentile_filter(peak_image, percentile=50, size=max(31, footprint_size * 2))
    threshold = bg_estimate + min_prominence * np.std(peak_image)

    peaks_mask = (peak_image == local_max) & (peak_image > threshold)
    peak_y, peak_x = np.where(peaks_mask)

    if len(peak_y) == 0:
        return []

    peaks = []
    H, W = peak_image.shape

    for py, px in zip(peak_y, peak_x):
        # Subpixel refinement via parabolic fit
        sx, sy = float(px), float(py)

        if 2 <= px < W - 2 and 2 <= py < H - 2:
            # X refinement
            left = peak_image[py, px - 1]
            center = peak_image[py, px]
            right = peak_image[py, px + 1]
            denom = 2 * center - left - right
            if denom > 0:
                sx = px + 0.5 * (left - right) / denom

            # Y refinement
            top = peak_image[py - 1, px]
            bottom = peak_image[py + 1, px]
            denom = 2 * center - top - bottom
            if denom > 0:
                sy = py + 0.5 * (top - bottom) / denom

        # Prominence
        local_bg = bg_estimate[py, px]
        prominence = peak_image[py, px] - local_bg

        peaks.append(SubpixelPeak(
            x=sx, y=sy,
            intensity=float(peak_image[py, px]),
            prominence=float(prominence),
        ))

    logger.info("Found %d peaks", len(peaks))
    return peaks


# ======================================================================
# Lattice validation
# ======================================================================

def validate_peak_lattice(peaks: List[SubpixelPeak],
                           expected_d_nm: float,
                           pixel_size_nm: float,
                           tolerance: float = 0.2,
                           adaptive: bool = True) -> LatticeValidation:
    """Validate detected peaks against expected lattice spacing.

    Checks NN distances against expected d-spacing.
    Gate G12: >= 50% of peaks with NN distance within tolerance of expected d.

    When adaptive=True and >= 10 peaks, tolerance is tightened based on the IQR
    of NN distances (floored at 0.1).
    """
    if len(peaks) < 2:
        result = LatticeValidation(
            nn_distances=np.array([]),
            fraction_valid=0.0,
            mean_nn_distance_nm=0.0,
            std_nn_distance_nm=0.0,
            expected_d_nm=expected_d_nm,
            min_separation_px_used=0,
            tolerance_used=tolerance,
        )
        evaluate_gate("G12", 0.0)
        return result

    positions = np.array([[p.x * pixel_size_nm, p.y * pixel_size_nm] for p in peaks])

    # Compute NN distances
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    nn_dists, _ = tree.query(positions, k=2)  # k=2: self + nearest
    nn_distances = nn_dists[:, 1]  # skip self

    # Adaptive tolerance from IQR
    tolerance_actual = tolerance
    if adaptive and len(peaks) >= 10:
        q75, q25 = np.percentile(nn_distances, [75, 25])
        iqr = q75 - q25
        tolerance_actual = min(tolerance, iqr / expected_d_nm)
        tolerance_actual = max(tolerance_actual, 0.1)
        if tolerance_actual < tolerance:
            logger.info("Adaptive tolerance: %.3f (IQR=%.3f nm, nominal=%.3f)",
                        tolerance_actual, iqr, tolerance)

    # Validate
    valid = np.abs(nn_distances - expected_d_nm) / expected_d_nm <= tolerance_actual
    fraction_valid = float(np.mean(valid))

    result = LatticeValidation(
        nn_distances=nn_distances,
        fraction_valid=fraction_valid,
        mean_nn_distance_nm=float(np.mean(nn_distances)),
        std_nn_distance_nm=float(np.std(nn_distances)),
        expected_d_nm=expected_d_nm,
        min_separation_px_used=max(2, 0.6 * expected_d_nm / pixel_size_nm),
        tolerance_used=tolerance_actual,
    )

    evaluate_gate("G12", fraction_valid)

    logger.info("Lattice validation: %.1f%% valid (mean_nn=%.3f nm, expected=%.3f nm, tol=%.3f)",
                fraction_valid * 100, result.mean_nn_distance_nm, expected_d_nm, tolerance_actual)

    return result
