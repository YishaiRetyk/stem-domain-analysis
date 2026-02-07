"""
Tile-based FFT Computation (WS3).

Refactored from radial_analysis.py. Uses FFTGrid for all coordinate conversions.
Supports ROI-aware processing: skips tiles where ROI coverage < threshold.

Gate G5: tile_size_px / (d_dom_nm / pixel_size_nm) >= 20 periods.
"""

import logging
import gc
import numpy as np
import psutil
from scipy.signal import windows
from scipy import ndimage
from typing import List, Optional, Tuple

from src.fft_coords import FFTGrid
from src.pipeline_config import TilePeak, TilePeakSet
from src.fft_features import tile_generator, get_tiling_info, create_2d_hann_window
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


def compute_tile_fft(tile: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Compute windowed FFT power spectrum for a tile.

    Returns shifted power spectrum with DC at centre.
    """
    windowed = tile.astype(np.float64) * window
    fft_result = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_result)
    return np.abs(fft_shifted) ** 2


def extract_tile_peaks(power: np.ndarray,
                       fft_grid: FFTGrid,
                       q_ranges: Optional[List[Tuple[float, float]]] = None,
                       dc_mask_radius: int = 3,
                       local_max_size: int = 5,
                       peak_threshold_frac: float = 0.1) -> List[TilePeak]:
    """Extract peaks from a tile power spectrum with optional subpixel refinement.

    Parameters
    ----------
    power : np.ndarray
        Shifted power spectrum.
    fft_grid : FFTGrid
        Tile-sized FFTGrid.
    q_ranges : list of (q_min, q_max), optional
        Seeded q-ranges from global FFT. If None, search everywhere.
    dc_mask_radius : int
        Pixels to mask around DC.
    local_max_size : int
        Footprint for local maximum detection.
    peak_threshold_frac : float
        Minimum peak intensity as fraction of max.

    Returns
    -------
    List of TilePeak.
    """
    tile_size = power.shape[0]
    center = fft_grid.dc_x

    # Mask DC
    y, x = np.ogrid[:tile_size, :tile_size]
    dc_mask = ((y - fft_grid.dc_y) ** 2 + (x - fft_grid.dc_x) ** 2) <= dc_mask_radius ** 2

    # When q_ranges are provided, many pixels get zeroed — need a full copy.
    # Otherwise, mask DC in-place and restore after detection.
    needs_copy = q_ranges is not None
    if needs_copy:
        power_masked = power.copy()
        power_masked[dc_mask] = 0
        q_mag = fft_grid.q_mag_grid()
        q_mask = np.zeros_like(power_masked, dtype=bool)
        for q_min, q_max in q_ranges:
            q_mask |= (q_mag >= q_min) & (q_mag <= q_max)
        power_masked[~q_mask] = 0
    else:
        dc_saved = power[dc_mask].copy()  # small (~28 pixels)
        power[dc_mask] = 0
        power_masked = power  # alias, no copy

    # Find local maxima
    footprint = np.ones((local_max_size, local_max_size))
    local_max = ndimage.maximum_filter(power_masked, footprint=footprint)

    max_val = power_masked.max()
    if max_val == 0:
        if not needs_copy:
            power[dc_mask] = dc_saved
        return []

    threshold = peak_threshold_frac * max_val
    peaks_mask = (power_masked == local_max) & (power_masked > threshold) & ~dc_mask

    peak_y, peak_x = np.where(peaks_mask)
    if len(peak_y) == 0:
        if not needs_copy:
            power[dc_mask] = dc_saved
        return []

    peaks = []
    for py, px in zip(peak_y, peak_x):
        # Subpixel refinement (parabolic interpolation)
        sx, sy = _subpixel_refine(power_masked, int(px), int(py))

        qx, qy = fft_grid.px_to_q(sx, sy)
        q_m = np.sqrt(qx ** 2 + qy ** 2)
        d = 1.0 / q_m if q_m > 1e-10 else 0
        angle = float(np.degrees(np.arctan2(qy, qx))) % 180

        peaks.append(TilePeak(
            qx=qx, qy=qy,
            q_mag=q_m,
            d_spacing=d,
            angle_deg=angle,
            intensity=float(power[int(py), int(px)]),
            fwhm=0.0,  # filled later by fft_peak_detection
        ))

    # Restore in-place DC modification
    if not needs_copy:
        power[dc_mask] = dc_saved

    return peaks


def _subpixel_refine(power: np.ndarray, px: int, py: int) -> Tuple[float, float]:
    """Parabolic subpixel peak refinement."""
    h, w = power.shape
    sx, sy = float(px), float(py)

    # X direction
    if 1 <= px < w - 1:
        left = power[py, px - 1]
        centre = power[py, px]
        right = power[py, px + 1]
        denom = 2 * centre - left - right
        if denom > 0:
            sx = px + 0.5 * (left - right) / denom

    # Y direction
    if 1 <= py < h - 1:
        top = power[py - 1, px]
        centre = power[py, px]
        bottom = power[py + 1, px]
        denom = 2 * centre - top - bottom
        if denom > 0:
            sy = py + 0.5 * (top - bottom) / denom

    return sx, sy


def check_tiling_adequacy(tile_size: int, d_dom_nm: float,
                          pixel_size_nm: float) -> Tuple[float, bool]:
    """Gate G5: check periods per tile.

    Returns (periods, passed).
    """
    d_px = d_dom_nm / pixel_size_nm
    if d_px < 1e-10:
        return 0, False
    periods = tile_size / d_px
    g5_result = evaluate_gate("G5", periods)
    return periods, g5_result.passed


def process_all_tiles(image_fft: np.ndarray,
                      roi_mask_grid: Optional[np.ndarray],
                      fft_grid: FFTGrid,
                      tile_size: int,
                      stride: int,
                      q_ranges: Optional[List[Tuple[float, float]]] = None,
                      ) -> Tuple[List[TilePeakSet], np.ndarray]:
    """Process all tiles, extracting peaks from each.

    Parameters
    ----------
    image_fft : np.ndarray
        Branch A image.
    roi_mask_grid : np.ndarray[bool] or None
        (n_rows, n_cols) mask. Tiles where False are skipped.
    fft_grid : FFTGrid
        Full-image grid (used only for pixel_size).
    tile_size, stride : int
    q_ranges : optional seeded q-ranges.

    Returns
    -------
    (peak_sets, skipped_mask) where skipped_mask[row, col] = True if skipped.
    """
    info = get_tiling_info(image_fft.shape, tile_size, stride)
    n_rows, n_cols = info["grid_shape"]
    n_total = n_rows * n_cols
    window = create_2d_hann_window(tile_size)

    tile_grid = FFTGrid(tile_size, tile_size, fft_grid.pixel_size_nm)
    peak_sets: List[TilePeakSet] = []
    skipped = np.zeros((n_rows, n_cols), dtype=bool)

    # Memory logging
    process = psutil.Process()
    log_interval = max(1, n_total // 20)  # Log ~20 times
    initial_mem = process.memory_info().rss / 1024**3
    logger.info(f"MEMORY: Starting tile processing. Initial RSS: {initial_mem:.2f} GB")
    print(f"MEMORY: Starting tile processing ({n_total} tiles). Initial RSS: {initial_mem:.2f} GB")

    tile_count = 0
    for tile, row, col, y, x in tile_generator(image_fft, tile_size, stride):
        tile_count += 1
        
        # ROI check
        if roi_mask_grid is not None and row < roi_mask_grid.shape[0] and col < roi_mask_grid.shape[1]:
            if not roi_mask_grid[row, col]:
                skipped[row, col] = True
                peak_sets.append(TilePeakSet(peaks=[], tile_row=row, tile_col=col))
                continue

        power = compute_tile_fft(tile, window)
        peaks = extract_tile_peaks(power, tile_grid, q_ranges=q_ranges)

        peak_sets.append(TilePeakSet(
            peaks=peaks,
            tile_row=row,
            tile_col=col,
            power_spectrum=power,
        ))

        # Log memory periodically
        if tile_count % log_interval == 0:
            mem_gb = process.memory_info().rss / 1024**3
            pct = 100 * tile_count / n_total
            logger.info(f"MEMORY: Tile {tile_count}/{n_total} ({pct:.0f}%) - RSS: {mem_gb:.2f} GB")
            print(f"MEMORY: Tile {tile_count}/{n_total} ({pct:.0f}%) - RSS: {mem_gb:.2f} GB", flush=True)

    final_mem = process.memory_info().rss / 1024**3
    logger.info(f"MEMORY: Tile processing complete. Final RSS: {final_mem:.2f} GB (delta: {final_mem - initial_mem:.2f} GB)")
    print(f"MEMORY: Tile processing complete. Final RSS: {final_mem:.2f} GB (delta: +{final_mem - initial_mem:.2f} GB)", flush=True)

    return peak_sets, skipped
