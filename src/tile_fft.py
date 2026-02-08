"""
Tile-based FFT Computation (WS3).

Refactored from radial_analysis.py. Uses FFTGrid for all coordinate conversions.
Supports ROI-aware processing: skips tiles where ROI coverage < threshold.
GPU-accelerated batched FFT via DeviceContext when available.

Gate G5: tile_size_px / (d_dom_nm / pixel_size_nm) >= 20 periods.
"""

import logging
import gc
import numpy as np
import psutil
from scipy.signal import windows
from scipy import ndimage
from typing import List, Optional, Tuple, TYPE_CHECKING

from src.fft_coords import FFTGrid
from src.pipeline_config import TilePeak, TilePeakSet
from src.fft_features import tile_generator, get_tiling_info, create_2d_hann_window
from src.gates import evaluate_gate

if TYPE_CHECKING:
    from src.gpu_backend import DeviceContext

logger = logging.getLogger(__name__)


def compute_tile_fft(tile: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Compute windowed FFT power spectrum for a tile.

    Returns shifted power spectrum with DC at centre.
    """
    windowed = tile.astype(np.float64) * window
    fft_result = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_result)
    return np.abs(fft_shifted) ** 2


def compute_tile_fft_batch(tiles: np.ndarray, window: np.ndarray,
                           ctx: "DeviceContext") -> np.ndarray:
    """Batch FFT for multiple tiles.

    Parameters
    ----------
    tiles : (B, H, W) float array
    window : (H, W) Hann window
    ctx : DeviceContext

    Returns
    -------
    power : (B, H, W) float64 power spectra (on host)
    """
    tiles_d = ctx.to_device(tiles.astype(np.float64))
    window_d = ctx.to_device(window)
    windowed = tiles_d * window_d          # broadcasts (B,H,W) * (H,W)
    del tiles_d, window_d
    fft_result = ctx.fft2(windowed, axes=(-2, -1))
    del windowed
    fft_shifted = ctx.fftshift(fft_result, axes=(-2, -1))
    del fft_result
    power = ctx.xp.abs(fft_shifted) ** 2
    del fft_shifted
    return ctx.to_host(power)


def extract_tile_peaks(power: np.ndarray,
                       fft_grid: FFTGrid,
                       q_ranges: Optional[List[Tuple[float, float]]] = None,
                       dc_mask_radius: int = 3,
                       local_max_size: int = 5,
                       peak_threshold_frac: float = 0.1,
                       effective_q_min: float = 0.0) -> List[TilePeak]:
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

    # Mask DC: q-based when effective_q_min > 0, pixel-based otherwise
    if effective_q_min > 0:
        q_mag = fft_grid.q_mag_grid()
        dc_mask = q_mag < effective_q_min
    else:
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
                      ctx: Optional["DeviceContext"] = None,
                      effective_q_min: float = 0.0,
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
    ctx : DeviceContext, optional
        When provided and using GPU, tiles are processed in batches.

    Returns
    -------
    (peak_sets, skipped_mask) where skipped_mask[row, col] = True if skipped.
    """
    info = get_tiling_info(image_fft.shape, tile_size, stride)
    n_rows, n_cols = info["grid_shape"]
    n_total = n_rows * n_cols
    window = create_2d_hann_window(tile_size)

    tile_grid = FFTGrid(tile_size, tile_size, fft_grid.pixel_size_nm)

    # Memory logging
    process = psutil.Process()
    log_interval = max(1, n_total // 20)  # Log ~20 times
    initial_mem = process.memory_info().rss / 1024**3
    logger.info(f"MEMORY: Starting tile processing. Initial RSS: {initial_mem:.2f} GB")
    print(f"MEMORY: Starting tile processing ({n_total} tiles). Initial RSS: {initial_mem:.2f} GB")

    # Dispatch to GPU-batched or sequential CPU path
    if ctx is not None and ctx.using_gpu:
        peak_sets, skipped = _process_tiles_gpu(
            image_fft, roi_mask_grid, tile_grid, window,
            tile_size, stride, n_rows, n_cols, q_ranges, ctx,
            process, log_interval, initial_mem,
            effective_q_min=effective_q_min,
        )
    else:
        peak_sets, skipped = _process_tiles_cpu(
            image_fft, roi_mask_grid, tile_grid, window,
            tile_size, stride, n_rows, n_cols, n_total, q_ranges,
            process, log_interval, initial_mem,
            effective_q_min=effective_q_min,
        )

    final_mem = process.memory_info().rss / 1024**3
    logger.info(f"MEMORY: Tile processing complete. Final RSS: {final_mem:.2f} GB (delta: {final_mem - initial_mem:.2f} GB)")
    print(f"MEMORY: Tile processing complete. Final RSS: {final_mem:.2f} GB (delta: +{final_mem - initial_mem:.2f} GB)", flush=True)

    return peak_sets, skipped


def _process_tiles_cpu(image_fft, roi_mask_grid, tile_grid, window,
                       tile_size, stride, n_rows, n_cols, n_total,
                       q_ranges, process, log_interval, initial_mem,
                       effective_q_min=0.0):
    """Sequential CPU tile processing (original path)."""
    peak_sets: List[TilePeakSet] = []
    skipped = np.zeros((n_rows, n_cols), dtype=bool)

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
        peaks = extract_tile_peaks(power, tile_grid, q_ranges=q_ranges,
                                   effective_q_min=effective_q_min)

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

    return peak_sets, skipped


def _process_tiles_gpu(image_fft, roi_mask_grid, tile_grid, window,
                       tile_size, stride, n_rows, n_cols, q_ranges, ctx,
                       process, log_interval, initial_mem,
                       effective_q_min=0.0):
    """Batched GPU tile processing with OOM recovery."""
    n_total = n_rows * n_cols

    # 1. Collect tile positions, separating valid from skipped
    valid_tiles = []   # (row, col, y, x)
    skipped = np.zeros((n_rows, n_cols), dtype=bool)

    for _tile, row, col, y, x in tile_generator(image_fft, tile_size, stride):
        if roi_mask_grid is not None and row < roi_mask_grid.shape[0] and col < roi_mask_grid.shape[1]:
            if not roi_mask_grid[row, col]:
                skipped[row, col] = True
                continue
        valid_tiles.append((row, col, y, x))

    # Pre-allocate result dict keyed by (row, col)
    result_map = {}

    # Add skipped entries
    for row in range(n_rows):
        for col in range(n_cols):
            if skipped[row, col]:
                result_map[(row, col)] = TilePeakSet(peaks=[], tile_row=row, tile_col=col)

    # 2. Auto-size batches
    batch_size = ctx.max_batch_tiles(tile_size)
    logger.info(f"GPU batch size: {batch_size} tiles")

    # 3. Process in batches
    processed_count = 0
    i = 0
    while i < len(valid_tiles):
        batch_entries = valid_tiles[i:i + batch_size]
        b = len(batch_entries)

        # Stack tiles into (B, H, W) array
        tile_stack = np.empty((b, tile_size, tile_size), dtype=np.float64)
        for j, (row, col, y, x) in enumerate(batch_entries):
            tile_stack[j] = image_fft[y:y + tile_size, x:x + tile_size]

        # Try GPU batch FFT
        try:
            power_batch = compute_tile_fft_batch(tile_stack, window, ctx)
        except Exception as e:
            err_name = type(e).__name__
            if "OutOfMemory" in err_name or "MemoryError" in err_name:
                # OOM recovery: halve batch size and retry
                ctx.clear_memory_pool()
                batch_size = max(1, batch_size // 2)
                logger.warning(f"GPU OOM — reducing batch to {batch_size}")
                if batch_size == 1 and b == 1:
                    # Batch=1 also failed; fall back to CPU for this tile
                    logger.warning("GPU batch=1 OOM — falling back to CPU for this tile")
                    row, col, y, x = batch_entries[0]
                    tile_data = image_fft[y:y + tile_size, x:x + tile_size]
                    power = compute_tile_fft(tile_data, window)
                    peaks = extract_tile_peaks(power, tile_grid, q_ranges=q_ranges,
                                               effective_q_min=effective_q_min)
                    result_map[(row, col)] = TilePeakSet(
                        peaks=peaks, tile_row=row, tile_col=col,
                        power_spectrum=power,
                    )
                    i += 1
                    processed_count += 1
                    continue
                # Retry with smaller batch (don't advance i)
                continue
            else:
                raise

        del tile_stack

        # Extract peaks on CPU for each tile in the batch
        for j, (row, col, y, x) in enumerate(batch_entries):
            power = power_batch[j]
            peaks = extract_tile_peaks(power, tile_grid, q_ranges=q_ranges,
                                       effective_q_min=effective_q_min)
            result_map[(row, col)] = TilePeakSet(
                peaks=peaks, tile_row=row, tile_col=col,
                power_spectrum=power,
            )

        del power_batch
        i += b
        processed_count += b

        # Memory logging
        if processed_count % max(1, log_interval) < b:
            mem_gb = process.memory_info().rss / 1024**3
            pct = 100 * processed_count / max(1, len(valid_tiles))
            logger.info(f"MEMORY: GPU batch tile {processed_count}/{len(valid_tiles)} ({pct:.0f}%) - RSS: {mem_gb:.2f} GB")
            print(f"MEMORY: GPU batch tile {processed_count}/{len(valid_tiles)} ({pct:.0f}%) - RSS: {mem_gb:.2f} GB", flush=True)

    # 4. Assemble results in tile_generator order
    peak_sets: List[TilePeakSet] = []
    for _tile, row, col, y, x in tile_generator(image_fft, tile_size, stride):
        peak_sets.append(result_map[(row, col)])

    return peak_sets, skipped
