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
from src.pipeline_config import TilePeak, TilePeakSet, TileFFTConfig, DCMaskConfig
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
                       effective_q_min: float = 0.0,
                       tile_fft_config: Optional[TileFFTConfig] = None,
                       dynamic_dc_q: float = 0.0,
                       dc_mask_config: Optional[DCMaskConfig] = None,
                       ) -> List[TilePeak]:
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
        Pixels to mask around DC (deprecated when tile_fft_config provided).
    local_max_size : int
        Footprint for local maximum detection.
    peak_threshold_frac : float
        Minimum peak intensity as fraction of max (secondary floor when
        tile_fft_config with peak_snr_threshold > 0 is provided).
    effective_q_min : float
        Low-q exclusion threshold (cycles/nm).
    tile_fft_config : TileFFTConfig, optional
        When provided, uses q_dc_min and peak_snr_threshold for unified
        DC suppression and SNR-first peak detection.
    dynamic_dc_q : float
        Global dynamic DC mask radius (cycles/nm). Tiles reuse global
        estimate. 0.0 = disabled.
    dc_mask_config : DCMaskConfig, optional
        When provided with ``soft_taper=True``, applies cosine taper
        instead of hard zeroing.

    Returns
    -------
    List of TilePeak.
    """
    tile_size = power.shape[0]

    # Deprecation warning: dc_mask_radius with tile_fft_config
    if tile_fft_config is not None and dc_mask_radius != 3:
        import warnings
        warnings.warn(
            "dc_mask_radius is deprecated when tile_fft_config is provided; "
            "use TileFFTConfig.q_dc_min instead",
            DeprecationWarning,
            stacklevel=2,
        )

    # Unified DC suppression: single path for q-based and pixel-based masking
    q_mag = fft_grid.q_mag_grid()
    q_dc_min = (max(effective_q_min, tile_fft_config.q_dc_min)
                if tile_fft_config is not None else effective_q_min)
    # Incorporate global dynamic DC estimate (tiles reuse global value)
    if dynamic_dc_q > 0:
        q_dc_min = max(q_dc_min, dynamic_dc_q)
    if q_dc_min > 0:
        dc_mask = q_mag < q_dc_min
    else:
        y, x = np.ogrid[:tile_size, :tile_size]
        dc_mask = ((y - fft_grid.dc_y) ** 2 + (x - fft_grid.dc_x) ** 2) <= dc_mask_radius ** 2

    # Soft taper mask (when dc_mask_config requests cosine taper)
    use_soft_taper = (dc_mask_config is not None and dc_mask_config.soft_taper
                      and q_dc_min > 0)

    # Use local_max_size from config when available
    if tile_fft_config is not None:
        local_max_size = tile_fft_config.local_max_size

    # When q_ranges are provided, many pixels get zeroed — need a full copy.
    # Otherwise, mask DC in-place and restore after detection.
    needs_copy = q_ranges is not None
    _restore_dc = False  # whether power was modified in-place and needs restoring
    ring_id_map = None
    if needs_copy:
        power_masked = power.copy()
        if use_soft_taper:
            from src.global_fft import build_dc_taper_mask
            taper = build_dc_taper_mask(q_mag, q_dc_min, soft_taper=True,
                                         taper_width_q=dc_mask_config.taper_width_q)
            power_masked *= taper
        else:
            power_masked[dc_mask] = 0
        q_mask = np.zeros_like(power_masked, dtype=bool)
        ring_id_map = np.full(power.shape, -1, dtype=np.int32)
        for idx, qr in enumerate(q_ranges):
            q_lo, q_hi = qr[0], qr[1]
            ring_id = qr[2] if len(qr) >= 3 else idx
            in_range = (q_mag >= q_lo) & (q_mag <= q_hi)
            ring_id_map[in_range & (ring_id_map == -1)] = ring_id
            q_mask |= in_range
        power_masked[~q_mask] = 0
    else:
        if use_soft_taper:
            from src.global_fft import build_dc_taper_mask
            taper = build_dc_taper_mask(q_mag, q_dc_min, soft_taper=True,
                                         taper_width_q=dc_mask_config.taper_width_q)
            power_masked = power * taper  # new array; power is untouched
        else:
            dc_saved = power[dc_mask].copy()  # small (~28 pixels)
            power[dc_mask] = 0
            power_masked = power  # alias, no copy
            _restore_dc = True

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

    # Build candidate peaks
    candidates = []
    for py, px in zip(peak_y, peak_x):
        # Subpixel refinement (parabolic interpolation)
        sx, sy = _subpixel_refine(power_masked, int(px), int(py))

        qx, qy = fft_grid.px_to_q(sx, sy)
        q_m = np.sqrt(qx ** 2 + qy ** 2)
        d = 1.0 / q_m if q_m > 1e-10 else 0
        angle = float(np.degrees(np.arctan2(qy, qx))) % 180

        assigned_ring = int(ring_id_map[int(py), int(px)]) if ring_id_map is not None else -1
        candidates.append((TilePeak(
            qx=qx, qy=qy,
            q_mag=q_m,
            d_spacing=d,
            angle_deg=angle,
            intensity=float(power[int(py), int(px)]),
            fwhm=0.0,  # filled later by fft_peak_detection
            ring_index=assigned_ring,
        ), int(py), int(px)))

    # SNR-first peak detection: reject weak candidates when tile_fft_config
    # provides a positive peak_snr_threshold.
    use_snr_filter = (tile_fft_config is not None
                      and tile_fft_config.peak_snr_threshold > 0)
    if use_snr_filter:
        peaks = []
        lw_method = tile_fft_config.lightweight_snr_method if tile_fft_config is not None else "ratio"
        for peak, py, px in candidates:
            snr = _lightweight_peak_snr(power_masked, q_mag, peak.q_mag,
                                        py, px, dc_mask,
                                        annulus_inner_factor=tile_fft_config.annulus_inner_factor,
                                        annulus_outer_factor=tile_fft_config.annulus_outer_factor,
                                        background_disk_r_sq=tile_fft_config.background_disk_r_sq,
                                        lightweight_snr_method=lw_method)
            if snr >= tile_fft_config.peak_snr_threshold:
                peaks.append(peak)
    else:
        peaks = [peak for peak, _py, _px in candidates]

    # Restore in-place DC modification (only when power was zeroed directly)
    if _restore_dc:
        power[dc_mask] = dc_saved

    return peaks


def _lightweight_peak_snr(power: np.ndarray, q_mag: np.ndarray,
                          peak_q: float, py: int, px: int,
                          dc_mask: np.ndarray,
                          annulus_inner_factor: float = 0.9,
                          annulus_outer_factor: float = 1.1,
                          background_disk_r_sq: int = 9,
                          lightweight_snr_method: str = "ratio") -> float:
    """Compute a lightweight SNR for a candidate peak.

    Parameters
    ----------
    lightweight_snr_method : str
        "ratio" (legacy) — signal / median(background).
        "zscore" — (signal - median) / (MAD * 1.4826 + eps).
    """
    signal = float(power[py, px])

    # Annulus: q in [inner_factor * peak_q, outer_factor * peak_q]  (cycles/nm)
    q_lo = annulus_inner_factor * peak_q
    q_hi = annulus_outer_factor * peak_q
    annulus = (q_mag >= q_lo) & (q_mag <= q_hi)

    # Exclude disk around peak
    H, W = power.shape
    yy, xx = np.ogrid[:H, :W]
    disk = ((yy - py) ** 2 + (xx - px) ** 2) <= background_disk_r_sq
    bg_mask = annulus & ~disk & ~dc_mask

    n_bg = int(np.sum(bg_mask))
    if n_bg == 0:
        return 0.0

    bg_values = power[bg_mask]
    bg_median = float(np.median(bg_values))

    if lightweight_snr_method == "zscore":
        bg_mad = float(np.median(np.abs(bg_values - bg_median)))
        bg_sigma = bg_mad * 1.4826
        return (signal - bg_median) / (bg_sigma + 1e-10)
    else:
        # Legacy "ratio" mode
        if bg_median <= 0:
            return signal  # treat as very high SNR when background is zero
        return signal / bg_median


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
                          pixel_size_nm: float,
                          d_max_nm: Optional[float] = None,
                          ) -> Tuple[float, bool]:
    """Gate G5: check periods per tile.

    Parameters
    ----------
    tile_size : int
    d_dom_nm : float
        Dominant d-spacing (nm).
    pixel_size_nm : float
    d_max_nm : float, optional
        When provided, use ``max(d_dom_nm, d_max_nm)`` for the period check
        so that the gate accounts for the largest expected spacing.

    Returns (periods, passed).
    """
    # Use d_dom directly — physics-bound filtering happens at d_dom selection time
    d_eff = d_dom_nm
    d_px = d_eff / pixel_size_nm
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
                      tile_fft_config: Optional[TileFFTConfig] = None,
                      dynamic_dc_q: float = 0.0,
                      dc_mask_config: Optional[DCMaskConfig] = None,
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
    tile_fft_config : TileFFTConfig, optional
        Tile FFT peak detection configuration.
    dynamic_dc_q : float
        Global dynamic DC mask radius (cycles/nm). 0.0 = disabled.
    dc_mask_config : DCMaskConfig, optional
        DC mask configuration (for soft taper).

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
            tile_fft_config=tile_fft_config,
            dynamic_dc_q=dynamic_dc_q,
            dc_mask_config=dc_mask_config,
        )
    else:
        peak_sets, skipped = _process_tiles_cpu(
            image_fft, roi_mask_grid, tile_grid, window,
            tile_size, stride, n_rows, n_cols, n_total, q_ranges,
            process, log_interval, initial_mem,
            effective_q_min=effective_q_min,
            tile_fft_config=tile_fft_config,
            dynamic_dc_q=dynamic_dc_q,
            dc_mask_config=dc_mask_config,
        )

    final_mem = process.memory_info().rss / 1024**3
    logger.info(f"MEMORY: Tile processing complete. Final RSS: {final_mem:.2f} GB (delta: {final_mem - initial_mem:.2f} GB)")
    print(f"MEMORY: Tile processing complete. Final RSS: {final_mem:.2f} GB (delta: +{final_mem - initial_mem:.2f} GB)", flush=True)

    return peak_sets, skipped


def _process_tiles_cpu(image_fft, roi_mask_grid, tile_grid, window,
                       tile_size, stride, n_rows, n_cols, n_total,
                       q_ranges, process, log_interval, initial_mem,
                       effective_q_min=0.0, tile_fft_config=None,
                       dynamic_dc_q=0.0, dc_mask_config=None):
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
                                   effective_q_min=effective_q_min,
                                   tile_fft_config=tile_fft_config,
                                   dynamic_dc_q=dynamic_dc_q,
                                   dc_mask_config=dc_mask_config)

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
                       effective_q_min=0.0, tile_fft_config=None,
                       dynamic_dc_q=0.0, dc_mask_config=None):
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
                                               effective_q_min=effective_q_min,
                                               tile_fft_config=tile_fft_config,
                                               dynamic_dc_q=dynamic_dc_q,
                                               dc_mask_config=dc_mask_config)
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
                                       effective_q_min=effective_q_min,
                                       tile_fft_config=tile_fft_config,
                                       dynamic_dc_q=dynamic_dc_q,
                                       dc_mask_config=dc_mask_config)
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
