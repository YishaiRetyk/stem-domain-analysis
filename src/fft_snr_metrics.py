"""
FFT SNR Aggregate Metrics (WS3).

Aggregate SNR/symmetry/FWHM maps, tier classification maps.
Gates G6, G7, G8.
"""

import logging
import os
import time
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from src.pipeline_config import (
    TilePeakSet, TileClassification, TierSummary, GatedTileGrid,
    TierConfig, PeakGateConfig, FWHMConfig,
)
from src.fft_coords import FFTGrid
from src.fft_peak_detection import classify_tile
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


def _classify_tile_batch(batch, tile_grid, tier_config, peak_gate_config,
                         fwhm_config, effective_q_min, precomputed_grids):
    """Classify a batch of tiles (worker function for ThreadPoolExecutor).

    Each item in *batch* is ``(index, TilePeakSet)``.
    Returns list of ``(index, TileClassification | None)``.
    """
    results = []
    for idx, ps in batch:
        tc = classify_tile(
            ps, tile_grid, tier_config, peak_gate_config,
            fwhm_config, effective_q_min=effective_q_min,
            _precomputed_grids=precomputed_grids,
        )
        results.append((idx, tc))
    return results


def build_gated_tile_grid(peak_sets: List[TilePeakSet],
                          skipped_mask: np.ndarray,
                          fft_grid: FFTGrid,
                          tile_size: int,
                          tier_config: TierConfig = None,
                          peak_gate_config: PeakGateConfig = None,
                          fwhm_config: FWHMConfig = None,
                          log_interval_s: float = 5.0,
                          effective_q_min: float = 0.0,
                          n_workers: int = 0,
                          ) -> GatedTileGrid:
    """Classify all tiles and build the unified GatedTileGrid.

    Parameters
    ----------
    peak_sets : list of TilePeakSet (one per tile, in row-major order)
    skipped_mask : (n_rows, n_cols) bool
    fft_grid : FFTGrid for individual tiles
    tile_size : int
    tier_config, peak_gate_config, fwhm_config : optional overrides
    log_interval_s : seconds between progress log lines (default 5)
    n_workers : int
        Number of parallel threads for tile classification.
        0 (default) = ``min(4, cpu_count)``.  1 = sequential (no threading).

    Returns
    -------
    GatedTileGrid
    """
    if tier_config is None:
        tier_config = TierConfig()
    if peak_gate_config is None:
        peak_gate_config = PeakGateConfig()
    if fwhm_config is None:
        fwhm_config = FWHMConfig()

    n_rows, n_cols = skipped_mask.shape
    tile_grid = FFTGrid(tile_size, tile_size, fft_grid.pixel_size_nm)

    classifications = np.empty((n_rows, n_cols), dtype=object)
    tier_map = np.full((n_rows, n_cols), "", dtype=object)
    snr_map = np.zeros((n_rows, n_cols))
    symmetry_map = np.zeros((n_rows, n_cols))
    fwhm_map = np.zeros((n_rows, n_cols))
    orientation_map = np.full((n_rows, n_cols), np.nan)

    # --- Pre-compute shared coordinate grids once (all tiles are tile_size×tile_size) ---
    precomputed_grids = {
        'y_grid': np.mgrid[:tile_size, :tile_size][0],
        'x_grid': np.mgrid[:tile_size, :tile_size][1],
    }

    # --- Separate classifiable tiles from skipped tiles ---
    classifiable = []  # (flat_index, TilePeakSet)
    for i, ps in enumerate(peak_sets):
        r, c = ps.tile_row, ps.tile_col
        if r >= n_rows or c >= n_cols:
            continue
        if skipped_mask[r, c]:
            classifications[r, c] = None
            continue
        classifiable.append((i, ps))

    # --- Progress tracking ---
    process = psutil.Process()
    n_total = len(peak_sets)
    n_classifiable = len(classifiable)
    initial_mem = process.memory_info().rss / 1024**3
    t_start = time.monotonic()

    # Resolve worker count
    if n_workers <= 0:
        n_workers = min(4, os.cpu_count() or 1)
    # Don't use threading for tiny workloads
    use_parallel = n_workers > 1 and n_classifiable > 32

    msg = (f"Classification: starting {n_total} tiles ({n_classifiable} classifiable, "
           f"workers={n_workers if use_parallel else 1}) "
           f"(RSS {initial_mem:.2f} GB)")
    logger.info(msg)
    print(msg, flush=True)

    # --- Classify tiles (parallel or sequential) ---
    tile_results = {}  # flat_index -> TileClassification

    if use_parallel:
        # Split classifiable tiles into chunks for thread workers
        chunk_size = max(8, n_classifiable // (n_workers * 4))
        chunks = [
            classifiable[i:i + chunk_size]
            for i in range(0, n_classifiable, chunk_size)
        ]

        completed = 0
        t_last_log = t_start
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _classify_tile_batch, chunk, tile_grid,
                    tier_config, peak_gate_config, fwhm_config,
                    effective_q_min, precomputed_grids,
                )
                for chunk in chunks
            ]
            for future in futures:
                batch_results = future.result()
                for idx, tc in batch_results:
                    tile_results[idx] = tc
                completed += len(batch_results)

                # Time-based progress logging
                t_now = time.monotonic()
                if t_now - t_last_log >= log_interval_s:
                    elapsed = t_now - t_start
                    tiles_per_s = completed / elapsed if elapsed > 0 else 0
                    remaining = (n_classifiable - completed) / tiles_per_s if tiles_per_s > 0 else 0
                    mem_gb = process.memory_info().rss / 1024**3
                    pct = 100 * completed / n_classifiable

                    msg = (f"Classification: {completed}/{n_classifiable} ({pct:.0f}%) | "
                           f"{tiles_per_s:.1f} tiles/s | "
                           f"ETA {remaining:.0f}s | RSS {mem_gb:.2f} GB")
                    logger.info(msg)
                    print(msg, flush=True)
                    t_last_log = t_now
    else:
        # Sequential path (n_workers=1 or small workload)
        tiles_done = 0
        peaks_done = 0
        t_last_log = t_start
        for idx, ps in classifiable:
            tc = classify_tile(
                ps, tile_grid, tier_config, peak_gate_config,
                fwhm_config, effective_q_min=effective_q_min,
                _precomputed_grids=precomputed_grids,
            )
            tile_results[idx] = tc
            tiles_done += 1
            peaks_done += len(ps.peaks)

            t_now = time.monotonic()
            if t_now - t_last_log >= log_interval_s:
                elapsed = t_now - t_start
                tiles_per_s = tiles_done / elapsed if elapsed > 0 else 0
                peaks_per_s = peaks_done / elapsed if elapsed > 0 else 0
                remaining = (n_classifiable - tiles_done) / tiles_per_s if tiles_per_s > 0 else 0
                mem_gb = process.memory_info().rss / 1024**3
                pct = 100 * tiles_done / n_classifiable

                msg = (f"Classification: {tiles_done}/{n_classifiable} ({pct:.0f}%) | "
                       f"{tiles_per_s:.1f} tiles/s, {peaks_per_s:.0f} peaks/s | "
                       f"ETA {remaining:.0f}s | RSS {mem_gb:.2f} GB")
                logger.info(msg)
                print(msg, flush=True)
                t_last_log = t_now

    # --- Merge results into output arrays ---
    tiles_done = 0
    peaks_done = 0
    for i, ps in enumerate(peak_sets):
        r, c = ps.tile_row, ps.tile_col
        if r >= n_rows or c >= n_cols:
            continue
        if i not in tile_results:
            continue

        tc = tile_results[i]
        classifications[r, c] = tc
        tier_map[r, c] = tc.tier
        snr_map[r, c] = tc.best_snr
        symmetry_map[r, c] = tc.symmetry_score
        orientation_map[r, c] = tc.best_orientation_deg

        if tc.peaks:
            valid_fwhms = [p["fwhm"] for p in tc.peaks if p.get("fwhm_valid", False)]
            if valid_fwhms:
                fwhm_map[r, c] = min(valid_fwhms)

        tiles_done += 1
        peaks_done += len(ps.peaks)

    t_total = time.monotonic() - t_start
    final_mem = process.memory_info().rss / 1024**3
    msg = (f"Classification: done {tiles_done} tiles, {peaks_done} peaks "
           f"in {t_total:.1f}s ({tiles_done/max(t_total,0.001):.1f} tiles/s) | "
           f"RSS {final_mem:.2f} GB (delta +{final_mem - initial_mem:.2f} GB)")
    logger.info(msg)
    print(msg, flush=True)

    # Tier summary
    tier_a_mask = tier_map == "A"
    tier_b_mask = tier_map == "B"
    rej_mask = tier_map == "REJECTED"

    n_tier_a = int(np.sum(tier_a_mask))
    n_tier_b = int(np.sum(tier_b_mask))
    n_rejected = int(np.sum(rej_mask))
    n_skipped = int(np.sum(skipped_mask))
    n_roi = n_rows * n_cols - n_skipped
    tier_a_fraction = n_tier_a / max(n_roi, 1)

    # Median SNR of Tier A
    tier_a_snrs = snr_map[tier_a_mask]
    median_snr_a = float(np.median(tier_a_snrs)) if len(tier_a_snrs) > 0 else 0.0

    tier_summary = TierSummary(
        n_tier_a=n_tier_a,
        n_tier_b=n_tier_b,
        n_rejected=n_rejected,
        n_skipped=n_skipped,
        tier_a_fraction=tier_a_fraction,
        median_snr_tier_a=median_snr_a,
    )

    # Gates G6, G7, G8
    g6 = evaluate_gate("G6", tier_a_fraction)
    g7 = evaluate_gate("G7", median_snr_a)

    # Mean symmetry of Tier A
    tier_a_sym = symmetry_map[tier_a_mask]
    mean_sym = float(np.mean(tier_a_sym)) if len(tier_a_sym) > 0 else 0.0
    g8 = evaluate_gate("G8", mean_sym)

    logger.info("Tier summary: A=%d, B=%d, rejected=%d, skipped=%d (fraction_A=%.3f)",
                n_tier_a, n_tier_b, n_rejected, n_skipped, tier_a_fraction)

    return GatedTileGrid(
        classifications=classifications,
        tier_map=tier_map,
        snr_map=snr_map,
        symmetry_map=symmetry_map,
        fwhm_map=fwhm_map,
        orientation_map=orientation_map,
        grid_shape=(n_rows, n_cols),
        skipped_mask=skipped_mask,
        tier_summary=tier_summary,
    )
