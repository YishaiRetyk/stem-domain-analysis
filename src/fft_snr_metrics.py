"""
FFT SNR Aggregate Metrics (WS3).

Aggregate SNR/symmetry/FWHM maps, tier classification maps.
Gates G6, G7, G8.
"""

import logging
import numpy as np
from typing import List, Optional

from src.pipeline_config import (
    TilePeakSet, TileClassification, TierSummary, GatedTileGrid,
    TierConfig, PeakGateConfig,
)
from src.fft_coords import FFTGrid
from src.fft_peak_detection import classify_tile
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


def build_gated_tile_grid(peak_sets: List[TilePeakSet],
                          skipped_mask: np.ndarray,
                          fft_grid: FFTGrid,
                          tile_size: int,
                          tier_config: TierConfig = None,
                          peak_gate_config: PeakGateConfig = None,
                          ) -> GatedTileGrid:
    """Classify all tiles and build the unified GatedTileGrid.

    Parameters
    ----------
    peak_sets : list of TilePeakSet (one per tile, in row-major order)
    skipped_mask : (n_rows, n_cols) bool
    fft_grid : FFTGrid for individual tiles
    tile_size : int
    tier_config, peak_gate_config : optional overrides

    Returns
    -------
    GatedTileGrid
    """
    if tier_config is None:
        tier_config = TierConfig()
    if peak_gate_config is None:
        peak_gate_config = PeakGateConfig()

    n_rows, n_cols = skipped_mask.shape
    tile_grid = FFTGrid(tile_size, tile_size, fft_grid.pixel_size_nm)

    classifications = np.empty((n_rows, n_cols), dtype=object)
    tier_map = np.full((n_rows, n_cols), "", dtype=object)
    snr_map = np.zeros((n_rows, n_cols))
    symmetry_map = np.zeros((n_rows, n_cols))
    fwhm_map = np.zeros((n_rows, n_cols))
    orientation_map = np.full((n_rows, n_cols), np.nan)

    for ps in peak_sets:
        r, c = ps.tile_row, ps.tile_col
        if r >= n_rows or c >= n_cols:
            continue

        if skipped_mask[r, c]:
            classifications[r, c] = None
            continue

        tc = classify_tile(ps, tile_grid, tier_config, peak_gate_config)
        classifications[r, c] = tc
        tier_map[r, c] = tc.tier
        snr_map[r, c] = tc.best_snr
        symmetry_map[r, c] = tc.symmetry_score
        orientation_map[r, c] = tc.best_orientation_deg

        # Best FWHM from peak metrics
        if tc.peaks:
            valid_fwhms = [p["fwhm"] for p in tc.peaks if p.get("fwhm_valid", False)]
            if valid_fwhms:
                fwhm_map[r, c] = min(valid_fwhms)

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
