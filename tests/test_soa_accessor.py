"""Tests for SoA accessor on GatedTileGrid (PR 5)."""

import numpy as np
import pytest

from src.pipeline_config import (
    GatedTileGrid, TileClassification, TierSummary,
)


def _make_mini_grid():
    """Create a minimal 2x2 GatedTileGrid for testing."""
    n_rows, n_cols = 2, 2
    classifications = np.empty((n_rows, n_cols), dtype=object)
    tier_map = np.full((n_rows, n_cols), "", dtype="U10")
    snr_map = np.zeros((n_rows, n_cols))
    pair_fraction_map = np.zeros((n_rows, n_cols))
    fwhm_map = np.zeros((n_rows, n_cols))
    orientation_map = np.full((n_rows, n_cols), np.nan)
    skipped_mask = np.zeros((n_rows, n_cols), dtype=bool)

    # Tile (0,0): Tier A with 2 peaks
    tc_00 = TileClassification(
        tier="A", peaks=[
            {"qx": 1.0, "qy": 0.0, "q_mag": 1.0, "d_spacing": 1.0,
             "angle_deg": 0, "snr": 8.0, "fwhm": 0.05, "fwhm_valid": True},
            {"qx": 0.0, "qy": 1.0, "q_mag": 1.0, "d_spacing": 1.0,
             "angle_deg": 90, "snr": 6.0, "fwhm": 0.06, "fwhm_valid": True},
        ],
        pair_fraction=0.8, n_non_collinear=2, best_snr=8.0,
        best_orientation_deg=45.0)
    classifications[0, 0] = tc_00
    tier_map[0, 0] = "A"
    snr_map[0, 0] = 8.0

    # Tile (0,1): Tier B with 1 peak
    tc_01 = TileClassification(
        tier="B", peaks=[
            {"qx": 0.5, "qy": 0.5, "q_mag": 0.707, "d_spacing": 1.414,
             "angle_deg": 45, "snr": 4.0, "fwhm": 0.1, "fwhm_valid": True},
        ],
        pair_fraction=0.3, n_non_collinear=1, best_snr=4.0)
    classifications[0, 1] = tc_01
    tier_map[0, 1] = "B"
    snr_map[0, 1] = 4.0

    # Tile (1,0): REJECTED, no peaks
    tc_10 = TileClassification(
        tier="REJECTED", peaks=[], pair_fraction=0.0,
        n_non_collinear=0, best_snr=1.0)
    classifications[1, 0] = tc_10
    tier_map[1, 0] = "REJECTED"

    # Tile (1,1): Skipped
    classifications[1, 1] = None
    skipped_mask[1, 1] = True

    tier_summary = TierSummary(
        n_tier_a=1, n_tier_b=1, n_rejected=1, n_skipped=1,
        tier_a_fraction=0.333, median_snr_tier_a=8.0)

    return GatedTileGrid(
        classifications=classifications,
        tier_map=tier_map,
        snr_map=snr_map,
        pair_fraction_map=pair_fraction_map,
        fwhm_map=fwhm_map,
        orientation_map=orientation_map,
        grid_shape=(n_rows, n_cols),
        skipped_mask=skipped_mask,
        tier_summary=tier_summary,
    )


class TestTierMapDtype:
    """tier_map should use native string dtype for vectorised ops."""

    def test_tier_map_u10_dtype(self):
        grid = _make_mini_grid()
        assert grid.tier_map.dtype.kind == "U"

    def test_tier_map_comparison(self):
        """Vectorised comparison should work without Python loop."""
        grid = _make_mini_grid()
        mask = grid.tier_map == "A"
        assert mask.dtype == bool
        assert mask[0, 0] is np.True_
        assert mask[0, 1] is np.False_


class TestToStructured:
    """to_structured() method tests."""

    def test_to_structured_keys_and_shapes(self):
        grid = _make_mini_grid()
        s = grid.to_structured()
        expected_keys = {
            "tier_map", "snr_map", "pair_fraction_map", "fwhm_map",
            "orientation_map", "skipped_mask", "n_peaks", "best_d_spacing",
        }
        assert expected_keys.issubset(set(s.keys()))

        for k in expected_keys:
            assert s[k].shape == (2, 2), f"{k} shape mismatch"

    def test_n_peaks_values(self):
        grid = _make_mini_grid()
        s = grid.to_structured()
        assert s["n_peaks"][0, 0] == 2  # Tier A, 2 peaks
        assert s["n_peaks"][0, 1] == 1  # Tier B, 1 peak
        assert s["n_peaks"][1, 0] == 0  # REJECTED
        assert s["n_peaks"][1, 1] == 0  # Skipped (None)

    def test_best_d_spacing(self):
        grid = _make_mini_grid()
        s = grid.to_structured()
        assert s["best_d_spacing"][0, 0] == pytest.approx(1.0)
        assert np.isnan(s["best_d_spacing"][1, 0])  # REJECTED, no peaks


class TestToPeaksDataframe:
    """to_peaks_dataframe() method tests."""

    def test_to_peaks_dataframe(self):
        grid = _make_mini_grid()
        rows = grid.to_peaks_dataframe()
        # 2 peaks from (0,0) + 1 peak from (0,1) + 0 from (1,0) + 0 from (1,1)
        assert len(rows) == 3

    def test_peak_row_fields(self):
        grid = _make_mini_grid()
        rows = grid.to_peaks_dataframe()
        first = rows[0]
        assert "tile_row" in first
        assert "tile_col" in first
        assert "tier" in first
        assert "peak_idx" in first
        assert "snr" in first
