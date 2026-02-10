"""
Tests for ring analysis: TilePeak ring_index, ring maps, feature vectors,
tile-averaged FFT.
"""

import numpy as np
import pytest

from src.pipeline_config import (
    TilePeak, TilePeakSet, TileClassification, GatedTileGrid,
    TierSummary, GlobalPeak,
)
from src.fft_coords import FFTGrid
from src.tile_fft import extract_tile_peaks
from src.ring_analysis import (
    build_ring_maps, build_ring_feature_vectors,
    compute_tile_averaged_fft, compute_cluster_averaged_ffts,
    compute_cluster_summaries, RingMaps,
)
from tests.synthetic import generate_single_crystal


# ======================================================================
# Helpers
# ======================================================================

def _make_gated_grid(n_rows, n_cols, peaks_per_tile=None, ring_index=0,
                     tier="A", snr=10.0, orientation=30.0):
    """Create a minimal GatedTileGrid for testing."""
    classifications = np.empty((n_rows, n_cols), dtype=object)
    tier_map = np.full((n_rows, n_cols), "", dtype=object)
    snr_map = np.zeros((n_rows, n_cols))
    pair_fraction_map = np.zeros((n_rows, n_cols))
    fwhm_map = np.zeros((n_rows, n_cols))
    orientation_map = np.zeros((n_rows, n_cols))
    skipped = np.zeros((n_rows, n_cols), dtype=bool)
    oc_map = np.zeros((n_rows, n_cols))

    for r in range(n_rows):
        for c in range(n_cols):
            if peaks_per_tile is not None and peaks_per_tile.get((r, c)) is not None:
                plist = peaks_per_tile[(r, c)]
            else:
                plist = [{"qx": 1.0, "qy": 0.0, "q_mag": 1.0, "snr": snr,
                          "fwhm": 0.05, "fwhm_valid": True, "fwhm_method": "proxy",
                          "fwhm_ok": True, "ring_index": ring_index}]
            tc = TileClassification(
                tier=tier, peaks=plist, pair_fraction=0.8,
                n_non_collinear=2, best_snr=snr,
                best_orientation_deg=orientation,
                orientation_confidence=0.9,
            )
            classifications[r, c] = tc
            tier_map[r, c] = tier
            snr_map[r, c] = snr
            pair_fraction_map[r, c] = 0.8
            orientation_map[r, c] = orientation
            oc_map[r, c] = 0.9

    ts = TierSummary(n_tier_a=n_rows * n_cols, n_tier_b=0, n_rejected=0,
                     n_skipped=0, tier_a_fraction=1.0, median_snr_tier_a=snr)

    return GatedTileGrid(
        classifications=classifications, tier_map=tier_map,
        snr_map=snr_map, pair_fraction_map=pair_fraction_map,
        fwhm_map=fwhm_map, orientation_map=orientation_map,
        grid_shape=(n_rows, n_cols), skipped_mask=skipped,
        tier_summary=ts, orientation_confidence_map=oc_map,
    )


def _make_global_peaks(n_rings=1, q_centers=None):
    """Create GlobalPeak list."""
    if q_centers is None:
        q_centers = [2.5 + i * 1.5 for i in range(n_rings)]
    peaks = []
    for i, qc in enumerate(q_centers):
        peaks.append(GlobalPeak(
            q_center=qc, q_fwhm=0.1, d_spacing=1.0 / qc,
            intensity=100.0, prominence=50.0, snr=10.0, index=i,
        ))
    return peaks


# ======================================================================
# TilePeak ring_index tests
# ======================================================================

def test_tilepeak_ring_index_default():
    """TilePeak().ring_index defaults to -1."""
    p = TilePeak(qx=1.0, qy=0.0, q_mag=1.0, d_spacing=1.0,
                 angle_deg=0.0, intensity=100.0, fwhm=0.05)
    assert p.ring_index == -1


def test_tilepeak_ring_index_assignment():
    """extract_tile_peaks with 3-tuple q_ranges tags peaks."""
    # Create a tile with known peak
    tile_size = 64
    pixel_size = 0.127
    d = 0.4
    img = generate_single_crystal((tile_size, tile_size), pixel_size, d,
                                   orientation_deg=0, noise_level=0.01)
    window = np.hanning(tile_size)[:, None] * np.hanning(tile_size)[None, :]
    fft = np.fft.fftshift(np.fft.fft2(img * window))
    power = np.abs(fft) ** 2

    grid = FFTGrid(tile_size, tile_size, pixel_size)
    q_center = 1.0 / d
    q_width = q_center * 0.15
    q_ranges = [(q_center - q_width, q_center + q_width, 7)]

    peaks = extract_tile_peaks(power, grid, q_ranges=q_ranges)
    if len(peaks) > 0:
        assert any(p.ring_index == 7 for p in peaks)


def test_tilepeak_ring_index_backward_compat():
    """2-tuple q_ranges still works (ring_index = positional idx)."""
    tile_size = 64
    pixel_size = 0.127
    d = 0.4
    img = generate_single_crystal((tile_size, tile_size), pixel_size, d,
                                   orientation_deg=0, noise_level=0.01)
    window = np.hanning(tile_size)[:, None] * np.hanning(tile_size)[None, :]
    fft = np.fft.fftshift(np.fft.fft2(img * window))
    power = np.abs(fft) ** 2

    grid = FFTGrid(tile_size, tile_size, pixel_size)
    q_center = 1.0 / d
    q_width = q_center * 0.15
    q_ranges = [(q_center - q_width, q_center + q_width)]

    peaks = extract_tile_peaks(power, grid, q_ranges=q_ranges)
    if len(peaks) > 0:
        assert any(p.ring_index == 0 for p in peaks)


# ======================================================================
# Ring maps tests
# ======================================================================

def test_build_ring_maps_single_ring():
    """Single ring: presence should be 1.0 for all non-skipped tiles."""
    grid = _make_gated_grid(4, 4, ring_index=0)
    gp = _make_global_peaks(1)
    rm = build_ring_maps(grid, gp)

    assert rm.n_rings == 1
    assert 0 in rm.presence
    assert np.all(rm.presence[0] == 1.0)


def test_build_ring_maps_two_rings():
    """Two rings: tiles with different ring_index."""
    peaks_per_tile = {}
    for r in range(4):
        for c in range(4):
            ri = 0 if c < 2 else 1
            peaks_per_tile[(r, c)] = [
                {"qx": 1.0, "qy": 0.0, "q_mag": 1.0, "snr": 8.0,
                 "fwhm": 0.05, "fwhm_valid": True, "fwhm_method": "proxy",
                 "fwhm_ok": True, "ring_index": ri}
            ]
    grid = _make_gated_grid(4, 4, peaks_per_tile=peaks_per_tile)
    gp = _make_global_peaks(2)
    rm = build_ring_maps(grid, gp)

    assert rm.n_rings == 2
    # Ring 0 present in left half
    assert np.all(rm.presence[0][:, :2] == 1.0)
    assert np.all(rm.presence[0][:, 2:] == 0.0)
    # Ring 1 present in right half
    assert np.all(rm.presence[1][:, 2:] == 1.0)
    assert np.all(rm.presence[1][:, :2] == 0.0)


def test_peak_count_map_values():
    """Count matches expected peaks per tile per ring."""
    peaks_per_tile = {}
    for r in range(3):
        for c in range(3):
            n_peaks = (r + c) % 3 + 1
            plist = [
                {"qx": 1.0, "qy": 0.0, "q_mag": 1.0, "snr": 8.0,
                 "fwhm": 0.05, "fwhm_valid": True, "fwhm_method": "proxy",
                 "fwhm_ok": True, "ring_index": 0}
                for _ in range(n_peaks)
            ]
            peaks_per_tile[(r, c)] = plist
    grid = _make_gated_grid(3, 3, peaks_per_tile=peaks_per_tile)
    gp = _make_global_peaks(1)
    rm = build_ring_maps(grid, gp)

    for r in range(3):
        for c in range(3):
            expected = (r + c) % 3 + 1
            assert rm.peak_count[0][r, c] == expected


def test_ring_maps_skipped_tiles():
    """Skipped tiles have 0 presence."""
    grid = _make_gated_grid(4, 4, ring_index=0)
    grid.skipped_mask[0, :] = True  # skip first row
    gp = _make_global_peaks(1)
    rm = build_ring_maps(grid, gp)

    assert np.all(rm.presence[0][0, :] == 0.0)
    assert np.all(rm.presence[0][1:, :] == 1.0)


# ======================================================================
# Feature vector tests
# ======================================================================

def test_ring_feature_vector_shape():
    """n_features = n_rings * 5 + 4."""
    grid = _make_gated_grid(4, 4, ring_index=0)
    gp = _make_global_peaks(2)
    rm = build_ring_maps(grid, gp)
    fv = build_ring_feature_vectors(grid, rm)

    assert fv.feature_matrix.shape == (16, 2 * 5 + 4)
    assert len(fv.feature_names) == 14


def test_ring_feature_vector_no_nans():
    """Valid tiles have no NaN/Inf."""
    grid = _make_gated_grid(4, 4, ring_index=0)
    gp = _make_global_peaks(1)
    rm = build_ring_maps(grid, gp)
    fv = build_ring_feature_vectors(grid, rm)

    valid_features = fv.feature_matrix[fv.valid_mask]
    assert not np.any(np.isnan(valid_features))
    assert not np.any(np.isinf(valid_features))


# ======================================================================
# Tile-averaged FFT tests
# ======================================================================

def test_tile_averaged_fft_shape():
    """Output power spectrum matches tile_size x tile_size."""
    img = generate_single_crystal((512, 512), 0.127, 0.4)
    tile_size = 64
    stride = 32
    skipped = np.zeros((15, 15), dtype=bool)

    result = compute_tile_averaged_fft(img, tile_size, stride, 0.127, skipped)
    assert result["mean_power"].shape == (tile_size, tile_size)
    assert result["n_tiles"] > 0
    assert len(result["q_values"]) == tile_size // 2


def test_tile_averaged_fft_radial_profile():
    """Radial profile should have positive values."""
    img = generate_single_crystal((256, 256), 0.127, 0.4)
    tile_size = 64
    stride = 32
    skipped = np.zeros((7, 7), dtype=bool)

    result = compute_tile_averaged_fft(img, tile_size, stride, 0.127, skipped)
    assert np.any(result["radial_profile"] > 0)


def test_cluster_averaged_ffts_streaming():
    """Streaming result: each cluster has a mean power spectrum."""
    img = generate_single_crystal((256, 256), 0.127, 0.4)
    tile_size = 64
    stride = 32
    n_rows, n_cols = 7, 7
    skipped = np.zeros((n_rows, n_cols), dtype=bool)
    labels = np.zeros((n_rows, n_cols), dtype=np.int32)
    labels[:, 4:] = 1

    result = compute_cluster_averaged_ffts(labels, img, tile_size, stride, 0.127, skipped)
    assert 0 in result
    assert 1 in result
    assert result[0]["mean_power"].shape == (tile_size, tile_size)
    assert result[1]["n_tiles"] > 0


# ======================================================================
# Cluster summaries tests
# ======================================================================

def test_cluster_summaries_per_cluster_rings():
    """Per-cluster dominant rings match expected."""
    peaks_per_tile = {}
    for r in range(4):
        for c in range(4):
            ri = 0 if c < 2 else 1
            peaks_per_tile[(r, c)] = [
                {"qx": 1.0, "qy": 0.0, "q_mag": 1.0, "snr": 8.0,
                 "fwhm": 0.05, "fwhm_valid": True, "fwhm_method": "proxy",
                 "fwhm_ok": True, "ring_index": ri}
            ]
    grid = _make_gated_grid(4, 4, peaks_per_tile=peaks_per_tile)
    gp = _make_global_peaks(2)
    rm = build_ring_maps(grid, gp)

    labels = np.zeros((4, 4), dtype=np.int32)
    labels[:, 2:] = 1

    summaries = compute_cluster_summaries(labels, rm, grid)
    assert 0 in summaries
    assert 1 in summaries
    # Cluster 0 (left) should have ring 0 dominant
    assert summaries[0]["dominant_rings"][0] == 1.0
    assert summaries[0]["dominant_rings"][1] == 0.0
    # Cluster 1 (right) should have ring 1 dominant
    assert summaries[1]["dominant_rings"][1] == 1.0
    assert summaries[1]["dominant_rings"][0] == 0.0
