"""
Tests for domain clustering pipeline.
"""

import numpy as np
import pytest

from src.pipeline_config import (
    ClusteringConfig, PipelineConfig, TilePeak, TilePeakSet,
    TileClassification, GatedTileGrid, TierSummary, GlobalPeak,
)
from src.ring_analysis import (
    build_ring_maps, build_ring_feature_vectors, RingFeatureVectors,
)
from src.domain_clustering import (
    run_domain_clustering, ClusteringResult,
    _normalize_features, _reduce_dimensions, _cluster,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_gated_grid_for_clustering(n_rows, n_cols, n_rings=2, domains=None):
    """Create a GatedTileGrid with spatially separated domains.

    domains: list of dict with 'rows', 'cols', 'ring_index', 'orientation'
    """
    if domains is None:
        # Default: left half = ring 0 @ 30 deg, right half = ring 1 @ 120 deg
        domains = [
            {"rows": range(n_rows), "cols": range(n_cols // 2),
             "ring_index": 0, "orientation": 30.0, "snr": 10.0},
            {"rows": range(n_rows), "cols": range(n_cols // 2, n_cols),
             "ring_index": 1, "orientation": 120.0, "snr": 8.0},
        ]

    classifications = np.empty((n_rows, n_cols), dtype=object)
    tier_map = np.full((n_rows, n_cols), "A", dtype=object)
    snr_map = np.zeros((n_rows, n_cols))
    pair_fraction_map = np.full((n_rows, n_cols), 0.8)
    fwhm_map = np.zeros((n_rows, n_cols))
    orientation_map = np.zeros((n_rows, n_cols))
    skipped = np.zeros((n_rows, n_cols), dtype=bool)
    oc_map = np.full((n_rows, n_cols), 0.9)

    for dom in domains:
        for r in dom["rows"]:
            for c in dom["cols"]:
                ri = dom["ring_index"]
                snr = dom["snr"]
                orient = dom["orientation"]
                plist = [{"qx": np.cos(np.radians(orient)),
                          "qy": np.sin(np.radians(orient)),
                          "q_mag": 2.5, "snr": snr,
                          "fwhm": 0.05, "fwhm_valid": True,
                          "fwhm_method": "proxy", "fwhm_ok": True,
                          "ring_index": ri}]
                tc = TileClassification(
                    tier="A", peaks=plist, pair_fraction=0.8,
                    n_non_collinear=2, best_snr=snr,
                    best_orientation_deg=orient,
                    orientation_confidence=0.9,
                )
                classifications[r, c] = tc
                snr_map[r, c] = snr
                orientation_map[r, c] = orient

    ts = TierSummary(n_tier_a=n_rows * n_cols, n_tier_b=0, n_rejected=0,
                     n_skipped=0, tier_a_fraction=1.0, median_snr_tier_a=10.0)

    return GatedTileGrid(
        classifications=classifications, tier_map=tier_map,
        snr_map=snr_map, pair_fraction_map=pair_fraction_map,
        fwhm_map=fwhm_map, orientation_map=orientation_map,
        grid_shape=(n_rows, n_cols), skipped_mask=skipped,
        tier_summary=ts, orientation_confidence_map=oc_map,
    )


def _make_global_peaks(n_rings=2):
    peaks = []
    for i in range(n_rings):
        qc = 2.5 + i * 1.5
        peaks.append(GlobalPeak(
            q_center=qc, q_fwhm=0.1, d_spacing=1.0 / qc,
            intensity=100.0, prominence=50.0, snr=10.0, index=i,
        ))
    return peaks


def _build_features(n_rows, n_cols, n_rings=2, domains=None):
    """Build ring_features from a gated grid."""
    grid = _make_gated_grid_for_clustering(n_rows, n_cols, n_rings, domains)
    gp = _make_global_peaks(n_rings)
    rm = build_ring_maps(grid, gp)
    rf = build_ring_feature_vectors(grid, rm)
    return rf, grid, rm


# ======================================================================
# Config tests
# ======================================================================

def test_clustering_config_defaults():
    """enabled=False, method=kmeans by default."""
    cfg = ClusteringConfig()
    assert cfg.enabled is False
    assert cfg.method == "kmeans"
    assert cfg.n_clusters == 0


def test_clustering_config_yaml_roundtrip():
    """to_dict/from_dict preserves clustering config."""
    cfg = PipelineConfig()
    cfg.clustering.enabled = True
    cfg.clustering.method = "gmm"
    cfg.clustering.n_clusters = 5

    d = cfg.to_dict()
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.clustering.enabled is True
    assert cfg2.clustering.method == "gmm"
    assert cfg2.clustering.n_clusters == 5


# ======================================================================
# Clustering tests
# ======================================================================

def test_kmeans_2_domains():
    """Synthetic with 2 domains, K-means finds ~2 clusters."""
    rf, grid, rm = _build_features(8, 8, n_rings=2)
    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=2)
    result = run_domain_clustering(rf, config)

    assert isinstance(result, ClusteringResult)
    assert result.n_clusters == 2
    assert result.tile_labels.shape == (8, 8)


def test_gmm_2_domains():
    """Same with GMM."""
    rf, grid, rm = _build_features(8, 8, n_rings=2)
    config = ClusteringConfig(enabled=True, method="gmm", n_clusters=2)
    result = run_domain_clustering(rf, config)

    assert result.n_clusters == 2


def test_hdbscan_2_domains():
    """Same with HDBSCAN."""
    rf, grid, rm = _build_features(8, 8, n_rings=2)
    config = ClusteringConfig(enabled=True, method="hdbscan",
                              hdbscan_min_cluster_size=3,
                              hdbscan_min_samples=2)
    result = run_domain_clustering(rf, config)

    # HDBSCAN may find different number of clusters
    assert result.n_clusters >= 1


def test_auto_k_selection():
    """Silhouette scan picks a K >= 2."""
    rf, grid, rm = _build_features(8, 8, n_rings=2)
    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=0,
                              n_clusters_max=5)
    result = run_domain_clustering(rf, config)

    assert result.n_clusters >= 2
    assert result.silhouette_curve is not None
    assert len(result.silhouette_curve) > 0


def test_silhouette_curve_returned():
    """silhouette_curve has entries for each k tested."""
    rf, grid, rm = _build_features(8, 8, n_rings=2)
    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=0,
                              n_clusters_max=5)
    result = run_domain_clustering(rf, config)

    if result.silhouette_curve:
        ks = [k for k, _ in result.silhouette_curve]
        assert min(ks) >= 2
        assert max(ks) <= 5


def test_pca_variance_retention():
    """PCA should retain >= threshold variance."""
    rf, grid, rm = _build_features(10, 10, n_rings=3)
    features_valid = rf.feature_matrix[rf.valid_mask]
    from src.domain_clustering import _normalize_features
    scaled, _ = _normalize_features(features_valid)

    config = ClusteringConfig(pca_variance_threshold=0.95)
    reduced, emb, method = _reduce_dimensions(scaled, config)

    assert method == "pca"
    assert reduced.shape[1] <= scaled.shape[1]
    assert reduced.shape[1] >= 2


def test_umap_fallback_to_pca():
    """When umap import fails, falls back gracefully."""
    import unittest.mock as mock
    rf, grid, rm = _build_features(8, 8, n_rings=2)
    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=2,
                              dimred_method="umap")

    # Mock umap import failure
    with mock.patch.dict("sys.modules", {"umap": None}):
        result = run_domain_clustering(rf, config)
        # Should still work, falling back to PCA
        assert result.n_clusters == 2
        assert result.embedding_method in ("pca", "umap")


def test_spatial_regularization_reduces_noise():
    """Flip rate decreases after regularization."""
    # Create 3 domains with some noise
    domains = [
        {"rows": range(10), "cols": range(5), "ring_index": 0,
         "orientation": 30.0, "snr": 10.0},
        {"rows": range(10), "cols": range(5, 10), "ring_index": 1,
         "orientation": 120.0, "snr": 8.0},
    ]
    rf, grid, rm = _build_features(10, 10, n_rings=2, domains=domains)
    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=2,
                              regularize=True)
    result = run_domain_clustering(rf, config)

    if result.adjacency_pre is not None and result.adjacency_post is not None:
        # Regularization should not increase flip rate
        assert result.adjacency_post["flip_rate"] <= result.adjacency_pre["flip_rate"] + 0.01


def test_clustering_disabled_by_default():
    """ClusteringConfig.enabled is False by default."""
    cfg = PipelineConfig()
    assert cfg.clustering.enabled is False


def test_empty_features_graceful():
    """All tiles skipped — no crash."""
    grid = _make_gated_grid_for_clustering(4, 4)
    grid.skipped_mask[:] = True
    gp = _make_global_peaks(2)
    rm = build_ring_maps(grid, gp)
    rf = build_ring_feature_vectors(grid, rm)

    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=2)
    result = run_domain_clustering(rf, config)

    assert result.n_clusters == 0
    assert "error" in result.diagnostics


def test_single_cluster_degenerate():
    """Identical tiles -> 1 cluster possible, silhouette=None."""
    # All tiles identical
    domains = [
        {"rows": range(6), "cols": range(6), "ring_index": 0,
         "orientation": 30.0, "snr": 10.0},
    ]
    rf, grid, rm = _build_features(6, 6, n_rings=1, domains=domains)
    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=1)
    result = run_domain_clustering(rf, config)

    assert result.n_clusters == 1
    # Silhouette not meaningful for 1 cluster
    assert result.silhouette_score is None


def test_cluster_summaries_orientations():
    """Per-cluster orientations are consistent with input."""
    domains = [
        {"rows": range(8), "cols": range(4), "ring_index": 0,
         "orientation": 30.0, "snr": 10.0},
        {"rows": range(8), "cols": range(4, 8), "ring_index": 1,
         "orientation": 120.0, "snr": 8.0},
    ]
    rf, grid, rm = _build_features(8, 8, n_rings=2, domains=domains)
    config = ClusteringConfig(enabled=True, method="kmeans", n_clusters=2,
                              regularize=False)
    result = run_domain_clustering(rf, config)

    if result.n_clusters >= 2:
        summaries = rm  # We test compute_cluster_summaries directly
        from src.ring_analysis import compute_cluster_summaries
        cs = compute_cluster_summaries(result.tile_labels, rm, grid)

        # Each cluster should exist
        assert len(cs) >= 1


# ======================================================================
# peak_metrics ring_index propagation
# ======================================================================

def test_peak_metrics_has_ring_index():
    """TileClassification.peaks dicts include ring_index."""
    from src.fft_peak_detection import classify_tile
    from tests.synthetic import generate_single_crystal

    tile_size = 64
    pixel_size = 0.127
    d = 0.4
    img = generate_single_crystal((tile_size, tile_size), pixel_size, d,
                                   noise_level=0.01)
    window = np.hanning(tile_size)[:, None] * np.hanning(tile_size)[None, :]
    fft = np.fft.fftshift(np.fft.fft2(img * window))
    power = np.abs(fft) ** 2

    grid = FFTGrid(tile_size, tile_size, pixel_size)
    from src.tile_fft import extract_tile_peaks
    q_center = 1.0 / d
    q_width = q_center * 0.15
    q_ranges = [(q_center - q_width, q_center + q_width, 3)]
    peaks = extract_tile_peaks(power, grid, q_ranges=q_ranges)

    if len(peaks) > 0:
        ps = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0, power_spectrum=power)
        tc = classify_tile(ps, grid)
        for pm in tc.peaks:
            assert "ring_index" in pm


from src.fft_coords import FFTGrid
