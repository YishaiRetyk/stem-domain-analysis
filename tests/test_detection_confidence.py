"""Tests for detection confidence heatmap (Option A) and ilastik comparison (Option B).

Required tests are implemented; deferred tests are documented at the bottom.
"""

from pathlib import Path

import numpy as np
import pytest

from src.pipeline_config import (
    ConfidenceConfig, TileClassification, TierConfig, PeakGateConfig,
    GatedTileGrid, TierSummary, PipelineConfig,
)
from src.fft_snr_metrics import compute_tile_confidence


# ======================================================================
# Helpers
# ======================================================================

def _make_tc(tier="A", best_snr=8.0, pair_fraction=0.7,
             orientation_confidence=0.8, n_non_collinear=3,
             peaks=None):
    """Build a TileClassification with sensible defaults."""
    if peaks is None:
        peaks = [
            {"fwhm": 0.01, "q_mag": 0.5, "fwhm_valid": True, "snr": 6.0},
            {"fwhm": 0.02, "q_mag": 0.5, "fwhm_valid": True, "snr": 4.0},
        ]
    return TileClassification(
        tier=tier,
        peaks=peaks,
        pair_fraction=pair_fraction,
        n_non_collinear=n_non_collinear,
        best_snr=best_snr,
        orientation_confidence=orientation_confidence,
    )


def _default_configs():
    return TierConfig(), PeakGateConfig(), ConfidenceConfig()


def _make_gated_grid(n_rows=4, n_cols=4):
    """Build a minimal GatedTileGrid with detection_confidence_map."""
    classifications = np.empty((n_rows, n_cols), dtype=object)
    tier_map = np.full((n_rows, n_cols), "", dtype=object)
    snr_map = np.zeros((n_rows, n_cols))
    pair_fraction_map = np.zeros((n_rows, n_cols))
    fwhm_map = np.zeros((n_rows, n_cols))
    orientation_map = np.full((n_rows, n_cols), np.nan)
    skipped_mask = np.zeros((n_rows, n_cols), dtype=bool)
    skipped_mask[0, :] = True  # first row skipped

    tier_config, pgc, cc = _default_configs()

    for r in range(n_rows):
        for c in range(n_cols):
            if skipped_mask[r, c]:
                classifications[r, c] = None
                continue
            if r == 1:
                tc = _make_tc(tier="A", best_snr=10.0, pair_fraction=0.9,
                              orientation_confidence=0.95)
            elif r == 2:
                tc = _make_tc(tier="B", best_snr=3.5, pair_fraction=0.4,
                              orientation_confidence=0.3, n_non_collinear=1)
            else:
                tc = _make_tc(tier="REJECTED", best_snr=1.0, pair_fraction=0.0)
            classifications[r, c] = tc
            tier_map[r, c] = tc.tier
            snr_map[r, c] = tc.best_snr
            pair_fraction_map[r, c] = tc.pair_fraction

    # Build confidence map
    conf_map = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            conf_map[r, c] = compute_tile_confidence(
                classifications[r, c], tier_config, pgc, cc)

    ts = TierSummary(
        n_tier_a=n_cols, n_tier_b=n_cols, n_rejected=n_cols,
        n_skipped=n_cols, tier_a_fraction=0.25, median_snr_tier_a=10.0,
    )

    return GatedTileGrid(
        classifications=classifications,
        tier_map=tier_map,
        snr_map=snr_map,
        pair_fraction_map=pair_fraction_map,
        fwhm_map=fwhm_map,
        orientation_map=orientation_map,
        grid_shape=(n_rows, n_cols),
        skipped_mask=skipped_mask,
        tier_summary=ts,
        orientation_confidence_map=np.zeros((n_rows, n_cols)),
        detection_confidence_map=conf_map,
    )


# ======================================================================
# Option A: Confidence scoring tests
# ======================================================================

class TestComputeTileConfidence:
    def test_rejected_tile_returns_zero(self):
        tc = _make_tc(tier="REJECTED", best_snr=10.0)
        tier_cfg, pgc, cc = _default_configs()
        assert compute_tile_confidence(tc, tier_cfg, pgc, cc) == 0.0

    def test_none_tile_returns_zero(self):
        tier_cfg, pgc, cc = _default_configs()
        assert compute_tile_confidence(None, tier_cfg, pgc, cc) == 0.0

    def test_strong_crystal_high_score(self):
        tc = _make_tc(tier="A", best_snr=10.0, pair_fraction=0.9,
                      orientation_confidence=0.95, n_non_collinear=3)
        tier_cfg, pgc, cc = _default_configs()
        score = compute_tile_confidence(tc, tier_cfg, pgc, cc)
        assert score > 0.7, f"Expected > 0.7, got {score}"

    def test_weak_tile_low_score(self):
        tc = _make_tc(tier="B", best_snr=3.5, pair_fraction=0.2,
                      orientation_confidence=0.2, n_non_collinear=1,
                      peaks=[{"fwhm": 0.1, "q_mag": 0.5, "fwhm_valid": True}])
        tier_cfg, pgc, cc = _default_configs()
        score = compute_tile_confidence(tc, tier_cfg, pgc, cc)
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_score_always_in_bounds(self):
        tier_cfg, pgc, cc = _default_configs()
        for snr in [0.0, 1.0, 3.0, 5.0, 10.0, 50.0, 100.0]:
            tc = _make_tc(tier="A", best_snr=snr)
            score = compute_tile_confidence(tc, tier_cfg, pgc, cc)
            assert 0.0 <= score <= 1.0, f"SNR={snr} gave score {score}"

    def test_tier_a_higher_than_tier_b(self):
        tier_cfg, pgc, cc = _default_configs()
        tc_a = _make_tc(tier="A", best_snr=7.0, pair_fraction=0.6,
                        orientation_confidence=0.6, n_non_collinear=3)
        tc_b = _make_tc(tier="B", best_snr=4.0, pair_fraction=0.6,
                        orientation_confidence=0.6, n_non_collinear=3)
        score_a = compute_tile_confidence(tc_a, tier_cfg, pgc, cc)
        score_b = compute_tile_confidence(tc_b, tier_cfg, pgc, cc)
        assert score_a > score_b

    def test_deterministic(self):
        tc = _make_tc()
        tier_cfg, pgc, cc = _default_configs()
        s1 = compute_tile_confidence(tc, tier_cfg, pgc, cc)
        s2 = compute_tile_confidence(tc, tier_cfg, pgc, cc)
        assert s1 == s2

    def test_custom_weights_snr_only(self):
        cc = ConfidenceConfig(
            w_snr=1.0, w_pair_fraction=0.0,
            w_orientation_confidence=0.0,
            w_non_collinearity=0.0, w_fwhm_quality=0.0,
        )
        tier_cfg = TierConfig()
        pgc = PeakGateConfig()

        # SNR at ceiling → 1.0
        tc_high = _make_tc(tier="A", best_snr=2.0 * tier_cfg.tier_a_snr)
        score_high = compute_tile_confidence(tc_high, tier_cfg, pgc, cc)
        assert abs(score_high - 1.0) < 1e-6

        # SNR at floor → 0.0
        tc_low = _make_tc(tier="B", best_snr=tier_cfg.tier_b_snr)
        score_low = compute_tile_confidence(tc_low, tier_cfg, pgc, cc)
        assert abs(score_low) < 1e-6


class TestConfidenceConfig:
    def test_defaults_sum_to_one(self):
        cc = ConfidenceConfig()
        total = (cc.w_snr + cc.w_pair_fraction + cc.w_orientation_confidence
                 + cc.w_non_collinearity + cc.w_fwhm_quality)
        assert abs(total - 1.0) < 1e-10

    def test_pipeline_config_has_confidence(self):
        cfg = PipelineConfig()
        assert hasattr(cfg, "confidence")
        assert isinstance(cfg.confidence, ConfidenceConfig)

    def test_from_dict_round_trip(self):
        d = {
            "confidence": {
                "enabled": False,
                "w_snr": 0.5,
                "w_pair_fraction": 0.1,
                "w_orientation_confidence": 0.1,
                "w_non_collinearity": 0.2,
                "w_fwhm_quality": 0.1,
            }
        }
        cfg = PipelineConfig.from_dict(d)
        assert cfg.confidence.enabled is False
        assert cfg.confidence.w_snr == 0.5
        assert cfg.confidence.w_non_collinearity == 0.2


class TestGatedTileGridConfidence:
    def test_grid_has_confidence_map(self):
        grid = _make_gated_grid()
        assert grid.detection_confidence_map is not None
        assert grid.detection_confidence_map.shape == grid.grid_shape

    def test_npy_artifact_saved(self, tmp_path):
        grid = _make_gated_grid()
        path = tmp_path / "detection_confidence_map.npy"
        np.save(path, grid.detection_confidence_map.astype(np.float32))
        loaded = np.load(path)
        np.testing.assert_allclose(loaded, grid.detection_confidence_map,
                                   atol=1e-6)

    def test_heatmap_renders(self, tmp_path):
        grid = _make_gated_grid()
        from src.hybrid_viz import _save_detection_confidence_heatmap
        path = _save_detection_confidence_heatmap(grid, tmp_path, dpi=72)
        assert path is not None
        assert Path(path).exists()
        import matplotlib
        matplotlib.pyplot.close("all")


class TestDiagnosticOnly:
    def test_gates_do_not_import_confidence(self):
        """Verify src/gates.py does not import confidence functions (DC-1)."""
        gates_path = Path("src/gates.py")
        if not gates_path.exists():
            pytest.skip("src/gates.py not found")
        content = gates_path.read_text()
        assert "compute_tile_confidence" not in content
        assert "ConfidenceConfig" not in content


# ======================================================================
# Option B: ilastik comparison tests
# ======================================================================

class TestIlastikFeatureStack:
    def test_export_feature_stack_shape(self, tmp_path):
        from src.ilastik_compare import export_feature_stack
        grid = _make_gated_grid(4, 5)
        path = export_feature_stack(grid, tmp_path / "features.npy")
        loaded = np.load(path)
        assert loaded.shape == (4, 5, 5)

    def test_load_npy_round_trip(self, tmp_path):
        from src.ilastik_compare import load_ilastik_probability_map
        arr = np.random.rand(4, 5).astype(np.float64)
        path = tmp_path / "prob.npy"
        np.save(path, arr)
        loaded = load_ilastik_probability_map(path)
        np.testing.assert_allclose(loaded, arr)


class TestNoIlastikDependency:
    def test_normal_imports_dont_pull_ilastik(self):
        """Main pipeline modules should not import ilastik_compare."""
        core_files = [
            Path("src/pipeline_config.py"),
            Path("src/fft_snr_metrics.py"),
            Path("src/gates.py"),
            Path("src/reporting.py"),
            Path("src/hybrid_viz.py"),
        ]
        for f in core_files:
            if not f.exists():
                continue
            content = f.read_text()
            assert "ilastik_compare" not in content, (
                f"{f} references ilastik_compare"
            )


# ======================================================================
# Deferred tests (documented, not implemented in this revision)
# ======================================================================
#
# These require real or realistic ilastik output:
#
# - test_perfect_agreement: identical maps → pearson ~1, agreement 100%
# - test_anti_correlated: inverted maps → pearson < -0.5
# - test_shape_mismatch_raises: ValueError on incompatible shapes
# - test_confusion_matrix_structure: correct keys and shape
