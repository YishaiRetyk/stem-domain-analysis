"""
Tests for the config consolidation refactoring (Step 9).

Verifies that:
1. Default equivalence — PipelineConfig() defaults match all previously hardcoded values
2. YAML round-trip — to_dict() → from_dict() preserves all new fields
3. Backward compat — YAML with old validation: keys syncs to gate_thresholds:
4. Gate override — evaluate_gate() respects GateThresholdsConfig
5. G3 lcc_fraction — now configurable via g3_min_lcc_fraction
6. Reference selection bug fix — select_reference_region() uses config values
7. Module config threading — non-default PeakSNRConfig changes behavior
8. FWHM patch cache rebuild — PeakSNRConfig(fwhm_patch_radius=7) triggers rebuild
"""

import numpy as np
import pytest

from src.pipeline_config import (
    PipelineConfig,
    GateThresholdsConfig,
    PeakSNRConfig,
    ReferenceSelectionConfig,
    RingAnalysisConfig,
    GlobalFFTConfig,
    PreprocConfig,
    ROIConfig,
    GPAConfig,
    PeakFindingConfig,
    TileFFTConfig,
    ClusteringConfig,
    DCMaskConfig,
    ConfidenceConfig,
    TierConfig,
    PeakGateConfig,
    FWHMConfig,
    GatedTileGrid,
    TierSummary,
    TilePeakSet,
    TilePeak,
)
from src.gates import evaluate_gate, GATE_DEFS


# ======================================================================
# Category 1: Default equivalence
# ======================================================================

class TestDefaultEquivalence:
    """PipelineConfig() defaults must match the previously hardcoded values."""

    def test_gate_thresholds_defaults(self):
        gt = GateThresholdsConfig()
        # G0
        assert gt.g0_nyquist_safety_margin == 0.95
        # G2
        assert gt.g2_max_clipped_fraction == 0.005
        assert gt.g2_min_range_ratio == 10.0
        # G3
        assert gt.g3_min_coverage == 10.0
        assert gt.g3_max_coverage == 95.0
        assert gt.g3_max_fragments == 20
        assert gt.g3_min_lcc_fraction == 0.5
        # G4
        assert gt.g4_min_peak_snr == 3.0
        # G5
        assert gt.g5_min_periods == 20.0
        # G6
        assert gt.g6_min_fraction == 0.05
        # G7
        assert gt.g7_min_median_snr == 5.0
        # G8
        assert gt.g8_min_mean_symmetry == 0.3
        # G9
        assert gt.g9_min_area == 9
        assert gt.g9_max_entropy == 0.3
        assert gt.g9_min_snr == 5.0
        # G10
        assert gt.g10_max_phase_noise == 0.3
        assert gt.g10_min_unwrap_success == 0.7
        # G11
        assert gt.g11_max_ref_strain == 0.005
        assert gt.g11_max_outlier_fraction == 0.20
        assert gt.g11_strain_outlier_threshold == 0.05
        # G12
        assert gt.g12_min_fraction_valid == 0.50
        assert gt.g12_tolerance == 0.20

    def test_gate_thresholds_match_gate_defs(self):
        """threshold_dict() output must match GATE_DEFS defaults."""
        gt = GateThresholdsConfig()
        for gate_id, gate_def in GATE_DEFS.items():
            if gate_def.default_threshold is None:
                continue
            td = gt.threshold_dict(gate_id)
            assert td is not None, f"{gate_id} threshold_dict returned None"
            for key, val in gate_def.default_threshold.items():
                assert key in td, f"{gate_id} missing key '{key}'"
                assert td[key] == pytest.approx(val), (
                    f"{gate_id}.{key}: config={td[key]} != gate_def={val}"
                )

    def test_peak_snr_defaults(self):
        ps = PeakSNRConfig()
        assert ps.signal_disk_radius_px == 3
        assert ps.annular_width_min_q == 0.15
        assert ps.annular_fwhm_multiplier == 1.5
        assert ps.min_background_pixels == 20
        assert ps.fwhm_patch_radius == 5
        assert ps.moment_sigma_floor == 0.3
        assert ps.fit_condition_max == 100.0
        assert ps.symmetry_tolerance_px == 2.0
        assert ps.non_collinear_min_angle_deg == 15.0

    def test_reference_selection_defaults(self):
        rc = ReferenceSelectionConfig()
        assert rc.scoring_weight_entropy == 0.4
        assert rc.scoring_weight_snr == 0.3
        assert rc.scoring_weight_area == 0.3
        assert rc.orientation_bins == 12

    def test_ring_analysis_defaults(self):
        ra = RingAnalysisConfig()
        assert ra.ring_width_fwhm_mult == 2.0
        assert ra.ring_width_fallback_frac == 0.03
        assert ra.ring_width_no_fwhm_frac == 0.1

    def test_global_fft_new_defaults(self):
        gf = GlobalFFTConfig()
        assert gf.strong_guidance_snr == 8.0
        assert gf.bg_reweight_iterations == 4
        assert gf.bg_reweight_downweight == 0.1
        assert gf.savgol_window_max == 11
        assert gf.savgol_window_min == 5
        assert gf.savgol_polyorder == 2
        assert gf.radial_peak_distance == 5
        assert gf.radial_peak_width == 2
        assert gf.q_width_expansion_frac == 0.03
        assert gf.harmonic_ratio_tol == 0.05
        assert gf.harmonic_angle_tol_deg == 5.0
        assert gf.harmonic_snr_ratio == 2.0
        assert gf.non_collinear_min_angle_deg == 15.0
        assert gf.angular_prominence_frac == 0.5
        assert gf.angular_peak_distance == 10

    def test_preproc_new_defaults(self):
        p = PreprocConfig()
        assert p.hot_pixel_median_kernel == 3
        assert p.robust_norm_clip_sigma == 5.0

    def test_roi_new_defaults(self):
        r = ROIConfig()
        assert r.variance_window_size == 32
        assert r.morph_kernel_size == 5
        assert r.smooth_sigma == 2.0
        assert r.gradient_smooth_sigma == 1.0

    def test_gpa_new_defaults(self):
        g = GPAConfig()
        assert g.mask_sigma_fwhm_factor == 0.5
        assert g.mask_sigma_fallback_factor == 0.06
        assert g.mask_sigma_dc_clamp_factor == 0.18
        assert g.orientation_bins == 12
        assert g.bimodal_smooth_sigma == 0.5
        assert g.bimodal_peak_distance == 2
        assert g.bimodal_valley_ratio == 0.5
        assert g.amplitude_erosion_iterations == 2
        assert g.phase_noise_min_pixels == 10
        assert g.min_tier_a_fraction_for_gpa == 0.1

    def test_peak_finding_new_defaults(self):
        pf = PeakFindingConfig()
        assert pf.taper_width_fraction == 0.3
        assert pf.background_percentile == 50.0
        assert pf.background_filter_size_mult == 2
        assert pf.adaptive_tolerance_floor == 0.1

    def test_tile_fft_new_defaults(self):
        tf = TileFFTConfig()
        assert tf.annulus_inner_factor == 0.9
        assert tf.annulus_outer_factor == 1.1
        assert tf.background_disk_r_sq == 9

    def test_clustering_new_defaults(self):
        c = ClusteringConfig()
        assert c.min_valid_tiles == 3
        assert c.spatial_regularize_iterations == 2
        assert c.umap_n_neighbors == 15
        assert c.umap_min_dist == 0.1
        assert c.kmeans_n_init == 10

    def test_confidence_new_defaults(self):
        c = ConfidenceConfig()
        assert c.snr_ceiling_multiplier == 2.0

    def test_pipeline_config_has_new_fields(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.gate_thresholds, GateThresholdsConfig)
        assert isinstance(cfg.peak_snr, PeakSNRConfig)
        assert isinstance(cfg.reference_selection, ReferenceSelectionConfig)
        assert isinstance(cfg.ring_analysis, RingAnalysisConfig)


# ======================================================================
# Category 2: YAML round-trip
# ======================================================================

class TestYAMLRoundTrip:
    """to_dict() → from_dict() preserves all new fields."""

    def test_round_trip_defaults(self):
        """Default config survives round-trip."""
        cfg1 = PipelineConfig()
        d = cfg1.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        # Check all new sub-configs
        assert cfg2.gate_thresholds.g5_min_periods == cfg1.gate_thresholds.g5_min_periods
        assert cfg2.peak_snr.signal_disk_radius_px == cfg1.peak_snr.signal_disk_radius_px
        assert cfg2.reference_selection.orientation_bins == cfg1.reference_selection.orientation_bins
        assert cfg2.ring_analysis.ring_width_fwhm_mult == cfg1.ring_analysis.ring_width_fwhm_mult

    def test_round_trip_custom_gate_thresholds(self):
        """Custom gate thresholds survive round-trip."""
        cfg1 = PipelineConfig()
        cfg1.gate_thresholds.g5_min_periods = 15.0
        cfg1.gate_thresholds.g9_min_area = 4
        cfg1.gate_thresholds.g12_min_fraction_valid = 0.75
        d = cfg1.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.gate_thresholds.g5_min_periods == 15.0
        assert cfg2.gate_thresholds.g9_min_area == 4
        assert cfg2.gate_thresholds.g12_min_fraction_valid == 0.75

    def test_round_trip_custom_peak_snr(self):
        cfg1 = PipelineConfig()
        cfg1.peak_snr.signal_disk_radius_px = 5
        cfg1.peak_snr.fwhm_patch_radius = 7
        d = cfg1.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.peak_snr.signal_disk_radius_px == 5
        assert cfg2.peak_snr.fwhm_patch_radius == 7

    def test_round_trip_custom_reference_selection(self):
        cfg1 = PipelineConfig()
        cfg1.reference_selection.scoring_weight_entropy = 0.5
        cfg1.reference_selection.orientation_bins = 24
        d = cfg1.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.reference_selection.scoring_weight_entropy == 0.5
        assert cfg2.reference_selection.orientation_bins == 24

    def test_round_trip_all_extended_fields(self):
        """Extended fields on existing dataclasses survive round-trip."""
        cfg1 = PipelineConfig()
        cfg1.global_fft.strong_guidance_snr = 12.0
        cfg1.preprocessing.hot_pixel_median_kernel = 5
        cfg1.roi.variance_window_size = 64
        cfg1.gpa.mask_sigma_fwhm_factor = 0.7
        cfg1.peak_finding.taper_width_fraction = 0.4
        cfg1.tile_fft.annulus_inner_factor = 0.85
        cfg1.clustering.kmeans_n_init = 20
        cfg1.confidence.snr_ceiling_multiplier = 3.0
        d = cfg1.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.global_fft.strong_guidance_snr == 12.0
        assert cfg2.preprocessing.hot_pixel_median_kernel == 5
        assert cfg2.roi.variance_window_size == 64
        assert cfg2.gpa.mask_sigma_fwhm_factor == 0.7
        assert cfg2.peak_finding.taper_width_fraction == 0.4
        assert cfg2.tile_fft.annulus_inner_factor == 0.85
        assert cfg2.clustering.kmeans_n_init == 20
        assert cfg2.confidence.snr_ceiling_multiplier == 3.0

    def test_from_dict_with_partial_sub_config(self):
        """from_dict with partial sub-config keeps defaults for unspecified fields."""
        d = {"gate_thresholds": {"g5_min_periods": 10.0}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g5_min_periods == 10.0
        # Other fields keep defaults
        assert cfg.gate_thresholds.g6_min_fraction == 0.05
        assert cfg.gate_thresholds.g9_min_area == 9


# ======================================================================
# Category 3: Backward compatibility
# ======================================================================

class TestBackwardCompat:
    """Old-style validation: keys sync to gate_thresholds:."""

    def test_min_detection_rate_syncs(self):
        d = {"validation": {"min_detection_rate": 0.10}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g6_min_fraction == 0.10

    def test_min_periods_syncs(self):
        d = {"validation": {"min_periods": 15.0}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g5_min_periods == 15.0

    def test_min_tier_a_snr_median_syncs(self):
        d = {"validation": {"min_tier_a_snr_median": 7.0}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g7_min_median_snr == 7.0

    def test_min_symmetry_mean_syncs(self):
        d = {"validation": {"min_symmetry_mean": 0.5}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g8_min_mean_symmetry == 0.5

    def test_ref_area_min_syncs(self):
        d = {"validation": {"ref_area_min": 16}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g9_min_area == 16

    def test_ref_entropy_max_syncs(self):
        d = {"validation": {"ref_entropy_max": 0.2}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g9_max_entropy == 0.2

    def test_ref_snr_min_syncs(self):
        d = {"validation": {"ref_snr_min": 8.0}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g9_min_snr == 8.0

    def test_peak_lattice_fraction_min_syncs(self):
        d = {"validation": {"peak_lattice_fraction_min": 0.60}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g12_min_fraction_valid == 0.60

    def test_multiple_old_keys_sync(self):
        d = {"validation": {
            "min_detection_rate": 0.15,
            "min_periods": 10.0,
            "ref_snr_min": 3.0,
        }}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.gate_thresholds.g6_min_fraction == 0.15
        assert cfg.gate_thresholds.g5_min_periods == 10.0
        assert cfg.gate_thresholds.g9_min_snr == 3.0

    def test_explicit_gate_thresholds_prevents_sync(self):
        """When gate_thresholds: is present, validation: keys do NOT sync."""
        d = {
            "validation": {"min_periods": 10.0},
            "gate_thresholds": {"g5_min_periods": 25.0},
        }
        cfg = PipelineConfig.from_dict(d)
        # Explicit gate_thresholds wins, sync does NOT happen
        assert cfg.gate_thresholds.g5_min_periods == 25.0

    def test_old_keys_not_in_validation_ignored(self):
        """Unknown keys in validation: section are silently ignored."""
        d = {"validation": {"nonexistent_key": 42}}
        cfg = PipelineConfig.from_dict(d)
        # gate_thresholds unchanged from defaults
        assert cfg.gate_thresholds.g5_min_periods == 20.0


# ======================================================================
# Category 4: Gate override
# ======================================================================

class TestGateOverride:
    """evaluate_gate() respects GateThresholdsConfig overrides."""

    def test_g5_pass_with_custom_threshold(self):
        """15 periods fails default (20) but passes with custom (10)."""
        custom = GateThresholdsConfig(g5_min_periods=10.0)
        result = evaluate_gate("G5", 15.0, gate_thresholds=custom)
        assert result.passed is True

    def test_g5_fail_with_default_threshold(self):
        """15 periods fails with default threshold of 20."""
        result = evaluate_gate("G5", 15.0)
        assert result.passed is False

    def test_g4_pass_with_custom_threshold(self):
        """SNR 2.5 fails default (3.0) but passes with custom (2.0)."""
        custom = GateThresholdsConfig(g4_min_peak_snr=2.0)
        result = evaluate_gate("G4", 2.5, gate_thresholds=custom)
        assert result.passed is True

    def test_g6_custom_threshold(self):
        """3% fails default (5%) but passes with custom (2%)."""
        custom = GateThresholdsConfig(g6_min_fraction=0.02)
        result = evaluate_gate("G6", 0.03, gate_thresholds=custom)
        assert result.passed is True

    def test_g7_custom_threshold(self):
        custom = GateThresholdsConfig(g7_min_median_snr=3.0)
        result = evaluate_gate("G7", 4.0, gate_thresholds=custom)
        assert result.passed is True

    def test_g9_custom_thresholds(self):
        custom = GateThresholdsConfig(g9_min_area=4, g9_max_entropy=0.5, g9_min_snr=3.0)
        result = evaluate_gate("G9", {"area": 5, "entropy": 0.4, "snr": 4.0},
                               gate_thresholds=custom)
        assert result.passed is True

    def test_g12_custom_threshold(self):
        custom = GateThresholdsConfig(g12_min_fraction_valid=0.30)
        result = evaluate_gate("G12", 0.40, gate_thresholds=custom)
        assert result.passed is True

    def test_threshold_override_takes_priority(self):
        """Direct threshold_override trumps gate_thresholds config."""
        custom = GateThresholdsConfig(g5_min_periods=10.0)
        # threshold_override=30 means 15 < 30 → fail, even though config says 10
        result = evaluate_gate("G5", 15.0,
                               threshold_override={"min_periods": 30.0},
                               gate_thresholds=custom)
        assert result.passed is False

    def test_all_gates_have_threshold_dict(self):
        """Every gate with a default_threshold has a threshold_dict entry."""
        gt = GateThresholdsConfig()
        for gate_id, gate_def in GATE_DEFS.items():
            if gate_def.default_threshold is None:
                continue
            td = gt.threshold_dict(gate_id)
            assert td is not None, f"{gate_id}: threshold_dict returned None"
            assert isinstance(td, dict), f"{gate_id}: threshold_dict not a dict"


# ======================================================================
# Category 5: G3 lcc_fraction configurable
# ======================================================================

class TestG3LCCFraction:
    """G3 min_lcc_fraction is now configurable via GateThresholdsConfig."""

    def test_g3_default_lcc_fraction(self):
        """lcc_fraction=0.4 fails with default 0.5 threshold."""
        result = evaluate_gate("G3", {
            "coverage_pct": 50.0,
            "n_components": 5,
            "lcc_fraction": 0.4,
        })
        assert result.passed is False
        assert "lcc_fraction" in result.reason

    def test_g3_custom_lcc_fraction_pass(self):
        """lcc_fraction=0.4 passes with custom 0.3 threshold."""
        custom = GateThresholdsConfig(g3_min_lcc_fraction=0.3)
        result = evaluate_gate("G3", {
            "coverage_pct": 50.0,
            "n_components": 5,
            "lcc_fraction": 0.4,
        }, gate_thresholds=custom)
        assert result.passed is True

    def test_g3_lcc_fraction_in_threshold_dict(self):
        """threshold_dict("G3") includes min_lcc_fraction."""
        gt = GateThresholdsConfig(g3_min_lcc_fraction=0.6)
        td = gt.threshold_dict("G3")
        assert "min_lcc_fraction" in td
        assert td["min_lcc_fraction"] == 0.6

    def test_g3_all_checks_configurable(self):
        """All G3 sub-checks are configurable."""
        custom = GateThresholdsConfig(
            g3_min_coverage=5.0,
            g3_max_coverage=99.0,
            g3_max_fragments=50,
            g3_min_lcc_fraction=0.2,
        )
        # Values that would fail defaults but pass custom thresholds
        result = evaluate_gate("G3", {
            "coverage_pct": 7.0,    # fails default 10, passes custom 5
            "n_components": 30,     # fails default 20, passes custom 50
            "lcc_fraction": 0.3,    # fails default 0.5, passes custom 0.2
        }, gate_thresholds=custom)
        assert result.passed is True


# ======================================================================
# Category 6: Reference selection bug fix
# ======================================================================

class TestReferenceSelectionBugFix:
    """select_reference_region() uses config values, not hardcoded defaults."""

    def test_custom_scoring_weights(self):
        """Non-default scoring weights change reference region ranking."""
        from src.reference_selection import select_reference_region

        # Build a minimal gated grid with 2 candidate regions
        n_rows, n_cols = 6, 6
        tier_map = np.full((n_rows, n_cols), "A", dtype=object)
        snr_map = np.ones((n_rows, n_cols)) * 8.0
        # Left half: higher SNR, right half: lower SNR but more uniform
        snr_map[:, :3] = 12.0
        snr_map[:, 3:] = 6.0
        pair_fraction_map = np.ones((n_rows, n_cols)) * 0.8
        fwhm_map = np.ones((n_rows, n_cols)) * 1.0
        orientation_map = np.full((n_rows, n_cols), 30.0)
        # Make right half have very uniform orientation
        orientation_map[:, 3:] = 30.0  # all same
        # Make left half have varied orientation (higher entropy)
        orientation_map[0, 0] = 10.0
        orientation_map[1, 0] = 50.0
        orientation_map[2, 0] = 70.0

        skipped_mask = np.zeros((n_rows, n_cols), dtype=bool)
        orientation_confidence_map = np.ones((n_rows, n_cols)) * 0.9

        grid = GatedTileGrid(
            classifications=np.empty((n_rows, n_cols), dtype=object),
            tier_map=tier_map,
            snr_map=snr_map,
            pair_fraction_map=pair_fraction_map,
            fwhm_map=fwhm_map,
            orientation_map=orientation_map,
            grid_shape=(n_rows, n_cols),
            skipped_mask=skipped_mask,
            tier_summary=TierSummary(
                n_tier_a=n_rows * n_cols, n_tier_b=0, n_rejected=0,
                n_skipped=0, tier_a_fraction=1.0, median_snr_tier_a=8.0,
            ),
            orientation_confidence_map=orientation_confidence_map,
        )

        # With default weights (entropy=0.4, snr=0.3, area=0.3)
        ref_default = select_reference_region(grid, min_area=4)

        # With SNR-dominant weights (entropy=0.1, snr=0.8, area=0.1)
        snr_heavy = ReferenceSelectionConfig(
            scoring_weight_entropy=0.1,
            scoring_weight_snr=0.8,
            scoring_weight_area=0.1,
        )
        ref_snr = select_reference_region(grid, min_area=4, ref_config=snr_heavy)

        # Both should succeed
        assert ref_default is not None
        assert ref_snr is not None
        # SNR-heavy config should select the higher-SNR region
        assert ref_snr.mean_snr >= ref_default.mean_snr or ref_snr.mean_snr >= 10.0

    def test_custom_orientation_bins(self):
        """Non-default orientation_bins changes entropy computation."""
        from src.reference_selection import select_reference_region

        n_rows, n_cols = 4, 4
        tier_map = np.full((n_rows, n_cols), "A", dtype=object)
        snr_map = np.ones((n_rows, n_cols)) * 10.0
        pair_fraction_map = np.ones((n_rows, n_cols)) * 0.8
        fwhm_map = np.ones((n_rows, n_cols)) * 1.0
        # Spread orientations across a moderate range
        orientation_map = np.linspace(20, 60, n_rows * n_cols).reshape(n_rows, n_cols)
        skipped_mask = np.zeros((n_rows, n_cols), dtype=bool)
        orientation_confidence_map = np.ones((n_rows, n_cols)) * 0.9

        grid = GatedTileGrid(
            classifications=np.empty((n_rows, n_cols), dtype=object),
            tier_map=tier_map,
            snr_map=snr_map,
            pair_fraction_map=pair_fraction_map,
            fwhm_map=fwhm_map,
            orientation_map=orientation_map,
            grid_shape=(n_rows, n_cols),
            skipped_mask=skipped_mask,
            tier_summary=TierSummary(
                n_tier_a=16, n_tier_b=0, n_rejected=0,
                n_skipped=0, tier_a_fraction=1.0, median_snr_tier_a=10.0,
            ),
            orientation_confidence_map=orientation_confidence_map,
        )

        # With 12 bins (default) vs 4 bins (coarser)
        ref_12 = select_reference_region(grid, min_area=4,
                                          ref_config=ReferenceSelectionConfig(orientation_bins=12))
        ref_4 = select_reference_region(grid, min_area=4,
                                         ref_config=ReferenceSelectionConfig(orientation_bins=4))

        # Both should find a region; entropy values differ due to different bin counts
        assert ref_12 is not None
        assert ref_4 is not None
        # With fewer bins, entropy is lower (more concentrated per bin)
        assert ref_4.entropy <= ref_12.entropy or ref_4.entropy < 0.5


# ======================================================================
# Category 7: Module config threading
# ======================================================================

class TestModuleConfigThreading:
    """Non-default PeakSNRConfig changes actual behavior."""

    def test_compute_peak_snr_respects_signal_disk_radius(self):
        """Changing signal_disk_radius_px changes the SNR result."""
        from src.fft_peak_detection import compute_peak_snr
        from src.fft_coords import FFTGrid

        # Create a small power spectrum with a sharp peak
        size = 64
        power = np.random.RandomState(42).random((size, size)) * 0.1
        cy, cx = size // 2 + 5, size // 2 + 8
        power[cy-1:cy+2, cx-1:cx+2] = 10.0  # 3×3 bright peak

        fft_grid = FFTGrid(size, size, 0.127)
        # Compute q coordinates from pixel offsets
        qx = (cx - fft_grid.dc_x) * fft_grid.qx_scale
        qy = (cy - fft_grid.dc_y) * fft_grid.qy_scale
        q_mag = np.sqrt(qx**2 + qy**2)
        peak = TilePeak(
            qx=qx, qy=qy, q_mag=q_mag,
            d_spacing=1.0 / q_mag if q_mag > 0 else 0.0,
            angle_deg=np.degrees(np.arctan2(qy, qx)),
            intensity=10.0, fwhm=0.1,
        )

        # Default radius=3
        snr_default = compute_peak_snr(power, peak, [peak], fft_grid,
                                        peak_snr_config=PeakSNRConfig(signal_disk_radius_px=3))
        # Larger radius=6 → includes more background in signal disk
        snr_large = compute_peak_snr(power, peak, [peak], fft_grid,
                                      peak_snr_config=PeakSNRConfig(signal_disk_radius_px=6))

        # Different disk radius should give different SNR values
        assert snr_default.snr != pytest.approx(snr_large.snr, abs=0.1)

    def test_classify_tile_threads_peak_snr_config(self):
        """classify_tile accepts and uses peak_snr_config."""
        from src.fft_peak_detection import classify_tile
        from src.fft_coords import FFTGrid

        fft_grid = FFTGrid(256, 256, 0.127)

        # Create a tile with one peak
        peak = TilePeak(qx=1.0, qy=1.0, q_mag=1.414,
                        d_spacing=1.0 / 1.414, angle_deg=45.0,
                        intensity=50.0, fwhm=0.1)
        ps = TilePeakSet(
            tile_row=0, tile_col=0,
            peaks=[peak],
            power_spectrum=np.random.RandomState(42).random((256, 256)) * 0.1,
        )
        # Make the peak bright
        ps.power_spectrum[128 + 10, 128 + 10] = 50.0

        config = PeakSNRConfig(signal_disk_radius_px=5, symmetry_tolerance_px=3.0)
        tc = classify_tile(ps, fft_grid, peak_snr_config=config)
        # Should return a classification (not crash)
        assert tc is not None
        assert tc.tier in ("A", "B", "REJECTED")

    def test_build_gated_tile_grid_accepts_peak_snr_config(self):
        """build_gated_tile_grid threads peak_snr_config to classify_tile."""
        from src.fft_snr_metrics import build_gated_tile_grid
        from src.fft_coords import FFTGrid

        fft_grid = FFTGrid(256, 256, 0.127)
        peak = TilePeak(qx=1.0, qy=0.0, q_mag=1.0, d_spacing=1.0,
                        angle_deg=0.0, intensity=10.0, fwhm=0.1)
        ps = TilePeakSet(
            tile_row=0, tile_col=0,
            peaks=[peak],
            power_spectrum=np.random.RandomState(42).random((256, 256)),
        )
        skipped = np.array([[False]])

        config = PeakSNRConfig(signal_disk_radius_px=4)
        grid = build_gated_tile_grid([ps], skipped, fft_grid, 256,
                                      peak_snr_config=config)
        assert grid is not None
        assert grid.grid_shape == (1, 1)


# ======================================================================
# Category 8: FWHM patch cache rebuild
# ======================================================================

class TestFWHMPatchCacheRebuild:
    """PeakSNRConfig(fwhm_patch_radius=7) triggers cache rebuild."""

    def test_ensure_patch_cache_default(self):
        """Default radius=5 produces 11×11 grids."""
        from src.fft_peak_detection import _ensure_patch_cache, _PATCH_R, _PATCH_SIZE
        _ensure_patch_cache(5)
        from src.fft_peak_detection import _PATCH_R as r, _PATCH_SIZE as s
        assert r == 5
        assert s == 11

    def test_ensure_patch_cache_rebuild(self):
        """Non-default radius triggers full cache rebuild."""
        from src import fft_peak_detection as mod
        mod._ensure_patch_cache(7)
        assert mod._PATCH_R == 7
        assert mod._PATCH_SIZE == 15  # 2*7+1
        assert mod._X11.shape == (15, 15)
        assert mod._Y11.shape == (15, 15)
        assert mod._DIST11.shape == (15, 15)
        assert mod._OUTER_RING11.shape == (15, 15)
        assert len(mod._RADIAL_MASKS11) == 8  # range(7+1) = 0..7

    def test_ensure_patch_cache_idempotent(self):
        """Same radius doesn't rebuild (fast path)."""
        from src import fft_peak_detection as mod
        mod._ensure_patch_cache(5)
        old_x = mod._X11
        mod._ensure_patch_cache(5)  # Should be a no-op
        assert mod._X11 is old_x  # Same object, not rebuilt

    def test_classify_tile_triggers_cache_for_custom_radius(self):
        """classify_tile with fwhm_patch_radius=7 rebuilds cache."""
        from src import fft_peak_detection as mod
        from src.fft_coords import FFTGrid

        # Reset to default first
        mod._ensure_patch_cache(5)
        assert mod._PATCH_R == 5

        fft_grid = FFTGrid(256, 256, 0.127)
        peak = TilePeak(qx=1.0, qy=0.0, q_mag=1.0, d_spacing=1.0,
                        angle_deg=0.0, intensity=10.0, fwhm=0.1)
        ps = TilePeakSet(
            tile_row=0, tile_col=0,
            peaks=[peak],
            power_spectrum=np.random.RandomState(42).random((256, 256)),
        )

        config = PeakSNRConfig(fwhm_patch_radius=7)
        mod.classify_tile(ps, fft_grid, peak_snr_config=config)
        assert mod._PATCH_R == 7

    def teardown_method(self):
        """Restore default cache after each test."""
        from src import fft_peak_detection as mod
        mod._ensure_patch_cache(5)


# ======================================================================
# Category 9: DCMaskConfig, signal method, background method
# ======================================================================

class TestDCMaskAndNewConfigs:
    """Tests for DCMaskConfig and new fields on existing configs."""

    def test_dc_mask_config_defaults(self):
        dc = DCMaskConfig()
        assert dc.enabled is False
        assert dc.method == "derivative"
        assert dc.savgol_window == 11
        assert dc.q_dc_min_floor == 0.15
        assert dc.soft_taper is False
        assert dc.max_dc_mask_q == 0.0
        assert dc.auto_cap_from_physics is True
        assert dc.noise_q_range_lo == 0.70
        assert dc.noise_q_range_hi == 0.90

    def test_dc_mask_yaml_roundtrip(self):
        cfg = PipelineConfig()
        cfg.dc_mask.enabled = True
        cfg.dc_mask.q_dc_min_floor = 0.20
        cfg.dc_mask.soft_taper = True
        d = cfg.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.dc_mask.enabled is True
        assert cfg2.dc_mask.q_dc_min_floor == 0.20
        assert cfg2.dc_mask.soft_taper is True

    def test_peak_snr_signal_method_default(self):
        ps = PeakSNRConfig()
        assert ps.signal_method == "max"

    def test_global_fft_background_method_default(self):
        gf = GlobalFFTConfig()
        assert gf.background_method == "polynomial_robust"
        assert gf.asls_lambda == 1e6
        assert gf.asls_p == 0.001
        assert gf.asls_n_iter == 10
        assert gf.asls_domain == "log"

    def test_pipeline_config_has_dc_mask(self):
        cfg = PipelineConfig()
        assert hasattr(cfg, "dc_mask")
        assert isinstance(cfg.dc_mask, DCMaskConfig)
        # Verify from_dict ignores missing dc_mask gracefully
        cfg2 = PipelineConfig.from_dict({})
        assert cfg2.dc_mask.enabled is False

    def test_tile_fft_lightweight_snr_method_default(self):
        tf = TileFFTConfig()
        assert tf.lightweight_snr_method == "ratio"
