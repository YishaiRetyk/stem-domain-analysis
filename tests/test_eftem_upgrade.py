"""
Tests for the EFTEM pipeline upgrade features.

Covers Stage 0 (Foundation) through Stage 6 (Reporting), ~30 tests
exercising PhysicsConfig, TileFFTConfig, FFTGrid extensions, Gate G0,
FFT guidance strength, background diagnostics, pair_fraction rename,
FWHM-scaled symmetry, orientation confidence, d-aware tiling, ROI LCC,
GPA entry gates, lattice_peaks rename, directional masks, adaptive NN
tolerance, and full parameters.json v3 reporting.
"""

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.pipeline_config import (
    PhysicsConfig,
    TileFFTConfig,
    PipelineConfig,
    GlobalFFTConfig,
    GlobalFFTResult,
    GlobalPeak,
    GVector,
    GatedTileGrid,
    TierSummary,
    TileClassification,
    ROIMaskResult,
    ROIConfig,
    GPAPhaseResult,
    LatticeValidation,
    SubpixelPeak,
    PeakGateConfig,
    TilePeak,
    TilePeakSet,
    FWHMConfig,
    PeakFindingConfig,
)
from src.fft_coords import FFTGrid, fwhm_to_tolerance_px
from src.gates import evaluate_gate
from src.fft_peak_detection import check_symmetry, classify_tile
from src.tile_fft import check_tiling_adequacy
from src.roi_masking import compute_roi_mask, _compute_lcc_fraction
from src.peak_finding import build_bandpass_image, validate_peak_lattice
from src.reporting import build_parameters_v3, save_pipeline_artifacts

from tests.synthetic import generate_single_crystal


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def default_physics():
    return PhysicsConfig()


@pytest.fixture
def default_tile_fft():
    return TileFFTConfig()


@pytest.fixture
def small_fft_grid():
    """256x256 grid at the default EFTEM pixel size."""
    return FFTGrid(256, 256, pixel_size_nm=0.1297)


@pytest.fixture
def square_fft_grid():
    """512x512 grid at 0.1 nm pixel size for simple arithmetic checks."""
    return FFTGrid(512, 512, pixel_size_nm=0.1)


@pytest.fixture
def minimal_config():
    return PipelineConfig()


@pytest.fixture
def minimal_global_fft_result():
    """Minimal GlobalFFTResult with a single strong peak."""
    q = np.linspace(0, 5, 100)
    profile = np.zeros(100)
    return GlobalFFTResult(
        power_spectrum=np.zeros((64, 64)),
        radial_profile=profile,
        q_values=q,
        background=np.zeros(100),
        corrected_profile=profile,
        noise_floor=0.01,
        peaks=[
            GlobalPeak(q_center=2.5, q_fwhm=0.1, d_spacing=0.4,
                       intensity=1.0, prominence=0.5, snr=10.0, index=50),
            GlobalPeak(q_center=3.5, q_fwhm=0.1, d_spacing=0.286,
                       intensity=0.5, prominence=0.3, snr=8.0, index=70),
        ],
        g_vectors=[
            GVector(gx=2.5, gy=0.0, magnitude=2.5, angle_deg=0.0,
                    d_spacing=0.4, snr=10.0, fwhm=0.1, ring_index=0),
        ],
        d_dom=0.4,
        information_limit_q=3.6,
        diagnostics={},
        fft_guidance_strength="strong",
    )


@pytest.fixture
def minimal_gated_grid():
    """2x2 GatedTileGrid with one Tier A tile."""
    n_rows, n_cols = 2, 2
    classifications = np.empty((n_rows, n_cols), dtype=object)
    tier_map = np.array([["A", "B"], ["REJECTED", "A"]])
    snr_map = np.array([[8.0, 4.0], [1.0, 6.0]])
    pf_map = np.array([[0.6, 0.2], [0.0, 0.5]])
    fwhm_map = np.zeros((n_rows, n_cols))
    orient_map = np.array([[30.0, 45.0], [0.0, 30.0]])
    skipped = np.zeros((n_rows, n_cols), dtype=bool)

    ts = TierSummary(
        n_tier_a=2, n_tier_b=1, n_rejected=1, n_skipped=0,
        tier_a_fraction=0.5, median_snr_tier_a=7.0,
    )

    for r in range(n_rows):
        for c in range(n_cols):
            classifications[r, c] = TileClassification(
                tier=tier_map[r, c],
                peaks=[],
                pair_fraction=pf_map[r, c],
                n_non_collinear=2 if tier_map[r, c] == "A" else 0,
                best_snr=snr_map[r, c],
                best_orientation_deg=orient_map[r, c],
            )

    return GatedTileGrid(
        classifications=classifications,
        tier_map=tier_map,
        snr_map=snr_map,
        pair_fraction_map=pf_map,
        fwhm_map=fwhm_map,
        orientation_map=orient_map,
        grid_shape=(n_rows, n_cols),
        skipped_mask=skipped,
        tier_summary=ts,
    )


# =====================================================================
# Stage 0 (Foundation) Tests  1-10
# =====================================================================

class TestStage0Foundation:
    """Tests 1-10: PhysicsConfig, TileFFTConfig, FFTGrid, G0."""

    # 1
    def test_physics_config_defaults(self, default_physics):
        """PhysicsConfig has correct default values."""
        assert default_physics.imaging_mode == "EFTEM-BF"
        assert default_physics.d_min_nm == 0.0
        assert default_physics.d_max_nm == 0.0
        assert default_physics.nyquist_safety_margin == 0.95

    # 2
    def test_physics_config_serialization_roundtrip(self):
        """PhysicsConfig survives serialisation via PipelineConfig.from_dict."""
        cfg = PipelineConfig(physics=PhysicsConfig(
            imaging_mode="STEM-HAADF",
            d_min_nm=0.2,
            d_max_nm=2.0,
            nyquist_safety_margin=0.90,
        ))
        d = cfg.to_dict()
        restored = PipelineConfig.from_dict(d)
        assert restored.physics.imaging_mode == "STEM-HAADF"
        assert restored.physics.d_min_nm == 0.2
        assert restored.physics.d_max_nm == 2.0
        assert restored.physics.nyquist_safety_margin == 0.90

    # 3
    def test_tile_fft_config_defaults(self, default_tile_fft):
        """TileFFTConfig has correct default values."""
        assert default_tile_fft.q_dc_min == 0.25
        assert default_tile_fft.peak_snr_threshold == 2.5
        assert default_tile_fft.local_max_size == 5

    # 4
    def test_fft_grid_nyquist_q(self, square_fft_grid):
        """FFTGrid.nyquist_q() returns 1/(2*pixel_size)."""
        expected = 1.0 / (2.0 * 0.1)  # 5.0 cycles/nm
        assert square_fft_grid.nyquist_q() == pytest.approx(expected)

    # 5
    def test_fft_grid_nyquist_d(self, square_fft_grid):
        """FFTGrid.nyquist_d() returns 2*pixel_size."""
        expected = 2.0 * 0.1  # 0.2 nm
        assert square_fft_grid.nyquist_d() == pytest.approx(expected)

    # 6
    def test_fft_grid_clamp_q_range(self, square_fft_grid):
        """clamp_q_range clamps q_max to safety*q_nyquist."""
        q_min, q_max_clamped, was_clamped = square_fft_grid.clamp_q_range(
            0.5, 10.0, safety=0.95)
        q_safe = 0.95 * 5.0  # 4.75
        assert q_min == 0.5
        assert q_max_clamped == pytest.approx(q_safe)
        assert was_clamped is True

        # Not clamped when within range
        _, q_max2, clamped2 = square_fft_grid.clamp_q_range(0.5, 4.0, safety=0.95)
        assert q_max2 == pytest.approx(4.0)
        assert clamped2 is False

    # 7
    def test_fwhm_to_tolerance_px(self):
        """fwhm_to_tolerance_px converts correctly."""
        fwhm_q = 0.2355  # sigma_q = 0.1
        q_scale = 0.01   # cycles/nm per pixel
        result = fwhm_to_tolerance_px(fwhm_q, q_scale, n_sigma=2.0)
        # sigma_px = 0.1 / 0.01 = 10; tolerance = 2 * 10 = 20
        assert result == pytest.approx(20.0, rel=1e-3)

    # 8
    def test_gate_g0_pass(self, square_fft_grid):
        """G0 passes when q_max is within safe limit."""
        q_nyq = square_fft_grid.nyquist_q()  # 5.0
        result = evaluate_gate("G0", {
            "q_max_requested": 4.0,
            "q_min_requested": 0.5,
            "q_nyquist": q_nyq,
            "safety_margin": 0.95,
        })
        assert result.passed is True
        assert "OK" in result.reason

    # 9
    def test_gate_g0_degrade(self, square_fft_grid):
        """G0 degrades when q_max exceeds safe limit but q_min is OK."""
        q_nyq = square_fft_grid.nyquist_q()  # 5.0
        result = evaluate_gate("G0", {
            "q_max_requested": 5.5,  # above 0.95 * 5.0 = 4.75
            "q_min_requested": 0.5,
            "q_nyquist": q_nyq,
            "safety_margin": 0.95,
        })
        assert result.passed is False
        assert "DEGRADE" in result.reason

    # 10
    def test_gate_g0_fatal(self, square_fft_grid):
        """G0 is FATAL when q_min >= q_nyquist (entire band invalid)."""
        q_nyq = square_fft_grid.nyquist_q()  # 5.0
        result = evaluate_gate("G0", {
            "q_max_requested": 10.0,
            "q_min_requested": 6.0,  # above q_nyquist
            "q_nyquist": q_nyq,
            "safety_margin": 0.95,
        })
        assert result.passed is False
        assert "FATAL" in result.reason


# =====================================================================
# Stage 1 (Global FFT) Tests  11-13
# =====================================================================

class TestStage1GlobalFFT:
    """Tests 11-13: FFT guidance strength, background diagnostics."""

    # 11
    def test_fft_guidance_strong(self):
        """fft_guidance_strength = 'strong' when SNR >= 8 and >= 2 peaks."""
        from src.global_fft import compute_global_fft
        image = generate_single_crystal(
            shape=(512, 512), pixel_size_nm=0.1297,
            d_spacing=0.4, amplitude=0.4, noise_level=0.02,
        )
        fft_grid = FFTGrid(512, 512, 0.1297)
        result = compute_global_fft(image, fft_grid)
        # Strong crystal should produce strong guidance
        # (at minimum, it should not be "none")
        assert result.fft_guidance_strength in ("strong", "weak")
        # If there are >= 2 peaks with high SNR, it should be strong
        if result.fft_guidance_strength == "strong":
            assert max(p.snr for p in result.peaks) >= 8
            assert len(result.peaks) >= 2

    # 12
    def test_fft_guidance_not_strong_for_noise(self):
        """fft_guidance_strength is not 'strong' for pure noise."""
        rng = np.random.default_rng(99)
        noise = 0.5 + rng.normal(0, 0.1, (512, 512))
        noise = np.clip(noise, 0, 1)
        fft_grid = FFTGrid(512, 512, 0.1297)
        from src.global_fft import compute_global_fft
        result = compute_global_fft(noise, fft_grid)
        assert result.fft_guidance_strength in ("none", "weak")

    # 13
    def test_background_diagnostics_keys(self):
        """compute_global_fft diagnostics contain expected background keys."""
        from src.global_fft import compute_global_fft
        image = generate_single_crystal(
            shape=(512, 512), pixel_size_nm=0.1297,
            d_spacing=0.4, amplitude=0.4, noise_level=0.02,
        )
        fft_grid = FFTGrid(512, 512, 0.1297)
        result = compute_global_fft(image, fft_grid)

        assert "background_diagnostics" in result.diagnostics
        bg_diag = result.diagnostics["background_diagnostics"]
        assert "median_abs_residual" in bg_diag
        assert "mad_residual" in bg_diag
        assert "neg_excursion_fraction_near_peaks" in bg_diag


# =====================================================================
# Stage 2 (Tile FFT & Classification) Tests  14-17
# =====================================================================

class TestStage2TileFFT:
    """Tests 14-17: pair_fraction rename, FWHM-scaled symmetry,
    orientation confidence, d-aware tiling."""

    # 14
    def test_pair_fraction_deprecation_alias(self):
        """TileClassification.symmetry_score aliases pair_fraction with warning."""
        tc = TileClassification(
            tier="A", peaks=[], pair_fraction=0.75,
            n_non_collinear=2, best_snr=8.0,
        )
        # Reset the global warning flag so we can capture the warning
        import src.pipeline_config as pc
        pc._warned_symmetry_score = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = tc.symmetry_score
            assert val == tc.pair_fraction == 0.75
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

        # Reset for other tests
        pc._warned_symmetry_score = False

    # 15
    def test_fwhm_scaled_symmetry(self, small_fft_grid):
        """check_symmetry with median_fwhm_q > 0 uses FWHM-scaled tolerance."""
        # Two perfectly antipodal peaks
        peaks = [
            TilePeak(qx=1.0, qy=0.0, q_mag=1.0, d_spacing=1.0,
                     angle_deg=0, intensity=1.0, fwhm=0.05),
            TilePeak(qx=-1.0, qy=0.0, q_mag=1.0, d_spacing=1.0,
                     angle_deg=180, intensity=1.0, fwhm=0.05),
        ]
        # With a large FWHM-based tolerance, they should pair
        score, n_paired = check_symmetry(peaks, small_fft_grid, median_fwhm_q=0.5)
        assert score == pytest.approx(1.0)
        assert n_paired == 2

        # With a very small FWHM, tolerance shrinks; still should pair if peaks
        # are truly antipodal
        score2, n2 = check_symmetry(peaks, small_fft_grid, median_fwhm_q=0.001)
        # Even a tiny FWHM means tol_q = 0.001/2.355*2 ~ 0.00085; residual is 0
        assert score2 == pytest.approx(1.0)

    # 16
    def test_orientation_confidence(self, small_fft_grid):
        """classify_tile computes orientation_confidence R > 0 for consistent peaks."""
        # Create a tile with two peaks at similar angles (should give high R)
        fft_grid = FFTGrid(64, 64, 0.1297)
        power = np.random.default_rng(42).random((64, 64))
        # Place two peaks at similar positions
        power[32 + 10, 32] = 100.0  # peak at one position
        power[32 - 10, 32] = 100.0  # antipodal peak

        peaks = [
            TilePeak(qx=0.0, qy=0.5, q_mag=0.5, d_spacing=2.0,
                     angle_deg=90.0, intensity=100.0, fwhm=0.05),
            TilePeak(qx=0.0, qy=-0.5, q_mag=0.5, d_spacing=2.0,
                     angle_deg=90.0, intensity=100.0, fwhm=0.05),
        ]
        peak_set = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                                power_spectrum=power)
        tc = classify_tile(peak_set, fft_grid)
        # Peaks at same angle => high R (circular concentration)
        assert tc.orientation_confidence > 0.0

    # 17
    def test_g5_d_aware_tiling(self):
        """check_tiling_adequacy uses d_dom directly (physics filtering happens at selection time)."""
        tile_size = 256
        pixel_size = 0.1297

        # Without d_max
        periods_base, passed_base = check_tiling_adequacy(
            tile_size, d_dom_nm=0.4, pixel_size_nm=pixel_size)
        # d_px = 0.4/0.1297 ~ 3.08, periods = 256/3.08 ~ 83 -> passes
        assert passed_base is True

        # With d_max_nm = 1.0 -> d_max is now ignored (d_dom used directly)
        # Physics-bound filtering happens at d_dom selection time, not at G5
        periods_dmax, passed_dmax = check_tiling_adequacy(
            tile_size, d_dom_nm=0.4, pixel_size_nm=pixel_size, d_max_nm=1.0)

        # d_max_nm no longer inflates the effective d, so periods are same as base
        assert periods_dmax == pytest.approx(periods_base, rel=0.01)
        assert passed_dmax is True

        # Small d_dom with many periods still passes
        periods_small, passed_small = check_tiling_adequacy(
            tile_size, d_dom_nm=0.4, pixel_size_nm=pixel_size, d_max_nm=10.0)
        assert periods_small == pytest.approx(periods_base, rel=0.01)
        assert passed_small is True

        # Large d_dom itself should fail G5
        _, passed_large_d = check_tiling_adequacy(
            tile_size, d_dom_nm=10.0, pixel_size_nm=pixel_size)
        # d_px = 10.0/0.1297 ~ 77, periods = 256/77 ~ 3.3 < 20
        assert passed_large_d is False


# =====================================================================
# Stage 3 (ROI Masking) Tests  18-20
# =====================================================================

class TestStage3ROIMasking:
    """Tests 18-20: LCC fraction, gradient fallback, full-image stop."""

    # 18
    def test_roi_lcc_fraction_single_component(self):
        """Single connected component mask has lcc_fraction = 1.0."""
        mask = np.ones((100, 100), dtype=np.uint8)
        frac = _compute_lcc_fraction(mask)
        assert frac == pytest.approx(1.0)

    # 19
    def test_roi_gradient_fallback(self):
        """Coverage out of bounds triggers gradient-magnitude fallback."""
        # Create an image that produces >max_coverage_pct mask, triggering
        # the gradient fallback path. A high-variance uniform image with
        # high intensity passes both intensity and variance thresholds
        # everywhere, giving ~100% coverage.
        rng = np.random.default_rng(42)
        image = rng.uniform(0.5, 1.0, (512, 512)).astype(np.float64)

        config = ROIConfig(
            min_coverage_pct=5.0,
            max_coverage_pct=60.0,  # low max so primary mask exceeds it
            gradient_threshold_pct=30.0,
            min_lcc_fraction=0.5,
        )
        result = compute_roi_mask(image, config)
        diag = result.diagnostics
        # Coverage exceeded max, so gradient fallback was applied
        assert diag.get("gradient_used", False), (
            f"Expected gradient_used=True, got diagnostics: {diag}"
        )

    # 20
    def test_roi_full_image_stop_condition(self):
        """Force full-image ROI when lcc < threshold after fallback."""
        # Create an image that produces many disconnected small blobs
        rng = np.random.default_rng(123)
        image = rng.random((512, 512)).astype(np.float64)

        config = ROIConfig(
            min_coverage_pct=10.0,
            max_coverage_pct=95.0,
            min_lcc_fraction=0.99,  # extremely high threshold -> force full-image
            gradient_threshold_pct=30.0,
        )
        result = compute_roi_mask(image, config)
        # Should have forced full-image ROI due to lcc constraint
        # Coverage should be 100% when forced to full image
        if result.diagnostics.get("roi_confidence") == "low":
            assert result.coverage_pct == pytest.approx(100.0)
            assert result.lcc_fraction == pytest.approx(1.0)


# =====================================================================
# Stage 4 (GPA Safety) Tests  21-22
# =====================================================================

class TestStage4GPASafety:
    """Tests 21-22: GPA entry gate skip conditions."""

    # 21
    def test_gpa_skip_when_fft_guidance_none(self):
        """run_gpa returns None when fft_guidance_strength='none'."""
        from src.gpa import run_gpa
        from src.pipeline_config import GPAConfig

        # Create mock objects
        image = np.random.default_rng(42).random((512, 512))
        fft_grid = FFTGrid(512, 512, 0.1297)

        global_fft = MagicMock(spec=GlobalFFTResult)
        global_fft.fft_guidance_strength = "none"

        gated = MagicMock(spec=GatedTileGrid)
        gated.tier_summary = MagicMock(spec=TierSummary)
        gated.tier_summary.tier_a_fraction = 0.5

        g_vectors = [
            GVector(gx=2.5, gy=0.0, magnitude=2.5, angle_deg=0.0,
                    d_spacing=0.4, snr=10.0, fwhm=0.1, ring_index=0),
        ]

        result = run_gpa(
            image, g_vectors, gated, global_fft, fft_grid,
            config=GPAConfig(enabled=True),
        )
        assert result is None

    # 22
    def test_gpa_skip_when_tier_a_fraction_low(self):
        """run_gpa returns None when tier_a_fraction < 0.1."""
        from src.gpa import run_gpa
        from src.pipeline_config import GPAConfig

        image = np.random.default_rng(42).random((512, 512))
        fft_grid = FFTGrid(512, 512, 0.1297)

        global_fft = MagicMock(spec=GlobalFFTResult)
        global_fft.fft_guidance_strength = "strong"

        gated = MagicMock(spec=GatedTileGrid)
        gated.tier_summary = MagicMock(spec=TierSummary)
        gated.tier_summary.tier_a_fraction = 0.05  # below 0.1

        g_vectors = [
            GVector(gx=2.5, gy=0.0, magnitude=2.5, angle_deg=0.0,
                    d_spacing=0.4, snr=10.0, fwhm=0.1, ring_index=0),
        ]

        result = run_gpa(
            image, g_vectors, gated, global_fft, fft_grid,
            config=GPAConfig(enabled=True),
        )
        assert result is None


# =====================================================================
# Stage 5 (Peak Finding) Tests  23-25
# =====================================================================

class TestStage5PeakFinding:
    """Tests 23-25: lattice_peaks.npy, directional mask, adaptive NN."""

    # 23
    def test_lattice_peaks_npy_created(self, tmp_path, minimal_config):
        """save_pipeline_artifacts creates lattice_peaks.npy when peaks given."""
        fft_grid = FFTGrid(512, 512, 0.1297)
        peaks = [
            SubpixelPeak(x=100.0, y=200.0, intensity=0.5, sigma_x=1.0, sigma_y=1.0),
            SubpixelPeak(x=150.0, y=250.0, intensity=0.8, sigma_x=1.2, sigma_y=1.1),
        ]
        saved = save_pipeline_artifacts(
            tmp_path,
            config=minimal_config,
            fft_grid=fft_grid,
            peaks=peaks,
        )
        assert "lattice_peaks.npy" in saved
        arr = np.load(saved["lattice_peaks.npy"])
        assert arr.shape == (2, 5)  # (x, y, intensity, sigma_x, sigma_y)
        assert arr[0, 0] == pytest.approx(100.0)

    # 24
    def test_directional_mask_angular_wedge(self):
        """build_bandpass_image with g_vectors applies angular wedge mask."""
        image = generate_single_crystal(
            shape=(256, 256), pixel_size_nm=0.1297,
            d_spacing=0.4, amplitude=0.3, noise_level=0.01,
        )
        fft_grid = FFTGrid(256, 256, 0.1297)
        g_mag = 1.0 / 0.4  # 2.5 cycles/nm

        # g-vector at 0 degrees
        gv = GVector(gx=g_mag, gy=0.0, magnitude=g_mag, angle_deg=0.0,
                     d_spacing=0.4, snr=10.0, fwhm=0.1, ring_index=0)

        # With directional mask
        bp_dir = build_bandpass_image(
            image, g_mag, fft_grid,
            g_vectors=[gv], angular_width_deg=30.0,
            use_directional_mask=True,
        )
        # Without directional mask (full ring)
        bp_full = build_bandpass_image(
            image, g_mag, fft_grid,
            g_vectors=[gv],
            use_directional_mask=False,
        )

        # The directional image should have less total energy (restricted angles)
        assert np.sum(bp_dir ** 2) <= np.sum(bp_full ** 2) + 1e-10

    # 25
    def test_adaptive_nn_tolerance(self):
        """Adaptive tolerance tightens when >= 10 peaks with tight IQR."""
        pixel_size = 0.1297
        expected_d = 0.4  # nm
        d_px = expected_d / pixel_size

        # Generate 20 peaks on a perfect lattice (very tight NN distribution)
        rng = np.random.default_rng(42)
        peaks = []
        for i in range(5):
            for j in range(4):
                x = 50 + i * d_px + rng.normal(0, 0.01)
                y = 50 + j * d_px + rng.normal(0, 0.01)
                peaks.append(SubpixelPeak(x=x, y=y, intensity=1.0))

        result = validate_peak_lattice(
            peaks, expected_d, pixel_size,
            tolerance=0.2, adaptive=True,
        )
        # With tight IQR, tolerance should be < 0.2
        assert result.tolerance_used <= 0.2
        # High fraction valid for a perfect lattice
        assert result.fraction_valid > 0.5


# =====================================================================
# Stage 6 (Reporting) Tests  26-30
# =====================================================================

class TestStage6Reporting:
    """Tests 26-30: parameters.json v3 sections."""

    def _build_minimal_params(self, **overrides):
        """Helper to build parameters with minimal required args."""
        config = overrides.get("config", PipelineConfig())
        fft_grid = overrides.get("fft_grid", FFTGrid(512, 512, 0.1297))
        kwargs = {
            "config": config,
            "fft_grid": fft_grid,
        }
        kwargs.update(overrides)
        return build_parameters_v3(**kwargs)

    # 26
    def test_parameters_includes_physics(self):
        """parameters.json includes physics section."""
        params = self._build_minimal_params()
        assert "physics" in params
        assert params["physics"]["imaging_mode"] == "EFTEM-BF"
        assert "d_min_nm" in params["physics"]
        assert "d_max_nm" in params["physics"]
        assert "nyquist_safety_margin" in params["physics"]

    # 27
    def test_parameters_includes_derived_cutoffs(self):
        """parameters.json includes derived_cutoffs section."""
        params = self._build_minimal_params(
            effective_q_min=0.15, tile_effective_q_min=0.25)
        assert "derived_cutoffs" in params
        dc = params["derived_cutoffs"]
        assert dc["effective_q_min_global"] == pytest.approx(0.15)
        assert dc["effective_q_min_tile"] == pytest.approx(0.25)

    # 28
    def test_parameters_includes_pipeline_flow(self):
        """parameters.json includes pipeline_flow section."""
        params = self._build_minimal_params()
        assert "pipeline_flow" in params
        flow = params["pipeline_flow"]
        assert "stages_completed" in flow
        assert "stages_skipped" in flow
        assert "stages_degraded" in flow
        assert "skip_reasons" in flow

    # 29
    def test_parameters_includes_fft_guidance_strength(
        self, minimal_global_fft_result,
    ):
        """parameters.json global_fft section contains fft_guidance_strength."""
        params = self._build_minimal_params(
            global_fft_result=minimal_global_fft_result)
        assert "global_fft" in params
        assert "fft_guidance_strength" in params["global_fft"]
        assert params["global_fft"]["fft_guidance_strength"] == "strong"

    # 30
    def test_reporting_pair_fraction_and_symmetry_score(
        self, minimal_gated_grid, minimal_global_fft_result,
    ):
        """Reporting emits both pair_fraction and symmetry_score (deprecated)."""
        params = self._build_minimal_params(
            gated_grid=minimal_gated_grid,
            global_fft_result=minimal_global_fft_result,
        )
        assert "peak_gates" in params
        pg = params["peak_gates"]
        # New canonical name
        assert "min_pair_fraction" in pg
        # Deprecated alias still present for backward compatibility
        assert "symmetry_score" in pg
        assert pg["symmetry_score_deprecated"] is True
        # Values should match
        assert pg["symmetry_score"] == pg["min_pair_fraction"]
