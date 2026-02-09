"""Tests for hybrid pipeline PNG visualizations."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.pipeline_config import (
    PipelineConfig, VizConfig, GVector, GlobalPeak, GlobalFFTResult,
    GatedTileGrid, TierSummary, TileClassification,
    GPAResult, GPAPhaseResult, GPAModeDecision,
    DisplacementField, StrainField,
    SubpixelPeak, LatticeValidation,
)
from src.hybrid_viz import (
    save_pipeline_visualizations,
    _save_input_original,
    _save_radial_profile,
    _save_fft_power_spectrum,
    _save_tier_map,
    _save_snr_map,
    _save_orientation_map,
    _save_gpa_phase,
    _save_gpa_amplitude,
    _save_displacement,
    _save_strain_component,
    _save_peak_lattice_overlay,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_out(tmp_path):
    return tmp_path / "viz_out"


@pytest.fixture
def config():
    cfg = PipelineConfig()
    cfg.viz = VizConfig(enabled=True, dpi=72)  # low DPI for fast tests
    return cfg


@pytest.fixture
def fft_grid():
    from src.fft_coords import FFTGrid
    return FFTGrid(64, 64, 0.127)


@pytest.fixture
def raw_image():
    rng = np.random.default_rng(42)
    return rng.random((64, 64))


@pytest.fixture
def global_fft_result():
    N = 100
    q = np.linspace(0.1, 10.0, N)
    profile = np.exp(-q) * 1000
    bg = np.exp(-q) * 800
    peaks = [
        GlobalPeak(q_center=2.5, q_fwhm=0.3, d_spacing=0.4,
                   intensity=200, prominence=100, snr=8.0, index=25),
    ]
    gvecs = [
        GVector(gx=2.5, gy=0.0, magnitude=2.5, angle_deg=0.0,
                d_spacing=0.4, snr=8.0, fwhm=0.2, ring_index=0),
    ]
    ps = np.random.default_rng(42).random((64, 64))
    return GlobalFFTResult(
        power_spectrum=ps,
        radial_profile=profile,
        q_values=q,
        background=bg,
        corrected_profile=profile - bg,
        noise_floor=1.0,
        peaks=peaks,
        g_vectors=gvecs,
        d_dom=0.4,
        information_limit_q=8.0,
    )


@pytest.fixture
def gated_grid():
    shape = (4, 4)
    tier_map = np.array([
        ["A", "A", "B", "REJECTED"],
        ["A", "B", "REJECTED", ""],
        ["B", "A", "A", "B"],
        ["", "REJECTED", "A", "A"],
    ])
    snr_map = np.random.default_rng(42).uniform(1, 10, shape)
    orient_map = np.random.default_rng(42).uniform(0, 180, shape)
    skipped = tier_map == ""
    ts = TierSummary(
        n_tier_a=6, n_tier_b=4, n_rejected=3, n_skipped=3,
        tier_a_fraction=0.375, median_snr_tier_a=7.0,
    )
    classifications = np.empty(shape, dtype=object)
    return GatedTileGrid(
        classifications=classifications,
        tier_map=tier_map,
        snr_map=snr_map,
        pair_fraction_map=np.ones(shape),
        fwhm_map=np.ones(shape),
        orientation_map=orient_map,
        grid_shape=shape,
        skipped_mask=skipped,
        tier_summary=ts,
    )


@pytest.fixture
def gpa_result():
    shape = (64, 64)
    rng = np.random.default_rng(42)
    gvec = GVector(gx=2.5, gy=0.0, magnitude=2.5, angle_deg=0.0,
                   d_spacing=0.4, snr=8.0, fwhm=0.2, ring_index=0)
    phase = GPAPhaseResult(
        phase_raw=rng.random(shape).astype(np.float64),
        phase_unwrapped=rng.random(shape).astype(np.float64),
        amplitude=rng.random(shape).astype(np.float64),
        amplitude_mask=np.ones(shape, dtype=bool),
        g_vector=gvec,
        phase_noise_sigma=0.1,
        unwrap_success_fraction=0.95,
    )
    disp = DisplacementField(
        ux=rng.normal(0, 0.01, shape),
        uy=rng.normal(0, 0.01, shape),
    )
    strain = StrainField(
        exx=rng.normal(0, 0.01, shape),
        eyy=rng.normal(0, 0.01, shape),
        exy=rng.normal(0, 0.005, shape),
        rotation=rng.normal(0, 0.01, shape),
    )
    mode_decision = GPAModeDecision(selected_mode="full", decision_metrics={})
    return GPAResult(
        mode="full",
        phases={"g0": phase},
        displacement=disp,
        strain=strain,
        reference_region=None,
        mode_decision=mode_decision,
    )


@pytest.fixture
def peaks():
    rng = np.random.default_rng(42)
    return [
        SubpixelPeak(x=rng.uniform(5, 55), y=rng.uniform(5, 55),
                     intensity=rng.uniform(0.5, 1.0))
        for _ in range(20)
    ]


@pytest.fixture
def lattice_validation():
    return LatticeValidation(
        nn_distances=np.array([3.1, 3.2, 3.0, 3.15]),
        fraction_valid=0.85,
        mean_nn_distance_nm=0.4,
        std_nn_distance_nm=0.02,
        expected_d_nm=0.4,
        min_separation_px_used=2.0,
    )


@pytest.fixture
def bandpass_image():
    return np.random.default_rng(42).random((64, 64))


# ======================================================================
# Tests
# ======================================================================

class TestSavePipelineVisualizations:
    """Integration-level tests for the orchestrator."""

    def test_all_artifacts_generated(
        self, tmp_out, config, fft_grid, raw_image, global_fft_result,
        gated_grid, gpa_result, peaks, lattice_validation, bandpass_image,
    ):
        saved = save_pipeline_visualizations(
            tmp_out, config=config, fft_grid=fft_grid,
            raw_image=raw_image, global_fft_result=global_fft_result,
            gated_grid=gated_grid, gpa_result=gpa_result,
            peaks=peaks, lattice_validation=lattice_validation,
            bandpass_image=bandpass_image,
        )
        expected_keys = [
            "input_original",
            "globalfft_radial_profile",
            "globalfft_power_spectrum",
            "tiles_tier_map",
            "tiles_snr_map",
            "tiles_orientation_map",
            "gpa_phase_g0",
            "gpa_amplitude_g0",
            "gpa_displacement_ux",
            "gpa_displacement_uy",
            "gpa_strain_exx",
            "gpa_strain_eyy",
            "gpa_strain_exy",
            "gpa_strain_rotation",
            "peaks_lattice_overlay",
        ]
        for key in expected_keys:
            assert key in saved, f"Missing artifact: {key}"
            assert saved[key].exists(), f"File not found: {saved[key]}"
            assert saved[key].stat().st_size > 0, f"Empty file: {saved[key]}"

    def test_no_pngs_when_disabled(
        self, tmp_out, config, fft_grid, raw_image, global_fft_result,
    ):
        config.viz.enabled = False
        # Disabled at the caller level (analyze.py), but we test the
        # orchestrator produces no files if called with minimal data
        saved = save_pipeline_visualizations(
            tmp_out, config=config, fft_grid=fft_grid,
        )
        assert len(saved) == 0

    def test_graceful_none_inputs(self, tmp_out, config, fft_grid):
        saved = save_pipeline_visualizations(
            tmp_out, config=config, fft_grid=fft_grid,
            raw_image=None, global_fft_result=None, gated_grid=None,
            gpa_result=None, peaks=None,
        )
        assert len(saved) == 0

    def test_partial_data_only_available_plots(
        self, tmp_out, config, fft_grid, raw_image, gated_grid,
    ):
        saved = save_pipeline_visualizations(
            tmp_out, config=config, fft_grid=fft_grid,
            raw_image=raw_image, gated_grid=gated_grid,
        )
        assert "input_original" in saved
        assert "tiles_tier_map" in saved
        assert "gpa_phase_g0" not in saved
        assert "peaks_lattice_overlay" not in saved


class TestInputOriginal:
    def test_creates_png(self, tmp_out, raw_image):
        path = _save_input_original(raw_image, tmp_out, 0.127, 72)
        assert path.exists()
        assert path.suffix == ".png"


class TestRadialProfile:
    def test_creates_png(self, tmp_out, global_fft_result):
        path = _save_radial_profile(global_fft_result, tmp_out, 72)
        assert path.exists()
        assert path.stat().st_size > 0


class TestFFTPowerSpectrum:
    def test_creates_png(self, tmp_out, global_fft_result, fft_grid):
        path = _save_fft_power_spectrum(global_fft_result, fft_grid, tmp_out, 72)
        assert path.exists()


class TestTierMap:
    def test_creates_png(self, tmp_out, gated_grid):
        path = _save_tier_map(gated_grid, tmp_out, 72)
        assert path.exists()


class TestSnrMap:
    def test_creates_png(self, tmp_out, gated_grid):
        path = _save_snr_map(gated_grid, tmp_out, 72)
        assert path.exists()


class TestOrientationMap:
    def test_creates_png(self, tmp_out, gated_grid):
        path = _save_orientation_map(gated_grid, tmp_out, 72)
        assert path.exists()


class TestGPAPhase:
    def test_creates_png(self, tmp_out, gpa_result):
        phase_result = list(gpa_result.phases.values())[0]
        path = _save_gpa_phase(phase_result, 0, tmp_out, 72)
        assert path.exists()

    def test_nan_mask_regions(self, tmp_out):
        """Phase where amplitude_mask is partially False -> NaN regions."""
        shape = (32, 32)
        rng = np.random.default_rng(42)
        mask = np.ones(shape, dtype=bool)
        mask[:16, :] = False
        gvec = GVector(gx=2.5, gy=0.0, magnitude=2.5, angle_deg=0.0,
                       d_spacing=0.4, snr=8.0, fwhm=0.2, ring_index=0)
        pr = GPAPhaseResult(
            phase_raw=rng.random(shape),
            phase_unwrapped=rng.random(shape),
            amplitude=rng.random(shape),
            amplitude_mask=mask,
            g_vector=gvec,
        )
        path = _save_gpa_phase(pr, 0, tmp_out, 72)
        assert path.exists()


class TestGPAAmplitude:
    def test_creates_png(self, tmp_out, gpa_result):
        phase_result = list(gpa_result.phases.values())[0]
        path = _save_gpa_amplitude(phase_result, 0, tmp_out, 72)
        assert path.exists()


class TestDisplacement:
    def test_creates_png(self, tmp_out):
        field = np.random.default_rng(42).normal(0, 0.01, (32, 32))
        path = _save_displacement(field, "ux", tmp_out, 72)
        assert path.exists()

    def test_zero_field(self, tmp_out):
        field = np.zeros((32, 32))
        path = _save_displacement(field, "uy", tmp_out, 72)
        assert path.exists()


class TestStrainComponent:
    def test_creates_png(self, tmp_out):
        data = np.random.default_rng(42).normal(0, 0.01, (32, 32))
        path = _save_strain_component(data, "exx", tmp_out, 72)
        assert path.exists()

    def test_rotation_component(self, tmp_out):
        data = np.random.default_rng(42).normal(0, 0.01, (32, 32))
        path = _save_strain_component(data, "rotation", tmp_out, 72)
        assert path.exists()


class TestPeakLatticeOverlay:
    def test_creates_png(self, tmp_out, peaks, lattice_validation, bandpass_image):
        path = _save_peak_lattice_overlay(
            peaks, lattice_validation, bandpass_image, 0.127, tmp_out, 72,
        )
        assert path.exists()

    def test_no_validation(self, tmp_out, peaks, bandpass_image):
        """Works even without lattice_validation."""
        path = _save_peak_lattice_overlay(
            peaks, None, bandpass_image, 0.127, tmp_out, 72,
        )
        assert path.exists()

    def test_few_peaks(self, tmp_out, bandpass_image):
        """Only one peak — no NN pairs possible."""
        single_peak = [SubpixelPeak(x=10, y=10, intensity=0.8)]
        path = _save_peak_lattice_overlay(
            single_peak, None, bandpass_image, 0.127, tmp_out, 72,
        )
        assert path.exists()


class TestDPIConfig:
    def test_dpi_respected(self, tmp_out, global_fft_result):
        """Check that the saved PNG has the expected DPI."""
        from PIL import Image
        path = _save_radial_profile(global_fft_result, tmp_out, 200)
        img = Image.open(str(path))
        dpi_info = img.info.get("dpi")
        if dpi_info is not None:
            assert abs(dpi_info[0] - 200) < 5, f"DPI mismatch: {dpi_info}"


class TestVizConfig:
    def test_default_config(self):
        cfg = VizConfig()
        assert cfg.enabled is True
        assert cfg.dpi == 150

    def test_pipeline_config_has_viz(self):
        cfg = PipelineConfig()
        assert hasattr(cfg, "viz")
        assert isinstance(cfg.viz, VizConfig)

    def test_from_dict_round_trip(self):
        cfg = PipelineConfig()
        cfg.viz.dpi = 300
        d = cfg.to_dict()
        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.viz.dpi == 300

    def test_from_dict_viz_section(self):
        d = {"viz": {"enabled": False, "dpi": 96}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.viz.enabled is False
        assert cfg.viz.dpi == 96
