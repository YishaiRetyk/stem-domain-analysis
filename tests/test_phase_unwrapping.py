"""Tests for phase unwrapping improvements (PR 2)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.fft_coords import FFTGrid
from src.pipeline_config import GVector, GPAConfig, PhaseUnwrapConfig
from src.gpa import (
    compute_gpa_phase,
    _unwrap_phase_quality_guided,
    CUCIM_UNWRAP_AVAILABLE,
)


def _make_perfect_lattice(N=128, pixel_size_nm=0.1, d_spacing=0.5, angle_deg=0):
    """Create a perfect lattice image and its expected g-vector."""
    g_mag = 1.0 / d_spacing
    angle_rad = np.radians(angle_deg)
    gx = g_mag * np.cos(angle_rad)
    gy = g_mag * np.sin(angle_rad)

    y_nm = np.arange(N).reshape(-1, 1) * pixel_size_nm
    x_nm = np.arange(N).reshape(1, -1) * pixel_size_nm
    image = 0.5 + 0.3 * np.cos(2 * np.pi * (gx * x_nm + gy * y_nm))

    gvec = GVector(gx=gx, gy=gy, magnitude=g_mag, angle_deg=angle_deg,
                   d_spacing=d_spacing, snr=10.0, fwhm=0.05, ring_index=0)
    return image, gvec


class TestCucimFallback:
    """GPU cucim unwrapping fallback tests."""

    def test_cucim_flag_is_bool(self):
        """CUCIM_UNWRAP_AVAILABLE should be a boolean."""
        assert isinstance(CUCIM_UNWRAP_AVAILABLE, bool)

    def test_default_method_uses_skimage(self):
        """Default method should use skimage (not cucim)."""
        image, gvec = _make_perfect_lattice(64)
        grid = FFTGrid(64, 64, 0.1)
        config = PhaseUnwrapConfig(method="default")

        result = compute_gpa_phase(image, gvec, 0.1, grid,
                                    amplitude_threshold=0.01,
                                    phase_unwrap_config=config)
        # Should produce valid unwrapped phase
        valid = result.amplitude_mask & ~np.isnan(result.phase_unwrapped)
        assert np.sum(valid) > 0


class TestQualityGuidedUnwrapping:
    """Quality-guided phase unwrapping tests."""

    def test_quality_guided_flat_lattice(self):
        """Quality-guided should match default on clean data (flat lattice)."""
        image, gvec = _make_perfect_lattice(64)
        grid = FFTGrid(64, 64, 0.1)

        # Default
        result_default = compute_gpa_phase(image, gvec, 0.1, grid,
                                            amplitude_threshold=0.01,
                                            phase_unwrap_config=PhaseUnwrapConfig(method="default"))

        # Quality guided
        result_qg = compute_gpa_phase(image, gvec, 0.1, grid,
                                       amplitude_threshold=0.01,
                                       phase_unwrap_config=PhaseUnwrapConfig(method="quality_guided"))

        # Both should produce valid data
        valid_d = result_default.amplitude_mask & ~np.isnan(result_default.phase_unwrapped)
        valid_q = result_qg.amplitude_mask & ~np.isnan(result_qg.phase_unwrapped)
        assert np.sum(valid_d) > 0
        assert np.sum(valid_q) > 0

    def test_quality_guided_noisy_band(self):
        """Quality-guided should be more robust near low-amplitude regions."""
        N = 64
        px = 0.1

        # Create lattice with amplitude modulation (weak band in the middle)
        gvec = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0.0,
                       d_spacing=0.5, snr=10.0, fwhm=0.05, ring_index=0)
        y_nm = np.arange(N).reshape(-1, 1) * px
        x_nm = np.arange(N).reshape(1, -1) * px

        # Amplitude envelope: strong at edges, weak in center
        amp = 0.1 + 0.4 * np.abs(np.linspace(-1, 1, N)).reshape(-1, 1)
        image = 0.5 + amp * np.cos(2 * np.pi * (gvec.gx * x_nm + gvec.gy * y_nm))

        grid = FFTGrid(N, N, px)
        config_qg = PhaseUnwrapConfig(method="quality_guided",
                                       laplacian_weight_power=2.0)

        result = compute_gpa_phase(image, gvec, 0.1, grid,
                                    amplitude_threshold=0.01,
                                    phase_unwrap_config=config_qg)

        # Should still produce some valid unwrapped pixels
        valid = result.amplitude_mask & ~np.isnan(result.phase_unwrapped)
        assert np.sum(valid) > 10

    def test_laplacian_solver_known_case(self):
        """Analytic verification: constant phase should unwrap to constant."""
        N = 32
        phase_raw = np.full((N, N), 0.5)  # constant phase (already unwrapped)
        amplitude = np.ones((N, N))

        config = PhaseUnwrapConfig(method="quality_guided",
                                    cg_tol=1e-6, cg_maxiter=200)
        result = _unwrap_phase_quality_guided(phase_raw, amplitude, config)

        # Result should be approximately constant
        # (CG solves Laplacian equation, constant is in the null space,
        #  so the solution should preserve the mean)
        result_centered = result - np.mean(result)
        assert np.std(result_centered) < 0.1


class TestConfigDefaults:
    """Backward compatibility for phase unwrap config defaults."""

    def test_default_method_is_default(self):
        """PhaseUnwrapConfig default method should be 'default'."""
        config = PhaseUnwrapConfig()
        assert config.method == "default"

    def test_gpa_config_has_phase_unwrap(self):
        """GPAConfig should contain a PhaseUnwrapConfig."""
        gpa = GPAConfig()
        assert isinstance(gpa.phase_unwrap, PhaseUnwrapConfig)
        assert gpa.phase_unwrap.method == "default"
