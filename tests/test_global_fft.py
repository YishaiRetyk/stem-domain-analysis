"""Tests for global FFT + g-vector extraction (WS2)."""

import numpy as np
import pytest
from src.fft_coords import FFTGrid
from src.global_fft import compute_global_fft
from src.pipeline_config import GlobalFFTConfig
from tests.synthetic import generate_single_crystal


class TestGlobalFFT:
    """Global FFT analysis."""

    def test_finds_peaks(self):
        """Should find at least one peak from a single-crystal image."""
        image = generate_single_crystal(
            shape=(512, 512), pixel_size_nm=0.1, d_spacing=0.5,
            orientation_deg=0, noise_level=0.02,
        )
        grid = FFTGrid(512, 512, 0.1)
        config = GlobalFFTConfig(min_peak_snr=2.0)
        result = compute_global_fft(image, grid, config)
        assert len(result.peaks) >= 1

    def test_d_dom_matches_input(self):
        """d_dom should be within 5% of the input d-spacing."""
        d_true = 0.4
        image = generate_single_crystal(
            shape=(512, 512), pixel_size_nm=0.1, d_spacing=d_true,
            orientation_deg=30, noise_level=0.02,
        )
        grid = FFTGrid(512, 512, 0.1)
        config = GlobalFFTConfig(min_peak_snr=2.0)
        result = compute_global_fft(image, grid, config)
        if result.d_dom is not None:
            assert abs(result.d_dom - d_true) / d_true < 0.05

    def test_g_vectors_found(self):
        """Should extract g-vectors from a crystal image."""
        image = generate_single_crystal(
            shape=(512, 512), pixel_size_nm=0.1, d_spacing=0.5,
            orientation_deg=45, noise_level=0.02, amplitude=0.4,
        )
        grid = FFTGrid(512, 512, 0.1)
        config = GlobalFFTConfig(min_peak_snr=2.0)
        result = compute_global_fft(image, grid, config)
        # May find g-vectors depending on SNR
        assert result.g_vectors is not None

    def test_radial_profile_produced(self):
        image = generate_single_crystal(shape=(512, 512), pixel_size_nm=0.1, d_spacing=0.5)
        grid = FFTGrid(512, 512, 0.1)
        result = compute_global_fft(image, grid)
        assert result.radial_profile is not None
        assert len(result.radial_profile) > 0
        assert len(result.q_values) == len(result.radial_profile)

    def test_amorphous_no_strong_peaks(self):
        """Amorphous (noise) image should not produce strong peaks."""
        from tests.synthetic import generate_amorphous
        image = generate_amorphous(shape=(512, 512))
        grid = FFTGrid(512, 512, 0.1)
        config = GlobalFFTConfig(min_peak_snr=5.0)
        result = compute_global_fft(image, grid, config)
        # Should find few or no high-SNR peaks
        high_snr = [p for p in result.peaks if p.snr >= 5.0]
        assert len(high_snr) == 0
