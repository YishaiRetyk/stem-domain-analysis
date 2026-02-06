"""Tests for tile FFT computation (WS3 - tile_fft.py)."""

import numpy as np
import pytest
from src.fft_coords import FFTGrid
from src.tile_fft import compute_tile_fft, extract_tile_peaks, check_tiling_adequacy
from tests.synthetic import generate_single_crystal


class TestComputeTileFft:
    """Tile-level FFT computation."""

    def test_power_spectrum_shape(self):
        tile = np.random.default_rng(42).normal(0, 1, (256, 256))
        window = np.outer(np.hanning(256), np.hanning(256))
        power = compute_tile_fft(tile, window)
        assert power.shape == (256, 256)

    def test_power_non_negative(self):
        tile = np.random.default_rng(42).normal(0, 1, (256, 256))
        window = np.outer(np.hanning(256), np.hanning(256))
        power = compute_tile_fft(tile, window)
        assert np.all(power >= 0)


class TestExtractTilePeaks:
    """Subpixel peak extraction from tile power spectrum."""

    def test_finds_peaks_in_crystal_tile(self):
        image = generate_single_crystal(
            shape=(256, 256), pixel_size_nm=0.1, d_spacing=0.5,
            orientation_deg=30, noise_level=0.02, amplitude=0.5,
        )
        window = np.outer(np.hanning(256), np.hanning(256))
        power = compute_tile_fft(image, window)
        grid = FFTGrid(256, 256, 0.1)

        q_ranges = [(1.5, 2.5)]  # 1/0.5 = 2.0 cycles/nm
        peaks = extract_tile_peaks(power, grid, q_ranges=q_ranges)
        assert len(peaks) > 0

    def test_no_peaks_in_noise(self):
        """Pure noise should find few or no peaks."""
        rng = np.random.default_rng(42)
        tile = rng.normal(0, 1, (256, 256))
        window = np.outer(np.hanning(256), np.hanning(256))
        power = compute_tile_fft(tile, window)
        grid = FFTGrid(256, 256, 0.1)

        peaks = extract_tile_peaks(power, grid, q_ranges=[(1.0, 3.0)],
                                    peak_threshold_frac=0.9)
        # With very high threshold, should find few peaks in noise
        assert len(peaks) < 20


class TestTilingAdequacy:
    """G5: tiling adequacy check."""

    def test_adequate(self):
        periods, passed = check_tiling_adequacy(tile_size=256, d_dom_nm=0.4,
                                                 pixel_size_nm=0.1)
        # 0.4 nm / 0.1 nm/px = 4 px per d-spacing
        # 256 / 4 = 64 periods -> adequate
        assert passed
        assert periods == pytest.approx(64.0)

    def test_inadequate(self):
        periods, passed = check_tiling_adequacy(tile_size=32, d_dom_nm=0.4,
                                                 pixel_size_nm=0.1)
        # 32 / 4 = 8 periods -> inadequate (< 20)
        assert not passed
        assert periods == pytest.approx(8.0)
