"""Tests for Tukey window support."""

import numpy as np
import pytest

from src.fft_features import (
    create_2d_hann_window,
    create_2d_tukey_window,
    create_window,
)


class TestTukeyWindow:
    """Tukey window construction tests."""

    def test_tukey_shape(self):
        """Correct dimensions."""
        w = create_2d_tukey_window(128)
        assert w.shape == (128, 128)

    def test_tukey_alpha_0_rectangular(self):
        """alpha=0 gives a nearly rectangular window (all ~1 in interior)."""
        w = create_2d_tukey_window(128, alpha=0.0)
        # Interior should be 1.0 (the outer product of rectangular windows)
        assert np.allclose(w[10:-10, 10:-10], 1.0)

    def test_tukey_alpha_1_matches_hann(self):
        """alpha=1 should closely approximate a Hann window."""
        w_tukey = create_2d_tukey_window(128, alpha=1.0)
        w_hann = create_2d_hann_window(128)
        np.testing.assert_allclose(w_tukey, w_hann, atol=1e-10)

    def test_tukey_center_unattenuated(self):
        """With alpha=0.2, the central 80% should be 1.0."""
        size = 256
        w = create_2d_tukey_window(size, alpha=0.2)
        margin = int(size * 0.1) + 2  # 10% taper zone + safety
        center = w[margin:-margin, margin:-margin]
        assert np.all(center >= 0.999), "Center region should be unattenuated"

    def test_tukey_symmetric(self):
        """Window should be symmetric in both axes."""
        w = create_2d_tukey_window(128, alpha=0.3)
        np.testing.assert_allclose(w, w[::-1, :], atol=1e-10)
        np.testing.assert_allclose(w, w[:, ::-1], atol=1e-10)


class TestCreateWindowDispatch:
    """create_window dispatch tests."""

    def test_dispatch_hann(self):
        """'hann' dispatches to Hann window."""
        w = create_window(64, window_type="hann")
        w_ref = create_2d_hann_window(64)
        np.testing.assert_allclose(w, w_ref)

    def test_dispatch_tukey(self):
        """'tukey' dispatches to Tukey window."""
        w = create_window(64, window_type="tukey", tukey_alpha=0.3)
        w_ref = create_2d_tukey_window(64, alpha=0.3)
        np.testing.assert_allclose(w, w_ref)


class TestTukeyInPipeline:
    """Integration: Tukey window through tile FFT pipeline."""

    def test_tukey_in_tile_pipeline(self):
        """Peaks should still be detected with Tukey window."""
        from src.fft_coords import FFTGrid
        from src.tile_fft import compute_tile_fft, extract_tile_peaks
        from src.pipeline_config import TileFFTConfig

        # Create a tile with known sinusoidal pattern
        size = 128
        px = 0.1
        d_spacing = 0.8  # nm
        g = 1.0 / d_spacing
        x = np.arange(size).reshape(1, -1) * px
        y = np.arange(size).reshape(-1, 1) * px
        tile = 0.5 + 0.4 * np.cos(2 * np.pi * g * x)

        grid = FFTGrid(size, size, px)

        # Hann window
        config_hann = TileFFTConfig(window_type="hann")
        from src.fft_features import create_window
        w_hann = create_window(size, "hann")
        power_hann = compute_tile_fft(tile, w_hann)
        peaks_hann = extract_tile_peaks(power_hann, grid, tile_fft_config=config_hann)

        # Tukey window
        config_tukey = TileFFTConfig(window_type="tukey", tukey_alpha=0.2)
        w_tukey = create_window(size, "tukey", 0.2)
        power_tukey = compute_tile_fft(tile, w_tukey)
        peaks_tukey = extract_tile_peaks(power_tukey, grid, tile_fft_config=config_tukey)

        # Both should detect peaks
        assert len(peaks_hann) > 0, "Hann should detect peaks"
        assert len(peaks_tukey) > 0, "Tukey should detect peaks"
