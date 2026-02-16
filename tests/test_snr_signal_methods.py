"""Tests for SNR signal extraction methods and lightweight zscore mode."""

import numpy as np
import pytest

from src.fft_coords import FFTGrid
from src.pipeline_config import TilePeak, TilePeakSet, PeakSNRConfig, TileFFTConfig
from src.fft_peak_detection import (
    _compute_signal_from_disk, compute_peak_snr,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_grid_and_power(tile_size=64, pixel_size_nm=0.1):
    """Create an FFTGrid and a power spectrum with a single Gaussian peak."""
    grid = FFTGrid(tile_size, tile_size, pixel_size_nm)
    power = np.random.RandomState(42).exponential(1.0, (tile_size, tile_size))
    # Place a broad Gaussian peak at a known location
    cy, cx = tile_size // 2, tile_size // 2
    peak_px_x = cx + 10
    peak_px_y = cy + 5
    yy, xx = np.mgrid[:tile_size, :tile_size]
    sigma = 2.5
    gaussian = 100 * np.exp(-((xx - peak_px_x)**2 + (yy - peak_px_y)**2) / (2 * sigma**2))
    power += gaussian
    qx, qy = grid.px_to_q(float(peak_px_x), float(peak_px_y))
    q_mag = np.sqrt(qx**2 + qy**2)
    peak = TilePeak(qx=qx, qy=qy, q_mag=q_mag,
                    d_spacing=1.0/q_mag if q_mag > 0 else 0,
                    angle_deg=float(np.degrees(np.arctan2(qy, qx))) % 180,
                    intensity=float(power[peak_px_y, peak_px_x]),
                    fwhm=sigma * 2.355 * grid.qx_scale,
                    ring_index=0)
    return grid, power, peak


# ======================================================================
# _compute_signal_from_disk tests
# ======================================================================

class TestComputeSignalFromDisk:
    def test_signal_max_is_default(self):
        power = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=float)
        disk = np.ones((3, 3), dtype=bool)
        assert _compute_signal_from_disk(power, disk, "max") == 9.0

    def test_signal_integrated_sum_broad_peak(self):
        power = np.array([[1, 2, 1],
                          [2, 5, 2],
                          [1, 2, 1]], dtype=float)
        disk = np.ones((3, 3), dtype=bool)
        result = _compute_signal_from_disk(power, disk, "integrated_sum")
        assert result == pytest.approx(17.0)

    def test_signal_integrated_median_robust_to_hot_pixel(self):
        power = np.array([[1, 1, 1],
                          [1, 1, 1000],  # hot pixel
                          [1, 1, 1]], dtype=float)
        disk = np.ones((3, 3), dtype=bool)
        assert _compute_signal_from_disk(power, disk, "integrated_median") == 1.0
        # max would pick 1000
        assert _compute_signal_from_disk(power, disk, "max") == 1000.0

    def test_signal_method_invalid_raises(self):
        power = np.ones((3, 3))
        disk = np.ones((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="Unknown signal_method"):
            _compute_signal_from_disk(power, disk, "invalid_method")

    def test_empty_disk_returns_zero(self):
        power = np.ones((3, 3))
        disk = np.zeros((3, 3), dtype=bool)
        assert _compute_signal_from_disk(power, disk, "max") == 0.0


# ======================================================================
# compute_peak_snr with signal_method tests
# ======================================================================

class TestComputePeakSNRSignalMethod:
    def test_snr_backward_compat_max(self):
        """Default signal_method='max' produces identical results to old code."""
        grid, power, peak = _make_grid_and_power()
        config_max = PeakSNRConfig(signal_method="max")
        config_default = PeakSNRConfig()  # should also be "max"
        result_max = compute_peak_snr(power, peak, [peak], grid,
                                       peak_snr_config=config_max)
        result_default = compute_peak_snr(power, peak, [peak], grid,
                                           peak_snr_config=config_default)
        assert result_max.snr == pytest.approx(result_default.snr)
        assert result_max.signal_peak == pytest.approx(result_default.signal_peak)

    def test_snr_scaling_integrated_sum(self):
        """Integrated sum with proper scaling produces reasonable SNR."""
        grid, power, peak = _make_grid_and_power()
        config = PeakSNRConfig(signal_method="integrated_sum")
        result = compute_peak_snr(power, peak, [peak], grid,
                                   peak_snr_config=config)
        # Signal_peak should be a sum (larger than max)
        config_max = PeakSNRConfig(signal_method="max")
        result_max = compute_peak_snr(power, peak, [peak], grid,
                                       peak_snr_config=config_max)
        assert result.signal_peak > result_max.signal_peak
        # SNR should still be positive and finite
        assert result.snr > 0
        assert np.isfinite(result.snr)

    def test_integrated_median_produces_valid_snr(self):
        """Integrated median produces valid SNR."""
        grid, power, peak = _make_grid_and_power()
        config = PeakSNRConfig(signal_method="integrated_median")
        result = compute_peak_snr(power, peak, [peak], grid,
                                   peak_snr_config=config)
        assert result.snr > 0
        assert np.isfinite(result.snr)


# ======================================================================
# Lightweight SNR zscore mode tests
# ======================================================================

class TestLightweightSNRZscore:
    def _make_tile_power(self, tile_size=64, pixel_size_nm=0.1):
        grid = FFTGrid(tile_size, tile_size, pixel_size_nm)
        rng = np.random.RandomState(42)
        power = rng.exponential(1.0, (tile_size, tile_size))
        # Add a peak
        cy, cx = tile_size // 2, tile_size // 2
        peak_y, peak_x = cy + 8, cx + 6
        power[peak_y, peak_x] = 50.0
        q_mag = grid.q_mag_grid()
        dc_mask = q_mag < 0.25 * grid.qx_scale * tile_size * 0.02
        return power, q_mag, dc_mask, peak_y, peak_x, grid

    def test_lightweight_snr_zscore_mode(self):
        from src.tile_fft import _lightweight_peak_snr
        power, q_mag, dc_mask, py, px, grid = self._make_tile_power()
        peak_q = q_mag[py, px]

        snr_ratio = _lightweight_peak_snr(power, q_mag, peak_q, py, px, dc_mask,
                                           lightweight_snr_method="ratio")
        snr_zscore = _lightweight_peak_snr(power, q_mag, peak_q, py, px, dc_mask,
                                            lightweight_snr_method="zscore")
        # Both should be positive but different values
        assert snr_ratio > 0
        assert snr_zscore > 0
        assert snr_ratio != pytest.approx(snr_zscore, rel=0.01)

    def test_lightweight_snr_ratio_default(self):
        from src.tile_fft import _lightweight_peak_snr
        power, q_mag, dc_mask, py, px, grid = self._make_tile_power()
        peak_q = q_mag[py, px]

        # Default should be ratio
        snr_default = _lightweight_peak_snr(power, q_mag, peak_q, py, px, dc_mask)
        snr_ratio = _lightweight_peak_snr(power, q_mag, peak_q, py, px, dc_mask,
                                           lightweight_snr_method="ratio")
        assert snr_default == pytest.approx(snr_ratio)
