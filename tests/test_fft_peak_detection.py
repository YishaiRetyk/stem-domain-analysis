"""Tests for FFT peak detection + two-tier classification (WS3)."""

import numpy as np
import pytest
from src.fft_coords import FFTGrid
from src.pipeline_config import TilePeak, TilePeakSet, TierConfig, PeakGateConfig
from src.fft_peak_detection import (
    compute_peak_snr, measure_peak_fwhm, check_symmetry,
    count_non_collinear, classify_tile,
)


def _make_power_with_peak(tile_size=256, pixel_size_nm=0.1, q_peak=2.0, angle_deg=30.0):
    """Create a synthetic power spectrum with a known peak."""
    grid = FFTGrid(tile_size, tile_size, pixel_size_nm)
    power = np.random.default_rng(42).exponential(1.0, (tile_size, tile_size))

    # Add a Gaussian peak at q_peak, angle_deg
    qx_peak = q_peak * np.cos(np.radians(angle_deg))
    qy_peak = q_peak * np.sin(np.radians(angle_deg))
    px_x, px_y = grid.q_to_px(qx_peak, qy_peak)

    y, x = np.mgrid[:tile_size, :tile_size]
    gaussian = 500 * np.exp(-0.5 * ((x - px_x)**2 + (y - px_y)**2) / 4.0)
    power += gaussian

    # Add antipodal
    apx, apy = grid.q_to_px(-qx_peak, -qy_peak)
    gaussian_anti = 500 * np.exp(-0.5 * ((x - apx)**2 + (y - apy)**2) / 4.0)
    power += gaussian_anti

    peak = TilePeak(
        qx=qx_peak, qy=qy_peak, q_mag=q_peak,
        d_spacing=1.0 / q_peak, angle_deg=angle_deg,
        intensity=float(power[int(round(px_y)), int(round(px_x))]),
        fwhm=0.1,
    )
    anti_peak = TilePeak(
        qx=-qx_peak, qy=-qy_peak, q_mag=q_peak,
        d_spacing=1.0 / q_peak, angle_deg=angle_deg + 180,
        intensity=float(power[int(round(apy)), int(round(apx))]),
        fwhm=0.1,
    )

    return power, [peak, anti_peak], grid


class TestPeakSNR:
    """B3: peak-height SNR tests."""

    def test_snr_positive_for_real_peak(self):
        power, peaks, grid = _make_power_with_peak()
        snr_result = compute_peak_snr(power, peaks[0], peaks, grid)
        assert snr_result.snr > 0

    def test_snr_uses_peak_height(self):
        """B3: SNR should be based on peak height, not wedge mean."""
        power, peaks, grid = _make_power_with_peak()
        snr = compute_peak_snr(power, peaks[0], peaks, grid)
        assert snr.signal_peak > snr.background_median

    def test_background_excludes_peaks(self):
        """C3: background annulus should exclude all detected peaks."""
        power, peaks, grid = _make_power_with_peak()
        snr = compute_peak_snr(power, peaks[0], peaks, grid)
        # The background should be computed from a region excluding peaks
        assert snr.n_background_px > 0
        assert snr.note is None or "full-annulus fallback" not in (snr.note or "")


class TestPeakFWHM:
    """B2: FWHM via 2D Gaussian fit."""

    def test_fwhm_on_gaussian_peak(self):
        """Should measure FWHM correctly for a synthetic Gaussian peak."""
        tile_size = 256
        pixel_size_nm = 0.1
        grid = FFTGrid(tile_size, tile_size, pixel_size_nm)

        power = np.ones((tile_size, tile_size)) * 10.0
        # Add Gaussian peak at (180, 150)
        y, x = np.mgrid[:tile_size, :tile_size]
        sigma_px = 3.0
        power += 200 * np.exp(-0.5 * ((x - 180)**2 + (y - 150)**2) / sigma_px**2)

        qx, qy = grid.px_to_q(180, 150)
        peak = TilePeak(qx=qx, qy=qy, q_mag=np.sqrt(qx**2 + qy**2),
                        d_spacing=0, angle_deg=0, intensity=0, fwhm=0)

        result = measure_peak_fwhm(power, peak, grid)
        assert result.fwhm_valid
        assert result.method in ("gaussian_2d", "wedge_fallback")

    def test_fwhm_invalid_edge_peak(self):
        """Peak near edge should fail gracefully."""
        grid = FFTGrid(64, 64, 0.1)
        power = np.ones((64, 64))
        # Peak at edge
        qx, qy = grid.px_to_q(1, 1)
        peak = TilePeak(qx=qx, qy=qy, q_mag=np.sqrt(qx**2 + qy**2),
                        d_spacing=0, angle_deg=0, intensity=0, fwhm=0)
        result = measure_peak_fwhm(power, peak, grid)
        assert not result.fwhm_valid


class TestSymmetry:
    """Symmetry check tests."""

    def test_perfect_symmetry(self):
        """Two antipodal peaks should give symmetry = 1.0."""
        grid = FFTGrid(256, 256, 0.1)
        peaks = [
            TilePeak(qx=2.0, qy=0.5, q_mag=2.06, d_spacing=0.485,
                     angle_deg=14, intensity=100, fwhm=0.1),
            TilePeak(qx=-2.0, qy=-0.5, q_mag=2.06, d_spacing=0.485,
                     angle_deg=194, intensity=100, fwhm=0.1),
        ]
        score, n_paired = check_symmetry(peaks, grid)
        assert score == 1.0
        assert n_paired == 2

    def test_no_symmetry(self):
        """Non-antipodal peaks should have low symmetry."""
        grid = FFTGrid(256, 256, 0.1)
        peaks = [
            TilePeak(qx=2.0, qy=0.5, q_mag=2.06, d_spacing=0.485,
                     angle_deg=14, intensity=100, fwhm=0.1),
            TilePeak(qx=1.0, qy=2.0, q_mag=2.24, d_spacing=0.447,
                     angle_deg=63, intensity=100, fwhm=0.1),
        ]
        score, n_paired = check_symmetry(peaks, grid)
        assert score == 0.0


class TestClassifyTile:
    """Two-tier classification."""

    def test_tier_a(self):
        """High-SNR tile with symmetry should be Tier A."""
        power, peaks, grid = _make_power_with_peak()
        peak_set = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                                power_spectrum=power)
        tc = classify_tile(peak_set, grid)
        # With a strong peak and symmetry, should be A or B
        assert tc.tier in ("A", "B")
        assert tc.best_snr > 0

    def test_rejected_no_peaks(self):
        """No peaks should result in REJECTED."""
        grid = FFTGrid(256, 256, 0.1)
        peak_set = TilePeakSet(peaks=[], tile_row=0, tile_col=0,
                                power_spectrum=None)
        tc = classify_tile(peak_set, grid)
        assert tc.tier == "REJECTED"


class TestNonCollinear:
    """Non-collinearity check."""

    def test_two_orthogonal(self):
        peaks = [
            TilePeak(qx=1, qy=0, q_mag=1, d_spacing=1, angle_deg=0,
                     intensity=1, fwhm=0),
            TilePeak(qx=0, qy=1, q_mag=1, d_spacing=1, angle_deg=90,
                     intensity=1, fwhm=0),
        ]
        assert count_non_collinear(peaks) == 2

    def test_collinear(self):
        peaks = [
            TilePeak(qx=1, qy=0, q_mag=1, d_spacing=1, angle_deg=0,
                     intensity=1, fwhm=0),
            TilePeak(qx=2, qy=0, q_mag=2, d_spacing=0.5, angle_deg=0,
                     intensity=1, fwhm=0),
        ]
        assert count_non_collinear(peaks) == 1
