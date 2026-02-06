"""Tests for FFTGrid canonical coordinate system (B1, F1)."""

import numpy as np
import pytest
from src.fft_coords import FFTGrid


class TestFFTGridBasic:
    """Basic construction and property tests."""

    def test_construction(self):
        grid = FFTGrid(256, 256, 0.127)
        assert grid.height == 256
        assert grid.width == 256
        assert grid.pixel_size_nm == 0.127
        assert grid.dc_x == 128
        assert grid.dc_y == 128
        assert grid.frequency_unit == "cycles/nm"

    def test_qx_scale(self):
        grid = FFTGrid(256, 256, 0.127)
        expected = 1.0 / (256 * 0.127)
        assert abs(grid.qx_scale - expected) < 1e-10

    def test_rectangular_separate_scales(self):
        """B1: rectangular images have separate qx_scale and qy_scale."""
        grid = FFTGrid(256, 512, 0.1)
        assert grid.qx_scale != grid.qy_scale
        assert abs(grid.qx_scale - 1.0 / (512 * 0.1)) < 1e-10
        assert abs(grid.qy_scale - 1.0 / (256 * 0.1)) < 1e-10

    def test_square_equal_scales(self):
        grid = FFTGrid(256, 256, 0.1)
        assert grid.qx_scale == grid.qy_scale

    def test_invalid_pixel_size(self):
        with pytest.raises(ValueError):
            FFTGrid(256, 256, 0)
        with pytest.raises(ValueError):
            FFTGrid(256, 256, -0.1)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            FFTGrid(0, 256, 0.1)


class TestCoordinateConversions:
    """Round-trip and correctness tests for px_to_q and q_to_px."""

    def test_dc_center_is_zero(self):
        grid = FFTGrid(256, 256, 0.127)
        qx, qy = grid.px_to_q(128, 128)
        assert abs(qx) < 1e-10
        assert abs(qy) < 1e-10

    def test_round_trip_square(self):
        """px_to_q -> q_to_px should be identity."""
        grid = FFTGrid(256, 256, 0.127)
        for px_x, px_y in [(0, 0), (128, 128), (200, 50), (255, 255)]:
            qx, qy = grid.px_to_q(px_x, px_y)
            rx, ry = grid.q_to_px(qx, qy)
            assert abs(rx - px_x) < 1e-10
            assert abs(ry - px_y) < 1e-10

    def test_round_trip_rectangular(self):
        grid = FFTGrid(256, 512, 0.1)
        for px_x, px_y in [(0, 0), (256, 128), (400, 200)]:
            qx, qy = grid.px_to_q(px_x, px_y)
            rx, ry = grid.q_to_px(qx, qy)
            assert abs(rx - px_x) < 1e-10
            assert abs(ry - px_y) < 1e-10

    def test_q_mag_at_dc(self):
        grid = FFTGrid(256, 256, 0.127)
        assert grid.q_mag(128, 128) == 0.0


class TestKnownSinusoid:
    """F1: verify g-vector extraction from known sinusoid."""

    def test_sinusoid_angle(self):
        """Known sinusoid at 30 deg should give g-vector at 30 deg."""
        N = 256
        px = 0.1
        d = 0.5  # nm
        angle_deg = 30.0
        angle_rad = np.radians(angle_deg)
        g_mag = 1.0 / d

        # Generate sinusoid
        y_nm = np.arange(N).reshape(-1, 1) * px
        x_nm = np.arange(N).reshape(1, -1) * px
        gx = g_mag * np.cos(angle_rad)
        gy = g_mag * np.sin(angle_rad)
        image = np.cos(2 * np.pi * (gx * x_nm + gy * y_nm))

        # FFT
        FT = np.fft.fftshift(np.fft.fft2(image))
        power = np.abs(FT) ** 2

        grid = FFTGrid(N, N, px)

        # Find peak (exclude DC)
        q_grid = grid.q_mag_grid()
        power_masked = power.copy()
        power_masked[q_grid < 0.5] = 0  # exclude DC region

        peak_idx = np.unravel_index(np.argmax(power_masked), power.shape)
        py, px_pos = peak_idx

        qx, qy = grid.px_to_q(px_pos, py)
        measured_angle = np.degrees(np.arctan2(qy, qx))
        measured_q = np.sqrt(qx**2 + qy**2)

        # Check angle within 2 degrees (may find +g or -g = antipodal)
        angle_diff = abs(((measured_angle - angle_deg) + 180) % 360 - 180)
        assert angle_diff < 2.0 or abs(angle_diff - 180) < 2.0, \
            f"Expected {angle_deg} (or antipodal), got {measured_angle}"
        # Check q magnitude: d = 1/|g|
        assert abs(measured_q - g_mag) < grid.qx_scale, \
            f"Expected q={g_mag}, got {measured_q}"

    def test_d_spacing_no_2pi(self):
        """F1: d = 1/|g| with no 2pi factor."""
        assert FFTGrid.d_spacing(2.0) == 0.5
        assert FFTGrid.d_spacing(1.0) == 1.0
        assert FFTGrid.d_spacing(10.0) == 0.1


class TestGridHelpers:
    """Test grid-wide helper methods."""

    def test_q_mag_grid_shape(self):
        grid = FFTGrid(256, 512, 0.1)
        q = grid.q_mag_grid()
        assert q.shape == (256, 512)

    def test_q_mag_grid_dc_is_zero(self):
        grid = FFTGrid(256, 256, 0.1)
        q = grid.q_mag_grid()
        assert q[128, 128] == 0.0

    def test_angle_grid_shape(self):
        grid = FFTGrid(256, 256, 0.1)
        angles = grid.angle_grid_deg()
        assert angles.shape == (256, 256)

    def test_to_dict(self):
        grid = FFTGrid(256, 256, 0.127)
        d = grid.to_dict()
        assert d["frequency_unit"] == "cycles/nm"
        assert d["d_spacing_formula"] == "d = 1/|g| (no 2pi)"
        assert d["pixel_size_nm"] == 0.127
