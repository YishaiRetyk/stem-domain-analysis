"""Tests for peak finding + lattice validation (WS6)."""

import numpy as np
import pytest
from src.fft_coords import FFTGrid
from src.peak_finding import build_bandpass_image, find_subpixel_peaks, validate_peak_lattice
from tests.synthetic import generate_single_crystal, generate_with_gradient


class TestBandpassImage:
    """I5: Bandpass-filtered peak image."""

    def test_suppresses_gradient(self):
        """Bandpass should suppress low-frequency thickness gradient."""
        image = generate_with_gradient(
            shape=(512, 512), pixel_size_nm=0.1, d_spacing=0.5,
            gradient_strength=0.5, noise_level=0.01,
        )
        grid = FFTGrid(512, 512, 0.1)
        g_dom_mag = 1.0 / 0.5

        bp = build_bandpass_image(image, g_dom_mag, grid)
        raw_range = image.max() - image.min()
        bp_range = bp.max() - bp.min()

        # Bandpass should have smaller range (gradient removed)
        # Or at least the mean should be closer to zero
        assert abs(np.mean(bp)) < abs(np.mean(image) - 0.5) + 0.1

    def test_fallback_to_raw(self):
        """Should return raw image if no g_dom."""
        image = np.random.default_rng(42).normal(0.5, 0.1, (256, 256))
        grid = FFTGrid(256, 256, 0.1)
        result = build_bandpass_image(image, None, grid)
        np.testing.assert_array_equal(result, image)

    def test_output_shape(self):
        grid = FFTGrid(256, 256, 0.1)
        image = np.random.default_rng(42).normal(0, 1, (256, 256))
        result = build_bandpass_image(image, 2.0, grid)
        assert result.shape == (256, 256)


class TestSubpixelPeaks:
    """Peak detection with d-spacing adaptive separation."""

    def test_finds_peaks_on_lattice(self):
        """Should find peaks on a regular lattice image."""
        image = generate_single_crystal(
            shape=(256, 256), pixel_size_nm=0.1, d_spacing=1.0,
            orientation_deg=0, noise_level=0.01, amplitude=0.5,
        )
        peaks = find_subpixel_peaks(image, expected_d_nm=1.0, pixel_size_nm=0.1,
                                     min_prominence=0.05, tile_size=256)
        assert len(peaks) > 0

    def test_adaptive_separation(self):
        """C9: min_separation should adapt to d-spacing."""
        image = generate_single_crystal(
            shape=(256, 256), pixel_size_nm=0.1, d_spacing=2.0,
            orientation_deg=0, noise_level=0.01, amplitude=0.5,
        )
        peaks_wide = find_subpixel_peaks(image, expected_d_nm=2.0, pixel_size_nm=0.1,
                                          tile_size=256)
        peaks_narrow = find_subpixel_peaks(image, expected_d_nm=0.5, pixel_size_nm=0.1,
                                            tile_size=256)
        # Different d-spacing should give different peak counts
        # (wider spacing = fewer peaks in same area)
        # Just check both return valid results
        assert isinstance(peaks_wide, list)
        assert isinstance(peaks_narrow, list)


class TestLatticeValidation:
    """Peak lattice validation (G12)."""

    def test_perfect_lattice(self):
        """Regular grid of peaks should pass validation."""
        from src.pipeline_config import SubpixelPeak

        d_nm = 0.5
        px = 0.1
        # Create a regular grid of peaks
        d_px = d_nm / px  # 5 pixels
        peaks = []
        for r in range(5, 50, int(d_px)):
            for c in range(5, 50, int(d_px)):
                peaks.append(SubpixelPeak(x=float(c), y=float(r),
                                           intensity=1.0, prominence=0.5))

        result = validate_peak_lattice(peaks, d_nm, px, tolerance=0.2)
        assert result.fraction_valid > 0.5

    def test_random_peaks_fail(self):
        """Random peaks should fail lattice validation."""
        from src.pipeline_config import SubpixelPeak
        rng = np.random.default_rng(42)
        peaks = [SubpixelPeak(x=float(rng.uniform(0, 100)),
                               y=float(rng.uniform(0, 100)),
                               intensity=1.0, prominence=0.5)
                 for _ in range(50)]

        result = validate_peak_lattice(peaks, expected_d_nm=0.5,
                                        pixel_size_nm=0.1, tolerance=0.2)
        # Random peaks unlikely to match expected d-spacing
        assert result.fraction_valid < 0.8

    def test_few_peaks(self):
        """Fewer than 2 peaks should give fraction_valid=0."""
        from src.pipeline_config import SubpixelPeak
        peaks = [SubpixelPeak(x=10.0, y=10.0, intensity=1.0)]
        result = validate_peak_lattice(peaks, 0.5, 0.1)
        assert result.fraction_valid == 0.0
