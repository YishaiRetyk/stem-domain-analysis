"""Tests for FFT-safe preprocessing (WS1a)."""

import numpy as np
import pytest
from src.preprocess_fft_safe import preprocess_fft_safe, _remove_hot_pixels
from src.pipeline_config import PreprocConfig


class TestHotPixelRemoval:
    """I1: hot-pixel removal replaces only outliers."""

    def test_replaces_hot_pixels(self):
        rng = np.random.default_rng(42)
        image = rng.normal(100, 5, (64, 64))
        # Insert hot pixels
        image[10, 10] = 10000
        image[20, 30] = 9000

        fixed, n_replaced = _remove_hot_pixels(image, sigma=5.0)
        # Hot pixels should be replaced (much lower than original)
        assert fixed[10, 10] < 200
        assert fixed[20, 30] < 200
        assert n_replaced >= 2

    def test_preserves_normal_pixels(self):
        rng = np.random.default_rng(42)
        image = rng.normal(100, 5, (512, 512))
        fixed, n_replaced = _remove_hot_pixels(image, sigma=5.0)
        # Most pixels should be unchanged (< 2% replaced on normal data)
        total = image.size
        assert n_replaced < total * 0.02


class TestPreprocessFFTSafe:
    """Branch A preprocessing."""

    def test_no_blur(self):
        """FFT-safe image should NOT be Gaussian blurred."""
        # Create an image with a small bright feature on a background
        rng = np.random.default_rng(42)
        image = rng.normal(500, 50, (512, 512))
        image[250:260, 250:260] = 1000  # bright patch

        config = PreprocConfig(hot_pixel_removal=False)
        result = preprocess_fft_safe(image, config)

        # The bright patch should still be relatively prominent
        patch_mean = result.image_fft[250:260, 250:260].mean()
        bg_mean = result.image_fft[100:110, 100:110].mean()
        assert patch_mean > bg_mean  # bright patch should stand out

    def test_output_range(self):
        rng = np.random.default_rng(42)
        image = rng.normal(1000, 100, (512, 512))
        config = PreprocConfig()
        result = preprocess_fft_safe(image, config)
        assert result.image_fft.min() >= 0.0
        assert result.image_fft.max() <= 1.0

    def test_confidence_normal(self):
        """Image with good dynamic range should get 'normal' confidence."""
        rng = np.random.default_rng(42)
        # Need (p99.9 - p0.1) / median > 10
        # Squared exponential gives low median with heavy tail
        image = rng.exponential(scale=50, size=(512, 512)) ** 2 + 1.0
        config = PreprocConfig()
        result = preprocess_fft_safe(image, config)
        assert result.confidence == "normal"

    def test_g2_degrade_fallback(self):
        """I3: G2 failure should fall back to min-max normalization."""
        # Image with almost no dynamic range -> low range_ratio
        image = np.ones((512, 512), dtype=np.float64) * 100
        image[:5, :] = 100.001
        config = PreprocConfig()
        result = preprocess_fft_safe(image, config)
        # Should still produce output (not crash)
        assert result.image_fft is not None
        assert result.image_fft.shape == (512, 512)
        assert result.confidence == "degraded"

    def test_diagnostics_present(self):
        rng = np.random.default_rng(42)
        image = rng.uniform(10, 1000, (512, 512))
        config = PreprocConfig()
        result = preprocess_fft_safe(image, config)
        assert "clipped_fraction" in result.diagnostics
        assert "intensity_range_ratio" in result.diagnostics

    def test_spectral_entropy_diagnostic(self):
        """C1: spectral entropy is a diagnostic, not a gate."""
        rng = np.random.default_rng(42)
        image = rng.uniform(10, 1000, (512, 512))
        config = PreprocConfig()
        result = preprocess_fft_safe(image, config)
        assert "spectral_entropy" in result.qc_metrics
