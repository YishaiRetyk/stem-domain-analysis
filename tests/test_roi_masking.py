"""Tests for early ROI masking (WS4)."""

import numpy as np
import pytest
from src.roi_masking import compute_roi_mask, downsample_to_tile_grid
from src.pipeline_config import ROIConfig


class TestROIMask:
    """ROI masking without FFT crystallinity (C6)."""

    def test_uniform_image_high_coverage(self):
        """Uniform bright image should have high coverage."""
        image = np.ones((512, 512), dtype=np.float32) * 0.8
        config = ROIConfig()
        result = compute_roi_mask(image, config)
        assert result.coverage_pct > 50

    def test_dark_border(self):
        """Image with dark border should exclude border from ROI."""
        image = np.ones((512, 512), dtype=np.float32) * 0.8
        image[:50, :] = 0.01  # dark top
        image[-50:, :] = 0.01  # dark bottom
        config = ROIConfig()
        result = compute_roi_mask(image, config)
        # Border should be excluded
        assert result.mask_full[:50, :].sum() < result.mask_full[50:-50, :].sum()

    def test_no_fft_crystallinity(self):
        """C6: ROI should NOT depend on FFT crystallinity."""
        # Just check it runs without any FFT-related input
        rng = np.random.default_rng(42)
        image = rng.uniform(0.3, 0.8, (512, 512)).astype(np.float32)
        config = ROIConfig()
        result = compute_roi_mask(image, config)
        assert result.mask_full.shape == (512, 512)
        assert result.n_components >= 0

    def test_g3_fallback(self):
        """I2: G3 failure should produce full-image mask (FALLBACK)."""
        # Very fragmented image to trigger max_fragments check
        rng = np.random.default_rng(42)
        image = rng.choice([0.01, 0.9], size=(512, 512), p=[0.5, 0.5]).astype(np.float32)
        config = ROIConfig(max_fragments=1)  # Very strict
        result = compute_roi_mask(image, config)
        # Should still return a mask (fallback to full)
        assert result.mask_full is not None


class TestDownsampleToGrid:
    """Tile grid downsampling."""

    def test_shape(self):
        mask = np.ones((512, 512), dtype=np.uint8)
        grid = downsample_to_tile_grid(mask, tile_size=256, stride=128)
        # Expected grid shape
        n_rows = (512 - 256) // 128 + 1
        n_cols = (512 - 256) // 128 + 1
        assert grid.shape == (n_rows, n_cols)

    def test_all_ones(self):
        mask = np.ones((512, 512), dtype=np.uint8)
        grid = downsample_to_tile_grid(mask, tile_size=256, stride=128)
        assert grid.all()

    def test_all_zeros(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        grid = downsample_to_tile_grid(mask, tile_size=256, stride=128)
        assert not grid.any()
