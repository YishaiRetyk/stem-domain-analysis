"""
Tests for GPU-accelerated tile FFT batching.

TestBatchCPU always runs (validates batch logic on CPU).
TestBatchGPU skipped when CuPy is absent.
"""

import numpy as np
import pytest

from src.gpu_backend import DeviceContext, GPU_AVAILABLE
from src.tile_fft import compute_tile_fft, compute_tile_fft_batch
from src.fft_features import create_2d_hann_window


# ======================================================================
# CPU batch tests (always run)
# ======================================================================

class TestBatchCPU:
    """Verify batch FFT on CPU matches N individual calls."""

    def _make_tiles(self, n_tiles=8, tile_size=64, seed=42):
        rng = np.random.default_rng(seed)
        tiles = rng.standard_normal((n_tiles, tile_size, tile_size))
        window = create_2d_hann_window(tile_size)
        return tiles, window

    def test_batch_matches_individual(self):
        tiles, window = self._make_tiles(n_tiles=8, tile_size=64)
        ctx = DeviceContext.create("cpu")

        batch_result = compute_tile_fft_batch(tiles, window, ctx)
        assert batch_result.shape == tiles.shape

        for i in range(tiles.shape[0]):
            individual = compute_tile_fft(tiles[i], window)
            np.testing.assert_allclose(batch_result[i], individual, rtol=1e-12)

    def test_batch_single_tile(self):
        tiles, window = self._make_tiles(n_tiles=1, tile_size=64)
        ctx = DeviceContext.create("cpu")

        batch_result = compute_tile_fft_batch(tiles, window, ctx)
        individual = compute_tile_fft(tiles[0], window)
        np.testing.assert_allclose(batch_result[0], individual, rtol=1e-12)

    def test_batch_non_negative_power(self):
        tiles, window = self._make_tiles(n_tiles=4, tile_size=64)
        ctx = DeviceContext.create("cpu")

        result = compute_tile_fft_batch(tiles, window, ctx)
        assert np.all(result >= 0)

    def test_batch_preserves_dtype(self):
        tiles, window = self._make_tiles(n_tiles=4, tile_size=64)
        ctx = DeviceContext.create("cpu")

        result = compute_tile_fft_batch(tiles, window, ctx)
        assert result.dtype == np.float64

    def test_batch_different_tile_sizes(self):
        """Test with 128x128 tiles."""
        tiles, window = self._make_tiles(n_tiles=4, tile_size=128)
        ctx = DeviceContext.create("cpu")

        batch_result = compute_tile_fft_batch(tiles, window, ctx)
        assert batch_result.shape == (4, 128, 128)

        for i in range(4):
            individual = compute_tile_fft(tiles[i], window)
            np.testing.assert_allclose(batch_result[i], individual, rtol=1e-12)


# ======================================================================
# GPU batch tests (skip if no GPU)
# ======================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy / GPU not available")
class TestBatchGPU:

    def _make_tiles(self, n_tiles=8, tile_size=64, seed=42):
        rng = np.random.default_rng(seed)
        tiles = rng.standard_normal((n_tiles, tile_size, tile_size))
        window = create_2d_hann_window(tile_size)
        return tiles, window

    def test_gpu_batch_matches_cpu(self):
        tiles, window = self._make_tiles(n_tiles=16, tile_size=64)
        ctx_cpu = DeviceContext.create("cpu")
        ctx_gpu = DeviceContext.create("gpu")

        cpu_result = compute_tile_fft_batch(tiles, window, ctx_cpu)
        gpu_result = compute_tile_fft_batch(tiles, window, ctx_gpu)

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_gpu_large_batch(self):
        tiles, window = self._make_tiles(n_tiles=128, tile_size=64)
        ctx = DeviceContext.create("gpu")

        result = compute_tile_fft_batch(tiles, window, ctx)
        assert result.shape == (128, 64, 64)
        assert np.all(result >= 0)

    def test_gpu_batch_256_tiles(self):
        """256x256 tiles matching real pipeline tile size."""
        tiles, window = self._make_tiles(n_tiles=8, tile_size=256)
        ctx_cpu = DeviceContext.create("cpu")
        ctx_gpu = DeviceContext.create("gpu")

        cpu_result = compute_tile_fft_batch(tiles, window, ctx_cpu)
        gpu_result = compute_tile_fft_batch(tiles, window, ctx_gpu)

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

    def test_gpu_result_is_numpy(self):
        """compute_tile_fft_batch should return host (numpy) arrays."""
        tiles, window = self._make_tiles(n_tiles=4, tile_size=64)
        ctx = DeviceContext.create("gpu")

        result = compute_tile_fft_batch(tiles, window, ctx)
        assert isinstance(result, np.ndarray)
