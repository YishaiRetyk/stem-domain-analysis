"""
Tests for src/gpu_backend — DeviceContext, transfers, FFT dispatch.

CPU-path tests always run; GPU-path tests are skipped when CuPy is absent.
"""

import numpy as np
import pytest

from src.gpu_backend import DeviceContext, GPU_AVAILABLE, get_gpu_info, GPUInfo


# ======================================================================
# CPU-path tests (always run)
# ======================================================================

class TestDeviceContextCPU:
    """Tests that exercise the CPU code path exclusively."""

    def test_create_cpu(self):
        ctx = DeviceContext.create("cpu")
        assert not ctx.using_gpu
        assert ctx.xp is np

    def test_create_auto_returns_context(self):
        ctx = DeviceContext.create("auto")
        # Should always succeed regardless of GPU presence
        assert isinstance(ctx, DeviceContext)
        assert ctx.xp is not None

    def test_to_device_noop_on_cpu(self):
        ctx = DeviceContext.create("cpu")
        arr = np.ones((4, 4))
        result = ctx.to_device(arr)
        # Should be the exact same object (no copy)
        assert result is arr

    def test_to_host_noop_on_cpu(self):
        ctx = DeviceContext.create("cpu")
        arr = np.ones((4, 4))
        result = ctx.to_host(arr)
        assert result is arr

    def test_fft2_matches_numpy(self):
        ctx = DeviceContext.create("cpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64))

        result = ctx.fft2(data)
        expected = np.fft.fft2(data)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_ifft2_matches_numpy(self):
        ctx = DeviceContext.create("cpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))

        result = ctx.ifft2(data)
        expected = np.fft.ifft2(data)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_fftshift_matches_numpy(self):
        ctx = DeviceContext.create("cpu")
        data = np.arange(16).reshape(4, 4).astype(float)

        result = ctx.fftshift(data)
        expected = np.fft.fftshift(data)
        np.testing.assert_array_equal(result, expected)

    def test_ifftshift_matches_numpy(self):
        ctx = DeviceContext.create("cpu")
        data = np.arange(16).reshape(4, 4).astype(float)

        result = ctx.ifftshift(data)
        expected = np.fft.ifftshift(data)
        np.testing.assert_array_equal(result, expected)

    def test_fft2_with_axes(self):
        ctx = DeviceContext.create("cpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, 64, 64))

        result = ctx.fft2(data, axes=(-2, -1))
        expected = np.fft.fft2(data, axes=(-2, -1))
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_free_memory_inf_on_cpu(self):
        ctx = DeviceContext.create("cpu")
        assert ctx.free_memory_gb() == float("inf")

    def test_max_batch_tiles_cpu(self):
        ctx = DeviceContext.create("cpu")
        n = ctx.max_batch_tiles(256)
        assert n == 1024

    def test_synchronize_noop_on_cpu(self):
        ctx = DeviceContext.create("cpu")
        ctx.synchronize()  # should not raise

    def test_clear_memory_pool_noop_on_cpu(self):
        ctx = DeviceContext.create("cpu")
        ctx.clear_memory_pool()  # should not raise


class TestGPUInfoCPU:
    """Test get_gpu_info when no GPU is present."""

    def test_get_gpu_info_no_gpu(self):
        if GPU_AVAILABLE:
            pytest.skip("GPU is available — testing no-GPU path not meaningful")
        info = get_gpu_info()
        assert isinstance(info, GPUInfo)
        assert not info.available


class TestGPUFallback:
    """Test that requesting GPU without CuPy falls back gracefully."""

    def test_gpu_request_without_cupy(self):
        if GPU_AVAILABLE:
            pytest.skip("CuPy available — fallback path not exercised")
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            ctx = DeviceContext.create("gpu")
        assert not ctx.using_gpu


# ======================================================================
# GPU-path tests (skip if no GPU)
# ======================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy / GPU not available")
class TestDeviceContextGPU:

    def test_create_gpu(self):
        ctx = DeviceContext.create("gpu")
        assert ctx.using_gpu
        import cupy
        assert ctx.xp is cupy

    def test_round_trip_transfer(self):
        ctx = DeviceContext.create("gpu")
        arr = np.arange(100, dtype=np.float64).reshape(10, 10)
        on_device = ctx.to_device(arr)
        back = ctx.to_host(on_device)
        np.testing.assert_array_equal(back, arr)

    def test_to_device_noop_if_already_on_device(self):
        import cupy
        ctx = DeviceContext.create("gpu")
        arr_gpu = cupy.ones((4, 4))
        result = ctx.to_device(arr_gpu)
        assert result is arr_gpu

    def test_to_host_noop_if_already_numpy(self):
        ctx = DeviceContext.create("gpu")
        arr = np.ones((4, 4))
        result = ctx.to_host(arr)
        assert result is arr

    def test_fft2_numerical_equivalence(self):
        ctx = DeviceContext.create("gpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((128, 128))

        data_d = ctx.to_device(data)
        result_d = ctx.fft2(data_d)
        result = ctx.to_host(result_d)

        expected = np.fft.fft2(data)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_fft2_batch_axes(self):
        ctx = DeviceContext.create("gpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 64, 64))

        data_d = ctx.to_device(data)
        result_d = ctx.fft2(data_d, axes=(-2, -1))
        result = ctx.to_host(result_d)

        expected = np.fft.fft2(data, axes=(-2, -1))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_fftshift_gpu(self):
        ctx = DeviceContext.create("gpu")
        data = np.arange(16).reshape(4, 4).astype(float)
        data_d = ctx.to_device(data)
        result = ctx.to_host(ctx.fftshift(data_d))
        expected = np.fft.fftshift(data)
        np.testing.assert_array_equal(result, expected)

    def test_free_memory_positive(self):
        ctx = DeviceContext.create("gpu")
        mem = ctx.free_memory_gb()
        assert mem > 0

    def test_max_batch_tiles_positive(self):
        ctx = DeviceContext.create("gpu")
        n = ctx.max_batch_tiles(256)
        assert n >= 1

    def test_gpu_info(self):
        info = get_gpu_info()
        assert info.available
        assert info.total_memory_gb > 0
        assert info.cupy_version is not None
        assert len(info.device_name) > 0
