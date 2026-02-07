"""
Tests for GPU-accelerated GPA module.

CPU-path tests always run; GPU-path tests are skipped when CuPy is absent.
"""

import numpy as np
import pytest
from scipy import ndimage as sp_ndimage

from src.gpu_backend import DeviceContext, GPU_AVAILABLE
from src.fft_coords import FFTGrid
from src.pipeline_config import GVector, GPAConfig, DisplacementField
from src.gpa import (
    compute_gpa_phase, smooth_displacement, compute_strain_field,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_perfect_lattice(N=128, pixel_size_nm=0.1, d_spacing=0.5, angle_deg=0):
    """Create a perfect lattice image and its expected g-vector."""
    g_mag = 1.0 / d_spacing
    angle_rad = np.radians(angle_deg)
    gx = g_mag * np.cos(angle_rad)
    gy = g_mag * np.sin(angle_rad)

    y_nm = np.arange(N).reshape(-1, 1) * pixel_size_nm
    x_nm = np.arange(N).reshape(1, -1) * pixel_size_nm
    image = 0.5 + 0.3 * np.cos(2 * np.pi * (gx * x_nm + gy * y_nm))

    gvec = GVector(gx=gx, gy=gy, magnitude=g_mag, angle_deg=angle_deg,
                   d_spacing=d_spacing, snr=10.0, fwhm=0.05, ring_index=0)
    return image, gvec


# ======================================================================
# DeviceContext new method tests (CPU — always run)
# ======================================================================

class TestDeviceContextNewMethodsCPU:
    """Test gaussian_filter and gradient on CPU path."""

    def test_gaussian_filter_matches_scipy(self):
        ctx = DeviceContext.create("cpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64))

        result = ctx.gaussian_filter(data, sigma=2.0)
        expected = sp_ndimage.gaussian_filter(data, sigma=2.0)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_gaussian_filter_with_kwargs(self):
        ctx = DeviceContext.create("cpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64))

        result = ctx.gaussian_filter(data, sigma=2.0, mode="constant", cval=0.0)
        expected = sp_ndimage.gaussian_filter(data, sigma=2.0, mode="constant", cval=0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_gradient_matches_numpy_axis0(self):
        ctx = DeviceContext.create("cpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64))

        result = ctx.gradient(data, axis=0)
        expected = np.gradient(data, axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_gradient_matches_numpy_axis1(self):
        ctx = DeviceContext.create("cpu")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64))

        result = ctx.gradient(data, axis=1)
        expected = np.gradient(data, axis=1)
        np.testing.assert_allclose(result, expected, rtol=1e-14)


# ======================================================================
# CPU backward compatibility tests (always run)
# ======================================================================

class TestGPAPhaseCPUCompat:
    """Verify ctx=None backward compat and ctx=CPU matches ctx=None."""

    def test_ctx_none_unchanged(self):
        """ctx=None should produce the same result as original code."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)

        result = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                   amplitude_threshold=0.01, ctx=None)

        assert result.phase_raw.shape == (128, 128)
        assert result.amplitude.shape == (128, 128)
        assert result.phase_raw.dtype == np.float32
        valid = result.amplitude_mask & ~np.isnan(result.phase_unwrapped)
        assert np.sum(valid) > 0

    def test_ctx_cpu_matches_ctx_none(self):
        """ctx=CPU should produce bit-identical results to ctx=None."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)
        ctx_cpu = DeviceContext.create("cpu")

        result_none = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                        amplitude_threshold=0.01, ctx=None)
        result_cpu = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                       amplitude_threshold=0.01, ctx=ctx_cpu)

        np.testing.assert_array_equal(result_none.phase_raw, result_cpu.phase_raw)
        np.testing.assert_array_equal(result_none.amplitude, result_cpu.amplitude)
        np.testing.assert_array_equal(result_none.amplitude_mask, result_cpu.amplitude_mask)


# ======================================================================
# FFT caching tests (CPU — always run)
# ======================================================================

class TestFFTCaching:
    """Verify _cached_ft_shifted produces identical results to uncached."""

    def test_cached_matches_uncached(self):
        """Phase and amplitude should be bit-identical with cache."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)

        # Uncached
        result_uncached = compute_gpa_phase(
            image, gvec, mask_sigma_q=0.1, fft_grid=grid,
            amplitude_threshold=0.01, ctx=None,
        )

        # Cached
        cached_ft = np.fft.fftshift(np.fft.fft2(image))
        result_cached = compute_gpa_phase(
            image, gvec, mask_sigma_q=0.1, fft_grid=grid,
            amplitude_threshold=0.01, ctx=None,
            _cached_ft_shifted=cached_ft,
        )

        np.testing.assert_array_equal(result_uncached.phase_raw, result_cached.phase_raw)
        np.testing.assert_array_equal(result_uncached.amplitude, result_cached.amplitude)
        np.testing.assert_array_equal(result_uncached.amplitude_mask, result_cached.amplitude_mask)

    def test_cached_with_ctx_cpu_matches(self):
        """Cache + ctx=CPU should match no-cache + ctx=None."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)
        ctx_cpu = DeviceContext.create("cpu")

        result_none = compute_gpa_phase(
            image, gvec, mask_sigma_q=0.1, fft_grid=grid,
            amplitude_threshold=0.01, ctx=None,
        )

        cached_ft = np.fft.fftshift(np.fft.fft2(image))
        result_cached = compute_gpa_phase(
            image, gvec, mask_sigma_q=0.1, fft_grid=grid,
            amplitude_threshold=0.01, ctx=ctx_cpu,
            _cached_ft_shifted=cached_ft,
        )

        np.testing.assert_array_equal(result_none.phase_raw, result_cached.phase_raw)
        np.testing.assert_array_equal(result_none.amplitude, result_cached.amplitude)

    def test_multiple_calls_same_cache(self):
        """Multiple calls with same cache should be identical."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)

        cached_ft = np.fft.fftshift(np.fft.fft2(image))

        results = []
        for _ in range(3):
            r = compute_gpa_phase(
                image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                amplitude_threshold=0.01, ctx=None,
                _cached_ft_shifted=cached_ft,
            )
            results.append(r)

        for r in results[1:]:
            np.testing.assert_array_equal(results[0].phase_raw, r.phase_raw)
            np.testing.assert_array_equal(results[0].amplitude, r.amplitude)

    def test_cache_not_mutated(self):
        """_cached_ft_shifted should not be modified by compute_gpa_phase."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)

        cached_ft = np.fft.fftshift(np.fft.fft2(image))
        cached_ft_copy = cached_ft.copy()

        compute_gpa_phase(
            image, gvec, mask_sigma_q=0.1, fft_grid=grid,
            amplitude_threshold=0.01, ctx=None,
            _cached_ft_shifted=cached_ft,
        )

        np.testing.assert_array_equal(cached_ft, cached_ft_copy)


# ======================================================================
# smooth_displacement + compute_strain_field with ctx (CPU — always run)
# ======================================================================

class TestSmoothDisplacementCPU:

    def test_ctx_none_matches_original(self):
        rng = np.random.default_rng(42)
        disp = DisplacementField(
            ux=rng.normal(0, 0.1, (64, 64)),
            uy=rng.normal(0, 0.1, (64, 64)),
        )
        result_none = smooth_displacement(disp, sigma=2.0, ctx=None)
        result_cpu = smooth_displacement(disp, sigma=2.0, ctx=DeviceContext.create("cpu"))

        np.testing.assert_allclose(result_none.ux, result_cpu.ux, rtol=1e-12)
        np.testing.assert_allclose(result_none.uy, result_cpu.uy, rtol=1e-12)

    def test_smoothing_reduces_noise(self):
        rng = np.random.default_rng(42)
        disp = DisplacementField(
            ux=rng.normal(0, 0.1, (64, 64)),
            uy=rng.normal(0, 0.1, (64, 64)),
        )
        ctx = DeviceContext.create("cpu")
        smoothed = smooth_displacement(disp, sigma=2.0, ctx=ctx)
        assert np.std(smoothed.ux) < np.std(disp.ux)


class TestComputeStrainFieldCPU:

    def test_ctx_none_matches_original(self):
        rng = np.random.default_rng(42)
        disp = DisplacementField(
            ux=rng.normal(0, 0.01, (64, 64)),
            uy=rng.normal(0, 0.01, (64, 64)),
        )
        strain_none = compute_strain_field(disp, pixel_size_nm=0.1, ctx=None)
        strain_cpu = compute_strain_field(disp, pixel_size_nm=0.1, ctx=DeviceContext.create("cpu"))

        np.testing.assert_allclose(strain_none.exx, strain_cpu.exx, rtol=1e-12)
        np.testing.assert_allclose(strain_none.eyy, strain_cpu.eyy, rtol=1e-12)
        np.testing.assert_allclose(strain_none.exy, strain_cpu.exy, rtol=1e-12)
        np.testing.assert_allclose(strain_none.rotation, strain_cpu.rotation, rtol=1e-12)

    def test_linear_displacement_constant_strain(self):
        N = 64
        px = 0.1
        strain_xx = 0.02
        x = np.arange(N).reshape(1, -1) * px
        ux = np.broadcast_to(strain_xx * x, (N, N)).copy()
        uy = np.zeros((N, N))

        disp = DisplacementField(ux=ux, uy=uy)
        strain = compute_strain_field(disp, px, ctx=DeviceContext.create("cpu"))

        interior = strain.exx[10:-10, 10:-10]
        assert abs(np.mean(interior) - strain_xx) < 0.005


# ======================================================================
# GPU-path tests (skip if no GPU)
# ======================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy / GPU not available")
class TestDeviceContextNewMethodsGPU:

    def test_gaussian_filter_gpu_matches_cpu(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64))

        ctx_gpu = DeviceContext.create("gpu")
        data_d = ctx_gpu.to_device(data)
        result_gpu = ctx_gpu.to_host(ctx_gpu.gaussian_filter(data_d, sigma=2.0))

        expected = sp_ndimage.gaussian_filter(data, sigma=2.0)
        np.testing.assert_allclose(result_gpu, expected, atol=1e-10)

    def test_gradient_gpu_matches_cpu(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64))

        ctx_gpu = DeviceContext.create("gpu")
        data_d = ctx_gpu.to_device(data)

        for axis in (0, 1):
            result_gpu = ctx_gpu.to_host(ctx_gpu.gradient(data_d, axis=axis))
            expected = np.gradient(data, axis=axis)
            np.testing.assert_allclose(result_gpu, expected, atol=1e-10)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy / GPU not available")
class TestGPAPhaseGPU:

    def test_gpu_matches_cpu(self):
        """GPU compute_gpa_phase should match CPU within atol=1e-6."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)

        result_cpu = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                       amplitude_threshold=0.01, ctx=None)
        ctx_gpu = DeviceContext.create("gpu")
        result_gpu = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                       amplitude_threshold=0.01, ctx=ctx_gpu)

        # All outputs should be numpy arrays
        assert isinstance(result_gpu.phase_raw, np.ndarray)
        assert isinstance(result_gpu.amplitude, np.ndarray)
        assert isinstance(result_gpu.amplitude_mask, np.ndarray)
        assert isinstance(result_gpu.phase_unwrapped, np.ndarray)

        np.testing.assert_allclose(result_gpu.phase_raw, result_cpu.phase_raw, atol=1e-6)
        np.testing.assert_allclose(result_gpu.amplitude, result_cpu.amplitude, atol=1e-6)

    def test_gpu_outputs_are_numpy(self):
        """All returned arrays must be host numpy, not CuPy."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)
        ctx_gpu = DeviceContext.create("gpu")

        result = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                   amplitude_threshold=0.01, ctx=ctx_gpu)

        assert type(result.phase_raw).__module__ == "numpy"
        assert type(result.amplitude).__module__ == "numpy"
        assert type(result.amplitude_mask).__module__ == "numpy"
        assert type(result.phase_unwrapped).__module__ == "numpy"

    def test_gpu_cached_matches_gpu_uncached(self):
        """GPU with cached FFT should match GPU uncached."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(128, 128, 0.1)
        ctx_gpu = DeviceContext.create("gpu")

        result_uncached = compute_gpa_phase(
            image, gvec, mask_sigma_q=0.1, fft_grid=grid,
            amplitude_threshold=0.01, ctx=ctx_gpu,
        )

        # Compute cache on GPU, bring to CPU (matches run_gpa_region pattern)
        img_d = ctx_gpu.to_device(image.astype(np.float64))
        cached_ft = ctx_gpu.to_host(ctx_gpu.fftshift(ctx_gpu.fft2(img_d)))
        del img_d

        result_cached = compute_gpa_phase(
            image, gvec, mask_sigma_q=0.1, fft_grid=grid,
            amplitude_threshold=0.01, ctx=ctx_gpu,
            _cached_ft_shifted=cached_ft,
        )

        np.testing.assert_allclose(result_uncached.phase_raw, result_cached.phase_raw, atol=1e-6)
        np.testing.assert_allclose(result_uncached.amplitude, result_cached.amplitude, atol=1e-6)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy / GPU not available")
class TestSmoothDisplacementGPU:

    def test_gpu_matches_cpu(self):
        rng = np.random.default_rng(42)
        disp = DisplacementField(
            ux=rng.normal(0, 0.1, (64, 64)),
            uy=rng.normal(0, 0.1, (64, 64)),
        )
        result_cpu = smooth_displacement(disp, sigma=2.0, ctx=None)
        result_gpu = smooth_displacement(disp, sigma=2.0, ctx=DeviceContext.create("gpu"))

        np.testing.assert_allclose(result_gpu.ux, result_cpu.ux, atol=1e-10)
        np.testing.assert_allclose(result_gpu.uy, result_cpu.uy, atol=1e-10)

        # Outputs must be numpy
        assert isinstance(result_gpu.ux, np.ndarray)
        assert isinstance(result_gpu.uy, np.ndarray)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy / GPU not available")
class TestComputeStrainFieldGPU:

    def test_gpu_matches_cpu(self):
        rng = np.random.default_rng(42)
        disp = DisplacementField(
            ux=rng.normal(0, 0.01, (64, 64)),
            uy=rng.normal(0, 0.01, (64, 64)),
        )
        strain_cpu = compute_strain_field(disp, pixel_size_nm=0.1, ctx=None)
        strain_gpu = compute_strain_field(disp, pixel_size_nm=0.1, ctx=DeviceContext.create("gpu"))

        np.testing.assert_allclose(strain_gpu.exx, strain_cpu.exx, atol=1e-10)
        np.testing.assert_allclose(strain_gpu.eyy, strain_cpu.eyy, atol=1e-10)
        np.testing.assert_allclose(strain_gpu.exy, strain_cpu.exy, atol=1e-10)
        np.testing.assert_allclose(strain_gpu.rotation, strain_cpu.rotation, atol=1e-10)

        # Outputs must be numpy
        assert isinstance(strain_gpu.exx, np.ndarray)
