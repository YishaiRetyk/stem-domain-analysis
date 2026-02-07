"""
GPU Backend Abstraction Layer.

Provides DeviceContext for transparent GPU/CPU dispatching.
CuPy is optional — the module gracefully falls back to NumPy when unavailable.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Detect CuPy at import time
try:
    import cupy as cp
    import cupy.fft as cufft
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    cufft = None
    GPU_AVAILABLE = False


@dataclass
class GPUInfo:
    """GPU device information."""
    available: bool
    device_name: str
    total_memory_gb: float
    free_memory_gb: float
    cupy_version: Optional[str]


def get_gpu_info(device_id: int = 0) -> GPUInfo:
    """Query GPU device information."""
    if not GPU_AVAILABLE:
        return GPUInfo(
            available=False,
            device_name="N/A",
            total_memory_gb=0.0,
            free_memory_gb=0.0,
            cupy_version=None,
        )
    try:
        with cp.cuda.Device(device_id):
            mem_free, mem_total = cp.cuda.Device(device_id).mem_info
            return GPUInfo(
                available=True,
                device_name=cp.cuda.runtime.getDeviceProperties(device_id)["name"].decode(),
                total_memory_gb=mem_total / 1024**3,
                free_memory_gb=mem_free / 1024**3,
                cupy_version=cp.__version__,
            )
    except Exception as e:
        logger.warning(f"Failed to query GPU info: {e}")
        return GPUInfo(
            available=False,
            device_name=f"error: {e}",
            total_memory_gb=0.0,
            free_memory_gb=0.0,
            cupy_version=cp.__version__ if cp else None,
        )


class DeviceContext:
    """Central interface for GPU/CPU dispatching.

    Carry through the pipeline and use ``ctx.xp`` instead of ``np``.
    """

    def __init__(self, use_gpu: bool, device_id: int = 0):
        self._use_gpu = use_gpu and GPU_AVAILABLE
        self._device_id = device_id

        if self._use_gpu:
            self._xp = cp
            self._device = cp.cuda.Device(device_id)
            self._device.use()
        else:
            self._xp = np
            self._device = None

    @classmethod
    def create(cls, device: str = "auto", device_id: int = 0) -> "DeviceContext":
        """Create a DeviceContext.

        Parameters
        ----------
        device : str
            ``"auto"`` (GPU if available), ``"gpu"`` (GPU required),
            or ``"cpu"`` (force CPU).
        device_id : int
            CUDA device ordinal.
        """
        if device == "cpu":
            return cls(use_gpu=False, device_id=device_id)
        elif device == "gpu":
            if not GPU_AVAILABLE:
                warnings.warn(
                    "GPU requested but CuPy not available — falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return cls(use_gpu=False, device_id=device_id)
            return cls(use_gpu=True, device_id=device_id)
        else:  # auto
            return cls(use_gpu=GPU_AVAILABLE, device_id=device_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def xp(self):
        """Array module — ``cupy`` or ``numpy``."""
        return self._xp

    @property
    def using_gpu(self) -> bool:
        return self._use_gpu

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    def to_device(self, arr: np.ndarray):
        """Transfer a NumPy array to the compute device (no-copy if already there)."""
        if not self._use_gpu:
            return arr
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)

    def to_host(self, arr) -> np.ndarray:
        """Transfer an array back to host RAM (no-copy if already NumPy)."""
        if not self._use_gpu:
            return np.asarray(arr)
        if isinstance(arr, np.ndarray):
            return arr
        return cp.asnumpy(arr)

    # ------------------------------------------------------------------
    # FFT helpers
    # ------------------------------------------------------------------

    def fft2(self, arr, axes=None):
        """2-D FFT dispatching to CuPy or NumPy."""
        if self._use_gpu:
            return cufft.fft2(arr, axes=axes)
        kw = {} if axes is None else {"axes": axes}
        return np.fft.fft2(arr, **kw)

    def ifft2(self, arr, axes=None):
        """Inverse 2-D FFT."""
        if self._use_gpu:
            return cufft.ifft2(arr, axes=axes)
        kw = {} if axes is None else {"axes": axes}
        return np.fft.ifft2(arr, **kw)

    def fftshift(self, arr, axes=None):
        """fftshift dispatching."""
        if self._use_gpu:
            return cp.fft.fftshift(arr, axes=axes)
        return np.fft.fftshift(arr, axes=axes)

    def ifftshift(self, arr, axes=None):
        """ifftshift dispatching."""
        if self._use_gpu:
            return cp.fft.ifftshift(arr, axes=axes)
        return np.fft.ifftshift(arr, axes=axes)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def free_memory_gb(self) -> float:
        """Free device memory in GB.  Returns ``inf`` for CPU."""
        if not self._use_gpu:
            return float("inf")
        mem_free, _ = cp.cuda.Device(self._device_id).mem_info
        return mem_free / 1024**3

    def synchronize(self):
        """Block until all GPU work completes (no-op for CPU)."""
        if self._use_gpu:
            cp.cuda.Stream.null.synchronize()

    def clear_memory_pool(self):
        """Release cached GPU memory back to the driver (no-op for CPU)."""
        if self._use_gpu:
            pool = cp.get_default_memory_pool()
            pool.free_all_blocks()

    def max_batch_tiles(self, tile_size: int, dtype=np.float64,
                        memory_fraction: float = 0.7) -> int:
        """Estimate how many tiles fit in one batch given free GPU memory.

        Accounts for: input float64 tile, complex128 FFT, float64 power.

        Returns at least 1.  Returns 1024 for CPU (unbounded in practice).
        """
        if not self._use_gpu:
            return 1024

        free_bytes = self.free_memory_gb() * 1024**3 * memory_fraction
        # Per tile: float64 input + complex128 FFT + float64 power = 4 x float64
        bytes_per_tile = 4 * tile_size * tile_size * np.dtype(dtype).itemsize
        n = int(free_bytes / bytes_per_tile)
        return max(1, n)
