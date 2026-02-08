"""
Canonical FFT Coordinate System (B1, F1).

All spatial frequencies in CYCLES/NM (not radians/nm).
d = 1/|g|, no 2pi factor anywhere.

Convention:
- FFT output: fftshift(fft2(image * window)), DC at (H//2, W//2)
- qx increases rightward, qy increases downward (image convention)
- q_scale = 1 / (N * pixel_size_nm) cycles/nm per pixel
- Rectangular images: separate qx_scale and qy_scale

All modules MUST use FFTGrid for coordinate conversions.
"""

import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


class FFTGrid:
    """Canonical FFT coordinate system for the pipeline.

    Parameters
    ----------
    height, width : int
        Image (or tile) dimensions in pixels.
    pixel_size_nm : float
        Real-space pixel pitch in nanometres.
    """

    frequency_unit: str = "cycles/nm"

    def __init__(self, height: int, width: int, pixel_size_nm: float):
        if pixel_size_nm <= 0:
            raise ValueError(f"pixel_size_nm must be positive, got {pixel_size_nm}")
        if height < 1 or width < 1:
            raise ValueError(f"dimensions must be positive, got {height}x{width}")

        self.height = height
        self.width = width
        self.pixel_size_nm = pixel_size_nm

        # DC center after fftshift
        self.dc_x = width // 2
        self.dc_y = height // 2

        # Separate scales for rectangular images
        self.qx_scale = 1.0 / (width * pixel_size_nm)   # cycles/nm per pixel in x
        self.qy_scale = 1.0 / (height * pixel_size_nm)  # cycles/nm per pixel in y

        # Grid cache — avoids recomputing full (H,W) arrays on repeated calls
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def px_to_q(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """Convert FFT pixel coordinates to spatial frequency (cycles/nm).

        Parameters
        ----------
        px_x, px_y : float
            Pixel position in the *shifted* FFT (can be fractional).

        Returns
        -------
        (qx, qy) in cycles/nm.
        """
        qx = (px_x - self.dc_x) * self.qx_scale
        qy = (px_y - self.dc_y) * self.qy_scale
        return (qx, qy)

    def q_to_px(self, qx: float, qy: float) -> Tuple[float, float]:
        """Convert spatial frequency (cycles/nm) to FFT pixel coordinates.

        Returns fractional pixel positions in the shifted FFT.
        """
        px_x = qx / self.qx_scale + self.dc_x
        px_y = qy / self.qy_scale + self.dc_y
        return (px_x, px_y)

    def q_mag(self, px_x: float, px_y: float) -> float:
        """Spatial-frequency magnitude |q| at a pixel position (cycles/nm)."""
        qx, qy = self.px_to_q(px_x, px_y)
        return float(np.sqrt(qx**2 + qy**2))

    # ------------------------------------------------------------------
    # Grid helpers (return full arrays)
    # ------------------------------------------------------------------

    def q_mag_grid(self) -> np.ndarray:
        """Return (H, W) array of |q| in cycles/nm. Cached after first call."""
        if 'q_mag' not in self._cache:
            y, x = np.mgrid[:self.height, :self.width]
            qx = (x - self.dc_x) * self.qx_scale
            qy = (y - self.dc_y) * self.qy_scale
            self._cache['q_mag'] = np.sqrt(qx**2 + qy**2)
        return self._cache['q_mag']

    def qx_grid(self) -> np.ndarray:
        """Return (H, W) array of qx values in cycles/nm. Cached after first call."""
        if 'qx' not in self._cache:
            x = np.arange(self.width)[np.newaxis, :]
            self._cache['qx'] = (x - self.dc_x) * self.qx_scale * np.ones((self.height, 1))
        return self._cache['qx']

    def qy_grid(self) -> np.ndarray:
        """Return (H, W) array of qy values in cycles/nm. Cached after first call."""
        if 'qy' not in self._cache:
            y = np.arange(self.height)[:, np.newaxis]
            self._cache['qy'] = (y - self.dc_y) * self.qy_scale * np.ones((1, self.width))
        return self._cache['qy']

    def angle_grid_deg(self) -> np.ndarray:
        """Return (H, W) array of angles in degrees from DC center. Cached.

        Uses arctan2(qy, qx), range (-180, 180].
        """
        if 'angle_deg' not in self._cache:
            y, x = np.mgrid[:self.height, :self.width]
            qx = (x - self.dc_x) * self.qx_scale
            qy = (y - self.dc_y) * self.qy_scale
            self._cache['angle_deg'] = np.degrees(np.arctan2(qy, qx))
        return self._cache['angle_deg']

    def angle_grid_rad(self) -> np.ndarray:
        """Return (H, W) array of angles in radians from DC center. Cached."""
        if 'angle_rad' not in self._cache:
            y, x = np.mgrid[:self.height, :self.width]
            qx = (x - self.dc_x) * self.qx_scale
            qy = (y - self.dc_y) * self.qy_scale
            self._cache['angle_rad'] = np.arctan2(qy, qx)
        return self._cache['angle_rad']

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def d_spacing(q_magnitude: float) -> float:
        """Convert |q| in cycles/nm to d-spacing in nm. No 2pi factor (F1)."""
        if q_magnitude <= 0:
            return float('inf')
        return 1.0 / q_magnitude

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise for parameters.json."""
        return {
            "frequency_unit": self.frequency_unit,
            "d_spacing_formula": "d = 1/|g| (no 2pi)",
            "dc_center": f"{self.dc_y}, {self.dc_x}",
            "qx_scale": round(self.qx_scale, 6),
            "qy_scale": round(self.qy_scale, 6),
            "qx_direction": "rightward",
            "qy_direction": "downward",
            "height": self.height,
            "width": self.width,
            "pixel_size_nm": self.pixel_size_nm,
        }

    def __repr__(self) -> str:
        return (f"FFTGrid({self.height}x{self.width}, px={self.pixel_size_nm}nm, "
                f"qx_scale={self.qx_scale:.6f}, qy_scale={self.qy_scale:.6f})")


def compute_effective_q_min(fft_grid: FFTGrid, *, enabled: bool = True,
                            q_min_cycles_per_nm: float = 0.1,
                            dc_bin_count: int = 3,
                            auto_q_min: bool = True) -> float:
    """Compute the effective low-q exclusion threshold for a given FFT grid.

    Parameters
    ----------
    fft_grid : FFTGrid
        The coordinate system (global or tile grid).
    enabled : bool
        Master switch. If False, returns 0.0.
    q_min_cycles_per_nm : float
        Physical floor in cycles/nm.
    dc_bin_count : int
        Number of radial bins to always exclude around DC.
    auto_q_min : bool
        If True, derive q_min from image geometry and floor.
        If False, use q_min_cycles_per_nm directly.

    Returns
    -------
    float
        Effective q_min in cycles/nm.
    """
    if not enabled:
        logger.debug("Low-q exclusion disabled, returning q_min=0.0")
        return 0.0

    if auto_q_min:
        q_scale_min = min(fft_grid.qx_scale, fft_grid.qy_scale)
        q_min_from_bins = dc_bin_count * q_scale_min
        effective = max(q_min_from_bins, q_min_cycles_per_nm)
        logger.info("Low-q exclusion (auto): dc_bins=%d × q_scale=%.6f = %.4f, "
                     "floor=%.4f → effective_q_min=%.4f cycles/nm",
                     dc_bin_count, q_scale_min, q_min_from_bins,
                     q_min_cycles_per_nm, effective)
    else:
        effective = q_min_cycles_per_nm
        logger.info("Low-q exclusion (manual): effective_q_min=%.4f cycles/nm",
                     effective)

    return effective
