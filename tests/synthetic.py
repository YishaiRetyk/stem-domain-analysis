"""
Synthetic Data Generators for Pipeline Testing.

Provides known-ground-truth images for unit and integration tests.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict


def generate_single_crystal(
    shape: Tuple[int, int] = (512, 512),
    pixel_size_nm: float = 0.127,
    d_spacing: float = 0.4,
    orientation_deg: float = 30.0,
    noise_level: float = 0.05,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a 2D sinusoidal lattice image with known g-vectors.

    Creates a single-crystal image with two orthogonal lattice frequencies
    at the given d-spacing and orientation.

    Parameters
    ----------
    shape : (H, W)
    pixel_size_nm : nm per pixel
    d_spacing : lattice spacing in nm
    orientation_deg : angle of first g-vector from x-axis
    noise_level : std of additive Gaussian noise relative to amplitude
    amplitude : strength of the lattice signal (0–1)

    Returns
    -------
    image : (H, W) float64 array in [0, 1]
    """
    H, W = shape
    y_nm = np.arange(H).reshape(-1, 1) * pixel_size_nm
    x_nm = np.arange(W).reshape(1, -1) * pixel_size_nm

    angle_rad = np.radians(orientation_deg)
    g_mag = 1.0 / d_spacing  # cycles/nm

    # Two orthogonal g-vectors
    gx1 = g_mag * np.cos(angle_rad)
    gy1 = g_mag * np.sin(angle_rad)
    gx2 = g_mag * np.cos(angle_rad + np.pi / 2)
    gy2 = g_mag * np.sin(angle_rad + np.pi / 2)

    lattice = (np.cos(2 * np.pi * (gx1 * x_nm + gy1 * y_nm)) +
               np.cos(2 * np.pi * (gx2 * x_nm + gy2 * y_nm)))

    # Normalise to [0, 1] range
    image = 0.5 + amplitude * lattice / 2.0
    if noise_level > 0:
        image += np.random.default_rng(42).normal(0, noise_level, shape)

    return np.clip(image, 0, 1).astype(np.float64)


def generate_polycrystalline(
    shape: Tuple[int, int] = (1024, 1024),
    pixel_size_nm: float = 0.127,
    domains: Optional[List[Dict]] = None,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Generate a polycrystalline image with multiple domains.

    Parameters
    ----------
    shape : (H, W)
    pixel_size_nm : nm per pixel
    domains : list of dict with keys: region (r_min, r_max, c_min, c_max),
              d_spacing, orientation_deg, amplitude
    noise_level : additive Gaussian noise std

    Returns
    -------
    image : (H, W) float64 array
    """
    H, W = shape
    if domains is None:
        domains = [
            {"region": (0, H // 2, 0, W // 2), "d_spacing": 0.4,
             "orientation_deg": 0, "amplitude": 0.3},
            {"region": (0, H // 2, W // 2, W), "d_spacing": 0.4,
             "orientation_deg": 45, "amplitude": 0.3},
            {"region": (H // 2, H, 0, W), "d_spacing": 0.35,
             "orientation_deg": 90, "amplitude": 0.25},
        ]

    y_nm = np.arange(H).reshape(-1, 1) * pixel_size_nm
    x_nm = np.arange(W).reshape(1, -1) * pixel_size_nm
    image = np.full(shape, 0.5, dtype=np.float64)

    for dom in domains:
        r_min, r_max, c_min, c_max = dom["region"]
        d = dom["d_spacing"]
        angle = np.radians(dom["orientation_deg"])
        amp = dom.get("amplitude", 0.3)
        g = 1.0 / d
        gx = g * np.cos(angle)
        gy = g * np.sin(angle)

        lattice = np.cos(2 * np.pi * (gx * x_nm + gy * y_nm))
        image[r_min:r_max, c_min:c_max] += amp * lattice[r_min:r_max, c_min:c_max]

    if noise_level > 0:
        image += np.random.default_rng(42).normal(0, noise_level, shape)

    return np.clip(image, 0, 1).astype(np.float64)


def generate_strained_crystal(
    shape: Tuple[int, int] = (512, 512),
    pixel_size_nm: float = 0.127,
    d_spacing: float = 0.4,
    orientation_deg: float = 0.0,
    strain_xx: float = 0.02,
    strain_yy: float = -0.01,
    noise_level: float = 0.02,
) -> np.ndarray:
    """Generate a crystal with known uniform strain.

    The lattice is distorted so that d_x = d_spacing * (1 + strain_xx)
    and d_y = d_spacing * (1 + strain_yy).

    Returns
    -------
    image : (H, W) float64 array
    """
    H, W = shape
    y_nm = np.arange(H).reshape(-1, 1) * pixel_size_nm
    x_nm = np.arange(W).reshape(1, -1) * pixel_size_nm

    angle = np.radians(orientation_deg)
    g_mag = 1.0 / d_spacing

    # Strained g-vectors: g = g0 / (1 + strain) → larger strain = smaller g
    gx1 = g_mag / (1 + strain_xx) * np.cos(angle)
    gy1 = g_mag / (1 + strain_xx) * np.sin(angle)
    gx2 = g_mag / (1 + strain_yy) * np.cos(angle + np.pi / 2)
    gy2 = g_mag / (1 + strain_yy) * np.sin(angle + np.pi / 2)

    lattice = (np.cos(2 * np.pi * (gx1 * x_nm + gy1 * y_nm)) +
               np.cos(2 * np.pi * (gx2 * x_nm + gy2 * y_nm)))

    image = 0.5 + 0.3 * lattice / 2.0
    if noise_level > 0:
        image += np.random.default_rng(42).normal(0, noise_level, shape)

    return np.clip(image, 0, 1).astype(np.float64)


def generate_amorphous(
    shape: Tuple[int, int] = (512, 512),
    noise_level: float = 0.15,
) -> np.ndarray:
    """Generate a pure noise image with no lattice structure."""
    rng = np.random.default_rng(42)
    return np.clip(0.5 + rng.normal(0, noise_level, shape), 0, 1).astype(np.float64)


def generate_mixed(
    shape: Tuple[int, int] = (1024, 1024),
    pixel_size_nm: float = 0.127,
    crystalline_fraction: float = 0.5,
    d_spacing: float = 0.4,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Generate an image with crystalline center and amorphous border.

    Good for testing ROI masking.
    """
    H, W = shape
    rng = np.random.default_rng(42)

    # Amorphous background
    image = 0.5 + rng.normal(0, noise_level * 2, shape)

    # Crystalline circular center
    cy, cx = H // 2, W // 2
    radius = int(np.sqrt(crystalline_fraction * H * W / np.pi))
    y, x = np.mgrid[:H, :W]
    mask = ((y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2

    crystal = generate_single_crystal(shape, pixel_size_nm, d_spacing,
                                       orientation_deg=15, noise_level=noise_level,
                                       amplitude=0.4)
    image[mask] = crystal[mask]

    return np.clip(image, 0, 1).astype(np.float64)


def generate_with_gradient(
    shape: Tuple[int, int] = (512, 512),
    pixel_size_nm: float = 0.127,
    d_spacing: float = 0.4,
    gradient_strength: float = 0.3,
    noise_level: float = 0.02,
) -> np.ndarray:
    """Generate a lattice with strong low-frequency intensity gradient.

    Tests robustness of ROI + FFT-safe preprocess + bandpass peak finding.
    """
    crystal = generate_single_crystal(shape, pixel_size_nm, d_spacing,
                                       noise_level=noise_level)
    H, W = shape
    gradient = gradient_strength * np.linspace(0, 1, W).reshape(1, -1) * np.ones((H, 1))
    return np.clip(crystal + gradient, 0, 1).astype(np.float64)
