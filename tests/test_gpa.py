"""Tests for GPA engine (WS5)."""

import numpy as np
import pytest
from src.fft_coords import FFTGrid
from src.pipeline_config import GVector, GPAConfig
from src.gpa import (
    compute_gpa_phase, compute_displacement_field,
    smooth_displacement, compute_strain_field,
)


def _make_perfect_lattice(N=256, pixel_size_nm=0.1, d_spacing=0.5, angle_deg=0):
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


class TestGPAPhase:
    """GPA phase extraction tests."""

    def test_flat_phase_for_perfect_lattice(self):
        """F2: Perfect lattice should give flat phase (no residual carrier)."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(256, 256, 0.1)

        phase_result = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                          amplitude_threshold=0.01)

        # Check phase in high-amplitude region
        valid = phase_result.amplitude_mask & ~np.isnan(phase_result.phase_unwrapped)
        if np.sum(valid) > 100:
            phase_vals = phase_result.phase_unwrapped[valid]
            phase_std = np.std(phase_vals)
            # Should be fairly flat (std < 0.5 rad for a perfect lattice)
            assert phase_std < 0.5, f"Phase std = {phase_std:.3f} rad (expected < 0.5)"

    def test_unwrap_success_fraction(self):
        """B5: unwrap_success_fraction should be reported."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(256, 256, 0.1)
        result = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid)
        assert 0 <= result.unwrap_success_fraction <= 1.0

    def test_amplitude_mask_excludes_low_signal(self):
        """B5: low-amplitude regions should be masked."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(256, 256, 0.1)
        result = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid,
                                    amplitude_threshold=0.5)
        # High threshold should exclude more pixels
        n_masked = np.sum(~result.amplitude_mask)
        assert n_masked > 0

    def test_no_zero_forcing(self):
        """F3: phase unwrapping should NOT zero-force masked regions."""
        image, gvec = _make_perfect_lattice()
        grid = FFTGrid(256, 256, 0.1)
        result = compute_gpa_phase(image, gvec, mask_sigma_q=0.1, fft_grid=grid)
        # Masked regions should be NaN, not zero
        outside_mask = ~result.amplitude_mask
        if np.sum(outside_mask) > 0:
            vals = result.phase_unwrapped[outside_mask]
            assert np.all(np.isnan(vals)), "Masked regions should be NaN, not zero"


class TestDisplacementStrain:
    """Displacement and strain computation tests."""

    def test_displacement_from_two_phases(self):
        """Should compute displacement from two phase fields."""
        N = 256
        grid = FFTGrid(N, N, 0.1)

        g1 = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)
        g2 = GVector(gx=0.0, gy=2.0, magnitude=2.0, angle_deg=90,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)

        image1, _ = _make_perfect_lattice(N, 0.1, 0.5, 0)
        image2, _ = _make_perfect_lattice(N, 0.1, 0.5, 90)

        phase1 = compute_gpa_phase(image1, g1, 0.1, grid)
        phase2 = compute_gpa_phase(image2, g2, 0.1, grid)

        disp = compute_displacement_field(phase1, phase2)
        assert disp is not None
        assert disp.ux.shape == (N, N)
        assert disp.uy.shape == (N, N)

    def test_collinear_g_vectors_returns_none(self):
        """Collinear g-vectors should return None (no displacement)."""
        N = 64
        grid = FFTGrid(N, N, 0.1)
        g1 = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)
        g2 = GVector(gx=4.0, gy=0.0, magnitude=4.0, angle_deg=0,
                      d_spacing=0.25, snr=10, fwhm=0.05, ring_index=0)

        image, _ = _make_perfect_lattice(N, 0.1, 0.5, 0)
        phase1 = compute_gpa_phase(image, g1, 0.1, grid)
        phase2 = compute_gpa_phase(image, g2, 0.1, grid)

        disp = compute_displacement_field(phase1, phase2)
        assert disp is None

    def test_smooth_displacement(self):
        """B5: smoothing should reduce noise in displacement."""
        from src.pipeline_config import DisplacementField
        rng = np.random.default_rng(42)
        noisy_ux = rng.normal(0, 0.1, (64, 64))
        noisy_uy = rng.normal(0, 0.1, (64, 64))
        disp = DisplacementField(ux=noisy_ux, uy=noisy_uy)

        smoothed = smooth_displacement(disp, sigma=2.0)
        assert np.std(smoothed.ux) < np.std(noisy_ux)
        assert np.std(smoothed.uy) < np.std(noisy_uy)

    def test_strain_field_computation(self):
        """B5: strain should be computed from smoothed displacement."""
        from src.pipeline_config import DisplacementField
        N = 64
        px = 0.1

        # Linear displacement → constant strain
        y = np.arange(N).reshape(-1, 1) * px
        x = np.arange(N).reshape(1, -1) * px
        strain_xx = 0.02
        ux = np.broadcast_to(strain_xx * x, (N, N)).copy()  # du_x/dx = strain_xx
        uy = np.zeros((N, N))

        disp = DisplacementField(ux=ux, uy=uy)
        strain = compute_strain_field(disp, px)

        # exx should be approximately strain_xx in interior
        interior = strain.exx[10:-10, 10:-10]
        mean_exx = np.mean(interior)
        assert abs(mean_exx - strain_xx) < 0.005, \
            f"Expected exx ≈ {strain_xx}, got {mean_exx:.4f}"


class TestCollinearityAngleCheck:
    """Angle-based collinearity rejection for g-vectors."""

    def test_collinear_rejection_by_angle(self):
        """G-vectors 5° apart should be rejected as collinear."""
        N = 64
        grid = FFTGrid(N, N, 0.1)
        # Two g-vectors 5° apart
        g1 = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)
        g2 = GVector(gx=2.0 * np.cos(np.radians(5)),
                      gy=2.0 * np.sin(np.radians(5)),
                      magnitude=2.0, angle_deg=5.0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)

        image, _ = _make_perfect_lattice(N, 0.1, 0.5, 0)
        phase1 = compute_gpa_phase(image, g1, 0.1, grid)
        phase2 = compute_gpa_phase(image, g2, 0.1, grid)

        disp = compute_displacement_field(phase1, phase2, min_gvector_angle_deg=15.0)
        assert disp is None

    def test_antiparallel_rejection(self):
        """G-vectors 175° apart (near-antiparallel) should be rejected."""
        N = 64
        grid = FFTGrid(N, N, 0.1)
        g1 = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)
        g2 = GVector(gx=-2.0 * np.cos(np.radians(5)),
                      gy=-2.0 * np.sin(np.radians(5)),
                      magnitude=2.0, angle_deg=175.0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)

        image, _ = _make_perfect_lattice(N, 0.1, 0.5, 0)
        phase1 = compute_gpa_phase(image, g1, 0.1, grid)
        phase2 = compute_gpa_phase(image, g2, 0.1, grid)

        disp = compute_displacement_field(phase1, phase2, min_gvector_angle_deg=15.0)
        assert disp is None

    def test_non_collinear_acceptance(self):
        """G-vectors 45° apart should succeed."""
        N = 64
        grid = FFTGrid(N, N, 0.1)
        g1 = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)
        g2 = GVector(gx=2.0 * np.cos(np.radians(45)),
                      gy=2.0 * np.sin(np.radians(45)),
                      magnitude=2.0, angle_deg=45.0,
                      d_spacing=0.5, snr=10, fwhm=0.05, ring_index=0)

        image, _ = _make_perfect_lattice(N, 0.1, 0.5, 0)
        phase1 = compute_gpa_phase(image, g1, 0.1, grid)
        phase2 = compute_gpa_phase(image, g2, 0.1, grid)

        disp = compute_displacement_field(phase1, phase2, min_gvector_angle_deg=15.0)
        assert disp is not None


class TestGPASchemaConsistency:
    """C12: Both GPA modes should produce identical GPAResult schema."""

    def test_gpa_result_fields(self):
        from src.pipeline_config import GPAResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(GPAResult)}
        expected = {"mode", "phases", "displacement", "strain",
                    "reference_region", "mode_decision", "qc", "diagnostics"}
        assert fields == expected
