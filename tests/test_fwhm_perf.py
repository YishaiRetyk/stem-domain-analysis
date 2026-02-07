"""Tests for FWHM performance optimizations.

Covers:
- Cached 11×11 grid objects are reused (B1)
- Proxy width is computed correctly (C1)
- curve_fit gating respects SNR threshold and per-tile cap (C1)
- Output stability on a fixed test tile (F)
"""

import numpy as np
import pytest
from src.fft_coords import FFTGrid
from src.pipeline_config import (
    TilePeak, TilePeakSet, TierConfig, PeakGateConfig, FWHMConfig, PeakFWHM,
)
from src.fft_peak_detection import (
    _PATCH_R, _PATCH_SIZE, _X11, _Y11, _DIST11, _OUTER_RING11, _RADIAL_MASKS11,
    measure_peak_fwhm, measure_peak_fwhm_proxy, classify_tile,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_power_with_peak(tile_size=256, pixel_size_nm=0.1,
                          q_peak=2.0, angle_deg=30.0, amplitude=500):
    """Create a synthetic power spectrum with known peak + antipodal."""
    grid = FFTGrid(tile_size, tile_size, pixel_size_nm)
    rng = np.random.default_rng(42)
    power = rng.exponential(1.0, (tile_size, tile_size))

    qx = q_peak * np.cos(np.radians(angle_deg))
    qy = q_peak * np.sin(np.radians(angle_deg))
    px_x, px_y = grid.q_to_px(qx, qy)

    y, x = np.mgrid[:tile_size, :tile_size]
    power += amplitude * np.exp(-0.5 * ((x - px_x)**2 + (y - px_y)**2) / 4.0)

    apx, apy = grid.q_to_px(-qx, -qy)
    power += amplitude * np.exp(-0.5 * ((x - apx)**2 + (y - apy)**2) / 4.0)

    peak = TilePeak(qx=qx, qy=qy, q_mag=q_peak, d_spacing=1.0/q_peak,
                    angle_deg=angle_deg,
                    intensity=float(power[int(round(px_y)), int(round(px_x))]),
                    fwhm=0.1)
    anti = TilePeak(qx=-qx, qy=-qy, q_mag=q_peak, d_spacing=1.0/q_peak,
                    angle_deg=angle_deg + 180,
                    intensity=float(power[int(round(apy)), int(round(apx))]),
                    fwhm=0.1)
    return power, [peak, anti], grid


def _make_many_peaks(n_peaks=12, tile_size=256, pixel_size_nm=0.1):
    """Create a tile with many peaks at different angles."""
    grid = FFTGrid(tile_size, tile_size, pixel_size_nm)
    rng = np.random.default_rng(99)
    power = rng.exponential(1.0, (tile_size, tile_size))
    y, x = np.mgrid[:tile_size, :tile_size]
    peaks = []

    for i in range(n_peaks):
        angle = i * 360.0 / n_peaks
        q = 2.0
        qx_p = q * np.cos(np.radians(angle))
        qy_p = q * np.sin(np.radians(angle))
        px_x, px_y = grid.q_to_px(qx_p, qy_p)

        amp = 600 if i < 4 else 30  # first 4 are high-SNR
        power += amp * np.exp(-0.5 * ((x - px_x)**2 + (y - px_y)**2) / 4.0)

        peaks.append(TilePeak(
            qx=qx_p, qy=qy_p, q_mag=q, d_spacing=1.0/q,
            angle_deg=angle, intensity=float(amp), fwhm=0.1,
        ))

    return power, peaks, grid


# ======================================================================
# B1: Cached grid constants
# ======================================================================

class TestCachedGrids:
    """Verify module-level cached grids are correct and reused."""

    def test_patch_size(self):
        assert _PATCH_R == 5
        assert _PATCH_SIZE == 11

    def test_grid_shapes(self):
        assert _X11.shape == (11, 11)
        assert _Y11.shape == (11, 11)
        assert _DIST11.shape == (11, 11)
        assert _OUTER_RING11.shape == (11, 11)

    def test_grid_identity(self):
        """Repeated access returns the same object (no re-allocation)."""
        from src import fft_peak_detection as mod
        assert mod._X11 is _X11
        assert mod._Y11 is _Y11
        assert mod._DIST11 is _DIST11

    def test_dist_center_is_zero(self):
        assert _DIST11[5, 5] == 0.0

    def test_outer_ring_excludes_center(self):
        assert not _OUTER_RING11[5, 5]
        # Corners should be in outer ring
        assert _OUTER_RING11[0, 0]

    def test_radial_masks_cover_patch(self):
        assert len(_RADIAL_MASKS11) == _PATCH_R + 1
        # Union of all masks should cover most of the patch
        union = np.zeros((11, 11), dtype=bool)
        for m in _RADIAL_MASKS11:
            union |= m
        # Not all pixels covered (gap at r >= R+1), but center is
        assert union[5, 5]


# ======================================================================
# C1: Proxy width
# ======================================================================

class TestProxyWidth:
    """Verify moment-based proxy produces valid FWHM estimates."""

    def test_proxy_returns_valid(self):
        power, peaks, grid = _make_power_with_peak()
        result = measure_peak_fwhm_proxy(power, peaks[0], grid)
        assert result.fwhm_valid
        assert result.method == "moment_proxy"
        assert result.fwhm_q > 0

    def test_proxy_edge_peak_fails(self):
        grid = FFTGrid(64, 64, 0.1)
        power = np.ones((64, 64))
        qx, qy = grid.px_to_q(1, 1)
        peak = TilePeak(qx=qx, qy=qy, q_mag=np.sqrt(qx**2 + qy**2),
                        d_spacing=0, angle_deg=0, intensity=0, fwhm=0)
        result = measure_peak_fwhm_proxy(power, peak, grid)
        assert not result.fwhm_valid

    def test_proxy_agrees_with_fit_order_of_magnitude(self):
        """Proxy and curve_fit should agree within a factor of ~3."""
        power, peaks, grid = _make_power_with_peak()
        proxy = measure_peak_fwhm_proxy(power, peaks[0], grid)
        fit = measure_peak_fwhm(power, peaks[0], grid, maxfev=2000)
        if proxy.fwhm_valid and fit.fwhm_valid:
            ratio = proxy.fwhm_q / fit.fwhm_q
            assert 0.2 < ratio < 5.0, f"proxy/fit ratio = {ratio}"


# ======================================================================
# C1: curve_fit gating
# ======================================================================

class TestCurveFitGating:
    """Verify curve_fit only runs for qualifying peaks."""

    def test_proxy_only_mode(self):
        """proxy_only mode should never produce gaussian_2d method."""
        power, peaks, grid = _make_power_with_peak()
        ps = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                         power_spectrum=power)
        cfg = FWHMConfig(method="proxy_only")
        tc = classify_tile(ps, grid, fwhm_config=cfg)
        for pm in tc.peaks:
            assert pm["fwhm_method"] != "gaussian_2d"

    def test_curve_fit_mode(self):
        """curve_fit mode should attempt fitting for all peaks."""
        power, peaks, grid = _make_power_with_peak()
        ps = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                         power_spectrum=power)
        cfg = FWHMConfig(method="curve_fit")
        tc = classify_tile(ps, grid, fwhm_config=cfg)
        methods = [pm["fwhm_method"] for pm in tc.peaks]
        # At least one should use gaussian_2d or wedge_fallback (not proxy)
        assert all(m != "moment_proxy" for m in methods)

    def test_max_per_tile_cap(self):
        """Only max_per_tile peaks should use curve_fit in auto mode."""
        power, peaks, grid = _make_many_peaks(n_peaks=12)
        ps = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                         power_spectrum=power)
        cfg = FWHMConfig(method="auto", max_per_tile=2, min_snr_for_fit=0.0)
        tc = classify_tile(ps, grid, fwhm_config=cfg)
        n_fit = sum(1 for pm in tc.peaks
                    if pm["fwhm_method"] in ("gaussian_2d", "wedge_fallback"))
        assert n_fit <= 2, f"Expected ≤2 fits, got {n_fit}"

    def test_snr_threshold_gates_fit(self):
        """Peaks below min_snr_for_fit should not trigger curve_fit."""
        power, peaks, grid = _make_many_peaks(n_peaks=12)
        ps = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                         power_spectrum=power)
        # Set threshold very high so no peak qualifies
        cfg = FWHMConfig(method="auto", min_snr_for_fit=9999.0, max_per_tile=10)
        tc = classify_tile(ps, grid, fwhm_config=cfg)
        n_fit = sum(1 for pm in tc.peaks
                    if pm["fwhm_method"] in ("gaussian_2d", "wedge_fallback"))
        assert n_fit == 0, f"Expected 0 fits, got {n_fit}"

    def test_disabled_fwhm(self):
        """enabled=False should skip curve_fit entirely."""
        power, peaks, grid = _make_power_with_peak()
        ps = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                         power_spectrum=power)
        cfg = FWHMConfig(enabled=False)
        tc = classify_tile(ps, grid, fwhm_config=cfg)
        for pm in tc.peaks:
            assert pm["fwhm_method"] != "gaussian_2d"


# ======================================================================
# F: Output stability on fixed test tile
# ======================================================================

class TestOutputStability:
    """Tier classification should be stable with default FWHMConfig."""

    def test_tier_unchanged(self):
        """A strong-peak tile should still be classified A or B."""
        power, peaks, grid = _make_power_with_peak()
        ps = TilePeakSet(peaks=peaks, tile_row=0, tile_col=0,
                         power_spectrum=power)
        tc = classify_tile(ps, grid)
        assert tc.tier in ("A", "B")
        assert tc.best_snr > 0

    def test_snr_unchanged(self):
        """SNR values should not change -- FWHM policy doesn't affect SNR."""
        power, peaks, grid = _make_power_with_peak()
        ps1 = TilePeakSet(peaks=list(peaks), tile_row=0, tile_col=0,
                          power_spectrum=power)
        ps2 = TilePeakSet(peaks=[TilePeak(qx=p.qx, qy=p.qy, q_mag=p.q_mag,
                                          d_spacing=p.d_spacing,
                                          angle_deg=p.angle_deg,
                                          intensity=p.intensity, fwhm=p.fwhm)
                                 for p in peaks],
                          tile_row=0, tile_col=0, power_spectrum=power)

        tc_auto = classify_tile(ps1, grid, fwhm_config=FWHMConfig(method="auto"))
        tc_fit = classify_tile(ps2, grid, fwhm_config=FWHMConfig(method="curve_fit"))

        # SNR should be identical (computed before FWHM)
        assert abs(tc_auto.best_snr - tc_fit.best_snr) < 1e-10

    def test_maxfev_configurable(self):
        """measure_peak_fwhm should accept and use custom maxfev."""
        power, peaks, grid = _make_power_with_peak()
        # Very low maxfev → fit likely fails → fallback
        result = measure_peak_fwhm(power, peaks[0], grid, maxfev=1)
        assert result.fwhm_valid  # fallback should still produce valid
        assert result.method in ("wedge_fallback", "gaussian_2d")
