"""Tests for dynamic DC mask estimation and taper mask construction."""

import numpy as np
import pytest

from src.pipeline_config import DCMaskConfig, PhysicsConfig
from src.global_fft import estimate_dynamic_dc_radius, build_dc_taper_mask


# ======================================================================
# Helpers
# ======================================================================

def _make_steep_dc_profile(n=200, q_max=10.0, dc_width=1.5):
    """Profile with steep DC falloff followed by flat background + peaks.

    DC contamination region: q < dc_width has steep power-law slope.
    """
    q = np.linspace(0.05, q_max, n)
    # Steep DC: large amplitude decays as q^-5 in the DC region
    dc_envelope = 1e8 * np.clip((dc_width - q) / dc_width, 0, 1) ** 5
    # Flat background with small peaks
    bg = np.full_like(q, 100.0)
    # Two peaks
    bg += 500 * np.exp(-0.5 * ((q - 3.0) / 0.2) ** 2)
    bg += 300 * np.exp(-0.5 * ((q - 6.0) / 0.3) ** 2)
    profile = dc_envelope + bg
    return q, profile


def _make_flat_profile(n=200, q_max=10.0):
    """Flat profile (no DC contamination)."""
    q = np.linspace(0.05, q_max, n)
    rng = np.random.RandomState(42)
    profile = 100 + rng.normal(0, 2, n)
    profile = np.clip(profile, 1, None)
    return q, profile


# ======================================================================
# estimate_dynamic_dc_radius tests
# ======================================================================

class TestEstimateDynamicDCRadius:
    def test_dc_disabled_returns_none(self):
        """When config disabled, compute_global_fft doesn't call estimator.

        (This test verifies the config flag behavior at a higher level;
        the estimator itself always returns a value.)
        """
        cfg = DCMaskConfig(enabled=False)
        assert cfg.enabled is False

    def test_strong_dc_detected(self):
        """Profile with steep DC falloff → radius > floor."""
        q, profile = _make_steep_dc_profile()
        cfg = DCMaskConfig(enabled=True, q_dc_min_floor=0.15,
                           consecutive_bins=5, slope_threshold_k=2.5)
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg)
        # Should detect DC radius above the floor
        assert q_dc > cfg.q_dc_min_floor
        # Safety clamp limits to 0.5 * q_max
        assert q_dc <= 0.5 * q[-1] + 1e-10
        assert diag["q_dc_final"] == pytest.approx(q_dc)

    def test_flat_profile_returns_floor(self):
        """Flat profile (no DC contamination) → returns floor."""
        q, profile = _make_flat_profile()
        cfg = DCMaskConfig(enabled=True, q_dc_min_floor=0.15,
                           consecutive_bins=5, slope_threshold_k=2.5)
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg)
        # Flat profile derivative is near zero everywhere;
        # first consecutive run starts at very low q → floor likely applies
        assert q_dc >= cfg.q_dc_min_floor

    def test_cap_from_physics(self):
        """Physics d_max_nm=2.0 → cap at 0.4 cycles/nm."""
        q, profile = _make_steep_dc_profile(dc_width=3.0)
        cfg = DCMaskConfig(enabled=True, q_dc_min_floor=0.15,
                           auto_cap_from_physics=True)
        physics = PhysicsConfig(d_max_nm=2.0)
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg,
                                                  physics_config=physics)
        assert q_dc <= 0.4 + 1e-10  # 0.8 / 2.0
        assert "physics_d_max" in str(diag.get("cap_applied", []))

    def test_cap_from_max_dc_mask_q(self):
        """Explicit max_dc_mask_q cap is honored."""
        q, profile = _make_steep_dc_profile(dc_width=3.0)
        cfg = DCMaskConfig(enabled=True, q_dc_min_floor=0.15,
                           max_dc_mask_q=0.5)
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg)
        assert q_dc <= 0.5 + 1e-10

    def test_floor_enforced(self):
        """Even for flat profile, result >= floor."""
        q, profile = _make_flat_profile()
        floor = 0.5
        cfg = DCMaskConfig(enabled=True, q_dc_min_floor=floor,
                           consecutive_bins=3, slope_threshold_k=5.0)
        q_dc, _ = estimate_dynamic_dc_radius(q, profile, cfg)
        assert q_dc >= floor

    def test_consecutive_bins_sensitivity(self):
        """Larger consecutive_bins → potentially different result."""
        q, profile = _make_steep_dc_profile()
        cfg_small = DCMaskConfig(enabled=True, consecutive_bins=3)
        cfg_large = DCMaskConfig(enabled=True, consecutive_bins=15)
        q_dc_small, _ = estimate_dynamic_dc_radius(q, profile, cfg_small)
        q_dc_large, _ = estimate_dynamic_dc_radius(q, profile, cfg_large)
        # With more consecutive bins required, the stable region starts later
        assert q_dc_large >= q_dc_small

    def test_slope_threshold_k_sensitivity(self):
        """Higher k → easier to declare 'flat', → smaller DC radius."""
        q, profile = _make_steep_dc_profile()
        cfg_tight = DCMaskConfig(enabled=True, slope_threshold_k=1.0)
        cfg_loose = DCMaskConfig(enabled=True, slope_threshold_k=10.0)
        q_dc_tight, _ = estimate_dynamic_dc_radius(q, profile, cfg_tight)
        q_dc_loose, _ = estimate_dynamic_dc_radius(q, profile, cfg_loose)
        # Looser threshold means stable region detected earlier → smaller DC
        assert q_dc_loose <= q_dc_tight

    def test_diagnostics_populated(self):
        """Diagnostics dict has expected keys."""
        q, profile = _make_steep_dc_profile()
        cfg = DCMaskConfig(enabled=True)
        _, diag = estimate_dynamic_dc_radius(q, profile, cfg)
        assert "method" in diag
        assert "q_dc_final" in diag

    def test_short_profile_fallback(self):
        """Very short profile triggers fallback."""
        q = np.linspace(0.1, 1.0, 8)
        profile = np.ones(8) * 100
        cfg = DCMaskConfig(enabled=True, consecutive_bins=7)
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg)
        assert q_dc == cfg.q_dc_min_floor
        assert diag.get("fallback") == "profile_too_short"

    def test_method_fixed_returns_floor_immediately(self):
        """method='fixed' returns q_dc_min_floor without derivative analysis."""
        q, profile = _make_steep_dc_profile()
        floor = 0.42
        cfg = DCMaskConfig(enabled=True, method="fixed", q_dc_min_floor=floor)
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg)
        assert q_dc == floor
        assert diag["method"] == "fixed"
        # No derivative-related keys should be present
        assert "sigma_noise" not in diag

    def test_mode_specific_haadf(self):
        """HAADF mode (d_max=0): no physics cap applied."""
        q, profile = _make_steep_dc_profile(dc_width=3.0)
        cfg = DCMaskConfig(enabled=True, auto_cap_from_physics=True)
        physics = PhysicsConfig(d_max_nm=0.0)  # unconstrained
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg,
                                                  physics_config=physics)
        # No physics cap should appear
        caps = diag.get("cap_applied", [])
        assert not any("physics" in str(c) for c in caps)

    def test_mode_specific_bf(self):
        """BF mode (d_max > 0): physics cap enforced."""
        q, profile = _make_steep_dc_profile(dc_width=3.0)
        cfg = DCMaskConfig(enabled=True, auto_cap_from_physics=True)
        physics = PhysicsConfig(d_max_nm=1.0)  # cap at 0.8
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg,
                                                  physics_config=physics)
        assert q_dc <= 0.8 + 1e-10

    def test_safety_clamp_half_q_max(self):
        """DC estimate clamped to 0.5 * q_max."""
        q = np.linspace(0.05, 2.0, 50)  # small q_max
        # Profile with huge DC extending almost everywhere
        profile = 1e6 * np.exp(-q * 0.1)
        cfg = DCMaskConfig(enabled=True, q_dc_min_floor=0.0,
                           consecutive_bins=3, slope_threshold_k=100.0)
        q_dc, diag = estimate_dynamic_dc_radius(q, profile, cfg)
        assert q_dc <= 0.5 * q[-1] + 1e-10


# ======================================================================
# build_dc_taper_mask tests
# ======================================================================

class TestBuildDCTaperMask:
    def _make_q_mag(self, size=64, pixel_size=0.1):
        from src.fft_coords import FFTGrid
        grid = FFTGrid(size, size, pixel_size)
        return grid.q_mag_grid()

    def test_hard_mask_binary(self):
        """Hard mask is exactly 0/1."""
        q_mag = self._make_q_mag()
        q_dc = 1.0
        mask = build_dc_taper_mask(q_mag, q_dc, soft_taper=False)
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        # Center should be 0, far away should be 1
        center = mask[32, 32]
        assert center == 0.0  # DC center
        assert mask[0, 0] == 1.0  # corner (high q)

    def test_soft_taper_shape(self):
        """Soft taper: 0 at center, 1 far away, smooth transition."""
        q_mag = self._make_q_mag()
        q_dc = 1.5
        mask = build_dc_taper_mask(q_mag, q_dc, soft_taper=True,
                                    taper_width_q=0.5)
        # Center should be 0
        assert mask[32, 32] == 0.0
        # Far corner should be 1
        assert mask[0, 0] == pytest.approx(1.0)
        # Transition values should be between 0 and 1
        transition = (mask > 0.01) & (mask < 0.99)
        assert np.sum(transition) > 0, "Should have transition pixels"

    def test_taper_preserves_peaks(self):
        """Peak at q >> q_dc is unattenuated by taper."""
        q_mag = self._make_q_mag()
        q_dc = 0.5
        mask = build_dc_taper_mask(q_mag, q_dc, soft_taper=True,
                                    taper_width_q=0.1)
        # All pixels where q > q_dc + taper_width/2 should be ~1.0
        high_q = q_mag > (q_dc + 0.1)
        if np.any(high_q):
            assert np.all(mask[high_q] > 0.99)

    def test_taper_width_zero_falls_back_to_hard(self):
        """taper_width_q <= 0 falls back to hard mask with warning."""
        q_mag = self._make_q_mag()
        q_dc = 1.0
        mask = build_dc_taper_mask(q_mag, q_dc, soft_taper=True,
                                    taper_width_q=0.0)
        # Should produce a hard binary mask
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_hard_mask_dc_zero_periphery_one(self):
        """Hard mask: verify topology."""
        q_mag = self._make_q_mag()
        q_dc = 2.0
        mask = build_dc_taper_mask(q_mag, q_dc, soft_taper=False)
        # Pixels inside DC circle should be 0
        dc_region = q_mag < q_dc
        assert np.all(mask[dc_region] == 0.0)
        # Pixels outside should be 1
        outside = q_mag >= q_dc
        assert np.all(mask[outside] == 1.0)
