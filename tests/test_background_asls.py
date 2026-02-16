"""Tests for AsLS background fitting and background dispatch."""

import numpy as np
import pytest

from src.pipeline_config import GlobalFFTConfig
from src.global_fft import (
    _fit_background, _fit_background_asls, fit_background_dispatch,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_powerlaw_profile(n=200, q_max=10.0, a=1e6, gamma=3.0, seed=42):
    """Power-law radial profile  a * q^{-gamma}  plus noise."""
    rng = np.random.RandomState(seed)
    q = np.linspace(0.05, q_max, n)
    profile = a * q ** (-gamma) + rng.normal(0, 5, n)
    profile = np.clip(profile, 1.0, None)
    return q, profile


def _make_profile_with_peaks(n=200, q_max=10.0, seed=42):
    """Power-law profile with two Gaussian peaks."""
    q, profile = _make_powerlaw_profile(n, q_max, seed=seed)
    # Add peaks
    peak1_q, peak1_amp, peak1_sigma = 2.5, 5000, 0.15
    peak2_q, peak2_amp, peak2_sigma = 5.0, 2000, 0.2
    profile += peak1_amp * np.exp(-0.5 * ((q - peak1_q) / peak1_sigma) ** 2)
    profile += peak2_amp * np.exp(-0.5 * ((q - peak2_q) / peak2_sigma) ** 2)
    return q, profile


# ======================================================================
# _fit_background_asls tests
# ======================================================================

class TestFitBackgroundAsLS:
    def test_asls_tracks_powerlaw(self):
        """AsLS baseline follows power-law envelope, not peaks."""
        q, profile = _make_profile_with_peaks()
        bg, model = _fit_background_asls(q, profile, lam=1e6, p=0.001, n_iter=10)
        assert model["type"] == "asls"
        # Baseline should be below peaks (positive corrected at peak locations)
        corrected = profile - bg
        peak1_idx = np.argmin(np.abs(q - 2.5))
        peak2_idx = np.argmin(np.abs(q - 5.0))
        assert corrected[peak1_idx] > 0, "Baseline should be below peak 1"
        assert corrected[peak2_idx] > 0, "Baseline should be below peak 2"
        # Baseline should track smooth envelope (not oscillate wildly)
        # Check that median of corrected away from peaks is near zero
        mask = np.ones(len(q), dtype=bool)
        mask[peak1_idx - 5:peak1_idx + 5] = False
        mask[peak2_idx - 5:peak2_idx + 5] = False
        mask[:10] = False  # skip DC region
        med_residual = np.median(np.abs(corrected[mask]))
        # Profile amplitudes are ~1e6 at low-q; residual < 2000 is < 0.2%
        assert med_residual < 2000, f"Residual too large: {med_residual}"

    def test_asls_vs_polynomial_similar(self):
        """AsLS and polynomial give similar baselines on clean power-law data."""
        q, profile = _make_powerlaw_profile()
        bg_asls, _ = _fit_background_asls(q, profile, lam=1e6, p=0.01, n_iter=15)
        bg_poly, _ = _fit_background(q, profile, degree=4)
        # Compare in the fit region (skip DC)
        fit_region = q >= 0.5
        mad_diff = np.median(np.abs(bg_asls[fit_region] - bg_poly[fit_region]))
        mad_ref = np.median(np.abs(bg_poly[fit_region]))
        relative_diff = mad_diff / (mad_ref + 1e-10)
        # Should be within ~30% for clean data (generous tolerance)
        assert relative_diff < 0.5, f"Relative MAD diff: {relative_diff:.3f}"

    def test_asls_excludes_dc_region(self):
        """Excluded DC region filled with raw profile values."""
        q, profile = _make_powerlaw_profile()
        q_fit_min = 1.0
        bg, _ = _fit_background_asls(q, profile, q_fit_min=q_fit_min)
        excluded = q < q_fit_min
        np.testing.assert_array_almost_equal(bg[excluded], profile[excluded])

    def test_asls_excludes_dc_effective_q_min(self):
        """effective_q_min also excludes DC region."""
        q, profile = _make_powerlaw_profile()
        eff_q_min = 0.8
        bg, _ = _fit_background_asls(q, profile, effective_q_min=eff_q_min)
        excluded = q < eff_q_min
        np.testing.assert_array_almost_equal(bg[excluded], profile[excluded])

    def test_asls_parameters_affect_result(self):
        """Different lambda/p values produce different baselines."""
        q, profile = _make_profile_with_peaks()
        bg_smooth, _ = _fit_background_asls(q, profile, lam=1e8, p=0.001)
        bg_rough, _ = _fit_background_asls(q, profile, lam=1e3, p=0.001)
        # Higher lambda → smoother → different from lower lambda
        diff = np.max(np.abs(bg_smooth - bg_rough))
        assert diff > 10, f"Baselines should differ, max diff = {diff}"

    def test_asls_linear_domain(self):
        """AsLS in linear domain produces non-negative baseline."""
        q, profile = _make_powerlaw_profile()
        bg, model = _fit_background_asls(q, profile, domain="linear")
        assert model["domain"] == "linear"
        # Linear domain clips to >= 0
        fit_mask = q >= 0.05
        assert np.all(bg[fit_mask] >= 0), "Linear AsLS baseline should be non-negative"

    def test_asls_log_domain_default(self):
        """Default domain is log."""
        q, profile = _make_powerlaw_profile()
        bg, model = _fit_background_asls(q, profile)
        assert model["domain"] == "log"

    def test_asls_short_profile_returns_zeros(self):
        """Profile with too few valid bins returns zero background."""
        q = np.array([0.1, 0.2, 0.3])
        profile = np.array([100.0, 50.0, 25.0])
        bg, model = _fit_background_asls(q, profile, q_fit_min=0.25)
        # Only 2 bins above q_fit_min → too few for AsLS
        assert model["type"] == "asls"
        # Should return zeros (graceful fallback)
        assert np.sum(bg) == pytest.approx(0.0, abs=1e-10)


# ======================================================================
# fit_background_dispatch tests
# ======================================================================

class TestFitBackgroundDispatch:
    def test_dispatch_polynomial_default(self):
        """Default config routes to polynomial_robust."""
        config = GlobalFFTConfig()
        assert config.background_method == "polynomial_robust"
        q, profile = _make_powerlaw_profile()
        bg, model = fit_background_dispatch(q, profile, config)
        assert model["type"] == "poly"

    def test_dispatch_asls_configured(self):
        """AsLS config routes to AsLS."""
        config = GlobalFFTConfig(background_method="asls")
        q, profile = _make_powerlaw_profile()
        bg, model = fit_background_dispatch(q, profile, config)
        assert model["type"] == "asls"

    def test_dispatch_passes_effective_q_min(self):
        """effective_q_min is forwarded to both methods."""
        q, profile = _make_powerlaw_profile()
        eff_q = 1.0

        config_poly = GlobalFFTConfig(background_method="polynomial_robust")
        bg_poly, _ = fit_background_dispatch(q, profile, config_poly,
                                              effective_q_min=eff_q)
        excluded = q < eff_q
        np.testing.assert_array_almost_equal(bg_poly[excluded], profile[excluded])

        config_asls = GlobalFFTConfig(background_method="asls")
        bg_asls, _ = fit_background_dispatch(q, profile, config_asls,
                                              effective_q_min=eff_q)
        np.testing.assert_array_almost_equal(bg_asls[excluded], profile[excluded])

    def test_polynomial_backward_compat(self):
        """Dispatch with defaults produces identical result to direct _fit_background call."""
        q, profile = _make_powerlaw_profile()
        config = GlobalFFTConfig()
        bg_dispatch, model_dispatch = fit_background_dispatch(q, profile, config)
        bg_degree = min(config.background_default_degree, config.background_max_degree)
        bg_direct, model_direct = _fit_background(
            q, profile, degree=bg_degree,
            q_fit_min=config.q_fit_min,
            bg_reweight_iterations=config.bg_reweight_iterations,
            bg_reweight_downweight=config.bg_reweight_downweight,
        )
        np.testing.assert_array_almost_equal(bg_dispatch, bg_direct)
        assert model_dispatch["type"] == model_direct["type"]

    def test_dispatch_unknown_method_raises(self):
        """Unknown background_method raises ValueError."""
        config = GlobalFFTConfig(background_method="unknown_method")
        q, profile = _make_powerlaw_profile()
        with pytest.raises(ValueError, match="Unknown background_method"):
            fit_background_dispatch(q, profile, config)
