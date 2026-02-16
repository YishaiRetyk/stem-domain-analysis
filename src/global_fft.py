"""
Global FFT + Multi-Ring G-Vector Extraction (WS2, B4).

Full-image FFT, radial profile, background fit, multi-ring g-vector extraction
with polar resampling, antipodal pairing, harmonic de-duplication.

Gate G4: >= 1 peak with SNR >= 3.0.
"""

import logging
import numpy as np
from scipy.signal import windows, find_peaks
from scipy.ndimage import map_coordinates
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import List, Optional, Tuple, TYPE_CHECKING

from src.fft_coords import FFTGrid
from src.pipeline_config import (
    GVector, GlobalPeak, GlobalFFTResult, GlobalFFTConfig,
    DCMaskConfig, PhysicsConfig,
)
from src.gates import evaluate_gate

if TYPE_CHECKING:
    from src.gpu_backend import DeviceContext

logger = logging.getLogger(__name__)


def _next_power_of_2(n: int) -> int:
    """Return the largest power of 2 <= n."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _angular_distance_deg(a1: float, a2: float) -> float:
    """Smallest angular distance between two angles in degrees."""
    d = abs(a1 - a2) % 360
    return min(d, 360 - d)


def select_d_dom(peaks, *, config=None, d_min_nm=None, d_max_nm=None):
    """Select dominant d-spacing from peaks, filtering by physics bounds.

    Picks highest-SNR peak within [d_min_nm, d_max_nm]. Falls back to
    global highest-SNR if no candidate is within bounds.
    """
    if not peaks:
        return None

    # Extract bounds from config if available
    if config is not None:
        if d_max_nm is None and hasattr(config, 'd_max_nm') and config.d_max_nm is not None:
            d_max_nm = config.d_max_nm
        if d_min_nm is None and hasattr(config, 'd_min_nm') and config.d_min_nm is not None:
            d_min_nm = config.d_min_nm

    candidates = []
    excluded = []
    for pk in peaks:
        d = pk.d_spacing
        if d_max_nm is not None and d > d_max_nm:
            excluded.append(pk)
            continue
        if d_min_nm is not None and d < d_min_nm:
            excluded.append(pk)
            continue
        candidates.append(pk)

    if candidates:
        best = max(candidates, key=lambda pk: pk.snr)
        if excluded:
            for ex in excluded:
                logger.info("Excluded peak d=%.3f nm (SNR=%.1f) — outside bounds "
                            "[%s, %s]", ex.d_spacing, ex.snr,
                            d_min_nm or '...', d_max_nm or '...')
        logger.info("Selected d_dom=%.3f nm (SNR=%.1f) within bounds [%s, %s]",
                     best.d_spacing, best.snr,
                     d_min_nm or '...', d_max_nm or '...')
        return best.d_spacing

    # Fallback: no peaks within bounds — use global best
    best = max(peaks, key=lambda pk: pk.snr)
    logger.warning("No peaks within d bounds [%s, %s]. "
                   "Falling back to d_dom=%.3f nm (SNR=%.1f)",
                   d_min_nm or '...', d_max_nm or '...',
                   best.d_spacing, best.snr)
    return best.d_spacing


def compute_global_fft(image_fft: np.ndarray,
                       fft_grid: FFTGrid,
                       config: GlobalFFTConfig = None,
                       ctx: Optional["DeviceContext"] = None,
                       effective_q_min: float = 0.0,
                       dc_mask_config: DCMaskConfig = None,
                       physics_config: PhysicsConfig = None) -> GlobalFFTResult:
    """Run full-image FFT and extract g-vectors.

    Parameters
    ----------
    image_fft : np.ndarray
        Branch A (FFT-safe) preprocessed image.
    fft_grid : FFTGrid
        Canonical coordinate system for the full image.
    config : GlobalFFTConfig, optional
    dc_mask_config : DCMaskConfig, optional
        When provided and ``enabled=True``, estimate dynamic DC radius.
    physics_config : PhysicsConfig, optional
        Used for physics-based caps on dynamic DC radius.

    Returns
    -------
    GlobalFFTResult
    """
    if config is None:
        config = GlobalFFTConfig()

    H, W = image_fft.shape
    diagnostics = {"frequency_unit": "cycles/nm"}

    # Cap image size (centre crop to power-of-2)
    max_size = config.max_image_size
    crop_h = min(_next_power_of_2(H), max_size)
    crop_w = min(_next_power_of_2(W), max_size)
    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    cropped = image_fft[y0:y0 + crop_h, x0:x0 + crop_w]

    diagnostics["crop_size"] = [crop_h, crop_w]

    # Hann window
    window = np.outer(windows.hann(crop_h), windows.hann(crop_w))

    # FFT (GPU or CPU)
    if ctx is not None and ctx.using_gpu:
        cropped_d = ctx.to_device(cropped.astype(np.float64))
        window_d = ctx.to_device(window)
        windowed_d = cropped_d * window_d
        del cropped_d, window_d
        fft_result = ctx.fft2(windowed_d)
        del windowed_d
        fft_shifted = ctx.fftshift(fft_result)
        del fft_result
        power_d = ctx.xp.abs(fft_shifted) ** 2
        del fft_shifted
        power = ctx.to_host(power_d)
        del power_d
    else:
        windowed = cropped.astype(np.float64) * window
        fft_result = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft_result)
        power = np.abs(fft_shifted) ** 2

    # Build FFTGrid for the cropped image
    crop_grid = FFTGrid(crop_h, crop_w, fft_grid.pixel_size_nm)

    # Radial profile
    q_mag = crop_grid.q_mag_grid()
    r_px = np.sqrt(
        (np.arange(crop_w)[np.newaxis, :] - crop_grid.dc_x) ** 2 +
        (np.arange(crop_h)[:, np.newaxis] - crop_grid.dc_y) ** 2
    )
    r_int = r_px.astype(int)
    max_r = min(crop_grid.dc_x, crop_grid.dc_y, r_int.max())

    r_flat = r_int.ravel()
    valid = r_flat <= max_r
    radial_sum = np.bincount(r_flat[valid], weights=power.ravel()[valid],
                             minlength=max_r + 1)
    radial_count = np.bincount(r_flat[valid], minlength=max_r + 1).astype(float)
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count

    q_scale = crop_grid.qx_scale  # same as qy_scale for square crops
    q_values = np.arange(max_r + 1) * q_scale

    # Dynamic DC mask estimation (global only — tiles reuse this value)
    dynamic_dc_q = None
    if dc_mask_config is not None and dc_mask_config.enabled:
        dynamic_dc_q, dc_diag = estimate_dynamic_dc_radius(
            q_values, radial_profile, dc_mask_config,
            physics_config=physics_config,
        )
        diagnostics["dc_mask"] = dc_diag
        logger.info("Dynamic DC radius: %.3f cycles/nm", dynamic_dc_q)

    # Effective q_min for background fit: max of low-q exclusion and dynamic DC
    bg_effective_q_min = effective_q_min
    if dynamic_dc_q is not None:
        bg_effective_q_min = max(effective_q_min, dynamic_dc_q)

    # Background fit (dispatch to polynomial_robust or AsLS)
    background, baseline_model = fit_background_dispatch(
        q_values, radial_profile, config,
        effective_q_min=bg_effective_q_min,
    )
    corrected = radial_profile - background
    diagnostics["baseline_model"] = baseline_model

    # Noise floor
    neg_vals = corrected[corrected < 0]
    if len(neg_vals) > 10:
        noise_floor = float(np.std(neg_vals) * 1.4826)
    else:
        noise_floor = float(np.std(corrected[corrected < np.median(corrected)]))

    # Find radial peaks
    radial_peaks = _find_radial_peaks(q_values, corrected, noise_floor,
                                       min_snr=config.min_peak_snr,
                                       effective_q_min=effective_q_min,
                                       savgol_window_max=config.savgol_window_max,
                                       savgol_window_min=config.savgol_window_min,
                                       savgol_polyorder=config.savgol_polyorder,
                                       radial_peak_distance=config.radial_peak_distance,
                                       radial_peak_width=config.radial_peak_width)
    diagnostics["n_radial_peaks"] = len(radial_peaks)
    diagnostics["effective_q_min"] = effective_q_min

    # Background residual diagnostics
    peak_indices = [rp["index"] for rp in radial_peaks]
    bg_diag = _compute_background_diagnostics(
        q_values, radial_profile, background, corrected,
        config.q_fit_min, peak_indices,
    )
    diagnostics["background_diagnostics"] = bg_diag

    # Convert to GlobalPeak list
    peaks = []
    for p in radial_peaks:
        peaks.append(GlobalPeak(
            q_center=p["q_center"],
            q_fwhm=p["q_width"],
            d_spacing=p["d_spacing"],
            intensity=p["intensity"],
            prominence=p["prominence"],
            snr=p["snr"],
            index=p["index"],
        ))

    # Dominant d-spacing — filter by physics bounds if available
    d_dom = None
    if peaks:
        _d_min = None
        _d_max = None
        if physics_config is not None:
            if physics_config.d_min_nm > 0:
                _d_min = physics_config.d_min_nm
            if physics_config.d_max_nm > 0:
                _d_max = physics_config.d_max_nm
        d_dom = select_d_dom(peaks, d_min_nm=_d_min, d_max_nm=_d_max)

    # Multi-ring g-vector extraction (B4)
    g_vectors = extract_all_g_vectors(
        power, radial_peaks, crop_grid,
        top_k=config.max_g_vectors,
        config=config,
    )

    # Information limit (highest q with detectable signal)
    info_limit_q = None
    if peaks:
        info_limit_q = max(p.q_center + p.q_fwhm for p in peaks)

    # Gate G4
    best_peak_snr = max((p.snr for p in peaks), default=0)
    g4_result = evaluate_gate("G4", best_peak_snr)
    diagnostics["g4_passed"] = g4_result.passed
    diagnostics["g4_reason"] = g4_result.reason

    # FFT guidance strength classification
    n_peaks = len(peaks)
    if best_peak_snr >= config.strong_guidance_snr and n_peaks >= 2:
        fft_guidance = "strong"
    elif best_peak_snr >= config.min_peak_snr:
        fft_guidance = "weak"
    else:
        fft_guidance = "none"

    return GlobalFFTResult(
        power_spectrum=power,
        radial_profile=radial_profile,
        q_values=q_values,
        background=background,
        corrected_profile=corrected,
        noise_floor=noise_floor,
        peaks=peaks,
        g_vectors=g_vectors,
        d_dom=d_dom,
        information_limit_q=info_limit_q,
        diagnostics=diagnostics,
        fft_guidance_strength=fft_guidance,
        dynamic_dc_q=dynamic_dc_q,
    )


# ======================================================================
# Background fitting (adapted from peak_discovery.fit_background)
# ======================================================================

def _fit_background(q_values: np.ndarray, profile: np.ndarray,
                    degree: int = 6, exclude_dc: int = 5,
                    effective_q_min: float = 0.0,
                    q_fit_min: float = 0.0,
                    bg_reweight_iterations: int = 4,
                    bg_reweight_downweight: float = 0.1) -> tuple:
    """Iterative reweighted polynomial background fit in log space.

    Returns
    -------
    background : np.ndarray
        Background estimate, same length as *profile*.
    baseline_model : dict
        Serialisable description of the fitted model.
    """
    # Cap polynomial degree at 4
    degree = min(degree, 4)

    # Build validity mask: positive values, above effective_q_min, above q_fit_min
    if effective_q_min > 0 or q_fit_min > 0:
        actual_min = max(effective_q_min, q_fit_min)
        valid = (profile > 0) & (q_values >= actual_min)
    else:
        valid = (profile > 0) & (np.arange(len(profile)) >= exclude_dc)
    if np.sum(valid) < degree + 1:
        baseline_model = {
            "type": "poly", "degree": degree,
            "q_fit_min": float(q_fit_min), "coeffs": [],
        }
        return np.zeros_like(profile), baseline_model

    q_fit = q_values[valid]
    log_profile = np.log10(profile[valid] + 1e-10)
    weights = np.ones_like(log_profile)

    _bg_reweight_iterations = bg_reweight_iterations
    _bg_reweight_downweight = bg_reweight_downweight
    coeffs = None
    for _ in range(_bg_reweight_iterations):
        coeffs = np.polyfit(q_fit, log_profile, degree, w=weights)
        fit = np.polyval(coeffs, q_fit)
        residual = log_profile - fit
        mad = np.median(np.abs(residual - np.median(residual)))
        threshold = 2.0 * mad * 1.4826
        weights = np.where(residual > threshold, _bg_reweight_downweight, 1.0)

    background = np.zeros_like(profile)
    if coeffs is not None:
        background[valid] = 10 ** np.polyval(coeffs, q_values[valid])
    # Fill excluded region with raw profile values
    actual_excl = max(effective_q_min, q_fit_min)
    if actual_excl > 0:
        excluded = q_values < actual_excl
        background[excluded] = profile[excluded]
    else:
        background[:exclude_dc] = profile[:exclude_dc]

    baseline_model = {
        "type": "poly",
        "degree": int(degree),
        "q_fit_min": float(q_fit_min),
        "coeffs": coeffs.tolist() if coeffs is not None else [],
    }
    return background, baseline_model


def _fit_background_asls(q_values: np.ndarray, profile: np.ndarray,
                         lam: float = 1e6, p: float = 0.001, n_iter: int = 10,
                         effective_q_min: float = 0.0, q_fit_min: float = 0.0,
                         domain: str = "log") -> Tuple[np.ndarray, dict]:
    """Asymmetric Least Squares (AsLS) background fit.

    Operates in log or linear domain. All q values are in cycles/nm.

    Parameters
    ----------
    q_values, profile : np.ndarray
        Radial profile and corresponding q values (cycles/nm).
    lam : float
        Smoothness penalty (larger = smoother baseline).
    p : float
        Asymmetry weight (small p penalises positive residuals less,
        so the baseline stays below peaks).
    n_iter : int
        Number of reweighted iterations.
    effective_q_min, q_fit_min : float
        Bins below max(effective_q_min, q_fit_min) are excluded from fit.
    domain : str
        ``"log"`` (default) — fit in log10 space; ``"linear"`` — raw values.

    Returns
    -------
    background : np.ndarray
        Background estimate, same length as *profile*.
    baseline_model : dict
        Serialisable description ``{"type": "asls", ...}``.
    """
    actual_min = max(effective_q_min, q_fit_min)

    # Build fit mask
    if actual_min > 0:
        fit_mask = (profile > 0) & (q_values >= actual_min)
    else:
        # Skip first 5 DC bins (same heuristic as polynomial fitter)
        fit_mask = (profile > 0) & (np.arange(len(profile)) >= 5)

    n_fit = int(np.sum(fit_mask))
    if n_fit < 4:
        baseline_model = {
            "type": "asls", "lam": lam, "p": p, "n_iter": n_iter,
            "domain": domain, "q_fit_min": float(q_fit_min),
        }
        return np.zeros_like(profile), baseline_model

    y_raw = profile[fit_mask]

    # Transform to working domain
    if domain == "log":
        eps = max(1e-10, float(y_raw[y_raw > 0].min()) * 1e-3) if np.any(y_raw > 0) else 1e-10
        y = np.log10(y_raw + eps)
    else:
        y = y_raw.copy()

    m = len(y)
    # Second-difference penalty matrix  D^T D
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m)).tocsc()
    DTD = D.T.dot(D)

    w = np.ones(m)
    z = y.copy()
    for _ in range(n_iter):
        W = sparse.diags(w, 0, shape=(m, m))
        C = W + lam * DTD
        z = spsolve(C, w * y)
        # Asymmetric weights: p for y > z (peaks), 1-p for y <= z (below baseline)
        w = np.where(y > z, p, 1 - p)

    # Back-transform
    if domain == "log":
        fitted_values = 10.0 ** z
    else:
        # Linear domain: clip to >= 0 to avoid negative baselines
        fitted_values = np.clip(z, 0, None)

    # Assemble full-length background
    background = np.zeros_like(profile)
    background[fit_mask] = fitted_values

    # Fill excluded region with raw profile (same as polynomial fitter)
    if actual_min > 0:
        excluded = q_values < actual_min
        background[excluded] = profile[excluded]
    else:
        background[:5] = profile[:5]

    baseline_model = {
        "type": "asls", "lam": lam, "p": p, "n_iter": n_iter,
        "domain": domain, "q_fit_min": float(q_fit_min),
    }
    return background, baseline_model


def fit_background_dispatch(q_values: np.ndarray, profile: np.ndarray,
                            config: GlobalFFTConfig,
                            effective_q_min: float = 0.0) -> Tuple[np.ndarray, dict]:
    """Route background fitting to the configured method.

    Parameters
    ----------
    q_values, profile : np.ndarray
    config : GlobalFFTConfig
        Uses ``background_method``, polynomial params, and AsLS params.
    effective_q_min : float
        Low-q exclusion boundary (cycles/nm).

    Returns
    -------
    background, baseline_model
    """
    if config.background_method == "asls":
        return _fit_background_asls(
            q_values, profile,
            lam=config.asls_lambda,
            p=config.asls_p,
            n_iter=config.asls_n_iter,
            effective_q_min=effective_q_min,
            q_fit_min=config.q_fit_min,
            domain=config.asls_domain,
        )
    elif config.background_method == "polynomial_robust":
        bg_degree = min(config.background_default_degree, config.background_max_degree)
        return _fit_background(
            q_values, profile,
            degree=bg_degree,
            effective_q_min=effective_q_min,
            q_fit_min=config.q_fit_min,
            bg_reweight_iterations=config.bg_reweight_iterations,
            bg_reweight_downweight=config.bg_reweight_downweight,
        )
    else:
        raise ValueError(
            f"Unknown background_method: {config.background_method!r}. "
            f"Choose from 'polynomial_robust', 'asls'."
        )


def estimate_dynamic_dc_radius(
    q_values: np.ndarray,
    radial_profile: np.ndarray,
    dc_mask_config: DCMaskConfig,
    physics_config: PhysicsConfig = None,
) -> Tuple[float, dict]:
    """Estimate dynamic DC center mask radius from the radial profile.

    All q values are in cycles/nm (d = 1/q).  Runs on the GLOBAL
    radial profile only; tiles reuse the returned ``q_dc`` value.

    Algorithm
    ---------
    1. Work in log10 space.
    2. Smooth with Savitzky-Golay (auto-adjusted window).
    3. Compute derivative ``dJ/dq``.
    4. Estimate noise sigma from MAD of derivative in high-q region.
    5. Scan from low q: first run of *consecutive_bins* where
       ``|deriv| < k * sigma_noise``.
    6. Clamp to ``[q_dc_min_floor, 0.5 * q_max]`` with optional physics cap.

    Returns ``(q_dc, diagnostics_dict)``.
    """
    cfg = dc_mask_config
    diagnostics: dict = {"method": cfg.method}

    # Fixed mode: return floor immediately without derivative analysis
    if cfg.method == "fixed":
        diagnostics["q_dc_final"] = cfg.q_dc_min_floor
        return cfg.q_dc_min_floor, diagnostics

    q_max = float(q_values[-1])  # lock to profile last bin (user note #1)

    # Short-profile fallback: need at least 2 × consecutive_bins
    if len(q_values) < 2 * cfg.consecutive_bins:
        diagnostics["fallback"] = "profile_too_short"
        return cfg.q_dc_min_floor, diagnostics

    # 1. Log-space transform
    eps = 1e-10
    log_profile = np.log10(radial_profile + eps)

    # 2. Savitzky-Golay smoothing (auto-adjust window)
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d

    win = cfg.savgol_window
    # Clamp to <= len(profile) - 1 and ensure odd
    win = min(win, len(log_profile) - 1)
    if win % 2 == 0:
        win -= 1

    if win < 7:
        # Fallback to Gaussian smoothing when profile too short
        smoothed = gaussian_filter1d(log_profile, sigma=1.5)
        diagnostics["smoothing"] = "gaussian_fallback"
    else:
        polyorder = min(cfg.savgol_polyorder, win - 1)
        smoothed = savgol_filter(log_profile, window_length=win, polyorder=polyorder)
        diagnostics["smoothing"] = "savgol"
        diagnostics["savgol_window_used"] = win

    # 3. Derivative
    deriv = np.gradient(smoothed, q_values)

    # 4. Noise sigma from high-q region
    lo_frac = cfg.noise_q_range_lo
    hi_frac = cfg.noise_q_range_hi
    noise_lo = lo_frac * q_max
    noise_hi = hi_frac * q_max
    noise_mask = (q_values >= noise_lo) & (q_values <= noise_hi)

    # Auto-widen if fewer than 20 bins
    if int(np.sum(noise_mask)) < 20:
        lo_frac = 0.60
        hi_frac = 0.95
        noise_lo = lo_frac * q_max
        noise_hi = hi_frac * q_max
        noise_mask = (q_values >= noise_lo) & (q_values <= noise_hi)
        diagnostics["noise_region_widened"] = True

    noise_deriv = deriv[noise_mask]
    if len(noise_deriv) < 3:
        diagnostics["fallback"] = "no_noise_region"
        return cfg.q_dc_min_floor, diagnostics

    # MAD-clip outliers in noise region
    med_nd = np.median(noise_deriv)
    mad_nd = np.median(np.abs(noise_deriv - med_nd))
    clip_threshold = 3.0 * mad_nd * 1.4826
    clipped = noise_deriv[np.abs(noise_deriv - med_nd) <= clip_threshold]
    if len(clipped) < 3:
        clipped = noise_deriv
    sigma_noise = float(np.median(np.abs(clipped - np.median(clipped))) * 1.4826)
    if sigma_noise < 1e-15:
        sigma_noise = float(np.std(clipped))

    diagnostics["sigma_noise"] = sigma_noise
    diagnostics["n_noise_bins"] = int(np.sum(noise_mask))

    # 5. Scan from low q for consecutive bins below threshold
    threshold = cfg.slope_threshold_k * sigma_noise
    abs_deriv = np.abs(deriv)
    below = abs_deriv < threshold

    q_dc_dynamic = None
    run_count = 0
    for i in range(len(below)):
        if below[i]:
            run_count += 1
            if run_count >= cfg.consecutive_bins:
                # First bin of the stable run
                start_idx = i - cfg.consecutive_bins + 1
                q_dc_dynamic = float(q_values[start_idx])
                break
        else:
            run_count = 0

    if q_dc_dynamic is None:
        # No stable region found → return floor
        diagnostics["fallback"] = "no_stable_region"
        return cfg.q_dc_min_floor, diagnostics

    diagnostics["q_dc_raw"] = q_dc_dynamic

    # 6. Enforce floor
    q_dc = max(cfg.q_dc_min_floor, q_dc_dynamic)
    diagnostics["floor_applied"] = q_dc > q_dc_dynamic

    # 7. Caps
    cap_applied = []

    # Physics cap: 0.8 / d_max_nm
    if cfg.auto_cap_from_physics and physics_config is not None and physics_config.d_max_nm > 0:
        physics_cap = 0.8 / physics_config.d_max_nm
        if q_dc > physics_cap:
            q_dc = physics_cap
            cap_applied.append(f"physics_d_max={physics_config.d_max_nm}")
        diagnostics["physics_cap"] = physics_cap

    # Explicit max_dc_mask_q cap
    if cfg.max_dc_mask_q > 0 and q_dc > cfg.max_dc_mask_q:
        q_dc = cfg.max_dc_mask_q
        cap_applied.append(f"max_dc_mask_q={cfg.max_dc_mask_q}")

    # Global safety clamp: 0.5 * q_max (use same q_max as profile last bin)
    q_safety = 0.5 * q_max
    if q_dc > q_safety:
        q_dc = q_safety
        cap_applied.append(f"0.5*q_max={q_safety:.3f}")

    diagnostics["cap_applied"] = cap_applied
    diagnostics["q_dc_final"] = q_dc
    return q_dc, diagnostics


def build_dc_taper_mask(q_mag_grid: np.ndarray, q_dc: float,
                        soft_taper: bool = False,
                        taper_width_q: float = 0.05) -> np.ndarray:
    """Build a DC center mask in frequency space.

    Parameters
    ----------
    q_mag_grid : (H, W) array
        Magnitude of q at each pixel (cycles/nm).
    q_dc : float
        DC mask radius (cycles/nm).
    soft_taper : bool
        If True, use cosine taper; if False, hard binary mask.
    taper_width_q : float
        Width of cosine transition region (cycles/nm).

    Returns
    -------
    mask : (H, W) float array in [0, 1].
        0 inside DC, 1 outside. For hard mask: exactly 0/1.
    """
    if not soft_taper:
        return (q_mag_grid >= q_dc).astype(np.float64)

    # Guard: non-positive taper width → fall back to hard mask
    if taper_width_q <= 0:
        logger.warning("taper_width_q=%.4f <= 0; falling back to hard DC mask",
                        taper_width_q)
        return (q_mag_grid >= q_dc).astype(np.float64)

    # Cosine taper: 0 for q < q_dc - w/2, 1 for q > q_dc + w/2
    w = taper_width_q
    lo = q_dc - w / 2
    hi = q_dc + w / 2
    mask = np.ones_like(q_mag_grid, dtype=np.float64)
    mask[q_mag_grid < lo] = 0.0
    transition = (q_mag_grid >= lo) & (q_mag_grid <= hi)
    # Cosine ramp: 0 at lo → 1 at hi
    frac = (q_mag_grid[transition] - lo) / (w + 1e-15)
    mask[transition] = 0.5 * (1 - np.cos(np.pi * frac))
    return mask


def _compute_background_diagnostics(q_values, profile, background, corrected,
                                     q_fit_min, peak_positions):
    """Compute residual diagnostics for the background fit.

    Parameters
    ----------
    q_values : np.ndarray
    profile, background, corrected : np.ndarray
    q_fit_min : float
        Lower bound of the fit region (cycles/nm).
    peak_positions : list of int
        Indices into *q_values* of detected radial peaks.

    Returns
    -------
    dict with keys: median_abs_residual, mad_residual,
    neg_excursion_fraction_near_peaks.
    """
    # Residual in the fit region (linear space)
    fit_mask = q_values >= q_fit_min if q_fit_min > 0 else np.ones(len(q_values), dtype=bool)
    residual = (profile - background)[fit_mask]

    if len(residual) == 0:
        return {
            "median_abs_residual": 0.0,
            "mad_residual": 0.0,
            "neg_excursion_fraction_near_peaks": 0.0,
        }

    abs_res = np.abs(residual)
    median_abs = float(np.median(abs_res))
    mad_res = float(np.median(np.abs(abs_res - np.median(abs_res))) * 1.4826)

    # Negative excursion fraction near peaks (within +/-2 bins)
    near_peak_mask = np.zeros(len(q_values), dtype=bool)
    for idx in peak_positions:
        lo = max(0, idx - 2)
        hi = min(len(q_values), idx + 3)
        near_peak_mask[lo:hi] = True

    near_peak_residual = corrected[near_peak_mask]
    if len(near_peak_residual) > 0:
        neg_frac = float(np.sum(near_peak_residual < 0) / len(near_peak_residual))
    else:
        neg_frac = 0.0

    return {
        "median_abs_residual": median_abs,
        "mad_residual": mad_res,
        "neg_excursion_fraction_near_peaks": neg_frac,
    }


# ======================================================================
# Radial peak finding (adapted from peak_discovery.find_diffraction_peaks)
# ======================================================================

def _find_radial_peaks(q_values, corrected, noise_floor,
                       min_q=0.5, max_q=15.0, min_snr=2.0,
                       effective_q_min: float = 0.0,
                       savgol_window_max: int = 11,
                       savgol_window_min: int = 5,
                       savgol_polyorder: int = 2,
                       radial_peak_distance: int = 5,
                       radial_peak_width: int = 2):
    """Find peaks in the background-corrected radial profile."""
    from scipy.signal import savgol_filter

    actual_min_q = max(min_q, effective_q_min)
    valid_mask = (q_values >= actual_min_q) & (q_values <= max_q)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 10:
        return []

    wl = min(savgol_window_max, len(valid_idx) // 2 * 2 + 1)
    if wl < savgol_window_min:
        wl = savgol_window_min
    smooth = savgol_filter(corrected[valid_mask], window_length=wl, polyorder=savgol_polyorder)

    min_prominence = noise_floor * min_snr
    peaks_idx, props = find_peaks(smooth, prominence=max(min_prominence, 1e-10),
                                  width=radial_peak_width, distance=radial_peak_distance)

    results = []
    for i, li in enumerate(peaks_idx):
        gi = valid_idx[li]
        pq = q_values[gi]
        pi = corrected[gi]
        prom = props["prominences"][i]
        w = props["widths"][i] if "widths" in props else 3
        q_step = q_values[1] - q_values[0] if len(q_values) > 1 else 0.01
        results.append({
            "q_center": float(pq),
            "q_width": float(w * q_step),
            "d_spacing": float(1.0 / pq) if pq > 0 else 0,
            "intensity": float(pi),
            "prominence": float(prom),
            "snr": float(prom / noise_floor) if noise_floor > 0 else 0,
            "index": int(gi),
        })

    results.sort(key=lambda x: x["prominence"], reverse=True)
    return results


# ======================================================================
# Multi-ring g-vector extraction (B4)
# ======================================================================

def extract_all_g_vectors(power_spectrum: np.ndarray,
                          radial_peaks: list,
                          fft_grid: FFTGrid,
                          top_k: int = 6,
                          config: GlobalFFTConfig = None) -> List[GVector]:
    """Multi-ring g-vector extraction with harmonic de-duplication.

    For each radial peak ring:
    1. Polar resampling -> angular profile
    2. Angular peak detection
    3. Cartesian antipodal pairing
    Then cross-ring harmonic de-duplication and top-K selection.
    """
    if config is None:
        config = GlobalFFTConfig()

    q_width_expansion = config.q_width_expansion_frac
    harmonic_ratio_tol = config.harmonic_ratio_tol
    harmonic_angle_tol = config.harmonic_angle_tol_deg
    harmonic_snr_ratio = config.harmonic_snr_ratio
    nc_min_angle = config.non_collinear_min_angle_deg
    angular_prom_frac = config.angular_prominence_frac
    angular_peak_dist = config.angular_peak_distance

    all_candidates: List[GVector] = []

    for ring_idx, rp in enumerate(radial_peaks[:8]):  # cap at 8 rings
        q_center = rp["q_center"]
        q_fwhm = rp["q_width"]
        q_width = max(q_fwhm * 2, q_center * q_width_expansion)

        ring_gvecs = _extract_g_vectors_single_ring(
            power_spectrum, q_center, q_width, fft_grid, ring_idx,
            ring_snr=rp.get("snr", 0),
            ring_fwhm=rp.get("q_width", 0.1),
            angular_prominence_frac=angular_prom_frac,
            angular_peak_distance=angular_peak_dist,
        )
        all_candidates.extend(ring_gvecs)

    if not all_candidates:
        return []

    # Cross-ring harmonic de-duplication
    all_candidates.sort(key=lambda v: v.magnitude)
    deduplicated: List[GVector] = []

    for v in all_candidates:
        is_harmonic = False
        for i, existing in enumerate(deduplicated):
            ratio = v.magnitude / existing.magnitude
            nearest_int = round(ratio)
            if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < harmonic_ratio_tol:
                if _angular_distance_deg(v.angle_deg, existing.angle_deg) < harmonic_angle_tol:
                    if v.snr > existing.snr * harmonic_snr_ratio:
                        deduplicated[i] = v
                    is_harmonic = True
                    break
        if not is_harmonic:
            deduplicated.append(v)

    # Top-K non-collinear selection
    deduplicated.sort(key=lambda v: v.snr, reverse=True)
    selected: List[GVector] = []
    for v in deduplicated:
        if len(selected) >= top_k:
            break
        if all(_angular_distance_deg(v.angle_deg, s.angle_deg) > nc_min_angle for s in selected):
            selected.append(v)

    return selected


def _extract_g_vectors_single_ring(power_spectrum, q_center, q_width,
                                    fft_grid: FFTGrid, ring_idx: int,
                                    ring_snr: float = 0,
                                    ring_fwhm: float = 0.1,
                                    angular_prominence_frac: float = 0.5,
                                    angular_peak_distance: int = 10) -> List[GVector]:
    """Extract g-vectors from a single q-ring via polar resampling."""
    H, W = power_spectrum.shape
    n_angle = 360

    # Radial range in pixels
    r_min_px = max(1, (q_center - q_width) / fft_grid.qx_scale)
    r_max_px = (q_center + q_width) / fft_grid.qx_scale
    n_radial = max(3, int(round(r_max_px - r_min_px)))

    angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    radii = np.linspace(r_min_px, r_max_px, n_radial)

    # Build angular profile by summing over radial bins
    angular_profile = np.zeros(n_angle)
    for i, angle in enumerate(angles):
        for r in radii:
            x = fft_grid.dc_x + r * np.cos(angle)
            y = fft_grid.dc_y + r * np.sin(angle)
            # Bilinear interpolation
            if 0 <= y < H - 1 and 0 <= x < W - 1:
                val = map_coordinates(power_spectrum, [[y], [x]],
                                      order=1, mode='nearest')[0]
                angular_profile[i] += val

    if np.max(angular_profile) < 1e-10:
        return []

    # Detect angular peaks
    med_profile = np.median(angular_profile)
    peaks_idx, props = find_peaks(
        angular_profile,
        prominence=max(med_profile * angular_prominence_frac, 1e-10),
        distance=angular_peak_distance,
    )
    if len(peaks_idx) == 0:
        return []

    # Cartesian antipodal pairing (B4)
    paired_vectors: List[GVector] = []
    used = set()
    tolerance_q = max(ring_fwhm * 2.0 / 2.355, q_center * 0.03)

    for i_idx in peaks_idx:
        if i_idx in used:
            continue
        angle_i = angles[i_idx]
        gx_i = q_center * np.cos(angle_i)
        gy_i = q_center * np.sin(angle_i)

        best_partner = None
        best_residual = float('inf')

        for j_idx in peaks_idx:
            if j_idx == i_idx or j_idx in used:
                continue
            gx_j = q_center * np.cos(angles[j_idx])
            gy_j = q_center * np.sin(angles[j_idx])
            residual = np.sqrt((gx_i + gx_j) ** 2 + (gy_i + gy_j) ** 2)
            if residual < tolerance_q and residual < best_residual:
                best_partner = j_idx
                best_residual = residual

        if best_partner is not None:
            used.add(i_idx)
            used.add(best_partner)

            # SNR from angular profile peak
            peak_val = angular_profile[i_idx]
            bg_vals = angular_profile[angular_profile < np.percentile(angular_profile, 50)]
            bg_med = np.median(bg_vals) if len(bg_vals) > 0 else 0
            bg_mad = np.median(np.abs(bg_vals - bg_med)) * 1.4826 if len(bg_vals) > 0 else 1
            snr = float((peak_val - bg_med) / (bg_mad + 1e-10))

            angle_deg = float(np.degrees(angle_i))
            paired_vectors.append(GVector(
                gx=float(gx_i),
                gy=float(gy_i),
                magnitude=float(q_center),
                angle_deg=angle_deg,
                d_spacing=float(1.0 / q_center) if q_center > 0 else 0,
                snr=snr,
                fwhm=ring_fwhm,
                ring_index=ring_idx,
            ))

    return paired_vectors
