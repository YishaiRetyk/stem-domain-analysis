"""
Geometric Phase Analysis (GPA) Engine (WS5).

Full-image and region-wise modes. Unified GPAResult schema (C12).
Subpixel-correct demodulation via real-space phase ramp (F2).
Phase unwrapping without zero-forcing (F3).
Strain via smoothed displacement fields (B5).

Gates G10 (phase noise + unwrap quality), G11 (strain sanity).
"""

import logging
import numpy as np
from scipy import ndimage
from typing import List, Optional, Dict

from src.fft_coords import FFTGrid
from src.pipeline_config import (
    GVector, GPAConfig, GPAPhaseResult, GPAResult, GPAModeDecision,
    DisplacementField, StrainField, ReferenceRegion, GatedTileGrid,
    GlobalFFTResult,
)
from src.reference_selection import select_reference_region, check_ref_region_exists
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


# ======================================================================
# GPA Mode Selection (C10, I4)
# ======================================================================

def select_gpa_mode(gated_grid: GatedTileGrid,
                    global_fft_result: GlobalFFTResult,
                    config: GPAConfig = None) -> GPAModeDecision:
    """Deterministic GPA mode auto-selection with guardrails (I4).

    Returns GPAModeDecision.
    """
    if config is None:
        config = GPAConfig()

    if config.mode != "auto":
        return GPAModeDecision(
            selected_mode=config.mode,
            decision_metrics={},
            reason="user override",
            decision_confidence=1.0,
            ref_region_exists=True,
            orientation_is_bimodal=False,
        )

    ts = gated_grid.tier_summary
    valid_tile_fraction = ts.tier_a_fraction

    # Orientation entropy from Tier A tiles
    tier_a_mask = gated_grid.tier_map == "A"
    orientations = gated_grid.orientation_map[tier_a_mask]
    valid_orient = orientations[~np.isnan(orientations)]
    if len(valid_orient) > 2:
        hist, _ = np.histogram(valid_orient, bins=12, range=(0, 180))
        probs = hist.astype(float) / hist.sum()
        probs_nz = probs[probs > 0]
        orientation_entropy = float(-np.sum(probs_nz * np.log2(probs_nz)) / np.log2(12))
    else:
        orientation_entropy = 1.0

    # Bimodal detection (I4)
    orientation_is_bimodal = False
    if len(valid_orient) > 10:
        from scipy.signal import find_peaks as _fp
        from scipy.ndimage import gaussian_filter1d
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=0.5)
        peaks_idx, _ = _fp(hist_smooth, distance=2)
        if len(peaks_idx) >= 2:
            top2 = sorted(peaks_idx, key=lambda i: hist_smooth[i], reverse=True)[:2]
            v_start, v_end = sorted(top2)
            valley_min = np.min(hist_smooth[v_start:v_end + 1])
            min_peak = min(hist_smooth[top2[0]], hist_smooth[top2[1]])
            if valley_min < min_peak * 0.5:
                orientation_is_bimodal = True

    # Global peak SNR
    global_peak_snr = max((p.snr for p in global_fft_result.peaks), default=0)

    # Ref region pre-check (I4)
    ref_region_exists = check_ref_region_exists(gated_grid, min_area=9)

    # Decision logic
    if not ref_region_exists:
        mode = "region"
        reason = "no connected Tier A region large enough for reference (>=9 tiles)"
    elif orientation_is_bimodal:
        mode = "region"
        reason = "bimodal orientation distribution detected -- multiple grains likely"
    elif (valid_tile_fraction >= config.auto_min_valid_tile_fraction and
          orientation_entropy <= config.auto_max_orientation_entropy and
          global_peak_snr >= config.auto_min_global_peak_snr):
        mode = "full"
        reason = "high Tier A coverage, unimodal low-entropy orientation, strong global peaks"
    else:
        mode = "region"
        reason = "conditions for full-image GPA not met"

    # Decision confidence (I4)
    vtf_margin = (valid_tile_fraction - config.auto_min_valid_tile_fraction) / max(config.auto_min_valid_tile_fraction, 0.01)
    ent_margin = (config.auto_max_orientation_entropy - orientation_entropy) / max(config.auto_max_orientation_entropy, 0.01)
    snr_margin = (global_peak_snr - config.auto_min_global_peak_snr) / max(config.auto_min_global_peak_snr, 0.01)
    margins = [np.clip(vtf_margin, -1, 1), np.clip(ent_margin, -1, 1), np.clip(snr_margin, -1, 1)]
    decision_confidence = float(np.clip(np.mean(margins) / 2 + 0.5, 0, 1))

    metrics = {
        "valid_tile_fraction": valid_tile_fraction,
        "orientation_entropy": orientation_entropy,
        "orientation_is_bimodal": orientation_is_bimodal,
        "global_peak_snr": global_peak_snr,
        "ref_region_exists": ref_region_exists,
    }

    decision = GPAModeDecision(
        selected_mode=mode,
        decision_metrics=metrics,
        thresholds_used={
            "vtf": config.auto_min_valid_tile_fraction,
            "entropy": config.auto_max_orientation_entropy,
            "snr": config.auto_min_global_peak_snr,
        },
        reason=reason,
        decision_confidence=decision_confidence,
        ref_region_exists=ref_region_exists,
        orientation_is_bimodal=orientation_is_bimodal,
    )

    logger.info("GPA mode decision: %s (confidence=%.2f, reason=%s)",
                mode, decision_confidence, reason)
    return decision


# ======================================================================
# GPA Phase Extraction (B5, F2, F3)
# ======================================================================

def compute_gpa_phase(image_fft: np.ndarray,
                      g_vector: GVector,
                      mask_sigma_q: float,
                      fft_grid: FFTGrid,
                      amplitude_threshold: float = 0.1) -> GPAPhaseResult:
    """GPA phase extraction following Hytch et al. (1998).

    Uses subpixel-correct real-space phase ramp (F2).
    Phase unwrapping on full array without zero-forcing (F3).
    """
    H, W = image_fft.shape
    px = fft_grid.pixel_size_nm

    # FFT
    FT = np.fft.fft2(image_fft)
    FT_shifted = np.fft.fftshift(FT)

    # Gaussian mask centred on g in frequency space
    g_px_x, g_px_y = fft_grid.q_to_px(g_vector.gx, g_vector.gy)
    sigma_px_x = mask_sigma_q / fft_grid.qx_scale
    sigma_px_y = mask_sigma_q / fft_grid.qy_scale

    y_freq, x_freq = np.mgrid[:H, :W]
    mask = np.exp(-0.5 * ((x_freq - g_px_x) ** 2 / (sigma_px_x ** 2 + 1e-10) +
                           (y_freq - g_px_y) ** 2 / (sigma_px_y ** 2 + 1e-10)))

    # Extract g-component
    filtered = FT_shifted * mask
    H_g_prime = np.fft.ifft2(np.fft.ifftshift(filtered))

    # Subpixel-correct demodulation via real-space phase ramp (F2)
    row_nm = np.arange(H).reshape(-1, 1) * px
    col_nm = np.arange(W).reshape(1, -1) * px
    phase_ramp = np.exp(-2j * np.pi * (g_vector.gx * col_nm + g_vector.gy * row_nm))

    H_g = H_g_prime * phase_ramp
    phase_raw = np.angle(H_g)
    amplitude = np.abs(H_g)

    # Amplitude mask (B5)
    amp_threshold = amplitude_threshold * np.max(amplitude)
    amplitude_mask = amplitude > amp_threshold

    # Phase unwrapping -- full array, NO zero-forcing (F3)
    try:
        from skimage.restoration import unwrap_phase
        phase_unwrapped = unwrap_phase(phase_raw)
    except ImportError:
        logger.warning("skimage.restoration.unwrap_phase not available, using raw phase")
        phase_unwrapped = phase_raw.copy()

    # Post-mask with eroded amplitude mask (F3)
    amplitude_mask_eroded = ndimage.binary_erosion(amplitude_mask, iterations=2)
    phase_unwrapped_masked = phase_unwrapped.copy()
    phase_unwrapped_masked[~amplitude_mask_eroded] = np.nan

    # Unwrap quality
    n_valid = int(np.sum(amplitude_mask_eroded & ~np.isnan(phase_unwrapped_masked)))
    n_amp_masked = int(np.sum(amplitude_mask))
    unwrap_success = n_valid / max(n_amp_masked, 1)

    return GPAPhaseResult(
        phase_raw=phase_raw.astype(np.float32),
        phase_unwrapped=phase_unwrapped_masked.astype(np.float32),
        amplitude=amplitude.astype(np.float32),
        amplitude_mask=amplitude_mask_eroded,
        g_vector=g_vector,
        phase_noise_sigma=None,  # computed later
        unwrap_success_fraction=float(unwrap_success),
    )


def check_phase_noise(phase_result: GPAPhaseResult,
                      ref_tiles: List,
                      tile_size: int,
                      stride: int) -> float:
    """Compute phase noise sigma in the reference region.

    Returns sigma_phi in radians on unwrapped, amplitude-masked phase.
    """
    phase = phase_result.phase_unwrapped
    mask = phase_result.amplitude_mask

    # Build reference region pixel mask from tile positions
    H, W = phase.shape
    ref_mask = np.zeros((H, W), dtype=bool)
    for (r, c) in ref_tiles:
        y0 = r * stride
        x0 = c * stride
        y1 = min(y0 + tile_size, H)
        x1 = min(x0 + tile_size, W)
        ref_mask[y0:y1, x0:x1] = True

    combined = ref_mask & mask & ~np.isnan(phase)
    if np.sum(combined) < 10:
        return float('inf')

    ref_phase = phase[combined]
    # Subtract mean to get noise
    ref_phase = ref_phase - np.mean(ref_phase)
    return float(np.std(ref_phase))


# ======================================================================
# Displacement and Strain (B5)
# ======================================================================

def compute_displacement_field(phase_g1: GPAPhaseResult,
                               phase_g2: GPAPhaseResult) -> Optional[DisplacementField]:
    """Compute displacement from two unwrapped phase fields.

    Convention (F1): g in cycles/nm, phase in radians.
    u(r) = -1/(2pi) * G^{-1} * [phi_g1, phi_g2]^T
    Result u is in nm.
    """
    g1, g2 = phase_g1.g_vector, phase_g2.g_vector
    det = g1.gx * g2.gy - g1.gy * g2.gx
    if abs(det) < 1e-10:
        logger.warning("G-vectors are collinear, cannot compute displacement")
        return None

    inv_det = 1.0 / det
    phi1 = phase_g1.phase_unwrapped
    phi2 = phase_g2.phase_unwrapped

    ux = -inv_det / (2 * np.pi) * (g2.gy * phi1 - g1.gy * phi2)
    uy = -inv_det / (2 * np.pi) * (-g2.gx * phi1 + g1.gx * phi2)

    return DisplacementField(ux=ux, uy=uy)


def smooth_displacement(displacement: DisplacementField,
                        sigma: float = 2.0) -> DisplacementField:
    """Gaussian smooth displacement fields before strain computation (B5)."""
    ux_smooth = ndimage.gaussian_filter(np.nan_to_num(displacement.ux), sigma=sigma)
    uy_smooth = ndimage.gaussian_filter(np.nan_to_num(displacement.uy), sigma=sigma)
    return DisplacementField(ux=ux_smooth, uy=uy_smooth)


def compute_strain_field(displacement: DisplacementField,
                         pixel_size_nm: float) -> StrainField:
    """Compute strain via central differences on smoothed displacement (B5).

    Displacement u is in nm, pixel_size_nm is real-space pixel pitch.
    Strain is dimensionless.
    """
    du_x_dx = np.gradient(displacement.ux, axis=1) / pixel_size_nm
    du_x_dy = np.gradient(displacement.ux, axis=0) / pixel_size_nm
    du_y_dx = np.gradient(displacement.uy, axis=1) / pixel_size_nm
    du_y_dy = np.gradient(displacement.uy, axis=0) / pixel_size_nm

    return StrainField(
        exx=du_x_dx,
        eyy=du_y_dy,
        exy=0.5 * (du_x_dy + du_y_dx),
        rotation=0.5 * (du_y_dx - du_x_dy),
    )


# ======================================================================
# GPA Execution (full and region modes)
# ======================================================================

def _determine_mask_sigma(g_vectors: List[GVector], config: GPAConfig) -> float:
    """Determine the Gaussian mask radius in q-space."""
    if config.mask_radius_q != "auto":
        return float(config.mask_radius_q)
    # Auto: 0.5 * min(peak FWHM)
    fwhms = [g.fwhm for g in g_vectors if g.fwhm > 0]
    if fwhms:
        return 0.5 * min(fwhms)
    return 0.06  # safe default


def run_gpa_full(image_fft: np.ndarray,
                 g_vectors: List[GVector],
                 ref_region: ReferenceRegion,
                 fft_grid: FFTGrid,
                 config: GPAConfig = None,
                 tile_size: int = 256,
                 stride: int = 128) -> GPAResult:
    """Full-image GPA mode."""
    if config is None:
        config = GPAConfig()

    mask_sigma = _determine_mask_sigma(g_vectors, config)
    phases: Dict[str, GPAPhaseResult] = {}
    qc: dict = {"frequency_unit": "cycles/nm"}

    for i, gv in enumerate(g_vectors[:2]):  # max 2 g-vectors
        key = f"g{i}"
        phase_result = compute_gpa_phase(
            image_fft, gv, mask_sigma, fft_grid,
            amplitude_threshold=config.amplitude_threshold,
        )
        # Phase noise in reference region
        sigma_phi = check_phase_noise(phase_result, ref_region.tiles,
                                      tile_size, stride)
        phase_result.phase_noise_sigma = sigma_phi

        # Subtract reference mean phase
        ref_phase_vals = []
        H, W = phase_result.phase_unwrapped.shape
        for (r, c) in ref_region.tiles:
            y0, x0 = r * stride, c * stride
            y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
            tile_phase = phase_result.phase_unwrapped[y0:y1, x0:x1]
            valid = tile_phase[phase_result.amplitude_mask[y0:y1, x0:x1] & ~np.isnan(tile_phase)]
            ref_phase_vals.extend(valid.tolist())

        if ref_phase_vals:
            ref_mean = np.mean(ref_phase_vals)
            phase_result.phase_unwrapped = phase_result.phase_unwrapped - ref_mean

        phases[key] = phase_result
        qc[f"phase_noise_{key}"] = sigma_phi
        qc[f"unwrap_success_{key}"] = phase_result.unwrap_success_fraction

    # Displacement and strain if 2 non-collinear g-vectors
    displacement = None
    strain = None
    if len(phases) == 2:
        displacement = compute_displacement_field(phases["g0"], phases["g1"])
        if displacement is not None:
            displacement = smooth_displacement(displacement, sigma=config.displacement_smooth_sigma)
            strain = compute_strain_field(displacement, fft_grid.pixel_size_nm)
            qc["displacement_smooth_sigma"] = config.displacement_smooth_sigma

    # Gate G10
    g10_value = {
        "phase_noise": {k: v.phase_noise_sigma for k, v in phases.items()},
        "unwrap_success": {k: v.unwrap_success_fraction for k, v in phases.items()},
    }
    g10 = evaluate_gate("G10", g10_value)
    qc["g10_passed"] = g10.passed

    # Gate G11
    if strain is not None:
        # Reference region strain
        ref_mask = phases["g0"].amplitude_mask.copy()
        for (r, c) in ref_region.tiles:
            y0, x0 = r * stride, c * stride
            y1 = min(y0 + tile_size, strain.exx.shape[0])
            x1 = min(x0 + tile_size, strain.exx.shape[1])
            # keep existing mask

        ref_exx = strain.exx[ref_mask & ~np.isnan(strain.exx)]
        ref_eyy = strain.eyy[ref_mask & ~np.isnan(strain.eyy)]
        ref_strain_max = max(
            abs(float(np.mean(ref_exx))) if len(ref_exx) > 0 else 0,
            abs(float(np.mean(ref_eyy))) if len(ref_eyy) > 0 else 0,
        )

        # Outlier fraction
        amp_mask = phases["g0"].amplitude_mask
        valid = amp_mask & ~np.isnan(strain.exx)
        total_valid = np.sum(valid)
        if total_valid > 0:
            outliers = (np.abs(strain.exx[valid]) > config.strain_outlier_threshold).sum()
            outliers += (np.abs(strain.eyy[valid]) > config.strain_outlier_threshold).sum()
            outlier_frac = outliers / (2 * total_valid)
        else:
            outlier_frac = 0

        g11_value = {
            "ref_strain_max": ref_strain_max,
            "outlier_fraction": float(outlier_frac),
        }
        g11 = evaluate_gate("G11", g11_value)
        qc["g11_passed"] = g11.passed
        qc["ref_strain_mean_exx"] = float(np.mean(ref_exx)) if len(ref_exx) > 0 else 0
        qc["ref_strain_mean_eyy"] = float(np.mean(ref_eyy)) if len(ref_eyy) > 0 else 0
        qc["strain_outlier_fraction"] = float(outlier_frac)

    mode_decision = GPAModeDecision(
        selected_mode="full",
        decision_metrics={},
        reason="direct call",
        decision_confidence=1.0,
    )

    return GPAResult(
        mode="full",
        phases=phases,
        displacement=displacement,
        strain=strain,
        reference_region=ref_region,
        mode_decision=mode_decision,
        qc=qc,
        diagnostics={"mask_sigma_q": mask_sigma},
    )


def run_gpa_region(image_fft: np.ndarray,
                   g_vectors: List[GVector],
                   gated_grid: GatedTileGrid,
                   fft_grid: FFTGrid,
                   config: GPAConfig = None,
                   tile_size: int = 256,
                   stride: int = 128) -> GPAResult:
    """Region-wise GPA mode.

    Segments Tier A tiles into connected domains.
    For each domain: select local reference, run GPA.
    Stitches results into a unified GPAResult.
    """
    if config is None:
        config = GPAConfig()

    H, W = image_fft.shape
    mask_sigma = _determine_mask_sigma(g_vectors, config)

    # Find connected domains of Tier A tiles
    tier_a_mask = (gated_grid.tier_map == "A")
    labeled, n_domains = ndimage.label(tier_a_mask)

    # Collect per-domain phases
    all_phases: Dict[str, GPAPhaseResult] = {}
    combined_displacement = None
    combined_strain = None
    qc: dict = {"frequency_unit": "cycles/nm", "mode": "region", "n_domains": n_domains}

    # Initialise combined arrays
    combined_phase_unwrapped = {f"g{i}": np.full((H, W), np.nan) for i in range(min(len(g_vectors), 2))}
    combined_amplitude_mask = {f"g{i}": np.zeros((H, W), dtype=bool) for i in range(min(len(g_vectors), 2))}

    best_ref_region = None

    for dom_id in range(1, n_domains + 1):
        dom_mask_grid = labeled == dom_id
        dom_tiles = list(zip(*np.where(dom_mask_grid)))

        if len(dom_tiles) < 4:
            continue

        # Local reference: lowest entropy tile cluster within domain
        # Simple: use the domain itself as reference region for subtraction
        local_ref = ReferenceRegion(
            center_tile=dom_tiles[0],
            tiles=dom_tiles,
            bounding_box=(
                min(t[0] for t in dom_tiles), max(t[0] for t in dom_tiles),
                min(t[1] for t in dom_tiles), max(t[1] for t in dom_tiles),
            ),
            orientation_mean=float(np.nanmean(gated_grid.orientation_map[dom_mask_grid])),
            orientation_std=float(np.nanstd(gated_grid.orientation_map[dom_mask_grid])),
            mean_snr=float(np.mean(gated_grid.snr_map[dom_mask_grid])),
            entropy=0.0,
            score=0.0,
        )

        if best_ref_region is None or local_ref.mean_snr > best_ref_region.mean_snr:
            best_ref_region = local_ref

        # Compute phase for this domain's bounding box region
        r_min, r_max, c_min, c_max = local_ref.bounding_box
        y_min = r_min * stride
        y_max = min((r_max + 1) * stride + tile_size, H)
        x_min = c_min * stride
        x_max = min((c_max + 1) * stride + tile_size, W)

        for gi, gv in enumerate(g_vectors[:2]):
            key = f"g{gi}"
            phase_result = compute_gpa_phase(
                image_fft, gv, mask_sigma, fft_grid,
                amplitude_threshold=config.amplitude_threshold,
            )

            # Extract domain region and insert into combined
            combined_phase_unwrapped[key][y_min:y_max, x_min:x_max] = \
                phase_result.phase_unwrapped[y_min:y_max, x_min:x_max]
            combined_amplitude_mask[key][y_min:y_max, x_min:x_max] |= \
                phase_result.amplitude_mask[y_min:y_max, x_min:x_max]

    # Build combined phase results
    for gi, gv in enumerate(g_vectors[:2]):
        key = f"g{gi}"
        phase = compute_gpa_phase(image_fft, gv, mask_sigma, fft_grid,
                                  amplitude_threshold=config.amplitude_threshold)
        phase.phase_unwrapped = combined_phase_unwrapped[key].astype(np.float32)
        phase.amplitude_mask = combined_amplitude_mask[key]

        if best_ref_region:
            sigma_phi = check_phase_noise(phase, best_ref_region.tiles, tile_size, stride)
            phase.phase_noise_sigma = sigma_phi
            qc[f"phase_noise_{key}"] = sigma_phi

        qc[f"unwrap_success_{key}"] = phase.unwrap_success_fraction
        all_phases[key] = phase

    # Displacement / strain
    if len(all_phases) == 2:
        combined_displacement = compute_displacement_field(all_phases["g0"], all_phases["g1"])
        if combined_displacement is not None:
            combined_displacement = smooth_displacement(combined_displacement, sigma=config.displacement_smooth_sigma)
            combined_strain = compute_strain_field(combined_displacement, fft_grid.pixel_size_nm)

    mode_decision = GPAModeDecision(
        selected_mode="region",
        decision_metrics={},
        reason="region-wise execution",
        decision_confidence=1.0,
    )

    return GPAResult(
        mode="region",
        phases=all_phases,
        displacement=combined_displacement,
        strain=combined_strain,
        reference_region=best_ref_region,
        mode_decision=mode_decision,
        qc=qc,
        diagnostics={"mask_sigma_q": mask_sigma, "n_domains": n_domains},
    )


# ======================================================================
# Top-level GPA runner
# ======================================================================

def run_gpa(image_fft: np.ndarray,
            g_vectors: List[GVector],
            gated_grid: GatedTileGrid,
            global_fft_result: GlobalFFTResult,
            fft_grid: FFTGrid,
            config: GPAConfig = None,
            tile_size: int = 256,
            stride: int = 128) -> Optional[GPAResult]:
    """Run GPA with mode auto-selection and failure handling.

    Returns GPAResult or None if skipped.
    """
    if config is None:
        config = GPAConfig()

    if not config.enabled:
        logger.info("GPA disabled by config")
        return None

    if not g_vectors:
        logger.warning("No g-vectors available, skipping GPA")
        return None

    mode_decision = select_gpa_mode(gated_grid, global_fft_result, config)

    try:
        if mode_decision.selected_mode == "full":
            ref_region = select_reference_region(gated_grid)
            result = run_gpa_full(image_fft, g_vectors, ref_region, fft_grid,
                                  config, tile_size, stride)
            result.mode_decision = mode_decision
            return result
        else:
            result = run_gpa_region(image_fft, g_vectors, gated_grid, fft_grid,
                                    config, tile_size, stride)
            result.mode_decision = mode_decision
            return result

    except Exception as e:
        logger.error("GPA failed: %s", e)

        if config.on_fail == "fallback_to_region" and mode_decision.selected_mode == "full":
            logger.info("Falling back to region-wise GPA")
            try:
                result = run_gpa_region(image_fft, g_vectors, gated_grid, fft_grid,
                                        config, tile_size, stride)
                result.mode_decision = mode_decision
                result.diagnostics["fallback"] = True
                return result
            except Exception as e2:
                logger.error("Region-wise fallback also failed: %s", e2)

        if config.on_fail == "error":
            raise
        # skip
        logger.info("GPA skipped due to failure")
        return None
