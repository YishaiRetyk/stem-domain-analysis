"""
Unified Pipeline Validation (WS7).

Evaluates all gates G1-G12, produces a structured ValidationReport.
Collects diagnostics: spectral entropy, roi_crystalline_retention, tier counts,
FWHM distribution, orientation dispersion, unwrap_success, decision_confidence.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np

from src.pipeline_config import (
    ValidationReport, GateResult, TierSummary,
    FFTPreprocRecord, ROIMaskResult, GlobalFFTResult,
    GatedTileGrid, GPAResult, LatticeValidation,
)
from src.gates import evaluate_gate, GATE_DEFS

logger = logging.getLogger(__name__)


def validate_pipeline(
    *,
    preproc_record: Optional[FFTPreprocRecord] = None,
    roi_result: Optional[ROIMaskResult] = None,
    global_fft_result: Optional[GlobalFFTResult] = None,
    gated_grid: Optional[GatedTileGrid] = None,
    gpa_result: Optional[GPAResult] = None,
    lattice_validation: Optional[LatticeValidation] = None,
    gate_results: Optional[Dict[str, GateResult]] = None,
    tile_size: int = 256,
    pixel_size_nm: float = 0.127,
    d_dom_nm: Optional[float] = None,
) -> ValidationReport:
    """Evaluate all gates and produce a unified ValidationReport.

    Parameters
    ----------
    gate_results : dict
        Pre-computed GateResults keyed by gate ID (G1-G12).
        Gates not in this dict are evaluated from available data.
    Other parameters provide raw data for evaluation and diagnostics.

    Returns
    -------
    ValidationReport
    """
    if gate_results is None:
        gate_results = {}

    gates: Dict[str, GateResult] = dict(gate_results)
    diagnostics: Dict[str, Any] = {}

    # --- G1: Input validation (typically already evaluated at load time) ---
    if "G1" not in gates:
        gates["G1"] = GateResult(
            gate_id="G1", name="Input validation",
            passed=True, value=None,
            threshold=None, failure_behavior="FATAL",
            reason="assumed valid (not re-evaluated)",
        )

    # --- G2: Preprocessing quality ---
    if "G2" not in gates and preproc_record is not None:
        g2_value = {
            "clipped_fraction": preproc_record.diagnostics.get("clipped_fraction", 0),
            "intensity_range_ratio": preproc_record.diagnostics.get("intensity_range_ratio", 100),
        }
        gates["G2"] = evaluate_gate("G2", g2_value)
        diagnostics["preproc_confidence"] = preproc_record.confidence
        if "spectral_entropy" in preproc_record.qc_metrics:
            diagnostics["spectral_entropy_fft_branch"] = preproc_record.qc_metrics["spectral_entropy"]

    # --- G3: ROI geometry ---
    if "G3" not in gates and roi_result is not None:
        g3_value = {
            "coverage_pct": roi_result.coverage_pct,
            "n_components": roi_result.n_components,
        }
        gates["G3"] = evaluate_gate("G3", g3_value)
        diagnostics["roi_coverage_pct"] = roi_result.coverage_pct

    # --- G4: Global FFT viability ---
    if "G4" not in gates and global_fft_result is not None:
        best_snr = max((p.snr for p in global_fft_result.peaks), default=0)
        gates["G4"] = evaluate_gate("G4", best_snr)

    # --- G5: Tiling adequacy ---
    if "G5" not in gates and d_dom_nm is not None and pixel_size_nm > 0:
        d_px = d_dom_nm / pixel_size_nm
        periods = tile_size / d_px if d_px > 0 else 0
        gates["G5"] = evaluate_gate("G5", periods)
        diagnostics["periods_per_tile"] = periods

    # --- G6, G7, G8: Tier classification gates ---
    tier_summary = None
    if gated_grid is not None:
        tier_summary = gated_grid.tier_summary

        if "G6" not in gates:
            gates["G6"] = evaluate_gate("G6", tier_summary.tier_a_fraction)
        if "G7" not in gates:
            gates["G7"] = evaluate_gate("G7", tier_summary.median_snr_tier_a)

        # G8: mean symmetry of Tier A
        if "G8" not in gates:
            tier_a_mask = gated_grid.tier_map == "A"
            tier_a_sym = gated_grid.symmetry_map[tier_a_mask]
            mean_sym = float(np.mean(tier_a_sym)) if len(tier_a_sym) > 0 else 0.0
            gates["G8"] = evaluate_gate("G8", mean_sym)

        # Diagnostics: FWHM distribution
        tier_a_fwhm = gated_grid.fwhm_map[gated_grid.tier_map == "A"]
        valid_fwhm = tier_a_fwhm[tier_a_fwhm > 0]
        if len(valid_fwhm) > 0:
            diagnostics["fwhm_distribution_median"] = float(np.median(valid_fwhm))
            diagnostics["fwhm_distribution_std"] = float(np.std(valid_fwhm))

        # Diagnostics: orientation dispersion
        tier_a_orient = gated_grid.orientation_map[gated_grid.tier_map == "A"]
        valid_orient = tier_a_orient[~np.isnan(tier_a_orient)]
        if len(valid_orient) > 0:
            diagnostics["orientation_std_deg"] = float(np.std(valid_orient))

    # --- G9: Reference region quality (from GPA) ---
    if "G9" not in gates and gpa_result is not None and gpa_result.reference_region is not None:
        ref = gpa_result.reference_region
        g9_value = {
            "area": len(ref.tiles),
            "entropy": ref.entropy,
            "snr": ref.mean_snr,
        }
        gates["G9"] = evaluate_gate("G9", g9_value)

    # --- G10, G11: GPA phase/strain gates ---
    if gpa_result is not None:
        if "G10" not in gates:
            g10_value = {
                "phase_noise": {k: v.phase_noise_sigma
                                for k, v in gpa_result.phases.items()
                                if v.phase_noise_sigma is not None},
                "unwrap_success": {k: v.unwrap_success_fraction
                                   for k, v in gpa_result.phases.items()},
            }
            gates["G10"] = evaluate_gate("G10", g10_value)

        if "G11" not in gates and gpa_result.strain is not None:
            g11_value = {
                "ref_strain_max": gpa_result.qc.get("ref_strain_mean_exx", 0),
                "outlier_fraction": gpa_result.qc.get("strain_outlier_fraction", 0),
            }
            gates["G11"] = evaluate_gate("G11", g11_value)

        diagnostics["gpa_mode"] = gpa_result.mode
        if gpa_result.mode_decision:
            diagnostics["gpa_decision_confidence"] = gpa_result.mode_decision.decision_confidence

    # --- G12: Peak lattice consistency ---
    if "G12" not in gates and lattice_validation is not None:
        gates["G12"] = evaluate_gate("G12", lattice_validation.fraction_valid)

    # --- Overall pass/fail ---
    # FATAL gates must pass. Others contribute to confidence.
    overall_pass = True
    fatal_failures = []
    degraded = []
    skipped_stages = []

    for gid in sorted(gates):
        gr = gates[gid]
        if not gr.passed:
            if gr.failure_behavior == "FATAL":
                overall_pass = False
                fatal_failures.append(gid)
            elif gr.failure_behavior == "SKIP_STAGE":
                skipped_stages.append(gid)
            elif gr.failure_behavior == "DEGRADE_CONFIDENCE":
                degraded.append(gid)
            elif gr.failure_behavior == "FALLBACK":
                degraded.append(gid)

    # Build summary
    parts = []
    if overall_pass:
        parts.append("Pipeline PASSED")
    else:
        parts.append(f"Pipeline FAILED (fatal: {', '.join(fatal_failures)})")
    if degraded:
        parts.append(f"degraded: {', '.join(degraded)}")
    if skipped_stages:
        parts.append(f"skipped: {', '.join(skipped_stages)}")
    n_passed = sum(1 for g in gates.values() if g.passed)
    n_total = len(gates)
    parts.append(f"{n_passed}/{n_total} gates passed")
    summary = "; ".join(parts)

    report = ValidationReport(
        gates=gates,
        overall_pass=overall_pass,
        tier_summary=tier_summary,
        diagnostics=diagnostics,
        summary=summary,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("Validation: %s", summary)
    return report
