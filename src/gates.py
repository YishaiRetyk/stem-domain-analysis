"""
Gate Registry.

Defines all quality gates (G1-G12) with their thresholds, failure behaviours,
and evaluation logic.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.pipeline_config import GateResult, GateThresholdsConfig

logger = logging.getLogger(__name__)


@dataclass
class GateDef:
    """Definition of a single quality gate."""
    gate_id: str
    name: str
    description: str
    failure_behavior: str       # FATAL / SKIP_STAGE / DEGRADE_CONFIDENCE / FALLBACK
    default_threshold: Any      # scalar, dict, or tuple


# Gate registry
GATE_DEFS: Dict[str, GateDef] = {
    "G0": GateDef(
        "G0", "Nyquist guard",
        "d_min >= 2*pixel_size; auto-clamp q_max to 0.95*q_nyquist if violated; "
        "FATAL if entire band invalid",
        "DEGRADE_CONFIDENCE",
        {"nyquist_safety_margin": 0.95},
    ),
    "G1": GateDef(
        "G1", "Input validation",
        "2D, float-convertible, no NaN/Inf, >=512x512, pixel_size_nm>0",
        "FATAL", None,
    ),
    "G2": GateDef(
        "G2", "Preprocessing quality",
        "clipped_fraction < 0.5%, intensity_range_ratio > 10",
        "DEGRADE_CONFIDENCE",
        {"max_clipped_fraction": 0.005, "min_range_ratio": 10.0},
    ),
    "G3": GateDef(
        "G3", "ROI geometry",
        "coverage in [10%, 95%], connected_components <= max_fragments",
        "FALLBACK",
        {"min_coverage": 10.0, "max_coverage": 95.0, "max_fragments": 20},
    ),
    "G4": GateDef(
        "G4", "Global FFT viability",
        "At least one peak with SNR >= 3.0 in global FFT",
        "FALLBACK",
        {"min_peak_snr": 3.0},
    ),
    "G5": GateDef(
        "G5", "Tiling adequacy",
        "tile_size / (d_dom / pixel_size) >= 20 periods",
        "FATAL",
        {"min_periods": 20.0},
    ),
    "G6": GateDef(
        "G6", "Tier A detection rate",
        "n_tier_a / n_roi_tiles >= 5%",
        "DEGRADE_CONFIDENCE",
        {"min_fraction": 0.05},
    ),
    "G7": GateDef(
        "G7", "Tier A SNR quality",
        "Median SNR of Tier A tiles >= 5.0",
        "DEGRADE_CONFIDENCE",
        {"min_median_snr": 5.0},
    ),
    "G8": GateDef(
        "G8", "Symmetry quality",
        "Mean symmetry score of Tier A tiles >= 0.3",
        "DEGRADE_CONFIDENCE",
        {"min_mean_symmetry": 0.3},
    ),
    "G9": GateDef(
        "G9", "Reference region quality",
        "area >= 9 tiles, entropy <= 0.3, mean SNR >= 5.0",
        "SKIP_STAGE",
        {"min_area": 9, "max_entropy": 0.3, "min_snr": 5.0},
    ),
    "G10": GateDef(
        "G10", "GPA phase noise + unwrap quality",
        "sigma_phi <= 0.3 rad per g, unwrap_success >= 0.7",
        "SKIP_STAGE",
        {"max_phase_noise": 0.3, "min_unwrap_success": 0.7},
    ),
    "G11": GateDef(
        "G11", "GPA strain sanity",
        "|ref_strain| <= 0.005, outlier_frac <= 0.20",
        "SKIP_STAGE",
        {"max_ref_strain": 0.005, "max_outlier_fraction": 0.20,
         "strain_outlier_threshold": 0.05},
    ),
    "G12": GateDef(
        "G12", "Peak lattice consistency",
        "fraction_valid >= 0.50 with tolerance 0.20",
        "DEGRADE_CONFIDENCE",
        {"min_fraction_valid": 0.50, "tolerance": 0.20},
    ),
}


def evaluate_gate(gate_id: str, value: Any,
                  threshold_override: Any = None,
                  gate_thresholds: Optional[GateThresholdsConfig] = None) -> GateResult:
    """Evaluate a gate and return a GateResult.

    Parameters
    ----------
    gate_id : str
        One of G1-G12.
    value : scalar or dict
        The measured value(s).
    threshold_override : optional
        Override the default threshold (takes precedence over *gate_thresholds*).
    gate_thresholds : GateThresholdsConfig, optional
        Consolidated config; used when *threshold_override* is None.

    Returns
    -------
    GateResult
    """
    gate_def = GATE_DEFS[gate_id]
    if threshold_override is not None:
        threshold = threshold_override
    elif gate_thresholds is not None:
        threshold = gate_thresholds.threshold_dict(gate_id)
        if threshold is None:
            threshold = gate_def.default_threshold
    else:
        threshold = gate_def.default_threshold

    passed = True
    reason = ""

    if gate_id == "G0":
        # value is dict: {q_max_requested, q_min_requested, q_nyquist, safety_margin}
        v = value if isinstance(value, dict) else {}
        q_max_req = v.get("q_max_requested", 0)
        q_min_req = v.get("q_min_requested", 0)
        q_nyquist = v.get("q_nyquist", float('inf'))
        safety = v.get("safety_margin", 0.95)
        q_safe = safety * q_nyquist

        if q_min_req >= q_nyquist:
            # Entire band invalid
            passed = False
            reason = (f"FATAL: entire q-band invalid — "
                      f"q_min={q_min_req:.4f} >= q_nyquist={q_nyquist:.4f}")
        elif q_max_req > q_safe:
            # Auto-clamp: degrade
            passed = False
            reason = (f"DEGRADE: q_max clamped {q_max_req:.4f} → {q_safe:.4f} "
                      f"(0.95 × q_nyquist={q_nyquist:.4f})")
        else:
            passed = True
            reason = (f"q_max={q_max_req:.4f} within safe limit "
                      f"{q_safe:.4f}: OK")

    elif gate_id == "G1":
        # value is dict: {is_2d, no_nan, min_dim, has_pixel_size}
        checks = value if isinstance(value, dict) else {}
        for k, v in checks.items():
            if not v:
                passed = False
                reason += f"{k} failed; "
        if passed:
            reason = "all input checks passed"

    elif gate_id == "G2":
        clipped = value.get("clipped_fraction", 1.0) if isinstance(value, dict) else 1.0
        ratio = value.get("intensity_range_ratio", 0.0) if isinstance(value, dict) else 0.0
        t = threshold if isinstance(threshold, dict) else {}
        if clipped > t.get("max_clipped_fraction", 0.005):
            passed = False
            reason += f"clipped_fraction={clipped:.4f} > {t.get('max_clipped_fraction', 0.005)}; "
        if ratio < t.get("min_range_ratio", 10.0):
            passed = False
            reason += f"range_ratio={ratio:.1f} < {t.get('min_range_ratio', 10.0)}; "
        if passed:
            reason = f"clipped={clipped:.4f}, ratio={ratio:.1f}: OK"

    elif gate_id == "G3":
        cov = value.get("coverage_pct", 0) if isinstance(value, dict) else 0
        ncomp = value.get("n_components", 999) if isinstance(value, dict) else 999
        lcc_frac = value.get("lcc_fraction", 1.0) if isinstance(value, dict) else 1.0
        t = threshold if isinstance(threshold, dict) else {}
        if cov < t.get("min_coverage", 10):
            passed = False
            reason += f"coverage={cov:.1f}% < min; "
        if cov > t.get("max_coverage", 95):
            passed = False
            reason += f"coverage={cov:.1f}% > max; "
        if ncomp > t.get("max_fragments", 20):
            passed = False
            reason += f"fragments={ncomp} > max; "
        if lcc_frac < t.get("min_lcc_fraction", 0.5):
            passed = False
            reason += f"lcc_fraction={lcc_frac:.2f} < {t.get('min_lcc_fraction', 0.5)}; "
        if passed:
            reason = f"coverage={cov:.1f}%, fragments={ncomp}, lcc={lcc_frac:.2f}: OK"

    elif gate_id == "G4":
        snr_val = value if isinstance(value, (int, float)) else 0
        t = threshold if isinstance(threshold, dict) else {"min_peak_snr": 3.0}
        if snr_val < t.get("min_peak_snr", 3.0):
            passed = False
            reason = f"best_peak_snr={snr_val:.1f} < {t.get('min_peak_snr', 3.0)}"
        else:
            reason = f"best_peak_snr={snr_val:.1f}: OK"

    elif gate_id == "G5":
        periods = value if isinstance(value, (int, float)) else 0
        t = threshold if isinstance(threshold, dict) else {"min_periods": 20.0}
        if periods < t.get("min_periods", 20.0):
            passed = False
            reason = f"periods={periods:.1f} < {t.get('min_periods', 20.0)}"
        else:
            reason = f"periods={periods:.1f}: OK"

    elif gate_id == "G6":
        frac = value if isinstance(value, (int, float)) else 0
        t = threshold if isinstance(threshold, dict) else {"min_fraction": 0.05}
        if frac < t.get("min_fraction", 0.05):
            passed = False
            reason = f"tier_a_fraction={frac:.3f} < {t.get('min_fraction', 0.05)}"
        else:
            reason = f"tier_a_fraction={frac:.3f}: OK"

    elif gate_id == "G7":
        snr_val = value if isinstance(value, (int, float)) else 0
        t = threshold if isinstance(threshold, dict) else {"min_median_snr": 5.0}
        if snr_val < t.get("min_median_snr", 5.0):
            passed = False
            reason = f"median_snr={snr_val:.1f} < {t.get('min_median_snr', 5.0)}"
        else:
            reason = f"median_snr={snr_val:.1f}: OK"

    elif gate_id == "G8":
        sym = value if isinstance(value, (int, float)) else 0
        t = threshold if isinstance(threshold, dict) else {"min_mean_symmetry": 0.3}
        if sym < t.get("min_mean_symmetry", 0.3):
            passed = False
            reason = f"mean_symmetry={sym:.2f} < {t.get('min_mean_symmetry', 0.3)}"
        else:
            reason = f"mean_symmetry={sym:.2f}: OK"

    elif gate_id == "G9":
        v = value if isinstance(value, dict) else {}
        t = threshold if isinstance(threshold, dict) else {}
        area = v.get("area", 0)
        ent = v.get("entropy", 1.0)
        snr_v = v.get("snr", 0)
        if area < t.get("min_area", 9):
            passed = False
            reason += f"area={area} < {t.get('min_area', 9)}; "
        if ent > t.get("max_entropy", 0.3):
            passed = False
            reason += f"entropy={ent:.2f} > {t.get('max_entropy', 0.3)}; "
        if snr_v < t.get("min_snr", 5.0):
            passed = False
            reason += f"snr={snr_v:.1f} < {t.get('min_snr', 5.0)}; "
        if passed:
            reason = f"area={area}, entropy={ent:.2f}, snr={snr_v:.1f}: OK"

    elif gate_id == "G10":
        v = value if isinstance(value, dict) else {}
        t = threshold if isinstance(threshold, dict) else {}
        for gname, sigma in v.get("phase_noise", {}).items():
            if sigma > t.get("max_phase_noise", 0.3):
                passed = False
                reason += f"{gname}: sigma={sigma:.2f} > max; "
        for gname, frac in v.get("unwrap_success", {}).items():
            if frac < t.get("min_unwrap_success", 0.7):
                passed = False
                reason += f"{gname}: unwrap={frac:.2f} < min; "
        if passed:
            reason = "phase noise and unwrap quality OK"

    elif gate_id == "G11":
        v = value if isinstance(value, dict) else {}
        t = threshold if isinstance(threshold, dict) else {}
        ref_strain = v.get("ref_strain_max", 0)
        outlier_frac = v.get("outlier_fraction", 0)
        if ref_strain > t.get("max_ref_strain", 0.005):
            passed = False
            reason += f"|ref_strain|={ref_strain:.4f} > max; "
        if outlier_frac > t.get("max_outlier_fraction", 0.20):
            passed = False
            reason += f"outlier_frac={outlier_frac:.2f} > max; "
        if passed:
            reason = f"ref_strain={ref_strain:.4f}, outliers={outlier_frac:.2f}: OK"

    elif gate_id == "G12":
        frac = value if isinstance(value, (int, float)) else 0
        t = threshold if isinstance(threshold, dict) else {"min_fraction_valid": 0.50}
        if frac < t.get("min_fraction_valid", 0.50):
            passed = False
            reason = f"fraction_valid={frac:.2f} < {t.get('min_fraction_valid', 0.50)}"
        else:
            reason = f"fraction_valid={frac:.2f}: OK"

    result = GateResult(
        gate_id=gate_id,
        name=gate_def.name,
        passed=passed,
        value=value,
        threshold=threshold,
        failure_behavior=gate_def.failure_behavior,
        reason=reason,
    )

    level = logging.INFO if passed else logging.WARNING
    logger.log(level, "Gate %s (%s): %s -- %s",
               gate_id, gate_def.name, "PASS" if passed else "FAIL", reason)

    return result
