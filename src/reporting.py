"""
Pipeline Reporting and Artifact Output (WS7).

JSON/artifact output with naming conventions.
Parameters.json v3.0 with all sections.
Report.json with gate results and diagnostics.
"""

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from src.fft_coords import FFTGrid
from src.pipeline_config import (
    PipelineConfig, FFTPreprocRecord, SegPreprocRecord, ROIMaskResult,
    GlobalFFTResult, GatedTileGrid, GPAResult, LatticeValidation,
    ValidationReport, GVector, SubpixelPeak,
)

logger = logging.getLogger(__name__)


# ======================================================================
# JSON serialisation helpers
# ======================================================================

def _make_serialisable(obj: Any) -> Any:
    """Convert an object to a JSON-serialisable form."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        if obj.size <= 100:
            return obj.tolist()
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        d = {}
        for k, v in asdict(obj).items():
            d[k] = _make_serialisable(v)
        return d
    return str(obj)


# ======================================================================
# Parameters.json v3.0
# ======================================================================

def build_parameters_v3(
    *,
    config: PipelineConfig,
    fft_grid: FFTGrid,
    preproc_record: Optional[FFTPreprocRecord] = None,
    seg_record: Optional[SegPreprocRecord] = None,
    roi_result: Optional[ROIMaskResult] = None,
    global_fft_result: Optional[GlobalFFTResult] = None,
    gated_grid: Optional[GatedTileGrid] = None,
    gpa_result: Optional[GPAResult] = None,
    lattice_validation: Optional[LatticeValidation] = None,
    validation_report: Optional[ValidationReport] = None,
    extra: Optional[Dict[str, Any]] = None,
    effective_q_min: float = 0.0,
    tile_effective_q_min: float = 0.0,
    ring_maps=None,
    clustering_result=None,
) -> dict:
    """Build the parameters.json v3.0 dict.

    All values are JSON-serialisable.
    """
    params: Dict[str, Any] = {
        "version": "3.0",
        "pixel_size_nm": config.pixel_size_nm,
        "tile_size": config.tile_size,
        "stride": config.stride,
        "fft_convention": fft_grid.to_dict(),
    }

    # Physics config
    params["physics"] = {
        "imaging_mode": config.physics.imaging_mode,
        "d_min_nm": config.physics.d_min_nm,
        "d_max_nm": config.physics.d_max_nm,
        "nyquist_safety_margin": config.physics.nyquist_safety_margin,
    }

    # Low-q exclusion
    low_q_cfg = config.low_q
    params["low_q_exclusion"] = {
        "enabled": low_q_cfg.enabled,
        "q_min_configured": low_q_cfg.q_min_cycles_per_nm,
        "dc_bin_count": low_q_cfg.dc_bin_count,
        "auto_q_min": low_q_cfg.auto_q_min,
        "effective_q_min_global": round(effective_q_min, 4),
        "effective_q_min_tile": round(tile_effective_q_min, 4),
    }

    # Preprocessing
    preproc_section: Dict[str, Any] = {}
    if preproc_record is not None:
        preproc_section["fft_branch"] = {
            "clip_percentile": config.preprocessing.clip_percentile,
            "normalize_method": config.preprocessing.normalize_method,
            "hot_pixel_removal": config.preprocessing.hot_pixel_removal,
            "clipped_fraction": preproc_record.diagnostics.get("clipped_fraction", 0),
            "hot_pixels_replaced": preproc_record.diagnostics.get("hot_pixels_replaced", 0),
            "confidence": preproc_record.confidence,
        }
    if seg_record is not None:
        preproc_section["seg_branch"] = {
            "clip_percentile": config.segmentation.clip_percentile,
            "smooth_sigma": config.segmentation.smooth_sigma,
        }
    if preproc_section:
        params["preprocessing"] = preproc_section

    # ROI
    if roi_result is not None:
        roi_section = {
            "coverage_pct": roi_result.coverage_pct,
            "n_components": roi_result.n_components,
            "lcc_fraction": roi_result.lcc_fraction,
            "method": "intensity+variance",
        }
        if roi_result.diagnostics:
            roi_section["gradient_used"] = roi_result.diagnostics.get("gradient_used", False)
            roi_section["roi_confidence"] = roi_result.diagnostics.get("roi_confidence", "normal")
        params["roi"] = roi_section

    # Global FFT
    if global_fft_result is not None:
        gf = global_fft_result
        peaks_list = []
        for p in gf.peaks:
            peaks_list.append({
                "q_center": p.q_center,
                "d_spacing": p.d_spacing,
                "snr": p.snr,
            })
        gvecs_list = []
        for g in gf.g_vectors:
            gvecs_list.append(g.to_dict())

        params["global_fft"] = {
            "d_dom": gf.d_dom,
            "information_limit_q": gf.information_limit_q,
            "n_peaks_found": len(gf.peaks),
            "peaks": peaks_list,
            "g_vectors": gvecs_list,
            "fft_guidance_strength": gf.fft_guidance_strength,
        }

        # Background diagnostics
        if gf.diagnostics:
            bg_diag = {}
            if "baseline_model" in gf.diagnostics:
                bg_diag["baseline_model"] = _make_serialisable(gf.diagnostics["baseline_model"])
            if "background_diagnostics" in gf.diagnostics:
                bg_diag.update(_make_serialisable(gf.diagnostics["background_diagnostics"]))
            if bg_diag:
                params["background_diagnostics"] = bg_diag

    # DC mask
    dc_cfg = config.dc_mask
    dc_section: Dict[str, Any] = {"enabled": dc_cfg.enabled}
    if dc_cfg.enabled and global_fft_result is not None:
        dc_section["dynamic_dc_q"] = global_fft_result.dynamic_dc_q
        dc_section["method"] = dc_cfg.method
        dc_section["q_dc_min_floor"] = dc_cfg.q_dc_min_floor
        dc_section["soft_taper"] = dc_cfg.soft_taper
        # Include diagnostics from global FFT if available
        if global_fft_result.diagnostics and "dc_mask" in global_fft_result.diagnostics:
            dc_section["diagnostics"] = _make_serialisable(
                global_fft_result.diagnostics["dc_mask"]
            )
    params["dc_mask"] = dc_section

    # Tile FFT
    if gated_grid is not None:
        ts = gated_grid.tier_summary
        d_dom = None
        if global_fft_result is not None:
            d_dom = global_fft_result.d_dom

        d_px = (d_dom / config.pixel_size_nm) if d_dom and config.pixel_size_nm > 0 else 0
        periods = config.tile_size / d_px if d_px > 0 else 0

        params["tile_fft"] = {
            "periods_per_tile": periods,
            "d_dom_used": d_dom,
            "tiles_processed": int(np.sum(~gated_grid.skipped_mask)),
            "tiles_skipped_roi": int(np.sum(gated_grid.skipped_mask)),
        }

        params["tier_config"] = {
            "tier_a_snr": config.tier.tier_a_snr,
            "tier_b_snr": config.tier.tier_b_snr,
        }

        params["tier_summary"] = {
            "n_tier_a": ts.n_tier_a,
            "n_tier_b": ts.n_tier_b,
            "n_rejected": ts.n_rejected,
            "n_skipped": ts.n_skipped,
            "tier_a_fraction": ts.tier_a_fraction,
            "median_snr_tier_a": ts.median_snr_tier_a,
        }

        params["peak_gates"] = {
            "max_fwhm_ratio": config.peak_gates.max_fwhm_ratio,
            "min_pair_fraction": config.peak_gates.min_pair_fraction,
            "symmetry_score": config.peak_gates.min_pair_fraction,
            "symmetry_score_deprecated": True,
            "min_non_collinear": config.peak_gates.min_non_collinear,
        }

    # Detection confidence (diagnostic only)
    if gated_grid is not None and gated_grid.detection_confidence_map is not None:
        conf_map = gated_grid.detection_confidence_map
        valid = conf_map[~gated_grid.skipped_mask]
        cc = config.confidence
        params["detection_confidence"] = {
            "note": "Ordinal diagnostic score for visualization. Not calibrated as probability.",
            "weights": {
                "snr": cc.w_snr,
                "pair_fraction": cc.w_pair_fraction,
                "orientation_confidence": cc.w_orientation_confidence,
                "non_collinearity": cc.w_non_collinearity,
                "fwhm_quality": cc.w_fwhm_quality,
            },
            "normalization": {
                "snr_floor": config.tier.tier_b_snr,
                "snr_ceil": 2.0 * config.tier.tier_a_snr,
                "fwhm_reference": config.peak_gates.max_fwhm_ratio,
                "non_collinear_threshold": config.peak_gates.min_non_collinear,
            },
            "statistics": {
                "mean": float(np.mean(valid)) if len(valid) > 0 else 0.0,
                "median": float(np.median(valid)) if len(valid) > 0 else 0.0,
                "std": float(np.std(valid)) if len(valid) > 0 else 0.0,
                "min": float(np.min(valid)) if len(valid) > 0 else 0.0,
                "max": float(np.max(valid)) if len(valid) > 0 else 0.0,
            },
        }

    # GPA
    if gpa_result is not None:
        gpa_section: Dict[str, Any] = {
            "config": {
                "mode": config.gpa.mode,
                "on_fail": config.gpa.on_fail,
            },
            "mode_selected": gpa_result.mode,
        }
        if gpa_result.mode_decision:
            gpa_section["mode_decision_metrics"] = _make_serialisable(
                gpa_result.mode_decision.decision_metrics
            )
            gpa_section["decision_confidence"] = gpa_result.mode_decision.decision_confidence

        gvecs_used = []
        for key, phase in gpa_result.phases.items():
            gv = phase.g_vector
            gvecs_used.append({"gx": gv.gx, "gy": gv.gy})
        gpa_section["g_vectors_used"] = gvecs_used
        gpa_section["mask_radius_q"] = gpa_result.diagnostics.get("mask_sigma_q")
        gpa_section["displacement_smooth_sigma"] = config.gpa.displacement_smooth_sigma
        gpa_section["amplitude_threshold"] = config.gpa.amplitude_threshold

        if gpa_result.reference_region is not None:
            ref = gpa_result.reference_region
            gpa_section["reference_region"] = {
                "center": list(ref.center_tile),
                "area_tiles": len(ref.tiles),
                "entropy": ref.entropy,
                "mean_snr": ref.mean_snr,
            }

        # Phase noise and unwrap success per g-vector
        phase_noise = {}
        unwrap_success = {}
        for key, phase in gpa_result.phases.items():
            if phase.phase_noise_sigma is not None:
                phase_noise[key] = phase.phase_noise_sigma
            unwrap_success[key] = phase.unwrap_success_fraction
        gpa_section["phase_noise"] = phase_noise
        gpa_section["unwrap_success"] = unwrap_success

        # Strain QC
        for qc_key in ("ref_strain_mean_exx", "ref_strain_mean_eyy", "strain_outlier_fraction"):
            if qc_key in gpa_result.qc:
                gpa_section[qc_key] = gpa_result.qc[qc_key]

        params["gpa"] = gpa_section

    # Peak finding
    if lattice_validation is not None:
        params["peak_finding"] = {
            "min_separation_px_used": lattice_validation.min_separation_px_used,
            "expected_d_nm": lattice_validation.expected_d_nm,
            "n_peaks_found": len(lattice_validation.nn_distances),
            "fraction_lattice_valid": lattice_validation.fraction_valid,
            "mean_nn_distance_nm": lattice_validation.mean_nn_distance_nm,
            "tolerance_used": lattice_validation.tolerance_used,
        }

    # Gates
    if validation_report is not None:
        gates_section = {}
        for gid, gr in sorted(validation_report.gates.items()):
            gates_section[gid] = {
                "passed": gr.passed,
                "value": _make_serialisable(gr.value),
                "behavior": gr.failure_behavior,
                "reason": gr.reason,
            }
        params["gates"] = gates_section

        # Diagnostics
        if validation_report.diagnostics:
            params["diagnostics"] = _make_serialisable(validation_report.diagnostics)

    # Derived cutoffs
    derived = {
        "effective_q_min_global": round(effective_q_min, 4),
        "effective_q_min_tile": round(tile_effective_q_min, 4),
    }
    if global_fft_result is not None and global_fft_result.diagnostics:
        bm = global_fft_result.diagnostics.get("baseline_model", {})
        if bm:
            derived["q_fit_min_used"] = bm.get("q_fit_min", 0)
    if gpa_result is not None and gpa_result.diagnostics:
        derived["mask_sigma_q_used"] = gpa_result.diagnostics.get("mask_sigma_q")
    if lattice_validation is not None:
        derived["nn_tolerance_used"] = lattice_validation.tolerance_used
    params["derived_cutoffs"] = derived

    # Ring analysis
    if ring_maps is not None:
        params["ring_analysis"] = {
            "n_rings": ring_maps.n_rings,
            "rings": [
                {"ring_index": ri.ring_index, "q_center": ri.q_center,
                 "d_spacing": ri.d_spacing, "q_width": ri.q_width}
                for ri in ring_maps.rings
            ],
        }

    # Domain clustering
    if clustering_result is not None:
        clust_section = {
            "method": clustering_result.method_used,
            "n_clusters": clustering_result.n_clusters,
            "silhouette_score": clustering_result.silhouette_score,
            "embedding_method": clustering_result.embedding_method,
        }
        if clustering_result.adjacency_pre:
            clust_section["adjacency_pre"] = clustering_result.adjacency_pre
        if clustering_result.adjacency_post:
            clust_section["adjacency_post"] = clustering_result.adjacency_post
        if clustering_result.silhouette_curve:
            clust_section["silhouette_curve"] = [
                {"k": k, "score": s} for k, s in clustering_result.silhouette_curve
            ]
        if clustering_result.cluster_summaries:
            clust_section["cluster_summaries"] = _make_serialisable(
                clustering_result.cluster_summaries)
        params["domain_clustering"] = clust_section

    # Pipeline flow
    flow: Dict[str, Any] = {"stages_completed": [], "stages_skipped": [],
                             "stages_degraded": [], "skip_reasons": {}}
    # Reconstruct flow from available data
    flow["stages_completed"].append("preprocessing")
    flow["stages_completed"].append("roi_masking")
    if global_fft_result is not None:
        flow["stages_completed"].append("global_fft")
    if gated_grid is not None:
        flow["stages_completed"].append("tile_fft")
    if ring_maps is not None:
        flow["stages_completed"].append("ring_analysis")
    else:
        flow["stages_skipped"].append("ring_analysis")
    if clustering_result is not None and clustering_result.n_clusters > 0:
        flow["stages_completed"].append("domain_clustering")
    else:
        flow["stages_skipped"].append("domain_clustering")
    if gpa_result is not None:
        flow["stages_completed"].append("gpa")
    elif config.gpa.enabled:
        flow["stages_skipped"].append("gpa")
        if global_fft_result is not None and global_fft_result.fft_guidance_strength == "none":
            flow["skip_reasons"]["gpa"] = "fft_guidance_strength=none"
    if lattice_validation is not None:
        flow["stages_completed"].append("peak_finding")
    elif config.peak_finding.enabled:
        flow["stages_skipped"].append("peak_finding")
    # Check for degraded stages
    if roi_result is not None and roi_result.diagnostics.get("roi_confidence") == "low":
        flow["stages_degraded"].append("roi")
    if validation_report is not None:
        for gid, gr in validation_report.gates.items():
            if not gr.passed and gr.failure_behavior == "DEGRADE_CONFIDENCE":
                flow["stages_degraded"].append(f"gate_{gid}")
    params["pipeline_flow"] = flow

    # Extra user-supplied data
    if extra:
        for k, v in extra.items():
            if k not in params:
                params[k] = _make_serialisable(v)

    return params


# ======================================================================
# Report.json
# ======================================================================

def build_report(validation_report: ValidationReport) -> dict:
    """Build report.json from a ValidationReport."""
    gates_out = {}
    for gid, gr in sorted(validation_report.gates.items()):
        gates_out[gid] = {
            "name": gr.name,
            "passed": gr.passed,
            "value": _make_serialisable(gr.value),
            "threshold": _make_serialisable(gr.threshold),
            "failure_behavior": gr.failure_behavior,
            "reason": gr.reason,
        }

    report = {
        "overall_pass": validation_report.overall_pass,
        "summary": validation_report.summary,
        "timestamp": validation_report.timestamp,
        "gates": gates_out,
    }

    if validation_report.tier_summary is not None:
        ts = validation_report.tier_summary
        report["tier_summary"] = {
            "n_tier_a": ts.n_tier_a,
            "n_tier_b": ts.n_tier_b,
            "n_rejected": ts.n_rejected,
            "n_skipped": ts.n_skipped,
            "tier_a_fraction": ts.tier_a_fraction,
            "median_snr_tier_a": ts.median_snr_tier_a,
        }

    if validation_report.diagnostics:
        report["diagnostics"] = _make_serialisable(validation_report.diagnostics)
        # Include pipeline_flow if present in diagnostics
        if "pipeline_flow" in validation_report.diagnostics:
            report["pipeline_flow"] = _make_serialisable(
                validation_report.diagnostics["pipeline_flow"]
            )

    return report


# ======================================================================
# Artifact saving
# ======================================================================

def save_json(data: dict, path: Path) -> None:
    """Save a dict as JSON with consistent formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved %s", path)


def save_npy(arr: np.ndarray, path: Path) -> None:
    """Save a NumPy array."""
    np.save(path, arr)
    logger.info("Saved %s", path)


def save_pipeline_artifacts(
    output_dir: Path,
    *,
    config: PipelineConfig,
    fft_grid: FFTGrid,
    preproc_record: Optional[FFTPreprocRecord] = None,
    seg_record: Optional[SegPreprocRecord] = None,
    roi_result: Optional[ROIMaskResult] = None,
    global_fft_result: Optional[GlobalFFTResult] = None,
    gated_grid: Optional[GatedTileGrid] = None,
    gpa_result: Optional[GPAResult] = None,
    lattice_validation: Optional[LatticeValidation] = None,
    peaks: Optional[List[SubpixelPeak]] = None,
    validation_report: Optional[ValidationReport] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    effective_q_min: float = 0.0,
    tile_effective_q_min: float = 0.0,
    ring_maps=None,
    ring_features=None,
    tile_avg_fft=None,
    clustering_result=None,
) -> Dict[str, Path]:
    """Save all pipeline artifacts to output directory.

    Returns dict mapping artifact name → file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    # --- parameters.json (v3.0) ---
    params_data = build_parameters_v3(
        config=config,
        fft_grid=fft_grid,
        preproc_record=preproc_record,
        seg_record=seg_record,
        roi_result=roi_result,
        global_fft_result=global_fft_result,
        gated_grid=gated_grid,
        gpa_result=gpa_result,
        lattice_validation=lattice_validation,
        validation_report=validation_report,
        extra=extra_params,
        effective_q_min=effective_q_min,
        tile_effective_q_min=tile_effective_q_min,
        ring_maps=ring_maps,
        clustering_result=clustering_result,
    )
    params_path = output_dir / "parameters.json"
    save_json(params_data, params_path)
    saved["parameters.json"] = params_path

    # --- report.json ---
    if validation_report is not None:
        report_data = build_report(validation_report)
        report_path = output_dir / "report.json"
        save_json(report_data, report_path)
        saved["report.json"] = report_path

    # --- global_g_vectors.json (C8) ---
    if global_fft_result is not None and global_fft_result.g_vectors:
        gvecs_data = {
            "frequency_unit": "cycles/nm",
            "g_vectors": [g.to_dict() for g in global_fft_result.g_vectors],
        }
        gvec_path = output_dir / "global_g_vectors.json"
        save_json(gvecs_data, gvec_path)
        saved["global_g_vectors.json"] = gvec_path

    # --- gpa_mode_decision.json (C10) ---
    if gpa_result is not None and gpa_result.mode_decision is not None:
        decision_data = _make_serialisable(gpa_result.mode_decision.to_dict())
        decision_path = output_dir / "gpa_mode_decision.json"
        save_json(decision_data, decision_path)
        saved["gpa_mode_decision.json"] = decision_path

    # --- gpa_reference.json ---
    if gpa_result is not None and gpa_result.reference_region is not None:
        ref = gpa_result.reference_region
        ref_data = {
            "center_tile": list(ref.center_tile),
            "n_tiles": len(ref.tiles),
            "bounding_box": list(ref.bounding_box),
            "orientation_mean": ref.orientation_mean,
            "orientation_std": ref.orientation_std,
            "mean_snr": ref.mean_snr,
            "entropy": ref.entropy,
            "score": ref.score,
        }
        ref_path = output_dir / "gpa_reference.json"
        save_json(ref_data, ref_path)
        saved["gpa_reference.json"] = ref_path

    # --- tier_map.npy ---
    if gated_grid is not None:
        # Encode: 0=skip, 1=rejected, 2=B, 3=A
        tier_map = gated_grid.tier_map
        encoded = np.zeros(tier_map.shape, dtype=np.uint8)
        encoded[gated_grid.skipped_mask] = 0
        encoded[tier_map == "REJECTED"] = 1
        encoded[tier_map == "B"] = 2
        encoded[tier_map == "A"] = 3
        tier_path = output_dir / "tier_map.npy"
        save_npy(encoded, tier_path)
        saved["tier_map.npy"] = tier_path

        # --- detection_confidence_map.npy ---
        if gated_grid.detection_confidence_map is not None:
            conf_path = output_dir / "detection_confidence_map.npy"
            save_npy(gated_grid.detection_confidence_map.astype(np.float32), conf_path)
            saved["detection_confidence_map.npy"] = conf_path

    # --- lattice_peaks.npy ---
    if peaks is not None and len(peaks) > 0:
        peak_arr = np.array(
            [(p.x, p.y, p.intensity, p.sigma_x, p.sigma_y) for p in peaks],
            dtype=np.float64,
        )
        peaks_path = output_dir / "lattice_peaks.npy"
        save_npy(peak_arr, peaks_path)
        saved["lattice_peaks.npy"] = peaks_path

    # --- peak_stats.json ---
    if lattice_validation is not None:
        peak_stats = {
            "fraction_lattice_valid": lattice_validation.fraction_valid,
            "mean_nn_distance_nm": lattice_validation.mean_nn_distance_nm,
            "std_nn_distance_nm": lattice_validation.std_nn_distance_nm,
            "expected_d_nm": lattice_validation.expected_d_nm,
            "min_separation_px_used": lattice_validation.min_separation_px_used,
            "n_peaks": len(lattice_validation.nn_distances),
        }
        stats_path = output_dir / "peak_stats.json"
        save_json(peak_stats, stats_path)
        saved["peak_stats.json"] = stats_path

    # --- ring_maps.npz ---
    if ring_maps is not None:
        ring_data = {}
        for ri in range(ring_maps.n_rings):
            ring_data[f"presence_{ri}"] = ring_maps.presence[ri]
            ring_data[f"peak_count_{ri}"] = ring_maps.peak_count[ri]
            ring_data[f"orientation_{ri}"] = ring_maps.orientation[ri]
            ring_data[f"snr_{ri}"] = ring_maps.snr[ri]
        ring_path = output_dir / "ring_maps.npz"
        np.savez_compressed(ring_path, **ring_data)
        saved["ring_maps.npz"] = ring_path
        logger.info("Saved %s", ring_path)

    # --- ring_feature_vectors.npy + ring_feature_names.json ---
    if ring_features is not None:
        fv_path = output_dir / "ring_feature_vectors.npy"
        save_npy(ring_features.feature_matrix, fv_path)
        saved["ring_feature_vectors.npy"] = fv_path
        fn_path = output_dir / "ring_feature_names.json"
        save_json(ring_features.feature_names, fn_path)
        saved["ring_feature_names.json"] = fn_path

    # --- average_tile_fft.npy + average_tile_radial_profile.json ---
    if tile_avg_fft is not None:
        avg_fft_path = output_dir / "average_tile_fft.npy"
        save_npy(tile_avg_fft["mean_power"], avg_fft_path)
        saved["average_tile_fft.npy"] = avg_fft_path
        avg_rp_path = output_dir / "average_tile_radial_profile.json"
        save_json({
            "q_values": tile_avg_fft["q_values"].tolist(),
            "profile": tile_avg_fft["radial_profile"].tolist(),
            "n_tiles": tile_avg_fft["n_tiles"],
        }, avg_rp_path)
        saved["average_tile_radial_profile.json"] = avg_rp_path

    # --- clustering artifacts ---
    if clustering_result is not None and clustering_result.n_clusters > 0:
        cl_path = output_dir / "cluster_labels.npy"
        save_npy(clustering_result.tile_labels, cl_path)
        saved["cluster_labels.npy"] = cl_path

        clr_path = output_dir / "cluster_labels_regularized.npy"
        save_npy(clustering_result.tile_labels_regularized, clr_path)
        saved["cluster_labels_regularized.npy"] = clr_path

        if clustering_result.cluster_averaged_ffts:
            fft_data = {}
            for lid, fft_info in clustering_result.cluster_averaged_ffts.items():
                fft_data[f"cluster_{lid}_power"] = fft_info["mean_power"]
                fft_data[f"cluster_{lid}_radial"] = fft_info["radial_profile"]
            fft_data["q_values"] = next(iter(
                clustering_result.cluster_averaged_ffts.values()))["q_values"]
            cafft_path = output_dir / "cluster_averaged_ffts.npz"
            np.savez_compressed(cafft_path, **fft_data)
            saved["cluster_averaged_ffts.npz"] = cafft_path
            logger.info("Saved %s", cafft_path)

        # clustering_results.json
        cr_data = {
            "method": clustering_result.method_used,
            "n_clusters": clustering_result.n_clusters,
            "silhouette_score": clustering_result.silhouette_score,
            "embedding_method": clustering_result.embedding_method,
        }
        if clustering_result.adjacency_pre:
            cr_data["adjacency_pre"] = clustering_result.adjacency_pre
        if clustering_result.adjacency_post:
            cr_data["adjacency_post"] = clustering_result.adjacency_post
        if clustering_result.silhouette_curve:
            cr_data["silhouette_curve"] = [
                {"k": k, "score": s} for k, s in clustering_result.silhouette_curve
            ]
        if clustering_result.cluster_summaries:
            cr_data["cluster_summaries"] = _make_serialisable(
                clustering_result.cluster_summaries)
        cr_path = output_dir / "clustering_results.json"
        save_json(cr_data, cr_path)
        saved["clustering_results.json"] = cr_path

    logger.info("Saved %d artifacts to %s", len(saved), output_dir)
    return saved
