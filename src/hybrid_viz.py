"""PNG Visualization Artifacts for the Hybrid Pipeline.

Generates publication-quality PNG images from pipeline results.
Reuses utilities from src/viz.py (scale bars, etc.).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

logger = logging.getLogger(__name__)


def save_pipeline_visualizations(
    output_dir,
    *,
    config,
    fft_grid,
    raw_image=None,
    global_fft_result=None,
    gated_grid=None,
    gpa_result=None,
    peaks=None,
    lattice_validation=None,
    bandpass_image=None,
    validation_report=None,
    effective_q_min: float = 0.0,
    roi_result=None,
    seg_record=None,
) -> Dict[str, Path]:
    """Orchestrate generation of all PNG visualization artifacts.

    Each plot is guarded by data availability and wrapped in try/except
    so that a single plotting failure does not abort the pipeline.

    Returns dict mapping artifact name to saved file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dpi = config.viz.dpi
    pixel_size_nm = config.pixel_size_nm
    saved: Dict[str, Path] = {}

    def _try(name, fn, *a, **kw):
        try:
            path = fn(*a, **kw)
            if path is not None:
                saved[name] = path
        except Exception:
            logger.warning("Failed to save %s", name, exc_info=True)

    # 1. Input original
    if raw_image is not None:
        _try("input_original",
             _save_input_original, raw_image, out, pixel_size_nm, dpi)

    # 2-3. Global FFT radial profile + power spectrum
    if global_fft_result is not None:
        _try("globalfft_radial_profile",
             _save_radial_profile, global_fft_result, out, dpi,
             effective_q_min)
        _try("globalfft_power_spectrum",
             _save_fft_power_spectrum, global_fft_result, fft_grid, out, dpi,
             effective_q_min)

    # 4-6. Tile maps
    if gated_grid is not None:
        _try("tiles_tier_map",
             _save_tier_map, gated_grid, out, dpi)
        _try("tiles_snr_map",
             _save_snr_map, gated_grid, out, dpi)
        _try("tiles_orientation_map",
             _save_orientation_map, gated_grid, out, dpi)

        if gated_grid.detection_confidence_map is not None:
            _try("detection_confidence_heatmap",
                 _save_detection_confidence_heatmap, gated_grid, out, dpi)

    # 7-10. GPA phase + amplitude
    if gpa_result is not None and gpa_result.phases:
        for i, (key, phase_result) in enumerate(gpa_result.phases.items()):
            _try(f"gpa_phase_g{i}",
                 _save_gpa_phase, phase_result, i, out, dpi)
            _try(f"gpa_amplitude_g{i}",
                 _save_gpa_amplitude, phase_result, i, out, dpi)

    # 11-12. Displacement
    if gpa_result is not None and gpa_result.displacement is not None:
        _try("gpa_displacement_ux",
             _save_displacement, gpa_result.displacement.ux, "ux", out, dpi)
        _try("gpa_displacement_uy",
             _save_displacement, gpa_result.displacement.uy, "uy", out, dpi)

    # 13-16. Strain
    if gpa_result is not None and gpa_result.strain is not None:
        s = gpa_result.strain
        for comp_name, comp_data in [("exx", s.exx), ("eyy", s.eyy),
                                      ("exy", s.exy), ("rotation", s.rotation)]:
            _try(f"gpa_strain_{comp_name}",
                 _save_strain_component, comp_data, comp_name, out, dpi)

    # 17. Peak lattice overlay
    if peaks is not None and len(peaks) > 0 and bandpass_image is not None:
        _try("peaks_lattice_overlay",
             _save_peak_lattice_overlay, peaks, lattice_validation,
             bandpass_image, pixel_size_nm, out, dpi)

    # --- New diagnostic visualizations ---

    # 18. ROI mask overlay
    if roi_result is not None and raw_image is not None:
        _try("roi_mask_overlay",
             _save_roi_mask_overlay, raw_image, roi_result, out, dpi)

    # 19. Background residual
    if global_fft_result is not None and global_fft_result.diagnostics:
        _try("globalfft_background_residual",
             _save_background_residual, global_fft_result, out, dpi)

    # 20. Reference boundary overlay on tier map
    if gated_grid is not None and gpa_result is not None and gpa_result.reference_region is not None:
        _try("tiles_reference_boundary",
             _save_reference_boundary, gated_grid, gpa_result.reference_region, out, dpi)

    # 21. NN distance histogram
    if lattice_validation is not None and len(lattice_validation.nn_distances) > 0:
        _try("peaks_nn_distance_histogram",
             _save_nn_distance_histogram, lattice_validation, out, dpi)

    # 22. GPA strain outlier map
    if gpa_result is not None and gpa_result.strain is not None:
        _try("gpa_strain_outlier_map",
             _save_strain_outlier_map, gpa_result, config, out, dpi)

    # 23. GPA phase noise histogram
    if gpa_result is not None and gpa_result.phases and gpa_result.reference_region is not None:
        _try("gpa_phase_noise_histogram",
             _save_phase_noise_histogram, gpa_result, config, out, dpi)

    # 24. Tile exemplar FFTs
    if gated_grid is not None and raw_image is not None:
        _try("tiles_exemplar_ffts",
             _save_tile_exemplar_ffts, gated_grid, raw_image, config, fft_grid, out, dpi)

    return saved


def _ensure_dir(path):
    """Create parent directory if it doesn't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Individual plotting functions
# ======================================================================

def _save_input_original(image, out_dir, pixel_size_nm, dpi):
    """Save the raw input image with scale bar."""
    from src.viz import save_image_with_scalebar
    path = str(out_dir / "input_original.png")
    save_image_with_scalebar(image, path, pixel_size_nm, cmap="gray",
                             title="Input Image", colorbar=False)
    return Path(path)


def _save_radial_profile(global_fft_result, out_dir, dpi,
                         effective_q_min=0.0):
    """Radial profile: q vs log-intensity, background curve, peak markers."""
    path = out_dir / "globalfft_radial_profile.png"
    _ensure_dir(path)
    r = global_fft_result

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(r.q_values, r.radial_profile, 'k-', lw=0.8, label="Radial profile")
    ax.semilogy(r.q_values, r.background, 'b--', lw=0.8, label="Background fit")

    # Shade excluded low-q region
    if effective_q_min > 0:
        ax.axvspan(0, effective_q_min, alpha=0.15, color='gray',
                   label=f"Excluded (q < {effective_q_min:.2f})")

    for p in r.peaks:
        ax.axvline(p.q_center, color="red", alpha=0.5, lw=0.7)
        ax.annotate(f"d={p.d_spacing:.3f} nm",
                    xy=(p.q_center, p.intensity),
                    xytext=(5, 10), textcoords="offset points",
                    fontsize=7, color="red")

    ax.set_xlabel("q (cycles/nm)")
    ax.set_ylabel("Intensity (log)")
    ax.set_title("Global FFT Radial Profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_fft_power_spectrum(global_fft_result, fft_grid, out_dir, dpi,
                             effective_q_min=0.0):
    """Log-scale power spectrum with g-vector arrows."""
    path = out_dir / "globalfft_power_spectrum.png"
    _ensure_dir(path)
    ps = global_fft_result.power_spectrum

    fig, ax = plt.subplots(figsize=(8, 8))
    display = np.log1p(ps)
    im = ax.imshow(display, cmap="inferno", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(Power + 1)")

    # Dashed circle at q_min exclusion boundary
    cy, cx = ps.shape[0] // 2, ps.shape[1] // 2
    if effective_q_min > 0:
        # Convert q_min to pixel radius: r_px = q_min / q_scale
        q_scale = min(fft_grid.qx_scale, fft_grid.qy_scale)
        r_px = effective_q_min / q_scale
        circle = plt.Circle((cx, cy), r_px, fill=False, edgecolor='white',
                             linestyle='--', linewidth=1.0, alpha=0.8)
        ax.add_patch(circle)

    # Annotate g-vectors as arrows from centre
    for gv in global_fft_result.g_vectors:
        # Convert cycles/nm to pixel coords in the power spectrum
        # gx,gy are in cycles/nm; power spectrum pixel = g / delta_q
        # delta_q = 1 / (N * pixel_size)
        px = cx + gv.gx * ps.shape[1] * fft_grid.pixel_size_nm
        py = cy + gv.gy * ps.shape[0] * fft_grid.pixel_size_nm
        ax.annotate("",
                    xy=(px, py), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))
        ax.annotate(f"d={gv.d_spacing:.2f}",
                    xy=(px, py), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="cyan")

    ax.set_title("Global FFT Power Spectrum")
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_tier_map(gated_grid, out_dir, dpi):
    """Discrete tier map: green=A, yellow=B, red=rejected, gray=skipped."""
    path = out_dir / "tiles_tier_map.png"
    _ensure_dir(path)
    tm = gated_grid.tier_map  # (n_rows, n_cols) str array

    # Encode as int: A=0, B=1, REJECTED=2, skipped/empty=3
    int_map = np.full(tm.shape, 3, dtype=int)
    int_map[tm == "A"] = 0
    int_map[tm == "B"] = 1
    int_map[tm == "REJECTED"] = 2

    colors = ["#2ecc71", "#f1c40f", "#e74c3c", "#95a5a6"]  # green, yellow, red, gray
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(int_map, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label=f"Tier A ({np.sum(tm == 'A')})"),
        Patch(facecolor="#f1c40f", label=f"Tier B ({np.sum(tm == 'B')})"),
        Patch(facecolor="#e74c3c", label=f"Rejected ({np.sum(tm == 'REJECTED')})"),
        Patch(facecolor="#95a5a6", label=f"Skipped ({np.sum((tm != 'A') & (tm != 'B') & (tm != 'REJECTED'))})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Tile Tier Map")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_snr_map(gated_grid, out_dir, dpi):
    """Viridis SNR heatmap with colorbar."""
    path = out_dir / "tiles_snr_map.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(8, 8))
    snr = gated_grid.snr_map.astype(float)
    snr_masked = np.ma.masked_where(gated_grid.skipped_mask, snr)
    im = ax.imshow(snr_masked, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Peak SNR")
    ax.set_title("Tile SNR Map")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_orientation_map(gated_grid, out_dir, dpi):
    """HSV cyclic orientation map, masked where skipped/rejected."""
    path = out_dir / "tiles_orientation_map.png"
    _ensure_dir(path)

    orient = gated_grid.orientation_map.astype(float)
    mask = gated_grid.skipped_mask | (gated_grid.tier_map == "REJECTED")
    orient_masked = np.ma.masked_where(mask, orient)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(orient_masked, cmap="hsv", vmin=0, vmax=180,
                   origin="upper", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Orientation (deg)")
    ax.set_title("Tile Orientation Map")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_detection_confidence_heatmap(gated_grid, out_dir, dpi):
    """Detection confidence heatmap (ordinal, diagnostic only)."""
    path = out_dir / "detection_confidence_heatmap.png"
    _ensure_dir(path)

    conf = gated_grid.detection_confidence_map.astype(float)
    conf_masked = np.ma.masked_where(gated_grid.skipped_mask, conf)

    cmap = plt.cm.magma.copy()
    cmap.set_bad("gray")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(conf_masked, cmap=cmap, vmin=0, vmax=1,
                   origin="upper", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Detection Confidence (ordinal)")
    ax.set_title("Detection Confidence Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_gpa_phase(phase_result, g_index, out_dir, dpi):
    """Unwrapped phase for one g-vector with twilight cmap."""
    path = out_dir / f"gpa_phase_g{g_index}.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(8, 8))
    phase = phase_result.phase_unwrapped.copy().astype(float)
    phase[~phase_result.amplitude_mask] = np.nan

    cmap = plt.cm.twilight.copy()
    cmap.set_bad("gray")
    im = ax.imshow(phase, cmap=cmap, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Phase (rad)")
    d = phase_result.g_vector.d_spacing
    ax.set_title(f"GPA Phase g{g_index} (d={d:.3f} nm)")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_gpa_amplitude(phase_result, g_index, out_dir, dpi):
    """Amplitude with mask contour overlay."""
    path = out_dir / f"gpa_amplitude_g{g_index}.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(phase_result.amplitude, cmap="inferno", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Amplitude")

    # Contour of amplitude mask boundary
    ax.contour(phase_result.amplitude_mask.astype(float),
               levels=[0.5], colors="cyan", linewidths=1.0)

    d = phase_result.g_vector.d_spacing
    ax.set_title(f"GPA Amplitude g{g_index} (d={d:.3f} nm)")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_displacement(field_component, label, out_dir, dpi):
    """Displacement field component (ux or uy), RdBu_r centred at 0."""
    path = out_dir / f"gpa_displacement_{label}.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = max(abs(np.nanmin(field_component)), abs(np.nanmax(field_component)))
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(field_component, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{label} (nm)")
    ax.set_title(f"GPA Displacement {label}")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_strain_component(data, comp_name, out_dir, dpi):
    """Strain component with RdBu_r, symmetric +/-5%."""
    path = out_dir / f"gpa_strain_{comp_name}.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(8, 8))
    if comp_name == "rotation":
        # Rotation in radians — convert to degrees for display
        display = np.degrees(data)
        vmax = 5.0  # +/-5 degrees
        label = "Rotation (deg)"
    else:
        display = data
        vmax = 0.05  # +/-5%
        label = comp_name

    im = ax.imshow(display, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title(f"GPA Strain: {comp_name}")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_peak_lattice_overlay(peaks, lattice_validation, bandpass_image,
                                pixel_size_nm, out_dir, dpi):
    """Peak positions + NN connections coloured by validity."""
    from scipy.spatial import cKDTree
    path = out_dir / "peaks_lattice_overlay.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bandpass_image, cmap="gray", origin="upper")

    xs = np.array([p.x for p in peaks])
    ys = np.array([p.y for p in peaks])

    # Draw NN connections
    if lattice_validation is not None and len(peaks) >= 2:
        expected_d_px = lattice_validation.expected_d_nm / pixel_size_nm
        tol = 0.2  # same tolerance used in pipeline
        tree = cKDTree(np.column_stack([xs, ys]))
        pairs = tree.query_pairs(expected_d_px * (1 + tol))

        for i, j in pairs:
            dist = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
            valid = abs(dist - expected_d_px) / expected_d_px < tol
            color = "lime" if valid else "red"
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color=color, lw=0.5, alpha=0.6)

    ax.scatter(xs, ys, s=8, c="cyan", edgecolors="none", zorder=5)
    ax.set_title(f"Peak Lattice Overlay ({len(peaks)} peaks)")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ======================================================================
# New diagnostic visualizations (Stage 6)
# ======================================================================

def _save_roi_mask_overlay(raw_image, roi_result, out_dir, dpi):
    """ROI boundary contour overlaid on raw image."""
    path = out_dir / "roi_mask_overlay.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(raw_image, cmap="gray", origin="upper")
    ax.contour(roi_result.mask_full.astype(float), levels=[0.5],
               colors="lime", linewidths=1.0)
    ax.set_title(f"ROI Mask (coverage={roi_result.coverage_pct:.1f}%, "
                 f"LCC={roi_result.lcc_fraction:.2f})")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_background_residual(global_fft_result, out_dir, dpi):
    """Background residual: profile - background with q_fit_min line."""
    path = out_dir / "globalfft_background_residual.png"
    _ensure_dir(path)

    r = global_fft_result
    if r.q_values is None or r.radial_profile is None or r.background is None:
        return None

    residual = r.radial_profile - r.background

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r.q_values, residual, 'k-', lw=0.8, label="Residual")
    ax.axhline(0, color='gray', ls='--', lw=0.5)

    # Mark q_fit_min if available
    bm = r.diagnostics.get("baseline_model", {}) if r.diagnostics else {}
    q_fit_min = bm.get("q_fit_min", 0)
    if q_fit_min > 0:
        ax.axvline(q_fit_min, color='blue', ls=':', lw=1.0,
                   label=f"q_fit_min={q_fit_min:.2f}")

    for p in r.peaks:
        ax.axvline(p.q_center, color="red", alpha=0.5, lw=0.7)

    ax.set_xlabel("q (cycles/nm)")
    ax.set_ylabel("Residual (profile - background)")
    ax.set_title("Global FFT Background Residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_reference_boundary(gated_grid, reference_region, out_dir, dpi):
    """Tier map with reference region contour overlaid."""
    path = out_dir / "tiles_reference_boundary.png"
    _ensure_dir(path)

    tm = gated_grid.tier_map
    int_map = np.full(tm.shape, 3, dtype=int)
    int_map[tm == "A"] = 0
    int_map[tm == "B"] = 1
    int_map[tm == "REJECTED"] = 2

    colors = ["#2ecc71", "#f1c40f", "#e74c3c", "#95a5a6"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(int_map, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")

    ref_mask = np.zeros(tm.shape, dtype=float)
    for (r, c) in reference_region.tiles:
        if 0 <= r < tm.shape[0] and 0 <= c < tm.shape[1]:
            ref_mask[r, c] = 1.0
    ax.contour(ref_mask, levels=[0.5], colors="cyan", linewidths=2.0)

    ax.set_title("Tier Map with Reference Region")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_nn_distance_histogram(lattice_validation, out_dir, dpi):
    """NN distance distribution with expected-d vertical line."""
    path = out_dir / "peaks_nn_distance_histogram.png"
    _ensure_dir(path)

    nn = np.array(lattice_validation.nn_distances)
    expected = lattice_validation.expected_d_nm
    tol = lattice_validation.tolerance_used

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nn, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(expected, color="red", ls="-", lw=1.5,
               label=f"Expected d={expected:.3f} nm")
    ax.axvspan(expected * (1 - tol), expected * (1 + tol),
               alpha=0.15, color="red", label=f"Tolerance ±{tol:.0%}")

    ax.set_xlabel("NN Distance (nm)")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-Neighbor Distance Distribution")
    ax.legend(fontsize=8)

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_strain_outlier_map(gpa_result, config, out_dir, dpi):
    """Binary overlay of pixels exceeding strain_outlier_threshold."""
    path = out_dir / "gpa_strain_outlier_map.png"
    _ensure_dir(path)

    strain = gpa_result.strain
    threshold = config.gpa.strain_outlier_threshold

    outlier_mask = (np.abs(strain.exx) > threshold) | (np.abs(strain.eyy) > threshold)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(outlier_mask.astype(float), cmap="Reds", origin="upper",
              vmin=0, vmax=1)
    ax.set_title(f"Strain Outlier Map (threshold={threshold:.3f})")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_phase_noise_histogram(gpa_result, config, out_dir, dpi):
    """Histogram of phase values per g-vector."""
    path = out_dir / "gpa_phase_noise_histogram.png"
    _ensure_dir(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, phase_result in gpa_result.phases.items():
        if phase_result.phase_noise_sigma is not None:
            phase = phase_result.phase_unwrapped.copy()
            valid = phase_result.amplitude_mask & ~np.isnan(phase)
            if np.any(valid):
                vals = phase[valid]
                ax.hist(vals, bins=50, alpha=0.5,
                        label=f"{key} (σ={phase_result.phase_noise_sigma:.3f} rad)")

    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Count")
    ax.set_title("GPA Phase Distribution")
    ax.legend(fontsize=8)

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_tile_exemplar_ffts(gated_grid, raw_image, config, fft_grid, out_dir, dpi):
    """3x2 grid showing representative tile power spectra."""
    path = out_dir / "tiles_exemplar_ffts.png"
    _ensure_dir(path)

    tile_size = config.tile_size
    stride = config.stride
    H, W = raw_image.shape

    tier_map = gated_grid.tier_map
    tiers = {"A": [], "B": [], "REJECTED": []}
    for r in range(tier_map.shape[0]):
        for c in range(tier_map.shape[1]):
            t = tier_map[r, c]
            if t in tiers:
                tiers[t].append((r, c))

    rng = np.random.default_rng(42)
    exemplars = []
    labels = []
    for tier_name in ("A", "B", "REJECTED"):
        candidates = tiers[tier_name]
        if len(candidates) >= 2:
            chosen = rng.choice(len(candidates), size=2, replace=False)
            for idx in chosen:
                exemplars.append(candidates[idx])
                labels.append(f"Tier {tier_name}")
        elif len(candidates) == 1:
            exemplars.append(candidates[0])
            labels.append(f"Tier {tier_name}")

    if not exemplars:
        return None

    n_plots = min(len(exemplars), 6)
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_plots):
        r, c = exemplars[idx]
        y0, x0 = r * stride, c * stride
        y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
        tile = raw_image[y0:y1, x0:x1]

        window = np.hanning(tile.shape[0])[:, None] * np.hanning(tile.shape[1])[None, :]
        ft = np.fft.fftshift(np.fft.fft2(tile * window))
        ps = np.log1p(np.abs(ft) ** 2)

        ax_r, ax_c = idx // n_cols, idx % n_cols
        ax = axes[ax_r, ax_c]
        ax.imshow(ps, cmap="inferno", origin="upper")
        ax.set_title(f"{labels[idx]} ({r},{c})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_plots, n_rows * n_cols):
        ax_r, ax_c = idx // n_cols, idx % n_cols
        axes[ax_r, ax_c].set_visible(False)

    fig.suptitle("Tile Exemplar FFTs", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path
