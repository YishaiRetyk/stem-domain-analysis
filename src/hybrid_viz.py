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
    ring_maps=None,
    clustering_result=None,
    tile_avg_fft=None,
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

        _try("peak_detection_heatmap",
             _save_peak_detection_heatmap, gated_grid, config, pixel_size_nm,
             out, dpi, effective_q_min)

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

    # --- Ring analysis visualizations ---
    if ring_maps is not None and global_fft_result is not None:
        _try("ring_presence_maps",
             _save_ring_presence_maps, ring_maps, out, dpi)
        _try("ring_peak_count_maps",
             _save_ring_peak_count_maps, ring_maps, out, dpi)
        _try("ring_orientation_maps",
             _save_ring_orientation_maps, ring_maps, out, dpi)
        if raw_image is not None:
            _try("ring_peak_locations_overlay",
                 _save_ring_peak_locations_overlay, ring_maps, raw_image, config, out, dpi)
            _try("ring_orientation_overlay",
                 _save_ring_orientation_overlay, ring_maps, raw_image, config, out, dpi)

    if tile_avg_fft is not None:
        _try("average_tile_fft",
             _save_average_tile_fft, tile_avg_fft, global_fft_result, out, dpi,
             effective_q_min)
        _try("average_tile_radial_profile",
             _save_average_tile_radial_profile, tile_avg_fft, global_fft_result, out, dpi,
             effective_q_min)

    # --- Clustering visualizations ---
    if clustering_result is not None and clustering_result.n_clusters > 0:
        _try("cluster_label_map",
             _save_cluster_label_map, clustering_result, out, dpi)
        if raw_image is not None:
            _try("cluster_overlay",
                 _save_cluster_overlay, clustering_result, raw_image, config, out, dpi)
        if clustering_result.embedding_2d is not None:
            _try("feature_embedding",
                 _save_feature_embedding_viz, clustering_result, out, dpi)
        if clustering_result.cluster_averaged_ffts:
            _try("cluster_averaged_ffts",
                 _save_cluster_averaged_ffts_viz, clustering_result, out, dpi,
                 effective_q_min)
            _try("cluster_radial_profiles",
                 _save_cluster_radial_profiles, clustering_result, out, dpi,
                 effective_q_min)
        if clustering_result.silhouette_curve:
            _try("silhouette_curve",
                 _save_silhouette_curve, clustering_result, out, dpi)
        if ring_maps is not None:
            _try("ring_vs_cluster_comparison",
                 _save_ring_vs_cluster_comparison, ring_maps, clustering_result, out, dpi)

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

    # Dynamic DC boundary (orange dashed line, distinct from gray low-q shading)
    if r.dynamic_dc_q is not None and r.dynamic_dc_q > 0:
        ax.axvline(r.dynamic_dc_q, color='orange', linestyle='--', lw=1.2,
                   label=f"Dynamic DC ({r.dynamic_dc_q:.2f})")

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
    valid = display[display > 0]
    vmin = np.percentile(valid, 5) if len(valid) > 0 else 0
    im = ax.imshow(display, cmap="inferno", origin="upper", vmin=vmin)
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

    # Solid orange circle at dynamic DC boundary (distinct from low-q dashed)
    if global_fft_result.dynamic_dc_q is not None and global_fft_result.dynamic_dc_q > 0:
        q_scale = min(fft_grid.qx_scale, fft_grid.qy_scale)
        dc_r_px = global_fft_result.dynamic_dc_q / q_scale
        dc_circle = plt.Circle((cx, cy), dc_r_px, fill=False, edgecolor='orange',
                                linestyle='-', linewidth=1.2, alpha=0.9)
        ax.add_patch(dc_circle)

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


def _save_peak_detection_heatmap(gated_grid, config, pixel_size_nm, out_dir, dpi,
                                 effective_q_min=0.0):
    """Accumulate detected FFT peaks in frequency space across all tiles.

    Produces a heatmap showing where peaks cluster in the shared tile-FFT
    coordinate system, revealing crystallographic orientations as bright spots
    on a ring at the dominant d-spacing.
    """
    from scipy.ndimage import gaussian_filter
    path = out_dir / "peak_detection_heatmap.png"
    _ensure_dir(path)

    tile_size = config.tile_size
    accumulator = np.zeros((tile_size, tile_size), dtype=np.float64)
    q_scale = 1.0 / (tile_size * pixel_size_nm)  # cycles/nm per pixel
    cy, cx = tile_size // 2, tile_size // 2

    n_peaks = 0
    n_tiles = 0
    n_rows, n_cols = gated_grid.classifications.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if gated_grid.skipped_mask[r, c]:
                continue
            tc = gated_grid.classifications[r, c]
            if tc is None or tc.tier == "REJECTED":
                continue
            n_tiles += 1
            for pm in tc.peaks:
                qx = pm.get("qx", None)
                qy = pm.get("qy", None)
                if qx is None or qy is None:
                    continue
                px = int(round(cx + qx / q_scale))
                py = int(round(cy + qy / q_scale))
                if 0 <= px < tile_size and 0 <= py < tile_size:
                    accumulator[py, px] += 1
                    n_peaks += 1

    if n_peaks == 0:
        return None

    # Light Gaussian smooth for visual clarity
    smoothed = gaussian_filter(accumulator, sigma=1.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(smoothed, cmap="inferno", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Detection count")

    # Dashed circle at q_min exclusion boundary
    if effective_q_min > 0:
        r_px = effective_q_min / q_scale
        circle = plt.Circle((cx, cy), r_px, fill=False, edgecolor='white',
                             linestyle='--', linewidth=1.0, alpha=0.8)
        ax.add_patch(circle)

    ax.set_title(f"Peak Detection Heatmap ({n_peaks} peaks from {n_tiles} tiles)")

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


# ======================================================================
# Ring analysis visualizations
# ======================================================================

def _save_ring_presence_maps(ring_maps, out_dir, dpi):
    """Grid of subplots: one per ring, viridis [0,1]."""
    path = out_dir / "ring_presence_maps.png"
    _ensure_dir(path)
    n_rings = ring_maps.n_rings
    if n_rings == 0:
        return None

    n_cols = min(n_rings, 3)
    n_rows = (n_rings + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    for i in range(n_rings):
        ax = axes[i // n_cols, i % n_cols]
        im = ax.imshow(ring_maps.presence[i], cmap="viridis", vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")
        ri = ring_maps.rings[i]
        ax.set_title(f"Ring {i} (d={ri.d_spacing:.3f} nm)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_rings, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("Ring Presence Maps", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_ring_peak_count_maps(ring_maps, out_dir, dpi):
    """Grid of subplots: peak counts per ring, integer colorbar."""
    path = out_dir / "ring_peak_count_maps.png"
    _ensure_dir(path)
    n_rings = ring_maps.n_rings
    if n_rings == 0:
        return None

    n_cols = min(n_rings, 3)
    n_rows = (n_rings + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    for i in range(n_rings):
        ax = axes[i // n_cols, i % n_cols]
        data = ring_maps.peak_count[i]
        im = ax.imshow(data, cmap="YlOrRd", vmin=0,
                       origin="upper", interpolation="nearest")
        ri = ring_maps.rings[i]
        ax.set_title(f"Ring {i} peaks (d={ri.d_spacing:.3f} nm)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_rings, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("Ring Peak Count Maps", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_ring_orientation_maps(ring_maps, out_dir, dpi):
    """Grid of subplots: HSV cyclic orientation per ring."""
    path = out_dir / "ring_orientation_maps.png"
    _ensure_dir(path)
    n_rings = ring_maps.n_rings
    if n_rings == 0:
        return None

    n_cols = min(n_rings, 3)
    n_rows = (n_rings + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    for i in range(n_rings):
        ax = axes[i // n_cols, i % n_cols]
        orient = ring_maps.orientation[i].copy()
        masked = np.ma.masked_where(np.isnan(orient), orient)
        im = ax.imshow(masked, cmap="hsv", vmin=0, vmax=180,
                       origin="upper", interpolation="nearest")
        ri = ring_maps.rings[i]
        ax.set_title(f"Ring {i} orient (d={ri.d_spacing:.3f} nm)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="deg")

    for i in range(n_rings, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("Ring Orientation Maps", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_ring_peak_locations_overlay(ring_maps, raw_image, config, out_dir, dpi):
    """Per-ring tile markers overlaid on original image."""
    path = out_dir / "ring_peak_locations_overlay.png"
    _ensure_dir(path)
    n_rings = ring_maps.n_rings
    if n_rings == 0:
        return None

    n_cols = min(n_rings, 3)
    n_rows = (n_rings + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_rings, 1)))

    for i in range(n_rings):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(raw_image, cmap="gray", origin="upper")
        pres = ring_maps.presence[i]
        snr_m = ring_maps.snr[i]
        rows, cols = np.where(pres > 0)
        if len(rows) > 0:
            # Convert tile coords to pixel coords (center of tile)
            py = rows * config.stride + config.tile_size // 2
            px = cols * config.stride + config.tile_size // 2
            sizes = np.clip(snr_m[rows, cols] * 3, 5, 100)
            ax.scatter(px, py, s=sizes, c=[colors[i]], alpha=0.6,
                       edgecolors="none")
        ri = ring_maps.rings[i]
        ax.set_title(f"Ring {i} (d={ri.d_spacing:.3f} nm)", fontsize=9)

    for i in range(n_rings, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("Ring Peak Locations", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_ring_orientation_overlay(ring_maps, raw_image, config, out_dir, dpi):
    """Per-ring orientation coloring overlaid on original image."""
    path = out_dir / "ring_orientation_overlay.png"
    _ensure_dir(path)
    n_rings = ring_maps.n_rings
    if n_rings == 0:
        return None

    n_cols = min(n_rings, 3)
    n_rows = (n_rings + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             squeeze=False)

    for i in range(n_rings):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(raw_image, cmap="gray", origin="upper")
        orient = ring_maps.orientation[i]
        pres = ring_maps.presence[i]
        rows, cols = np.where((pres > 0) & ~np.isnan(orient))
        if len(rows) > 0:
            py = rows * config.stride + config.tile_size // 2
            px = cols * config.stride + config.tile_size // 2
            angles = orient[rows, cols]
            ax.scatter(px, py, c=angles, cmap="hsv", vmin=0, vmax=180,
                       s=20, alpha=0.6, edgecolors="none")
        ri = ring_maps.rings[i]
        ax.set_title(f"Ring {i} orient overlay (d={ri.d_spacing:.3f} nm)", fontsize=9)

    for i in range(n_rings, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("Ring Orientation Overlay", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_average_tile_fft(tile_avg_fft, global_fft_result, out_dir, dpi,
                           effective_q_min=0.0):
    """Annotated all-tiles average power spectrum (log scale)."""
    path = out_dir / "average_tile_fft.png"
    _ensure_dir(path)

    ps = tile_avg_fft["mean_power"]
    fig, ax = plt.subplots(figsize=(8, 8))
    display = np.log1p(ps)
    # Clip vmin to suppress residual low-q glow
    valid = display[display > 0]
    vmin = np.percentile(valid, 5) if len(valid) > 0 else 0
    im = ax.imshow(display, cmap="inferno", origin="upper", vmin=vmin)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(Power + 1)")

    # Draw dashed circle at q_min exclusion boundary
    eq_min = tile_avg_fft.get("effective_q_min", effective_q_min)
    if eq_min > 0:
        cy, cx = ps.shape[0] // 2, ps.shape[1] // 2
        q_vals = tile_avg_fft.get("q_values", None)
        if q_vals is not None and len(q_vals) >= 2:
            dq = q_vals[1] - q_vals[0]  # q_max / n_bins = pixel q_scale
            r_px = eq_min / dq if dq > 0 else 0
            if r_px > 1:
                circle = plt.Circle((cx, cy), r_px, fill=False, edgecolor='white',
                                    linestyle='--', linewidth=1.0, alpha=0.8)
                ax.add_patch(circle)

    ax.set_title(f"Tile-Averaged Power Spectrum ({tile_avg_fft['n_tiles']} tiles)")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_average_tile_radial_profile(tile_avg_fft, global_fft_result, out_dir, dpi,
                                      effective_q_min=0.0):
    """Radial profile from tile-averaged FFT with peak markers."""
    path = out_dir / "average_tile_radial_profile.png"
    _ensure_dir(path)

    q = tile_avg_fft["q_values"]
    profile = tile_avg_fft["radial_profile"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(q, profile, 'k-', lw=0.8, label="Tile-averaged radial profile")

    # Gray shading for low-q exclusion zone
    eq_min = tile_avg_fft.get("effective_q_min", effective_q_min)
    if eq_min > 0:
        ax.axvspan(0, eq_min, alpha=0.15, color='gray',
                   label=f"Excluded (q < {eq_min:.2f})")

    if global_fft_result is not None:
        for p in global_fft_result.peaks:
            ax.axvline(p.q_center, color="red", alpha=0.5, lw=0.7)
            ax.annotate(f"d={p.d_spacing:.3f} nm",
                        xy=(p.q_center, profile[np.argmin(np.abs(q - p.q_center))]
                            if len(q) > 0 else 1),
                        xytext=(5, 10), textcoords="offset points",
                        fontsize=7, color="red")

    ax.set_xlabel("q (cycles/nm)")
    ax.set_ylabel("Intensity (log)")
    ax.set_title(f"Tile-Averaged Radial Profile ({tile_avg_fft['n_tiles']} tiles)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ======================================================================
# Clustering visualizations
# ======================================================================

def _save_cluster_label_map(clustering_result, out_dir, dpi):
    """Discrete colormap cluster label map."""
    path = out_dir / "cluster_label_map.png"
    _ensure_dir(path)

    labels = clustering_result.tile_labels_regularized
    n_clusters = clustering_result.n_clusters

    fig, ax = plt.subplots(figsize=(8, 8))
    unique = sorted(set(int(x) for x in np.unique(labels)))
    if -1 in unique:
        # Map -1 to gray
        colors_list = ["#95a5a6"]
        offset = 1
    else:
        colors_list = []
        offset = 0
    tab_colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    for i in range(n_clusters):
        colors_list.append(tab_colors[i % 10])

    if len(colors_list) > 0:
        cmap = ListedColormap(colors_list)
        # Remap labels for display
        display = np.full_like(labels, 0, dtype=int)
        for i, u in enumerate(unique):
            display[labels == u] = i
        im = ax.imshow(display, cmap=cmap, origin="upper", interpolation="nearest")

        from matplotlib.patches import Patch
        legend_elements = []
        for i, u in enumerate(unique):
            name = "Noise" if u == -1 else f"Cluster {u}"
            count = int(np.sum(labels == u))
            legend_elements.append(Patch(facecolor=colors_list[i], label=f"{name} ({count})"))
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title(f"Cluster Label Map ({n_clusters} clusters)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_cluster_overlay(clustering_result, raw_image, config, out_dir, dpi):
    """Domain boundaries on raw image."""
    path = out_dir / "cluster_overlay.png"
    _ensure_dir(path)

    from src.cluster_domains import upsample_labels
    labels = clustering_result.tile_labels_regularized
    label_image = upsample_labels(labels, raw_image.shape[:2],
                                  config.tile_size, config.stride)

    # Find boundaries
    from scipy import ndimage
    boundaries = np.zeros_like(label_image, dtype=bool)
    for shift in [(0, 1), (1, 0)]:
        shifted = np.roll(label_image, shift, axis=(0, 1))
        boundaries |= (label_image != shifted)

    img_display = raw_image.copy().astype(float)
    p1, p99 = np.percentile(img_display, [1, 99])
    img_display = np.clip((img_display - p1) / (p99 - p1 + 1e-8), 0, 1)
    rgb = np.stack([img_display] * 3, axis=-1)
    rgb[boundaries] = [1, 0, 0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper")
    ax.set_title("Cluster Overlay")
    ax.axis("off")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_feature_embedding_viz(clustering_result, out_dir, dpi):
    """2D PCA/UMAP scatter, colored by cluster."""
    path = out_dir / "feature_embedding.png"
    _ensure_dir(path)

    emb = clustering_result.embedding_2d
    labels = clustering_result.tile_labels.flatten()

    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(set(int(x) for x in np.unique(labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))

    for i, u in enumerate(unique):
        mask = labels == u
        if not np.any(mask):
            continue
        name = "Noise" if u == -1 else f"Cluster {u}"
        c = "gray" if u == -1 else colors[i % 10]
        alpha = 0.3 if u == -1 else 0.7
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[c], s=15,
                   alpha=alpha, label=name, edgecolors="none")

    ax.set_xlabel(f"{clustering_result.embedding_method.upper()} 1")
    ax.set_ylabel(f"{clustering_result.embedding_method.upper()} 2")
    ax.set_title(f"Feature Embedding ({clustering_result.embedding_method.upper()})")
    ax.legend(fontsize=8, loc="best")

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_cluster_averaged_ffts_viz(clustering_result, out_dir, dpi,
                                    effective_q_min=0.0):  # noqa: ARG001
    """Grid of mean power spectra per cluster (log scale).

    effective_q_min kept in signature for consistency; DC suppression is
    handled at data level (ring_analysis.py) and via shared vmin clipping.
    """
    path = out_dir / "cluster_averaged_ffts.png"
    _ensure_dir(path)

    ffts = clustering_result.cluster_averaged_ffts
    n = len(ffts)
    if n == 0:
        return None

    n_cols = min(n, 3)
    n_rows_plot = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows_plot, n_cols, figsize=(5 * n_cols, 4 * n_rows_plot),
                             squeeze=False)

    # Compute shared vmin across all clusters for consistent colorscale
    all_displays = []
    for lid, fft_info in sorted(ffts.items()):
        all_displays.append(np.log1p(fft_info["mean_power"]))
    all_valid = np.concatenate([d[d > 0].ravel() for d in all_displays])
    vmin = np.percentile(all_valid, 5) if len(all_valid) > 0 else 0

    for idx, (lid, fft_info) in enumerate(sorted(ffts.items())):
        ax = axes[idx // n_cols, idx % n_cols]
        ps = all_displays[idx]
        ax.imshow(ps, cmap="inferno", origin="upper", vmin=vmin)
        ax.set_title(f"Cluster {lid} ({fft_info['n_tiles']} tiles)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, n_rows_plot * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Cluster-Averaged Power Spectra", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_cluster_radial_profiles(clustering_result, out_dir, dpi,
                                  effective_q_min=0.0):
    """Overlaid radial profiles colored by cluster."""
    path = out_dir / "cluster_radial_profiles.png"
    _ensure_dir(path)

    ffts = clustering_result.cluster_averaged_ffts
    if not ffts:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ffts), 1)))

    # Gray shading for low-q exclusion zone
    if effective_q_min > 0:
        ax.axvspan(0, effective_q_min, alpha=0.15, color='gray',
                   label=f"Excluded (q < {effective_q_min:.2f})")

    for idx, (lid, fft_info) in enumerate(sorted(ffts.items())):
        ax.semilogy(fft_info["q_values"], fft_info["radial_profile"],
                    color=colors[idx % 10], lw=1.0,
                    label=f"Cluster {lid} ({fft_info['n_tiles']} tiles)")

    ax.set_xlabel("q (cycles/nm)")
    ax.set_ylabel("Intensity (log)")
    ax.set_title("Cluster Radial Profiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_silhouette_curve(clustering_result, out_dir, dpi):
    """Silhouette score vs K plot."""
    path = out_dir / "silhouette_curve.png"
    _ensure_dir(path)

    curve = clustering_result.silhouette_curve
    if not curve:
        return None

    ks = [k for k, _ in curve]
    scores = [s for _, s in curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, 'o-', color="steelblue", lw=1.5)
    ax.axvline(clustering_result.n_clusters, color="red", ls="--", lw=1.0,
               label=f"Selected K={clustering_result.n_clusters}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs K")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def _save_ring_vs_cluster_comparison(ring_maps, clustering_result, out_dir, dpi):
    """Side-by-side: ring presence maps (left) vs cluster map (right)."""
    path = out_dir / "ring_vs_cluster_comparison.png"
    _ensure_dir(path)

    n_rings = ring_maps.n_rings
    n_rows_plot = max(n_rings, 1)
    fig, axes = plt.subplots(n_rows_plot, 2, figsize=(10, 4 * n_rows_plot),
                             squeeze=False)

    for i in range(n_rings):
        ax = axes[i, 0]
        im = ax.imshow(ring_maps.presence[i], cmap="viridis", vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")
        ri = ring_maps.rings[i]
        ax.set_title(f"Ring {i} (d={ri.d_spacing:.3f} nm)", fontsize=9)
        ax.set_ylabel("Row")

    # Right column: cluster map (repeated)
    labels = clustering_result.tile_labels_regularized
    n_clusters = clustering_result.n_clusters
    unique = sorted(set(int(x) for x in np.unique(labels)))
    tab_colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    colors_list = []
    for u in unique:
        if u == -1:
            colors_list.append([0.6, 0.6, 0.6])
        else:
            colors_list.append(tab_colors[u % 10][:3])
    if colors_list:
        cmap_c = ListedColormap(colors_list)
        display = np.full_like(labels, 0, dtype=int)
        for i_u, u in enumerate(unique):
            display[labels == u] = i_u

    for i in range(n_rows_plot):
        ax = axes[i, 1]
        if colors_list:
            ax.imshow(display, cmap=cmap_c, origin="upper", interpolation="nearest")
        ax.set_title(f"Cluster Map ({n_clusters} clusters)", fontsize=9)

    fig.suptitle("Ring Presence vs Cluster Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path
