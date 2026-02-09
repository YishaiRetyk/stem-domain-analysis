"""ilastik Comparison Module (exploratory, optional).

Exports feature stacks, loads external probability maps, and computes
comparison metrics. Never imported by the main pipeline unless the user
passes --ilastik-map.

Agreement with ilastik does not validate correctness of either method;
this is exploratory and qualitative (DC-5).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import ndimage, stats

logger = logging.getLogger(__name__)


def export_feature_stack(gated_grid, output_path):
    """Export a 5-channel feature stack for ilastik comparison.

    Channels: [snr, pair_fraction, orientation_confidence, fwhm, tier_encoded]
    tier_encoded: 0=skipped, 1=REJECTED, 2=B, 3=A

    Returns the saved path.
    """
    n_rows, n_cols = gated_grid.grid_shape
    stack = np.zeros((n_rows, n_cols, 5), dtype=np.float32)

    stack[:, :, 0] = gated_grid.snr_map
    stack[:, :, 1] = gated_grid.pair_fraction_map

    if gated_grid.orientation_confidence_map is not None:
        stack[:, :, 2] = gated_grid.orientation_confidence_map

    stack[:, :, 3] = gated_grid.fwhm_map

    # Tier encoding
    tm = gated_grid.tier_map
    tier_enc = np.zeros((n_rows, n_cols), dtype=np.float32)
    tier_enc[tm == "REJECTED"] = 1
    tier_enc[tm == "B"] = 2
    tier_enc[tm == "A"] = 3
    stack[:, :, 4] = tier_enc

    output_path = Path(output_path)
    np.save(output_path, stack)
    logger.info("Saved feature stack: %s", output_path)
    return output_path


def load_ilastik_probability_map(path):
    """Load an ilastik probability map from .npy or .h5 file.

    Returns a 2D float array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ilastik map not found: {path}")

    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix in (".h5", ".hdf5"):
        import h5py  # lazy import
        with h5py.File(path, "r") as f:
            # Use the first dataset found
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No datasets found in {path}")
            arr = f[keys[0]][:]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix} (expected .npy or .h5)")

    # Squeeze extra dimensions
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    return arr.astype(np.float64)


def compute_comparison_metrics(pipeline_conf, ilastik_prob, valid_mask):
    """Compute comparison metrics between pipeline and ilastik maps.

    These describe correlation, not accuracy (DC-5).

    Returns dict with pearson_r, spearman_r, agreement_fraction.
    """
    if pipeline_conf.shape != ilastik_prob.shape:
        raise ValueError(
            f"Shape mismatch: pipeline {pipeline_conf.shape} vs "
            f"ilastik {ilastik_prob.shape}")

    p = pipeline_conf[valid_mask]
    i = ilastik_prob[valid_mask]

    if len(p) < 2:
        return {
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "agreement_fraction": 0.0,
            "n_valid": int(len(p)),
        }

    pearson_r, _ = stats.pearsonr(p, i)
    spearman_r, _ = stats.spearmanr(p, i)

    # Agreement: both above or both below 0.5
    agree = ((p >= 0.5) & (i >= 0.5)) | ((p < 0.5) & (i < 0.5))
    agreement_fraction = float(np.mean(agree))

    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "agreement_fraction": agreement_fraction,
        "n_valid": int(len(p)),
    }


def save_comparison_overlay(pipeline_conf, ilastik_prob, valid_mask, out_dir, dpi):
    """Save a 3-panel comparison PNG: pipeline / ilastik / difference."""
    import matplotlib.pyplot as plt

    path = Path(out_dir) / "ilastik_comparison_overlay.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    mask_float = valid_mask.astype(float)
    p_masked = np.ma.masked_where(~valid_mask, pipeline_conf)
    i_masked = np.ma.masked_where(~valid_mask, ilastik_prob)
    diff = pipeline_conf - ilastik_prob
    d_masked = np.ma.masked_where(~valid_mask, diff)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cmap = plt.cm.magma.copy()
    cmap.set_bad("gray")

    im0 = axes[0].imshow(p_masked, cmap=cmap, vmin=0, vmax=1, origin="upper")
    axes[0].set_title("Pipeline Confidence")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(i_masked, cmap=cmap, vmin=0, vmax=1, origin="upper")
    axes[1].set_title("ilastik Probability")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff_cmap = plt.cm.RdBu_r.copy()
    diff_cmap.set_bad("gray")
    vmax_d = max(abs(np.nanmin(diff[valid_mask])), abs(np.nanmax(diff[valid_mask]))) if np.any(valid_mask) else 1.0
    if vmax_d == 0:
        vmax_d = 1.0
    im2 = axes[2].imshow(d_masked, cmap=diff_cmap, vmin=-vmax_d, vmax=vmax_d, origin="upper")
    axes[2].set_title("Difference (Pipeline - ilastik)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("ilastik Comparison (exploratory)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def run_ilastik_comparison(gated_grid, ilastik_map_path, output_dir, dpi=150):
    """Orchestrate ilastik comparison: export, load, compare, visualize.

    Returns dict of saved artifact paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    # Export feature stack
    stack_path = output_dir / "ilastik_feature_stack.npy"
    export_feature_stack(gated_grid, stack_path)
    saved["ilastik_feature_stack.npy"] = stack_path

    # Load ilastik map
    ilastik_prob = load_ilastik_probability_map(ilastik_map_path)

    # Resize if needed (ilastik may be pixel-resolution)
    grid_shape = gated_grid.grid_shape
    if ilastik_prob.shape != grid_shape:
        logger.info("Resizing ilastik map from %s to %s", ilastik_prob.shape, grid_shape)
        zoom_factors = (grid_shape[0] / ilastik_prob.shape[0],
                        grid_shape[1] / ilastik_prob.shape[1])
        ilastik_prob = ndimage.zoom(ilastik_prob, zoom_factors, order=1)

    # Get pipeline confidence and valid mask
    pipeline_conf = gated_grid.detection_confidence_map
    if pipeline_conf is None:
        raise ValueError("detection_confidence_map not available on gated_grid")

    valid_mask = ~gated_grid.skipped_mask

    # Compute comparison metrics
    metrics = compute_comparison_metrics(pipeline_conf, ilastik_prob, valid_mask)

    # Save comparison JSON
    comparison_data = {
        "note": "Exploratory comparison, not validation. Metrics describe correlation, not accuracy.",
        "ilastik_map_path": str(ilastik_map_path),
        "ilastik_map_shape_original": list(load_ilastik_probability_map(ilastik_map_path).shape),
        "grid_shape": list(grid_shape),
        "metrics": metrics,
    }
    json_path = output_dir / "ilastik_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    saved["ilastik_comparison.json"] = json_path
    logger.info("Saved: %s", json_path)

    # Save overlay
    overlay_path = save_comparison_overlay(
        pipeline_conf, ilastik_prob, valid_mask, output_dir, dpi)
    saved["ilastik_comparison_overlay.png"] = overlay_path

    return saved
