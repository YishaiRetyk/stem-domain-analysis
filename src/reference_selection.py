"""
GPA Reference Region Selection (WS5).

Selects a reference region from Tier A tiles only (C2).
Scores regions by: 0.4*(1-entropy) + 0.3*snr_norm + 0.3*area_norm.

Gate G9: area >= 9, entropy <= 0.3, mean_snr >= 5.0.
"""

import logging
import numpy as np
from scipy import ndimage

from src.pipeline_config import ReferenceRegion, GatedTileGrid
from src.gates import evaluate_gate

logger = logging.getLogger(__name__)


def select_reference_region(gated_grid: GatedTileGrid,
                            min_area: int = 9,
                            max_entropy: float = 0.3,
                            min_snr: float = 5.0) -> ReferenceRegion:
    """Select the best reference region from Tier A tiles.

    Uses connected components of Tier A tiles.
    Score = 0.4*(1-entropy) + 0.3*snr_norm + 0.3*area_norm.

    Returns
    -------
    ReferenceRegion
        Best region. Raises ValueError if no valid region found.
    """
    tier_map = gated_grid.tier_map
    snr_map = gated_grid.snr_map
    orientation_map = gated_grid.orientation_map

    tier_a_mask = (tier_map == "A")
    labeled, n_components = ndimage.label(tier_a_mask)

    if n_components == 0:
        raise ValueError("No Tier A tiles found for reference selection")

    # Evaluate each connected component
    candidates = []
    max_area = 0

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        tiles = list(zip(*np.where(comp_mask)))
        area = len(tiles)

        if area < min_area:
            continue

        max_area = max(max_area, area)

        # Orientation entropy (12 bins, normalised)
        orientations = orientation_map[comp_mask]
        valid_orient = orientations[~np.isnan(orientations)]
        if len(valid_orient) > 0:
            hist, _ = np.histogram(valid_orient, bins=12, range=(0, 180))
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)) / np.log2(12))
        else:
            entropy = 1.0

        # Mean SNR
        mean_snr = float(np.mean(snr_map[comp_mask]))

        # Centre tile (closest to component centroid)
        rows, cols = np.where(comp_mask)
        centroid_r = np.mean(rows)
        centroid_c = np.mean(cols)
        dists = (rows - centroid_r) ** 2 + (cols - centroid_c) ** 2
        center_idx = np.argmin(dists)
        center_tile = (int(rows[center_idx]), int(cols[center_idx]))

        # Bounding box
        bbox = (int(rows.min()), int(rows.max()),
                int(cols.min()), int(cols.max()))

        candidates.append({
            "tiles": tiles,
            "area": area,
            "entropy": entropy,
            "mean_snr": mean_snr,
            "center_tile": center_tile,
            "bbox": bbox,
            "orientation_mean": float(np.nanmean(valid_orient)) if len(valid_orient) > 0 else 0,
            "orientation_std": float(np.nanstd(valid_orient)) if len(valid_orient) > 0 else 0,
        })

    if not candidates:
        raise ValueError(f"No connected Tier A region with area >= {min_area}")

    # Normalise and score
    max_snr_val = max(c["mean_snr"] for c in candidates)
    max_area_val = max(c["area"] for c in candidates)

    for c in candidates:
        snr_norm = c["mean_snr"] / (max_snr_val + 1e-10)
        area_norm = c["area"] / (max_area_val + 1e-10)
        c["score"] = 0.4 * (1.0 - c["entropy"]) + 0.3 * snr_norm + 0.3 * area_norm

    # Select best
    best = max(candidates, key=lambda c: c["score"])

    ref = ReferenceRegion(
        center_tile=best["center_tile"],
        tiles=best["tiles"],
        bounding_box=best["bbox"],
        orientation_mean=best["orientation_mean"],
        orientation_std=best["orientation_std"],
        mean_snr=best["mean_snr"],
        entropy=best["entropy"],
        score=best["score"],
    )

    # Gate G9
    g9_result = evaluate_gate("G9", {
        "area": best["area"],
        "entropy": best["entropy"],
        "snr": best["mean_snr"],
    })

    logger.info("Reference region: center=%s, area=%d, entropy=%.2f, snr=%.1f, score=%.3f",
                ref.center_tile, len(ref.tiles), ref.entropy, ref.mean_snr, ref.score)

    if not g9_result.passed:
        logger.warning("G9 failed: %s", g9_result.reason)

    return ref


def check_ref_region_exists(gated_grid: GatedTileGrid,
                            min_area: int = 9) -> bool:
    """Quick pre-check: does a valid reference region exist? (I4)."""
    tier_a_mask = (gated_grid.tier_map == "A")
    labeled, n_components = ndimage.label(tier_a_mask)
    for comp_id in range(1, n_components + 1):
        if np.sum(labeled == comp_id) >= min_area:
            return True
    return False
