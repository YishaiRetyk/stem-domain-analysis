"""
Per-Ring Spatial Maps, Feature Vectors, and Tile-Averaged FFTs.

Builds per-ring presence/count/orientation/SNR maps from classified tiles,
constructs feature vectors for clustering, and computes streaming tile-averaged
and cluster-averaged FFT spectra.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pipeline_config import GatedTileGrid, GlobalPeak, RingAnalysisConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Dataclasses
# ======================================================================

@dataclass
class RingInfo:
    """Metadata for a single diffraction ring."""
    ring_index: int
    q_center: float       # cycles/nm
    d_spacing: float      # nm
    q_width: float        # cycles/nm (half-width used for matching)


@dataclass
class RingMaps:
    """Per-ring 2D spatial maps."""
    n_rings: int
    rings: List[RingInfo]
    presence: Dict[int, np.ndarray]          # ring_idx -> (n_rows, n_cols) float [0,1]
    peak_count: Dict[int, np.ndarray]        # ring_idx -> (n_rows, n_cols) int
    orientation: Dict[int, np.ndarray]       # ring_idx -> (n_rows, n_cols) float deg
    snr: Dict[int, np.ndarray]               # ring_idx -> (n_rows, n_cols) float
    angular_variance: Dict[int, np.ndarray]  # ring_idx -> (n_rows, n_cols) float
    grid_shape: Tuple[int, int]


@dataclass
class RingFeatureVectors:
    """Feature matrix built from ring maps for clustering."""
    feature_matrix: np.ndarray    # (n_tiles, n_features)
    feature_names: List[str]
    tile_positions: np.ndarray    # (n_tiles, 2) row,col
    valid_mask: np.ndarray        # (n_tiles,) bool
    grid_shape: Tuple[int, int]
    n_rings: int


# ======================================================================
# Ring map construction
# ======================================================================

def build_ring_maps(
    gated_grid: GatedTileGrid,
    global_peaks: List[GlobalPeak],
    ring_config: RingAnalysisConfig = None,
) -> RingMaps:
    """Build per-ring spatial maps from classified tile grid.

    Iterates over all tiles, extracts per-peak ring_index from the
    TileClassification.peaks metric dicts, and accumulates per-ring
    presence, count, orientation, and SNR maps.
    """
    if ring_config is None:
        ring_config = RingAnalysisConfig()

    n_rows, n_cols = gated_grid.grid_shape
    n_rings = len(global_peaks)

    rings = []
    for i, gp in enumerate(global_peaks):
        q_width = (max(gp.q_fwhm * ring_config.ring_width_fwhm_mult,
                       gp.q_center * ring_config.ring_width_fallback_frac)
                   if gp.q_fwhm > 0
                   else gp.q_center * ring_config.ring_width_no_fwhm_frac)
        rings.append(RingInfo(
            ring_index=i,
            q_center=gp.q_center,
            d_spacing=gp.d_spacing,
            q_width=q_width,
        ))

    presence = {i: np.zeros((n_rows, n_cols), dtype=np.float64) for i in range(n_rings)}
    peak_count = {i: np.zeros((n_rows, n_cols), dtype=np.int32) for i in range(n_rings)}
    orientation = {i: np.full((n_rows, n_cols), np.nan, dtype=np.float64) for i in range(n_rings)}
    snr_map = {i: np.zeros((n_rows, n_cols), dtype=np.float64) for i in range(n_rings)}
    angular_var = {i: np.zeros((n_rows, n_cols), dtype=np.float64) for i in range(n_rings)}

    for r in range(n_rows):
        for c in range(n_cols):
            if gated_grid.skipped_mask[r, c]:
                continue
            tc = gated_grid.classifications[r, c]
            if tc is None or tc.tier == "REJECTED":
                continue

            # Group peaks by ring
            ring_peaks: Dict[int, list] = {}
            for pm in tc.peaks:
                ri = pm.get("ring_index", -1)
                if 0 <= ri < n_rings:
                    ring_peaks.setdefault(ri, []).append(pm)

            for ri, plist in ring_peaks.items():
                presence[ri][r, c] = 1.0
                peak_count[ri][r, c] = len(plist)

                # Best SNR peak for orientation and SNR
                best = max(plist, key=lambda p: p.get("snr", 0))
                snr_map[ri][r, c] = best.get("snr", 0)

                # Mean orientation (circular mean for 180-deg periodicity)
                angles = [p.get("qx", 0) for p in plist]  # placeholder
                angles_deg = []
                for p in plist:
                    qx, qy = p.get("qx", 0), p.get("qy", 0)
                    angles_deg.append(float(np.degrees(np.arctan2(qy, qx))) % 180)
                if angles_deg:
                    # Circular mean for doubled angles
                    doubled = np.array(angles_deg) * np.pi / 90
                    mean_angle = float(np.degrees(np.arctan2(
                        np.mean(np.sin(doubled)),
                        np.mean(np.cos(doubled))
                    )) / 2) % 180
                    orientation[ri][r, c] = mean_angle

                    # Circular variance
                    R = np.sqrt(np.mean(np.cos(doubled))**2 + np.mean(np.sin(doubled))**2)
                    angular_var[ri][r, c] = 1.0 - R

    return RingMaps(
        n_rings=n_rings,
        rings=rings,
        presence=presence,
        peak_count=peak_count,
        orientation=orientation,
        snr=snr_map,
        angular_variance=angular_var,
        grid_shape=(n_rows, n_cols),
    )


# ======================================================================
# Feature vector construction
# ======================================================================

def build_ring_feature_vectors(
    gated_grid: GatedTileGrid,
    ring_maps: RingMaps,
) -> RingFeatureVectors:
    """Build per-tile feature vectors from ring maps.

    Features per ring (n_rings * 5):
        - presence (0/1)
        - peak count
        - mean SNR
        - mean orientation
        - angular variance

    Global features (4):
        - total peak count
        - best SNR across all rings
        - pair_fraction
        - orientation_confidence
    """
    n_rows, n_cols = ring_maps.grid_shape
    n_rings = ring_maps.n_rings
    per_ring = 5
    n_global = 4
    n_features = n_rings * per_ring + n_global

    feature_names = []
    for ri in range(n_rings):
        feature_names.extend([
            f"ring{ri}_presence",
            f"ring{ri}_count",
            f"ring{ri}_snr",
            f"ring{ri}_orientation",
            f"ring{ri}_angular_var",
        ])
    feature_names.extend([
        "total_peak_count",
        "best_snr",
        "pair_fraction",
        "orientation_confidence",
    ])

    n_tiles = n_rows * n_cols
    features = np.zeros((n_tiles, n_features), dtype=np.float64)
    positions = np.zeros((n_tiles, 2), dtype=np.int32)
    valid = np.zeros(n_tiles, dtype=bool)

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            positions[idx] = [r, c]
            if gated_grid.skipped_mask[r, c]:
                idx += 1
                continue

            tc = gated_grid.classifications[r, c]
            if tc is None or tc.tier == "REJECTED":
                idx += 1
                continue

            valid[idx] = True
            f = 0
            for ri in range(n_rings):
                features[idx, f] = ring_maps.presence[ri][r, c]
                features[idx, f + 1] = ring_maps.peak_count[ri][r, c]
                features[idx, f + 2] = ring_maps.snr[ri][r, c]
                orient = ring_maps.orientation[ri][r, c]
                features[idx, f + 3] = orient if not np.isnan(orient) else 0.0
                features[idx, f + 4] = ring_maps.angular_variance[ri][r, c]
                f += per_ring

            # Global features
            features[idx, f] = sum(
                ring_maps.peak_count[ri][r, c] for ri in range(n_rings)
            )
            features[idx, f + 1] = gated_grid.snr_map[r, c]
            features[idx, f + 2] = gated_grid.pair_fraction_map[r, c]
            oc_map = gated_grid.orientation_confidence_map
            features[idx, f + 3] = oc_map[r, c] if oc_map is not None else 0.0

            idx += 1

    return RingFeatureVectors(
        feature_matrix=features,
        feature_names=feature_names,
        tile_positions=positions,
        valid_mask=valid,
        grid_shape=(n_rows, n_cols),
        n_rings=n_rings,
    )


# ======================================================================
# Tile-averaged FFT (all tiles)
# ======================================================================

def compute_tile_averaged_fft(
    image_fft: np.ndarray,
    tile_size: int,
    stride: int,
    pixel_size_nm: float,
    skipped_mask: np.ndarray,
    effective_q_min: float = 0.0,
) -> dict:
    """Streaming mean of all valid tile power spectra.

    Parameters
    ----------
    effective_q_min : float
        Bins with q < effective_q_min are zeroed (DC / low-q suppression).

    Returns dict with mean_power (tile_size, tile_size),
    radial_profile, q_values, n_tiles, effective_q_min.
    """
    from src.fft_coords import FFTGrid
    from src.fft_features import get_tiling_info, create_2d_hann_window

    H, W = image_fft.shape
    info = get_tiling_info(image_fft.shape, tile_size, stride)
    n_rows, n_cols = info["grid_shape"]
    window = create_2d_hann_window(tile_size)

    # Pre-compute low-q mask
    grid = FFTGrid(tile_size, tile_size, pixel_size_nm)
    q_mag = grid.q_mag_grid()
    lowq_mask = q_mag < effective_q_min if effective_q_min > 0 else None

    mean_power = np.zeros((tile_size, tile_size), dtype=np.float64)
    n_tiles = 0

    for r in range(n_rows):
        for c in range(n_cols):
            if r < skipped_mask.shape[0] and c < skipped_mask.shape[1]:
                if skipped_mask[r, c]:
                    continue
            y0 = r * stride
            x0 = c * stride
            if y0 + tile_size > H or x0 + tile_size > W:
                continue
            tile = image_fft[y0:y0 + tile_size, x0:x0 + tile_size]
            windowed = tile.astype(np.float64) * window
            fft_shifted = np.fft.fftshift(np.fft.fft2(windowed))
            power = np.abs(fft_shifted) ** 2
            if lowq_mask is not None:
                power[lowq_mask] = 0.0
            mean_power += power
            n_tiles += 1

    if n_tiles > 0:
        mean_power /= n_tiles

    # Compute radial profile (skip low-q bins)
    q_max = q_mag.max()
    n_bins = tile_size // 2
    bin_edges = np.linspace(0, q_max, n_bins + 1)
    q_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    radial_profile = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if effective_q_min > 0 and hi <= effective_q_min:
            continue  # skip bins entirely within exclusion zone
        mask = (q_mag >= max(lo, effective_q_min)) & (q_mag < hi)
        if np.any(mask):
            radial_profile[i] = np.mean(mean_power[mask])

    return {
        "mean_power": mean_power,
        "radial_profile": radial_profile,
        "q_values": q_values,
        "n_tiles": n_tiles,
        "effective_q_min": effective_q_min,
    }


# ======================================================================
# Cluster-averaged FFTs (streaming)
# ======================================================================

def compute_cluster_averaged_ffts(
    cluster_labels: np.ndarray,
    image_fft: np.ndarray,
    tile_size: int,
    stride: int,
    pixel_size_nm: float,
    skipped_mask: np.ndarray,
    effective_q_min: float = 0.0,
) -> Dict[int, dict]:
    """Streaming per-cluster mean power spectra.

    Parameters
    ----------
    cluster_labels : (n_rows, n_cols) int array
    image_fft : Branch A image
    tile_size, stride : tiling params
    pixel_size_nm : pixel size
    skipped_mask : (n_rows, n_cols) bool
    effective_q_min : float
        Bins with q < effective_q_min are zeroed (DC / low-q suppression).

    Returns
    -------
    Dict[cluster_id, {"mean_power", "radial_profile", "q_values", "n_tiles"}]
    """
    from src.fft_coords import FFTGrid
    from src.fft_features import get_tiling_info, create_2d_hann_window

    H, W = image_fft.shape
    info = get_tiling_info(image_fft.shape, tile_size, stride)
    n_rows, n_cols = info["grid_shape"]
    window = create_2d_hann_window(tile_size)

    # Pre-compute low-q mask
    grid = FFTGrid(tile_size, tile_size, pixel_size_nm)
    q_mag = grid.q_mag_grid()
    lowq_mask = q_mag < effective_q_min if effective_q_min > 0 else None

    unique_labels = set(int(x) for x in np.unique(cluster_labels) if x >= 0)
    accum = {
        lid: {"sum_power": np.zeros((tile_size, tile_size), dtype=np.float64), "count": 0}
        for lid in unique_labels
    }

    for r in range(n_rows):
        for c in range(n_cols):
            if r < skipped_mask.shape[0] and c < skipped_mask.shape[1]:
                if skipped_mask[r, c]:
                    continue
            if r >= cluster_labels.shape[0] or c >= cluster_labels.shape[1]:
                continue
            lid = int(cluster_labels[r, c])
            if lid < 0:
                continue
            y0 = r * stride
            x0 = c * stride
            if y0 + tile_size > H or x0 + tile_size > W:
                continue
            tile = image_fft[y0:y0 + tile_size, x0:x0 + tile_size]
            windowed = tile.astype(np.float64) * window
            fft_shifted = np.fft.fftshift(np.fft.fft2(windowed))
            power = np.abs(fft_shifted) ** 2
            if lowq_mask is not None:
                power[lowq_mask] = 0.0
            accum[lid]["sum_power"] += power
            accum[lid]["count"] += 1

    # Convert to results
    q_max = q_mag.max()
    n_bins = tile_size // 2
    bin_edges = np.linspace(0, q_max, n_bins + 1)
    q_values = (bin_edges[:-1] + bin_edges[1:]) / 2

    results = {}
    for lid, acc in accum.items():
        n = acc["count"]
        if n == 0:
            continue
        mean_power = acc["sum_power"] / n
        radial_profile = np.zeros(n_bins)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if effective_q_min > 0 and hi <= effective_q_min:
                continue
            mask = (q_mag >= max(lo, effective_q_min)) & (q_mag < hi)
            if np.any(mask):
                radial_profile[i] = np.mean(mean_power[mask])
        results[lid] = {
            "mean_power": mean_power,
            "radial_profile": radial_profile,
            "q_values": q_values,
            "n_tiles": n,
        }

    return results


# ======================================================================
# Cluster physics summaries
# ======================================================================

def compute_cluster_summaries(
    cluster_labels: np.ndarray,
    ring_maps: RingMaps,
    gated_grid: GatedTileGrid,
) -> Dict[int, dict]:
    """Per-cluster physics: dominant ring presences, mean orientations, peak counts.

    Returns {cluster_id: {"dominant_rings", "mean_orientations", "peak_count_dist", "n_tiles"}}
    """
    n_rows, n_cols = ring_maps.grid_shape
    unique_labels = set(int(x) for x in np.unique(cluster_labels) if x >= 0)

    summaries = {}
    for lid in sorted(unique_labels):
        mask = cluster_labels == lid
        n_tiles = int(np.sum(mask))

        # Ring presences: fraction of tiles in cluster with each ring present
        dominant_rings = {}
        mean_orientations = {}
        peak_count_dist = {}

        for ri in range(ring_maps.n_rings):
            ring_presence = ring_maps.presence[ri][mask]
            frac = float(np.mean(ring_presence)) if n_tiles > 0 else 0.0
            dominant_rings[ri] = frac

            # Mean orientation for tiles with this ring
            orient = ring_maps.orientation[ri][mask]
            valid_orient = orient[~np.isnan(orient)]
            if len(valid_orient) > 0:
                doubled = valid_orient * np.pi / 90
                mean_angle = float(np.degrees(np.arctan2(
                    np.mean(np.sin(doubled)),
                    np.mean(np.cos(doubled))
                )) / 2) % 180
                mean_orientations[ri] = mean_angle
            else:
                mean_orientations[ri] = None

            # Peak count distribution
            counts = ring_maps.peak_count[ri][mask]
            peak_count_dist[ri] = {
                "mean": float(np.mean(counts)),
                "max": int(np.max(counts)),
                "nonzero_fraction": float(np.mean(counts > 0)),
            }

        summaries[lid] = {
            "dominant_rings": dominant_rings,
            "mean_orientations": mean_orientations,
            "peak_count_dist": peak_count_dist,
            "n_tiles": n_tiles,
        }

    return summaries
