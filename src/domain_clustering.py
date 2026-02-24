"""
Domain Clustering Pipeline.

Normalizes ring feature vectors, applies PCA/UMAP dimensionality reduction,
runs K-means/GMM/HDBSCAN clustering, and optionally regularizes spatially.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler

from src.pipeline_config import ClusteringConfig
from src.ring_analysis import RingFeatureVectors

logger = logging.getLogger(__name__)


# ======================================================================
# Spatial helpers (previously in cluster_domains.py)
# ======================================================================

def compute_adjacency_metrics(tile_labels: np.ndarray) -> Dict[str, Any]:
    """Compute adjacency-based metrics for spatial coherence."""
    n_rows, n_cols = tile_labels.shape

    h_diff = tile_labels[:, :-1] != tile_labels[:, 1:]
    h_adjacencies = n_rows * (n_cols - 1)
    h_flips = int(np.sum(h_diff))

    v_diff = tile_labels[:-1, :] != tile_labels[1:, :]
    v_adjacencies = (n_rows - 1) * n_cols
    v_flips = int(np.sum(v_diff))

    total_adjacencies = h_adjacencies + v_adjacencies
    total_flips = h_flips + v_flips
    flip_rate = total_flips / total_adjacencies if total_adjacencies > 0 else 0.0

    unique_labels = np.unique(tile_labels)
    micro_domain_count = 0
    for label in unique_labels:
        if label == -1:
            continue
        mask = tile_labels == label
        labeled_regions, n_regions = ndimage.label(mask)
        for region_id in range(1, n_regions + 1):
            if np.sum(labeled_regions == region_id) < 5:
                micro_domain_count += 1

    return {
        "flip_rate": float(flip_rate),
        "n_adjacencies": int(total_adjacencies),
        "n_flips": int(total_flips),
        "micro_domain_count": int(micro_domain_count),
    }


def spatial_regularize(tile_labels: np.ndarray, params: dict) -> np.ndarray:
    """Apply spatial regularization (mode filter + tiny-cluster removal)."""
    iterations = params.get("regularize_iterations", 2)
    min_domain_size = params.get("min_domain_size", 5)
    labels_reg = tile_labels.copy()

    for _ in range(iterations):
        labels_new = labels_reg.copy()
        n_rows, n_cols = labels_reg.shape
        for i in range(n_rows):
            for j in range(n_cols):
                i_min, i_max = max(0, i - 1), min(n_rows, i + 2)
                j_min, j_max = max(0, j - 1), min(n_cols, j + 2)
                neighborhood = labels_reg[i_min:i_max, j_min:j_max].flatten()
                valid = neighborhood[neighborhood >= 0]
                if len(valid) > 0:
                    mode_result = mode(valid, keepdims=False)
                    labels_new[i, j] = mode_result.mode
        labels_reg = labels_new

    unique_labels = np.unique(labels_reg)
    for label in unique_labels:
        if label == -1:
            continue
        mask = labels_reg == label
        labeled_regions, n_regions = ndimage.label(mask)
        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            if np.sum(region_mask) < min_domain_size:
                dilated = ndimage.binary_dilation(region_mask)
                neighbor_mask = dilated & ~region_mask
                if np.any(neighbor_mask):
                    neighbor_labels = labels_reg[neighbor_mask]
                    valid_neighbors = neighbor_labels[neighbor_labels >= 0]
                    if len(valid_neighbors) > 0:
                        mode_result = mode(valid_neighbors, keepdims=False)
                        labels_reg[region_mask] = mode_result.mode
                    else:
                        labels_reg[region_mask] = -1
    return labels_reg


def upsample_labels(tile_labels_reg: np.ndarray, image_shape: Tuple[int, int],
                    tile_size: int, stride: int) -> np.ndarray:
    """Upsample tile labels to full image resolution (nearest-neighbor)."""
    H, W = image_shape
    n_rows, n_cols = tile_labels_reg.shape
    label_image = np.full((H, W), -1, dtype=np.int32)
    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + tile_size, H)
            x_end = min(x_start + tile_size, W)
            label_image[y_start:y_end, x_start:x_end] = tile_labels_reg[i, j]
    return label_image


# ======================================================================
# Result dataclass
# ======================================================================

@dataclass
class ClusteringResult:
    """Output of domain clustering."""
    tile_labels: np.ndarray                  # (n_rows, n_cols) int
    tile_labels_regularized: np.ndarray      # (n_rows, n_cols) int
    n_clusters: int
    method_used: str
    feature_names: List[str]
    embedding_2d: Optional[np.ndarray]       # (n_tiles, 2) or None
    embedding_method: str                    # "pca" | "umap" | "none"
    silhouette_score: Optional[float]
    silhouette_curve: Optional[List[Tuple[int, float]]]  # [(k, score), ...]
    adjacency_pre: Optional[dict]
    adjacency_post: Optional[dict]
    cluster_averaged_ffts: Optional[Dict[int, dict]] = None
    cluster_summaries: Optional[Dict[int, dict]] = None
    diagnostics: dict = field(default_factory=dict)


# ======================================================================
# Main orchestrator
# ======================================================================

def run_domain_clustering(
    ring_features: RingFeatureVectors,
    config: ClusteringConfig,
    image_fft: Optional[np.ndarray] = None,
    tile_size: int = 256,
    stride: int = 128,
    pixel_size_nm: float = 0.127,
    skipped_mask: Optional[np.ndarray] = None,
    effective_q_min: float = 0.0,
) -> ClusteringResult:
    """Run the full domain clustering pipeline.

    1. Extract valid features
    2. Normalize (StandardScaler + NaN imputation)
    3. PCA reduce for clustering
    4. Cluster (K-means / GMM / HDBSCAN)
    5. Spatial regularization (optional)
    6. 2D embedding for visualization
    7. Cluster-averaged FFTs (if image_fft provided)
    """
    valid = ring_features.valid_mask
    n_valid = int(np.sum(valid))

    if n_valid < config.min_valid_tiles:
        logger.warning("Too few valid tiles (%d) for clustering", n_valid)
        n_rows, n_cols = ring_features.grid_shape
        empty_labels = np.full((n_rows, n_cols), -1, dtype=np.int32)
        return ClusteringResult(
            tile_labels=empty_labels,
            tile_labels_regularized=empty_labels,
            n_clusters=0,
            method_used=config.method,
            feature_names=ring_features.feature_names,
            embedding_2d=None,
            embedding_method="none",
            silhouette_score=None,
            silhouette_curve=None,
            adjacency_pre=None,
            adjacency_post=None,
            diagnostics={"error": "too_few_valid_tiles", "n_valid": n_valid},
        )

    features_valid = ring_features.feature_matrix[valid]

    # 1. Normalize
    features_scaled, scaler = _normalize_features(features_valid)

    # 2. PCA reduce for clustering
    features_reduced, pca_embedding, dimred_method = _reduce_dimensions(
        features_scaled, config)

    # 3. Cluster
    labels_valid, n_clusters, sil_score, sil_curve = _cluster(
        features_reduced, config)

    # 4. Map back to grid
    n_rows, n_cols = ring_features.grid_shape
    tile_labels = np.full((n_rows, n_cols), -1, dtype=np.int32)
    valid_idx = 0
    for i in range(n_rows * n_cols):
        if valid[i]:
            r, c = ring_features.tile_positions[i]
            tile_labels[r, c] = labels_valid[valid_idx]
            valid_idx += 1

    # 5. Spatial regularization
    if config.regularize and n_clusters > 1:
        adj_pre = compute_adjacency_metrics(tile_labels)
        tile_labels_reg = spatial_regularize(tile_labels, {
            "regularize_iterations": config.spatial_regularize_iterations,
            "min_domain_size": config.min_domain_size,
        })
        adj_post = compute_adjacency_metrics(tile_labels_reg)
    else:
        tile_labels_reg = tile_labels.copy()
        adj_pre = None
        adj_post = None

    # 6. 2D embedding for visualization
    embedding_2d = pca_embedding  # from PCA reduction
    embedding_method = dimred_method
    if config.dimred_method == "umap" and features_scaled.shape[0] >= 10:
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=config.random_state,
                           n_neighbors=min(config.umap_n_neighbors, features_scaled.shape[0] - 1),
                           min_dist=config.umap_min_dist)
            embedding_2d_full = np.zeros((n_rows * n_cols, 2), dtype=np.float64)
            embedding_2d_full[valid] = reducer.fit_transform(features_scaled)
            embedding_2d = embedding_2d_full
            embedding_method = "umap"
        except ImportError:
            logger.info("UMAP not available, using PCA embedding")
        except Exception as e:
            logger.warning("UMAP failed: %s, using PCA embedding", e)

    # Expand embedding to full grid if PCA
    if embedding_2d is not None and embedding_2d.shape[0] == n_valid:
        full_embedding = np.zeros((n_rows * n_cols, 2), dtype=np.float64)
        full_embedding[valid] = embedding_2d
        embedding_2d = full_embedding

    # 7. Cluster-averaged FFTs
    cluster_ffts = None
    if image_fft is not None and skipped_mask is not None and n_clusters > 0:
        from src.ring_analysis import compute_cluster_averaged_ffts
        cluster_ffts = compute_cluster_averaged_ffts(
            tile_labels_reg, image_fft, tile_size, stride,
            pixel_size_nm, skipped_mask, effective_q_min=effective_q_min)

    return ClusteringResult(
        tile_labels=tile_labels,
        tile_labels_regularized=tile_labels_reg,
        n_clusters=n_clusters,
        method_used=config.method,
        feature_names=ring_features.feature_names,
        embedding_2d=embedding_2d,
        embedding_method=embedding_method,
        silhouette_score=sil_score,
        silhouette_curve=sil_curve,
        adjacency_pre=adj_pre,
        adjacency_post=adj_post,
        cluster_averaged_ffts=cluster_ffts,
        diagnostics={
            "n_valid_tiles": n_valid,
            "n_features": features_valid.shape[1],
            "n_features_reduced": features_reduced.shape[1] if features_reduced is not None else 0,
        },
    )


# ======================================================================
# Normalization
# ======================================================================

def _normalize_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """StandardScaler + NaN/Inf imputation (column median)."""
    features_clean = features.copy()

    # Impute NaN/Inf with column median
    for col in range(features_clean.shape[1]):
        col_data = features_clean[:, col]
        bad = np.isnan(col_data) | np.isinf(col_data)
        if np.any(bad):
            good = col_data[~bad]
            fill = float(np.median(good)) if len(good) > 0 else 0.0
            features_clean[bad, col] = fill

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    return features_scaled, scaler


# ======================================================================
# Dimensionality reduction
# ======================================================================

def _reduce_dimensions(
    features_scaled: np.ndarray,
    config: ClusteringConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """PCA for clustering (retain variance threshold), optional 2D embedding.

    Returns (features_reduced, embedding_2d, method).
    """
    if config.dimred_method == "none" or features_scaled.shape[1] <= 2:
        emb = features_scaled[:, :2] if features_scaled.shape[1] >= 2 else features_scaled
        return features_scaled, emb, "none"

    from sklearn.decomposition import PCA

    # Full PCA for variance analysis
    n_components = min(features_scaled.shape[0], features_scaled.shape[1])
    pca_full = PCA(n_components=n_components, random_state=config.random_state)
    transformed = pca_full.fit_transform(features_scaled)

    # Select components retaining threshold variance
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, config.pca_variance_threshold) + 1)
    n_keep = max(2, min(n_keep, n_components))

    features_reduced = transformed[:, :n_keep]
    embedding_2d = transformed[:, :2]

    logger.info("PCA: %d -> %d components (%.1f%% variance)",
                features_scaled.shape[1], n_keep, cumvar[n_keep - 1] * 100)

    return features_reduced, embedding_2d, "pca"


# ======================================================================
# Clustering
# ======================================================================

def _cluster(
    features: np.ndarray,
    config: ClusteringConfig,
) -> Tuple[np.ndarray, int, Optional[float], Optional[List[Tuple[int, float]]]]:
    """Dispatch to K-means, GMM, or HDBSCAN.

    Returns (labels, n_clusters, silhouette_score, silhouette_curve).
    """
    if config.method == "hdbscan":
        return _cluster_hdbscan(features, config)
    elif config.method == "gmm":
        return _cluster_gmm(features, config)
    else:
        return _cluster_kmeans(features, config)


def _cluster_kmeans(
    features: np.ndarray,
    config: ClusteringConfig,
) -> Tuple[np.ndarray, int, Optional[float], Optional[List[Tuple[int, float]]]]:
    """K-means with optional auto-K via silhouette scan."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = features.shape[0]

    if config.n_clusters > 0:
        # Fixed K
        k = min(config.n_clusters, n_samples)
        km = KMeans(n_clusters=k, random_state=config.random_state, n_init=config.kmeans_n_init)
        labels = km.fit_predict(features)
        sil = _safe_silhouette(features, labels)
        return labels, k, sil, None

    # Auto-K: silhouette scan
    k_max = min(config.n_clusters_max, n_samples - 1)
    if k_max < 2:
        return np.zeros(n_samples, dtype=np.int32), 1, None, None

    sil_curve = []
    best_k = 2
    best_score = -1.0
    best_labels = None

    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=config.random_state, n_init=config.kmeans_n_init)
        labels = km.fit_predict(features)
        score = _safe_silhouette(features, labels)
        if score is not None:
            sil_curve.append((k, score))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=np.int32)
        best_k = 1

    return best_labels, best_k, best_score if best_score > -1 else None, sil_curve or None


def _cluster_gmm(
    features: np.ndarray,
    config: ClusteringConfig,
) -> Tuple[np.ndarray, int, Optional[float], Optional[List[Tuple[int, float]]]]:
    """GMM with optional auto-K via silhouette scan."""
    from sklearn.mixture import GaussianMixture

    n_samples = features.shape[0]

    if config.n_clusters > 0:
        k = min(config.n_clusters, n_samples)
        gmm = GaussianMixture(n_components=k, random_state=config.random_state)
        labels = gmm.fit_predict(features)
        sil = _safe_silhouette(features, labels)
        return labels, k, sil, None

    k_max = min(config.n_clusters_max, n_samples - 1)
    if k_max < 2:
        return np.zeros(n_samples, dtype=np.int32), 1, None, None

    sil_curve = []
    best_k = 2
    best_score = -1.0
    best_labels = None

    for k in range(2, k_max + 1):
        gmm = GaussianMixture(n_components=k, random_state=config.random_state)
        labels = gmm.fit_predict(features)
        score = _safe_silhouette(features, labels)
        if score is not None:
            sil_curve.append((k, score))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=np.int32)
        best_k = 1

    return best_labels, best_k, best_score if best_score > -1 else None, sil_curve or None


def _cluster_hdbscan(
    features: np.ndarray,
    config: ClusteringConfig,
) -> Tuple[np.ndarray, int, Optional[float], Optional[List[Tuple[int, float]]]]:
    """HDBSCAN clustering."""
    from sklearn.cluster import HDBSCAN

    n_samples = features.shape[0]
    min_cluster_size = min(config.hdbscan_min_cluster_size, max(2, n_samples // 3))
    min_samples = min(config.hdbscan_min_samples, min_cluster_size)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(features)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    sil = _safe_silhouette(features, labels)
    return labels, n_clusters, sil, None


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Compute silhouette score, returning None if impossible."""
    n_labels = len(set(labels) - {-1})
    if n_labels < 2 or n_labels >= features.shape[0]:
        return None
    try:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(features, labels))
    except Exception:
        return None
