"""
Domain segmentation via clustering (SA4).

Clusters extracted tile features using HDBSCAN, applies spatial regularization,
and produces full-resolution domain label maps.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def check_feature_health(features: np.ndarray) -> Dict[str, Any]:
    """
    Gate G5: Check features for NaN/Inf and compute health metrics.
    
    Returns dict with health metrics and cleaned features.
    """
    n_tiles, n_features = features.shape
    
    # Check for NaN/Inf
    nan_mask = np.isnan(features)
    inf_mask = np.isinf(features)
    
    nan_count = np.sum(nan_mask)
    inf_count = np.sum(inf_mask)
    nan_tiles = np.sum(np.any(nan_mask, axis=1))
    inf_tiles = np.sum(np.any(inf_mask, axis=1))
    
    health = {
        'n_tiles': n_tiles,
        'n_features': n_features,
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'nan_tiles': int(nan_tiles),
        'inf_tiles': int(inf_tiles),
        'healthy': nan_count == 0 and inf_count == 0
    }
    
    # Clean features - replace NaN/Inf with column median
    features_clean = features.copy()
    if nan_count > 0 or inf_count > 0:
        for col in range(n_features):
            col_data = features[:, col]
            bad_mask = np.isnan(col_data) | np.isinf(col_data)
            if np.any(bad_mask):
                good_vals = col_data[~bad_mask]
                if len(good_vals) > 0:
                    median_val = np.median(good_vals)
                else:
                    median_val = 0.0
                features_clean[bad_mask, col] = median_val
    
    health['features_clean'] = features_clean
    return health


def cluster_tiles(
    features: np.ndarray,
    tile_confidence: np.ndarray,
    params: dict
) -> np.ndarray:
    """
    Cluster tiles using HDBSCAN from sklearn.
    
    Args:
        features: (n_tiles, n_features) feature array
        tile_confidence: (n_tiles,) confidence scores (unused in basic HDBSCAN)
        params: dict with min_cluster_size, min_samples, metric
    
    Returns:
        tile_labels: (n_tiles,) with -1 for noise
    """
    min_cluster_size = params.get('min_cluster_size', 5)
    min_samples = params.get('min_samples', 3)
    metric = params.get('metric', 'euclidean')
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # HDBSCAN clustering
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric
    )
    
    labels = clusterer.fit_predict(features_scaled)
    
    return labels


def compute_adjacency_metrics(tile_labels: np.ndarray) -> Dict[str, Any]:
    """
    Compute adjacency-based metrics for spatial coherence.
    
    Returns:
        dict with flip_rate, n_adjacencies, micro_domain_count
    """
    n_rows, n_cols = tile_labels.shape
    
    # Count horizontal adjacencies with different labels
    h_diff = tile_labels[:, :-1] != tile_labels[:, 1:]
    h_adjacencies = n_rows * (n_cols - 1)
    h_flips = np.sum(h_diff)
    
    # Count vertical adjacencies with different labels
    v_diff = tile_labels[:-1, :] != tile_labels[1:, :]
    v_adjacencies = (n_rows - 1) * n_cols
    v_flips = np.sum(v_diff)
    
    total_adjacencies = h_adjacencies + v_adjacencies
    total_flips = h_flips + v_flips
    flip_rate = total_flips / total_adjacencies if total_adjacencies > 0 else 0.0
    
    # Count micro-domains (connected components with < 5 tiles)
    unique_labels = np.unique(tile_labels)
    micro_domain_count = 0
    
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        mask = tile_labels == label
        labeled_regions, n_regions = ndimage.label(mask)
        for region_id in range(1, n_regions + 1):
            region_size = np.sum(labeled_regions == region_id)
            if region_size < 5:
                micro_domain_count += 1
    
    return {
        'flip_rate': float(flip_rate),
        'n_adjacencies': int(total_adjacencies),
        'n_flips': int(total_flips),
        'micro_domain_count': int(micro_domain_count)
    }


def spatial_regularize(
    tile_labels: np.ndarray,
    params: dict
) -> np.ndarray:
    """
    Apply spatial regularization to smooth label boundaries.
    
    1. Mode filter (most common neighbor label)
    2. Remove tiny clusters (< min_size tiles)
    
    Args:
        tile_labels: (n_rows, n_cols) label array
        params: dict with regularize_iterations, min_domain_size
    
    Returns:
        regularized labels (n_rows, n_cols)
    """
    iterations = params.get('regularize_iterations', 2)
    min_domain_size = params.get('min_domain_size', 5)
    
    labels_reg = tile_labels.copy()
    
    # Mode filter iterations
    for _ in range(iterations):
        labels_new = labels_reg.copy()
        n_rows, n_cols = labels_reg.shape
        
        for i in range(n_rows):
            for j in range(n_cols):
                # Get 3x3 neighborhood
                i_min, i_max = max(0, i-1), min(n_rows, i+2)
                j_min, j_max = max(0, j-1), min(n_cols, j+2)
                
                neighborhood = labels_reg[i_min:i_max, j_min:j_max].flatten()
                
                # Exclude noise (-1) from mode calculation unless all are noise
                valid = neighborhood[neighborhood >= 0]
                if len(valid) > 0:
                    # scipy.stats.mode returns ModeResult
                    mode_result = mode(valid, keepdims=False)
                    labels_new[i, j] = mode_result.mode
                # If all noise, keep as noise
        
        labels_reg = labels_new
    
    # Remove tiny connected components
    unique_labels = np.unique(labels_reg)
    for label in unique_labels:
        if label == -1:
            continue
        
        mask = labels_reg == label
        labeled_regions, n_regions = ndimage.label(mask)
        
        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = np.sum(region_mask)
            
            if region_size < min_domain_size:
                # Replace with most common neighbor label
                # Dilate to find neighbors
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


def upsample_labels(
    tile_labels_reg: np.ndarray,
    image_shape: Tuple[int, int],
    tile_size: int,
    stride: int
) -> np.ndarray:
    """
    Upsample tile labels to full image resolution using nearest-neighbor.
    
    Args:
        tile_labels_reg: (n_rows, n_cols) regularized tile labels
        image_shape: (H, W) target image dimensions
        tile_size: tile size in pixels
        stride: stride between tiles
    
    Returns:
        label_image: (H, W) full resolution labels
    """
    H, W = image_shape
    n_rows, n_cols = tile_labels_reg.shape
    
    # Create output array
    label_image = np.full((H, W), -1, dtype=np.int32)
    
    # Fill in each tile's region
    for i in range(n_rows):
        for j in range(n_cols):
            label = tile_labels_reg[i, j]
            
            # Calculate tile boundaries
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + tile_size, H)
            x_end = min(x_start + tile_size, W)
            
            # For overlapping tiles, later tiles overwrite earlier
            # This creates clean boundaries at tile centers
            label_image[y_start:y_end, x_start:x_end] = label
    
    return label_image


def create_domain_colormap(n_domains: int) -> np.ndarray:
    """Create a distinct colormap for domains."""
    if n_domains <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_domains]
    elif n_domains <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_domains]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_domains))
    return colors


def save_domain_labels_image(
    label_image: np.ndarray,
    output_path: Path,
    title: str = "Domain Labels"
):
    """Save color-coded domain label image."""
    unique_labels = np.unique(label_image)
    n_labels = len(unique_labels)
    
    # Create color mapping
    colors = create_domain_colormap(n_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    
    # Create RGB image
    H, W = label_image.shape
    rgb_image = np.zeros((H, W, 3))
    
    for label, idx in label_to_idx.items():
        if label == -1:
            # Noise in gray
            rgb_image[label_image == label] = [0.5, 0.5, 0.5]
        else:
            rgb_image[label_image == label] = colors[idx][:3]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_image)
    ax.set_title(title)
    ax.axis('off')
    
    # Add legend
    patches = []
    for label in sorted(unique_labels):
        if label == -1:
            color = [0.5, 0.5, 0.5]
            name = "Noise"
        else:
            color = colors[label_to_idx[label]][:3]
            name = f"Domain {label}"
        patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=name))
    
    ax.legend(handles=patches, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_domain_overlay(
    label_image: np.ndarray,
    original_image: np.ndarray,
    output_path: Path,
    title: str = "Domain Overlay"
):
    """Save domain boundaries overlaid on original image."""
    from skimage import segmentation
    
    # Normalize original image for display
    img_display = original_image.copy().astype(float)
    p1, p99 = np.percentile(img_display, [1, 99])
    img_display = np.clip((img_display - p1) / (p99 - p1 + 1e-8), 0, 1)
    
    # Find boundaries
    boundaries = segmentation.find_boundaries(label_image, mode='thick')
    
    # Create RGB overlay
    if img_display.ndim == 2:
        rgb_img = np.stack([img_display] * 3, axis=-1)
    else:
        rgb_img = img_display
    
    # Mark boundaries in red
    rgb_img[boundaries] = [1, 0, 0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_img)
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_feature_embedding(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Create and save UMAP embedding visualization.
    
    Returns dict with embedding quality metrics.
    """
    try:
        from umap import UMAP
        use_umap = True
    except ImportError:
        from sklearn.decomposition import PCA
        use_umap = False
        warnings.warn("UMAP not available, falling back to PCA")
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute embedding
    if use_umap:
        reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(features_scaled)
        method = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(features_scaled)
        method = "PCA"
    
    # Check for degenerate embedding
    x_range = np.ptp(embedding[:, 0])
    y_range = np.ptp(embedding[:, 1])
    x_std = np.std(embedding[:, 0])
    y_std = np.std(embedding[:, 1])
    
    # Degenerate if all points collapse to nearly same location
    is_degenerate = (x_std < 1e-6) or (y_std < 1e-6)
    has_structure = x_std > 0.1 and y_std > 0.1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = create_domain_colormap(len(unique_labels))
    
    for i, label in enumerate(sorted(unique_labels)):
        mask = labels == label
        if label == -1:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c='gray', alpha=0.3, s=20, label='Noise')
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[colors[i]], alpha=0.7, s=30, label=f'Domain {label}')
    
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title(f'Feature Embedding ({method})')
    ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'method': method,
        'x_range': float(x_range),
        'y_range': float(y_range),
        'x_std': float(x_std),
        'y_std': float(y_std),
        'is_degenerate': bool(is_degenerate),
        'has_structure': bool(has_structure)
    }


def run_clustering_pipeline(
    features_path: Path,
    image_path: Path,
    artifacts_dir: Path,
    outputs_dir: Path,
    params: dict = None
) -> Dict[str, Any]:
    """
    Run the full clustering pipeline.
    
    Args:
        features_path: Path to tile_features.npz
        image_path: Path to original image (for overlay)
        artifacts_dir: Directory for artifacts
        outputs_dir: Directory for output images
        params: Clustering parameters
    
    Returns:
        dict with all metrics and gate results
    """
    if params is None:
        params = {
            'min_cluster_size': 5,
            'min_samples': 3,
            'metric': 'euclidean',
            'random_state': 42,
            'regularize_iterations': 2,
            'min_domain_size': 5
        }
    
    # Load features
    data = np.load(features_path)
    features = data['features']
    tile_confidence = data.get('tile_confidence', np.ones(features.shape[0]))
    tile_positions = data['tile_positions']
    n_rows = int(data['n_rows'])
    n_cols = int(data['n_cols'])
    tile_size = int(data['tile_size'])
    stride = int(data['stride'])
    image_shape = tuple(data['image_shape'])
    
    results = {
        'n_tiles': features.shape[0],
        'n_features': features.shape[1],
        'n_rows': n_rows,
        'n_cols': n_cols,
        'tile_size': tile_size,
        'stride': stride,
        'image_shape': image_shape
    }
    
    # Gate G5: Feature Health
    print("Gate G5: Checking feature health...")
    health = check_feature_health(features)
    results['g5_health'] = {k: v for k, v in health.items() if k != 'features_clean'}
    features_clean = health['features_clean']
    
    print(f"  NaN count: {health['nan_count']}, Inf count: {health['inf_count']}")
    print(f"  Features shape: {features.shape}")
    
    # Cluster tiles
    print("Clustering tiles with HDBSCAN...")
    tile_labels_flat = cluster_tiles(features_clean, tile_confidence, params)
    
    n_clusters = len(set(tile_labels_flat)) - (1 if -1 in tile_labels_flat else 0)
    n_noise = np.sum(tile_labels_flat == -1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise tiles")
    
    results['n_clusters'] = n_clusters
    results['n_noise_tiles'] = int(n_noise)
    
    # Reshape to grid
    tile_labels_grid = tile_labels_flat.reshape(n_rows, n_cols)
    
    # Compute pre-regularization metrics
    print("Computing pre-regularization metrics...")
    pre_metrics = compute_adjacency_metrics(tile_labels_grid)
    results['pre_regularization'] = pre_metrics
    print(f"  Pre-reg flip rate: {pre_metrics['flip_rate']:.4f}")
    print(f"  Pre-reg micro-domains: {pre_metrics['micro_domain_count']}")
    
    # Spatial regularization
    print("Applying spatial regularization...")
    tile_labels_reg = spatial_regularize(tile_labels_grid, params)
    
    # Compute post-regularization metrics
    post_metrics = compute_adjacency_metrics(tile_labels_reg)
    results['post_regularization'] = post_metrics
    print(f"  Post-reg flip rate: {post_metrics['flip_rate']:.4f}")
    print(f"  Post-reg micro-domains: {post_metrics['micro_domain_count']}")
    
    # Gate G6: Coherence check
    g6_flip_improved = post_metrics['flip_rate'] <= pre_metrics['flip_rate']
    g6_micro_improved = post_metrics['micro_domain_count'] <= pre_metrics['micro_domain_count']
    g6_pass = g6_flip_improved and g6_micro_improved
    
    results['g6_coherence'] = {
        'flip_rate_improved': g6_flip_improved,
        'micro_domains_improved': g6_micro_improved,
        'pass': g6_pass
    }
    
    # Upsample to full resolution
    print("Upsampling to full resolution...")
    label_image = upsample_labels(tile_labels_reg, image_shape, tile_size, stride)
    
    # Save artifacts
    print("Saving artifacts...")
    np.save(artifacts_dir / 'tile_labels.npy', tile_labels_reg)
    np.save(artifacts_dir / 'label_image.npy', label_image)
    
    # Save embedding plot (part of G5)
    print("Creating feature embedding plot...")
    embedding_metrics = save_feature_embedding(
        features_clean,
        tile_labels_flat,
        outputs_dir / 'feature_embedding.png',
        random_state=params.get('random_state', 42)
    )
    results['embedding'] = embedding_metrics
    
    # G5 final check
    g5_pass = health['healthy'] or (health['nan_count'] + health['inf_count'] < features.size * 0.01)
    g5_pass = g5_pass and embedding_metrics['has_structure']
    results['g5_health']['embedding_has_structure'] = embedding_metrics['has_structure']
    results['g5_health']['pass'] = g5_pass
    
    # Save domain labels image
    print("Saving domain labels image...")
    save_domain_labels_image(
        label_image,
        outputs_dir / 'domain_labels.png',
        title=f"Domain Segmentation ({n_clusters} domains)"
    )
    
    # Save overlay if original image available
    if image_path.exists():
        print("Saving domain overlay...")
        # Load original image
        if str(image_path).endswith('.npy'):
            original_image = np.load(image_path)
        else:
            from PIL import Image
            original_image = np.array(Image.open(image_path))
        
        save_domain_overlay(
            label_image,
            original_image,
            outputs_dir / 'domain_overlay.png',
            title="Domain Boundaries Overlay"
        )
    
    return results


if __name__ == '__main__':
    import json
    
    # Paths
    features_path = Path('artifacts/tile_features.npz')
    artifacts_dir = Path('artifacts')
    outputs_dir = Path('outputs')
    
    # Find original image
    image_candidates = [
        Path('artifacts/preprocessed_image.npy'),
        Path('data/input.dm4'),
        Path('data/input.tif'),
        Path('data/input.png')
    ]
    image_path = None
    for p in image_candidates:
        if p.exists():
            image_path = p
            break
    
    if image_path is None:
        image_path = Path('artifacts/preprocessed_image.npy')  # Will be checked later
    
    # Check dependency
    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Waiting for SA3 to complete.")
        exit(1)
    
    # Run pipeline
    params = {
        'min_cluster_size': 5,
        'min_samples': 3,
        'metric': 'euclidean',
        'random_state': 42,
        'regularize_iterations': 2,
        'min_domain_size': 5
    }
    
    results = run_clustering_pipeline(
        features_path=features_path,
        image_path=image_path,
        artifacts_dir=artifacts_dir,
        outputs_dir=outputs_dir,
        params=params
    )
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING PIPELINE RESULTS")
    print("="*60)
    print(f"Tiles: {results['n_tiles']} ({results['n_rows']}x{results['n_cols']})")
    print(f"Features: {results['n_features']}")
    print(f"Clusters found: {results['n_clusters']}")
    print(f"Noise tiles: {results['n_noise_tiles']}")
    
    print("\n--- Gate G5: Feature Health ---")
    g5 = results['g5_health']
    print(f"NaN count: {g5['nan_count']}, Inf count: {g5['inf_count']}")
    print(f"Embedding has structure: {g5['embedding_has_structure']}")
    print(f"G5 PASS: {g5['pass']}")
    
    print("\n--- Gate G6: Domain Coherence ---")
    pre = results['pre_regularization']
    post = results['post_regularization']
    g6 = results['g6_coherence']
    print(f"Flip rate: {pre['flip_rate']:.4f} -> {post['flip_rate']:.4f}")
    print(f"Micro-domains: {pre['micro_domain_count']} -> {post['micro_domain_count']}")
    print(f"G6 PASS: {g6['pass']}")
    
    print("\n--- Outputs ---")
    print(f"  artifacts/tile_labels.npy")
    print(f"  artifacts/label_image.npy")
    print(f"  outputs/domain_labels.png")
    print(f"  outputs/domain_overlay.png")
    print(f"  outputs/feature_embedding.png")
    
    # Save results JSON
    with open(outputs_dir / 'clustering_results.json', 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        results_json = json.loads(json.dumps(results, default=convert))
        json.dump(results_json, f, indent=2)
