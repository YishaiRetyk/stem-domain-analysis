"""
Test script for cluster_domains.py using synthetic data.
Validates SA4 module works correctly before real data is available.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Create synthetic tile features that simulate 3 crystal domains + noise
np.random.seed(42)

# Image and tiling parameters
image_shape = (1024, 1024)
tile_size = 64
stride = 32
n_rows = (image_shape[0] - tile_size) // stride + 1
n_cols = (image_shape[1] - tile_size) // stride + 1
n_tiles = n_rows * n_cols

print(f"Synthetic test setup:")
print(f"  Image shape: {image_shape}")
print(f"  Grid: {n_rows} x {n_cols} = {n_tiles} tiles")

# Create tile positions
tile_positions = []
for i in range(n_rows):
    for j in range(n_cols):
        tile_positions.append([i, j, i*stride, j*stride])
tile_positions = np.array(tile_positions)

# Create 3 distinct crystal domains spatially
# Domain 0: top-left quadrant - d-spacing ~1.0nm, orientation ~45°
# Domain 1: top-right and bottom-left - d-spacing ~0.8nm, orientation ~0°
# Domain 2: bottom-right quadrant - d-spacing ~1.2nm, orientation ~90°

n_features = 5  # n_peaks, n_paired, dominant_d, dominant_theta, crystallinity
features = np.zeros((n_tiles, n_features))
tile_confidence = np.zeros(n_tiles)

for idx in range(n_tiles):
    row, col = tile_positions[idx, :2]
    cy, cx = row / n_rows, col / n_cols  # Normalized coordinates
    
    # Determine domain based on spatial location
    if cy < 0.5 and cx < 0.5:
        # Domain 0: top-left
        n_peaks = np.random.randint(4, 8)
        n_paired = np.random.randint(2, min(n_peaks, 6))
        d = 1.0 + np.random.normal(0, 0.05)
        theta = 45 + np.random.normal(0, 5)
    elif cy >= 0.5 and cx >= 0.5:
        # Domain 2: bottom-right
        n_peaks = np.random.randint(3, 7)
        n_paired = np.random.randint(2, min(n_peaks, 5))
        d = 1.2 + np.random.normal(0, 0.05)
        theta = 90 + np.random.normal(0, 5)
    else:
        # Domain 1: other areas
        n_peaks = np.random.randint(4, 9)
        n_paired = np.random.randint(3, min(n_peaks, 7))
        d = 0.8 + np.random.normal(0, 0.04)
        theta = np.random.normal(0, 5)
    
    # Add some noise tiles (amorphous regions)
    if np.random.random() < 0.05:
        n_peaks = np.random.randint(0, 2)
        n_paired = 0
        d = np.random.uniform(0.5, 1.5)
        theta = np.random.uniform(0, 180)
    
    crystallinity = (n_paired / max(n_peaks, 1)) * min(1.0, n_paired / 4.0)
    confidence = 0.3 + 0.6 * crystallinity
    
    features[idx] = [n_peaks, n_paired, d, theta % 180, crystallinity]
    tile_confidence[idx] = confidence

# Save synthetic features
artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)
outputs_dir = Path('outputs')
outputs_dir.mkdir(exist_ok=True)

np.savez(
    artifacts_dir / 'tile_features.npz',
    features=features,
    tile_confidence=tile_confidence,
    tile_positions=tile_positions,
    n_rows=n_rows,
    n_cols=n_cols,
    tile_size=tile_size,
    stride=stride,
    image_shape=np.array(image_shape)
)
print(f"Saved synthetic tile_features.npz to {artifacts_dir}")

# Create synthetic image for overlay visualization
synthetic_image = np.zeros(image_shape, dtype=np.float32)
for idx in range(n_tiles):
    row, col = tile_positions[idx, :2]
    y, x = tile_positions[idx, 2:]
    cy, cx = row / n_rows, col / n_cols
    
    # Create different patterns for different domains
    yy, xx = np.meshgrid(np.arange(tile_size), np.arange(tile_size), indexing='ij')
    
    if cy < 0.5 and cx < 0.5:
        pattern = np.sin(2*np.pi*(yy + xx) / 10)  # Diagonal stripes
    elif cy >= 0.5 and cx >= 0.5:
        pattern = np.sin(2*np.pi*xx / 12)  # Vertical stripes
    else:
        pattern = np.sin(2*np.pi*yy / 8)  # Horizontal stripes
    
    pattern = (pattern + 1) / 2  # Normalize to [0, 1]
    pattern += np.random.normal(0, 0.1, pattern.shape)
    
    y_end = min(y + tile_size, image_shape[0])
    x_end = min(x + tile_size, image_shape[1])
    synthetic_image[y:y_end, x:x_end] = pattern[:y_end-y, :x_end-x]

np.save(artifacts_dir / 'preprocessed_image.npy', synthetic_image)
print(f"Saved synthetic preprocessed_image.npy")

# Now run the clustering pipeline
print("\n" + "="*60)
print("Running SA4 Clustering Pipeline on Synthetic Data")
print("="*60 + "\n")

from cluster_domains import run_clustering_pipeline

params = {
    'min_cluster_size': 5,
    'min_samples': 3,
    'metric': 'euclidean',
    'random_state': 42,
    'regularize_iterations': 2,
    'min_domain_size': 5
}

results = run_clustering_pipeline(
    features_path=artifacts_dir / 'tile_features.npz',
    image_path=artifacts_dir / 'preprocessed_image.npy',
    artifacts_dir=artifacts_dir,
    outputs_dir=outputs_dir,
    params=params
)

# Print results
print("\n" + "="*60)
print("SA4 CLUSTERING RESULTS (Synthetic Data)")
print("="*60)
print(f"Tiles: {results['n_tiles']} ({results['n_rows']}x{results['n_cols']})")
print(f"Features: {results['n_features']}")
print(f"Clusters found: {results['n_clusters']}")
print(f"Noise tiles: {results['n_noise_tiles']}")

print("\n--- Gate G5: Feature Health ---")
g5 = results['g5_health']
print(f"NaN count: {g5['nan_count']}, Inf count: {g5['inf_count']}")
print(f"Embedding has structure: {g5['embedding_has_structure']}")
g5_status = "PASS ✓" if g5['pass'] else "FAIL ✗"
print(f"G5 RESULT: {g5_status}")

print("\n--- Gate G6: Domain Coherence ---")
pre = results['pre_regularization']
post = results['post_regularization']
g6 = results['g6_coherence']
print(f"Flip rate: {pre['flip_rate']:.4f} -> {post['flip_rate']:.4f}")
print(f"Micro-domains: {pre['micro_domain_count']} -> {post['micro_domain_count']}")
g6_status = "PASS ✓" if g6['pass'] else "FAIL ✗"
print(f"G6 RESULT: {g6_status}")

print("\n--- Outputs Generated ---")
for f in ['tile_labels.npy', 'label_image.npy']:
    path = artifacts_dir / f
    if path.exists():
        print(f"  ✓ {path}")
    else:
        print(f"  ✗ {path} (MISSING)")

for f in ['domain_labels.png', 'domain_overlay.png', 'feature_embedding.png']:
    path = outputs_dir / f
    if path.exists():
        print(f"  ✓ {path}")
    else:
        print(f"  ✗ {path} (MISSING)")

print("\n" + "="*60)
print("SA4 TEST COMPLETE")
print("="*60)
