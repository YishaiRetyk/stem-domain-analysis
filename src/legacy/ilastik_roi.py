"""
ilastik-style ROI Cleanup Module

Generates ROI masks for background/sample separation WITHOUT pretrained models.
Uses intensity + FFT crystallinity to create masks that serve same purpose as
ilastik pixel classification.

This module implements Option A from the integration plan:
- No pretrained models
- ilastik used ONLY for ROI cleanup
- FFT remains authoritative for crystallographic decisions
"""

import numpy as np
from scipy import ndimage
from scipy.signal import windows
from typing import Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt


def compute_local_variance(image: np.ndarray, window_size: int = 32) -> np.ndarray:
    """Compute local variance as texture indicator."""
    # Local mean
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = ndimage.convolve(image.astype(np.float64), kernel, mode='reflect')
    
    # Local variance
    local_sq_mean = ndimage.convolve(image.astype(np.float64)**2, kernel, mode='reflect')
    local_var = local_sq_mean - local_mean**2
    
    return np.maximum(local_var, 0)  # Ensure non-negative


def compute_tile_crystallinity(image: np.ndarray, tile_size: int = 256, 
                                stride: int = 128, pixel_size: float = 0.0667,
                                q_range: Tuple[float, float] = (2.3, 2.6),
                                threshold_pct: float = 70) -> np.ndarray:
    """
    Compute per-tile crystallinity score based on FFT peak detection.
    
    Returns a grid of crystallinity scores (0-1) for each tile.
    """
    h, w = image.shape
    n_rows = (h - tile_size) // stride + 1
    n_cols = (w - tile_size) // stride + 1
    
    # Pre-compute window and frequency grid
    hann_1d = windows.hann(tile_size)
    window = np.outer(hann_1d, hann_1d)
    
    q_scale = 1.0 / (tile_size * pixel_size)
    center = tile_size // 2
    y, x = np.ogrid[:tile_size, :tile_size]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    q = r * q_scale
    q_mask = (q >= q_range[0]) & (q <= q_range[1])
    
    # DC mask
    dc_radius = 3
    dc_mask = ((y - center)**2 + (x - center)**2) <= dc_radius**2
    
    # Compute crystallinity for each tile
    crystallinity = np.zeros((n_rows, n_cols), dtype=np.float32)
    max_powers = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * stride
            x_start = col * stride
            tile = image[y_start:y_start + tile_size, x_start:x_start + tile_size]
            
            windowed = tile.astype(np.float64) * window
            fft = np.fft.fft2(windowed)
            fft_shifted = np.fft.fftshift(fft)
            power = np.abs(fft_shifted)**2
            
            # Mask DC
            power_masked = power.copy()
            power_masked[dc_mask] = 0
            
            # Get max power in q-range
            target_power = power_masked * q_mask
            max_power = np.max(target_power)
            max_powers.append(max_power)
    
    # Convert to array and compute threshold
    max_powers = np.array(max_powers).reshape(n_rows, n_cols)
    threshold = np.percentile(max_powers, threshold_pct)
    
    # Crystallinity score: 1 if above threshold, scaled otherwise
    crystallinity = np.clip(max_powers / threshold, 0, 1)
    
    return crystallinity


def create_roi_mask(image: np.ndarray, 
                    pixel_size: float = 0.0667,
                    tile_size: int = 256,
                    stride: int = 128,
                    intensity_threshold_pct: float = 10,
                    variance_threshold_pct: float = 20,
                    crystallinity_weight: float = 0.5,
                    smooth_sigma: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Create ROI mask combining multiple features (ilastik-style without ML).
    
    Features used:
    1. Intensity - dark regions are likely background
    2. Local variance - low variance regions lack texture
    3. FFT crystallinity - tiles with crystalline peaks should be kept
    
    Returns dict with:
    - 'mask': Binary mask (1 = sample, 0 = background)
    - 'probability': Soft probability map
    - 'intensity_mask': Intensity-based mask
    - 'variance_mask': Variance-based mask
    - 'crystallinity': Crystallinity grid
    """
    h, w = image.shape
    
    print("[ilastik ROI] Computing intensity mask...")
    # 1. Intensity mask - exclude very dark regions
    intensity_thresh = np.percentile(image, intensity_threshold_pct)
    intensity_mask = image > intensity_thresh
    
    print("[ilastik ROI] Computing local variance...")
    # 2. Local variance mask - exclude smooth/uniform regions
    local_var = compute_local_variance(image, window_size=32)
    var_thresh = np.percentile(local_var, variance_threshold_pct)
    variance_mask = local_var > var_thresh
    
    print("[ilastik ROI] Computing FFT crystallinity...")
    # 3. FFT crystallinity grid
    cryst_grid = compute_tile_crystallinity(
        image, tile_size, stride, pixel_size,
        q_range=(2.3, 2.6), threshold_pct=70
    )
    
    # Upsample crystallinity to full resolution
    n_rows, n_cols = cryst_grid.shape
    crystallinity_full = np.zeros((h, w), dtype=np.float32)
    
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * stride
            x_start = col * stride
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            
            # Use maximum of overlapping tiles
            crystallinity_full[y_start:y_end, x_start:x_end] = np.maximum(
                crystallinity_full[y_start:y_end, x_start:x_end],
                cryst_grid[row, col]
            )
    
    print("[ilastik ROI] Combining features...")
    # 4. Combine features into probability
    # Weight: intensity (0.3) + variance (0.2) + crystallinity (0.5)
    intensity_prob = (image - image.min()) / (image.max() - image.min() + 1e-10)
    variance_prob = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-10)
    
    probability = (
        0.3 * intensity_prob + 
        0.2 * variance_prob + 
        crystallinity_weight * crystallinity_full
    )
    
    # Normalize
    probability = probability / probability.max()
    
    # 5. Create binary mask
    # Threshold at 0.3 (conservative - keep more)
    mask_threshold = 0.3
    mask = probability > mask_threshold
    
    # 6. Clean up mask with morphological operations
    print("[ilastik ROI] Cleaning mask...")
    # Fill small holes
    mask = ndimage.binary_fill_holes(mask)
    
    # Remove small objects
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
    
    # Smooth edges
    if smooth_sigma > 0:
        mask_smooth = ndimage.gaussian_filter(mask.astype(float), sigma=smooth_sigma)
        mask = mask_smooth > 0.5
    
    return {
        'mask': mask.astype(np.uint8),
        'probability': probability,
        'intensity_mask': intensity_mask.astype(np.uint8),
        'variance_mask': variance_mask.astype(np.uint8),
        'crystallinity': cryst_grid,
        'crystallinity_full': crystallinity_full,
    }


def validate_roi_mask(mask: np.ndarray, 
                       crystallinity_grid: np.ndarray,
                       tile_size: int = 256,
                       stride: int = 128) -> Dict[str, Any]:
    """
    Validate ROI mask against FFT crystallinity (Gate G2b).
    
    Returns validation results dict.
    """
    h, w = mask.shape
    n_rows, n_cols = crystallinity_grid.shape
    
    # Count tiles retained by mask
    tiles_total = n_rows * n_cols
    tiles_crystalline = np.sum(crystallinity_grid > 0.5)  # > 50% crystallinity score
    
    tiles_retained = 0
    crystalline_retained = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * stride
            x_start = col * stride
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            
            tile_mask = mask[y_start:y_end, x_start:x_end]
            tile_coverage = np.mean(tile_mask)
            
            if tile_coverage > 0.5:  # Tile is mostly kept
                tiles_retained += 1
                if crystallinity_grid[row, col] > 0.5:
                    crystalline_retained += 1
    
    # Calculate metrics
    pct_retained = tiles_retained / tiles_total * 100 if tiles_total > 0 else 0
    pct_masked = 100 - pct_retained
    
    if tiles_crystalline > 0:
        pct_crystalline_retained = crystalline_retained / tiles_crystalline * 100
    else:
        pct_crystalline_retained = 100  # No crystalline tiles to lose
    
    # G2b Pass criteria: ≥80% crystalline tiles retained
    g2b_pass = pct_crystalline_retained >= 80
    
    return {
        'tiles_total': tiles_total,
        'tiles_crystalline': int(tiles_crystalline),
        'tiles_retained': tiles_retained,
        'crystalline_retained': crystalline_retained,
        'pct_retained': pct_retained,
        'pct_masked': pct_masked,
        'pct_crystalline_retained': pct_crystalline_retained,
        'g2b_pass': g2b_pass,
        'g2b_reason': f"{pct_crystalline_retained:.1f}% crystalline tiles retained (threshold: 80%)"
    }


def save_roi_visualizations(image: np.ndarray, 
                            roi_results: Dict[str, np.ndarray],
                            output_dir: str = 'outputs'):
    """Save ROI mask visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mask = roi_results['mask']
    probability = roi_results['probability']
    
    # 1. Mask overlay
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(image, cmap='gray', alpha=1.0)
    
    # Create colored overlay
    overlay = np.zeros((*image.shape, 4))
    overlay[mask == 1, 1] = 0.5  # Green for sample
    overlay[mask == 1, 3] = 0.3  # Alpha
    overlay[mask == 0, 0] = 0.5  # Red for background
    overlay[mask == 0, 3] = 0.3
    
    ax.imshow(overlay)
    ax.set_title('ilastik ROI Mask Overlay\nGreen = Sample, Red = Background')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ilastik_mask_overlay.png', dpi=150)
    plt.close()
    
    # 2. Probability map
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(probability, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Sample Probability')
    ax.set_title('ROI Probability Map')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ilastik_probability.png', dpi=150)
    plt.close()
    
    # 3. Multi-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(roi_results['crystallinity_full'], cmap='hot')
    axes[0, 1].set_title('FFT Crystallinity')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(probability, cmap='viridis')
    axes[1, 0].set_title('Combined Probability')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title(f'Final Mask ({np.mean(mask)*100:.1f}% sample)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ilastik_roi_summary.png', dpi=150)
    plt.close()
    
    print(f"[ilastik ROI] Saved visualizations to {output_path}/")


def run_ilastik_roi(image: np.ndarray,
                    pixel_size: float = 0.0667,
                    output_dir: str = 'outputs',
                    artifacts_dir: str = 'artifacts',
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Run complete ilastik-style ROI analysis.
    
    Returns dict with mask, validation results, and metadata.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ILASTIK ROI CLEANUP (Option A - No Pretraining)")
        print("=" * 60)
    
    # Create ROI mask
    roi_results = create_roi_mask(
        image, 
        pixel_size=pixel_size,
        tile_size=256,
        stride=128
    )
    
    # Validate mask
    if verbose:
        print("\n[G2b] Validating ROI mask against FFT crystallinity...")
    
    validation = validate_roi_mask(
        roi_results['mask'],
        roi_results['crystallinity'],
        tile_size=256,
        stride=128
    )
    
    if verbose:
        print(f"  Tiles retained: {validation['tiles_retained']}/{validation['tiles_total']} ({validation['pct_retained']:.1f}%)")
        print(f"  Crystalline tiles retained: {validation['crystalline_retained']}/{validation['tiles_crystalline']} ({validation['pct_crystalline_retained']:.1f}%)")
        print(f"  G2b: {'PASS ✓' if validation['g2b_pass'] else 'FAIL ✗'} - {validation['g2b_reason']}")
    
    # Save artifacts
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    np.save(artifacts_path / 'ilastik_mask.npy', roi_results['mask'])
    np.save(artifacts_path / 'ilastik_prob.npy', roi_results['probability'])
    
    if verbose:
        print(f"\n  Saved: {artifacts_path}/ilastik_mask.npy")
        print(f"  Saved: {artifacts_path}/ilastik_prob.npy")
    
    # Save visualizations
    save_roi_visualizations(image, roi_results, output_dir)
    
    if verbose:
        print("\n" + "=" * 60)
        print("ILASTIK ROI COMPLETE")
        print("=" * 60)
    
    return {
        'mask': roi_results['mask'],
        'probability': roi_results['probability'],
        'validation': validation,
        'crystallinity': roi_results['crystallinity'],
        'metadata': {
            'method': 'intensity + variance + FFT crystallinity',
            'pretrained': False,
            'tile_size': 256,
            'stride': 128,
        }
    }


if __name__ == '__main__':
    """Test ROI module on preprocessed image."""
    import sys
    
    artifacts_path = Path('artifacts')
    if not (artifacts_path / 'preprocessed.npy').exists():
        print("Error: Run preprocessing first")
        sys.exit(1)
    
    image = np.load(artifacts_path / 'preprocessed.npy')
    
    # Load pixel size
    import json
    meta_path = artifacts_path / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        pixel_size = meta.get('pixel_size_nm', 0.0667)
    else:
        pixel_size = 0.0667
    
    print(f"Image shape: {image.shape}")
    print(f"Pixel size: {pixel_size} nm")
    
    results = run_ilastik_roi(image, pixel_size, verbose=True)
    
    print(f"\nFinal mask: {np.mean(results['mask'])*100:.1f}% sample area")
