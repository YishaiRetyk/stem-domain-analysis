"""
FFT Feature Extraction Module for STEM-HAADF Crystal Domain Segmentation.

SA3 - Tile-based FFT analysis and peak detection.
"""

from dataclasses import dataclass
from typing import Generator, Tuple, Dict, Any
import numpy as np
from scipy import ndimage
from scipy.signal import windows
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class PeakSet:
    """Container for FFT peak analysis results."""
    g_vectors: np.ndarray      # (N, 2) g-vectors in nm^-1
    d_list: np.ndarray         # (N,) d-spacings in nm
    theta_list: np.ndarray     # (N,) angles in degrees [0, 180)
    amps: np.ndarray           # (N,) peak amplitudes
    paired_mask: np.ndarray    # (N,) bool - True if ±symmetry paired


# Default parameters
DEFAULT_PARAMS = {
    'tile_size': 256,
    'stride': 128,
    'window': 'hann',
    'peak_threshold': 0.1,  # relative to max after DC removal
    'd_range': (0.5, 1.5),  # nm
    'pixel_size': 0.127,    # nm/px
    'dc_mask_radius': 2,    # pixels to mask around DC
    'local_max_size': 5,    # footprint for local maximum detection
    'pair_tolerance': 2,    # pixels tolerance for ±symmetry pairing
}


def tile_generator(image: np.ndarray, tile_size: int = 256, stride: int = 128
                   ) -> Generator[Tuple[np.ndarray, int, int, int, int], None, None]:
    """
    Generate overlapping tiles from an image.
    
    Yields:
        (tile_data, row_idx, col_idx, y_start, x_start) for each tile.
    """
    h, w = image.shape[:2]
    row_idx = 0
    for y in range(0, h - tile_size + 1, stride):
        col_idx = 0
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            yield tile, row_idx, col_idx, y, x
            col_idx += 1
        row_idx += 1


def get_tiling_info(image_shape: Tuple[int, int], tile_size: int = 256, stride: int = 128
                    ) -> Dict[str, Any]:
    """Get tiling grid information without generating tiles."""
    h, w = image_shape[:2]
    n_rows = (h - tile_size) // stride + 1
    n_cols = (w - tile_size) // stride + 1
    n_tiles = n_rows * n_cols
    return {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_tiles': n_tiles,
        'grid_shape': (n_rows, n_cols),
    }


def create_2d_hann_window(size: int) -> np.ndarray:
    """Create a 2D Hann window for FFT."""
    hann_1d = windows.hann(size)
    return np.outer(hann_1d, hann_1d)


def tile_fft_peaks(tile: np.ndarray, px_nm: float, params: dict = None) -> PeakSet:
    """
    Compute FFT and extract peaks from a tile.
    
    Algorithm:
    1. Apply Hann window
    2. Compute 2D FFT, shift to center
    3. Compute power spectrum
    4. Find local maxima above threshold
    5. Filter to d_range [0.5, 1.5] nm
    6. Pair peaks with ±symmetry (g and -g)
    
    Args:
        tile: 2D array of tile data
        px_nm: pixel size in nm
        params: optional parameter override dict
        
    Returns:
        PeakSet with detected peaks
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    tile_size = tile.shape[0]
    
    # 1. Apply Hann window
    window = create_2d_hann_window(tile_size)
    windowed = tile.astype(np.float64) * window
    
    # 2. Compute 2D FFT and shift
    fft_result = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # 3. Compute power spectrum
    power = np.abs(fft_shifted) ** 2
    
    # 4. Mask DC component
    center = tile_size // 2
    dc_radius = p['dc_mask_radius']
    y_dc, x_dc = np.ogrid[:tile_size, :tile_size]
    dc_mask = ((y_dc - center)**2 + (x_dc - center)**2) <= dc_radius**2
    power_masked = power.copy()
    power_masked[dc_mask] = 0
    
    # Find local maxima
    footprint = np.ones((p['local_max_size'], p['local_max_size']))
    local_max = ndimage.maximum_filter(power_masked, footprint=footprint)
    
    # Threshold
    max_val = power_masked.max()
    if max_val == 0:
        # No signal - return empty PeakSet
        return PeakSet(
            g_vectors=np.zeros((0, 2)),
            d_list=np.zeros(0),
            theta_list=np.zeros(0),
            amps=np.zeros(0),
            paired_mask=np.zeros(0, dtype=bool)
        )
    
    threshold = p['peak_threshold'] * max_val
    peaks_mask = (power_masked == local_max) & (power_masked > threshold) & ~dc_mask
    
    # Get peak positions
    peak_y, peak_x = np.where(peaks_mask)
    
    if len(peak_y) == 0:
        return PeakSet(
            g_vectors=np.zeros((0, 2)),
            d_list=np.zeros(0),
            theta_list=np.zeros(0),
            amps=np.zeros(0),
            paired_mask=np.zeros(0, dtype=bool)
        )
    
    # 5. Convert to spatial frequencies
    # Frequency axes: center is DC (0 frequency)
    # freq = (pixel_index - center) / (tile_size * pixel_size)
    freq_scale = 1.0 / (tile_size * px_nm)  # nm^-1 per pixel
    
    gx = (peak_x - center) * freq_scale  # nm^-1
    gy = (peak_y - center) * freq_scale  # nm^-1
    g_mag = np.sqrt(gx**2 + gy**2)  # |g| in nm^-1
    
    # Convert to d-spacing: d = 1/|g|
    # Avoid division by zero
    valid_g = g_mag > 1e-10
    d_spacing = np.zeros_like(g_mag)
    d_spacing[valid_g] = 1.0 / g_mag[valid_g]
    
    # Filter by d_range
    d_min, d_max = p['d_range']
    in_range = (d_spacing >= d_min) & (d_spacing <= d_max)
    
    if not np.any(in_range):
        return PeakSet(
            g_vectors=np.zeros((0, 2)),
            d_list=np.zeros(0),
            theta_list=np.zeros(0),
            amps=np.zeros(0),
            paired_mask=np.zeros(0, dtype=bool)
        )
    
    # Apply filter
    gx = gx[in_range]
    gy = gy[in_range]
    d_spacing = d_spacing[in_range]
    peak_x_filt = peak_x[in_range]
    peak_y_filt = peak_y[in_range]
    
    # Get amplitudes
    amps = power[peak_y_filt, peak_x_filt]
    
    # Calculate angles in [0, 180) - direction of g-vector
    theta = np.degrees(np.arctan2(gy, gx))
    theta = theta % 180  # Map to [0, 180)
    
    # 6. Pair peaks with ±symmetry
    g_vectors = np.column_stack([gx, gy])
    n_peaks = len(gx)
    paired_mask = np.zeros(n_peaks, dtype=bool)
    
    tol = p['pair_tolerance'] * freq_scale  # tolerance in nm^-1
    
    for i in range(n_peaks):
        if paired_mask[i]:
            continue
        # Look for -g partner
        target_g = -g_vectors[i]
        for j in range(i + 1, n_peaks):
            if paired_mask[j]:
                continue
            dist = np.linalg.norm(g_vectors[j] - target_g)
            if dist < tol:
                paired_mask[i] = True
                paired_mask[j] = True
                break
    
    return PeakSet(
        g_vectors=g_vectors,
        d_list=d_spacing,
        theta_list=theta,
        amps=amps,
        paired_mask=paired_mask
    )


def extract_tile_features(peak_set: PeakSet, qc: dict = None) -> Tuple[np.ndarray, float]:
    """
    Extract feature vector from peak set.
    
    Feature vector: [n_peaks, n_paired, dominant_d, dominant_theta, crystallinity_score]
    Confidence: based on n_paired_peaks and peak SNR
    
    Args:
        peak_set: PeakSet from tile_fft_peaks
        qc: optional quality control parameters
        
    Returns:
        (features, confidence) tuple
    """
    n_peaks = len(peak_set.d_list)
    n_paired = np.sum(peak_set.paired_mask)
    
    if n_peaks == 0:
        # Amorphous/empty tile
        features = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        confidence = 0.0
        return features, confidence
    
    # Find dominant d-spacing (highest amplitude)
    max_amp_idx = np.argmax(peak_set.amps)
    dominant_d = peak_set.d_list[max_amp_idx]
    dominant_theta = peak_set.theta_list[max_amp_idx]
    
    # Crystallinity score: based on paired peaks and their regularity
    # Higher score = more crystalline
    if n_peaks > 0:
        pair_ratio = n_paired / n_peaks
    else:
        pair_ratio = 0
    
    # Crystallinity = weighted combination of pair ratio and peak count
    crystallinity = pair_ratio * min(1.0, n_paired / 4.0)  # saturates at 4 paired peaks
    
    features = np.array([
        float(n_peaks),
        float(n_paired),
        dominant_d,
        dominant_theta,
        crystallinity
    ])
    
    # Confidence based on n_paired and amplitude SNR
    if n_paired >= 4:
        confidence = 0.9
    elif n_paired >= 2:
        confidence = 0.6
    elif n_peaks >= 2:
        confidence = 0.3
    else:
        confidence = 0.1
    
    # Adjust by amplitude consistency
    if n_peaks > 1:
        amp_cv = np.std(peak_set.amps) / (np.mean(peak_set.amps) + 1e-10)
        if amp_cv < 0.5:  # Consistent amplitudes
            confidence = min(1.0, confidence + 0.1)
    
    return features, confidence


def process_all_tiles(image: np.ndarray, params: dict = None, verbose: bool = True
                      ) -> Dict[str, np.ndarray]:
    """
    Process all tiles in an image and extract features.
    
    Args:
        image: 2D preprocessed image
        params: parameter dict
        verbose: print progress
        
    Returns:
        dict with tile_coords, features, confidence, peaks_summary
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    px_nm = p['pixel_size']
    tile_size = p['tile_size']
    stride = p['stride']
    
    info = get_tiling_info(image.shape, tile_size, stride)
    n_tiles = info['n_tiles']
    
    if verbose:
        print(f"[G3] Tiling Sanity Check:", flush=True)
        print(f"  Image shape: {image.shape}", flush=True)
        print(f"  Tile size: {tile_size} px, Stride: {stride} px", flush=True)
        print(f"  Grid shape: {info['grid_shape']}", flush=True)
        print(f"  Total tiles: {n_tiles}", flush=True)
        # Check period coverage
        d_max = p['d_range'][1]
        periods_per_tile = tile_size / (d_max / px_nm)
        print(f"  Periods at d={d_max}nm: {tile_size}/({d_max}/{px_nm}) = {periods_per_tile:.1f}", flush=True)
        if periods_per_tile >= 20:
            print(f"  [G3] PASS: {periods_per_tile:.1f} >= 20 periods", flush=True)
        else:
            print(f"  [G3] WARN: {periods_per_tile:.1f} < 20 periods (may affect resolution)", flush=True)
    
    # Allocate output arrays
    tile_coords = np.zeros((n_tiles, 4), dtype=np.int32)  # row, col, y, x
    features = np.zeros((n_tiles, 5), dtype=np.float32)
    confidence = np.zeros(n_tiles, dtype=np.float32)
    peaks_summary = []  # List of (n_peaks, n_paired, dominant_d) per tile
    
    for i, (tile, row, col, y, x) in enumerate(tile_generator(image, tile_size, stride)):
        tile_coords[i] = [row, col, y, x]
        
        # Extract peaks
        peak_set = tile_fft_peaks(tile, px_nm, p)
        
        # Extract features
        feat, conf = extract_tile_features(peak_set)
        features[i] = feat
        confidence[i] = conf
        
        # Summary
        peaks_summary.append({
            'n_peaks': int(feat[0]),
            'n_paired': int(feat[1]),
            'dominant_d': feat[2] if feat[0] > 0 else np.nan,
        })
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_tiles} tiles...", flush=True)
    
    # G4 metrics
    if verbose:
        n_paired_arr = features[:, 1]
        crystalline_mask = n_paired_arr >= 4
        frac_crystalline = np.mean(crystalline_mask)
        
        print(f"\n[G4] Peak Detection Viability:", flush=True)
        print(f"  Tiles with ≥4 paired peaks: {np.sum(crystalline_mask)}/{n_tiles} ({frac_crystalline*100:.1f}%)", flush=True)
        
        # D-spacing distribution for crystalline tiles
        if np.any(crystalline_mask):
            d_vals = features[crystalline_mask, 2]
            print(f"  Dominant d-spacing (crystalline): mean={np.mean(d_vals):.3f}nm, std={np.std(d_vals):.3f}nm", flush=True)
        
        # Total peaks and paired peaks
        total_peaks = np.sum(features[:, 0])
        total_paired = np.sum(features[:, 1])
        print(f"  Total peaks detected: {int(total_peaks)}", flush=True)
        print(f"  Total paired peaks: {int(total_paired)}", flush=True)
        
        if frac_crystalline > 0.1:
            print(f"  [G4] PASS: {frac_crystalline*100:.1f}% > 10% crystalline tiles", flush=True)
        else:
            # Check if amorphous
            if total_peaks < n_tiles * 0.5:  # Less than 0.5 peaks per tile on average
                print(f"  [G4] PASS (amorphous): Sample appears largely amorphous", flush=True)
            else:
                print(f"  [G4] WARN: Only {frac_crystalline*100:.1f}% crystalline, {total_peaks/n_tiles:.1f} avg peaks/tile", flush=True)
    
    return {
        'tile_coords': tile_coords,
        'features': features,
        'confidence': confidence,
        'peaks_summary': peaks_summary,
        'grid_shape': info['grid_shape'],
    }


def save_visualizations(image: np.ndarray, results: Dict[str, np.ndarray],
                        output_dir: str = 'outputs', params: dict = None):
    """Generate and save visualization outputs."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tile_size = p['tile_size']
    stride = p['stride']
    grid_shape = results['grid_shape']
    tile_coords = results['tile_coords']
    features = results['features']
    confidence = results['confidence']
    
    # 1. Tile grid overlay
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='gray')
    
    # Draw tile boundaries
    for i in range(len(tile_coords)):
        row, col, y, x = tile_coords[i]
        rect = plt.Rectangle((x, y), tile_size, tile_size,
                             fill=False, edgecolor='cyan', linewidth=0.5, alpha=0.5)
        ax.add_patch(rect)
    
    ax.set_title(f'Tile Grid Overlay ({grid_shape[0]}×{grid_shape[1]} tiles)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(output_path / 'tile_grid_overlay.png', dpi=150)
    plt.close()
    
    # 2. Peak count map
    n_rows, n_cols = grid_shape
    peak_count_grid = np.zeros((n_rows, n_cols))
    for i in range(len(tile_coords)):
        row, col = tile_coords[i, 0], tile_coords[i, 1]
        peak_count_grid[row, col] = features[i, 0]  # n_peaks
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(peak_count_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Number of Peaks')
    ax.set_title('Peak Count per Tile')
    ax.set_xlabel('Tile Column')
    ax.set_ylabel('Tile Row')
    plt.tight_layout()
    plt.savefig(output_path / 'peak_count_map.png', dpi=150)
    plt.close()
    
    # 3. Confidence map
    confidence_grid = np.zeros((n_rows, n_cols))
    for i in range(len(tile_coords)):
        row, col = tile_coords[i, 0], tile_coords[i, 1]
        confidence_grid[row, col] = confidence[i]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confidence_grid, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Crystallinity Confidence')
    ax.set_title('Tile Crystallinity Confidence')
    ax.set_xlabel('Tile Column')
    ax.set_ylabel('Tile Row')
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_map.png', dpi=150)
    plt.close()
    
    print(f"Saved visualizations to {output_path}/", flush=True)


def main():
    """Main entry point for FFT feature extraction."""
    import json
    import sys
    
    # Unbuffered output
    print("=" * 60, flush=True)
    print("SA3 - FFT Feature Extraction", flush=True)
    print("=" * 60, flush=True)
    
    # Load preprocessed image
    input_path = Path('artifacts/preprocessed.npy')
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed image not found at {input_path}")
    
    print(f"Loading preprocessed image from {input_path}...", flush=True)
    image = np.load(input_path)
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}", flush=True)
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]", flush=True)
    
    # Process tiles
    print("\nProcessing tiles...", flush=True)
    results = process_all_tiles(image, verbose=True)
    
    # Save artifacts
    artifacts_path = Path('artifacts')
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    # Convert peaks_summary to array-friendly format
    peaks_summary_arr = np.array([
        [p['n_peaks'], p['n_paired'], p['dominant_d'] if not np.isnan(p.get('dominant_d', np.nan)) else 0]
        for p in results['peaks_summary']
    ], dtype=np.float32)
    
    np.savez(
        artifacts_path / 'tile_features.npz',
        tile_coords=results['tile_coords'],
        features=results['features'],
        confidence=results['confidence'],
        peaks_summary=peaks_summary_arr,
        grid_shape=np.array(results['grid_shape']),
    )
    print(f"\nSaved tile features to {artifacts_path}/tile_features.npz", flush=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...", flush=True)
    save_visualizations(image, results)
    
    print("\n" + "=" * 60, flush=True)
    print("SA3 Complete", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
