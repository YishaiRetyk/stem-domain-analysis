"""
Radial Profile Analysis Module for STEM-HAADF Crystal Domain Segmentation.

Implements radial FFT analysis similar to the reference pipeline:
- Radial profile extraction in q-space (1/nm)
- Peak detection at specific q-ranges
- Peak location maps and orientation maps
"""

import numpy as np
from scipy import ndimage
from scipy.signal import windows, find_peaks
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path


@dataclass
class RadialProfile:
    """Container for radial profile results."""
    q_values: np.ndarray      # q in nm^-1
    intensity: np.ndarray     # radial average intensity
    q_min: float
    q_max: float
    pixel_size_nm: float
    tile_size: int


@dataclass  
class TilePeakResult:
    """Results of peak detection for a single tile."""
    has_peak: bool
    peak_intensity: float
    peak_q: float
    orientation_deg: float    # dominant orientation in degrees
    confidence: float


DEFAULT_PARAMS = {
    'tile_size': 256,
    'stride': 128,
    'q_range': (2.3, 2.6),    # nm^-1 - target q-range for crystalline peaks
    'intensity_threshold': 45000,  # minimum peak intensity 
    'window': 'hann',
    'dc_mask_radius': 3,
}


def compute_radial_profile(fft_power: np.ndarray, pixel_size_nm: float,
                           tile_size: int) -> RadialProfile:
    """
    Compute radial average of FFT power spectrum.
    
    Args:
        fft_power: 2D FFT power spectrum (shifted, DC at center)
        pixel_size_nm: pixel size in nm
        tile_size: tile size in pixels
        
    Returns:
        RadialProfile with q_values and intensity
    """
    center = fft_power.shape[0] // 2
    
    # Create radial distance array
    y, x = np.ogrid[:fft_power.shape[0], :fft_power.shape[1]]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Convert radius to q (spatial frequency in nm^-1)
    # q = r / (tile_size * pixel_size) 
    q_scale = 1.0 / (tile_size * pixel_size_nm)  # nm^-1 per pixel in freq space
    q = r * q_scale
    
    # Bin by integer radius
    r_int = r.astype(int)
    max_r = min(center, r_int.max())
    
    # Compute radial average
    radial_sum = ndimage.sum(fft_power, r_int, index=np.arange(max_r + 1))
    radial_count = ndimage.sum(np.ones_like(fft_power), r_int, index=np.arange(max_r + 1))
    
    # Avoid division by zero
    radial_count[radial_count == 0] = 1
    radial_avg = radial_sum / radial_count
    
    # Convert bin indices to q values
    q_values = np.arange(max_r + 1) * q_scale
    
    return RadialProfile(
        q_values=q_values,
        intensity=radial_avg,
        q_min=q_values[1] if len(q_values) > 1 else 0,
        q_max=q_values[-1] if len(q_values) > 0 else 0,
        pixel_size_nm=pixel_size_nm,
        tile_size=tile_size
    )


def analyze_tile_peaks(tile: np.ndarray, pixel_size_nm: float, 
                       params: dict = None) -> TilePeakResult:
    """
    Analyze a tile for crystalline peaks in target q-range.
    
    Args:
        tile: 2D tile data
        pixel_size_nm: pixel size in nm
        params: analysis parameters
        
    Returns:
        TilePeakResult with peak detection results
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    tile_size = tile.shape[0]
    q_min, q_max = p['q_range']
    threshold = p['intensity_threshold']
    
    # Apply Hann window
    hann_1d = windows.hann(tile_size)
    window = np.outer(hann_1d, hann_1d)
    windowed = tile.astype(np.float64) * window
    
    # Compute FFT
    fft = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted)**2
    
    center = tile_size // 2
    
    # Mask DC component
    dc_radius = p['dc_mask_radius']
    y, x = np.ogrid[:tile_size, :tile_size]
    dc_mask = ((y - center)**2 + (x - center)**2) <= dc_radius**2
    power_masked = power.copy()
    power_masked[dc_mask] = 0
    
    # Create q-space mask for target range
    r = np.sqrt((x - center)**2 + (y - center)**2)
    q_scale = 1.0 / (tile_size * pixel_size_nm)
    q = r * q_scale
    
    q_mask = (q >= q_min) & (q <= q_max)
    
    # Find intensity in target q-range
    target_power = power_masked * q_mask
    intensities_in_annulus = target_power[q_mask]

    # Select peak detection method
    peak_method = p.get('peak_method', 'percentile95')

    if peak_method == 'percentile95':
        # Robust method: mean of top 5% intensities
        if len(intensities_in_annulus) > 0:
            p95_threshold = np.percentile(intensities_in_annulus, 95)
            high_intensities = intensities_in_annulus[intensities_in_annulus > p95_threshold]
            if len(high_intensities) > 0:
                peak_intensity = np.mean(high_intensities)
            else:
                peak_intensity = np.max(intensities_in_annulus)
        else:
            peak_intensity = 0
    elif peak_method == 'max':
        # Original method: single pixel maximum
        peak_intensity = np.max(target_power)
    else:
        raise ValueError(f"Unknown peak_method: {peak_method}")

    if peak_intensity < threshold:
        return TilePeakResult(
            has_peak=False,
            peak_intensity=peak_intensity,
            peak_q=0,
            orientation_deg=0,
            confidence=0
        )

    # Find peak location
    if peak_method == 'max':
        # Find single brightest pixel
        peak_idx = np.unravel_index(np.argmax(target_power), target_power.shape)
        peak_y, peak_x = peak_idx
    else:  # percentile95
        # Find centroid of top 5% intensities
        p95_threshold = np.percentile(intensities_in_annulus, 95)
        high_intensity_mask = (target_power > p95_threshold) & q_mask

        if np.any(high_intensity_mask):
            # Weighted centroid
            weights = target_power[high_intensity_mask]
            coords = np.argwhere(high_intensity_mask)
            peak_y = np.average(coords[:, 0], weights=weights)
            peak_x = np.average(coords[:, 1], weights=weights)
        else:
            # Fallback to max
            peak_idx = np.unravel_index(np.argmax(target_power), target_power.shape)
            peak_y, peak_x = peak_idx
    
    # Compute q at peak
    peak_r = np.sqrt((peak_x - center)**2 + (peak_y - center)**2)
    peak_q = peak_r * q_scale
    
    # Compute orientation (angle from center)
    # atan2 gives angle from -180 to 180
    dx = peak_x - center
    dy = peak_y - center
    orientation = np.degrees(np.arctan2(dy, dx))
    
    # Confidence based on peak intensity relative to threshold
    confidence = min(1.0, peak_intensity / (threshold * 3))
    
    return TilePeakResult(
        has_peak=True,
        peak_intensity=peak_intensity,
        peak_q=peak_q,
        orientation_deg=orientation,
        confidence=confidence
    )


def compute_global_radial_profile(image: np.ndarray, pixel_size_nm: float,
                                   tile_size: int = 256) -> RadialProfile:
    """
    Compute global radial profile from center tile of image.
    
    Args:
        image: full preprocessed image
        pixel_size_nm: pixel size in nm
        tile_size: tile size to use for FFT
        
    Returns:
        RadialProfile from center region
    """
    h, w = image.shape
    
    # Use center tile
    cy, cx = h // 2, w // 2
    half = tile_size // 2
    
    center_tile = image[cy - half:cy + half, cx - half:cx + half]
    
    # Apply window
    hann_1d = windows.hann(tile_size)
    window = np.outer(hann_1d, hann_1d)
    windowed = center_tile.astype(np.float64) * window
    
    # FFT
    fft = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted)**2
    
    return compute_radial_profile(power, pixel_size_nm, tile_size)


def process_tiles_for_peaks(image: np.ndarray, pixel_size_nm: float,
                            params: dict = None, verbose: bool = True
                            ) -> Dict[str, np.ndarray]:
    """
    Process all tiles and detect peaks in target q-range.
    
    Returns maps of:
    - peak_mask: bool array where peaks detected
    - orientation_map: angle values where peaks detected
    - intensity_map: peak intensities
    - confidence_map: detection confidence
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    tile_size = p['tile_size']
    stride = p['stride']
    
    h, w = image.shape
    n_rows = (h - tile_size) // stride + 1
    n_cols = (w - tile_size) // stride + 1
    
    if verbose:
        print(f"[Radial Analysis] Processing {n_rows}x{n_cols} = {n_rows*n_cols} tiles")
        print(f"  Q-range: {p['q_range'][0]:.2f} - {p['q_range'][1]:.2f} nm^-1")
        print(f"  Threshold: {p['intensity_threshold']}")
    
    # Output grids
    peak_mask = np.zeros((n_rows, n_cols), dtype=bool)
    orientation_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    intensity_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    confidence_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    n_peaks = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * stride
            x = col * stride
            
            tile = image[y:y + tile_size, x:x + tile_size]
            
            result = analyze_tile_peaks(tile, pixel_size_nm, params)
            
            peak_mask[row, col] = result.has_peak
            orientation_map[row, col] = result.orientation_deg
            intensity_map[row, col] = result.peak_intensity
            confidence_map[row, col] = result.confidence
            
            if result.has_peak:
                n_peaks += 1
    
    if verbose:
        total = n_rows * n_cols
        pct = n_peaks / total * 100 if total > 0 else 0
        print(f"  Peaks detected: {n_peaks}/{total} tiles ({pct:.1f}%)")
    
    return {
        'peak_mask': peak_mask,
        'orientation_map': orientation_map,
        'intensity_map': intensity_map,
        'confidence_map': confidence_map,
        'grid_shape': (n_rows, n_cols),
        'n_peaks': n_peaks,
        'params': p,
    }


def save_radial_profile_plot(profile: RadialProfile, q_range: Tuple[float, float],
                              output_path: str, title: str = None, log_scale: bool = True):
    """Save radial profile plot with highlighted q-range."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter out zero/negative values for log scale
    q_vals = profile.q_values
    intensity = profile.intensity
    
    if log_scale:
        # Mask zeros for log scale
        mask = intensity > 0
        q_plot = q_vals[mask]
        i_plot = intensity[mask]
    else:
        q_plot = q_vals
        i_plot = intensity
    
    ax.plot(q_plot, i_plot, 'b-', linewidth=1)
    
    # Highlight selected range
    q_min, q_max = q_range
    ax.axvspan(q_min, q_max, alpha=0.3, color='red', label='Selected Range')
    
    ax.set_xlabel('q (1/nm)')
    ax.set_ylabel('Intensity')
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Intensity (log scale)')
    
    ax.set_title(title or f'Enhanced Radial Profile (q={q_min}-{q_max})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_peak_location_map(image: np.ndarray, peak_mask: np.ndarray,
                           stride: int, tile_size: int,
                           output_path: str, params: dict = None,
                           pixel_size_nm: float = None):
    """
    Save peak location map overlaid on original image.
    Green = peak detected, gray = no peak
    """
    p = params or {}
    q_range = p.get('q_range', (2.3, 2.6))
    threshold = p.get('intensity_threshold', 45000)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Show original image
    ax.imshow(image, cmap='gray', alpha=1.0)

    # Create overlay
    h, w = image.shape
    overlay = np.zeros((h, w, 4))  # RGBA

    n_rows, n_cols = peak_mask.shape

    for row in range(n_rows):
        for col in range(n_cols):
            y = row * stride
            x = col * stride

            if peak_mask[row, col]:
                # Green with alpha
                overlay[y:y+tile_size, x:x+tile_size, 1] = 0.6  # G
                overlay[y:y+tile_size, x:x+tile_size, 3] = 0.7  # A

    ax.imshow(overlay)

    # Add scale bar
    if pixel_size_nm:
        from src.viz import add_scalebar
        add_scalebar(ax, pixel_size_nm, image.shape, location='lower right', color='white')

    ax.set_title(f'Peak Locations (Green)\nq={q_range[0]}-{q_range[1]} | Thresh={threshold}')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_orientation_map(image: np.ndarray, peak_mask: np.ndarray,
                         orientation_map: np.ndarray,
                         stride: int, tile_size: int,
                         output_path: str, params: dict = None,
                         pixel_size_nm: float = None):
    """
    Save orientation map with color-coded angles.
    Uses cyclic colormap for angle visualization.
    """
    p = params or {}
    q_range = p.get('q_range', (2.3, 2.6))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Show original image
    ax.imshow(image, cmap='gray', alpha=1.0)

    # Create colored overlay based on orientation
    h, w = image.shape
    overlay = np.zeros((h, w, 4))  # RGBA

    n_rows, n_cols = peak_mask.shape

    for row in range(n_rows):
        for col in range(n_cols):
            if not peak_mask[row, col]:
                continue

            y = row * stride
            x = col * stride
            angle = orientation_map[row, col]

            # Map angle (-180 to 180) to hue (0 to 1)
            hue = (angle + 180) / 360

            # HSV to RGB
            rgb = hsv_to_rgb([hue, 0.8, 0.9])

            overlay[y:y+tile_size, x:x+tile_size, 0] = rgb[0]
            overlay[y:y+tile_size, x:x+tile_size, 1] = rgb[1]
            overlay[y:y+tile_size, x:x+tile_size, 2] = rgb[2]
            overlay[y:y+tile_size, x:x+tile_size, 3] = 0.7

    ax.imshow(overlay)

    # Add scale bar
    if pixel_size_nm:
        from src.viz import add_scalebar
        add_scalebar(ax, pixel_size_nm, image.shape, location='lower right', color='white')

    # Add colorbar for angle
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(-180, 180))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Angle (Degrees)')

    ax.set_title(f'Orientation Map\nq={q_range[0]}-{q_range[1]}')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_detection_heatmap(intensity_map: np.ndarray, output_path: str,
                           title: str = "Detection Heatmap"):
    """Save intensity heatmap of detections."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(intensity_map, cmap='plasma', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Detections')
    
    ax.set_title(title)
    ax.set_xlabel('Tile Column')
    ax.set_ylabel('Tile Row')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_fft_power_spectrum(image: np.ndarray, pixel_size_nm: float,
                            output_path: str, q_range: Tuple[float, float] = None,
                            tile_size: int = 256):
    """
    Save 2D FFT power spectrum visualization from center tile.
    
    Shows log-scale power spectrum with optional q-range annulus overlay.
    """
    h, w = image.shape
    cy, cx = h // 2, w // 2
    half = tile_size // 2
    
    # Extract center tile
    center_tile = image[cy - half:cy + half, cx - half:cx + half]
    
    # Apply Hann window
    hann_1d = windows.hann(tile_size)
    window = np.outer(hann_1d, hann_1d)
    windowed = center_tile.astype(np.float64) * window
    
    # Compute FFT
    fft = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted)**2
    
    # Log scale for visualization
    log_power = np.log10(power + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(log_power, cmap='inferno', origin='lower')
    plt.colorbar(im, ax=ax, label='Log₁₀(Power + 1)')
    
    # Add q-range annulus if specified
    if q_range is not None:
        q_min, q_max = q_range
        q_scale = 1.0 / (tile_size * pixel_size_nm)
        r_min = q_min / q_scale
        r_max = q_max / q_scale
        
        center = tile_size // 2
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Inner circle
        x_inner = center + r_min * np.cos(theta)
        y_inner = center + r_min * np.sin(theta)
        ax.plot(x_inner, y_inner, 'c--', linewidth=1.5, label=f'q={q_min:.2f} nm⁻¹')
        
        # Outer circle
        x_outer = center + r_max * np.cos(theta)
        y_outer = center + r_max * np.sin(theta)
        ax.plot(x_outer, y_outer, 'c-', linewidth=1.5, label=f'q={q_max:.2f} nm⁻¹')
        
        ax.legend(loc='upper right')
    
    ax.set_title(f'FFT Power Spectrum (Center {tile_size}×{tile_size} tile)\nPixel: {pixel_size_nm:.4f} nm')
    ax.set_xlabel('Frequency X')
    ax.set_ylabel('Frequency Y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_radial_analysis(image: np.ndarray, pixel_size_nm: float,
                        output_dir: str = 'outputs',
                        params: dict = None,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Run complete radial analysis pipeline.
    
    Generates:
    1. Radial profile plot
    2. Peak location map
    3. Orientation map
    4. Detection heatmap
    
    Returns results dict with all computed data.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "=" * 60)
        print("RADIAL ANALYSIS")
        print("=" * 60)
    
    # 1. Compute global radial profile
    if verbose:
        print("\n[1] Computing radial profile...")
    
    profile = compute_global_radial_profile(
        image, pixel_size_nm, tile_size=p['tile_size']
    )
    
    save_radial_profile_plot(
        profile, p['q_range'],
        str(output_path / '2_Radial_Profile.png'),
        title=f"Enhanced Radial Profile (q={p['q_range'][0]}-{p['q_range'][1]})"
    )

    # Also save 2D FFT power spectrum
    save_fft_power_spectrum(
        image, pixel_size_nm,
        str(output_path / '3_FFT_Power_Spectrum.png'),
        q_range=p['q_range'],
        tile_size=p['tile_size']
    )

    if verbose:
        print(f"  Q range: {profile.q_min:.3f} - {profile.q_max:.3f} nm^-1")
        print(f"  Saved: 2_Radial_Profile.png")
        print(f"  Saved: 3_FFT_Power_Spectrum.png")
    
    # 2. Process tiles for peak detection
    if verbose:
        print("\n[2] Detecting peaks in target q-range...")
    
    peak_results = process_tiles_for_peaks(
        image, pixel_size_nm, params=p, verbose=verbose
    )
    
    # 3. Save peak location map
    if verbose:
        print("\n[3] Generating peak location map...")
    
    save_peak_location_map(
        image, peak_results['peak_mask'],
        p['stride'], p['tile_size'],
        str(output_path / '4_Peak_Location_Map.png'),
        params=p,
        pixel_size_nm=pixel_size_nm
    )

    if verbose:
        print(f"  Saved: 4_Peak_Location_Map.png")

    # 4. Save orientation map
    if verbose:
        print("\n[4] Generating orientation map...")

    save_orientation_map(
        image, peak_results['peak_mask'],
        peak_results['orientation_map'],
        p['stride'], p['tile_size'],
        str(output_path / '5_Orientation_Map.png'),
        params=p,
        pixel_size_nm=pixel_size_nm
    )

    if verbose:
        print(f"  Saved: 5_Orientation_Map.png")
    
    # 5. Save detection heatmap
    if verbose:
        print("\n[5] Generating detection heatmap...")
    
    # For heatmap, count detections per grid cell (like Ilastik)
    save_detection_heatmap(
        peak_results['intensity_map'],
        str(output_path / 'Heatmap.png'),
        title="Peak Detection Heatmap"
    )
    
    if verbose:
        print(f"  Saved: Heatmap.png")
        print("\n" + "=" * 60)
        print("RADIAL ANALYSIS COMPLETE")
        print("=" * 60)
    
    return {
        'profile': profile,
        'peak_results': peak_results,
        'params': p,
    }


if __name__ == '__main__':
    """Test radial analysis on preprocessed image."""
    import sys
    
    # Load preprocessed image
    artifacts_path = Path('artifacts')
    if not (artifacts_path / 'preprocessed.npy').exists():
        print("Error: Run preprocessing first (artifacts/preprocessed.npy not found)")
        sys.exit(1)
    
    image = np.load(artifacts_path / 'preprocessed.npy')
    
    # Load pixel size from metadata
    import json
    meta_path = artifacts_path / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        pixel_size = meta.get('pixel_size_nm', 0.127)
    else:
        pixel_size = 0.127
    
    print(f"Image shape: {image.shape}")
    print(f"Pixel size: {pixel_size} nm")
    
    # Run analysis
    results = run_radial_analysis(
        image, pixel_size,
        output_dir='outputs',
        verbose=True
    )
    
    print(f"\nPeaks detected: {results['peak_results']['n_peaks']}")
