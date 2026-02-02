"""
Peak Discovery Module for STEM Domain Analysis.

Automatically discovers crystalline diffraction peaks in the radial FFT profile
without requiring prior knowledge of d-spacing.

Key functions:
- discover_peaks(): Main entry point - finds candidate q-ranges
- compute_enhanced_radial_profile(): Background-subtracted radial profile
- validate_spatial_coherence(): Check if results are meaningful
"""

import numpy as np
from scipy import ndimage
from scipy.signal import windows, find_peaks, savgol_filter
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
import warnings


@dataclass
class DiscoveredPeak:
    """A discovered diffraction peak."""
    q_center: float          # Peak center in nm⁻¹
    q_width: float           # Estimated FWHM in nm⁻¹
    d_spacing: float         # Corresponding d-spacing in nm
    prominence: float        # Peak prominence (height above background)
    intensity: float         # Absolute peak intensity
    snr: float              # Signal-to-noise ratio
    suggested_threshold: float  # Recommended threshold for this peak
    confidence: str          # 'high', 'medium', 'low'


@dataclass
class DiscoveryResult:
    """Results from peak discovery."""
    peaks: List[DiscoveredPeak]
    q_values: np.ndarray
    raw_profile: np.ndarray
    background: np.ndarray
    corrected_profile: np.ndarray
    noise_floor: float
    best_peak_idx: Optional[int]  # Index of recommended peak
    message: str


def compute_enhanced_radial_profile(
    image: np.ndarray,
    pixel_size_nm: float,
    tile_size: int = 256,
    n_sample_tiles: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute averaged radial profile from multiple tiles across the image.
    
    Sampling multiple tiles reduces noise and gives a representative profile.
    
    Args:
        image: Preprocessed image
        pixel_size_nm: Pixel size in nm
        tile_size: Size of FFT tiles
        n_sample_tiles: Number of tiles to sample
        seed: Random seed for reproducibility
        
    Returns:
        (q_values, averaged_radial_profile)
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape
    
    # Ensure we can fit tiles
    if h < tile_size or w < tile_size:
        raise ValueError(f"Image too small ({h}x{w}) for tile size {tile_size}")
    
    # Generate random tile positions (with some margin)
    margin = tile_size // 4
    n_tiles = min(n_sample_tiles, ((h - tile_size) // margin) * ((w - tile_size) // margin))
    
    y_positions = rng.integers(0, h - tile_size, size=n_tiles)
    x_positions = rng.integers(0, w - tile_size, size=n_tiles)
    
    # Prepare FFT infrastructure
    hann = np.outer(windows.hann(tile_size), windows.hann(tile_size))
    center = tile_size // 2
    
    # Radial distance array
    y_grid, x_grid = np.ogrid[:tile_size, :tile_size]
    r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
    r_int = r.astype(int)
    max_r = min(center, r_int.max())
    
    # Accumulate radial profiles
    radial_sum = np.zeros(max_r + 1)
    radial_count = np.zeros(max_r + 1)
    
    for y, x in zip(y_positions, x_positions):
        tile = image[y:y+tile_size, x:x+tile_size].astype(np.float64) * hann
        fft = np.fft.fftshift(np.fft.fft2(tile))
        power = np.abs(fft)**2
        
        # Radial binning
        for ri in range(max_r + 1):
            mask = r_int == ri
            radial_sum[ri] += np.sum(power[mask])
            radial_count[ri] += np.sum(mask)
    
    # Average
    radial_count[radial_count == 0] = 1
    radial_avg = radial_sum / radial_count / n_tiles
    
    # Convert to q-space
    q_scale = 1.0 / (tile_size * pixel_size_nm)
    q_values = np.arange(max_r + 1) * q_scale
    
    return q_values, radial_avg


def fit_background(q_values: np.ndarray, profile: np.ndarray,
                   degree: int = 5, exclude_dc: int = 3) -> np.ndarray:
    """
    Fit polynomial background to the radial profile.
    
    Uses iterative fitting that ignores peaks (values significantly above fit).
    
    Args:
        q_values: Q values (nm⁻¹)
        profile: Radial intensity profile
        degree: Polynomial degree
        exclude_dc: Number of low-q points to exclude (DC component)
        
    Returns:
        Fitted background curve
    """
    # Work in log space for better polynomial fit of power-law decay
    valid = (profile > 0) & (np.arange(len(profile)) >= exclude_dc)
    q_fit = q_values[valid]
    log_profile = np.log10(profile[valid] + 1e-10)
    
    # Initial fit
    weights = np.ones_like(log_profile)
    
    # Iterative reweighting to ignore peaks
    for _ in range(3):
        coeffs = np.polyfit(q_fit, log_profile, degree, w=weights)
        fit = np.polyval(coeffs, q_fit)
        residual = log_profile - fit
        
        # Downweight points significantly above fit (potential peaks)
        mad = np.median(np.abs(residual - np.median(residual)))
        threshold = 2.0 * mad * 1.4826
        weights = np.where(residual > threshold, 0.1, 1.0)
    
    # Generate full background
    background = np.zeros_like(profile)
    background[valid] = 10**np.polyval(coeffs, q_values[valid])
    background[:exclude_dc] = profile[:exclude_dc]  # Don't subtract DC region
    
    return background


def find_diffraction_peaks(
    q_values: np.ndarray,
    corrected_profile: np.ndarray,
    noise_floor: float,
    min_q: float = 0.5,
    max_q: float = 15.0,
    min_prominence_snr: float = 2.0
) -> List[Dict]:
    """
    Find diffraction peaks in background-corrected radial profile.
    
    Args:
        q_values: Q values (nm⁻¹)
        corrected_profile: Background-subtracted profile
        noise_floor: Estimated noise level
        min_q: Minimum q to consider (avoid DC artifacts)
        max_q: Maximum q to consider
        min_prominence_snr: Minimum peak prominence in units of noise
        
    Returns:
        List of peak dictionaries with properties
    """
    # Apply q-range filter
    valid_mask = (q_values >= min_q) & (q_values <= max_q)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 10:
        return []
    
    # Smooth profile slightly for peak detection
    profile_smooth = savgol_filter(corrected_profile[valid_mask], 
                                   window_length=min(11, len(valid_indices)//2*2+1),
                                   polyorder=2)
    
    # Find peaks with prominence filter
    min_prominence = noise_floor * min_prominence_snr
    peaks_idx, properties = find_peaks(
        profile_smooth,
        prominence=min_prominence,
        width=2,
        distance=5
    )
    
    # Convert back to original indices
    peak_results = []
    for i, local_idx in enumerate(peaks_idx):
        global_idx = valid_indices[local_idx]
        
        peak_q = q_values[global_idx]
        peak_intensity = corrected_profile[global_idx]
        prominence = properties['prominences'][i]
        
        # Estimate width (FWHM)
        if 'widths' in properties:
            width_samples = properties['widths'][i]
            q_step = q_values[1] - q_values[0] if len(q_values) > 1 else 0.01
            width_q = width_samples * q_step
        else:
            width_q = 0.2  # Default width
        
        peak_results.append({
            'q_center': float(peak_q),
            'q_width': float(width_q),
            'd_spacing': float(1.0 / peak_q) if peak_q > 0 else 0,
            'intensity': float(peak_intensity),
            'prominence': float(prominence),
            'snr': float(prominence / noise_floor) if noise_floor > 0 else 0,
            'index': int(global_idx),
        })
    
    # Sort by prominence (strongest first)
    peak_results.sort(key=lambda x: x['prominence'], reverse=True)
    
    return peak_results


def estimate_threshold_for_peak(
    image: np.ndarray,
    pixel_size_nm: float,
    q_center: float,
    q_width: float,
    tile_size: int = 256,
    n_sample_tiles: int = 200,
    target_detection_pct: float = 15.0
) -> Tuple[float, Dict]:
    """
    Estimate appropriate threshold for a specific q-range.
    
    Samples tiles and finds threshold that gives target detection percentage.
    
    Args:
        image: Preprocessed image
        pixel_size_nm: Pixel size
        q_center: Center of q-range
        q_width: Width of q-range (±)
        tile_size: Tile size
        n_sample_tiles: Number of tiles to sample
        target_detection_pct: Target percentage of tiles to mark as crystalline
        
    Returns:
        (suggested_threshold, stats_dict)
    """
    rng = np.random.default_rng(42)
    h, w = image.shape
    
    q_min = q_center - q_width
    q_max = q_center + q_width
    
    # Sample tiles
    n_tiles = min(n_sample_tiles, (h // tile_size) * (w // tile_size))
    y_positions = rng.integers(0, h - tile_size, size=n_tiles)
    x_positions = rng.integers(0, w - tile_size, size=n_tiles)
    
    # Setup
    hann = np.outer(windows.hann(tile_size), windows.hann(tile_size))
    center = tile_size // 2
    y_grid, x_grid = np.ogrid[:tile_size, :tile_size]
    r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
    q_scale = 1.0 / (tile_size * pixel_size_nm)
    q = r * q_scale
    q_mask = (q >= q_min) & (q <= q_max)
    
    intensities = []
    for y, x in zip(y_positions, x_positions):
        tile = image[y:y+tile_size, x:x+tile_size].astype(np.float64) * hann
        fft = np.fft.fftshift(np.fft.fft2(tile))
        power = np.abs(fft)**2
        peak_intensity = np.max(power * q_mask)
        intensities.append(peak_intensity)
    
    intensities = np.array(intensities)
    
    # Find threshold for target detection percentage
    target_percentile = 100 - target_detection_pct
    suggested_threshold = np.percentile(intensities, target_percentile)
    
    stats = {
        'min': float(np.min(intensities)),
        'max': float(np.max(intensities)),
        'mean': float(np.mean(intensities)),
        'median': float(np.median(intensities)),
        'std': float(np.std(intensities)),
        'p50': float(np.percentile(intensities, 50)),
        'p75': float(np.percentile(intensities, 75)),
        'p85': float(np.percentile(intensities, 85)),
        'p90': float(np.percentile(intensities, 90)),
        'p95': float(np.percentile(intensities, 95)),
        'n_samples': len(intensities),
        'target_pct': target_detection_pct,
    }
    
    return suggested_threshold, stats


def discover_peaks(
    image: np.ndarray,
    pixel_size_nm: float,
    tile_size: int = 256,
    min_q: float = 0.3,
    max_q: float = 15.0,
    target_detection_pct: float = 15.0,
    verbose: bool = True
) -> DiscoveryResult:
    """
    Main entry point: Discover crystalline diffraction peaks automatically.
    
    Args:
        image: Preprocessed image
        pixel_size_nm: Pixel size in nm
        tile_size: FFT tile size
        min_q: Minimum q to search (nm⁻¹)
        max_q: Maximum q to search (nm⁻¹)
        target_detection_pct: Target % of tiles to detect as crystalline
        verbose: Print progress
        
    Returns:
        DiscoveryResult with found peaks and analysis data
    """
    if verbose:
        print("\n" + "=" * 60)
        print("AUTOMATIC PEAK DISCOVERY")
        print("=" * 60)
    
    # Step 1: Compute averaged radial profile
    if verbose:
        print("\n[1] Computing averaged radial profile from sampled tiles...")
    
    q_values, raw_profile = compute_enhanced_radial_profile(
        image, pixel_size_nm, tile_size, n_sample_tiles=200
    )
    
    if verbose:
        print(f"    Q-range covered: {q_values[1]:.3f} - {q_values[-1]:.3f} nm⁻¹")
    
    # Step 2: Fit and subtract background
    if verbose:
        print("\n[2] Fitting polynomial background...")
    
    background = fit_background(q_values, raw_profile, degree=6, exclude_dc=5)
    corrected_profile = raw_profile - background
    
    # Estimate noise floor from negative regions (should be near zero if good fit)
    negative_vals = corrected_profile[corrected_profile < 0]
    if len(negative_vals) > 10:
        noise_floor = np.std(negative_vals) * 1.4826  # MAD-based estimate
    else:
        noise_floor = np.std(corrected_profile[corrected_profile < np.median(corrected_profile)])
    
    if verbose:
        print(f"    Estimated noise floor: {noise_floor:.1f}")
    
    # Step 3: Find peaks
    if verbose:
        print("\n[3] Searching for diffraction peaks...")
    
    raw_peaks = find_diffraction_peaks(
        q_values, corrected_profile, noise_floor,
        min_q=min_q, max_q=max_q, min_prominence_snr=2.0
    )
    
    if verbose:
        print(f"    Found {len(raw_peaks)} candidate peaks")
    
    # Step 4: Characterize each peak and estimate thresholds
    discovered_peaks = []
    
    for i, p in enumerate(raw_peaks[:5]):  # Top 5 peaks max
        if verbose:
            print(f"\n[4.{i+1}] Characterizing peak at q={p['q_center']:.3f} nm⁻¹ (d={p['d_spacing']:.3f} nm)...")
        
        # Estimate threshold for this peak
        q_width = max(0.15, p['q_width'] * 1.5)  # At least 0.15 nm⁻¹ width
        threshold, stats = estimate_threshold_for_peak(
            image, pixel_size_nm,
            p['q_center'], q_width,
            tile_size=tile_size,
            target_detection_pct=target_detection_pct
        )
        
        # Assign confidence
        if p['snr'] > 5:
            confidence = 'high'
        elif p['snr'] > 3:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        discovered_peak = DiscoveredPeak(
            q_center=p['q_center'],
            q_width=q_width,
            d_spacing=p['d_spacing'],
            prominence=p['prominence'],
            intensity=p['intensity'],
            snr=p['snr'],
            suggested_threshold=threshold,
            confidence=confidence
        )
        discovered_peaks.append(discovered_peak)
        
        if verbose:
            print(f"        SNR: {p['snr']:.1f}, Confidence: {confidence}")
            print(f"        Suggested threshold: {threshold:.0f}")
            print(f"        Intensity range: {stats['min']:.0f} - {stats['max']:.0f}")
    
    # Determine best peak
    best_idx = None
    if discovered_peaks:
        # Prefer high-confidence peaks, then highest SNR
        high_conf = [(i, p) for i, p in enumerate(discovered_peaks) if p.confidence == 'high']
        if high_conf:
            best_idx = max(high_conf, key=lambda x: x[1].snr)[0]
        else:
            med_conf = [(i, p) for i, p in enumerate(discovered_peaks) if p.confidence == 'medium']
            if med_conf:
                best_idx = max(med_conf, key=lambda x: x[1].snr)[0]
            elif discovered_peaks:
                best_idx = 0  # Highest prominence peak
    
    # Generate message
    if not discovered_peaks:
        message = "No significant diffraction peaks found. Image may be amorphous or parameters need adjustment."
    elif best_idx is not None:
        bp = discovered_peaks[best_idx]
        message = (f"Recommended: q={bp.q_center:.2f}±{bp.q_width:.2f} nm⁻¹ "
                   f"(d={bp.d_spacing:.3f} nm), threshold={bp.suggested_threshold:.0f} "
                   f"[{bp.confidence} confidence]")
    else:
        message = f"Found {len(discovered_peaks)} peaks but none with high confidence."
    
    if verbose:
        print("\n" + "=" * 60)
        print("DISCOVERY SUMMARY")
        print("=" * 60)
        print(f"  {message}")
        if discovered_peaks:
            print(f"\n  All discovered peaks:")
            print(f"  {'#':<3} {'q (nm⁻¹)':<10} {'d (nm)':<10} {'SNR':<8} {'Threshold':<12} {'Conf.':<8}")
            print(f"  {'-'*55}")
            for i, p in enumerate(discovered_peaks):
                marker = " *" if i == best_idx else "  "
                print(f"{marker}{i+1:<3} {p.q_center:<10.3f} {p.d_spacing:<10.3f} "
                      f"{p.snr:<8.1f} {p.suggested_threshold:<12.0f} {p.confidence:<8}")
        print("=" * 60)
    
    return DiscoveryResult(
        peaks=discovered_peaks,
        q_values=q_values,
        raw_profile=raw_profile,
        background=background,
        corrected_profile=corrected_profile,
        noise_floor=noise_floor,
        best_peak_idx=best_idx,
        message=message
    )


def validate_spatial_coherence(
    peak_mask: np.ndarray,
    orientation_map: np.ndarray,
    min_domain_size: int = 4
) -> Dict:
    """
    Validate that detected peaks form spatially coherent domains.
    
    High coherence = real crystalline domains
    Low coherence = likely noise (random detections)
    
    Args:
        peak_mask: Boolean mask of detected peaks (grid)
        orientation_map: Orientation values for each tile
        min_domain_size: Minimum connected region size to count
        
    Returns:
        Dict with coherence metrics
    """
    from scipy import ndimage as ndi
    
    n_detected = np.sum(peak_mask)
    total_tiles = peak_mask.size
    detection_rate = n_detected / total_tiles if total_tiles > 0 else 0
    
    if n_detected < 10:
        return {
            'coherence_score': 0.0,
            'detection_rate': detection_rate,
            'n_domains': 0,
            'largest_domain': 0,
            'orientation_entropy': 1.0,
            'interpretation': 'insufficient_detections',
            'is_valid': False,
        }
    
    # Find connected components
    labeled, n_domains = ndi.label(peak_mask)
    domain_sizes = ndi.sum(peak_mask, labeled, index=range(1, n_domains + 1))
    
    # Filter small domains
    significant_domains = np.sum(domain_sizes >= min_domain_size)
    largest_domain = np.max(domain_sizes) if len(domain_sizes) > 0 else 0
    
    # Compute orientation entropy (how random are the orientations?)
    orientations = orientation_map[peak_mask]
    # Bin orientations into 12 sectors (30° each)
    orientation_bins = np.histogram(orientations, bins=12, range=(-180, 180))[0]
    orientation_probs = orientation_bins / orientation_bins.sum()
    orientation_probs = orientation_probs[orientation_probs > 0]
    orientation_entropy = -np.sum(orientation_probs * np.log2(orientation_probs)) / np.log2(12)  # Normalized
    
    # Compute local coherence (do neighbors have similar orientations?)
    # This is more indicative than global entropy
    neighbor_agreement = 0
    neighbor_count = 0
    
    n_rows, n_cols = peak_mask.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if not peak_mask[r, c]:
                continue
            angle = orientation_map[r, c]
            
            # Check 4-neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and peak_mask[nr, nc]:
                    neighbor_angle = orientation_map[nr, nc]
                    # Angle difference (handling wrap-around)
                    diff = abs(angle - neighbor_angle)
                    diff = min(diff, 360 - diff)
                    if diff < 30:  # Within 30° counts as agreement
                        neighbor_agreement += 1
                    neighbor_count += 1
    
    local_coherence = neighbor_agreement / neighbor_count if neighbor_count > 0 else 0
    
    # Overall coherence score
    # High score = spatially clustered detections + locally similar orientations
    spatial_clustering = largest_domain / n_detected if n_detected > 0 else 0
    coherence_score = 0.5 * local_coherence + 0.3 * spatial_clustering + 0.2 * (1 - orientation_entropy)
    
    # Interpretation
    if coherence_score > 0.5:
        interpretation = 'good_domains'
        is_valid = True
    elif coherence_score > 0.3:
        interpretation = 'weak_domains'
        is_valid = True
    elif detection_rate > 0.5:
        interpretation = 'likely_noise_high_detection'
        is_valid = False
    else:
        interpretation = 'sparse_or_noise'
        is_valid = detection_rate > 0.05  # At least some signal
    
    return {
        'coherence_score': float(coherence_score),
        'detection_rate': float(detection_rate),
        'n_domains': int(significant_domains),
        'largest_domain': int(largest_domain),
        'local_coherence': float(local_coherence),
        'orientation_entropy': float(orientation_entropy),
        'interpretation': interpretation,
        'is_valid': is_valid,
    }


def save_discovery_plot(
    result: DiscoveryResult,
    output_path: str,
    title: str = "Peak Discovery Analysis"
):
    """Save visualization of peak discovery results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Raw profile with background
    ax1 = axes[0]
    ax1.semilogy(result.q_values, result.raw_profile, 'b-', label='Raw Profile', linewidth=1)
    ax1.semilogy(result.q_values, result.background, 'r--', label='Fitted Background', linewidth=1.5)
    ax1.set_xlabel('q (nm⁻¹)')
    ax1.set_ylabel('Intensity (log scale)')
    ax1.set_title('Radial Profile with Polynomial Background Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(15, result.q_values[-1]))
    
    # Bottom: Background-corrected with peaks marked
    ax2 = axes[1]
    ax2.plot(result.q_values, result.corrected_profile, 'b-', linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axhline(y=result.noise_floor, color='orange', linestyle='--', 
                label=f'Noise floor: {result.noise_floor:.0f}', linewidth=1)
    ax2.axhline(y=-result.noise_floor, color='orange', linestyle='--', linewidth=1)
    
    # Mark discovered peaks
    colors = ['green', 'red', 'purple', 'brown', 'pink']
    for i, peak in enumerate(result.peaks):
        color = colors[i % len(colors)]
        marker = '*' if i == result.best_peak_idx else 'o'
        ax2.axvline(x=peak.q_center, color=color, linestyle=':', alpha=0.7)
        ax2.scatter([peak.q_center], [peak.intensity], c=color, s=100, marker=marker, zorder=5)
        ax2.annotate(f'd={peak.d_spacing:.2f}nm\nSNR={peak.snr:.1f}',
                     (peak.q_center, peak.intensity),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=9, color=color)
    
    ax2.set_xlabel('q (nm⁻¹)')
    ax2.set_ylabel('Corrected Intensity')
    ax2.set_title(f'Background-Corrected Profile with Detected Peaks\n{result.message}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(15, result.q_values[-1]))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# Convenience function to get recommended parameters
def get_recommended_params(result: DiscoveryResult) -> Optional[Dict]:
    """
    Get recommended analysis parameters from discovery result.
    
    Returns None if no good peaks found.
    """
    if result.best_peak_idx is None or not result.peaks:
        return None
    
    peak = result.peaks[result.best_peak_idx]
    
    return {
        'd_min': peak.d_spacing - 0.02,  # Small margin
        'd_max': peak.d_spacing + 0.02,
        'q_min': peak.q_center - peak.q_width,
        'q_max': peak.q_center + peak.q_width,
        'intensity_threshold': peak.suggested_threshold,
        'confidence': peak.confidence,
        'snr': peak.snr,
    }
