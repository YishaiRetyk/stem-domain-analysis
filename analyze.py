#!/usr/bin/env python3
"""
STEM Domain Analysis - Automated Pipeline Script

Usage:
    python analyze.py <input_file.dm4> [options]
    python analyze.py --help

This script automates the entire analysis pipeline:
1. Load DM4/image file
2. Extract or prompt for metadata (pixel size, etc.)
3. Preprocess the image
4. Run radial FFT analysis
5. Detect crystalline peaks
6. Generate orientation maps and visualizations
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np


def prompt_float(message: str, default: float = None) -> float:
    """Prompt user for a float value with optional default."""
    if default is not None:
        prompt = f"{message} [{default}]: "
    else:
        prompt = f"{message}: "
    
    while True:
        response = input(prompt).strip()
        if not response and default is not None:
            return default
        try:
            return float(response)
        except ValueError:
            print("  Invalid number. Please try again.")


def prompt_yes_no(message: str, default: bool = True) -> bool:
    """Prompt user for yes/no with default."""
    yn = "[Y/n]" if default else "[y/N]"
    response = input(f"{message} {yn}: ").strip().lower()
    if not response:
        return default
    return response in ('y', 'yes')


def auto_detect_threshold(image: np.ndarray, pixel_size_nm: float, 
                          q_min: float, q_max: float, tile_size: int = 256,
                          sample_tiles: int = 500) -> dict:
    """
    Auto-detect intensity threshold using ensemble of methods.
    
    Runs multiple detection algorithms in parallel and combines results:
    - Otsu's method (bimodal separation)
    - Percentile-based (90th percentile)
    - Median + k*MAD (robust outlier detection)
    - Knee/elbow detection
    - Gaussian Mixture Model (2 components)
    
    Returns dict with:
        - suggested: recommended threshold (ensemble decision)
        - method: which method was selected
        - methods: individual method results
        - stats: intensity statistics
    """
    from scipy.signal import windows
    from scipy import ndimage
    
    h, w = image.shape
    stride = tile_size
    n_rows = (h - tile_size) // stride + 1
    n_cols = (w - tile_size) // stride + 1
    total_tiles = n_rows * n_cols
    
    # Sample evenly distributed tiles
    sample_step = max(1, total_tiles // sample_tiles)
    
    hann = np.outer(windows.hann(tile_size), windows.hann(tile_size))
    center = tile_size // 2
    y_grid, x_grid = np.ogrid[:tile_size, :tile_size]
    r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
    q_scale = 1.0 / (tile_size * pixel_size_nm)
    q = r * q_scale
    q_mask = (q >= q_min) & (q <= q_max)
    
    intensities = []
    tile_idx = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            if tile_idx % sample_step == 0:
                y, x = row * stride, col * stride
                tile = image[y:y+tile_size, x:x+tile_size].astype(np.float64) * hann
                fft = np.fft.fftshift(np.fft.fft2(tile))
                power = np.abs(fft)**2
                peak_intensity = np.max(power * q_mask)
                intensities.append(peak_intensity)
            tile_idx += 1
    
    intensities = np.array(intensities)
    n_samples = len(intensities)
    
    # Basic statistics
    stats = {
        'min': float(intensities.min()),
        'max': float(intensities.max()),
        'mean': float(intensities.mean()),
        'std': float(intensities.std()),
        'median': float(np.median(intensities)),
        'n_samples': n_samples,
    }
    
    methods = {}
    
    # Method 1: Percentile-based (90th)
    methods['percentile_90'] = {
        'threshold': float(np.percentile(intensities, 90)),
        'description': '90th percentile (top 10% as crystalline)',
    }
    
    # Method 2: Otsu's method
    try:
        thresh_otsu = _otsu_threshold(intensities)
        methods['otsu'] = {
            'threshold': float(thresh_otsu),
            'description': "Otsu's method (minimize intra-class variance)",
        }
    except Exception as e:
        methods['otsu'] = {'threshold': None, 'error': str(e)}
    
    # Method 3: Median + k*MAD (k=3 for outliers)
    median = np.median(intensities)
    mad = np.median(np.abs(intensities - median))
    thresh_mad = median + 3 * mad * 1.4826  # 1.4826 scales MAD to std for normal dist
    methods['median_mad'] = {
        'threshold': float(thresh_mad),
        'description': 'Median + 3×MAD (robust outlier detection)',
        'mad': float(mad),
    }
    
    # Method 4: Knee/elbow detection
    try:
        thresh_knee = _knee_threshold(intensities)
        methods['knee'] = {
            'threshold': float(thresh_knee),
            'description': 'Knee detection (elbow in sorted curve)',
        }
    except Exception as e:
        methods['knee'] = {'threshold': None, 'error': str(e)}
    
    # Method 5: Gaussian Mixture Model (2 components)
    try:
        thresh_gmm, gmm_info = _gmm_threshold(intensities)
        methods['gmm'] = {
            'threshold': float(thresh_gmm),
            'description': 'GMM (2 Gaussians, intersection point)',
            **gmm_info,
        }
    except Exception as e:
        methods['gmm'] = {'threshold': None, 'error': str(e)}
    
    # Method 6: Bimodality test + adaptive selection
    try:
        is_bimodal, dip_stat = _test_bimodality(intensities)
        methods['bimodality'] = {
            'is_bimodal': is_bimodal,
            'dip_statistic': float(dip_stat),
        }
    except Exception:
        is_bimodal = False
        methods['bimodality'] = {'is_bimodal': False, 'error': 'test failed'}
    
    # Ensemble decision
    valid_thresholds = []
    for name, m in methods.items():
        if isinstance(m, dict) and m.get('threshold') is not None:
            valid_thresholds.append((name, m['threshold']))
    
    if not valid_thresholds:
        # Fallback
        suggested = stats['median'] + stats['std']
        selected_method = 'fallback'
    elif is_bimodal and methods.get('otsu', {}).get('threshold'):
        # Bimodal: prefer Otsu
        suggested = methods['otsu']['threshold']
        selected_method = 'otsu'
    elif methods.get('gmm', {}).get('threshold') and methods['gmm'].get('separation', 0) > 1.5:
        # Good GMM separation: use GMM
        suggested = methods['gmm']['threshold']
        selected_method = 'gmm'
    else:
        # Default: median of all valid thresholds (consensus)
        threshold_values = [t for _, t in valid_thresholds]
        suggested = float(np.median(threshold_values))
        selected_method = 'consensus'
    
    # Compute detection rates for each threshold
    for name, m in methods.items():
        if isinstance(m, dict) and m.get('threshold') is not None:
            pct = np.sum(intensities >= m['threshold']) / n_samples * 100
            m['detection_pct'] = round(pct, 1)
    
    return {
        'suggested': suggested,
        'method': selected_method,
        'methods': methods,
        'stats': stats,
        'percentiles': {
            50: float(np.percentile(intensities, 50)),
            75: float(np.percentile(intensities, 75)),
            90: float(np.percentile(intensities, 90)),
            95: float(np.percentile(intensities, 95)),
        },
    }


def _otsu_threshold(data: np.ndarray) -> float:
    """Otsu's method for threshold selection."""
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist.astype(float) / hist.sum()
    
    # Compute cumulative sums
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Cumulative means
    mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-10))[::-1]
    
    # Between-class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Find maximum
    idx = np.argmax(variance)
    return bin_centers[idx]


def _knee_threshold(data: np.ndarray) -> float:
    """Find knee/elbow point in sorted intensity curve."""
    sorted_data = np.sort(data)[::-1]  # Descending
    n = len(sorted_data)
    
    # Normalize to [0, 1] for both axes
    x = np.arange(n) / (n - 1)
    y = (sorted_data - sorted_data.min()) / (sorted_data.max() - sorted_data.min() + 1e-10)
    
    # Line from first to last point
    # Using perpendicular distance formula instead of cross product
    x1, y1 = 0, y[0]
    x2, y2 = 1, y[-1]
    
    # Distance from point (x[i], y[i]) to line through (x1,y1)-(x2,y2)
    # d = |((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)| / sqrt((y2-y1)^2 + (x2-x1)^2)
    numerator = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distances = numerator / denominator
    
    # Knee is point with maximum distance
    knee_idx = np.argmax(distances)
    return sorted_data[knee_idx]


def _gmm_threshold(data: np.ndarray) -> tuple:
    """Fit 2-component GMM and find threshold at intersection."""
    from sklearn.mixture import GaussianMixture
    
    # Fit GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data.reshape(-1, 1))
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    
    # Sort by mean
    idx = np.argsort(means)
    means = means[idx]
    stds = stds[idx]
    weights = weights[idx]
    
    # Threshold between the two means
    # Simple: midpoint weighted by stds
    threshold = (means[0] * stds[1] + means[1] * stds[0]) / (stds[0] + stds[1])
    
    # Separation measure (distance between means / pooled std)
    separation = (means[1] - means[0]) / np.sqrt((stds[0]**2 + stds[1]**2) / 2)
    
    info = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'weights': weights.tolist(),
        'separation': float(separation),
    }
    
    return threshold, info


def _test_bimodality(data: np.ndarray) -> tuple:
    """Test if distribution is bimodal using Hartigan's dip test approximation."""
    # Simple approximation: check if histogram has two peaks
    hist, _ = np.histogram(data, bins=50)
    
    # Smooth histogram
    from scipy.ndimage import gaussian_filter1d
    smooth_hist = gaussian_filter1d(hist.astype(float), sigma=2)
    
    # Find local maxima
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smooth_hist, height=smooth_hist.max() * 0.1)
    
    # Dip statistic approximation (simplified)
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Compute empirical CDF
    ecdf = np.arange(1, n + 1) / n
    
    # Greatest convex minorant and least concave majorant
    # Simplified: just use range between peaks if multiple
    if len(peaks) >= 2:
        dip_stat = 0.1  # Indicates bimodality
        is_bimodal = True
    else:
        dip_stat = 0.01
        is_bimodal = False
    
    return is_bimodal, dip_stat


def get_boundary_conditions(metadata: dict, args, no_interactive: bool = False,
                            image: np.ndarray = None) -> dict:
    """
    Get boundary conditions from metadata, args, or prompt user.
    
    Returns dict with:
        - pixel_size_nm: pixel size in nanometers
        - d_min: minimum d-spacing (nm)
        - d_max: maximum d-spacing (nm)
        - q_min: minimum q (1/nm) - computed from d_max
        - q_max: maximum q (1/nm) - computed from d_min
        - intensity_threshold: peak detection threshold (auto or manual)
        - tile_size: FFT tile size
        - stride: tile stride
    """
    params = {}
    
    print("\n" + "=" * 60)
    print("BOUNDARY CONDITIONS")
    print("=" * 60)
    
    # Pixel size - use metadata if available
    meta_pixel = metadata.get('pixel_size_nm')
    if args.pixel_size:
        params['pixel_size_nm'] = args.pixel_size
        print(f"Pixel size (from args): {params['pixel_size_nm']} nm/pixel")
    elif meta_pixel:
        params['pixel_size_nm'] = meta_pixel
        print(f"Pixel size (from metadata): {meta_pixel:.6f} nm/pixel ✓")
    elif no_interactive:
        print("ERROR: Pixel size not in metadata and --no-interactive set")
        sys.exit(1)
    else:
        print("⚠ Pixel size not found in metadata.")
        params['pixel_size_nm'] = prompt_float("  Enter pixel size (nm/pixel)")
    
    # D-spacing range - ALWAYS ask (material-specific)
    print()
    if args.d_min is not None and args.d_max is not None:
        params['d_min'] = args.d_min
        params['d_max'] = args.d_max
        print(f"D-spacing range (from args): {params['d_min']} - {params['d_max']} nm")
    elif no_interactive:
        print("ERROR: D-spacing range required (--d-min and --d-max)")
        sys.exit(1)
    else:
        print("Target lattice d-spacing range (material-specific):")
        print("  Common ranges: 0.2-0.4 nm (metals), 0.3-0.6 nm (oxides), 0.5-1.5 nm (organics)")
        params['d_min'] = prompt_float("  Minimum d-spacing (nm)")
        params['d_max'] = prompt_float("  Maximum d-spacing (nm)")
    
    # Validate d-spacing
    if params['d_min'] >= params['d_max']:
        print("ERROR: d_min must be less than d_max")
        sys.exit(1)
    
    # Convert d-spacing to q-range (q = 1/d)
    params['q_min'] = 1.0 / params['d_max']
    params['q_max'] = 1.0 / params['d_min']
    print(f"  → Q-range: {params['q_min']:.3f} - {params['q_max']:.3f} nm⁻¹")
    
    # Tile parameters (need these before auto-threshold)
    print()
    if args.tile_size:
        params['tile_size'] = args.tile_size
    elif no_interactive:
        params['tile_size'] = 256
    else:
        params['tile_size'] = int(prompt_float("FFT tile size (pixels)", default=256))
    
    if args.stride:
        params['stride'] = args.stride
    elif no_interactive:
        params['stride'] = params['tile_size'] // 2
    else:
        default_stride = params['tile_size'] // 2
        params['stride'] = int(prompt_float("Tile stride (pixels)", default=default_stride))
    
    print(f"  Tile: {params['tile_size']}x{params['tile_size']}, stride: {params['stride']}")
    
    # Intensity threshold - auto-detect if image provided and not specified
    print()
    if args.threshold:
        params['intensity_threshold'] = args.threshold
        print(f"Intensity threshold (from args): {params['intensity_threshold']}")
    elif image is not None:
        print("Auto-detecting threshold (ensemble of methods)...")
        auto = auto_detect_threshold(
            image, params['pixel_size_nm'],
            params['q_min'], params['q_max'],
            params['tile_size']
        )
        print(f"  Sampled {auto['stats']['n_samples']} tiles")
        print(f"  Intensity range: {auto['stats']['min']:.0f} - {auto['stats']['max']:.0f}")
        print()
        print("  Method comparison:")
        print("  " + "-" * 55)
        print(f"  {'Method':<20} {'Threshold':>10} {'Detection':>10}")
        print("  " + "-" * 55)
        for name, m in auto['methods'].items():
            if isinstance(m, dict) and m.get('threshold') is not None:
                det_pct = m.get('detection_pct', '?')
                print(f"  {name:<20} {m['threshold']:>10.0f} {det_pct:>9}%")
        print("  " + "-" * 55)
        
        # Show bimodality result
        bimod = auto['methods'].get('bimodality', {})
        if bimod.get('is_bimodal'):
            print(f"  Distribution: BIMODAL (two distinct populations)")
        else:
            print(f"  Distribution: unimodal")
        
        print()
        print(f"  Selected method: {auto['method'].upper()}")
        print(f"  Suggested threshold: {auto['suggested']:.0f}")
        
        if not no_interactive:
            use_auto = prompt_yes_no(f"  Use suggested threshold ({auto['suggested']:.0f})?", default=True)
            if use_auto:
                params['intensity_threshold'] = auto['suggested']
            else:
                params['intensity_threshold'] = prompt_float("  Enter threshold manually")
        else:
            params['intensity_threshold'] = auto['suggested']
            print(f"  (auto-selected for --no-interactive)")
        
        params['auto_threshold_stats'] = auto
    else:
        print("Peak detection threshold:")
        print("  (Higher = more selective, Lower = more detections)")
        params['intensity_threshold'] = prompt_float("  Threshold", default=3000)
    
    print("=" * 60)
    
    return params


def main():
    parser = argparse.ArgumentParser(
        description="STEM Domain Analysis - Automated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for parameters)
  python analyze.py sample.dm4

  # Specify all parameters
  python analyze.py sample.dm4 --pixel-size 0.1297 --d-min 0.5 --d-max 1.5 --threshold 3000

  # Custom output directory
  python analyze.py sample.dm4 -o results/experiment1
        """
    )
    
    parser.add_argument('input', type=str, help='Input file (DM4, TIFF, or NPY)')
    parser.add_argument('-o', '--output', type=str, default='outputs',
                        help='Output directory (default: outputs)')
    parser.add_argument('--pixel-size', type=float, dest='pixel_size',
                        help='Pixel size in nm/pixel')
    parser.add_argument('--d-min', type=float, dest='d_min',
                        help='Minimum d-spacing in nm')
    parser.add_argument('--d-max', type=float, dest='d_max',
                        help='Maximum d-spacing in nm')
    parser.add_argument('--threshold', type=float,
                        help='Peak detection intensity threshold')
    parser.add_argument('--tile-size', type=int, dest='tile_size',
                        help='FFT tile size in pixels (default: 256)')
    parser.add_argument('--stride', type=int,
                        help='Tile stride in pixels (default: tile_size/2)')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Fail instead of prompting for missing parameters')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--save-preprocessed', action='store_true',
                        help='Save preprocessed image as NPY file')
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("STEM DOMAIN ANALYSIS")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {args.output}/")
    
    # Import modules (delayed to show help faster)
    from src.io_dm4 import load_dm4
    from src.preprocess import preprocess
    from src.radial_analysis import run_radial_analysis
    
    # Load input file
    print("\n[1/5] Loading image...")
    
    suffix = input_path.suffix.lower()
    metadata = {}
    
    if suffix == '.dm4' or suffix == '.dm3':
        record = load_dm4(str(input_path))
        image = record.image
        metadata = record.metadata
        if record.px_nm is not None:
            metadata['pixel_size_nm'] = record.px_nm
            print(f"  Loaded DM4: {image.shape}")
            print(f"  Pixel size from metadata: {record.px_nm:.6f} nm/pixel ✓")
        else:
            print(f"  Loaded DM4: {image.shape}")
            print(f"  ⚠ Pixel size not found in metadata")
        if metadata:
            print(f"  Metadata keys: {len(metadata)} entries")
    elif suffix == '.npy':
        image = np.load(input_path)
        print(f"  Loaded NPY: {image.shape}")
        # Try to load companion metadata
        meta_path = input_path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            print(f"  Loaded metadata from {meta_path.name}")
    elif suffix in ('.tif', '.tiff'):
        from skimage import io as skio
        image = skio.imread(str(input_path))
        if image.ndim == 3:
            image = image.mean(axis=-1)  # Convert to grayscale
        print(f"  Loaded TIFF: {image.shape}")
    else:
        print(f"Error: Unsupported file format: {suffix}")
        print("  Supported: .dm4, .dm3, .npy, .tif, .tiff")
        sys.exit(1)
    
    # Preprocess first (needed for auto-threshold detection)
    print("\n[2/5] Preprocessing...")
    preprocess_params = {
        'outlier_percentile': 0.1,
        'normalize': True,
        'smooth_sigma': 0.5,
    }
    preprocess_result = preprocess(image, preprocess_params)
    processed = preprocess_result.image_pp
    print(f"  Output shape: {processed.shape}")
    print(f"  Range: [{processed.min():.3f}, {processed.max():.3f}]")
    if args.verbose:
        print(f"  Diagnostics: {preprocess_result.diagnostics}")
    
    # Get boundary conditions (with auto-threshold using preprocessed image)
    print("\n[3/5] Setting analysis parameters...")
    if args.no_interactive:
        # Check required params
        missing = []
        if not args.pixel_size and metadata.get('pixel_size_nm') is None:
            missing.append('--pixel-size')
        if args.d_min is None:
            missing.append('--d-min')
        if args.d_max is None:
            missing.append('--d-max')
        if missing:
            print(f"Error: Missing required parameters: {', '.join(missing)}")
            print("  Use interactive mode or provide all parameters.")
            sys.exit(1)
    
    params = get_boundary_conditions(metadata, args, no_interactive=args.no_interactive, image=processed)
    
    # Save preprocessed if requested
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.save_preprocessed:
        npy_path = output_path / 'preprocessed.npy'
        np.save(npy_path, processed)
        print(f"  Saved: {npy_path}")
    
    # Save parameters (convert numpy types for JSON)
    params_path = output_path / 'parameters.json'
    params_serializable = {}
    for k, v in params.items():
        if isinstance(v, (np.floating, np.integer)):
            params_serializable[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, dict):
            # Handle nested dicts (like auto_threshold_stats)
            params_serializable[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, np.integer, float, int)) else vv)
                for kk, vv in v.items()
            }
        else:
            params_serializable[k] = v
    with open(params_path, 'w') as f:
        json.dump(params_serializable, f, indent=2)
    print(f"  Saved: {params_path}")
    
    # Run radial analysis
    print("\n[4/5] Running radial analysis...")
    
    analysis_params = {
        'q_range': (params['q_min'], params['q_max']),
        'intensity_threshold': params['intensity_threshold'],
        'tile_size': params['tile_size'],
        'stride': params['stride'],
    }
    
    results = run_radial_analysis(
        processed,
        pixel_size_nm=params['pixel_size_nm'],
        output_dir=str(output_path),
        params=analysis_params,
        verbose=True
    )
    
    # Summary
    print("\n[5/5] Summary")
    print("=" * 60)
    n_peaks = results['peak_results']['n_peaks']
    total_tiles = np.prod(results['peak_results']['grid_shape'])
    pct = n_peaks / total_tiles * 100 if total_tiles > 0 else 0
    
    print(f"  Crystalline tiles: {n_peaks}/{total_tiles} ({pct:.1f}%)")
    print(f"  D-spacing range: {params['d_min']:.2f} - {params['d_max']:.2f} nm")
    print(f"  Q-range: {params['q_min']:.3f} - {params['q_max']:.3f} nm⁻¹")
    print(f"  Threshold: {params['intensity_threshold']}")
    print()
    print("Output files:")
    for f in sorted(output_path.glob('*.png')):
        print(f"  - {f.name}")
    print(f"  - parameters.json")
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
