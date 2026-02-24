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
import gc
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
                # Use percentile95 method for robust threshold estimation
                intensities_in_annulus = power[q_mask]
                if len(intensities_in_annulus) > 0:
                    p95_threshold = np.percentile(intensities_in_annulus, 95)
                    high_intensities = intensities_in_annulus[intensities_in_annulus > p95_threshold]
                    peak_intensity = np.mean(high_intensities) if len(high_intensities) > 0 else np.max(intensities_in_annulus)
                else:
                    peak_intensity = 0
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
    
    # Auto-discover peaks if requested and no d-spacing provided
    if args.auto_discover and image is not None and args.d_min is None and args.d_max is None:
        from src.peak_discovery import discover_peaks, get_recommended_params, save_discovery_plot
        
        print("\n" + "=" * 60)
        print("AUTOMATIC PEAK DISCOVERY")
        print("=" * 60)
        print("Searching for crystalline diffraction peaks...")
        
        discovery_result = discover_peaks(
            image, params['pixel_size_nm'],
            tile_size=args.tile_size or 256,
            verbose=True
        )
        
        # Save discovery plot
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_discovery_plot(discovery_result, str(output_path / '1_Peak_Discovery.png'))
        print(f"\n  Saved: 1_Peak_Discovery.png")
        
        # Store discovery results
        params['discovery_result'] = {
            'n_peaks': len(discovery_result.peaks),
            'best_peak_idx': discovery_result.best_peak_idx,
            'noise_floor': discovery_result.noise_floor,
            'message': discovery_result.message,
            'peaks': [
                {
                    'q_center': p.q_center,
                    'd_spacing': p.d_spacing,
                    'snr': p.snr,
                    'confidence': p.confidence,
                    'suggested_threshold': p.suggested_threshold,
                }
                for p in discovery_result.peaks
            ]
        }
        
        recommended = get_recommended_params(discovery_result)
        
        if recommended:
            print(f"\n  RECOMMENDED PARAMETERS:")
            print(f"    D-spacing: {recommended['d_min']:.3f} - {recommended['d_max']:.3f} nm")
            print(f"    Q-range:   {recommended['q_min']:.3f} - {recommended['q_max']:.3f} nm⁻¹")
            print(f"    Threshold: {recommended['intensity_threshold']:.0f}")
            print(f"    Confidence: {recommended['confidence']} (SNR={recommended['snr']:.1f})")
            
            if not no_interactive:
                use_recommended = prompt_yes_no("\n  Use recommended parameters?", default=True)
                if use_recommended:
                    params['d_min'] = recommended['d_min']
                    params['d_max'] = recommended['d_max']
                    params['intensity_threshold'] = recommended['intensity_threshold']
                    params['auto_discovered'] = True
                    print("  ✓ Using auto-discovered parameters")
                else:
                    print("\n  Manual parameter entry:")
            else:
                # Non-interactive: use recommended
                params['d_min'] = recommended['d_min']
                params['d_max'] = recommended['d_max']
                params['intensity_threshold'] = recommended['intensity_threshold']
                params['auto_discovered'] = True
                print("  ✓ Using auto-discovered parameters (--no-interactive)")

            # Handle multi-plane mode
            if args.multi_plane and recommended:
                params['multi_plane_mode'] = True
                params['discovered_peaks'] = discovery_result.peaks[:args.max_planes]

                if args.interactive and not no_interactive:
                    # Interactive plane selection
                    print("\n  Select planes to analyze:")
                    print(f"  {'#':<3} {'q (nm⁻¹)':<10} {'d (nm)':<10} {'SNR':<8} {'Threshold':<12}")
                    print(f"  {'-'*50}")

                    selected_peaks = []
                    for i, p in enumerate(params['discovered_peaks']):
                        prompt_text = f"  {i+1:<3} {p.q_center:<10.3f} {p.d_spacing:<10.3f} {p.snr:<8.1f} {p.suggested_threshold:<12.0f}"
                        include = prompt_yes_no(prompt_text, default=True)
                        if include:
                            selected_peaks.append(p)

                    params['discovered_peaks'] = selected_peaks
                    print(f"\n  Selected {len(selected_peaks)} planes for analysis")
                else:
                    print(f"\n  ✓ Will analyze top {len(params['discovered_peaks'])} planes")

                # Don't set d_min/d_max (will use per-plane ranges)
                # Remove them if they were set
                if 'd_min' in params:
                    del params['d_min']
                if 'd_max' in params:
                    del params['d_max']
                if 'intensity_threshold' in params:
                    del params['intensity_threshold']

                return params

        else:
            print("\n  ⚠ No strong peaks found. Manual parameter entry required.")
            if no_interactive:
                print("ERROR: Auto-discovery found no peaks and --no-interactive set")
                sys.exit(1)
    
    # D-spacing range - use discovered, args, or prompt
    print()
    if 'd_min' in params and 'd_max' in params:
        # Already set from auto-discovery
        print(f"D-spacing range (auto-discovered): {params['d_min']:.3f} - {params['d_max']:.3f} nm")
    elif args.d_min is not None and args.d_max is not None:
        params['d_min'] = args.d_min
        params['d_max'] = args.d_max
        print(f"D-spacing range (from args): {params['d_min']} - {params['d_max']} nm")
    elif no_interactive:
        print("ERROR: D-spacing range required (--d-min and --d-max, or use --auto-discover)")
        sys.exit(1)
    else:
        print("Target lattice d-spacing range (material-specific):")
        print("  Common ranges: 0.2-0.4 nm (metals), 0.3-0.6 nm (oxides), 0.5-1.5 nm (organics)")
        print("  TIP: Use --auto-discover to automatically find peaks!")
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
    if 'intensity_threshold' in params:
        # Already set from auto-discovery
        print(f"Intensity threshold (auto-discovered): {params['intensity_threshold']:.0f}")
    elif args.threshold:
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


def run_interactive_threshold_loop(
    processed: np.ndarray,
    pixel_size_nm: float,
    params: dict,
    output_path: Path,
    verbose: bool = True
) -> dict:
    """
    Interactive threshold tuning loop.

    Runs analysis, shows results, prompts for threshold adjustment.
    Continues until user accepts results.

    Args:
        processed: Preprocessed image
        pixel_size_nm: Pixel size in nm
        params: Initial analysis parameters (must include all except threshold)
        output_path: Output directory
        verbose: Verbose output

    Returns:
        Final analysis results dict
    """
    from src.radial_analysis import run_radial_analysis

    current_threshold = params['intensity_threshold']
    iteration = 0

    while True:
        iteration += 1
        print("\n" + "=" * 60)
        print(f"THRESHOLD ITERATION {iteration}")
        print("=" * 60)
        print(f"Current threshold: {current_threshold:.0f}")
        print()

        # Update params with current threshold
        analysis_params = {
            'q_range': (params['q_min'], params['q_max']),
            'intensity_threshold': current_threshold,
            'tile_size': params['tile_size'],
            'stride': params['stride'],
            'peak_method': params.get('peak_method', 'percentile95'),
        }

        # Run analysis
        print("[Running analysis with current threshold...]")
        results = run_radial_analysis(
            processed,
            pixel_size_nm=pixel_size_nm,
            output_dir=str(output_path),
            params=analysis_params,
            verbose=verbose
        )

        # Display results summary
        print("\n" + "-" * 60)
        print("RESULTS SUMMARY")
        print("-" * 60)
        n_peaks = results['peak_results']['n_peaks']
        total_tiles = np.prod(results['peak_results']['grid_shape'])
        pct = n_peaks / total_tiles * 100 if total_tiles > 0 else 0

        print(f"  Crystalline tiles: {n_peaks}/{total_tiles} ({pct:.1f}%)")
        print(f"  D-spacing range: {params['d_min']:.2f} - {params['d_max']:.2f} nm")
        print(f"  Q-range: {params['q_min']:.3f} - {params['q_max']:.3f} nm⁻¹")
        print(f"  Threshold: {current_threshold:.0f}")
        print()

        # Guidance
        if pct < 5:
            print("  💡 Low detection rate - consider lowering threshold")
        elif pct > 30:
            print("  💡 High detection rate - consider raising threshold")
        else:
            print("  ✓ Detection rate looks reasonable")

        print()
        print(f"  Review output files in: {output_path}/")
        print(f"    - 4_Peak_Location_Map.png (spatial distribution)")
        print(f"    - 5_Orientation_Map.png (crystal orientations)")
        print()

        # Prompt for adjustment
        accept = prompt_yes_no("  Accept these results?", default=True)

        if accept:
            print("\n  ✓ Results accepted!")
            # Update final params
            params['intensity_threshold'] = current_threshold
            params['interactive_iterations'] = iteration
            return results
        else:
            # Get new threshold
            print()
            print(f"  Current threshold: {current_threshold:.0f}")
            print("  Enter new threshold (or press Ctrl+C to abort):")

            while True:
                try:
                    new_threshold = prompt_float("    New threshold", default=current_threshold)
                    if new_threshold <= 0:
                        print("    Threshold must be positive. Try again.")
                        continue
                    current_threshold = new_threshold
                    break
                except KeyboardInterrupt:
                    print("\n\n  Aborted by user.")
                    sys.exit(1)


def run_hybrid_pipeline(image: np.ndarray, args, output_path: Path,
                        metadata: dict) -> int:
    """Run the new hybrid FFT + GPA + Peak-Finding pipeline.

    Pipeline order:
    1. LOAD (already done) + G1 input validation + FFTGrid
    2. BRANCH A: preprocess_fft_safe
    3. BRANCH B: preprocess_segmentation
    4. EARLY ROI (uses Branch B)
    5. GLOBAL FFT (uses Branch A)
    6. TILE FFT + Two-Tier SNR (uses Branch A + ROI)
    7. GPA (optional)
    8. PEAK FINDING (optional)
    9. VALIDATION + REPORTING
    """
    import logging
    import psutil
    from src.fft_coords import FFTGrid, compute_effective_q_min
    from src.pipeline_config import PipelineConfig, GPAConfig, TierConfig, PeakSNRConfig, ReferenceSelectionConfig, RingAnalysisConfig
    from src.preprocess_fft_safe import preprocess_fft_safe
    from src.preprocess_segmentation import preprocess_segmentation
    from src.roi_masking import compute_roi_mask, downsample_to_tile_grid
    from src.global_fft import compute_global_fft
    from src.tile_fft import process_all_tiles, check_tiling_adequacy
    from src.fft_snr_metrics import build_gated_tile_grid
    from src.gates import evaluate_gate
    from src.validation import validate_pipeline
    from src.reporting import save_pipeline_artifacts, save_json
    
    # Memory tracking
    process = psutil.Process()
    def log_memory(stage: str):
        mem = process.memory_info()
        rss_gb = mem.rss / 1024**3
        vms_gb = mem.vms / 1024**3
        print(f"MEMORY [{stage}]: RSS={rss_gb:.2f} GB, VMS={vms_gb:.2f} GB", flush=True)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(name)s %(levelname)s: %(message)s')

    # --- Load config ---
    config = PipelineConfig()
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f)
            config = PipelineConfig.from_dict(cfg_dict)
            print(f"  Loaded config from {args.config}")
        except ImportError:
            print("  WARNING: pyyaml not installed, using defaults")
        except Exception as e:
            print(f"  WARNING: Failed to load config: {e}, using defaults")

    # Apply CLI overrides
    pixel_size = args.pixel_size or metadata.get('pixel_size_nm') or config.pixel_size_nm
    config.pixel_size_nm = pixel_size
    if args.tile_size:
        config.tile_size = args.tile_size
    if args.stride:
        config.stride = args.stride
    else:
        config.stride = config.tile_size // 2
    config.tier.tier_a_snr = args.snr_tier_a
    config.tier.tier_b_snr = args.snr_tier_b
    config.gpa.mode = args.gpa_mode
    config.gpa.on_fail = args.gpa_on_fail
    if args.no_gpa:
        config.gpa.enabled = False
    if args.no_peak_finding:
        config.peak_finding.enabled = False
    if args.no_viz:
        config.viz.enabled = False
    config.viz.dpi = args.viz_dpi
    if args.no_low_q_exclusion:
        config.low_q.enabled = False
    if args.q_min_override is not None:
        config.low_q.q_min_cycles_per_nm = args.q_min_override
        config.low_q.auto_q_min = False
        config.low_q.enabled = True
    # Physics config CLI overrides
    if args.physics_d_min is not None:
        config.physics.d_min_nm = args.physics_d_min
    if args.physics_d_max is not None:
        config.physics.d_max_nm = args.physics_d_max
    if args.imaging_mode is not None:
        config.physics.imaging_mode = args.imaging_mode
    # Clustering config CLI overrides
    if args.cluster:
        config.clustering.enabled = True
    config.clustering.method = args.cluster_method
    config.clustering.n_clusters = args.cluster_n
    config.clustering.dimred_method = args.cluster_dimred
    if args.strong_guidance_snr is not None:
        config.global_fft.strong_guidance_snr = args.strong_guidance_snr
    # DC masking and SNR CLI overrides
    if args.dynamic_dc:
        config.dc_mask.enabled = True
    if args.dc_floor is not None:
        config.dc_mask.q_dc_min_floor = args.dc_floor
    if args.dc_soft_taper:
        config.dc_mask.soft_taper = True
    if args.snr_signal_method is not None:
        config.peak_snr.signal_method = args.snr_signal_method
    if args.bg_method is not None:
        config.global_fft.background_method = args.bg_method
    if args.window_type is not None:
        config.tile_fft.window_type = args.window_type
    if args.tukey_alpha is not None:
        config.tile_fft.tukey_alpha = args.tukey_alpha
    if args.phase_unwrap_method is not None:
        config.gpa.phase_unwrap.method = args.phase_unwrap_method

    # Extract config sub-objects for threading
    gt = config.gate_thresholds

    # --- GPU / device context ---
    from src.gpu_backend import DeviceContext, get_gpu_info
    device_choice = getattr(args, 'device', config.device.device)
    ctx = DeviceContext.create(device_choice, device_id=config.device.device_id)
    if ctx.using_gpu:
        info = get_gpu_info(config.device.device_id)
        print(f"  GPU: {info.device_name} ({info.free_memory_gb:.1f}/{info.total_memory_gb:.1f} GB free)")
    else:
        print(f"  Device: CPU")

    H, W = image.shape
    print(f"\n  Image: {H}x{W}, pixel_size={pixel_size:.6f} nm")
    print(f"  Tile: {config.tile_size}, stride: {config.stride}")
    print(f"  Tier A SNR: {config.tier.tier_a_snr}, Tier B: {config.tier.tier_b_snr}")

    # --- Step 1: Input validation (G1) ---
    log_memory("Start")
    print("\n[1/9] Input validation...")
    g1_checks = {
        "is_2d": image.ndim == 2,
        "no_nan": not np.any(np.isnan(image)),
        "no_inf": not np.any(np.isinf(image)),
        "min_dim": min(H, W) >= 512,
        "has_pixel_size": pixel_size > 0,
    }
    g1 = evaluate_gate("G1", g1_checks, gate_thresholds=gt)
    gate_results = {"G1": g1}
    if not g1.passed:
        print(f"  FATAL: Input validation failed: {g1.reason}")
        return 1
    print("  G1 PASS")

    # Instantiate FFTGrid (canonical coordinate system)
    fft_grid = FFTGrid(H, W, pixel_size)

    # Compute effective q_min for global and tile grids
    low_q = config.low_q
    effective_q_min = compute_effective_q_min(
        fft_grid, enabled=low_q.enabled,
        q_min_cycles_per_nm=low_q.q_min_cycles_per_nm,
        dc_bin_count=low_q.dc_bin_count, auto_q_min=low_q.auto_q_min)
    tile_fft_grid_tmp = FFTGrid(config.tile_size, config.tile_size, pixel_size)
    tile_effective_q_min = compute_effective_q_min(
        tile_fft_grid_tmp, enabled=low_q.enabled,
        q_min_cycles_per_nm=low_q.q_min_cycles_per_nm,
        dc_bin_count=low_q.dc_bin_count, auto_q_min=low_q.auto_q_min)
    if low_q.enabled:
        print(f"  Low-q exclusion: global={effective_q_min:.4f}, tile={tile_effective_q_min:.4f} cycles/nm")
    else:
        print("  Low-q exclusion: disabled")

    # --- Gate G0: Nyquist guard ---
    physics = config.physics
    effective_d_min = physics.d_min_nm
    effective_d_max = physics.d_max_nm
    if physics.d_min_nm > 0 or physics.d_max_nm > 0:
        q_nyquist = fft_grid.nyquist_q()
        q_max_req = 1.0 / physics.d_min_nm if physics.d_min_nm > 0 else 0
        q_min_req = 1.0 / physics.d_max_nm if physics.d_max_nm > 0 else 0
        g0 = evaluate_gate("G0", {
            "q_max_requested": q_max_req,
            "q_min_requested": q_min_req,
            "q_nyquist": q_nyquist,
            "safety_margin": physics.nyquist_safety_margin,
        }, gate_thresholds=gt)
        gate_results["G0"] = g0
        if not g0.passed and "FATAL" in g0.reason:
            print(f"  FATAL: {g0.reason}")
            return 1
        elif not g0.passed:
            # DEGRADE: clamp d_min to Nyquist-safe value
            q_safe = physics.nyquist_safety_margin * q_nyquist
            effective_d_min = 1.0 / q_safe
            print(f"  G0 DEGRADE: d_min clamped {physics.d_min_nm:.4f} → {effective_d_min:.4f} nm")
        else:
            print("  G0 PASS")
    else:
        print("  G0 skipped (unconstrained d-range)")

    # --- Step 2: Branch A preprocessing (FFT-safe) ---
    log_memory("Before Branch A")
    print("\n[2/9] Branch A: FFT-safe preprocessing...")
    preproc_record = preprocess_fft_safe(image, config.preprocessing)
    gate_results["G2"] = evaluate_gate("G2", {
        "clipped_fraction": preproc_record.diagnostics.get("clipped_fraction", 0),
        "intensity_range_ratio": preproc_record.diagnostics.get("intensity_range_ratio", 100),
    }, gate_thresholds=gt)
    print(f"  Confidence: {preproc_record.confidence}")
    print(f"  G2 {'PASS' if gate_results['G2'].passed else 'FAIL (DEGRADE)'}")

    # --- Step 3: Branch B preprocessing (segmentation) ---
    log_memory("Before Branch B")
    print("\n[3/9] Branch B: Segmentation preprocessing...")
    seg_record = preprocess_segmentation(image, config.segmentation)
    print(f"  Output range: [{seg_record.image_seg.min():.3f}, {seg_record.image_seg.max():.3f}]")

    # --- Step 4: Early ROI ---
    log_memory("Before ROI")
    print("\n[4/9] Early ROI masking...")
    roi_result = compute_roi_mask(seg_record.image_seg, config.roi)
    roi_grid = downsample_to_tile_grid(
        roi_result.mask_full, config.tile_size, config.stride,
        min_coverage=config.roi.min_tile_coverage
    )
    gate_results["G3"] = evaluate_gate("G3", {
        "coverage_pct": roi_result.coverage_pct,
        "n_components": roi_result.n_components,
    }, gate_thresholds=gt)
    print(f"  Coverage: {roi_result.coverage_pct:.1f}%, components: {roi_result.n_components}")
    print(f"  G3 {'PASS' if gate_results['G3'].passed else 'FAIL (FALLBACK)'}")
    if not gate_results["G3"].passed:
        roi_grid = np.ones(roi_grid.shape, dtype=bool)
        print("  Using full-image ROI (fallback)")

    # --- Step 5: Global FFT ---
    log_memory("Before Global FFT")
    print("\n[5/9] Global FFT analysis...")
    global_fft_result = compute_global_fft(preproc_record.image_fft, fft_grid, config.global_fft, ctx=ctx,
                                           effective_q_min=effective_q_min,
                                           dc_mask_config=config.dc_mask,
                                           physics_config=config.physics)
    d_dom = global_fft_result.d_dom

    best_global_snr = max((p.snr for p in global_fft_result.peaks), default=0)
    gate_results["G4"] = evaluate_gate("G4", best_global_snr, gate_thresholds=gt)
    print(f"  d_dom: {d_dom:.3f} nm" if d_dom else "  d_dom: not found")
    print(f"  Peaks found: {len(global_fft_result.peaks)}, g-vectors: {len(global_fft_result.g_vectors)}")
    print(f"  G4 {'PASS' if gate_results['G4'].passed else 'FAIL (FALLBACK)'}")

    # Determine q_ranges for tile FFT from global peaks (3-tuple: q_lo, q_hi, ring_idx)
    q_ranges = []
    if global_fft_result.peaks:
        for ring_idx, p in enumerate(global_fft_result.peaks):
            q_width = max(p.q_fwhm * 2, p.q_center * 0.03) if p.q_fwhm > 0 else p.q_center * 0.1
            q_ranges.append((p.q_center - q_width, p.q_center + q_width, ring_idx))
    elif args.d_min is not None and args.d_max is not None:
        q_ranges.append((1.0 / args.d_max, 1.0 / args.d_min))

    # --- Step 6: Tile FFT + Two-tier classification ---
    log_memory("Before Tile FFT")
    print("\n[6/9] Tile FFT + Two-tier classification...")

    # G5: tiling adequacy — use selected d_dom (already filtered by physics bounds)
    d_for_g5 = d_dom if d_dom else (args.d_max if args.d_max else 1.0)
    d_px = d_for_g5 / pixel_size
    periods = config.tile_size / d_px if d_px > 0 else 0
    print(f"  G5: periods/tile={periods:.1f} using d_used={d_for_g5:.3f} nm")
    gate_results["G5"] = evaluate_gate("G5", periods, gate_thresholds=gt)
    if not gate_results["G5"].passed:
        print(f"  FATAL: Only {periods:.1f} periods per tile (need >=20)")
        # Save what we have and exit
        report = validate_pipeline(
            preproc_record=preproc_record, roi_result=roi_result,
            global_fft_result=global_fft_result,
            gate_results=gate_results, tile_size=config.tile_size,
            pixel_size_nm=pixel_size, d_dom_nm=d_dom,
            effective_q_min=effective_q_min,
            tile_effective_q_min=tile_effective_q_min,
            gate_thresholds=gt,
        )
        save_pipeline_artifacts(
            output_path, config=config, fft_grid=fft_grid,
            preproc_record=preproc_record, seg_record=seg_record,
            roi_result=roi_result, global_fft_result=global_fft_result,
            validation_report=report,
            effective_q_min=effective_q_min,
            tile_effective_q_min=tile_effective_q_min,
        )
        if config.viz.enabled:
            from src.hybrid_viz import save_pipeline_visualizations
            save_pipeline_visualizations(
                output_path, config=config, fft_grid=fft_grid,
                raw_image=image, global_fft_result=global_fft_result,
                effective_q_min=effective_q_min,
            )
        return 1
    print(f"  Periods/tile: {periods:.1f} (G5 PASS)")

    # Dynamic DC passes from global → tiles
    _dynamic_dc_q = global_fft_result.dynamic_dc_q or 0.0
    if _dynamic_dc_q > 0:
        print(f"  Dynamic DC mask: {_dynamic_dc_q:.3f} cycles/nm")

    peak_sets, skipped_mask = process_all_tiles(
        preproc_record.image_fft, roi_grid, fft_grid,
        tile_size=config.tile_size, stride=config.stride,
        q_ranges=q_ranges if q_ranges else None,
        ctx=ctx,
        effective_q_min=tile_effective_q_min,
        tile_fft_config=config.tile_fft,
        dynamic_dc_q=_dynamic_dc_q,
        dc_mask_config=config.dc_mask,
    )

    tile_fft_grid = FFTGrid(config.tile_size, config.tile_size, pixel_size)
    gated_grid = build_gated_tile_grid(
        peak_sets, skipped_mask, tile_fft_grid, config.tile_size,
        tier_config=config.tier, peak_gate_config=config.peak_gates,
        effective_q_min=tile_effective_q_min,
        confidence_config=config.confidence,
        peak_snr_config=config.peak_snr,
    )

    # Free tile power spectra — no longer needed after classification
    for ps in peak_sets:
        ps.power_spectrum = None
    gc.collect()

    ts = gated_grid.tier_summary
    print(f"  Tier A: {ts.n_tier_a}, Tier B: {ts.n_tier_b}, "
          f"Rejected: {ts.n_rejected}, Skipped: {ts.n_skipped}")
    print(f"  Tier A fraction: {ts.tier_a_fraction:.3f}")

    gate_results["G6"] = evaluate_gate("G6", ts.tier_a_fraction, gate_thresholds=gt)
    gate_results["G7"] = evaluate_gate("G7", ts.median_snr_tier_a, gate_thresholds=gt)
    tier_a_sym = gated_grid.symmetry_map[gated_grid.tier_map == "A"]
    mean_sym = float(np.mean(tier_a_sym)) if len(tier_a_sym) > 0 else 0.0
    gate_results["G8"] = evaluate_gate("G8", mean_sym, gate_thresholds=gt)
    for gid in ("G6", "G7", "G8"):
        print(f"  {gid} {'PASS' if gate_results[gid].passed else 'FAIL'}")

    # --- Step 7: Ring Analysis ---
    ring_maps = None
    ring_features = None
    tile_avg_fft = None
    clustering_result = None
    if global_fft_result.peaks:
        print("\n[7/11] Ring analysis...")
        from src.ring_analysis import (
            build_ring_maps, build_ring_feature_vectors,
            compute_tile_averaged_fft, compute_cluster_summaries,
        )

        ring_maps = build_ring_maps(gated_grid, global_fft_result.peaks,
                                     ring_config=config.ring_analysis)
        ring_features = build_ring_feature_vectors(gated_grid, ring_maps)
        print(f"  Rings: {ring_maps.n_rings}, features: {ring_features.feature_matrix.shape[1]}")
        print(f"  Valid tiles: {int(np.sum(ring_features.valid_mask))}")

        tile_avg_fft = compute_tile_averaged_fft(
            preproc_record.image_fft, config.tile_size, config.stride,
            pixel_size, gated_grid.skipped_mask,
            effective_q_min=tile_effective_q_min)
        print(f"  Tile-averaged FFT: {tile_avg_fft['n_tiles']} tiles")

        # --- Step 8: Domain Clustering ---
        if config.clustering.enabled and ring_features is not None:
            print("\n[8/11] Domain clustering...")
            from src.domain_clustering import run_domain_clustering

            clustering_result = run_domain_clustering(
                ring_features, config.clustering,
                image_fft=preproc_record.image_fft,
                tile_size=config.tile_size,
                stride=config.stride,
                pixel_size_nm=pixel_size,
                skipped_mask=gated_grid.skipped_mask,
                effective_q_min=tile_effective_q_min,
            )
            print(f"  Method: {clustering_result.method_used}")
            print(f"  Clusters: {clustering_result.n_clusters}")
            if clustering_result.silhouette_score is not None:
                print(f"  Silhouette: {clustering_result.silhouette_score:.3f}")

            # Compute per-cluster physics summaries
            if clustering_result.n_clusters > 0:
                clustering_result.cluster_summaries = compute_cluster_summaries(
                    clustering_result.tile_labels_regularized, ring_maps, gated_grid)
                print(f"  Cluster summaries: {len(clustering_result.cluster_summaries)} clusters")
        else:
            if not config.clustering.enabled:
                print("\n[8/11] Domain clustering skipped (not enabled)")
            else:
                print("\n[8/11] Domain clustering skipped (no ring features)")
    else:
        print("\n[7/11] Ring analysis skipped (no global peaks)")
        print("[8/11] Domain clustering skipped")

    # --- Step 9: GPA ---
    gpa_result = None
    if config.gpa.enabled and global_fft_result.g_vectors:
        print(f"\n[9/11] GPA (mode={config.gpa.mode})...")
        from src.gpa import run_gpa

        gpa_result = run_gpa(
            preproc_record.image_fft,
            global_fft_result.g_vectors,
            gated_grid, global_fft_result, fft_grid,
            config=config.gpa,
            tile_size=config.tile_size,
            stride=config.stride,
            ctx=ctx,
            effective_q_min=effective_q_min,
            gate_thresholds=gt,
            ref_config=config.reference_selection,
        )
        if gpa_result is not None:
            print(f"  Mode used: {gpa_result.mode}")
            print(f"  Phases computed: {len(gpa_result.phases)}")
            print(f"  Displacement: {'yes' if gpa_result.displacement else 'no'}")
            print(f"  Strain: {'yes' if gpa_result.strain else 'no'}")
            # Collect GPA gate results if present
            if gpa_result.qc.get("g10_passed") is not None:
                g10_val = {
                    "phase_noise": {k: v.phase_noise_sigma
                                    for k, v in gpa_result.phases.items()
                                    if v.phase_noise_sigma is not None},
                    "unwrap_success": {k: v.unwrap_success_fraction
                                       for k, v in gpa_result.phases.items()},
                }
                gate_results["G10"] = evaluate_gate("G10", g10_val, gate_thresholds=gt)
                print(f"  G10 {'PASS' if gate_results['G10'].passed else 'FAIL'}")
            if gpa_result.qc.get("g11_passed") is not None:
                g11_val = {
                    "ref_strain_max": gpa_result.qc.get("ref_strain_mean_exx", 0),
                    "outlier_fraction": gpa_result.qc.get("strain_outlier_fraction", 0),
                }
                gate_results["G11"] = evaluate_gate("G11", g11_val, gate_thresholds=gt)
                print(f"  G11 {'PASS' if gate_results['G11'].passed else 'FAIL'}")
        else:
            print("  GPA skipped")
    else:
        print("\n[9/11] GPA skipped")
        if not config.gpa.enabled:
            print("  (disabled by config)")
        elif not global_fft_result.g_vectors:
            print("  (no g-vectors available)")

    # --- Step 10: Peak finding ---
    lattice_validation = None
    peaks = None
    peak_image = None
    if config.peak_finding.enabled and d_dom is not None:
        print("\n[10/11] Peak finding...")
        from src.peak_finding import (
            build_bandpass_image, find_subpixel_peaks, validate_peak_lattice,
        )

        g_dom_mag = 1.0 / d_dom if d_dom > 0 else None
        peak_image = build_bandpass_image(
            preproc_record.image_fft, g_dom_mag, fft_grid,
            bandwidth_fraction=config.peak_finding.bandpass_bandwidth,
            effective_q_min=effective_q_min,
        )

        peaks = find_subpixel_peaks(
            peak_image, d_dom, pixel_size,
            min_prominence=config.peak_finding.min_prominence,
            tile_size=config.tile_size,
        )
        print(f"  Peaks found: {len(peaks)}")

        if len(peaks) >= 2:
            lattice_validation = validate_peak_lattice(
                peaks, d_dom, pixel_size,
                tolerance=config.peak_finding.lattice_tolerance,
            )
            gate_results["G12"] = evaluate_gate("G12", lattice_validation.fraction_valid, gate_thresholds=gt)
            print(f"  Lattice valid: {lattice_validation.fraction_valid:.1%}")
            print(f"  G12 {'PASS' if gate_results['G12'].passed else 'FAIL'}")
    else:
        print("\n[10/11] Peak finding skipped")

    # --- Step 11: Validation + Reporting ---
    print("\n[11/11] Validation and reporting...")
    report = validate_pipeline(
        preproc_record=preproc_record,
        roi_result=roi_result,
        global_fft_result=global_fft_result,
        gated_grid=gated_grid,
        gpa_result=gpa_result,
        lattice_validation=lattice_validation,
        gate_results=gate_results,
        tile_size=config.tile_size,
        pixel_size_nm=pixel_size,
        d_dom_nm=d_dom,
        effective_q_min=effective_q_min,
        tile_effective_q_min=tile_effective_q_min,
        gate_thresholds=gt,
    )

    saved = save_pipeline_artifacts(
        output_path,
        config=config,
        fft_grid=fft_grid,
        preproc_record=preproc_record,
        seg_record=seg_record,
        roi_result=roi_result,
        global_fft_result=global_fft_result,
        gated_grid=gated_grid,
        gpa_result=gpa_result,
        lattice_validation=lattice_validation,
        peaks=peaks,
        validation_report=report,
        effective_q_min=effective_q_min,
        tile_effective_q_min=tile_effective_q_min,
        ring_maps=ring_maps,
        ring_features=ring_features,
        tile_avg_fft=tile_avg_fft,
        clustering_result=clustering_result,
    )

    # --- ilastik comparison (exploratory, optional) ---
    if args.ilastik_map and gated_grid is not None:
        try:
            from src.ilastik_compare import run_ilastik_comparison
            ilastik_result = run_ilastik_comparison(
                gated_grid, args.ilastik_map, output_path, config.viz.dpi)
            print(f"  ilastik comparison saved ({len(ilastik_result)} artifacts)")
        except Exception as e:
            logger.warning("ilastik comparison failed: %s", e, exc_info=True)
            print(f"  Warning: ilastik comparison failed: {e}")

    # --- PNG visualizations ---
    if config.viz.enabled:
        from src.hybrid_viz import save_pipeline_visualizations
        viz_saved = save_pipeline_visualizations(
            output_path, config=config, fft_grid=fft_grid,
            raw_image=image, global_fft_result=global_fft_result,
            gated_grid=gated_grid, gpa_result=gpa_result,
            peaks=peaks, lattice_validation=lattice_validation,
            bandpass_image=peak_image, validation_report=report,
            effective_q_min=effective_q_min,
            roi_result=roi_result, seg_record=seg_record,
            ring_maps=ring_maps, clustering_result=clustering_result,
            tile_avg_fft=tile_avg_fft,
        )
        saved.update(viz_saved)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("HYBRID PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  {report.summary}")
    print()
    if gated_grid:
        print(f"  Tier A: {ts.n_tier_a} tiles ({ts.tier_a_fraction:.1%})")
        print(f"  Tier B: {ts.n_tier_b} tiles")
        print(f"  Rejected: {ts.n_rejected} tiles")
    if gpa_result:
        print(f"  GPA mode: {gpa_result.mode}")
    if lattice_validation:
        print(f"  Peak lattice valid: {lattice_validation.fraction_valid:.1%}")
    print()
    print("Output files:")
    for name, path in sorted(saved.items()):
        print(f"  - {name}")
    print()
    print("=" * 60)
    if report.overall_pass:
        print("PIPELINE COMPLETE")
    else:
        print("PIPELINE COMPLETE (with failures)")
    print("=" * 60)

    return 0 if report.overall_pass else 1


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
                        help='Output directory (default: outputs/<input_filename>)')
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
    parser.add_argument('--auto-discover', action='store_true',
                        help='Automatically discover diffraction peaks (recommended if d-spacing unknown)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate results by checking spatial coherence')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--save-preprocessed', action='store_true',
                        help='Save preprocessed image as NPY file')
    parser.add_argument('--peak-method', type=str,
                        choices=['max', 'percentile95'],
                        default='percentile95',
                        help='Peak intensity detection method: max (single pixel) or percentile95 (mean of top 5%%)')
    parser.add_argument('--interactive-threshold', action='store_true',
                        help='Enable interactive threshold tuning (prompt to adjust after seeing results)')
    parser.add_argument('--multi-plane', action='store_true',
                        help='Analyze all discovered peaks (requires --auto-discover)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode: select peaks and/or tune threshold')
    parser.add_argument('--max-planes', type=int, default=5,
                        help='Maximum number of planes to analyze (default: 5)')

    # --- New hybrid pipeline flags (WS8) ---
    parser.add_argument('--hybrid', action='store_true',
                        help='Run the new hybrid FFT+GPA+Peak-Finding pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file for hybrid pipeline')
    parser.add_argument('--gpa-mode', type=str, choices=['auto', 'full', 'region'],
                        default='auto', dest='gpa_mode',
                        help='GPA execution mode (default: auto)')
    parser.add_argument('--gpa-on-fail', type=str,
                        choices=['fallback_to_region', 'skip', 'error'],
                        default='fallback_to_region', dest='gpa_on_fail',
                        help='GPA failure behavior (default: fallback_to_region)')
    parser.add_argument('--snr-tier-a', type=float, default=5.0, dest='snr_tier_a',
                        help='Tier A SNR threshold (default: 5.0)')
    parser.add_argument('--snr-tier-b', type=float, default=3.0, dest='snr_tier_b',
                        help='Tier B SNR threshold (default: 3.0)')
    parser.add_argument('--no-gpa', action='store_true', dest='no_gpa',
                        help='Skip GPA stage entirely')
    parser.add_argument('--no-peak-finding', action='store_true', dest='no_peak_finding',
                        help='Skip peak-finding stage')
    parser.add_argument('--report-format', type=str, choices=['json', 'html', 'both'],
                        default='json', dest='report_format',
                        help='Report output format (default: json)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu'],
                        help='Compute device (default: auto-detect GPU)')
    parser.add_argument('--viz-dpi', type=int, default=150, dest='viz_dpi',
                        help='DPI for PNG visualizations (default: 150)')
    parser.add_argument('--no-viz', action='store_true', dest='no_viz',
                        help='Skip PNG visualization generation')
    parser.add_argument('--q-min', type=float, default=None, dest='q_min_override',
                        help='Override low-q exclusion threshold (cycles/nm), disables auto')
    parser.add_argument('--no-low-q-exclusion', action='store_true', dest='no_low_q_exclusion',
                        help='Disable low-q / DC exclusion entirely')
    parser.add_argument('--physics-d-min', type=float, default=None, dest='physics_d_min',
                        help='Expected minimum d-spacing in nm (for Nyquist guard)')
    parser.add_argument('--physics-d-max', type=float, default=None, dest='physics_d_max',
                        help='Expected maximum d-spacing in nm')
    parser.add_argument('--imaging-mode', type=str, default=None, dest='imaging_mode',
                        help='Imaging mode label (e.g., EFTEM-BF, STEM-HAADF)')
    parser.add_argument('--ilastik-map', type=str, default=None, dest='ilastik_map',
                        help='Path to ilastik probability map (.npy or .h5) for comparison')

    # --- Domain clustering flags ---
    parser.add_argument('--cluster', action='store_true',
                        help='Enable domain clustering')
    parser.add_argument('--cluster-method', type=str, default='kmeans',
                        choices=['kmeans', 'gmm', 'hdbscan'], dest='cluster_method',
                        help='Clustering method (default: kmeans)')
    parser.add_argument('--cluster-n', type=int, default=0, dest='cluster_n',
                        help='Number of clusters, 0=auto (default: 0)')
    parser.add_argument('--cluster-dimred', type=str, default='pca',
                        choices=['pca', 'umap', 'none'], dest='cluster_dimred',
                        help='Dimensionality reduction for clustering (default: pca)')
    parser.add_argument('--strong-guidance-snr', type=float, default=None,
                        dest='strong_guidance_snr',
                        help='SNR threshold for strong FFT guidance (default: 8.0)')

    # --- DC masking and SNR flags ---
    parser.add_argument('--dynamic-dc', action='store_true', dest='dynamic_dc',
                        help='Enable dynamic DC mask estimation from radial profile')
    parser.add_argument('--dc-floor', type=float, default=None, dest='dc_floor',
                        help='Override minimum DC mask radius (cycles/nm)')
    parser.add_argument('--dc-soft-taper', action='store_true', dest='dc_soft_taper',
                        help='Use cosine taper instead of hard DC mask')
    parser.add_argument('--snr-signal-method', type=str, default=None,
                        choices=['max', 'integrated_sum', 'integrated_median'],
                        dest='snr_signal_method',
                        help='SNR signal extraction method (default: max)')
    parser.add_argument('--bg-method', type=str, default=None,
                        choices=['polynomial_robust', 'asls'], dest='bg_method',
                        help='Background fitting method (default: polynomial_robust)')
    parser.add_argument('--window-type', type=str, default=None,
                        choices=['hann', 'tukey'], dest='window_type',
                        help='Tile FFT window function (default: hann)')
    parser.add_argument('--tukey-alpha', type=float, default=None, dest='tukey_alpha',
                        help='Tukey window alpha parameter (default: 0.2)')
    parser.add_argument('--phase-unwrap-method', type=str, default=None,
                        choices=['default', 'quality_guided'],
                        dest='phase_unwrap_method',
                        help='Phase unwrapping method (default: default)')

    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Generate output directory name from input file if using default
    if args.output == 'outputs':
        # Extract base name without extension
        input_stem = input_path.stem
        # Create output folder: outputs/<input_basename>/
        output_dir = Path('outputs') / input_stem
    else:
        output_dir = Path(args.output)

    # Update args.output
    args.output = str(output_dir)

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

    # Save original image to output directory (before preprocessing)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save original as NPY
    original_npy_path = output_path / 'original.npy'
    np.save(original_npy_path, image)
    print(f"  Saved: {original_npy_path.name}")

    # Save original as PNG with scale bar (if pixel size known)
    if metadata.get('pixel_size_nm'):
        from src.viz import save_image_with_scalebar
        save_image_with_scalebar(
            image,
            str(output_path / '0_Original.png'),
            pixel_size_nm=metadata['pixel_size_nm'],
            title='Original STEM-HAADF Image',
            cmap='gray'
        )
        print(f"  Saved: 0_Original.png")

    # --- Dispatch to hybrid pipeline if requested ---
    if args.hybrid:
        print("\n  Running HYBRID pipeline (FFT + GPA + Peak-Finding)")
        return run_hybrid_pipeline(image, args, output_path, metadata)

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
        # Check required params (--auto-discover can substitute for --d-min/--d-max)
        missing = []
        if not args.pixel_size and metadata.get('pixel_size_nm') is None:
            missing.append('--pixel-size')
        if args.d_min is None and args.d_max is None and not args.auto_discover:
            missing.append('--d-min and --d-max (or use --auto-discover)')
        elif args.d_min is None and args.d_max is not None:
            missing.append('--d-min')
        elif args.d_max is None and args.d_min is not None:
            missing.append('--d-max')
        if missing:
            print(f"Error: Missing required parameters: {', '.join(missing)}")
            print("  Use interactive mode or provide all parameters.")
            sys.exit(1)
    
    params = get_boundary_conditions(metadata, args, no_interactive=args.no_interactive, image=processed)

    # Add peak detection method to params
    params['peak_method'] = args.peak_method

    # Save preprocessed if requested
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.save_preprocessed:
        npy_path = output_path / 'preprocessed.npy'
        np.save(npy_path, processed)
        print(f"  Saved: {npy_path}")
    
    # Add interactive mode flag if enabled
    if args.interactive_threshold and not args.no_interactive:
        params['interactive_threshold_mode'] = True

    # Save parameters (convert numpy types for JSON)
    from dataclasses import is_dataclass, asdict

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
        elif isinstance(v, list):
            # Handle lists (e.g., discovered_peaks with DiscoveredPeak objects)
            params_serializable[k] = [
                asdict(item) if is_dataclass(item) else item
                for item in v
            ]
        elif is_dataclass(v):
            # Handle individual dataclass objects
            params_serializable[k] = asdict(v)
        else:
            params_serializable[k] = v
    with open(params_path, 'w') as f:
        json.dump(params_serializable, f, indent=2)
    print(f"  Saved: {params_path}")
    
    # Run radial analysis
    print("\n[4/5] Running radial analysis...")

    # Check if multi-plane mode
    if params.get('multi_plane_mode', False):
        # MULTI-PLANE MODE
        from src.multi_plane_analysis import analyze_multiple_planes

        print(f"  Analyzing {len(params['discovered_peaks'])} planes...")

        multi_results = analyze_multiple_planes(
            processed,
            pixel_size_nm=params['pixel_size_nm'],
            peaks=params['discovered_peaks'],
            tile_size=params['tile_size'],
            stride=params['stride'],
            peak_method=args.peak_method,
            verbose=True
        )

        # Save multi-plane visualizations
        from src.viz import save_multi_plane_composite, save_per_plane_maps

        print("\n[4.1] Saving multi-plane composite...")
        save_multi_plane_composite(
            processed,
            multi_results,
            params['stride'],
            params['tile_size'],
            str(output_path / '6_Multi_Plane_Composite.png'),
            pixel_size_nm=params['pixel_size_nm']
        )

        print("[4.2] Saving per-plane orientation maps...")
        save_per_plane_maps(
            processed,
            multi_results,
            params['stride'],
            params['tile_size'],
            output_path,
            pixel_size_nm=params['pixel_size_nm']
        )

        # For backwards compatibility, store first plane as 'results'
        results = {
            'profile': multi_results.radial_profile,
            'peak_results': {
                'peak_mask': multi_results.planes[0].peak_mask,
                'orientation_map': multi_results.planes[0].orientation_map,
                'intensity_map': multi_results.planes[0].intensity_map,
                'grid_shape': multi_results.planes[0].peak_mask.shape,
                'n_peaks': multi_results.planes[0].n_detections,
            },
            'multi_plane_results': multi_results,  # Store full multi-plane data
        }

        # Add multi-plane metadata to params for JSON
        params['planes'] = [
            {
                'plane_id': plane.plane_id,
                'd_spacing': plane.d_spacing,
                'q_center': plane.q_center,
                'q_range': plane.q_range,
                'n_detections': plane.n_detections,
                'detection_rate': plane.detection_rate,
                'coherence_score': plane.coherence_score,
                'is_valid': plane.is_valid,
            }
            for plane in multi_results.planes
        ]

    elif args.interactive_threshold and not args.no_interactive:
        # INTERACTIVE THRESHOLD TUNING MODE
        results = run_interactive_threshold_loop(
            processed,
            pixel_size_nm=params['pixel_size_nm'],
            params=params,
            output_path=output_path,
            verbose=True
        )
    else:
        # STANDARD SINGLE-PLANE MODE
        analysis_params = {
            'q_range': (params['q_min'], params['q_max']),
            'intensity_threshold': params['intensity_threshold'],
            'tile_size': params['tile_size'],
            'stride': params['stride'],
            'peak_method': args.peak_method,
        }

        results = run_radial_analysis(
            processed,
            pixel_size_nm=params['pixel_size_nm'],
            output_dir=str(output_path),
            params=analysis_params,
            verbose=True
        )

    # Re-save parameters if interactive mode was used (threshold may have changed)
    if args.interactive_threshold and not args.no_interactive:
        params_serializable = {}
        for k, v in params.items():
            if isinstance(v, (np.floating, np.integer)):
                params_serializable[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, dict):
                params_serializable[k] = {
                    kk: (float(vv) if isinstance(vv, (np.floating, np.integer, float, int)) else vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                # Handle lists (e.g., discovered_peaks with DiscoveredPeak objects)
                params_serializable[k] = [
                    asdict(item) if is_dataclass(item) else item
                    for item in v
                ]
            elif is_dataclass(v):
                # Handle individual dataclass objects
                params_serializable[k] = asdict(v)
            else:
                params_serializable[k] = v
        with open(params_path, 'w') as f:
            json.dump(params_serializable, f, indent=2)

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
    
    # Validate results if requested
    if args.validate:
        from src.peak_discovery import validate_spatial_coherence
        
        print("\n" + "-" * 60)
        print("VALIDATION")
        print("-" * 60)
        
        validation = validate_spatial_coherence(
            results['peak_results']['peak_mask'],
            results['peak_results']['orientation_map']
        )
        
        print(f"  Detection rate: {validation['detection_rate']*100:.1f}%")
        print(f"  Coherence score: {validation['coherence_score']:.2f}")
        print(f"  Local coherence: {validation['local_coherence']:.2f}")
        print(f"  Orientation entropy: {validation['orientation_entropy']:.2f}")
        print(f"  Significant domains: {validation['n_domains']}")
        print(f"  Largest domain: {validation['largest_domain']} tiles")
        print(f"  Interpretation: {validation['interpretation']}")
        
        if not validation['is_valid']:
            print()
            print("  ⚠ WARNING: Results may not be meaningful!")
            if validation['interpretation'] == 'likely_noise_high_detection':
                print("    High detection rate with low coherence suggests wrong parameters.")
                print("    Try: --auto-discover to find correct d-spacing range")
            elif validation['orientation_entropy'] > 0.8:
                print("    Orientations appear random (high entropy).")
                print("    This could indicate noise rather than real domains.")
        else:
            print()
            print("  ✓ Results appear valid")
        
        params['validation'] = validation
        
        # Update parameters file with validation
        with open(params_path, 'w') as f:
            json.dump(params_serializable, f, indent=2)
    
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
