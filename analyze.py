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


def get_boundary_conditions(metadata: dict, args) -> dict:
    """
    Get boundary conditions from metadata, args, or prompt user.
    
    Returns dict with:
        - pixel_size_nm: pixel size in nanometers
        - d_min: minimum d-spacing (nm)
        - d_max: maximum d-spacing (nm)
        - q_min: minimum q (1/nm) - computed from d_max
        - q_max: maximum q (1/nm) - computed from d_min
        - intensity_threshold: peak detection threshold
        - tile_size: FFT tile size
        - stride: tile stride
    """
    params = {}
    
    print("\n" + "=" * 60)
    print("BOUNDARY CONDITIONS")
    print("=" * 60)
    
    # Pixel size
    meta_pixel = metadata.get('pixel_size_nm')
    if args.pixel_size:
        params['pixel_size_nm'] = args.pixel_size
        print(f"Pixel size (from args): {params['pixel_size_nm']} nm/pixel")
    elif meta_pixel:
        print(f"Pixel size (from metadata): {meta_pixel} nm/pixel")
        if prompt_yes_no("  Use this value?", default=True):
            params['pixel_size_nm'] = meta_pixel
        else:
            params['pixel_size_nm'] = prompt_float("  Enter pixel size (nm/pixel)")
    else:
        print("Pixel size not found in metadata.")
        params['pixel_size_nm'] = prompt_float("  Enter pixel size (nm/pixel)")
    
    # D-spacing range
    print()
    if args.d_min and args.d_max:
        params['d_min'] = args.d_min
        params['d_max'] = args.d_max
        print(f"D-spacing range (from args): {params['d_min']} - {params['d_max']} nm")
    else:
        print("Target lattice d-spacing range:")
        params['d_min'] = prompt_float("  Minimum d-spacing (nm)", default=0.5)
        params['d_max'] = prompt_float("  Maximum d-spacing (nm)", default=1.5)
    
    # Convert d-spacing to q-range (q = 1/d)
    params['q_min'] = 1.0 / params['d_max']
    params['q_max'] = 1.0 / params['d_min']
    print(f"  → Q-range: {params['q_min']:.3f} - {params['q_max']:.3f} nm⁻¹")
    
    # Intensity threshold
    print()
    if args.threshold:
        params['intensity_threshold'] = args.threshold
        print(f"Intensity threshold (from args): {params['intensity_threshold']}")
    else:
        print("Peak detection threshold:")
        print("  (Higher = more selective, Lower = more detections)")
        params['intensity_threshold'] = prompt_float("  Threshold", default=3000)
    
    # Tile parameters
    print()
    if args.tile_size:
        params['tile_size'] = args.tile_size
    else:
        params['tile_size'] = int(prompt_float("FFT tile size (pixels)", default=256))
    
    if args.stride:
        params['stride'] = args.stride
    else:
        default_stride = params['tile_size'] // 2
        params['stride'] = int(prompt_float("Tile stride (pixels)", default=default_stride))
    
    print(f"  Tile: {params['tile_size']}x{params['tile_size']}, stride: {params['stride']}")
    
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
    from src.io_dm4 import load_dm4_image
    from src.preprocess import preprocess_image
    from src.radial_analysis import run_radial_analysis
    
    # Load input file
    print("\n[1/4] Loading image...")
    
    suffix = input_path.suffix.lower()
    metadata = {}
    
    if suffix == '.dm4' or suffix == '.dm3':
        image, metadata = load_dm4_image(str(input_path))
        print(f"  Loaded DM4: {image.shape}")
        if metadata:
            print(f"  Metadata: {list(metadata.keys())}")
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
    
    # Get boundary conditions
    if args.no_interactive:
        # Check all required params are present
        missing = []
        if not args.pixel_size and 'pixel_size_nm' not in metadata:
            missing.append('--pixel-size')
        if not args.d_min:
            missing.append('--d-min')
        if not args.d_max:
            missing.append('--d-max')
        if missing:
            print(f"Error: Missing required parameters: {', '.join(missing)}")
            print("  Use interactive mode or provide all parameters.")
            sys.exit(1)
    
    params = get_boundary_conditions(metadata, args)
    
    # Preprocess
    print("\n[2/4] Preprocessing...")
    processed = preprocess_image(image, verbose=args.verbose)
    print(f"  Output shape: {processed.shape}")
    print(f"  Range: [{processed.min():.1f}, {processed.max():.1f}]")
    
    # Save preprocessed if requested
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.save_preprocessed:
        npy_path = output_path / 'preprocessed.npy'
        np.save(npy_path, processed)
        print(f"  Saved: {npy_path}")
    
    # Save parameters
    params_path = output_path / 'parameters.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {params_path}")
    
    # Run radial analysis
    print("\n[3/4] Running radial analysis...")
    
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
    print("\n[4/4] Summary")
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
