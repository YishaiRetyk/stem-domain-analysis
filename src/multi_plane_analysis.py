"""Multi-plane discrimination for STEM domain analysis."""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.legacy.radial_analysis import (
    process_tiles_for_peaks,
    compute_global_radial_profile,
    RadialProfile
)
from src.peak_discovery import DiscoveredPeak, DiscoveryResult


@dataclass
class PlaneAnalysisResult:
    """Results for a single lattice plane."""
    plane_id: int
    q_center: float
    q_range: Tuple[float, float]
    d_spacing: float

    # Tile-grid results
    peak_mask: np.ndarray           # (n_rows, n_cols) bool
    orientation_map: np.ndarray     # (n_rows, n_cols) angles
    intensity_map: np.ndarray       # (n_rows, n_cols) peak intensities
    confidence_map: np.ndarray      # (n_rows, n_cols) confidence

    # Statistics
    n_detections: int
    detection_rate: float
    mean_intensity: float

    # Validation
    coherence_score: float
    is_valid: bool


@dataclass
class MultiPlaneResult:
    """Results from multi-plane analysis."""
    planes: List[PlaneAnalysisResult]

    # Global radial profile (shared)
    radial_profile: RadialProfile

    # Discovery metadata
    discovery_result: Optional[DiscoveryResult]

    # Composite maps
    dominant_plane_map: np.ndarray   # (n_rows, n_cols) int - which plane is strongest
    multi_plane_mask: np.ndarray     # (n_rows, n_cols) bool - any plane detected
    overlay_intensity_map: np.ndarray  # (n_rows, n_cols) - max across planes


def analyze_multiple_planes(
    image: np.ndarray,
    pixel_size_nm: float,
    peaks: List[DiscoveredPeak],
    tile_size: int = 256,
    stride: int = 128,
    peak_method: str = 'percentile95',
    verbose: bool = True
) -> MultiPlaneResult:
    """
    Analyze multiple lattice planes simultaneously.

    Args:
        image: Preprocessed image
        pixel_size_nm: Pixel size in nm
        peaks: List of discovered peaks to analyze
        tile_size: FFT tile size
        stride: Tile stride
        peak_method: Peak detection method
        verbose: Print progress

    Returns:
        MultiPlaneResult with per-plane and composite data
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"MULTI-PLANE ANALYSIS ({len(peaks)} planes)")
        print("=" * 60)

    # Compute global radial profile (shared across planes)
    if verbose:
        print("\n[1] Computing global radial profile...")
    radial_profile = compute_global_radial_profile(
        image, pixel_size_nm, tile_size
    )

    # Analyze each plane
    plane_results = []
    for i, peak in enumerate(peaks):
        if verbose:
            print(f"\n[2.{i+1}] Analyzing plane {i}: d={peak.d_spacing:.3f} nm "
                  f"(q={peak.q_center:.2f} nm⁻¹)...")

        # Define q-range for this plane
        q_min = peak.q_center - peak.q_width
        q_max = peak.q_center + peak.q_width

        # Run peak detection for this q-range
        params = {
            'q_range': (q_min, q_max),
            'intensity_threshold': peak.suggested_threshold,
            'tile_size': tile_size,
            'stride': stride,
            'peak_method': peak_method,
        }

        tile_results = process_tiles_for_peaks(
            image, pixel_size_nm, params, verbose=False
        )

        # Validate coherence
        from src.peak_discovery import validate_spatial_coherence
        validation = validate_spatial_coherence(
            tile_results['peak_mask'],
            tile_results['orientation_map']
        )

        # Create plane result
        plane_result = PlaneAnalysisResult(
            plane_id=i,
            q_center=peak.q_center,
            q_range=(q_min, q_max),
            d_spacing=peak.d_spacing,
            peak_mask=tile_results['peak_mask'],
            orientation_map=tile_results['orientation_map'],
            intensity_map=tile_results['intensity_map'],
            confidence_map=tile_results['confidence_map'],
            n_detections=tile_results['n_peaks'],
            detection_rate=tile_results['n_peaks'] / np.prod(tile_results['grid_shape']),
            mean_intensity=float(np.mean(tile_results['intensity_map'][tile_results['peak_mask']]))
                           if tile_results['n_peaks'] > 0 else 0,
            coherence_score=validation['coherence_score'],
            is_valid=validation['is_valid'],
        )

        plane_results.append(plane_result)

        if verbose:
            print(f"    Detections: {plane_result.n_detections} "
                  f"({plane_result.detection_rate*100:.1f}%)")
            print(f"    Coherence: {plane_result.coherence_score:.2f}")
            print(f"    Status: {'VALID' if plane_result.is_valid else 'WEAK'}")

    # Generate composite maps
    if verbose:
        print(f"\n[3] Generating composite maps...")

    composite_maps = _generate_composite_maps(plane_results, verbose=verbose)

    return MultiPlaneResult(
        planes=plane_results,
        radial_profile=radial_profile,
        discovery_result=None,  # Set by caller if using auto-discovery
        **composite_maps
    )


def _generate_composite_maps(
    planes: List[PlaneAnalysisResult],
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate composite maps showing all planes.

    Args:
        planes: List of plane results
        verbose: Print progress

    Returns:
        Dict with composite maps
    """
    if not planes:
        raise ValueError("No planes to composite")

    # Get grid shape from first plane
    grid_shape = planes[0].peak_mask.shape

    # Initialize composite maps
    dominant_plane_map = np.full(grid_shape, -1, dtype=np.int8)  # -1 = no detection
    multi_plane_mask = np.zeros(grid_shape, dtype=bool)
    overlay_intensity_map = np.zeros(grid_shape, dtype=np.float32)

    # For each tile position, find dominant plane
    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            # Collect intensities from all planes at this position
            intensities = []
            plane_ids = []

            for plane in planes:
                if plane.peak_mask[row, col]:
                    intensities.append(plane.intensity_map[row, col])
                    plane_ids.append(plane.plane_id)

            if intensities:
                # Mark as detected
                multi_plane_mask[row, col] = True

                # Find dominant plane (highest intensity)
                max_idx = np.argmax(intensities)
                dominant_plane_map[row, col] = plane_ids[max_idx]
                overlay_intensity_map[row, col] = max(intensities)

    if verbose:
        n_detected = np.sum(multi_plane_mask)
        total = np.prod(grid_shape)
        print(f"    Multi-plane detections: {n_detected}/{total} "
              f"({n_detected/total*100:.1f}%)")

        for plane in planes:
            n_dominant = np.sum(dominant_plane_map == plane.plane_id)
            if n_dominant > 0:
                print(f"    Plane {plane.plane_id} (d={plane.d_spacing:.3f} nm): "
                      f"{n_dominant} tiles dominant")

    return {
        'dominant_plane_map': dominant_plane_map,
        'multi_plane_mask': multi_plane_mask,
        'overlay_intensity_map': overlay_intensity_map,
    }
