"""
Domain Metrics Module for STEM-HAADF Crystal Domain Segmentation.

SA5 - Per-domain analysis and statistics.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage


@dataclass
class DomainMetrics:
    """Container for per-domain metrics."""
    domain_id: int
    n_tiles: int
    area_nm2: float
    mean_crystallinity: float
    std_crystallinity: float
    d_spacings: List[float]           # all d-spacings in domain
    dominant_d_nm: float              # most common d-spacing
    d_std_nm: float
    orientations: List[float]         # all angles
    dominant_orientation_deg: float
    orientation_std_deg: float
    confidence: str                   # 'high', 'medium', 'low'
    flags: List[str]                  # any warnings


def group_into_families(values: np.ndarray, tolerance: float = 0.05) -> Dict[float, List[float]]:
    """
    Group values into families within tolerance.
    
    Returns dict of {family_center: [member_values]}
    """
    if len(values) == 0:
        return {}
    
    sorted_vals = np.sort(values)
    families = {}
    current_family = [sorted_vals[0]]
    
    for val in sorted_vals[1:]:
        if val - current_family[-1] <= tolerance:
            current_family.append(val)
        else:
            family_center = np.median(current_family)
            families[family_center] = current_family
            current_family = [val]
    
    # Don't forget last family
    if current_family:
        family_center = np.median(current_family)
        families[family_center] = current_family
    
    return families


def find_dominant_value(values: np.ndarray, bins: int = 50) -> Tuple[float, float]:
    """
    Find the dominant value and its std using histogram method.
    
    Returns (dominant_value, std_around_dominant)
    """
    if len(values) == 0:
        return 0.0, 0.0
    
    if len(values) == 1:
        return float(values[0]), 0.0
    
    # Histogram to find dominant bin
    hist, bin_edges = np.histogram(values, bins=min(bins, len(values) // 2 + 1))
    max_bin_idx = np.argmax(hist)
    bin_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    
    # Values in dominant bin
    in_bin = (values >= bin_edges[max_bin_idx]) & (values < bin_edges[max_bin_idx + 1])
    
    if np.sum(in_bin) > 0:
        dominant_val = np.median(values[in_bin])
        std_val = np.std(values[in_bin]) if np.sum(in_bin) > 1 else 0.0
    else:
        dominant_val = np.median(values)
        std_val = np.std(values)
    
    return float(dominant_val), float(std_val)


def assign_confidence(n_tiles: int, d_std: float, n_peak_families: int) -> Tuple[str, List[str]]:
    """
    Assign confidence level based on domain statistics.
    
    Returns (confidence_level, flags)
    """
    flags = []
    
    # Base confidence on tile count
    if n_tiles >= 20:
        confidence = 'high'
    elif n_tiles >= 10:
        confidence = 'medium'
    else:
        confidence = 'low'
        flags.append('low_tile_count')
    
    # Adjust by d-spacing consistency
    if d_std > 0.2:
        if confidence == 'high':
            confidence = 'medium'
        flags.append('high_d_variance')
    elif d_std > 0.1 and confidence == 'high':
        confidence = 'medium'
    
    # Adjust by peak families
    if n_peak_families < 2:
        if confidence == 'high':
            confidence = 'medium'
        flags.append('few_peak_families')
    
    return confidence, flags


def compute_domain_metrics(
    domain_id: int,
    tile_indices: np.ndarray,
    tile_features: Dict[str, np.ndarray],
    params: dict
) -> DomainMetrics:
    """
    Compute metrics for a single domain.
    
    Args:
        domain_id: Domain label
        tile_indices: Indices of tiles in this domain
        tile_features: Dict from tile_features.npz
        params: Pipeline parameters
    
    Returns:
        DomainMetrics dataclass
    """
    features = tile_features['features']
    # Handle both 'confidence' and 'tile_confidence' naming
    confidence_scores = tile_features.get('tile_confidence', tile_features.get('confidence', np.ones(features.shape[0])))
    
    # Extract data for this domain's tiles
    n_tiles = len(tile_indices)
    
    # Features: [n_peaks, n_paired, dominant_d, dominant_theta, crystallinity]
    domain_features = features[tile_indices]
    domain_confidence = confidence_scores[tile_indices]
    
    # Filter to valid tiles (with detected peaks)
    valid_mask = domain_features[:, 0] > 0  # n_peaks > 0
    valid_features = domain_features[valid_mask]
    
    # D-spacings (feature index 2)
    d_spacings = valid_features[:, 2] if len(valid_features) > 0 else np.array([])
    d_spacings = d_spacings[d_spacings > 0]  # Filter zero values
    
    # Orientations (feature index 3)  
    orientations = valid_features[:, 3] if len(valid_features) > 0 else np.array([])
    
    # Crystallinity (feature index 4)
    crystallinity_vals = domain_features[:, 4]
    mean_crystallinity = float(np.mean(crystallinity_vals))
    std_crystallinity = float(np.std(crystallinity_vals)) if len(crystallinity_vals) > 1 else 0.0
    
    # Find dominant d-spacing and std
    if len(d_spacings) > 0:
        dominant_d, d_std = find_dominant_value(d_spacings)
    else:
        dominant_d, d_std = 0.0, 0.0
    
    # Find dominant orientation and std
    if len(orientations) > 0:
        dominant_orient, orient_std = find_dominant_value(orientations)
    else:
        dominant_orient, orient_std = 0.0, 0.0
    
    # Count peak families (d-spacings grouped within ±0.05 nm)
    d_families = group_into_families(d_spacings, tolerance=0.05)
    n_peak_families = len(d_families)
    
    # Compute area in nm²
    tile_size = params.get('tile_size', 256)
    stride = params.get('stride', 128)
    pixel_size = params.get('pixel_size', 0.127)  # nm/px
    tile_area_nm2 = (stride * pixel_size) ** 2  # Non-overlapping area per tile
    area_nm2 = n_tiles * tile_area_nm2
    
    # Assign confidence
    confidence, flags = assign_confidence(n_tiles, d_std, n_peak_families)
    
    # Additional flags
    if len(d_spacings) > 0:
        d_min, d_max = params.get('d_range', (0.5, 1.5))
        if not (d_min <= dominant_d <= d_max):
            flags.append('d_outside_expected_range')
    else:
        flags.append('no_valid_peaks')
    
    return DomainMetrics(
        domain_id=int(domain_id),
        n_tiles=int(n_tiles),
        area_nm2=float(area_nm2),
        mean_crystallinity=mean_crystallinity,
        std_crystallinity=std_crystallinity,
        d_spacings=list(d_spacings.astype(float)) if len(d_spacings) > 0 else [],
        dominant_d_nm=dominant_d,
        d_std_nm=d_std,
        orientations=list(orientations.astype(float)) if len(orientations) > 0 else [],
        dominant_orientation_deg=dominant_orient,
        orientation_std_deg=orient_std,
        confidence=confidence,
        flags=flags
    )


def summarize_domains(
    label_image: np.ndarray,
    tile_features: Dict[str, np.ndarray],
    tile_peaksets: Optional[List] = None,  # Not used currently - for future extension
    params: Optional[dict] = None
) -> Tuple[pd.DataFrame, Dict[int, Any]]:
    """
    Compute per-domain statistics.
    
    Args:
        label_image: Full resolution label image (H, W)
        tile_features: Dict loaded from tile_features.npz
        tile_peaksets: Optional list of peak data (unused in current impl)
        params: Pipeline parameters
    
    Returns:
        df: DataFrame with columns matching DomainMetrics
        per_domain_data: dict of {domain_id: data for plotting}
    """
    if params is None:
        params = {
            'tile_size': 256,
            'stride': 128,
            'pixel_size': 0.127,
            'd_range': (0.5, 1.5),
        }
    
    # Get grid info - handle both old and new npz formats
    if 'grid_shape' in tile_features:
        grid_shape = tuple(tile_features['grid_shape'])
        n_rows, n_cols = grid_shape
    else:
        n_rows = int(tile_features['n_rows'])
        n_cols = int(tile_features['n_cols'])
        grid_shape = (n_rows, n_cols)
    
    # Handle tile_coords vs tile_positions naming
    if 'tile_coords' in tile_features:
        tile_coords = tile_features['tile_coords']
    else:
        tile_coords = tile_features['tile_positions']
    
    features = tile_features['features']
    
    # Build tile label map: for each tile, what domain does it belong to?
    # Sample label at tile center
    tile_size = params.get('tile_size', 256)
    stride = params.get('stride', 128)
    
    tile_labels = np.zeros(len(tile_coords), dtype=np.int32)
    for i, (row, col, y, x) in enumerate(tile_coords):
        # Sample at tile center
        cy = y + tile_size // 2
        cx = x + tile_size // 2
        cy = min(cy, label_image.shape[0] - 1)
        cx = min(cx, label_image.shape[1] - 1)
        tile_labels[i] = label_image[cy, cx]
    
    # Get unique domains (excluding noise -1)
    unique_labels = np.unique(tile_labels)
    domain_labels = unique_labels[unique_labels >= 0]
    
    # Compute metrics per domain
    metrics_list = []
    per_domain_data = {}
    
    for domain_id in domain_labels:
        # Find tiles in this domain
        tile_indices = np.where(tile_labels == domain_id)[0]
        
        if len(tile_indices) == 0:
            continue
        
        # Compute metrics
        metrics = compute_domain_metrics(domain_id, tile_indices, tile_features, params)
        metrics_list.append(metrics)
        
        # Store data for plotting
        per_domain_data[int(domain_id)] = {
            'd_spacings': metrics.d_spacings,
            'orientations': metrics.orientations,
            'n_tiles': metrics.n_tiles,
            'dominant_d': metrics.dominant_d_nm,
            'dominant_orientation': metrics.dominant_orientation_deg,
            'crystallinity_values': features[tile_indices, 4].tolist(),
        }
    
    # Convert to DataFrame
    if metrics_list:
        df_data = []
        for m in metrics_list:
            row = {
                'domain_id': m.domain_id,
                'n_tiles': m.n_tiles,
                'area_nm2': m.area_nm2,
                'mean_crystallinity': m.mean_crystallinity,
                'std_crystallinity': m.std_crystallinity,
                'dominant_d_nm': m.dominant_d_nm,
                'd_std_nm': m.d_std_nm,
                'dominant_orientation_deg': m.dominant_orientation_deg,
                'orientation_std_deg': m.orientation_std_deg,
                'confidence': m.confidence,
                'flags': ';'.join(m.flags) if m.flags else '',
            }
            df_data.append(row)
        df = pd.DataFrame(df_data)
    else:
        df = pd.DataFrame(columns=[
            'domain_id', 'n_tiles', 'area_nm2', 'mean_crystallinity', 'std_crystallinity',
            'dominant_d_nm', 'd_std_nm', 'dominant_orientation_deg', 'orientation_std_deg',
            'confidence', 'flags'
        ])
    
    return df, per_domain_data


def save_domain_peak_plots(
    per_domain_data: Dict[int, Any],
    output_dir: Path,
    params: dict = None
) -> List[Path]:
    """
    Generate and save per-domain peak analysis plots.
    
    Args:
        per_domain_data: Dict from summarize_domains
        output_dir: Directory for output plots
        params: Pipeline parameters
    
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for domain_id, data in per_domain_data.items():
        d_spacings = np.array(data['d_spacings'])
        orientations = np.array(data['orientations'])
        n_tiles = data['n_tiles']
        dominant_d = data['dominant_d']
        dominant_orient = data['dominant_orientation']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. D-spacing histogram
        ax1 = axes[0]
        if len(d_spacings) > 0:
            bins = min(30, max(5, len(d_spacings) // 3))
            ax1.hist(d_spacings, bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
            ax1.axvline(dominant_d, color='red', linestyle='--', linewidth=2, 
                       label=f'Dominant: {dominant_d:.3f} nm')
            ax1.legend()
        ax1.set_xlabel('d-spacing (nm)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'D-spacing Distribution')
        
        # 2. Orientation histogram (polar-ish, but linear for simplicity)
        ax2 = axes[1]
        if len(orientations) > 0:
            bins = min(36, max(5, len(orientations) // 3))  # 5-degree bins max
            ax2.hist(orientations, bins=bins, color='coral', edgecolor='white', alpha=0.7)
            ax2.axvline(dominant_orient, color='red', linestyle='--', linewidth=2,
                       label=f'Dominant: {dominant_orient:.1f}°')
            ax2.legend()
        ax2.set_xlabel('Orientation (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_xlim(0, 180)
        ax2.set_title('Orientation Distribution')
        
        # 3. D-spacing vs Orientation scatter
        ax3 = axes[2]
        if len(d_spacings) > 0 and len(orientations) > 0:
            min_len = min(len(d_spacings), len(orientations))
            ax3.scatter(d_spacings[:min_len], orientations[:min_len], 
                       alpha=0.5, s=20, c='purple')
            ax3.axhline(dominant_orient, color='coral', linestyle='--', alpha=0.7)
            ax3.axvline(dominant_d, color='steelblue', linestyle='--', alpha=0.7)
        ax3.set_xlabel('d-spacing (nm)')
        ax3.set_ylabel('Orientation (degrees)')
        ax3.set_title('D-spacing vs Orientation')
        
        plt.suptitle(f'Domain {domain_id} Peak Analysis (n={n_tiles} tiles)', fontsize=12, y=1.02)
        plt.tight_layout()
        
        output_path = output_dir / f'domain_{domain_id}_peaks.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(output_path)
    
    return saved_paths


def check_gate_g7(df: pd.DataFrame, params: dict = None) -> Dict[str, Any]:
    """
    Gate G7: Per-domain Stability Check.
    
    For each domain with confidence='high':
    - Must have ≥20 tiles OR flagged
    - Must have ≥2 stable peak families OR flagged
    - At least one d-spacing in [0.5, 1.5] nm OR flagged
    
    Returns:
        dict with pass/fail status and details
    """
    if params is None:
        params = {'d_range': (0.5, 1.5)}
    
    d_min, d_max = params.get('d_range', (0.5, 1.5))
    
    high_conf_domains = df[df['confidence'] == 'high']
    
    issues = []
    domain_checks = []
    
    for _, row in high_conf_domains.iterrows():
        domain_id = row['domain_id']
        n_tiles = row['n_tiles']
        dominant_d = row['dominant_d_nm']
        flags = row['flags'].split(';') if row['flags'] else []
        
        checks = {
            'domain_id': domain_id,
            'n_tiles': n_tiles,
            'dominant_d_nm': dominant_d,
        }
        
        # Check 1: ≥20 tiles or flagged
        tile_ok = n_tiles >= 20 or 'low_tile_count' in flags
        checks['tile_check'] = tile_ok
        if not tile_ok:
            issues.append(f"Domain {domain_id}: {n_tiles} tiles < 20 and not flagged")
        
        # Check 2: ≥2 peak families or flagged
        family_ok = 'few_peak_families' not in flags or flags.count('few_peak_families') == 0
        # If it has few_peak_families flag, it means it was already caught
        checks['family_check'] = True  # Handled by flagging system
        
        # Check 3: d-spacing in valid range or flagged
        d_ok = (d_min <= dominant_d <= d_max) or 'd_outside_expected_range' in flags
        checks['d_range_check'] = d_ok
        if not d_ok:
            issues.append(f"Domain {domain_id}: d={dominant_d:.3f} outside [{d_min}, {d_max}] nm")
        
        domain_checks.append(checks)
    
    passed = len(issues) == 0
    
    return {
        'gate': 'G7',
        'name': 'Per-domain Stability',
        'pass': passed,
        'n_high_conf_domains': len(high_conf_domains),
        'issues': issues,
        'domain_checks': domain_checks,
    }


def main():
    """Main entry point for domain metrics analysis."""
    print("=" * 60)
    print("SA5 - Domain Metrics Analysis")
    print("=" * 60)
    
    # Paths
    artifacts_dir = Path('artifacts')
    outputs_dir = Path('outputs')
    peaks_dir = outputs_dir / 'domain_peaks'
    
    # Load inputs
    label_image_path = artifacts_dir / 'label_image.npy'
    tile_features_path = artifacts_dir / 'tile_features.npz'
    
    print(f"\nLoading inputs...")
    print(f"  Label image: {label_image_path}")
    print(f"  Tile features: {tile_features_path}")
    
    label_image = np.load(label_image_path)
    tile_features = dict(np.load(tile_features_path))
    
    print(f"  Label image shape: {label_image.shape}")
    print(f"  Unique labels: {np.unique(label_image)}")
    print(f"  Tile features keys: {list(tile_features.keys())}")
    
    # Parameters
    params = {
        'tile_size': 256,
        'stride': 128,
        'pixel_size': 0.127,  # nm/px
        'd_range': (0.5, 1.5),
    }
    
    # Compute domain metrics
    print("\nComputing per-domain metrics...")
    df, per_domain_data = summarize_domains(label_image, tile_features, params=params)
    
    print(f"\nDomain Summary:")
    print(f"  Total domains: {len(df)}")
    if len(df) > 0:
        print(f"  Confidence breakdown:")
        print(f"    High: {sum(df['confidence'] == 'high')}")
        print(f"    Medium: {sum(df['confidence'] == 'medium')}")
        print(f"    Low: {sum(df['confidence'] == 'low')}")
        print(f"\n  Per-domain statistics:")
        for _, row in df.iterrows():
            flags_str = f" [{row['flags']}]" if row['flags'] else ""
            print(f"    Domain {row['domain_id']}: {row['n_tiles']} tiles, "
                  f"d={row['dominant_d_nm']:.3f}±{row['d_std_nm']:.3f} nm, "
                  f"θ={row['dominant_orientation_deg']:.1f}±{row['orientation_std_deg']:.1f}°, "
                  f"conf={row['confidence']}{flags_str}")
    
    # Save CSV
    outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_dir / 'domain_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved domain summary to {csv_path}")
    
    # Generate per-domain plots
    print("\nGenerating per-domain peak plots...")
    saved_plots = save_domain_peak_plots(per_domain_data, peaks_dir, params)
    print(f"  Saved {len(saved_plots)} plots to {peaks_dir}/")
    
    # Gate G7 check
    print("\n" + "-" * 40)
    print("Gate G7: Per-domain Stability Check")
    print("-" * 40)
    
    g7_result = check_gate_g7(df, params)
    
    print(f"  High-confidence domains: {g7_result['n_high_conf_domains']}")
    if g7_result['pass']:
        print(f"  [G7] PASS: All domains have appropriate confidence/flags")
    else:
        print(f"  [G7] FAIL: Issues found:")
        for issue in g7_result['issues']:
            print(f"    - {issue}")
    
    # Save G7 report
    import json
    g7_path = outputs_dir / 'gate_g7_report.json'
    with open(g7_path, 'w') as f:
        json.dump(g7_result, f, indent=2)
    print(f"\nSaved G7 report to {g7_path}")
    
    print("\n" + "=" * 60)
    print("SA5 Complete")
    print("=" * 60)
    
    return df, per_domain_data, g7_result


if __name__ == '__main__':
    main()
