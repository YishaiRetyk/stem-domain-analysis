"""DM4 file loading and metadata extraction for STEM-HAADF images."""

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Pipeline constant
PIPELINE_PX_NM = 0.127  # Expected pixel size in nm/px


@dataclass
class ImageRecord:
    """Container for loaded DM4 image data."""
    image: np.ndarray  # shape (H, W), float32
    px_nm: float       # pixel size in nm
    metadata: dict     # full DM4 metadata


def load_dm4(path: str) -> ImageRecord:
    """
    Load DM4 file using hyperspy.
    
    Args:
        path: Path to DM4 file
        
    Returns:
        ImageRecord with image data, pixel size, and metadata
    """
    import hyperspy.api as hs
    
    logger.info(f"Loading DM4 file: {path}")
    
    # Load with hyperspy
    s = hs.load(path)
    
    # Handle case where load returns a list
    if isinstance(s, list):
        logger.info(f"Multiple signals found, using first of {len(s)}")
        s = s[0]
    
    # Extract image data as float32
    image = s.data.astype(np.float32)
    logger.info(f"Raw image shape: {image.shape}, dtype: {image.dtype}")
    
    # Extract pixel size from calibration
    axes = s.axes_manager
    signal_axes = axes.signal_axes
    metadata_px_nm = None
    
    # Try to get pixel size from signal axes (spatial axes)
    if len(signal_axes) >= 2:
        # Get scale from first spatial axis
        axis0 = signal_axes[0]
        axis1 = signal_axes[1]
        
        scale0 = axis0.scale
        scale1 = axis1.scale
        units0 = axis0.units
        units1 = axis1.units
        
        logger.info(f"Axis 0 ({axis0.name}): scale={scale0}, units={units0}")
        logger.info(f"Axis 1 ({axis1.name}): scale={scale1}, units={units1}")
        
        # Convert to nm if needed
        if units0 == 'nm':
            metadata_px_nm = scale0
        elif units0 == 'µm' or units0 == 'um':
            metadata_px_nm = scale0 * 1000
        elif units0 == 'Å' or units0 == 'A':
            metadata_px_nm = scale0 / 10
        elif units0 == 'm':
            metadata_px_nm = scale0 * 1e9
        else:
            logger.warning(f"Unknown unit: {units0}, assuming nm")
            metadata_px_nm = scale0
    
    # Use pipeline constant, log discrepancy if any
    px_nm = PIPELINE_PX_NM
    if metadata_px_nm is not None:
        discrepancy = abs(metadata_px_nm - PIPELINE_PX_NM) / PIPELINE_PX_NM * 100
        if discrepancy > 1:  # More than 1% difference
            logger.warning(f"Pixel size discrepancy: metadata={metadata_px_nm:.6f} nm/px, "
                          f"pipeline={PIPELINE_PX_NM:.6f} nm/px ({discrepancy:.1f}% diff)")
        else:
            logger.info(f"Pixel size from metadata: {metadata_px_nm:.6f} nm/px (matches pipeline)")
    
    # Build metadata dict
    metadata = {
        'original_metadata': _serialize_metadata(s.original_metadata.as_dictionary()),
        'axes': [
            {
                'name': ax.name,
                'scale': ax.scale,
                'offset': ax.offset,
                'units': ax.units,
                'size': ax.size
            }
            for ax in signal_axes
        ],
        'pixel_size_nm_metadata': metadata_px_nm,
        'pixel_size_nm_used': px_nm,
        'shape': list(image.shape),
        'dtype': str(image.dtype),
        'signal_type': s.metadata.get_item('Signal.signal_type', 'unknown'),
    }
    
    return ImageRecord(image=image, px_nm=px_nm, metadata=metadata)


def _serialize_metadata(obj):
    """Convert metadata to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _serialize_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_metadata(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def validate_g1(record: ImageRecord) -> dict:
    """
    Validate G1 gate requirements.
    
    Returns dict with validation results.
    """
    results = {
        'shape': record.image.shape,
        'dtype': str(record.image.dtype),
        'min': float(np.nanmin(record.image)),
        'max': float(np.nanmax(record.image)),
        'has_nan': bool(np.any(np.isnan(record.image))),
        'has_inf': bool(np.any(np.isinf(record.image))),
        'px_nm': record.px_nm,
        'checks': {}
    }
    
    # Check 1: 2D array
    results['checks']['is_2d'] = len(record.image.shape) == 2
    
    # Check 2: float32
    results['checks']['is_float32'] = record.image.dtype == np.float32
    
    # Check 3: No NaN/Inf
    results['checks']['no_nan_inf'] = not results['has_nan'] and not results['has_inf']
    
    # Check 4: Dimensions ≥512
    h, w = record.image.shape
    results['checks']['min_512'] = h >= 512 and w >= 512
    
    # Overall pass
    results['G1_PASS'] = all(results['checks'].values())
    
    return results


def contrast_stretch(image: np.ndarray, percentile: tuple = (1, 99)) -> np.ndarray:
    """Apply contrast stretching for visualization."""
    p_low, p_high = np.percentile(image, percentile)
    stretched = np.clip((image - p_low) / (p_high - p_low), 0, 1)
    return (stretched * 255).astype(np.uint8)


def main():
    """Load DM4, validate, and save artifacts."""
    import matplotlib.pyplot as plt
    
    dm4_path = "dm4_input/HWAO17_Phe_50mg-mL_t0_surface_wall_0045.dm4"
    
    # Load
    record = load_dm4(dm4_path)
    
    # Validate G1
    g1 = validate_g1(record)
    
    # Print metrics
    print("\n" + "="*60)
    print("G1 GATE VALIDATION")
    print("="*60)
    print(f"Image shape: {g1['shape']}")
    print(f"Data type: {g1['dtype']}")
    print(f"Data range: [{g1['min']:.2f}, {g1['max']:.2f}]")
    print(f"Has NaN: {g1['has_nan']}")
    print(f"Has Inf: {g1['has_inf']}")
    print(f"Pixel size (metadata): {record.metadata.get('pixel_size_nm_metadata', 'N/A')} nm/px")
    print(f"Pixel size (used): {g1['px_nm']} nm/px")
    print("-"*60)
    print("Checks:")
    for check, passed in g1['checks'].items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    print("-"*60)
    print(f"G1 GATE: {'PASS ✓' if g1['G1_PASS'] else 'FAIL ✗'}")
    print("="*60 + "\n")
    
    # Save artifacts
    np.save("artifacts/image.npy", record.image)
    logger.info("Saved artifacts/image.npy")
    
    with open("artifacts/metadata.json", "w") as f:
        json.dump(record.metadata, f, indent=2)
    logger.info("Saved artifacts/metadata.json")
    
    # Save visualization
    vis = contrast_stretch(record.image)
    plt.figure(figsize=(10, 10))
    plt.imshow(vis, cmap='gray')
    plt.title(f"STEM-HAADF: {record.image.shape[0]}×{record.image.shape[1]} px, {record.px_nm} nm/px")
    plt.colorbar(label='Intensity (stretched)')
    plt.tight_layout()
    plt.savefig("outputs/original.png", dpi=150)
    plt.close()
    logger.info("Saved outputs/original.png")
    
    return g1['G1_PASS']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
