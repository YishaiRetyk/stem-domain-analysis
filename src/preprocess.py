"""
STEM-HAADF Image Preprocessing Module (SA2)

Implements preprocessing with QC metrics for Gate G2 validation.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import json


@dataclass
class PreprocRecord:
    """Container for preprocessed image and metrics."""
    image_pp: np.ndarray    # preprocessed image, float32 [0,1]
    diagnostics: dict       # preprocessing stats
    qc_metrics: dict        # G2 gate metrics


def compute_fft_metrics(image: np.ndarray) -> dict:
    """
    Compute FFT-based QC metrics for an image.
    
    Returns:
        dict with:
        - log_power_mean: mean log-power (DC-masked)
        - spectral_entropy: -sum(p * log(p)) 
        - peak_energy_fraction: sum of top 1% bins / total
    """
    # Compute 2D FFT and shift DC to center
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2
    
    # Mask DC component (center)
    h, w = power.shape
    cy, cx = h // 2, w // 2
    dc_mask = np.ones_like(power, dtype=bool)
    dc_mask[cy-1:cy+2, cx-1:cx+2] = False  # 3x3 DC region
    
    power_no_dc = power[dc_mask]
    
    # Log power mean (avoid log(0))
    log_power = np.log(power_no_dc + 1e-10)
    log_power_mean = float(np.mean(log_power))
    
    # Spectral entropy: normalize power to probability distribution
    p = power_no_dc / (np.sum(power_no_dc) + 1e-10)
    # Avoid log(0)
    p_safe = np.clip(p, 1e-10, 1.0)
    spectral_entropy = float(-np.sum(p_safe * np.log(p_safe)))
    
    # Peak energy fraction: top 1% bins / total
    threshold = np.percentile(power_no_dc, 99)
    top_1pct_sum = np.sum(power_no_dc[power_no_dc >= threshold])
    total_sum = np.sum(power_no_dc)
    peak_energy_fraction = float(top_1pct_sum / (total_sum + 1e-10))
    
    return {
        'log_power_mean': log_power_mean,
        'spectral_entropy': spectral_entropy,
        'peak_energy_fraction': peak_energy_fraction
    }


def save_fft_visualization(image: np.ndarray, output_path: Path, title: str):
    """Save FFT log-power spectrum visualization."""
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log(magnitude + 1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(log_magnitude, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Log Magnitude')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def preprocess(image: np.ndarray, params: dict) -> PreprocRecord:
    """
    Preprocess STEM-HAADF image.
    
    Steps:
    1. Clip outliers (hot pixels) at configured percentiles
    2. Normalize to [0, 1]
    3. Optional: Gaussian smoothing for noise reduction
    
    Args:
        image: Input image as numpy array
        params: Dict with keys:
            - outlier_percentile: float (default 0.1), clips at p and 100-p
            - normalize: bool (default True)
            - smooth_sigma: float (default 0.5), 0 to disable
    
    Returns:
        PreprocRecord with processed image and metrics
    """
    # Extract parameters with defaults
    outlier_pct = params.get('outlier_percentile', 0.1)
    normalize = params.get('normalize', True)
    smooth_sigma = params.get('smooth_sigma', 0.5)
    
    # Track diagnostics
    diagnostics = {
        'input_shape': image.shape,
        'input_dtype': str(image.dtype),
        'input_min': float(np.min(image)),
        'input_max': float(np.max(image)),
        'input_mean': float(np.mean(image)),
        'input_std': float(np.std(image)),
    }
    
    # Convert to float32 for processing
    img = image.astype(np.float32)
    
    # Step 1: Clip outliers (hot pixels)
    low_pct = outlier_pct
    high_pct = 100 - outlier_pct
    p_low = np.percentile(img, low_pct)
    p_high = np.percentile(img, high_pct)
    
    diagnostics['clip_low'] = float(p_low)
    diagnostics['clip_high'] = float(p_high)
    diagnostics['pixels_clipped_low'] = int(np.sum(img < p_low))
    diagnostics['pixels_clipped_high'] = int(np.sum(img > p_high))
    
    img = np.clip(img, p_low, p_high)
    
    # Step 2: Normalize to [0, 1]
    if normalize:
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        diagnostics['normalized'] = True
    else:
        diagnostics['normalized'] = False
    
    # Step 3: Optional Gaussian smoothing
    if smooth_sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=smooth_sigma)
        diagnostics['smoothed'] = True
        diagnostics['smooth_sigma'] = smooth_sigma
    else:
        diagnostics['smoothed'] = False
        diagnostics['smooth_sigma'] = 0
    
    # Final stats
    diagnostics['output_min'] = float(np.min(img))
    diagnostics['output_max'] = float(np.max(img))
    diagnostics['output_mean'] = float(np.mean(img))
    diagnostics['output_std'] = float(np.std(img))
    
    # Compute QC metrics on preprocessed image
    qc_metrics = compute_fft_metrics(img)
    
    return PreprocRecord(
        image_pp=img.astype(np.float32),
        diagnostics=diagnostics,
        qc_metrics=qc_metrics
    )


def validate_g2_gate(raw_metrics: dict, pp_metrics: dict) -> tuple[bool, str]:
    """
    Validate Gate G2: preprocessing should not degrade spectral quality.
    
    PASS if: spectral entropy is similar or lower (within 10% tolerance)
    """
    raw_entropy = raw_metrics['spectral_entropy']
    pp_entropy = pp_metrics['spectral_entropy']
    
    # Allow up to 10% increase in entropy
    tolerance = 0.10
    max_allowed = raw_entropy * (1 + tolerance)
    
    passed = pp_entropy <= max_allowed
    
    reason = f"Entropy: raw={raw_entropy:.4f}, preprocessed={pp_entropy:.4f}"
    if passed:
        if pp_entropy < raw_entropy:
            reason += " (improved)"
        else:
            reason += f" (within {tolerance*100:.0f}% tolerance)"
    else:
        reason += f" (DEGRADED by {(pp_entropy/raw_entropy - 1)*100:.1f}%)"
    
    return passed, reason


def main():
    """Main entry point for preprocessing."""
    # Paths
    artifacts_dir = Path('artifacts')
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    # Load input
    image_path = artifacts_dir / 'image.npy'
    metadata_path = artifacts_dir / 'metadata.json'
    
    print("=" * 60)
    print("SA2: Preprocessing Module")
    print("=" * 60)
    
    print(f"\nLoading image from {image_path}...")
    image = np.load(image_path)
    print(f"  Shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Range: [{image.min():.2f}, {image.max():.2f}]")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"  Metadata: {metadata.get('source', 'unknown')}")
    
    # Compute raw FFT metrics
    print("\nComputing raw image FFT metrics...")
    raw_metrics = compute_fft_metrics(image.astype(np.float32))
    for k, v in raw_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save raw FFT visualization
    save_fft_visualization(
        image.astype(np.float32), 
        outputs_dir / 'global_fft_raw.png',
        'Raw Image FFT (Log Power)'
    )
    print(f"  Saved: outputs/global_fft_raw.png")
    
    # Preprocess
    params = {
        'outlier_percentile': 0.1,
        'normalize': True,
        'smooth_sigma': 0.5
    }
    
    print(f"\nPreprocessing with params: {params}")
    result = preprocess(image, params)
    
    print("\nDiagnostics:")
    for k, v in result.diagnostics.items():
        print(f"  {k}: {v}")
    
    print("\nPreprocessed FFT metrics:")
    for k, v in result.qc_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save outputs
    np.save(artifacts_dir / 'preprocessed.npy', result.image_pp)
    print(f"\nSaved: artifacts/preprocessed.npy")
    
    # Save preprocessed visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Raw Image')
    axes[0].axis('off')
    
    axes[1].imshow(result.image_pp, cmap='gray')
    axes[1].set_title('Preprocessed Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(outputs_dir / 'preprocessed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: outputs/preprocessed.png")
    
    # Save preprocessed FFT visualization
    save_fft_visualization(
        result.image_pp,
        outputs_dir / 'global_fft_preprocessed.png',
        'Preprocessed Image FFT (Log Power)'
    )
    print(f"Saved: outputs/global_fft_preprocessed.png")
    
    # Gate G2 Validation
    print("\n" + "=" * 60)
    print("GATE G2 VALIDATION")
    print("=" * 60)
    
    passed, reason = validate_g2_gate(raw_metrics, result.qc_metrics)
    
    print(f"\nMetrics Comparison:")
    print(f"  {'Metric':<25} {'Raw':>15} {'Preprocessed':>15} {'Delta':>10}")
    print(f"  {'-'*65}")
    for key in raw_metrics:
        raw_val = raw_metrics[key]
        pp_val = result.qc_metrics[key]
        delta = pp_val - raw_val
        delta_pct = (delta / raw_val * 100) if raw_val != 0 else 0
        print(f"  {key:<25} {raw_val:>15.6f} {pp_val:>15.6f} {delta_pct:>+9.1f}%")
    
    print(f"\nValidation: {reason}")
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{'='*60}")
    print(f"GATE G2: {status}")
    print(f"{'='*60}")
    
    # Save gate result
    gate_result = {
        'gate': 'G2',
        'passed': passed,
        'reason': reason,
        'raw_metrics': raw_metrics,
        'preprocessed_metrics': result.qc_metrics,
        'diagnostics': result.diagnostics
    }
    
    with open(artifacts_dir / 'g2_result.json', 'w') as f:
        json.dump(gate_result, f, indent=2)
    print(f"\nSaved: artifacts/g2_result.json")
    
    return passed


if __name__ == '__main__':
    main()
