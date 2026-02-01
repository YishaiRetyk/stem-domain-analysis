# STEM Domain Analysis

Automated crystal domain segmentation and orientation mapping for STEM-HAADF (Scanning Transmission Electron Microscopy - High-Angle Annular Dark-Field) images.

## Overview

This toolkit processes large-scale STEM images to:
- **Detect crystalline domains** via FFT-based radial profile analysis
- **Map crystal orientations** using diffraction peak localization
- **Segment domain regions** with clustering algorithms
- **Generate quantitative metrics** for domain size, shape, and orientation distributions

## Features

- 📊 **Radial Profile Analysis** — Extract and visualize FFT intensity profiles in q-space (nm⁻¹)
- 🎯 **Peak Detection** — Identify crystalline diffraction peaks within specified d-spacing ranges
- 🗺️ **Orientation Mapping** — Color-coded visualization of crystal orientations across the sample
- 📐 **Domain Metrics** — Compute area, perimeter, circularity, and orientation statistics
- 🔬 **DM4 Support** — Native reading of Gatan Digital Micrograph files with metadata extraction

## Installation

### Requirements

- Python 3.10+
- NumPy, SciPy, scikit-image
- Matplotlib
- ncempy (for DM4 file reading)

### Setup

```bash
# Clone the repository
git clone https://github.com/YishaiRetyk/stem-domain-analysis.git
cd stem-domain-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy scipy scikit-image matplotlib ncempy
```

## Quick Start

### Automated Script (Recommended)

The easiest way to run the analysis:

```bash
# Interactive mode - prompts for any missing parameters
python analyze.py sample.dm4

# Specify all parameters via command line
python analyze.py sample.dm4 \
    --pixel-size 0.1297 \
    --d-min 0.5 \
    --d-max 1.5 \
    --threshold 3000 \
    -o results/

# Non-interactive mode (fails if parameters missing)
python analyze.py sample.dm4 --pixel-size 0.1297 --d-min 0.5 --d-max 1.5 --no-interactive
```

The script will:
1. Load the DM4/TIFF/NPY file
2. Extract metadata (pixel size) if available
3. **Prompt for missing parameters** (d-spacing range, threshold, etc.)
4. Preprocess the image
5. Run radial FFT analysis
6. Generate all output visualizations

### Python API

For programmatic use:

```python
from src.io_dm4 import load_dm4_image
from src.preprocess import preprocess_image
from src.radial_analysis import run_radial_analysis

# Load DM4 file
image, metadata = load_dm4_image('sample.dm4')
pixel_size = metadata.get('pixel_size_nm', 0.1297)  # nm/pixel

# Preprocess (normalize, denoise)
processed = preprocess_image(image)

# Target d-spacing: 0.5-1.5 nm → q-range: 0.67-2.0 nm⁻¹
params = {
    'q_range': (0.67, 2.0),       # 1/nm
    'intensity_threshold': 3000,  # Peak detection threshold
    'tile_size': 256,             # FFT tile size in pixels
    'stride': 128,                # Tile overlap
}

results = run_radial_analysis(
    processed, 
    pixel_size_nm=pixel_size,
    output_dir='outputs',
    params=params,
    verbose=True
)

print(f"Detected {results['peak_results']['n_peaks']} crystalline tiles")
```

### Output Files

The analysis generates:

| File | Description |
|------|-------------|
| `1_Radial_Profile.png` | FFT radial intensity profile (log scale) with highlighted q-range |
| `3_Peak_Location_Map.png` | Spatial map of detected crystalline peaks (green overlay) |
| `4_Orientation_Map.png` | Color-coded crystal orientation map |
| `Heatmap.png` | Peak intensity heatmap across tile grid |

## Parameters

### Radial Analysis

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_range` | `(2.3, 2.6)` | Target q-range in nm⁻¹ for peak detection |
| `intensity_threshold` | `45000` | Minimum peak intensity for detection |
| `tile_size` | `256` | FFT window size in pixels |
| `stride` | `128` | Step between tiles (overlap = tile_size - stride) |
| `window` | `'hann'` | Windowing function for FFT |
| `dc_mask_radius` | `3` | Radius to mask central DC component |

### Converting d-spacing to q

```
q (nm⁻¹) = 1 / d (nm)
```

| d-spacing (nm) | q (nm⁻¹) |
|----------------|----------|
| 0.5 | 2.0 |
| 1.0 | 1.0 |
| 1.5 | 0.67 |
| 2.0 | 0.5 |

## Module Reference

### `src/io_dm4.py`
DM4 file reader with metadata extraction (pixel size, acquisition parameters).

### `src/preprocess.py`
Image preprocessing: normalization, background subtraction, denoising.

### `src/radial_analysis.py`
Core FFT analysis: radial profiles, peak detection, orientation mapping.

### `src/fft_features.py`
Feature extraction from FFT power spectra for machine learning.

### `src/cluster_domains.py`
Domain clustering and segmentation algorithms.

### `src/domain_metrics.py`
Quantitative metrics: area, perimeter, orientation statistics.

### `src/viz.py`
Visualization utilities for results and diagnostics.

### `src/ilastik_roi.py`
ROI export compatible with Ilastik segmentation workflow.

## Example Results

### Radial Profile (Log Scale)
Shows FFT intensity vs. spatial frequency with target q-range highlighted:
- Central beam dominates at q≈0
- Crystalline peaks visible in selected range
- Log scale reveals weak diffraction features

### Orientation Map
Color-coded by crystal orientation angle (-180° to +180°):
- Similar colors indicate same crystallographic orientation
- Domain boundaries visible where colors change
- Useful for grain boundary analysis

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{stem_domain_analysis,
  title={STEM Domain Analysis},
  author={Yishai Retyk},
  year={2025},
  url={https://github.com/YishaiRetyk/stem-domain-analysis}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
