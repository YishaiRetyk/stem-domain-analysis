# STEM Domain Analysis

Automated crystal domain segmentation and orientation mapping for STEM-HAADF and EFTEM bright-field (zero-loss filtered) images.

## Overview

This toolkit processes large-scale STEM images to:
- **Detect crystalline domains** via FFT-based radial profile analysis
- **Automatically discover diffraction peaks** without prior material knowledge
- **Map crystal orientations** using diffraction peak localization
- **Analyze per-ring spatial distributions** with presence, peak count, orientation, and SNR maps
- **Cluster crystal domains** via K-means, GMM, or HDBSCAN with automatic K selection
- **Extract strain fields** via Geometric Phase Analysis (GPA)
- **Validate results** with a 13-gate quality control system (G0-G12)
- **Generate quantitative metrics** for domain size, shape, and orientation distributions

## Pipeline Architecture

The pipeline supports two modes: the **classic** tile-FFT workflow and the new **hybrid** FFT + GPA + Peak-Finding pipeline activated with `--hybrid`.

### Classic Pipeline

```mermaid
flowchart TD
    subgraph Input["Input"]
        A[DM4/TIFF/NPY Image]
    end

    subgraph Loading["1. Load & Preprocess"]
        B[Load Image<br/>Extract Metadata]
        C[Preprocess<br/>Normalize, Denoise]
    end

    subgraph Discovery["2. Peak Discovery --auto-discover"]
        D[Sample Tiles<br/>Compute Radial Profile]
        E[Fit Polynomial Background]
        F[Background Subtraction]
        G[Find Diffraction Peaks<br/>Prominence Filter]
        H[Estimate Threshold<br/>per Peak]
    end

    subgraph Analysis["3. Radial Analysis"]
        I[Tile-based FFT]
        J[Peak Detection<br/>in q-range]
        K[Orientation Extraction]
    end

    subgraph Validation["4. Validation --validate"]
        L[Spatial Coherence<br/>Score]
        M[Orientation Entropy]
        N[Domain Clustering<br/>Analysis]
    end

    subgraph Output["Output"]
        O[Peak Discovery Plot]
        P[Radial Profile]
        Q[Peak Location Map]
        R[Orientation Map]
        S[Heatmap]
        T[parameters.json]
    end

    A --> B --> C
    C --> D --> E --> F --> G --> H
    H --> I
    C -.->|"Manual params"| I
    I --> J --> K
    K --> L --> M --> N

    G --> O
    I --> P
    J --> Q
    K --> R
    J --> S
    H --> T
    N --> T

    style Discovery fill:#e1f5fe
    style Validation fill:#fff3e0
```

### Hybrid Pipeline (`--hybrid`)

```mermaid
flowchart TD
    subgraph Input["Input"]
        A[DM4/TIFF/NPY Image<br/>Gate G0: Nyquist Guard<br/>Gate G1: Input Validation]
    end

    subgraph Preprocess["Two-Branch Preprocessing"]
        B1["Branch A: FFT-Safe<br/>Hot-pixel removal, robust normalize<br/>NO Gaussian blur<br/>Gate G2"]
        B2["Branch B: Segmentation<br/>Clip + normalize + blur"]
    end

    subgraph ROI["Early ROI"]
        C["Intensity + Variance Mask<br/>Gradient fallback + LCC fraction<br/>Gate G3"]
    end

    subgraph GlobalFFT["Global FFT"]
        D["Constrained Background Fit<br/>G-vector Extraction<br/>FFT Guidance Strength<br/>Gate G4"]
    end

    subgraph TileFFT["Tile FFT + Two-Tier SNR"]
        E["SNR-first Peak Detection<br/>Unified q-based DC suppression<br/>Gate G5: Tiling Adequacy"]
        F["Two-Tier Classification<br/>Tier A: SNR >= 5.0<br/>Tier B: SNR 3.0-5.0<br/>FWHM-scaled tolerances<br/>Gates G6, G7, G8"]
    end

    subgraph RingAnalysis["Ring Analysis (always-on)"]
        RA["Per-ring spatial maps<br/>Ring feature vectors<br/>Tile-averaged FFT"]
    end

    subgraph Clustering["Domain Clustering (--cluster)"]
        CL["K-means / GMM / HDBSCAN<br/>PCA / UMAP dimensionality reduction<br/>Auto-K via silhouette scan<br/>Spatial regularization"]
    end

    subgraph GPA["GPA (optional, skips on weak evidence)"]
        G["Entry gate: guidance + tier check<br/>Restricted phase unwrapping<br/>Ref-region unwrap success<br/>Gates G9, G10, G11"]
    end

    subgraph PeakFinding["Peak Finding (optional)"]
        H["Directional ±g Bandpass Mask<br/>Subpixel Peak Detection<br/>Adaptive NN-Distance Validation<br/>Gate G12"]
    end

    subgraph Reporting["Validation + Reporting"]
        I["13-Gate Evaluation (G0-G12)<br/>parameters.json v3.0<br/>report.json + pipeline_flow"]
    end

    A --> B1 & B2
    B2 --> C
    B1 --> D
    C --> E
    D --> E
    E --> F
    F --> RA
    RA --> CL
    CL --> G & H
    G --> I
    H --> I
    RA --> I
    CL --> I

    style Preprocess fill:#e8f5e9
    style TileFFT fill:#e1f5fe
    style RingAnalysis fill:#e8eaf6
    style Clustering fill:#fce4ec
    style GPA fill:#f3e5f5
    style Reporting fill:#fff3e0
```

---

## What's New (v3.4) -- Dynamic DC Masking & SNR Enhancements

### Dynamic DC Center Masking (`--dynamic-dc`)
- **Derivative-based DC estimation**: Automatically determines the DC contamination radius from the global radial profile by finding where `|dJ/dq|` drops below a noise-derived threshold for N consecutive bins
- **Tiles reuse global estimate**: The DC boundary is computed once on the full-image FFT and propagated to all tiles (no per-tile recomputation)
- **Hard mask or cosine taper**: Default is hard cutoff; `--dc-soft-taper` enables a smooth cosine transition to reduce ringing
- **Configurable via `DCMaskConfig`** (13 fields): Savitzky-Golay smoothing, slope threshold, noise region, floor/cap bounds, taper width
- **CLI flags**: `--dynamic-dc` (enable), `--dc-floor <float>` (override minimum radius in cycles/nm), `--dc-soft-taper` (cosine taper)

### AsLS Background Fitting (`--bg-method asls`)
- **Asymmetric Least Squares** baseline estimation as an alternative to the iterative-reweighted polynomial background fit
- Uses `scipy.sparse` for efficient banded-matrix operations
- Supports both log-domain and linear-domain fitting
- **CLI flag**: `--bg-method {polynomial_robust,asls}`

### SNR Signal Extraction Methods (`--snr-signal-method`)
- Configurable signal aggregation from the peak measurement disk:
  - `max` -- legacy default, single brightest pixel
  - `integrated_sum` -- sum of all pixels in the signal disk
  - `integrated_median` -- median of all pixels in the signal disk
- **CLI flag**: `--snr-signal-method {max,integrated_sum,integrated_median}`

### Lightweight SNR Zscore Mode
- Tile-level peak SNR can use MAD-based zscore instead of the simple signal/background ratio
- Configured via `TileFFTConfig.lightweight_snr_method`

### Visualization
- Orange dashed line on radial profile marks the dynamic DC boundary (distinct from the gray low-q exclusion region)
- Solid orange circle on the 2D power spectrum shows the DC mask radius

### Reporting
- `dc_mask` section added to `parameters.json` with `enabled`, `dynamic_dc_q`, `method`, and `diagnostics` fields

---

## What's New (v3.3) -- Configurable Pipeline Parameters

### Fully Configurable Gate Thresholds
- All 13 quality gate thresholds (G0-G12) are now configurable via YAML config or `GateThresholdsConfig`
- Previously hardcoded values are exposed as named fields with unchanged defaults
- `GateThresholdsConfig.threshold_dict(gate_id)` provides per-gate threshold dicts
- G3 `min_lcc_fraction` is now configurable (was hardcoded to 0.5)

### New Config Dataclasses
- **`GateThresholdsConfig`**: 22 fields controlling all gate pass/fail thresholds
- **`PeakSNRConfig`**: 9 fields for peak SNR measurement (signal disk radius, annular width, FWHM patch radius, symmetry tolerance, non-collinear angle)
- **`ReferenceSelectionConfig`**: 4 fields for reference region scoring (entropy/SNR/area weights, orientation bins)
- **`RingAnalysisConfig`**: 3 fields for ring width calculation

### Extended Existing Configs
- **`GlobalFFTConfig`**: 15 new fields (background fitting, Savitzky-Golay parameters, harmonic rejection, angular peak detection)
- **`GPAConfig`**: 10 new fields (mask sigma factors, bimodal detection, amplitude erosion, phase noise thresholds)
- **`PreprocConfig`**: 2 new fields (hot pixel median kernel, robust normalization clip sigma)
- **`ROIConfig`**: 4 new fields (variance window, morphological kernel, smoothing sigma, gradient sigma)
- **`PeakFindingConfig`**: 4 new fields (taper width, background percentile, filter size, adaptive tolerance floor)
- **`TileFFTConfig`**: 3 new fields (annulus factors, background disk radius)
- **`ClusteringConfig`**: 5 new fields (min valid tiles, regularization iterations, UMAP/K-means parameters)
- **`ConfidenceConfig`**: 1 new field (SNR ceiling multiplier)

### YAML Configuration
All parameters are loadable from YAML via `--config`. Example:
```yaml
gate_thresholds:
  g5_min_periods: 15.0
  g9_min_area: 4
  g12_min_fraction_valid: 0.30
peak_snr:
  signal_disk_radius_px: 5
  fwhm_patch_radius: 7
reference_selection:
  scoring_weight_entropy: 0.5
  scoring_weight_snr: 0.3
  scoring_weight_area: 0.2
global_fft:
  strong_guidance_snr: 10.0
```

### Backward Compatibility
- Old-style `validation:` YAML keys (e.g., `min_periods`, `ref_snr_min`) automatically sync to the new `gate_thresholds:` fields when `gate_thresholds:` is absent
- All defaults match the previously hardcoded values -- zero behavioral change without explicit configuration

### Bug Fixes
- **Reference selection config threading**: `select_reference_region()` now correctly receives config values from `run_gpa()` (was ignoring caller-provided `min_area`/`max_entropy`/`min_snr`)
- **G3 LCC fraction**: Now configurable via `gate_thresholds.g3_min_lcc_fraction` (was hardcoded 0.5 in gate evaluation logic)

### New CLI Flag
- `--strong-guidance-snr FLOAT` -- Override the SNR threshold for "strong" FFT guidance classification (default: 8.0)

---

## What's New (v3.2) -- Multi-Ring Spatial Maps + Domain Clustering

### Per-Ring Spatial Analysis
- **Ring-indexed peaks**: Each tile peak is tagged with a `ring_index` linking it to the global diffraction ring it belongs to
- **Per-ring spatial maps**: Presence, peak count, orientation, SNR, and angular variance maps for each diffraction ring
- **Ring feature vectors**: Per-tile feature matrix (5 features per ring + 4 global) for downstream clustering
- **Tile-averaged FFT**: Streaming mean of all valid tile power spectra with annotated radial profile

### Domain Clustering (`--cluster`)
- **Three clustering methods**: K-means, GMM, HDBSCAN via `--cluster-method`
- **Automatic K selection**: Silhouette scan over k=2..10 when `--cluster-n 0` (default)
- **Dimensionality reduction**: PCA (default, 95% variance threshold) or UMAP (`--cluster-dimred umap`) with automatic PCA fallback
- **Spatial regularization**: Majority-vote smoothing to reduce label noise at domain boundaries
- **Per-cluster physics summaries**: Dominant ring presences, mean orientations, peak count distributions per cluster
- **Cluster-averaged FFTs**: Streaming mean power spectra per cluster for spectral comparison

### New CLI Flags
- `--cluster` -- Enable domain clustering (opt-in)
- `--cluster-method {kmeans,gmm,hdbscan}` -- Clustering algorithm (default: kmeans)
- `--cluster-n N` -- Number of clusters, 0=auto (default: 0)
- `--cluster-dimred {pca,umap,none}` -- Dimensionality reduction method (default: pca)

### New Output Artifacts
- 9 new data files: `ring_maps.npz`, `ring_feature_vectors.npy`, `ring_feature_names.json`, `average_tile_fft.npy`, `average_tile_radial_profile.json`, `cluster_labels.npy`, `cluster_labels_regularized.npy`, `cluster_averaged_ffts.npz`, `clustering_results.json`
- 14 new PNG visualizations: per-ring presence/peak count/orientation maps, ring overlays, tile-averaged FFT, cluster label map, cluster overlay, feature embedding, cluster-averaged FFTs, radial profiles, silhouette curve, ring-vs-cluster comparison

### Pipeline Stages
- Pipeline expanded from 9 to 11 stages: Ring Analysis (7) and Domain Clustering (8) inserted after Tile FFT (6)
- GPA renumbered to stage 9, Peak Finding to 10, Validation to 11
- Ring analysis runs automatically when global peaks exist; clustering is opt-in via `--cluster`

---

## What's New (v3.1) -- EFTEM Bright-Field Upgrade

### Physics-Aware Configuration
- **`PhysicsConfig`** with d-range bounds (`d_min_nm`, `d_max_nm`), imaging mode, and Nyquist safety margin
- **Gate G0 (Nyquist Guard)**: Auto-clamps analysis q-range to 0.95 x q_nyquist when d_min violates sampling theorem; FATAL if entire band is invalid
- New CLI flags: `--physics-d-min`, `--physics-d-max`, `--imaging-mode`

### Robust Background Fitting
- Constrained polynomial background: degree 3 (cap 4) with `q_fit_min` to exclude low-q oscillations
- **FFT guidance strength** classification ("strong"/"weak"/"none") controls downstream GPA eligibility
- Background residual diagnostics: median|res|, MAD, neg-excursion fraction near peaks
- FWHM-scaled antipodal pairing tolerance (was fixed 5% of q)

### Improved Tile Classification
- **SNR-first peak detection** replaces amplitude-percentage thresholds
- Unified q-based DC suppression via `TileFFTConfig.q_dc_min` (single code path)
- `symmetry_score` renamed to `pair_fraction` (with one-cycle deprecation alias)
- FWHM-scaled symmetry tolerance (scales with measured peak width)
- **Orientation confidence** (circular concentration R) per tile -- diagnostic-only

### ROI Robustness
- **Gradient-magnitude fallback** when primary mask fails coverage or LCC check
- **LCC fraction** (largest connected component / total mask) as primary G3 metric
- Full-image ROI forced with `roi_confidence = "low"` when all fallbacks exhausted

### GPA Safety Gates
- GPA skips when FFT guidance is "none" or Tier A fraction < 10%
- Phase unwrapping restricted to amplitude-valid regions (zero before, NaN after)
- **Reference-region unwrap success** metric for G10 (was global metric)
- Mask sigma uses median(FWHM) instead of min (robust to outliers)

### Lattice-Feature Validation
- **Directional ±g angular wedge masks** replace full-ring bandpass (suppresses non-lattice noise)
- **Adaptive NN tolerance** from IQR when >= 10 peaks (tighter for regular lattices)
- `atom_peaks.npy` renamed to `lattice_peaks.npy`

### Expanded Diagnostics
- 7 new diagnostic visualizations: ROI overlay, background residual, reference boundary, tile exemplar FFTs, phase noise histogram, strain outlier map, NN distance histogram
- `parameters.json` includes `derived_cutoffs`, `physics`, `background_diagnostics`, and `pipeline_flow` sections
- Pipeline flow tracking: stages completed, skipped, degraded, with reasons

---

## What's New (v3.0) -- Hybrid Pipeline

### Two-Branch Preprocessing
- **Branch A (FFT-safe)**: Hot-pixel removal + robust normalization with **no Gaussian blur**, preserving FFT peak integrity
- **Branch B (Segmentation)**: Standard preprocessing with Gaussian smooth for ROI mask computation

### Canonical FFT Coordinate System
- All spatial frequencies in **cycles/nm** throughout the pipeline (`d = 1/|g|`, no 2pi)
- `FFTGrid` class handles coordinate conversions, supporting both square and rectangular images

### Two-Tier SNR Classification
- **Tier A (SNR >= 5.0)**: High-confidence tiles used for reference selection, GPA, and clustering
- **Tier B (3.0 <= SNR < 5.0)**: Weak-evidence tiles for visualization and diagnostics only
- Peak-height SNR with annular background that excludes all detected peaks

### Geometric Phase Analysis (GPA)
- **Full-image mode**: Single reference region, global phase/strain extraction
- **Region-wise mode**: Per-domain analysis with automatic stitching
- **Auto-selection**: Deterministic mode choice based on orientation entropy, Tier A coverage, and reference region viability
- Subpixel-correct demodulation via real-space phase ramp
- Phase unwrapping without zero-forcing, with eroded amplitude post-mask

### 13-Gate Quality Control
- Gates G0-G12 covering Nyquist guard, input validation, preprocessing, ROI geometry, FFT viability, tiling adequacy, SNR quality, pair fraction, reference region, phase noise, strain sanity, and lattice consistency
- Four failure behaviors: FATAL, SKIP_STAGE, DEGRADE_CONFIDENCE, FALLBACK

### Peak Finding with Lattice Validation
- Bandpass-filtered image tuned to lattice frequency suppresses thickness gradients
- Adaptive minimum separation derived from expected d-spacing
- Nearest-neighbor lattice validation against expected d-spacing

---

## What's New (v2.0)

### E1: Enhanced Output & Presentation
- **Original image preserved** -- `0_Original.png` saved before any processing
- **Scale bars** on all output images (using matplotlib-scalebar)
- **Smart folder naming** -- Output folder named after input file (`outputs/<filename>/`)

### E2: Interactive Threshold Tuning
- **`--interactive-threshold`** -- Iteratively adjust threshold until satisfied
- Visual feedback with detection rate guidance (too low/high/reasonable)
- Iteration count tracked in `parameters.json`

### E3: Robust Peak Detection
- **`--peak-method percentile95`** (new default) -- Uses mean of top 5% intensities
- More robust for polycrystalline samples with diffuse ring patterns
- Weighted centroid for accurate peak localization
- Legacy mode: `--peak-method max` for single-pixel maximum

### E4: Multi-Plane Discrimination
- **`--multi-plane`** -- Analyze multiple lattice planes simultaneously
- **`--interactive`** -- Select which discovered peaks to analyze
- **`--max-planes N`** -- Limit number of planes (default: 5)
- Color-coded composite maps showing dominant plane per region
- Per-plane orientation maps (`5a_Orientation_Plane0.png`, `5b_...`, etc.)

---

## Features

- **Automatic Peak Discovery** -- Finds crystalline diffraction peaks without requiring d-spacing input
- **Background-Subtracted Radial Profile** -- Reveals true peaks above noise floor
- **Adaptive Threshold Calibration** -- Sets threshold based on actual peak intensity
- **Two-Tier SNR Gating** -- Separates high-confidence from weak-evidence tiles
- **GPA Strain Mapping** -- Full-image or region-wise strain field extraction
- **13-Gate Validation** -- Comprehensive quality control from Nyquist guard to output, all thresholds configurable
- **Per-Ring Spatial Maps** -- Presence, peak count, orientation, and SNR maps per diffraction ring
- **Domain Clustering** -- K-means, GMM, or HDBSCAN with auto-K selection and spatial regularization
- **Spatial Coherence Validation** -- Detects when results are noise vs. real domains
- **Orientation Mapping** -- Color-coded visualization of crystal orientations
- **Domain Metrics** -- Area, perimeter, circularity, orientation statistics
- **DM4 Support** -- Native reading of Gatan Digital Micrograph files
- **Scale Bars** -- Calibrated scale bars on all output images
- **Multi-Plane Analysis** -- Simultaneous analysis of multiple lattice planes

## Installation

### Requirements

- Python 3.10+
- NumPy, SciPy, scikit-image, scikit-learn
- Matplotlib, matplotlib-scalebar
- HyperSpy, ncempy (for DM4 file reading)

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
pip install -r requirements.txt
```

## Quick Start

### Hybrid Pipeline (Recommended for new work)

```bash
# Run the full hybrid pipeline with automatic peak discovery
python analyze.py sample.dm4 --hybrid --auto-discover -o results/

# With GPA strain analysis in full-image mode
python analyze.py sample.dm4 --hybrid --auto-discover --gpa-mode full

# Region-wise GPA for polycrystalline samples
python analyze.py sample.dm4 --hybrid --auto-discover --gpa-mode region

# Skip GPA (FFT analysis + peak finding only)
python analyze.py sample.dm4 --hybrid --auto-discover --no-gpa

# Custom SNR thresholds
python analyze.py sample.dm4 --hybrid --auto-discover --snr-tier-a 4.0 --snr-tier-b 2.5

# EFTEM bright-field with physics constraints
python analyze.py sample.dm4 --hybrid --auto-discover --physics-d-min 0.4 --physics-d-max 1.5 --imaging-mode EFTEM-BF

# Domain clustering with auto-K selection
python analyze.py sample.dm4 --hybrid --auto-discover --cluster

# Clustering with GMM and fixed 3 clusters
python analyze.py sample.dm4 --hybrid --auto-discover --cluster --cluster-method gmm --cluster-n 3

# HDBSCAN clustering with UMAP visualization
python analyze.py sample.dm4 --hybrid --auto-discover --cluster --cluster-method hdbscan --cluster-dimred umap

# Custom gate thresholds and parameters via YAML config
python analyze.py sample.dm4 --hybrid --auto-discover --config custom_config.yaml

# Override strong guidance SNR threshold from CLI
python analyze.py sample.dm4 --hybrid --auto-discover --strong-guidance-snr 10.0
```

### Classic Auto-Discovery Mode

**Don't know the d-spacing? Use `--auto-discover`:**

```bash
# Auto-discover peaks and validate results
python analyze.py sample.dm4 --auto-discover --validate

# Non-interactive (for scripts/automation)
python analyze.py sample.dm4 --auto-discover --validate --no-interactive -o results/
```

The auto-discovery will:
1. Compute background-subtracted radial profile
2. Find diffraction peaks using prominence detection
3. Recommend optimal d-spacing range and threshold
4. Validate results for spatial coherence

### Manual Mode

If you know the material's d-spacing:

```bash
# Specify parameters directly
python analyze.py sample.dm4 \
    --d-min 0.39 \
    --d-max 0.43 \
    --threshold 45000 \
    -o results/

# Interactive mode - prompts for missing parameters
python analyze.py sample.dm4
```

### Full CLI Reference

```bash
python analyze.py <input> [options]

Positional:
  input                   Input file (DM4, DM3, TIFF, or NPY)

Core Options:
  -o, --output DIR        Output directory (default: outputs/<input_filename>/)
  --pixel-size FLOAT      Pixel size in nm/pixel (auto-detected from DM4)
  --d-min FLOAT           Minimum d-spacing in nm
  --d-max FLOAT           Maximum d-spacing in nm
  --threshold FLOAT       Peak detection intensity threshold
  --tile-size INT         FFT tile size in pixels (default: 256)
  --stride INT            Tile stride in pixels (default: tile_size/2)
  --auto-discover         Automatically discover diffraction peaks
  --validate              Validate results with coherence analysis
  --no-interactive        Fail instead of prompting for missing params
  --save-preprocessed     Save preprocessed image as NPY
  -v, --verbose           Verbose output
  --help                  Show all available options

Hybrid Pipeline:
  --hybrid                Run the hybrid FFT + GPA + Peak-Finding pipeline
  --config PATH           YAML config file for hybrid pipeline parameters
  --gpa-mode {auto,full,region}
                          GPA execution mode (default: auto)
  --gpa-on-fail {fallback_to_region,skip,error}
                          GPA failure behavior (default: fallback_to_region)
  --snr-tier-a FLOAT      Tier A SNR threshold (default: 5.0)
  --snr-tier-b FLOAT      Tier B SNR threshold (default: 3.0)
  --no-gpa                Skip GPA stage entirely
  --no-peak-finding       Skip peak-finding stage
  --report-format {json,html,both}
                          Report output format (default: json)
  --strong-guidance-snr FLOAT
                          SNR threshold for "strong" FFT guidance (default: 8.0)
  --dynamic-dc            Enable derivative-based DC center masking
  --dc-floor FLOAT        Override minimum DC mask radius (cycles/nm)
  --dc-soft-taper         Use cosine taper instead of hard DC mask
  --bg-method {polynomial_robust,asls}
                          Background fitting method (default: polynomial_robust)
  --snr-signal-method {max,integrated_sum,integrated_median}
                          Peak signal aggregation method (default: max)

Physics & EFTEM:
  --physics-d-min FLOAT   Expected minimum d-spacing in nm (default: 0.4)
  --physics-d-max FLOAT   Expected maximum d-spacing in nm (default: 1.5)
  --imaging-mode STR      Imaging mode label (default: EFTEM-BF)

Domain Clustering:
  --cluster               Enable domain clustering (opt-in)
  --cluster-method {kmeans,gmm,hdbscan}
                          Clustering algorithm (default: kmeans)
  --cluster-n N           Number of clusters, 0=auto (default: 0)
  --cluster-dimred {pca,umap,none}
                          Dimensionality reduction method (default: pca)

Peak Detection (E3):
  --peak-method {max,percentile95}
                          Peak intensity method (default: percentile95)
                          - max: Single brightest pixel
                          - percentile95: Mean of top 5% (more robust)

Interactive Threshold (E2):
  --interactive-threshold
                          Enable iterative threshold adjustment
                          Shows results, allows tweaking until satisfied

Multi-Plane Analysis (E4):
  --multi-plane           Analyze all discovered peaks simultaneously
                          (requires --auto-discover)
  --interactive           Interactive mode: select which peaks to analyze
  --max-planes N          Maximum planes to analyze (default: 5)
```

### Getting Help

```bash
# Show all available options
python analyze.py --help
```

### Usage Examples

```bash
# Hybrid pipeline (recommended)
python analyze.py sample.dm4 --hybrid --auto-discover

# Hybrid with GPA strain analysis
python analyze.py sample.dm4 --hybrid --auto-discover --gpa-mode auto

# Classic auto-discovery
python analyze.py sample.dm4 --auto-discover

# With interactive threshold tuning
python analyze.py sample.dm4 --auto-discover --interactive-threshold

# Multi-plane analysis (all discovered peaks)
python analyze.py sample.dm4 --auto-discover --multi-plane

# Interactive plane selection
python analyze.py sample.dm4 --auto-discover --multi-plane --interactive

# Non-interactive batch mode
python analyze.py *.dm4 --hybrid --auto-discover --no-interactive

# Domain clustering (auto-K, K-means)
python analyze.py sample.dm4 --hybrid --auto-discover --cluster

# Clustering with HDBSCAN
python analyze.py sample.dm4 --hybrid --auto-discover --cluster --cluster-method hdbscan

# Custom config from YAML (gate thresholds, SNR params, etc.)
python analyze.py sample.dm4 --hybrid --auto-discover --config my_config.yaml
```

## Output Files

Output is saved to `outputs/<input_filename>/` by default.

### Classic Pipeline Outputs

| File | Description |
|------|-------------|
| `0_Original.png` | Original image before processing (with scale bar) |
| `original.npy` | Original image data (NumPy format) |
| `1_Peak_Discovery.png` | Background-subtracted radial profile with detected peaks *(auto-discover only)* |
| `2_Radial_Profile.png` | FFT radial intensity profile with highlighted q-range |
| `3_FFT_Power_Spectrum.png` | 2D FFT power spectrum visualization |
| `4_Peak_Location_Map.png` | Spatial map of detected crystalline peaks (green overlay, with scale bar) |
| `5_Orientation_Map.png` | Color-coded crystal orientation map (-180 to +180 degrees, with scale bar) |
| `5a_Orientation_Plane0.png` | Per-plane orientation map *(multi-plane mode only)* |
| `6_Multi_Plane_Composite.png` | Color-coded composite showing dominant plane per region *(multi-plane mode only)* |
| `Heatmap.png` | Peak intensity heatmap across tile grid |
| `parameters.json` | All analysis parameters, validation results, and plane metadata |

### Hybrid Pipeline Outputs (`--hybrid`)

| File | Description |
|------|-------------|
| `parameters.json` | v3.0 schema with FFT convention, tier summary, GPA decision, all gate results |
| `report.json` | Gate results and diagnostics |
| `global_g_vectors.json` | Extracted g-vectors with per-vector SNR, FWHM, angle, d-spacing |
| `tier_map.npy` | Tile classification map (0=skip, 1=rejected, 2=Tier B, 3=Tier A) |
| `gpa_mode_decision.json` | GPA mode selection metrics and confidence *(when GPA enabled)* |
| `gpa_reference.json` | Reference region details *(when GPA enabled)* |
| `lattice_peaks.npy` | Subpixel lattice-feature positions (N x 5: x, y, intensity, sigma_x, sigma_y) *(when peak-finding enabled)* |
| `peak_stats.json` | NN-distance statistics and lattice validation *(when peak-finding enabled)* |
| `ring_maps.npz` | Per-ring spatial maps: presence, peak count, orientation, SNR, angular variance *(when global peaks exist)* |
| `ring_feature_vectors.npy` | Feature matrix (n_tiles x n_features) for clustering *(when global peaks exist)* |
| `ring_feature_names.json` | Feature name list for the feature matrix *(when global peaks exist)* |
| `average_tile_fft.npy` | Mean power spectrum across all valid tiles *(when global peaks exist)* |
| `average_tile_radial_profile.json` | Radial profile from tile-averaged FFT *(when global peaks exist)* |
| `cluster_labels.npy` | Tile-level cluster assignments *(when `--cluster` enabled)* |
| `cluster_labels_regularized.npy` | Spatially regularized cluster labels *(when `--cluster` enabled)* |
| `cluster_averaged_ffts.npz` | Per-cluster mean power spectra *(when `--cluster` enabled)* |
| `clustering_results.json` | Method, n_clusters, silhouette score, per-cluster summaries *(when `--cluster` enabled)* |

## How Auto-Discovery Works

The `--auto-discover` flag enables automatic parameter selection:

```mermaid
flowchart LR
    subgraph Profile["Radial Profile"]
        A[Sample 200 tiles] --> B[Average FFT]
        B --> C[Fit polynomial<br/>background]
        C --> D[Subtract background]
    end

    subgraph Peaks["Peak Finding"]
        D --> E[scipy.find_peaks<br/>with prominence]
        E --> F[Rank by SNR]
        F --> G[Select best peak]
    end

    subgraph Threshold["Calibration"]
        G --> H[Sample tiles at<br/>peak q-range]
        H --> I[Set threshold for<br/>~15% detection]
    end

    I --> J[Run Analysis]
```

### Why Background Subtraction?

Raw FFT profiles show a strong power-law decay that masks diffraction peaks:

| Without Background Subtraction | With Background Subtraction |
|-------------------------------|----------------------------|
| Monotonic decay | Clear peaks visible |
| Peaks hidden in noise | SNR quantifiable |
| Manual d-spacing required | Automatic peak finding |

## Quality Gates (Hybrid Pipeline)

The hybrid pipeline evaluates 13 quality gates (G0-G12) at each stage. All thresholds are configurable via `GateThresholdsConfig` in YAML or Python API (defaults shown below):

| Gate | Metric | Default Threshold | On Failure |
|------|--------|-------------------|------------|
| **G0** | Nyquist guard (d_min vs sampling) | q_max < 0.95 × q_nyquist | DEGRADE (auto-clamp) / FATAL |
| **G1** | Input validity (2D, no NaN, >=512x512) | pass/fail | FATAL |
| **G2** | Preprocessing quality (clipped fraction, range ratio) | <0.5%, >10 | DEGRADE_CONFIDENCE |
| **G3** | ROI geometry (coverage, LCC fraction, fragments) | 10-95%, LCC>=0.5, <=20 | FALLBACK |
| **G4** | Global FFT viability (best peak SNR) | >=3.0 | FALLBACK |
| **G5** | Tiling adequacy (periods per tile, d-aware) | >=20 | FATAL |
| **G6** | Tier A detection rate | >=5% of ROI tiles | DEGRADE_CONFIDENCE |
| **G7** | Tier A SNR quality (median) | >=5.0 | DEGRADE_CONFIDENCE |
| **G8** | Pair fraction quality (mean score) | >=0.3 | DEGRADE_CONFIDENCE |
| **G9** | Reference region quality | area>=9, entropy<=0.3, SNR>=5.0 | SKIP_STAGE |
| **G10** | GPA phase noise + ref-region unwrap | sigma<=0.3 rad, ref unwrap>=70% | SKIP_STAGE |
| **G11** | GPA strain sanity | ref strain<=0.005, outliers<=20% | SKIP_STAGE |
| **G12** | Peak lattice consistency | >=50% valid NN distances | DEGRADE_CONFIDENCE |

**Failure behaviors:**
- **FATAL**: Pipeline halts immediately
- **SKIP_STAGE**: Stage skipped, downstream stages that depend on it also skipped
- **DEGRADE_CONFIDENCE**: Output marked as degraded, pipeline continues
- **FALLBACK**: Use fallback path (e.g., full-image mask, user-specified d-ranges)

## Validation Metrics (Classic Pipeline)

When using `--validate`, the tool computes:

| Metric | Good Value | Meaning |
|--------|------------|---------|
| **Coherence Score** | > 0.5 | Combined spatial + orientation coherence |
| **Local Coherence** | > 0.5 | Neighboring tiles have similar orientations |
| **Orientation Entropy** | < 0.5 | Orientations cluster (not random) |
| **Detection Rate** | 5-30% | Reasonable crystalline fraction |

### Interpreting Results

| Interpretation | What It Means | Action |
|----------------|---------------|--------|
| `good_domains` | Real crystalline domains detected | Results valid |
| `weak_domains` | Some structure, possibly noisy | Consider adjusting threshold |
| `likely_noise_high_detection` | Too many detections, random orientations | Wrong q-range or threshold too low |
| `sparse_or_noise` | Very few detections | Threshold too high or no crystallinity |

## Parameters Reference

### FFT Convention

All spatial frequencies are in **cycles/nm** (not radians/nm). The d-spacing relation is `d = 1/|g|` with no 2pi factor.

```
q (cycles/nm) = 1 / d (nm)
q_scale = 1 / (N * pixel_size_nm)    cycles/nm per FFT pixel
```

| Material Type | d-spacing (nm) | q (cycles/nm) |
|--------------|----------------|---------------|
| Metals | 0.2 - 0.4 | 2.5 - 5.0 |
| Oxides | 0.3 - 0.6 | 1.7 - 3.3 |
| Organics | 0.5 - 1.5 | 0.7 - 2.0 |
| Proteins | 1.0 - 5.0 | 0.2 - 1.0 |

### Radial Analysis

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_range` | Auto or `(2.3, 2.6)` | Target q-range in cycles/nm |
| `intensity_threshold` | Auto or `45000` | Minimum peak intensity |
| `tile_size` | `256` | FFT window size in pixels |
| `stride` | `128` | Step between tiles |
| `window` | `hann` | Windowing function for FFT |

### Hybrid Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `physics.d_min_nm` | 0.4 | Expected minimum d-spacing (nm) |
| `physics.d_max_nm` | 1.5 | Expected maximum d-spacing (nm) |
| `physics.imaging_mode` | EFTEM-BF | Imaging mode label |
| `physics.nyquist_safety_margin` | 0.95 | Clamp q_max to this fraction of q_nyquist |
| `tier_a_snr` | 5.0 | High-confidence SNR threshold |
| `tier_b_snr` | 3.0 | Weak-evidence SNR threshold |
| `tile_fft.q_dc_min` | 0.25 | DC suppression cutoff (cycles/nm) |
| `tile_fft.peak_snr_threshold` | 2.5 | SNR-first tile peak detection threshold |
| `global_fft.q_fit_min` | 0.30 | Background fit low-q exclusion (cycles/nm) |
| `global_fft.background_default_degree` | 3 | Polynomial degree (max 4) |
| `gpa.mode` | auto | GPA mode: auto, full, or region |
| `gpa.displacement_smooth_sigma` | 2.0 px | Gaussian smoothing before strain gradients |
| `gpa.amplitude_threshold` | 0.1 | Fraction of max amplitude for phase mask |
| `gpa.max_phase_noise` | 0.3 rad | G10 phase noise threshold |
| `peak_finding.min_separation_factor` | 0.6 | min_sep = factor * d_expected_px |
| `peak_finding.bandpass_bandwidth` | 0.3 | Bandwidth as fraction of \|g_dom\| |
| `peak_finding.angular_width_deg` | 30.0 | Half-width of directional wedge mask |
| `roi.min_lcc_fraction` | 0.5 | LCC fraction threshold for G3 |
| `clustering.enabled` | false | Enable domain clustering |
| `clustering.method` | kmeans | Clustering algorithm: kmeans, gmm, hdbscan |
| `clustering.n_clusters` | 0 | Number of clusters (0 = auto via silhouette scan) |
| `clustering.n_clusters_max` | 10 | Maximum K for auto-K scan |
| `clustering.dimred_method` | pca | Dimensionality reduction: pca, umap, none |
| `clustering.pca_variance_threshold` | 0.95 | Cumulative variance to retain in PCA |
| `clustering.regularize` | true | Apply spatial regularization to labels |
| `clustering.min_domain_size` | 5 | Minimum domain size (tiles) |
| `clustering.kmeans_n_init` | 10 | Number of K-means initializations |
| `clustering.umap_n_neighbors` | 15 | UMAP n_neighbors parameter |
| `clustering.umap_min_dist` | 0.1 | UMAP min_dist parameter |

#### Gate Thresholds (`gate_thresholds.*`)

All gate pass/fail thresholds can be overridden via `GateThresholdsConfig`:

| Parameter | Default | Gate |
|-----------|---------|------|
| `gate_thresholds.g0_nyquist_safety_margin` | 0.95 | G0 |
| `gate_thresholds.g2_max_clipped_fraction` | 0.005 | G2 |
| `gate_thresholds.g2_min_range_ratio` | 10.0 | G2 |
| `gate_thresholds.g3_min_coverage` | 10.0 | G3 |
| `gate_thresholds.g3_max_coverage` | 95.0 | G3 |
| `gate_thresholds.g3_max_fragments` | 20 | G3 |
| `gate_thresholds.g3_min_lcc_fraction` | 0.5 | G3 |
| `gate_thresholds.g4_min_peak_snr` | 3.0 | G4 |
| `gate_thresholds.g5_min_periods` | 20.0 | G5 |
| `gate_thresholds.g6_min_fraction` | 0.05 | G6 |
| `gate_thresholds.g7_min_median_snr` | 5.0 | G7 |
| `gate_thresholds.g8_min_mean_symmetry` | 0.3 | G8 |
| `gate_thresholds.g9_min_area` | 9 | G9 |
| `gate_thresholds.g9_max_entropy` | 0.3 | G9 |
| `gate_thresholds.g9_min_snr` | 5.0 | G9 |
| `gate_thresholds.g10_max_phase_noise` | 0.3 | G10 |
| `gate_thresholds.g10_min_unwrap_success` | 0.7 | G10 |
| `gate_thresholds.g11_max_ref_strain` | 0.005 | G11 |
| `gate_thresholds.g11_max_outlier_fraction` | 0.20 | G11 |
| `gate_thresholds.g11_strain_outlier_threshold` | 0.05 | G11 |
| `gate_thresholds.g12_min_fraction_valid` | 0.50 | G12 |
| `gate_thresholds.g12_tolerance` | 0.20 | G12 |

#### Peak SNR Measurement (`peak_snr.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `peak_snr.signal_disk_radius_px` | 3 | Pixel radius of signal measurement disk |
| `peak_snr.annular_width_min_q` | 0.15 | Minimum annular background width (cycles/nm) |
| `peak_snr.annular_fwhm_multiplier` | 1.5 | Multiplier for FWHM-based annular width |
| `peak_snr.min_background_pixels` | 20 | Minimum pixels for valid background estimate |
| `peak_snr.fwhm_patch_radius` | 5 | Patch radius for FWHM measurement (px) |
| `peak_snr.moment_sigma_floor` | 0.3 | Minimum sigma for moment-based FWHM proxy |
| `peak_snr.fit_condition_max` | 100.0 | Maximum condition number for curve_fit |
| `peak_snr.symmetry_tolerance_px` | 2.0 | Tolerance for antipodal peak pairing (px) |
| `peak_snr.non_collinear_min_angle_deg` | 15.0 | Minimum angle between non-collinear peaks |

#### Reference Selection (`reference_selection.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reference_selection.scoring_weight_entropy` | 0.4 | Weight for orientation entropy in scoring |
| `reference_selection.scoring_weight_snr` | 0.3 | Weight for mean SNR in scoring |
| `reference_selection.scoring_weight_area` | 0.3 | Weight for region area in scoring |
| `reference_selection.orientation_bins` | 12 | Number of bins for orientation histogram |

#### Ring Analysis (`ring_analysis.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ring_analysis.ring_width_fwhm_mult` | 2.0 | Ring width = mult * FWHM |
| `ring_analysis.ring_width_fallback_frac` | 0.03 | Fallback ring width as fraction of q_center |
| `ring_analysis.ring_width_no_fwhm_frac` | 0.1 | Ring width when no FWHM available |

#### DC Mask (`dc_mask.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dc_mask.enabled` | false | Enable dynamic DC masking (opt-in) |
| `dc_mask.method` | derivative | Estimation method: derivative or fixed |
| `dc_mask.savgol_window` | 11 | Savitzky-Golay smoothing window (auto-clamped, odd) |
| `dc_mask.savgol_polyorder` | 2 | Savitzky-Golay polynomial order |
| `dc_mask.slope_threshold_k` | 2.5 | \|dJ/dq\| < k * sigma_noise for DC boundary |
| `dc_mask.consecutive_bins` | 7 | Required consecutive bins below threshold |
| `dc_mask.noise_q_range_lo` | 0.70 | Noise region lower bound (fraction of q_max) |
| `dc_mask.noise_q_range_hi` | 0.90 | Noise region upper bound (fraction of q_max) |
| `dc_mask.q_dc_min_floor` | 0.15 | Absolute minimum DC mask (cycles/nm) |
| `dc_mask.soft_taper` | false | Cosine taper vs hard mask |
| `dc_mask.taper_width_q` | 0.05 | Taper transition width (cycles/nm) |
| `dc_mask.max_dc_mask_q` | 0.0 | Hard upper bound (0 = uncapped) |
| `dc_mask.auto_cap_from_physics` | true | Derive cap from d_max_nm |

## Python API

### Classic Pipeline

```python
from src.io_dm4 import load_dm4
from src.preprocess import preprocess
from src.peak_discovery import discover_peaks, get_recommended_params
from src.radial_analysis import run_radial_analysis

# Load and preprocess
record = load_dm4('sample.dm4')
processed = preprocess(record.image)

# Auto-discover peaks
discovery = discover_peaks(
    processed,
    record.pixel_size_nm,
    tile_size=256,
    verbose=True
)

# Get recommended parameters
params = get_recommended_params(discovery)
if params:
    print(f"Recommended d-spacing: {params['d_min']:.3f} - {params['d_max']:.3f} nm")
    print(f"Recommended threshold: {params['intensity_threshold']:.0f}")

# Run analysis with discovered parameters
results = run_radial_analysis(
    processed,
    pixel_size_nm=record.pixel_size_nm,
    output_dir='outputs',
    params={
        'q_range': (params['q_min'], params['q_max']),
        'intensity_threshold': params['intensity_threshold'],
    },
    verbose=True
)
```

### Hybrid Pipeline

```python
from src.fft_coords import FFTGrid
from src.pipeline_config import PipelineConfig
from src.preprocess_fft_safe import preprocess_fft_safe
from src.preprocess_segmentation import preprocess_segmentation
from src.roi_masking import compute_roi_mask, downsample_to_tile_grid
from src.global_fft import compute_global_fft
from src.tile_fft import process_all_tiles
from src.fft_snr_metrics import build_gated_tile_grid
from src.gpa import run_gpa
from src.peak_finding import build_bandpass_image, find_subpixel_peaks, validate_peak_lattice
from src.ring_analysis import build_ring_maps, build_ring_feature_vectors, compute_tile_averaged_fft
from src.domain_clustering import run_domain_clustering
from src.validation import validate_pipeline
from src.reporting import save_pipeline_artifacts

# Configure (all parameters customizable via YAML or Python)
config = PipelineConfig(pixel_size_nm=0.1297)

# Optional: customize gate thresholds
config.gate_thresholds.g5_min_periods = 15.0    # relax tiling requirement
config.gate_thresholds.g9_min_area = 4           # smaller reference region OK

# Optional: customize peak SNR measurement
config.peak_snr.signal_disk_radius_px = 5

# Or load from YAML:
# config = PipelineConfig.from_dict(yaml.safe_load(open("config.yaml")))

fft_grid = FFTGrid(height, width, config.pixel_size_nm)

# Branch A: FFT-safe preprocessing (no blur)
preproc = preprocess_fft_safe(image, config.preprocessing)

# Branch B: Segmentation preprocessing
seg = preprocess_segmentation(image, config.segmentation)

# Early ROI mask
roi = compute_roi_mask(seg.image_seg, config.roi)
roi_grid = downsample_to_tile_grid(roi.mask_full, config.tile_size, config.stride)

# Global FFT: radial profile, g-vector extraction
global_result = compute_global_fft(preproc.image_fft, fft_grid, config.global_fft)

# Tile FFT with two-tier SNR classification
peak_sets, skipped = process_all_tiles(
    preproc.image_fft, roi_grid, fft_grid,
    tile_size=config.tile_size, stride=config.stride,
)
tile_grid = FFTGrid(config.tile_size, config.tile_size, config.pixel_size_nm)
gated = build_gated_tile_grid(peak_sets, skipped, tile_grid, config.tile_size,
                               peak_snr_config=config.peak_snr)

# Ring analysis (always-on when global peaks exist)
ring_maps = build_ring_maps(gated, global_result.peaks,
                             ring_config=config.ring_analysis)
ring_features = build_ring_feature_vectors(gated, ring_maps)
tile_avg_fft = compute_tile_averaged_fft(
    preproc.image_fft, config.tile_size, config.stride,
    config.pixel_size_nm, gated.skipped_mask)

# Domain clustering (opt-in)
if config.clustering.enabled:
    clustering_result = run_domain_clustering(ring_features, config.clustering)

# Validation (with configurable gate thresholds)
report = validate_pipeline(
    preproc_record=preproc, roi_result=roi,
    global_fft_result=global_result, gated_grid=gated,
    gate_thresholds=config.gate_thresholds,
)

# Save all artifacts
save_pipeline_artifacts(output_dir, config=config, fft_grid=fft_grid,
                        preproc_record=preproc, validation_report=report, ...)
```

## Module Reference

### Hybrid Pipeline Modules

| Module | Purpose |
|--------|---------|
| `src/fft_coords.py` | Canonical FFT coordinate system (`FFTGrid` class) |
| `src/pipeline_config.py` | All parameter dataclasses, config classes, defaults, and YAML serialization |
| `src/gates.py` | Gate registry G0-G12 with configurable thresholds and failure behaviors |
| `src/preprocess_fft_safe.py` | Branch A: hot-pixel removal + robust normalize, no blur |
| `src/preprocess_segmentation.py` | Branch B: clip + normalize + Gaussian smooth |
| `src/roi_masking.py` | Early ROI mask (intensity + variance, geometric gates) |
| `src/global_fft.py` | Full-image FFT, radial profile, g-vector extraction |
| `src/tile_fft.py` | Tile-based FFT with ROI-aware processing |
| `src/fft_peak_detection.py` | Per-peak SNR, FWHM, symmetry, two-tier classification |
| `src/fft_snr_metrics.py` | Aggregate SNR/symmetry/FWHM maps, tier classification |
| `src/reference_selection.py` | GPA reference region selection (Tier A tiles only) |
| `src/gpa.py` | GPA engine: full-image and region-wise modes, strain fields |
| `src/peak_finding.py` | Subpixel peak detection with lattice validation |
| `src/validation.py` | Unified 13-gate evaluation (G0-G12) and `ValidationReport` |
| `src/ring_analysis.py` | Per-ring spatial maps, feature vectors, tile/cluster-averaged FFTs |
| `src/domain_clustering.py` | Domain clustering: K-means/GMM/HDBSCAN, PCA/UMAP, regularization |
| `src/reporting.py` | JSON/artifact output, parameters.json v3.0 |
| `src/hybrid_viz.py` | PNG visualization orchestrator (38 artifacts) |

### Legacy Modules

| Module | Purpose |
|--------|---------|
| `src/io_dm4.py` | DM4/DM3 file loading with metadata extraction |
| `src/preprocess.py` | Image normalization, denoising, background subtraction |
| `src/peak_discovery.py` | Automatic peak finding and threshold calibration |
| `src/radial_analysis.py` | FFT analysis, peak detection, orientation mapping |
| `src/fft_features.py` | Feature extraction for ML pipelines |
| `src/cluster_domains.py` | HDBSCAN clustering and spatial regularization |
| `src/domain_metrics.py` | Per-domain statistics and quality gates |
| `src/viz.py` | Visualization utilities |
| `src/ilastik_roi.py` | ROI export for Ilastik workflows |

## Example Workflow

```bash
# 1. First run: hybrid pipeline with auto-discovery
python analyze.py myimage.dm4 --hybrid --auto-discover -o test_run/

# 2. Review parameters.json for gate results and tier summary
# 3. Check report.json for detailed diagnostics

# 4. If GPA strain maps are needed:
python analyze.py myimage.dm4 --hybrid --auto-discover --gpa-mode full -o strain_run/

# 5. Domain clustering with auto-K:
python analyze.py myimage.dm4 --hybrid --auto-discover --cluster -o cluster_run/

# 6. Clustering with fixed 3 clusters and GMM:
python analyze.py myimage.dm4 --hybrid --auto-discover --cluster --cluster-method gmm --cluster-n 3

# 7. For batch processing:
python analyze.py *.dm4 --hybrid --auto-discover --no-interactive --no-gpa
```

## Troubleshooting

### "No significant peaks found"
- Image may be amorphous (no crystalline regions)
- Try preprocessing with different parameters
- Check if pixel size is correct

### High detection rate with random orientations
- Threshold too low -- increase `--threshold`
- Wrong q-range -- check material's expected d-spacing
- Use `--auto-discover` to find correct parameters

### Low detection rate
- Threshold too high -- decrease `--threshold`
- Peaks may be at different q-range
- Sample may have weak crystallinity

### Hybrid pipeline: G2 gate fails (degraded confidence)
- Image has extreme clipping or very narrow dynamic range
- Pipeline continues with min-max normalization fallback
- Check `parameters.json` for `preprocessing.confidence` field

### Hybrid pipeline: GPA stage skipped
- Reference region too small (G9) or phase noise too high (G10)
- Try `--gpa-mode region` for polycrystalline samples
- Check `gpa_mode_decision.json` for decision metrics

## Citation

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
