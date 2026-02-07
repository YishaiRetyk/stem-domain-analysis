"""
Pipeline Configuration and Parameter Dataclasses.

All parameter dataclasses, defaults, validation, and serialisation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


# ======================================================================
# Core data containers
# ======================================================================

@dataclass
class GVector:
    """A reciprocal-lattice vector (g-vector) in cycles/nm."""
    gx: float               # cycles/nm
    gy: float               # cycles/nm
    magnitude: float         # |g| cycles/nm
    angle_deg: float
    d_spacing: float         # 1/|g| nm
    snr: float
    fwhm: float              # cycles/nm (from 2D Gaussian fit)
    ring_index: int          # which radial peak ring this came from

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GlobalPeak:
    """A peak found in the global radial profile."""
    q_center: float          # cycles/nm
    q_fwhm: float            # cycles/nm
    d_spacing: float         # nm
    intensity: float
    prominence: float
    snr: float
    index: int               # index in the radial profile array


# ======================================================================
# Preprocessing records
# ======================================================================

@dataclass
class FFTPreprocRecord:
    """Branch A: FFT-safe preprocessed image."""
    image_fft: np.ndarray           # float64, robust-normalised, no blur
    diagnostics: dict
    qc_metrics: dict
    confidence: str = "normal"      # 'normal' or 'degraded'


@dataclass
class SegPreprocRecord:
    """Branch B: Segmentation preprocessed image."""
    image_seg: np.ndarray           # float32, [0,1], blurred
    diagnostics: dict


# ======================================================================
# ROI
# ======================================================================

@dataclass
class ROIMaskResult:
    """Result of early ROI masking."""
    mask_full: np.ndarray           # (H, W) uint8
    mask_grid: np.ndarray           # (n_rows, n_cols) bool
    coverage_pct: float
    n_components: int
    diagnostics: dict


# ======================================================================
# Tile FFT
# ======================================================================

@dataclass
class TilePeak:
    """A single peak detected in a tile FFT."""
    qx: float
    qy: float
    q_mag: float
    d_spacing: float
    angle_deg: float
    intensity: float
    fwhm: float              # measured FWHM in cycles/nm


@dataclass
class PeakSNR:
    """Peak-height SNR result (B3)."""
    signal_peak: float
    background_median: float
    background_mad_sigma: float
    snr: float
    n_background_px: int
    note: Optional[str] = None


@dataclass
class PeakFWHM:
    """FWHM measurement via 2D Gaussian fit (B2)."""
    fwhm_q: float = 0.0
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    theta: float = 0.0
    fwhm_valid: bool = False
    method: str = "failed"


@dataclass
class TilePeakSet:
    """All peaks from a single tile."""
    peaks: List[TilePeak]
    tile_row: int
    tile_col: int
    power_spectrum: Optional[np.ndarray] = None


@dataclass
class TileClassification:
    """Two-tier classification result for a tile."""
    tier: str                       # 'A', 'B', 'REJECTED'
    peaks: list                     # list of dicts with per-peak metrics
    symmetry_score: float
    n_non_collinear: int
    best_snr: float
    best_orientation_deg: float = 0.0
    gate_details: dict = field(default_factory=dict)


@dataclass
class TierSummary:
    """Aggregate tier statistics."""
    n_tier_a: int
    n_tier_b: int
    n_rejected: int
    n_skipped: int
    tier_a_fraction: float
    median_snr_tier_a: float


@dataclass
class GatedTileGrid:
    """Unified output for all downstream consumers (C12)."""
    classifications: np.ndarray     # (n_rows, n_cols) object array of TileClassification
    tier_map: np.ndarray            # (n_rows, n_cols) str: 'A'/'B'/'REJECTED'/''
    snr_map: np.ndarray
    symmetry_map: np.ndarray
    fwhm_map: np.ndarray
    orientation_map: np.ndarray
    grid_shape: tuple
    skipped_mask: np.ndarray        # (n_rows, n_cols) bool
    tier_summary: TierSummary


# ======================================================================
# Reference selection
# ======================================================================

@dataclass
class ReferenceRegion:
    """Selected reference region for GPA."""
    center_tile: Tuple[int, int]
    tiles: List[Tuple[int, int]]
    bounding_box: Tuple[int, int, int, int]  # row_min, row_max, col_min, col_max
    orientation_mean: float
    orientation_std: float
    mean_snr: float
    entropy: float
    score: float


# ======================================================================
# GPA
# ======================================================================

@dataclass
class GPAModeDecision:
    """Decision about GPA execution mode."""
    selected_mode: str              # 'full' or 'region'
    decision_metrics: dict
    thresholds_used: dict = field(default_factory=dict)
    reason: str = ""
    decision_confidence: float = 0.0
    ref_region_exists: bool = False
    orientation_is_bimodal: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GPAPhaseResult:
    """Phase extraction result for one g-vector."""
    phase_raw: np.ndarray           # (H, W) radians, wrapped
    phase_unwrapped: np.ndarray     # (H, W) radians, unwrapped
    amplitude: np.ndarray           # (H, W) float
    amplitude_mask: np.ndarray      # (H, W) bool (eroded)
    g_vector: GVector
    phase_noise_sigma: Optional[float] = None
    unwrap_success_fraction: float = 0.0


@dataclass
class DisplacementField:
    """Displacement field from GPA."""
    ux: np.ndarray                  # (H, W) nm
    uy: np.ndarray                  # (H, W) nm


@dataclass
class StrainField:
    """Strain field from GPA."""
    exx: np.ndarray                 # (H, W) dimensionless
    eyy: np.ndarray
    exy: np.ndarray
    rotation: np.ndarray            # (H, W) radians


@dataclass
class GPAResult:
    """Unified GPA result -- identical schema for full and region modes (C12)."""
    mode: str                       # 'full' or 'region'
    phases: Dict[str, GPAPhaseResult]
    displacement: Optional[DisplacementField]
    strain: Optional[StrainField]
    reference_region: Optional[ReferenceRegion]
    mode_decision: GPAModeDecision
    qc: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


# ======================================================================
# Peak finding
# ======================================================================

@dataclass
class SubpixelPeak:
    """A peak with subpixel localisation."""
    x: float
    y: float
    intensity: float
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    prominence: float = 0.0


@dataclass
class LatticeValidation:
    """Nearest-neighbour lattice validation."""
    nn_distances: np.ndarray
    fraction_valid: float
    mean_nn_distance_nm: float
    std_nn_distance_nm: float
    expected_d_nm: float
    min_separation_px_used: float


# ======================================================================
# Gates
# ======================================================================

@dataclass
class GateResult:
    """Result of evaluating a single gate."""
    gate_id: str
    name: str
    passed: bool
    value: Any = None
    threshold: Any = None
    failure_behavior: str = "FATAL"
    reason: str = ""


@dataclass
class ValidationReport:
    """Full pipeline validation report."""
    gates: Dict[str, GateResult]
    overall_pass: bool
    tier_summary: Optional[TierSummary] = None
    diagnostics: dict = field(default_factory=dict)
    summary: str = ""
    timestamp: str = ""


# ======================================================================
# Global FFT result
# ======================================================================

@dataclass
class GlobalFFTResult:
    """Result of global FFT analysis."""
    power_spectrum: np.ndarray
    radial_profile: np.ndarray
    q_values: np.ndarray
    background: np.ndarray
    corrected_profile: np.ndarray
    noise_floor: float
    peaks: List[GlobalPeak]
    g_vectors: List[GVector]
    d_dom: Optional[float]          # dominant d-spacing (nm)
    information_limit_q: Optional[float]
    diagnostics: dict = field(default_factory=dict)


# ======================================================================
# Configuration dataclasses
# ======================================================================

@dataclass
class PreprocConfig:
    """FFT-safe preprocessing configuration."""
    clip_percentile: float = 0.1
    normalize_method: str = "robust"        # 'robust' or 'minmax'
    hot_pixel_removal: bool = True          # I1
    hot_pixel_sigma: float = 5.0            # I1: outlier threshold in MAD units


@dataclass
class SegmentationPreprocConfig:
    """Segmentation preprocessing configuration."""
    clip_percentile: float = 0.1
    smooth_sigma: float = 0.5


@dataclass
class ROIConfig:
    """ROI masking configuration."""
    min_coverage_pct: float = 10.0
    max_coverage_pct: float = 95.0
    max_fragments: int = 20
    min_tile_coverage: float = 0.5
    intensity_threshold_pct: float = 10.0
    variance_threshold_pct: float = 20.0


@dataclass
class TierConfig:
    """Two-tier SNR configuration."""
    tier_a_snr: float = 5.0
    tier_b_snr: float = 3.0


@dataclass
class PeakGateConfig:
    """Peak quality gate thresholds."""
    max_fwhm_ratio: float = 0.15
    min_symmetry: float = 0.3
    min_non_collinear: int = 2


@dataclass
class GlobalFFTConfig:
    """Global FFT configuration."""
    max_image_size: int = 4096
    background_poly_degree: int = 6
    min_peak_snr: float = 3.0
    max_g_vectors: int = 6


@dataclass
class GPAConfig:
    """GPA configuration."""
    enabled: bool = True
    mode: str = "auto"                      # auto | full | region
    on_fail: str = "fallback_to_region"     # fallback_to_region | skip | error
    mask_radius_q: str = "auto"             # 'auto' or float
    max_phase_noise: float = 0.3
    amplitude_threshold: float = 0.1
    displacement_smooth_sigma: float = 2.0
    min_unwrap_success: float = 0.7
    ref_strain_tolerance: float = 0.005
    strain_outlier_threshold: float = 0.05
    max_strain_outlier_fraction: float = 0.20
    # Auto-selection thresholds
    auto_min_valid_tile_fraction: float = 0.3
    auto_max_orientation_entropy: float = 0.5
    auto_min_global_peak_snr: float = 5.0


@dataclass
class PeakFindingConfig:
    """Peak-finding configuration."""
    enabled: bool = True
    min_separation_factor: float = 0.6
    min_separation_px_floor: int = 2
    min_prominence: float = 0.1
    lattice_tolerance: float = 0.2
    bandpass_bandwidth: float = 0.3


@dataclass
class ValidationConfig:
    """Validation gate thresholds."""
    min_detection_rate: float = 0.05
    min_periods: float = 20.0
    min_tier_a_snr_median: float = 5.0
    min_symmetry_mean: float = 0.3
    ref_area_min: int = 9
    ref_entropy_max: float = 0.3
    ref_snr_min: float = 5.0
    peak_lattice_fraction_min: float = 0.50


@dataclass
class DeviceConfig:
    """GPU / compute-device configuration."""
    device: str = "auto"           # "auto" | "gpu" | "cpu"
    device_id: int = 0
    tile_batch_size: int = 0       # 0 = auto-size from GPU memory
    gpu_memory_fraction: float = 0.7


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    pixel_size_nm: float = 0.127
    tile_size: int = 256
    stride: int = 128
    preprocessing: PreprocConfig = field(default_factory=PreprocConfig)
    segmentation: SegmentationPreprocConfig = field(default_factory=SegmentationPreprocConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    tier: TierConfig = field(default_factory=TierConfig)
    peak_gates: PeakGateConfig = field(default_factory=PeakGateConfig)
    global_fft: GlobalFFTConfig = field(default_factory=GlobalFFTConfig)
    gpa: GPAConfig = field(default_factory=GPAConfig)
    peak_finding: PeakFindingConfig = field(default_factory=PeakFindingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Create from a flat or nested dict (e.g. YAML config)."""
        cfg = cls()
        # Top-level scalars
        for key in ("pixel_size_nm", "tile_size", "stride"):
            if key in d:
                setattr(cfg, key, d[key])
        # Sub-configs
        _mapping = {
            "preprocessing": (PreprocConfig, "preprocessing"),
            "segmentation": (SegmentationPreprocConfig, "segmentation"),
            "roi": (ROIConfig, "roi"),
            "tier": (TierConfig, "tier"),
            "peak_gates": (PeakGateConfig, "peak_gates"),
            "global_fft": (GlobalFFTConfig, "global_fft"),
            "gpa": (GPAConfig, "gpa"),
            "peak_finding": (PeakFindingConfig, "peak_finding"),
            "validation": (ValidationConfig, "validation"),
            "device": (DeviceConfig, "device"),
        }
        for attr, (klass, key) in _mapping.items():
            if key in d and isinstance(d[key], dict):
                sub = klass()
                for k, v in d[key].items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
                setattr(cfg, attr, sub)
        return cfg
