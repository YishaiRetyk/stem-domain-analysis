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
    lcc_fraction: float = 1.0       # largest connected component / total mask area


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
    ring_index: int = -1     # -1 = unassigned, >=0 = matched to global ring


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


import warnings as _warnings

_warned_symmetry_score = False


@dataclass
class TileClassification:
    """Two-tier classification result for a tile."""
    tier: str                       # 'A', 'B', 'REJECTED'
    peaks: list                     # list of dicts with per-peak metrics
    pair_fraction: float            # was symmetry_score; fraction of peaks that are ±g paired
    n_non_collinear: int
    best_snr: float
    best_orientation_deg: float = 0.0
    gate_details: dict = field(default_factory=dict)
    orientation_confidence: float = 0.0     # circular concentration R (diagnostic-only)

    @property
    def symmetry_score(self) -> float:
        """Deprecated alias for pair_fraction. Use pair_fraction instead."""
        global _warned_symmetry_score
        if not _warned_symmetry_score:
            _warnings.warn(
                "TileClassification.symmetry_score is deprecated, "
                "use pair_fraction instead",
                DeprecationWarning,
                stacklevel=2,
            )
            _warned_symmetry_score = True
        return self.pair_fraction


@dataclass
class TierSummary:
    """Aggregate tier statistics."""
    n_tier_a: int
    n_tier_b: int
    n_rejected: int
    n_skipped: int
    tier_a_fraction: float
    median_snr_tier_a: float


_warned_symmetry_map = False


@dataclass
class GatedTileGrid:
    """Unified output for all downstream consumers (C12)."""
    classifications: np.ndarray     # (n_rows, n_cols) object array of TileClassification
    tier_map: np.ndarray            # (n_rows, n_cols) str: 'A'/'B'/'REJECTED'/''
    snr_map: np.ndarray
    pair_fraction_map: np.ndarray   # was symmetry_map
    fwhm_map: np.ndarray
    orientation_map: np.ndarray
    grid_shape: tuple
    skipped_mask: np.ndarray        # (n_rows, n_cols) bool
    tier_summary: TierSummary
    orientation_confidence_map: Optional[np.ndarray] = None  # (n_rows, n_cols) float
    detection_confidence_map: Optional[np.ndarray] = None  # (n_rows, n_cols) float [0,1], diagnostic only

    @property
    def symmetry_map(self) -> np.ndarray:
        """Deprecated alias for pair_fraction_map. Use pair_fraction_map instead."""
        global _warned_symmetry_map
        if not _warned_symmetry_map:
            _warnings.warn(
                "GatedTileGrid.symmetry_map is deprecated, "
                "use pair_fraction_map instead",
                DeprecationWarning,
                stacklevel=2,
            )
            _warned_symmetry_map = True
        return self.pair_fraction_map


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
    unwrap_success_ref_fraction: float = 0.0  # computed within reference region only


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
    tolerance_used: float = 0.2             # actual tolerance used (may be adaptive)


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
    fft_guidance_strength: str = "none"     # "strong", "weak", or "none"


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
    hot_pixel_median_kernel: int = 3        # median filter kernel size for hot pixel removal
    robust_norm_clip_sigma: float = 5.0     # clip range in sigma units for robust norm


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
    min_lcc_fraction: float = 0.5           # largest connected component / total mask area
    gradient_threshold_pct: float = 30.0    # percentile for gradient-magnitude fallback
    variance_window_size: int = 32          # window size for local variance computation
    morph_kernel_size: int = 5              # morphological opening kernel size
    smooth_sigma: float = 2.0              # Gaussian smoothing sigma for mask edges
    gradient_smooth_sigma: float = 1.0      # Gaussian smoothing for gradient computation


@dataclass
class TierConfig:
    """Two-tier SNR configuration."""
    tier_a_snr: float = 5.0
    tier_b_snr: float = 3.0


@dataclass
class FWHMConfig:
    """FWHM measurement configuration."""
    enabled: bool = True
    method: str = "auto"           # auto | proxy_only | curve_fit
    maxfev: int = 500              # max iterations for curve_fit (was 2000)
    max_per_tile: int = 2          # max curve_fit calls per tile
    min_snr_for_fit: float = 5.0   # only curve_fit if peak SNR >= this


@dataclass
class ConfidenceConfig:
    """Detection confidence heatmap weights.

    Produces an ordinal score for visualization only — not consumed by
    gates, tier assignment, or any classification logic (DC-1, DC-2).
    Weights are normalized at runtime so they need not sum to 1.0.
    """
    enabled: bool = True
    w_snr: float = 0.40
    w_pair_fraction: float = 0.20
    w_orientation_confidence: float = 0.15
    w_non_collinearity: float = 0.10
    w_fwhm_quality: float = 0.15
    snr_ceiling_multiplier: float = 2.0


@dataclass
class PeakGateConfig:
    """Peak quality gate thresholds."""
    max_fwhm_ratio: float = 0.15
    min_pair_fraction: float = 0.3          # was min_symmetry
    min_non_collinear: int = 2

    @property
    def min_symmetry(self) -> float:
        """Deprecated alias for min_pair_fraction."""
        return self.min_pair_fraction

    @min_symmetry.setter
    def min_symmetry(self, value: float):
        """Deprecated setter for min_pair_fraction."""
        self.min_pair_fraction = value


@dataclass
class GlobalFFTConfig:
    """Global FFT configuration."""
    max_image_size: int = 4096
    background_poly_degree: int = 6
    min_peak_snr: float = 3.0
    max_g_vectors: int = 6
    q_fit_min: float = 0.30                 # cycles/nm; 0 = derive from PhysicsConfig.d_max_nm
    background_max_degree: int = 4          # hard cap on polynomial degree
    background_default_degree: int = 3      # used unless overridden
    strong_guidance_snr: float = 8.0        # SNR threshold for "strong" FFT guidance
    bg_reweight_iterations: int = 4         # iterative reweight cycles for background fit
    bg_reweight_downweight: float = 0.1     # weight for outlier bins in reweighted fit
    savgol_window_max: int = 11             # max Savitzky-Golay window for radial smoothing
    savgol_window_min: int = 5              # min Savitzky-Golay window
    savgol_polyorder: int = 2               # Savitzky-Golay polynomial order
    radial_peak_distance: int = 5           # min distance between radial peaks
    radial_peak_width: int = 2              # min width for radial peak detection
    q_width_expansion_frac: float = 0.03    # fractional q-width expansion for ring extraction
    harmonic_ratio_tol: float = 0.05        # tolerance for harmonic ratio matching
    harmonic_angle_tol_deg: float = 5.0     # angle tolerance for harmonic de-duplication
    harmonic_snr_ratio: float = 2.0         # SNR ratio to prefer harmonic over fundamental
    non_collinear_min_angle_deg: float = 15.0  # min angle between non-collinear g-vectors
    angular_prominence_frac: float = 0.5    # angular prominence as fraction of median
    angular_peak_distance: int = 10         # min distance between angular peaks


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
    # GPA mask sigma determination
    mask_sigma_fwhm_factor: float = 0.5     # sigma = factor * median(fwhm)
    mask_sigma_fallback_factor: float = 0.06  # sigma = factor * min(|g|) when no fwhm
    mask_sigma_dc_clamp_factor: float = 0.18  # DC safety clamp: sigma <= factor * min(|g|)
    # Orientation histogram for mode selection
    orientation_bins: int = 12
    bimodal_smooth_sigma: float = 0.5
    bimodal_peak_distance: int = 2
    bimodal_valley_ratio: float = 0.5
    # Phase extraction
    amplitude_erosion_iterations: int = 2
    phase_noise_min_pixels: int = 10
    min_tier_a_fraction_for_gpa: float = 0.1


@dataclass
class PeakFindingConfig:
    """Peak-finding configuration."""
    enabled: bool = True
    min_separation_factor: float = 0.6
    min_separation_px_floor: int = 2
    min_prominence: float = 0.1
    lattice_tolerance: float = 0.2
    bandpass_bandwidth: float = 0.3
    use_directional_mask: bool = True       # g-oriented wedge masks instead of full ring
    angular_width_deg: float = 30.0         # half-width of angular wedge per g-vector
    adaptive_tolerance: bool = True         # adaptive NN tolerance from IQR
    taper_width_fraction: float = 0.3       # cosine taper width as fraction of bandwidth
    background_percentile: float = 50.0     # percentile for local background estimation
    background_filter_size_mult: int = 2    # multiplier for background filter size
    adaptive_tolerance_floor: float = 0.1   # minimum adaptive tolerance


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
class VizConfig:
    """PNG visualization configuration."""
    enabled: bool = True
    dpi: int = 150


@dataclass
class PhysicsConfig:
    """Physical constraints and imaging mode configuration."""
    imaging_mode: str = "EFTEM-BF"          # informational, logged in report
    d_min_nm: float = 0.0                   # expected minimum d-spacing (nm); 0 = unconstrained
    d_max_nm: float = 0.0                   # expected maximum d-spacing (nm); 0 = unconstrained
    nyquist_safety_margin: float = 0.95     # clamp q_max to this fraction of q_nyquist


@dataclass
class TileFFTConfig:
    """Tile FFT peak detection configuration."""
    q_dc_min: float = 0.25                  # cycles/nm, replaces pixel-based DC mask
    peak_snr_threshold: float = 2.5         # replaces peak_threshold_frac for primary detection
    local_max_size: int = 5                 # footprint for local max detection (pixels)
    annulus_inner_factor: float = 0.9       # inner annulus bound = factor * peak_q
    annulus_outer_factor: float = 1.1       # outer annulus bound = factor * peak_q
    background_disk_r_sq: int = 9           # squared radius for peak exclusion disk (r=3)


@dataclass
class LowQExclusionConfig:
    """Unified low-q / DC exclusion configuration."""
    enabled: bool = True
    q_min_cycles_per_nm: float = 0.1   # physical floor (d < 10 nm)
    dc_bin_count: int = 3              # radial bins always excluded
    auto_q_min: bool = True            # derive q_min from image geometry


@dataclass
class ClusteringConfig:
    """Domain clustering configuration (opt-in via --cluster)."""
    enabled: bool = False
    method: str = "kmeans"          # kmeans | gmm | hdbscan
    n_clusters: int = 0             # 0 = auto (silhouette scan / hdbscan auto)
    n_clusters_max: int = 10
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    regularize: bool = True
    min_domain_size: int = 5
    dimred_method: str = "pca"      # pca | umap | none
    pca_variance_threshold: float = 0.95
    random_state: int = 42
    min_valid_tiles: int = 3                # minimum valid tiles for clustering
    spatial_regularize_iterations: int = 2  # iterations for spatial regularization
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    kmeans_n_init: int = 10


@dataclass
class GateThresholdsConfig:
    """All gate thresholds (G0-G12) in a single dataclass.

    ``threshold_dict(gate_id)`` returns the dict that ``evaluate_gate()``
    already expects via *threshold_override*.
    """
    # G0
    g0_nyquist_safety_margin: float = 0.95
    # G2
    g2_max_clipped_fraction: float = 0.005
    g2_min_range_ratio: float = 10.0
    # G3
    g3_min_coverage: float = 10.0
    g3_max_coverage: float = 95.0
    g3_max_fragments: int = 20
    g3_min_lcc_fraction: float = 0.5
    # G4
    g4_min_peak_snr: float = 3.0
    # G5
    g5_min_periods: float = 20.0
    # G6
    g6_min_fraction: float = 0.05
    # G7
    g7_min_median_snr: float = 5.0
    # G8
    g8_min_mean_symmetry: float = 0.3
    # G9
    g9_min_area: int = 9
    g9_max_entropy: float = 0.3
    g9_min_snr: float = 5.0
    # G10
    g10_max_phase_noise: float = 0.3
    g10_min_unwrap_success: float = 0.7
    # G11
    g11_max_ref_strain: float = 0.005
    g11_max_outlier_fraction: float = 0.20
    g11_strain_outlier_threshold: float = 0.05
    # G12
    g12_min_fraction_valid: float = 0.50
    g12_tolerance: float = 0.20

    def threshold_dict(self, gate_id: str) -> Optional[dict]:
        """Return the threshold dict matching GATE_DEFS format for *gate_id*."""
        _map = {
            "G0": {"nyquist_safety_margin": self.g0_nyquist_safety_margin},
            "G2": {"max_clipped_fraction": self.g2_max_clipped_fraction,
                    "min_range_ratio": self.g2_min_range_ratio},
            "G3": {"min_coverage": self.g3_min_coverage,
                    "max_coverage": self.g3_max_coverage,
                    "max_fragments": self.g3_max_fragments,
                    "min_lcc_fraction": self.g3_min_lcc_fraction},
            "G4": {"min_peak_snr": self.g4_min_peak_snr},
            "G5": {"min_periods": self.g5_min_periods},
            "G6": {"min_fraction": self.g6_min_fraction},
            "G7": {"min_median_snr": self.g7_min_median_snr},
            "G8": {"min_mean_symmetry": self.g8_min_mean_symmetry},
            "G9": {"min_area": self.g9_min_area,
                    "max_entropy": self.g9_max_entropy,
                    "min_snr": self.g9_min_snr},
            "G10": {"max_phase_noise": self.g10_max_phase_noise,
                     "min_unwrap_success": self.g10_min_unwrap_success},
            "G11": {"max_ref_strain": self.g11_max_ref_strain,
                     "max_outlier_fraction": self.g11_max_outlier_fraction,
                     "strain_outlier_threshold": self.g11_strain_outlier_threshold},
            "G12": {"min_fraction_valid": self.g12_min_fraction_valid,
                     "tolerance": self.g12_tolerance},
        }
        return _map.get(gate_id)


@dataclass
class PeakSNRConfig:
    """Peak SNR and FWHM measurement tunables."""
    signal_disk_radius_px: int = 3
    annular_width_min_q: float = 0.15
    annular_fwhm_multiplier: float = 1.5
    min_background_pixels: int = 20
    fwhm_patch_radius: int = 5
    moment_sigma_floor: float = 0.3
    fit_condition_max: float = 100.0
    symmetry_tolerance_px: float = 2.0
    non_collinear_min_angle_deg: float = 15.0


@dataclass
class ReferenceSelectionConfig:
    """Reference region scoring weights and bins."""
    scoring_weight_entropy: float = 0.4
    scoring_weight_snr: float = 0.3
    scoring_weight_area: float = 0.3
    orientation_bins: int = 12


@dataclass
class RingAnalysisConfig:
    """Ring analysis tunables."""
    ring_width_fwhm_mult: float = 2.0
    ring_width_fallback_frac: float = 0.03
    ring_width_no_fwhm_frac: float = 0.1


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    pixel_size_nm: float = 0.1297
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
    viz: VizConfig = field(default_factory=VizConfig)
    low_q: LowQExclusionConfig = field(default_factory=LowQExclusionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    tile_fft: TileFFTConfig = field(default_factory=TileFFTConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    gate_thresholds: GateThresholdsConfig = field(default_factory=GateThresholdsConfig)
    peak_snr: PeakSNRConfig = field(default_factory=PeakSNRConfig)
    reference_selection: ReferenceSelectionConfig = field(default_factory=ReferenceSelectionConfig)
    ring_analysis: RingAnalysisConfig = field(default_factory=RingAnalysisConfig)

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
            "viz": (VizConfig, "viz"),
            "low_q": (LowQExclusionConfig, "low_q"),
            "physics": (PhysicsConfig, "physics"),
            "tile_fft": (TileFFTConfig, "tile_fft"),
            "confidence": (ConfidenceConfig, "confidence"),
            "clustering": (ClusteringConfig, "clustering"),
            "gate_thresholds": (GateThresholdsConfig, "gate_thresholds"),
            "peak_snr": (PeakSNRConfig, "peak_snr"),
            "reference_selection": (ReferenceSelectionConfig, "reference_selection"),
            "ring_analysis": (RingAnalysisConfig, "ring_analysis"),
        }
        for attr, (klass, key) in _mapping.items():
            if key in d and isinstance(d[key], dict):
                sub = klass()
                for k, v in d[key].items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
                setattr(cfg, attr, sub)

        # Backward-compat: sync old-style validation keys → gate_thresholds
        if "validation" in d and isinstance(d["validation"], dict) and "gate_thresholds" not in d:
            v = d["validation"]
            gt = cfg.gate_thresholds
            _compat = {
                "min_detection_rate": "g6_min_fraction",
                "min_periods": "g5_min_periods",
                "min_tier_a_snr_median": "g7_min_median_snr",
                "min_symmetry_mean": "g8_min_mean_symmetry",
                "ref_area_min": "g9_min_area",
                "ref_entropy_max": "g9_max_entropy",
                "ref_snr_min": "g9_min_snr",
                "peak_lattice_fraction_min": "g12_min_fraction_valid",
            }
            for old_key, new_attr in _compat.items():
                if old_key in v:
                    setattr(gt, new_attr, v[old_key])

        return cfg
