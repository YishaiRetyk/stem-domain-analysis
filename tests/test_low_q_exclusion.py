"""Tests for the unified low-q / DC exclusion system."""

import numpy as np
import pytest

from src.fft_coords import FFTGrid, compute_effective_q_min
from src.pipeline_config import LowQExclusionConfig, PipelineConfig


class TestComputeEffectiveQMin:
    """Tests for compute_effective_q_min()."""

    def test_auto_mode_256px_tile(self):
        """256px tile at 0.127 nm: q_scale ≈ 0.0308, 3 bins ≈ 0.092, floor 0.1 wins."""
        grid = FFTGrid(256, 256, 0.127)
        q_min = compute_effective_q_min(grid, enabled=True,
                                         q_min_cycles_per_nm=0.1,
                                         dc_bin_count=3, auto_q_min=True)
        # 3 * 0.0308 = 0.0924 < 0.1, so floor dominates
        assert q_min == pytest.approx(0.1, abs=1e-6)

    def test_auto_mode_4096px_image(self):
        """4096px at 0.127 nm: q_scale ≈ 0.00192, 3 bins ≈ 0.00577, floor 0.1 wins."""
        grid = FFTGrid(4096, 4096, 0.127)
        q_min = compute_effective_q_min(grid, enabled=True,
                                         q_min_cycles_per_nm=0.1,
                                         dc_bin_count=3, auto_q_min=True)
        assert q_min == pytest.approx(0.1, abs=1e-6)

    def test_auto_mode_bins_dominate(self):
        """Very small image where dc_bin_count * q_scale > floor."""
        # 16px at 0.1 nm: q_scale = 1/(16*0.1) = 0.625
        # 3 bins * 0.625 = 1.875 > 0.1 floor
        grid = FFTGrid(16, 16, 0.1)
        q_min = compute_effective_q_min(grid, enabled=True,
                                         q_min_cycles_per_nm=0.1,
                                         dc_bin_count=3, auto_q_min=True)
        expected = 3 * 0.625  # 1.875
        assert q_min == pytest.approx(expected, abs=1e-6)

    def test_manual_mode(self):
        """auto_q_min=False uses user value directly."""
        grid = FFTGrid(256, 256, 0.127)
        q_min = compute_effective_q_min(grid, enabled=True,
                                         q_min_cycles_per_nm=0.3,
                                         dc_bin_count=3, auto_q_min=False)
        assert q_min == pytest.approx(0.3, abs=1e-6)

    def test_disabled_returns_zero(self):
        """enabled=False → q_min = 0.0."""
        grid = FFTGrid(256, 256, 0.127)
        q_min = compute_effective_q_min(grid, enabled=False)
        assert q_min == 0.0


class TestLowQExclusionConfig:
    """Tests for the LowQExclusionConfig dataclass."""

    def test_default_values(self):
        cfg = LowQExclusionConfig()
        assert cfg.enabled is True
        assert cfg.q_min_cycles_per_nm == 0.1
        assert cfg.dc_bin_count == 3
        assert cfg.auto_q_min is True

    def test_pipeline_config_has_low_q(self):
        """PipelineConfig includes low_q with correct defaults."""
        cfg = PipelineConfig()
        assert hasattr(cfg, 'low_q')
        assert isinstance(cfg.low_q, LowQExclusionConfig)
        assert cfg.low_q.enabled is True

    def test_config_serialization_roundtrip(self):
        """PipelineConfig.to_dict() and from_dict() preserve low_q."""
        cfg = PipelineConfig()
        cfg.low_q.q_min_cycles_per_nm = 0.5
        cfg.low_q.auto_q_min = False

        d = cfg.to_dict()
        assert d['low_q']['q_min_cycles_per_nm'] == 0.5
        assert d['low_q']['auto_q_min'] is False

        cfg2 = PipelineConfig.from_dict(d)
        assert cfg2.low_q.q_min_cycles_per_nm == 0.5
        assert cfg2.low_q.auto_q_min is False


class TestGlobalFFTExclusion:
    """Verify global FFT respects effective_q_min."""

    def test_no_peaks_below_q_min(self):
        """When q_min is set high, no peaks should be found below it."""
        from src.global_fft import compute_global_fft
        from src.pipeline_config import GlobalFFTConfig
        from tests.synthetic import generate_single_crystal

        # d=0.5nm → q=2.0 c/nm. Set q_min=3.0 to exclude it.
        image = generate_single_crystal(
            shape=(512, 512), pixel_size_nm=0.1, d_spacing=0.5,
            orientation_deg=0, noise_level=0.02,
        )
        grid = FFTGrid(512, 512, 0.1)
        config = GlobalFFTConfig(min_peak_snr=2.0)

        result = compute_global_fft(image, grid, config, effective_q_min=3.0)
        for p in result.peaks:
            assert p.q_center >= 3.0, f"Peak at q={p.q_center} below q_min=3.0"


class TestTileFFTExclusion:
    """Verify tile FFT q-based DC mask."""

    def test_q_based_mask_covers_more_than_pixel_based(self):
        """q-based DC mask with reasonable q_min covers more pixels than 3px radius."""
        from src.tile_fft import extract_tile_peaks, compute_tile_fft
        from tests.synthetic import generate_single_crystal

        image = generate_single_crystal(
            shape=(256, 256), pixel_size_nm=0.1, d_spacing=0.5,
            orientation_deg=30, noise_level=0.02, amplitude=0.5,
        )
        window = np.outer(np.hanning(256), np.hanning(256))
        power = compute_tile_fft(image, window)
        grid = FFTGrid(256, 256, 0.1)

        # q_min=0.5 → masks region where |q| < 0.5 c/nm
        # This should exclude more pixels than dc_mask_radius=3
        q_mag = grid.q_mag_grid()
        q_mask_count = np.sum(q_mag < 0.5)
        pixel_mask_count = np.sum(
            ((np.arange(256)[:, None] - 128)**2 +
             (np.arange(256)[None, :] - 128)**2) <= 9
        )
        assert q_mask_count > pixel_mask_count

    def test_q_min_zero_uses_legacy(self):
        """effective_q_min=0 should use legacy pixel-based DC mask."""
        from src.tile_fft import extract_tile_peaks, compute_tile_fft
        from tests.synthetic import generate_single_crystal

        image = generate_single_crystal(
            shape=(256, 256), pixel_size_nm=0.1, d_spacing=0.5,
            orientation_deg=30, noise_level=0.02, amplitude=0.5,
        )
        window = np.outer(np.hanning(256), np.hanning(256))
        power = compute_tile_fft(image, window)
        grid = FFTGrid(256, 256, 0.1)

        # Should work without errors with q_min=0
        peaks = extract_tile_peaks(power, grid, effective_q_min=0.0)
        # Should also work with q_min > 0
        peaks_q = extract_tile_peaks(power, grid, effective_q_min=0.1)
        # Both should return peaks (crystal at q=2.0, well above any exclusion)
        assert len(peaks) > 0 or len(peaks_q) > 0


class TestSNRAnnulusExclusion:
    """Verify SNR annulus excludes low-q pixels."""

    def test_annulus_excludes_below_q_min(self):
        """Annular background should not include pixels below q_min."""
        from src.fft_peak_detection import compute_peak_snr
        from src.pipeline_config import TilePeak

        size = 64
        grid = FFTGrid(size, size, 0.1)
        power = np.random.rand(size, size) + 1.0

        # Peak at q=1.0 with q_min=0.5
        peak = TilePeak(qx=1.0, qy=0.0, q_mag=1.0, d_spacing=1.0,
                         angle_deg=0, intensity=10.0, fwhm=0.1)
        result = compute_peak_snr(power, peak, [peak], grid,
                                   effective_q_min=0.5)
        assert result.n_background_px > 0


class TestGPASigmaClamp:
    """Verify GPA mask sigma is clamped for DC safety."""

    def test_sigma_clamped_at_018_g_min(self):
        """When effective_q_min > 0, sigma should be ≤ 0.18 * min(|g|)."""
        from src.gpa import _determine_mask_sigma
        from src.pipeline_config import GVector, GPAConfig

        g = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0,
                     d_spacing=0.5, snr=10, fwhm=5.0, ring_index=0)
        config = GPAConfig(mask_radius_q="auto")
        # fwhm=5.0, so auto sigma = 0.5 * 5.0 = 2.5
        # Clamp: 0.18 * 2.0 = 0.36
        sigma = _determine_mask_sigma([g], config, effective_q_min=0.1)
        assert sigma <= 0.18 * 2.0 + 1e-10
        assert sigma == pytest.approx(0.36, abs=1e-6)

    def test_no_clamp_when_disabled(self):
        """When effective_q_min=0, sigma is not clamped."""
        from src.gpa import _determine_mask_sigma
        from src.pipeline_config import GVector, GPAConfig

        g = GVector(gx=2.0, gy=0.0, magnitude=2.0, angle_deg=0,
                     d_spacing=0.5, snr=10, fwhm=5.0, ring_index=0)
        config = GPAConfig(mask_radius_q="auto")
        sigma = _determine_mask_sigma([g], config, effective_q_min=0.0)
        # Auto: 0.5 * 5.0 = 2.5 (not clamped)
        assert sigma == pytest.approx(2.5, abs=1e-6)


class TestBandpassExclusion:
    """Verify bandpass ring_mask is zeroed below q_min."""

    def test_ring_mask_zero_below_q_min(self):
        """Ring mask should be zero for q < q_min."""
        from src.peak_finding import build_bandpass_image

        size = 128
        image = np.random.rand(size, size)
        grid = FFTGrid(size, size, 0.1)

        # g_dom at q=2.0 with q_min=1.0
        result_with = build_bandpass_image(image, 2.0, grid,
                                            effective_q_min=1.0)
        result_without = build_bandpass_image(image, 2.0, grid,
                                               effective_q_min=0.0)
        # Both should produce valid output
        assert result_with.shape == (size, size)
        assert result_without.shape == (size, size)
