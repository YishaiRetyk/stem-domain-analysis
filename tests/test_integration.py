"""Integration tests for the full hybrid pipeline."""

import json
import numpy as np
import pytest
from pathlib import Path

from src.fft_coords import FFTGrid
from src.pipeline_config import PipelineConfig
from src.preprocess_fft_safe import preprocess_fft_safe
from src.preprocess_segmentation import preprocess_segmentation
from src.roi_masking import compute_roi_mask, downsample_to_tile_grid
from src.global_fft import compute_global_fft
from src.tile_fft import process_all_tiles
from src.fft_snr_metrics import build_gated_tile_grid
from src.validation import validate_pipeline
from src.reporting import save_pipeline_artifacts, build_parameters_v3
from tests.synthetic import generate_single_crystal, generate_polycrystalline


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "test_output"


class TestFullPipelineSingleCrystal:
    """End-to-end test on a single-crystal synthetic image."""

    def test_pipeline_runs(self, tmp_output):
        """Full pipeline should run without errors."""
        image = generate_single_crystal(
            shape=(512, 512), pixel_size_nm=0.1, d_spacing=0.5,
            orientation_deg=30, noise_level=0.03, amplitude=0.4,
        )
        config = PipelineConfig(pixel_size_nm=0.1)
        fft_grid = FFTGrid(512, 512, 0.1)

        # Branch A
        preproc = preprocess_fft_safe(image, config.preprocessing)
        assert preproc.image_fft.shape == (512, 512)

        # Branch B
        seg = preprocess_segmentation(image, config.segmentation)
        assert seg.image_seg.shape == (512, 512)

        # ROI
        roi = compute_roi_mask(seg.image_seg, config.roi)
        roi_grid = downsample_to_tile_grid(
            roi.mask_full, config.tile_size, config.stride)

        # Global FFT
        global_result = compute_global_fft(preproc.image_fft, fft_grid,
                                            config.global_fft)
        assert global_result.radial_profile is not None

        # Tile FFT
        q_ranges = None
        if global_result.peaks:
            q_ranges = [(p.q_center - 0.3, p.q_center + 0.3)
                        for p in global_result.peaks]

        peak_sets, skipped = process_all_tiles(
            preproc.image_fft, roi_grid, fft_grid,
            tile_size=config.tile_size, stride=config.stride,
            q_ranges=q_ranges,
        )

        tile_grid = FFTGrid(config.tile_size, config.tile_size, 0.1)
        gated = build_gated_tile_grid(peak_sets, skipped, tile_grid,
                                       config.tile_size)
        assert gated.tier_map is not None

        # Validation
        report = validate_pipeline(
            preproc_record=preproc, roi_result=roi,
            global_fft_result=global_result, gated_grid=gated,
            tile_size=config.tile_size, pixel_size_nm=0.1,
            d_dom_nm=global_result.d_dom,
        )
        assert report.gates is not None

        # Save artifacts
        saved = save_pipeline_artifacts(
            tmp_output, config=config, fft_grid=fft_grid,
            preproc_record=preproc, seg_record=seg,
            roi_result=roi, global_fft_result=global_result,
            gated_grid=gated, validation_report=report,
        )
        assert "parameters.json" in saved
        assert "report.json" in saved

    def test_parameters_json_v3(self, tmp_output):
        """Parameters JSON should be version 3.0."""
        image = generate_single_crystal(shape=(512, 512), pixel_size_nm=0.1,
                                         d_spacing=0.5)
        config = PipelineConfig(pixel_size_nm=0.1)
        fft_grid = FFTGrid(512, 512, 0.1)
        preproc = preprocess_fft_safe(image, config.preprocessing)

        params = build_parameters_v3(config=config, fft_grid=fft_grid,
                                      preproc_record=preproc)
        assert params["version"] == "3.0"
        assert params["fft_convention"]["frequency_unit"] == "cycles/nm"

    def test_tier_map_saved(self, tmp_output):
        """tier_map.npy should be saved."""
        image = generate_single_crystal(shape=(512, 512), pixel_size_nm=0.1,
                                         d_spacing=0.5)
        config = PipelineConfig(pixel_size_nm=0.1)
        fft_grid = FFTGrid(512, 512, 0.1)

        preproc = preprocess_fft_safe(image, config.preprocessing)
        seg = preprocess_segmentation(image, config.segmentation)
        roi = compute_roi_mask(seg.image_seg, config.roi)
        roi_grid = downsample_to_tile_grid(roi.mask_full, config.tile_size,
                                            config.stride)
        global_result = compute_global_fft(preproc.image_fft, fft_grid)
        peak_sets, skipped = process_all_tiles(
            preproc.image_fft, roi_grid, fft_grid,
            tile_size=config.tile_size, stride=config.stride,
        )
        tile_grid = FFTGrid(config.tile_size, config.tile_size, 0.1)
        gated = build_gated_tile_grid(peak_sets, skipped, tile_grid,
                                       config.tile_size)

        saved = save_pipeline_artifacts(
            tmp_output, config=config, fft_grid=fft_grid,
            gated_grid=gated,
        )
        assert "tier_map.npy" in saved
        tier_map = np.load(saved["tier_map.npy"])
        assert set(np.unique(tier_map)).issubset({0, 1, 2, 3})


class TestFFTGridConsistency:
    """Verify FFTGrid is used consistently across modules."""

    def test_frequency_unit_in_artifacts(self, tmp_output):
        """F1: all artifacts should log frequency_unit as cycles/nm."""
        config = PipelineConfig(pixel_size_nm=0.1)
        fft_grid = FFTGrid(512, 512, 0.1)

        params = build_parameters_v3(config=config, fft_grid=fft_grid)
        assert params["fft_convention"]["frequency_unit"] == "cycles/nm"
        assert "no 2pi" in params["fft_convention"]["d_spacing_formula"].lower()
