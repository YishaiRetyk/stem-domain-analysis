"""Tests for CPU-parallel tile processing (PR 3)."""

import numpy as np
import pytest

from src.fft_coords import FFTGrid
from src.pipeline_config import TileFFTConfig, ParallelConfig
from src.tile_fft import process_all_tiles


def _make_sinusoid_image(size=512, tile_size=128, pixel_size_nm=0.1, d_spacing=0.8):
    """Create a simple sinusoidal test image."""
    g = 1.0 / d_spacing
    x = np.arange(size).reshape(1, -1) * pixel_size_nm
    y = np.arange(size).reshape(-1, 1) * pixel_size_nm
    image = 0.5 + 0.3 * np.cos(2 * np.pi * g * x) + 0.2 * np.cos(2 * np.pi * g * y)
    return image


class TestParallelTiles:
    """CPU-parallel tile processing tests."""

    def test_parallel_matches_sequential(self):
        """Parallel and sequential should give identical results."""
        image = _make_sinusoid_image(256, 128)
        grid = FFTGrid(256, 256, 0.1)
        tile_fft_config = TileFFTConfig()

        # Sequential
        peaks_seq, skipped_seq = process_all_tiles(
            image, None, grid, 128, 64,
            tile_fft_config=tile_fft_config,
            parallel_config=ParallelConfig(enabled=False),
        )

        # Parallel
        peaks_par, skipped_par = process_all_tiles(
            image, None, grid, 128, 64,
            tile_fft_config=tile_fft_config,
            parallel_config=ParallelConfig(enabled=True, cpu_workers=2),
        )

        assert len(peaks_seq) == len(peaks_par)
        np.testing.assert_array_equal(skipped_seq, skipped_par)

        # Compare peak counts per tile
        for ps_s, ps_p in zip(peaks_seq, peaks_par):
            assert len(ps_s.peaks) == len(ps_p.peaks), (
                f"Tile ({ps_s.tile_row},{ps_s.tile_col}): "
                f"{len(ps_s.peaks)} vs {len(ps_p.peaks)} peaks"
            )

    def test_parallel_with_roi_mask(self):
        """Skipped tiles should be handled correctly in parallel mode."""
        image = _make_sinusoid_image(256, 128)
        grid = FFTGrid(256, 256, 0.1)

        # Mask out some tiles
        from src.fft_features import get_tiling_info
        info = get_tiling_info(image.shape, 128, 64)
        n_rows, n_cols = info["grid_shape"]
        roi_mask = np.ones((n_rows, n_cols), dtype=bool)
        roi_mask[0, :] = False  # skip first row

        peaks_par, skipped_par = process_all_tiles(
            image, roi_mask, grid, 128, 64,
            tile_fft_config=TileFFTConfig(),
            parallel_config=ParallelConfig(enabled=True, cpu_workers=2),
        )

        # First row should be skipped
        assert np.all(skipped_par[0, :])
        # Non-first rows should not be skipped
        assert not np.any(skipped_par[1:, :])

    def test_parallel_single_worker(self):
        """Degenerate case: single worker should still work."""
        image = _make_sinusoid_image(256, 128)
        grid = FFTGrid(256, 256, 0.1)

        peaks, skipped = process_all_tiles(
            image, None, grid, 128, 64,
            tile_fft_config=TileFFTConfig(),
            parallel_config=ParallelConfig(enabled=True, cpu_workers=1),
        )
        assert len(peaks) > 0

    def test_parallel_disabled_by_default(self):
        """ParallelConfig default should have enabled=False."""
        cfg = ParallelConfig()
        assert cfg.enabled is False
        assert cfg.cpu_workers == 0
