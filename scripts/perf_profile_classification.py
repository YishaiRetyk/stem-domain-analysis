#!/usr/bin/env python3
"""
Performance profiling for tile classification (measure_peak_fwhm + classify_tile).

Generates synthetic tile data representative of a 94MP image workload,
instruments key functions with timing counters, and saves results to
artifacts/perf/.

Usage:
    python3 scripts/perf_profile_classification.py [--tag baseline|after]
"""
import argparse
import json
import os
import sys
import time
import statistics

import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fft_coords import FFTGrid
from src.pipeline_config import TilePeak, TilePeakSet, TierConfig, PeakGateConfig
from src.fft_peak_detection import classify_tile, measure_peak_fwhm


# ---------------------------------------------------------------------------
# Synthetic workload generator
# ---------------------------------------------------------------------------

def make_tile_with_peaks(grid, tile_size, rng, n_peaks=6, high_snr=True):
    """Create a synthetic power spectrum tile with n_peaks Gaussian peaks."""
    power = rng.exponential(1.0, (tile_size, tile_size)).astype(np.float64)
    peaks = []

    for i in range(n_peaks):
        angle_deg = (i * 360.0 / n_peaks) + rng.uniform(-5, 5)
        q_peak = rng.uniform(1.0, 3.0)
        qx = q_peak * np.cos(np.radians(angle_deg))
        qy = q_peak * np.sin(np.radians(angle_deg))
        px_x, px_y = grid.q_to_px(qx, qy)

        # Add Gaussian peak to power spectrum
        y, x = np.mgrid[:tile_size, :tile_size]
        amp = rng.uniform(200, 800) if high_snr else rng.uniform(10, 50)
        sigma_px = rng.uniform(1.5, 3.5)
        gaussian = amp * np.exp(-0.5 * ((x - px_x)**2 + (y - px_y)**2) / sigma_px**2)
        power += gaussian

        peak = TilePeak(
            qx=qx, qy=qy, q_mag=q_peak,
            d_spacing=1.0 / q_peak, angle_deg=angle_deg,
            intensity=float(amp),
            fwhm=0.1,
        )
        peaks.append(peak)

        # Add antipodal
        apx, apy = grid.q_to_px(-qx, -qy)
        gaussian_anti = amp * np.exp(-0.5 * ((x - apx)**2 + (y - apy)**2) / sigma_px**2)
        power += gaussian_anti
        anti_peak = TilePeak(
            qx=-qx, qy=-qy, q_mag=q_peak,
            d_spacing=1.0 / q_peak, angle_deg=angle_deg + 180,
            intensity=float(amp),
            fwhm=0.1,
        )
        peaks.append(anti_peak)

    return power, peaks


def generate_workload(n_tiles=500, tile_size=256, pixel_size_nm=0.1,
                      peaks_per_tile=6, seed=42):
    """Generate a list of TilePeakSet objects simulating a real workload."""
    rng = np.random.default_rng(seed)
    grid = FFTGrid(tile_size, tile_size, pixel_size_nm)

    peak_sets = []
    n_cols = int(np.ceil(np.sqrt(n_tiles)))
    for i in range(n_tiles):
        r = i // n_cols
        c = i % n_cols
        high_snr = rng.random() > 0.3  # 70% high-SNR
        power, peaks = make_tile_with_peaks(
            grid, tile_size, rng, n_peaks=peaks_per_tile, high_snr=high_snr
        )
        ps = TilePeakSet(peaks=peaks, tile_row=r, tile_col=c,
                         power_spectrum=power)
        peak_sets.append(ps)

    return peak_sets, grid


# ---------------------------------------------------------------------------
# Profiling harness
# ---------------------------------------------------------------------------

def profile_classification(n_tiles=500, tile_size=256, peaks_per_tile=6):
    """Run classification on synthetic tiles, collect timing stats."""
    print(f"Generating {n_tiles} synthetic tiles ({peaks_per_tile} peaks each, "
          f"tile_size={tile_size})...")
    peak_sets, grid = generate_workload(n_tiles, tile_size,
                                         peaks_per_tile=peaks_per_tile)

    tier_config = TierConfig()
    peak_gate_config = PeakGateConfig()

    # Counters
    total_peaks = 0
    tiles_classified = 0
    tier_counts = {"A": 0, "B": 0, "REJECTED": 0}

    # Timing for classify_tile
    tile_times = []

    # Timing for FWHM (both curve_fit and proxy paths)
    fwhm_fit_times = []     # curve_fit path
    fwhm_proxy_times = []   # proxy path
    fwhm_methods = {}

    # Monkey-patch both FWHM functions for timing
    import src.fft_peak_detection as mod
    _orig_fwhm = mod.measure_peak_fwhm
    _orig_proxy = mod.measure_peak_fwhm_proxy

    def _timed_fwhm(power, peak, fft_grid, **kwargs):
        t0 = time.perf_counter()
        result = _orig_fwhm(power, peak, fft_grid, **kwargs)
        dt = time.perf_counter() - t0
        fwhm_fit_times.append(dt)
        fwhm_methods[result.method] = fwhm_methods.get(result.method, 0) + 1
        return result

    def _timed_proxy(power, peak, fft_grid):
        t0 = time.perf_counter()
        result = _orig_proxy(power, peak, fft_grid)
        dt = time.perf_counter() - t0
        fwhm_proxy_times.append(dt)
        fwhm_methods[result.method] = fwhm_methods.get(result.method, 0) + 1
        return result

    mod.measure_peak_fwhm = _timed_fwhm
    mod.measure_peak_fwhm_proxy = _timed_proxy

    print(f"Running classification on {n_tiles} tiles...")
    t_start = time.perf_counter()

    for ps in peak_sets:
        t0 = time.perf_counter()
        tc = classify_tile(ps, grid, tier_config, peak_gate_config)
        dt = time.perf_counter() - t0
        tile_times.append(dt)

        tiles_classified += 1
        total_peaks += len(ps.peaks)
        tier_counts[tc.tier] = tier_counts.get(tc.tier, 0) + 1

    t_total = time.perf_counter() - t_start

    # Restore originals
    mod.measure_peak_fwhm = _orig_fwhm
    mod.measure_peak_fwhm_proxy = _orig_proxy

    # Compute stats
    all_fwhm_times = fwhm_fit_times + fwhm_proxy_times
    n_fwhm_calls = len(all_fwhm_times)
    fwhm_total = sum(all_fwhm_times)
    fwhm_mean = statistics.mean(all_fwhm_times) if all_fwhm_times else 0
    fwhm_median = statistics.median(all_fwhm_times) if all_fwhm_times else 0
    fwhm_p95 = sorted(all_fwhm_times)[int(0.95 * len(all_fwhm_times))] if all_fwhm_times else 0

    fit_total = sum(fwhm_fit_times)
    proxy_total = sum(fwhm_proxy_times)
    fit_mean = statistics.mean(fwhm_fit_times) if fwhm_fit_times else 0
    proxy_mean = statistics.mean(fwhm_proxy_times) if fwhm_proxy_times else 0

    tile_mean = statistics.mean(tile_times)
    tile_median = statistics.median(tile_times)

    results = {
        "n_tiles": n_tiles,
        "tile_size": tile_size,
        "peaks_per_tile": peaks_per_tile,
        "total_peaks": total_peaks,
        "tiles_classified": tiles_classified,
        "tier_counts": tier_counts,
        "total_time_s": round(t_total, 3),
        "tiles_per_sec": round(n_tiles / t_total, 2),
        "peaks_per_sec": round(total_peaks / t_total, 2),
        "tile_time_mean_ms": round(tile_mean * 1000, 3),
        "tile_time_median_ms": round(tile_median * 1000, 3),
        "fwhm_calls": n_fwhm_calls,
        "fwhm_fit_calls": len(fwhm_fit_times),
        "fwhm_proxy_calls": len(fwhm_proxy_times),
        "fwhm_total_s": round(fwhm_total, 3),
        "fwhm_fit_total_s": round(fit_total, 3),
        "fwhm_proxy_total_s": round(proxy_total, 3),
        "fwhm_fraction_of_total": round(fwhm_total / t_total, 4) if t_total > 0 else 0,
        "fwhm_mean_ms": round(fwhm_mean * 1000, 4),
        "fwhm_median_ms": round(fwhm_median * 1000, 4),
        "fwhm_p95_ms": round(fwhm_p95 * 1000, 4),
        "fwhm_fit_mean_ms": round(fit_mean * 1000, 4),
        "fwhm_proxy_mean_ms": round(proxy_mean * 1000, 4),
        "fwhm_methods": fwhm_methods,
    }
    return results


def format_results(results):
    """Format results as human-readable text."""
    lines = [
        "=" * 60,
        "Classification Performance Profile",
        "=" * 60,
        f"Tiles:           {results['n_tiles']} ({results['tile_size']}×{results['tile_size']})",
        f"Peaks/tile:      {results['peaks_per_tile']} ({results['total_peaks']} total)",
        f"Tier counts:     {results['tier_counts']}",
        "",
        "--- Overall ---",
        f"Total time:      {results['total_time_s']:.3f} s",
        f"Throughput:      {results['tiles_per_sec']:.1f} tiles/s, "
        f"{results['peaks_per_sec']:.0f} peaks/s",
        f"Tile time (mean/median): {results['tile_time_mean_ms']:.2f} / "
        f"{results['tile_time_median_ms']:.2f} ms",
        "",
        "--- FWHM (all paths) ---",
        f"Total calls:     {results['fwhm_calls']} "
        f"(fit: {results.get('fwhm_fit_calls', '?')}, "
        f"proxy: {results.get('fwhm_proxy_calls', '?')})",
        f"Total time:      {results['fwhm_total_s']:.3f} s "
        f"({results['fwhm_fraction_of_total']*100:.1f}% of total)",
        f"  curve_fit:     {results.get('fwhm_fit_total_s', '?')} s "
        f"(mean {results.get('fwhm_fit_mean_ms', '?')} ms)",
        f"  proxy:         {results.get('fwhm_proxy_total_s', '?')} s "
        f"(mean {results.get('fwhm_proxy_mean_ms', '?')} ms)",
        f"Mean/median:     {results['fwhm_mean_ms']:.3f} / "
        f"{results['fwhm_median_ms']:.3f} ms",
        f"P95:             {results['fwhm_p95_ms']:.3f} ms",
        f"Methods:         {results['fwhm_methods']}",
        "=" * 60,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline",
                        help="Tag for output files (baseline or after)")
    parser.add_argument("--tiles", type=int, default=500)
    parser.add_argument("--peaks", type=int, default=6)
    args = parser.parse_args()

    results = profile_classification(n_tiles=args.tiles,
                                      peaks_per_tile=args.peaks)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "perf")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"perf_{args.tag}.json")
    txt_path = os.path.join(out_dir, f"perf_{args.tag}.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    txt = format_results(results)
    with open(txt_path, "w") as f:
        f.write(txt + "\n")

    print(txt)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
