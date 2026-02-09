# Detection Confidence Heatmap Plan

## Context

The hybrid pipeline classifies tiles into tiers (A/B/REJECTED) but lacks a single composite confidence score suitable for visual heatmap comparison with external tools. The user requests:

- **Option A (Required):** Pipeline-derived detection confidence heatmap from internal signals (SNR, pair_fraction, orientation_confidence, non-collinearity, FWHM). Deterministic, reproducible, diagnostic-only. Outputs `detection_confidence_heatmap.png` + `detection_confidence_map.npy`.
- **Option B (Optional):** ilastik comparison hooks — export feature stacks, load external probability maps, compute agreement metrics. Zero runtime dependency on ilastik. Pipeline must run fully without ilastik.

---

## Design Constraints

### DC-1: Diagnostic-Only Semantics (Locked)

**`detection_confidence` is never consumed by gates, tier assignment, GPA eligibility, or peak finding in this revision.** It is a purely diagnostic/visualization output derived *after* classification is complete. No pipeline logic reads the confidence score. This constraint is enforced by:
- Computing confidence only after the classification loop finishes
- Storing it on `GatedTileGrid` as a read-only output field
- No gate in `gates.py` references it
- Tests verify no import of confidence functions from classification code paths

This ensures the score cannot create feedback loops or alter existing pipeline behavior.

### DC-2: Ordinal Score, Not Probabilistic

The confidence score is **ordinal/comparative**: it provides a relative ranking of tile quality for visual inspection. It is **not calibrated** and does **not** represent the probability of crystal presence. A tile with score 0.8 is not "80% likely to be crystalline" — it simply has stronger combined signals than a tile scoring 0.4.

**Why a weighted linear model is sufficient:** The score's sole purpose is to drive a heatmap colormap for human visual comparison. For this use case:
- Monotonicity is what matters (stronger signals → warmer colors), not calibrated magnitudes
- A linear combination of clipped/normalized features preserves monotonicity and is trivially interpretable
- More complex models (logistic, learned) would add complexity without benefit since there is no classification decision boundary to optimize
- The weights are user-tunable via `ConfidenceConfig`, enabling domain experts to emphasize whichever signal they trust most

### DC-3: Non-Collinearity Is Binary

Non-collinearity is treated as **binary** (0 or 1) because it encodes **lattice dimensionality** (1D vs 2D), not signal strength. A tile either has enough non-collinear peak pairs to evidence a 2D lattice, or it doesn't. There is no meaningful intermediate value — "partially 2D" is not physically meaningful for a crystallographic lattice. The `min_non_collinear` threshold from `PeakGateConfig` defines the boundary.

### DC-4: FWHM Normalization

**What `max_fwhm_ratio` represents physically:** It is the maximum acceptable ratio of a peak's full-width-at-half-maximum to its spatial frequency magnitude (FWHM / |q|). Physically, this bounds how diffuse a Bragg peak can be relative to its position in reciprocal space. A ratio near `max_fwhm_ratio` means the peak is barely resolved from the background; a ratio near 0 means a sharp, well-defined peak.

**Why best peak (min FWHM/q) rather than median:** The best peak represents the tile's strongest crystallographic evidence. A tile with one sharp peak and several broad ones is more likely crystalline than one where all peaks are moderately broad. Using the best peak avoids penalizing tiles where secondary peaks are weaker due to texture or tilt, which is the common case in STEM-HAADF.

**Clipping vs sigmoid:** Hard clipping (`np.clip`) is used rather than a sigmoid because:
- The score is ordinal (DC-2), so smooth tails at extremes add no interpretive value
- Clipping is transparent and trivially auditable — the user can predict exactly when saturation occurs
- If future use requires smoother gradients, a sigmoid can be substituted without changing the API

### DC-5: ilastik Comparison Is Exploratory

**Agreement with ilastik does not validate correctness of either method.** The ilastik comparison module is **exploratory and qualitative** — it helps users visually identify regions where the two methods agree or disagree, prompting further investigation. Labels throughout (code, docs, output filenames) use "comparison" not "validation". Metrics (Pearson r, agreement fraction) describe correlation, not ground-truth accuracy.

---

## Step 1: Add `ConfidenceConfig` and update `GatedTileGrid`

**File: `src/pipeline_config.py`**

Add dataclass after `FWHMConfig`:

```python
@dataclass
class ConfidenceConfig:
    """Detection confidence heatmap weights.

    Produces an ordinal score for visualization only — not consumed by
    gates, tier assignment, or any classification logic (see DC-1, DC-2).
    Weights are normalized at runtime so they need not sum to 1.0.
    """
    enabled: bool = True
    w_snr: float = 0.40
    w_pair_fraction: float = 0.20
    w_orientation_confidence: float = 0.15
    w_non_collinearity: float = 0.10
    w_fwhm_quality: float = 0.15
```

Add field to `GatedTileGrid` (after `orientation_confidence_map`):

```python
detection_confidence_map: Optional[np.ndarray] = None  # (n_rows, n_cols) float [0,1], diagnostic only
```

Add to `PipelineConfig` (after `tile_fft`):

```python
confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
```

Add to `_mapping` in `from_dict()`:

```python
"confidence": (ConfidenceConfig, "confidence"),
```

---

## Step 2: Confidence scoring in `fft_snr_metrics.py`

**File: `src/fft_snr_metrics.py`**

Add `compute_tile_confidence()` function:

```python
def compute_tile_confidence(tc, tier_config, peak_gate_config, conf_config):
    """Per-tile detection confidence — ordinal score in [0, 1].

    Diagnostic only: not consumed by gates or tier assignment (DC-1).
    Score is comparative/ordinal, not probabilistic (DC-2).
    """
    if tc is None or tc.tier == "REJECTED":
        return 0.0

    # SNR: piecewise linear, 0 at tier_b_snr, 1 at 2×tier_a_snr
    snr_floor = tier_config.tier_b_snr
    snr_ceil = 2.0 * tier_config.tier_a_snr
    snr_norm = np.clip((tc.best_snr - snr_floor) / (snr_ceil - snr_floor + 1e-10), 0, 1)

    pf = np.clip(tc.pair_fraction, 0, 1)
    oc = np.clip(tc.orientation_confidence, 0, 1)

    # Binary: encodes lattice dimensionality (1D vs 2D), not strength (DC-3)
    nc = 1.0 if tc.n_non_collinear >= peak_gate_config.min_non_collinear else 0.0

    # FWHM quality: best peak's fwhm/q_mag ratio vs max_fwhm_ratio (DC-4)
    fwhm_q = 0.0
    if tc.peaks:
        valid = [p for p in tc.peaks if p.get("fwhm_valid") and p.get("q_mag", 0) > 0]
        if valid:
            best = min(valid, key=lambda p: p["fwhm"] / p["q_mag"])
            fwhm_q = np.clip(
                1.0 - (best["fwhm"] / best["q_mag"]) / peak_gate_config.max_fwhm_ratio,
                0, 1
            )

    w = conf_config
    w_total = w.w_snr + w.w_pair_fraction + w.w_orientation_confidence + w.w_non_collinearity + w.w_fwhm_quality
    score = (w.w_snr * snr_norm +
             w.w_pair_fraction * pf +
             w.w_orientation_confidence * oc +
             w.w_non_collinearity * nc +
             w.w_fwhm_quality * fwhm_q) / (w_total + 1e-10)

    return float(np.clip(score, 0, 1))
```

Modify `build_gated_tile_grid()`:
- Add param `confidence_config: ConfidenceConfig = None`
- **After** the classification loop completes (not during), compute confidence per tile and populate `detection_confidence_map`
- Pass it to the `GatedTileGrid(...)` constructor

This ordering enforces DC-1: confidence is computed post-classification, cannot influence it.

---

## Step 3: Save confidence artifact

**File: `src/reporting.py`**

In `save_pipeline_artifacts()`, after tier_map.npy block:

```python
if gated_grid is not None and gated_grid.detection_confidence_map is not None:
    save_npy(gated_grid.detection_confidence_map.astype(np.float32),
             output_dir / "detection_confidence_map.npy")
```

In `build_parameters_v3()`, add `detection_confidence` section with:
- `weights` dict (from `config.confidence`)
- `normalization` dict (snr_floor, snr_ceil, fwhm_reference, non_collinear_threshold)
- `statistics` dict (mean, median, std, min, max of valid tiles)
- `note`: "Ordinal diagnostic score for visualization. Not calibrated as probability."

---

## Step 4: Visualize heatmap

**File: `src/hybrid_viz.py`**

Add `_save_detection_confidence_heatmap(gated_grid, out_dir, dpi)`:
- Reads `gated_grid.detection_confidence_map`
- Masks skipped tiles (gray)
- `magma` colormap, vmin=0, vmax=1, colorbar labeled "Detection Confidence (ordinal)"
- Title: "Detection Confidence Heatmap"
- Returns saved path

Register in `save_pipeline_visualizations()` orchestrator after tile maps:

```python
if gated_grid is not None and gated_grid.detection_confidence_map is not None:
    _try("detection_confidence_heatmap",
         _save_detection_confidence_heatmap, gated_grid, out, dpi)
```

---

## Step 5: Wire in `analyze.py`

**File: `analyze.py`**

Update the `build_gated_tile_grid()` call to pass `confidence_config=config.confidence`.

No new CLI flags needed — confidence is always-on by default. Can be disabled via YAML `confidence.enabled: false`.

---

## Step 6: ilastik Comparison Module (Option B — Exploratory)

**New file: `src/ilastik_compare.py`**

Completely isolated module. Never imported unless user passes `--ilastik-map <path>`.

**All outputs labeled "comparison", never "validation" (DC-5).**

Functions:
- `export_feature_stack(gated_grid, output_path)` → saves `ilastik_feature_stack.npy` (n_rows, n_cols, 5) with channels: [snr, pair_fraction, orientation_confidence, fwhm, tier_encoded]
- `load_ilastik_probability_map(path)` → loads .npy or .h5, returns 2D float array
- `compute_comparison_metrics(pipeline_conf, ilastik_prob, valid_mask)` → returns dict with pearson_r, spearman_r, agreement_fraction. Note: these describe correlation, not accuracy.
- `save_comparison_overlay(pipeline_conf, ilastik_prob, valid_mask, out_dir, dpi)` → 3-panel PNG: pipeline / ilastik / difference
- `run_ilastik_comparison(gated_grid, ilastik_map_path, output_dir, dpi)` → orchestrates all of the above, saves `ilastik_comparison.json`

Shape mismatch handling: if ilastik map is pixel-resolution, zoom to tile grid shape.

h5py imported lazily (only if .h5 file provided).

---

## Step 7: CLI flag for ilastik comparison

**File: `analyze.py`**

Add argument `--ilastik-map` (type=str, default=None).

In `run_hybrid_pipeline()`, after artifact saving and before viz:

```python
if args.ilastik_map:
    from src.ilastik_compare import run_ilastik_comparison
    ilastik_result = run_ilastik_comparison(gated_grid, args.ilastik_map, output_path, config.viz.dpi)
```

Wrapped in try/except — failure logged but does not abort pipeline.

---

## Step 8: Tests

**New file: `tests/test_detection_confidence.py`**

### Required Tests (implement now):

1. `test_rejected_tile_returns_zero` — REJECTED → 0.0
2. `test_none_tile_returns_zero` — None → 0.0
3. `test_strong_crystal_high_score` — Tier A, SNR=10, pf=0.9, oc=0.95 → score > 0.7
4. `test_weak_tile_low_score` — Tier B, SNR=3.5, pf=0.2 → score < 0.5
5. `test_score_always_in_bounds` — multiple SNR values → all in [0, 1]
6. `test_tier_a_higher_than_tier_b` — matched params, A > B
7. `test_deterministic` — same input → same output
8. `test_custom_weights` — w_snr=1.0 all others 0 → score = normalized SNR
9. `test_grid_has_confidence_map` — GatedTileGrid field exists and correct shape
10. `test_npy_artifact_saved` — round-trip save/load
11. `test_heatmap_renders` — PNG created without error
12. `test_confidence_config_defaults` — weights sum to 1.0
13. `test_pipeline_config_has_confidence` — PipelineConfig().confidence exists
14. `test_from_dict_round_trip` — serialize/deserialize preserves values
15. `test_diagnostic_only_not_imported_by_gates` — `src/gates.py` does not import confidence functions
16. `test_export_feature_stack_shape` — 5 channels, correct shape
17. `test_load_npy_round_trip` — save/load identity
18. `test_no_ilastik_dependency` — normal pipeline imports don't pull in ilastik_compare

### Deferred Tests (not implemented in this revision):

These require real or realistic ilastik output and are deferred until ilastik integration is exercised on actual data:

- `test_perfect_agreement` — identical maps → pearson ~1, agreement 100%
- `test_anti_correlated` — inverted maps → pearson < -0.5
- `test_shape_mismatch_raises` — ValueError on incompatible shapes
- `test_confusion_matrix_structure` — correct keys and shape

---

## Verification

1. Run `pytest tests/test_detection_confidence.py -v` — all required tests pass
2. Run `pytest tests/ --ignore=tests/test_cluster_domains.py` — all existing + new tests pass
3. Run hybrid pipeline on dm4 file: `python analyze.py inputs/good_examples/HWAO_PHe_40mgml_2h_ms_blot_0003.dm4 --hybrid --no-gpa`
   - Verify `detection_confidence_map.npy` saved
   - Verify `detection_confidence_heatmap.png` rendered
   - Verify `parameters.json` has `detection_confidence` section with ordinal note
4. Verify ilastik comparison is NOT triggered without `--ilastik-map`

---

## File Summary

| File | Change |
|------|--------|
| `src/pipeline_config.py` | Add `ConfidenceConfig`, field on `GatedTileGrid`, field on `PipelineConfig`, `from_dict` entry |
| `src/fft_snr_metrics.py` | Add `compute_tile_confidence()` (post-classification), integrate into `build_gated_tile_grid()` |
| `src/reporting.py` | Save `detection_confidence_map.npy`, add metadata to `parameters.json` with ordinal note |
| `src/hybrid_viz.py` | Add `_save_detection_confidence_heatmap()`, register in orchestrator |
| `analyze.py` | Pass `confidence_config` to `build_gated_tile_grid()`, add `--ilastik-map` CLI flag |
| `src/ilastik_compare.py` | **New** — isolated ilastik comparison module (labeled "comparison" throughout) |
| `tests/test_detection_confidence.py` | **New** — 18 required tests + 4 deferred (documented, not implemented) |
