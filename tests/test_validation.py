"""Tests for unified pipeline validation (WS7)."""

import numpy as np
import pytest
from src.pipeline_config import (
    GateResult, TierSummary, FFTPreprocRecord, ROIMaskResult,
    GlobalFFTResult, GlobalPeak, GVector,
)
from src.validation import validate_pipeline
from src.gates import evaluate_gate, GATE_DEFS


class TestGateEvaluation:
    """Individual gate evaluation tests."""

    def test_g1_passes(self):
        result = evaluate_gate("G1", {
            "is_2d": True, "no_nan": True, "no_inf": True,
            "min_dim": True, "has_pixel_size": True,
        })
        assert result.passed
        assert result.failure_behavior == "FATAL"

    def test_g1_fails(self):
        result = evaluate_gate("G1", {"is_2d": False, "no_nan": True})
        assert not result.passed

    def test_g2_passes(self):
        result = evaluate_gate("G2", {
            "clipped_fraction": 0.001,
            "intensity_range_ratio": 50.0,
        })
        assert result.passed
        assert result.failure_behavior == "DEGRADE_CONFIDENCE"

    def test_g2_fails_clipping(self):
        result = evaluate_gate("G2", {
            "clipped_fraction": 0.01,  # > 0.5%
            "intensity_range_ratio": 50.0,
        })
        assert not result.passed

    def test_g3_passes(self):
        result = evaluate_gate("G3", {"coverage_pct": 50.0, "n_components": 3})
        assert result.passed
        assert result.failure_behavior == "FALLBACK"

    def test_g3_fails_coverage(self):
        result = evaluate_gate("G3", {"coverage_pct": 5.0, "n_components": 3})
        assert not result.passed

    def test_g4_passes(self):
        result = evaluate_gate("G4", 5.0)
        assert result.passed

    def test_g4_fails(self):
        result = evaluate_gate("G4", 1.0)
        assert not result.passed

    def test_g5_passes(self):
        result = evaluate_gate("G5", 25.0)
        assert result.passed
        assert result.failure_behavior == "FATAL"

    def test_g5_fails(self):
        result = evaluate_gate("G5", 10.0)
        assert not result.passed

    def test_g6_passes(self):
        result = evaluate_gate("G6", 0.10)
        assert result.passed

    def test_g6_fails(self):
        result = evaluate_gate("G6", 0.02)
        assert not result.passed

    def test_g12_passes(self):
        result = evaluate_gate("G12", 0.75)
        assert result.passed

    def test_g12_fails(self):
        result = evaluate_gate("G12", 0.30)
        assert not result.passed

    def test_no_spectral_entropy_gate(self):
        """C1: there should be no spectral entropy gate."""
        for gid, gdef in GATE_DEFS.items():
            assert "spectral_entropy" not in gdef.description.lower()


class TestValidatePipeline:
    """Integrated validation report tests."""

    def test_empty_pipeline(self):
        """Should produce a report even with no data."""
        report = validate_pipeline()
        assert report.gates is not None
        assert report.timestamp

    def test_overall_pass_with_passing_gates(self):
        gate_results = {
            "G1": GateResult("G1", "Input", True, failure_behavior="FATAL"),
            "G2": GateResult("G2", "Preproc", True, failure_behavior="DEGRADE_CONFIDENCE"),
        }
        report = validate_pipeline(gate_results=gate_results)
        assert report.overall_pass

    def test_overall_fail_on_fatal(self):
        gate_results = {
            "G1": GateResult("G1", "Input", False, failure_behavior="FATAL",
                             reason="bad input"),
        }
        report = validate_pipeline(gate_results=gate_results)
        assert not report.overall_pass

    def test_degrade_does_not_fail_overall(self):
        gate_results = {
            "G6": GateResult("G6", "Tier A rate", False,
                             failure_behavior="DEGRADE_CONFIDENCE"),
        }
        report = validate_pipeline(gate_results=gate_results)
        assert report.overall_pass  # DEGRADE doesn't cause overall failure

    def test_skip_stage_in_summary(self):
        gate_results = {
            "G10": GateResult("G10", "GPA phase", False,
                              failure_behavior="SKIP_STAGE"),
        }
        report = validate_pipeline(gate_results=gate_results)
        assert "skipped" in report.summary.lower() or "G10" in report.summary
