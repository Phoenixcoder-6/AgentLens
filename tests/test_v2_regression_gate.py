"""
tests/test_v2_regression_gate.py — Day 66: v2.0 Extended Regression Gate Unit Tests
=====================================================================================
Validates the v2.0 extended regression runner, certification thresholds, report
generation, and causal pipeline integration.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_v2_regression import (
    MIN_BASELINE_ACCURACY_THRESHOLD,
    MIN_BINARY_ACCURACY_THRESHOLD,
    MIN_EXACT_ACCURACY_THRESHOLD,
    TARGET_DEMO_RUNS,
    _map_predicted_category,
    load_labeled_trace,
    run_v2_regression,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_category_mapping_helper() -> None:
    """Validate mapping from Arbiter verdict & primary_cause to label category."""
    assert _map_predicted_category("PASS", "unknown") == "pass"
    assert _map_predicted_category("FAIL", "reasoning") == "reasoning_failure"
    assert _map_predicted_category("FAIL", "execution") == "execution_failure"
    assert _map_predicted_category("FAIL", "workflow") == "workflow_failure"
    assert _map_predicted_category("FAIL", "verification") == "verification_failure"


def test_load_labeled_trace_valid() -> None:
    """Verify loading labeled trace extracts valid RunTrace and normalizes tool calls."""
    trace = load_labeled_trace("run_lbl_pass_01")
    assert trace.run_id == "run_lbl_pass_01"
    assert len(trace.steps) == 3
    assert trace.workflow == "research_report_pipeline"
    assert trace.steps[0].agent == "researcher"


def test_target_demo_runs_count() -> None:
    """Verify exactly 5 canonical enterprise demo runs are included."""
    assert len(TARGET_DEMO_RUNS) == 5
    assert "demo_normal_pass_01" in TARGET_DEMO_RUNS
    assert "demo_grounded_p1_01" in TARGET_DEMO_RUNS
    assert "demo_heuristic_p2_hallucination" in TARGET_DEMO_RUNS
    assert "demo_heuristic_p3_workflow" in TARGET_DEMO_RUNS
    assert "demo_diff_pair_diverged_b" in TARGET_DEMO_RUNS


def test_run_v2_regression_runner_execution(tmp_path: Path) -> None:
    """Verify running the regression runner in a temp path produces valid reports."""
    out_json = tmp_path / "test_reg_v2.json"
    out_md = tmp_path / "test_reg_v2.md"

    res = run_v2_regression(output_json=out_json, output_md=out_md)

    assert res["passed"] is True
    assert res["release_version"] == "v2.0.0"
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert len(payload["comparisons"]) == 30

    md_content = out_md.read_text(encoding="utf-8")
    assert "AgentLens v2.0 Final Extended Regression Report" in md_content
    assert "RELEASE GATE PASSED" in md_content


def test_regression_attribution_threshold() -> None:
    """Verify exact causal attribution accuracy achieves >= 85.0%."""
    reg_json = REPO_ROOT / "validation" / "regression_v2.json"
    assert reg_json.exists(), "validation/regression_v2.json must exist"
    data = json.loads(reg_json.read_text(encoding="utf-8"))

    achieved = data["gates"]["exact_attribution_gate"]["achieved"]
    assert achieved >= MIN_EXACT_ACCURACY_THRESHOLD, (
        f"Attribution accuracy {achieved * 100:.1f}% fell below threshold {MIN_EXACT_ACCURACY_THRESHOLD * 100:.1f}%"
    )
    assert data["gates"]["exact_attribution_gate"]["passed"] is True


def test_regression_binary_detection_threshold() -> None:
    """Verify binary failure detection achieves 100.0%."""
    reg_json = REPO_ROOT / "validation" / "regression_v2.json"
    data = json.loads(reg_json.read_text(encoding="utf-8"))

    achieved = data["gates"]["binary_detection_gate"]["achieved"]
    assert achieved == MIN_BINARY_ACCURACY_THRESHOLD, (
        f"Binary accuracy {achieved * 100:.1f}% was not 100%"
    )
    assert data["gates"]["binary_detection_gate"]["passed"] is True


def test_regression_baseline_preservation() -> None:
    """Verify Day 43 frozen benchmark (16/20 = 80.0%) is strictly preserved."""
    reg_json = REPO_ROOT / "validation" / "regression_v2.json"
    data = json.loads(reg_json.read_text(encoding="utf-8"))

    achieved = data["gates"]["baseline_preservation_gate"]["achieved"]
    assert achieved >= MIN_BASELINE_ACCURACY_THRESHOLD, (
        f"Baseline accuracy {achieved * 100:.1f}% fell below threshold {MIN_BASELINE_ACCURACY_THRESHOLD * 100:.1f}%"
    )
    assert data["gates"]["baseline_preservation_gate"]["correct"] >= 16
    assert data["gates"]["baseline_preservation_gate"]["passed"] is True


def test_regression_causal_and_remediation_coverage() -> None:
    """Verify all 30 traces produce valid causal nodes and remediation patch records."""
    reg_json = REPO_ROOT / "validation" / "regression_v2.json"
    data = json.loads(reg_json.read_text(encoding="utf-8"))

    for row in data["comparisons"]:
        assert row["causal_nodes"] > 0, f"Run {row['run_id']} has 0 causal nodes"
        if row["predicted_verdict"] == "FAIL":
            # Failed runs should attempt remediation
            assert row["remediation_status"] != "", f"Run {row['run_id']} has empty remediation"
        else:
            assert row["remediation_status"] == "none"
