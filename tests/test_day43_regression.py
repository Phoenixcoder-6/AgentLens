"""
tests/test_day43_regression.py -- Day 43 Automated Regression Run & CI Gate Tests
==================================================================================
Tests for the Day 43 automated 20-trace regression runner (`scripts/run_day43_regression.py`)
and the GitHub Actions CI workflow (`.github/workflows/ci.yml`).

Verifies:
  1. Re-running all 20 frozen labeled traces after Week 8 hardening (Days 39-42)
     produces zero regressions vs. Day 35 (`16/20 = 80.0%` exact attribution).
  2. `evaluate_regression()` fails (`passed = False`) when accuracy drops below 75%.
  3. `evaluate_regression()` fails (`passed = False`) when any previously-passing
     labeled run (`exact_match == True`) fails, even if overall accuracy >= 75%.
  4. `evaluate_regression()` fails (`passed = False`) when a binary PASS/FAIL
     classification regresses or a run is missing.
  5. `.github/workflows/ci.yml` runs on every PR to `main` and executes
     `scripts/run_day43_regression.py` as a required CI step.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from scripts.run_day43_regression import (
    BASELINE_DAY35_PATH,
    REGRESSION_JSON_PATH,
    REGRESSION_REPORT_PATH,
    ensure_day35_baseline,
    evaluate_regression,
    run_day43_regression,
)

ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"


class TestDay43RegressionSuite:
    """End-to-end and unit tests for the Day 43 automated regression runner."""

    def test_full_20_trace_regression_passes_with_zero_regressions(self) -> None:
        """All 20 frozen labeled traces pass regression check vs. Day 35 baseline."""
        report = run_day43_regression()

        assert report["passed"] is True
        assert report["accuracy_gate_passed"] is True
        assert report["no_regressions_passed"] is True
        assert report["total_runs"] == 20
        assert report["day35_exact_correct"] == 16
        assert report["day43_exact_correct"] >= 16
        assert report["day43_exact_accuracy"] >= 0.75
        assert report["regressed_count"] == 0
        assert report["regressed_runs"] == []
        assert report["binary_regressed_count"] == 0
        assert report["verdict_drift_count"] == 0
        assert report["failure_reasons"] == []

        assert BASELINE_DAY35_PATH.exists()
        assert REGRESSION_JSON_PATH.exists()
        assert REGRESSION_REPORT_PATH.exists()

    def test_fails_when_accuracy_drops_below_75_percent(self) -> None:
        """CI gate fails when overall exact attribution accuracy drops below 75%."""
        baseline = ensure_day35_baseline()
        degraded = copy.deepcopy(baseline)
        degraded["metrics"]["exact_attribution_correct"] = 14
        degraded["metrics"]["exact_attribution_accuracy"] = 0.70

        report = evaluate_regression(
            current_eval=degraded,
            baseline_eval=baseline,
            min_accuracy=0.75,
        )

        assert report["accuracy_gate_passed"] is False
        assert report["passed"] is False
        assert any("Accuracy dropped below 75.0%" in r for r in report["failure_reasons"])

    def test_fails_when_previously_passing_run_regresses_even_if_accuracy_above_75(self) -> None:
        """
        CI gate fails if ANY previously-passing run (`exact_match == True`) now fails,
        even when overall accuracy stays >= 75% (e.g., 15/20 = 75.0% or 16/20 = 80.0%).
        """
        baseline = ensure_day35_baseline()
        mutated = copy.deepcopy(baseline)

        # Flip one previously-passing run (run_lbl_pass_01) to exact_match=False
        # and flip one previously-failing run (run_lbl_verification_01) to exact_match=True
        # so overall accuracy stays 16/20 (80.0% >= 75.0%), testing the per-run regression guard.
        for row in mutated["comparisons"]:
            if row["run_id"] == "run_lbl_pass_01":
                assert row["exact_match"] is True
                row["exact_match"] = False
                row["predicted_category"] = "unknown"
            elif row["run_id"] == "run_lbl_verification_01":
                row["exact_match"] = True
                row["predicted_category"] = row["expected_category"]
                row["predicted_primary_agent"] = row["expected_primary_agent"]

        report = evaluate_regression(
            current_eval=mutated,
            baseline_eval=baseline,
            min_accuracy=0.75,
        )

        assert report["accuracy_gate_passed"] is True
        assert report["no_regressions_passed"] is False
        assert report["passed"] is False
        assert report["regressed_count"] == 1
        assert report["regressed_runs"][0]["run_id"] == "run_lbl_pass_01"
        assert report["improved_count"] == 1
        assert report["improved_runs"][0]["run_id"] == "run_lbl_verification_01"
        assert any("run_lbl_pass_01" in reason for reason in report["failure_reasons"])

    def test_fails_when_binary_pass_fail_regresses_or_run_missing(self) -> None:
        """CI gate fails if a binary PASS/FAIL match regresses or a trace is missing."""
        baseline = ensure_day35_baseline()
        mutated = copy.deepcopy(baseline)

        # Flip binary_match on run_lbl_pass_02 and drop run_lbl_workflow_04
        mutated["comparisons"] = [
            r for r in mutated["comparisons"] if r["run_id"] != "run_lbl_workflow_04"
        ]
        for row in mutated["comparisons"]:
            if row["run_id"] == "run_lbl_pass_02":
                row["binary_match"] = False
                row["predicted_verdict"] = "FAIL"

        report = evaluate_regression(
            current_eval=mutated,
            baseline_eval=baseline,
            min_accuracy=0.75,
        )

        assert report["passed"] is False
        assert report["no_regressions_passed"] is False
        assert report["binary_regressed_count"] == 1
        assert any(r["run_id"] == "run_lbl_workflow_04" for r in report["regressed_runs"])

    def test_github_actions_ci_workflow_wires_day43_regression_on_pr_to_main(self) -> None:
        """Verify `.github/workflows/ci.yml` runs on PRs to `main` and executes Day 43 regression."""
        assert CI_WORKFLOW_PATH.exists(), ".github/workflows/ci.yml must exist"
        raw_text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
        workflow: dict[Any, Any] = yaml.safe_load(raw_text)

        # PyYAML parses `on:` key as boolean True or string "on"
        triggers = workflow.get("on") or workflow.get(True) or {}
        assert "pull_request" in triggers
        assert "main" in triggers["pull_request"].get("branches", [])

        jobs = workflow.get("jobs", {})
        assert "test" in jobs
        assert "validation-gate" in jobs

        assert "--cov-fail-under=75" in raw_text
        assert "python scripts/run_day43_regression.py" in raw_text
        assert "python scripts/verify_day43.py" in raw_text
        assert "validation/regression_day43.json" in raw_text
