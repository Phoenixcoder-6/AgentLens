"""
scripts/verify_day43.py -- Day 43 Verification Script (8 Checks)
=================================================================
Verifies Day 43 requirements:
  1. Frozen Day 35 baseline (`validation/baseline_day35.json`) exists with 20 runs & 16/20 accuracy.
  2. Full 20-trace regression run (`scripts/run_day43_regression.py`) completes and writes
     `validation/regression_day43.json` and `validation/REGRESSION_REPORT_DAY43.md`.
  3. Minimum accuracy gate passes (`day43_exact_accuracy >= 0.75`, actual `16/20 = 80.0%`).
  4. Zero previously-passing labeled runs regressed (`regressed_count == 0`, `binary_regressed_count == 0`).
  5. Zero unintended verdict drift across all 20 traces vs. Day 35 (`verdict_drift_count == 0`).
  6. Negative gate test: `evaluate_regression()` fails when accuracy < 75% (`0.70 < 0.75`).
  7. Negative gate test: `evaluate_regression()` fails when a previously-passing run flips to
     `exact_match=False`, even if overall accuracy remains >= 75%.
  8. GitHub Actions CI workflow (`.github/workflows/ci.yml`) triggers on PRs to `main` and
     runs `python scripts/run_day43_regression.py` with `--cov-fail-under=75`.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_day43_regression import (  # noqa: E402
    BASELINE_DAY35_PATH,
    REGRESSION_JSON_PATH,
    REGRESSION_REPORT_PATH,
    ensure_day35_baseline,
    evaluate_regression,
    run_day43_regression,
)

CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"


def check_1_baseline_snapshot_exists() -> str:
    baseline = ensure_day35_baseline()
    assert BASELINE_DAY35_PATH.exists(), "validation/baseline_day35.json missing"
    total = baseline.get("total_runs")
    correct = baseline.get("metrics", {}).get("exact_attribution_correct")
    acc = baseline.get("metrics", {}).get("exact_attribution_accuracy")
    assert total == 20, f"Expected 20 baseline runs, got {total}"
    assert correct == 16, f"Expected 16/20 baseline correct, got {correct}"
    assert acc == 0.8, f"Expected 0.80 baseline accuracy, got {acc}"
    return f"baseline_day35.json verified ({correct}/{total} = {acc * 100:.1f}%)"


def check_2_regression_runner_executes_and_writes_artifacts() -> str:
    report = run_day43_regression()
    assert REGRESSION_JSON_PATH.exists(), "validation/regression_day43.json missing"
    assert REGRESSION_REPORT_PATH.exists(), "validation/REGRESSION_REPORT_DAY43.md missing"
    loaded = json.loads(REGRESSION_JSON_PATH.read_text(encoding="utf-8"))
    assert loaded["total_runs"] == 20
    assert len(loaded["run_comparisons"]) == 20
    assert report["passed"] is True
    return "regression_day43.json & REGRESSION_REPORT_DAY43.md generated (20/20 traces)"


def check_3_minimum_accuracy_gate_passes() -> str:
    loaded = json.loads(REGRESSION_JSON_PATH.read_text(encoding="utf-8"))
    acc = loaded["day43_exact_accuracy"]
    correct = loaded["day43_exact_correct"]
    total = loaded["total_runs"]
    threshold = loaded["min_accuracy_threshold"]
    assert loaded["accuracy_gate_passed"] is True, f"Accuracy gate failed: {acc} < {threshold}"
    assert acc >= 0.75, f"Expected accuracy >= 0.75, got {acc}"
    return f"Day 43 accuracy = {correct}/{total} ({acc * 100:.1f}% >= {threshold * 100:.1f}%)"


def check_4_zero_previously_passing_runs_regressed() -> str:
    loaded = json.loads(REGRESSION_JSON_PATH.read_text(encoding="utf-8"))
    assert loaded["no_regressions_passed"] is True
    assert loaded["regressed_count"] == 0, f"Regressed runs: {loaded['regressed_runs']}"
    assert loaded["binary_regressed_count"] == 0, (
        f"Binary regressed runs: {loaded['binary_regressed_runs']}"
    )
    return "0/16 previously-passing runs regressed; 20/20 binary PASS/FAIL matches preserved"


def check_5_zero_verdict_drift_vs_day35() -> str:
    loaded = json.loads(REGRESSION_JSON_PATH.read_text(encoding="utf-8"))
    drift_count = loaded["verdict_drift_count"]
    assert drift_count == 0, f"Unexpected verdict drift vs. Day 35: {loaded['verdict_drift_runs']}"
    statuses = [r["status"] for r in loaded["run_comparisons"]]
    assert all(s in ("STABLE", "IMPROVED") for s in statuses)
    return "All 20 traces match Day 35 verdicts, categories, agents, and priorities (0 drift)"


def check_6_negative_test_accuracy_below_75_fails_gate() -> str:
    baseline = ensure_day35_baseline()
    degraded = copy.deepcopy(baseline)
    degraded["metrics"]["exact_attribution_correct"] = 14
    degraded["metrics"]["exact_attribution_accuracy"] = 0.70
    res = evaluate_regression(current_eval=degraded, baseline_eval=baseline, min_accuracy=0.75)
    assert res["accuracy_gate_passed"] is False
    assert res["passed"] is False
    assert len(res["failure_reasons"]) >= 1
    return "Gate accurately rejects <75% accuracy (14/20 = 70.0% -> FAIL)"


def check_7_negative_test_single_run_regression_fails_gate() -> str:
    baseline = ensure_day35_baseline()
    mutated = copy.deepcopy(baseline)
    for row in mutated["comparisons"]:
        if row["run_id"] == "run_lbl_pass_01":
            row["exact_match"] = False
            row["predicted_category"] = "unknown"
        elif row["run_id"] == "run_lbl_verification_01":
            row["exact_match"] = True
    res = evaluate_regression(current_eval=mutated, baseline_eval=baseline, min_accuracy=0.75)
    assert res["accuracy_gate_passed"] is True
    assert res["no_regressions_passed"] is False
    assert res["passed"] is False
    assert res["regressed_count"] == 1
    assert res["regressed_runs"][0]["run_id"] == "run_lbl_pass_01"
    return "Gate accurately rejects single-run regression (run_lbl_pass_01) even at 80.0% overall accuracy"


def check_8_github_actions_ci_wiring() -> str:
    assert CI_WORKFLOW_PATH.exists(), ".github/workflows/ci.yml not found"
    raw = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    parsed: dict[Any, Any] = yaml.safe_load(raw)
    triggers = parsed.get("on") or parsed.get(True) or {}
    assert "pull_request" in triggers, "Missing pull_request trigger in ci.yml"
    assert "main" in triggers["pull_request"].get("branches", []), "PR trigger does not target main"
    assert "--cov-fail-under=75" in raw, "Coverage gate not set to 75% in ci.yml"
    assert "python scripts/run_day43_regression.py" in raw, (
        "run_day43_regression.py not wired in ci.yml"
    )
    assert "python scripts/verify_day43.py" in raw, "verify_day43.py not wired in ci.yml"
    return (
        ".github/workflows/ci.yml wired for PRs to main with run_day43_regression.py & 75% cov gate"
    )


def main() -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Check 1: Day 35 Frozen Baseline Snapshot", check_1_baseline_snapshot_exists),
        (
            "Check 2: 20-Trace Regression Execution & Artifacts",
            check_2_regression_runner_executes_and_writes_artifacts,
        ),
        ("Check 3: Minimum Accuracy Gate (>= 75%)", check_3_minimum_accuracy_gate_passes),
        (
            "Check 4: Zero Previously-Passing Runs Regressed",
            check_4_zero_previously_passing_runs_regressed,
        ),
        ("Check 5: Zero Verdict Drift vs. Day 35", check_5_zero_verdict_drift_vs_day35),
        (
            "Check 6: Negative Gate Test (Accuracy < 75%)",
            check_6_negative_test_accuracy_below_75_fails_gate,
        ),
        (
            "Check 7: Negative Gate Test (Single-Run Regression)",
            check_7_negative_test_single_run_regression_fails_gate,
        ),
        ("Check 8: GitHub Actions CI Workflow Wiring", check_8_github_actions_ci_wiring),
    ]

    print("=" * 72)
    print("AgentLens -- Day 43 Verification Suite (Automated Regression & CI Gate)")
    print("=" * 72)

    passed = 0
    for title, fn in checks:
        try:
            detail = fn()
            print(f"[PASS] {title}")
            print(f"       -> {detail}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {title}")
            print(f"       -> {exc}")

    print("-" * 72)
    print(f"Day 43 Verification Summary: {passed}/{len(checks)} checks passed.")
    print("=" * 72)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
