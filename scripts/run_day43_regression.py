"""
scripts/run_day43_regression.py -- Day 43 Automated Regression Runner & CI Gate
================================================================================
Re-runs all 20 frozen labeled traces from Day 15 (`sample_data/labels.json`)
after the Week 8 hardening changes (Days 39-42) and compares the verdicts
against the Day 35 baseline (`validation/baseline_day35.json`).

CI Failure Conditions (exits with code 1 if ANY occur):
  1. Overall exact attribution accuracy drops below 75% (< 15/20).
  2. Any previously-passing labeled run in Day 35 (`exact_match == True`) now fails.
  3. Any previously-matching binary PASS/FAIL classification now fails.

Outputs:
  - `validation/baseline_day35.json` (frozen snapshot of Day 35 baseline)
  - `validation/regression_day43.json` (structured regression report)
  - `validation/REGRESSION_REPORT_DAY43.md` (human-readable Markdown report)
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACCURACY_DAY35_PATH = ROOT / "validation" / "accuracy_day35.json"
BASELINE_DAY35_PATH = ROOT / "validation" / "baseline_day35.json"
REGRESSION_JSON_PATH = ROOT / "validation" / "regression_day43.json"
REGRESSION_REPORT_PATH = ROOT / "validation" / "REGRESSION_REPORT_DAY43.md"

MIN_ACCURACY_THRESHOLD = 0.75


def ensure_day35_baseline() -> dict[str, Any]:
    """
    Ensure `validation/baseline_day35.json` exists as a frozen snapshot of Day 35
    before re-running the pipeline.
    """
    if not BASELINE_DAY35_PATH.exists():
        if not ACCURACY_DAY35_PATH.exists():
            from scripts.evaluate_day35_accuracy import evaluate_day35
            from scripts.run_day34_validation import run_validation

            run_validation()
            evaluate_day35()
        BASELINE_DAY35_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ACCURACY_DAY35_PATH, BASELINE_DAY35_PATH)

    return json.loads(BASELINE_DAY35_PATH.read_text(encoding="utf-8"))


def evaluate_regression(
    current_eval: dict[str, Any],
    baseline_eval: dict[str, Any],
    min_accuracy: float = MIN_ACCURACY_THRESHOLD,
) -> dict[str, Any]:
    """
    Compare current 20-trace evaluation payload against the frozen Day 35 baseline.

    Fails (`passed = False`) if:
      - `current_accuracy < min_accuracy` (default 0.75)
      - Any run with `exact_match == True` in `baseline_eval` has `exact_match == False` in `current_eval`
      - Any run with `binary_match == True` in `baseline_eval` has `binary_match == False` in `current_eval`
    """
    cur_metrics = current_eval.get("metrics", {})
    base_metrics = baseline_eval.get("metrics", {})

    cur_accuracy = float(cur_metrics.get("exact_attribution_accuracy", 0.0))
    base_accuracy = float(base_metrics.get("exact_attribution_accuracy", 0.0))
    cur_correct = int(cur_metrics.get("exact_attribution_correct", 0))
    base_correct = int(base_metrics.get("exact_attribution_correct", 0))
    total_runs = int(current_eval.get("total_runs", 20))

    base_by_id: dict[str, dict[str, Any]] = {
        r["run_id"]: r for r in baseline_eval.get("comparisons", [])
    }
    cur_by_id: dict[str, dict[str, Any]] = {
        r["run_id"]: r for r in current_eval.get("comparisons", [])
    }

    regressed_runs: list[dict[str, Any]] = []
    binary_regressed_runs: list[dict[str, Any]] = []
    improved_runs: list[dict[str, Any]] = []
    verdict_drift_runs: list[dict[str, Any]] = []
    run_comparisons: list[dict[str, Any]] = []

    for run_id, base_row in base_by_id.items():
        cur_row = cur_by_id.get(run_id)
        if cur_row is None:
            regressed_runs.append(
                {
                    "run_id": run_id,
                    "reason": "Run missing from current validation output",
                    "day35_exact_match": bool(base_row.get("exact_match", False)),
                    "day43_exact_match": False,
                }
            )
            continue

        was_exact = bool(base_row.get("exact_match", False))
        now_exact = bool(cur_row.get("exact_match", False))
        was_binary = bool(base_row.get("binary_match", False))
        now_binary = bool(cur_row.get("binary_match", False))

        if was_exact and not now_exact:
            regressed_runs.append(
                {
                    "run_id": run_id,
                    "expected_category": base_row.get("expected_category"),
                    "expected_primary_agent": base_row.get("expected_primary_agent"),
                    "day35_category": base_row.get("predicted_category"),
                    "day35_agent": base_row.get("predicted_primary_agent"),
                    "day43_category": cur_row.get("predicted_category"),
                    "day43_agent": cur_row.get("predicted_primary_agent"),
                    "reason": (
                        f"Previously-passing run '{run_id}' regressed: "
                        f"Day 35=({base_row.get('predicted_category')}, {base_row.get('predicted_primary_agent')}) -> "
                        f"Day 43=({cur_row.get('predicted_category')}, {cur_row.get('predicted_primary_agent')})"
                    ),
                }
            )
        elif not was_exact and now_exact:
            improved_runs.append(
                {
                    "run_id": run_id,
                    "day35_category": base_row.get("predicted_category"),
                    "day43_category": cur_row.get("predicted_category"),
                }
            )

        if was_binary and not now_binary:
            binary_regressed_runs.append(
                {
                    "run_id": run_id,
                    "day35_verdict": base_row.get("predicted_verdict"),
                    "day43_verdict": cur_row.get("predicted_verdict"),
                }
            )

        drifted = (
            base_row.get("predicted_category") != cur_row.get("predicted_category")
            or base_row.get("predicted_primary_agent") != cur_row.get("predicted_primary_agent")
            or base_row.get("predicted_verdict") != cur_row.get("predicted_verdict")
            or base_row.get("priority_level") != cur_row.get("priority_level")
        )
        if drifted:
            verdict_drift_runs.append(
                {
                    "run_id": run_id,
                    "day35": {
                        "verdict": base_row.get("predicted_verdict"),
                        "category": base_row.get("predicted_category"),
                        "agent": base_row.get("predicted_primary_agent"),
                        "priority": base_row.get("priority_level"),
                    },
                    "day43": {
                        "verdict": cur_row.get("predicted_verdict"),
                        "category": cur_row.get("predicted_category"),
                        "agent": cur_row.get("predicted_primary_agent"),
                        "priority": cur_row.get("priority_level"),
                    },
                }
            )

        run_comparisons.append(
            {
                "run_id": run_id,
                "expected_category": cur_row.get("expected_category"),
                "expected_primary_agent": cur_row.get("expected_primary_agent"),
                "day35_category": base_row.get("predicted_category"),
                "day35_agent": base_row.get("predicted_primary_agent"),
                "day35_exact_match": was_exact,
                "day43_category": cur_row.get("predicted_category"),
                "day43_agent": cur_row.get("predicted_primary_agent"),
                "day43_exact_match": now_exact,
                "day43_priority": cur_row.get("priority_level"),
                "day43_rules_fired": cur_row.get("rules_fired", []),
                "status": (
                    "REGRESSED"
                    if (was_exact and not now_exact)
                    else ("IMPROVED" if (not was_exact and now_exact) else "STABLE")
                ),
            }
        )

    accuracy_gate_passed = cur_accuracy >= min_accuracy
    no_regressions_passed = len(regressed_runs) == 0 and len(binary_regressed_runs) == 0
    overall_passed = accuracy_gate_passed and no_regressions_passed

    failure_reasons: list[str] = []
    if not accuracy_gate_passed:
        failure_reasons.append(
            f"Accuracy dropped below {min_accuracy * 100:.1f}% threshold: "
            f"{cur_correct}/{total_runs} ({cur_accuracy * 100:.1f}%)"
        )
    if regressed_runs:
        ids = [r["run_id"] for r in regressed_runs]
        failure_reasons.append(
            f"{len(regressed_runs)} previously-passing labeled run(s) regressed: {ids}"
        )
    if binary_regressed_runs:
        ids = [r["run_id"] for r in binary_regressed_runs]
        failure_reasons.append(
            f"{len(binary_regressed_runs)} binary PASS/FAIL match(es) regressed: {ids}"
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": "1.0",
        "passed": overall_passed,
        "accuracy_gate_passed": accuracy_gate_passed,
        "no_regressions_passed": no_regressions_passed,
        "min_accuracy_threshold": min_accuracy,
        "total_runs": total_runs,
        "day35_exact_correct": base_correct,
        "day35_exact_accuracy": base_accuracy,
        "day43_exact_correct": cur_correct,
        "day43_exact_accuracy": cur_accuracy,
        "accuracy_delta": round(cur_accuracy - base_accuracy, 4),
        "regressed_count": len(regressed_runs),
        "regressed_runs": regressed_runs,
        "binary_regressed_count": len(binary_regressed_runs),
        "binary_regressed_runs": binary_regressed_runs,
        "improved_count": len(improved_runs),
        "improved_runs": improved_runs,
        "verdict_drift_count": len(verdict_drift_runs),
        "verdict_drift_runs": verdict_drift_runs,
        "failure_reasons": failure_reasons,
        "run_comparisons": run_comparisons,
    }


def _build_regression_markdown(report: dict[str, Any]) -> str:
    status_str = "PASS (0 Regressions)" if report["passed"] else "FAIL"
    lines: list[str] = [
        "# AgentLens Automated Regression Report (Day 43)",
        "",
        f"**Generated At:** `{report['generated_at']}`  ",
        f"**Overall CI Gate Status:** **`{status_str}`**  ",
        f"**Day 35 Baseline Accuracy:** `{report['day35_exact_correct']}/{report['total_runs']} ({report['day35_exact_accuracy'] * 100:.1f}%)`  ",
        f"**Day 43 Post-Hardening Accuracy:** `{report['day43_exact_correct']}/{report['total_runs']} ({report['day43_exact_accuracy'] * 100:.1f}%)` (Threshold `>= {report['min_accuracy_threshold'] * 100:.1f}%`)  ",
        f"**Previously-Passing Runs Regressed:** `{report['regressed_count']}`  ",
        f"**Verdict Drift Count vs. Day 35:** `{report['verdict_drift_count']}`  ",
        "",
        "---",
        "",
        "## 1. Regression Gate Summary",
        "",
        "| Gate Check | Requirement | Actual | Status |",
        "|---|---|---|---|",
        f"| **Minimum Accuracy Gate** | `>= {report['min_accuracy_threshold'] * 100:.1f}%` (`15/20`) | `{report['day43_exact_correct']}/{report['total_runs']} ({report['day43_exact_accuracy'] * 100:.1f}%)` | **{'PASS' if report['accuracy_gate_passed'] else 'FAIL'}** |",
        f"| **Zero Per-Run Regressions** | `0` previously-passing runs fail | `{report['regressed_count']}` regressed | **{'PASS' if report['no_regressions_passed'] else 'FAIL'}** |",
        "",
        "---",
        "",
        "## 2. Run-by-Run Regression Comparison (Day 35 vs. Day 43)",
        "",
        "| Run ID | Expected (Category / Agent) | Day 35 (Category / Agent) | Day 43 (Category / Agent) | Day 43 Priority | Status |",
        "|---|---|---|---|---|---|",
    ]

    for row in report["run_comparisons"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['expected_category']}` / `{row['expected_primary_agent']}` | "
            f"`{row['day35_category']}` / `{row['day35_agent']}` | "
            f"`{row['day43_category']}` / `{row['day43_agent']}` | "
            f"`{row['day43_priority']}` | **{row['status']}** |"
        )

    lines.append("")
    return "\n".join(lines)


def run_day43_regression() -> dict[str, Any]:
    """
    Execute the full Day 43 regression workflow:
      1. Ensure frozen Day 35 baseline exists (`validation/baseline_day35.json`).
      2. Re-run all 20 labeled traces through the hardened pipeline (`run_validation()`).
      3. Evaluate accuracy (`evaluate_day35()`).
      4. Compare against Day 35 baseline (`evaluate_regression()`).
      5. Write `validation/regression_day43.json` and `validation/REGRESSION_REPORT_DAY43.md`.
    """
    from scripts.evaluate_day35_accuracy import evaluate_day35
    from scripts.run_day34_validation import run_validation

    baseline_eval = ensure_day35_baseline()
    run_validation()
    current_eval = evaluate_day35()

    report = evaluate_regression(current_eval=current_eval, baseline_eval=baseline_eval)

    REGRESSION_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGRESSION_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REGRESSION_REPORT_PATH.write_text(_build_regression_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    print("=" * 72)
    print("AgentLens -- Day 43: Automated 20-Trace Regression Run (vs. Day 35)")
    print("=" * 72)

    report = run_day43_regression()
    total = report["total_runs"]

    print("-" * 72)
    print(
        f"  Day 35 Baseline Accuracy : {report['day35_exact_correct']}/{total} "
        f"({report['day35_exact_accuracy'] * 100:.1f}%)"
    )
    print(
        f"  Day 43 Current Accuracy  : {report['day43_exact_correct']}/{total} "
        f"({report['day43_exact_accuracy'] * 100:.1f}%) "
        f"[Threshold >= {report['min_accuracy_threshold'] * 100:.1f}%]"
    )
    print(f"  Previously-Passing Runs Regressed : {report['regressed_count']}")
    print(f"  Verdict Drift Count vs. Day 35    : {report['verdict_drift_count']}")
    print(f"  Overall Regression Gate           : {'PASS' if report['passed'] else 'FAIL'}")
    if report["failure_reasons"]:
        for reason in report["failure_reasons"]:
            print(f"    [FAILURE] {reason}")
    print("-" * 72)
    print(f"  Saved JSON   -> {REGRESSION_JSON_PATH}")
    print(f"  Saved Report -> {REGRESSION_REPORT_PATH}")
    print("=" * 72)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
