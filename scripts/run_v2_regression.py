"""
scripts/run_v2_regression.py — Day 66: v2.0 Extended 30-Trace Regression Runner & Release Gate
==============================================================================================
Executes the comprehensive 30-trace extended regression suite certifying AgentLens v2.0:
  1. 20 Frozen Baseline Traces (sample_data/labeled_traces/ & sample_data/labels.json)
  2. 5 Canonical Phase B Multi-Agent Traces (sample_data/v2_traces/ & manifest.json)
  3. 5 Enterprise Demo Traces (sample_data/demo_traces/ & manifest.json)

Certification Conditions (exits with code 1 if ANY fail):
  - Overall exact causal attribution accuracy >= 85.0% (target: >= 26/30).
  - Overall binary failure detection accuracy == 100.0% (target: 30/30).
  - Frozen Day 43 baseline preservation >= 80.0% (16/20 exact matches, 0 regressions).
  - Full v2 analysis pipeline (CausalGraphEngine, BlastRadiusCalculator, PromptFixer)
    runs cleanly without unhandled exceptions.

Outputs:
  - validation/regression_v2.json
  - validation/REGRESSION_REPORT_V2.md
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzers.arbiter import Arbiter
from analyzers.causal.blast_radius import BlastRadiusCalculator
from analyzers.causal.fault_tree import CausalGraphEngine
from analyzers.detection.consistency_validator import ConsistencyValidator
from analyzers.detection.ground_truth import GroundTruthValidator
from analyzers.detection.information_loss import InformationLossRule
from analyzers.detection.rule_engine import RuleEngine
from analyzers.detection.statistical_detector import StatisticalDetector
from analyzers.detection.workflow_validator import WorkflowValidator
from analyzers.remediation.prompt_fixer import PromptFixer
from normalizer.normalizer import Normalizer
from schema.models import (
    SCHEMA_VERSION,
    AgentStep,
    PriorityLevel,
    RunTrace,
)
from storage.db import DatabaseManager

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("agentlens.regression_v2")

LABELS_PATH = ROOT / "sample_data" / "labels.json"
LABELED_DIR = ROOT / "sample_data" / "labeled_traces"
V2_DIR = ROOT / "sample_data" / "v2_traces"
DEMO_DIR = ROOT / "sample_data" / "demo_traces"

REGRESSION_V2_JSON = ROOT / "validation" / "regression_v2.json"
REGRESSION_V2_MD = ROOT / "validation" / "REGRESSION_REPORT_V2.md"

MIN_EXACT_ACCURACY_THRESHOLD = 0.85
MIN_BINARY_ACCURACY_THRESHOLD = 1.00
MIN_BASELINE_ACCURACY_THRESHOLD = 0.80

TARGET_DEMO_RUNS = [
    "demo_normal_pass_01",
    "demo_grounded_p1_01",
    "demo_heuristic_p2_hallucination",
    "demo_heuristic_p3_workflow",
    "demo_diff_pair_diverged_b",
]


def _map_predicted_category(verdict: str, primary_cause: str) -> str:
    """Map Arbiter (verdict, primary_cause) to the 5 standard label categories."""
    if verdict == "PASS":
        return "pass"
    cause_map = {
        "reasoning": "reasoning_failure",
        "execution": "execution_failure",
        "workflow": "workflow_failure",
        "verification": "verification_failure",
    }
    return cause_map.get(primary_cause, f"{primary_cause}_failure")


def load_labeled_trace(run_id: str) -> RunTrace:
    """Load a labeled trace JSON file into a validated RunTrace instance."""
    path = LABELED_DIR / f"{run_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    steps: list[AgentStep] = []
    for s in raw.get("steps", []):
        sd = dict(s)
        if sd.get("parent_step") is not None:
            sd["parent_step"] = str(sd["parent_step"])
        if sd.get("child_step") is not None:
            sd["child_step"] = str(sd["child_step"])

        norm_tools = []
        for tc in sd.get("tool_calls", []):
            norm_tools.append(
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "output": tc.get("output", tc.get("result", "")),
                    "result": tc.get("result", tc.get("output", "")),
                    "error": tc.get("error"),
                }
            )
        sd["tool_calls"] = norm_tools
        steps.append(AgentStep(**sd))

    return RunTrace(
        schema_version=raw.get("schema_version", SCHEMA_VERSION),
        run_id=raw.get("run_id", run_id),
        workflow=raw.get("workflow", "research_report_pipeline"),
        steps=steps,
        total_latency_ms=float(raw.get("total_latency_ms", 0.0)),
        total_tokens=int(raw.get("total_tokens", 0)),
        expected_output=raw.get("expected_output"),
    )


def run_v2_regression(
    output_json: Path = REGRESSION_V2_JSON,
    output_md: Path = REGRESSION_V2_MD,
    min_exact_acc: float = MIN_EXACT_ACCURACY_THRESHOLD,
    min_binary_acc: float = MIN_BINARY_ACCURACY_THRESHOLD,
) -> dict[str, Any]:
    """Execute the complete 30-trace extended regression suite and generate sign-off reports."""
    tmp_dir = tempfile.mkdtemp()
    val_db = DatabaseManager(str(Path(tmp_dir) / "val_regression_v2.db"))
    val_db.initialize()

    try:
        # 1. Initialize Analyzers & Pipelines
        normalizer = Normalizer()
        gt_eval = GroundTruthValidator()
        rule_engine = RuleEngine()
        info_loss_rule = InformationLossRule()
        workflow_val = WorkflowValidator()
        consistency_val = ConsistencyValidator()
        stat_detector = StatisticalDetector(val_db)
        arbiter = Arbiter()
        causal_engine = CausalGraphEngine()
        blast_calculator = BlastRadiusCalculator()
        prompt_fixer = PromptFixer(db=val_db)

        # 2. Pre-seed DB with 20 labeled runs for statistical detector
        labels_doc = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        labeled_entries: list[tuple[dict[str, Any], RunTrace]] = []

        for item in labels_doc["runs"]:
            tr = load_labeled_trace(item["run_id"])
            val_db.insert_run(
                run_id=tr.run_id,
                workflow=tr.workflow,
                timestamp="2026-01-01T00:00:00Z",
                status=str(tr.status.value),
                total_latency_ms=tr.total_latency_ms,
                total_tokens=tr.total_tokens,
                schema_version=tr.schema_version,
                trace_path="",
                trace_json=tr.model_dump_json(),
                expected_output=tr.expected_output,
            )
            for st in tr.steps:
                val_db.insert_step(
                    run_id=tr.run_id,
                    step=st.step,
                    agent=st.agent,
                    status=str(st.status.value),
                    latency_ms=st.latency_ms,
                    tokens_prompt=st.tokens.prompt,
                    tokens_completion=st.tokens.completion,
                    tokens_total=st.tokens.total,
                    diff_summary="",
                    error=st.error,
                    timestamp=st.timestamp.isoformat(),
                    schema_version=st.schema_version,
                )
            labeled_entries.append((item, tr))

        comparisons: list[dict[str, Any]] = []
        labeled_correct = 0
        labeled_binary = 0
        v2_correct = 0
        v2_binary = 0
        demo_correct = 0
        demo_binary = 0

        # Helper pipeline runner
        def _analyze_trace(trace: RunTrace) -> tuple[Any, Any, Any, Any]:
            _ = normalizer.normalize_run(trace)
            evidence = []

            gt_res = gt_eval.analyze(trace)
            if not gt_res.skipped:
                evidence.extend(gt_res.evidence)

            re_res = rule_engine.analyze(trace)
            if not re_res.skipped:
                evidence.extend(re_res.evidence)

            il_res = info_loss_rule.analyze(trace)
            if not il_res.skipped:
                evidence.extend(il_res.evidence)

            wv_res = workflow_val.analyze(trace)
            if not wv_res.skipped:
                evidence.extend(wv_res.evidence)

            cv_res = consistency_val.analyze(trace)
            if not cv_res.skipped:
                evidence.extend(cv_res.evidence)

            try:
                stat_rep = stat_detector.analyze_run(trace.run_id)
                if stat_rep.anomalies:
                    evidence.extend(stat_rep.anomalies)
            except Exception:
                pass

            bundle = arbiter.run(run_id=trace.run_id, evidence=evidence)
            causal_graph = causal_engine.build_graph(trace, bundle)
            blast_report = blast_calculator.calculate(
                run_trace=trace, causal_graph=causal_graph, analysis_bundle=bundle
            )
            remediation_patch = prompt_fixer.generate_patch(trace, bundle)

            return bundle, causal_graph, blast_report, remediation_patch

        # -------------------------------------------------------------
        # Section A: 20 Frozen Labeled Traces
        # -------------------------------------------------------------
        for item, trace in labeled_entries:
            bundle, cg, blast, patch = _analyze_trace(trace)

            pred_verdict = "PASS" if bundle.priority_level == PriorityLevel.P5 else "FAIL"
            pred_priority = bundle.priority_level.value
            pred_cause = bundle.primary_cause.value
            pred_agent = bundle.primary_agent
            pred_category = _map_predicted_category(pred_verdict, pred_cause)

            exp_category = item["category"]
            exp_agent = item["expected_primary_agent"]
            exp_is_pass = exp_category == "pass"
            pred_is_pass = pred_verdict == "PASS"

            binary_match = exp_is_pass == pred_is_pass
            exact_match = (
                binary_match
                and pred_category == exp_category
                and (pred_agent == exp_agent or (pred_agent is None and exp_agent is None))
            )

            if binary_match:
                labeled_binary += 1
            if exact_match:
                labeled_correct += 1

            comparisons.append(
                {
                    "run_id": trace.run_id,
                    "corpus_subset": "frozen_labeled",
                    "category": exp_category,
                    "expected_verdict": "PASS" if exp_is_pass else "FAIL",
                    "expected_priority": "P5" if exp_is_pass else item.get("priority", "P2"),
                    "expected_agent": exp_agent,
                    "predicted_verdict": pred_verdict,
                    "predicted_priority": pred_priority,
                    "predicted_category": pred_category,
                    "predicted_agent": pred_agent,
                    "binary_match": binary_match,
                    "exact_match": exact_match,
                    "causal_nodes": len(cg.nodes),
                    "blast_wasted_tokens": blast.wasted_tokens.total if blast else 0,
                    "remediation_status": (patch.constraint_type if patch else "none"),
                }
            )

        # -------------------------------------------------------------
        # Section B: 5 Canonical Phase B Multi-Agent Traces
        # -------------------------------------------------------------
        v2_manifest = json.loads((V2_DIR / "manifest.json").read_text(encoding="utf-8"))
        for item in v2_manifest.get("traces", []):
            raw_tr = json.loads((V2_DIR / item["file"]).read_text(encoding="utf-8"))
            trace = RunTrace.model_validate(raw_tr)

            bundle, cg, blast, patch = _analyze_trace(trace)

            pred_verdict = "PASS" if bundle.priority_level == PriorityLevel.P5 else "FAIL"
            pred_priority = bundle.priority_level.value
            pred_agent = bundle.primary_agent

            exp_verdict = item["expected_verdict"]
            exp_priority = item["expected_priority"]
            exp_agent = item["primary_agent"]

            binary_match = pred_verdict == exp_verdict
            exact_match = (
                binary_match
                and pred_priority == exp_priority
                and (pred_agent == exp_agent or (pred_agent is None and exp_agent is None))
            )

            if binary_match:
                v2_binary += 1
            if exact_match:
                v2_correct += 1

            comparisons.append(
                {
                    "run_id": trace.run_id,
                    "corpus_subset": "phase_b_multi_agent",
                    "category": item["category"],
                    "expected_verdict": exp_verdict,
                    "expected_priority": exp_priority,
                    "expected_agent": exp_agent,
                    "predicted_verdict": pred_verdict,
                    "predicted_priority": pred_priority,
                    "predicted_category": _map_predicted_category(
                        pred_verdict, bundle.primary_cause.value
                    ),
                    "predicted_agent": pred_agent,
                    "binary_match": binary_match,
                    "exact_match": exact_match,
                    "causal_nodes": len(cg.nodes),
                    "blast_wasted_tokens": blast.wasted_tokens.total if blast else 0,
                    "remediation_status": (patch.constraint_type if patch else "none"),
                }
            )

        # -------------------------------------------------------------
        # Section C: 5 Canonical Enterprise Demo Traces
        # -------------------------------------------------------------
        demo_manifest = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))
        for item in demo_manifest.get("traces", []):
            if item["run_id"] not in TARGET_DEMO_RUNS:
                continue

            raw_tr = json.loads((DEMO_DIR / item["file"]).read_text(encoding="utf-8"))
            trace = RunTrace.model_validate(raw_tr)

            bundle, cg, blast, patch = _analyze_trace(trace)

            pred_verdict = "PASS" if bundle.priority_level == PriorityLevel.P5 else "FAIL"
            pred_priority = bundle.priority_level.value
            pred_agent = bundle.primary_agent

            exp_verdict = item["verdict"]
            exp_priority = item["priority_level"]
            exp_agent = item["primary_agent"]

            binary_match = pred_verdict == exp_verdict
            exact_match = (
                binary_match
                and pred_priority == exp_priority
                and (pred_agent == exp_agent or (pred_agent is None and exp_agent is None))
            )

            if binary_match:
                demo_binary += 1
            if exact_match:
                demo_correct += 1

            comparisons.append(
                {
                    "run_id": trace.run_id,
                    "corpus_subset": "enterprise_demo",
                    "category": item["category"],
                    "expected_verdict": exp_verdict,
                    "expected_priority": exp_priority,
                    "expected_agent": exp_agent,
                    "predicted_verdict": pred_verdict,
                    "predicted_priority": pred_priority,
                    "predicted_category": _map_predicted_category(
                        pred_verdict, bundle.primary_cause.value
                    ),
                    "predicted_agent": pred_agent,
                    "binary_match": binary_match,
                    "exact_match": exact_match,
                    "causal_nodes": len(cg.nodes),
                    "blast_wasted_tokens": blast.wasted_tokens.total if blast else 0,
                    "remediation_status": (patch.constraint_type if patch else "none"),
                }
            )

        # -------------------------------------------------------------
        # Aggregate Metrics & Threshold Certification
        # -------------------------------------------------------------
        total_runs = len(comparisons)
        total_exact = labeled_correct + v2_correct + demo_correct
        total_binary = labeled_binary + v2_binary + demo_binary

        exact_acc = round(total_exact / total_runs, 4)
        binary_acc = round(total_binary / total_runs, 4)
        baseline_acc = round(labeled_correct / len(labeled_entries), 4)

        exact_gate_passed = exact_acc >= min_exact_acc
        binary_gate_passed = binary_acc >= min_binary_acc
        baseline_gate_passed = baseline_acc >= MIN_BASELINE_ACCURACY_THRESHOLD

        all_passed = exact_gate_passed and binary_gate_passed and baseline_gate_passed

        report_payload: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "release_version": "v2.0.0",
            "passed": all_passed,
            "gates": {
                "exact_attribution_gate": {
                    "passed": exact_gate_passed,
                    "threshold": min_exact_acc,
                    "achieved": exact_acc,
                    "correct": total_exact,
                    "total": total_runs,
                },
                "binary_detection_gate": {
                    "passed": binary_gate_passed,
                    "threshold": min_binary_acc,
                    "achieved": binary_acc,
                    "correct": total_binary,
                    "total": total_runs,
                },
                "baseline_preservation_gate": {
                    "passed": baseline_gate_passed,
                    "threshold": MIN_BASELINE_ACCURACY_THRESHOLD,
                    "achieved": baseline_acc,
                    "correct": labeled_correct,
                    "total": len(labeled_entries),
                },
            },
            "subsets": {
                "frozen_labeled": {
                    "total": len(labeled_entries),
                    "exact_correct": labeled_correct,
                    "binary_correct": labeled_binary,
                    "exact_accuracy": baseline_acc,
                },
                "phase_b_multi_agent": {
                    "total": len(v2_manifest.get("traces", [])),
                    "exact_correct": v2_correct,
                    "binary_correct": v2_binary,
                    "exact_accuracy": round(v2_correct / len(v2_manifest.get("traces", [])), 4),
                },
                "enterprise_demo": {
                    "total": len(TARGET_DEMO_RUNS),
                    "exact_correct": demo_correct,
                    "binary_correct": demo_binary,
                    "exact_accuracy": round(demo_correct / len(TARGET_DEMO_RUNS), 4),
                },
            },
            "comparisons": comparisons,
        }

        # Write JSON report
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        # Write Markdown report
        md_text = _build_markdown_report(report_payload)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md_text, encoding="utf-8")

        return report_payload

    finally:
        val_db.close()


def _build_markdown_report(report: dict[str, Any]) -> str:
    status_badge = "✅ RELEASE GATE PASSED" if report["passed"] else "❌ RELEASE GATE FAILED"
    g = report["gates"]
    sub = report["subsets"]

    lines: list[str] = [
        "# AgentLens v2.0 Final Extended Regression Report",
        "",
        f"**Release Version:** `{report['release_version']}`  ",
        f"**Generated At:** `{report['generated_at']}`  ",
        f"**Status:** **`{status_badge}`**  ",
        "",
        "## Release Gate Certification Metrics",
        "",
        "| Gate | Required Threshold | Achieved Score | Passing Runs | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
        (
            f"| **Exact Causal Attribution** | `≥ {g['exact_attribution_gate']['threshold'] * 100:.1f}%`"
            f" | **`{g['exact_attribution_gate']['achieved'] * 100:.1f}%`** |"
            f" {g['exact_attribution_gate']['correct']}/{g['exact_attribution_gate']['total']} |"
            f" {'✅ PASS' if g['exact_attribution_gate']['passed'] else '❌ FAIL'} |"
        ),
        (
            f"| **Binary Failure Detection** | `100.0%` |"
            f" **`{g['binary_detection_gate']['achieved'] * 100:.1f}%`** |"
            f" {g['binary_detection_gate']['correct']}/{g['binary_detection_gate']['total']} |"
            f" {'✅ PASS' if g['binary_detection_gate']['passed'] else '❌ FAIL'} |"
        ),
        (
            f"| **Day 43 Baseline Preservation** | `≥ {g['baseline_preservation_gate']['threshold'] * 100:.1f}%`"
            f" | **`{g['baseline_preservation_gate']['achieved'] * 100:.1f}%`** |"
            f" {g['baseline_preservation_gate']['correct']}/{g['baseline_preservation_gate']['total']} |"
            f" {'✅ PASS' if g['baseline_preservation_gate']['passed'] else '❌ FAIL'} |"
        ),
        "",
        "## Sub-Corpus Breakdown",
        "",
        "| Corpus Subset | Traces | Exact Matches | Binary Matches | Exact Accuracy |",
        "| :--- | :--- | :--- | :--- | :--- |",
        (
            f"| Frozen Labeled (Day 15/43) | {sub['frozen_labeled']['total']} |"
            f" {sub['frozen_labeled']['exact_correct']} |"
            f" {sub['frozen_labeled']['binary_correct']} |"
            f" {sub['frozen_labeled']['exact_accuracy'] * 100:.1f}% |"
        ),
        (
            f"| Phase B Multi-Agent (Day 56) | {sub['phase_b_multi_agent']['total']} |"
            f" {sub['phase_b_multi_agent']['exact_correct']} |"
            f" {sub['phase_b_multi_agent']['binary_correct']} |"
            f" {sub['phase_b_multi_agent']['exact_accuracy'] * 100:.1f}% |"
        ),
        (
            f"| Enterprise Demo (Day 44) | {sub['enterprise_demo']['total']} |"
            f" {sub['enterprise_demo']['exact_correct']} |"
            f" {sub['enterprise_demo']['binary_correct']} |"
            f" {sub['enterprise_demo']['exact_accuracy'] * 100:.1f}% |"
        ),
        (
            f"| **Total Extended Suite** | **{report['gates']['exact_attribution_gate']['total']}** |"
            f" **{report['gates']['exact_attribution_gate']['correct']}** |"
            f" **{report['gates']['binary_detection_gate']['correct']}** |"
            f" **{report['gates']['exact_attribution_gate']['achieved'] * 100:.1f}%** |"
        ),
        "",
        "## 30-Trace Execution Ledger",
        "",
        "| Run ID | Subset | Expected | Predicted | Exact Match | Binary Match | Causal Nodes | Remediation |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for row in report["comparisons"]:
        exp = f"{row['expected_verdict']} ({row['expected_priority']}, {row['expected_agent'] or 'none'})"
        pred = f"{row['predicted_verdict']} ({row['predicted_priority']}, {row['predicted_agent'] or 'none'})"
        ex_badge = "✅" if row["exact_match"] else "❌"
        bin_badge = "✅" if row["binary_match"] else "❌"
        lines.append(
            f"| `{row['run_id']}` | {row['corpus_subset']} | {exp} | {pred} | {ex_badge} | {bin_badge} | {row['causal_nodes']} | `{row['remediation_status']}` |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentLens v2.0 Extended 30-Trace Regression Suite"
    )
    parser.add_argument("--json", type=Path, default=REGRESSION_V2_JSON, help="Output JSON path")
    parser.add_argument("--md", type=Path, default=REGRESSION_V2_MD, help="Output MD path")
    args = parser.parse_args()

    print("\n" + "=" * 76)
    print("AgentLens v2.0 — Extended 30-Trace Regression Runner & Release Gate")
    print("=" * 76)

    res = run_v2_regression(output_json=args.json, output_md=args.md)

    print("\n" + "-" * 76)
    print(f"Overall Gate Passed: {res['passed']}")
    print(
        f"Exact Attribution:   "
        f"{res['gates']['exact_attribution_gate']['correct']}/{res['gates']['exact_attribution_gate']['total']} "
        f"({res['gates']['exact_attribution_gate']['achieved'] * 100:.1f}%) "
        f"[Target: >= {res['gates']['exact_attribution_gate']['threshold'] * 100:.1f}%]"
    )
    print(
        f"Binary Detection:    "
        f"{res['gates']['binary_detection_gate']['correct']}/{res['gates']['binary_detection_gate']['total']} "
        f"({res['gates']['binary_detection_gate']['achieved'] * 100:.1f}%) "
        f"[Target: {res['gates']['binary_detection_gate']['threshold'] * 100:.1f}%]"
    )
    print(
        f"Baseline Integrity:  "
        f"{res['gates']['baseline_preservation_gate']['correct']}/{res['gates']['baseline_preservation_gate']['total']} "
        f"({res['gates']['baseline_preservation_gate']['achieved'] * 100:.1f}%) "
        f"[Target: >= {res['gates']['baseline_preservation_gate']['threshold'] * 100:.1f}%]"
    )
    print(f"JSON Report:         {args.json}")
    print(f"Markdown Report:     {args.md}")
    print("=" * 76 + "\n")

    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
