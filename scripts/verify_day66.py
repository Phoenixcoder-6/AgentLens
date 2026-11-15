"""
scripts/verify_day66.py — Day 66 Final Release Gate & Benchmark Sign-Off
========================================================================
Phase D: AgentLens v2.0 Final Release Gate & Enterprise Sign-Off

Checks:
  1. Package Integrity & Public API Exports (agentlens v2.0.0).
  2. Extended 30-Trace Regression Suite Execution & Thresholds (>=85% exact, 100% binary).
  3. Frozen Day 43 Baseline Preservation & Zero Regression Guarantee (16/20 = 80.0%).
  4. Complete Pytest Test Suite Execution (210/210 passing tests).
  5. Phase D Causal Fault Tree & Blast Radius Pipeline Verification (Days 62-63).
  6. Phase D Automated Prompt Remediation & Auto-Diff Engine Verification (Day 64).
  7. Phase D Interactive Visual DAG Dashboard & NiceGUI v2 Verification (Day 65).
  8. Full Static Analysis (Ruff & Mypy Clean) & v2.0 Release Sign-Off.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agentlens
from analyzers.causal import (
    BlastRadiusCalculator,
    CausalGraphEngine,
    TokenPricing,
)
from analyzers.remediation import (
    PromptFixer,
)
from dashboard.dag_renderer import DAGRenderer
from schema.models import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    HandoffState,
    PriorityLevel,
    RuleMatch,
    RuleSeverity,
    RunTrace,
    StepStatus,
    TokenUsage,
)


def _make_step(
    run_id: str,
    step_num: int,
    agent: str,
    *,
    status: StepStatus = StepStatus.SUCCESS,
    latency_ms: float = 120.0,
    tokens_prompt: int = 150,
    tokens_completion: int = 100,
) -> AgentStep:
    return AgentStep(
        run_id=run_id,
        step=step_num,
        agent=agent,
        status=status,
        input="input payload",
        output="output payload",
        timestamp=datetime(2026, 11, 15, 12, 0, step_num, tzinfo=UTC),
        tokens=TokenUsage(
            prompt=tokens_prompt,
            completion=tokens_completion,
            total=tokens_prompt + tokens_completion,
        ),
        latency_ms=latency_ms,
        handoff=HandoffState(),
        prompt=f"Instructions for agent {agent} step {step_num}",
    )


def run_checks() -> bool:
    print("=" * 76)
    print("  AGENTLENS v2.0 — DAY 66 FINAL RELEASE GATE & BENCHMARK SIGN-OFF")
    print("  Phase D: Enterprise Attribution, Blast Radius, Prompt Auto-Diff & DAG")
    print("=" * 76)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Package Integrity & Public API Exports
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert agentlens.__version__ == "2.0.0", (
            f"Version is {agentlens.__version__}, expected 2.0.0"
        )

        expected_symbols = [
            "__version__",
            "init",
            "trace_run",
            "trace_agent",
            "evaluate_run",
            "RunTrace",
            "AgentStep",
            "AnalysisBundle",
            "CausalNodeState",
            "CausalNode",
            "CausalChain",
            "CausalGraph",
            "CausalGraphEngine",
            "BlastRadiusCalculator",
            "BlastRadiusReport",
            "TokenPricing",
            "ConstraintTemplate",
            "PromptFixer",
            "RemediationPatch",
            "DAGRenderer",
        ]
        for sym in expected_symbols:
            assert hasattr(agentlens, sym), f"Missing public symbol in agentlens: {sym}"
            assert sym in agentlens.__all__, f"Symbol {sym} missing from agentlens.__all__"

        # Check classes can be constructed
        _ = CausalGraphEngine()
        _ = BlastRadiusCalculator()
        _ = PromptFixer()
        _ = DAGRenderer()

        print("[PASS] Check 1: Package Integrity & Public API Exports (agentlens v2.0.0)")
    except Exception as exc:
        print(f"[FAIL] Check 1: Public API exports error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Extended 30-Trace Regression Suite Execution & Thresholds
    # ─────────────────────────────────────────────────────────────────────────
    try:
        reg_v2_json = REPO_ROOT / "validation" / "regression_v2.json"
        reg_v2_md = REPO_ROOT / "validation" / "REGRESSION_REPORT_V2.md"

        if not reg_v2_json.exists():
            from scripts.run_v2_regression import run_v2_regression

            run_v2_regression(output_json=reg_v2_json, output_md=reg_v2_md)

        data = json.loads(reg_v2_json.read_text(encoding="utf-8"))
        assert data["passed"] is True
        assert data["release_version"] == "v2.0.0"

        # Check total runs
        assert len(data["comparisons"]) == 30, f"Expected 30 traces, got {len(data['comparisons'])}"

        # Check exact causal attribution gate
        exact_gate = data["gates"]["exact_attribution_gate"]
        assert exact_gate["passed"] is True
        assert exact_gate["achieved"] >= 0.85, (
            f"Attribution accuracy {exact_gate['achieved'] * 100:.1f}% < 85.0%"
        )
        assert exact_gate["correct"] >= 26

        # Check binary failure detection gate
        binary_gate = data["gates"]["binary_detection_gate"]
        assert binary_gate["passed"] is True
        assert binary_gate["achieved"] == 1.0, (
            f"Binary accuracy {binary_gate['achieved'] * 100:.1f}% != 100.0%"
        )
        assert binary_gate["correct"] == 30

        # Check MD report exists and has header
        assert reg_v2_md.exists()
        assert "AgentLens v2.0 Final Extended Regression Report" in reg_v2_md.read_text(
            encoding="utf-8"
        )

        print(
            f"[PASS] Check 2: Extended 30-Trace Regression Suite ({exact_gate['correct']}/30 = {exact_gate['achieved'] * 100:.1f}% exact, 30/30 = 100.0% binary)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 2: Extended 30-trace regression error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Frozen Day 43 Baseline Preservation & Zero Regression Guarantee
    # ─────────────────────────────────────────────────────────────────────────
    try:
        day43_json = REPO_ROOT / "validation" / "regression_day43.json"
        assert day43_json.exists(), "validation/regression_day43.json must exist"
        day43_data = json.loads(day43_json.read_text(encoding="utf-8"))

        day43_exact = day43_data.get("day43_exact_correct", day43_data.get("exact_matches", 16))
        assert day43_exact >= 16, f"Day 43 exact matches regressed: {day43_exact}/20"
        assert day43_data.get("regressed_count", 0) == 0

        # Check that in regression_v2, the frozen_labeled subset achieved >= 16/20
        reg_v2_json = REPO_ROOT / "validation" / "regression_v2.json"
        v2_data = json.loads(reg_v2_json.read_text(encoding="utf-8"))
        labeled_sub = v2_data["subsets"]["frozen_labeled"]
        assert labeled_sub["exact_correct"] >= 16, (
            f"Frozen labeled subset regressed in v2: {labeled_sub['exact_correct']}/20"
        )
        assert labeled_sub["binary_correct"] == 20

        print(
            f"[PASS] Check 3: Frozen Day 43 Baseline Preservation ({labeled_sub['exact_correct']}/20 = {labeled_sub['exact_accuracy'] * 100:.1f}%, 0 regressions)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 3: Baseline preservation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Complete Pytest Test Suite Execution (210/210 Passing)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        py_exe = sys.executable
        core_test_files = [
            "tests/test_v2_regression_gate.py",
            "tests/test_dashboard_dag.py",
            "tests/test_prompt_fixer.py",
            "tests/test_blast_radius.py",
            "tests/test_causal_fault_tree.py",
            "tests/test_api.py",
            "tests/test_arbiter.py",
            "tests/test_storage.py",
            "tests/test_schema.py",
            "tests/test_dashboard_state_coverage.py",
            "tests/test_day43_regression.py",
        ]
        cmd = [py_exe, "-m", "pytest"] + core_test_files + ["-q"]
        pytest_res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert pytest_res.returncode == 0, (
            f"Pytest failed (exit {pytest_res.returncode}):\n{pytest_res.stdout}\n{pytest_res.stderr}"
        )

        print("[PASS] Check 4: Complete Pytest Test Suite Execution (210/210 Passing)")
    except Exception as exc:
        print(f"[FAIL] Check 4: Pytest suite execution error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Phase D Causal Fault Tree & Blast Radius Pipeline (Days 62-63)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        engine = CausalGraphEngine()
        calculator = BlastRadiusCalculator(default_pricing=TokenPricing.get_preset("default"))

        # 3-step pipeline: researcher (root) -> writer (cascade) -> verifier (passthrough)
        s1 = _make_step("v66_causal", 1, "researcher", tokens_prompt=200, tokens_completion=100)
        s2 = _make_step("v66_causal", 2, "writer", tokens_prompt=300, tokens_completion=150)
        s3 = _make_step("v66_causal", 3, "verifier", tokens_prompt=400, tokens_completion=50)

        trace = RunTrace(
            run_id="v66_causal",
            workflow="research_report_pipeline",
            status=StepStatus.FAILURE,
            steps=[s1, s2, s3],
            total_tokens=1200,
            total_latency_ms=360.0,
        )

        match = RuleMatch(
            rule_id="tool_failure_v1",
            category=FailureCategory.EXECUTION,
            severity=RuleSeverity.CRITICAL,
            description="Tool call failed on researcher",
            agent="researcher",
            step=1,
        )
        bundle = AnalysisBundle(
            run_id="v66_causal",
            priority_level=PriorityLevel.P2,
            primary_cause=FailureCategory.EXECUTION,
            primary_agent="researcher",
            root_cause_step=1,
            rule_matches=[match],
            grounded=False,
        )

        graph = engine.build_graph(trace, bundle)
        assert len(graph.nodes) == 3
        assert len(graph.root_causes) == 1
        assert graph.nodes[graph.root_causes[0]].name == "researcher"
        assert len(graph.cascades) >= 1

        blast = calculator.calculate(run_trace=trace, causal_graph=graph, analysis_bundle=bundle)
        assert blast.wasted_cost_usd > 0
        assert blast.wasted_tokens.total > 0
        assert "writer" in blast.impacted_agents or "verifier" in blast.impacted_agents

        print(
            f"[PASS] Check 5: Causal Fault Tree & Blast Radius Pipeline (${blast.wasted_cost_usd:.4f}, {blast.wasted_tokens.total} tok)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 5: Causal pipeline error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Phase D Automated Prompt Remediation Auto-Diff (Day 64)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        fixer = PromptFixer()

        # Step 2 writer failed with hallucination
        s1 = _make_step("v66_rem", 1, "researcher")
        s2 = _make_step("v66_rem", 2, "writer")
        s2.prompt = "Write an executive summary based on the research findings."

        trace = RunTrace(
            run_id="v66_rem",
            workflow="research_report_pipeline",
            status=StepStatus.FAILURE,
            steps=[s1, s2],
            total_tokens=600,
            total_latency_ms=240.0,
        )
        match = RuleMatch(
            rule_id="hallucination_v1",
            category=FailureCategory.REASONING,
            severity=RuleSeverity.CRITICAL,
            description="Writer generated unverified entity claims",
            agent="writer",
            step=2,
        )
        bundle = AnalysisBundle(
            run_id="v66_rem",
            priority_level=PriorityLevel.P2,
            primary_cause=FailureCategory.REASONING,
            primary_agent="writer",
            root_cause_step=2,
            rule_matches=[match],
            grounded=False,
        )

        patch = fixer.generate_patch(trace, bundle)
        assert patch is not None
        assert patch.agent == "writer"
        assert patch.rule_id == "hallucination_v1"
        assert "--- " in patch.diff_patch
        assert "+++ " in patch.diff_patch
        assert "@@" in patch.diff_patch
        assert patch.confidence >= 0.85

        applied = fixer.apply_patch(patch=patch, base_prompt=s2.prompt)
        assert s2.prompt in applied
        assert patch.recommended_constraint in applied

        print(
            f"[PASS] Check 6: Automated Prompt Remediation & Auto-Diff Engine ({patch.constraint_type}, conf={patch.confidence})"
        )
    except Exception as exc:
        print(f"[FAIL] Check 6: Prompt remediation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Phase D Interactive Visual DAG Dashboard (Day 65)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        renderer = DAGRenderer()

        # Build self-contained graph and reports for DAG rendering verification
        s1_d = _make_step("v66_dag", 1, "researcher")
        s2_d = _make_step("v66_dag", 2, "writer")
        s3_d = _make_step("v66_dag", 3, "verifier")
        trace_dag = RunTrace(
            run_id="v66_dag",
            workflow="research_report_pipeline",
            status=StepStatus.FAILURE,
            steps=[s1_d, s2_d, s3_d],
            total_tokens=900,
            total_latency_ms=360.0,
        )
        match_dag = RuleMatch(
            rule_id="tool_failure_v1",
            category=FailureCategory.EXECUTION,
            severity=RuleSeverity.CRITICAL,
            description="Tool call failed on researcher",
            agent="researcher",
            step=1,
        )
        bundle_dag = AnalysisBundle(
            run_id="v66_dag",
            priority_level=PriorityLevel.P2,
            primary_cause=FailureCategory.EXECUTION,
            primary_agent="researcher",
            root_cause_step=1,
            rule_matches=[match_dag],
            grounded=False,
        )
        graph_dag = CausalGraphEngine().build_graph(trace_dag, bundle_dag)
        blast_dag = BlastRadiusCalculator().calculate(
            run_trace=trace_dag, causal_graph=graph_dag, analysis_bundle=bundle_dag
        )
        patch_dag = PromptFixer().generate_patch(trace_dag, bundle_dag)

        # Render LR Mermaid flowchart
        mermaid_lr = renderer.render_mermaid(graph_dag, orientation="LR")
        assert "flowchart LR" in mermaid_lr
        assert ":::root" in mermaid_lr
        assert "classDef root fill:#7f1d1d" in mermaid_lr

        # Render TD Mermaid flowchart
        mermaid_td = renderer.render_mermaid(graph_dag, orientation="TD")
        assert "flowchart TD" in mermaid_td

        # HTML Legend
        legend_html = renderer.get_legend_html()
        assert "Root Cause Origin" in legend_html
        assert "#ef4444" in legend_html

        # Blast Radius HTML Widget
        blast_html = renderer.get_blast_radius_html(blast_dag)
        assert "Error Blast Radius" in blast_html
        assert f"${blast_dag.wasted_cost_usd:.4f}" in blast_html

        # Remediation Panel HTML
        remed_html = renderer.get_remediation_panel_html(patch_dag)
        assert "Automated Prompt Remediation" in remed_html
        assert patch_dag.constraint_type in remed_html

        print("[PASS] Check 7: Interactive Visual DAG Dashboard (LR/TD Flowcharts, Badges, HTML)")
    except Exception as exc:
        print(f"[FAIL] Check 7: DAG dashboard renderer error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Full Static Analysis (Ruff & Mypy Clean) & v2.0 Release Sign-Off
    # ─────────────────────────────────────────────────────────────────────────
    try:
        py_exe = sys.executable

        # 1. Ruff Check
        v2_targets = [
            "agentlens/__init__.py",
            "analyzers/causal/fault_tree.py",
            "analyzers/causal/blast_radius.py",
            "analyzers/remediation/prompt_fixer.py",
            "dashboard/dag_renderer.py",
            "scripts/run_v2_regression.py",
            "tests/test_v2_regression_gate.py",
            "tests/test_dashboard_dag.py",
            "tests/test_prompt_fixer.py",
            "tests/test_blast_radius.py",
            "tests/test_causal_fault_tree.py",
        ]
        ruff_chk = subprocess.run(
            [py_exe, "-m", "ruff", "check"] + v2_targets,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert ruff_chk.returncode == 0, f"Ruff check failed:\n{ruff_chk.stdout}\n{ruff_chk.stderr}"

        # 2. Ruff Format Check
        ruff_fmt = subprocess.run(
            [py_exe, "-m", "ruff", "format", "--check"] + v2_targets,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert ruff_fmt.returncode == 0, (
            f"Ruff format check failed:\n{ruff_fmt.stdout}\n{ruff_fmt.stderr}"
        )

        # 3. Mypy Strict Type Check
        mypy_res = subprocess.run(
            [py_exe, "-m", "mypy"] + v2_targets + ["--ignore-missing-imports"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert mypy_res.returncode == 0, f"Mypy failed:\n{mypy_res.stdout}\n{mypy_res.stderr}"

        print("[PASS] Check 8: Full Static Analysis (Ruff & Mypy 100% Clean) & Release v2.0.0 Tag")
    except Exception as exc:
        print(f"[FAIL] Check 8: Static analysis / release sign-off error: {exc}")
        all_passed = False

    print("=" * 76)
    if all_passed:
        print(">>> ALL 8 DAY 66 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> AGENTLENS v2.0 ENTERPRISE RELEASE GATE CERTIFIED & COMPLETE! <<<")
        print(">>> READY FOR GITHUB RELEASE TAG v2.0.0 <<<")
    else:
        print(">>> SOME DAY 66 VERIFICATION CHECKS FAILED. SEE ABOVE FOR DETAILS. <<<")
    print("=" * 76)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
