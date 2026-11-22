"""
scripts/verify_day56.py — Day 56 Verification Gate
===================================================
Phase B Sign-Off: Universal Core Verification & Accuracy Gate (AgentLens v2.0)

Checks:
  1. Multi-Agent Canonical v2 Traces & Manifest Verification (5 canonical traces).
  2. Multi-Agent Reflection Loop Tracking & Alignment (trace_v2_loop_reflection_pass -> PASS/P5).
  3. Concurrent Parallel Fork DAG Execution (trace_v2_parallel_fork_divergence -> FAIL/P2 on financial_analyst).
  4. Semantic 1-to-1 Entity Substitution Detection (trace_v2_entity_substitution_swap -> FAIL/P2 on writer).
  5. Semantic Relation Contradiction Detection (trace_v2_semantic_contradiction -> FAIL/P2 on writer).
  6. Cyclic Non-Convergence Deadlock Detection (trace_v2_loop_deadlock_unresolved -> FAIL/P2 on researcher).
  7. Automated Phase B Accuracy Evaluation Engine (validation/results_phase_b.json, >= 90.0% accuracy).
  8. Static Analysis (Ruff/Mypy Clean) & Day 43 Frozen Benchmark Gate (16/20 = 80.0%).
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyzers.arbiter import Arbiter
from analyzers.detection.rule_engine import RuleEngine
from analyzers.semantic.entity_tracker import EntityTracker
from analyzers.semantic.nli_engine import get_nli_engine
from diff_engine import align_spans
from sample_data.generate_v2_traces import DEFAULT_V2_OUTPUT_DIR, generate_v2_traces
from schema.models import RunTrace
from scripts.evaluate_phase_b_accuracy import run_phase_b_evaluation


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 56 VERIFICATION GATE")
    print("  Phase B Universal Core Verification & Accuracy Gate")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Multi-Agent Canonical v2 Traces & Manifest Verification
    # ─────────────────────────────────────────────────────────────────────────
    try:
        traces_dir = REPO_ROOT / DEFAULT_V2_OUTPUT_DIR
        manifest_file = traces_dir / "manifest.json"
        if not manifest_file.exists():
            generate_v2_traces(output_dir=traces_dir)

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["total_traces"] == 5, f"Expected 5 traces, got {manifest['total_traces']}"
        assert len(manifest["traces"]) == 5

        categories = {t["category"] for t in manifest["traces"]}
        expected_cats = {
            "loop_reflection",
            "parallel_fork",
            "entity_substitution",
            "semantic_contradiction",
            "loop_deadlock",
        }
        assert categories == expected_cats, f"Unexpected categories: {categories}"

        for rec in manifest["traces"]:
            t_path = traces_dir / rec["file"]
            assert t_path.exists(), f"Trace file {t_path} does not exist"
            tr = RunTrace.model_validate_json(t_path.read_text(encoding="utf-8"))
            assert tr.run_id == rec["run_id"]
            assert len(tr.steps) == rec["steps_count"]
            assert len(tr.spans) == rec["spans_count"]

        print("[PASS] Check 1: Multi-Agent Canonical v2 Traces & Manifest Verification")
    except Exception as exc:
        print(f"[FAIL] Check 1: Manifest verification failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Multi-Agent Reflection Loop Tracking & Alignment
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_path = REPO_ROOT / DEFAULT_V2_OUTPUT_DIR / "trace_v2_loop_reflection_pass.json"
        tr_loop = RunTrace.model_validate_json(t_path.read_text(encoding="utf-8"))

        assert len(tr_loop.steps) == 4
        assert len(tr_loop.spans) == 4

        # Span alignment & loop iteration discovery
        alignment = align_spans(tr_loop.spans, tr_loop.spans)
        assert alignment.matched_count == 4
        assert alignment.get_loop_iterations() == {0, 1}

        # Arbiter verdict: PASS (P5)
        res = RuleEngine().analyze(tr_loop)
        assert len(res.evidence) == 0, f"Expected 0 rules, got {len(res.evidence)}"
        bundle = Arbiter().run(run_id=tr_loop.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P5"
        assert bundle.primary_agent is None

        print("[PASS] Check 2: Multi-Agent Reflection Loop Tracking & Alignment (PASS/P5)")
    except Exception as exc:
        print(f"[FAIL] Check 2: Reflection loop check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Concurrent Parallel Fork DAG Execution
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_path = REPO_ROOT / DEFAULT_V2_OUTPUT_DIR / "trace_v2_parallel_fork_divergence.json"
        tr_fork = RunTrace.model_validate_json(t_path.read_text(encoding="utf-8"))

        assert len(tr_fork.steps) == 4
        # Coordinator step 1 -> Market analyst (2) and Financial analyst (3) -> Synthesizer (4)
        assert tr_fork.steps[1].parent_step == "1"
        assert tr_fork.steps[2].parent_step == "1"
        assert tr_fork.steps[3].parent_step == "2,3"

        # Arbiter verdict: tool_failure_v1 on financial_analyst (P2)
        res = RuleEngine().analyze(tr_fork)
        rules = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "tool_failure_v1" in rules

        bundle = Arbiter().run(run_id=tr_fork.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "financial_analyst"

        print(
            "[PASS] Check 3: Concurrent Parallel Fork DAG Execution (FAIL/P2 on financial_analyst)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 3: Parallel fork check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Semantic 1-to-1 Entity Substitution Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_path = REPO_ROOT / DEFAULT_V2_OUTPUT_DIR / "trace_v2_entity_substitution_swap.json"
        tr_sub = RunTrace.model_validate_json(t_path.read_text(encoding="utf-8"))

        delta = EntityTracker.compare_steps(tr_sub.steps[0], tr_sub.steps[1])
        assert delta.substitution_detected is True
        assert delta.invented_entities == {"moderna", "pfizer", "biontech"}
        assert delta.dropped_entities == {"microsoft", "amazon", "google"}

        res = RuleEngine().analyze(tr_sub)
        rules = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "hallucination_v2" in rules

        bundle = Arbiter().run(run_id=tr_sub.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "writer"

        print("[PASS] Check 4: Semantic 1-to-1 Entity Substitution Detection (FAIL/P2 on writer)")
    except Exception as exc:
        print(f"[FAIL] Check 4: Entity substitution check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Semantic Relation Contradiction Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_path = REPO_ROOT / DEFAULT_V2_OUTPUT_DIR / "trace_v2_semantic_contradiction.json"
        tr_contra = RunTrace.model_validate_json(t_path.read_text(encoding="utf-8"))

        nli_engine = get_nli_engine()
        report = nli_engine.evaluate_steps(tr_contra.steps[0], tr_contra.steps[1])
        assert report.has_contradiction is True

        res = RuleEngine().analyze(tr_contra)
        rules = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "contradiction_v1" in rules

        bundle = Arbiter().run(run_id=tr_contra.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "writer"

        print("[PASS] Check 5: Semantic Relation Contradiction Detection (FAIL/P2 on writer)")
    except Exception as exc:
        print(f"[FAIL] Check 5: Relation contradiction check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Cyclic Non-Convergence Deadlock Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_path = REPO_ROOT / DEFAULT_V2_OUTPUT_DIR / "trace_v2_loop_deadlock_unresolved.json"
        tr_deadlock = RunTrace.model_validate_json(t_path.read_text(encoding="utf-8"))

        assert len(tr_deadlock.steps) == 6

        res = RuleEngine().analyze(tr_deadlock)
        rules = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "researcher_quality_v1" in rules

        bundle = Arbiter().run(run_id=tr_deadlock.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "researcher"

        print("[PASS] Check 6: Cyclic Non-Convergence Deadlock Detection (FAIL/P2 on researcher)")
    except Exception as exc:
        print(f"[FAIL] Check 6: Loop deadlock check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Automated Phase B Accuracy Evaluation Engine
    # ─────────────────────────────────────────────────────────────────────────
    try:
        eval_res = run_phase_b_evaluation()
        metrics = eval_res["metrics"]
        assert eval_res["total_runs"] == 5
        assert metrics["exact_attribution_correct"] == 5
        assert metrics["exact_attribution_accuracy"] == 1.0
        assert metrics["binary_detection_accuracy"] == 1.0
        assert metrics["passed"] is True

        assert (REPO_ROOT / "validation" / "results_phase_b.json").exists()
        assert (REPO_ROOT / "validation" / "EVALUATION_REPORT_PHASE_B.md").exists()

        print("[PASS] Check 7: Automated Phase B Accuracy Evaluation Engine (100.0% >= 90.0%)")
    except Exception as exc:
        print(f"[FAIL] Check 7: Accuracy evaluation runner failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis & Day 43 Frozen Benchmark Gate
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # 1. Ruff linting
        ruff_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "sample_data/generate_v2_traces.py",
                "scripts/evaluate_phase_b_accuracy.py",
                "tests/test_day56_phase_b_gate.py",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if ruff_res.returncode != 0:
            raise RuntimeError(f"Ruff check failed: {ruff_res.stdout} {ruff_res.stderr}")

        # 2. Mypy type-checking
        mypy_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "sample_data/generate_v2_traces.py",
                "scripts/evaluate_phase_b_accuracy.py",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if mypy_res.returncode != 0:
            raise RuntimeError(f"Mypy check failed: {mypy_res.stdout} {mypy_res.stderr}")

        # 3. Day 43 Benchmark Regression Check
        bench_script = REPO_ROOT / "scripts" / "run_day43_regression.py"
        bench_res = subprocess.run(
            [sys.executable, str(bench_script)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if bench_res.returncode != 0 or "16/20 (80.0%)" not in bench_res.stdout:
            raise RuntimeError(
                f"Day 43 benchmark regression failed:\n{bench_res.stdout}\n{bench_res.stderr}"
            )

        print(
            "[PASS] Check 8: Static Analysis (Ruff/Mypy clean) & Day 43 Benchmark Preserved (16/20 = 80.0%)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 8: Static analysis / regression gate failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    if all_passed:
        print(">>> ALL 8 DAY 56 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> PHASE B UNIVERSAL CORE VERIFICATION COMPLETE & CERTIFIED! <<<")
    else:
        print(">>> SOME DAY 56 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
