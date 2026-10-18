"""
scripts/verify_day54.py — Day 54 Verification Gate
===================================================
Phase B: Semantic Entity Set Overlap Engine (AgentLens v2.0)

Checks:
  1. Multi-Pattern Named Entity Extraction (proper nouns, codes, acronyms, metrics)
  2. Mathematical Set Difference & Preservation Metrics (E_invented, E_dropped, E_shared)
  3. Entity Substitution Detection (where scalar counts match 3 vs 3)
  4. Zero False Positives on Faithful Paraphrasing Synthesis
  5. Step-Level & Trace-Level Entity Analysis (compare_steps, analyze_trace)
  6. RuleEngine Integration: hallucination_v2 Firing on Substitution Bug
  7. Execution Failure Precedence (Suppresses Hallucination on Tool Errors)
  8. Static Analysis (Ruff & Mypy clean) & Day 43 Frozen Benchmark Gate (16/20 = 80.0%)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from analyzers.detection.rule_engine import RuleEngine
from analyzers.semantic.entity_tracker import (
    EntityDeltaReport,
    EntityTracker,
    compare_entity_sets,
    extract_entities,
    normalize_entity,
)
from schema.models import AgentStep, FailureCategory, RunTrace, StepStatus

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 54 VERIFICATION GATE")
    print("  Semantic Entity Set Overlap Engine (v2.0 Phase B)")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Multi-Pattern Named Entity Extraction
    # ─────────────────────────────────────────────────────────────────────────
    try:
        text = (
            "Satya Nadella announced Microsoft Azure enhancements with OpenAI "
            "complying with FDA guidelines, outperforming Claude-3-Opus and GPT-4o."
        )
        entities = extract_entities(text)
        assert "openai" in entities, "Missing 'openai'"
        assert "fda" in entities, "Missing 'fda'"
        assert "gpt-4o" in entities, "Missing 'gpt-4o'"
        assert "claude-3-opus" in entities, "Missing 'claude-3-opus'"
        assert normalize_entity("  Microsoft's, ") == "microsoft"
        print("[PASS] Check 1: Multi-Pattern Named Entity Extraction & Normalization")
    except Exception as exc:
        print(f"[FAIL] Check 1: Entity extraction failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Mathematical Set Difference & Preservation Metrics
    # ─────────────────────────────────────────────────────────────────────────
    try:
        up_set = {"apple", "google", "microsoft", "amazon"}
        down_set = {"google", "microsoft", "tesla"}

        report = compare_entity_sets(up_set, down_set, "researcher", "writer")
        assert isinstance(report, EntityDeltaReport)
        assert report.shared_entities == {"google", "microsoft"}
        assert report.invented_entities == {"tesla"}
        assert report.dropped_entities == {"apple", "amazon"}
        assert abs(report.preservation_ratio - 0.5) < 1e-4, (
            f"Expected 0.5, got {report.preservation_ratio}"
        )
        assert abs(report.hallucination_rate - (1 / 3)) < 1e-4, (
            f"Expected 1/3, got {report.hallucination_rate}"
        )
        assert report.has_hallucination is True
        assert report.has_information_loss is True
        print("[PASS] Check 2: Mathematical Set Difference & Preservation Metrics")
    except Exception as exc:
        print(f"[FAIL] Check 2: Set difference calculation failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Entity Substitution Detection (Scalar counts 3 vs 3)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        up_text = "Analysis focused on Apple, Google, and Amazon performance."
        down_text = "Analysis focused on Pfizer, Moderna, and BioNTech performance."

        delta = EntityTracker.compare_texts(up_text, down_text, "researcher", "writer")
        assert delta.has_hallucination is True
        assert delta.substitution_detected is True
        assert "pfizer" in delta.invented_entities
        assert "moderna" in delta.invented_entities
        assert "biontech" in delta.invented_entities
        assert "apple" in delta.dropped_entities
        assert delta.preservation_ratio == 0.0
        print("[PASS] Check 3: Entity Substitution Detection (3 vs 3 scalar blind spot closed)")
    except Exception as exc:
        print(f"[FAIL] Check 3: Substitution detection failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Zero False Positives on Faithful Paraphrasing Synthesis
    # ─────────────────────────────────────────────────────────────────────────
    try:
        up_text = "Earnings report: Microsoft and Google achieved strong results in Seattle."
        down_text = (
            "Executive Summary: Microsoft and Google delivered solid performance in Seattle."
        )

        delta = EntityTracker.compare_texts(up_text, down_text, "researcher", "writer")
        assert delta.invented_entities == set(), (
            f"Unexpected invented entities: {delta.invented_entities}"
        )
        assert delta.has_hallucination is False
        assert delta.substitution_detected is False
        assert delta.preservation_ratio == 1.0
        print("[PASS] Check 4: Zero False Positives on Faithful Paraphrasing Synthesis")
    except Exception as exc:
        print(f"[FAIL] Check 4: Zero false positives check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Step-Level & Trace-Level Entity Analysis
    # ─────────────────────────────────────────────────────────────────────────
    try:
        s1 = AgentStep(
            run_id="r1", step=1, agent="researcher", output="Findings on Tesla and SpaceX."
        )
        s2 = AgentStep(
            run_id="r1",
            step=2,
            agent="writer",
            output="Report on Tesla, SpaceX, and Boring Company.",
            parent_step="1",
        )
        s3 = AgentStep(
            run_id="r1",
            step=3,
            agent="verifier",
            output="Verified report on Tesla and SpaceX.",
            parent_step="2",
        )
        trace = RunTrace(run_id="r1", workflow="tech_report", steps=[s1, s2, s3])

        reports = EntityTracker.analyze_trace(trace)
        assert len(reports) == 2
        # Transition 1 -> 2: Boring Company invented
        assert "boring company" in reports[0].invented_entities or any(
            "boring" in e for e in reports[0].invented_entities
        )
        assert reports[0].has_hallucination is True
        # Transition 2 -> 3: Boring Company dropped
        assert reports[1].has_information_loss is True
        print("[PASS] Check 5: Step-Level & Trace-Level Entity Analysis (analyze_trace)")
    except Exception as exc:
        print(f"[FAIL] Check 5: Step/trace level analysis failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: RuleEngine Integration: hallucination_v2 Firing
    # ─────────────────────────────────────────────────────────────────────────
    try:
        s1 = AgentStep(
            run_id="r2",
            step=1,
            agent="researcher",
            output="Research on Apple, Google, and Microsoft.",
            status=StepStatus.SUCCESS,
        )
        s2 = AgentStep(
            run_id="r2",
            step=2,
            agent="writer",
            output="Report on Pfizer, Moderna, and BioNTech.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )
        trace = RunTrace(run_id="r2", workflow="wf", steps=[s1, s2])

        engine = RuleEngine()
        res = engine.analyze(trace)
        rule_ids = [e.rule_match.rule_id for e in res.evidence if e.rule_match]

        assert "hallucination_v1" not in rule_ids, (
            "hallucination_v1 should not fire on identical count"
        )
        assert "hallucination_v2" in rule_ids, "hallucination_v2 MUST fire on entity substitution"
        v2_record = next(
            e for e in res.evidence if e.rule_match and e.rule_match.rule_id == "hallucination_v2"
        )
        assert v2_record.rule_match.category == FailureCategory.REASONING
        assert v2_record.agent == "writer"
        print("[PASS] Check 6: RuleEngine Integration: hallucination_v2 Firing on Substitution")
    except Exception as exc:
        print(f"[FAIL] Check 6: RuleEngine hallucination_v2 integration failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Execution Failure Precedence
    # ─────────────────────────────────────────────────────────────────────────
    try:
        s1 = AgentStep(
            run_id="r3",
            step=1,
            agent="researcher",
            output="",
            tool_calls=[{"name": "search", "error": "ConnectionRefusedError"}],
            status=StepStatus.FAILURE,
        )
        s2 = AgentStep(
            run_id="r3",
            step=2,
            agent="writer",
            output="Report discussing market trends.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )
        trace = RunTrace(run_id="r3", workflow="wf", steps=[s1, s2])

        engine = RuleEngine()
        res = engine.analyze(trace)
        rule_ids = [e.rule_match.rule_id for e in res.evidence if e.rule_match]

        assert "tool_failure_v1" in rule_ids
        assert "hallucination_v2" not in rule_ids, (
            "hallucination_v2 must be suppressed on upstream execution failure"
        )
        print(
            "[PASS] Check 7: Execution Failure Precedence (Upstream Crash Suppresses Downstream Hallucination)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 7: Execution failure precedence check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis & Day 43 Frozen Benchmark Gate
    # ─────────────────────────────────────────────────────────────────────────
    try:
        py_exe = sys.executable

        # Ruff check
        ruff_res = subprocess.run(
            [
                py_exe,
                "-m",
                "ruff",
                "check",
                "analyzers/semantic/entity_tracker.py",
                "tests/test_entity_tracker.py",
                "scripts/verify_day54.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert ruff_res.returncode == 0, f"Ruff failed: {ruff_res.stdout}\n{ruff_res.stderr}"

        # Mypy check
        mypy_res = subprocess.run(
            [
                py_exe,
                "-m",
                "mypy",
                "analyzers/semantic/entity_tracker.py",
                "tests/test_entity_tracker.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert mypy_res.returncode == 0, f"Mypy failed: {mypy_res.stdout}\n{mypy_res.stderr}"

        # Day 43 frozen benchmark regression check
        reg_cmd = [py_exe, "scripts/run_day43_regression.py"]
        reg_res = subprocess.run(reg_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert "16/20 (80.0%)" in reg_res.stdout and "Overall Regression Gate" in reg_res.stdout, (
            f"Day 43 benchmark gate check failed:\n{reg_res.stdout}"
        )
        print(
            "[PASS] Check 8: Static Analysis (Ruff/Mypy clean) & Day 43 Benchmark Preserved (16/20 = 80.0%)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 8: Static analysis / regression gate failed: {exc}")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> ALL 8 DAY 54 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> DAY 54 SEMANTIC ENTITY SET OVERLAP ENGINE CERTIFIED <<<")
    else:
        print(">>> SOME DAY 54 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
