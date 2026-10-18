"""
scripts/verify_day55.py — Day 55 Verification Gate
===================================================
Phase B: Local Cross-Encoder NLI Engine (AgentLens v2.0)

Checks:
  1. NLIEngine Initialization & Fallback Mode (offline/deterministic mode)
  2. Proposition / Claim Decomposition (split_claims with markdown & list filtering)
  3. Direct Polar Contradiction Detection (P(Contradiction) >= 0.70)
  4. Opposing Verbs & Action Inversion (e.g. rejected vs approved, decreased vs increased)
  5. Faithful Entailment Scoring (P(Entailment) >= 0.75)
  6. Multi-Claim Aggregate Report & Step Evaluation (evaluate_steps)
  7. RuleEngine Integration: contradiction_v1 Firing & Execution Error Precedence
  8. Static Analysis (Ruff & Mypy clean) & Day 43 Frozen Benchmark Gate (16/20 = 80.0%)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from analyzers.detection.rule_engine import RuleEngine
from analyzers.semantic.nli_engine import (
    NLIClaimScore,
    NLIEngine,
    NLIEvaluationReport,
    NLILabel,
    evaluate_pair_lexical_fallback,
    split_claims,
)
from schema.models import AgentStep, FailureCategory, RunTrace, StepStatus

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 55 VERIFICATION GATE")
    print("  Local Cross-Encoder NLI Engine (v2.0 Phase B)")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: NLIEngine Initialization & Fallback Mode
    # ─────────────────────────────────────────────────────────────────────────
    try:
        engine = NLIEngine(mode="fallback")
        assert engine.mode == "fallback", "Engine mode should be fallback"
        assert engine.contradiction_threshold == 0.70
        assert engine.entailment_threshold == 0.75
        p_ent, p_neu, p_con = evaluate_pair_lexical_fallback("A", "B")
        assert abs((p_ent + p_neu + p_con) - 1.0) < 1e-4, "Probabilities must sum to 1.0"
        print("[PASS] Check 1: NLIEngine Initialization & Fallback Mode")
    except Exception as exc:
        print(f"[FAIL] Check 1: Engine initialization failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Proposition / Claim Decomposition
    # ─────────────────────────────────────────────────────────────────────────
    try:
        text = (
            "# Executive Report\n"
            "1. DeepMind introduced AlphaFold to accurately predict 3D protein structures.\n"
            "- It leverages evoformer attention to model spatial coordinates.\n"
            "> The database covers over 200 million protein structures worldwide.\n"
            "End."
        )
        claims = split_claims(text)
        assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"
        assert "AlphaFold to accurately predict 3D protein structures" in claims[0]
        assert "evoformer attention" in claims[1]
        assert "covers over 200 million protein structures" in claims[2]
        print("[PASS] Check 2: Proposition & Claim Decomposition (split_claims)")
    except Exception as exc:
        print(f"[FAIL] Check 2: Claim decomposition failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Direct Polar Contradiction Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        engine = NLIEngine(mode="fallback")
        premise = "Clinical study NCT-0391 failed to meet its primary safety endpoints."
        hypothesis = "Clinical study NCT-0391 successfully met its primary safety endpoints."

        score = engine.score_claim(premise, hypothesis)
        assert isinstance(score, NLIClaimScore)
        assert score.is_contradiction is True, "Expected is_contradiction=True"
        assert score.predicted_label == NLILabel.CONTRADICTION
        assert score.contradiction_prob >= 0.70, (
            f"Expected P(contra) >= 0.70, got {score.contradiction_prob}"
        )
        print("[PASS] Check 3: Direct Polar Contradiction Detection (P(Contra) >= 0.70)")
    except Exception as exc:
        print(f"[FAIL] Check 3: Polar contradiction detection failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Opposing Verbs & Action Inversion
    # ─────────────────────────────────────────────────────────────────────────
    try:
        engine = NLIEngine(mode="fallback")
        p_pairs = [
            (
                "The FDA review panel rejected the therapeutic application.",
                "The FDA review panel approved the therapeutic application.",
            ),
            (
                "Quarterly cloud division revenue increased by 22 percent.",
                "Quarterly cloud division revenue decreased by 22 percent.",
            ),
            (
                "The spacecraft was lost during atmospheric reentry.",
                "The spacecraft survived and was recovered safely.",
            ),
        ]
        for p, h in p_pairs:
            s = engine.score_claim(p, h)
            assert s.is_contradiction is True, (
                f"Failed contradiction for '{p}' vs '{h}' (P(Contra)={s.contradiction_prob})"
            )
        print("[PASS] Check 4: Opposing Verbs & Action Inversion Detection")
    except Exception as exc:
        print(f"[FAIL] Check 4: Opposing verbs detection failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Faithful Entailment Scoring
    # ─────────────────────────────────────────────────────────────────────────
    try:
        engine = NLIEngine(mode="fallback")
        premise = (
            "NASA selected SpaceX to develop the human landing system for the Artemis program."
        )
        hypothesis = "SpaceX was selected by NASA to develop the human landing system for the Artemis lunar mission."

        score = engine.score_claim(premise, hypothesis)
        assert score.is_entailed is True, "Expected is_entailed=True"
        assert score.is_contradiction is False
        assert score.predicted_label == NLILabel.ENTAILMENT
        assert score.entailment_prob >= 0.75, (
            f"Expected P(entail) >= 0.75, got {score.entailment_prob}"
        )
        print("[PASS] Check 5: Faithful Entailment Scoring (P(Entail) >= 0.75)")
    except Exception as exc:
        print(f"[FAIL] Check 5: Entailment scoring failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Multi-Claim Aggregate Report & Step Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        engine = NLIEngine(mode="fallback")
        s1 = AgentStep(
            run_id="run_v55_01",
            step=1,
            agent="researcher",
            output=(
                "Apollo 11 landed on July 20, 1969. "
                "Neil Armstrong and Buzz Aldrin walked on the lunar surface."
            ),
            status=StepStatus.SUCCESS,
        )
        s2 = AgentStep(
            run_id="run_v55_01",
            step=2,
            agent="writer",
            output=(
                "Apollo 11 landed on July 20, 1969. "
                "Neil Armstrong and Buzz Aldrin failed to land on the Moon."
            ),
            status=StepStatus.SUCCESS,
            parent_step="1",
        )

        report = engine.evaluate_steps(s1, s2)
        assert isinstance(report, NLIEvaluationReport)
        assert report.upstream_agent == "researcher"
        assert report.downstream_agent == "writer"
        assert report.has_contradiction is True
        assert len(report.contradicted_claims) == 1
        assert "failed to land" in report.contradicted_claims[0].claim
        assert len(report.entailed_claims) == 1
        print("[PASS] Check 6: Multi-Claim Aggregate Report & Step Evaluation")
    except Exception as exc:
        print(f"[FAIL] Check 6: Aggregate report failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: RuleEngine contradiction_v1 Firing & Precedence
    # ─────────────────────────────────────────────────────────────────────────
    try:
        rule_engine = RuleEngine()

        # Sub-check A: contradiction_v1 fires on reasoning contradiction
        step1 = AgentStep(
            run_id="test_run",
            step=1,
            agent="researcher",
            output="Clinical trial NCT-0391 failed to meet its primary safety endpoints.",
            status=StepStatus.SUCCESS,
        )
        step2 = AgentStep(
            run_id="test_run",
            step=2,
            agent="writer",
            output="Clinical trial NCT-0391 successfully met its primary safety endpoints.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )
        trace_contra = RunTrace(
            run_id="test_run", workflow="research_report_pipeline", steps=[step1, step2]
        )
        res_contra = rule_engine.analyze(trace_contra)
        rule_ids = [e.rule_match.rule_id for e in res_contra.evidence if e.rule_match]
        assert "contradiction_v1" in rule_ids, (
            f"Expected contradiction_v1 in rules fired, got: {rule_ids}"
        )

        contra_ev = next(
            e
            for e in res_contra.evidence
            if e.rule_match and e.rule_match.rule_id == "contradiction_v1"
        )

        rule_match = contra_ev.rule_match
        assert rule_match is not None
        assert rule_match.category == FailureCategory.REASONING
        assert contra_ev.agent == "writer"

        # Sub-check B: Upstream execution error suppresses downstream contradiction
        step1_err = AgentStep(
            run_id="test_run_err",
            step=1,
            agent="researcher",
            output="",
            tool_calls=[{"name": "fetch_clinical", "error": "ConnectionTimeout"}],
            status=StepStatus.FAILURE,
        )
        trace_err = RunTrace(
            run_id="test_run_err", workflow="research_report_pipeline", steps=[step1_err, step2]
        )
        res_err = rule_engine.analyze(trace_err)
        err_rules = [e.rule_match.rule_id for e in res_err.evidence if e.rule_match]
        assert "tool_failure_v1" in err_rules
        assert "contradiction_v1" not in err_rules, (
            "contradiction_v1 should be suppressed on upstream execution failure"
        )

        print("[PASS] Check 7: RuleEngine contradiction_v1 Firing & Precedence")
    except Exception as exc:
        print(f"[FAIL] Check 7: RuleEngine integration failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis & Day 43 Benchmark Gate
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # 1. Ruff check
        ruff_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "analyzers/semantic/",
                "tests/test_nli_engine.py",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if ruff_res.returncode != 0:
            raise RuntimeError(f"Ruff failed: {ruff_res.stdout} {ruff_res.stderr}")

        # 2. Mypy check
        mypy_res = subprocess.run(
            [sys.executable, "-m", "mypy", "analyzers/semantic/nli_engine.py"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if mypy_res.returncode != 0:
            raise RuntimeError(f"Mypy failed: {mypy_res.stdout} {mypy_res.stderr}")

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
        print(">>> ALL 8 DAY 55 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> DAY 55 LOCAL CROSS-ENCODER NLI ENGINE CERTIFIED <<<")
    else:
        print(">>> SOME DAY 55 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
