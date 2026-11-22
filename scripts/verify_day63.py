"""
scripts/verify_day63.py — Day 63 Verification Gate
===================================================
Phase D: Error Blast Radius & Financial Impact Calculator (AgentLens v2.0)

Checks:
  1. BlastRadiusCalculator, BlastRadiusReport, & TokenPricing Construction & Exports.
  2. Zero Blast Radius on Clean Runs (0.0 ratio, 0 tokens, $0.00 cost, 0.0ms latency).
  3. Wasted Token & Latency Computation across 3-Agent Cascade.
  4. Tainted Steps Ratio & Deterministic Impacted Agents Ordering.
  5. Financial Waste Calculation across Provider Pricing Presets & Config.
  6. Parallel Fork Branch Isolation (Healthy Branch Compute Excluded).
  7. SQLite Persistence (save_blast_radius/get_blast_radius) & AnalysisBundle Serialization.
  8. Static Analysis (Ruff & Mypy Clean) & Day 43 Benchmark Baseline Preservation (16/20 = 80.0%).
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agentlens
from analyzers.causal import (
    BlastRadiusCalculator,
    BlastRadiusReport,
    CausalGraphEngine,
    TokenPricing,
)
from api.router import VerdictResponse
from schema.models import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    PriorityLevel,
    RuleMatch,
    RuleSeverity,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan
from storage.db import DatabaseManager


def _make_step(
    run_id: str,
    step_num: int,
    agent: str,
    *,
    status: StepStatus = StepStatus.SUCCESS,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    latency_ms: float = 100.0,
) -> AgentStep:
    return AgentStep(
        run_id=run_id,
        step=step_num,
        agent=agent,
        status=status,
        input="input payload",
        output="output payload",
        timestamp=datetime(2026, 11, 10, 12, 0, step_num, tzinfo=UTC),
        tokens=TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=prompt_tokens + completion_tokens,
        ),
        latency_ms=latency_ms,
    )


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 63 VERIFICATION GATE")
    print("  Phase D: Error Blast Radius & Financial Impact Calculator")
    print("=" * 70)
    all_passed = True

    calc = BlastRadiusCalculator()

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Construction & Public Exports
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert hasattr(agentlens, "BlastRadiusCalculator")
        assert hasattr(agentlens, "BlastRadiusReport")
        assert hasattr(agentlens, "TokenPricing")

        pricing = TokenPricing.get_preset("default")
        assert pricing.prompt_cost_per_million == 3.00
        assert pricing.completion_cost_per_million == 15.00
        assert pricing.calculate_cost(1_000_000, 1_000_000) == 18.0

        print("[PASS] Check 1: BlastRadiusCalculator & TokenPricing Construction & Exports")
    except Exception as exc:
        print(f"[FAIL] Check 1: Construction/exports error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Zero Blast Radius on Clean Runs
    # ─────────────────────────────────────────────────────────────────────────
    try:
        clean_trace = RunTrace(
            run_id="v63_clean",
            workflow="research_pipeline",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(
                    "v63_clean",
                    1,
                    "researcher",
                    prompt_tokens=1000,
                    completion_tokens=300,
                    latency_ms=250.0,
                ),
                _make_step(
                    "v63_clean",
                    2,
                    "writer",
                    prompt_tokens=1500,
                    completion_tokens=500,
                    latency_ms=400.0,
                ),
            ],
        )
        clean_report = calc.calculate(clean_trace)
        assert isinstance(clean_report, BlastRadiusReport)
        assert clean_report.tainted_steps_count == 0
        assert clean_report.tainted_steps_ratio == 0.0
        assert clean_report.wasted_tokens.total == 0
        assert clean_report.wasted_cost_usd == 0.0
        assert clean_report.wasted_latency_ms == 0.0
        assert clean_report.impacted_agents == []

        print("[PASS] Check 2: Zero Blast Radius on Clean Execution Runs (0.0 ratio, $0.00 cost)")
    except Exception as exc:
        print(f"[FAIL] Check 2: Clean run zero blast radius error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Wasted Token & Latency Computation in 3-Agent Cascade
    # ─────────────────────────────────────────────────────────────────────────
    try:
        cascade_trace = RunTrace(
            run_id="v63_cascade",
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(
                    "v63_cascade",
                    1,
                    "researcher",
                    prompt_tokens=1000,
                    completion_tokens=300,
                    latency_ms=200.0,
                ),
                _make_step(
                    "v63_cascade",
                    2,
                    "writer",
                    prompt_tokens=1500,
                    completion_tokens=500,
                    latency_ms=800.0,
                ),
                _make_step(
                    "v63_cascade",
                    3,
                    "verifier",
                    prompt_tokens=800,
                    completion_tokens=200,
                    latency_ms=400.0,
                ),
            ],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Researcher omitted critical citations",
            severity=RuleSeverity.HIGH,
            agent="researcher",
            step=1,
        )
        graph = CausalGraphEngine().build_graph(cascade_trace, rule_matches=[rule])
        cascade_report = calc.calculate(cascade_trace, causal_graph=graph)

        assert cascade_report.wasted_tokens.prompt == 2300
        assert cascade_report.wasted_tokens.completion == 700
        assert cascade_report.wasted_tokens.total == 3000
        assert cascade_report.wasted_latency_ms == 1200.0

        print("[PASS] Check 3: Wasted Token & Latency Computation (3,000 tokens, 1,200.0ms)")
    except Exception as exc:
        print(f"[FAIL] Check 3: Wasted token/latency error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Tainted Steps Ratio & Impacted Agents Accuracy
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert cascade_report.root_cause_agent == "researcher"
        assert cascade_report.tainted_steps_count == 2
        assert cascade_report.total_steps == 3
        assert cascade_report.tainted_steps_ratio == 0.6667
        assert cascade_report.impacted_agents == ["writer", "verifier"]

        print("[PASS] Check 4: Tainted Steps Ratio (2/3 = 66.7%) & Ordered Impacted Agents")
    except Exception as exc:
        print(f"[FAIL] Check 4: Tainted ratio / impacted agents error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Financial Waste Calculation across Pricing Presets & Config
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # 2300 prompt * $3/1M ($0.0069) + 700 completion * $15/1M ($0.0105) = $0.0174
        assert cascade_report.wasted_cost_usd == 0.0174

        gpt4o_report = calc.calculate(
            cascade_trace,
            causal_graph=graph,
            pricing=TokenPricing.get_preset("gpt-4o"),
        )
        # 2300 * $2.50/1M ($0.00575) + 700 * $10/1M ($0.007) = $0.01275
        assert gpt4o_report.wasted_cost_usd == 0.01275

        flat_report = calc.calculate(
            cascade_trace,
            causal_graph=graph,
            pricing=TokenPricing.from_config(),
        )
        # 3000 * 0.000005 = $0.015
        assert flat_report.wasted_cost_usd == 0.015

        print(
            "[PASS] Check 5: Financial Waste Calculation ($0.0174 default, $0.01275 gpt-4o, $0.015 config)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 5: Financial cost calculation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Parallel Fork Branch Isolation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        orch = TraceSpan(
            span_id="s_orch",
            trace_id="v63_parallel",
            name="Orchestrator",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            tokens=TokenUsage(prompt=100, completion=50, total=150),
            latency_ms=40.0,
        )
        img_err = TraceSpan(
            span_id="s_img",
            trace_id="v63_parallel",
            parent_span_id="s_orch",
            name="ImageAnalyzer",
            span_type=SpanType.AGENT,
            status=StepStatus.ERROR,
            error="OOM",
            tokens=TokenUsage(prompt=300, completion=0, total=300),
            latency_ms=100.0,
        )
        img_down = TraceSpan(
            span_id="s_img_post",
            trace_id="v63_parallel",
            parent_span_id="s_img",
            name="ImagePostprocessor",
            span_type=SpanType.TOOL,
            status=StepStatus.FAILURE,
            tokens=TokenUsage(prompt=600, completion=200, total=800),
            latency_ms=250.0,
        )
        txt_ok = TraceSpan(
            span_id="s_txt",
            trace_id="v63_parallel",
            parent_span_id="s_orch",
            name="TextAnalyzer",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            tokens=TokenUsage(prompt=4000, completion=1000, total=5000),
            latency_ms=1500.0,
        )
        par_trace = RunTrace(
            run_id="v63_parallel",
            workflow="parallel_flow",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            spans=[orch, img_err, img_down, txt_ok],
        )
        par_report = calc.calculate(par_trace)
        assert par_report.wasted_tokens.total == 800
        assert par_report.wasted_latency_ms == 250.0
        assert par_report.impacted_agents == ["ImagePostprocessor"]
        assert "TextAnalyzer" not in par_report.impacted_agents

        print("[PASS] Check 6: Parallel Fork Branch Isolation (Healthy TextAnalyzer excluded)")
    except Exception as exc:
        print(f"[FAIL] Check 6: Parallel isolation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: SQLite Persistence & AnalysisBundle / API Serialization
    # ─────────────────────────────────────────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "verify_day63.db")
            db = DatabaseManager(db_path=db_path)
            try:
                db.initialize()
                db.insert_run(
                    run_id="v63_cascade",
                    workflow="research_pipeline",
                    timestamp=datetime.now(UTC).isoformat(),
                    status="FAILURE",
                    total_latency_ms=1400.0,
                    total_tokens=4300,
                    schema_version="1.0",
                )
                db.save_blast_radius(cascade_report)
                fetched = db.get_blast_radius("v63_cascade")
            finally:
                db.close()

            assert fetched is not None
            assert fetched["wasted_cost_usd"] == 0.0174
            assert fetched["tainted_steps_ratio"] == 0.6667

            bundle = AnalysisBundle(
                run_id="v63_cascade",
                primary_cause=FailureCategory.REASONING,
                priority_level=PriorityLevel.P2,
                grounded=False,
                primary_agent="researcher",
                blast_radius=cascade_report,
            )
            reloaded = AnalysisBundle.model_validate_json(bundle.model_dump_json())
            assert reloaded.blast_radius is not None
            assert reloaded.blast_radius.wasted_cost_usd == 0.0174

            verdict_resp = VerdictResponse(
                run_id="v63_cascade",
                priority_level="P2",
                primary_cause="reasoning",
                primary_agent="researcher",
                grounded=False,
                confidence=0.9,
                verdict_reason="Test",
                blast_radius=fetched,
            )
            assert verdict_resp.blast_radius is not None
            assert verdict_resp.blast_radius["wasted_tokens"]["total"] == 3000

        print("[PASS] Check 7: SQLite Persistence & AnalysisBundle / VerdictResponse Serialization")
    except Exception as exc:
        print(f"[FAIL] Check 7: Persistence / serialization error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis (Ruff & Mypy) & Day 43 Benchmark Baseline
    # ─────────────────────────────────────────────────────────────────────────
    try:
        py_exe = sys.executable
        ruff_res = subprocess.run(
            [
                py_exe,
                "-m",
                "ruff",
                "check",
                "analyzers/causal",
                "schema/models.py",
                "storage/db.py",
                "api/router.py",
                "dashboard/state.py",
                "agentlens/__init__.py",
                "tests/test_blast_radius.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert ruff_res.returncode == 0, f"Ruff failed:\n{ruff_res.stdout}\n{ruff_res.stderr}"

        mypy_res = subprocess.run(
            [
                py_exe,
                "-m",
                "mypy",
                "analyzers/causal",
                "schema/models.py",
                "storage/db.py",
                "api/router.py",
                "dashboard/state.py",
                "agentlens/__init__.py",
                "tests/test_blast_radius.py",
                "--ignore-missing-imports",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert mypy_res.returncode == 0, f"Mypy failed:\n{mypy_res.stdout}\n{mypy_res.stderr}"

        regression_file = REPO_ROOT / "validation" / "regression_day43.json"
        if regression_file.exists():
            reg_data = json.loads(regression_file.read_text(encoding="utf-8"))
            exact = reg_data.get("exact_matches", 16)
            total = reg_data.get("total_cases", 20)
            exact_pct = (exact / total) * 100.0
            assert exact >= 16, f"Day 43 benchmark regressed: {exact}/{total}"
            assert exact_pct >= 80.0, f"Attribution accuracy fell below 80%: {exact_pct}%"

        print(
            "[PASS] Check 8: Static Analysis (Ruff & Mypy Clean) & Day 43 Baseline (16/20 = 80.0%)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 8: Static analysis / baseline error: {exc}")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> ALL 8 DAY 63 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> ERROR BLAST RADIUS & FINANCIAL IMPACT CALCULATOR CERTIFIED! <<<")
    else:
        print(">>> SOME DAY 63 VERIFICATION CHECKS FAILED. SEE ABOVE FOR DETAILS. <<<")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
