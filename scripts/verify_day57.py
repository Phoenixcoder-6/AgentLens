"""
scripts/verify_day57.py — Day 57 Verification Gate
===================================================
Phase C Sign-Off: Unified Top-Level Public SDK (AgentLens v2.0)

Checks:
  1. Public SDK Imports & Package Surface Parity (agentlens v2.0.0).
  2. Multi-Agent Pipeline Tracing in Under 5 Lines of Code.
  3. Decorator Syntax Permutations (bare, empty parens, custom role & name).
  4. Asynchronous Coroutine Agent Tracing (async def with async with trace_run).
  5. Automatic Framework Inspection (args, docstrings, handoff state).
  6. Exception Propagation & Error Span Attribution (transparent error handling).
  7. Database Persistence & String Run-ID Evaluation (agentlens.evaluate_run).
  8. Static Analysis (Ruff/Mypy Clean) & Day 43 Frozen Benchmark Gate (16/20 = 80.0%).
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agentlens
from schema.models import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    PriorityLevel,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 57 VERIFICATION GATE")
    print("  Unified Top-Level Public SDK (Phase C Day 57)")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Public SDK Imports & Package Surface Parity
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert agentlens.__version__ == "2.0.0", f"Unexpected version: {agentlens.__version__}"
        assert hasattr(agentlens, "init")
        assert hasattr(agentlens, "trace_run")
        assert hasattr(agentlens, "trace_agent")
        assert hasattr(agentlens, "get_current_trace")
        assert hasattr(agentlens, "evaluate_run")

        assert issubclass(agentlens.RunTrace, RunTrace)
        assert issubclass(agentlens.AgentStep, AgentStep)
        assert issubclass(agentlens.TraceSpan, TraceSpan)
        assert issubclass(agentlens.AnalysisBundle, AnalysisBundle)
        assert issubclass(agentlens.FailureCategory, FailureCategory)
        assert issubclass(agentlens.PriorityLevel, PriorityLevel)
        assert issubclass(agentlens.StepStatus, StepStatus)
        assert issubclass(agentlens.TokenUsage, TokenUsage)
        assert issubclass(agentlens.SpanType, SpanType)

        print("[PASS] Check 1: Public SDK Imports & Package Surface Parity (agentlens v2.0.0)")
    except Exception as exc:
        print(f"[FAIL] Check 1: Public imports check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Multi-Agent Pipeline Tracing in Under 5 Lines of Code
    # ─────────────────────────────────────────────────────────────────────────
    try:

        @agentlens.trace_agent(role="researcher")
        def research(topic: str) -> dict[str, Any]:
            return {"findings": f"Core notes on {topic}", "source_count": 2, "entity_count": 2}

        @agentlens.trace_agent(role="writer")
        def write(findings: dict[str, Any]) -> dict[str, Any]:
            return {
                "report": f"Report: {findings['findings']}",
                "source_count": 2,
                "entity_count": 2,
            }

        with agentlens.trace_run("five_lines_pipeline") as run:
            write(research("Deep Learning"))

        assert len(run.steps) == 2, f"Expected 2 steps, got {len(run.steps)}"
        assert len(run.spans) == 2, f"Expected 2 spans, got {len(run.spans)}"
        assert run.steps[0].agent == "research"
        assert run.steps[1].agent == "write"

        bundle = agentlens.evaluate_run(run)
        assert bundle.priority_level == PriorityLevel.P5
        assert bundle.primary_agent is None

        print("[PASS] Check 2: Multi-Agent Pipeline Tracing in Under 5 Lines of Code (PASS/P5)")
    except Exception as exc:
        print(f"[FAIL] Check 2: Under 5 lines pipeline failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Decorator Syntax Permutations (Bare, Empty Parens, Custom Kwargs)
    # ─────────────────────────────────────────────────────────────────────────
    try:

        @agentlens.trace_agent
        def step_bare(state: dict[str, Any]) -> dict[str, Any]:
            return {"msg": "bare"}

        @agentlens.trace_agent()
        def step_parens(state: dict[str, Any]) -> dict[str, Any]:
            return {"msg": "parens"}

        @agentlens.trace_agent(role="compliance", name="safety_guardrail")
        def step_custom(state: dict[str, Any]) -> dict[str, Any]:
            return {"msg": "custom"}

        with agentlens.trace_run("syntax_permutations") as run:
            step_bare({})
            step_parens({})
            step_custom({})

        assert len(run.steps) == 3
        assert run.steps[0].agent == "step_bare"
        assert run.steps[1].agent == "step_parens"
        assert run.steps[2].agent == "safety_guardrail"
        assert run.spans[2].attributes.get("role") == "compliance"

        print(
            "[PASS] Check 3: Decorator Syntax Permutations (bare, empty parens, custom role & name)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 3: Decorator permutations failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Asynchronous Coroutine Agent Tracing
    # ─────────────────────────────────────────────────────────────────────────
    try:

        @agentlens.trace_agent(role="async_gatherer")
        async def fetch_async(url: str) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"url": url, "source_count": 1, "entity_count": 1}

        async def run_async_pipeline() -> RunTrace:
            async with agentlens.trace_run("async_verify_run") as r:
                await fetch_async("https://api.agentlens.ai")
                return r

        async_run = asyncio.run(run_async_pipeline())
        assert len(async_run.steps) == 1
        assert len(async_run.spans) == 1
        assert async_run.steps[0].agent == "fetch_async"
        assert async_run.steps[0].latency_ms >= 5.0

        print("[PASS] Check 4: Asynchronous Coroutine Agent Tracing (async def / async with)")
    except Exception as exc:
        print(f"[FAIL] Check 4: Async tracing check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Automatic Framework Inspection (Args, Docstrings, Handoffs)
    # ─────────────────────────────────────────────────────────────────────────
    try:

        @agentlens.trace_agent(role="evaluator")
        def score_model(model_id: str, threshold: float = 0.85) -> dict[str, Any]:
            """Compute benchmark fidelity metric for evaluation dataset."""
            return {"model": model_id, "score": 0.92, "accepted": True}

        with agentlens.trace_run("inspection_verify_run") as run:
            score_model("claude-3.5-sonnet", threshold=0.90)

        step = run.steps[0]
        assert "model_id" in step.input
        assert "claude-3.5-sonnet" in step.input
        assert "0.9" in step.input
        assert "Compute benchmark fidelity metric" in step.prompt

        print("[PASS] Check 5: Automatic Framework Inspection (args, docstrings, handoff state)")
    except Exception as exc:
        print(f"[FAIL] Check 5: Auto-inspection check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Exception Propagation & Error Span Attribution
    # ─────────────────────────────────────────────────────────────────────────
    try:

        @agentlens.trace_agent(role="database_reader")
        def flaky_db_query(table: str) -> None:
            raise ConnectionError(f"Failed to connect to table {table}")

        exception_caught = False
        try:
            with agentlens.trace_run("error_capture_run") as run:
                flaky_db_query("users")
        except ConnectionError as ce:
            exception_caught = True
            assert "Failed to connect to table users" in str(ce)

        assert exception_caught, "Expected ConnectionError to be propagated"
        assert len(run.steps) == 1
        assert run.steps[0].status == StepStatus.ERROR
        assert "ConnectionError" in str(run.steps[0].error)

        assert len(run.spans) == 1
        assert run.spans[0].status == StepStatus.ERROR
        assert "ConnectionError" in str(run.spans[0].error)

        print(
            "[PASS] Check 6: Exception Propagation & Error Span Attribution (transparent error handling)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 6: Error capture check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Database Persistence & String Run-ID Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        import os
        import tempfile

        from storage.db import DatabaseManager

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            temp_db = Path(tmp_dir) / "verify_sdk.db"
            agentlens.init(db_path=temp_db)

            @agentlens.trace_agent(role="researcher")
            def query_kb() -> dict[str, Any]:
                return {
                    "result": "Knowledgebase article found",
                    "source_count": 2,
                    "entity_count": 2,
                }

            with agentlens.trace_run("db_verify_pipeline", sync=True) as run:
                query_kb()

            # Evaluate directly by run_id string fetched from SQLite
            bundle = agentlens.evaluate_run(run.run_id, db_path=temp_db)
            assert bundle.run_id == run.run_id
            assert bundle.priority_level == PriorityLevel.P5

            DatabaseManager(db_path=temp_db).close()
            os.environ.pop("DB_PATH", None)

        print(
            "[PASS] Check 7: Database Persistence & String Run-ID Evaluation (agentlens.evaluate_run)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 7: Database persistence evaluation failed: {exc}")
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
                "agentlens/",
                "tests/test_public_sdk.py",
                "scripts/verify_day57.py",
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
                "agentlens/",
                "tests/test_public_sdk.py",
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
        print(">>> ALL 8 DAY 57 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> AGENTLENS PUBLIC SDK CERTIFIED & OPERATIONAL! <<<")
    else:
        print(">>> SOME DAY 57 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
