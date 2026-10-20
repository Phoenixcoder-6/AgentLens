"""
scripts/verify_day60.py — Day 60 Verification Gate
===================================================
Phase C Sign-Off: OpenTelemetry (OTLP) Ingestion Endpoint (AgentLens v2.0)

Checks:
  1. OTLP Ingestion Surface & REST Endpoints (POST /v1/traces and POST /api/v1/traces).
  2. Standard OTLP JSON Trace Normalization (OTLPNormalizer -> RunTrace).
  3. OpenInference Semantic Attributes & Span Kinds Mapping (AGENT, TOOL, LLM).
  4. Nested Tool Call Extraction & LLM Token Aggregation into step.tokens.
  5. Multi-Iteration Cyclic Reflection Loop Tracking with iteration.index.
  6. Error Span Ingestion & Arbiter Root-Cause Attribution (tool_failure_v1).
  7. SQLite Database Persistence & Trace Reloading via DatabaseManager.
  8. Static Analysis (Ruff/Mypy Clean) & Day 43 Frozen Benchmark Gate (16/20 = 80.0%).
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

import agentlens
from api.main import fastapi_app
from api.router import otlp_router
from normalizer.otlp_normalizer import OTLPNormalizer
from schema.models import FailureCategory, PriorityLevel, StepStatus
from schema.span import SpanType
from storage.db import DatabaseManager


def _build_otlp_payload(
    workflow: str,
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Helper to build an OTLP JSON trace payload."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": workflow}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "openinference", "version": "0.1.0"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 60 VERIFICATION GATE")
    print("  OpenTelemetry (OTLP) Ingestion Endpoint (Phase C Day 60)")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: OTLP Ingestion Surface & REST Endpoints
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert otlp_router is not None
        routes = [r.path for r in otlp_router.routes]  # type: ignore[attr-defined]
        assert "/v1/traces" in routes, f"Missing /v1/traces route: {routes}"
        assert "/traces" in routes, f"Missing /traces route: {routes}"

        client = TestClient(fastapi_app, raise_server_exceptions=False)
        # Verify 400 when missing payload
        res = client.post("/v1/traces", json={})
        assert res.status_code == 400

        print(
            "[PASS] Check 1: OTLP Ingestion Surface & REST Endpoints (POST /v1/traces & /api/v1/traces)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 1: Surface endpoint check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Standard OTLP JSON Trace Normalization
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_id = uuid.uuid4().hex
        s_id = uuid.uuid4().hex[:16]

        spans = [
            {
                "traceId": t_id,
                "spanId": s_id,
                "name": "orchestrator",
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000001000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "input.value", "value": {"stringValue": "initiate job"}},
                    {"key": "output.value", "value": {"stringValue": "job completed"}},
                ],
                "status": {"code": 1},
            }
        ]

        payload = _build_otlp_payload("batch_orchestration", spans)
        normalizer = OTLPNormalizer()
        traces = normalizer.normalize_otlp_payload(payload)

        assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"
        trace = traces[0]
        assert trace.run_id == t_id
        assert trace.workflow == "batch_orchestration"
        assert trace.status == StepStatus.SUCCESS
        assert len(trace.steps) == 1
        assert trace.steps[0].agent == "orchestrator"

        print("[PASS] Check 2: Standard OTLP JSON Trace Normalization (OTLPNormalizer)")
    except Exception as exc:
        print(f"[FAIL] Check 2: Normalization check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: OpenInference Semantic Attributes & Span Kinds Mapping
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_id = uuid.uuid4().hex
        spans = [
            {
                "traceId": t_id,
                "spanId": "s_agent",
                "name": "search_agent",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}}
                ],
            },
            {
                "traceId": t_id,
                "spanId": "s_tool",
                "parentSpanId": "s_agent",
                "name": "bing_search",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}}
                ],
            },
            {
                "traceId": t_id,
                "spanId": "s_llm",
                "parentSpanId": "s_agent",
                "name": "claude-3-5-sonnet",
                "attributes": [{"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}],
            },
            {
                "traceId": t_id,
                "spanId": "s_router",
                "parentSpanId": "s_agent",
                "name": "intent_router",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "ROUTER"}}
                ],
            },
        ]

        payload = _build_otlp_payload("semantic_kinds_wf", spans)
        trace = OTLPNormalizer().normalize_otlp_payload(payload)[0]

        span_types = {s.span_id: s.span_type for s in trace.spans}
        assert span_types["s_agent"] == SpanType.AGENT
        assert span_types["s_tool"] == SpanType.TOOL
        assert span_types["s_llm"] == SpanType.LLM
        assert span_types["s_router"] == SpanType.ROUTER

        print("[PASS] Check 3: OpenInference Semantic Attributes & Span Kinds Mapping")
    except Exception as exc:
        print(f"[FAIL] Check 3: Semantic kinds check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Nested Tool Call Extraction & LLM Token Aggregation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_id = uuid.uuid4().hex
        agent_id = "agent_math"

        spans = [
            {
                "traceId": t_id,
                "spanId": agent_id,
                "name": "math_solver",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "input.value", "value": {"stringValue": "Calculate 15 * 8"}},
                    {"key": "output.value", "value": {"stringValue": "120"}},
                ],
            },
            {
                "traceId": t_id,
                "spanId": "tool_calc",
                "parentSpanId": agent_id,
                "name": "calculator_tool",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}},
                    {"key": "tool.name", "value": {"stringValue": "calculator_tool"}},
                    {"key": "tool.parameters", "value": {"stringValue": "15 * 8"}},
                    {"key": "tool.output", "value": {"stringValue": "120"}},
                ],
            },
            {
                "traceId": t_id,
                "spanId": "llm_eval",
                "parentSpanId": agent_id,
                "name": "deepseek-coder",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                    {"key": "llm.token_count.prompt", "value": {"intValue": 110}},
                    {"key": "llm.token_count.completion", "value": {"intValue": 40}},
                    {"key": "llm.token_count.total", "value": {"intValue": 150}},
                ],
            },
        ]

        payload = _build_otlp_payload("math_pipeline", spans)
        trace = OTLPNormalizer().normalize_otlp_payload(payload)[0]

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0]["name"] == "calculator_tool"
        assert step.tokens.prompt == 110
        assert step.tokens.completion == 40
        assert step.tokens.total == 150
        assert trace.total_tokens == 150

        print(
            "[PASS] Check 4: Nested Tool Call Extraction & LLM Token Aggregation into step.tokens"
        )
    except Exception as exc:
        print(f"[FAIL] Check 4: Tool and token aggregation check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Multi-Iteration Cyclic Reflection Loop Tracking
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_id = uuid.uuid4().hex
        spans = [
            {
                "traceId": t_id,
                "spanId": "pass_0",
                "name": "reflector",
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000001000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "iteration.index", "value": {"intValue": 0}},
                ],
            },
            {
                "traceId": t_id,
                "spanId": "pass_1",
                "name": "reflector",
                "startTimeUnixNano": "1700000002000000000",
                "endTimeUnixNano": "1700000003000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "iteration.index", "value": {"intValue": 1}},
                ],
            },
        ]

        payload = _build_otlp_payload("cyclic_pipeline", spans)
        trace = OTLPNormalizer().normalize_otlp_payload(payload)[0]

        assert len(trace.steps) == 2
        reflector_spans = [s for s in trace.spans if s.name == "reflector"]
        assert len(reflector_spans) == 2
        assert reflector_spans[0].iteration_index == 0
        assert reflector_spans[1].iteration_index == 1

        print("[PASS] Check 5: Multi-Iteration Cyclic Reflection Loop Tracking (iteration.index)")
    except Exception as exc:
        print(f"[FAIL] Check 5: Cyclic loop tracking check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Error Span Ingestion & Arbiter Root-Cause Attribution
    # ─────────────────────────────────────────────────────────────────────────
    try:
        t_id = uuid.uuid4().hex
        agent_id = "failing_worker"

        spans = [
            {
                "traceId": t_id,
                "spanId": agent_id,
                "name": "database_sync_agent",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}}
                ],
                "status": {"code": 2, "message": "Failed database transaction"},
            },
            {
                "traceId": t_id,
                "spanId": "tool_fail",
                "parentSpanId": agent_id,
                "name": "postgres_write",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}},
                    {"key": "tool.name", "value": {"stringValue": "postgres_write"}},
                    {
                        "key": "error.message",
                        "value": {"stringValue": "OperationalError: connection to server was lost"},
                    },
                ],
                "status": {"code": 2, "message": "OperationalError"},
            },
        ]

        payload = _build_otlp_payload("sync_pipeline", spans)
        trace = OTLPNormalizer().normalize_otlp_payload(payload)[0]
        assert trace.status == StepStatus.FAILURE

        bundle = agentlens.evaluate_run(trace)
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_agent == "database_sync_agent"

        print(
            "[PASS] Check 6: Error Span Ingestion & Arbiter Root-Cause Attribution (tool_failure_v1)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 6: Error span attribution check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: SQLite Database Persistence & Trace Reloading
    # ─────────────────────────────────────────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            temp_db = Path(tmp_dir) / "otlp_verify.db"
            agentlens.init(db_path=temp_db)

            client = TestClient(fastapi_app, raise_server_exceptions=False)
            t_id = uuid.uuid4().hex

            spans = [
                {
                    "traceId": t_id,
                    "spanId": uuid.uuid4().hex[:16],
                    "name": "persisted_agent",
                    "attributes": [
                        {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                        {"key": "input.value", "value": {"stringValue": "hello"}},
                        {"key": "output.value", "value": {"stringValue": "world"}},
                    ],
                    "status": {"code": 1},
                }
            ]

            payload = _build_otlp_payload("persistence_test_wf", spans)
            resp = client.post("/v1/traces", json=payload)
            assert resp.status_code == 200
            assert t_id in resp.json()["run_ids"]

            # Reload from DB
            db = DatabaseManager(db_path=temp_db)
            row = db.get_run(t_id)
            assert row is not None
            assert row["workflow"] == "persistence_test_wf"
            assert row["status"] == "SUCCESS"

            db.close()
            os.environ.pop("DB_PATH", None)

        print("[PASS] Check 7: SQLite Database Persistence & Trace Reloading via DatabaseManager")
    except Exception as exc:
        print(f"[FAIL] Check 7: Database persistence check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis & Day 43 Frozen Benchmark Gate
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # 1. Ruff check
        ruff_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "normalizer/otlp_normalizer.py",
                "api/router.py",
                "api/main.py",
                "tests/test_otlp_ingestion.py",
                "scripts/verify_day60.py",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if ruff_res.returncode != 0:
            raise RuntimeError(f"Ruff check failed: {ruff_res.stdout} {ruff_res.stderr}")

        # 2. Mypy check
        mypy_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "normalizer/otlp_normalizer.py",
                "api/router.py",
                "api/main.py",
                "tests/test_otlp_ingestion.py",
                "scripts/verify_day60.py",
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
        print(">>> ALL 8 DAY 60 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> OPENTELEMETRY (OTLP) INGESTION CERTIFIED & OPERATIONAL! <<<")
    else:
        print(">>> SOME DAY 60 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
