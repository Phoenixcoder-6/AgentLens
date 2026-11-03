"""
tests/test_otlp_ingestion.py — Day 60: OpenTelemetry & OpenInference Ingestion Tests
=====================================================================================
Tests:
  1. Standard OTLP JSON payload ingestion at POST /v1/traces.
  2. Namespaced OTLP payload ingestion at POST /api/v1/traces.
  3. OpenInference semantic attributes mapping (AGENT, TOOL, LLM hierarchy & tokens).
  4. Multi-iteration cyclic reflection loop tracking with iteration.index.
  5. Error span ingestion and automated Arbiter root-cause diagnosis.
  6. Flat spans payload format compatibility.
  7. Invalid payload schema validation (HTTP 400 on empty/malformed payload).
  8. SQLite database persistence and trace reloading via DatabaseManager.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import agentlens
from api.main import fastapi_app
from schema.models import FailureCategory, PriorityLevel, StepStatus
from schema.span import SpanType
from storage.db import DatabaseManager


@pytest.fixture
def client(tmp_path: Path):
    """TestClient wired to a temporary test SQLite database."""
    test_db = tmp_path / "otlp_test.db"
    agentlens.init(db_path=test_db)
    # Ensure the dashboard state singleton uses the test DB too
    import dashboard.state as _ds
    _ds._db = None  # Force re-initialization on next get_db() call
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c


def _generate_otlp_payload(
    workflow: str = "customer_support",
    spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Helper to generate a standard OTLP JSON trace structure."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": workflow}},
                        {"key": "service.namespace", "value": {"stringValue": "agent_services"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "openinference.instrumentation",
                            "version": "0.1.0",
                        },
                        "spans": spans or [],
                    }
                ],
            }
        ]
    }


class TestOTLPIngestion:
    def test_otlp_standard_payload_ingestion(self, client: TestClient, tmp_path: Path) -> None:
        """Verify standard OTLP JSON payload is ingested at POST /v1/traces with HTTP 200."""
        trace_id = uuid.uuid4().hex
        span_id_1 = uuid.uuid4().hex[:16]
        span_id_2 = uuid.uuid4().hex[:16]

        spans = [
            {
                "traceId": trace_id,
                "spanId": span_id_1,
                "name": "triage_agent",
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000001000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {
                        "key": "input.value",
                        "value": {"stringValue": json.dumps({"ticket": "Cannot login"})},
                    },
                    {
                        "key": "output.value",
                        "value": {"stringValue": json.dumps({"intent": "auth_reset"})},
                    },
                ],
                "status": {"code": 1},
            },
            {
                "traceId": trace_id,
                "spanId": span_id_2,
                "parentSpanId": span_id_1,
                "name": "auth_agent",
                "startTimeUnixNano": "1700000001100000000",
                "endTimeUnixNano": "1700000002000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {
                        "key": "input.value",
                        "value": {"stringValue": json.dumps({"intent": "auth_reset"})},
                    },
                    {
                        "key": "output.value",
                        "value": {"stringValue": json.dumps({"status": "reset_link_sent"})},
                    },
                ],
                "status": {"code": 1},
            },
        ]

        payload = _generate_otlp_payload("auth_support_workflow", spans=spans)
        response = client.post("/v1/traces", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["traces_ingested"] == 1
        assert trace_id in data["run_ids"]

        # Read back from DB
        db = DatabaseManager(tmp_path / "otlp_test.db")
        row = db.get_run(trace_id)
        assert row is not None
        assert row["workflow"] == "auth_support_workflow"
        assert row["status"] == "SUCCESS"

    def test_otlp_api_namespaced_endpoint(self, client: TestClient) -> None:
        """Verify namespaced POST /api/v1/traces endpoint also functions correctly."""
        trace_id = uuid.uuid4().hex
        spans = [
            {
                "traceId": trace_id,
                "spanId": uuid.uuid4().hex[:16],
                "name": "ping_agent",
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000000500000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "input.value", "value": {"stringValue": "ping"}},
                    {"key": "output.value", "value": {"stringValue": "pong"}},
                ],
                "status": {"code": 1},
            }
        ]

        payload = _generate_otlp_payload("ping_workflow", spans=spans)
        response = client.post("/api/v1/traces", json=payload)

        assert response.status_code == 200
        assert response.json()["traces_ingested"] == 1

    def test_otlp_openinference_agent_tool_llm_hierarchy(self, client: TestClient) -> None:
        """Verify OpenInference AGENT, TOOL, and LLM spans map into step.tool_calls and tokens."""
        trace_id = uuid.uuid4().hex
        agent_span_id = uuid.uuid4().hex[:16]
        tool_span_id = uuid.uuid4().hex[:16]
        llm_span_id = uuid.uuid4().hex[:16]

        spans = [
            # 1. Agent Span
            {
                "traceId": trace_id,
                "spanId": agent_span_id,
                "name": "financial_researcher",
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000003000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "input.value", "value": {"stringValue": "Fetch Q3 income statement"}},
                    {"key": "output.value", "value": {"stringValue": "Net income: $20B"}},
                ],
                "status": {"code": 1},
            },
            # 2. Nested Tool Span
            {
                "traceId": trace_id,
                "spanId": tool_span_id,
                "parentSpanId": agent_span_id,
                "name": "edgar_sec_fetch",
                "startTimeUnixNano": "1700000000200000000",
                "endTimeUnixNano": "1700000001000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}},
                    {"key": "tool.name", "value": {"stringValue": "edgar_sec_fetch"}},
                    {"key": "tool.parameters", "value": {"stringValue": '{"ticker": "AAPL"}'}},
                    {"key": "tool.output", "value": {"stringValue": '{"10-Q": "verified"}'}},
                ],
                "status": {"code": 1},
            },
            # 3. Nested LLM Span
            {
                "traceId": trace_id,
                "spanId": llm_span_id,
                "parentSpanId": agent_span_id,
                "name": "gpt-4o",
                "startTimeUnixNano": "1700000001200000000",
                "endTimeUnixNano": "1700000002800000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                    {"key": "llm.token_count.prompt", "value": {"intValue": 150}},
                    {"key": "llm.token_count.completion", "value": {"intValue": 50}},
                    {"key": "llm.token_count.total", "value": {"intValue": 200}},
                ],
                "status": {"code": 1},
            },
        ]

        payload = _generate_otlp_payload("sec_filing_pipeline", spans=spans)
        resp = client.post("/v1/traces", json=payload)
        assert resp.status_code == 200

        # Verify normalized trace from normalizer directly
        from normalizer.otlp_normalizer import OTLPNormalizer

        traces = OTLPNormalizer().normalize_otlp_payload(payload)
        assert len(traces) == 1
        trace = traces[0]

        # 1 agent step was reconstructed
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.agent == "financial_researcher"

        # Tool calls mapped
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0]["name"] == "edgar_sec_fetch"

        # Tokens aggregated from child LLM span
        assert step.tokens.prompt == 150
        assert step.tokens.completion == 50
        assert step.tokens.total == 200
        assert trace.total_tokens == 200

        # Spans hierarchy
        assert len(trace.spans) == 3
        tool_s = next(s for s in trace.spans if s.span_type == SpanType.TOOL)
        assert tool_s.parent_span_id == agent_span_id

    def test_otlp_cyclic_reflection_iteration_index(self, client: TestClient) -> None:
        """Verify cyclic passes of an agent preserve iteration.index attribute."""
        trace_id = uuid.uuid4().hex
        spans = [
            {
                "traceId": trace_id,
                "spanId": uuid.uuid4().hex[:16],
                "name": "drafter",
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000001000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "iteration.index", "value": {"intValue": 0}},
                ],
                "status": {"code": 1},
            },
            {
                "traceId": trace_id,
                "spanId": uuid.uuid4().hex[:16],
                "name": "drafter",
                "startTimeUnixNano": "1700000002000000000",
                "endTimeUnixNano": "1700000003000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                    {"key": "iteration.index", "value": {"intValue": 1}},
                ],
                "status": {"code": 1},
            },
        ]

        payload = _generate_otlp_payload("cyclic_pipeline", spans=spans)
        resp = client.post("/v1/traces", json=payload)
        assert resp.status_code == 200

        from normalizer.otlp_normalizer import OTLPNormalizer

        trace = OTLPNormalizer().normalize_otlp_payload(payload)[0]
        assert len(trace.steps) == 2
        drafter_spans = [s for s in trace.spans if s.name == "drafter"]
        assert drafter_spans[0].iteration_index == 0
        assert drafter_spans[1].iteration_index == 1

    def test_otlp_error_span_and_arbiter_evaluation(self, client: TestClient) -> None:
        """Verify error in tool span triggers Arbiter tool_failure_v1 root cause attribution."""
        trace_id = uuid.uuid4().hex
        agent_id = uuid.uuid4().hex[:16]
        tool_id = uuid.uuid4().hex[:16]

        spans = [
            {
                "traceId": trace_id,
                "spanId": agent_id,
                "name": "data_engineer",
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000002000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                ],
                "status": {"code": 2, "message": "Sub-operation failed"},
            },
            {
                "traceId": trace_id,
                "spanId": tool_id,
                "parentSpanId": agent_id,
                "name": "sql_exec",
                "startTimeUnixNano": "1700000000500000000",
                "endTimeUnixNano": "1700000001000000000",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}},
                    {"key": "tool.name", "value": {"stringValue": "sql_exec"}},
                    {
                        "key": "error.message",
                        "value": {"stringValue": "ConnectionRefusedError: port 5432 unreachable"},
                    },
                ],
                "status": {"code": 2, "message": "ConnectionRefusedError"},
            },
        ]

        payload = _generate_otlp_payload("etl_pipeline", spans=spans)
        resp = client.post("/v1/traces?evaluate=true", json=payload)
        assert resp.status_code == 200

        from normalizer.otlp_normalizer import OTLPNormalizer

        trace = OTLPNormalizer().normalize_otlp_payload(payload)[0]
        assert trace.status == StepStatus.FAILURE

        bundle = agentlens.evaluate_run(trace)
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_agent == "data_engineer"

    def test_otlp_flat_spans_payload_compatibility(self, client: TestClient) -> None:
        """Verify flat span payload format without resourceSpans is parsed seamlessly."""
        trace_id = uuid.uuid4().hex
        payload = {
            "spans": [
                {
                    "traceId": trace_id,
                    "spanId": uuid.uuid4().hex[:16],
                    "name": "simple_worker",
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000000200000000",
                    "attributes": {
                        "openinference.span.kind": "AGENT",
                        "input.value": "work",
                        "output.value": "done",
                    },
                    "status": {"code": 1},
                }
            ]
        }

        resp = client.post("/v1/traces", json=payload)
        assert resp.status_code == 200
        assert resp.json()["traces_ingested"] == 1

    def test_otlp_invalid_payload_returns_400(self, client: TestClient) -> None:
        """Verify empty or non-OTLP payloads return HTTP 400."""
        # Completely empty payload
        resp1 = client.post("/v1/traces", json={})
        assert resp1.status_code == 400
        assert "missing resourceSpans" in resp1.json()["detail"]

        # Malformed object missing spans
        resp2 = client.post("/v1/traces", json={"foo": "bar"})
        assert resp2.status_code == 400

    def test_otlp_empty_spans_returns_400(self, client: TestClient) -> None:
        """Verify resourceSpans with empty spans list returns HTTP 400."""
        payload: dict[str, Any] = {"resourceSpans": [{"scopeSpans": [{"spans": []}]}]}
        resp = client.post("/v1/traces", json=payload)
        assert resp.status_code == 400
        assert "No valid traces could be reconstructed" in resp.json()["detail"]
