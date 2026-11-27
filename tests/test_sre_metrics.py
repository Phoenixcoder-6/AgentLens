"""
tests/test_sre_metrics.py — Day 70 Prometheus SRE Metrics & Safe Deployment Tests
==================================================================================
Phase E: Data Durability, Storage Reliability & SRE

Tests:
1. Prometheus text exposition format conforms to OpenMetrics/Prometheus 0.0.4 standard.
2. Content negotiation: Accept header determines text/plain vs application/json.
3. Dedicated scrape paths: /api/metrics/prometheus and root /metrics.
4. Live telemetry accuracy: queue depth, dropped spans, active/completed jobs, run count.
5. Per-agent labeled metrics: latency seconds and token counters.
6. Deployment safety guard: auto-seed disabled in production by default.
7. Explicit AGENTLENS_AUTO_SEED override control.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import dashboard.state as _state
from analyzers.worker import JobStatus, get_job_manager
from api.metrics import generate_prometheus_metrics
from api.router import health_router, router
from capture.queue import get_global_queue
from dashboard.app import is_auto_seed_enabled
from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
from storage.db import DatabaseManager


@pytest.fixture
def sre_client(tmp_path: Path, monkeypatch) -> TestClient:
    """FastAPI TestClient with isolated temporary database and clean queue."""
    test_db_path = str(tmp_path / "test_sre_agentlens.db")
    db = DatabaseManager(test_db_path)
    db.initialize()

    monkeypatch.setattr(_state, "get_db", lambda: db)
    monkeypatch.setattr(_state, "_db", db)
    _state._analysis_cache.clear()

    # Reset worker manager
    manager = get_job_manager()
    manager.clear()

    # Reset ingestion queue
    q = get_global_queue()
    q.clear()

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.include_router(health_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_evidence_extractor(monkeypatch):
    """Bypass remote LLM extraction calls to prevent connection timeouts in CI."""
    from analyzers.evidence_extraction.extractor import EvidenceExtractor, ExtractedEvidence

    def fake_extract(self, raw_output: str, agent: str = "") -> ExtractedEvidence:
        claim = raw_output[:80] if raw_output else "sample claim"
        return ExtractedEvidence(
            claims=[claim],
            source_count=1,
            entity_count=1,
            token_cost=50,
        )

    monkeypatch.setattr(EvidenceExtractor, "extract", fake_extract)


def _seed_sre_run(run_id: str = "sre_run_01") -> None:
    """Insert a test run with steps into the active DB."""
    db = _state.get_db()
    step1 = AgentStep(
        run_id=run_id,
        step=1,
        agent="researcher",
        status=StepStatus.SUCCESS,
        output="Market analysis complete.",
        tokens=TokenUsage(prompt=30, completion=20, total=50),
        latency_ms=100.0,
    )
    step2 = AgentStep(
        run_id=run_id,
        step=2,
        agent="writer",
        status=StepStatus.SUCCESS,
        output="Final report generated.",
        tokens=TokenUsage(prompt=50, completion=25, total=75),
        latency_ms=150.0,
    )
    trace = RunTrace(
        run_id=run_id,
        workflow="market_analysis",
        status=StepStatus.SUCCESS,
        total_latency_ms=250.0,
        total_tokens=125,
        steps=[step1, step2],
    )
    db.insert_run(
        run_id=run_id,
        workflow=trace.workflow,
        timestamp="2026-11-28T00:00:00Z",
        status=trace.status,
        total_latency_ms=trace.total_latency_ms,
        total_tokens=trace.total_tokens,
        schema_version="1.0",
        trace_json=trace.model_dump_json(),
    )
    for s in [step1, step2]:
        db.insert_step(
            run_id=run_id,
            step=s.step,
            agent=s.agent,
            status=str(s.status),
            latency_ms=s.latency_ms,
            tokens_prompt=s.tokens.prompt,
            tokens_completion=s.tokens.completion,
            tokens_total=s.tokens.total,
            diff_summary="no diff",
            timestamp="2026-11-28T00:00:00Z",
            schema_version="1.0",
        )


def test_prometheus_exposition_format(sre_client: TestClient):
    """GET /api/metrics?format=prometheus returns standard OpenMetrics/Prometheus 0.0.4 text."""
    _seed_sre_run("sre_format_run")
    resp = sre_client.get("/api/metrics?format=prometheus")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "version=0.0.4" in resp.headers["content-type"]

    body = resp.text
    # Assert standard Prometheus HELP and TYPE annotations
    assert "# HELP agentlens_traces_ingested_total" in body
    assert "# TYPE agentlens_traces_ingested_total counter" in body
    assert "# HELP agentlens_queue_depth" in body
    assert "# TYPE agentlens_queue_depth gauge" in body
    assert "# HELP agentlens_dropped_spans_total" in body
    assert "# TYPE agentlens_dropped_spans_total counter" in body
    assert "# HELP agentlens_active_jobs" in body
    assert "# TYPE agentlens_active_jobs gauge" in body
    assert "# HELP agentlens_analysis_duration_seconds" in body
    assert "# TYPE agentlens_analysis_duration_seconds gauge" in body


def test_metrics_content_negotiation(sre_client: TestClient):
    """Content-negotiation via Accept header routes between text and JSON."""
    _seed_sre_run("sre_negotiation_run")

    # 1. Accept text/plain -> Prometheus format
    resp_text = sre_client.get("/api/metrics", headers={"Accept": "text/plain"})
    assert resp_text.status_code == 200
    assert "text/plain" in resp_text.headers["content-type"]
    assert "agentlens_traces_ingested_total" in resp_text.text

    # 2. Accept application/json -> Backward compatible JSON
    resp_json = sre_client.get("/api/metrics", headers={"Accept": "application/json"})
    assert resp_json.status_code == 200
    assert "application/json" in resp_json.headers["content-type"]
    data = resp_json.json()
    assert "agents" in data
    assert "total_runs" in data
    assert data["total_runs"] >= 1


def test_dedicated_prometheus_endpoints(sre_client: TestClient):
    """Scrape paths /api/metrics/prometheus and root /metrics return text format."""
    resp_api = sre_client.get("/api/metrics/prometheus")
    assert resp_api.status_code == 200
    assert "text/plain" in resp_api.headers["content-type"]
    assert "agentlens_queue_depth" in resp_api.text

    resp_root = sre_client.get("/metrics")
    assert resp_root.status_code == 200
    assert "text/plain" in resp_root.headers["content-type"]
    assert "agentlens_traces_ingested_total" in resp_root.text


def test_live_telemetry_accuracy(sre_client: TestClient):
    """Prometheus values accurately reflect database runs, queue state, and worker jobs."""
    _seed_sre_run("telemetry_run_01")
    _seed_sre_run("telemetry_run_02")

    manager = get_job_manager()
    job = manager.submit_job("telemetry_run_01")
    finished_job = manager.wait_for_job(job.job_id, timeout=25.0)
    assert finished_job is not None
    assert finished_job.status == JobStatus.COMPLETED

    text = generate_prometheus_metrics()

    # Parse key metric lines
    metrics_map: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 2 and "{" not in parts[0]:
                metrics_map[parts[0]] = float(parts[1])

    assert metrics_map.get("agentlens_traces_ingested_total", 0) >= 2.0
    assert metrics_map.get("agentlens_total_runs", 0) >= 2.0
    assert metrics_map.get("agentlens_queue_depth", -1) >= 0.0
    assert metrics_map.get("agentlens_dropped_spans_total", -1) >= 0.0
    assert metrics_map.get("agentlens_completed_jobs_total", 0) >= 1.0


def test_per_agent_latency_and_token_labels(sre_client: TestClient):
    """Per-agent step metrics surface with standard label syntax."""
    _seed_sre_run("labeled_run_01")

    resp = sre_client.get("/api/metrics/prometheus")
    text = resp.text

    assert 'agentlens_agent_latency_seconds{agent="researcher"}' in text
    assert 'agentlens_agent_latency_seconds{agent="writer"}' in text
    assert 'agentlens_agent_tokens_total{agent="researcher"}' in text
    assert 'agentlens_agent_tokens_total{agent="writer"}' in text


def test_auto_seed_disabled_in_production(monkeypatch):
    """Auto-seeding is suppressed by default when running in production."""
    # Ensure explicit env is unset
    monkeypatch.delenv("AGENTLENS_AUTO_SEED", raising=False)

    monkeypatch.setenv("ENVIRONMENT", "production")
    assert is_auto_seed_enabled() is False

    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("AGENTLENS_ENV", "production")
    assert is_auto_seed_enabled() is False

    monkeypatch.delenv("AGENTLENS_ENV", raising=False)
    monkeypatch.setenv("NODE_ENV", "prod")
    assert is_auto_seed_enabled() is False


def test_auto_seed_explicit_override(monkeypatch):
    """Explicit AGENTLENS_AUTO_SEED variable overrides environment defaults."""
    # Explicit 0 blocks seeding even in development
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("AGENTLENS_ENV", raising=False)
    monkeypatch.setenv("AGENTLENS_AUTO_SEED", "0")
    assert is_auto_seed_enabled() is False

    monkeypatch.setenv("AGENTLENS_AUTO_SEED", "false")
    assert is_auto_seed_enabled() is False

    # Explicit 1 forces seeding even in production
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("AGENTLENS_AUTO_SEED", "1")
    assert is_auto_seed_enabled() is True
