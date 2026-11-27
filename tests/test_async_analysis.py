"""
tests/test_async_analysis.py — Day 69 Asynchronous Analysis Job Tests
======================================================================
Phase E: Data Durability, Storage Reliability & SRE

Tests:
1. POST /api/analyze/{run_id} returns HTTP 202 Accepted immediately (<50ms).
2. Lifecycle state transitions: QUEUED -> PROCESSING -> COMPLETED.
3. Polling endpoint GET /api/jobs/{job_id} returns complete analysis results.
4. High-concurrency burst: 20 simultaneous analysis requests return 202 without timeouts.
5. Error handling: invalid run transitions to FAILED with descriptive error message.
6. Backward compatibility: POST /api/analyze/{run_id}?sync=true returns 200 OK.
7. Webhook dispatch: completed job delivers HTTP callback payload to configured URL.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import dashboard.state as _state
from analyzers.worker import JobStatus, get_job_manager
from api.router import router
from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
from storage.db import DatabaseManager


@pytest.fixture(autouse=True)
def clean_job_manager():
    """Ensure clean job manager state before and after each test."""
    manager = get_job_manager()
    manager.clear()
    yield manager
    manager.clear()


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    """Create a FastAPI test client backed by a temporary test database."""
    test_db_path = str(tmp_path / "test_async_agentlens.db")
    db = DatabaseManager(test_db_path)
    db.initialize()

    monkeypatch.setattr(_state, "get_db", lambda: db)
    monkeypatch.setattr(_state, "_db", db)
    _state._analysis_cache.clear()

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _seed_sample_run(client: TestClient, run_id: str) -> None:
    """Helper to seed a valid RunTrace into the active database."""
    db = _state.get_db()
    step1 = AgentStep(
        run_id=run_id,
        step=1,
        agent="researcher",
        status=StepStatus.SUCCESS,
        output="The company revenue grew by 15% in Q3 reaching $10M.",
        tokens=TokenUsage(prompt=50, completion=30, total=80),
        latency_ms=120.0,
    )
    step2 = AgentStep(
        run_id=run_id,
        step=2,
        agent="writer",
        status=StepStatus.SUCCESS,
        output="Report: Revenue declined by 10% in Q3.",
        tokens=TokenUsage(prompt=80, completion=40, total=120),
        latency_ms=150.0,
    )
    trace = RunTrace(
        run_id=run_id,
        workflow="financial_report",
        status=StepStatus.FAILURE,
        total_latency_ms=270.0,
        total_tokens=200,
        steps=[step1, step2],
        expected_output="The company revenue grew by 15% in Q3 reaching $10M.",
    )
    db.insert_run(
        run_id=run_id,
        workflow=trace.workflow,
        timestamp="2026-11-27T20:00:00Z",
        status=trace.status,
        total_latency_ms=trace.total_latency_ms,
        total_tokens=trace.total_tokens,
        schema_version="1.0",
        trace_json=trace.model_dump_json(),
        expected_output=trace.expected_output,
    )


def test_async_analysis_202_accepted(client: TestClient):
    """POST /api/analyze/{run_id} returns HTTP 202 Accepted in <50ms with job metadata."""
    run_id = "async_run_01"
    _seed_sample_run(client, run_id)

    start_time = time.perf_counter()
    resp = client.post(f"/api/analyze/{run_id}")
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    assert resp.status_code == 202
    assert duration_ms < 50.0, f"Submission latency {duration_ms:.2f}ms exceeded 50ms SLA"

    data = resp.json()
    assert data["run_id"] == run_id
    assert data["status"] == "queued"
    assert "job_id" in data
    assert data["poll_url"] == f"/api/jobs/{data['job_id']}"
    assert "created_at" in data


def test_job_lifecycle_transitions(client: TestClient):
    """Background worker transitions job state from queued -> processing -> completed."""
    run_id = "lifecycle_run_01"
    _seed_sample_run(client, run_id)

    resp = client.post(f"/api/analyze/{run_id}")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    manager = get_job_manager()
    job = manager.wait_for_job(job_id, timeout=8.0)

    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None
    assert job.execution_time_ms is not None
    assert job.execution_time_ms > 0.0


def test_job_polling_result_structure(client: TestClient):
    """GET /api/jobs/{job_id} surfaces priority, verdict reason, and causal artifacts."""
    run_id = "polling_run_01"
    _seed_sample_run(client, run_id)

    resp = client.post(f"/api/analyze/{run_id}")
    job_id = resp.json()["job_id"]

    manager = get_job_manager()
    job = manager.wait_for_job(job_id, timeout=8.0)
    assert job is not None and job.status == JobStatus.COMPLETED

    poll_resp = client.get(f"/api/jobs/{job_id}")
    assert poll_resp.status_code == 200
    data = poll_resp.json()

    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert data["priority_level"] in ("P1", "P2", "P3", "P4", "P5")
    assert data["primary_cause"] is not None
    assert "execution_time_ms" in data


def test_concurrent_analysis_burst(client: TestClient):
    """Burst of 20 concurrent requests return 202 without thread starvation or timeouts."""
    run_ids = [f"burst_run_{i:02d}" for i in range(20)]
    for r in run_ids:
        _seed_sample_run(client, r)

    def trigger(rid: str) -> tuple[int, dict[str, Any]]:
        t0 = time.perf_counter()
        r = client.post(f"/api/analyze/{rid}")
        ms = (time.perf_counter() - t0) * 1000.0
        return r.status_code, {"ms": ms, "body": r.json()}

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(trigger, run_ids))
    total_submit_ms = (time.perf_counter() - start_total) * 1000.0

    # Assert all 20 accepted
    for code, info in results:
        assert code == 202
        assert info["body"]["status"] == "queued"
    # Total submission time under 500ms
    assert total_submit_ms < 500.0, f"Total burst submission time {total_submit_ms:.2f}ms too slow"

    # Wait for all jobs to complete
    manager = get_job_manager()
    for _, info in results:
        jid = info["body"]["job_id"]
        job = manager.wait_for_job(jid, timeout=25.0)
        assert job is not None
        assert job.status in (JobStatus.COMPLETED, JobStatus.FAILED)


def test_failed_job_handling(client: TestClient):
    """Analysis on non-existent or invalid run gracefully records FAILED state."""
    run_id = "nonexistent_fake_run_999"
    resp = client.post(f"/api/analyze/{run_id}")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    manager = get_job_manager()
    job = manager.wait_for_job(job_id, timeout=5.0)

    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error is not None
    assert "trace_json not found" in job.error or "failed" in job.error.lower()

    poll_resp = client.get(f"/api/jobs/{job_id}")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["status"] == "failed"


def test_sync_mode_backward_compatibility(client: TestClient):
    """POST /api/analyze/{run_id}?sync=true processes synchronously and returns 200 OK."""
    run_id = "sync_compat_run_01"
    _seed_sample_run(client, run_id)

    resp = client.post(f"/api/analyze/{run_id}?sync=true")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "complete"
    assert data["run_id"] == run_id
    assert data["priority_level"] is not None


def test_webhook_dispatch_on_complete(client: TestClient):
    """Completed job dispatches HTTP POST callback to configured webhook URL."""
    run_id = "webhook_run_01"
    _seed_sample_run(client, run_id)

    # Lightweight local HTTP server to receive webhook
    received_payloads: list[dict[str, Any]] = []
    server_error: list[str] = []

    class WebhookHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                received_payloads.append(data)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"received"}')
            except Exception as e:
                server_error.append(str(e))
                self.send_response(500)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress console log spam

    server = http.server.HTTPServer(("127.0.0.1", 0), WebhookHandler)
    server_port = server.server_address[1]
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    webhook_url = f"http://127.0.0.1:{server_port}/callback"
    resp = client.post(f"/api/analyze/{run_id}", json={"webhook_url": webhook_url})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    manager = get_job_manager()
    job = manager.wait_for_job(job_id, timeout=8.0)
    assert job is not None and job.status == JobStatus.COMPLETED

    server_thread.join(timeout=3.0)
    server.server_close()

    assert len(received_payloads) == 1
    assert received_payloads[0]["job_id"] == job_id
    assert received_payloads[0]["status"] == "completed"
    assert job.webhook_status == "delivered"
