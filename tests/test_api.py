"""
tests/test_api.py — REST API tests (Day 29)

Uses FastAPI's built-in TestClient (Starlette), which supports ASGI apps
synchronously — no async test runner needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Synchronous TestClient backed by the FastAPI app."""
    from fastapi.testclient import TestClient

    from api.main import fastapi_app
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        body = client.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert "uptime_seconds" in body
        assert "db_status" in body
        assert "version" in body


class TestRunsListEndpoint:
    def test_runs_returns_200(self, client):
        resp = client.get("/api/runs")
        assert resp.status_code == 200

    def test_runs_returns_list(self, client):
        body = client.get("/api/runs").json()
        assert isinstance(body, list)

    def test_runs_with_limit(self, client):
        resp = client.get("/api/runs?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) <= 5

    def test_runs_schema(self, client):
        body = client.get("/api/runs?limit=1").json()
        if body:
            run = body[0]
            assert "run_id" in run
            assert "verdict_level" in run
            assert "latency_ms" in run


class TestRunDetailEndpoint:
    def test_missing_run_returns_404(self, client):
        resp = client.get("/api/runs/nonexistent-run-id-xyz")
        assert resp.status_code == 404

    def test_existing_run_if_any(self, client):
        """If any runs exist, fetch the first one and verify schema."""
        runs = client.get("/api/runs?limit=1").json()
        if not runs:
            pytest.skip("No runs in DB — skipping detail test")
        run_id = runs[0]["run_id"]
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run_id
        assert "steps" in body
        assert isinstance(body["steps"], list)


class TestVerdictEndpoint:
    def test_unanalyzed_run_returns_404(self, client):
        """A run that has not been analyzed has no verdict → 404."""
        runs = client.get("/api/runs?limit=1").json()
        if not runs:
            pytest.skip("No runs in DB")
        run_id = runs[0]["run_id"]
        resp = client.get(f"/api/runs/{run_id}/verdict")
        # Acceptable: 200 (if already cached) or 404 (if not analyzed)
        assert resp.status_code in (200, 404)

    def test_nonexistent_run_verdict_404(self, client):
        resp = client.get("/api/runs/totally-fake-run-id/verdict")
        assert resp.status_code == 404


class TestAnalyzeEndpoint:
    def test_nonexistent_run_returns_error_payload(self, client):
        resp = client.post("/api/analyze/nonexistent-run-id")
        # Should return 200 with status=error (not 404) since we surface errors as payload
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert body["run_id"] == "nonexistent-run-id"

    def test_analyze_existing_run_if_any(self, client):
        """If a run exists, trigger analysis and verify response shape."""
        runs = client.get("/api/runs?limit=1").json()
        if not runs:
            pytest.skip("No runs in DB")
        run_id = runs[0]["run_id"]
        resp = client.post(f"/api/analyze/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["run_id"] == run_id
        assert body["status"] in ("complete", "error")


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200

    def test_metrics_schema(self, client):
        body = client.get("/api/metrics").json()
        assert "agents" in body
        assert "total_runs" in body
        assert isinstance(body["agents"], list)
        if body["agents"]:
            ag = body["agents"][0]
            assert "agent" in ag
            assert "avg_latency_ms" in ag
            assert "run_count" in ag
