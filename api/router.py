"""
api/router.py — AgentLens REST API (Day 29)
============================================
FastAPI router exposing 6 endpoints.  Mounted inside the NiceGUI process
via ``app.mount("/api", fastapi_app)`` in dashboard/app.py.

Endpoints
---------
GET  /api/runs                    → list[RunSummary]
GET  /api/runs/{run_id}           → RunDetail
GET  /api/runs/{run_id}/verdict   → VerdictResponse
POST /api/analyze/{run_id}        → AnalysisJobResponse
GET  /api/metrics                 → MetricsResponse
GET  /health                      → HealthResponse  (mounted at root level)

Auth
----
Optional.  Set ``api_key`` under the ``api`` section in config.yaml::

    api:
      api_key: "your-secret"

Then callers must pass ``X-AgentLens-Key: your-secret``.
If no key is configured the header is ignored.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from pydantic import BaseModel, Field

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dashboard.state as _state  # noqa: E402

# ── Optional auth ─────────────────────────────────────────────────────────────


def _configured_api_key() -> str | None:
    """Return the API key from config.yaml, or None if not set."""
    try:
        from config.config_loader import get as cfg_get

        return cfg_get("api", "api_key") or None
    except Exception:
        return None


def verify_api_key(x_agentlens_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce API key if one is configured."""
    required = _configured_api_key()
    if required and x_agentlens_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing X-AgentLens-Key header")


# ── Response models ───────────────────────────────────────────────────────────


class RunSummary(BaseModel):
    run_id: str
    workflow: str
    topic: str
    timestamp: str
    status: str
    latency_ms: float
    tokens_total: int
    step_count: int
    verdict_level: str


class RunDetail(RunSummary):
    steps: list[dict[str, Any]]


class VerdictResponse(BaseModel):
    run_id: str
    priority_level: str
    primary_cause: str
    primary_agent: str | None
    grounded: bool
    confidence: float | None
    verdict_reason: str | None


class AnalysisJobResponse(BaseModel):
    run_id: str
    status: str  # "complete" | "error"
    priority_level: str | None
    primary_agent: str | None
    error: str | None


class AgentMetrics(BaseModel):
    agent: str
    avg_latency_ms: float
    max_latency_ms: float
    avg_tokens: float
    total_tokens: int
    run_count: int


class MetricsResponse(BaseModel):
    agents: list[AgentMetrics]
    total_runs: int


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    db: str  # "connected" | "disconnected" (Day 44)
    llm: str  # "reachable" | "unreachable" (Day 44)
    uptime_seconds: float
    db_status: str = "ok"  # "ok" | "error" (Day 29 backward compat)
    version: str = "1.0.0"


class OTLPResponse(BaseModel):
    partialSuccess: dict[str, Any] = Field(default_factory=dict)
    traces_ingested: int = 0
    run_ids: list[str] = Field(default_factory=list)


# ── Startup time for uptime calculation ───────────────────────────────────────
_START_TIME = datetime.datetime.now(datetime.UTC)

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = 50) -> list[RunSummary]:
    """List runs with cached verdict levels, newest first."""
    rows = _state.list_runs(limit=limit)
    return [
        RunSummary(
            run_id=r.run_id,
            workflow=r.workflow,
            topic=r.topic,
            timestamp=r.timestamp,
            status=r.status,
            latency_ms=r.latency_ms,
            tokens_total=r.tokens_total,
            step_count=r.step_count,
            verdict_level=r.verdict_level,
        )
        for r in rows
    ]


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    """Full run detail including step list."""
    db = _state.get_db()
    row = db.get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    steps = _state.get_steps(run_id)
    step_dicts = [
        {
            "step": s.step,
            "agent": s.agent,
            "status": s.status,
            "latency_ms": s.latency_ms,
            "tokens_total": s.tokens_total,
        }
        for s in steps
    ]

    runs = _state.list_runs(limit=500)
    run_row = next((r for r in runs if r.run_id == run_id), None)
    lat = run_row.latency_ms if run_row else 0.0
    tok = run_row.tokens_total if run_row else 0
    verdict_level = run_row.verdict_level if run_row else "UNANALYZED"

    return RunDetail(
        run_id=run_id,
        workflow=row.get("workflow", "unknown"),
        topic=run_row.topic if run_row else row.get("workflow", "unknown"),
        timestamp=(row.get("timestamp", "")[:19] or "").replace("T", " "),
        status=row.get("status", "unknown"),
        latency_ms=lat,
        tokens_total=tok,
        step_count=len(steps),
        verdict_level=verdict_level,
        steps=step_dicts,
    )


@router.get("/runs/{run_id}/verdict", response_model=VerdictResponse)
def get_verdict(run_id: str) -> VerdictResponse:
    """Return the cached analysis verdict for a run.  404 if not yet analyzed."""
    cached = _state.get_cached(run_id)
    if not cached or not cached.bundle:
        raise HTTPException(
            status_code=404,
            detail=f"No verdict for run '{run_id}' — trigger POST /api/analyze/{run_id} first",
        )
    bundle = cached.bundle
    conf: float | None = None
    if cached.loss_result:
        conf = cached.loss_result.confidence

    return VerdictResponse(
        run_id=run_id,
        priority_level=bundle.priority_level.value,
        primary_cause=bundle.primary_cause.value,
        primary_agent=bundle.primary_agent,
        grounded=bundle.grounded,
        confidence=conf,
        verdict_reason=bundle.summary,
    )


@router.post("/analyze/{run_id}", response_model=AnalysisJobResponse)
def trigger_analysis(run_id: str) -> AnalysisJobResponse:
    """Run the full analysis pipeline for a run synchronously and return the verdict."""
    # Clear cache so re-analysis is always fresh
    _state._analysis_cache.pop(run_id, None)
    result = _state.run_full_analysis(run_id)

    if result.error or not result.bundle:
        return AnalysisJobResponse(
            run_id=run_id,
            status="error",
            priority_level=None,
            primary_agent=None,
            error=result.error or "Analysis failed",
        )

    return AnalysisJobResponse(
        run_id=run_id,
        status="complete",
        priority_level=result.bundle.priority_level.value,
        primary_agent=result.bundle.primary_agent,
        error=None,
    )


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Aggregate per-agent latency and token metrics across all stored runs."""
    data = _state.get_metrics_data()
    db = _state.get_db()
    total_runs = len(db.list_runs(limit=10000))

    agents = [
        AgentMetrics(
            agent=ag,
            avg_latency_ms=round(v["avg_latency_ms"], 2),
            max_latency_ms=round(v["max_latency_ms"], 2),
            avg_tokens=round(v["avg_tokens"], 1),
            total_tokens=v["total_tokens"],
            run_count=v["run_count"],
        )
        for ag, v in sorted(data.items())
    ]
    return MetricsResponse(agents=agents, total_runs=total_runs)


# ── Health endpoint (separate, no auth) ───────────────────────────────────────
health_router = APIRouter()


def _check_llm_reachability() -> str:
    """
    Lightweight readiness check for the LLM subsystem (Day 44).
    Verifies LLM configuration and driver availability without burning API tokens
    on every 30s container healthcheck poll.
    """
    import os

    if os.getenv("AGENTLENS_FORCE_LLM_UNREACHABLE") == "1":
        return "unreachable"
    try:
        from config.config_loader import get as cfg_get

        model_name = cfg_get("llm", "model")
        if not model_name:
            return "unreachable"
        import langchain_groq  # noqa: F401

        return "reachable"
    except Exception:
        return "unreachable"


@health_router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness / readiness check (Day 29 & Day 44)."""
    uptime = max(0.0, (datetime.datetime.now(datetime.UTC) - _START_TIME).total_seconds())

    db_status = "ok"
    db_conn = "connected"
    try:
        db = _state.get_db()
        db.list_runs(limit=1)
    except Exception:
        db_status = "error"
        db_conn = "disconnected"

    llm_state = _check_llm_reachability()
    overall = "ok" if (db_conn == "connected" and llm_state == "reachable") else "degraded"

    return HealthResponse(
        status=overall,
        db=db_conn,
        llm=llm_state,
        uptime_seconds=round(uptime, 1),
        db_status=db_status,
        version="1.0.0",
    )


# ── OTLP Ingestion Handler (Day 60) ──────────────────────────────────────────


def _process_otlp_ingestion(
    payload: dict[str, Any],
    evaluate: bool = False,
) -> OTLPResponse:
    """Core logic to normalize and persist an OpenTelemetry / OpenInference trace payload."""
    if not payload or ("resourceSpans" not in payload and "spans" not in payload):
        raise HTTPException(
            status_code=400,
            detail="Invalid OTLP trace payload: missing resourceSpans or spans",
        )

    from normalizer.normalizer import Normalizer
    from normalizer.otlp_normalizer import OTLPNormalizer
    from storage.writer import StorageWriter

    normalizer = OTLPNormalizer()
    run_traces = normalizer.normalize_otlp_payload(payload)

    if not run_traces:
        raise HTTPException(
            status_code=400,
            detail="No valid traces could be reconstructed from OTLP payload",
        )

    db = _state.get_db()
    writer = StorageWriter(db)
    legacy_normalizer = Normalizer()

    saved_run_ids: list[str] = []
    for rt in run_traces:
        norm_run = legacy_normalizer.normalize_run(rt)
        writer.write_run(norm_run, trace_json=rt.model_dump_json(indent=2))
        saved_run_ids.append(rt.run_id)

        if evaluate:
            try:
                _state.run_full_analysis(rt.run_id)
            except Exception:
                pass

    return OTLPResponse(
        partialSuccess={},
        traces_ingested=len(saved_run_ids),
        run_ids=saved_run_ids,
    )


# OTLP Router mounted at root level (/v1/traces and /traces)
otlp_router = APIRouter(dependencies=[Depends(verify_api_key)])


@otlp_router.post("/v1/traces", response_model=OTLPResponse)
@otlp_router.post("/traces", response_model=OTLPResponse)
def ingest_otlp_traces_root(
    payload: dict[str, Any] = Body(...),
    evaluate: bool = False,
) -> OTLPResponse:
    """Standard OpenTelemetry HTTP traces ingestion endpoint."""
    return _process_otlp_ingestion(payload, evaluate=evaluate)


# Also register on the /api router for clients configured with /api prefix (/api/v1/traces)
@router.post("/v1/traces", response_model=OTLPResponse)
@router.post("/traces", response_model=OTLPResponse)
def ingest_otlp_traces_api(
    payload: dict[str, Any] = Body(...),
    evaluate: bool = False,
) -> OTLPResponse:
    """Namespaced /api/v1/traces endpoint."""
    return _process_otlp_ingestion(payload, evaluate=evaluate)
