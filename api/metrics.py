"""
api/metrics.py — Prometheus Telemetry Exposition for AgentLens SRE
===================================================================
Day 70: Enterprise observability & SRE Prometheus metrics exporter.
Generates OpenMetrics / Prometheus 0.0.4 text exposition format.
"""

from __future__ import annotations

import logging
from typing import Any

import dashboard.state as _state
from analyzers.worker import JobStatus, get_job_manager

logger = logging.getLogger(__name__)


def generate_prometheus_metrics() -> str:
    """
    Generate Prometheus text exposition (OpenMetrics/Prometheus 0.0.4 format).

    Exposes:
      - agentlens_traces_ingested_total (counter)
      - agentlens_queue_depth (gauge)
      - agentlens_dropped_spans_total (counter)
      - agentlens_spilled_traces_total (counter)
      - agentlens_active_jobs (gauge)
      - agentlens_completed_jobs_total (counter)
      - agentlens_failed_jobs_total (counter)
      - agentlens_analysis_duration_seconds (gauge)
      - agentlens_total_runs (gauge)
      - agentlens_agent_latency_seconds (gauge, labeled by agent)
      - agentlens_agent_tokens_total (counter, labeled by agent)
    """
    lines: list[str] = []

    # 1. Ingestion Queue Metrics
    queue_depth = 0
    dropped_spans = 0
    spilled_traces = 0
    try:
        from capture.queue import get_global_queue

        q_stats: dict[str, Any] = get_global_queue().stats()
        queue_depth = int(q_stats.get("size", 0))
        dropped_spans = int(q_stats.get("dropped", 0))
        spilled_traces = int(q_stats.get("spilled", 0))
    except Exception as exc:
        logger.debug("[Prometheus] Could not retrieve queue stats: %s", exc)

    # 2. Database Run Count
    total_runs = 0
    try:
        db = _state.get_db()
        total_runs = len(db.list_runs(limit=100000))
    except Exception as exc:
        logger.debug("[Prometheus] Could not retrieve run count from DB: %s", exc)

    traces_ingested = total_runs

    # 3. Worker Job Manager Metrics
    active_jobs = 0
    completed_jobs = 0
    failed_jobs = 0
    last_duration_s = 0.0
    try:
        manager = get_job_manager()
        jobs = manager.list_jobs(limit=1000)
        active_jobs = sum(1 for j in jobs if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING))
        completed_jobs = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
        failed_jobs = sum(1 for j in jobs if j.status == JobStatus.FAILED)

        durations = [
            j.execution_time_ms / 1000.0
            for j in jobs
            if j.status == JobStatus.COMPLETED and j.execution_time_ms is not None
        ]
        if durations:
            last_duration_s = durations[0]
    except Exception as exc:
        logger.debug("[Prometheus] Could not retrieve job manager stats: %s", exc)

    # 4. Per-Agent Latency & Tokens
    agent_data: dict[str, dict[str, Any]] = {}
    try:
        agent_data = _state.get_metrics_data()
        if not agent_data:
            import json

            db = _state.get_db()
            runs = db.list_runs(limit=200)
            raw_agents: dict[str, dict[str, Any]] = {}
            for r in runs:
                trace_json_str = r.get("trace_json")
                if trace_json_str:
                    try:
                        t_data = json.loads(trace_json_str)
                        for s in t_data.get("steps", []):
                            ag = s.get("agent", "unknown")
                            lat = float(s.get("latency_ms", 0.0) or 0.0)
                            toks_val = s.get("tokens", {})
                            tok = (
                                int(toks_val.get("total", 0))
                                if isinstance(toks_val, dict)
                                else int(s.get("tokens_total", 0) or 0)
                            )
                            if ag not in raw_agents:
                                raw_agents[ag] = {"latencies": [], "tokens": []}
                            raw_agents[ag]["latencies"].append(lat)
                            raw_agents[ag]["tokens"].append(tok)
                    except Exception:
                        pass
            for ag, d in raw_agents.items():
                lats = d["latencies"]
                toks = d["tokens"]
                agent_data[ag] = {
                    "avg_latency_ms": sum(lats) / len(lats) if lats else 0.0,
                    "max_latency_ms": max(lats) if lats else 0.0,
                    "avg_tokens": sum(toks) / len(toks) if toks else 0.0,
                    "total_tokens": sum(toks),
                    "run_count": len(lats),
                }
    except Exception as exc:
        logger.debug("[Prometheus] Could not retrieve agent metrics: %s", exc)

    # Build Prometheus text lines
    lines.append(
        "# HELP agentlens_traces_ingested_total Total number of traces ingested into AgentLens."
    )
    lines.append("# TYPE agentlens_traces_ingested_total counter")
    lines.append(f"agentlens_traces_ingested_total {traces_ingested}")
    lines.append("")

    lines.append(
        "# HELP agentlens_queue_depth Current number of traces waiting in the in-memory ingestion queue."
    )
    lines.append("# TYPE agentlens_queue_depth gauge")
    lines.append(f"agentlens_queue_depth {queue_depth}")
    lines.append("")

    lines.append(
        "# HELP agentlens_dropped_spans_total Total number of traces/spans dropped due to queue saturation."
    )
    lines.append("# TYPE agentlens_dropped_spans_total counter")
    lines.append(f"agentlens_dropped_spans_total {dropped_spans}")
    lines.append("")

    lines.append(
        "# HELP agentlens_spilled_traces_total Total number of traces spilled to disk Write-Ahead Log."
    )
    lines.append("# TYPE agentlens_spilled_traces_total counter")
    lines.append(f"agentlens_spilled_traces_total {spilled_traces}")
    lines.append("")

    lines.append(
        "# HELP agentlens_active_jobs Current number of background analysis jobs in queued or processing state."
    )
    lines.append("# TYPE agentlens_active_jobs gauge")
    lines.append(f"agentlens_active_jobs {active_jobs}")
    lines.append("")

    lines.append(
        "# HELP agentlens_completed_jobs_total Total background analysis jobs completed successfully."
    )
    lines.append("# TYPE agentlens_completed_jobs_total counter")
    lines.append(f"agentlens_completed_jobs_total {completed_jobs}")
    lines.append("")

    lines.append("# HELP agentlens_failed_jobs_total Total background analysis jobs that failed.")
    lines.append("# TYPE agentlens_failed_jobs_total counter")
    lines.append(f"agentlens_failed_jobs_total {failed_jobs}")
    lines.append("")

    lines.append(
        "# HELP agentlens_analysis_duration_seconds Execution duration of the most recent analysis job in seconds."
    )
    lines.append("# TYPE agentlens_analysis_duration_seconds gauge")
    lines.append(f"agentlens_analysis_duration_seconds {last_duration_s:.4f}")
    lines.append("")

    lines.append("# HELP agentlens_total_runs Total number of runs stored in the active database.")
    lines.append("# TYPE agentlens_total_runs gauge")
    lines.append(f"agentlens_total_runs {total_runs}")
    lines.append("")

    if agent_data:
        lines.append(
            "# HELP agentlens_agent_latency_seconds Average execution latency per agent step in seconds."
        )
        lines.append("# TYPE agentlens_agent_latency_seconds gauge")
        for ag in sorted(agent_data.keys()):
            v = agent_data[ag]
            lat_sec = float(v.get("avg_latency_ms", 0.0)) / 1000.0
            safe_ag = ag.replace('"', '\\"')
            lines.append(f'agentlens_agent_latency_seconds{{agent="{safe_ag}"}} {lat_sec:.6f}')
        lines.append("")

        lines.append("# HELP agentlens_agent_tokens_total Total tokens consumed per agent.")
        lines.append("# TYPE agentlens_agent_tokens_total counter")
        for ag in sorted(agent_data.keys()):
            v = agent_data[ag]
            tokens = int(v.get("total_tokens", 0))
            safe_ag = ag.replace('"', '\\"')
            lines.append(f'agentlens_agent_tokens_total{{agent="{safe_ag}"}} {tokens}')
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
