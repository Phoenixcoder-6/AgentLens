"""
analyzers/metrics_analyzer.py — Step Metrics Analyzer
======================================================
Day 11: Computes latency, token usage, and execution time per step
from the SQLite database and flags anomalies against config thresholds.

Three metric types:
    latency_ms        — LLM call duration in milliseconds
    tokens            — prompt / completion / total token counts
    execution_time_ms — wall-clock step time (= latency_ms for current pipeline)

Anomaly detection:
    - Flag steps where latency_ms > latency_threshold_ms (from config)
    - Flag steps where tokens_total > mean + N * stddev (when enough runs exist)
    - Requires min_runs_for_baseline runs before statistical detection activates

Output: MetricsReport — structured per-step and run-level metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from schema.models import SCHEMA_VERSION
from storage.db import DatabaseManager
from config.config_loader import get


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    """Metrics for a single agent step."""
    schema_version: str
    run_id: str
    step: int
    agent: str

    # Latency
    latency_ms: float
    latency_flag: bool          # True if latency exceeds threshold
    latency_threshold_ms: float

    # Tokens
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int

    # Execution time (wall-clock — same as latency_ms for current pipeline)
    execution_time_ms: float

    # Anomaly
    is_anomalous: bool
    anomaly_reasons: list[str] = field(default_factory=list)


@dataclass
class RunMetrics:
    """Aggregated metrics for a complete pipeline run."""
    schema_version: str
    run_id: str
    workflow: str

    # Step-level breakdown
    steps: list[StepMetrics] = field(default_factory=list)

    # Run-level aggregates
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_execution_time_ms: float = 0.0
    step_count: int = 0

    # Per-agent summaries (agent_name → latency)
    agent_latency: dict[str, float] = field(default_factory=dict)

    # Slowest and fastest steps
    slowest_step: Optional[str] = None   # agent name
    fastest_step: Optional[str] = None   # agent name

    # Anomaly summary
    anomalous_steps: list[str] = field(default_factory=list)
    has_anomalies: bool = False

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# MetricsAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class MetricsAnalyzer:
    """
    Reads step data from SQLite and computes structured metrics per run.

    Usage:
        db = DatabaseManager()
        analyzer = MetricsAnalyzer(db)
        report = analyzer.analyze_run("run_abc123")
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self._latency_threshold = float(get("metrics", "latency_threshold_ms", 5000))
        self._min_runs_baseline = int(get("metrics", "min_runs_for_baseline", 5))
        self._token_stddev_mult = float(get("metrics", "token_stddev_multiplier", 2.5))
        self._latency_stddev_mult = float(get("metrics", "latency_stddev_multiplier", 2.5))

    def analyze_run(self, run_id: str) -> Optional[RunMetrics]:
        """
        Compute metrics for one run.

        Returns:
            RunMetrics with per-step breakdown and run-level aggregates.
            None if run not found.
        """
        run_row = self.db.get_run(run_id)
        if not run_row:
            return None

        step_rows = self.db.get_steps_for_run(run_id)
        if not step_rows:
            return RunMetrics(
                schema_version=SCHEMA_VERSION,
                run_id=run_id,
                workflow=run_row.get("workflow", ""),
            )

        # Compute statistical baseline from historical runs
        baseline = self._get_baseline()

        step_metrics = []
        for row in step_rows:
            sm = self._analyze_step(row, baseline)
            step_metrics.append(sm)

        return self._build_run_metrics(run_row, step_metrics)

    def analyze_latest_run(self) -> Optional[RunMetrics]:
        """Convenience: analyze the most recent run in the database."""
        runs = self.db.list_runs(limit=1)
        if not runs:
            return None
        return self.analyze_run(runs[0]["run_id"])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _analyze_step(self, row: dict, baseline: dict) -> StepMetrics:
        """Compute StepMetrics for one step row from the DB."""
        latency    = float(row.get("latency_ms", 0.0))
        tokens_p   = int(row.get("tokens_prompt", 0))
        tokens_c   = int(row.get("tokens_completion", 0))
        tokens_t   = int(row.get("tokens_total", 0))
        agent      = row.get("agent", "")
        step_num   = int(row.get("step", 0))

        anomaly_reasons = []

        # ── Latency threshold check ───────────────────────────────────────
        latency_flag = latency > self._latency_threshold
        if latency_flag:
            anomaly_reasons.append(
                f"latency {latency:.0f}ms > threshold {self._latency_threshold:.0f}ms"
            )

        # ── Statistical anomaly check (only when baseline available) ─────
        if baseline.get("active"):
            # Latency stddev check
            lat_mean = baseline.get("latency_mean", 0.0)
            lat_std  = baseline.get("latency_std", 0.0)
            if lat_std > 0 and latency > lat_mean + self._latency_stddev_mult * lat_std:
                anomaly_reasons.append(
                    f"latency {latency:.0f}ms > statistical threshold "
                    f"({lat_mean:.0f} + {self._latency_stddev_mult}σ)"
                )

            # Token stddev check
            tok_mean = baseline.get("token_mean", 0.0)
            tok_std  = baseline.get("token_std", 0.0)
            if tok_std > 0 and tokens_t > tok_mean + self._token_stddev_mult * tok_std:
                anomaly_reasons.append(
                    f"tokens {tokens_t} > statistical threshold "
                    f"({tok_mean:.0f} + {self._token_stddev_mult}σ)"
                )

        return StepMetrics(
            schema_version       = SCHEMA_VERSION,
            run_id               = row.get("run_id", ""),
            step                 = step_num,
            agent                = agent,
            latency_ms           = latency,
            latency_flag         = latency_flag,
            latency_threshold_ms = self._latency_threshold,
            tokens_prompt        = tokens_p,
            tokens_completion    = tokens_c,
            tokens_total         = tokens_t,
            execution_time_ms    = latency,   # wall-clock = latency for current pipeline
            is_anomalous         = len(anomaly_reasons) > 0,
            anomaly_reasons      = anomaly_reasons,
        )

    def _build_run_metrics(self, run_row: dict, steps: list[StepMetrics]) -> RunMetrics:
        """Aggregate step metrics into a RunMetrics object."""
        total_latency  = sum(s.latency_ms for s in steps)
        total_tokens   = sum(s.tokens_total for s in steps)
        total_exec     = sum(s.execution_time_ms for s in steps)
        agent_latency  = {s.agent: s.latency_ms for s in steps}

        slowest = max(steps, key=lambda s: s.latency_ms).agent if steps else None
        fastest = min(steps, key=lambda s: s.latency_ms).agent if steps else None

        anomalous = [s.agent for s in steps if s.is_anomalous]

        return RunMetrics(
            schema_version         = SCHEMA_VERSION,
            run_id                 = run_row.get("run_id", ""),
            workflow               = run_row.get("workflow", ""),
            steps                  = steps,
            total_latency_ms       = total_latency,
            total_tokens           = total_tokens,
            total_execution_time_ms= total_exec,
            step_count             = len(steps),
            agent_latency          = agent_latency,
            slowest_step           = slowest,
            fastest_step           = fastest,
            anomalous_steps        = anomalous,
            has_anomalies          = len(anomalous) > 0,
        )

    def _get_baseline(self) -> dict:
        """
        Compute statistical baseline from all stored runs.
        Returns empty baseline if fewer than min_runs_for_baseline runs exist.
        """
        runs = self.db.list_runs(limit=1000)
        if len(runs) < self._min_runs_baseline:
            return {"active": False}

        latencies = []
        tokens    = []
        for run in runs:
            steps = self.db.get_steps_for_run(run["run_id"])
            for s in steps:
                latencies.append(float(s.get("latency_ms", 0.0)))
                tokens.append(int(s.get("tokens_total", 0)))

        if not latencies:
            return {"active": False}

        return {
            "active":       True,
            "latency_mean": _mean(latencies),
            "latency_std":  _stddev(latencies),
            "token_mean":   _mean(tokens),
            "token_std":    _stddev(tokens),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
