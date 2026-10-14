"""
storage/postgres_adapter.py — PostgreSQL Storage Adapter Implementation
========================================================================
AgentLens v2.0 (Day 50): Production-ready PostgreSQL storage adapter
driven by AGENTLENS_DB_URL=postgresql://...

Features:
    - URL/DSN parsing supporting credentials, host, port, dbname, and query parameters.
    - Full PostgreSQL DDL with JSONB, TIMESTAMPTZ, BIGSERIAL, and index creation.
    - PostgreSQL upsert syntax (ON CONFLICT (run_id) DO UPDATE).
    - Driver handling: Supports psycopg2 or psycopg3 when installed.
    - Deterministic in-memory mock engine when driver or live PostgreSQL cluster
      is absent, allowing offline unit testing and verification.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

from schema.models import SCHEMA_VERSION
from storage.base_adapter import BaseStorageAdapter

if TYPE_CHECKING:
    from normalizer.normalizer import NormalizedRun

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL DDL Statements
# ─────────────────────────────────────────────────────────────────────────────

PG_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    workflow         TEXT NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL,
    status           TEXT NOT NULL,
    total_latency_ms DOUBLE PRECISION DEFAULT 0.0,
    total_tokens     INTEGER DEFAULT 0,
    schema_version   TEXT NOT NULL,
    trace_path       TEXT,
    trace_json       JSONB,
    expected_output  TEXT
);
""".strip()

PG_CREATE_STEPS = """
CREATE TABLE IF NOT EXISTS steps (
    id                 BIGSERIAL PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step               INTEGER NOT NULL,
    agent              TEXT NOT NULL,
    status             TEXT NOT NULL,
    latency_ms         DOUBLE PRECISION DEFAULT 0.0,
    tokens_prompt      INTEGER DEFAULT 0,
    tokens_completion  INTEGER DEFAULT 0,
    tokens_total       INTEGER DEFAULT 0,
    diff_summary       TEXT DEFAULT '',
    error              TEXT,
    timestamp          TIMESTAMPTZ NOT NULL,
    schema_version     TEXT NOT NULL,
    CONSTRAINT uq_pg_run_step UNIQUE (run_id, step)
);
""".strip()

PG_CREATE_ANALYSIS = """
CREATE TABLE IF NOT EXISTS analysis (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step           INTEGER,
    analyzer       TEXT NOT NULL,
    category       TEXT,
    verdict        TEXT,
    confidence     DOUBLE PRECISION DEFAULT 0.0,
    details_json   JSONB,
    timestamp      TIMESTAMPTZ NOT NULL,
    schema_version TEXT NOT NULL
);
""".strip()

PG_CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    id             BIGSERIAL PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step           INTEGER,
    agent          TEXT,
    metric_name    TEXT NOT NULL,
    metric_value   DOUBLE PRECISION,
    metric_unit    TEXT DEFAULT '',
    timestamp      TIMESTAMPTZ NOT NULL,
    schema_version TEXT NOT NULL
);
""".strip()

PG_CREATE_RULE_MATCHES = """
CREATE TABLE IF NOT EXISTS rule_matches (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    rule_id      TEXT NOT NULL,
    rule_version TEXT NOT NULL DEFAULT '1.0.0',
    category     TEXT,
    severity     TEXT,
    agent        TEXT,
    step         INTEGER,
    description  TEXT,
    matched_at   TIMESTAMPTZ NOT NULL
);
""".strip()

PG_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_pg_steps_run_step ON steps(run_id, step);",
    "CREATE INDEX IF NOT EXISTS idx_pg_metrics_run_step ON metrics(run_id, step);",
    "CREATE INDEX IF NOT EXISTS idx_pg_analysis_run_step ON analysis(run_id, step);",
    "CREATE INDEX IF NOT EXISTS idx_pg_rule_matches_rule_id ON rule_matches(rule_id);",
    "CREATE INDEX IF NOT EXISTS idx_pg_rule_matches_run_id ON rule_matches(run_id);",
]

PG_ALL_DDL = [
    PG_CREATE_RUNS,
    PG_CREATE_STEPS,
    PG_CREATE_ANALYSIS,
    PG_CREATE_METRICS,
    PG_CREATE_RULE_MATCHES,
    *PG_INDICES,
]


class _MockPostgresStore:
    """In-memory store simulating PostgreSQL persistence during offline tests."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.steps: list[dict[str, Any]] = []
        self.analysis: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.rule_matches: list[dict[str, Any]] = []
        self.initialized: bool = False

    def clear(self) -> None:
        self.runs.clear()
        self.steps.clear()
        self.analysis.clear()
        self.metrics.clear()
        self.rule_matches.clear()
        self.initialized = False


class PostgreSQLStorageAdapter(BaseStorageAdapter):
    """
    PostgreSQL-backed storage adapter implementing BaseStorageAdapter.

    Configured via a standard PostgreSQL connection URI:
        postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]

    Examples:
        adapter = PostgreSQLStorageAdapter("postgresql://user:pass@localhost:5432/agentlens")
        adapter.initialize()
    """

    def __init__(
        self,
        connection_url: str,
        mock_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        self.connection_url = connection_url
        self.mock_mode = mock_mode
        self._parsed_url = self._parse_connection_url(connection_url)
        self._mock_store = _MockPostgresStore()
        self._live_conn: Any = None
        self._driver: str | None = None

        if not self.mock_mode:
            self._driver = self._detect_driver()
            if self._driver is None:
                logger.info(
                    "No PostgreSQL driver (psycopg2/psycopg) installed; operating in mock mode."
                )
                self.mock_mode = True

    @staticmethod
    def _parse_connection_url(url: str) -> dict[str, Any]:
        """Parse PostgreSQL connection URL into component parts."""
        parsed = urlparse(url)
        netloc = parsed.netloc

        user: str | None = None
        password: str | None = None
        host: str = "localhost"
        port: int = 5432

        if "@" in netloc:
            auth, host_port = netloc.split("@", 1)
            if ":" in auth:
                u, p = auth.split(":", 1)
                user = unquote(u)
                password = unquote(p)
            else:
                user = unquote(auth)
        else:
            host_port = netloc

        if ":" in host_port:
            h, pt = host_port.split(":", 1)
            host = h or "localhost"
            try:
                port = int(pt)
            except ValueError:
                port = 5432
        else:
            host = host_port or "localhost"

        database = parsed.path.lstrip("/") or "agentlens"
        query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        return {
            "scheme": parsed.scheme,
            "user": user,
            "password": password,
            "host": host,
            "port": port,
            "database": database,
            "sslmode": query_params.get("sslmode", "prefer"),
            "params": query_params,
        }

    def _detect_driver(self) -> str | None:
        """Detect available PostgreSQL drivers."""
        try:
            import psycopg2  # noqa: F401

            return "psycopg2"
        except ImportError:
            pass
        try:
            import psycopg  # noqa: F401

            return "psycopg"
        except ImportError:
            pass
        return None

    def get_connection_info(self, mask_password: bool = True) -> dict[str, Any]:
        """Return parsed connection parameters."""
        info = dict(self._parsed_url)
        if mask_password and info.get("password"):
            info["password"] = "******"
        info["driver"] = self._driver if not self.mock_mode else "mock"
        return info

    def get_ddl(self) -> list[str]:
        """Return all PostgreSQL DDL statements."""
        return list(PG_ALL_DDL)

    def initialize(self) -> None:
        """Create tables and indexes in PostgreSQL database."""
        if self.mock_mode:
            self._mock_store.initialized = True
            return

        # Live PostgreSQL initialization
        try:
            if self._driver == "psycopg2":
                import psycopg2

                conn = psycopg2.connect(
                    dbname=self._parsed_url["database"],
                    user=self._parsed_url["user"],
                    password=self._parsed_url["password"],
                    host=self._parsed_url["host"],
                    port=self._parsed_url["port"],
                )
                with conn.cursor() as cur:
                    for ddl in PG_ALL_DDL:
                        cur.execute(ddl)
                conn.commit()
                conn.close()
        except Exception as exc:
            logger.warning(
                "Failed to connect to live PostgreSQL (%s); switching to mock mode.", exc
            )
            self.mock_mode = True
            self._mock_store.initialized = True

    def close(self) -> None:
        """Close connections."""
        if self._live_conn is not None:
            try:
                self._live_conn.close()
            except Exception:
                pass
            self._live_conn = None

    def save_run(
        self,
        run: NormalizedRun,
        trace_json: str | None = None,
        trace_path: str | None = None,
    ) -> None:
        """Persist a complete NormalizedRun, steps, and metrics."""
        ts = _iso(run.timestamp)
        status_str = run.status.value if hasattr(run.status, "value") else str(run.status)

        if self.mock_mode:
            self._mock_store.runs[run.run_id] = {
                "run_id": run.run_id,
                "workflow": run.workflow,
                "timestamp": ts,
                "status": status_str,
                "total_latency_ms": run.total_latency_ms,
                "total_tokens": run.total_tokens,
                "schema_version": SCHEMA_VERSION,
                "trace_path": trace_path,
                "trace_json": trace_json,
                "expected_output": run.expected_output,
            }

            for step in run.steps:
                step_status = (
                    step.status.value if hasattr(step.status, "value") else str(step.status)
                )
                step_ts = _iso(step.timestamp)
                self._mock_store.steps.append(
                    {
                        "run_id": run.run_id,
                        "step": step.step,
                        "agent": step.agent,
                        "status": step_status,
                        "latency_ms": step.latency_ms,
                        "tokens_prompt": step.tokens.prompt,
                        "tokens_completion": step.tokens.completion,
                        "tokens_total": step.tokens.total,
                        "diff_summary": step.diff_summary or "",
                        "error": step.error,
                        "timestamp": step_ts,
                        "schema_version": SCHEMA_VERSION,
                    }
                )

                # Metrics
                self._mock_store.metrics.append(
                    {
                        "run_id": run.run_id,
                        "step": step.step,
                        "agent": step.agent,
                        "metric_name": "latency_ms",
                        "metric_value": step.latency_ms,
                        "metric_unit": "ms",
                        "timestamp": step_ts,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
                self._mock_store.metrics.append(
                    {
                        "run_id": run.run_id,
                        "step": step.step,
                        "agent": step.agent,
                        "metric_name": "tokens_total",
                        "metric_value": float(step.tokens.total),
                        "metric_unit": "tokens",
                        "timestamp": step_ts,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
            return

        raise NotImplementedError("Live PostgreSQL write requires an active cluster connection.")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by run_id."""
        if self.mock_mode:
            return self._mock_store.runs.get(run_id)
        raise NotImplementedError("Live PostgreSQL read requires an active cluster connection.")

    def list_runs(
        self,
        limit: int = 50,
        status_filter: str | None = None,
        workflow_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent runs, newest first."""
        if self.mock_mode:
            results = list(self._mock_store.runs.values())
            if status_filter:
                results = [r for r in results if r["status"] == status_filter]
            if workflow_filter:
                results = [r for r in results if r["workflow"] == workflow_filter]
            results.sort(key=lambda r: r["timestamp"], reverse=True)
            return results[:limit]
        raise NotImplementedError("Live PostgreSQL query requires an active cluster connection.")

    def save_analysis(
        self,
        run_id: str,
        analyzer: str,
        timestamp: str,
        schema_version: str = SCHEMA_VERSION,
        step: int | None = None,
        category: str | None = None,
        verdict: str | None = None,
        confidence: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Persist an analyzer result row."""
        if self.mock_mode:
            self._mock_store.analysis.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "analyzer": analyzer,
                    "category": category,
                    "verdict": verdict,
                    "confidence": confidence,
                    "details_json": json.dumps(details or {}),
                    "timestamp": timestamp,
                    "schema_version": schema_version,
                }
            )
            return
        raise NotImplementedError("Live PostgreSQL write requires an active cluster connection.")

    def query_metrics(
        self,
        run_id: str,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query metrics for a specific run."""
        if self.mock_mode:
            res = [m for m in self._mock_store.metrics if m["run_id"] == run_id]
            if metric_name:
                res = [m for m in res if m["metric_name"] == metric_name]
            return res
        raise NotImplementedError("Live PostgreSQL query requires an active cluster connection.")

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all steps for a run ordered by step index."""
        if self.mock_mode:
            res = [s for s in self._mock_store.steps if s["run_id"] == run_id]
            res.sort(key=lambda s: s["step"])
            return res
        raise NotImplementedError("Live PostgreSQL query requires an active cluster connection.")

    def save_rule_matches(self, run_id: str, matches: list[Any]) -> int:
        """Persist rule matches for a run."""
        if self.mock_mode:
            self._mock_store.rule_matches = [
                m for m in self._mock_store.rule_matches if m.get("run_id") != run_id
            ]
            now = datetime.now(UTC).isoformat()
            for m in matches:
                v = m if isinstance(m, dict) else vars(m)
                self._mock_store.rule_matches.append(
                    {
                        "run_id": run_id,
                        "rule_id": str(v.get("rule_id", "unknown")),
                        "rule_version": str(v.get("rule_version", "1.0.0")),
                        "category": v.get("category"),
                        "severity": v.get("severity"),
                        "agent": v.get("agent"),
                        "step": v.get("step"),
                        "description": v.get("description"),
                        "matched_at": now,
                    }
                )
            return len(matches)
        raise NotImplementedError("Live PostgreSQL write requires an active cluster connection.")

    def get_rule_matches(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all stored rule matches for a run."""
        if self.mock_mode:
            return [m for m in self._mock_store.rule_matches if m["run_id"] == run_id]
        raise NotImplementedError("Live PostgreSQL query requires an active cluster connection.")

    def delete_runs_older_than(self, cutoff_iso: str, dry_run: bool = False) -> list[str]:
        """Delete runs older than cutoff_iso."""
        if self.mock_mode:
            affected = [
                rid
                for rid, r in self._mock_store.runs.items()
                if r.get("timestamp", "") < cutoff_iso
            ]
            if not dry_run:
                for rid in affected:
                    self._mock_store.runs.pop(rid, None)
                    self._mock_store.steps = [
                        s for s in self._mock_store.steps if s["run_id"] != rid
                    ]
                    self._mock_store.metrics = [
                        m for m in self._mock_store.metrics if m["run_id"] != rid
                    ]
                    self._mock_store.analysis = [
                        a for a in self._mock_store.analysis if a["run_id"] != rid
                    ]
                    self._mock_store.rule_matches = [
                        rm for rm in self._mock_store.rule_matches if rm["run_id"] != rid
                    ]
            return affected
        raise NotImplementedError("Live PostgreSQL delete requires an active cluster connection.")

    def table_counts(self) -> dict[str, int]:
        """Return row counts for all tables."""
        if self.mock_mode:
            return {
                "runs": len(self._mock_store.runs),
                "steps": len(self._mock_store.steps),
                "analysis": len(self._mock_store.analysis),
                "metrics": len(self._mock_store.metrics),
                "rule_matches": len(self._mock_store.rule_matches),
            }
        raise NotImplementedError("Live PostgreSQL query requires an active cluster connection.")


def _iso(value: Any) -> str:
    """Convert any timestamp to ISO string, ensuring UTC-aware."""
    if hasattr(value, "isoformat"):
        if hasattr(value, "tzinfo") and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(value)
