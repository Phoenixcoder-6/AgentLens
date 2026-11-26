"""
storage/postgres_adapter.py — PostgreSQL Storage Adapter Implementation
========================================================================
AgentLens v2.1 (Day 68): Production-grade, fail-loud PostgreSQL storage adapter
driven by AGENTLENS_DB_URL=postgresql://...

Features:
    - URL/DSN parsing supporting credentials, host, port, dbname, and query parameters.
    - Full PostgreSQL DDL with JSONB, TIMESTAMPTZ, BIGSERIAL, and index creation.
    - Fail-loud architecture: Eliminates silent mock fallback on connection failures.
    - Exponential backoff retry with jitter on transient network/cluster disconnects.
    - Custom diagnostic exceptions: PostgresDriverMissingError and PostgresConnectionError.
    - Live query execution for PostgreSQL when driver and cluster are present.
    - Deterministic in-memory mock engine when explicitly requested via mock_mode=True.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

from schema.models import SCHEMA_VERSION
from storage.base_adapter import BaseStorageAdapter

if TYPE_CHECKING:
    from normalizer.normalizer import NormalizedRun

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL Storage Exceptions (Day 68 Fail-Loud Architecture)
# ─────────────────────────────────────────────────────────────────────────────


class PostgresStorageError(Exception):
    """Base exception for PostgreSQL storage adapter failures."""


class PostgresDriverMissingError(PostgresStorageError):
    """Raised when neither psycopg2 nor psycopg is installed and mock_mode is False."""


class PostgresConnectionError(PostgresStorageError):
    """Raised when live PostgreSQL connection fails after exhausting retries."""

    def __init__(
        self,
        message: str,
        host: str,
        port: int,
        database: str,
        user: str | None = None,
        attempts: int = 1,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.attempts = attempts
        self.original_exception = original_exception


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

    Fail-Loud Architecture (Day 68):
        - When mock_mode=False (default), missing drivers or unreachable databases
          raise explicit, structured exceptions rather than silently falling back
          to in-memory mock storage.
        - Network transient failures retry with exponential backoff and randomized jitter.
    """

    def __init__(
        self,
        connection_url: str,
        mock_mode: bool = False,
        max_retries: int = 3,
        retry_delay: float = 0.2,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        **kwargs: Any,
    ) -> None:
        self.connection_url = connection_url
        self.mock_mode = mock_mode
        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.01, retry_delay)
        self.backoff_factor = max(1.0, backoff_factor)
        self.jitter = jitter

        self._parsed_url = self._parse_connection_url(connection_url)
        self._mock_store = _MockPostgresStore()
        self._live_conn: Any = None
        self._driver: str | None = None

        if not self.mock_mode:
            self._driver = self._detect_driver()
            # In Day 68 fail-loud architecture, we NEVER silently flip self.mock_mode = True.

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

    def _raw_connect(self) -> Any:
        """Establish a single live PostgreSQL connection using the detected driver."""
        if self._driver == "psycopg2":
            import psycopg2

            return psycopg2.connect(
                dbname=self._parsed_url["database"],
                user=self._parsed_url["user"],
                password=self._parsed_url["password"],
                host=self._parsed_url["host"],
                port=self._parsed_url["port"],
                sslmode=self._parsed_url.get("sslmode", "prefer"),
            )
        elif self._driver == "psycopg":
            import psycopg

            return psycopg.connect(
                dbname=self._parsed_url["database"],
                user=self._parsed_url["user"],
                password=self._parsed_url["password"],
                host=self._parsed_url["host"],
                port=self._parsed_url["port"],
                sslmode=self._parsed_url.get("sslmode", "prefer"),
            )
        raise PostgresDriverMissingError(
            "No PostgreSQL driver installed (neither psycopg2 nor psycopg found). "
            f"Cannot connect to PostgreSQL at {self._parsed_url['host']}:{self._parsed_url['port']}. "
            "Please install psycopg2-binary or psycopg, or explicitly pass mock_mode=True for offline testing."
        )

    def _connect_with_retry(self) -> Any:
        """
        Connect to PostgreSQL with structured retry backoff and randomized jitter.
        Fails loud if driver is missing or all attempts are exhausted.
        """
        if self._driver is None:
            raise PostgresDriverMissingError(
                "No PostgreSQL driver installed (neither psycopg2 nor psycopg found). "
                f"Cannot connect to PostgreSQL at {self._parsed_url['host']}:{self._parsed_url['port']}. "
                "Please install psycopg2-binary or psycopg, or explicitly pass mock_mode=True for offline testing."
            )

        host = self._parsed_url["host"]
        port = self._parsed_url["port"]
        database = self._parsed_url["database"]
        user = self._parsed_url.get("user")

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                conn = self._raw_connect()
                logger.info(
                    "[PostgreSQLStorageAdapter] Connected to %s:%d/%s (attempt %d/%d).",
                    host,
                    port,
                    database,
                    attempt,
                    self.max_retries,
                )
                return conn
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[PostgreSQLStorageAdapter] Connection attempt %d/%d to %s:%d/%s failed: %s",
                    attempt,
                    self.max_retries,
                    host,
                    port,
                    database,
                    exc,
                )
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                    if self.jitter:
                        delay += random.uniform(0.01, max(0.02, 0.1 * delay))
                    time.sleep(delay)

        # Fail-loud: Never fall back to mock store in production!
        err_msg = (
            f"Failed to connect to PostgreSQL at {host}:{port}/{database} after "
            f"{self.max_retries} attempts: {last_exc}"
        )
        logger.error("[PostgreSQLStorageAdapter] %s", err_msg)
        raise PostgresConnectionError(
            err_msg,
            host=host,
            port=port,
            database=database,
            user=user,
            attempts=self.max_retries,
            original_exception=last_exc,
        ) from last_exc

    def get_connection_info(self, mask_password: bool = True) -> dict[str, Any]:
        """Return parsed connection parameters."""
        info = dict(self._parsed_url)
        if mask_password and info.get("password"):
            info["password"] = "******"
        info["driver"] = self._driver if not self.mock_mode else "mock"
        info["mock_mode"] = self.mock_mode
        return info

    def get_ddl(self) -> list[str]:
        """Return all PostgreSQL DDL statements."""
        return list(PG_ALL_DDL)

    def initialize(self) -> None:
        """Create tables and indexes in PostgreSQL database."""
        if self.mock_mode:
            self._mock_store.initialized = True
            return

        conn = self._connect_with_retry()
        try:
            with conn.cursor() as cur:
                for ddl in PG_ALL_DDL:
                    cur.execute(ddl)
            conn.commit()
            self._live_conn = conn
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            raise PostgresConnectionError(
                f"Failed to initialize PostgreSQL schema at {self._parsed_url['host']}: {exc}",
                host=self._parsed_url["host"],
                port=self._parsed_url["port"],
                database=self._parsed_url["database"],
                user=self._parsed_url["user"],
                original_exception=exc,
            ) from exc

    def is_healthy(self) -> bool:
        """Check if storage connection is healthy and responsive."""
        if self.mock_mode:
            return self._mock_store.initialized
        try:
            return self.ping()
        except Exception:
            return False

    def ping(self) -> bool:
        """Verify database connectivity by executing 'SELECT 1;'."""
        if self.mock_mode:
            return True
        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()
        with self._live_conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return True

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

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        with self._live_conn.cursor() as cur:
            # 1. Upsert runs
            upsert_run_sql = """
            INSERT INTO runs (run_id, workflow, timestamp, status, total_latency_ms, total_tokens, schema_version, trace_path, trace_json, expected_output)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                workflow = EXCLUDED.workflow,
                timestamp = EXCLUDED.timestamp,
                status = EXCLUDED.status,
                total_latency_ms = EXCLUDED.total_latency_ms,
                total_tokens = EXCLUDED.total_tokens,
                schema_version = EXCLUDED.schema_version,
                trace_path = EXCLUDED.trace_path,
                trace_json = EXCLUDED.trace_json,
                expected_output = EXCLUDED.expected_output;
            """
            cur.execute(
                upsert_run_sql,
                (
                    run.run_id,
                    run.workflow,
                    ts,
                    status_str,
                    run.total_latency_ms,
                    run.total_tokens,
                    SCHEMA_VERSION,
                    trace_path,
                    trace_json,
                    run.expected_output,
                ),
            )

            # 2. Upsert steps
            upsert_step_sql = """
            INSERT INTO steps (run_id, step, agent, status, latency_ms, tokens_prompt, tokens_completion, tokens_total, diff_summary, error, timestamp, schema_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, step) DO UPDATE SET
                agent = EXCLUDED.agent,
                status = EXCLUDED.status,
                latency_ms = EXCLUDED.latency_ms,
                tokens_prompt = EXCLUDED.tokens_prompt,
                tokens_completion = EXCLUDED.tokens_completion,
                tokens_total = EXCLUDED.tokens_total,
                diff_summary = EXCLUDED.diff_summary,
                error = EXCLUDED.error,
                timestamp = EXCLUDED.timestamp,
                schema_version = EXCLUDED.schema_version;
            """
            for step in run.steps:
                step_status = (
                    step.status.value if hasattr(step.status, "value") else str(step.status)
                )
                step_ts = _iso(step.timestamp)
                cur.execute(
                    upsert_step_sql,
                    (
                        run.run_id,
                        step.step,
                        step.agent,
                        step_status,
                        step.latency_ms,
                        step.tokens.prompt,
                        step.tokens.completion,
                        step.tokens.total,
                        step.diff_summary or "",
                        step.error,
                        step_ts,
                        SCHEMA_VERSION,
                    ),
                )

                # Metrics
                insert_metric_sql = """
                INSERT INTO metrics (run_id, step, agent, metric_name, metric_value, metric_unit, timestamp, schema_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """
                cur.execute(
                    insert_metric_sql,
                    (
                        run.run_id,
                        step.step,
                        step.agent,
                        "latency_ms",
                        step.latency_ms,
                        "ms",
                        step_ts,
                        SCHEMA_VERSION,
                    ),
                )
                cur.execute(
                    insert_metric_sql,
                    (
                        run.run_id,
                        step.step,
                        step.agent,
                        "tokens_total",
                        float(step.tokens.total),
                        "tokens",
                        step_ts,
                        SCHEMA_VERSION,
                    ),
                )

        self._live_conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by run_id."""
        if self.mock_mode:
            return self._mock_store.runs.get(run_id)

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        with self._live_conn.cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE run_id = %s;", (run_id,))
            row = cur.fetchone()
            if row is None:
                return None
            colnames = [desc[0] for desc in cur.description]
            return dict(zip(colnames, row, strict=False))

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

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        query = (
            "SELECT run_id, workflow, timestamp, status, total_latency_ms, total_tokens FROM runs"
        )
        params: list[Any] = []
        filters: list[str] = []
        if status_filter:
            filters.append("status = %s")
            params.append(status_filter)
        if workflow_filter:
            filters.append("workflow = %s")
            params.append(workflow_filter)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY timestamp DESC LIMIT %s;"
        params.append(limit)

        with self._live_conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, r, strict=False)) for r in rows]

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

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        sql = """
        INSERT INTO analysis (run_id, step, analyzer, category, verdict, confidence, details_json, timestamp, schema_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        with self._live_conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    run_id,
                    step,
                    analyzer,
                    category,
                    verdict,
                    confidence,
                    json.dumps(details or {}),
                    timestamp,
                    schema_version,
                ),
            )
        self._live_conn.commit()

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

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        query = "SELECT * FROM metrics WHERE run_id = %s"
        params: list[Any] = [run_id]
        if metric_name:
            query += " AND metric_name = %s"
            params.append(metric_name)
        query += " ORDER BY step ASC, metric_name ASC;"

        with self._live_conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, r, strict=False)) for r in rows]

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all steps for a run ordered by step index."""
        if self.mock_mode:
            res = [s for s in self._mock_store.steps if s["run_id"] == run_id]
            res.sort(key=lambda s: s["step"])
            return res

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        with self._live_conn.cursor() as cur:
            cur.execute("SELECT * FROM steps WHERE run_id = %s ORDER BY step ASC;", (run_id,))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, r, strict=False)) for r in rows]

    def save_rule_matches(self, run_id: str, matches: list[Any]) -> int:
        """Persist rule matches for a run."""
        now = datetime.now(UTC).isoformat()
        if self.mock_mode:
            self._mock_store.rule_matches = [
                m for m in self._mock_store.rule_matches if m.get("run_id") != run_id
            ]
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

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        with self._live_conn.cursor() as cur:
            cur.execute("DELETE FROM rule_matches WHERE run_id = %s;", (run_id,))
            insert_sql = """
            INSERT INTO rule_matches (run_id, rule_id, rule_version, category, severity, agent, step, description, matched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            for m in matches:
                v = m if isinstance(m, dict) else vars(m)
                cur.execute(
                    insert_sql,
                    (
                        run_id,
                        str(v.get("rule_id", "unknown")),
                        str(v.get("rule_version", "1.0.0")),
                        v.get("category"),
                        v.get("severity"),
                        v.get("agent"),
                        v.get("step"),
                        v.get("description"),
                        now,
                    ),
                )
        self._live_conn.commit()
        return len(matches)

    def get_rule_matches(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all stored rule matches for a run."""
        if self.mock_mode:
            return [m for m in self._mock_store.rule_matches if m["run_id"] == run_id]

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        with self._live_conn.cursor() as cur:
            cur.execute("SELECT * FROM rule_matches WHERE run_id = %s;", (run_id,))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, r, strict=False)) for r in rows]

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

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        with self._live_conn.cursor() as cur:
            cur.execute("SELECT run_id FROM runs WHERE timestamp < %s;", (cutoff_iso,))
            affected = [r[0] for r in cur.fetchall()]
            if not dry_run and affected:
                cur.execute("DELETE FROM runs WHERE timestamp < %s;", (cutoff_iso,))
                self._live_conn.commit()
            return affected

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

        if self._live_conn is None:
            self._live_conn = self._connect_with_retry()

        counts = {}
        with self._live_conn.cursor() as cur:
            for tbl in ("runs", "steps", "analysis", "metrics", "rule_matches"):
                cur.execute(f"SELECT COUNT(*) FROM {tbl};")  # noqa: S608
                row = cur.fetchone()
                counts[tbl] = row[0] if row else 0
        return counts


def _iso(value: Any) -> str:
    """Convert any timestamp to ISO string, ensuring UTC-aware."""
    if hasattr(value, "isoformat"):
        if hasattr(value, "tzinfo") and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(value)
