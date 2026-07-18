"""
storage/db.py — SQLite Database Manager
=========================================
Day 8: Creates and manages the AgentLens SQLite database.

Four tables:
    runs     — one row per pipeline run (+ full JSON blob)
    steps    — one row per agent step within a run
    analysis — one row per analyzer result (populated by future analyzers)
    metrics  — one row per named metric per step (latency, tokens, etc.)

Design decisions:
    - JSON blobs for full trace payloads (human-readable, portable)
    - Foreign keys enforced for data integrity
    - Upsert (INSERT OR REPLACE) so re-running a pipeline is idempotent
    - Single file DB at data/agentlens.db (configurable via DB_PATH)
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Default DB path
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "data/agentlens.db"


# ─────────────────────────────────────────────────────────────────────────────
# DDL — Table definitions
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    workflow         TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    status           TEXT NOT NULL,
    total_latency_ms REAL DEFAULT 0.0,
    total_tokens     INTEGER DEFAULT 0,
    schema_version   TEXT NOT NULL,
    trace_path       TEXT,
    trace_json       TEXT
);
"""

_CREATE_STEPS = """
CREATE TABLE IF NOT EXISTS steps (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT NOT NULL,
    step               INTEGER NOT NULL,
    agent              TEXT NOT NULL,
    status             TEXT NOT NULL,
    latency_ms         REAL DEFAULT 0.0,
    tokens_prompt      INTEGER DEFAULT 0,
    tokens_completion  INTEGER DEFAULT 0,
    tokens_total       INTEGER DEFAULT 0,
    diff_summary       TEXT DEFAULT '',
    error              TEXT,
    timestamp          TEXT NOT NULL,
    schema_version     TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, step)
);
"""

_CREATE_ANALYSIS = """
CREATE TABLE IF NOT EXISTS analysis (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    step           INTEGER,
    analyzer       TEXT NOT NULL,
    category       TEXT,
    verdict        TEXT,
    confidence     REAL DEFAULT 0.0,
    details_json   TEXT,
    timestamp      TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

_CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    step           INTEGER,
    agent          TEXT,
    metric_name    TEXT NOT NULL,
    metric_value   REAL,
    metric_unit    TEXT DEFAULT '',
    timestamp      TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

_ALL_DDL = [_CREATE_RUNS, _CREATE_STEPS, _CREATE_ANALYSIS, _CREATE_METRICS]


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseManager
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Manages the SQLite connection and all CRUD operations.

    Usage:
        db = DatabaseManager()           # uses default path
        db = DatabaseManager("custom.db")

        with db.connection() as conn:    # context manager for safe connections
            ...

        db.initialize()                  # create tables if not exist
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a connection and commits/rolls back."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create all four tables if they do not already exist."""
        with self.connection() as conn:
            for ddl in _ALL_DDL:
                conn.execute(ddl)

    # ── Runs ──────────────────────────────────────────────────────────────────

    def insert_run(
        self,
        run_id: str,
        workflow: str,
        timestamp: str,
        status: str,
        total_latency_ms: float,
        total_tokens: int,
        schema_version: str,
        trace_path: Optional[str] = None,
        trace_json: Optional[str] = None,
    ) -> None:
        """Insert or replace a run record."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, workflow, timestamp, status,
                     total_latency_ms, total_tokens, schema_version,
                     trace_path, trace_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, workflow, timestamp, status,
                 total_latency_ms, total_tokens, schema_version,
                 trace_path, trace_json),
            )

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Fetch a run row by run_id. Returns None if not found."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(
        self,
        limit: int = 50,
        status_filter: Optional[str] = None,
        workflow_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List run summaries, newest first."""
        query = "SELECT run_id, workflow, timestamp, status, total_latency_ms, total_tokens FROM runs"
        params: list[Any] = []
        filters = []
        if status_filter:
            filters.append("status = ?")
            params.append(status_filter)
        if workflow_filter:
            filters.append("workflow = ?")
            params.append(workflow_filter)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Steps ─────────────────────────────────────────────────────────────────

    def insert_step(
        self,
        run_id: str,
        step: int,
        agent: str,
        status: str,
        latency_ms: float,
        tokens_prompt: int,
        tokens_completion: int,
        tokens_total: int,
        diff_summary: str,
        timestamp: str,
        schema_version: str,
        error: Optional[str] = None,
    ) -> None:
        """Insert or replace a step record."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO steps
                    (run_id, step, agent, status, latency_ms,
                     tokens_prompt, tokens_completion, tokens_total,
                     diff_summary, error, timestamp, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, step, agent, status, latency_ms,
                 tokens_prompt, tokens_completion, tokens_total,
                 diff_summary, error, timestamp, schema_version),
            )

    def get_steps_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all steps for a run, ordered by step number."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY step ASC",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Analysis ──────────────────────────────────────────────────────────────

    def insert_analysis(
        self,
        run_id: str,
        analyzer: str,
        timestamp: str,
        schema_version: str,
        step: Optional[int] = None,
        category: Optional[str] = None,
        verdict: Optional[str] = None,
        confidence: float = 0.0,
        details: Optional[dict] = None,
    ) -> None:
        """Insert an analysis result row."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis
                    (run_id, step, analyzer, category, verdict,
                     confidence, details_json, timestamp, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, step, analyzer, category, verdict,
                 confidence, json.dumps(details or {}), timestamp, schema_version),
            )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def insert_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
        timestamp: str,
        schema_version: str,
        step: Optional[int] = None,
        agent: Optional[str] = None,
        metric_unit: str = "",
    ) -> None:
        """Insert a named metric row."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics
                    (run_id, step, agent, metric_name, metric_value,
                     metric_unit, timestamp, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, step, agent, metric_name, metric_value,
                 metric_unit, timestamp, schema_version),
            )

    def get_metrics_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all metrics for a run."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY step ASC, metric_name ASC",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Utility ───────────────────────────────────────────────────────────────

    def table_counts(self) -> dict[str, int]:
        """Return row count for each table — useful for checkpoint verification."""
        counts = {}
        with self.connection() as conn:
            for table in ("runs", "steps", "analysis", "metrics"):
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0]
        return counts
