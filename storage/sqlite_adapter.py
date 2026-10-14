"""
storage/sqlite_adapter.py — SQLite Storage Adapter Implementation
==================================================================
AgentLens v2.0 (Day 50): Concrete implementation of BaseStorageAdapter
for SQLite with WAL mode, connection pooling, and atomic transaction support.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from schema.models import SCHEMA_VERSION
from storage.base_adapter import BaseStorageAdapter
from storage.db import DatabaseManager
from storage.writer import StorageWriter

if TYPE_CHECKING:
    from normalizer.normalizer import NormalizedRun


class SQLiteStorageAdapter(BaseStorageAdapter):
    """
    SQLite-backed storage adapter implementing BaseStorageAdapter.

    Combines DatabaseManager (connection pooling, WAL mode, CRUD)
    and StorageWriter (atomic multi-step run writes) behind the
    polymorphic BaseStorageAdapter interface.

    Usage:
        adapter = SQLiteStorageAdapter("data/agentlens.db")
        adapter.initialize()
        adapter.save_run(normalized_run)
        run = adapter.get_run("run_123")
    """

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        pool_size: int | None = None,
        db: DatabaseManager | None = None,
    ) -> None:
        if db is not None:
            self.db = db
        else:
            self.db = DatabaseManager(db_path=db_path, pool_size=pool_size)

        self.writer = StorageWriter(self.db)

    @property
    def db_path(self) -> str:
        """Return the underlying SQLite database file path."""
        return self.db.db_path

    def initialize(self) -> None:
        """Initialize SQLite database tables and indexes."""
        self.db.initialize()

    def close(self) -> None:
        """Close all connections in the SQLite connection pool."""
        self.db.close()

    def save_run(
        self,
        run: NormalizedRun,
        trace_json: str | None = None,
        trace_path: str | None = None,
    ) -> None:
        """
        Persist a complete NormalizedRun, its steps, and its metrics atomically
        within a single transaction.
        """
        self.writer.write_run(run=run, trace_json=trace_json, trace_path=trace_path)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by run_id."""
        return self.db.get_run(run_id)

    def list_runs(
        self,
        limit: int = 50,
        status_filter: str | None = None,
        workflow_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent runs, newest first."""
        return self.db.list_runs(
            limit=limit,
            status_filter=status_filter,
            workflow_filter=workflow_filter,
        )

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
        self.db.insert_analysis(
            run_id=run_id,
            analyzer=analyzer,
            timestamp=timestamp,
            schema_version=schema_version,
            step=step,
            category=category,
            verdict=verdict,
            confidence=confidence,
            details=details,
        )

    def query_metrics(
        self,
        run_id: str,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query metrics for a specific run, with optional filtering by metric_name.
        """
        all_metrics = self.db.get_metrics_for_run(run_id)
        if metric_name is None:
            return all_metrics
        return [m for m in all_metrics if m.get("metric_name") == metric_name]

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all steps for a run ordered by step index."""
        return self.db.get_steps_for_run(run_id)

    def save_rule_matches(self, run_id: str, matches: list[Any]) -> int:
        """Persist rule matches for a run. Returns count inserted."""
        return self.db.insert_rule_matches(run_id, matches)

    def get_rule_matches(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all stored rule matches for a run."""
        return self.db.get_rule_matches_for_run(run_id)

    def delete_runs_older_than(self, cutoff_iso: str, dry_run: bool = False) -> list[str]:
        """Delete runs older than cutoff_iso (retention policy). Returns affected run_ids."""
        return self.db.delete_runs_older_than(cutoff_iso, dry_run=dry_run)

    def table_counts(self) -> dict[str, int]:
        """Return row counts for all tables."""
        return self.db.table_counts()

    def get_pragmas(self) -> dict[str, Any]:
        """Return active SQLite PRAGMA configuration values."""
        return self.db.get_pragmas()
