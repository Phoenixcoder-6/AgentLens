"""
storage/base_adapter.py — Pluggable Storage Adapter Base Class & Factory
========================================================================
AgentLens v2.0 (Day 50): Defines the pluggable storage abstraction layer.

Allows AgentLens to persist and query runs, steps, metrics, analyses,
and rule matches across different storage backends (SQLite, PostgreSQL, etc.)
behind a uniform, polymorphic interface.

Key Classes:
    - BaseStorageAdapter: Abstract base class specifying the storage interface.
    - get_storage_adapter: Factory function instantiating the appropriate
      adapter based on a connection URL, file path, or environment variables.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from schema.models import SCHEMA_VERSION

if TYPE_CHECKING:
    from normalizer.normalizer import NormalizedRun


class BaseStorageAdapter(ABC):
    """
    Abstract base class for all AgentLens storage adapters.

    Subclasses must implement:
        - save_run: Persist a normalized run, its steps, and metrics.
        - get_run: Fetch a run record by run_id.
        - list_runs: List recent runs with optional filters.
        - save_analysis: Record an analyzer verdict/bundle.
        - query_metrics: Query step or agent metrics for a run.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Create tables, indexes, and initialize schema."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connections, flush pools, and release resources."""
        pass

    @abstractmethod
    def save_run(
        self,
        run: NormalizedRun,
        trace_json: str | None = None,
        trace_path: str | None = None,
    ) -> None:
        """
        Persist a complete NormalizedRun, its steps, and its metrics atomically.

        Args:
            run: The normalized run to persist.
            trace_json: Full JSON string of the raw RunTrace.
            trace_path: Path to the JSON file on disk (if available).
        """
        pass

    @abstractmethod
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Fetch a single run by run_id.

        Returns:
            Dictionary of run fields, or None if not found.
        """
        pass

    @abstractmethod
    def list_runs(
        self,
        limit: int = 50,
        status_filter: str | None = None,
        workflow_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List recent runs, newest first.

        Args:
            limit: Maximum number of runs to return.
            status_filter: Filter by run status (e.g. 'SUCCESS', 'FAILED').
            workflow_filter: Filter by workflow name.
        """
        pass

    @abstractmethod
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
        """
        Persist an analyzer result row.

        Args:
            run_id: ID of the run analyzed.
            analyzer: Name of the analyzer (e.g. 'diff_engine', 'rule_engine').
            timestamp: ISO timestamp of analysis.
            schema_version: Schema version string.
            step: Optional step index.
            category: Optional error category.
            verdict: Analyzer verdict (PASS/FAIL/WARN).
            confidence: Confidence score between 0.0 and 1.0.
            details: Arbitrary analyzer detail dictionary.
        """
        pass

    @abstractmethod
    def query_metrics(
        self,
        run_id: str,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query metrics for a specific run.

        Args:
            run_id: Run ID to query.
            metric_name: Optional filter for specific metric name
                         (e.g. 'latency_ms', 'tokens_total').
        """
        pass

    # -- Additional Standard Methods with Default / Delegated Implementations --

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all steps for a run ordered by step index."""
        return []

    def save_rule_matches(self, run_id: str, matches: list[Any]) -> int:
        """Persist rule matches for a run. Returns count inserted."""
        return 0

    def get_rule_matches(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all stored rule matches for a run."""
        return []

    def delete_runs_older_than(self, cutoff_iso: str, dry_run: bool = False) -> list[str]:
        """Delete runs older than cutoff_iso (retention policy). Returns affected run_ids."""
        return []

    def table_counts(self) -> dict[str, int]:
        """Return row counts for each primary table."""
        return {}


def get_storage_adapter(
    url_or_path: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> BaseStorageAdapter:
    """
    Factory function to resolve and instantiate the appropriate storage adapter.

    Resolution precedence:
        1. Explicit `url_or_path` argument
        2. `AGENTLENS_DB_URL` environment variable
        3. `DATABASE_URL` environment variable
        4. `storage.db_url` from config.yaml
        5. `DB_PATH` environment variable
        6. `storage.db_path` from config.yaml
        7. Default SQLite path: `data/agentlens.db`

    Supported URL schemes:
        - `postgresql://...` or `postgres://...` -> PostgreSQLStorageAdapter
        - `sqlite:///...` or file path / `:memory:` -> SQLiteStorageAdapter
    """
    target = url_or_path

    if target is None:
        target = os.getenv("AGENTLENS_DB_URL") or os.getenv("DATABASE_URL")

    if target is None:
        try:
            from config.config_loader import get

            target = get("storage", "db_url")
        except Exception:
            target = None

    if target is None:
        target = os.getenv("DB_PATH")

    if target is None:
        try:
            from config.config_loader import get

            target = get("storage", "db_path")
        except Exception:
            target = None

    if target is None:
        from storage.db import DEFAULT_DB_PATH

        target = DEFAULT_DB_PATH

    target_str = str(target).strip()

    if target_str.startswith("postgresql://") or target_str.startswith("postgres://"):
        from storage.postgres_adapter import PostgreSQLStorageAdapter

        return PostgreSQLStorageAdapter(connection_url=target_str, **kwargs)

    # SQLite resolution
    sqlite_path = target_str
    if sqlite_path.startswith("sqlite:///"):
        sqlite_path = sqlite_path[len("sqlite:///") :]

    from storage.sqlite_adapter import SQLiteStorageAdapter

    return SQLiteStorageAdapter(db_path=sqlite_path, **kwargs)
