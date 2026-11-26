"""
tests/test_storage_durability.py — Day 68 Storage Fail-Loud & Indexing Tests
=============================================================================
Phase E: Data Durability, Storage Reliability & SRE

Tests:
1. PostgreSQL missing driver fails loud with PostgresDriverMissingError (no silent mock fallback).
2. PostgreSQL connection failures raise PostgresConnectionError with masked credentials and diagnostics.
3. PostgreSQL reconnect retry with exponential backoff and randomized jitter on transient disconnects.
4. Explicit mock_mode=True remains safely isolated for offline unit testing.
5. SQLite secondary composite indices are created idempotently on database initialization.
6. EXPLAIN QUERY PLAN verifies index utilization (USING INDEX) eliminating full table scans.
7. High-volume query benchmark verifying sub-millisecond telemetry lookups.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from storage.db import DatabaseManager
from storage.postgres_adapter import (
    PostgresConnectionError,
    PostgresDriverMissingError,
    PostgreSQLStorageAdapter,
)


def test_postgres_missing_driver_fail_loud():
    """Verify missing postgres driver raises PostgresDriverMissingError when mock_mode=False."""
    url = "postgresql://agentlens:secret@prod-db.internal:5432/agentlens_db"
    adapter = PostgreSQLStorageAdapter(url, mock_mode=False)

    # Force driver to None to simulate missing psycopg2/psycopg
    adapter._driver = None

    # Crucial assertion: mock_mode must NOT be silently flipped to True!
    assert adapter.mock_mode is False

    with pytest.raises(PostgresDriverMissingError) as exc_info:
        adapter.initialize()

    err = str(exc_info.value)
    assert "No PostgreSQL driver installed" in err
    assert "prod-db.internal:5432" in err
    # Confirm password is not leaked in error message
    assert "secret" not in err


def test_postgres_fail_loud_on_invalid_connection(monkeypatch):
    """Verify unreachable PostgreSQL cluster raises PostgresConnectionError with diagnostics."""
    url = "postgresql://agentlens:super_secret@127.0.0.1:54399/unreachable_db"
    adapter = PostgreSQLStorageAdapter(
        url,
        mock_mode=False,
        max_retries=3,
        retry_delay=0.01,
        jitter=False,
    )

    # Simulate that driver is present
    adapter._driver = "psycopg2"

    # Mock raw connect to simulate connection refused
    def mock_fail_connect():
        raise ConnectionRefusedError("Connection refused by peer")

    monkeypatch.setattr(adapter, "_raw_connect", mock_fail_connect)

    with pytest.raises(PostgresConnectionError) as exc_info:
        adapter.initialize()

    err = exc_info.value
    assert err.host == "127.0.0.1"
    assert err.port == 54399
    assert err.database == "unreachable_db"
    assert err.user == "agentlens"
    assert err.attempts == 3
    assert "super_secret" not in str(err)
    # mock_mode must remain False
    assert adapter.mock_mode is False


def test_postgres_reconnect_retry_backoff(monkeypatch):
    """Verify exponential backoff retry succeeds if connection recovers within retry window."""
    url = "postgresql://user:pass@127.0.0.1:5432/agentlens"
    adapter = PostgreSQLStorageAdapter(
        url,
        mock_mode=False,
        max_retries=4,
        retry_delay=0.02,
        backoff_factor=1.5,
        jitter=True,
    )
    adapter._driver = "psycopg2"

    attempt_count = 0
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    def mock_transient_connect():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionResetError(f"Transient network blip {attempt_count}")
        return mock_conn

    monkeypatch.setattr(adapter, "_raw_connect", mock_transient_connect)

    start_time = time.perf_counter()
    conn = adapter._connect_with_retry()
    elapsed = time.perf_counter() - start_time

    assert conn is mock_conn
    assert attempt_count == 3
    # Verifies retry delay took place (> 0.02s)
    assert elapsed >= 0.02


def test_explicit_mock_mode_isolated():
    """Verify that explicit mock_mode=True operates cleanly in memory for offline testing."""
    url = "postgresql://user:pass@localhost:5432/agentlens"
    adapter = PostgreSQLStorageAdapter(url, mock_mode=True)

    assert adapter.mock_mode is True
    adapter.initialize()
    assert adapter.is_healthy() is True
    assert adapter.ping() is True

    info = adapter.get_connection_info(mask_password=True)
    assert info["driver"] == "mock"
    assert info["mock_mode"] is True
    assert info["password"] == "******"

    # Table counts
    counts = adapter.table_counts()
    assert counts["runs"] == 0
    adapter.close()


def test_sqlite_secondary_indices_exist():
    """Verify that all composite secondary indices exist in SQLite master table."""
    db = DatabaseManager(":memory:")
    db.initialize()

    with db.connection() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index';").fetchall()
        index_names = {r[0] for r in rows}

    expected_indices = {
        "idx_analysis_run_step",
        "idx_analysis_run_id",
        "idx_metrics_run_step",
        "idx_metrics_run_name",
        "idx_runs_timestamp",
        "idx_steps_run_status",
        "idx_rule_matches_rule_id",
        "idx_rule_matches_run_id",
    }

    for idx in expected_indices:
        assert idx in index_names, f"Expected secondary index '{idx}' was not found in database!"


def test_sqlite_explain_query_plan_uses_index():
    """EXPLAIN QUERY PLAN confirms indexed queries use SEARCH TABLE USING INDEX (zero SCAN TABLE)."""
    db = DatabaseManager(":memory:")
    db.initialize()

    with db.connection() as conn:
        # Seed test rows
        conn.execute(
            """
            INSERT INTO runs (run_id, workflow, timestamp, status, schema_version)
            VALUES ('run_idx_01', 'wf_test', '2026-11-27T00:00:00Z', 'COMPLETED', '1.0');
            """
        )
        conn.execute(
            """
            INSERT INTO analysis (run_id, step, analyzer, category, timestamp, schema_version)
            VALUES ('run_idx_01', 1, 'blast_radius', 'P2', '2026-11-27T00:00:00Z', '1.0');
            """
        )
        conn.execute(
            """
            INSERT INTO metrics (run_id, step, metric_name, metric_value, timestamp, schema_version)
            VALUES ('run_idx_01', 1, 'latency_ms', 120.5, '2026-11-27T00:00:00Z', '1.0');
            """
        )

        # 1. Check analysis query plan
        plan_analysis = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM analysis WHERE run_id = 'run_idx_01' AND step = 1;"
        ).fetchall()
        plan_analysis_str = " ".join([str(col) for row in plan_analysis for col in row])
        assert "idx_analysis_run_step" in plan_analysis_str
        assert "SEARCH" in plan_analysis_str
        assert "SCAN TABLE analysis" not in plan_analysis_str

        # 2. Check metrics query plan
        plan_metrics = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM metrics WHERE run_id = 'run_idx_01' AND step = 1;"
        ).fetchall()
        plan_metrics_str = " ".join([str(col) for row in plan_metrics for col in row])
        assert "idx_metrics_run_step" in plan_metrics_str
        assert "SEARCH" in plan_metrics_str
        assert "SCAN TABLE metrics" not in plan_metrics_str

        # 3. Check runs timestamp query plan
        plan_runs = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM runs ORDER BY timestamp DESC LIMIT 10;"
        ).fetchall()
        plan_runs_str = " ".join([str(col) for row in plan_runs for col in row])
        assert "idx_runs_timestamp" in plan_runs_str or "SEARCH" in plan_runs_str


def test_sqlite_high_volume_index_benchmark():
    """Benchmark: 5,000 analysis & metrics rows queried via secondary indices execute in <1ms."""
    db = DatabaseManager(":memory:")
    db.initialize()

    # Bulk insert synthetic records
    with db.connection() as conn:
        analysis_data: list[tuple[Any, ...]] = []
        metrics_data: list[tuple[Any, ...]] = []
        runs_data: list[tuple[Any, ...]] = []

        for r in range(50):
            run_id = f"bench_run_{r:03d}"
            runs_data.append((run_id, "test_wf", f"2026-11-27T{r:02d}:00:00Z", "SUCCESS", "1.0"))
            for s in range(50):
                analysis_data.append(
                    (run_id, s, "arbiter", "P2", "PASS", 0.95, "{}", "2026-11-27T00:00:00Z", "1.0")
                )
                metrics_data.append(
                    (run_id, s, "latency_ms", float(s * 10), "2026-11-27T00:00:00Z", "1.0")
                )

        conn.executemany(
            "INSERT INTO runs (run_id, workflow, timestamp, status, schema_version) VALUES (?, ?, ?, ?, ?);",
            runs_data,
        )
        conn.executemany(
            "INSERT INTO analysis (run_id, step, analyzer, category, verdict, confidence, details_json, timestamp, schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
            analysis_data,
        )
        conn.executemany(
            "INSERT INTO metrics (run_id, step, metric_name, metric_value, timestamp, schema_version) VALUES (?, ?, ?, ?, ?, ?);",
            metrics_data,
        )
        conn.execute("ANALYZE;")

    # Benchmark 100 random point lookups
    with db.connection() as conn:
        start_time = time.perf_counter()
        for i in range(100):
            target_run = f"bench_run_{i % 50:03d}"
            target_step = (i * 3) % 50
            row = conn.execute(
                "SELECT id, verdict FROM analysis WHERE run_id = ? AND step = ?;",
                (target_run, target_step),
            ).fetchone()
            assert row is not None
        duration = time.perf_counter() - start_time
        avg_latency_ms = (duration / 100) * 1000.0

        # Assert sub-millisecond query time
        assert avg_latency_ms < 1.0, (
            f"Average query latency {avg_latency_ms:.4f}ms exceeded 1.0ms SLA!"
        )
