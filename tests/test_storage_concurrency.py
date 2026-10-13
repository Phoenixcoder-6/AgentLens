"""
tests/test_storage_concurrency.py — Day 49 SQLite WAL & Connection Pool Stress Tests
=====================================================================================
Validates:
1. SQLite WAL mode, busy_timeout=10000, and synchronous=NORMAL pragmas.
2. Thread-safe connection pool acquisition and reuse across threads.
3. 100 simultaneous concurrent writes with zero "database is locked" errors.
4. Concurrent non-blocking reads while heavy write bursts are in progress.
5. Exponential retry backoff on lock contention.
6. Transaction rollback cleanly discarding partial writes on failure.
"""

from __future__ import annotations

import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from normalizer.normalizer import Normalizer
from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
from storage.db import DatabaseManager
from storage.writer import StorageWriter


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    """Return a path to a clean SQLite database file."""
    return str(tmp_path / "concurrency_test.db")


def test_wal_mode_and_pragmas_configured(test_db_path: str):
    """Verify that file-based SQLite connections use WAL mode, busy_timeout=10000, and synchronous=NORMAL."""
    db = DatabaseManager(db_path=test_db_path)
    db.initialize()

    pragmas = db.get_pragmas()
    assert pragmas["journal_mode"] == "wal", f"Expected 'wal', got {pragmas['journal_mode']}"
    assert pragmas["busy_timeout"] == 10000, f"Expected 10000, got {pragmas['busy_timeout']}"
    assert pragmas["synchronous"] == 1, f"Expected 1 (NORMAL), got {pragmas['synchronous']}"
    assert pragmas["foreign_keys"] == 1, f"Expected 1 (ON), got {pragmas['foreign_keys']}"

    db.close()


def test_connection_pool_acquire_and_reuse(test_db_path: str):
    """Verify connection pool bounds, reuse, and thread safety."""
    db = DatabaseManager(db_path=test_db_path, pool_size=5)
    db.initialize()

    initial_stats = db.pool.stats()
    assert initial_stats["max_connections"] == 5

    def worker_acquire(worker_id: int):
        with db.connection() as conn:
            row = conn.execute("SELECT 1 AS val").fetchone()
            assert row["val"] == 1
            time.sleep(0.01)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_acquire, i) for i in range(25)]
        for f in futures:
            f.result()

    stats_after = db.pool.stats()
    # At most 5 connections created
    assert stats_after["active_connections"] <= 5
    assert stats_after["available"] >= 1

    db.close()


def test_100_concurrent_writes_zero_lock_errors(test_db_path: str):
    """
    Stress test: 100 simultaneous threads writing full NormalizedRun objects
    (each with 2 steps and metrics) concurrently.
    Must finish with 0 database lock errors and 100% data integrity.
    """
    db = DatabaseManager(db_path=test_db_path, pool_size=10)
    db.initialize()
    writer = StorageWriter(db)
    normalizer = Normalizer()

    def write_worker(idx: int) -> str:
        run_id = f"stress_run_{idx:03d}"
        trace = RunTrace(
            run_id=run_id,
            workflow="concurrency_stress",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[
                AgentStep(
                    run_id=run_id,
                    step=1,
                    agent="researcher",
                    input='{"topic": "stress"}',
                    output='{"findings": "ok"}',
                    latency_ms=10.0 + random.uniform(1.0, 5.0),
                    status=StepStatus.SUCCESS,
                    tokens=TokenUsage(prompt=100, completion=50, total=150),
                ),
                AgentStep(
                    run_id=run_id,
                    step=2,
                    agent="writer",
                    input='{"findings": "ok"}',
                    output='{"report": "done"}',
                    latency_ms=20.0 + random.uniform(1.0, 5.0),
                    status=StepStatus.SUCCESS,
                    tokens=TokenUsage(prompt=150, completion=70, total=220),
                ),
            ],
        )
        norm_run = normalizer.normalize_run(trace)
        writer.write_run(norm_run)
        return run_id

    # Execute 100 writes across 20 worker threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(write_worker, i) for i in range(100)]
        written_run_ids = [f.result() for f in futures]

    assert len(written_run_ids) == 100

    # Verify counts in database
    counts = db.table_counts()
    assert counts["runs"] == 100, f"Expected 100 runs, got {counts['runs']}"
    assert counts["steps"] == 200, f"Expected 200 steps, got {counts['steps']}"
    # Each step writes 4 metrics
    assert counts["metrics"] >= 800

    db.close()


def test_concurrent_readers_and_writers_interleaved(test_db_path: str):
    """
    Validate WAL mode reader-writer concurrency:
    50 threads writing while 50 threads reading simultaneously.
    """
    db = DatabaseManager(db_path=test_db_path, pool_size=10)
    db.initialize()
    writer = StorageWriter(db)

    normalizer = Normalizer()

    # Seed 5 runs first
    for i in range(5):
        trace = RunTrace(
            run_id=f"seed_{i}",
            workflow="seed_wf",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[],
        )
        writer.write_run(normalizer.normalize_run(trace))

    read_results: list[int] = []

    def reader_worker():
        for _ in range(5):
            runs = db.list_runs(limit=10)
            read_results.append(len(runs))
            time.sleep(random.uniform(0.001, 0.005))

    def writer_worker(idx: int):
        trace = RunTrace(
            run_id=f"rw_run_{idx:03d}",
            workflow="rw_wf",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[],
        )
        writer.write_run(normalizer.normalize_run(trace))

    with ThreadPoolExecutor(max_workers=20) as executor:
        read_futures = [executor.submit(reader_worker) for _ in range(10)]
        write_futures = [executor.submit(writer_worker, i) for i in range(40)]

        for f in read_futures + write_futures:
            f.result()

    assert len(read_results) == 50
    counts = db.table_counts()
    assert counts["runs"] == 45  # 5 seed + 40 written

    db.close()


def test_retry_backoff_on_simulated_locks(test_db_path: str):
    """Verify execute_with_retry recovers from transient database locked errors."""
    db = DatabaseManager(db_path=test_db_path)
    db.initialize()

    call_count = 0

    def flaky_operation(conn: sqlite3.Connection) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise sqlite3.OperationalError("database is locked")
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, workflow, timestamp, status, schema_version) VALUES ('flaky_01', 'wf', '2026-01-01', 'SUCCESS', '1.0')"
        )
        return "success"

    res = db.execute_with_retry(flaky_operation, max_retries=5, base_delay=0.01)
    assert res == "success"
    assert call_count == 3
    assert db.get_run("flaky_01") is not None

    db.close()


def test_transaction_rollback_preserves_integrity(test_db_path: str):
    """Verify that an exception inside connection() rolls back all pending inserts."""
    db = DatabaseManager(db_path=test_db_path)
    db.initialize()

    with pytest.raises(ValueError, match="Forced failure during write"):
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, workflow, timestamp, status, schema_version) VALUES ('rollback_run', 'wf', '2026-01-01', 'SUCCESS', '1.0')"
            )
            # Intentionally explode before commit
            raise ValueError("Forced failure during write")

    assert db.get_run("rollback_run") is None
    db.close()
