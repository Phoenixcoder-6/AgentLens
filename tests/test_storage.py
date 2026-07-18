"""
tests/test_storage.py — Day 8: Storage Layer tests
====================================================
Tests for:
    - DatabaseManager: table creation, CRUD for all 4 tables
    - StorageWriter: writes NormalizedRun correctly
    - Checkpoint: full run insert + query back
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from schema.models import (
    AgentStep, HandoffState, RunTrace, StepStatus, SCHEMA_VERSION, TokenUsage
)
from normalizer.normalizer import Normalizer, NormalizedRun, NormalizedStep
from storage.db import DatabaseManager
from storage.writer import StorageWriter


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path) -> DatabaseManager:
    """Fresh in-memory-equivalent DB using a temp file for each test."""
    db_path = str(tmp_path / "test_agentlens.db")
    db = DatabaseManager(db_path=db_path)
    db.initialize()
    return db


@pytest.fixture
def sample_run() -> RunTrace:
    """A minimal 3-step RunTrace for testing."""
    def make_step(n, agent):
        return AgentStep(
            run_id="run_test999",
            step=n,
            agent=agent,
            input='{"topic": "AI"}',
            output='{"result": "done"}',
            latency_ms=float(n * 100),
            status=StepStatus.SUCCESS,
            prompt=f"added=['key_{n}']",
            timestamp=datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc),
            tokens=TokenUsage(prompt=10, completion=20, total=30),
            handoff=HandoffState(
                input_state={"topic": "AI"},
                filtered_state={"result": "done"},
                output_state={"topic": "AI", "result": "done"},
            ),
        )

    return RunTrace(
        run_id="run_test999",
        workflow="test_pipeline",
        timestamp=datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc),
        steps=[make_step(1, "researcher"), make_step(2, "writer"), make_step(3, "verifier")],
        total_latency_ms=600.0,
        total_tokens=90,
        status=StepStatus.SUCCESS,
    )


@pytest.fixture
def normalized_run(sample_run) -> NormalizedRun:
    return Normalizer().normalize_run(sample_run)


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseManager — initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseManagerInit:

    def test_initialize_creates_all_tables(self, tmp_db):
        counts = tmp_db.table_counts()
        assert set(counts.keys()) == {"runs", "steps", "analysis", "metrics"}

    def test_initialize_is_idempotent(self, tmp_db):
        """Calling initialize() twice should not raise."""
        tmp_db.initialize()
        counts = tmp_db.table_counts()
        assert all(v == 0 for v in counts.values())

    def test_all_tables_start_empty(self, tmp_db):
        counts = tmp_db.table_counts()
        assert all(v == 0 for v in counts.values())


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseManager — runs table
# ─────────────────────────────────────────────────────────────────────────────

class TestRunsTable:

    def test_insert_and_get_run(self, tmp_db):
        tmp_db.insert_run(
            run_id="run_abc",
            workflow="test_wf",
            timestamp="2026-07-17T12:00:00+00:00",
            status="SUCCESS",
            total_latency_ms=1234.5,
            total_tokens=100,
            schema_version=SCHEMA_VERSION,
        )
        row = tmp_db.get_run("run_abc")
        assert row is not None
        assert row["run_id"] == "run_abc"
        assert row["workflow"] == "test_wf"
        assert row["status"] == "SUCCESS"
        assert row["total_latency_ms"] == 1234.5
        assert row["schema_version"] == SCHEMA_VERSION

    def test_get_run_returns_none_for_missing(self, tmp_db):
        assert tmp_db.get_run("nonexistent") is None

    def test_insert_run_is_upsert(self, tmp_db):
        """Inserting with same run_id replaces the row."""
        tmp_db.insert_run("run_x", "wf", "2026-07-17T12:00:00+00:00", "SUCCESS",
                           100.0, 0, SCHEMA_VERSION)
        tmp_db.insert_run("run_x", "wf", "2026-07-17T12:00:00+00:00", "ERROR",
                           100.0, 0, SCHEMA_VERSION)
        row = tmp_db.get_run("run_x")
        assert row["status"] == "ERROR"

    def test_list_runs_empty(self, tmp_db):
        assert tmp_db.list_runs() == []

    def test_list_runs_returns_all(self, tmp_db):
        for i in range(3):
            tmp_db.insert_run(f"run_{i}", "wf", "2026-07-17T12:00:00+00:00",
                              "SUCCESS", 100.0, 0, SCHEMA_VERSION)
        rows = tmp_db.list_runs()
        assert len(rows) == 3

    def test_list_runs_status_filter(self, tmp_db):
        tmp_db.insert_run("run_ok", "wf", "2026-07-17T12:00:00+00:00",
                          "SUCCESS", 100.0, 0, SCHEMA_VERSION)
        tmp_db.insert_run("run_bad", "wf", "2026-07-17T12:00:00+00:00",
                          "ERROR", 100.0, 0, SCHEMA_VERSION)
        rows = tmp_db.list_runs(status_filter="SUCCESS")
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run_ok"


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseManager — steps table
# ─────────────────────────────────────────────────────────────────────────────

class TestStepsTable:

    def _insert_run(self, db):
        db.insert_run("run_s", "wf", "2026-07-17T12:00:00+00:00",
                      "SUCCESS", 300.0, 0, SCHEMA_VERSION)

    def test_insert_and_get_steps(self, tmp_db):
        self._insert_run(tmp_db)
        tmp_db.insert_step("run_s", 1, "researcher", "SUCCESS", 100.0,
                           0, 0, 0, "added=['key']", "2026-07-17T12:00:00+00:00", SCHEMA_VERSION)
        rows = tmp_db.get_steps_for_run("run_s")
        assert len(rows) == 1
        assert rows[0]["agent"] == "researcher"
        assert rows[0]["latency_ms"] == 100.0

    def test_steps_ordered_by_step_number(self, tmp_db):
        self._insert_run(tmp_db)
        for n, agent in [(3, "verifier"), (1, "researcher"), (2, "writer")]:
            tmp_db.insert_step("run_s", n, agent, "SUCCESS", float(n * 100),
                               0, 0, 0, "", "2026-07-17T12:00:00+00:00", SCHEMA_VERSION)
        rows = tmp_db.get_steps_for_run("run_s")
        assert [r["agent"] for r in rows] == ["researcher", "writer", "verifier"]

    def test_steps_empty_for_unknown_run(self, tmp_db):
        assert tmp_db.get_steps_for_run("nonexistent") == []


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseManager — metrics table
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsTable:

    def _insert_run(self, db):
        db.insert_run("run_m", "wf", "2026-07-17T12:00:00+00:00",
                      "SUCCESS", 300.0, 0, SCHEMA_VERSION)

    def test_insert_and_get_metrics(self, tmp_db):
        self._insert_run(tmp_db)
        tmp_db.insert_metric("run_m", "latency_ms", 234.5,
                             "2026-07-17T12:00:00+00:00", SCHEMA_VERSION,
                             step=1, agent="researcher", metric_unit="ms")
        rows = tmp_db.get_metrics_for_run("run_m")
        assert len(rows) == 1
        assert rows[0]["metric_name"] == "latency_ms"
        assert rows[0]["metric_value"] == 234.5
        assert rows[0]["metric_unit"] == "ms"


# ─────────────────────────────────────────────────────────────────────────────
# StorageWriter
# ─────────────────────────────────────────────────────────────────────────────

class TestStorageWriter:

    def test_write_run_inserts_run_row(self, tmp_db, normalized_run):
        writer = StorageWriter(tmp_db)
        writer.write_run(normalized_run)
        row = tmp_db.get_run("run_test999")
        assert row is not None
        assert row["workflow"] == "test_pipeline"
        assert row["schema_version"] == SCHEMA_VERSION

    def test_write_run_inserts_all_steps(self, tmp_db, normalized_run):
        writer = StorageWriter(tmp_db)
        writer.write_run(normalized_run)
        steps = tmp_db.get_steps_for_run("run_test999")
        assert len(steps) == 3

    def test_write_run_step_agents_correct(self, tmp_db, normalized_run):
        writer = StorageWriter(tmp_db)
        writer.write_run(normalized_run)
        steps = tmp_db.get_steps_for_run("run_test999")
        assert [s["agent"] for s in steps] == ["researcher", "writer", "verifier"]

    def test_write_run_inserts_latency_metrics(self, tmp_db, normalized_run):
        writer = StorageWriter(tmp_db)
        writer.write_run(normalized_run)
        metrics = tmp_db.get_metrics_for_run("run_test999")
        latency_metrics = [m for m in metrics if m["metric_name"] == "latency_ms"]
        assert len(latency_metrics) == 3  # one per step

    def test_write_run_stores_trace_json(self, tmp_db, normalized_run):
        trace_json = '{"run_id": "run_test999", "workflow": "test_pipeline"}'
        writer = StorageWriter(tmp_db)
        writer.write_run(normalized_run, trace_json=trace_json)
        row = tmp_db.get_run("run_test999")
        assert row["trace_json"] == trace_json

    def test_write_run_idempotent(self, tmp_db, normalized_run):
        """Writing same run twice should not raise (upsert)."""
        writer = StorageWriter(tmp_db)
        writer.write_run(normalized_run)
        writer.write_run(normalized_run)
        assert tmp_db.get_run("run_test999") is not None


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint — full end-to-end: capture → normalize → store → query
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:
    """
    Day 8 Checkpoint: one full run completely captured, normalized, stored,
    and queryable from SQLite.
    """

    def test_full_pipeline_capture_normalize_store(self, tmp_db, sample_run):
        """Simulates the full Day 8 path for one complete pipeline run."""
        # Step 1: Normalize
        normalized = Normalizer().normalize_run(sample_run)

        # Step 2: Store
        writer = StorageWriter(tmp_db)
        writer.write_run(normalized, trace_json=sample_run.model_dump_json())

        # Step 3: Query back
        run_row = tmp_db.get_run("run_test999")
        step_rows = tmp_db.get_steps_for_run("run_test999")
        metric_rows = tmp_db.get_metrics_for_run("run_test999")
        counts = tmp_db.table_counts()

        # Assertions
        assert run_row is not None, "Run row must exist"
        assert run_row["schema_version"] == SCHEMA_VERSION
        assert len(step_rows) == 3, "All 3 steps must be stored"
        assert len(metric_rows) >= 3, "At least one latency metric per step"
        assert counts["runs"] == 1
        assert counts["steps"] == 3
        assert counts["metrics"] >= 3
        assert counts["analysis"] == 0  # populated by future analyzers

        # Verify trace_json blob is stored and valid
        stored_json = run_row["trace_json"]
        assert stored_json is not None
        parsed = json.loads(stored_json)
        assert parsed["run_id"] == "run_test999"

        print("\n✅ CHECKPOINT PASSED")
        print(f"   runs     : {counts['runs']}")
        print(f"   steps    : {counts['steps']}")
        print(f"   metrics  : {counts['metrics']}")
        print(f"   analysis : {counts['analysis']} (empty — awaiting analyzers)")
