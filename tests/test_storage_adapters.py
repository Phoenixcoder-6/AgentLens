"""
tests/test_storage_adapters.py — Day 50 Storage Adapter Test Suite
==================================================================
Tests for:
    - BaseStorageAdapter abstract contract enforcement
    - SQLiteStorageAdapter CRUD parity and metrics filtering
    - StorageWriter polymorphism with DatabaseManager and SQLiteStorageAdapter
    - get_storage_adapter factory URL resolution and environment variable routing
    - PostgreSQLStorageAdapter connection URL parsing, DDL compilation, and mock mode CRUD
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from normalizer.normalizer import Normalizer
from schema.models import AgentStep, HandoffState, RunTrace, StepStatus, TokenUsage
from storage.base_adapter import BaseStorageAdapter, get_storage_adapter
from storage.db import DatabaseManager
from storage.postgres_adapter import PostgreSQLStorageAdapter
from storage.sqlite_adapter import SQLiteStorageAdapter
from storage.writer import StorageWriter


@pytest.fixture
def sample_trace() -> RunTrace:
    """Create a sample 2-step RunTrace for storage adapter testing."""
    steps = [
        AgentStep(
            run_id="run_adapter_test_01",
            step=1,
            agent="Researcher",
            input='{"topic": "quantum"}',
            output='{"findings": "quantum computing advances"}',
            latency_ms=250.0,
            status=StepStatus.SUCCESS,
            prompt="Research quantum",
            timestamp=datetime(2026, 7, 15, 10, 0, 0, tzinfo=UTC),
            tokens=TokenUsage(prompt=120, completion=80, total=200),
            handoff=HandoffState(
                input_state={"topic": "quantum"},
                filtered_state={"findings": "quantum computing advances"},
                output_state={"topic": "quantum", "findings": "quantum computing advances"},
            ),
        ),
        AgentStep(
            run_id="run_adapter_test_01",
            step=2,
            agent="Writer",
            input='{"findings": "quantum computing advances"}',
            output='{"article": "Summary of quantum computing"}',
            latency_ms=450.0,
            status=StepStatus.SUCCESS,
            prompt="Write article",
            timestamp=datetime(2026, 7, 15, 10, 0, 1, tzinfo=UTC),
            tokens=TokenUsage(prompt=250, completion=150, total=400),
            handoff=HandoffState(
                input_state={"findings": "quantum computing advances"},
                filtered_state={"article": "Summary of quantum computing"},
                output_state={"article": "Summary of quantum computing"},
            ),
        ),
    ]

    return RunTrace(
        run_id="run_adapter_test_01",
        workflow="ResearchWorkflow",
        timestamp=datetime(2026, 7, 15, 10, 0, 0, tzinfo=UTC),
        status=StepStatus.SUCCESS,
        total_latency_ms=700.0,
        total_tokens=600,
        steps=steps,
        expected_output="Final quantum article",
    )


def test_base_storage_adapter_abstract_enforcement():
    """Verify that BaseStorageAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        BaseStorageAdapter()  # type: ignore[abstract]

    err_str = str(exc_info.value)
    assert "Can't instantiate abstract class" in err_str
    for required in ["save_run", "get_run", "list_runs", "save_analysis", "query_metrics"]:
        assert required in err_str


def test_sqlite_adapter_crud_parity(tmp_path: Path, sample_trace: RunTrace):
    """Verify SQLiteStorageAdapter provides full CRUD functionality."""
    db_file = str(tmp_path / "sqlite_adapter_test.db")
    adapter = SQLiteStorageAdapter(db_path=db_file)
    adapter.initialize()

    normalized = Normalizer().normalize_run(sample_trace)
    adapter.save_run(normalized, trace_json=sample_trace.model_dump_json())

    # 1. get_run
    run_row = adapter.get_run("run_adapter_test_01")
    assert run_row is not None
    assert run_row["run_id"] == "run_adapter_test_01"
    assert run_row["workflow"] == "ResearchWorkflow"
    assert run_row["total_tokens"] == 600
    assert run_row["expected_output"] == "Final quantum article"

    # 2. list_runs
    runs_all = adapter.list_runs()
    assert len(runs_all) == 1
    assert runs_all[0]["run_id"] == "run_adapter_test_01"

    runs_filtered = adapter.list_runs(status_filter="SUCCESS", workflow_filter="ResearchWorkflow")
    assert len(runs_filtered) == 1

    runs_none = adapter.list_runs(status_filter="FAILED")
    assert len(runs_none) == 0

    # 3. get_steps
    steps = adapter.get_steps("run_adapter_test_01")
    assert len(steps) == 2
    assert steps[0]["agent"] == "Researcher"
    assert steps[1]["agent"] == "Writer"

    # 4. query_metrics (all and filtered)
    all_metrics = adapter.query_metrics("run_adapter_test_01")
    assert len(all_metrics) >= 6  # latency, execution_time, tokens per step

    latency_metrics = adapter.query_metrics("run_adapter_test_01", metric_name="latency_ms")
    assert len(latency_metrics) == 2
    assert {m["agent"] for m in latency_metrics} == {"Researcher", "Writer"}

    token_metrics = adapter.query_metrics("run_adapter_test_01", metric_name="tokens_total")
    assert len(token_metrics) == 2

    # 5. save_analysis
    adapter.save_analysis(
        run_id="run_adapter_test_01",
        analyzer="rule_engine",
        timestamp=datetime.now(UTC).isoformat(),
        verdict="PASS",
        confidence=0.95,
        details={"checks_passed": 10},
    )

    # 6. save_rule_matches & get_rule_matches
    count = adapter.save_rule_matches(
        run_id="run_adapter_test_01",
        matches=[
            {
                "rule_id": "P1_ROLE_MISMATCH",
                "category": "role",
                "severity": "HIGH",
                "agent": "Writer",
                "step": 2,
                "description": "Test match",
            }
        ],
    )
    assert count == 1
    matches = adapter.get_rule_matches("run_adapter_test_01")
    assert len(matches) == 1
    assert matches[0]["rule_id"] == "P1_ROLE_MISMATCH"

    # 7. table_counts
    counts = adapter.table_counts()
    assert counts["runs"] == 1
    assert counts["steps"] == 2
    assert counts["analysis"] == 1

    adapter.close()


def test_storage_writer_interoperability(tmp_path: Path, sample_trace: RunTrace):
    """Verify StorageWriter accepts both DatabaseManager and SQLiteStorageAdapter."""
    db_file = str(tmp_path / "writer_interop.db")
    adapter = SQLiteStorageAdapter(db_path=db_file)
    adapter.initialize()

    # Pass adapter directly to StorageWriter
    writer = StorageWriter(adapter)
    normalized = Normalizer().normalize_run(sample_trace)
    writer.write_run(normalized)

    # Verify run persisted
    run_record = adapter.get_run("run_adapter_test_01")
    assert run_record is not None
    assert run_record["run_id"] == "run_adapter_test_01"

    # Also check DatabaseManager.get_adapter()
    db_mgr = DatabaseManager(db_path=db_file)
    adapter_from_db = db_mgr.get_adapter()
    assert isinstance(adapter_from_db, SQLiteStorageAdapter)
    assert adapter_from_db.get_run("run_adapter_test_01") is not None

    adapter.close()
    db_mgr.close()


def test_factory_get_storage_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify get_storage_adapter() properly resolves backends from URLs and env vars."""
    db_file = str(tmp_path / "factory_test.db")

    # 1. Explicit sqlite path
    adapter1 = get_storage_adapter(db_file)
    assert isinstance(adapter1, SQLiteStorageAdapter)
    assert adapter1.db_path == db_file
    adapter1.close()

    # 2. sqlite:/// URI
    adapter2 = get_storage_adapter(f"sqlite:///{db_file}")
    assert isinstance(adapter2, SQLiteStorageAdapter)
    assert adapter2.db_path == db_file
    adapter2.close()

    # 3. :memory:
    adapter3 = get_storage_adapter(":memory:")
    assert isinstance(adapter3, SQLiteStorageAdapter)
    assert adapter3.db_path == ":memory:"
    adapter3.close()

    # 4. postgresql:// URI
    adapter4 = get_storage_adapter(
        "postgresql://agentlens:secret@db.prod.internal:5432/traces_db?sslmode=require"
    )
    assert isinstance(adapter4, PostgreSQLStorageAdapter)
    info = adapter4.get_connection_info(mask_password=True)
    assert info["host"] == "db.prod.internal"
    assert info["port"] == 5432
    assert info["user"] == "agentlens"
    assert info["password"] == "******"
    assert info["database"] == "traces_db"
    assert info["sslmode"] == "require"

    # 5. AGENTLENS_DB_URL environment variable routing
    monkeypatch.setenv("AGENTLENS_DB_URL", "postgresql://user:pass@127.0.0.1:5433/custom_agentlens")
    adapter5 = get_storage_adapter()
    assert isinstance(adapter5, PostgreSQLStorageAdapter)
    assert adapter5.get_connection_info()["port"] == 5433
    assert adapter5.get_connection_info()["database"] == "custom_agentlens"

    monkeypatch.delenv("AGENTLENS_DB_URL", raising=False)


def test_postgresql_adapter_ddl_and_mock_crud(sample_trace: RunTrace):
    """Verify PostgreSQLStorageAdapter DDL compilation and mock mode CRUD."""
    url = "postgresql://user:pass@localhost:5432/agentlens"
    adapter = PostgreSQLStorageAdapter(url, mock_mode=True)
    adapter.initialize()

    # Verify DDL contains PostgreSQL specific types
    ddls = adapter.get_ddl()
    assert len(ddls) >= 5
    joined_ddl = "\n".join(ddls)
    assert "TIMESTAMPTZ" in joined_ddl
    assert "JSONB" in joined_ddl
    assert "BIGSERIAL" in joined_ddl
    assert "DOUBLE PRECISION" in joined_ddl

    # Test CRUD operations in mock mode
    normalized = Normalizer().normalize_run(sample_trace)
    adapter.save_run(normalized, trace_json=sample_trace.model_dump_json())

    run_row = adapter.get_run("run_adapter_test_01")
    assert run_row is not None
    assert run_row["run_id"] == "run_adapter_test_01"
    assert run_row["workflow"] == "ResearchWorkflow"

    runs = adapter.list_runs(status_filter="SUCCESS")
    assert len(runs) == 1

    steps = adapter.get_steps("run_adapter_test_01")
    assert len(steps) == 2

    metrics = adapter.query_metrics("run_adapter_test_01", metric_name="latency_ms")
    assert len(metrics) == 2

    adapter.save_analysis(
        run_id="run_adapter_test_01",
        analyzer="diff_engine",
        timestamp=datetime.now(UTC).isoformat(),
        verdict="PASS",
    )

    adapter.save_rule_matches(
        run_id="run_adapter_test_01",
        matches=[{"rule_id": "P3_HALLUCINATION", "severity": "MEDIUM"}],
    )
    matches = adapter.get_rule_matches("run_adapter_test_01")
    assert len(matches) == 1

    counts = adapter.table_counts()
    assert counts["runs"] == 1
    assert counts["steps"] == 2
    assert counts["analysis"] == 1
    assert counts["rule_matches"] == 1

    # Retention delete
    deleted = adapter.delete_runs_older_than(datetime.now(UTC).isoformat())
    assert "run_adapter_test_01" in deleted
    assert adapter.get_run("run_adapter_test_01") is None
    assert adapter.table_counts()["runs"] == 0

    adapter.close()
