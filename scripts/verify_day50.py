"""
scripts/verify_day50.py — Day 50 Pluggable Storage Adapter Verification Suite
=============================================================================
Automated verification of the Day 50 storage adapter abstraction layer:
1. BaseStorageAdapter abstract contract enforcement
2. SQLiteStorageAdapter initialization & WAL mode / high-concurrency pragmas
3. SQLiteStorageAdapter full CRUD parity & atomic transaction persistence
4. SQLiteStorageAdapter query metrics filtering
5. StorageWriter interoperability & DatabaseManager.get_adapter()
6. get_storage_adapter() dynamic factory URL & env var resolution
7. PostgreSQLStorageAdapter DSN parsing & schema / DDL validation
8. Day 43 regression suite benchmark preservation (16/20 exact attribution)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from normalizer.normalizer import Normalizer
from schema.models import AgentStep, HandoffState, RunTrace, StepStatus, TokenUsage
from storage.base_adapter import BaseStorageAdapter, get_storage_adapter
from storage.db import DatabaseManager
from storage.postgres_adapter import PostgreSQLStorageAdapter
from storage.sqlite_adapter import SQLiteStorageAdapter
from storage.writer import StorageWriter


def _create_sample_run(run_id: str = "run_verify_day50") -> RunTrace:
    steps = [
        AgentStep(
            run_id=run_id,
            step=1,
            agent="Researcher",
            input='{"topic": "quantum-ml"}',
            output='{"findings": "quantum neural network architectures"}',
            latency_ms=210.0,
            status=StepStatus.SUCCESS,
            prompt="Investigate QML",
            timestamp=datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC),
            tokens=TokenUsage(prompt=150, completion=90, total=240),
            handoff=HandoffState(
                input_state={"topic": "quantum-ml"},
                filtered_state={"findings": "quantum neural network architectures"},
                output_state={"findings": "quantum neural network architectures"},
            ),
        ),
        AgentStep(
            run_id=run_id,
            step=2,
            agent="Analyst",
            input='{"findings": "quantum neural network architectures"}',
            output='{"verdict": "high potential for optimization problems"}',
            latency_ms=380.0,
            status=StepStatus.SUCCESS,
            prompt="Analyze QML findings",
            timestamp=datetime(2026, 7, 10, 12, 0, 1, tzinfo=UTC),
            tokens=TokenUsage(prompt=240, completion=110, total=350),
            handoff=HandoffState(
                input_state={"findings": "quantum neural network architectures"},
                filtered_state={"verdict": "high potential for optimization problems"},
                output_state={"verdict": "high potential for optimization problems"},
            ),
        ),
    ]

    return RunTrace(
        run_id=run_id,
        workflow="QuantumQMLPipeline",
        timestamp=datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC),
        status=StepStatus.SUCCESS,
        total_latency_ms=590.0,
        total_tokens=590,
        steps=steps,
        expected_output="high potential for optimization problems",
    )


def check_1_base_adapter_contract() -> bool:
    print("\n--- Check 1: BaseStorageAdapter Abstract Contract Enforcement ---")
    try:
        BaseStorageAdapter()  # type: ignore[abstract]
        print("[FAIL] BaseStorageAdapter allowed direct instantiation without implementing abstract methods.")
        return False
    except TypeError as exc:
        err = str(exc)
        for meth in ["save_run", "get_run", "list_runs", "save_analysis", "query_metrics"]:
            if meth not in err:
                print(f"[FAIL] Expected abstract method '{meth}' in TypeError, got: {err}")
                return False

    print("[PASS] BaseStorageAdapter correctly enforces abstract methods (save_run, get_run, list_runs, save_analysis, query_metrics).")
    return True


def check_2_sqlite_adapter_initialization(tmp_dir: Path) -> bool:
    print("\n--- Check 2: SQLiteStorageAdapter Initialization & Pragmas ---")
    db_file = str(tmp_dir / "sqlite_init.db")
    adapter = SQLiteStorageAdapter(db_path=db_file)
    adapter.initialize()

    pragmas = adapter.get_pragmas()
    if pragmas.get("journal_mode") != "wal":
        print(f"[FAIL] Expected journal_mode='wal', got '{pragmas.get('journal_mode')}'")
        adapter.close()
        return False

    if pragmas.get("busy_timeout") != 10000:
        print(f"[FAIL] Expected busy_timeout=10000, got {pragmas.get('busy_timeout')}")
        adapter.close()
        return False

    if pragmas.get("synchronous") != 1:
        print(f"[FAIL] Expected synchronous=1 (NORMAL), got {pragmas.get('synchronous')}")
        adapter.close()
        return False

    adapter.close()
    print("[PASS] SQLiteStorageAdapter initialized with WAL mode, busy_timeout=10000, and synchronous=NORMAL.")
    return True


def check_3_sqlite_crud_parity(tmp_dir: Path) -> bool:
    print("\n--- Check 3: SQLiteStorageAdapter Full CRUD Parity ---")
    db_file = str(tmp_dir / "sqlite_crud.db")
    adapter = SQLiteStorageAdapter(db_path=db_file)
    adapter.initialize()

    trace = _create_sample_run("run_crud_01")
    normalized = Normalizer().normalize_run(trace)
    adapter.save_run(normalized, trace_json=trace.model_dump_json())

    run_row = adapter.get_run("run_crud_01")
    if not run_row or run_row["run_id"] != "run_crud_01":
        print(f"[FAIL] Failed to retrieve run record by ID: {run_row}")
        adapter.close()
        return False

    if run_row["workflow"] != "QuantumQMLPipeline" or run_row["total_tokens"] != 590:
        print(f"[FAIL] Stored run attributes mismatch: {run_row}")
        adapter.close()
        return False

    # Check list_runs
    runs_list = adapter.list_runs(status_filter="SUCCESS", workflow_filter="QuantumQMLPipeline")
    if len(runs_list) != 1 or runs_list[0]["run_id"] != "run_crud_01":
        print(f"[FAIL] list_runs filter returned unexpected count: {len(runs_list)}")
        adapter.close()
        return False

    # Check steps
    steps = adapter.get_steps("run_crud_01")
    if len(steps) != 2:
        print(f"[FAIL] Expected 2 steps, got {len(steps)}")
        adapter.close()
        return False

    adapter.close()
    print("[PASS] SQLiteStorageAdapter CRUD parity verified (save_run, get_run, list_runs, get_steps).")
    return True


def check_4_sqlite_metrics_query(tmp_dir: Path) -> bool:
    print("\n--- Check 4: SQLiteStorageAdapter Metrics Querying & Filtering ---")
    db_file = str(tmp_dir / "sqlite_metrics.db")
    adapter = SQLiteStorageAdapter(db_path=db_file)
    adapter.initialize()

    trace = _create_sample_run("run_metrics_01")
    normalized = Normalizer().normalize_run(trace)
    adapter.save_run(normalized)

    all_metrics = adapter.query_metrics("run_metrics_01")
    if len(all_metrics) < 6:
        print(f"[FAIL] Expected at least 6 total metrics, got {len(all_metrics)}")
        adapter.close()
        return False

    latency_metrics = adapter.query_metrics("run_metrics_01", metric_name="latency_ms")
    if len(latency_metrics) != 2:
        print(f"[FAIL] Expected 2 latency_ms metrics, got {len(latency_metrics)}")
        adapter.close()
        return False

    token_metrics = adapter.query_metrics("run_metrics_01", metric_name="tokens_total")
    if len(token_metrics) != 2:
        print(f"[FAIL] Expected 2 tokens_total metrics, got {len(token_metrics)}")
        adapter.close()
        return False

    adapter.close()
    print("[PASS] SQLiteStorageAdapter metrics querying and metric_name filtering verified.")
    return True


def check_5_storage_writer_interoperability(tmp_dir: Path) -> bool:
    print("\n--- Check 5: StorageWriter & DatabaseManager Interoperability ---")
    db_file = str(tmp_dir / "writer_compat.db")
    adapter = SQLiteStorageAdapter(db_path=db_file)
    adapter.initialize()

    # Pass adapter to StorageWriter
    writer = StorageWriter(adapter)
    trace = _create_sample_run("run_interop_01")
    normalized = Normalizer().normalize_run(trace)
    writer.write_run(normalized)

    if adapter.get_run("run_interop_01") is None:
        print("[FAIL] StorageWriter failed to persist run when passed SQLiteStorageAdapter.")
        adapter.close()
        return False

    # Verify DatabaseManager.get_adapter()
    db_mgr = DatabaseManager(db_path=db_file)
    adapter_from_db = db_mgr.get_adapter()
    if not isinstance(adapter_from_db, SQLiteStorageAdapter):
        print(f"[FAIL] DatabaseManager.get_adapter() returned {type(adapter_from_db)}")
        adapter.close()
        db_mgr.close()
        return False

    if adapter_from_db.get_run("run_interop_01") is None:
        print("[FAIL] Adapter derived from DatabaseManager cannot read persisted runs.")
        adapter.close()
        db_mgr.close()
        return False

    adapter.close()
    db_mgr.close()
    print("[PASS] StorageWriter polymorphism and DatabaseManager.get_adapter() verified.")
    return True


def check_6_factory_resolution(tmp_dir: Path) -> bool:
    print("\n--- Check 6: get_storage_adapter() Factory Resolution ---")
    db_file = str(tmp_dir / "factory_resolve.db")

    # 1. Plain file path
    a1 = get_storage_adapter(db_file)
    if not isinstance(a1, SQLiteStorageAdapter) or a1.db_path != db_file:
        print(f"[FAIL] Failed resolving plain path: {type(a1)}")
        return False
    a1.close()

    # 2. sqlite:/// URI
    a2 = get_storage_adapter(f"sqlite:///{db_file}")
    if not isinstance(a2, SQLiteStorageAdapter) or a2.db_path != db_file:
        print(f"[FAIL] Failed resolving sqlite:/// URI: {type(a2)}")
        return False
    a2.close()

    # 3. postgresql:// URI
    pg_url = "postgresql://user:secret@postgres.host.internal:5432/agentlens_prod?sslmode=require"
    a3 = get_storage_adapter(pg_url)
    if not isinstance(a3, PostgreSQLStorageAdapter):
        print(f"[FAIL] Failed resolving postgresql:// URI: {type(a3)}")
        return False
    a3.close()

    # 4. AGENTLENS_DB_URL environment variable routing
    old_env = os.environ.get("AGENTLENS_DB_URL")
    try:
        os.environ["AGENTLENS_DB_URL"] = "postgresql://user:pass@127.0.0.1:5432/env_db"
        a4 = get_storage_adapter()
        if not isinstance(a4, PostgreSQLStorageAdapter):
            print(f"[FAIL] Failed resolving from AGENTLENS_DB_URL: {type(a4)}")
            return False
        if a4.get_connection_info()["database"] != "env_db":
            print(f"[FAIL] Database mismatch in env resolution: {a4.get_connection_info()}")
            return False
        a4.close()
    finally:
        if old_env is not None:
            os.environ["AGENTLENS_DB_URL"] = old_env
        else:
            os.environ.pop("AGENTLENS_DB_URL", None)

    print("[PASS] get_storage_adapter() successfully resolved SQLite paths, sqlite:/// URIs, postgresql:// URIs, and AGENTLENS_DB_URL.")
    return True


def check_7_postgresql_adapter_schema() -> bool:
    print("\n--- Check 7: PostgreSQLStorageAdapter DSN Parsing & Schema Validation ---")
    url = "postgresql://agent_user:mypassword@db.cloud.service:5432/production_traces?sslmode=verify-full"
    adapter = PostgreSQLStorageAdapter(url, mock_mode=True)
    adapter.initialize()

    info = adapter.get_connection_info(mask_password=True)
    if info.get("host") != "db.cloud.service":
        print(f"[FAIL] Host parse mismatch: {info.get('host')}")
        return False
    if info.get("port") != 5432:
        print(f"[FAIL] Port parse mismatch: {info.get('port')}")
        return False
    if info.get("user") != "agent_user":
        print(f"[FAIL] User parse mismatch: {info.get('user')}")
        return False
    if info.get("password") != "******":
        print(f"[FAIL] Password not masked: {info.get('password')}")
        return False
    if info.get("database") != "production_traces":
        print(f"[FAIL] Database parse mismatch: {info.get('database')}")
        return False
    if info.get("sslmode") != "verify-full":
        print(f"[FAIL] SSL mode parse mismatch: {info.get('sslmode')}")
        return False

    ddl_list = adapter.get_ddl()
    full_ddl = "\n".join(ddl_list)
    for expected_keyword in ["TIMESTAMPTZ", "JSONB", "BIGSERIAL", "DOUBLE PRECISION", "CREATE INDEX"]:
        if expected_keyword not in full_ddl:
            print(f"[FAIL] PostgreSQL DDL missing expected keyword '{expected_keyword}'")
            return False

    # Test CRUD operations in mock mode
    trace = _create_sample_run("run_pg_01")
    normalized = Normalizer().normalize_run(trace)
    adapter.save_run(normalized, trace_json=trace.model_dump_json())

    run_row = adapter.get_run("run_pg_01")
    if not run_row or run_row["run_id"] != "run_pg_01":
        print(f"[FAIL] PostgreSQLStorageAdapter mock mode failed to retrieve run: {run_row}")
        return False

    adapter.close()
    print("[PASS] PostgreSQLStorageAdapter DSN parsing, DDL compilation, and mock mode verified.")
    return True


def check_8_day43_regression_preservation() -> bool:
    print("\n--- Check 8: Day 43 Regression Benchmark Preservation (Zero Regressions) ---")
    script_path = Path(__file__).resolve().parent / "run_day43_regression.py"
    if not script_path.exists():
        print(f"[FAIL] Could not locate regression script at {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[FAIL] Day 43 regression failed with returncode {result.returncode}:\n{result.stderr}\n{result.stdout}")
        return False

    if "16/20 (80.0%)" not in result.stdout or "Overall Regression Gate" not in result.stdout:
        print(f"[FAIL] Regression output did not contain expected success message:\n{result.stdout}")
        return False

    print("[PASS] Day 43 regression benchmark preserved: 16/20 (80.0%) exact attribution, zero regressions.")
    return True


def main() -> int:
    print("=====================================================================")
    print("AgentLens v2.0 Day 50 Verification: Pluggable Storage Adapters")
    print("=====================================================================")

    tmp_dir = Path(tempfile.mkdtemp(prefix="agentlens_day50_verify_"))
    try:
        checks = [
            check_1_base_adapter_contract(),
            check_2_sqlite_adapter_initialization(tmp_dir),
            check_3_sqlite_crud_parity(tmp_dir),
            check_4_sqlite_metrics_query(tmp_dir),
            check_5_storage_writer_interoperability(tmp_dir),
            check_6_factory_resolution(tmp_dir),
            check_7_postgresql_adapter_schema(),
            check_8_day43_regression_preservation(),
        ]

        passed = sum(1 for c in checks if c)
        total = len(checks)
        print("\n=====================================================================")
        print(f"Day 50 Verification Summary: {passed}/{total} checks PASSED")
        print("=====================================================================")

        if passed == total:
            print("\n>>> DAY 50 STORAGE ADAPTER ABSTRACTION FULLY CERTIFIED <<<")
            return 0
        else:
            print(f"\n>>> DAY 50 FAILED: {total - passed} checks failed <<<")
            return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
