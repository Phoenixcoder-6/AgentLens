"""
tests/test_queue_durability.py — Day 67 Telemetry Durability, WAL & Backpressure Tests
=======================================================================================
Phase E: Data Durability, Storage Reliability & SRE

Tests:
1. Disk WAL file creation, atomic appending, and fsync synchronization.
2. Crash recovery & replay: unpersisted traces on disk are re-enqueued on restart.
3. Memory overflow disk spillover: queue capacity overflow spills safely to disk without drops.
4. Saturation detection and backpressure signaling (HTTP 429 + Retry-After).
5. Automatic WAL chunk file pruning once traces are safely persisted to storage.
6. API endpoint GET /api/queue/status reporting runtime health and queue depth.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.router import router
from capture.queue import (
    TraceIngestionQueue,
    WALDiskSpooler,
    get_global_queue,
)
from schema.models import RunTrace


@pytest.fixture
def spool_dir(tmp_path: Path) -> Path:
    spool = tmp_path / "spool"
    spool.mkdir(parents=True, exist_ok=True)
    return spool


def test_wal_spool_creation_and_append(spool_dir: Path):
    """Verify traces enqueued with wal_enabled=True are appended to disk WAL files."""
    q = TraceIngestionQueue(
        max_size=50,
        spool_dir=spool_dir,
        wal_enabled=True,
    )
    assert q.spooler is not None

    t1 = RunTrace(run_id="wal_run_01", workflow="order_processing")
    t2 = RunTrace(run_id="wal_run_02", workflow="order_processing")

    assert q.enqueue(t1) is True
    assert q.enqueue(t2) is True
    assert q.size() == 2

    # Verify WAL files exist on disk
    wal_files = list(spool_dir.glob("wal_*.jsonl"))
    assert len(wal_files) == 1

    content = wal_files[0].read_text(encoding="utf-8")
    assert "wal_run_01" in content
    assert "wal_run_02" in content
    q.clear()


def test_wal_crash_recovery(spool_dir: Path):
    """Simulate ungraceful process abort: traces on disk must be fully recovered on reboot."""
    spooler = WALDiskSpooler(spool_dir=spool_dir)

    # 1. Enqueue 5 traces and simulate worker crash before persistence
    for i in range(5):
        spooler.append_trace(RunTrace(run_id=f"crash_run_{i}", workflow="recovery_pipeline"))
    spooler.close()

    # 2. Simulate new process startup
    q_reboot = TraceIngestionQueue(
        max_size=50,
        spool_dir=spool_dir,
        wal_enabled=True,
    )
    recovered_count = q_reboot.recover_from_disk()
    assert recovered_count == 5
    assert q_reboot.size() == 5

    batch = q_reboot.dequeue_batch(max_items=10)
    assert len(batch) == 5
    run_ids = [t.run_id for t in batch]
    assert run_ids == [f"crash_run_{i}" for i in range(5)]
    q_reboot.clear()


def test_memory_overflow_disk_spillover(spool_dir: Path):
    """Queue overflow spills excess traces directly to disk rather than dropping them."""
    q = TraceIngestionQueue(
        max_size=5,
        spool_dir=spool_dir,
        wal_enabled=False,
        overflow_spill=True,
    )

    # Enqueue 12 traces (5 capacity + 7 overflow)
    for i in range(12):
        res = q.enqueue(RunTrace(run_id=f"overflow_run_{i}", workflow="burst_test"))
        assert res is True  # All accepted, none dropped!

    stats = q.stats()
    assert stats["size"] == 5
    assert stats["spilled"] == 7
    assert stats["dropped"] == 0

    overflow_files = list(spool_dir.glob("overflow_*.jsonl"))
    assert len(overflow_files) > 0

    # Recover overflow traces
    recovered = q.spooler.recover_overflow(delete_after_read=True) if q.spooler else []
    assert len(recovered) == 7
    q.clear()


def test_saturation_and_backpressure_detection():
    """Verify is_saturated triggers when queue exceeds high-watermark threshold."""
    q = TraceIngestionQueue(max_size=10, high_watermark=0.80)
    assert q.is_saturated() is False

    for i in range(7):
        q.enqueue(RunTrace(run_id=f"sat_run_{i}", workflow="test"))
    assert q.is_saturated() is False

    # 8th item reaches 80% watermark
    q.enqueue(RunTrace(run_id="sat_run_7", workflow="test"))
    assert q.is_saturated() is True
    assert q.stats()["saturated"] is True


def test_wal_chunk_pruning_after_persistence(spool_dir: Path):
    """WAL chunk files are cleanly unlinked once all contained traces are persisted."""
    q = TraceIngestionQueue(
        max_size=50,
        spool_dir=spool_dir,
        wal_enabled=True,
    )
    assert q.spooler is not None

    q.enqueue(RunTrace(run_id="prune_run_1", workflow="test"))
    q.enqueue(RunTrace(run_id="prune_run_2", workflow="test"))

    wal_files_before = list(spool_dir.glob("wal_*.jsonl"))
    assert len(wal_files_before) == 1
    chunk_name = wal_files_before[0].name

    # Rotate chunk so the first chunk is eligible for deletion
    q.spooler._rotate_chunk_locked()

    # Mark all traces in chunk as persisted
    pruned = q.spooler.mark_persisted(["prune_run_1", "prune_run_2"])
    assert chunk_name in pruned

    wal_files_after = list(spool_dir.glob("wal_*.jsonl"))
    # Only the active empty chunk remains
    assert chunk_name not in [f.name for f in wal_files_after]
    q.clear()


def test_api_otlp_backpressure_429(monkeypatch):
    """OTLP ingestion endpoint returns HTTP 429 with Retry-After when queue is saturated."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Force queue to saturated state
    q = get_global_queue()
    monkeypatch.setattr(q, "is_saturated", lambda: True)

    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace_backpressure_01",
                                "spanId": "span_01",
                                "name": "researcher",
                            }
                        ]
                    }
                ]
            }
        ]
    }

    response = client.post("/v1/traces", json=payload)
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "5"
    assert response.headers.get("X-AgentLens-Queue-Status") == "saturated"
    assert "saturated" in response.json()["detail"]


def test_queue_status_endpoint():
    """GET /api/queue/status returns detailed queue telemetry and saturation status."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/queue/status")
    assert response.status_code == 200
    data = response.json()
    assert "max_size" in data
    assert "size" in data
    assert "enqueued" in data
    assert "dropped" in data
    assert "spilled" in data
    assert "saturated" in data
