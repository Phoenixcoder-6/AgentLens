"""
capture/queue.py — Non-Blocking Async Telemetry Ring Buffer & Ingestion Worker
==============================================================================
Day 48: High-throughput bounded ring buffer queue and background ingestion worker.
Decouples trace capture from disk/database persistence, eliminating I/O latency
from the host agent's critical execution path.
"""

from __future__ import annotations

import atexit
import io
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from schema.models import RunTrace

logger = logging.getLogger("agentlens.capture.queue")


class WALDiskSpooler:
    """
    Local Write-Ahead Logging (WAL) disk spooler for crash-proof telemetry ingestion.
    Writes traces synchronously to append-only JSONL files with fsync durability.
    Automatically rotates chunk files and unlinks chunks once persisted by workers.
    """

    def __init__(
        self,
        spool_dir: str | Path = "data/spool",
        max_chunk_records: int = 500,
        sync_on_write: bool = False,
    ) -> None:
        self.spool_dir = Path(spool_dir)
        self.max_chunk_records = max_chunk_records
        self.sync_on_write = sync_on_write
        self._lock = threading.RLock()
        self._current_file: io.TextIOWrapper | None = None
        self._current_chunk_path: Path | None = None
        self._current_records_count = 0
        self._chunk_seq = 0
        self._active_chunks: dict[str, set[str]] = {}  # filename -> set(run_id)

        try:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(
                "[WALDiskSpooler] Failed to create spool_dir %s: %s", self.spool_dir, exc
            )

    def _rotate_chunk_locked(self) -> None:
        if self._current_file is not None and not self._current_file.closed:
            try:
                self._current_file.flush()
                if self.sync_on_write:
                    os.fsync(self._current_file.fileno())
                self._current_file.close()
            except Exception as exc:
                logger.warning("[WALDiskSpooler] Error closing previous WAL chunk: %s", exc)

        self._chunk_seq += 1
        ts = int(time.time() * 1000)
        chunk_name = f"wal_{ts}_{uuid.uuid4().hex[:6]}_{self._chunk_seq:04d}.jsonl"
        self._current_chunk_path = self.spool_dir / chunk_name
        self._current_records_count = 0
        try:
            self._current_file = open(self._current_chunk_path, "a", encoding="utf-8")
            self._active_chunks[chunk_name] = set()
        except Exception as exc:
            logger.error(
                "[WALDiskSpooler] Failed to open new WAL chunk %s: %s",
                self._current_chunk_path,
                exc,
            )
            self._current_file = None

    def append_trace(self, trace: RunTrace) -> str | None:
        """Write trace to current WAL chunk file. Returns chunk filename or None on failure."""
        payload = trace.model_dump_json()
        run_id = trace.run_id or "unknown"
        with self._lock:
            if self._current_file is None or self._current_records_count >= self.max_chunk_records:
                self._rotate_chunk_locked()

            if self._current_file is None:
                return None

            try:
                self._current_file.write(payload + "\n")
                self._current_file.flush()
                if self.sync_on_write:
                    os.fsync(self._current_file.fileno())
                self._current_records_count += 1
                chunk_name = (
                    self._current_chunk_path.name if self._current_chunk_path else "unknown"
                )
                self._active_chunks.setdefault(chunk_name, set()).add(run_id)
                return chunk_name
            except Exception as exc:
                logger.error("[WALDiskSpooler] Failed to write trace %s to WAL: %s", run_id, exc)
                return None

    def spill_overflow(self, trace: RunTrace) -> Path | None:
        """Spill overflow trace to a dedicated overflow chunk."""
        payload = trace.model_dump_json()
        ts = int(time.time() * 1000)
        path = self.spool_dir / f"overflow_{ts}_{uuid.uuid4().hex[:6]}.jsonl"
        with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")
                    f.flush()
                    if self.sync_on_write:
                        os.fsync(f.fileno())
                return path
            except Exception as exc:
                logger.error("[WALDiskSpooler] Failed to write overflow trace: %s", exc)
                return None

    def mark_persisted(self, run_ids: list[str]) -> list[str]:
        """Prune chunk files whose traces have all been persisted to storage."""
        pruned_chunks: list[str] = []
        with self._lock:
            for r_id in run_ids:
                for _chunk_name, chunk_set in list(self._active_chunks.items()):
                    chunk_set.discard(r_id)

            # Check for chunks that can be safely deleted
            active_name = self._current_chunk_path.name if self._current_chunk_path else ""
            for chunk_name, chunk_set in list(self._active_chunks.items()):
                if len(chunk_set) == 0 and chunk_name != active_name:
                    chunk_path = self.spool_dir / chunk_name
                    try:
                        if chunk_path.exists():
                            chunk_path.unlink()
                        pruned_chunks.append(chunk_name)
                        del self._active_chunks[chunk_name]
                    except Exception as exc:
                        logger.warning(
                            "[WALDiskSpooler] Could not remove chunk %s: %s", chunk_name, exc
                        )
        return pruned_chunks

    def recover_unpersisted_wal(self) -> list[RunTrace]:
        """Scan spool_dir for any unpersisted wal_*.jsonl files and parse RunTrace models."""
        recovered: list[RunTrace] = []
        if not self.spool_dir.exists():
            return recovered

        # Sync and close current file handle before scanning to ensure consistency
        with self._lock:
            if self._current_file is not None and not self._current_file.closed:
                try:
                    self._current_file.flush()
                    if self.sync_on_write:
                        os.fsync(self._current_file.fileno())
                    self._current_file.close()
                except Exception:
                    pass
                self._current_file = None
                self._current_chunk_path = None

        chunk_files = sorted(self.spool_dir.glob("wal_*.jsonl"))
        for cf in chunk_files:
            try:
                with open(cf, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            trace = RunTrace.model_validate_json(line)
                            recovered.append(trace)
                            with self._lock:
                                self._active_chunks.setdefault(cf.name, set()).add(trace.run_id)
            except Exception as exc:
                logger.error("[WALDiskSpooler] Error recovering from %s: %s", cf, exc)

        return recovered

    def recover_overflow(self, delete_after_read: bool = True) -> list[RunTrace]:
        """Scan spool_dir for overflow_*.jsonl files."""
        recovered: list[RunTrace] = []
        if not self.spool_dir.exists():
            return recovered

        overflow_files = sorted(self.spool_dir.glob("overflow_*.jsonl"))
        for of in overflow_files:
            try:
                with open(of, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            trace = RunTrace.model_validate_json(line)
                            recovered.append(trace)
                if delete_after_read:
                    of.unlink(missing_ok=True)
            except Exception as exc:
                logger.error("[WALDiskSpooler] Error recovering overflow %s: %s", of, exc)
        return recovered

    def close(self) -> None:
        """Close open file handles."""
        with self._lock:
            if self._current_file is not None and not self._current_file.closed:
                try:
                    self._current_file.flush()
                    if self.sync_on_write:
                        os.fsync(self._current_file.fileno())
                    self._current_file.close()
                except Exception:
                    pass
                self._current_file = None

    def clear(self) -> None:
        """Clean all spool files (used for testing and resets)."""
        self.close()
        with self._lock:
            self._active_chunks.clear()
            if self.spool_dir.exists():
                for p in self.spool_dir.glob("*.jsonl"):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass


class TraceIngestionQueue:
    """
    Thread-safe, memory-bounded ring buffer for completed RunTrace payloads.
    Provides non-blocking enqueue with optional crash-proof disk WAL spooling,
    overflow disk spillover, and backpressure accounting.
    """

    def __init__(
        self,
        max_size: int = 5000,
        spool_dir: str | Path | None = None,
        wal_enabled: bool = False,
        overflow_spill: bool = False,
        high_watermark: float = 0.85,
    ) -> None:
        self.max_size = max_size
        self._queue: queue.Queue[RunTrace] = queue.Queue(maxsize=max_size)
        self._lock = threading.RLock()
        self._enqueued_count = 0
        self._processed_count = 0
        self._dropped_count = 0
        self._spilled_count = 0
        self._inflight = 0

        self.wal_enabled = wal_enabled
        self.overflow_spill = overflow_spill
        self.high_watermark = high_watermark
        self.spooler: WALDiskSpooler | None = (
            WALDiskSpooler(spool_dir=spool_dir or "data/spool")
            if (spool_dir is not None or wal_enabled or overflow_spill)
            else None
        )

    def _is_saturated_locked(self) -> bool:
        q_len = self._queue.qsize()
        limit = int(self.max_size * self.high_watermark)
        return q_len >= max(1, limit) or self._queue.full()

    def is_saturated(self) -> bool:
        """True if memory queue size exceeds high-watermark threshold or is full."""
        with self._lock:
            return self._is_saturated_locked()

    def enqueue(self, trace: RunTrace, allow_spillover: bool | None = None) -> bool:
        """
        Enqueues a trace without blocking the caller.
        If wal_enabled is True, synchronizes to disk WAL before memory queuing.
        If the queue is full:
            - If overflow_spill is True (or allow_spillover is True), spills to disk and returns True.
            - Otherwise, increments dropped_count and returns False.
        """
        # 1. Write to local disk WAL first if enabled (Day 67)
        if self.wal_enabled and self.spooler is not None:
            self.spooler.append_trace(trace)

        # 2. Try enqueue into memory ring buffer
        try:
            with self._lock:
                self._queue.put_nowait(trace)
                self._enqueued_count += 1
                self._inflight += 1
            return True
        except queue.Full:
            should_spill = self.overflow_spill if allow_spillover is None else bool(allow_spillover)
            if should_spill and self.spooler is not None:
                spill_res = self.spooler.spill_overflow(trace)
                if spill_res is not None:
                    with self._lock:
                        self._spilled_count += 1
                        self._enqueued_count += 1
                        self._inflight += 1
                    logger.info(
                        "[TraceIngestionQueue] Buffer full (%d items). Spilled trace %s to disk: %s",
                        self.max_size,
                        getattr(trace, "run_id", "unknown"),
                        spill_res.name,
                    )
                    return True

            with self._lock:
                self._dropped_count += 1
            logger.warning(
                "[TraceIngestionQueue] Buffer full (%d items). Dropping trace %s. Total dropped: %d",
                self.max_size,
                getattr(trace, "run_id", "unknown"),
                self._dropped_count,
            )
            return False

    def dequeue_batch(self, max_items: int = 50, timeout: float = 0.1) -> list[RunTrace]:
        """
        Pulls a batch of traces from the queue.
        Waits up to `timeout` for the first item, then drains up to `max_items - 1`
        additional items without blocking.
        """
        batch: list[RunTrace] = []
        try:
            first = self._queue.get(timeout=timeout)
            batch.append(first)
            self._queue.task_done()
        except queue.Empty:
            return batch

        # Pull remaining available items without waiting
        while len(batch) < max_items:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
                self._queue.task_done()
            except queue.Empty:
                break

        with self._lock:
            self._processed_count += len(batch)

        return batch

    def mark_done(self, count: int, run_ids: list[str] | None = None) -> None:
        """Signal that `count` dequeued traces have finished persisting and prune WAL chunks."""
        with self._lock:
            self._inflight = max(0, self._inflight - count)

        if self.spooler is not None and run_ids:
            self.spooler.mark_persisted(run_ids)

    def recover_from_disk(self) -> int:
        """Recover unpersisted traces from WAL spool and overflow into the queue."""
        if self.spooler is None:
            return 0
        wal_traces = self.spooler.recover_unpersisted_wal()
        overflow_traces = self.spooler.recover_overflow(delete_after_read=True)
        total_traces = wal_traces + overflow_traces

        recovered_count = 0
        for t in total_traces:
            try:
                self._queue.put_nowait(t)
                with self._lock:
                    self._enqueued_count += 1
                    self._inflight += 1
                recovered_count += 1
            except queue.Full:
                break
        return recovered_count

    def unfinished(self) -> int:
        """Traces enqueued but not yet fully persisted (queued + in-flight)."""
        with self._lock:
            return self._inflight

    def size(self) -> int:
        """Current number of items in the queue."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """True if the queue is empty."""
        return self._queue.empty()

    def stats(self) -> dict[str, Any]:
        """Snapshot of queue metrics."""
        with self._lock:
            return {
                "max_size": self.max_size,
                "size": self._queue.qsize(),
                "enqueued": self._enqueued_count,
                "processed": self._processed_count,
                "dropped": self._dropped_count,
                "spilled": self._spilled_count,
                "saturated": self._is_saturated_locked(),
                "wal_enabled": self.wal_enabled,
                "overflow_spill": self.overflow_spill,
                "spool_dir": str(self.spooler.spool_dir) if self.spooler else None,
            }

    def clear(self) -> None:
        """Drain and discard all items (primarily for tests)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        with self._lock:
            self._enqueued_count = 0
            self._processed_count = 0
            self._dropped_count = 0
            self._spilled_count = 0
            self._inflight = 0
        if self.spooler is not None:
            self.spooler.clear()


class TraceIngestionWorker:
    """
    Background daemon thread that continuously batches and persists traces
    from a TraceIngestionQueue to disk and database storage.
    """

    def __init__(
        self,
        ingestion_queue: TraceIngestionQueue,
        persist_fn: Callable[[RunTrace], Any],
        flush_interval: float = 0.1,
        batch_size: int = 50,
    ):
        self.queue = ingestion_queue
        self.persist_fn = persist_fn
        self.flush_interval = flush_interval
        self.batch_size = batch_size

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """Start the background ingestion worker thread if not already running."""
        with self._lock:
            if self._started and self._thread is not None and self._thread.is_alive():
                return

            # Recover any unpersisted traces from disk WAL before processing
            if self.queue.wal_enabled and self.queue.spooler is not None:
                try:
                    recovered = self.queue.recover_from_disk()
                    if recovered > 0:
                        logger.info(
                            "[TraceIngestionWorker] Recovered %d unpersisted traces from WAL spool.",
                            recovered,
                        )
                except Exception as exc:
                    logger.warning("[TraceIngestionWorker] Spool recovery error: %s", exc)

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="AgentLensTraceIngestionWorker",
                daemon=True,
            )
            self._thread.start()
            self._started = True

    def _run_loop(self) -> None:
        """Main worker loop."""
        while not self._stop_event.is_set():
            batch = self.queue.dequeue_batch(
                max_items=self.batch_size,
                timeout=self.flush_interval,
            )
            if batch:
                self._process_batch(batch)

        # Final drain on stop
        self._drain_remaining()

    def _process_batch(self, batch: list[RunTrace]) -> None:
        persisted_ids: list[str] = []
        try:
            for trace in batch:
                try:
                    self.persist_fn(trace)
                    persisted_ids.append(trace.run_id)
                except Exception as exc:
                    logger.error(
                        "[TraceIngestionWorker] Error persisting trace %s: %s",
                        getattr(trace, "run_id", "unknown"),
                        exc,
                    )
        finally:
            self.queue.mark_done(len(batch), run_ids=persisted_ids)

    def _drain_remaining(self) -> None:
        """Persist all traces left in the queue before exit."""
        while not self.queue.is_empty():
            batch = self.queue.dequeue_batch(max_items=self.batch_size, timeout=0.01)
            if batch:
                self._process_batch(batch)
            else:
                break

    def flush(self, timeout: float = 2.0) -> bool:
        """
        Block until all enqueued traces have been fully persisted (queued AND
        in-flight), up to `timeout` seconds. Returns True if fully flushed.
        """
        deadline = time.perf_counter() + timeout
        while self.queue.unfinished() > 0 and time.perf_counter() < deadline:
            time.sleep(0.005)
        return self.queue.unfinished() == 0

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker thread to stop, drain the queue, and join."""
        with self._lock:
            if not self._started or self._thread is None:
                return
            self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        with self._lock:
            self._started = False


# ── Global Worker & Queue Singletons ─────────────────────────────────────────

_global_queue: TraceIngestionQueue | None = None
_global_worker: TraceIngestionWorker | None = None
_global_lock = threading.RLock()  # Reentrant: get_global_worker() calls get_global_queue()
_async_logging_enabled: bool | None = None


def is_async_logging_enabled() -> bool:
    """
    Check if non-blocking async trace persistence is enabled.
    Priority:
        1. Runtime toggle via set_async_logging_enabled()
        2. Environment variable AGENTLENS_ASYNC_LOGGING ("true" / "1" / "false" / "0")
        3. config.yaml capture.async_logging (default True)
    """
    if _async_logging_enabled is not None:
        return _async_logging_enabled

    env_val = os.environ.get("AGENTLENS_ASYNC_LOGGING")
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")

    try:
        from config.config_loader import get

        cfg_val = get("capture", "async_logging", True)
        return bool(cfg_val)
    except Exception:
        return True


def set_async_logging_enabled(enabled: bool | None) -> None:
    """Override async logging at runtime (pass None to revert to config/env)."""
    global _async_logging_enabled
    _async_logging_enabled = enabled


def get_global_queue() -> TraceIngestionQueue:
    """Retrieve or initialize the global TraceIngestionQueue singleton."""
    global _global_queue
    with _global_lock:
        if _global_queue is None:
            max_size = 5000
            wal_enabled = True
            spool_dir = "data/spool"
            overflow_spill = True
            high_watermark = 0.85
            try:
                from config.config_loader import get

                max_size = int(get("capture", "queue_max_size", 5000) or 5000)
                wal_enabled = bool(get("capture", "wal_enabled", True))
                spool_dir = str(get("capture", "spool_dir", "data/spool") or "data/spool")
                overflow_spill = bool(get("capture", "overflow_spill_enabled", True))
                high_watermark = float(get("capture", "high_watermark", 0.85) or 0.85)
            except Exception:
                pass
            _global_queue = TraceIngestionQueue(
                max_size=max_size,
                spool_dir=spool_dir,
                wal_enabled=wal_enabled,
                overflow_spill=overflow_spill,
                high_watermark=high_watermark,
            )
    return _global_queue


def get_global_worker() -> TraceIngestionWorker:
    """Retrieve or initialize the global TraceIngestionWorker daemon singleton."""
    global _global_worker
    with _global_lock:
        if _global_worker is None:
            from capture.session import CaptureSession

            ingestion_queue = get_global_queue()

            flush_interval = 0.1
            batch_size = 50
            try:
                from config.config_loader import get

                flush_interval = float(get("capture", "batch_flush_interval_sec", 0.1) or 0.1)
                batch_size = int(get("capture", "batch_flush_size", 50) or 50)
            except Exception:
                pass

            _global_worker = TraceIngestionWorker(
                ingestion_queue=ingestion_queue,
                persist_fn=lambda t: CaptureSession._persist_trace(t),
                flush_interval=flush_interval,
                batch_size=batch_size,
            )
            # Register graceful flush + stop at application exit
            atexit.register(_shutdown_global_worker)
            _install_sigterm_handler()
            _global_worker.start()
    return _global_worker


def _shutdown_global_worker(timeout: float = 10.0) -> None:
    """atexit hook: persist every pending trace, then stop the worker thread."""
    worker = _global_worker
    if worker is None:
        return
    if not worker.flush(timeout=timeout):
        logger.warning(
            "[TraceIngestionWorker] Shutdown flush timed out with %d trace(s) unpersisted",
            worker.queue.unfinished(),
        )
    worker.stop(timeout=2.0)


def _install_sigterm_handler() -> None:
    """
    Route SIGTERM through sys.exit() so atexit hooks (and therefore the final
    flush) run. Only installed from the main thread and only if no custom
    handler is already registered. SIGINT already raises KeyboardInterrupt,
    which unwinds normally and triggers atexit.
    """
    import signal
    import sys

    if threading.current_thread() is not threading.main_thread():
        return
    try:
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
            signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    except (ValueError, OSError, AttributeError):
        pass


def flush_global_queue(timeout: float = 2.0) -> bool:
    """Flush all pending traces in the global queue. Returns True if fully flushed."""
    if _global_worker is not None:
        return _global_worker.flush(timeout=timeout)
    return True
