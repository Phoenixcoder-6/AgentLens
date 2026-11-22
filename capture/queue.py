"""
capture/queue.py — Non-Blocking Async Telemetry Ring Buffer & Ingestion Worker
==============================================================================
Day 48: High-throughput bounded ring buffer queue and background ingestion worker.
Decouples trace capture from disk/database persistence, eliminating I/O latency
from the host agent's critical execution path.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from schema.models import RunTrace

logger = logging.getLogger("agentlens.capture.queue")


class TraceIngestionQueue:
    """
    Thread-safe, memory-bounded ring buffer for completed RunTrace payloads.
    Provides non-blocking enqueue with dropped-span accounting on overflow.
    """

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._queue: queue.Queue[RunTrace] = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._enqueued_count = 0
        self._processed_count = 0
        self._dropped_count = 0
        self._inflight = 0

    def enqueue(self, trace: RunTrace) -> bool:
        """
        Enqueues a trace without blocking the caller.
        If the queue is full, increments dropped_count and returns False.
        """
        try:
            with self._lock:
                self._queue.put_nowait(trace)
                self._enqueued_count += 1
                self._inflight += 1
            return True
        except queue.Full:
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

    def mark_done(self, count: int) -> None:
        """Signal that `count` dequeued traces have finished persisting."""
        with self._lock:
            self._inflight = max(0, self._inflight - count)

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

    def stats(self) -> dict[str, int]:
        """Snapshot of queue metrics."""
        with self._lock:
            return {
                "max_size": self.max_size,
                "size": self._queue.qsize(),
                "enqueued": self._enqueued_count,
                "processed": self._processed_count,
                "dropped": self._dropped_count,
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
            self._inflight = 0


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
        try:
            for trace in batch:
                try:
                    self.persist_fn(trace)
                except Exception as exc:
                    logger.error(
                        "[TraceIngestionWorker] Error persisting trace %s: %s",
                        getattr(trace, "run_id", "unknown"),
                        exc,
                    )
        finally:
            self.queue.mark_done(len(batch))

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
            try:
                from config.config_loader import get

                max_size = int(get("capture", "queue_max_size", 5000) or 5000)
            except Exception:
                pass
            _global_queue = TraceIngestionQueue(max_size=max_size)
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
