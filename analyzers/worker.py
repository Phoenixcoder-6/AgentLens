"""
analyzers/worker.py — Background Analysis Job Worker & Queue Manager
=====================================================================
AgentLens v2.1 (Day 69): Asynchronous analysis worker pool.
Decouples heavy multi-agent analysis (DAG alignment, NLI cross-encoding,
causal blast radius, prompt auto-diff, and LLM explanation) from the
synchronous HTTP request-response cycle.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

import dashboard.state as _state

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisJob(BaseModel):
    job_id: str
    run_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    execution_time_ms: float | None = None
    priority_level: str | None = None
    primary_cause: str | None = None
    primary_agent: str | None = None
    grounded: bool | None = None
    verdict_reason: str | None = None
    blast_radius: dict[str, Any] | None = None
    remediation_patch: dict[str, Any] | None = None
    error: str | None = None
    webhook_url: str | None = None
    webhook_status: str | None = None
    poll_url: str | None = None


class AnalysisJobManager:
    """
    In-process background worker pool managing asynchronous analysis jobs.
    Maintains thread-safe job state, queues tasks, and dispatches optional webhooks.
    """

    def __init__(
        self,
        max_workers: int = 4,
        job_ttl_seconds: int = 86400,
        webhook_timeout_sec: float = 5.0,
    ) -> None:
        self.max_workers = max(1, max_workers)
        self.job_ttl_seconds = max(60, job_ttl_seconds)
        self.webhook_timeout_sec = max(0.5, webhook_timeout_sec)
        self._lock = threading.RLock()
        self._jobs: dict[str, AnalysisJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="AgentLensAnalysisWorker",
        )
        self._closed = False

    def submit_job(
        self,
        run_id: str,
        webhook_url: str | None = None,
    ) -> AnalysisJob:
        """Queue a new analysis job for background execution."""
        with self._lock:
            if self._closed:
                raise RuntimeError("AnalysisJobManager has been stopped")

            job_id = f"job_{uuid.uuid4().hex[:12]}"
            now = datetime.now(UTC).isoformat()
            job = AnalysisJob(
                job_id=job_id,
                run_id=run_id,
                status=JobStatus.QUEUED,
                created_at=now,
                webhook_url=webhook_url,
                poll_url=f"/api/jobs/{job_id}",
            )
            self._jobs[job_id] = job
            future = self._executor.submit(self._run_job, job_id)
            self._futures[job_id] = future
            self._prune_expired_jobs_locked()
            return job

    def get_job(self, job_id: str) -> AnalysisJob | None:
        """Fetch a job by job_id."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, run_id: str | None = None, limit: int = 50) -> list[AnalysisJob]:
        """List jobs, optionally filtered by run_id, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
            if run_id:
                jobs = [j for j in jobs if j.run_id == run_id]
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    def _run_job(self, job_id: str) -> None:
        """Worker thread entry point."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(UTC).isoformat()

        start_time = time.perf_counter()
        error_msg: str | None = None
        state: Any = None

        try:
            # Clear cache so re-analysis is completely fresh
            _state._analysis_cache.pop(job.run_id, None)
            state = _state.run_full_analysis(job.run_id)
            if state.error or not state.bundle:
                error_msg = state.error or "Analysis produced no verdict bundle"
        except Exception as exc:
            logger.exception(
                "[AnalysisJobManager] Job %s for run %s failed: %s", job_id, job.run_id, exc
            )
            error_msg = str(exc)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        now_completed = datetime.now(UTC).isoformat()

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.completed_at = now_completed
            job.execution_time_ms = round(elapsed_ms, 2)

            if error_msg or not state or not state.bundle:
                job.status = JobStatus.FAILED
                job.error = error_msg
            else:
                bundle = state.bundle
                job.status = JobStatus.COMPLETED
                job.priority_level = (
                    bundle.priority_level.value
                    if hasattr(bundle.priority_level, "value")
                    else str(bundle.priority_level)
                )
                job.primary_cause = (
                    bundle.primary_cause.value
                    if hasattr(bundle.primary_cause, "value")
                    else str(bundle.primary_cause)
                )
                job.primary_agent = bundle.primary_agent
                job.grounded = bundle.grounded
                job.verdict_reason = bundle.summary

                # Causal blast radius & remediation patch
                br_dict = bundle.blast_radius.model_dump() if bundle.blast_radius else None
                if not br_dict:
                    try:
                        br_dict = _state.get_db().get_blast_radius(job.run_id)
                    except Exception:
                        pass
                job.blast_radius = br_dict

                rem_dict = (
                    bundle.remediation_patch.model_dump() if bundle.remediation_patch else None
                )
                if not rem_dict:
                    try:
                        rem_dict = _state.get_db().get_remediation_patch(job.run_id)
                    except Exception:
                        pass
                job.remediation_patch = rem_dict

        if job.webhook_url:
            self._dispatch_webhook(job)

    def _dispatch_webhook(self, job: AnalysisJob) -> None:
        """Send HTTP POST callback with job result payload."""
        if not job.webhook_url:
            return
        try:
            payload = json.dumps(job.model_dump()).encode("utf-8")
            req = urllib.request.Request(
                job.webhook_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "AgentLens-Webhook/2.1",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.webhook_timeout_sec) as resp:
                code = getattr(resp, "status", getattr(resp, "code", 200))
                with self._lock:
                    job.webhook_status = "delivered" if 200 <= code < 300 else "failed"
        except Exception as exc:
            logger.warning(
                "[AnalysisJobManager] Webhook delivery failed for %s: %s", job.webhook_url, exc
            )
            with self._lock:
                job.webhook_status = "failed"

    def wait_for_job(self, job_id: str, timeout: float = 10.0) -> AnalysisJob | None:
        """Wait synchronously for job completion or failure."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    return job
            time.sleep(0.02)
        with self._lock:
            return self._jobs.get(job_id)

    def _prune_expired_jobs_locked(self) -> None:
        """Prune jobs that have exceeded TTL if count is high."""
        if len(self._jobs) <= 500:
            return
        now_ts = time.time()
        for jid, job in list(self._jobs.items()):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED) and job.completed_at:
                try:
                    dt = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00"))
                    if (now_ts - dt.timestamp()) > self.job_ttl_seconds:
                        del self._jobs[jid]
                        self._futures.pop(jid, None)
                except Exception:
                    pass

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background thread pool executor."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait)

    def clear(self) -> None:
        """Clear all stored job state (primarily for testing resets)."""
        with self._lock:
            self._jobs.clear()
            self._futures.clear()


_job_manager: AnalysisJobManager | None = None
_job_manager_lock = threading.Lock()


def get_job_manager() -> AnalysisJobManager:
    """Retrieve or initialize the global AnalysisJobManager singleton."""
    global _job_manager
    with _job_manager_lock:
        if _job_manager is None:
            max_workers = 4
            job_ttl_hours = 24
            webhook_timeout = 5.0
            try:
                from config.config_loader import get

                max_workers = int(get("analysis_worker", "max_workers", 4) or 4)
                job_ttl_hours = int(get("analysis_worker", "job_ttl_hours", 24) or 24)
                webhook_timeout = float(get("analysis_worker", "webhook_timeout_sec", 5.0) or 5.0)
            except Exception:
                pass
            _job_manager = AnalysisJobManager(
                max_workers=max_workers,
                job_ttl_seconds=job_ttl_hours * 3600,
                webhook_timeout_sec=webhook_timeout,
            )
    return _job_manager
