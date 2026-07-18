"""
storage/writer.py — Storage Writer
====================================
Day 8: Takes a NormalizedRun and writes it into all four SQLite tables.

Writes in this order:
    1. runs    — the top-level run record + full JSON blob
    2. steps   — one row per agent step
    3. metrics — latency + token metrics per step (queryable without parsing JSON)

analysis table is left empty here — it will be populated by future analyzers
(Diff Engine, Rule Engine, etc.) when they produce verdicts.
"""

from __future__ import annotations

import json
from datetime import timezone
from typing import Optional

from schema.models import SCHEMA_VERSION
from normalizer.normalizer import NormalizedRun, NormalizedStep
from normalizer.serializer import to_serializable
from storage.db import DatabaseManager


class StorageWriter:
    """
    Writes a NormalizedRun into the SQLite database.

    Usage:
        db = DatabaseManager()
        db.initialize()
        writer = StorageWriter(db)
        writer.write_run(normalized_run, trace_json=raw_json_string)
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def write_run(
        self,
        run: NormalizedRun,
        trace_json: Optional[str] = None,
        trace_path: Optional[str] = None,
    ) -> None:
        """
        Write a complete NormalizedRun to the database.

        Args:
            run:        The normalized run to persist.
            trace_json: Full JSON string of the raw RunTrace (optional blob).
            trace_path: Path to the JSON file on disk (optional).
        """
        ts = _iso(run.timestamp)
        status_str = run.status.value if hasattr(run.status, "value") else str(run.status)

        # 1. Insert the run row
        self.db.insert_run(
            run_id           = run.run_id,
            workflow         = run.workflow,
            timestamp        = ts,
            status           = status_str,
            total_latency_ms = run.total_latency_ms,
            total_tokens     = run.total_tokens,
            schema_version   = SCHEMA_VERSION,
            trace_path       = trace_path,
            trace_json       = trace_json,
        )

        # 2. Insert each step + its metrics
        for step in run.steps:
            self._write_step(run.run_id, step)

    def _write_step(self, run_id: str, step: NormalizedStep) -> None:
        """Write one step row and its per-step metrics."""
        ts = _iso(step.timestamp)
        status_str = step.status.value if hasattr(step.status, "value") else str(step.status)

        self.db.insert_step(
            run_id            = run_id,
            step              = step.step,
            agent             = step.agent,
            status            = status_str,
            latency_ms        = step.latency_ms,
            tokens_prompt     = step.tokens.prompt,
            tokens_completion = step.tokens.completion,
            tokens_total      = step.tokens.total,
            diff_summary      = step.diff_summary or "",
            error             = step.error,
            timestamp         = ts,
            schema_version    = SCHEMA_VERSION,
        )

        # Write latency metric
        self.db.insert_metric(
            run_id       = run_id,
            step         = step.step,
            agent        = step.agent,
            metric_name  = "latency_ms",
            metric_value = step.latency_ms,
            metric_unit  = "ms",
            timestamp    = ts,
            schema_version = SCHEMA_VERSION,
        )

        # Write token metrics (only if non-zero)
        if step.tokens.total > 0:
            for name, value in [
                ("tokens_prompt",     step.tokens.prompt),
                ("tokens_completion", step.tokens.completion),
                ("tokens_total",      step.tokens.total),
            ]:
                self.db.insert_metric(
                    run_id         = run_id,
                    step           = step.step,
                    agent          = step.agent,
                    metric_name    = name,
                    metric_value   = float(value),
                    metric_unit    = "tokens",
                    timestamp      = ts,
                    schema_version = SCHEMA_VERSION,
                )


def _iso(value) -> str:
    """Convert any timestamp to ISO string, ensuring UTC-aware."""
    if hasattr(value, "isoformat"):
        if hasattr(value, "tzinfo") and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)
