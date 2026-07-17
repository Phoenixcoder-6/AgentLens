"""
normalizer/normalizer.py — Raw Event → Canonical Schema
========================================================
Day 7: Converts raw captured AgentStep events into clean, fully-typed
canonical records with schema_version stamped on every output.

The Normalizer sits between the Capture layer and all downstream analyzers:

    Capture (@trace_step)
        ↓  raw AgentStep with JSON strings, untyped dicts
    Normalizer
        ↓  NormalizedStep with typed fields, schema_version stamped
    Analyzers (Diff Engine, Rule Engine, Metrics Analyzer, ...)

Every NormalizedStep produced here is guaranteed to:
    - Have schema_version == SCHEMA_VERSION
    - Have input_state and output_state as proper dicts (not strings)
    - Have timestamp as a timezone-aware datetime object
    - Have latency_ms as a float
    - Have status as a StepStatus enum value
    - Have all nested values JSON-safe (via safe_loads + to_serializable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from schema.models import (
    AgentStep,
    HandoffState,
    RunTrace,
    SCHEMA_VERSION,
    StepStatus,
    TokenUsage,
)
from normalizer.serializer import safe_loads, to_serializable


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedStep — the canonical output of the Normalizer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedStep:
    """
    A fully-typed, canonical representation of a single agent invocation.

    Produced by Normalizer.normalize_step(). Every field is typed —
    no raw strings, no untyped dicts, no ambiguous values.

    schema_version is stamped on every record so schema migrations
    are fully traceable across stored records.
    """
    # Identity
    schema_version: str
    run_id: str
    step: int
    agent: str

    # Typed states (parsed from raw JSON strings)
    input_state: dict[str, Any]
    filtered_state: dict[str, Any]
    output_state: dict[str, Any]

    # Performance
    latency_ms: float
    tokens: TokenUsage

    # Status
    status: StepStatus
    error: Optional[str]

    # Timing — always UTC-aware datetime
    timestamp: datetime

    # Diff summary (stored in AgentStep.prompt field by tracer)
    diff_summary: str

    # Raw strings preserved for downstream text analysis
    raw_input: str
    raw_output: str


@dataclass
class NormalizedRun:
    """
    A fully-typed, canonical representation of a complete pipeline run.

    Produced by Normalizer.normalize_run(). Contains all NormalizedSteps
    in execution order.
    """
    schema_version: str
    run_id: str
    workflow: str
    timestamp: datetime
    status: StepStatus
    total_latency_ms: float
    total_tokens: int
    steps: list[NormalizedStep] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """
    Converts raw AgentStep / RunTrace objects into NormalizedStep / NormalizedRun.

    Usage:
        normalizer = Normalizer()

        # Normalize a single step
        norm_step = normalizer.normalize_step(agent_step)

        # Normalize an entire run
        norm_run = normalizer.normalize_run(run_trace)
    """

    def normalize_step(self, step: AgentStep) -> NormalizedStep:
        """
        Convert one raw AgentStep into a NormalizedStep.

        Transformations applied:
            input / output strings  → parsed dicts via safe_loads
            handoff states          → parsed dicts via safe_loads
            timestamp string        → timezone-aware datetime
            status string           → StepStatus enum
            all nested values       → JSON-safe via to_serializable
            schema_version          → stamped as SCHEMA_VERSION
        """
        # ── Parse state dicts from raw JSON strings ───────────────────────
        input_state    = self._parse_state(step.handoff.input_state or step.input)
        filtered_state = self._parse_state(step.handoff.filtered_state)
        output_state   = self._parse_state(step.handoff.output_state or step.output)

        # ── Normalize timestamp → UTC-aware datetime ──────────────────────
        timestamp = self._normalize_timestamp(step.timestamp)

        # ── Normalize status → StepStatus enum ───────────────────────────
        status = self._normalize_status(step.status)

        # ── Ensure all state values are JSON-safe ─────────────────────────
        input_state    = to_serializable(input_state)
        filtered_state = to_serializable(filtered_state)
        output_state   = to_serializable(output_state)

        return NormalizedStep(
            schema_version = SCHEMA_VERSION,          # ← stamped here
            run_id         = step.run_id,
            step           = step.step,
            agent          = step.agent,
            input_state    = input_state    if isinstance(input_state, dict)    else {},
            filtered_state = filtered_state if isinstance(filtered_state, dict) else {},
            output_state   = output_state   if isinstance(output_state, dict)   else {},
            latency_ms     = float(step.latency_ms),
            tokens         = step.tokens,
            status         = status,
            error          = step.error,
            timestamp      = timestamp,
            diff_summary   = step.prompt or "",       # diff stored in prompt field
            raw_input      = step.input  or "",
            raw_output     = step.output or "",
        )

    def normalize_run(self, run: RunTrace) -> NormalizedRun:
        """
        Convert a complete RunTrace into a NormalizedRun.
        All steps are normalized in order.
        """
        timestamp = self._normalize_timestamp(run.timestamp)
        status    = self._normalize_status(run.status)

        normalized_steps = [self.normalize_step(s) for s in run.steps]

        return NormalizedRun(
            schema_version   = SCHEMA_VERSION,        # ← stamped here
            run_id           = run.run_id,
            workflow         = run.workflow,
            timestamp        = timestamp,
            status           = status,
            total_latency_ms = float(run.total_latency_ms),
            total_tokens     = int(run.total_tokens),
            steps            = normalized_steps,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_state(value: Any) -> dict[str, Any]:
        """
        Parse a raw value into a dict.
        Handles: dict (pass-through), JSON string, empty string, None.
        """
        if isinstance(value, dict):
            return value
        if not value:
            return {}
        parsed = safe_loads(value)
        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _normalize_timestamp(value: Any) -> datetime:
        """
        Convert any timestamp representation to a UTC-aware datetime.
        Handles: datetime (with or without tzinfo), ISO string, None.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        if isinstance(value, str) and value:
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass

        # Fallback: use current UTC time
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_status(value: Any) -> StepStatus:
        """
        Convert any status representation to a StepStatus enum.
        Handles: StepStatus enum, string, None.
        """
        if isinstance(value, StepStatus):
            return value
        if isinstance(value, str):
            try:
                return StepStatus(value.upper())
            except ValueError:
                pass
        return StepStatus.SUCCESS
