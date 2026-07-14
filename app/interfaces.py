"""
app/interfaces.py — AgentLens Architecture Contracts
=====================================================
Four Protocol interfaces that define the shape every component must implement.
No logic lives here — only method signatures and docstrings.

These interfaces are what allow every component to be:
  - Independently testable (swap real impl for a fake in tests)
  - Independently swappable (change Groq → local model in one place)
  - Consistently structured (Arbiter calls all Analyzers the same way)

Implementations:
  Analyzer         → EvidenceExtractor, DiffEngine, MetricsAnalyzer, DetectionLayer
  CaptureProvider  → capture/tracer.py  (@trace_step decorator)
  StorageProvider  → storage/sqlite_store.py
  LLMProvider      → app/groq_provider.py  (or any local model adapter)

Rule: if a class claims to implement one of these interfaces, it MUST define
every method listed here — no partial implementations allowed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from schema import (
    AgentStep,
    AnalysisBundle,
    EvidenceRecord,
    RunTrace,
)


# ─────────────────────────────────────────────────────────────────────────────
# Return types used by interfaces
# (lightweight dataclasses — not full Pydantic models, kept simple here)
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    """
    The standard return type for every Analyzer.analyze() call.

    All analyzers produce this shape — the Arbiter collects a list of
    AnalysisResults and resolves them into a single AnalysisBundle.

    Fields:
        evidence    : list of EvidenceRecord objects produced by this analyzer
        analyzer_id : identifier for which analyzer produced this result
        skipped     : True if the analyzer could not run (e.g. extraction failed)
        skip_reason : why it was skipped, for dashboard display
    """
    evidence: list[EvidenceRecord] = field(default_factory=list)
    analyzer_id: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class TraceEvent:
    """
    A single raw capture event — produced by CaptureProvider, consumed by Normalizer.

    This is intentionally un-typed (raw dicts) because the Normalizer is
    responsible for converting it into a typed AgentStep.
    """
    agent: str
    raw_input: Any
    raw_output: Any
    latency_ms: float
    timestamp: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Interface 1: Analyzer
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Analyzer(Protocol):
    """
    The shared interface for every analysis component in AgentLens.

    Implemented by:
        EvidenceExtractor       → extracts structured facts from agent outputs
        DiffEngine              → compares two runs and finds divergence
        MetricsAnalyzer         → computes latency, token, and anomaly statistics
        DetectionLayer          → runs rule engine + validators

    Contract:
        - analyze() receives the full RunTrace for one run
        - analyze() returns an AnalysisResult containing EvidenceRecords
        - analyze() MUST NOT raise exceptions — return skipped=True instead
        - analyze() MUST NOT modify the RunTrace it receives (read-only)

    The Arbiter calls analyze() on each registered Analyzer and aggregates
    the results. It does not know or care which concrete class it's calling.
    """

    @property
    def analyzer_id(self) -> str:
        """
        Unique identifier for this analyzer, e.g. 'evidence_extraction'.
        Used in logging, AnalysisResult attribution, and the dashboard.
        """
        ...

    def analyze(self, trace: RunTrace) -> AnalysisResult:
        """
        Run analysis on a complete RunTrace and return structured evidence.

        Args:
            trace: The full RunTrace for one workflow run (read-only).

        Returns:
            AnalysisResult with a list of EvidenceRecords.
            If analysis cannot complete, return AnalysisResult(skipped=True, skip_reason=...).
            Never raise — graceful degradation is a hard requirement.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Interface 2: CaptureProvider
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class CaptureProvider(Protocol):
    """
    The interface for the Capture layer — records raw agent events.

    Implemented by:
        capture/tracer.py   → @trace_step decorator on LangGraph agent nodes

    Contract:
        - capture_step() records one agent invocation and returns a TraceEvent
        - capture_step() MUST NOT raise or alter the agent pipeline if it fails
          (a capture failure must be silent — it must never crash the workflow)
        - capture_step() is called by the decorator, not by analyzers

    The Normalizer consumes TraceEvents and converts them to typed AgentSteps.

    Key design note: CaptureProvider is intentionally separate from Normalizer.
    This decoupling means an OpenTelemetry adapter can replace CaptureProvider
    later without changing any downstream layer.
    """

    def capture_step(
        self,
        agent: str,
        raw_input: Any,
        raw_output: Any,
        latency_ms: float,
        tool_calls: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """
        Capture a single agent invocation and return a raw TraceEvent.

        Args:
            agent       : Agent name, e.g. 'researcher'
            raw_input   : Raw input passed to the agent (any type)
            raw_output  : Raw output produced by the agent (any type)
            latency_ms  : Wall-clock time for this invocation in milliseconds
            tool_calls  : List of tool invocations made during this step
            metadata    : Any additional key-value metadata to attach

        Returns:
            TraceEvent with raw (un-normalized) event data.
        """
        ...

    def start_run(self, workflow: str) -> str:
        """
        Start a new run and return its run_id.
        All subsequent capture_step() calls will be associated with this run_id.

        Args:
            workflow: The workflow name, e.g. 'research_report_pipeline'

        Returns:
            run_id: Unique run identifier, e.g. 'run_8f21ac'
        """
        ...

    def end_run(self, run_id: str, status: str = "SUCCESS") -> None:
        """
        Mark a run as complete. Called after the last agent step finishes.

        Args:
            run_id : The run_id returned by start_run()
            status : 'SUCCESS' or 'FAILURE'
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Interface 3: StorageProvider
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class StorageProvider(Protocol):
    """
    The interface for persistent storage of runs, steps, and analysis results.

    Implemented by:
        storage/sqlite_store.py  → SQLite + JSON blob storage

    Contract:
        - save_run() persists a complete RunTrace (SQLite index + JSON blob)
        - load_run() retrieves a RunTrace by run_id
        - list_runs() returns a summary list for the Run Explorer dashboard page
        - save_analysis() persists an AnalysisBundle linked to a run_id
        - load_analysis() retrieves an AnalysisBundle by run_id

    Storage layout (from config.yaml):
        SQLite   → data/agentlens.db  (Runs, Steps, Analysis, Metrics tables)
        JSON     → data/traces/<run_id>.json  (full trace payloads)

    The JSON blob keeps traces portable and human-readable.
    SQLite handles fast indexing and queries for the dashboard.
    """

    def save_run(self, run: RunTrace) -> str:
        """
        Persist a complete RunTrace. Returns the trace_path (JSON blob location).

        Args:
            run: The complete RunTrace to persist.

        Returns:
            trace_path: Relative path to the JSON blob, e.g. 'data/traces/run_8f21ac.json'
        """
        ...

    def load_run(self, run_id: str) -> RunTrace:
        """
        Load a RunTrace by its run_id.

        Args:
            run_id: The run identifier, e.g. 'run_8f21ac'

        Returns:
            The complete RunTrace including all AgentSteps.

        Raises:
            KeyError: If run_id does not exist in storage.
        """
        ...

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Return a summary list of recent runs for the Run Explorer dashboard page.

        Args:
            limit: Maximum number of runs to return (from config.yaml dashboard.default_run_limit)

        Returns:
            List of dicts with keys: run_id, workflow, timestamp, status, grounded
            (lightweight summary only — not full RunTrace objects)
        """
        ...

    def save_analysis(self, bundle: AnalysisBundle) -> None:
        """
        Persist an AnalysisBundle linked to its run_id.

        Args:
            bundle: The complete AnalysisBundle from the Arbiter.
        """
        ...

    def load_analysis(self, run_id: str) -> AnalysisBundle:
        """
        Load an AnalysisBundle by run_id.

        Args:
            run_id: The run identifier.

        Returns:
            The AnalysisBundle for this run.

        Raises:
            KeyError: If no analysis exists for this run_id.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Interface 4: LLMProvider
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """
    The interface for LLM calls — used by Evidence Extraction and LLM Explainer.

    Implemented by:
        app/groq_provider.py   → Groq API (llama-3.3-70b-versatile)
        app/local_provider.py  → local model adapter (future)

    Contract:
        - complete() takes a prompt string and returns a response string
        - complete() handles retries internally (config.yaml: llm.retry_attempts)
        - complete() MUST NOT be called with raw trace data — only with
          the structured prompt built from an AnalysisBundle or EvidenceBundle
        - If the call fails after retries, raise LLMProviderError (not a silent fail)

    Critical architectural note:
        The LLM is NEVER used for detection or classification.
        It is ONLY used for:
          1. Evidence Extraction  → schema-constrained fact extraction from agent output
          2. LLM Explainer        → generating a human-readable explanation of the
                                    AnalysisBundle that the Arbiter already produced

        Swapping Groq for a local model = implement this interface, update config.yaml.
        Nothing else changes.
    """

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Make a single LLM call and return the response text.

        Args:
            prompt       : The user-turn prompt to send
            system_prompt: Optional system prompt override
            max_tokens   : Override config.yaml llm.max_tokens if provided
            temperature  : Override config.yaml llm.temperature if provided

        Returns:
            The LLM response as a plain string.

        Raises:
            LLMProviderError: If the call fails after all retry attempts.
        """
        ...

    def complete_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """
        Make an LLM call with a JSON schema constraint and return parsed output.
        Used by Evidence Extraction to enforce structured fact extraction.

        Args:
            prompt          : The user-turn prompt
            response_schema : JSON schema the response must conform to
            system_prompt   : Optional system prompt override

        Returns:
            Parsed dict matching the response_schema.

        Raises:
            LLMProviderError    : If the call fails after all retry attempts.
            ExtractionError     : If response does not conform to schema after retry.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class LLMProviderError(Exception):
    """Raised when an LLMProvider call fails after all retry attempts."""
    pass


class ExtractionError(Exception):
    """Raised when structured extraction output is malformed after retry."""
    pass


class StorageError(Exception):
    """Raised when a StorageProvider operation fails."""
    pass


class CaptureError(Exception):
    """
    Raised internally within the CaptureProvider — never propagated to the
    agent pipeline. Capture failures are logged and silently skipped.
    """
    pass
