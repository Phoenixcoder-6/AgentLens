# tests/test_interfaces.py
"""
Tests for app/interfaces.py — the 4 Protocol interfaces and custom exceptions.

Covers:
  - All interfaces and return types import cleanly
  - Custom exceptions are proper Exception subclasses
  - AnalysisResult and TraceEvent instantiate with defaults
  - Protocol structure is correct (runtime_checkable)
"""

import pytest
from app.interfaces import (
    Analyzer,
    CaptureProvider,
    StorageProvider,
    LLMProvider,
    AnalysisResult,
    TraceEvent,
    LLMProviderError,
    ExtractionError,
    StorageError,
    CaptureError,
)


# ── Import completeness ───────────────────────────────────────────────────────

def test_all_interfaces_importable():
    assert Analyzer is not None
    assert CaptureProvider is not None
    assert StorageProvider is not None
    assert LLMProvider is not None


def test_all_return_types_importable():
    assert AnalysisResult is not None
    assert TraceEvent is not None


def test_all_exceptions_importable():
    assert LLMProviderError is not None
    assert ExtractionError is not None
    assert StorageError is not None
    assert CaptureError is not None


# ── Exception hierarchy ───────────────────────────────────────────────────────

def test_llm_provider_error_is_exception():
    assert issubclass(LLMProviderError, Exception)


def test_extraction_error_is_exception():
    assert issubclass(ExtractionError, Exception)


def test_storage_error_is_exception():
    assert issubclass(StorageError, Exception)


def test_capture_error_is_exception():
    assert issubclass(CaptureError, Exception)


def test_exceptions_can_be_raised_and_caught():
    with pytest.raises(LLMProviderError):
        raise LLMProviderError("Groq API failed after 1 retry")

    with pytest.raises(ExtractionError):
        raise ExtractionError("Malformed JSON in extraction output")


# ── AnalysisResult defaults ───────────────────────────────────────────────────

def test_analysis_result_default_instantiation():
    result = AnalysisResult()
    assert result.evidence == []
    assert result.analyzer_id == ""
    assert result.skipped is False
    assert result.skip_reason == ""


def test_analysis_result_skipped_flag():
    result = AnalysisResult(skipped=True, skip_reason="LLM extraction failed")
    assert result.skipped is True
    assert result.skip_reason == "LLM extraction failed"


def test_analysis_result_with_evidence(sample_evidence_record):
    result = AnalysisResult(
        evidence=[sample_evidence_record],
        analyzer_id="rule_engine",
    )
    assert len(result.evidence) == 1
    assert result.analyzer_id == "rule_engine"


# ── TraceEvent defaults ───────────────────────────────────────────────────────

def test_trace_event_instantiation():
    from datetime import datetime, timezone
    event = TraceEvent(
        agent="researcher",
        raw_input="Summarize 10 sources",
        raw_output="Tesla was founded...",
        latency_ms=1400.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    assert event.agent == "researcher"
    assert event.tool_calls == []
    assert event.metadata == {}


# ── Protocol runtime checkability ─────────────────────────────────────────────

def test_analyzer_is_runtime_checkable():
    """Analyzer is a runtime_checkable Protocol — isinstance() works on it."""
    from typing import runtime_checkable, Protocol

    # A class that implements Analyzer correctly
    class FakeAnalyzer:
        @property
        def analyzer_id(self) -> str:
            return "fake"

        def analyze(self, trace):
            return AnalysisResult()

    fa = FakeAnalyzer()
    assert isinstance(fa, Analyzer)
