"""
tests/test_extractor_retry.py — Day 21: Schema Validation & Retry tests
=========================================================================
Tests the retry-once behaviour of EvidenceExtractor, _parse_response
validation, the retried/token_cost fields, and config-driven max_tokens.

All tests are pure unit tests — no API key required.
"""

from __future__ import annotations

import json

import pytest

from analyzers.evidence_extraction.extractor import (
    EvidenceExtractor,
    ExtractedEvidence,
    _parse_response,
)
from config.config_loader import get

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_extractor_with_responses(*responses) -> EvidenceExtractor:
    """
    Create a mocked EvidenceExtractor where successive _call_llm calls
    return the provided (content_str, tokens) tuples — or raise an Exception.
    """
    extractor = EvidenceExtractor.__new__(EvidenceExtractor)
    call_iter = iter(responses)

    def mock_call_llm(system, user_content):
        resp = next(call_iter)
        if isinstance(resp, Exception):
            raise resp
        content, tokens = resp
        return content, tokens

    extractor._call_llm = mock_call_llm  # type: ignore[method-assign]
    return extractor


def _valid_json(**overrides) -> str:
    base = {
        "source_count": 3,
        "entity_count": 5,
        "tool_calls": [],
        "claims": [],
        "references": [],
        "numbers": [],
        "dates": [],
    }
    base.update(overrides)
    return json.dumps(base)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_response validation
# ─────────────────────────────────────────────────────────────────────────────


class TestParseResponse:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_response("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_response("   \n  ")

    def test_non_json_raises(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_response("not json at all")

    def test_source_count_string_raises(self):
        with pytest.raises(ValueError, match="source_count"):
            _parse_response(json.dumps({"source_count": "five", "entity_count": 2}))

    def test_entity_count_float_raises(self):
        with pytest.raises(ValueError, match="entity_count"):
            _parse_response(json.dumps({"source_count": 2, "entity_count": 3.5}))

    def test_valid_json_returns_dict(self):
        parsed = _parse_response(_valid_json(source_count=2, entity_count=4))
        assert parsed["source_count"] == 2
        assert parsed["entity_count"] == 4

    def test_missing_optional_fields_ok(self):
        """claims/references/numbers/dates are optional — missing is fine."""
        parsed = _parse_response(json.dumps({"source_count": 1, "entity_count": 2}))
        assert parsed["source_count"] == 1

    def test_zero_counts_valid(self):
        parsed = _parse_response(_valid_json(source_count=0, entity_count=0))
        assert parsed["source_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Retry behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryBehaviour:
    def test_clean_success_no_retry(self):
        extractor = _make_extractor_with_responses((_valid_json(source_count=2), 80))
        result = extractor.extract("clean output")
        assert not result.extraction_failed
        assert not result.retried
        assert result.source_count == 2
        assert result.token_cost == 80

    def test_malformed_first_retry_succeeds(self):
        valid = _valid_json(source_count=4, entity_count=6)
        extractor = _make_extractor_with_responses(
            ("not json", 50),  # attempt 1 fails
            (valid, 120),  # attempt 2 succeeds
        )
        result = extractor.extract("some output")
        assert not result.extraction_failed
        assert result.retried is True
        assert result.source_count == 4
        assert result.token_cost == 170  # 50 + 120

    def test_empty_first_retry_succeeds(self):
        valid = _valid_json(source_count=1)
        extractor = _make_extractor_with_responses(
            ("", 30),  # attempt 1: empty
            (valid, 90),  # attempt 2: valid
        )
        result = extractor.extract("some output")
        assert not result.extraction_failed
        assert result.retried is True

    def test_both_attempts_fail_returns_fallback(self):
        extractor = _make_extractor_with_responses(
            ("bad json", 40),
            ("", 60),
        )
        result = extractor.extract("some output")
        assert result.extraction_failed is True
        assert result.retried is True
        assert result.source_count == 0
        assert result.token_cost == 100  # 40 + 60

    def test_exception_on_first_retry_succeeds(self):
        valid = _valid_json(source_count=3)
        extractor = _make_extractor_with_responses(
            Exception("network error"),  # attempt 1 raises
            (valid, 100),  # attempt 2 succeeds
        )
        result = extractor.extract("some output")
        assert not result.extraction_failed
        assert result.retried is True
        assert result.source_count == 3

    def test_exception_on_both_returns_fallback(self):
        extractor = _make_extractor_with_responses(
            Exception("timeout"),
            Exception("timeout again"),
        )
        result = extractor.extract("some output")
        assert result.extraction_failed is True
        assert "Retry failed" in result.error_message

    def test_empty_input_skips_llm_call(self):
        """Empty input should be caught before any LLM call."""
        extractor = EvidenceExtractor.__new__(EvidenceExtractor)
        # _call_llm should never be called
        extractor._call_llm = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("LLM called for empty input")
        )
        result = extractor.extract("")
        assert result.extraction_failed is True
        assert "Empty output" in result.error_message

    def test_retry_preserves_all_day20_fields(self):
        """After a retry, all Day-20 fields should still be populated."""
        valid = _valid_json(
            source_count=2,
            claims=["Claim A"],
            dates=["July 1969"],
            numbers=["382 kg"],
            references=["NASA (2023)"],
        )
        extractor = _make_extractor_with_responses(
            ("bad", 20),
            (valid, 100),
        )
        result = extractor.extract("text")
        assert result.retried
        assert result.claims == ["Claim A"]
        assert result.dates == ["July 1969"]
        assert result.numbers == ["382 kg"]
        assert result.references == ["NASA (2023)"]


# ─────────────────────────────────────────────────────────────────────────────
# Config-driven max_tokens
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigDrivenMaxTokens:
    def test_extraction_max_tokens_in_config(self):
        val = get("llm", "extraction_max_tokens", None)
        assert val is not None, "extraction_max_tokens must exist in config.yaml"
        assert int(val) > 0

    def test_explanation_max_tokens_in_config(self):
        val = get("llm", "explanation_max_tokens", None)
        assert val is not None, "explanation_max_tokens must exist in config.yaml"
        assert int(val) > 0

    def test_legacy_max_tokens_removed(self):
        """The old flat max_tokens key should no longer be the only token config."""
        extr = get("llm", "extraction_max_tokens", None)
        expl = get("llm", "explanation_max_tokens", None)
        assert extr is not None and expl is not None, (
            "Both extraction_max_tokens and explanation_max_tokens must be present"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedEvidence retried field
# ─────────────────────────────────────────────────────────────────────────────


class TestRetiredField:
    def test_retried_defaults_false(self):
        ev = ExtractedEvidence()
        assert ev.retried is False

    def test_retried_settable(self):
        ev = ExtractedEvidence(retried=True)
        assert ev.retried is True

    def test_retried_round_trips(self):
        ev = ExtractedEvidence(retried=True, token_cost=200)
        restored = ExtractedEvidence.model_validate(ev.model_dump())
        assert restored.retried is True
        assert restored.token_cost == 200
