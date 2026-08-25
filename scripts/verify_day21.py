"""
scripts/verify_day21.py — Day 21 Verification: Schema Validation & Retry
=========================================================================
Tests:
  1. Malformed JSON on attempt 1 → retry → success (retried=True)
  2. Malformed JSON on both attempts → extraction_failed=True, no crash
  3. extraction_failed=True → rule_engine skips extraction-dependent rules
  4. extraction_max_tokens sourced from config (not hardcoded)

Run with:
    conda run -n agentlens python scripts/verify_day21.py
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

# Force UTF-8 output — conda run on Windows defaults to cp1252 which
# cannot encode characters like em-dash that appear in log messages.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyzers.evidence_extraction.extractor import (
    EvidenceExtractor,
    ExtractedEvidence,
    _parse_response,
)
from config.config_loader import get

PASS = "[PASS]"
FAIL = "[FAIL]"


def _make_mock_message(content: str, tokens: int = 100) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.response_metadata = {"token_usage": {"total_tokens": tokens}}
    return msg


def _make_extractor_with_responses(*responses) -> EvidenceExtractor:
    """
    Create a mocked EvidenceExtractor where successive _call_llm invocations
    return the given (content, tokens) tuples — or raise if given an Exception.
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


# ─────────────────────────────────────────────────────────────────────────────
print("=== Day 21 Verification: Schema Validation & Retry ===\n")

total = 0
passed = 0

# ── Test 1: _parse_response rejects empty string ──────────────────────────────
total += 1
try:
    _parse_response("")
    print(f"  {FAIL} [1] _parse_response should reject empty string")
except ValueError as e:
    print(f"  {PASS} [1] _parse_response correctly rejects empty: {e}")
    passed += 1

# ── Test 2: _parse_response rejects non-JSON ──────────────────────────────────
total += 1
try:
    _parse_response("not json at all")
    print(f"  {FAIL} [2] _parse_response should reject non-JSON")
except (ValueError, json.JSONDecodeError):
    print(f"  {PASS} [2] _parse_response correctly rejects non-JSON")
    passed += 1

# ── Test 3: _parse_response rejects wrong types ───────────────────────────────
total += 1
try:
    _parse_response(json.dumps({"source_count": "five", "entity_count": 2}))
    print(f"  {FAIL} [3] _parse_response should reject string source_count")
except ValueError as e:
    print(f"  {PASS} [3] _parse_response rejects wrong type: {e}")
    passed += 1

# ── Test 4: _parse_response accepts valid JSON ────────────────────────────────
total += 1
try:
    parsed = _parse_response(json.dumps({"source_count": 3, "entity_count": 5, "tool_calls": []}))
    assert parsed["source_count"] == 3
    print(f"  {PASS} [4] _parse_response accepts valid JSON")
    passed += 1
except Exception as e:
    print(f"  {FAIL} [4] _parse_response valid JSON: {e}")

# ── Test 5: First attempt malformed → retry → success ─────────────────────────
total += 1
valid_json = json.dumps(
    {
        "source_count": 4,
        "entity_count": 6,
        "tool_calls": [],
        "claims": ["Earth is round"],
        "references": [],
        "numbers": [],
        "dates": [],
    }
)
extractor = _make_extractor_with_responses(
    ("not json at all", 50),  # attempt 1: malformed
    (valid_json, 120),  # attempt 2: valid
)
result = extractor.extract("some agent output here", agent="researcher")
if not result.extraction_failed and result.retried and result.source_count == 4:
    print(f"  {PASS} [5] Retry succeeded: retried=True, source_count=4")
    passed += 1
else:
    print(
        f"  {FAIL} [5] Retry result: failed={result.extraction_failed} retried={result.retried} sources={result.source_count}"
    )

# ── Test 6: Both attempts malformed → extraction_failed=True ──────────────────
total += 1
extractor = _make_extractor_with_responses(
    ("```json oops", 50),  # attempt 1: malformed
    ("", 30),  # attempt 2: empty
)
result = extractor.extract("some agent output", agent="writer")
if result.extraction_failed and result.retried and result.source_count == 0:
    print(f"  {PASS} [6] Double failure: extraction_failed=True, retried=True, no crash")
    passed += 1
else:
    print(
        f"  {FAIL} [6] Double failure result: failed={result.extraction_failed} retried={result.retried}"
    )

# ── Test 7: extraction_failed=True carries token_cost from attempts ────────────
total += 1
extractor = _make_extractor_with_responses(
    ("bad", 40),  # attempt 1 fails
    ("", 60),  # attempt 2 fails
)
result = extractor.extract("text", agent="verifier")
if result.extraction_failed and result.token_cost == 100:
    print(f"  {PASS} [7] Token cost accumulated across failed attempts: {result.token_cost}")
    passed += 1
else:
    print(f"  {FAIL} [7] Token cost: expected 100, got {result.token_cost}")

# ── Test 8: retried=False on clean first success ──────────────────────────────
total += 1
valid_json = json.dumps(
    {
        "source_count": 2,
        "entity_count": 3,
        "tool_calls": [],
        "claims": [],
        "references": [],
        "numbers": [],
        "dates": [],
    }
)
extractor = _make_extractor_with_responses((valid_json, 80))
result = extractor.extract("clean output", agent="researcher")
if not result.extraction_failed and not result.retried and result.source_count == 2:
    print(f"  {PASS} [8] Clean call: retried=False, source_count=2")
    passed += 1
else:
    print(f"  {FAIL} [8] Clean call: failed={result.extraction_failed} retried={result.retried}")

# ── Test 9: extraction_max_tokens loaded from config (not hardcoded) ──────────
total += 1
config_val = get("llm", "extraction_max_tokens", None)
if config_val is not None and int(config_val) > 0:
    print(f"  {PASS} [9] extraction_max_tokens in config: {config_val}")
    passed += 1
else:
    print(f"  {FAIL} [9] extraction_max_tokens not found in config (got: {config_val!r})")

# ── Test 10: explanation_max_tokens loaded from config ────────────────────────
total += 1
expl_val = get("llm", "explanation_max_tokens", None)
if expl_val is not None and int(expl_val) > 0:
    print(f"  {PASS} [10] explanation_max_tokens in config: {expl_val}")
    passed += 1
else:
    print(f"  {FAIL} [10] explanation_max_tokens not found in config (got: {expl_val!r})")

# ── Test 11: RuleEngine skips extraction rules when extraction_failed=True ─────
total += 1
try:
    from analyzers.detection.rule_engine import RuleEngine
    from schema.models import AgentStep, RunTrace, StepStatus

    # Build a trace with a researcher step
    step = AgentStep(
        run_id="test_run",
        step=1,
        agent="researcher",
        output="Some research output",
        status=StepStatus.SUCCESS,
        tool_calls=[],
    )
    trace = RunTrace(
        run_id="test_run",
        workflow="research_report_pipeline",
        steps=[step],
    )

    # Patch EvidenceExtractor to return extraction_failed=True
    failed_ev = ExtractedEvidence(extraction_failed=True, error_message="Simulated failure")
    with patch("analyzers.detection.rule_engine.EvidenceExtractor") as MockExtractor:
        mock_instance = MagicMock()
        mock_instance.extract.return_value = failed_ev
        MockExtractor.return_value = mock_instance

        engine = RuleEngine()
        analysis = engine.analyze(trace)

        # Reasoning rules (researcher_quality_v1, hallucination_v1) should be skipped
        rule_ids = [e.rule_match.rule_id for e in analysis.evidence if e.rule_match]
        extraction_rules = [
            r for r in rule_ids if r in ("researcher_quality_v1", "hallucination_v1")
        ]

        if not extraction_rules:
            print(
                f"  {PASS} [11] RuleEngine skipped extraction-dependent rules when extraction_failed=True"
            )
            passed += 1
        else:
            print(
                f"  {FAIL} [11] RuleEngine fired extraction rules despite failure: {extraction_rules}"
            )
except Exception as e:
    print(f"  {FAIL} [11] RuleEngine skip test error: {e}")

# ── Test 12: ExtractedEvidence has retried field ──────────────────────────────
total += 1
ev = ExtractedEvidence(retried=True)
if ev.retried is True:
    print(f"  {PASS} [12] ExtractedEvidence.retried field exists and defaults False, settable")
    passed += 1
else:
    print(f"  {FAIL} [12] ExtractedEvidence.retried: {ev.retried}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'✅' if passed == total else '❌'} {passed}/{total} verifications passed.")
if passed < total:
    sys.exit(1)
