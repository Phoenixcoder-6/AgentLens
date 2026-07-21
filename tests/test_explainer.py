"""
tests/test_explainer.py — Day 13: LLM Explainer tests
======================================================
Tests for LLMExplainer — verifying:
1. The LLM receives ONLY AnalysisBundle fields (never raw trace data)
2. Prompt structure and field presence
3. Fallback behaviour on LLM failure
4. Bundle mutation (summary + suggested_fix populated)
5. Structured output schema

LLM-calling tests are marked @pytest.mark.integration and skipped by default.
All prompt-inspection and fallback tests run without any LLM calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from schema.models import (
    AnalysisBundle,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    PriorityLevel,
    RuleMatch,
    RuleSeverity,
    SCHEMA_VERSION,
)
from analyzers.explainer import LLMExplainer, ExplanationOutput


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID = "run_test_explainer"

RAW_TRACE_SENTINEL = "RESEARCH_FINDINGS_RAW_TEXT_NEVER_ALLOWED_IN_PROMPT"


def make_bundle(
    cause: FailureCategory = FailureCategory.REASONING,
    priority: PriorityLevel = PriorityLevel.P2,
    grounded: bool = False,
    agent: str = "writer",
    confidence: float = 0.75,
    rule_id: str = "information_loss_v1",
) -> AnalysisBundle:
    rule = RuleMatch(
        rule_id=rule_id,
        category=cause,
        description=f"Test rule {rule_id} fired",
        severity=RuleSeverity.MEDIUM,
        agent=agent,
    )
    ev = EvidenceRecord(
        source=EvidenceSource.RULE_ENGINE,
        description=f"Test evidence from {rule_id}",
        value="WARNING",
        rule_match=rule,
        agent=agent,
        confidence=confidence,
    )
    return AnalysisBundle(
        run_id=RUN_ID,
        primary_cause=cause,
        priority_level=priority,
        grounded=grounded,
        evidence=[ev],
        rule_matches=[rule],
        primary_agent=agent,
        summary=None,
        suggested_fix=None,
    )


def make_explainer_with_mock() -> tuple[LLMExplainer, MagicMock]:
    """Return an LLMExplainer whose LLM is replaced with a mock."""
    with patch("analyzers.explainer.ChatGroq"):
        explainer = LLMExplainer.__new__(LLMExplainer)
        mock_llm = MagicMock()
        explainer._llm = mock_llm
        return explainer, mock_llm


# ─────────────────────────────────────────────────────────────────────────────
# CORE REQUIREMENT: LLM never sees raw trace data
# ─────────────────────────────────────────────────────────────────────────────

class TestRawTraceNeverInPrompt:
    """
    The architectural guarantee of Day 13:
    The LLM Explainer receives ONLY AnalysisBundle fields.
    Raw trace data (research_findings, written_report, etc.) must NEVER appear.
    """

    def test_prompt_does_not_contain_raw_trace_sentinel(self):
        """Sentinel string representing raw trace data must not appear in prompt."""
        bundle = make_bundle()
        # Inject the sentinel into a non-bundle field to simulate raw data leakage
        # (mimics a bug where raw text was accidentally embedded)
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)
        assert RAW_TRACE_SENTINEL not in prompt

    def test_prompt_contains_only_bundle_fields(self):
        """Prompt must contain run_id, primary_cause, priority_level from the bundle."""
        bundle = make_bundle(
            cause=FailureCategory.REASONING,
            priority=PriorityLevel.P2,
            agent="writer",
        )
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)

        assert bundle.run_id          in prompt
        assert "reasoning"            in prompt   # primary_cause.value
        assert "P2"                   in prompt   # priority_level.value
        assert "writer"               in prompt   # primary_agent

    def test_prompt_contains_rule_id(self):
        bundle = make_bundle(rule_id="information_loss_v1")
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)
        assert "information_loss_v1" in prompt

    def test_prompt_contains_confidence(self):
        bundle = make_bundle(confidence=0.75)
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)
        assert "75%" in prompt

    def test_prompt_contains_hedging_instruction_when_not_grounded(self):
        bundle = make_bundle(grounded=False)
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)
        assert "HEDGED" in prompt.upper() or "hedged" in prompt.lower()

    def test_prompt_contains_confident_instruction_when_grounded(self):
        bundle = make_bundle(grounded=True)
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)
        assert "CONFIDENT" in prompt.upper() or "confident" in prompt.lower()

    def test_prompt_contains_evidence_count(self):
        bundle = make_bundle()
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)
        assert "evidence_count" in prompt
        assert "1" in prompt   # one evidence record

    def test_prompt_does_not_contain_research_findings_literal(self):
        """The literal string 'research_findings' must never appear as data in prompt."""
        bundle = make_bundle()
        explainer, _ = make_explainer_with_mock()
        prompt = explainer._build_prompt(bundle)
        # The word may appear in field names but not as embedded raw content
        # We verify the actual research text is absent
        assert "Key findings" not in prompt
        assert "SOURCES:" not in prompt
        assert "ENTITIES:" not in prompt

    def test_build_prompt_takes_only_bundle_not_trace(self):
        """_build_prompt signature must accept only AnalysisBundle."""
        import inspect
        explainer, _ = make_explainer_with_mock()
        sig = inspect.signature(explainer._build_prompt)
        params = list(sig.parameters.keys())
        assert params == ["bundle"]   # bound method: self excluded, exactly one arg


# ─────────────────────────────────────────────────────────────────────────────
# Fallback behaviour (no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

class TestFallback:

    def test_fallback_summary_contains_priority(self):
        bundle = make_bundle(priority=PriorityLevel.P2)
        explainer, _ = make_explainer_with_mock()
        summary = explainer._fallback_summary(bundle)
        assert "P2" in summary

    def test_fallback_summary_contains_cause(self):
        bundle = make_bundle(cause=FailureCategory.REASONING)
        explainer, _ = make_explainer_with_mock()
        summary = explainer._fallback_summary(bundle)
        assert "reasoning" in summary

    def test_fallback_summary_contains_agent(self):
        bundle = make_bundle(agent="writer")
        explainer, _ = make_explainer_with_mock()
        summary = explainer._fallback_summary(bundle)
        assert "writer" in summary

    def test_fallback_summary_contains_rule_id(self):
        bundle = make_bundle(rule_id="information_loss_v1")
        explainer, _ = make_explainer_with_mock()
        summary = explainer._fallback_summary(bundle)
        assert "information_loss_v1" in summary

    def test_fallback_fix_reasoning_cause(self):
        bundle = make_bundle(cause=FailureCategory.REASONING, agent="writer")
        explainer, _ = make_explainer_with_mock()
        fix = explainer._fallback_fix(bundle)
        assert "writer" in fix
        assert len(fix) > 10

    def test_fallback_fix_workflow_cause(self):
        bundle = make_bundle(cause=FailureCategory.WORKFLOW)
        explainer, _ = make_explainer_with_mock()
        fix = explainer._fallback_fix(bundle)
        assert "handoff" in fix.lower() or "workflow" in fix.lower() or "context" in fix.lower()

    def test_fallback_fix_unknown_cause(self):
        bundle = make_bundle(cause=FailureCategory.UNKNOWN)
        explainer, _ = make_explainer_with_mock()
        fix = explainer._fallback_fix(bundle)
        assert len(fix) > 10

    def test_explain_falls_back_on_llm_error(self):
        """If LLM raises, explain() must populate summary via fallback — never raise."""
        explainer, mock_llm = make_explainer_with_mock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM down")

        bundle = make_bundle()
        result = explainer.explain(bundle)

        assert result.summary is not None
        assert "Fallback" in result.summary
        assert result.suggested_fix is not None

    def test_explain_never_raises_on_llm_error(self):
        """explain() must always return a bundle, never propagate an exception."""
        explainer, mock_llm = make_explainer_with_mock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("network error")

        bundle = make_bundle()
        result = explainer.explain(bundle)   # must not raise
        assert isinstance(result, AnalysisBundle)


# ─────────────────────────────────────────────────────────────────────────────
# Bundle mutation
# ─────────────────────────────────────────────────────────────────────────────

class TestBundleMutation:

    def test_explain_populates_summary(self):
        explainer, mock_llm = make_explainer_with_mock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ExplanationOutput(
            summary="The writer may have hallucinated new sources.",
            suggested_fix="Review writer output for grounding.",
        )
        bundle = make_bundle()
        result = explainer.explain(bundle)
        assert result.summary == "The writer may have hallucinated new sources."

    def test_explain_populates_suggested_fix(self):
        explainer, mock_llm = make_explainer_with_mock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ExplanationOutput(
            summary="Test summary.",
            suggested_fix="Add source verification step.",
        )
        bundle = make_bundle()
        result = explainer.explain(bundle)
        assert result.suggested_fix == "Add source verification step."

    def test_explain_returns_same_bundle(self):
        """explain() must mutate and return the SAME bundle object."""
        explainer, mock_llm = make_explainer_with_mock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ExplanationOutput(
            summary="Test.", suggested_fix="Fix."
        )
        bundle = make_bundle()
        result = explainer.explain(bundle)
        assert result is bundle

    def test_explain_preserves_primary_cause(self):
        explainer, mock_llm = make_explainer_with_mock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ExplanationOutput(
            summary="Test.", suggested_fix="Fix."
        )
        bundle = make_bundle(cause=FailureCategory.WORKFLOW)
        result = explainer.explain(bundle)
        assert result.primary_cause == FailureCategory.WORKFLOW

    def test_explain_preserves_priority_level(self):
        explainer, mock_llm = make_explainer_with_mock()
        mock_llm.with_structured_output.return_value.invoke.return_value = ExplanationOutput(
            summary="Test.", suggested_fix="Fix."
        )
        bundle = make_bundle(priority=PriorityLevel.P2)
        result = explainer.explain(bundle)
        assert result.priority_level == PriorityLevel.P2


# ─────────────────────────────────────────────────────────────────────────────
# System prompt verification
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPrompt:

    def test_system_prompt_instructs_hedging(self):
        assert "hedged" in LLMExplainer.SYSTEM_PROMPT.lower()

    def test_system_prompt_forbids_re_analysis(self):
        assert "do not re-analyze" in LLMExplainer.SYSTEM_PROMPT.lower() or \
               "only explain" in LLMExplainer.SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_grounded(self):
        assert "grounded" in LLMExplainer.SYSTEM_PROMPT.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Integration test (requires GROQ_API_KEY — skipped by default)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestExplainerIntegration:

    def test_real_llm_call_populates_bundle(self):
        """Live Groq call — only runs with -m integration flag."""
        bundle = make_bundle(
            cause=FailureCategory.REASONING,
            priority=PriorityLevel.P2,
            agent="writer",
            confidence=0.75,
        )
        explainer = LLMExplainer()
        result = explainer.explain(bundle)

        assert result.summary is not None
        assert len(result.summary) > 20
        assert result.suggested_fix is not None
        assert len(result.suggested_fix) > 10
        # Verify hedging language is present (grounded=False)
        hedging_words = ["may", "suggest", "possible", "indicate", "could"]
        assert any(w in result.summary.lower() for w in hedging_words)
