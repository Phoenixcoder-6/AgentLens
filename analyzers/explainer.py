"""
analyzers/explainer.py — LLM Explainer
=======================================
Day 13: Produces a natural-language explanation of the Arbiter's verdict.

CRITICAL ARCHITECTURAL RULE:
    The LLM Explainer receives ONLY the AnalysisBundle.
    It NEVER sees the raw RunTrace, AgentStep data, or any pipeline text
    (research_findings, written_report, etc.).

    This is what makes explanations trustworthy:
        The LLM explains what the deterministic Arbiter established — not
        what it independently guesses from raw data.

What the LLM receives (from AnalysisBundle only):
    - primary_cause      : which failure category was detected
    - priority_level     : P2–P5 (how confident the Arbiter was)
    - grounded           : whether a ground-truth comparison was possible
    - primary_agent      : which agent is responsible
    - rule_matches       : which rules fired and why (rule_id, description)
    - evidence summaries : confidence scores and brief descriptions

What the LLM never receives:
    - Raw research_findings text
    - Raw written_report text
    - AgentStep input/output
    - Any pipeline state

Output (written back onto the same AnalysisBundle):
    - bundle.summary       : 2–3 sentence root cause explanation
    - bundle.suggested_fix : 1–2 sentence actionable recommendation

Hedging rule:
    If bundle.grounded is False (P2–P5), the prompt instructs the LLM to
    use hedged language ("may indicate", "suggests", "possible").
    If bundle.grounded is True (P1), the LLM may use confident language.
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from schema.models import AnalysisBundle, SCHEMA_VERSION
from config.config_loader import get

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Structured output schema for the LLM
# ─────────────────────────────────────────────────────────────────────────────

class ExplanationOutput(BaseModel):
    """Schema for the LLM's structured explanation output."""
    summary: str = Field(
        description=(
            "2–3 sentence root cause explanation. "
            "Use hedged language ('may indicate', 'suggests') when grounded=False."
        )
    )
    suggested_fix: str = Field(
        description=(
            "1–2 sentence actionable recommendation for fixing the detected issue."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLMExplainer
# ─────────────────────────────────────────────────────────────────────────────

class LLMExplainer:
    """
    Generates a natural-language explanation of the Arbiter's verdict.

    Usage:
        explainer = LLMExplainer()
        bundle = explainer.explain(bundle)
        print(bundle.summary)        # root cause explanation
        print(bundle.suggested_fix)  # actionable recommendation

    The AnalysisBundle is mutated in place (summary + suggested_fix populated)
    and also returned for chaining.

    On LLM error: falls back to a rule-based explanation — never raises.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are an AI observability analyst. Your job is to explain the output
        of a deterministic analysis system that monitors multi-agent AI pipelines.

        You will receive a structured analysis verdict. Explain:
        1. What was detected and why it matters (2-3 sentences)
        2. A concrete, actionable fix (1-2 sentences)

        IMPORTANT RULES:
        - You only explain what the analysis detected. You do NOT re-analyze.
        - If grounded=False, use hedged language: "may indicate", "suggests", "possible".
        - If grounded=True, you may use confident language.
        - Be specific: mention the rule ID, agent name, and failure category.
        - Keep the summary under 80 words.
        - Keep the suggested_fix under 40 words.
    """)

    def __init__(self) -> None:
        self._llm = self._build_llm()

    def _build_llm(self) -> ChatGroq:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Copy .env.example to .env and set your key."
            )
        return ChatGroq(
            model=get("llm", "model"),
            temperature=0.0,          # deterministic output
            max_tokens=512,           # explanations are short
            api_key=api_key,
        )

    def explain(self, bundle: AnalysisBundle) -> AnalysisBundle:
        """
        Generate explanation for the Arbiter verdict and write it onto the bundle.

        Args:
            bundle: The AnalysisBundle from the Arbiter. Only this object is
                    passed to the LLM — never raw trace data.

        Returns:
            The same bundle with summary and suggested_fix populated.
            On LLM failure, falls back to a rule-based explanation.
        """
        prompt = self._build_prompt(bundle)
        try:
            structured_llm = self._llm.with_structured_output(ExplanationOutput)
            result: ExplanationOutput = structured_llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            bundle.summary       = result.summary
            bundle.suggested_fix = result.suggested_fix
        except Exception as exc:
            # Never crash the pipeline — fall back to rule-based explanation
            bundle.summary       = self._fallback_summary(bundle)
            bundle.suggested_fix = self._fallback_fix(bundle)

        return bundle

    # ── Prompt construction (only uses AnalysisBundle fields) ─────────────────

    def _build_prompt(self, bundle: AnalysisBundle) -> str:
        """
        Serialize only AnalysisBundle fields into the LLM prompt.

        CRITICAL: This method must NEVER receive or embed raw trace data.
                  Only bundle.* fields are allowed here.
        """
        # Evidence summary — only descriptions and confidence scores
        evidence_lines = []
        for i, ev in enumerate(bundle.evidence, 1):
            rule_id = ev.rule_match.rule_id if ev.rule_match else "no_rule"
            evidence_lines.append(
                f"  [{i}] rule={rule_id}  "
                f"confidence={ev.confidence:.0%}  "
                f"description={ev.description[:120]}"
            )

        # Rule match summary
        rule_lines = []
        for rm in bundle.rule_matches:
            rule_lines.append(
                f"  - rule_id={rm.rule_id}  "
                f"category={rm.category.value}  "
                f"severity={rm.severity.value}  "
                f"agent={rm.agent or 'unknown'}"
            )

        evidence_block = "\n".join(evidence_lines) or "  (none)"
        rule_block     = "\n".join(rule_lines)     or "  (none)"

        return textwrap.dedent(f"""\
            ANALYSIS VERDICT
            ════════════════
            run_id         : {bundle.run_id}
            primary_cause  : {bundle.primary_cause.value}
            priority_level : {bundle.priority_level.value}
            primary_agent  : {bundle.primary_agent or 'unknown'}
            grounded       : {bundle.grounded}
            evidence_count : {len(bundle.evidence)}

            EVIDENCE RECORDS
            ────────────────
            {evidence_block}

            FIRED RULES
            ────────────
            {rule_block}

            HEDGING INSTRUCTION
            ───────────────────
            grounded={bundle.grounded} → {"Use CONFIDENT language." if bundle.grounded else "Use HEDGED language (may indicate, suggests, possible)."}
        """)

    # ── Fallback (no LLM call) ─────────────────────────────────────────────────

    def _fallback_summary(self, bundle: AnalysisBundle) -> str:
        """Rule-based summary when LLM is unavailable."""
        agent    = bundle.primary_agent or "an agent"
        cause    = bundle.primary_cause.value
        priority = bundle.priority_level.value
        hedge    = "may indicate" if not bundle.grounded else "indicates"
        rule_id  = (
            bundle.rule_matches[0].rule_id
            if bundle.rule_matches else "unknown rule"
        )
        return (
            f"[Fallback — LLM unavailable] "
            f"The {priority} analysis {hedge} a {cause} issue attributed to {agent}. "
            f"Rule '{rule_id}' fired during the Arbiter evaluation."
        )

    def _fallback_fix(self, bundle: AnalysisBundle) -> str:
        """Rule-based fix suggestion when LLM is unavailable."""
        cause = bundle.primary_cause.value
        agent = bundle.primary_agent or "the agent"
        fixes = {
            "reasoning":     f"Review {agent}'s output for hallucinated facts not present in the research.",
            "workflow":      f"Check the handoff from the previous agent to {agent} for dropped context.",
            "execution":     f"Investigate {agent} for tool errors, timeouts, or missing outputs.",
            "verification":  f"Review the verifier's approval criteria — it may have passed a flawed output.",
            "unknown":       "Inspect the pipeline trace manually — no specific rule matched.",
        }
        return fixes.get(cause, "Inspect the pipeline trace for anomalies.")
