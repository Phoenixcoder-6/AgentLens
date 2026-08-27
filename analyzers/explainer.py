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

import json
import os
import textwrap
import time
from typing import Any, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr

from config.config_loader import get
from config.logging_config import get_logger
from schema.models import AnalysisBundle
from storage.llm_cache import LLMCache

log = get_logger("explainer")

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Structured output schema for the LLM
# ─────────────────────────────────────────────────────────────────────────────


class ExplanationOutput(BaseModel):
    """Schema for the LLM's structured explanation output."""

    summary: str = Field(
        description=(
            "A detailed root cause analysis in 4-6 sentences covering: "
            "(1) what was detected and which rule fired, "
            "(2) what the specific evidence numbers mean (e.g. source count changes), "
            "(3) why this matters for output quality, "
            "(4) which agent is responsible and how the failure likely occurred. "
            "Use hedged language ('may indicate', 'suggests') when grounded=False."
        )
    )
    suggested_fix: str = Field(
        description=(
            "2-3 concrete, actionable steps to fix the detected issue. "
            "Be specific about what to check, what to change in the prompt or pipeline, "
            "and how to verify the fix worked."
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
        You are an expert AI observability analyst reviewing multi-agent pipeline failures.
        Your job is to write a detailed, insightful explanation of a deterministic analysis verdict.

        Structure your response as follows:
        1. summary: A thorough root cause analysis (4-6 sentences) that covers:
           - WHAT was detected: name the specific rule that fired and the evidence numbers
           - WHY this is a problem: explain what information gain/loss means for output quality
           - HOW it likely happened: describe the mechanism (e.g. hallucination, context drop)
           - WHO is responsible: name the specific agent and their role in the failure
           - IMPACT: what downstream effect this could have on the final output

        2. suggested_fix: 2-3 specific, actionable steps to diagnose and fix the issue.
           Reference concrete things to check (prompts, context windows, temperature settings).

        IMPORTANT RULES:
        - You only explain what the analysis detected. You do NOT re-analyze.
        - If grounded=False, use hedged language throughout: "may indicate", "suggests", "could mean".
        - If grounded=True, you may use confident language.
        - Be specific: always mention the rule ID, the exact evidence numbers, and the agent name.
        - Write for a senior ML engineer who wants actionable insight, not a general audience.
        - Do NOT add bullet points or headers inside the summary — write in flowing prose.
    """)

    @property
    def _cache(self) -> LLMCache:
        if not hasattr(self, "_cache_instance"):
            self._cache_instance = LLMCache()
        return self._cache_instance

    @_cache.setter
    def _cache(self, value: Any) -> None:
        self._cache_instance = value

    def __init__(self) -> None:
        self._cache = LLMCache()
        self._primary_model_name = str(get("llm", "model", "openai/gpt-oss-120b"))
        self._fallback_model_name = get("llm", "fallback_model")
        self._llm = self._build_llm(self._primary_model_name)
        self._fallback_llm = (
            self._build_llm(str(self._fallback_model_name))
            if self._fallback_model_name
            else None
        )

    def _build_llm(self, model_name: str) -> ChatGroq:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise OSError("GROQ_API_KEY not found. Copy .env.example to .env and set your key.")
        return ChatGroq(
            model=model_name,
            temperature=0.0,
            max_tokens=int(get("llm", "explanation_max_tokens", 1024)),
            api_key=SecretStr(api_key),
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

        primary_model = getattr(self, "_primary_model_name", "openai/gpt-oss-120b")
        fallback_model = getattr(self, "_fallback_model_name", None)
        fallback_llm = getattr(self, "_fallback_llm", None)

        is_llm_mocked = (
            hasattr(self._llm, "return_value")
            or hasattr(self._llm, "side_effect")
            or hasattr(self._llm, "assert_called")
        )

        # 1. Check cache hit (skip if LLM is mocked in unit tests)
        if not is_llm_mocked:
            cached = self._cache.get(
                self.SYSTEM_PROMPT, prompt, model=primary_model, temperature=0.0
            )
            if cached is not None:
                try:
                    cached_data = json.loads(cached[0])
                    bundle.summary = cached_data.get("summary", "")
                    bundle.suggested_fix = cached_data.get("suggested_fix", "")
                    return bundle
                except Exception:
                    pass

        try:
            start_t = time.time()
            target_model = primary_model
            try:
                structured_llm = self._llm.with_structured_output(ExplanationOutput)
                result = cast(
                    ExplanationOutput,
                    structured_llm.invoke(
                        [
                            SystemMessage(content=self.SYSTEM_PROMPT),
                            HumanMessage(content=prompt),
                        ]
                    ),
                )
            except Exception as primary_exc:
                if fallback_llm and fallback_model:
                    log.warning(
                        f"Primary explainer model '{primary_model}' failed: {primary_exc}. "
                        f"Retrying with fallback model '{fallback_model}'."
                    )
                    target_model = str(fallback_model)
                    structured_fallback = fallback_llm.with_structured_output(
                        ExplanationOutput
                    )
                    result = cast(
                        ExplanationOutput,
                        structured_fallback.invoke(
                            [
                                SystemMessage(content=self.SYSTEM_PROMPT),
                                HumanMessage(content=prompt),
                            ]
                        ),
                    )
                else:
                    raise primary_exc

            latency_ms = (time.time() - start_t) * 1000

            log.info(
                "Explanation LLM call completed",
                extra={
                    "extra_fields": {
                        "model": target_model,
                        "latency_ms": round(latency_ms, 2),
                        "run_id": bundle.run_id,
                        "cost_estimate": 0.0,
                    }
                },
            )

            bundle.summary = result.summary
            bundle.suggested_fix = result.suggested_fix

            # Store in cache as JSON string
            cache_payload = json.dumps(
                {"summary": result.summary, "suggested_fix": result.suggested_fix}
            )
            self._cache.set(
                self.SYSTEM_PROMPT,
                prompt,
                model=target_model,
                response_text=cache_payload,
                token_cost=0,
                temperature=0.0,
            )
        except Exception as exc:
            log.warning(
                "Explanation LLM call failed, falling back to rule-based",
                extra={"extra_fields": {"error": str(exc), "run_id": bundle.run_id}},
            )
            # Never crash the pipeline — fall back to rule-based explanation
            bundle.summary = self._fallback_summary(bundle)
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
        rule_block = "\n".join(rule_lines) or "  (none)"

        return textwrap.dedent(f"""\
            ANALYSIS VERDICT
            ════════════════
            run_id         : {bundle.run_id}
            primary_cause  : {bundle.primary_cause.value}
            priority_level : {bundle.priority_level.value}
            primary_agent  : {bundle.primary_agent or "unknown"}
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
        agent = bundle.primary_agent or "an agent"
        cause = bundle.primary_cause.value
        priority = bundle.priority_level.value
        hedge = "may indicate" if not bundle.grounded else "indicates"
        rule_id = bundle.rule_matches[0].rule_id if bundle.rule_matches else "unknown rule"
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
            "reasoning": f"Review {agent}'s output for hallucinated facts not present in the research.",
            "workflow": f"Check the handoff from the previous agent to {agent} for dropped context.",
            "execution": f"Investigate {agent} for tool errors, timeouts, or missing outputs.",
            "verification": "Review the verifier's approval criteria — it may have passed a flawed output.",
            "unknown": "Inspect the pipeline trace manually — no specific rule matched.",
        }
        return fixes.get(cause, "Inspect the pipeline trace for anomalies.")
