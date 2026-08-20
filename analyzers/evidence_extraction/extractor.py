"""
analyzers/evidence_extraction/extractor.py — Tiny Evidence Extractor
======================================================================
Day 9: Extract exactly three fields from a raw agent output string
via a JSON-mode LLM call.

Fields extracted:
    source_count  — number of sources referenced in the output
    entity_count  — number of named entities mentioned
    tool_calls    — list of any tool names invoked (empty list if none)

Uses JSON mode (response_format: json_object) instead of function calling
for broad model compatibility across all Groq-hosted models.
If extraction fails, a safe fallback ExtractedEvidence(0, 0, []) is returned
and the error is logged.
"""

from __future__ import annotations

import json
import os
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config.config_loader import get
from config.logging_config import get_logger
from schema.models import SCHEMA_VERSION

log = get_logger("extractor")

# ─────────────────────────────────────────────────────────────────────────────
# ExtractedEvidence — the schema the LLM must return
# ─────────────────────────────────────────────────────────────────────────────


class ExtractedEvidence(BaseModel):
    """
    Structured evidence extracted from a single agent's output.
    Produced by a JSON-mode LLM call — not text parsing.

    schema_version is stamped so records are traceable across schema upgrades.
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION, description="Schema version stamped on every record"
    )
    source_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct sources or citations referenced in the output",
    )
    entity_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct named entities (people, orgs, places, concepts) mentioned",
    )
    tool_calls: list[str] = Field(
        default_factory=list,
        description="Names of any tools explicitly invoked during this step. Empty list if none.",
    )
    extraction_failed: bool = Field(
        default=False, description="True if extraction fell back to defaults due to LLM error"
    )
    error_message: str | None = Field(
        default=None, description="Error details if extraction_failed is True"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise evidence extractor for an AI observability system.

Your job: read the agent output below and extract exactly three facts.
Return ONLY a valid JSON object with these keys — no explanation, no extra text.

Rules:
- source_count: count distinct sources, citations, books, reports, or URLs explicitly listed
- entity_count: count distinct named entities (people, organizations, cities, technologies, programs)
- tool_calls: list the name of any tool explicitly called (e.g. "web_search", "calculator").
  If no tools were called, return an empty list [].

Return format (JSON only):
{
  "source_count": <integer>,
  "entity_count": <integer>,
  "tool_calls": [<string>, ...]
}

Be precise. Count carefully. When in doubt, undercount rather than overcount."""

_USER_PROMPT = """Agent output to analyze:

{output}

Extract source_count, entity_count, and tool_calls from the above. Return valid JSON only."""


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceExtractor
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceExtractor:
    """
    Extracts structured evidence from a raw agent output string.

    Uses JSON mode (response_format: json_object) for broad model compatibility.
    This avoids the 400 "tool not in request" errors from with_structured_output()
    on models that don't support function calling.

    Usage:
        extractor = EvidenceExtractor()
        evidence = extractor.extract(raw_output="SOURCES:\\n- Book A\\nENTITIES:\\n- NITI Aayog")
        # ExtractedEvidence(source_count=1, entity_count=1, tool_calls=[])
    """

    def __init__(self) -> None:
        self._llm = self._build_llm()

    def _build_llm(self):
        """Build an LLM in JSON mode for structured extraction."""
        from dotenv import load_dotenv

        load_dotenv()

        model = get("llm", "model", "openai/gpt-oss-120b")
        temperature = float(get("llm", "temperature", 0.0))

        # json_object response format — broadly supported across Groq models
        # without needing tool/function calling capability.
        return ChatGroq(
            model=model,
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def extract(self, raw_output: str, agent: str = "") -> ExtractedEvidence:
        """
        Run a JSON-mode LLM call to extract evidence from raw_output.

        Args:
            raw_output: The raw string output from a single agent step.
            agent:      Optional agent name for logging context.

        Returns:
            ExtractedEvidence with schema_version stamped.
            On LLM failure, returns safe fallback with extraction_failed=True.
        """
        if not raw_output or not raw_output.strip():
            return ExtractedEvidence(
                extraction_failed=True, error_message="Empty output — nothing to extract"
            )

        # Truncate very long outputs to stay within token limits
        truncated = raw_output[:4000] if len(raw_output) > 4000 else raw_output

        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=_USER_PROMPT.format(output=truncated)),
            ]

            start_t = time.time()
            response = self._llm.invoke(messages)
            latency_ms = (time.time() - start_t) * 1000

            # Parse JSON response into ExtractedEvidence
            raw_json = response.content if hasattr(response, "content") else str(response)
            parsed = json.loads(raw_json)
            result = ExtractedEvidence(
                source_count=int(parsed.get("source_count", 0)),
                entity_count=int(parsed.get("entity_count", 0)),
                tool_calls=list(parsed.get("tool_calls", [])),
            )

            log.info(
                "Extraction LLM call completed",
                extra={
                    "extra_fields": {
                        "model": get("llm", "model", "unknown"),
                        "latency_ms": round(latency_ms, 2),
                        "agent": agent,
                        "cost_estimate": 0.0,
                    }
                },
            )

            result.schema_version = SCHEMA_VERSION
            return result

        except Exception as exc:
            log.warning(
                "Extraction LLM call failed",
                extra={"extra_fields": {"error": str(exc), "agent": agent}},
            )
            return ExtractedEvidence(
                extraction_failed=True, error_message=f"{type(exc).__name__}: {exc}"
            )

    def extract_run(self, steps: list[dict]) -> dict[int, ExtractedEvidence]:
        """
        Extract evidence for all steps in a run.

        Args:
            steps: List of step dicts from db.get_steps_for_run() or NormalizedStep list.

        Returns:
            Dict mapping step number → ExtractedEvidence
        """
        results = {}
        for step in steps:
            # Support both NormalizedStep objects and plain dicts
            step_num = step["step"] if isinstance(step, dict) else step.step
            raw_output = step.get("raw_output", "") if isinstance(step, dict) else step.raw_output
            agent = step.get("agent", "") if isinstance(step, dict) else step.agent

            results[step_num] = self.extract(raw_output=raw_output, agent=agent)

        return results
