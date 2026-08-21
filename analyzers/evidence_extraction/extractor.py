"""
analyzers/evidence_extraction/extractor.py — Evidence Extractor
================================================================
Day 9:  Extract source_count, entity_count, tool_calls.
Day 20: Broaden extraction to claims, references, numbers, dates.
        All new fields are extracted in the SAME single LLM call.
        Token cost is now logged per call for the cost tracker.

Fields extracted:
    source_count   — number of sources / citations referenced
    entity_count   — number of distinct named entities mentioned
    tool_calls     — tool names explicitly invoked (empty if none)
    claims         — key factual claims made in the output
    references     — explicit citation strings (author/title/URL)
    numbers        — notable numeric facts (stats, figures, years)
    dates          — date / time expressions mentioned

Uses JSON mode (response_format: json_object) for broad model
compatibility across all Groq-hosted models. If extraction fails,
a safe fallback ExtractedEvidence is returned and the error logged.
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

    Day 20 extends the schema with claims, references, numbers, dates.
    All fields have safe defaults so older callers remain compatible.
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION, description="Schema version stamped on every record"
    )

    # ── Original Day-9 fields ────────────────────────────────────────────────
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

    # ── Day-20 extended fields ───────────────────────────────────────────────
    claims: list[str] = Field(
        default_factory=list,
        description=(
            "Key factual claims made in the output "
            "(e.g. 'Apollo 11 landed on July 20, 1969'). "
            "Max 10 most important claims."
        ),
    )
    references: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit citation strings — author names, book/article titles, URLs, "
            "or report names. Empty list if none."
        ),
    )
    numbers: list[str] = Field(
        default_factory=list,
        description=(
            "Notable numeric facts: statistics, percentages, counts, measurements "
            "(e.g. '382 kg of lunar rocks', '$1 trillion'). Max 10 items."
        ),
    )
    dates: list[str] = Field(
        default_factory=list,
        description=(
            "Date and time expressions mentioned "
            "(e.g. 'July 20, 1969', 'early 1970s'). Max 10 items."
        ),
    )

    # ── Extraction metadata ──────────────────────────────────────────────────
    extraction_failed: bool = Field(
        default=False, description="True if extraction fell back to defaults due to LLM error"
    )
    error_message: str | None = Field(
        default=None, description="Error details if extraction_failed is True"
    )
    token_cost: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed by this extraction call (prompt + completion).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise evidence extractor for an AI observability system.

Your job: read the agent output below and extract ALL of the following fields.
Return ONLY a valid JSON object — no explanation, no extra text, no markdown.

Field definitions:
- source_count  : integer — count of distinct sources, citations, books, reports, or URLs
- entity_count  : integer — count of distinct named entities (people, orgs, cities, technologies)
- tool_calls    : list[str] — tool names explicitly invoked (e.g. "web_search"). Empty list if none.
- claims        : list[str] — key factual claims stated in the text. Max 10, most important first.
- references    : list[str] — explicit citation strings (author, title, URL, or report name). Empty if none.
- numbers       : list[str] — notable numeric facts, stats, or measurements (e.g. "382 kg", "$1 trillion"). Max 10.
- dates         : list[str] — date/time expressions mentioned (e.g. "July 20, 1969", "early 1970s"). Max 10.

Return format (JSON only):
{
  "source_count": <integer>,
  "entity_count": <integer>,
  "tool_calls": [<string>, ...],
  "claims": [<string>, ...],
  "references": [<string>, ...],
  "numbers": [<string>, ...],
  "dates": [<string>, ...]
}

Be precise. Count carefully. When in doubt, undercount rather than overcount.
For list fields, return [] if nothing applies."""

_USER_PROMPT = """Agent output to analyze:

{output}

Extract all fields and return valid JSON only."""


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceExtractor
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceExtractor:
    """
    Extracts structured evidence from a raw agent output string.

    Day 9:  source_count, entity_count, tool_calls
    Day 20: + claims, references, numbers, dates + token_cost logging

    Uses JSON mode (response_format: json_object) for broad model compatibility.
    This avoids the 400 "tool not in request" errors from with_structured_output()
    on models that don't support function calling.

    Usage:
        extractor = EvidenceExtractor()
        evidence = extractor.extract(raw_output="SOURCES:\\n- Book A\\nENTITIES:\\n- NITI Aayog")
        # ExtractedEvidence(source_count=1, entity_count=1, claims=[...], dates=[...])
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
            ExtractedEvidence with schema_version and token_cost stamped.
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

            # Extract token cost from response metadata if available
            token_cost = 0
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("token_usage", {})
                token_cost = int(usage.get("total_tokens", 0))

            result = ExtractedEvidence(
                source_count=int(parsed.get("source_count", 0)),
                entity_count=int(parsed.get("entity_count", 0)),
                tool_calls=list(parsed.get("tool_calls", [])),
                claims=list(parsed.get("claims", [])),
                references=list(parsed.get("references", [])),
                numbers=list(parsed.get("numbers", [])),
                dates=list(parsed.get("dates", [])),
                token_cost=token_cost,
            )

            log.info(
                "Extraction LLM call completed",
                extra={
                    "extra_fields": {
                        "model": get("llm", "model", "unknown"),
                        "latency_ms": round(latency_ms, 2),
                        "agent": agent,
                        "token_cost": token_cost,
                        "claims_count": len(result.claims),
                        "dates_count": len(result.dates),
                        "numbers_count": len(result.numbers),
                        "references_count": len(result.references),
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
