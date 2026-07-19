"""
analyzers/evidence_extraction/extractor.py — Tiny Evidence Extractor
======================================================================
Day 9: Extract exactly three fields from a raw agent output string
via a schema-constrained LLM call.

Fields extracted:
    source_count  — number of sources referenced in the output
    entity_count  — number of named entities mentioned
    tool_calls    — list of any tool names invoked (empty list if none)

"Schema-constrained" means the LLM is forced to return a Pydantic model
via with_structured_output() — not free text. If extraction fails, a safe
fallback ExtractedEvidence(0, 0, []) is returned and the error is logged.

Why only these three fields?
    - source_count and entity_count come directly from the pipeline state,
      making them ground-truth verifiable.
    - tool_calls are needed to detect silent tool failures later.
    - Nothing broader yet — Day 9 is intentionally tiny and reliable.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config.config_loader import get
from schema.models import SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedEvidence — the schema the LLM must return
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedEvidence(BaseModel):
    """
    Structured evidence extracted from a single agent's output.
    Produced by a schema-constrained LLM call — not text parsing.

    schema_version is stamped so records are traceable across schema upgrades.
    """
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version stamped on every record"
    )
    source_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct sources or citations referenced in the output"
    )
    entity_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct named entities (people, orgs, places, concepts) mentioned"
    )
    tool_calls: list[str] = Field(
        default_factory=list,
        description="Names of any tools explicitly invoked during this step. Empty list if none."
    )
    extraction_failed: bool = Field(
        default=False,
        description="True if extraction fell back to defaults due to LLM error"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if extraction_failed is True"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceExtractor
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise evidence extractor for an AI observability system.

Your job: read the agent output below and extract exactly three facts.
Return ONLY the structured fields — no explanation, no extra text.

Rules:
- source_count: count distinct sources, citations, books, reports, or URLs explicitly listed
- entity_count: count distinct named entities (people, organizations, cities, technologies, programs)
- tool_calls: list the name of any tool explicitly called (e.g. "web_search", "calculator"). 
  If no tools were called, return an empty list [].

Be precise. Count carefully. When in doubt, undercount rather than overcount."""

_USER_PROMPT = """Agent output to analyze:

{output}

Extract source_count, entity_count, and tool_calls from the above."""


class EvidenceExtractor:
    """
    Extracts structured evidence from a raw agent output string.

    Usage:
        extractor = EvidenceExtractor()
        evidence = extractor.extract(raw_output="SOURCES:\\n- Book A\\nENTITIES:\\n- NITI Aayog")
        # ExtractedEvidence(source_count=1, entity_count=1, tool_calls=[])
    """

    def __init__(self) -> None:
        self._llm = self._build_llm()

    def _build_llm(self):
        """Build a schema-constrained LLM that returns ExtractedEvidence."""
        from dotenv import load_dotenv
        load_dotenv()

        model = get("llm", "model", "llama-3.3-70b-versatile")
        temperature = float(get("llm", "temperature", 0.0))

        # with_structured_output forces the LLM to return a valid ExtractedEvidence
        # Pydantic model — no free-text parsing needed
        base_llm = ChatGroq(model=model, temperature=temperature)
        return base_llm.with_structured_output(ExtractedEvidence)

    def extract(self, raw_output: str, agent: str = "") -> ExtractedEvidence:
        """
        Run a schema-constrained LLM call to extract evidence from raw_output.

        Args:
            raw_output: The raw string output from a single agent step.
            agent:      Optional agent name for logging context.

        Returns:
            ExtractedEvidence with schema_version stamped.
            On LLM failure, returns safe fallback with extraction_failed=True.
        """
        if not raw_output or not raw_output.strip():
            return ExtractedEvidence(
                extraction_failed=True,
                error_message="Empty output — nothing to extract"
            )

        # Truncate very long outputs to stay within token limits
        truncated = raw_output[:4000] if len(raw_output) > 4000 else raw_output

        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=_USER_PROMPT.format(output=truncated)),
            ]
            result: ExtractedEvidence = self._llm.invoke(messages)
            # Stamp schema_version (Pydantic default does this, but enforce it)
            result.schema_version = SCHEMA_VERSION
            return result

        except Exception as exc:
            return ExtractedEvidence(
                extraction_failed=True,
                error_message=f"{type(exc).__name__}: {exc}"
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
