"""
analyzers/evidence_extraction/extractor.py — Evidence Extractor
================================================================
Day 9:  Extract source_count, entity_count, tool_calls.
Day 20: Broaden extraction to claims, references, numbers, dates.
        Token cost logged per call.
Day 21: Retry-once on malformed/empty JSON response.
        extraction_max_tokens read from config.yaml (no hardcoding).
        extraction_failed=True skips downstream rules gracefully.

All new fields are extracted in the SAME single LLM call — no extra cost.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config.config_loader import get
from config.logging_config import get_logger
from schema.models import SCHEMA_VERSION
from storage.llm_cache import LLMCache

log = get_logger("extractor")

# ─────────────────────────────────────────────────────────────────────────────
# ExtractedEvidence — the schema the LLM must return
# ─────────────────────────────────────────────────────────────────────────────


class ExtractedEvidence(BaseModel):
    """
    Structured evidence extracted from a single agent's output.
    Produced by a JSON-mode LLM call — not text parsing.

    Day 20 extends the schema with claims, references, numbers, dates.
    Day 21 adds token_cost and retry metadata.
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
    retried: bool = Field(
        default=False,
        description="True if a retry was needed due to a malformed first response.",
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

# Retry prompt — used when the first response was malformed
_RETRY_SYSTEM_PROMPT = (
    _SYSTEM_PROMPT + "\n\nIMPORTANT: Your previous response was not valid JSON. "
    "Return ONLY the raw JSON object. No markdown, no code fences, no explanation."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_response(raw_json: str) -> dict:
    """
    Parse and validate a raw JSON string from the LLM.

    Raises:
        ValueError: if the string is empty, not valid JSON, or missing
                    required integer keys.
    """
    raw_json = raw_json.strip()
    if not raw_json:
        raise ValueError("LLM returned empty response")

    parsed = json.loads(raw_json)  # raises json.JSONDecodeError on bad JSON

    # Validate that numeric fields are actually integers (not strings/None)
    for key in ("source_count", "entity_count"):
        val = parsed.get(key, 0)
        if not isinstance(val, int):
            raise ValueError(f"Field '{key}' must be an integer, got {type(val).__name__}: {val!r}")

    return parsed


def _build_evidence(parsed: dict, token_cost: int, retried: bool) -> ExtractedEvidence:
    """Hydrate a parsed JSON dict into an ExtractedEvidence object."""
    return ExtractedEvidence(
        source_count=int(parsed.get("source_count", 0)),
        entity_count=int(parsed.get("entity_count", 0)),
        tool_calls=list(parsed.get("tool_calls", [])),
        claims=list(parsed.get("claims", [])),
        references=list(parsed.get("references", [])),
        numbers=list(parsed.get("numbers", [])),
        dates=list(parsed.get("dates", [])),
        token_cost=token_cost,
        retried=retried,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit backoff helper
# ─────────────────────────────────────────────────────────────────────────────

_RATE_LIMIT_DEFAULT_WAIT_S = 15  # fallback wait when we can't parse the suggested time


def _rate_limit_wait(exc: Exception) -> float:
    """
    Return seconds to sleep before retrying.

    - If the exception is a Groq 429 rate-limit error, parse the suggested
      wait time from the error message (e.g. "Please try again in 7.19s").
      Add a 1-second buffer and cap at 60s.
    - For all other errors (JSON parse, type validation, network) return 0
      so the retry fires immediately.
    """
    msg = str(exc)
    if "rate_limit_exceeded" in msg or "429" in msg:
        import re

        match = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
        if match:
            return min(float(match.group(1)) + 1.0, 60.0)
        return float(_RATE_LIMIT_DEFAULT_WAIT_S)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceExtractor
# ─────────────────────────────────────────────────────────────────────────────


class EvidenceExtractor:
    """
    Extracts structured evidence from a raw agent output string.

    Day 9:  source_count, entity_count, tool_calls
    Day 20: + claims, references, numbers, dates + token_cost logging
    Day 21: + retry-once on malformed JSON + extraction_max_tokens from config

    Uses JSON mode (response_format: json_object) for broad model compatibility.
    On malformed first response → retries once with a stricter prompt.
    On two consecutive failures → returns extraction_failed=True fallback.
    Downstream rules check extraction_failed before using extraction data.

    Usage:
        extractor = EvidenceExtractor()
        evidence = extractor.extract(raw_output="SOURCES:\\n- Book A\\nENTITIES:\\n- NITI Aayog")
    """

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
        self._temperature = float(get("llm", "temperature", 0.0))
        self._llm = self._build_llm(self._primary_model_name)
        self._fallback_llm = (
            self._build_llm(str(self._fallback_model_name))
            if self._fallback_model_name
            else None
        )

    def _build_llm(self, model_name: str) -> ChatGroq:
        """Build an LLM in JSON mode. max_tokens pulled from config."""
        from dotenv import load_dotenv

        load_dotenv()

        max_tokens = int(get("llm", "extraction_max_tokens", 1024))

        return ChatGroq(
            model=model_name,
            temperature=self._temperature,
            max_tokens=max_tokens,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def _call_llm(self, system: str, user_content: str) -> tuple[str, int]:
        """
        Invoke the LLM with caching and fallback model support.
        Returns (raw_content_string, total_tokens).
        Raises on network/API errors if both primary and fallback fail.
        """
        primary_model = getattr(self, "_primary_model_name", "openai/gpt-oss-120b")
        fallback_model = getattr(self, "_fallback_model_name", None)
        temperature = getattr(self, "_temperature", 0.0)
        fallback_llm = getattr(self, "_fallback_llm", None)

        is_llm_mocked = (
            hasattr(self._llm, "return_value")
            or hasattr(self._llm, "side_effect")
            or hasattr(self._llm, "assert_called")
        )

        # 1. Check cache hit (skip if LLM is mocked in unit tests)
        if not is_llm_mocked:
            cached = self._cache.get(
                system, user_content, model=primary_model, temperature=temperature
            )
            if cached is not None:
                return cached

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ]

        # 2. Primary model call
        target_model = primary_model
        try:
            response = self._llm.invoke(messages)
        except Exception as primary_exc:
            if fallback_llm and fallback_model:
                log.warning(
                    f"Primary extraction model '{primary_model}' failed: {primary_exc}. "
                    f"Retrying with fallback model '{fallback_model}'."
                )
                target_model = str(fallback_model)
                response = fallback_llm.invoke(messages)
            else:
                raise primary_exc

        raw: str = str(response.content) if hasattr(response, "content") else str(response)

        token_cost = 0
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            token_cost = int(usage.get("total_tokens", 0))

        # 3. Store in cache
        self._cache.set(
            system,
            user_content,
            model=target_model,
            response_text=raw,
            token_cost=token_cost,
            temperature=temperature,
        )

        return raw, token_cost

    def extract(self, raw_output: str, agent: str = "") -> ExtractedEvidence:
        """
        Run a JSON-mode LLM call to extract evidence from raw_output.

        Day 21 behaviour:
        - On malformed/empty first response → retry once with stricter prompt.
        - On second failure → return extraction_failed=True (no crash).
        - extraction_failed=True signals downstream rules to skip themselves.

        Args:
            raw_output: The raw string output from a single agent step.
            agent:      Optional agent name for logging context.

        Returns:
            ExtractedEvidence with schema_version and token_cost stamped.
        """
        if not raw_output or not raw_output.strip():
            return ExtractedEvidence(
                extraction_failed=True, error_message="Empty output — nothing to extract"
            )

        # Truncate very long outputs to stay within token limits
        truncated = raw_output[:4000] if len(raw_output) > 4000 else raw_output
        user_content = _USER_PROMPT.format(output=truncated)

        start_t = time.time()
        retried = False
        total_tokens = 0

        # ── Attempt 1 ────────────────────────────────────────────────────────
        try:
            raw, tokens = self._call_llm(_SYSTEM_PROMPT, user_content)
            total_tokens += tokens
            parsed = _parse_response(raw)

        except Exception as first_exc:
            wait_s = _rate_limit_wait(first_exc)
            log.warning(
                "Extraction attempt 1 failed — retrying",
                extra={
                    "extra_fields": {
                        "error": str(first_exc),
                        "agent": agent,
                        "retry_wait_s": wait_s,
                    }
                },
            )
            retried = True
            if wait_s > 0:
                time.sleep(wait_s)

            # ── Attempt 2 (retry with stricter prompt) ────────────────────────
            try:
                raw, tokens = self._call_llm(_RETRY_SYSTEM_PROMPT, user_content)
                total_tokens += tokens
                parsed = _parse_response(raw)

            except Exception as second_exc:
                latency_ms = (time.time() - start_t) * 1000
                log.warning(
                    "Extraction failed after retry",
                    extra={
                        "extra_fields": {
                            "error": str(second_exc),
                            "agent": agent,
                            "latency_ms": round(latency_ms, 2),
                            "token_cost": total_tokens,
                        }
                    },
                )
                return ExtractedEvidence(
                    extraction_failed=True,
                    error_message=f"Retry failed: {type(second_exc).__name__}: {second_exc}",
                    token_cost=total_tokens,
                    retried=True,
                )

        latency_ms = (time.time() - start_t) * 1000
        result = _build_evidence(parsed, total_tokens, retried)

        log.info(
            "Extraction LLM call completed",
            extra={
                "extra_fields": {
                    "model": get("llm", "model", "unknown"),
                    "latency_ms": round(latency_ms, 2),
                    "agent": agent,
                    "token_cost": total_tokens,
                    "retried": retried,
                    "claims_count": len(result.claims),
                    "dates_count": len(result.dates),
                    "numbers_count": len(result.numbers),
                    "references_count": len(result.references),
                }
            },
        )

        result.schema_version = SCHEMA_VERSION
        return result

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
            step_num = step["step"] if isinstance(step, dict) else step.step
            raw_output = step.get("raw_output", "") if isinstance(step, dict) else step.raw_output
            agent = step.get("agent", "") if isinstance(step, dict) else step.agent

            results[step_num] = self.extract(raw_output=raw_output, agent=agent)

        return results
