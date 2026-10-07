"""
capture/pii_scrubber.py — Config-Driven PII Scrubber
=====================================================
Day 39: Redacts sensitive PII (emails, phone numbers, SSNs, API keys,
credit card numbers) from captured agent steps and traces before storage.

Features:
    - Driven by `capture.pii_scrubbing` in `config/config.yaml`
    - Configurable regex patterns + replacement tokens
    - Optional NER-based scrubbing via spaCy (`en_core_web_sm`) when installed,
      with graceful fallback to regex-only when spaCy or the model is absent
    - Fail-safe: scrubbing errors never crash the capture or agent pipeline
"""

from __future__ import annotations

import logging
import re
from typing import Any

from config.config_loader import get
from schema.models import AgentStep, RunTrace

logger = logging.getLogger(__name__)

# Default regex patterns used if config.yaml patterns are missing or overridden
DEFAULT_PII_PATTERNS: dict[str, dict[str, str]] = {
    "email": {
        "regex": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "replacement": "[REDACTED_EMAIL]",
    },
    "phone": {
        "regex": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b",
        "replacement": "[REDACTED_PHONE]",
    },
    "ssn": {
        "regex": r"\b\d{3}-\d{2}-\d{4}\b",
        "replacement": "[REDACTED_SSN]",
    },
    "api_key": {
        "regex": r"\b(?:gsk_[a-zA-Z0-9]{16,}|sk-[a-zA-Z0-9_-]{16,})\b",
        "replacement": "[REDACTED_API_KEY]",
    },
    "credit_card": {
        "regex": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "replacement": "[REDACTED_CREDIT_CARD]",
    },
}

_NER_LABEL_MAP: dict[str, str] = {
    "PERSON": "[REDACTED_PERSON]",
    "ORG": "[REDACTED_ORG]",
    "GPE": "[REDACTED_LOCATION]",
}

_SPACY_NLP_CACHE: Any = False  # False = uninitialized, None = unavailable


def _get_spacy_nlp() -> Any:
    """
    Lazily load spaCy `en_core_web_sm` if installed.
    Returns None when spaCy or the model is not installed (graceful fallback).
    """
    global _SPACY_NLP_CACHE
    if _SPACY_NLP_CACHE is not False:
        return _SPACY_NLP_CACHE

    try:
        import spacy  # type: ignore[import-not-found]

        _SPACY_NLP_CACHE = spacy.load("en_core_web_sm")
    except Exception as exc:
        logger.debug("spaCy NER unavailable, falling back to regex-only PII scrubbing: %s", exc)
        _SPACY_NLP_CACHE = None

    return _SPACY_NLP_CACHE


class PIIScrubber:
    """
    Config-driven PII scrubber for strings, nested dicts/lists, AgentSteps, and RunTraces.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        patterns: dict[str, dict[str, str]] | None = None,
        use_spacy_ner: bool | None = None,
    ) -> None:
        pii_cfg = get("capture", "pii_scrubbing", {}) or {}

        if enabled is None:
            self.enabled = bool(pii_cfg.get("enabled", False))
        else:
            self.enabled = bool(enabled)

        if use_spacy_ner is None:
            self.use_spacy_ner = bool(pii_cfg.get("use_spacy_ner", False))
        else:
            self.use_spacy_ner = bool(use_spacy_ner)

        raw_patterns = (
            patterns if patterns is not None else pii_cfg.get("patterns", DEFAULT_PII_PATTERNS)
        )
        if not raw_patterns:
            raw_patterns = DEFAULT_PII_PATTERNS

        self._compiled_patterns: list[tuple[str, re.Pattern[str], str]] = []
        for name, spec in raw_patterns.items():
            if not isinstance(spec, dict):
                continue
            regex_str = spec.get("regex", "")
            replacement = spec.get("replacement", f"[REDACTED_{name.upper()}]")
            if regex_str:
                try:
                    self._compiled_patterns.append((name, re.compile(regex_str), replacement))
                except re.error as exc:
                    logger.warning("Invalid PII regex for '%s': %s", name, exc)

    def scrub_text(self, text: str) -> str:
        """
        Redact PII from a string if scrubbing is enabled.
        Never raises an exception — returns original text on unexpected errors.
        """
        if not self.enabled or not text or not isinstance(text, str):
            return text

        try:
            scrubbed = text
            # 1. Regex pass (emails, phones, SSNs, API keys, credit cards)
            for _name, pattern, replacement in self._compiled_patterns:
                scrubbed = pattern.sub(replacement, scrubbed)

            # 2. Optional spaCy NER pass (if enabled and spaCy is available)
            if self.use_spacy_ner:
                nlp = _get_spacy_nlp()
                if nlp is not None:
                    doc = nlp(scrubbed)
                    # Replace entities from right to left to preserve character offsets
                    ents = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
                    for ent in ents:
                        repl = _NER_LABEL_MAP.get(ent.label_)
                        if repl and not scrubbed[ent.start_char : ent.end_char].startswith(
                            "[REDACTED_"
                        ):
                            scrubbed = scrubbed[: ent.start_char] + repl + scrubbed[ent.end_char :]
            return scrubbed
        except Exception as exc:
            logger.warning("PIIScrubber.scrub_text failed safely: %s", exc)
            return text

    def scrub_value(self, value: Any) -> Any:
        """Recursively scrub strings inside dicts, lists, and tuples."""
        if not self.enabled:
            return value

        try:
            if isinstance(value, str):
                return self.scrub_text(value)
            if isinstance(value, dict):
                return {k: self.scrub_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [self.scrub_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(self.scrub_value(item) for item in value)
            return value
        except Exception as exc:
            logger.warning("PIIScrubber.scrub_value failed safely: %s", exc)
            return value

    def scrub_step(self, step: AgentStep) -> AgentStep:
        """Scrub all text and state fields on an AgentStep in-place."""
        if not self.enabled or step is None:
            return step

        try:
            if step.input:
                step.input = self.scrub_text(step.input)
            if step.output:
                step.output = self.scrub_text(step.output)
            if step.prompt:
                step.prompt = self.scrub_text(step.prompt)
            if step.error:
                step.error = self.scrub_text(step.error)
            if step.handoff is not None:
                step.handoff.input_state = self.scrub_value(step.handoff.input_state)
                step.handoff.filtered_state = self.scrub_value(step.handoff.filtered_state)
                step.handoff.output_state = self.scrub_value(step.handoff.output_state)
        except Exception as exc:
            logger.warning("PIIScrubber.scrub_step failed safely: %s", exc)

        return step

    def scrub_trace(self, trace: RunTrace) -> RunTrace:
        """Scrub all steps and expected_output on a RunTrace in-place."""
        if not self.enabled or trace is None:
            return trace

        try:
            if getattr(trace, "expected_output", None):
                trace.expected_output = self.scrub_text(str(trace.expected_output))
            for step in trace.steps:
                self.scrub_step(step)
        except Exception as exc:
            logger.warning("PIIScrubber.scrub_trace failed safely: %s", exc)

        return trace
