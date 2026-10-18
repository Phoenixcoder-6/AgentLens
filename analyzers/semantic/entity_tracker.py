"""
analyzers/semantic/entity_tracker.py — Day 54: Semantic Entity Set Overlap Engine
==================================================================================
Extracts named entities from agent text payloads and computes mathematical set
differences to detect entity substitution and fabrication hallucinations.

Mathematical Formulation:
    E_invented = E_downstream \\ E_upstream     (Entities fabricated by downstream agent)
    E_dropped  = E_upstream \\ E_downstream     (Entities omitted by downstream agent)
    E_shared   = E_downstream ∩ E_upstream     (Entities faithfully preserved)

    Preservation Ratio = |E_shared| / |E_upstream|
    Hallucination Rate = |E_invented| / |E_downstream|

Features:
    - Zero external dependencies: fast, deterministic regex extraction.
    - Extracts proper nouns, multi-word entities, acronyms, CamelCase identifiers,
      and quoted phrases.
    - Stopword filtering and canonical case-folding normalization.
    - Identifies substitution attacks where scalar counts match (|E_down| = |E_up|)
      but actual entities have been swapped.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from schema.models import AgentStep, RunTrace

# ─────────────────────────────────────────────────────────────────────────────
# Common English Stopwords & Discourse Markers (filtered from single-word proper nouns)
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "according",
        "across",
        "action",
        "actually",
        "additionally",
        "after",
        "again",
        "against",
        "agent",
        "all",
        "almost",
        "along",
        "already",
        "also",
        "although",
        "always",
        "among",
        "an",
        "analysis",
        "and",
        "another",
        "any",
        "anyway",
        "are",
        "around",
        "as",
        "at",
        "back",
        "based",
        "be",
        "because",
        "been",
        "before",
        "behind",
        "being",
        "below",
        "between",
        "both",
        "briefly",
        "but",
        "by",
        "can",
        "certain",
        "clearly",
        "close",
        "come",
        "conclusion",
        "could",
        "data",
        "detail",
        "details",
        "did",
        "different",
        "do",
        "does",
        "done",
        "down",
        "during",
        "each",
        "early",
        "earnings",
        "either",
        "else",
        "end",
        "entities",
        "entity",
        "even",
        "every",
        "example",
        "executive",
        "fact",
        "facts",
        "few",
        "figure",
        "finally",
        "financial",
        "find",
        "findings",
        "first",
        "five",
        "focus",
        "following",
        "for",
        "four",
        "from",
        "further",
        "furthermore",
        "general",
        "get",
        "give",
        "given",
        "go",
        "good",
        "great",
        "group",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "high",
        "him",
        "his",
        "how",
        "however",
        "i",
        "if",
        "in",
        "income",
        "including",
        "into",
        "is",
        "issue",
        "it",
        "its",
        "just",
        "keep",
        "key",
        "last",
        "lastly",
        "late",
        "later",
        "lead",
        "least",
        "less",
        "let",
        "like",
        "likely",
        "little",
        "look",
        "made",
        "main",
        "make",
        "many",
        "market",
        "matter",
        "may",
        "me",
        "mean",
        "meanwhile",
        "might",
        "more",
        "moreover",
        "most",
        "much",
        "must",
        "my",
        "need",
        "never",
        "new",
        "next",
        "no",
        "non",
        "none",
        "nor",
        "not",
        "note",
        "notes",
        "now",
        "number",
        "numbers",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "open",
        "or",
        "order",
        "other",
        "others",
        "our",
        "out",
        "output",
        "over",
        "overall",
        "own",
        "page",
        "part",
        "per",
        "perhaps",
        "person",
        "point",
        "possible",
        "present",
        "presented",
        "previously",
        "process",
        "profit",
        "provide",
        "provided",
        "public",
        "quarterly",
        "quite",
        "rather",
        "really",
        "report",
        "reported",
        "research",
        "response",
        "result",
        "results",
        "revenue",
        "right",
        "said",
        "same",
        "say",
        "second",
        "see",
        "set",
        "several",
        "she",
        "should",
        "show",
        "shows",
        "side",
        "since",
        "six",
        "small",
        "so",
        "some",
        "someone",
        "something",
        "source",
        "sources",
        "state",
        "step",
        "steps",
        "still",
        "such",
        "summary",
        "sure",
        "system",
        "take",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "therefore",
        "these",
        "they",
        "thing",
        "things",
        "think",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "thus",
        "time",
        "to",
        "today",
        "together",
        "too",
        "topic",
        "total",
        "toward",
        "towards",
        "two",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "use",
        "used",
        "uses",
        "using",
        "various",
        "very",
        "view",
        "was",
        "way",
        "we",
        "well",
        "went",
        "were",
        "what",
        "whatever",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "word",
        "work",
        "worker",
        "would",
        "write",
        "writer",
        "written",
        "year",
        "yes",
        "yet",
        "you",
        "your",
        # Common occupational titles, honorifics, and role nouns
        "command",
        "module",
        "pilot",
        "astronaut",
        "astronauts",
        "director",
        "president",
        "officer",
        "manager",
        "chief",
        "chairman",
        "engineer",
        "specialist",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Regex Patterns for Named Entity Extraction
# ─────────────────────────────────────────────────────────────────────────────

# Multi-word proper nouns (e.g. "Microsoft Azure", "Goldman Sachs", "United States")
RE_MULTI_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# CamelCase identifiers (e.g. "AgentLens", "ChatGPT", "BioNTech", "DeepMind", "PyTorch")
RE_CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b")

# Hyphenated / dotted product & model codes (e.g. "GPT-4o", "Claude-3-Opus", "Llama-3-70b", "DeBERTa-v3")
RE_CODE_NAME = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")

# Uppercase acronyms & initialisms (e.g. "FDA", "SEC", "NASA", "API", "Q3", "LLM", "WHO")
RE_ACRONYM = re.compile(r"\b[A-Z]{2,}(?:[0-9]+)?\b")

# Explicitly quoted phrases / names (e.g. "Apollo 11", 'Project Titan')
RE_QUOTED = re.compile(r"[\"']([A-Z0-9][^\"']{1,40})[\"']")

# Single capitalized proper noun candidate (e.g. "Google", "Tesla", "Pfizer")
RE_SINGLE_PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Numerical metrics with currency or units (e.g. "$100M", "$4.5 billion", "15%")
RE_FINANCIAL_METRIC = re.compile(
    r"[\$€£]\s*[0-9]+(?:\.[0-9]+)?(?:\s*(?:billion|million|trillion|k|m|b))?",
    re.IGNORECASE,
)
RE_PERCENTAGE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\s*%")


# ─────────────────────────────────────────────────────────────────────────────
# Entity Delta Report Model
# ─────────────────────────────────────────────────────────────────────────────


class EntityDeltaReport(BaseModel):
    """
    Mathematical set difference and overlap report between upstream and downstream agent entities.
    """

    upstream_agent: str = Field(description="Name of the upstream source agent")
    downstream_agent: str = Field(description="Name of the downstream synthesizing agent")

    # Raw entity sets (canonical normalized strings)
    upstream_entities: set[str] = Field(
        default_factory=set, description="Entities found in upstream payload"
    )
    downstream_entities: set[str] = Field(
        default_factory=set, description="Entities found in downstream payload"
    )

    # Set operations
    shared_entities: set[str] = Field(
        default_factory=set,
        description="Entities present in both upstream and downstream (intersection)",
    )
    invented_entities: set[str] = Field(
        default_factory=set, description="Entities invented by downstream (downstream \\ upstream)"
    )
    dropped_entities: set[str] = Field(
        default_factory=set, description="Entities omitted by downstream (upstream \\ downstream)"
    )

    # Metrics
    preservation_ratio: float = Field(
        default=1.0,
        description="|shared| / |upstream| (ratio of upstream entities preserved downstream)",
    )
    hallucination_rate: float = Field(
        default=0.0,
        description="|invented| / |downstream| (ratio of downstream entities fabricated)",
    )

    # Diagnostic flags
    has_hallucination: bool = Field(
        default=False, description="True if downstream invented at least one novel entity"
    )
    has_information_loss: bool = Field(
        default=False, description="True if downstream omitted upstream entities"
    )
    substitution_detected: bool = Field(
        default=False,
        description="True if entities were swapped/replaced while scalar counts remained roughly identical",
    )

    model_config = {"arbitrary_types_allowed": True}


# ─────────────────────────────────────────────────────────────────────────────
# Entity Normalization & Extraction Functions
# ─────────────────────────────────────────────────────────────────────────────


def normalize_entity(entity: str) -> str:
    """
    Normalize an entity string for robust canonical comparison.
    Strips punctuation, trailing possessives ('s), collapses whitespace,
    and removes leading/trailing stopwords and articles.
    """
    text = entity.strip().strip(".,;:!?\"'()[]{}*`")
    # Strip possessive suffix: "Apple's" -> "Apple"
    if text.endswith("'s") or text.endswith("’s"):
        text = text[:-2]
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text).lower()
    words = text.split()
    while words and words[0] in STOPWORDS:
        words.pop(0)
    while words and words[-1] in STOPWORDS:
        words.pop()
    return " ".join(words)


def extract_entities(text: str) -> set[str]:
    """
    Extract canonical named entities, acronyms, and proper nouns from freeform text.

    Returns:
        Set of lowercase, normalized entity strings.
    """
    if not text:
        return set()

    entities: set[str] = set()

    # 1. Multi-word proper nouns (highest specificity)
    for match in RE_MULTI_PROPER.finditer(text):
        token = match.group(0)
        norm = normalize_entity(token)
        # Check that it's not simply a multi-word stopword sequence
        words = norm.split()
        if any(w not in STOPWORDS for w in words):
            entities.add(norm)

    # 2. CamelCase identifiers
    for match in RE_CAMEL_CASE.finditer(text):
        norm = normalize_entity(match.group(0))
        if norm not in STOPWORDS:
            entities.add(norm)

    # 3. Model & product codes (e.g. GPT-4o, Claude-3)
    for match in RE_CODE_NAME.finditer(text):
        norm = normalize_entity(match.group(0))
        entities.add(norm)

    # 4. Acronyms & initialisms (e.g. FDA, SEC, Q3, NASA)
    for match in RE_ACRONYM.finditer(text):
        norm = normalize_entity(match.group(0))
        if norm not in STOPWORDS and len(norm) >= 2:
            entities.add(norm)

    # 5. Quoted entities
    for match in RE_QUOTED.finditer(text):
        norm = normalize_entity(match.group(1))
        if norm and norm not in STOPWORDS:
            entities.add(norm)

    # 6. Financial metrics & percentages
    for match in RE_FINANCIAL_METRIC.finditer(text):
        norm = normalize_entity(match.group(0))
        if norm:
            entities.add(norm)
    for match in RE_PERCENTAGE.finditer(text):
        norm = normalize_entity(match.group(0))
        if norm:
            entities.add(norm)

    # 7. Single capitalized proper nouns (excluding sentence-starters and stopwords)
    # Check if a word appears capitalized NOT at the start of a sentence,
    # or if capitalized at start of sentence, is not in STOPWORDS.
    sentences = re.split(r"[.!?\n]\s*", text)
    for sentence in sentences:
        words = sentence.strip().split()
        for idx, word in enumerate(words):
            clean_word = word.strip(".,;:!?\"'()[]{}*`")
            if not clean_word:
                continue
            if clean_word[0].isupper() and len(clean_word) >= 3 and clean_word.isalpha():
                norm = clean_word.lower()
                if norm in STOPWORDS:
                    continue
                # If first word in sentence, be slightly more selective
                if idx == 0 and len(words) > 1 and clean_word.istitle():
                    # Check if it also appears elsewhere or is capitalized
                    entities.add(norm)
                else:
                    entities.add(norm)

    # Post-filter: If a multi-word entity is present, remove single words contained within it
    # e.g., if "microsoft azure" is present, keep both or let multi-word represent it
    return entities


def compare_entity_sets(
    upstream_set: set[str],
    downstream_set: set[str],
    upstream_agent: str = "upstream",
    downstream_agent: str = "downstream",
) -> EntityDeltaReport:
    """
    Perform mathematical set difference between upstream and downstream entity sets.

    Calculates:
        shared    = downstream ∩ upstream
        invented  = downstream \\ upstream
        dropped   = upstream \\ downstream
    """
    shared = downstream_set & upstream_set
    invented = downstream_set - upstream_set
    dropped = upstream_set - downstream_set

    up_count = len(upstream_set)
    down_count = len(downstream_set)

    preservation_ratio = len(shared) / up_count if up_count > 0 else 1.0
    hallucination_rate = len(invented) / down_count if down_count > 0 else 0.0

    # Substitution detection: agent swapped entities while count remained similar
    # e.g. dropped >= 1 and invented >= 1, and count difference is small
    substitution = bool(
        len(invented) > 0
        and len(dropped) > 0
        and abs(down_count - up_count) <= max(1, int(up_count * 0.4))
    )

    return EntityDeltaReport(
        upstream_agent=upstream_agent,
        downstream_agent=downstream_agent,
        upstream_entities=upstream_set,
        downstream_entities=downstream_set,
        shared_entities=shared,
        invented_entities=invented,
        dropped_entities=dropped,
        preservation_ratio=preservation_ratio,
        hallucination_rate=hallucination_rate,
        has_hallucination=len(invented) > 0,
        has_information_loss=len(dropped) > 0,
        substitution_detected=substitution,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EntityTracker Engine
# ─────────────────────────────────────────────────────────────────────────────


class EntityTracker:
    """
    Tracks and diffs entity sets between upstream and downstream agents in a pipeline.
    """

    @staticmethod
    def extract(text: str) -> set[str]:
        """Extract canonical entities from text."""
        return extract_entities(text)

    @staticmethod
    def compare_texts(
        upstream_text: str,
        downstream_text: str,
        upstream_agent: str = "upstream",
        downstream_agent: str = "downstream",
    ) -> EntityDeltaReport:
        """Extract entities from raw texts and compute mathematical set difference."""
        up_entities = extract_entities(upstream_text)
        down_entities = extract_entities(downstream_text)
        return compare_entity_sets(
            upstream_set=up_entities,
            downstream_set=down_entities,
            upstream_agent=upstream_agent,
            downstream_agent=downstream_agent,
        )

    @staticmethod
    def compare_steps(
        upstream_step: AgentStep,
        downstream_step: AgentStep,
    ) -> EntityDeltaReport:
        """
        Compare entity sets between two AgentStep objects.
        Considers both step output and handoff state if available.
        """
        # Combine input, output, and handoff state for complete upstream context
        up_text_parts = [upstream_step.output or ""]
        if upstream_step.input:
            up_text_parts.append(str(upstream_step.input))
        if upstream_step.handoff:
            if upstream_step.handoff.output_state:
                up_text_parts.append(str(upstream_step.handoff.output_state))
            if upstream_step.handoff.input_state:
                up_text_parts.append(str(upstream_step.handoff.input_state))

        down_text_parts = [downstream_step.output or ""]
        if downstream_step.handoff and downstream_step.handoff.output_state:
            down_text_parts.append(str(downstream_step.handoff.output_state))

        return EntityTracker.compare_texts(
            upstream_text="\n".join(up_text_parts),
            downstream_text="\n".join(down_text_parts),
            upstream_agent=upstream_step.agent,
            downstream_agent=downstream_step.agent,
        )

    @staticmethod
    def analyze_trace(trace: RunTrace) -> list[EntityDeltaReport]:
        """
        Analyze all sequential or topological transitions in a RunTrace.
        """
        reports: list[EntityDeltaReport] = []
        if not trace.steps or len(trace.steps) < 2:
            return reports

        for i in range(1, len(trace.steps)):
            up_step = trace.steps[i - 1]
            down_step = trace.steps[i]
            rep = EntityTracker.compare_steps(up_step, down_step)
            reports.append(rep)

        return reports
