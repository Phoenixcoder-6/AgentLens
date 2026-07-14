"""
analyzers/arbiter.py — The AgentLens Arbiter
=============================================
The Arbiter is the central decision-making component of AgentLens.
It collects evidence from all analyzers and resolves it into a single,
deterministic verdict using the P1–P5 priority system.

Core architectural principle:
    SAME EVIDENCE IN → SAME OUTPUT OUT, EVERY TIME.
    The Arbiter is deterministic by design. No randomness, no LLM calls,
    no probabilistic reasoning. This is what makes its output trustworthy.

The LLM is called AFTER the Arbiter — only to explain the verdict,
never to produce it.

─────────────────────────────────────────────────────────────────────────────
PRIORITY TABLE (locked — do not change without updating the schema)
─────────────────────────────────────────────────────────────────────────────

┌──────┬──────────────────────────┬────────────────────────────────────────────┐
│  Pri │  Source                  │  Condition                                 │
├──────┼──────────────────────────┼────────────────────────────────────────────┤
│  P1  │  ground_truth_mismatch   │  expected_output present AND output differs │
│  P2  │  rule_match              │  deterministic rule fired                  │
│  P3  │  workflow_violation      │  workflow validator flagged handoff issue  │
│  P4  │  statistical_anomaly     │  metrics analyzer flagged an outlier       │
│  P5  │  unknown                 │  no evidence matched — fallback             │
└──────┴──────────────────────────┴────────────────────────────────────────────┘

TIE-BREAK RULE:
    If two pieces of evidence share the same priority level,
    the one with the lower rule_id (ascending alphabetical/numeric) wins.
    This guarantees a deterministic verdict even in ambiguous cases.

GROUNDED FLAG:
    grounded = True   → verdict backed by P1 evidence (expected_output comparison)
    grounded = False  → verdict is heuristic only (P2–P5)
    The dashboard always shows this flag. Heuristic verdicts use hedged language.

─────────────────────────────────────────────────────────────────────────────
PSEUDOCODE — determine_primary_cause(evidence_list)
─────────────────────────────────────────────────────────────────────────────

    FUNCTION determine_primary_cause(evidence: list[EvidenceRecord]) → AnalysisBundle:

        IF evidence is empty:
            RETURN AnalysisBundle(primary_cause=UNKNOWN, priority=P5, grounded=False)

        # Step 1: Check P1 — ground truth mismatch (highest priority)
        p1_evidence = [e for e in evidence if e.source == GROUND_TRUTH]
        IF p1_evidence is not empty:
            best = p1_evidence[0]   # only one ground truth per run
            RETURN AnalysisBundle(
                primary_cause = best.rule_match.category,
                priority      = P1,
                grounded      = True,
                evidence      = evidence,
                primary_agent = best.agent
            )

        # Step 2: Check P2 — rule match (deterministic rules fired)
        p2_evidence = [e for e in evidence if e.source == RULE_ENGINE]
        IF p2_evidence is not empty:
            best = sort(p2_evidence, by=rule_id ascending)[0]  # tie-break
            RETURN AnalysisBundle(
                primary_cause = best.rule_match.category,
                priority      = P2,
                grounded      = False,
                evidence      = evidence,
                primary_agent = best.agent
            )

        # Step 3: Check P3 — workflow violation (workflow/consistency validators)
        p3_evidence = [e for e in evidence
                       if e.source in (WORKFLOW_VALIDATOR, CONSISTENCY_VALIDATOR)]
        IF p3_evidence is not empty:
            best = sort(p3_evidence, by=rule_id ascending)[0]  # tie-break
            RETURN AnalysisBundle(
                primary_cause = best.rule_match.category,
                priority      = P3,
                grounded      = False,
                evidence      = evidence,
                primary_agent = best.agent
            )

        # Step 4: Check P4 — statistical anomaly (metrics analyzer)
        p4_evidence = [e for e in evidence if e.source == METRICS_ANALYZER]
        IF p4_evidence is not empty:
            best = sort(p4_evidence, by=confidence descending)[0]  # highest confidence
            RETURN AnalysisBundle(
                primary_cause = EXECUTION,   # anomalies default to execution category
                priority      = P4,
                grounded      = False,
                evidence      = evidence,
                primary_agent = best.agent
            )

        # Step 5: P5 fallback — nothing matched
        RETURN AnalysisBundle(
            primary_cause = UNKNOWN,
            priority      = P5,
            grounded      = False,
            evidence      = evidence,
            primary_agent = None
        )

─────────────────────────────────────────────────────────────────────────────
NOTE ON DAY 18a (from the build plan):
─────────────────────────────────────────────────────────────────────────────
    The P3 tier (workflow_violation) is wired explicitly in this pseudocode.
    When implementing determine_primary_cause() on Day 12, P3 must be a
    distinct code path — not merged into P2 — or workflow violations will
    never surface as secondary causes, even when no rule fires.

    Unit test required (Day 18a):
        A workflow violation with no rule match still surfaces as P3,
        correctly ranked below P1/P2 and above P4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from schema import (
    AnalysisBundle,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    PriorityLevel,
    RunTrace,
)

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Priority resolution helpers (locked — no changes after Day 3)
# ─────────────────────────────────────────────────────────────────────────────

# Maps EvidenceSource → PriorityLevel for the Arbiter's decision logic.
# This table is the authoritative source — config.yaml mirrors it for display.
SOURCE_TO_PRIORITY: dict[EvidenceSource, PriorityLevel] = {
    EvidenceSource.GROUND_TRUTH:             PriorityLevel.P1,
    EvidenceSource.RULE_ENGINE:              PriorityLevel.P2,
    EvidenceSource.WORKFLOW_VALIDATOR:       PriorityLevel.P3,
    EvidenceSource.CONSISTENCY_VALIDATOR:    PriorityLevel.P3,
    EvidenceSource.METRICS_ANALYZER:         PriorityLevel.P4,
    EvidenceSource.EVIDENCE_EXTRACTION:      PriorityLevel.P4,  # supporting evidence only
    EvidenceSource.DIFF_ENGINE:              PriorityLevel.P4,  # supporting evidence only
}

# Priority level ordering — lower index = higher priority
PRIORITY_ORDER: list[PriorityLevel] = [
    PriorityLevel.P1,
    PriorityLevel.P2,
    PriorityLevel.P3,
    PriorityLevel.P4,
    PriorityLevel.P5,
]


def _priority_rank(level: PriorityLevel) -> int:
    """Return numeric rank for a PriorityLevel (lower = higher priority)."""
    return PRIORITY_ORDER.index(level)


def _tiebreak_key(evidence: EvidenceRecord) -> str:
    """
    Tie-break key for same-priority evidence.
    Sorts by rule_id ascending (alphabetical/numeric).
    Evidence without a rule_match sorts last.
    """
    if evidence.rule_match and evidence.rule_match.rule_id:
        return evidence.rule_match.rule_id
    return "zzz"  # no rule_id → sorts to end


# ─────────────────────────────────────────────────────────────────────────────
# Arbiter — stub (implementation on Day 12, P3 wiring on Day 18a)
# ─────────────────────────────────────────────────────────────────────────────

def determine_primary_cause(
    evidence: list[EvidenceRecord],
    run_id: str,
) -> AnalysisBundle:
    """
    Resolve a list of EvidenceRecords into a single deterministic verdict.

    Priority order: P1 → P2 → P3 → P4 → P5 (fallback)
    Tie-break: same priority → lowest rule_id ascending

    Args:
        evidence : All EvidenceRecords collected from all analyzers for this run.
        run_id   : The run this analysis belongs to.

    Returns:
        AnalysisBundle with primary_cause, priority_level, grounded flag,
        and the full evidence list attached.

    Guarantees:
        - Always returns an AnalysisBundle (never raises)
        - Same input always produces same output (deterministic)
        - P5 is returned when evidence is empty or nothing matches

    Implementation: Day 12 (MVP: P2 + P5 only)
                    Day 17 (add P1)
                    Day 18a (wire P3 explicitly — separate code path, not merged into P2)
                    Day 27 (wire P4)
    """
    # ── STUB ─────────────────────────────────────────────────────────────────
    # Full implementation begins Day 12.
    # Pseudocode is documented in the module docstring above.
    # Do NOT implement logic here yet — Day 3 is contracts only.
    raise NotImplementedError(
        "determine_primary_cause() is a stub. "
        "Implementation begins on Day 12. "
        "See module docstring for full pseudocode."
    )


class Arbiter:
    """
    The Arbiter orchestrates all analyzers and produces the final verdict.

    Usage (Day 12+):
        arbiter = Arbiter(analyzers=[evidence_extractor, rule_engine, metrics_analyzer])
        bundle = arbiter.run(trace)

    Implementation: Day 12
    """

    def __init__(self, analyzers: list) -> None:
        """
        Args:
            analyzers: List of objects implementing the Analyzer interface.
                       Order does not matter — priority table determines precedence.
        """
        self.analyzers = analyzers

    def run(self, trace: RunTrace) -> AnalysisBundle:
        """
        Run all registered analyzers on the trace and return the verdict.

        Steps:
            1. Call analyzer.analyze(trace) for each registered analyzer
            2. Collect all EvidenceRecords from all AnalysisResults
            3. Skip any analyzer that returned skipped=True (log the skip)
            4. Pass the full evidence list to determine_primary_cause()
            5. Return the AnalysisBundle

        Implementation: Day 12
        """
        # ── STUB ─────────────────────────────────────────────────────────────
        raise NotImplementedError("Arbiter.run() implementation begins Day 12.")
