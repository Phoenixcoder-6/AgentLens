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
Day 12 MVP: P2 (rule match) + P5 (unknown fallback) only.
P1 → Day 17 | P3 → Day 18a | P4 → Day 27
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from analyzers.detection.information_loss import InformationLossResult
from config.logging_config import get_logger
from schema.models import (
    AnalysisBundle,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    PriorityLevel,
    RuleMatch,
    RuleSeverity,
)

log = get_logger("arbiter")

# ─────────────────────────────────────────────────────────────────────────────
# Priority resolution helpers (locked — no changes after Day 3)
# ─────────────────────────────────────────────────────────────────────────────

# Maps EvidenceSource → PriorityLevel for the Arbiter's decision logic.
SOURCE_TO_PRIORITY: dict[EvidenceSource, PriorityLevel] = {
    EvidenceSource.GROUND_TRUTH: PriorityLevel.P1,
    EvidenceSource.RULE_ENGINE: PriorityLevel.P2,
    EvidenceSource.WORKFLOW_VALIDATOR: PriorityLevel.P3,
    EvidenceSource.CONSISTENCY_VALIDATOR: PriorityLevel.P3,
    EvidenceSource.METRICS_ANALYZER: PriorityLevel.P4,
    EvidenceSource.STATISTICAL_ANOMALY: PriorityLevel.P4,  # Day 27 — stddev outlier detector
    EvidenceSource.EVIDENCE_EXTRACTION: PriorityLevel.P4,  # supporting evidence only
    EvidenceSource.DIFF_ENGINE: PriorityLevel.P4,  # supporting evidence only
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
# Evidence conversion helper
# ─────────────────────────────────────────────────────────────────────────────


def evidence_from_information_loss(result: InformationLossResult) -> EvidenceRecord | None:
    """
    Convert an InformationLossResult (Day 10) into an EvidenceRecord for the Arbiter.

    Returns None if:
        - The rule was skipped (extraction_failed)
        - The verdict is PASS (nothing to report)
    """
    if result.rule_failed or result.verdict == "PASS":
        return None

    # Map verdict → failure category + severity
    if result.verdict == "FAIL":
        category = FailureCategory.WORKFLOW
        severity = RuleSeverity.HIGH
        src_r = result.source_diff.researcher_value if result.source_diff else "N/A"
        src_w = result.source_diff.writer_value if result.source_diff else "N/A"
        ent_r = result.entity_diff.researcher_value if result.entity_diff else "N/A"
        ent_w = result.entity_diff.writer_value if result.entity_diff else "N/A"
        description = (
            f"Information loss detected: sources or entities were dropped "
            f"in the Researcher → Writer handoff. "
            f"source: {src_r}→{src_w}, "
            f"entity: {ent_r}→{ent_w}"
        )
    else:  # WARNING
        category = FailureCategory.REASONING
        severity = RuleSeverity.MEDIUM
        src_r = result.source_diff.researcher_value if result.source_diff else "N/A"
        src_w = result.source_diff.writer_value if result.source_diff else "N/A"
        ent_r = result.entity_diff.researcher_value if result.entity_diff else "N/A"
        ent_w = result.entity_diff.writer_value if result.entity_diff else "N/A"
        description = (
            f"Information gain detected: Writer introduced sources or entities "
            f"not present in research findings (hallucination risk). "
            f"source: {src_r}→{src_w}, "
            f"entity: {ent_r}→{ent_w}"
        )

    rule = RuleMatch(
        rule_id=result.rule_id,
        category=category,
        description=description,
        severity=severity,
        agent="writer",
        evidence_detail=result.summary,
    )

    return EvidenceRecord(
        source=EvidenceSource.RULE_ENGINE,
        description=description,
        value=result.verdict,
        rule_match=rule,
        agent="writer",
        confidence=result.confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grounded Check
# ─────────────────────────────────────────────────────────────────────────────


def compute_grounded(evidence: list[EvidenceRecord]) -> bool:
    """
    Returns True if at least one P1 (Ground Truth) evidence record exists.
    Otherwise False.
    """
    return any(SOURCE_TO_PRIORITY.get(e.source) == PriorityLevel.P1 for e in evidence)


# ─────────────────────────────────────────────────────────────────────────────
# determine_primary_cause — Day 12 MVP: P2 + P5 only
# ─────────────────────────────────────────────────────────────────────────────


def determine_primary_cause(
    evidence: list[EvidenceRecord],
    run_id: str,
) -> AnalysisBundle:
    """
    Resolve a list of EvidenceRecords into a single deterministic verdict.

    Day 12 MVP: P2 (rule match) and P5 (unknown fallback) only.
    P1/P3/P4 slots are explicit reserved no-ops — separate code paths,
    ready to be filled without restructuring on future days.

    Priority order (Day 12): P2 → P5
    Tie-break: same priority → lowest rule_id ascending (alphabetical)

    Guarantees:
        - Always returns an AnalysisBundle (never raises)
        - Same evidence list → same output, every time (deterministic)
        - P5 is returned when evidence is empty or nothing matches P2
    """
    # ── P5 immediate fallback for empty evidence ──────────────────────────────
    if not evidence:
        return _make_bundle(
            run_id=run_id,
            primary_cause=FailureCategory.UNKNOWN,
            priority=PriorityLevel.P5,
            grounded=False,
            evidence=[],
            primary_agent=None,
            verdict_reason="No evidence collected — P5 fallback.",
        )

    is_grounded = compute_grounded(evidence)

    # ── P1: ground truth mismatch (Day 17) ────────────────────────────────────
    p1_evidence = [e for e in evidence if e.source == EvidenceSource.GROUND_TRUTH]
    if p1_evidence:
        best = sorted(p1_evidence, key=_tiebreak_key)[0]
        category = best.rule_match.category if best.rule_match else FailureCategory.UNKNOWN
        return _make_bundle(
            run_id=run_id,
            primary_cause=category,
            priority=PriorityLevel.P1,
            grounded=True,
            evidence=evidence,
            primary_agent=best.agent,
            verdict_reason=(
                f"P1 ground truth mismatch: {best.rule_match.rule_id if best.rule_match else 'unknown'} "
                f"(confidence={best.confidence:.0%})"
            ),
        )

    # ── P2: rule match (deterministic rules fired) ────────────────────────────
    p2_evidence = [e for e in evidence if e.source == EvidenceSource.RULE_ENGINE]
    if p2_evidence:
        # Tie-break: sort by rule_id ascending — guarantees determinism
        best = sorted(p2_evidence, key=_tiebreak_key)[0]
        category = best.rule_match.category if best.rule_match else FailureCategory.UNKNOWN
        return _make_bundle(
            run_id=run_id,
            primary_cause=category,
            priority=PriorityLevel.P2,
            grounded=is_grounded,
            evidence=evidence,
            primary_agent=best.agent,
            verdict_reason=(
                f"P2 rule match: {best.rule_match.rule_id if best.rule_match else 'unknown'} "
                f"(confidence={best.confidence:.0%})"
            ),
        )

    # ── P3: workflow violation ────────────────────────────────────────────────
    p3_evidence = [e for e in evidence if e.source == EvidenceSource.WORKFLOW_VALIDATOR]
    if p3_evidence:
        best = sorted(p3_evidence, key=_tiebreak_key)[0]
        return _make_bundle(
            run_id=run_id,
            primary_cause=FailureCategory.WORKFLOW,
            priority=PriorityLevel.P3,
            grounded=is_grounded,
            evidence=evidence,
            primary_agent=best.agent,
            verdict_reason=f"P3 Workflow violation: {best.rule_match.rule_id if best.rule_match else 'unknown'} (confidence={best.confidence:.0%})",
        )

    # ── P4: statistical anomaly (reserved — Day 27) ───────────────────────────
    # p4_evidence = [e for e in evidence if e.source == EvidenceSource.METRICS_ANALYZER]
    # → Day 27

    # ── P5: fallback — evidence present but nothing matched ───────────────────
    return _make_bundle(
        run_id=run_id,
        primary_cause=FailureCategory.UNKNOWN,
        priority=PriorityLevel.P5,
        grounded=is_grounded,
        evidence=evidence,
        primary_agent=None,
        verdict_reason="No P2 rule matched — P5 fallback.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Arbiter class
# ─────────────────────────────────────────────────────────────────────────────


class Arbiter:
    """
    The Arbiter resolves EvidenceRecords into a final deterministic verdict.

    Day 12 MVP: accepts evidence directly via run().
    Evidence conversion helpers (e.g. evidence_from_information_loss) translate
    analyzer results into EvidenceRecord format before calling run().

    Usage:
        from analyzers.arbiter import Arbiter, evidence_from_information_loss

        loss_ev = evidence_from_information_loss(loss_result)
        evidence = [ev for ev in [loss_ev] if ev is not None]

        bundle = Arbiter().run(run_id="run_abc", evidence=evidence)
        print(bundle.primary_cause)    # FailureCategory.REASONING
        print(bundle.priority_level)   # PriorityLevel.P2
        print(bundle.grounded)         # False
    """

    def run(
        self,
        run_id: str,
        evidence: list[EvidenceRecord],
    ) -> AnalysisBundle:
        """
        Resolve evidence into a final verdict.

        Input evidence is sorted before processing to guarantee that the
        outcome is independent of the order in which evidence was collected.

        Args:
            run_id:   The pipeline run ID.
            evidence: EvidenceRecords from all analyzers for this run.

        Returns:
            AnalysisBundle — always (never raises).
        """
        # Sort for determinism: priority first, then rule_id tie-break
        # This ensures input ordering NEVER affects the output verdict
        sorted_evidence = sorted(
            evidence,
            key=lambda e: (
                _priority_rank(SOURCE_TO_PRIORITY.get(e.source, PriorityLevel.P5)),
                _tiebreak_key(e),
            ),
        )
        bundle = determine_primary_cause(sorted_evidence, run_id)

        log.info(
            "Analysis completed",
            extra={
                "extra_fields": {
                    "run_id": run_id,
                    "rules_fired": len(bundle.rule_matches),
                    "evidence_count": len(bundle.evidence),
                    "verdict": bundle.primary_cause.value
                    if hasattr(bundle.primary_cause, "value")
                    else str(bundle.primary_cause),
                    "primary_agent": bundle.primary_agent,
                    "grounded": bundle.grounded,
                }
            },
        )

        return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────


def _make_bundle(
    run_id: str,
    primary_cause: FailureCategory,
    priority: PriorityLevel,
    grounded: bool,
    evidence: list[EvidenceRecord],
    primary_agent: str | None,
    verdict_reason: str,
) -> AnalysisBundle:
    """Build a complete AnalysisBundle with all fields populated."""
    rule_matches = [e.rule_match for e in evidence if e.rule_match is not None]
    return AnalysisBundle(
        run_id=run_id,
        primary_cause=primary_cause,
        priority_level=priority,
        grounded=grounded,
        evidence=evidence,
        rule_matches=rule_matches,
        primary_agent=primary_agent,
        summary=verdict_reason,
    )
