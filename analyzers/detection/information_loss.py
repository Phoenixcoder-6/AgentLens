"""
analyzers/detection/information_loss.py — Information Loss Rule
================================================================
Day 10: One rule that compares ExtractedEvidence from Researcher → Writer
and flags when information was dropped OR unexpectedly added.

Consumes structured ExtractedEvidence from Day 9 — NOT raw text parsing.

Two signals detected:
    DROPPED  — writer count < researcher count (information lost in handoff)
    ADDED    — writer count > researcher count (new info introduced = hallucination risk)

Results are written to the analysis table in SQLite.

Design:
    - Operates on source_count and entity_count only (the two verifiable fields)
    - tool_calls comparison reserved for future rules
    - Returns InformationLossResult with full verdict + confidence
    - Never raises — safe fallback on any error
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from analyzers.evidence_extraction.extractor import ExtractedEvidence
from schema.models import SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldDiff:
    """Diff result for a single numeric field."""
    field_name: str
    researcher_value: int
    writer_value: int
    delta: int                    # writer - researcher (negative = dropped, positive = added)
    signal: str                   # "DROPPED" | "ADDED" | "PRESERVED"
    severity: str                 # "HIGH" | "MEDIUM" | "LOW" | "NONE"


@dataclass
class InformationLossResult:
    """
    Full verdict for the Information Loss rule on one Researcher → Writer handoff.

    schema_version is stamped so records are traceable.
    """
    schema_version: str
    run_id: str
    rule_id: str = "information_loss_v1"
    rule_category: str = "workflow"

    # Per-field diffs
    source_diff: Optional[FieldDiff] = None
    entity_diff: Optional[FieldDiff] = None

    # Overall verdict
    verdict: str = "PASS"          # "PASS" | "WARNING" | "FAIL"
    confidence: float = 1.0        # 0.0–1.0
    summary: str = ""

    # Flags
    has_information_loss: bool = False
    has_information_gain: bool = False
    rule_failed: bool = False
    error_message: Optional[str] = None

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# InformationLossRule
# ─────────────────────────────────────────────────────────────────────────────

class InformationLossRule:
    """
    Compares ExtractedEvidence from Researcher and Writer to detect
    information loss or unexpected information gain.

    Usage:
        rule = InformationLossRule()
        result = rule.evaluate(
            run_id="run_abc",
            researcher_evidence=ev_researcher,
            writer_evidence=ev_writer,
        )
        print(result.verdict)   # "PASS" | "WARNING" | "FAIL"
        print(result.summary)
    """

    # How much delta triggers each severity level
    _SEVERE_THRESHOLD = 3      # |delta| >= 3 → HIGH severity
    _MODERATE_THRESHOLD = 1    # |delta| >= 1 → MEDIUM severity

    def evaluate(
        self,
        run_id: str,
        researcher_evidence: ExtractedEvidence,
        writer_evidence: ExtractedEvidence,
    ) -> InformationLossResult:
        """
        Run the information loss rule.

        Args:
            run_id:               The pipeline run ID (for storage linkage).
            researcher_evidence:  ExtractedEvidence from the Researcher step.
            writer_evidence:      ExtractedEvidence from the Writer step.

        Returns:
            InformationLossResult with verdict, per-field diffs, and summary.
        """
        result = InformationLossResult(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
        )

        # If either extraction failed, we can't run the rule reliably
        if researcher_evidence.extraction_failed or writer_evidence.extraction_failed:
            result.rule_failed = True
            result.verdict = "PASS"  # don't flag when evidence is unreliable
            result.confidence = 0.0
            result.error_message = (
                f"Researcher extraction_failed={researcher_evidence.extraction_failed}, "
                f"Writer extraction_failed={writer_evidence.extraction_failed}"
            )
            result.summary = "Rule skipped — evidence extraction failed on one or both steps."
            return result

        # ── Compute per-field diffs ───────────────────────────────────────
        source_diff = self._compute_diff(
            "source_count",
            researcher_evidence.source_count,
            writer_evidence.source_count,
        )
        entity_diff = self._compute_diff(
            "entity_count",
            researcher_evidence.entity_count,
            writer_evidence.entity_count,
        )

        result.source_diff = source_diff
        result.entity_diff = entity_diff

        # ── Determine overall verdict ─────────────────────────────────────
        dropped_fields = [
            d for d in [source_diff, entity_diff] if d.signal == "DROPPED"
        ]
        added_fields = [
            d for d in [source_diff, entity_diff] if d.signal == "ADDED"
        ]

        result.has_information_loss = len(dropped_fields) > 0
        result.has_information_gain = len(added_fields) > 0

        # Verdict logic:
        #   FAIL    — any field DROPPED with HIGH severity
        #   WARNING — any DROPPED (moderate) OR any ADDED (high/moderate)
        #   PASS    — all fields preserved or minor delta
        high_severity_drops = [d for d in dropped_fields if d.severity == "HIGH"]
        any_drops = len(dropped_fields) > 0
        high_severity_gains = [d for d in added_fields if d.severity in ("HIGH", "MEDIUM")]

        if high_severity_drops:
            result.verdict = "FAIL"
        elif any_drops or high_severity_gains:
            result.verdict = "WARNING"
        else:
            result.verdict = "PASS"

        # Scalable confidence — reflects magnitude of delta, not just verdict bucket
        result.confidence = self._compute_confidence(source_diff, entity_diff, result.verdict)

        result.summary = self._build_summary(source_diff, entity_diff, result.verdict)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_diff(
        self, field_name: str, researcher_val: int, writer_val: int
    ) -> FieldDiff:
        """Compute diff between researcher and writer for one numeric field."""
        delta = writer_val - researcher_val

        if delta < 0:
            signal = "DROPPED"
        elif delta > 0:
            signal = "ADDED"
        else:
            signal = "PRESERVED"

        severity = self._severity(abs(delta), signal)

        return FieldDiff(
            field_name=field_name,
            researcher_value=researcher_val,
            writer_value=writer_val,
            delta=delta,
            signal=signal,
            severity=severity,
        )

    def _severity(self, abs_delta: int, signal: str) -> str:
        """Map absolute delta to severity level."""
        if abs_delta == 0:
            return "NONE"
        if abs_delta >= self._SEVERE_THRESHOLD:
            return "HIGH"
        if abs_delta >= self._MODERATE_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _compute_confidence(
        source_diff: FieldDiff,
        entity_diff: FieldDiff,
        verdict: str,
    ) -> float:
        """
        Scalable confidence — scales with delta magnitude, not a fixed lookup.

        Formula:
            ratio    = abs(delta) / max(researcher_count, 1)   # 0.0 → 1.0
            avg_ratio = mean(source_ratio, entity_ratio)

            PASS     → 1.0
            FAIL     → clamp(0.80 + avg_ratio × 0.19, 0.80, 0.99)
            WARNING  → clamp(0.60 + avg_ratio × 0.24, 0.60, 0.84)

        Examples:
            source 8→7  (small drop):   WARNING, ratio=0.125 → confidence=0.63
            source 8→4  (half dropped): FAIL,    ratio=0.500 → confidence=0.895
            source 8→0  (all dropped):  FAIL,    ratio=1.000 → confidence=0.99
            source 8→11 (gain 3):       WARNING, ratio=0.375 → confidence=0.69
        """
        if verdict == "PASS":
            return 1.0

        # Ratio per field: how large is the change relative to baseline?
        source_ratio = abs(source_diff.delta) / max(source_diff.researcher_value, 1)
        entity_ratio = abs(entity_diff.delta) / max(entity_diff.researcher_value, 1)
        avg_ratio = (source_ratio + entity_ratio) / 2.0

        if verdict == "FAIL":
            # Range: [0.80, 0.99]
            raw = 0.80 + avg_ratio * 0.19
            return round(min(0.99, max(0.80, raw)), 4)
        else:  # WARNING
            # Range: [0.60, 0.84]
            raw = 0.60 + avg_ratio * 0.24
            return round(min(0.84, max(0.60, raw)), 4)

    @staticmethod
    def _build_summary(
        source_diff: FieldDiff,
        entity_diff: FieldDiff,
        verdict: str,
    ) -> str:
        """Build a human-readable summary of the rule result."""
        lines = [f"Verdict: {verdict}"]
        for diff in [source_diff, entity_diff]:
            if diff.signal == "DROPPED":
                lines.append(
                    f"  {diff.field_name}: {diff.researcher_value} → {diff.writer_value} "
                    f"(DROPPED {abs(diff.delta)}, severity={diff.severity})"
                )
            elif diff.signal == "ADDED":
                lines.append(
                    f"  {diff.field_name}: {diff.researcher_value} → {diff.writer_value} "
                    f"(ADDED {diff.delta}, severity={diff.severity})"
                )
            else:
                lines.append(
                    f"  {diff.field_name}: {diff.researcher_value} → {diff.writer_value} "
                    f"(PRESERVED)"
                )
        return "\n".join(lines)
