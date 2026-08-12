"""
analyzers/detection/ground_truth.py
Implements the P1 Ground Truth Validator.
Compares the final pipeline output with expected_output (if present)
and triggers a P1 evidence record if they diverge.
"""

from __future__ import annotations

import difflib

from app.interfaces import AnalysisResult, Analyzer
from config import config_loader
from schema.models import (
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    RuleMatch,
    RuleSeverity,
    RunTrace,
)


class GroundTruthValidator(Analyzer):
    """
    Checks if the run has an expected_output. If so, compares it to the
    pipeline's actual final output. If the string similarity is below
    the configured threshold, it returns a P1 evidence record.
    """

    @property
    def analyzer_id(self) -> str:
        return "ground_truth_validator"

    def analyze(self, trace: RunTrace) -> AnalysisResult:
        if not trace.expected_output:
            return AnalysisResult(
                skipped=True,
                skip_reason="No expected_output present in trace.",
                analyzer_id="ground_truth_validator",
            )

        if not trace.steps:
            return AnalysisResult(
                skipped=True,
                skip_reason="No steps present in trace.",
                analyzer_id="ground_truth_validator",
            )

        # Get final output from the last agent step
        final_step = trace.steps[-1]
        actual_output = final_step.output

        # Compare strings
        similarity = difflib.SequenceMatcher(None, trace.expected_output, actual_output).ratio()

        # Load threshold. config_loader.get("arbiter") returns the arbiter section dict
        arbiter_config = config_loader.get("arbiter") or {}
        ground_truth_config = arbiter_config.get("ground_truth", {})
        threshold = ground_truth_config.get("p1_similarity_threshold", 0.85)

        if similarity < threshold:
            rule = RuleMatch(
                rule_id="gt_mismatch_v1",
                category=FailureCategory.REASONING,
                description=(
                    f"Pipeline final output diverged from expected output. "
                    f"Similarity: {similarity:.2f} (Threshold: {threshold:.2f})"
                ),
                severity=RuleSeverity.CRITICAL,
                agent=final_step.agent,
                evidence_detail=f"Expected:\n{trace.expected_output}\n\nActual:\n{actual_output}",
            )

            evidence = EvidenceRecord(
                source=EvidenceSource.GROUND_TRUTH,
                description="Ground truth mismatch detected.",
                value="FAIL",
                rule_match=rule,
                agent=final_step.agent,
                confidence=1.0 - similarity,  # Lower similarity -> higher confidence of mismatch
            )
            return AnalysisResult(
                evidence=[evidence],
                analyzer_id="ground_truth_validator",
            )

        return AnalysisResult(
            evidence=[],
            analyzer_id="ground_truth_validator",
            skipped=False,
        )
