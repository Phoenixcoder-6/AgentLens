"""
analyzers/detection/consistency_validator.py — Day 22
======================================================
Implements Verification & cross-step consistency rules.

Extracted from rule_engine.py so each file has a single, clear
responsibility:

  rule_engine.py            → Execution + Reasoning rules
  workflow_validator.py     → Workflow / ordering rules
  consistency_validator.py  → Verification + claim consistency rules (this file)

Rules implemented:
  verifier_passthrough_v1   — verifier approved hallucinated entities unchanged
  claim_drift_v1            — key claims changed between researcher → writer
                              (uses Day-20 ExtractedEvidence.claims)

Implements the Analyzer interface; safe to register in the Arbiter alongside
RuleEngine and WorkflowValidator.
"""

from __future__ import annotations

import os

from analyzers.evidence_extraction.extractor import EvidenceExtractor
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


class ConsistencyValidator(Analyzer):
    """
    Validates cross-step evidence consistency and verifier behaviour.

    Day 22 rules:
      - verifier_passthrough_v1 : verifier passed hallucinated entities unchanged
      - claim_drift_v1          : claims added/dropped between researcher → writer
                                  (requires Day-20 extraction data)

    When the GROQ_API_KEY is absent, extraction-dependent rules are skipped
    gracefully — the same guard used in rule_engine.py.
    """

    @property
    def analyzer_id(self) -> str:
        return "consistency_validator"

    def analyze(self, trace: RunTrace) -> AnalysisResult:
        if not trace.steps:
            return AnalysisResult(
                skipped=True,
                skip_reason="No steps in trace",
                analyzer_id=self.analyzer_id,
            )

        evidence: list[EvidenceRecord] = []

        # Load config thresholds
        reasoning_cfg = config_loader.get("arbiter", "reasoning", {})
        entity_gain_threshold: int = reasoning_cfg.get(
            "hallucination_entity_gain_threshold", 0
        )

        # ── Run LLM extraction only if API key present ────────────────────────
        extractor = None
        if os.getenv("GROQ_API_KEY"):
            extractor = EvidenceExtractor()

        # Identify agent steps
        researcher_steps = [s for s in trace.steps if s.agent == "researcher"]
        writer_steps = [s for s in trace.steps if s.agent == "writer"]
        verifier_steps = [s for s in trace.steps if s.agent == "verifier"]

        res_step = researcher_steps[-1] if researcher_steps else None
        wr_step = writer_steps[-1] if writer_steps else None
        ver_step = verifier_steps[-1] if verifier_steps else None

        res_ev = None
        wr_ev = None
        ver_ev = None

        if extractor:
            if res_step:
                res_ev = extractor.extract(res_step.output, agent="researcher")
            if wr_step:
                wr_ev = extractor.extract(wr_step.output, agent="writer")
            if ver_step:
                ver_ev = extractor.extract(ver_step.output, agent="verifier")

        # ── Rule: verifier_passthrough_v1 ─────────────────────────────────────
        # Fires when the verifier's entity count equals the writer's entity count
        # AND the writer had hallucinated new entities (gain > threshold).
        # This means the verifier "passed through" hallucinated content unchanged.
        if (
            res_step
            and wr_step
            and ver_step
            and res_ev is not None
            and wr_ev is not None
            and ver_ev is not None
            and not res_ev.extraction_failed
            and not wr_ev.extraction_failed
            and not ver_ev.extraction_failed
        ):
            entity_gain = wr_ev.entity_count - res_ev.entity_count
            if (
                wr_ev.entity_count == ver_ev.entity_count
                and entity_gain > entity_gain_threshold
            ):
                evidence.append(
                    self._make_record(
                        rule_id="verifier_passthrough_v1",
                        category=FailureCategory.VERIFICATION,
                        description=(
                            f"Verifier passed through {entity_gain} hallucinated "
                            "entities without flagging them."
                        ),
                        agent="verifier",
                        step_idx=ver_step.step,
                    )
                )

        # ── Rule: claim_drift_v1 ──────────────────────────────────────────────
        # Fires when the writer introduces claims that were NOT in the researcher
        # output, indicating unsupported or hallucinated factual claims.
        # Requires Day-20 extraction: both res_ev.claims and wr_ev.claims.
        if (
            res_step
            and wr_step
            and res_ev is not None
            and wr_ev is not None
            and not res_ev.extraction_failed
            and not wr_ev.extraction_failed
            and res_ev.claims  # only fire if researcher had extractable claims
        ):
            res_claim_set = set(c.lower().strip() for c in res_ev.claims)
            wr_claim_set = set(c.lower().strip() for c in wr_ev.claims)

            # Claims in writer output not present in researcher output
            new_claims = wr_claim_set - res_claim_set
            if new_claims:
                evidence.append(
                    self._make_record(
                        rule_id="claim_drift_v1",
                        category=FailureCategory.VERIFICATION,
                        description=(
                            f"Writer introduced {len(new_claims)} claim(s) not found "
                            f"in researcher output: {list(new_claims)[:3]}"
                        ),
                        agent="writer",
                        step_idx=wr_step.step,
                    )
                )

        return AnalysisResult(evidence=evidence, analyzer_id=self.analyzer_id)

    def _make_record(
        self,
        rule_id: str,
        category: FailureCategory,
        description: str,
        agent: str,
        step_idx: int,
    ) -> EvidenceRecord:
        rule = RuleMatch(
            rule_id=rule_id,
            rule_version="1.0.0",
            category=category,
            description=description,
            severity=RuleSeverity.HIGH,
            agent=agent,
            step=step_idx,
        )
        return EvidenceRecord(
            source=EvidenceSource.RULE_ENGINE,
            description=description,
            value="FAIL",
            rule_match=rule,
            agent=agent,
            step=step_idx,
            confidence=1.0,
        )
