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

import json
import os

from analyzers.detection.rule_engine import _prestructured_evidence
from analyzers.evidence_extraction.extractor import EvidenceExtractor
from app.interfaces import AnalysisResult, Analyzer
from config import config_loader
from schema.models import (
    AgentStep,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    RuleMatch,
    RuleSeverity,
    RunTrace,
)


def _is_rubber_stamp_verifier(step: AgentStep | None) -> bool:
    """Return True when a verifier step explicitly rubber-stamps flagged/unverified content."""
    if step is None or not step.output:
        return False
    raw = step.output.strip()
    if not raw.startswith("{"):
        return False
    try:
        data = json.loads(raw)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    approved = (
        data.get("verified") is True
        or data.get("approved") is True
        or str(data.get("verification_result", "")).upper() == "APPROVED"
    )
    if not approved:
        return False
    return bool(
        data.get("always_approve") is True
        or int(data.get("unverified_claims", 0) or 0) > 0
        or int(data.get("flagged_claims", 0) or 0) > 0
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
        entity_gain_threshold: int = reasoning_cfg.get("hallucination_entity_gain_threshold", 0)

        # ── Run LLM extraction only if API key present ────────────────────────
        extractor = None
        if os.getenv("GROQ_API_KEY"):
            extractor = EvidenceExtractor()

        # Identify agent steps by configured topology role (Day 40a)
        from config.topology import (
            ROLE_CHECKER,
            ROLE_GATHERER,
            ROLE_SYNTHESIZER,
            get_topology,
        )

        topo = get_topology()
        res_step = topo.find_step_for_role(trace.steps, ROLE_GATHERER)
        wr_step = topo.find_step_for_role(trace.steps, ROLE_SYNTHESIZER)
        ver_step = topo.find_step_for_role(trace.steps, ROLE_CHECKER)
        upstream_step = topo.find_upstream_step(trace.steps, wr_step) or res_step

        res_ev = _prestructured_evidence(upstream_step)
        wr_ev = _prestructured_evidence(wr_step)
        ver_ev = _prestructured_evidence(ver_step)

        if extractor:
            if res_ev is None and upstream_step:
                res_ev = extractor.extract(upstream_step.output, agent=upstream_step.agent)
            if wr_ev is None and wr_step:
                wr_ev = extractor.extract(wr_step.output, agent=wr_step.agent)
            if ver_ev is None and ver_step:
                ver_ev = extractor.extract(ver_step.output, agent=ver_step.agent)

        # ── Rule: verifier_passthrough_v1 ─────────────────────────────────────
        # Fires when the quality_checker's entity count equals the synthesizer's entity count
        # AND the synthesizer had hallucinated new entities (gain > threshold).
        passthrough_fired = False
        if (
            upstream_step
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
            if wr_ev.entity_count == ver_ev.entity_count and entity_gain > entity_gain_threshold:
                passthrough_fired = True
                evidence.append(
                    self._make_record(
                        rule_id="verifier_passthrough_v1",
                        category=FailureCategory.VERIFICATION,
                        description=(
                            f"Agent '{ver_step.agent}' passed through {entity_gain} hallucinated "
                            "entities without flagging them."
                        ),
                        agent=ver_step.agent,
                        step_idx=ver_step.step,
                    )
                )

        if not passthrough_fired and ver_step and _is_rubber_stamp_verifier(ver_step):
            evidence.append(
                self._make_record(
                    rule_id="verifier_passthrough_v1",
                    category=FailureCategory.VERIFICATION,
                    description=(
                        f"Always-approve quality checker '{ver_step.agent}' rubber-stamped output "
                        "without rejecting flagged or unverified claims."
                    ),
                    agent=ver_step.agent,
                    step_idx=ver_step.step,
                )
            )

        # ── Rule: claim_drift_v1 ──────────────────────────────────────────────
        # Fires when the synthesizer introduces claims that were NOT in the upstream
        # gatherer output, indicating unsupported or hallucinated factual claims.
        if (
            upstream_step
            and wr_step
            and res_ev is not None
            and wr_ev is not None
            and not res_ev.extraction_failed
            and not wr_ev.extraction_failed
            and res_ev.claims  # only fire if upstream gatherer had extractable claims
        ):
            res_claim_set = set(c.lower().strip() for c in res_ev.claims)
            wr_claim_set = set(c.lower().strip() for c in wr_ev.claims)

            new_claims = wr_claim_set - res_claim_set
            if new_claims:
                evidence.append(
                    self._make_record(
                        rule_id="claim_drift_v1",
                        category=FailureCategory.VERIFICATION,
                        description=(
                            f"Agent '{wr_step.agent}' introduced {len(new_claims)} claim(s) not found "
                            f"in '{upstream_step.agent}' output: {list(new_claims)[:3]}"
                        ),
                        agent=wr_step.agent,
                        step_idx=wr_step.step,
                    )
                )

        return AnalysisResult(evidence=evidence, analyzer_id=self.analyzer_id)

    def run(self, trace: RunTrace) -> AnalysisResult:
        """Standard Analyzer execution alias."""
        return self.analyze(trace)

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
