"""
analyzers/detection/rule_engine.py — Day 18, updated Day 22
============================================================
Implements Execution and Reasoning deterministic rules only.

Day 22: Verification rules (verifier_passthrough_v1) moved to
        consistency_validator.py. RuleEngine now has a single,
        clear scope: Execution + Reasoning.

Rules:
  Execution:  missing_tool_output_v1, tool_failure_v1
  Reasoning:  researcher_quality_v1, hallucination_v1
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


class RuleEngine(Analyzer):
    """
    Evaluates Execution, Reasoning, and Verification deterministic rules.
    """

    @property
    def analyzer_id(self) -> str:
        return "rule_engine"

    def analyze(self, trace: RunTrace) -> AnalysisResult:
        if not trace.steps:
            return AnalysisResult(
                skipped=True, skip_reason="No steps in trace", analyzer_id=self.analyzer_id
            )

        evidence: list[EvidenceRecord] = []

        # Load config
        reasoning_config = config_loader.get("arbiter", "reasoning", {})
        min_sources = reasoning_config.get("researcher_min_sources", 1)
        entity_gain_threshold = reasoning_config.get("hallucination_entity_gain_threshold", 0)

        extractor = EvidenceExtractor() if os.getenv("GROQ_API_KEY") else None

        # 1. Execution Rules
        for step in trace.steps:
            for tool_call in step.tool_calls:
                # Execution: missing_tool_output_v1
                output = tool_call.get("output", "")
                error = tool_call.get("error", "")

                if not output and not error:
                    evidence.append(
                        self._make_record(
                            rule_id="missing_tool_output_v1",
                            category=FailureCategory.EXECUTION,
                            description=f"Agent '{step.agent}' called tool '{tool_call.get('name')}' but no output was recorded.",
                            agent=step.agent,
                            step_idx=step.step,
                        )
                    )

                # Execution: tool_failure_v1
                if error or (
                    isinstance(output, str) and ("Error:" in output or "Exception:" in output)
                ):
                    evidence.append(
                        self._make_record(
                            rule_id="tool_failure_v1",
                            category=FailureCategory.EXECUTION,
                            description=f"Tool '{tool_call.get('name')}' returned an error.",
                            agent=step.agent,
                            step_idx=step.step,
                        )
                    )

        # 2. Reasoning Rules
        if extractor:
            # Find researcher and writer steps
            researcher_steps = [s for s in trace.steps if s.agent == "researcher"]
            writer_steps = [s for s in trace.steps if s.agent == "writer"]
            # Note: verifier step collection moved to consistency_validator.py (Day 22)

            res_step = researcher_steps[-1] if researcher_steps else None
            wr_step = writer_steps[-1] if writer_steps else None

            res_ev = extractor.extract(res_step.output, agent="researcher") if res_step else None
            wr_ev = extractor.extract(wr_step.output, agent="writer") if wr_step else None
            # Note: verifier extraction is done in consistency_validator.py (Day 22 split)

            # Reasoning: researcher_quality_v1
            if res_step and res_ev and not res_ev.extraction_failed:
                if res_ev.source_count < min_sources:
                    evidence.append(
                        self._make_record(
                            rule_id="researcher_quality_v1",
                            category=FailureCategory.REASONING,
                            description=f"Researcher source count ({res_ev.source_count}) below threshold ({min_sources}).",
                            agent="researcher",
                            step_idx=res_step.step,
                        )
                    )

            # Reasoning: hallucination_v1
            if (
                res_step
                and wr_step
                and res_ev
                and wr_ev
                and not res_ev.extraction_failed
                and not wr_ev.extraction_failed
            ):
                entity_gain = wr_ev.entity_count - res_ev.entity_count
                if entity_gain > entity_gain_threshold:
                    evidence.append(
                        self._make_record(
                            rule_id="hallucination_v1",
                            category=FailureCategory.REASONING,
                            description=f"Writer hallucinated entities (gain of {entity_gain}).",
                            agent="writer",
                            step_idx=wr_step.step,
                        )
                    )

            # Note: verifier_passthrough_v1 and claim_drift_v1 are owned by
            # consistency_validator.py (Day 22 split).

        return AnalysisResult(evidence=evidence, analyzer_id=self.analyzer_id)

    def _make_record(
        self, rule_id: str, category: FailureCategory, description: str, agent: str, step_idx: int
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
