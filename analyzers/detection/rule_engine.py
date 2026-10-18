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

import json
import os

from analyzers.evidence_extraction.extractor import EvidenceExtractor, ExtractedEvidence
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


def _prestructured_evidence(step: AgentStep | None) -> ExtractedEvidence | None:
    """Hydrate ExtractedEvidence directly when step.output is pre-structured JSON or in handoff."""
    if step is None:
        return None
    handoff_ev = (getattr(getattr(step, "handoff", None), "output_state", {}) or {}).get(
        "extracted_evidence"
    )
    if isinstance(handoff_ev, dict):
        try:
            return ExtractedEvidence(**handoff_ev)
        except Exception:
            pass
    if not step.output:
        return None
    raw = step.output.strip()
    if not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if (
        isinstance(data, dict)
        and isinstance(data.get("source_count"), int)
        and isinstance(data.get("entity_count"), int)
    ):
        return ExtractedEvidence(
            source_count=data["source_count"],
            entity_count=data["entity_count"],
            claims=list(data.get("claims", [])),
            references=list(data.get("references", [])),
            numbers=list(data.get("numbers", [])),
            dates=list(data.get("dates", [])),
        )
    return None


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
                # Execution: missing_tool_output_v1 (support both 'output' and 'result' keys)
                output = tool_call.get("output") or tool_call.get("result") or ""
                error = tool_call.get("error") or ""

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

        # 2. Reasoning Rules (Day 40a: role-based topology resolution)
        from config.topology import ROLE_GATHERER, ROLE_SYNTHESIZER, get_topology

        topo = get_topology()
        res_step = topo.find_step_for_role(trace.steps, ROLE_GATHERER)
        wr_step = topo.find_step_for_role(trace.steps, ROLE_SYNTHESIZER)
        upstream_step = topo.find_upstream_step(trace.steps, wr_step) or res_step

        res_ev = _prestructured_evidence(res_step)
        wr_ev = _prestructured_evidence(wr_step)
        up_ev = res_ev if upstream_step is res_step else _prestructured_evidence(upstream_step)

        if extractor:
            if res_ev is None and res_step:
                res_ev = extractor.extract(res_step.output, agent=res_step.agent)
            if wr_ev is None and wr_step:
                wr_ev = extractor.extract(wr_step.output, agent=wr_step.agent)
            if up_ev is None and upstream_step:
                up_ev = (
                    res_ev
                    if upstream_step is res_step
                    else extractor.extract(upstream_step.output, agent=upstream_step.agent)
                )

        # Reasoning: researcher_quality_v1 (fires on information_gatherer role)
        if res_step:
            has_res_exec_failure = any(e.agent == res_step.agent for e in evidence)
            if res_ev and not res_ev.extraction_failed and not has_res_exec_failure:
                if res_ev.source_count < min_sources:
                    evidence.append(
                        self._make_record(
                            rule_id="researcher_quality_v1",
                            category=FailureCategory.REASONING,
                            description=(
                                f"Agent '{res_step.agent}' source count ({res_ev.source_count}) "
                                f"below threshold ({min_sources})."
                            ),
                            agent=res_step.agent,
                            step_idx=res_step.step,
                        )
                    )

        # Reasoning: hallucination_v1 (fires on synthesizer role vs its upstream receives_from agent)
        if (
            upstream_step
            and wr_step
            and up_ev
            and wr_ev
            and not up_ev.extraction_failed
            and not wr_ev.extraction_failed
        ):
            entity_gain = wr_ev.entity_count - up_ev.entity_count
            if entity_gain > entity_gain_threshold:
                evidence.append(
                    self._make_record(
                        rule_id="hallucination_v1",
                        category=FailureCategory.REASONING,
                        description=(
                            f"Agent '{wr_step.agent}' hallucinated entities "
                            f"(gain of {entity_gain} over '{upstream_step.agent}')."
                        ),
                        agent=wr_step.agent,
                        step_idx=wr_step.step,
                    )
                )

        # Reasoning: hallucination_v2 (Day 54: Semantic entity set difference & substitution)
        v2_enabled = reasoning_config.get("hallucination_v2_enabled", True)
        invented_threshold = reasoning_config.get("hallucination_invented_threshold", 1)

        if v2_enabled and upstream_step and wr_step:
            # Guard: If upstream had execution failure, don't flag downstream hallucination
            has_up_exec_failure = any(
                e.agent == upstream_step.agent
                and e.rule_match
                and e.rule_match.category == FailureCategory.EXECUTION
                for e in evidence
            )
            if not has_up_exec_failure:
                from analyzers.semantic.entity_tracker import EntityTracker

                delta = EntityTracker.compare_steps(upstream_step, wr_step)
                should_fire = delta.substitution_detected or (
                    delta.has_hallucination
                    and len(delta.invented_entities) >= invented_threshold
                    and (delta.preservation_ratio < 0.9 or delta.hallucination_rate >= 0.3)
                )
                if should_fire:
                    inv_sample = sorted(list(delta.invented_entities))[:5]
                    desc = (
                        f"Agent '{wr_step.agent}' hallucinated invented entities "
                        f"{inv_sample} not found in upstream '{upstream_step.agent}'."
                    )
                    if delta.substitution_detected:
                        desc += " (Entity substitution detected)."
                    evidence.append(
                        self._make_record(
                            rule_id="hallucination_v2",
                            category=FailureCategory.REASONING,
                            description=desc,
                            agent=wr_step.agent,
                            step_idx=wr_step.step,
                        )
                    )

        return AnalysisResult(evidence=evidence, analyzer_id=self.analyzer_id)

    def run(self, trace: RunTrace) -> AnalysisResult:
        """Standard Analyzer execution alias."""
        return self.analyze(trace)

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
