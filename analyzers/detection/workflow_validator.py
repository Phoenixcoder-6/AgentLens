"""
analyzers/detection/workflow_validator.py

Implements Workflow rules (skipped steps, wrong order).
"""

from __future__ import annotations

from app.interfaces import AnalysisResult, Analyzer
from schema.models import (
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    RuleMatch,
    RuleSeverity,
    RunTrace,
)


class WorkflowValidator(Analyzer):
    """
    Validates that the pipeline execution followed the expected workflow graph.
    """

    @property
    def analyzer_id(self) -> str:
        return "workflow_validator"

    def analyze(self, trace: RunTrace) -> AnalysisResult:
        if not trace.steps:
            return AnalysisResult(
                skipped=True, skip_reason="No steps in trace", analyzer_id=self.analyzer_id
            )

        evidence: list[EvidenceRecord] = []

        # Load topology from config (Day 40a: supports pipeline.agents + arbiter.workflow)
        from config.topology import get_topology

        topo = get_topology()
        required_agents = topo.required_agent_order()
        handoff_edges = topo.handoff_edges()

        executed_agents = [step.agent for step in trace.steps]

        # Workflow: skipped_step_v1
        for required in required_agents:
            if required not in executed_agents:
                evidence.append(
                    self._make_record(
                        rule_id="skipped_step_v1",
                        category=FailureCategory.WORKFLOW,
                        description=f"Required agent '{required}' is missing from the trace.",
                        agent=required,
                        step_idx=-1,  # Not attached to a specific step since it's missing
                    )
                )

        # Workflow: wrong_order_v1
        # Check each (upstream -> downstream) handoff edge from topology
        for agent_a, agent_b in handoff_edges:
            if agent_a in executed_agents and agent_b in executed_agents:
                idx_a = executed_agents.index(agent_a)
                idx_b = executed_agents.index(agent_b)

                if idx_b < idx_a:
                    evidence.append(
                        self._make_record(
                            rule_id="wrong_order_v1",
                            category=FailureCategory.WORKFLOW,
                            description=f"Agent '{agent_b}' executed before '{agent_a}'.",
                            agent=agent_b,
                            step_idx=idx_b + 1,
                        )
                    )

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
            step=step_idx if step_idx != -1 else None,
        )
        return EvidenceRecord(
            source=EvidenceSource.WORKFLOW_VALIDATOR,
            description=description,
            value="FAIL",
            rule_match=rule,
            agent=agent,
            step=step_idx if step_idx != -1 else None,
            confidence=1.0,
        )

    def run(self, trace: RunTrace) -> AnalysisResult:
        """Standard Analyzer execution alias."""
        return self.analyze(trace)
