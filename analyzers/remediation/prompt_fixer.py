"""
analyzers/remediation/prompt_fixer.py — Automated Prompt Auto-Diff Engine
==========================================================================
Day 64: Generates targeted prompt constraints, GitHub-style unified diffs,
and regression-avoidance scores by comparing failed agent runs against
historical clean executions.

Classes:
  - ConstraintTemplate: Definition of a targeted constraint rule.
  - PromptFixer: Engine that analyzes failures and generates RemediationPatches.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

from schema.models import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    RemediationPatch,
    RuleMatch,
    RunTrace,
    StepStatus,
)
from storage.db import DatabaseManager


@dataclass(frozen=True)
class ConstraintTemplate:
    """A parameterized prompt constraint template."""

    constraint_type: str
    rule_prefix: str
    category: FailureCategory
    name: str
    constraint_text: str
    explanation: str
    default_regression_risk: float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Targeted Constraint Catalog
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINT_CATALOG: list[ConstraintTemplate] = [
    ConstraintTemplate(
        constraint_type="SOURCE_PRESERVATION",
        rule_prefix="R-IL",
        category=FailureCategory.REASONING,
        name="Source Citation & Evidence Retention",
        constraint_text=(
            "CRITICAL CONSTRAINT: You must preserve all source citations ([1], [2], ...) "
            "and retain all factual claims and numerical figures from the input research findings. "
            "Do not omit, condense away, or drop any verified sources."
        ),
        explanation=(
            "Forces the synthesizer agent to retain all citations and facts from upstream gatherers, "
            "directly preventing information loss failures (R-IL)."
        ),
        default_regression_risk=0.05,
    ),
    ConstraintTemplate(
        constraint_type="ENTITY_FIDELITY",
        rule_prefix="R-HL",
        category=FailureCategory.REASONING,
        name="Closed-Domain Grounding & Entity Fidelity",
        constraint_text=(
            "CRITICAL CONSTRAINT: Strictly ground your synthesis on the provided context only. "
            "Do not introduce external named entities, unverified companies, or facts not present "
            "in the input. If information is missing, state that it is unavailable."
        ),
        explanation=(
            "Enforces closed-domain hallucination boundaries, forbidding the agent from hallucinating "
            "or inventing entities not present in input context (R-HL)."
        ),
        default_regression_risk=0.08,
    ),
    ConstraintTemplate(
        constraint_type="CONTRADICTION_GUARD",
        rule_prefix="R-CT",
        category=FailureCategory.REASONING,
        name="Logical Premise & Consistency Guard",
        constraint_text=(
            "CRITICAL CONSTRAINT: Do not contradict any established premise or finding from "
            "upstream agent outputs. Ensure logical entailment with input facts and maintain "
            "consistent timelines and factual attributions."
        ),
        explanation=(
            "Prevents the agent from asserting claims that contradict upstream evidence or premises (R-CT)."
        ),
        default_regression_risk=0.07,
    ),
    ConstraintTemplate(
        constraint_type="SCHEMA_CONFORMANCE",
        rule_prefix="R-WF",
        category=FailureCategory.WORKFLOW,
        name="State Handoff & Schema Integrity",
        constraint_text=(
            "CRITICAL CONSTRAINT: Preserve all expected dictionary keys in the output payload. "
            "Never delete, omit, or overwrite upstream state keys. Concur with the strict JSON output schema."
        ),
        explanation=(
            "Enforces dictionary schema integrity and prevents dropped state variables during agent handoffs (R-WF)."
        ),
        default_regression_risk=0.04,
    ),
    ConstraintTemplate(
        constraint_type="EXECUTION_FALLBACK",
        rule_prefix="R-EX",
        category=FailureCategory.EXECUTION,
        name="Execution Timeout & Error Resilience",
        constraint_text=(
            "CRITICAL CONSTRAINT: Handle tool timeouts and external API errors gracefully. "
            "Provide valid fallback responses with explicit error metadata rather than unhandled aborts."
        ),
        explanation=(
            "Ensures external tool failures are captured and returned safely rather than crashing the pipeline (R-EX)."
        ),
        default_regression_risk=0.05,
    ),
    ConstraintTemplate(
        constraint_type="QUALITY_GUARD",
        rule_prefix="R-VF",
        category=FailureCategory.VERIFICATION,
        name="Independent Verification & Rigor",
        constraint_text=(
            "CRITICAL CONSTRAINT: Explicitly cross-check all claims against source findings. "
            "Reject and flag any ungrounded assertions or missing citations rather than approving questionable outputs."
        ),
        explanation=("Prevents verifiers from rubber-stamping invalid upstream outputs (R-VF)."),
        default_regression_risk=0.06,
    ),
]

_FALLBACK_TEMPLATE = ConstraintTemplate(
    constraint_type="GENERIC_GUARD",
    rule_prefix="",
    category=FailureCategory.UNKNOWN,
    name="Execution Integrity Guard",
    constraint_text=(
        "CRITICAL CONSTRAINT: Adhere strictly to the required task instructions and preserve "
        "all required input context, citations, and structural constraints without dropping information."
    ),
    explanation="Adds strict task integrity and context retention constraints to prevent failures.",
    default_regression_risk=0.10,
)


# ─────────────────────────────────────────────────────────────────────────────
# PromptFixer Engine
# ─────────────────────────────────────────────────────────────────────────────


class PromptFixer:
    """
    Automated prompt remediation engine.

    Analyzes trace failures, matches them against the constraint catalog,
    compares against historical successful runs, and outputs GitHub-style
    unified diff recommendations.
    """

    def __init__(
        self,
        db: DatabaseManager | None = None,
        default_prompts: dict[str, str] | None = None,
    ) -> None:
        self.db = db
        self.default_prompts: dict[str, str] = default_prompts or {
            "researcher": (
                "You are an expert research agent. Gather verified sources, citations, and named entities "
                "on the given topic."
            ),
            "writer": (
                "You are an expert synthesis agent. Summarize the research findings into a comprehensive report "
                "with introduction, key findings, and conclusion."
            ),
            "verifier": (
                "You are a quality-control verification agent. Review the synthesis report for accuracy, "
                "completeness, and adherence to source findings."
            ),
        }

    def generate_patch(
        self,
        run_trace: RunTrace,
        analysis_bundle: AnalysisBundle | None = None,
        causal_graph: Any | None = None,
        historical_traces: list[RunTrace] | None = None,
        target_agent: str | None = None,
        override_prompt: str | None = None,
    ) -> RemediationPatch | None:
        """
        Generate an automated prompt remediation patch for a failed run.

        Args:
            run_trace: The trace containing the failure.
            analysis_bundle: Optional Arbiter verdict bundle.
            causal_graph: Optional CausalGraph from Day 62.
            historical_traces: Optional list of past successful runs for baseline comparison.
            target_agent: Optional explicit agent to remediate.
            override_prompt: Optional base prompt string to patch.

        Returns:
            RemediationPatch if failure is remediable, or None if clean run.
        """
        # Clean run check: if run succeeded and no rules fired, no patch needed
        all_rules: list[RuleMatch] = []
        if analysis_bundle and analysis_bundle.rule_matches:
            all_rules.extend(analysis_bundle.rule_matches)

        has_failure = run_trace.status in (StepStatus.FAILURE, StepStatus.ERROR)
        if not has_failure and not all_rules and not (causal_graph and causal_graph.root_causes):
            return None

        # Determine target agent
        agent = target_agent
        if not agent and analysis_bundle and analysis_bundle.primary_agent:
            agent = analysis_bundle.primary_agent
        if not agent and causal_graph:
            root_node = causal_graph.get_root_cause()
            if root_node:
                agent = root_node.name
        if not agent:
            # Pick first step with error/failure
            for step in run_trace.steps:
                if step.status in (StepStatus.ERROR, StepStatus.FAILURE):
                    agent = step.agent
                    break
        if not agent:
            agent = "writer"

        # Find matching rule and category
        category = analysis_bundle.primary_cause if analysis_bundle else FailureCategory.REASONING
        rule_match: RuleMatch | None = None
        for rm in all_rules:
            if rm.agent == agent or not rule_match:
                rule_match = rm

        rule_id = rule_match.rule_id if rule_match else None
        if rule_match and rule_match.category:
            category = rule_match.category

        # Select constraint template
        template = self._select_template(rule_id, category)

        # Retrieve original base prompt
        original_prompt = self._resolve_prompt(
            agent=agent,
            run_trace=run_trace,
            override_prompt=override_prompt,
        )

        # Build remediated prompt
        remediated_prompt = self._inject_constraint(original_prompt, template.constraint_text)

        # Generate GitHub-style unified diff
        diff_patch = self._format_unified_diff(
            agent=agent,
            original_prompt=original_prompt,
            remediated_prompt=remediated_prompt,
        )

        # Compare against historical successful runs if available
        baseline_stats = self._compare_with_history(
            agent=agent,
            current_trace=run_trace,
            historical_traces=historical_traces,
        )

        confidence = 0.90
        regression_risk = template.default_regression_risk
        ref_run_id = baseline_stats.get("reference_run_id")

        if baseline_stats.get("historical_match"):
            confidence = min(0.98, confidence + 0.05)
            # If historical clean runs exhibited clean outputs, regression risk is lower
            regression_risk = max(0.02, regression_risk - 0.03)

        return RemediationPatch(
            run_id=run_trace.run_id,
            agent=agent,
            rule_id=rule_id,
            category=category,
            constraint_type=template.constraint_type,
            target_prompt_type="system",
            original_prompt_snippet=original_prompt,
            recommended_constraint=template.constraint_text,
            diff_patch=diff_patch,
            confidence=round(confidence, 2),
            regression_risk=round(regression_risk, 2),
            explanation=template.explanation,
            historical_reference_run_id=ref_run_id,
            applied=False,
        )

    def apply_patch(self, patch: RemediationPatch, base_prompt: str | None = None) -> str:
        """Apply a RemediationPatch constraint to a base prompt string."""
        prompt = base_prompt if base_prompt is not None else patch.original_prompt_snippet
        return self._inject_constraint(prompt, patch.recommended_constraint)

    def _select_template(
        self,
        rule_id: str | None,
        category: FailureCategory,
    ) -> ConstraintTemplate:
        """Select the best matching constraint template for the failure."""
        if rule_id:
            rule_upper = rule_id.upper()
            for tmpl in CONSTRAINT_CATALOG:
                if tmpl.rule_prefix and rule_upper.startswith(tmpl.rule_prefix):
                    return tmpl

        for tmpl in CONSTRAINT_CATALOG:
            if tmpl.category == category:
                return tmpl

        return _FALLBACK_TEMPLATE

    def _resolve_prompt(
        self,
        agent: str,
        run_trace: RunTrace,
        override_prompt: str | None = None,
    ) -> str:
        """Resolve the base prompt text for the target agent."""
        if override_prompt:
            return override_prompt.strip()

        # Check trace steps for an explicit prompt
        for step in run_trace.steps:
            if step.agent == agent:
                if step.system_prompt and step.system_prompt.strip():
                    return step.system_prompt.strip()
                if step.prompt and step.prompt.strip():
                    return step.prompt.strip()

        # Check configured / default prompts
        if agent in self.default_prompts:
            return self.default_prompts[agent].strip()

        return f"You are the {agent} agent. Execute your assigned multi-agent workflow task."

    def _inject_constraint(self, original_prompt: str, constraint: str) -> str:
        """Append or inject the constraint cleanly into the prompt."""
        clean_orig = original_prompt.rstrip()
        if not clean_orig:
            return constraint.strip()
        return f"{clean_orig}\n\n{constraint.strip()}"

    def _format_unified_diff(
        self,
        agent: str,
        original_prompt: str,
        remediated_prompt: str,
    ) -> str:
        """Generate a standard unified Git diff string."""
        orig_lines = original_prompt.splitlines(keepends=True)
        rem_lines = remediated_prompt.splitlines(keepends=True)

        # Ensure trailing newlines for difflib consistency
        orig_lines = [line if line.endswith("\n") else line + "\n" for line in orig_lines]
        rem_lines = [line if line.endswith("\n") else line + "\n" for line in rem_lines]

        diff_gen = difflib.unified_diff(
            orig_lines,
            rem_lines,
            fromfile=f"SYSTEM PROMPT (agent={agent}:original)",
            tofile=f"SYSTEM PROMPT (agent={agent}:remediated)",
            lineterm="",
        )
        return "\n".join(diff_gen)

    def _compare_with_history(
        self,
        agent: str,
        current_trace: RunTrace,
        historical_traces: list[RunTrace] | None = None,
    ) -> dict[str, Any]:
        """Compare current run against historical successful runs."""
        clean_traces: list[RunTrace] = []

        if historical_traces:
            clean_traces = [
                t
                for t in historical_traces
                if t.status == StepStatus.SUCCESS and t.run_id != current_trace.run_id
            ]

        if not clean_traces and self.db:
            try:
                successful_rows = self.db.list_runs(limit=10, status_filter="SUCCESS")
                for row in successful_rows:
                    rid = row.get("run_id")
                    if rid and rid != current_trace.run_id:
                        full_row = self.db.get_run(rid)
                        if full_row and full_row.get("trace_json"):
                            t = RunTrace(**json.loads(full_row["trace_json"]))
                            clean_traces.append(t)
            except Exception:
                pass

        if not clean_traces:
            return {"historical_match": False, "reference_run_id": None}

        # Select the closest historical run of the same workflow
        ref_trace = next(
            (t for t in clean_traces if t.workflow == current_trace.workflow),
            clean_traces[0],
        )

        ref_step: AgentStep | None = None
        for step in ref_trace.steps:
            if step.agent == agent and step.status == StepStatus.SUCCESS:
                ref_step = step
                break

        return {
            "historical_match": True,
            "reference_run_id": ref_trace.run_id,
            "reference_agent_status": ref_step.status if ref_step else None,
            "clean_run_count": len(clean_traces),
        }
