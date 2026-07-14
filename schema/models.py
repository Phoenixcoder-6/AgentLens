"""
schema/models.py — AgentLens Canonical Trace Schema
=====================================================
All Pydantic v2 models that define the data contracts for the entire system.
Every layer — Capture, Normalizer, Analyzers, Arbiter, Dashboard — reads and
writes these models. Nothing passes raw dicts between layers.

Schema version: 1.0  (stamped on every record)

Models defined here:
    RunTrace         — top-level container for a full workflow run
    AgentStep        — a single agent invocation (atomic unit of analysis)
    WorkflowState    — shared pipeline state between agents
    HandoffState     — before/after state snapshot at each agent handoff
    AnalysisBundle   — Arbiter's final verdict (what the LLM Explainer sees)
    EvidenceRecord   — a single piece of evidence from any analyzer
    RuleMatch        — a fired rule from the rule engine or validators
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_serializer

# ─────────────────────────────────────────────────────────────────────────────
# Schema Version — stamped on every record
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    """Execution outcome of a single agent step or overall run."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class NodeType(str, Enum):
    """The type of node in the LangGraph pipeline."""
    LLM = "llm"
    TOOL = "tool"
    ROUTER = "router"
    HUMAN = "human"


class FailureCategory(str, Enum):
    """
    Four-category failure taxonomy from the architecture spec.
    Every verdict maps to exactly one of these.

    - EXECUTION:    tool timeout, API error, crash, missing output
    - REASONING:    hallucination, wrong facts, missed sources
    - WORKFLOW:     information loss in handoff, skipped node, state mutation
    - VERIFICATION: verifier approved a bad answer
    - UNKNOWN:      no evidence matched (P5 — Arbiter fallback)
    """
    EXECUTION = "execution"
    REASONING = "reasoning"
    WORKFLOW = "workflow"
    VERIFICATION = "verification"
    UNKNOWN = "unknown"


class PriorityLevel(str, Enum):
    """
    Arbiter priority levels — determines which evidence wins in a conflict.
    Tie-break: same-priority ties resolved by rule_id ascending.

    P1: ground_truth_mismatch  — expected_output present and output differs
    P2: rule_match             — deterministic rule fired
    P3: workflow_violation     — workflow validator flagged handoff issue
    P4: statistical_anomaly    — metrics analyzer detected outlier
    P5: unknown                — no evidence matched
    """
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"


class EvidenceSource(str, Enum):
    """Which analyzer produced an EvidenceRecord."""
    RULE_ENGINE = "rule_engine"
    WORKFLOW_VALIDATOR = "workflow_validator"
    CONSISTENCY_VALIDATOR = "consistency_validator"
    METRICS_ANALYZER = "metrics_analyzer"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    DIFF_ENGINE = "diff_engine"
    GROUND_TRUTH = "ground_truth"


class RuleSeverity(str, Enum):
    """How serious a rule match is — used for dashboard display priority."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models (building blocks used inside the main models)
# ─────────────────────────────────────────────────────────────────────────────

class TokenUsage(BaseModel):
    """Token counts for a single agent step."""
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute total if not explicitly provided."""
        if self.total == 0 and (self.prompt > 0 or self.completion > 0):
            object.__setattr__(self, "total", self.prompt + self.completion)


class GenerationParams(BaseModel):
    """LLM generation parameters captured per step."""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: HandoffState
# ─────────────────────────────────────────────────────────────────────────────

class HandoffState(BaseModel):
    """
    Records the shared state at each agent handoff.

    This three-field snapshot is what makes information-loss detection possible:
        input_state:    what the agent received from the previous agent
        filtered_state: what it selected / processed internally
        output_state:   what it passed forward to the next agent

    Comparing input_state vs output_state reveals dropped keys, mutated values,
    and silent context loss — failure modes invisible in plain input/output logs.
    """
    input_state: dict[str, Any] = Field(
        default_factory=dict,
        description="State received from the previous agent or pipeline entry"
    )
    filtered_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Intermediate state after agent's internal filtering/processing"
    )
    output_state: dict[str, Any] = Field(
        default_factory=dict,
        description="State passed forward to the next agent"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: WorkflowState
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowState(BaseModel):
    """
    The shared state object flowing through the entire LangGraph pipeline.
    Captured at the workflow level (not per individual step).

    Used by the Workflow Validator to detect graph-level anomalies:
    skipped nodes, unexpected transitions, missing required keys.
    """
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version — stamped on every record"
    )
    run_id: str = Field(description="Shared run identifier across all steps")
    step_index: int = Field(description="Position of this state snapshot in the workflow")
    state_data: dict[str, Any] = Field(
        default_factory=dict,
        description="The full LangGraph state dict at this point in the pipeline"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: AgentStep
# ─────────────────────────────────────────────────────────────────────────────

class AgentStep(BaseModel):
    """
    A single agent invocation — the atomic unit of analysis in AgentLens.

    Captured by the @trace_step decorator on every agent call.
    Fields map directly to the Canonical Trace Schema (Appendix A of the ARD).

    The handoff field is the key differentiator from plain input/output logging:
    it records the full state transformation, enabling information-loss detection.
    """
    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str = Field(description="Links this step to its parent RunTrace")
    step: int = Field(description="Sequential step number within the run (1-indexed)")
    agent: str = Field(description="Agent name, e.g. 'researcher', 'writer', 'verifier'")
    node_type: NodeType = Field(default=NodeType.LLM)

    # Input / output
    input: str = Field(default="", description="Raw input passed to this agent")
    output: str = Field(default="", description="Raw output produced by this agent")
    expected_output: Optional[str] = Field(
        default=None,
        description="Ground-truth expected output, if available (enables P1 grounded evaluation)"
    )

    # Prompt details
    prompt: str = Field(default="", description="User-turn prompt sent to the LLM")
    system_prompt: str = Field(default="", description="System prompt sent to the LLM")
    model: str = Field(default="", description="Model ID used for this step")
    generation_params: GenerationParams = Field(default_factory=GenerationParams)

    # Performance metrics
    latency_ms: float = Field(default=0.0, description="Wall-clock time for this step in ms")
    tokens: TokenUsage = Field(default_factory=TokenUsage)

    # Tool usage
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool invocations made during this step"
    )

    # Handoff — captures state transformation (not just input/output)
    handoff: HandoffState = Field(default_factory=HandoffState)

    # Graph relationships
    parent_step: Optional[str] = Field(
        default=None,
        description="Step ID of the agent that produced input to this step"
    )
    child_step: Optional[str] = Field(
        default=None,
        description="Step ID of the agent that will receive this step's output"
    )

    # Status
    status: StepStatus = Field(default=StepStatus.SUCCESS)
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is FAILURE or ERROR"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer('timestamp')
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat()

    model_config = {}


# ─────────────────────────────────────────────────────────────────────────────
# Model 4: RunTrace
# ─────────────────────────────────────────────────────────────────────────────

class RunTrace(BaseModel):
    """
    Top-level container for a complete workflow execution.

    One RunTrace groups all AgentSteps under a shared run_id.
    Stored as a JSON blob in data/traces/ and indexed in SQLite.

    The trace_path field links the SQLite record to its full JSON payload,
    keeping traces portable and human-readable while SQLite handles queries.
    """
    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str = Field(
        default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}",
        description="Unique run identifier, e.g. 'run_8f21ac'"
    )
    workflow: str = Field(description="Workflow name, e.g. 'research_report_pipeline'")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this run started"
    )

    # All steps in this run, in order
    steps: list[AgentStep] = Field(default_factory=list)

    # Aggregate metrics (computed after capture)
    total_latency_ms: float = Field(default=0.0)
    total_tokens: int = Field(default=0)
    status: StepStatus = Field(
        default=StepStatus.SUCCESS,
        description="Overall run status — FAILURE if any step failed"
    )

    # Ground truth (optional — enables P1 grounded evaluation in Arbiter)
    expected_output: Optional[str] = Field(
        default=None,
        description="Expected final output for this run, if labeled"
    )

    # Storage reference
    trace_path: Optional[str] = Field(
        default=None,
        description="Relative path to the full JSON trace blob in data/traces/"
    )

    @field_serializer('timestamp')
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat()

    model_config = {}


# ─────────────────────────────────────────────────────────────────────────────
# Model 5: RuleMatch
# ─────────────────────────────────────────────────────────────────────────────

class RuleMatch(BaseModel):
    """
    A single fired rule from the rule engine or validators.

    Produced by:
        rule_engine.py          → Execution, Reasoning, Verification rules
        workflow_validator.py   → Workflow rules (handoff, skipped nodes)
        consistency_validator.py → Verification rules (verifier approved bad answer)

    Consumed by the Arbiter as P2 or P3 evidence.
    """
    rule_id: str = Field(description="Unique rule identifier, e.g. 'R-WF-001'")
    category: FailureCategory = Field(description="Which failure category this rule belongs to")
    description: str = Field(description="Human-readable description of what was detected")
    severity: RuleSeverity = Field(default=RuleSeverity.MEDIUM)
    agent: Optional[str] = Field(
        default=None,
        description="Which agent triggered this rule (if attributable)"
    )
    step: Optional[int] = Field(
        default=None,
        description="Which step index triggered this rule"
    )
    evidence_detail: Optional[str] = Field(
        default=None,
        description="Specific detail about what was found, e.g. 'sources dropped: 10 → 3'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 6: EvidenceRecord
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceRecord(BaseModel):
    """
    A single piece of evidence from any analyzer.

    The Arbiter collects EvidenceRecords from all analyzers and resolves
    them into a single verdict using the P1–P5 priority system.

    Standardizing all evidence into this shape means the Arbiter compares
    apples to apples — it doesn't need to know which analyzer produced what.
    """
    source: EvidenceSource = Field(description="Which analyzer produced this evidence")
    description: str = Field(description="What was found")
    value: Any = Field(
        default=None,
        description="Numeric value, boolean, or string — the raw finding"
    )
    rule_match: Optional[RuleMatch] = Field(
        default=None,
        description="Populated when source is rule_engine or a validator"
    )
    agent: Optional[str] = Field(
        default=None,
        description="Which agent this evidence is attributed to"
    )
    step: Optional[int] = Field(
        default=None,
        description="Which step this evidence came from"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence 0.0–1.0; deterministic rules = 1.0, statistical = variable"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 7: AnalysisBundle
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisBundle(BaseModel):
    """
    The Arbiter's final output — the complete analysis verdict for one run.

    CRITICAL: This is the ONLY thing the LLM Explainer ever sees.
    The LLM never touches the raw RunTrace or AgentStep data.
    This is the architectural property that makes verdicts trustworthy:
    the model explains what deterministic analysis established, not what it guesses.

    The 'grounded' flag is the most important field for trust:
        True  → verdict backed by expected_output comparison (P1 evidence)
        False → verdict is heuristic only (P2–P5), hedging language should be used
    """
    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str = Field(description="Links this bundle to its RunTrace")

    # Arbiter verdict
    primary_cause: FailureCategory = Field(
        description="The single most likely failure category"
    )
    priority_level: PriorityLevel = Field(
        description="Which priority tier produced the primary cause"
    )
    grounded: bool = Field(
        description="True if expected_output was available for comparison"
    )

    # Evidence used to reach the verdict
    evidence: list[EvidenceRecord] = Field(
        default_factory=list,
        description="All evidence records considered by the Arbiter"
    )
    rule_matches: list[RuleMatch] = Field(
        default_factory=list,
        description="All rules that fired during analysis"
    )

    # Attribution
    primary_agent: Optional[str] = Field(
        default=None,
        description="Which agent is most responsible for the primary cause"
    )

    # LLM Explainer output (populated after explanation step)
    summary: Optional[str] = Field(
        default=None,
        description="LLM-generated root cause summary (filled by explanation layer)"
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="LLM-generated suggested fix (filled by explanation layer)"
    )

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer('timestamp')
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat()

    model_config = {}
