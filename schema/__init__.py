"""
schema/__init__.py
Exports all AgentLens models and enums from a single import point.

Usage:
    from schema import RunTrace, AgentStep, HandoffState, AnalysisBundle
    from schema import FailureCategory, PriorityLevel, StepStatus
"""

from schema.models import (
    # Version
    SCHEMA_VERSION,
    AgentStep,
    AnalysisBundle,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    GenerationParams,
    # Core models
    HandoffState,
    NodeType,
    PriorityLevel,
    # Analysis models
    RuleMatch,
    RuleSeverity,
    RunTrace,
    # Enums
    StepStatus,
    # Sub-models
    TokenUsage,
    WorkflowState,
)
from schema.span import (
    SpanType,
    TraceSpan,
    spans_to_steps,
    steps_to_spans,
)

__all__ = [
    "SCHEMA_VERSION",
    "StepStatus",
    "NodeType",
    "SpanType",
    "FailureCategory",
    "PriorityLevel",
    "EvidenceSource",
    "RuleSeverity",
    "TokenUsage",
    "GenerationParams",
    "HandoffState",
    "WorkflowState",
    "AgentStep",
    "RunTrace",
    "TraceSpan",
    "RuleMatch",
    "EvidenceRecord",
    "AnalysisBundle",
    "steps_to_spans",
    "spans_to_steps",
]
