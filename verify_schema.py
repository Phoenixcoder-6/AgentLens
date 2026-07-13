"""
verify_schema.py — Day 2 smoke test
Confirms all 7 Pydantic models instantiate correctly and schema_version is stamped.
Run: conda run -n agentlens python verify_schema.py
"""

import sys
import json
sys.path.insert(0, ".")

from schema import (
    SCHEMA_VERSION,
    RunTrace, AgentStep, WorkflowState, HandoffState,
    AnalysisBundle, EvidenceRecord, RuleMatch,
    FailureCategory, PriorityLevel, EvidenceSource,
    StepStatus, NodeType, RuleSeverity,
    TokenUsage, GenerationParams,
)

PASS = "[OK]  "
FAIL = "[FAIL]"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label} {detail}")
        errors.append(label)


# ── 1. SCHEMA_VERSION ────────────────────────────────────────────────────────
check("SCHEMA_VERSION == '1.0'", SCHEMA_VERSION == "1.0")


# ── 2. HandoffState ──────────────────────────────────────────────────────────
hs = HandoffState(
    input_state={"sources": 10},
    filtered_state={"sources": 3},
    output_state={"summary_sources_cited": 3},
)
check("HandoffState instantiates", hs is not None)
check("HandoffState.input_state has 'sources'", "sources" in hs.input_state)


# ── 3. WorkflowState ─────────────────────────────────────────────────────────
ws = WorkflowState(run_id="run_test01", step_index=1, state_data={"topic": "AI"})
check("WorkflowState instantiates", ws is not None)
check("WorkflowState.schema_version stamped", ws.schema_version == SCHEMA_VERSION)


# ── 4. AgentStep ─────────────────────────────────────────────────────────────
step = AgentStep(
    run_id="run_test01",
    step=1,
    agent="researcher",
    node_type=NodeType.LLM,
    input="Summarize findings from 10 sources",
    output="Tesla was founded in 2003...",
    latency_ms=1400,
    tokens=TokenUsage(prompt=512, completion=180),
    handoff=hs,
    status=StepStatus.SUCCESS,
)
check("AgentStep instantiates", step is not None)
check("AgentStep.schema_version stamped", step.schema_version == SCHEMA_VERSION)
check("AgentStep.tokens.total auto-computed", step.tokens.total == 692)
check("AgentStep.handoff captured", step.handoff.input_state["sources"] == 10)


# ── 5. RunTrace ──────────────────────────────────────────────────────────────
run = RunTrace(workflow="research_report_pipeline", steps=[step])
check("RunTrace instantiates", run is not None)
check("RunTrace.schema_version stamped", run.schema_version == SCHEMA_VERSION)
check("RunTrace.run_id auto-generated", run.run_id.startswith("run_"))
check("RunTrace has 1 step", len(run.steps) == 1)


# ── 6. RuleMatch ─────────────────────────────────────────────────────────────
rule = RuleMatch(
    rule_id="R-WF-001",
    category=FailureCategory.WORKFLOW,
    description="Information loss detected: sources dropped from 10 to 3",
    severity=RuleSeverity.HIGH,
    agent="writer",
    step=2,
    evidence_detail="sources: 10 -> 3",
)
check("RuleMatch instantiates", rule is not None)
check("RuleMatch.category is WORKFLOW", rule.category == FailureCategory.WORKFLOW)


# ── 7. EvidenceRecord ────────────────────────────────────────────────────────
ev = EvidenceRecord(
    source=EvidenceSource.RULE_ENGINE,
    description="Information loss in writer handoff",
    rule_match=rule,
    agent="writer",
    step=2,
    confidence=1.0,
)
check("EvidenceRecord instantiates", ev is not None)
check("EvidenceRecord.confidence == 1.0", ev.confidence == 1.0)
check("EvidenceRecord.rule_match attached", ev.rule_match.rule_id == "R-WF-001")


# ── 8. AnalysisBundle ────────────────────────────────────────────────────────
bundle = AnalysisBundle(
    run_id=run.run_id,
    primary_cause=FailureCategory.WORKFLOW,
    priority_level=PriorityLevel.P2,
    grounded=False,
    evidence=[ev],
    rule_matches=[rule],
    primary_agent="writer",
)
check("AnalysisBundle instantiates", bundle is not None)
check("AnalysisBundle.schema_version stamped", bundle.schema_version == SCHEMA_VERSION)
check("AnalysisBundle.grounded=False (heuristic)", bundle.grounded == False)
check("AnalysisBundle.priority_level == P2", bundle.priority_level == PriorityLevel.P2)


# ── 9. JSON round-trip ───────────────────────────────────────────────────────
run_json = run.model_dump_json()
run_restored = RunTrace.model_validate_json(run_json)
check("RunTrace JSON round-trip", run_restored.run_id == run.run_id)
check("RunTrace round-trip preserves schema_version", run_restored.schema_version == SCHEMA_VERSION)

bundle_json = bundle.model_dump_json()
bundle_restored = AnalysisBundle.model_validate_json(bundle_json)
check("AnalysisBundle JSON round-trip", bundle_restored.run_id == bundle.run_id)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
if not errors:
    print("Day 2 schema verification PASSED. All 7 models OK.")
else:
    print(f"FAILED: {len(errors)} check(s) failed: {errors}")
    sys.exit(1)
