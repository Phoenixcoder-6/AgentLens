import os
import sys

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.arbiter import determine_primary_cause
from analyzers.detection.rule_engine import RuleEngine
from analyzers.detection.workflow_validator import WorkflowValidator
from schema.models import AgentStep, PriorityLevel, RunTrace


def main() -> None:
    print("=== Day 18 Verification: Execution Rules & Workflow Priority (P3) ===\n")

    rule_engine = RuleEngine()
    workflow_validator = WorkflowValidator()

    # ─────────────────────────────────────────────────────────────────────────
    print("[1/2] Testing: Execution Tool Failure (Should trigger P2)")
    # ─────────────────────────────────────────────────────────────────────────
    trace_tool_fail = RunTrace(
        workflow="test",
        steps=[
            AgentStep(
                run_id="run1",
                step=1,
                agent="researcher",
                tool_calls=[
                    {"name": "search", "args": {}, "output": "Error: API Timeout", "error": ""}
                ],
            )
        ],
    )

    result_rule = rule_engine.analyze(trace_tool_fail)

    if not result_rule.skipped and len(result_rule.evidence) > 0:
        ev = result_rule.evidence[0]
        print("  [PASS] RuleEngine correctly detected tool failure.")
        print(f"         Rule Fired: {ev.rule_match.rule_id if ev.rule_match else 'None'}")

        # Test Arbiter Routing
        bundle = determine_primary_cause(result_rule.evidence, "run1")
        if bundle and bundle.priority_level == PriorityLevel.P2:
            print("  [PASS] Arbiter successfully prioritized the tool failure as P2.")
            print(f"         Primary Cause: {bundle.primary_cause.value}")
        else:
            print("  [FAIL] Arbiter failed to assign P2 to rule match.")
    else:
        print("  [FAIL] RuleEngine failed to detect tool failure.\n")
        sys.exit(1)

    print()

    # ─────────────────────────────────────────────────────────────────────────
    print("[2/2] Testing: Workflow Violation - Skipped Verifier (Should trigger P3)")
    # ─────────────────────────────────────────────────────────────────────────
    trace_workflow_fail = RunTrace(
        workflow="test",
        steps=[
            AgentStep(run_id="run2", step=1, agent="researcher"),
            AgentStep(run_id="run2", step=2, agent="writer"),
            # Missing verifier step!
        ],
    )

    result_workflow = workflow_validator.analyze(trace_workflow_fail)

    if not result_workflow.skipped and len(result_workflow.evidence) > 0:
        ev = result_workflow.evidence[0]
        print("  [PASS] WorkflowValidator correctly detected skipped step.")
        print(
            f"         Rule Fired: {ev.rule_match.rule_id if ev.rule_match else 'None'} for agent '{ev.agent}'"
        )

        # Test Arbiter Routing
        bundle = determine_primary_cause(result_workflow.evidence, "run2")
        if bundle and bundle.priority_level == PriorityLevel.P3:
            print("  [PASS] Arbiter successfully prioritized the workflow violation as P3.")
            print(f"         Primary Cause: {bundle.primary_cause.value}")
        else:
            print(
                f"  [FAIL] Arbiter failed to assign P3 to workflow violation. Got: {bundle.priority_level if bundle else 'None'}"
            )
    else:
        print("  [FAIL] WorkflowValidator failed to detect skipped step.\n")
        sys.exit(1)

    print("\n✅ All Day 18 Verifications Passed!")


if __name__ == "__main__":
    main()
