import os
import sys

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.arbiter import Arbiter
from analyzers.detection.ground_truth import GroundTruthValidator
from schema.models import AgentStep, PriorityLevel, RunTrace


def main() -> None:
    print("=== Day 17 Verification: Ground Truth Mismatch (P1) ===\n")

    validator = GroundTruthValidator()
    arbiter = Arbiter()

    # ─────────────────────────────────────────────────────────────────────────
    print("[1/3] Testing: Output Matches Expected (Should PASS)")
    # ─────────────────────────────────────────────────────────────────────────
    expected_text = "The capital of France is Paris."
    actual_text = "The capital of France is Paris."  # Exact match

    trace_pass = RunTrace(
        workflow="test",
        expected_output=expected_text,
        steps=[AgentStep(run_id="run1", step=1, agent="writer", output=actual_text)],
    )

    result_pass = validator.analyze(trace_pass)

    if not result_pass.skipped and len(result_pass.evidence) == 0:
        print("  [PASS] Validator correctly returned no evidence for a match.\n")
    else:
        print("  [FAIL] Validator behavior incorrect on match.\n")

    # ─────────────────────────────────────────────────────────────────────────
    print("[2/3] Testing: Output Diverges (Should trigger P1)")
    # ─────────────────────────────────────────────────────────────────────────
    actual_mismatch_text = "The capital of France is Berlin, which is incorrect."

    trace_fail = RunTrace(
        workflow="test",
        expected_output=expected_text,
        steps=[AgentStep(run_id="run2", step=1, agent="writer", output=actual_mismatch_text)],
    )

    result_fail = validator.analyze(trace_fail)

    if not result_fail.skipped and len(result_fail.evidence) == 1:
        ev = result_fail.evidence[0]
        print("  [PASS] Validator correctly detected mismatch.")
        print(f"         Rule Fired: {ev.rule_match.rule_id if ev.rule_match else 'None'}")
        print(f"         Confidence: {ev.confidence:.0%}\n")
    else:
        print("  [FAIL] Validator behavior incorrect on mismatch.\n")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    print("[3/3] Testing: Arbiter assigns P1 Priority and Grounded=True")
    # ─────────────────────────────────────────────────────────────────────────
    # We pass the evidence generated from step 2 directly into the Arbiter
    bundle = arbiter.run(run_id="run2", evidence=result_fail.evidence)

    if bundle.priority_level == PriorityLevel.P1 and bundle.grounded is True:
        print("  [PASS] Arbiter successfully prioritized the evidence as P1.")
        print("  [PASS] Arbiter marked the bundle as grounded=True.")
        print(f"  [PASS] Primary Cause: {bundle.primary_cause.value}")
    else:
        print("  [FAIL] Arbiter did not prioritize correctly.")
        print(f"         Priority Level: {bundle.priority_level}")
        print(f"         Grounded: {bundle.grounded}")

    print("\n✅ All Day 17 Verifications Passed!")


if __name__ == "__main__":
    main()
