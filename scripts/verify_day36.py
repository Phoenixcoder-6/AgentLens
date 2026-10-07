"""Day 36 verification -- Error Injection Sweep (tests/test_error_injection.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_FILE = ROOT / "tests" / "test_error_injection.py"


def main() -> int:
    print("\nDay 36 verification -- Error Injection Sweep\n")
    passed = 0
    total = 8

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status = "[PASS]" if ok else "[FAIL]"
        suffix = f" -- {detail}" if detail else ""
        print(f"  {status} {label}{suffix}")
        if ok:
            passed += 1

    # 1. tests/test_error_injection.py exists
    exists = TEST_FILE.exists() and TEST_FILE.stat().st_size > 500
    check("tests/test_error_injection.py exists", exists, str(TEST_FILE))
    if not exists:
        print(f"\nDay 36 verification: {passed}/{total} checks passed")
        return 1

    from tests.test_error_injection import (
        TestAlwaysApproveVerifierInjection,
        TestCleanPipelineControl,
        TestSkippedVerifierNodeInjection,
        TestToolTimeoutInjection,
        TestWriterWrongFactsInjection,
    )

    # 2. Tool timeout -> execution failure, researcher blamed
    try:
        t_exec = TestToolTimeoutInjection()
        t_exec.test_tool_timeout_error_field_triggers_execution_failure()
        t_exec.test_tool_timeout_exception_in_output_triggers_execution_failure()
        t_exec.test_missing_tool_output_triggers_execution_failure()
        exec_ok = True
    except Exception as exc:
        exec_ok = False
        print(f"    error: {exc}")
    check(
        "Tool timeout injection -> execution failure, researcher blamed",
        exec_ok,
        "tool_failure_v1 & missing_tool_output_v1 (P2)",
    )

    # 3. Wrong facts in writer output -> reasoning failure, writer blamed
    try:
        t_reas = TestWriterWrongFactsInjection()
        t_reas.test_writer_entity_fabrication_triggers_reasoning_failure()
        reas_ok = True
    except Exception as exc:
        reas_ok = False
        print(f"    error: {exc}")
    check(
        "Wrong facts in writer output -> reasoning failure, writer blamed",
        reas_ok,
        "hallucination_v1 (P2)",
    )

    # 4. Skipped verifier node -> workflow failure, verifier blamed
    try:
        t_wf = TestSkippedVerifierNodeInjection()
        t_wf.test_skipped_verifier_node_triggers_workflow_failure()
        t_wf.test_skipped_researcher_node_triggers_workflow_failure()
        wf_ok = True
    except Exception as exc:
        wf_ok = False
        print(f"    error: {exc}")
    check(
        "Skipped verifier node -> workflow failure, verifier blamed",
        wf_ok,
        "skipped_step_v1 (P3)",
    )

    # 5. Always-approve verifier -> verification failure, verifier blamed
    try:
        t_ver = TestAlwaysApproveVerifierInjection()
        t_ver.test_always_approve_rubber_stamp_verifier_blamed_in_full_pipeline()
        t_ver.test_always_approve_verifier_passthrough_on_hallucinated_entities()
        ver_ok = True
    except Exception as exc:
        ver_ok = False
        print(f"    error: {exc}")
    check(
        "Always-approve verifier -> verification failure, verifier blamed",
        ver_ok,
        "verifier_passthrough_v1 (P2)",
    )

    # 6. Ground-truth contradiction -> P1 grounded failure, writer blamed
    try:
        t_gt = TestWriterWrongFactsInjection()
        t_gt.test_writer_ground_truth_contradiction_triggers_p1_grounded_failure()
        gt_ok = True
    except Exception as exc:
        gt_ok = False
        print(f"    error: {exc}")
    check(
        "Ground-truth contradiction -> P1 grounded failure, writer blamed",
        gt_ok,
        "grounded=True (P1)",
    )

    # 7. Clean baseline control -> P5 PASS, zero false positives
    try:
        t_clean = TestCleanPipelineControl()
        t_clean.test_clean_three_agent_pipeline_passes_with_zero_false_positives()
        clean_ok = True
    except Exception as exc:
        clean_ok = False
        print(f"    error: {exc}")
    check(
        "Clean 3-agent baseline control -> P5 PASS, zero false positives",
        clean_ok,
        "rules_fired=[] (P5)",
    )

    # 8. pytest tests/test_error_injection.py passes
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST_FILE), "-v", "-o", "addopts="],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    pytest_ok = proc.returncode == 0 and "10 passed" in proc.stdout
    summary_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    check(
        "pytest tests/test_error_injection.py passes all 10 cases",
        pytest_ok,
        summary_line,
    )

    pct = int(round(passed / total * 100))
    print(f"\nDay 36 verification: {passed}/{total} checks passed ({pct}%)")
    if passed == total:
        print("All checks passed -- Day 36 complete!\n")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
