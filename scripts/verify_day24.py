"""
scripts/verify_day24.py — Day 24 Verification Script
=====================================================
Verifies 10 critical Day 24 Graph Alignment requirements:
  1. Package: diff_engine importable with align_traces and models
  2. Type: align_traces returns GraphAlignmentResult
  3. Identical Alignment: identical traces produce 100% MATCHED steps
  4. Missing in B: skipped step in Trace B produces MISSING_IN_B
  5. Missing in A: extra step in Trace B produces MISSING_IN_A
  6. Middle Insertion: step inserted in sequence is matched gracefully
  7. Empty Trace: empty traces produce 0 pairs without throwing exceptions
  8. Matched Helper: get_matched_pairs filters MATCHED steps accurately
  9. Missing Helper: get_missing_steps filters MISSING steps accurately
 10. Agent Filtering: get_pairs_by_agent filters pairs by agent identity
"""

from __future__ import annotations

import os
import sys

# Force UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schema.models import AgentStep, RunTrace, StepStatus

PASS = "[PASS]"
FAIL = "[FAIL]"


def _make_step(agent: str, step: int, output: str = "") -> AgentStep:
    return AgentStep(
        run_id="test_run",
        step=step,
        agent=agent,
        output=output or f"{agent} output",
        status=StepStatus.SUCCESS,
    )


def _make_trace(run_id: str, *steps: AgentStep) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        workflow="test_pipeline",
        steps=list(steps),
    )


def main() -> int:
    print("=== Day 24 Verification: Graph Alignment Engine ===\n")
    passed = 0
    total = 0

    # ── Test 1: Package imports ───────────────────────────────────────────────
    total += 1
    try:
        from diff_engine import (
            AlignmentStatus,
            GraphAlignmentResult,
            align_traces,
        )

        print(f"  {PASS} [1] diff_engine package and alignment models imported successfully")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [1] diff_engine import error: {e}")

    # ── Test 2: Function signature and return type ────────────────────────────
    total += 1
    try:
        t1 = _make_trace("r1", _make_step("researcher", 1))
        t2 = _make_trace("r2", _make_step("researcher", 1))
        res = align_traces(t1, t2)
        if isinstance(res, GraphAlignmentResult):
            print(f"  {PASS} [2] align_traces returned GraphAlignmentResult instance")
            passed += 1
        else:
            print(f"  {FAIL} [2] Unexpected return type: {type(res)}")
    except Exception as e:
        print(f"  {FAIL} [2] align_traces signature error: {e}")

    # ── Test 3: Identical trace alignment ────────────────────────────────────
    total += 1
    try:
        t1 = _make_trace(
            "r1", _make_step("researcher", 1), _make_step("writer", 2), _make_step("verifier", 3)
        )
        t2 = _make_trace(
            "r2", _make_step("researcher", 1), _make_step("writer", 2), _make_step("verifier", 3)
        )
        res = align_traces(t1, t2)
        if res.matched_count == 3 and res.is_fully_aligned:
            print(f"  {PASS} [3] Identical traces aligned with 100% MATCHED steps")
            passed += 1
        else:
            print(f"  {FAIL} [3] Identical alignment failed: matched={res.matched_count}")
    except Exception as e:
        print(f"  {FAIL} [3] Identical trace error: {e}")

    # ── Test 4: Missing step in Trace B ──────────────────────────────────────
    total += 1
    try:
        t1 = _make_trace(
            "r1", _make_step("researcher", 1), _make_step("writer", 2), _make_step("verifier", 3)
        )
        t2 = _make_trace("r2", _make_step("researcher", 1), _make_step("writer", 2))
        res = align_traces(t1, t2)
        missing_b = [p for p in res.pairs if p.status == AlignmentStatus.MISSING_IN_B]
        if len(missing_b) == 1 and missing_b[0].agent == "verifier" and not res.is_fully_aligned:
            print(f"  {PASS} [4] Missing step in Trace B correctly identified as MISSING_IN_B")
            passed += 1
        else:
            print(f"  {FAIL} [4] Missing step in B check failed: missing_b={missing_b}")
    except Exception as e:
        print(f"  {FAIL} [4] Missing step in B error: {e}")

    # ── Test 5: Extra step in Trace B (Missing in A) ─────────────────────────
    total += 1
    try:
        t1 = _make_trace("r1", _make_step("researcher", 1), _make_step("writer", 2))
        t2 = _make_trace(
            "r2", _make_step("researcher", 1), _make_step("writer", 2), _make_step("verifier", 3)
        )
        res = align_traces(t1, t2)
        missing_a = [p for p in res.pairs if p.status == AlignmentStatus.MISSING_IN_A]
        if len(missing_a) == 1 and missing_a[0].agent == "verifier":
            print(f"  {PASS} [5] Extra step in Trace B correctly identified as MISSING_IN_A")
            passed += 1
        else:
            print(f"  {FAIL} [5] Extra step in B check failed: missing_a={missing_a}")
    except Exception as e:
        print(f"  {FAIL} [5] Extra step in B error: {e}")

    # ── Test 6: Middle step insertion ────────────────────────────────────────
    total += 1
    try:
        t1 = _make_trace("r1", _make_step("researcher", 1), _make_step("writer", 2))
        t2 = _make_trace(
            "r2", _make_step("researcher", 1), _make_step("editor", 2), _make_step("writer", 3)
        )
        res = align_traces(t1, t2)
        matched = [p.agent for p in res.get_matched_pairs()]
        if "researcher" in matched and "writer" in matched and res.missing_in_a_count == 1:
            print(f"  {PASS} [6] Middle insertion ('editor') aligned without index mismatch errors")
            passed += 1
        else:
            print(f"  {FAIL} [6] Middle insertion alignment failed: matched={matched}")
    except Exception as e:
        print(f"  {FAIL} [6] Middle insertion error: {e}")

    # ── Test 7: Empty trace handling ─────────────────────────────────────────
    total += 1
    try:
        t1 = _make_trace("r1")
        t2 = _make_trace("r2")
        res = align_traces(t1, t2)
        if len(res.pairs) == 0 and res.is_fully_aligned:
            print(f"  {PASS} [7] Empty traces aligned smoothly with 0 pairs")
            passed += 1
        else:
            print(f"  {FAIL} [7] Empty trace alignment failed")
    except Exception as e:
        print(f"  {FAIL} [7] Empty trace error: {e}")

    # ── Test 8: get_matched_pairs helper ─────────────────────────────────────
    total += 1
    try:
        t1 = _make_trace("r1", _make_step("researcher", 1), _make_step("writer", 2))
        t2 = _make_trace("r2", _make_step("researcher", 1))
        res = align_traces(t1, t2)
        matched_pairs = res.get_matched_pairs()
        if len(matched_pairs) == 1 and matched_pairs[0].agent == "researcher":
            print(f"  {PASS} [8] get_matched_pairs helper filtered correctly")
            passed += 1
        else:
            print(f"  {FAIL} [8] get_matched_pairs helper failed: {matched_pairs}")
    except Exception as e:
        print(f"  {FAIL} [8] get_matched_pairs error: {e}")

    # ── Test 9: get_missing_steps helper ─────────────────────────────────────
    total += 1
    try:
        missing_pairs = res.get_missing_steps()
        if len(missing_pairs) == 1 and missing_pairs[0].agent == "writer":
            print(f"  {PASS} [9] get_missing_steps helper filtered correctly")
            passed += 1
        else:
            print(f"  {FAIL} [9] get_missing_steps helper failed: {missing_pairs}")
    except Exception as e:
        print(f"  {FAIL} [9] get_missing_steps error: {e}")

    # ── Test 10: get_pairs_by_agent helper ────────────────────────────────────
    total += 1
    try:
        res_pairs = res.get_pairs_by_agent("researcher")
        if len(res_pairs) == 1 and res_pairs[0].is_matched:
            print(f"  {PASS} [10] get_pairs_by_agent helper filtered correctly")
            passed += 1
        else:
            print(f"  {FAIL} [10] get_pairs_by_agent helper failed: {res_pairs}")
    except Exception as e:
        print(f"  {FAIL} [10] get_pairs_by_agent error: {e}")

    print(f"\n{passed}/{total} verifications passed.")
    if passed == total:
        print("✅ Day 24 Verification Complete — All 10 checks passed!")
        return 0
    else:
        print(f"❌ {total - passed} verifications failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
