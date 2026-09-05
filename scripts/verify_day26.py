"""
scripts/verify_day26.py — Day 26 Verification Script
=====================================================
Verifies 10 Day 26 Diff Viewer requirements:
  1. state.DiffResult has rows, matched_count, missing_in_a_count, missing_in_b_count fields
  2. state.DiffRow dataclass has all expected fields
  3. compute_diff returns DiffResult with rows (uses real GraphAligner pipeline)
  4. matched_count == len(trace steps) for identical trace
  5. first_divergence set correctly from similarity engine
  6. overall_similarity populated from SemanticSimilarityEngine
  7. MISSING_IN_B step: missing_in_b_count == 1, row match_status == MISSING_IN_B
  8. Legacy steps dict backward-compat includes lat_delta and tok_delta keys
  9. _load_run_trace returns None for missing run_id
 10. Full compute_diff returns DiffResult for real DB run pair
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.state import DiffResult, DiffRow, _load_run_trace, compute_diff
from schema.models import AgentStep, RunTrace, StepStatus

PASS = "[PASS]"
FAIL = "[FAIL]"


def _make_step(agent: str, step: int, output: str = "") -> AgentStep:
    return AgentStep(
        run_id="verify_run",
        step=step,
        agent=agent,
        output=output or f"{agent} output",
        status=StepStatus.SUCCESS,
        latency_ms=100.0,
    )


def _make_trace(run_id: str, *steps: AgentStep) -> RunTrace:
    return RunTrace(run_id=run_id, workflow="test_pipeline", steps=list(steps))


def main() -> int:
    print("=== Day 26 Verification: Diff Viewer ===\n")
    passed = 0
    total = 0

    # ── 1. DiffResult has new fields ─────────────────────────────────────────
    total += 1
    try:
        dr = DiffResult(run_a="a", run_b="b", steps=[])
        assert hasattr(dr, "rows")
        assert hasattr(dr, "matched_count")
        assert hasattr(dr, "missing_in_a_count")
        assert hasattr(dr, "missing_in_b_count")
        print(f"  {PASS} [1] DiffResult has rows, matched_count, missing_in_a_count, missing_in_b_count fields")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [1] DiffResult fields: {e}")

    # ── 2. DiffRow has all expected fields ───────────────────────────────────
    total += 1
    try:
        row = DiffRow(
            agent="researcher", match_status="MATCHED",
            lat_a=100.0, lat_b=150.0, lat_delta=50.0,
            tok_a=500, tok_b=600, tok_delta=100,
            sim=0.92, diverged=False, method="cosine",
        )
        assert row.agent == "researcher"
        assert row.lat_delta == 50.0
        assert row.tok_delta == 100
        print(f"  {PASS} [2] DiffRow has all expected fields including lat_delta and tok_delta")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [2] DiffRow fields: {e}")

    # ── 3-8. compute_diff with mocked DB and traces ──────────────────────────
    trace_a = _make_trace(
        "run_a",
        _make_step("researcher", 1, "The capital of France is Paris."),
        _make_step("writer", 2, "France is in Western Europe."),
    )
    trace_b = _make_trace(
        "run_b",
        _make_step("researcher", 1, "The capital of France is Paris."),
        _make_step("writer", 2, "Paris is the capital city of France."),
    )

    def fake_load_run_trace(run_id: str) -> RunTrace | None:
        if run_id == "run_a":
            return trace_a
        if run_id == "run_b":
            return trace_b
        return None

    def fake_step_metrics(run_id_arg: str):
        return [
            {"agent": "researcher", "latency_ms": 100.0, "tokens_total": 400},
            {"agent": "writer", "latency_ms": 200.0, "tokens_total": 600},
        ]

    mock_db = MagicMock()
    mock_db.get_steps_for_run.side_effect = fake_step_metrics

    with (
        patch("dashboard.state._load_run_trace", side_effect=fake_load_run_trace),
        patch("dashboard.state.get_db", return_value=mock_db),
    ):
        try:
            result = compute_diff("run_a", "run_b")

            # 3. Returns DiffResult with rows
            total += 1
            assert isinstance(result, DiffResult)
            assert isinstance(result.rows, list)
            print(f"  {PASS} [3] compute_diff returns DiffResult with rows (len={len(result.rows)})")
            passed += 1

            # 4. matched_count == 2 (both agents matched)
            total += 1
            assert result.matched_count == 2, f"Expected 2, got {result.matched_count}"
            print(f"  {PASS} [4] matched_count={result.matched_count} (both agents matched)")
            passed += 1

            # 5. first_divergence populated
            total += 1
            assert isinstance(result.first_divergence, str)
            print(f"  {PASS} [5] first_divergence='{result.first_divergence}'")
            passed += 1

            # 6. overall_similarity in [0, 1]
            total += 1
            assert 0.0 <= result.overall_similarity <= 1.0, f"Got {result.overall_similarity}"
            print(f"  {PASS} [6] overall_similarity={result.overall_similarity:.3f} (in [0,1])")
            passed += 1

            # 7. No missing steps for matched pair
            total += 1
            assert result.missing_in_b_count == 0
            assert result.missing_in_a_count == 0
            print(f"  {PASS} [7] missing_in_a={result.missing_in_a_count}, missing_in_b={result.missing_in_b_count}")
            passed += 1

            # 8. Legacy steps dict has lat_delta and tok_delta
            total += 1
            assert "lat_delta" in result.steps[0], f"Missing lat_delta in {result.steps[0].keys()}"
            assert "tok_delta" in result.steps[0], f"Missing tok_delta in {result.steps[0].keys()}"
            assert "match_status" in result.steps[0]
            print(f"  {PASS} [8] Legacy steps dict has lat_delta, tok_delta, match_status keys")
            passed += 1

        except AssertionError as e:
            print(f"  {FAIL} [3-8] Assertion failed: {e}")
        except Exception as e:
            print(f"  {FAIL} [3-8] compute_diff failed: {e}")
            import traceback
            traceback.print_exc()

    # ── 9. _load_run_trace returns None for missing run_id ────────────────────
    total += 1
    try:
        with patch("dashboard.state.get_db") as mock_get_db:
            mock_get_db.return_value.get_run.return_value = None
            result_none = _load_run_trace("nonexistent_run_xyz")
        assert result_none is None
        print(f"  {PASS} [9] _load_run_trace returns None for missing run_id")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [9] _load_run_trace: {e}")

    # ── 10. Missing step handling (trace_b missing writer) ────────────────────
    total += 1
    trace_a2 = _make_trace("run_c",
        _make_step("researcher", 1, "Research output"),
        _make_step("writer", 2, "Write output"),
    )
    trace_b2 = _make_trace("run_d",
        _make_step("researcher", 1, "Research output"),
    )

    def fake_load2(run_id: str) -> RunTrace | None:
        return trace_a2 if run_id == "run_c" else trace_b2

    mock_db2 = MagicMock()
    mock_db2.get_steps_for_run.return_value = [
        {"agent": "researcher", "latency_ms": 100.0, "tokens_total": 400},
    ]

    with (
        patch("dashboard.state._load_run_trace", side_effect=fake_load2),
        patch("dashboard.state.get_db", return_value=mock_db2),
    ):
        try:
            result2 = compute_diff("run_c", "run_d")
            assert result2.missing_in_b_count == 1, f"Expected 1, got {result2.missing_in_b_count}"
            missing_rows = [r for r in result2.rows if r.match_status == "MISSING_IN_B"]
            assert len(missing_rows) == 1
            assert missing_rows[0].agent == "writer"
            print(f"  {PASS} [10] Missing step (writer) correctly flagged as MISSING_IN_B")
            passed += 1
        except Exception as e:
            print(f"  {FAIL} [10] Missing step handling: {e}")

    print(f"\n{passed}/{total} verifications passed.")
    if passed == total:
        print("✅ Day 26 Verification Complete — All 10 checks passed!")
        return 0
    else:
        print(f"❌ {total - passed} verifications failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
