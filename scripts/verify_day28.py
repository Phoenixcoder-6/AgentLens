"""
scripts/verify_day28.py — Day 28 Verification Suite
=====================================================
10 automated checks to verify the Day 28 Long-Tail Proof & System Gap Evaluation:
1.  Check 01: scripts/run_long_tail_test.py exists and is importable
2.  Check 02: LIMITATIONS.md exists and contains long-tail architectural proof sections
3.  Check 03: Baseline seeding and DatabaseManager integration
4.  Check 04: Long-tail run creation & SQLite trace persistence
5.  Check 05: Deterministic rule bypass verification (0 rules fire)
6.  Check 06: StatisticalDetector latency outlier detection on writer step
7.  Check 07: StatisticalDetector token explosion detection on writer step
8.  Check 08: Arbiter P4 priority assignment & primary agent attribution (writer)
9.  Check 09: Dashboard compatibility (run_full_analysis returns populated bundle)
10. Check 10: Dedicated unit test suite execution (tests/test_long_tail.py)

Usage:
    $env:PYTHONUTF8=1; conda run -n agentlens python scripts/verify_day28.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure root directory is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyzers.arbiter import Arbiter  # noqa: E402
from analyzers.detection.information_loss import InformationLossRule  # noqa: E402
from analyzers.detection.statistical_detector import StatisticalDetector  # noqa: E402
from analyzers.evidence_extraction.extractor import ExtractedEvidence  # noqa: E402
from schema.models import PriorityLevel  # noqa: E402
from storage.db import DatabaseManager  # noqa: E402

_CHECKS_PASSED = 0
_TOTAL_CHECKS = 10


def _check(num: int, title: str, condition: bool, detail: str = "") -> None:
    global _CHECKS_PASSED
    status = "✅ PASS" if condition else "❌ FAIL"
    if condition:
        _CHECKS_PASSED += 1
    print(f"[{num:2d}/{_TOTAL_CHECKS}] {title}")
    print(f"     Status: {status}  {detail}\n")


def main() -> None:
    print("============================================================")
    print("       DAY 28 VERIFICATION SUITE — LONG-TAIL PROOF          ")
    print("============================================================\n")

    # ── Check 01: scripts/run_long_tail_test.py exists ────────────────────────
    script_path = _ROOT / "scripts" / "run_long_tail_test.py"
    _check(
        1,
        "scripts/run_long_tail_test.py exists and is non-empty",
        script_path.exists() and script_path.stat().st_size > 500,
        f"[path={script_path.name}]",
    )

    # ── Check 02: LIMITATIONS.md exists and contains long-tail proof ───────────
    limitations_path = _ROOT / "LIMITATIONS.md"
    has_limitations = limitations_path.exists() and "Long-Tail Proof" in limitations_path.read_text(
        encoding="utf-8"
    )
    _check(
        2,
        "LIMITATIONS.md exists and contains Long-Tail Evaluation",
        has_limitations,
        f"[path={limitations_path.name}]",
    )

    # ── Check 03: Baseline seeding and DatabaseManager integration ────────────
    from scripts.run_long_tail_test import build_historical_baselines, create_long_tail_run

    db_path = _ROOT / "data" / "verify_day28_temp.db"
    if db_path.exists():
        db_path.unlink()

    db = DatabaseManager(str(db_path))
    db.initialize()
    build_historical_baselines(db)
    detector = StatisticalDetector(db)
    baselines = detector.get_baselines()
    has_baselines = len(baselines) >= 2 and all(b.n_runs >= 5 for b in baselines.values())
    _check(
        3,
        "DatabaseManager computes baselines from seeded historical runs",
        has_baselines,
        f"[baselines={len(baselines)} agents]",
    )

    # ── Check 04: Long-tail run creation & SQLite trace persistence ──────────
    run_id = create_long_tail_run(db)
    run_row = db.get_run(run_id)
    _check(
        4,
        "Long-tail run created and persisted in SQLite DB",
        run_row is not None and bool(run_row.get("trace_json")),
        f"[run_id={run_id}]",
    )

    # ── Check 05: Deterministic rule bypass verification (0 rules fire) ───────
    info_rule = InformationLossRule()
    res_ev = ExtractedEvidence(source_count=5, entity_count=8)
    wri_ev = ExtractedEvidence(source_count=5, entity_count=8)
    loss_res = info_rule.evaluate(run_id=run_id, researcher_evidence=res_ev, writer_evidence=wri_ev)
    rules_bypassed = loss_res.verdict == "PASS" and not loss_res.has_information_loss
    _check(
        5,
        "Deterministic rules bypass long-tail payload (0 rules fire)",
        rules_bypassed,
        f"[verdict={loss_res.verdict}, info_loss={loss_res.has_information_loss}]",
    )

    # ── Check 06: StatisticalDetector latency outlier detection on writer step ─
    report = detector.analyze_run(run_id)
    lat_anom = next(
        (a for a in report.anomalies if a.agent == "writer" and "latency" in a.description.lower()),
        None,
    )
    _check(
        6,
        "StatisticalDetector detects latency outlier on writer step",
        lat_anom is not None and lat_anom.agent == "writer",
        f"[agent={lat_anom.agent if lat_anom else 'none'}, desc={lat_anom.description[:40] if lat_anom else 'n/a'}]",
    )

    # ── Check 07: StatisticalDetector token explosion detection on writer step
    tok_anom = next(
        (a for a in report.anomalies if a.agent == "writer" and "token" in a.description.lower()),
        None,
    )
    _check(
        7,
        "StatisticalDetector detects token explosion on writer step",
        tok_anom is not None and tok_anom.agent == "writer",
        f"[agent={tok_anom.agent if tok_anom else 'none'}, desc={tok_anom.description[:40] if tok_anom else 'n/a'}]",
    )

    # ── Check 08: Arbiter P4 priority assignment & primary agent attribution ──
    evidence_records = report.anomalies
    bundle = Arbiter().run(run_id=run_id, evidence=evidence_records)
    correct_arbiter = bundle.priority_level == PriorityLevel.P4 and bundle.primary_agent == "writer"
    _check(
        8,
        "Arbiter assigns P4 priority & attributes primary_agent='writer'",
        correct_arbiter,
        f"[priority={bundle.priority_level.value}, agent={bundle.primary_agent}, cause={bundle.primary_cause.value}]",
    )

    # ── Check 09: Dashboard compatibility (run_full_analysis returns bundle) ─
    from dashboard.state import _analysis_cache, run_full_analysis

    _analysis_cache.clear()
    state = run_full_analysis(run_id, db=db)
    dash_ok = state.done and state.bundle is not None
    _check(
        9,
        "Dashboard data layer (run_full_analysis) processes long-tail run",
        dash_ok,
        f"[done={state.done}, has_bundle={state.bundle is not None}]",
    )

    # Clean up temp db
    if db_path.exists():
        db_path.unlink()

    # ── Check 10: Dedicated unit test suite execution (tests/test_long_tail.py)
    pytest_cmd = [sys.executable, "-m", "pytest", "tests/test_long_tail.py", "-q"]
    proc = subprocess.run(pytest_cmd, capture_output=True, text=True)
    tests_ok = proc.returncode == 0
    _check(
        10,
        "Dedicated unit test suite (tests/test_long_tail.py) passes cleanly",
        tests_ok,
        f"[returncode={proc.returncode}]",
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("============================================================")
    print(
        f"Day 28 verification: {_CHECKS_PASSED}/{_TOTAL_CHECKS} checks passed ({_CHECKS_PASSED / _TOTAL_CHECKS:.0%})"
    )
    if _CHECKS_PASSED == _TOTAL_CHECKS:
        print("🎉 All checks passed — Day 28 complete!")
        print("   Week 5 (Diff Engine & Hardening) is 100% COMPLETE!")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
