"""
scripts/verify_day34.py -- Day 34 automated verification (8 checks)

Run from project root:
    conda run -n agentlens python scripts/verify_day34.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if ok:
        passed += 1
    else:
        failed += 1


print("\nDay 34 verification -- Run the Labeled Set (20 traces)\n")

results_path = _ROOT / "validation" / "results_day34.json"

# 1: Run validation if results_day34.json does not exist yet
if not results_path.exists():
    try:
        from scripts.run_day34_validation import run_validation

        run_validation()
    except Exception as exc:
        check("run_day34_validation executes cleanly", False, str(exc))

check("validation/results_day34.json exists", results_path.exists(), str(results_path))

data = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
runs = data.get("runs", [])

# 2: Exactly 20 labeled runs recorded
check(
    "20 labeled runs recorded",
    len(runs) == 20 and data.get("total_runs") == 20,
    f"count={len(runs)}",
)

# 3: Required fields present on every run record
required_fields = {
    "run_id",
    "label_id",
    "verdict",
    "priority_level",
    "primary_cause",
    "primary_agent",
    "confidence",
    "grounded",
    "rules_fired",
}
missing_any = [r.get("run_id") for r in runs if not required_fields.issubset(set(r.keys()))]
check(
    "Every run record has verdict, primary_agent, confidence, grounded",
    len(runs) == 20 and len(missing_any) == 0,
)

# 4: Zero false positives on all 4 PASS runs (all resolve to P5 / PASS)
pass_runs = [r for r in runs if r.get("label_category") == "pass"]
pass_ok = len(pass_runs) == 4 and all(
    r["verdict"] == "PASS" and r["priority_level"] == "P5" and r["primary_agent"] is None
    for r in pass_runs
)
check("All 4 PASS runs resolve to PASS (P5, zero false positives)", pass_ok)

# 5: All 4 execution_failure runs resolve to P2 execution with researcher blamed
exec_runs = [r for r in runs if r.get("label_category") == "execution_failure"]
exec_ok = len(exec_runs) == 4 and all(
    r["priority_level"] == "P2"
    and r["primary_cause"] == "execution"
    and r["primary_agent"] == "researcher"
    for r in exec_runs
)
check("All 4 execution_failure runs attributed to researcher (P2)", exec_ok)

# 6: All 4 reasoning_failure runs resolve to P2 reasoning with writer blamed
reas_runs = [r for r in runs if r.get("label_category") == "reasoning_failure"]
reas_ok = len(reas_runs) == 4 and all(
    r["priority_level"] == "P2"
    and r["primary_cause"] == "reasoning"
    and r["primary_agent"] == "writer"
    for r in reas_runs
)
check("All 4 reasoning_failure runs attributed to writer (P2)", reas_ok)

# 7: All 4 workflow_failure runs resolve to P3 workflow with expected skipped agent blamed
wf_runs = [r for r in runs if r.get("label_category") == "workflow_failure"]
wf_ok = len(wf_runs) == 4 and all(
    r["priority_level"] == "P3"
    and r["primary_cause"] == "workflow"
    and r["primary_agent"] == r["expected_primary_agent"]
    for r in wf_runs
)
check("All 4 workflow_failure runs attributed to skipped agent (P3)", wf_ok)

# 8: Rule matches persisted in main DB rule_matches table
try:
    from storage.db import DatabaseManager

    db = DatabaseManager()
    db.initialize()
    rm_count = db.rule_match_count()
    check("rule_matches table populated in SQLite DB", rm_count >= 16, f"rows={rm_count}")
except Exception as exc:
    check("rule_matches table populated in SQLite DB", False, str(exc))

total = passed + failed
print(f"\nDay 34 verification: {passed}/{total} checks passed ({100 * passed // total}%)")
if failed:
    print(f"{failed} check(s) failed -- review above")
    sys.exit(1)
print("All checks passed -- Day 34 complete!")
