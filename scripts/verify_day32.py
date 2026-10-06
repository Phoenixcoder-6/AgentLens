"""
scripts/verify_day32.py -- Day 32 verification (8 checks)

Run from project root:
    conda run -n agentlens python scripts/verify_day32.py
"""

from __future__ import annotations

import sys
import tempfile
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


print("\nDay 32 verification -- Diff polish + Rule Explorer\n")

# 1: rule_matches table is created
try:
    from storage.db import DatabaseManager

    tmp = tempfile.mkdtemp()
    db = DatabaseManager(str(Path(tmp) / "v.db"))
    db.initialize()
    db.initialize()
    check("rule_matches table created (idempotent init)", db.rule_match_count() == 0)
except Exception as e:
    check("rule_matches table created", False, str(e))
    sys.exit(1)

# 2: insert / aggregate round trip
try:
    db.insert_run("r1", "wf", "2026-01-01T00:00:00", "success", 1.0, 1, "1.0")
    db.insert_rule_matches(
        "r1",
        [{"rule_id": "hallucination_v1", "category": "reasoning", "description": "x"}],
    )
    stats = db.get_rule_stats()
    check(
        "insert + get_rule_stats round trip",
        len(stats) == 1 and stats[0]["times_fired"] == 1 and stats[0]["example_run"] == "r1",
    )
except Exception as e:
    check("insert + get_rule_stats round trip", False, str(e))

# 3: catalog completeness
try:
    from analyzers.rule_catalog import RULE_CATALOG, normalize_rule_id

    check("catalog has >= 12 rules", len(RULE_CATALOG) >= 12, str(len(RULE_CATALOG)))
except Exception as e:
    check("catalog", False, str(e))

# 4: statistical ID normalization
try:
    check(
        "STAT ids normalized",
        normalize_rule_id("STAT-LAT-WRITER-001") == "STAT-LAT"
        and normalize_rule_id("STAT-TOK-WRITER-002") == "STAT-TOK",
    )
except Exception as e:
    check("STAT ids normalized", False, str(e))

# 5: state.get_rule_stats lists never-fired rules
try:
    from dashboard import state

    rows = state.get_rule_stats(db=db)
    never = [r for r in rows if r["times_fired"] == 0]
    check("get_rule_stats includes never-fired rules", len(never) >= 10, str(len(never)))
except Exception as e:
    check("get_rule_stats", False, str(e))

# 6: category filter
try:
    rows = state.get_rule_stats(category="workflow", db=db)
    check(
        "category filter works",
        bool(rows) and all(r["category"] == "workflow" for r in rows),
    )
except Exception as e:
    check("category filter", False, str(e))

# 7: persistence wired into run_full_analysis
try:
    src = (_ROOT / "dashboard" / "state.py").read_text(encoding="utf-8")
    check("persist_rule_matches wired in run_full_analysis", "persist_rule_matches(run_id" in src)
except Exception as e:
    check("persistence wired", False, str(e))

# 8: /rules page + nav tab + diff guards present
try:
    app_src = (_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    check(
        "/rules page, nav tab and diff guards present",
        '@ui.page("/rules")' in app_src
        and '"/rules"' in app_src
        and "semantically identical" in app_src
        and "same run" in app_src,
    )
except Exception as e:
    check("app.py updates", False, str(e))

total = passed + failed
print(f"\nDay 32 verification: {passed}/{total} checks passed")
if failed:
    print(f"{failed} check(s) failed -- review above")
    sys.exit(1)
print("All checks passed -- Day 32 complete!")
