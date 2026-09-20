"""
scripts/verify_day30.py  --  Day 30 automated verification (8 checks)

Run from project root:
    conda run -n agentlens python scripts/verify_day30.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    sym = "PASS" if ok else "FAIL"
    print(f"  [{sym}] {name}" + (f" -- {detail}" if detail else ""))
    if ok:
        passed += 1
    else:
        failed += 1


print("\nDay 30 verification -- Timeline & Workflow State Viewer\n")

# -- Check 1: get_step_handoff_detail exists in dashboard.state ---------------
try:
    from dashboard.state import get_step_handoff_detail
    sig = inspect.signature(get_step_handoff_detail)
    params = set(sig.parameters.keys())
    check("get_step_handoff_detail(run_id, agent) exists", {"run_id", "agent"}.issubset(params))
except Exception as e:
    check("get_step_handoff_detail exists", False, str(e))

# -- Check 2: get_timeline_data exists in dashboard.state ---------------------
try:
    from dashboard.state import get_timeline_data
    sig2 = inspect.signature(get_timeline_data)
    check("get_timeline_data(run_id) exists", "run_id" in sig2.parameters)
except Exception as e:
    check("get_timeline_data exists", False, str(e))

# -- Check 3: both return correct types for unknown run_id --------------------
try:
    from unittest.mock import patch

    from dashboard.state import get_step_handoff_detail, get_timeline_data  # noqa: F811
    with patch("dashboard.state.get_trace_steps", return_value=[]):
        r1 = get_step_handoff_detail("fake-run", "agent-x")
        r2 = get_timeline_data("fake-run")
    check(
        "Both return empty/list for unknown run",
        r1 == {} and r2 == [],
        f"detail={r1!r} timeline={r2!r}",
    )
except Exception as e:
    check("Graceful empty return", False, str(e))

# -- Check 4: diff_key_badge exists in dashboard.theme ------------------------
try:
    from dashboard.theme import diff_key_badge
    html = diff_key_badge("sources", "added")
    check("diff_key_badge('sources', 'added') returns HTML", "sources" in html and "+" in html)
except Exception as e:
    check("diff_key_badge exists", False, str(e))

# -- Check 5: DIFF_COLORS has all four categories ----------------------------
try:
    from dashboard.theme import DIFF_COLORS
    required = {"added", "modified", "dropped", "unchanged"}
    check(
        "DIFF_COLORS has added/modified/dropped/unchanged",
        required.issubset(DIFF_COLORS.keys()),
        str(set(DIFF_COLORS.keys())),
    )
except Exception as e:
    check("DIFF_COLORS keys", False, str(e))

# -- Check 6: .al-state-card CSS present in GLOBAL_CSS -----------------------
try:
    from dashboard.theme import GLOBAL_CSS
    check(".al-state-card CSS defined", ".al-state-card" in GLOBAL_CSS)
except Exception as e:
    check(".al-state-card CSS", False, str(e))

# -- Check 7: .al-diff-badge CSS present in GLOBAL_CSS -----------------------
try:
    from dashboard.theme import GLOBAL_CSS  # noqa: F811
    check(".al-diff-badge CSS defined", ".al-diff-badge" in GLOBAL_CSS)
except Exception as e:
    check(".al-diff-badge CSS", False, str(e))

# -- Check 8: trace_page function accepts run_id param -----------------------
try:
    # Import the module but do NOT call ui.run -- just inspect the function
    # Only check the function exists and accepts run_id
    import ast
    import pathlib
    src = pathlib.Path("dashboard/app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "trace_page":
            args = [a.arg for a in node.args.args]
            if "run_id" in args:
                found = True
            break
    check("trace_page(run_id) function exists in app.py", found)
except Exception as e:
    check("trace_page function", False, str(e))

# -- Summary ------------------------------------------------------------------
total = passed + failed
pct = 100 * passed // total if total else 0
print(f"\nDay 30 verification: {passed}/{total} checks passed ({pct}%)")
if failed == 0:
    print("All checks passed -- Day 30 complete!")
else:
    print(f"{failed} check(s) failed -- review above")
    sys.exit(1)
