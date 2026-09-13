"""
scripts/verify_day29.py -- Day 29 automated verification (8 checks)

Run from project root:
    conda run -n agentlens python scripts/verify_day29.py
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


print("\nDay 29 verification -- Run Explorer Polish + REST API Layer\n")

# -- Check 1: RunRow has verdict_level and stale_verdict ----------------------
try:
    from dashboard.state import RunRow

    rr = RunRow(
        run_id="x",
        workflow="w",
        topic="t",
        timestamp="ts",
        status="ok",
        latency_ms=0,
        tokens_total=0,
        step_count=0,
    )
    check("RunRow has verdict_level default UNANALYZED", rr.verdict_level == "UNANALYZED")
    check("RunRow has stale_verdict default False", rr.stale_verdict is False)
except Exception as e:
    check("RunRow verdict_level / stale_verdict", False, str(e))
    check("RunRow stale_verdict field", False, "skipped")

# -- Check 2: list_runs filter params accepted --------------------------------
try:
    from dashboard.state import list_runs

    sig = inspect.signature(list_runs)
    params = set(sig.parameters.keys())
    has_filters = {"agent_filter", "verdict_filter", "date_from", "date_to", "sort_by"}.issubset(
        params
    )
    check("list_runs accepts filter/sort params", has_filters, str(params))
except Exception as e:
    check("list_runs filter params", False, str(e))

# -- Check 3: get_unique_agents returns list ----------------------------------
try:
    from dashboard.state import get_unique_agents

    agents = get_unique_agents()
    check("get_unique_agents() returns list", isinstance(agents, list))
except Exception as e:
    check("get_unique_agents()", False, str(e))

# -- Check 4: get_aggregate_stats works with empty list -----------------------
try:
    from dashboard.state import RunRow, get_aggregate_stats  # noqa: F811

    stats = get_aggregate_stats([])
    keys = {"total", "analyzed", "p1_p2_count", "avg_latency", "total_tokens", "top_failing_agent"}
    check("get_aggregate_stats() returns expected keys", keys.issubset(stats.keys()))
except Exception as e:
    check("get_aggregate_stats()", False, str(e))

# -- Check 5: theme has ROW_TINT_P, priority_row_bg, stale_badge --------------
try:
    from dashboard.theme import ROW_TINT_P, priority_row_bg, stale_badge

    check(
        "ROW_TINT_P has P1-P5 and UNANALYZED",
        set(ROW_TINT_P.keys()) >= {"P1", "P2", "P3", "P4", "P5", "UNANALYZED"},
    )
    check("priority_row_bg('P1') returns non-empty string", bool(priority_row_bg("P1")))
    sb = stale_badge()
    check("stale_badge() returns HTML with Stale text", "Stale" in sb)
except Exception as e:
    check("theme Day 29 helpers", False, str(e))
    check("priority_row_bg", False, "skipped")
    check("stale_badge", False, "skipped")

# -- Check 6: api.router imports cleanly --------------------------------------
try:
    from api.router import health_router, router

    routes = [r.path for r in router.routes]
    check("api.router has /runs endpoint", any("/runs" in r for r in routes), str(routes))
    health_routes = [r.path for r in health_router.routes]
    check("health_router has /health endpoint", any("/health" in r for r in health_routes))
except Exception as e:
    check("api.router imports", False, str(e))
    check("health_router imports", False, "skipped")

# -- Check 7: FastAPI app creates successfully --------------------------------
try:
    from api.main import fastapi_app

    check("api.main.fastapi_app is FastAPI", hasattr(fastapi_app, "routes"))
except Exception as e:
    check("api.main fastapi_app", False, str(e))

# -- Check 8: /health endpoint via TestClient ---------------------------------
try:
    from fastapi.testclient import TestClient

    from api.main import fastapi_app  # noqa: F811

    with TestClient(fastapi_app, raise_server_exceptions=False) as tc:
        resp = tc.get("/health")
        body = resp.json()
    check(
        "/health returns 200 with status field",
        resp.status_code == 200 and "status" in body,
        f"status={body.get('status')} db={body.get('db_status')}",
    )
except Exception as e:
    check("/health endpoint smoke test", False, str(e))

# -- Summary ------------------------------------------------------------------
total = passed + failed
pct = 100 * passed // total if total else 0
print(f"\nDay 29 verification: {passed}/{total} checks passed ({pct}%)")
if failed == 0:
    print("All checks passed -- Day 29 complete!")
else:
    print(f"{failed} check(s) failed -- review above")
    sys.exit(1)
