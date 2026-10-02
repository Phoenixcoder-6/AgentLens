"""
scripts/verify_day31.py -- Day 31 automated verification (8 checks)

Run from project root:
    conda run -n agentlens python scripts/verify_day31.py
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


print("\nDay 31 verification -- Evidence Panel + Metrics + Alerting\n")

# -- Check 1: Alerter class exists and importable ----------------------------
try:
    from analyzers.alerter import Alerter
    check("analyzers.alerter.Alerter importable", True)
except Exception as e:
    check("analyzers.alerter.Alerter importable", False, str(e))

# -- Check 2: Alerter.fire() accepts run_id, bundle, dashboard_base_url ------
try:
    from analyzers.alerter import Alerter  # noqa: F811
    sig = inspect.signature(Alerter.fire)
    params = set(sig.parameters.keys())
    check(
        "Alerter.fire() signature correct",
        {"run_id", "bundle", "dashboard_base_url"}.issubset(params),
        str(params),
    )
except Exception as e:
    check("Alerter.fire() signature", False, str(e))

# -- Check 3: Alerter.should_alert() works correctly -------------------------
try:
    from analyzers.alerter import Alerter  # noqa: F811
    alerter = Alerter.__new__(Alerter)
    alerter._enabled = True
    alerter._on_verdict = ["P1", "P2"]
    ok = alerter.should_alert("P1") is True and alerter.should_alert("P5") is False
    check("Alerter.should_alert() filters correctly", ok)
except Exception as e:
    check("Alerter.should_alert()", False, str(e))

# -- Check 4: alerting section exists in config.yaml -------------------------
try:
    from config.config_loader import get
    enabled = get("alerting.enabled", None)
    on_verdict = get("alerting.on_verdict", None)
    channel = get("alerting.channel", None)
    check(
        "alerting: section in config.yaml",
        enabled is not None and on_verdict is not None and channel is not None,
        f"enabled={enabled}, on_verdict={on_verdict}, channel={channel}",
    )
except Exception as e:
    check("alerting: config section", False, str(e))

# -- Check 5: alerter wired into run_full_analysis ---------------------------
try:
    src = Path("dashboard/state.py").read_text(encoding="utf-8")
    check(
        "Alerter.fire() called in run_full_analysis",
        "Alerter" in src and "alerter.fire" in src.lower() or "Alerter().fire" in src,
        "looked for Alerter().fire in state.py",
    )
except Exception as e:
    check("Alerter wired in state.py", False, str(e))

# -- Check 6: get_failure_timeline exists in dashboard.state -----------------
try:
    from dashboard.state import get_failure_timeline
    sig3 = inspect.signature(get_failure_timeline)
    check("get_failure_timeline(days) exists", "days" in sig3.parameters)
except Exception as e:
    check("get_failure_timeline exists", False, str(e))

# -- Check 7: get_cause_breakdown exists in dashboard.state ------------------
try:
    from dashboard.state import get_cause_breakdown
    result = get_cause_breakdown()
    check("get_cause_breakdown() returns list", isinstance(result, list))
except Exception as e:
    check("get_cause_breakdown exists", False, str(e))

# -- Check 8: metrics_page has failure timeline in app.py --------------------
try:

    src = Path("dashboard/app.py").read_text(encoding="utf-8")
    check(
        "metrics_page has failure_timeline and cause_breakdown",
        "get_failure_timeline" in src and "get_cause_breakdown" in src,
    )
except Exception as e:
    check("metrics_page updated", False, str(e))

# -- Summary -----------------------------------------------------------------
total = passed + failed
pct = 100 * passed // total if total else 0
print(f"\nDay 31 verification: {passed}/{total} checks passed ({pct}%)")
if failed == 0:
    print("All checks passed -- Day 31 complete!")
else:
    print(f"{failed} check(s) failed -- review above")
    sys.exit(1)
