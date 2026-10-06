"""
scripts/verify_day33.py -- Day 33 & Week 6 Checkpoint verification (8 checks)

Run from project root:
    conda run -n agentlens python scripts/verify_day33.py
"""

from __future__ import annotations

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


print("\nDay 33 verification -- Full Walkthrough + Architecture Docs (Week 6 Checkpoint)\n")

arch_path = _ROOT / "ARCHITECTURE.md"
arch_text = arch_path.read_text(encoding="utf-8") if arch_path.exists() else ""

# 1: ARCHITECTURE.md exists and is substantial
check(
    "ARCHITECTURE.md exists and is non-empty",
    arch_path.exists() and len(arch_text) > 2000,
    f"{len(arch_text)} chars",
)

# 2: Component diagram covers full pipeline
required_components = [
    "CaptureSession",
    "Normalizer",
    "EvidenceExtractor",
    "RuleEngine",
    "WorkflowValidator",
    "ConsistencyValidator",
    "Arbiter",
    "LLMExplainer",
]
missing_comp = [c for c in required_components if c not in arch_text]
check(
    "Component diagram covers all pipeline stages",
    len(missing_comp) == 0,
    "all present" if not missing_comp else f"missing: {missing_comp}",
)

# 3: Data flow contracts documented
required_models = [
    "RunTrace",
    "NormalizedRun",
    "ExtractedEvidence",
    "EvidenceRecord",
    "AnalysisBundle",
]
missing_models = [m for m in required_models if m not in arch_text]
check(
    "Data flow contracts (input/output types) documented",
    len(missing_models) == 0,
    "all 5 core models documented" if not missing_models else f"missing: {missing_models}",
)

# 4: DB schema diagram covers all 6 tables
required_tables = ["runs", "steps", "analysis", "metrics", "llm_cache", "rule_matches"]
missing_tables = [t for t in required_tables if t not in arch_text]
check(
    "DB schema diagram covers all 6 SQLite tables",
    "erDiagram" in arch_text and len(missing_tables) == 0,
    "6/6 tables in erDiagram",
)

# 5: Step-by-step guide: How to add a new rule
check(
    "Guide: How to add a new deterministic rule",
    "How to Add a New Deterministic Rule" in arch_text and "RULE_CATALOG" in arch_text,
)

# 6: Step-by-step guide: How to add a new agent extractor
check(
    "Guide: How to add a new agent extractor",
    "How to Add a New Agent Extractor" in arch_text and "_SYSTEM_PROMPT" in arch_text,
)

# 7: All 7 dashboard views registered in dashboard/app.py
app_src = (_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
expected_routes = [
    '@ui.page("/")',
    '@ui.page("/run/{run_id}")',
    '@ui.page("/run/{run_id}/evidence")',
    '@ui.page("/run/{run_id}/explain")',
    '@ui.page("/diff")',
    '@ui.page("/rules")',
    '@ui.page("/metrics")',
]
missing_routes = [r for r in expected_routes if r not in app_src]
check(
    "All 7 dashboard views present in dashboard/app.py",
    len(missing_routes) == 0,
    f"{len(expected_routes) - len(missing_routes)}/{len(expected_routes)} routes",
)

# 8: Week 6 checkpoint modules (REST API + Alerter + Rule Catalog) work end-to-end
try:
    from fastapi.testclient import TestClient

    from analyzers.alerter import Alerter
    from analyzers.rule_catalog import RULE_CATALOG
    from api.main import fastapi_app

    with TestClient(fastapi_app, raise_server_exceptions=False) as tc:
        resp = tc.get("/health")
    check(
        "Week 6 checkpoint modules (API, Alerter, RuleCatalog) verified",
        len(RULE_CATALOG) >= 12 and callable(Alerter) and resp.status_code == 200,
        f"rules={len(RULE_CATALOG)}, /health={resp.status_code}",
    )
except Exception as exc:
    check("Week 6 checkpoint modules verified", False, str(exc))

total = passed + failed
print(f"\nDay 33 verification: {passed}/{total} checks passed ({100 * passed // total}%)")
if failed:
    print(f"{failed} check(s) failed -- review above")
    sys.exit(1)
print("All checks passed -- Day 33 & Week 6 Checkpoint complete!")
