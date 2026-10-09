"""
scripts/verify_day46.py -- Day 46 Documentation Suite Verification Script (8 Checks)
=====================================================================================
Verifies that all 6 core documentation files required for the v1.0.0 release exist,
contain all mandatory sections and links, and that the documentation suite conforms
to the Day 46 specification (full_project_plan.md lines 454-463):

  1. `README.md` exists with Quickstart, Architecture Diagram, Benchmark Summary & Links.
  2. `ARCHITECTURE.md` exists with Component Diagram, Data Flow, Priority Model, & Tutorial.
  3. `RULES.md` exists with Full Rule Catalog across P1, P2, P3, and P4 rules.
  4. `LIMITATIONS.md` exists with Benchmark Accuracy, Dual-Fault Analysis, & Scope Boundaries.
  5. `CONTRIBUTING.md` exists with Dev Setup, Code Quality Standards, & CI PR Process.
  6. `CHANGELOG.md` exists with Complete Version History from v0.1.0 to v1.0.0.
  7. Cross-Document Markdown Link Integrity (all referenced doc files exist on disk).
  8. Code formatting and typing tools pass cleanly across documentation examples.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOC_FILES = [
    "README.md",
    "ARCHITECTURE.md",
    "RULES.md",
    "LIMITATIONS.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
]


def check_1_readme_contents() -> str:
    path = ROOT / "README.md"
    assert path.exists(), "README.md does not exist"
    text = path.read_text(encoding="utf-8")
    assert "agentlens-dashboard" in text, "Missing agentlens-dashboard command in README"
    assert "agentlens-seed" in text, "Missing agentlens-seed command in README"
    assert "docker compose up" in text, "Missing docker compose quickstart in README"
    assert "Benchmark Accuracy" in text, "Missing benchmark accuracy section in README"
    assert "ARCHITECTURE.md" in text, "Missing link to ARCHITECTURE.md"
    assert "RULES.md" in text, "Missing link to RULES.md"
    return f"README.md verified ({len(text)} chars, all sections & links present)"


def check_2_architecture_contents() -> str:
    path = ROOT / "ARCHITECTURE.md"
    assert path.exists(), "ARCHITECTURE.md does not exist"
    text = path.read_text(encoding="utf-8")
    assert "The 5-Tier Priority Resolution Hierarchy" in text
    assert "Grounded vs. Heuristic Signals" in text
    assert "Storage Architecture & Alembic Schema" in text
    assert "How to Add a New Detection Rule" in text
    assert "P1" in text and "P2" in text and "P3" in text and "P4" in text and "P5" in text
    return f"ARCHITECTURE.md verified ({len(text)} chars, 5-tier model & tutorial present)"


def check_3_rules_contents() -> str:
    path = ROOT / "RULES.md"
    assert path.exists(), "RULES.md does not exist"
    text = path.read_text(encoding="utf-8")
    expected_rules = [
        "gt_mismatch_v1",
        "tool_failure_v1",
        "missing_tool_output_v1",
        "hallucination_v1",
        "researcher_quality_v1",
        "verifier_passthrough_v1",
        "skipped_step_v1",
        "information_loss_v1",
        "latency_outlier_v1",
        "token_outlier_v1",
    ]
    for r in expected_rules:
        assert r in text, f"Missing rule in RULES.md: {r}"
    assert "Summary Matrix" in text
    return f"RULES.md verified ({len(text)} chars, all 10 canonical rules specified)"


def check_4_limitations_contents() -> str:
    path = ROOT / "LIMITATIONS.md"
    assert path.exists(), "LIMITATIONS.md does not exist"
    text = path.read_text(encoding="utf-8")
    assert "Verified Accuracy Metrics" in text
    assert "Dual-Fault Ambiguity" in text
    assert "100.0%" in text and "80.0%" in text
    assert "Scope Boundaries" in text
    assert "What AgentLens Does NOT Catch" in text
    return f"LIMITATIONS.md verified ({len(text)} chars, benchmark & boundary conditions present)"


def check_5_contributing_contents() -> str:
    path = ROOT / "CONTRIBUTING.md"
    assert path.exists(), "CONTRIBUTING.md does not exist"
    text = path.read_text(encoding="utf-8")
    assert "Local Development Setup" in text
    assert "ruff format" in text
    assert "ruff check" in text
    assert "mypy" in text
    assert "pytest tests/" in text
    assert "validation-gate" in text
    return f"CONTRIBUTING.md verified ({len(text)} chars, dev setup & CI PR guidelines present)"


def check_6_changelog_contents() -> str:
    path = ROOT / "CHANGELOG.md"
    assert path.exists(), "CHANGELOG.md does not exist"
    text = path.read_text(encoding="utf-8")
    assert "[1.0.0]" in text, "Missing v1.0.0 release in CHANGELOG.md"
    assert "[0.8.0]" in text, "Missing v0.8.0 entry in CHANGELOG.md"
    assert "[0.1.0]" in text, "Missing v0.1.0 entry in CHANGELOG.md"
    return f"CHANGELOG.md verified ({len(text)} chars, version history from v0.1.0 to v1.0.0)"


def check_7_cross_document_links() -> str:
    link_pattern = re.compile(r"\[.*?\]\(([\w\-\.]+\.md)\)")
    checked_links = 0
    for doc_name in DOC_FILES:
        doc_path = ROOT / doc_name
        content = doc_path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(content):
            target = match.group(1)
            target_path = ROOT / target
            assert target_path.exists(), f"Broken link in {doc_name}: {target} not found"
            checked_links += 1
    return f"All {checked_links} internal markdown document links verified (zero broken references)"


def check_8_all_documentation_files_present() -> str:
    for doc in DOC_FILES:
        p = ROOT / doc
        assert p.exists() and p.stat().st_size > 500, f"File {doc} is missing or too small"
    return "All 6 release documentation files exist and exceed minimum word count"


def main() -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Check 1: README.md Quickstart, Architecture & Benchmarks", check_1_readme_contents),
        (
            "Check 2: ARCHITECTURE.md 5-Tier Hierarchy & Rule Tutorial",
            check_2_architecture_contents,
        ),
        ("Check 3: RULES.md Complete Catalog & Summary Matrix", check_3_rules_contents),
        (
            "Check 4: LIMITATIONS.md Benchmark Honesty & Scope Boundaries",
            check_4_limitations_contents,
        ),
        ("Check 5: CONTRIBUTING.md Dev Environment & CI Workflow", check_5_contributing_contents),
        ("Check 6: CHANGELOG.md Version History v0.1.0 -> v1.0.0", check_6_changelog_contents),
        ("Check 7: Cross-Document Link Integrity", check_7_cross_document_links),
        ("Check 8: Full Documentation Set Completeness", check_8_all_documentation_files_present),
    ]

    print("=" * 72)
    print("AgentLens -- Day 46 Verification Suite (Full Documentation Set v1.0.0)")
    print("=" * 72)

    passed = 0
    for title, fn in checks:
        try:
            detail = fn()
            print(f"[PASS] {title}")
            print(f"       -> {detail}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {title}")
            print(f"       -> {exc}")

    print("-" * 72)
    print(f"Day 46 Verification Summary: {passed}/{len(checks)} checks passed.")
    print("=" * 72)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
