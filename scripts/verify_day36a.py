"""Day 36a verification -- CI/CD Hardening (.github/workflows/ci.yml & README.md)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"
README_PATH = ROOT / "README.md"


def main() -> int:
    print("\nDay 36a verification -- CI/CD Hardening\n")
    passed = 0
    total = 8

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status = "[PASS]" if ok else "[FAIL]"
        suffix = f" -- {detail}" if detail else ""
        print(f"  {status} {label}{suffix}")
        if ok:
            passed += 1

    # 1. .github/workflows/ci.yml exists and is valid YAML
    ci_exists = CI_PATH.exists() and CI_PATH.stat().st_size > 200
    ci_text = CI_PATH.read_text(encoding="utf-8") if ci_exists else ""
    try:
        ci_doc = yaml.safe_load(ci_text) if ci_exists else {}
        yaml_valid = isinstance(ci_doc, dict) and "jobs" in ci_doc
    except Exception:
        ci_doc = {}
        yaml_valid = False
    check(
        ".github/workflows/ci.yml exists and is valid YAML",
        ci_exists and yaml_valid,
        f"jobs={list((ci_doc.get('jobs') or {}).keys())}",
    )

    # 2. lint job runs ruff check, ruff format --check, and mypy . --ignore-missing-imports
    has_lint_cmds = (
        "ruff check ." in ci_text
        and "ruff format --check ." in ci_text
        and "mypy . --ignore-missing-imports" in ci_text
    )
    check(
        "CI lint job runs ruff check, ruff format, and mypy --ignore-missing-imports",
        has_lint_cmds,
    )

    # 3. test job runs pytest tests/ with --cov=. and --cov-fail-under=70
    has_cov_gate = (
        "pytest tests/" in ci_text and "--cov=." in ci_text and "--cov-fail-under=70" in ci_text
    )
    check(
        "CI test job enforces pytest tests/ --cov=. --cov-fail-under=70",
        has_cov_gate,
    )

    # 4. validation-gate job runs Day 34 validation set and Day 35 accuracy gate (>= 75%)
    has_val_gate = (
        "validation-gate" in (ci_doc.get("jobs") or {})
        and "python scripts/run_day34_validation.py" in ci_text
        and "python scripts/evaluate_day35_accuracy.py" in ci_text
    )
    check(
        "CI validation-gate job runs Day 34 validation and fails if accuracy < 75%",
        has_val_gate,
    )

    # 5. README.md has CI status badge
    readme_text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    has_badge = "actions/workflows/ci.yml/badge.svg" in readme_text
    check(
        "README.md includes GitHub Actions CI status badge",
        has_badge,
    )

    # 6. README.md documents branch protection rules
    has_branch_protection = (
        "Branch Protection" in readme_text
        and "Require status checks to pass before merging" in readme_text
    )
    check(
        "README.md documents branch protection rules for main",
        has_branch_protection,
    )

    # 7. Local ruff check on Day 34-36a scripts and tests passes
    ruff_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "scripts/run_day34_validation.py",
            "scripts/evaluate_day35_accuracy.py",
            "tests/test_error_injection.py",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    check(
        "Ruff lint passes cleanly on validation & error-injection suite",
        ruff_proc.returncode == 0,
        ruff_proc.stdout.strip().splitlines()[-1] if ruff_proc.stdout.strip() else "clean",
    )

    # 8. Accuracy gate (evaluate_day35_accuracy.py) passes (>= 75% accuracy)
    from scripts.evaluate_day35_accuracy import evaluate_day35

    acc_payload = evaluate_day35()
    acc_metrics = acc_payload.get("metrics", {})
    acc_val = acc_metrics.get("exact_attribution_accuracy", 0.0)
    check(
        "Validation accuracy gate (evaluate_day35_accuracy.py) passes (>= 75%)",
        acc_payload.get("target_met") is True and acc_val >= 0.75,
        f"accuracy={acc_val * 100:.1f}% ({acc_metrics.get('exact_attribution_correct')}/{acc_payload.get('total_runs')})",
    )

    pct = int(round(passed / total * 100))
    print(f"\nDay 36a verification: {passed}/{total} checks passed ({pct}%)")
    if passed == total:
        print("All checks passed -- Day 36a complete!\n")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
