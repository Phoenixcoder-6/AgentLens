"""Day 38 verification -- Document Results (LIMITATIONS.md, RULES.md & Week 7 Checkpoint)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIMITATIONS_PATH = ROOT / "LIMITATIONS.md"
RULES_PATH = ROOT / "RULES.md"
RESULTS_DAY34_PATH = ROOT / "validation" / "results_day34.json"
ACCURACY_DAY35_PATH = ROOT / "validation" / "accuracy_day35.json"
ERROR_INJECTION_PATH = ROOT / "tests" / "test_error_injection.py"
REPLAY_PATH = ROOT / "replay.py"


def main() -> int:
    print("\nDay 38 verification -- Document Results (Week 7 Checkpoint)\n")
    passed = 0
    total = 8

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status = "[PASS]" if ok else "[FAIL]"
        suffix = f" -- {detail}" if detail else ""
        print(f"  {status} {label}{suffix}")
        if ok:
            passed += 1

    lim_text = LIMITATIONS_PATH.read_text(encoding="utf-8") if LIMITATIONS_PATH.exists() else ""
    rules_text = RULES_PATH.read_text(encoding="utf-8") if RULES_PATH.exists() else ""

    # 1. LIMITATIONS.md exists and documents empirical accuracy rate
    has_acc = (
        LIMITATIONS_PATH.exists()
        and "16 / 20" in lim_text
        and "80.0%" in lim_text
        and "Empirical Validation Results" in lim_text
    )
    check(
        "LIMITATIONS.md documents empirical validation accuracy rate (16/20 = 80.0%)",
        has_acc,
        f"{len(lim_text)} chars",
    )

    # 2. LIMITATIONS.md documents known failure modes
    has_failure_modes = (
        "Known Failure Modes" in lim_text
        and "Dual-Fault" in lim_text
        and "verifier_passthrough_v1" in lim_text
        and "hallucination_v1" in lim_text
    )
    check(
        "LIMITATIONS.md documents known failure modes & P2 tie-breaking",
        has_failure_modes,
    )

    # 3. LIMITATIONS.md documents known false positives per rule for all 12 rules
    from analyzers.rule_catalog import RULE_CATALOG

    catalog_ids = list(RULE_CATALOG.keys())
    all_in_lim = all(rid in lim_text for rid in catalog_ids) and (
        "Known False Positives" in lim_text
    )
    check(
        "LIMITATIONS.md documents known false positives & blind spots for all 12 rules",
        all_in_lim,
        f"rules={len(catalog_ids)}/12",
    )

    # 4. LIMITATIONS.md documents the Day 28 long-tail proof
    has_long_tail = (
        "Empirical Long-Tail Proof" in lim_text
        and "Day 28" in lim_text
        and "22,500 ms" in lim_text
        and "9,500 tokens" in lim_text
    )
    check(
        "LIMITATIONS.md documents the Day 28 long-tail proof",
        has_long_tail,
    )

    # 5. RULES.md exists and covers all 12 rules in RULE_CATALOG
    all_in_rules = (
        RULES_PATH.exists()
        and len(rules_text) > 1500
        and all(rid in rules_text for rid in catalog_ids)
    )
    check(
        "RULES.md covers all 12 rules across P1-P4",
        all_in_rules,
        f"rules={len(catalog_ids)}/12, {len(rules_text)} chars",
    )

    # 6. RULES.md includes validation results per rule (TP/FP counts from labeled set)
    has_tp_fp = (
        "Validation Results Per Rule" in rules_text
        and "Condition-Level TP" in rules_text
        and "Condition-Level FP" in rules_text
        and "Arbiter-Winner TP" in rules_text
        and "Arbiter-Winner FP" in rules_text
    )
    check(
        "RULES.md includes TP/FP validation counts per rule from labeled set",
        has_tp_fp,
    )

    # 7. Week 7 Checkpoint: 20-run labeled set validated (>= 75% accuracy)
    val_ok = False
    if RESULTS_DAY34_PATH.exists() and ACCURACY_DAY35_PATH.exists():
        acc_data = json.loads(ACCURACY_DAY35_PATH.read_text(encoding="utf-8"))
        val_ok = (
            acc_data.get("total_runs") == 20
            and acc_data.get("target_met") is True
            and acc_data.get("metrics", {}).get("exact_attribution_accuracy", 0.0) >= 0.75
        )
    check(
        "Week 7 Checkpoint: 20-run labeled set validated (>= 75% accuracy)",
        val_ok,
        "16/20 (80.0%)",
    )

    # 8. Week 7 Checkpoint: Error injection suite + Replay CLI with exit codes
    import replay

    _, code_pass = replay.execute_replay("run_lbl_pass_01", dry_run=True)
    _, code_fail = replay.execute_replay("run_lbl_execution_01", dry_run=True)
    checkpoint_ok = (
        ERROR_INJECTION_PATH.exists() and REPLAY_PATH.exists() and code_pass == 0 and code_fail == 2
    )
    check(
        "Week 7 Checkpoint: Error injection suite & Replay CLI exit codes verified",
        checkpoint_ok,
        f"exit_pass={code_pass}, exit_fail={code_fail}",
    )

    pct = int(round(passed / total * 100))
    print(f"\nDay 38 verification: {passed}/{total} checks passed ({pct}%)")
    if passed == total:
        print("All checks passed -- Day 38 & Week 7 Checkpoint complete!\n")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
