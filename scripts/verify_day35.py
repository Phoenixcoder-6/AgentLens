"""Day 35 verification -- Compare Human vs. AgentLens (Accuracy & Disagreements)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACCURACY_JSON_PATH = ROOT / "validation" / "accuracy_day35.json"
ACCURACY_REPORT_PATH = ROOT / "validation" / "ACCURACY_REPORT.md"

EXPECTED_CATEGORIES = {
    "pass",
    "reasoning_failure",
    "execution_failure",
    "workflow_failure",
    "verification_failure",
}


def main() -> int:
    print("\nDay 35 verification -- Compare Human vs. AgentLens\n")
    passed = 0
    total = 8

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status = "[PASS]" if ok else "[FAIL]"
        suffix = f" -- {detail}" if detail else ""
        print(f"  {status} {label}{suffix}")
        if ok:
            passed += 1

    # 1. validation/accuracy_day35.json exists
    exists_json = ACCURACY_JSON_PATH.exists() and ACCURACY_JSON_PATH.stat().st_size > 100
    check("validation/accuracy_day35.json exists", exists_json, str(ACCURACY_JSON_PATH))
    if not exists_json:
        print(f"\nDay 35 verification: {passed}/{total} checks passed")
        return 1

    data = json.loads(ACCURACY_JSON_PATH.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    comparisons = data.get("comparisons", [])
    per_cat = data.get("per_category_metrics", {})
    confusion = data.get("confusion_matrix", {})
    disagreements = data.get("disagreements", [])

    # 2. validation/ACCURACY_REPORT.md exists and contains required sections
    exists_md = ACCURACY_REPORT_PATH.exists() and ACCURACY_REPORT_PATH.stat().st_size > 500
    md_text = ACCURACY_REPORT_PATH.read_text(encoding="utf-8") if exists_md else ""
    has_sections = all(
        sec in md_text
        for sec in (
            "Aggregate Accuracy Summary",
            "Per-Category Precision, Recall & F1",
            "Confusion Matrix",
            "Disagreement Log",
            "Run-by-Run Comparison Table",
        )
    )
    check(
        "validation/ACCURACY_REPORT.md exists with all 5 sections",
        exists_md and has_sections,
        f"{len(md_text)} chars",
    )

    # 3. All 20 labeled runs compared
    check(
        "All 20 labeled runs compared against labels.json",
        len(comparisons) == 20 and data.get("total_runs") == 20,
        f"count={len(comparisons)}",
    )

    # 4. Target accuracy >= 15/20 (75%) met
    exact_correct = metrics.get("exact_attribution_correct", 0)
    exact_acc = metrics.get("exact_attribution_accuracy", 0.0)
    check(
        "Target attribution accuracy >= 15/20 (75%) achieved",
        exact_correct >= 15 and exact_acc >= 0.75 and data.get("target_met") is True,
        f"{exact_correct}/20 ({exact_acc * 100:.1f}%)",
    )

    # 5. Binary PASS/FAIL detection is 20/20 (100%)
    bin_correct = metrics.get("binary_pass_fail_correct", 0)
    bin_acc = metrics.get("binary_pass_fail_accuracy", 0.0)
    check(
        "Binary PASS/FAIL detection accuracy is 20/20 (100%)",
        bin_correct == 20 and bin_acc == 1.0,
        f"{bin_correct}/20 ({bin_acc * 100:.1f}%)",
    )

    # 6. Per-category Precision, Recall, F1, and Rule-Level Recall computed for all 5 categories
    cat_keys_ok = set(per_cat.keys()) == EXPECTED_CATEGORIES
    fields_ok = all(
        {"precision", "recall", "f1", "support", "tp", "fp", "fn", "rule_level_recall"}.issubset(
            per_cat[c].keys()
        )
        for c in per_cat
    )
    macro_rule_recall = metrics.get("macro_rule_level_recall", 0.0)
    check(
        "Per-category precision/recall/F1 computed for all 5 categories",
        cat_keys_ok and fields_ok and macro_rule_recall == 1.0,
        f"categories={len(per_cat)}, macro_rule_recall={macro_rule_recall * 100:.0f}%",
    )

    # 7. 5x5 Confusion matrix sums to 20
    cm_sum = sum(sum(row.values()) for row in confusion.values() if isinstance(row, dict))
    check(
        "5x5 Confusion matrix populated and sums to 20",
        set(confusion.keys()) == EXPECTED_CATEGORIES and cm_sum == 20,
        f"sum={cm_sum}",
    )

    # 8. Disagreements logged with root-cause notes on why
    expected_disagreements = 20 - exact_correct
    disagreements_valid = (
        len(disagreements) == expected_disagreements
        and len(disagreements) > 0
        and all(
            bool(d.get("disagreement_reason")) and bool(d.get("human_notes")) for d in disagreements
        )
    )
    check(
        "All disagreements logged with root-cause explanation notes",
        disagreements_valid,
        f"disagreements={len(disagreements)}",
    )

    pct = int(round(passed / total * 100))
    print(f"\nDay 35 verification: {passed}/{total} checks passed ({pct}%)")
    if passed == total:
        print("All checks passed -- Day 35 complete!\n")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
