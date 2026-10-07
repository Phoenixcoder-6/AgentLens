"""Day 37 verification -- Replay CLI (replay.py)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPLAY_PATH = ROOT / "replay.py"
README_PATH = ROOT / "README.md"
TEST_FILE = ROOT / "tests" / "test_replay.py"


def main() -> int:
    print("\nDay 37 verification -- Replay CLI (replay.py)\n")
    passed = 0
    total = 8

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status = "[PASS]" if ok else "[FAIL]"
        suffix = f" -- {detail}" if detail else ""
        print(f"  {status} {label}{suffix}")
        if ok:
            passed += 1

    # 1. replay.py exists at project root
    exists = REPLAY_PATH.exists() and REPLAY_PATH.stat().st_size > 500
    check("replay.py exists at project root", exists, str(REPLAY_PATH))
    if not exists:
        print(f"\nDay 37 verification: {passed}/{total} checks passed")
        return 1

    import replay

    # 2. --dry-run on PASS run returns exit code 0 (PASS)
    payload_pass, code_pass = replay.execute_replay("run_lbl_pass_01", dry_run=True)
    check(
        "--dry-run on PASS trace returns exit_code=0 (PASS)",
        code_pass == 0 and payload_pass["verdict"] == "PASS" and payload_pass["dry_run"] is True,
        f"exit_code={code_pass}, verdict={payload_pass['verdict']}",
    )

    # 3. --json outputs machine-readable JSON
    buf = io.StringIO()
    with redirect_stdout(buf):
        code_json = replay.main(["run_lbl_pass_01", "--dry-run", "--json"])
    try:
        json_out = json.loads(buf.getvalue())
        json_ok = (
            code_json == 0
            and json_out.get("run_id") == "run_lbl_pass_01"
            and "verdict" in json_out
            and "exit_code" in json_out
            and "primary_agent" in json_out
        )
    except Exception:
        json_ok = False
    check(
        "--json flag outputs valid machine-readable JSON verdict",
        json_ok,
        f"keys={len(json_out) if json_ok else 0}",
    )

    # 4. --override-topic updates effective_topic and preserves attribution stability
    payload_ov, code_ov = replay.execute_replay(
        "run_lbl_reasoning_01",
        dry_run=True,
        override_topic="Solid-state batteries",
    )
    check(
        "--override-topic updates topic and preserves attribution stability",
        (
            code_ov == 2
            and payload_ov["topic_overridden"] is True
            and payload_ov["effective_topic"] == "Solid-state batteries"
            and payload_ov["primary_agent"] == "writer"
        ),
        f"topic='{payload_ov['effective_topic']}', agent={payload_ov['primary_agent']}",
    )

    # 5. Exit code 1 (WARNING) on P3 workflow violation
    payload_warn, code_warn = replay.execute_replay("run_lbl_workflow_01", dry_run=True)
    check(
        "Exit code 1 (WARNING) returned for P3 workflow violation",
        code_warn == 1 and payload_warn["verdict"] == "WARNING",
        f"exit_code={code_warn}, priority={payload_warn['priority_level']}",
    )

    # 6. Exit code 2 (FAIL) on P2 execution failure
    payload_fail, code_fail = replay.execute_replay("run_lbl_execution_01", dry_run=True)
    check(
        "Exit code 2 (FAIL) returned for P1/P2 critical failure",
        code_fail == 2 and payload_fail["verdict"] == "FAIL",
        f"exit_code={code_fail}, agent={payload_fail['primary_agent']}",
    )

    # 7. Exit code 3 (ERROR) on missing run_id
    payload_err, code_err = replay.execute_replay("nonexistent_run_id_000", dry_run=True)
    check(
        "Exit code 3 (ERROR) returned when run_id is not found",
        code_err == 3 and payload_err["verdict"] == "ERROR",
        f"exit_code={code_err}",
    )

    # 8. README.md note & pytest tests/test_replay.py passes
    readme_text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    has_readme_note = (
        "replay.py" in readme_text
        and "deliberate addition beyond the original MVP" in readme_text
        and "--override-topic" in readme_text
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST_FILE), "-v", "-o", "addopts="],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    pytest_ok = proc.returncode == 0 and "11 passed" in proc.stdout
    check(
        "README.md documents replay.py MVP note and pytest test_replay.py passes",
        has_readme_note and pytest_ok,
        "11/11 tests passed",
    )

    pct = int(round(passed / total * 100))
    print(f"\nDay 37 verification: {passed}/{total} checks passed ({pct}%)")
    if passed == total:
        print("All checks passed -- Day 37 complete!\n")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
