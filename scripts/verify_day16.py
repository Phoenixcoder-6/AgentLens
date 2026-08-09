import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzers.arbiter import Arbiter
from storage.db import DEFAULT_DB_PATH


def verify_day16():
    print("=== Day 16 Verification ===\n")

    # 1. Check DB Schema
    print("[1/3] Checking SQLite Schema...")
    if not os.path.exists(DEFAULT_DB_PATH):
        print(f"  [!] DB not found at {DEFAULT_DB_PATH}. Run Day 15 verify first.")
    else:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(runs)")
        columns = [row[1] for row in cursor.fetchall()]
        if "expected_output" in columns:
            print("  [PASS] 'expected_output' column successfully added to 'runs' table.")
        else:
            print("  [FAIL] 'expected_output' column missing!")
        conn.close()

    # 2. Check Arbiter Logging & Grounded Logic
    print("\n[2/3] Checking Arbiter Logging & Grounded Logic...")
    arbiter = Arbiter()

    # Send an empty run to trigger P5 fallback and logging
    print("  Triggering Arbiter analysis to generate logs...")
    bundle = arbiter.run("verify_run_001", [])
    if not bundle.grounded and bundle.priority_level.value == "P5":
        print("  [PASS] Arbiter correctly handled ungrounded P5 run.")
    else:
        print("  [FAIL] Arbiter did not return P5 or flagged ungrounded run as grounded.")

    # 3. Check Log File Output
    print("\n[3/3] Checking Structured JSON Logs...")
    log_file = "logs/agentlens.log"
    if not os.path.exists(log_file):
        print(f"  [FAIL] Log file {log_file} was not created.")
    else:
        print(f"  [PASS] Log file {log_file} exists.")

        # Read the last line to confirm it's JSON
        with open(log_file, encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_log = lines[-1].strip()
                try:
                    log_data = json.loads(last_log)
                    print("  [PASS] Log entries are valid JSON.")
                    if log_data.get("logger") == "arbiter":
                        print(f"  [PASS] Successfully found Arbiter log: {log_data.get('message')}")
                        print(f"         Fields recorded: {list(log_data.keys())}")
                except json.JSONDecodeError:
                    print("  [FAIL] Log entries are NOT valid JSON.")
            else:
                print("  [FAIL] Log file is empty.")


if __name__ == "__main__":
    verify_day16()
