import json
import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, ".")

from storage.db import DatabaseManager  # noqa: E402


def main() -> None:
    with open("sample_data/labels.json") as f:
        d = json.load(f)
    print(f"labels.json : {d['total']} runs, frozen={d['frozen']}")
    print(f"Categories  : {d['category_counts']}")

    traces = os.listdir("sample_data/labeled_traces")
    print(f"Trace files : {len(traces)} files in sample_data/labeled_traces/")

    db = DatabaseManager("data/agentlens.db")
    with db.connection() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id LIKE 'run_lbl_%'").fetchone()
        print(f"DB rows     : {rows[0]} labeled runs in DB")

    print("\nDay 15 COMPLETE ✓")


if __name__ == "__main__":
    main()
