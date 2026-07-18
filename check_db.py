"""Quick script to verify Day 8 — checks what's stored in the SQLite database."""

from storage.db import DatabaseManager

db = DatabaseManager()
counts = db.table_counts()

print("=== TABLE COUNTS ===")
for table, count in counts.items():
    print(f"  {table:10} : {count} rows")

runs = db.list_runs()
print(f"\n=== RUNS ({len(runs)} total) ===")
for r in runs:
    print(f"  {r['run_id']} | {r['status']} | {r['total_latency_ms']:.0f}ms")

if runs:
    latest = runs[0]["run_id"]
    steps = db.get_steps_for_run(latest)
    print(f"\n=== STEPS for {latest} ===")
    for s in steps:
        print(f"  Step {s['step']} [{s['agent']}] | {s['status']} | {s['latency_ms']:.0f}ms")
        print(f"         diff: {s['diff_summary']}")

    metrics = db.get_metrics_for_run(latest)
    print(f"\n=== METRICS for {latest} ({len(metrics)} rows) ===")
    for m in metrics:
        print(f"  Step {m['step']} [{m['agent']}] {m['metric_name']} = {m['metric_value']:.1f} {m['metric_unit']}")
else:
    print("\n  No runs found. Run the pipeline first:")
    print("  python app/main.py --topic \"Rise of AI in India\"")
