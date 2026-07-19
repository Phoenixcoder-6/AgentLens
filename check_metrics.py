"""Quick script to test Day 11 — shows metrics report for your latest pipeline run."""

from storage.db import DatabaseManager
from analyzers.metrics_analyzer import MetricsAnalyzer

db = DatabaseManager()
db.initialize()

analyzer = MetricsAnalyzer(db)
report = analyzer.analyze_latest_run()

if not report:
    print("No runs found. Run: python app/main.py --topic 'Rise of AI in India'")
    exit()

print(f"=== METRICS REPORT: {report.run_id} ===\n")
print(f"  Workflow          : {report.workflow}")
print(f"  Total latency     : {report.total_latency_ms:.0f}ms")
print(f"  Total exec time   : {report.total_execution_time_ms:.0f}ms")
print(f"  Total tokens      : {report.total_tokens}")
print(f"  Step count        : {report.step_count}")
print(f"  Slowest step      : {report.slowest_step}")
print(f"  Fastest step      : {report.fastest_step}")
print(f"  Has anomalies     : {report.has_anomalies}")
if report.anomalous_steps:
    print(f"  Anomalous steps   : {report.anomalous_steps}")

print(f"\n=== PER-STEP BREAKDOWN ===\n")
for step in report.steps:
    flag = " ⚠️ ANOMALY" if step.is_anomalous else ""
    print(f"  Step {step.step} [{step.agent.upper()}]{flag}")
    print(f"    latency_ms       : {step.latency_ms:.0f}ms  (threshold={step.latency_threshold_ms:.0f}ms)")
    print(f"    execution_time   : {step.execution_time_ms:.0f}ms")
    print(f"    tokens_total     : {step.tokens_total}")
    print(f"    tokens_prompt    : {step.tokens_prompt}")
    print(f"    tokens_completion: {step.tokens_completion}")
    if step.anomaly_reasons:
        for reason in step.anomaly_reasons:
            print(f"    ⚠️  {reason}")
    print()
