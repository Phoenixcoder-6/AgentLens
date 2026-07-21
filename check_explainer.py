"""
check_explainer.py — Day 13 demo
Full pipeline: Evidence → Info Loss → Arbiter → LLM Explainer
Demonstrates that the Explainer receives ONLY the AnalysisBundle, never raw trace data.
"""

import json

from storage.db import DatabaseManager
from normalizer.normalizer import Normalizer
from schema.models import RunTrace
from analyzers.evidence_extraction.extractor import EvidenceExtractor
from analyzers.detection.information_loss import InformationLossRule
from analyzers.arbiter import Arbiter, evidence_from_information_loss
from analyzers.explainer import LLMExplainer

db = DatabaseManager()
db.initialize()

# ── Load latest run ───────────────────────────────────────────────────────────
runs = db.list_runs(limit=1)
if not runs:
    print("No runs found. Run: python app/main.py --topic 'Rise of AI in India'")
    exit()

run_id  = runs[0]["run_id"]
run_row = db.get_run(run_id)
print(f"=== FULL ANALYSIS: {run_id} ===\n")

trace_json_str = run_row.get("trace_json", "")
run  = RunTrace(**json.loads(trace_json_str))
norm = Normalizer().normalize_run(run)

# ── Step 1: Evidence extraction (Day 9) ───────────────────────────────────────
extractor = EvidenceExtractor()
extracted = {}
for step in norm.steps:
    ev = extractor.extract(step.raw_output, agent=step.agent)
    extracted[step.agent] = ev

researcher_ev = extracted.get("researcher")
writer_ev     = extracted.get("writer")

# ── Step 2: Information Loss rule (Day 10) ────────────────────────────────────
result = InformationLossRule().evaluate(
    researcher_evidence=researcher_ev,
    writer_evidence=writer_ev,
    run_id=run_id,
)
evidence_record = evidence_from_information_loss(result)

# ── Step 3: Arbiter (Day 12) ──────────────────────────────────────────────────
all_evidence = [e for e in [evidence_record] if e is not None]
bundle = Arbiter().run(run_id=run_id, evidence=all_evidence)

print(f"  Arbiter verdict  : {bundle.primary_cause.value}  ({bundle.priority_level.value})")
print(f"  Primary agent    : {bundle.primary_agent or 'N/A'}")
print(f"  Evidence count   : {len(bundle.evidence)}")

# ── Step 4: LLM Explainer (Day 13) ───────────────────────────────────────────
print(f"\n  [Calling LLM Explainer — receives ONLY AnalysisBundle, not raw trace]\n")
bundle = LLMExplainer().explain(bundle)

print(f"=== LLM EXPLANATION ===\n")
print(f"  Summary:\n  {bundle.summary}\n")
print(f"  Suggested Fix:\n  {bundle.suggested_fix}")
