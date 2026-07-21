"""
check_arbiter.py — Day 12 demo
Runs the full Day 9→10→12 pipeline on the latest stored trace.
Shows: evidence extracted → information loss verdict → Arbiter final verdict.

Uses the same extraction path as check_information_loss.py:
    trace_json → RunTrace → Normalizer → NormalizedStep.raw_output → EvidenceExtractor
This gives the LLM-extracted source/entity counts from the actual text content,
not the regex counts stored in the pipeline node output dict.
"""

import json

from storage.db import DatabaseManager
from normalizer.normalizer import Normalizer
from schema.models import RunTrace
from analyzers.evidence_extraction.extractor import EvidenceExtractor
from analyzers.detection.information_loss import InformationLossRule
from analyzers.arbiter import Arbiter, evidence_from_information_loss

db = DatabaseManager()
db.initialize()

# ── Get latest run ────────────────────────────────────────────────────────────
runs = db.list_runs(limit=1)
if not runs:
    print("No runs found. Run: python app/main.py --topic 'Rise of AI in India'")
    exit()

run_id  = runs[0]["run_id"]
run_row = db.get_run(run_id)
print(f"=== ARBITER VERDICT: {run_id} ===\n")

trace_json_str = run_row.get("trace_json", "")
if not trace_json_str:
    print("ERROR: trace_json not found in DB for this run.")
    print("Re-run the pipeline: python app/main.py --topic 'Rise of AI in India'")
    exit()

# ── Reconstruct RunTrace + normalize (same as check_information_loss.py) ──────
run  = RunTrace(**json.loads(trace_json_str))
norm = Normalizer().normalize_run(run)

# ── Step 1: Extract evidence from full text content (LLM-based) ───────────────
print("=== STEP 1: EVIDENCE EXTRACTION ===")
extractor = EvidenceExtractor()
extracted = {}
for step in norm.steps:
    ev = extractor.extract(step.raw_output, agent=step.agent)
    extracted[step.agent] = ev
    print(f"  [{step.agent.upper()}] source={ev.source_count}  entity={ev.entity_count}  tool_calls={ev.tool_calls}")

# ── Step 2: Information Loss rule (Day 10) ────────────────────────────────────
print("\n=== STEP 2: INFORMATION LOSS ===")
rule = InformationLossRule()
researcher_ev = extracted.get("researcher")
writer_ev     = extracted.get("writer")

evidence_record = None
if researcher_ev and writer_ev:
    result = rule.evaluate(
        researcher_evidence=researcher_ev,
        writer_evidence=writer_ev,
        run_id=run_id,
    )
    print(f"  Verdict    : {result.verdict}")
    print(f"  Confidence : {result.confidence:.0%}")
    print(f"  source     : {result.source_diff.researcher_value} → {result.source_diff.writer_value}  ({result.source_diff.signal})")
    print(f"  entity     : {result.entity_diff.researcher_value} → {result.entity_diff.writer_value}  ({result.entity_diff.signal})")
    evidence_record = evidence_from_information_loss(result)
else:
    print("  [SKIP] Could not find researcher/writer steps")

# ── Step 3: Arbiter (Day 12) ──────────────────────────────────────────────────
all_evidence = [e for e in [evidence_record] if e is not None]
bundle = Arbiter().run(run_id=run_id, evidence=all_evidence)

print(f"\n=== STEP 3: ARBITER FINAL VERDICT ===\n")
print(f"  Primary cause  : {bundle.primary_cause.value}")
print(f"  Priority       : {bundle.priority_level.value}")
print(f"  Grounded       : {bundle.grounded}")
print(f"  Primary agent  : {bundle.primary_agent or 'N/A'}")
print(f"  Evidence count : {len(bundle.evidence)}")
print(f"  Summary        : {bundle.summary}")
