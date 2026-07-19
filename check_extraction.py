"""Quick script to test Day 9 — runs evidence extraction on your latest real pipeline trace."""

import json, glob, os
from normalizer.normalizer import Normalizer
from schema.models import RunTrace
from analyzers.evidence_extraction.extractor import EvidenceExtractor

# Load largest trace (real pipeline run)
files = glob.glob("data/traces/*.json")
if not files:
    print("No traces found. Run: python app/main.py --topic 'Rise of AI in India'")
    exit()

latest = max(files, key=os.path.getsize)
print(f"Reading: {latest}\n")

run = RunTrace(**json.load(open(latest)))
norm = Normalizer().normalize_run(run)

extractor = EvidenceExtractor()
print("=== EVIDENCE EXTRACTION RESULTS ===\n")
for step in norm.steps:
    ev = extractor.extract(step.raw_output, agent=step.agent)
    print(f"  Step {step.step} [{step.agent.upper()}]")
    print(f"    source_count     : {ev.source_count}")
    print(f"    entity_count     : {ev.entity_count}")
    print(f"    tool_calls       : {ev.tool_calls}")
    print(f"    schema_version   : {ev.schema_version}")
    print(f"    extraction_failed: {ev.extraction_failed}")
    if ev.extraction_failed:
        print(f"    error            : {ev.error_message}")
    print()
