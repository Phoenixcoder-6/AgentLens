"""Quick script to test Day 10 — runs the Information Loss rule on your latest trace."""

import json, glob, os
from normalizer.normalizer import Normalizer
from schema.models import RunTrace
from analyzers.evidence_extraction.extractor import EvidenceExtractor
from analyzers.detection.information_loss import InformationLossRule

# Load largest trace (real pipeline run)
files = glob.glob("data/traces/*.json")
latest = max(files, key=os.path.getsize)
print(f"Reading: {latest}\n")

run = RunTrace(**json.load(open(latest)))
norm = Normalizer().normalize_run(run)

# Step 1: Extract evidence from Researcher and Writer
extractor = EvidenceExtractor()
print("=== STEP 1: EVIDENCE EXTRACTION ===")
evidence = {}
for step in norm.steps:
    ev = extractor.extract(step.raw_output, agent=step.agent)
    evidence[step.agent] = ev
    print(f"  [{step.agent.upper()}] source_count={ev.source_count}  entity_count={ev.entity_count}  tool_calls={ev.tool_calls}")

# Step 2: Run Information Loss rule
print("\n=== STEP 2: INFORMATION LOSS RULE ===")
rule = InformationLossRule()
result = rule.evaluate(
    run_id=norm.run_id,
    researcher_evidence=evidence.get("researcher"),
    writer_evidence=evidence.get("writer"),
)

print(f"\n  Verdict    : {result.verdict}")
print(f"  Confidence : {result.confidence:.0%}")
print(f"  Has Loss   : {result.has_information_loss}")
print(f"  Has Gain   : {result.has_information_gain}")
print(f"\n  Source diff  : {result.source_diff.researcher_value} → {result.source_diff.writer_value}  ({result.source_diff.signal}, severity={result.source_diff.severity})")
print(f"  Entity diff  : {result.entity_diff.researcher_value} → {result.entity_diff.writer_value}  ({result.entity_diff.signal}, severity={result.entity_diff.severity})")
print(f"\n  Summary:\n{result.summary}")
