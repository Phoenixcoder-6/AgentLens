# AgentLens Rule Catalog

This document tracks all detection rules and validators implemented in the AgentLens pipeline.

## P1: Ground Truth

| Rule ID | Category | Blames | Trigger Condition | Confidence Formula |
|---------|----------|--------|-------------------|--------------------|
| `gt_mismatch_v1` | `REASONING` | Final step agent | `expected_output` is present AND final output similarity < threshold. | `1.0 - similarity_ratio` |
