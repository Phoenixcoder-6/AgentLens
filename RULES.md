# AgentLens Rule Catalog

This document tracks all detection rules and validators implemented in the AgentLens pipeline.

## P1: Ground Truth

| Rule ID | Category | Blames | Trigger Condition | Confidence Formula |
|---------|----------|--------|-------------------|--------------------|
| `gt_mismatch_v1` | `REASONING` | Final step agent | `expected_output` is present AND final output similarity < threshold. | `1.0 - similarity_ratio` |
| `tool_failure_v1` | `EXECUTION` | Step agent | Tool call returned an error or exception string. | `1.0` |
| `missing_tool_output_v1` | `EXECUTION` | Step agent | Tool call was made but no output was recorded. | `1.0` |
| `hallucination_v1` | `REASONING` | Writer | Writer entity count > researcher entity count. | `1.0` |
| `researcher_quality_v1` | `REASONING` | Researcher | Researcher source count < minimum threshold. | `1.0` |
| `skipped_step_v1` | `WORKFLOW` | Missing agent | An expected agent is absent from the execution trace. | `1.0` |
| `wrong_order_v1` | `WORKFLOW` | Agent out of order | Agent steps appear in an unexpected sequence. | `1.0` |
| `verifier_passthrough_v1` | `VERIFICATION` | Verifier | Verifier passed through hallucinated entities unchallenged. | `1.0` |
