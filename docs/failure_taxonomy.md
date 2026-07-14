# AgentLens Failure Taxonomy
**Version:** 1.0 (Locked — matches `schema/models.py::FailureCategory`)
**Status:** Approved for implementation

Every verdict produced by AgentLens maps to exactly one of the four categories below.
This taxonomy is the shared language between the Rule Engine, Workflow Validator,
Consistency Validator, Arbiter, and Dashboard.

---

## Why Four Categories?

Multi-agent systems fail through mechanisms that do not exist in single-agent systems.
A flat "error" label is not enough — an engineer debugging a failure needs to know
**which kind of failure** it is, because each category has different root causes,
different detection strategies, and different fixes.

---

## Category 1 — EXECUTION

> The agent could not complete its task due to a technical or infrastructure failure.

**Definition:** Something went wrong at the execution level — the agent attempted to act
but was blocked by a system-level problem. The agent's reasoning may have been correct,
but the failure occurred before or during the action.

### Example Failure Modes

| Mode | Description |
|---|---|
| Tool timeout | An external tool call exceeded its time limit and returned no result |
| API error | The LLM API returned a non-200 response or raised an exception |
| Empty output | The agent returned an empty string or null where content was expected |
| Crash / exception | An unhandled exception terminated the agent step |
| Missing dependency | A required input (file, key, external resource) was not available |
| Retry exhaustion | The agent retried a failed operation and exhausted all attempts |

### Detection Strategy
- `status == FAILURE or ERROR` on `AgentStep`
- `output == ""` or `output is None` when `status == SUCCESS` (silent failure)
- `latency_ms == 0` when output is non-empty (suspiciously fast — possible cache/skip)
- Tool call present in `tool_calls` with no corresponding output

### Rule IDs Reserved
`R-EX-001` through `R-EX-099`

---

## Category 2 — REASONING

> The agent completed its task but produced an incorrect, incomplete, or misleading output.

**Definition:** The agent ran successfully — no technical failure — but the content of
its output is wrong. This is the hardest category to detect without ground truth,
because the system reported SUCCESS while the actual answer was bad.

### Example Failure Modes

| Mode | Description |
|---|---|
| Hallucination | The agent stated facts that are not in its inputs or are demonstrably false |
| Wrong facts | Factual claims in the output contradict the source material provided |
| Missed sources | The agent was given 10 sources but cited only 2, dropping key context |
| Fabricated entities | Named entities (people, companies, dates) that do not appear in inputs |
| Incomplete answer | The output addresses only part of the task — key sub-questions ignored |
| Contradiction | The output contradicts an earlier step's output in the same run |

### Detection Strategy
- Evidence extraction: `source_count` in output < `source_count` in input
- Evidence extraction: entities in output not present in input sources
- Ground truth comparison (P1): output semantic similarity to expected_output < threshold
- Consistency validator: output of step N contradicts output of step N-1

### Rule IDs Reserved
`R-RS-001` through `R-RS-099`

---

## Category 3 — WORKFLOW

> The pipeline structure broke down — agents did not hand off information correctly.

**Definition:** The failure is not in what an individual agent did, but in how information
moved between agents. An agent may have produced correct output, but the next agent
received a degraded, incomplete, or mutated version of it.

This category only exists because AgentLens captures `HandoffState` —
plain input/output logging cannot detect these failures.

### Example Failure Modes

| Mode | Description |
|---|---|
| Information loss | `input_state` had 10 keys, `output_state` has 3 — 7 keys silently dropped |
| State mutation | A key was modified in transit without the receiving agent being aware |
| Missing required key | The next agent expected a key that was not in `output_state` |
| Skipped node | A node in the LangGraph graph was never executed (no AgentStep recorded) |
| Wrong agent order | Agents executed in a different sequence than the defined graph |
| Orphaned step | A step has no `parent_step` or `child_step` but should have both |

### Detection Strategy
- `HandoffState.input_state` key count vs `HandoffState.output_state` key count
- Keys present in `input_state` but absent in `output_state` (dropped keys)
- Values in `output_state` that differ from `input_state` without a filter step
- Expected agent sequence from LangGraph graph definition vs captured step order
- `parent_step` / `child_step` chain integrity check

### Rule IDs Reserved
`R-WF-001` through `R-WF-099`

---

## Category 4 — VERIFICATION

> The verification step failed to catch a bad answer, or skipped verification entirely.

**Definition:** A Verifier agent exists in the pipeline specifically to catch mistakes.
This category fires when the Verifier approved an incorrect answer, or when the
Verifier's behaviour suggests it is not performing a real check (e.g., always approving).

This is a meta-failure — the safety net failed. It means errors from other categories
may have passed through undetected.

### Example Failure Modes

| Mode | Description |
|---|---|
| Approved bad answer | Verifier returned "APPROVED" for an output that contains known errors |
| Always-approve pattern | Verifier approved every run across N consecutive executions (threshold-based) |
| Skipped verification | No Verifier step was captured in a run that should have one |
| Trivial check | Verifier output is shorter than a minimum threshold — likely a rubber-stamp |
| Circular verification | Verifier is checking its own output rather than the Writer's output |

### Detection Strategy
- Verifier `output` contains "APPROVED" or "PASS" + ground truth shows output was wrong (P1)
- Verifier `output` never contains "REJECTED" or "FAIL" across sliding window of N runs
- No AgentStep with `agent == "verifier"` (or configured verifier name) in the run
- Verifier `output` token count < configured minimum (too short to be a real check)
- Verifier `input` matches Verifier's own previous `output` (circular reference)

### Rule IDs Reserved
`R-VF-001` through `R-VF-099`

---

## Category 5 — UNKNOWN

> No evidence matched any of the four categories above.

**Definition:** The Arbiter exhausted all evidence sources (P1–P4) and found nothing
that maps to a known failure pattern. The run may have failed for a novel reason,
or it may have succeeded (in which case UNKNOWN with no failure signals is correct).

This is the P5 fallback — it is always returned when nothing else fires.
It is never a crash state.

### Dashboard Display
- Shown with a grey badge: "No attribution found"
- Heuristic language: "AgentLens could not identify a specific cause"
- Suggests: "Review the raw trace or add more labeled examples"

---

## Rule ID Naming Convention

All rule IDs follow the format: `R-{CATEGORY}-{NUMBER}`

| Category | Prefix | Range |
|---|---|---|
| Execution | `R-EX` | 001–099 |
| Reasoning | `R-RS` | 001–099 |
| Workflow | `R-WF` | 001–099 |
| Verification | `R-VF` | 001–099 |

Example: `R-WF-001` = Workflow rule #1 = "Information Loss at Handoff"

Rule IDs are used by the Arbiter's tie-break logic:
> *Same-priority ties resolved by rule ID ascending (alphabetical sort)*

This means `R-EX-001` beats `R-EX-002` if both fire at the same priority.

---

## Taxonomy ↔ Arbiter Priority Mapping

| Category | Detected by | Arbiter Priority |
|---|---|---|
| Any (with ground truth) | Ground truth comparison | P1 |
| EXECUTION / REASONING / WORKFLOW / VERIFICATION | Rule Engine | P2 |
| WORKFLOW | Workflow Validator | P3 |
| EXECUTION (anomaly) | Metrics Analyzer | P4 |
| UNKNOWN | Fallback | P5 |

---

*This document is locked at v1.0. Any additions require a schema version bump and
corresponding update to `schema/models.py::FailureCategory`.*
