# AgentLens Rule Catalog (v1.0.0)

This document is the definitive specification of all deterministic rules implemented in AgentLens v1.0.0.

Every rule has a unique `rule_id`, semantic `version`, assigned `FailureCategory`, `RuleSeverity`, assigned `PriorityLevel`, and an immutable root-cause attribution contract.

---

## 1. Ground Truth Rules ($P_1$)

Grounded rules operate against verified external baselines or reference contracts (`expected_output`).

### `gt_mismatch_v1`
- **Category:** `FailureCategory.REASONING`
- **Severity:** `RuleSeverity.CRITICAL`
- **Priority:** `PriorityLevel.P1`
- **Source:** `EvidenceSource.GROUND_TRUTH`
- **Trigger Condition:**  
  The final output of the pipeline is compared against `trace.expected_output` using sequence similarity. If similarity falls below the configured threshold (default: `0.85`), this rule fires.
- **Blamed Agent:** Final synthesis agent (`writer`).
- **Grounded Status:** `grounded = True`
- **Example Trace:** `demo_grounded_p1_01` (Eiffel tower constructed in 1925 instead of 1889).

---

## 2. Execution Failure Rules ($P_2$)

Execution rules detect tool crashes, unhandled exceptions, and missing tool outputs.

### `tool_failure_v1`
- **Category:** `FailureCategory.EXECUTION`
- **Severity:** `RuleSeverity.HIGH`
- **Priority:** `PriorityLevel.P2`
- **Source:** `EvidenceSource.RULE_ENGINE`
- **Trigger Condition:**  
  A step's tool call dictionary contains a non-empty `error` key, or the tool output string contains standard error signatures (`"Error:"`, `"Exception:"`, `"Traceback"`, `"HTTP 500"`).
- **Blamed Agent:** The agent executing the tool call (e.g., `researcher`).
- **Grounded Status:** `grounded = False`

### `missing_tool_output_v1`
- **Category:** `FailureCategory.EXECUTION`
- **Severity:** `RuleSeverity.HIGH`
- **Priority:** `PriorityLevel.P2`
- **Source:** `EvidenceSource.RULE_ENGINE`
- **Trigger Condition:**  
  An agent invokes a tool, but the output payload is null, empty string `""`, or completely missing without an explicit error recorded.
- **Blamed Agent:** The agent executing the tool call.
- **Grounded Status:** `grounded = False`

---

## 3. Reasoning & Hallucination Rules ($P_2$)

Reasoning rules detect ungrounded facts, entity fabrication, and poor research quality.

### `hallucination_v1`
- **Category:** `FailureCategory.REASONING`
- **Severity:** `RuleSeverity.HIGH`
- **Priority:** `PriorityLevel.P2`
- **Source:** `EvidenceSource.RULE_ENGINE`
- **Trigger Condition:**  
  The downstream synthesizer agent introduces more named entities or cited sources than were provided by the upstream research agent ($\Delta_{\text{entities}} > \text{threshold}$, default: $>0$).
- **Blamed Agent:** Synthesizer agent (`writer`).
- **Grounded Status:** `grounded = False`
- **Example Trace:** `demo_heuristic_p2_hallucination`.

### `researcher_quality_v1`
- **Category:** `FailureCategory.REASONING`
- **Severity:** `RuleSeverity.MEDIUM`
- **Priority:** `PriorityLevel.P2`
- **Source:** `EvidenceSource.RULE_ENGINE`
- **Trigger Condition:**  
  The research agent returns fewer sources or citations than the configured minimum threshold (`researcher_min_sources`, default: $<1$).
- **Blamed Agent:** Research agent (`researcher`).
- **Grounded Status:** `grounded = False`

---

## 4. Verification Failure Rules ($P_2$)

Verification rules detect quality-checker negligence, rubber-stamping, and verifier bypasses.

### `verifier_passthrough_v1`
- **Category:** `FailureCategory.VERIFICATION`
- **Severity:** `RuleSeverity.HIGH`
- **Priority:** `PriorityLevel.P2`
- **Source:** `EvidenceSource.CONSISTENCY_VALIDATOR`
- **Trigger Condition:**  
  The verifier marks output as `approved = True` despite severe upstream entity inflation ($\ge 3$ new entities) or obvious tool errors.
- **Blamed Agent:** Quality checker (`verifier`).
- **Grounded Status:** `grounded = False`
- **Note on Co-occurrence:** If `hallucination_v1` (Writer) and `verifier_passthrough_v1` (Verifier) both fire, the Arbiter tie-breaks by ascending `rule_id` (`hallucination_v1` < `verifier_passthrough_v1`), correctly isolating upstream fabrication as the primary root cause.

---

## 5. Workflow & Handoff Rules ($P_3$)

Workflow rules inspect topology adherence, agent step sequences, and information drops.

### `skipped_step_v1`
- **Category:** `FailureCategory.WORKFLOW`
- **Severity:** `RuleSeverity.HIGH`
- **Priority:** `PriorityLevel.P3`
- **Source:** `EvidenceSource.WORKFLOW_VALIDATOR`
- **Trigger Condition:**  
  An agent specified in `pipeline.required_agents` (or topology nodes) does not execute a step in the trace.
- **Blamed Agent:** The omitted agent (e.g., `verifier`).
- **Grounded Status:** `grounded = False`
- **Example Trace:** `demo_heuristic_p3_workflow`.

### `information_loss_v1`
- **Category:** `FailureCategory.WORKFLOW`
- **Severity:** `RuleSeverity.MEDIUM` / `RuleSeverity.HIGH`
- **Priority:** `PriorityLevel.P3`
- **Source:** `EvidenceSource.WORKFLOW_VALIDATOR` (via `InformationLossRule`)
- **Trigger Condition:**  
  During agent handoff, the downstream agent drops critical sources or entities ($\text{delta} \le -1 \implies \text{MEDIUM}$, $\text{delta} \le -3 \implies \text{HIGH}$).
- **Blamed Agent:** Downstream synthesizer agent (`writer`).
- **Grounded Status:** `grounded = False`

---

## 6. Statistical Outlier Rules ($P_4$)

Statistical rules monitor system performance baselines across historical runs.

### `latency_outlier_v1` & `token_outlier_v1`
- **Category:** `FailureCategory.PERFORMANCE`
- **Severity:** `RuleSeverity.LOW` / `RuleSeverity.MEDIUM`
- **Priority:** `PriorityLevel.P4`
- **Source:** `EvidenceSource.STATISTICAL_ANOMALY`
- **Trigger Condition:**  
  An agent step's latency or token count exceeds $>2.5\sigma$ (standard deviations) from that specific agent's historical running mean (requires $\ge 5$ historical runs).
- **Blamed Agent:** The anomalous agent.
- **Grounded Status:** `grounded = True` (grounded in mathematical distribution).

---

## Summary Matrix

| Rule ID | Version | Category | Priority | Severity | Default Blame |
|---|---|---|---|---|---|
| `gt_mismatch_v1` | 1.0.0 | Reasoning | **P1** | Critical | `writer` |
| `tool_failure_v1` | 1.0.0 | Execution | **P2** | High | `step.agent` |
| `missing_tool_output_v1` | 1.0.0 | Execution | **P2** | High | `step.agent` |
| `hallucination_v1` | 1.0.0 | Reasoning | **P2** | High | `writer` |
| `researcher_quality_v1` | 1.0.0 | Reasoning | **P2** | Medium | `researcher` |
| `verifier_passthrough_v1` | 1.0.0 | Verification | **P2** | High | `verifier` |
| `skipped_step_v1` | 1.0.0 | Workflow | **P3** | High | `missing_agent` |
| `information_loss_v1` | 1.0.0 | Workflow | **P3** | High/Med | `writer` |
| `latency_outlier_v1` | 1.0.0 | Performance | **P4** | Low | `step.agent` |
| `token_outlier_v1` | 1.0.0 | Performance | **P4** | Low | `step.agent` |
