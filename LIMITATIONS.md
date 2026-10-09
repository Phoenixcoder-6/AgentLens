# AgentLens v1.0.0 Limitations & Boundary Conditions

Transparent, honest reporting of system boundaries is essential for any production-grade AI observability platform. This document outlines the verified accuracy metrics, known failure modes, scope boundaries, and edge cases of AgentLens v1.0.0.

---

## 1. Verified Accuracy Metrics

AgentLens is benchmarked against a frozen 20-trace dataset (`sample_data/labels.json`):

- **Binary PASS / FAIL Detection:** **100.0% (20/20)**  
  Every clean pipeline run is recognized as PASS; every broken run is recognized as FAIL. False positive rate on clean runs is 0.0%.
- **Exact Category & Agent Attribution:** **80.0% (16/20)**  
  AgentLens achieves exact root-cause attribution matching human consensus across 16 of the 20 benchmark runs.

### The 4 Disagreement Traces: Dual-Fault Ambiguity
The 4 disagreements in the benchmark occur exclusively on **`verification_failure`** test traces (`run_lbl_verification_01` through `04`).

#### Why the Disagreement Occurs:
In these 4 traces, a human annotator labeled the root cause as `verification_failure` because the Verifier rubber-stamped an unverified report.  
However, to trigger that test scenario, the Writer *first* hallucinated 12+ entities.  

As a result, two $P_2$ rules fire simultaneously:
1. `hallucination_v1` ($P_2$, agent: `writer`)
2. `verifier_passthrough_v1` ($P_2$, agent: `verifier`)

Because both are $P_2$ rules with equal confidence ($1.0$), the Arbiter breaks the tie by ascending `rule_id` (`hallucination_v1` < `verifier_passthrough_v1`). The Arbiter attributes root cause to the upstream Writer fabrication rather than the downstream Verifier oversight.

Both attributions are technically valid interpretations:
- *Upstream perspective:* The failure originated with the Writer. If the Writer hadn't hallucinated, the Verifier wouldn't have passed bad output.
- *Downstream perspective:* The Verifier was explicitly tasked with catching bad output and failed its job.

AgentLens deterministically favors the **upstream root cause**.

---

## 2. Scope Boundaries (What AgentLens Does NOT Catch in v1.0)

To avoid false expectations, engineers should be aware of what is outside the v1.0 design scope:

1. **Non-Numeric Semantic Replacement (Equal Entity Count Hallucinations):**  
   If an agent replaces 5 real entities with 5 completely fabricated entities without changing the total count, the count difference is zero. Without an explicit external ground-truth contract ($P_1$), count-based rules will not flag this as an entity gain.
2. **Subtle Tone or Stylistic Drift:**  
   AgentLens evaluates factual preservation, schema contracts, execution status, and latency. It does not grade stylistic tone, politeness, or prose elegance unless explicitly captured by custom rules.
3. **Multi-Agent Cyclic Loops (Graph Iterations):**  
   AgentLens v1.0 is optimized for directed acyclic workflows (DAGs) and linear chains. If two agents loop back and forth 5 times in a feedback cycle, steps are recorded sequentially by index (`step_idx`), but iteration clustering is not natively aggregated in v1.0.
4. **Adversarial Jailbreaks within Safe Schemas:**  
   If an agent is jailbroken but still returns valid JSON conforming to the schema without tool errors or entity inflation, AgentLens will treat the execution as technically valid unless a semantic or ground-truth check is attached.

---

## 3. Concurrency & Performance Thresholds

- **Storage Engine:** SQLite default file database. Recommended for up to ~100 concurrent requests/sec. For high-volume enterprise ingestion, an external PostgreSQL / ClickHouse adapter is recommended.
- **Trace Size Limits:** Recommended trace size is $<10\text{ MB}$ per run. Traces containing massive binary payloads (e.g., raw images in tool outputs) should store references or S3 URLs rather than embedding base64 payloads into SQLite JSON fields.
- **Statistical Detector Warmup:** `StatisticalDetector` requires at least `min_runs_for_baseline = 5` historical runs per agent before flagging latency or token outliers.

---

## 4. Summary of Guarantees

| Guarantee | AgentLens v1.0 Status |
|---|---|
| Deterministic Tie-Breaking (Same Input $\to$ Same Verdict) | **100% Guaranteed** |
| Zero Pipeline Crashes from Telemetry (Fail-Safe Decorator) | **100% Guaranteed** |
| Grounded vs. Heuristic Separation | **100% Guaranteed** |
| Zero False Positives on Clean Reference Runs | **100% Verified** |
| Upstream Root-Cause Prioritization | **100% Guaranteed** |
