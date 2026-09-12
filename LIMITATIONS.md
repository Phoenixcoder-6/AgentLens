# AgentLens System Boundaries & Long-Tail Failure Evaluation

This document outlines the architectural boundaries of **AgentLens**, detailing what deterministic rules cover, what statistical anomaly detection handles, and how the system addresses the "long-tail" gap in multi-agent LLM observability.

---

## 1. Observability Architecture Overview

AgentLens employs a **dual-layer detection architecture** to evaluate multi-agent pipeline executions:

```text
                             [ Agent Run Trace ]
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
┌──────────────────────────────┐                      ┌──────────────────────────────┐
│  Deterministic Rule Engine   │                      │ Statistical Anomaly Detector │
│   (Information Loss, Rules)   │                      │  (Per-Agent Latency/Tokens)  │
└──────────────┬───────────────┘                      └──────────────┬───────────────┘
               │                                                     │
               ▼                                                     ▼
     [ Known Failure Modes ]                                [ Long-Tail Outliers ]
     (P1 Contradiction, P2 Loss)                            (P4 Latency/Token Spikes)
               │                                                     │
               └──────────────────────────┬──────────────────────────┘
                                          ▼
                              ┌──────────────────────┐
                              │    Arbiter Engine    │
                              │  (Priority Merge P1-5)│
                              └──────────────────────┘
```

---

## 2. Deterministic Rule Engine Scope & Boundaries

The **Deterministic Rule Engine** evaluates explicit structural and semantic handoff invariants between agents:

### Supported Known Failure Patterns:
* **Information Loss (P2)**: Detects when downstream agents drop sources, citations, or factual claims provided by upstream agents.
* **Ungrounded Information Addition (P2)**: Flags when an agent introduces ungrounded entities or claims not present in research inputs.
* **Factual Contradictions (P1)**: Identifies conflicting numerical values, dates, or statements between pipeline steps.
* **Workflow Step Skips (P3)**: Detects out-of-order agent execution or missing required handoff nodes.

### System Boundaries of Rules:
* **Rule Limitations**: Rules can only detect failure modes for which explicit code logic has been written.
* **The Long-Tail Risk**: An agent can produce text that is 100% syntactically valid and faithful to research inputs, yet still experience a critical failure (e.g., an internal LLM reasoning loop, token explosion, or extreme latency delay).

---

## 3. Statistical Anomaly Detection (Closing the Long-Tail Gap)

To catch unseen, novel, or un-ruled failure modes, AgentLens incorporates the **Statistical Detector (`StatisticalDetector`)**:

### How It Works:
1. **Per-Agent Baselines**: Automatically tracks rolling mean ($\mu$) and standard deviation ($\sigma$) for latency and token consumption per agent identity.
2. **Outlier Detection**: Calculates Z-scores ($Z = \frac{x - \mu}{\sigma}$) on every step. Any step exceeding a configurable multiplier (default: $> 2.5\sigma$) is flagged.
3. **P4 Evidence Generation**: Emits `EvidenceSource.STATISTICAL_ANOMALY` records with dynamic confidence scaling ($0.50$ at threshold up to $0.99$ cap).

---

## 4. Empirical Long-Tail Proof (Day 28 Evaluation)

In **Day 28 (`scripts/run_long_tail_test.py`)**, AgentLens was subjected to a synthetic long-tail failure mode:

### Test Scenario:
* **Agent Behavior**: The `writer` agent output text that was 100% valid, cited all 5 research sources, and introduced 0 factual contradictions.
* **Hidden Anomaly**: An internal LLM reasoning loop caused:
  - **Latency**: `22,500 ms` (5.5x baseline mean of `2,200 ms`)
  - **Tokens**: `9,500 tokens` (6.0x baseline mean of `1,200 tokens`)

### Experimental Results:

| Detection Layer | Result | Description |
| :--- | :--- | :--- |
| **Deterministic Rules** | **0 Rules Fired** | Payload passed all schema, citation, and factual checks cleanly. |
| **Statistical Detector** | ⚡ **FIRED** | Flagged `writer` step for 5.5x latency and 6.0x token outlier ($Z > 4.2\sigma$). |
| **Arbiter Final Verdict** | 🔴 **P4 ANOMALY** | Attributed primary cause to `STATISTICAL_ANOMALY` and primary agent to `writer`. |

### Conclusion:
This experiment proves that AgentLens closes the long-tail observability gap. Even when an agent failure matches no pre-existing hardcoded rule, the statistical detection layer reliably captures the anomaly and attributes it to the exact malfunctioning component.
