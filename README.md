# AgentLens

**Multi-Agent Failure Attribution, Trace Diffing & Explainability Platform**

AgentLens answers a single question precisely: *why did a multi-agent workflow fail, and which agent was responsible?*

> The model never decides what happened — it only explains what deterministic analysis has already established.

## Architecture

```
agentlens/
├── app/                         # Application entry point
├── capture/                     # @trace_step decorator — captures input/output/handoff
├── normalizer/                  # Converts raw events → Canonical Trace Schema
├── schema/                      # Pydantic models (RunTrace, AgentStep, etc.)
├── storage/                     # SQLite + JSON blob storage
├── analyzers/
│   ├── evidence_extraction/     # Schema-constrained LLM fact extraction
│   ├── detection/
│   │   ├── rule_engine.py       # Deterministic rules for known failure patterns
│   │   ├── workflow_validator.py# Handoff/workflow violation detection (P3)
│   │   └── consistency_validator.py  # Verifier behavior checking
│   ├── diff_engine.py           # Graph-aligned cross-run comparison
│   ├── metrics_analyzer.py      # Latency, token, statistical anomaly detection (P4)
│   └── arbiter.py               # Priority-ranked evidence merger → final verdict
├── dashboard/                   # Streamlit UI
├── replay.py                    # CLI to re-run a workflow with original inputs
├── tests/
├── sample_data/
├── config/
│   └── config.yaml              # All thresholds, models, paths — nothing hardcoded
└── docs/
```

## Quickstart

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd AgentLensCode

# 2. Create the conda environment (bundles MSVC runtime — fixes PyTorch DLL issues on Windows)
conda create -n agentlens python=3.12 -y
conda activate agentlens

# 3. Install PyTorch via conda FIRST (properly handles C++ runtime dependencies)
conda install pytorch cpuonly -c pytorch -y

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Set your API key
copy .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

# 6. Run the verification smoke test
python verify_groq.py

# 7. Run the dashboard
streamlit run dashboard/app.py
```

> **Windows note:** Using conda (not pip venv) is required on Windows to ensure PyTorch's C++ runtime DLLs are correctly installed alongside the package.

## Design Principles

- **Evidence before verdicts** — every finding is backed by a verifiable trace record
- **Deterministic core** — the Arbiter's P1–P5 priority system produces the same output for the same input, always
- **LLM confined to explanation** — the model explains what analysis established; it never assigns blame
- **Configuration over code** — all thresholds, model choices, and paths live in `config/config.yaml`

## Arbiter Priority System

| Priority | Source | Example |
|---|---|---|
| P1 | Ground truth mismatch | Output ≠ `expected_output` |
| P2 | Rule match | Information loss detected by rule engine |
| P3 | Workflow violation | Handoff dropped a key |
| P4 | Statistical anomaly | Latency spike beyond threshold |
| P5 | Unknown | No evidence matched |

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| Schema/Validation | Pydantic v2 |
| Storage | SQLite + JSON blobs |
| Evidence/Explanation | Groq API |
| Diff Engine | sentence-transformers (local) |
| Dashboard | Streamlit |

## Status

🚧 **Active development — v1.0 build in progress (45-day plan)**

See `docs/` for the full Architecture & Requirements Document.

## Limitations

See `LIMITATIONS.md` (generated after validation in Week 7).
