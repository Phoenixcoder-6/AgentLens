# AgentLens_demo

[![AgentLens CI](https://github.com/Phoenixcoder-6/AgentLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Phoenixcoder-6/AgentLens/actions/workflows/ci.yml)

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

# 7. Apply database migrations (indexes and future schema changes)
alembic upgrade head

# 8. Run the dashboard
streamlit run dashboard/app.py
```

> **Windows note:** Using conda (not pip venv) is required on Windows to ensure PyTorch's C++ runtime DLLs are correctly installed alongside the package.

> **Database migrations:** AgentLens uses [Alembic](https://alembic.sqlalchemy.org/) for schema migrations.
> After pulling new code, always run `alembic upgrade head` to apply pending migrations.
> Migration files live in `alembic/versions/` and are version-controlled alongside the source code.

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

## Replay CLI (`replay.py`)

> **Note:** `replay.py` is a deliberate addition beyond the original MVP, included because it directly supports cross-run diff validation and CI/CD pipeline gating.

Re-run or deterministically evaluate a captured workflow run by `run_id`:

```bash
# Re-run or evaluate a run by ID
python replay.py run_lbl_pass_01

# Dry-run mode (evaluate deterministically without calling LLMs)
python replay.py run_lbl_pass_01 --dry-run

# Machine-readable JSON output for CI scripting
python replay.py run_lbl_execution_01 --dry-run --json

# Replay with a different topic to test attribution stability and diff validation
python replay.py run_lbl_reasoning_01 --dry-run --override-topic "Quantum error correction"
```

### Exit Codes for CI Gating

| Exit Code | Verdict | Meaning |
|---|---|---|
| `0` | `PASS` | `P5` — No failures or anomalies detected |
| `1` | `WARNING` | `P3` / `P4` — Workflow violation or statistical outlier |
| `2` | `FAIL` | `P1` / `P2` — Ground-truth mismatch or critical rule failure |
| `3` | `ERROR` | Run ID not found, malformed trace, or runtime error |

```bash
python replay.py <run_id> --dry-run --json
if [ $? -ge 2 ]; then exit 1; fi
```

## CI/CD & Branch Protection

Every push to `main` / `dev` and every pull request targeting `main` runs the GitHub Actions workflow in `.github/workflows/ci.yml`:

1. **Lint & Type Check (`lint`)**: `ruff check .`, `ruff format --check .`, and `mypy . --ignore-missing-imports`.
2. **Unit & Error Injection Tests (`test`)**: `pytest tests/ --cov=. --cov-fail-under=70`.
3. **Validation Accuracy Gate (`validation-gate`)**: Runs the 20-trace frozen labeled dataset (`scripts/run_day34_validation.py` + `scripts/evaluate_day35_accuracy.py`) and fails the build if exact root-cause attribution accuracy drops below **75% (15/20)**.

### Branch Protection Setup (`main`)
Pull requests targeting `main` require all three CI status checks to pass before merging:
- Enable **Require a pull request before merging** in GitHub `Settings -> Branches -> Branch protection rules (main)`.
- Enable **Require status checks to pass before merging** and select:
  - `Lint & Type Check`
  - `Unit & Error Injection Tests (Coverage >= 70%)`
  - `Validation Accuracy Gate (>= 75%)`

## Data Privacy, PII Scrubbing & Trace Retention

> **Warning:** Traces contain full LLM I/O. Enable `pii_scrubbing` before using on user data.

- **Fail-Safe Capture (`capture.fail_safe: true`)**: Exceptions inside `CaptureSession`, `HandoffCapture`, `@trace_step`, or `StorageWriter` are caught and logged as warnings; they never crash or alter the underlying agent workflow execution.
- **Config-Driven PII Scrubbing (`capture/pii_scrubber.py`)**: Set `capture.pii_scrubbing.enabled: true` in `config/config.yaml` to redact emails (`[REDACTED_EMAIL]`), phone numbers (`[REDACTED_PHONE]`), SSNs (`[REDACTED_SSN]`), API keys (`[REDACTED_API_KEY]`), and credit cards (`[REDACTED_CREDIT_CARD]`) via configurable regex patterns, plus optional `spaCy` (`en_core_web_sm`) NER entity redaction when installed.
- **Trace Retention Policy (`capture.retention_days: 90`)**: Run the cleanup utility to delete traces older than `retention_days` from `data/traces/*.json` and `data/agentlens.db`:

```bash
# Preview traces older than 90 days (dry-run)
python scripts/cleanup_old_traces.py --dry-run

# Delete traces older than 30 days
python scripts/cleanup_old_traces.py --days 30
```

## Status

🚧 **Active development — v1.0 build in progress (45-day plan)**

See `docs/` for the full Architecture & Requirements Document.

## Limitations

See `LIMITATIONS.md` (generated after validation in Week 7).


