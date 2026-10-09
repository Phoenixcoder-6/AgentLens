# Contributing to AgentLens

Thank you for your interest in contributing to AgentLens! We welcome bug fixes, documentation improvements, new deterministic analyzers, and benchmark traces.

---

## 1. Local Development Setup

### Prerequisites
- Python 3.11 or 3.12 (Python 3.12 recommended)
- Git
- Conda (optional, but recommended for clean environment isolation)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Phoenixcoder-6/AgentLens.git
cd AgentLens

# 2. Create virtual environment or conda env
conda create -n agentlens python=3.12 -y
conda activate agentlens

# 3. Install editable package with dev dependencies
pip install -e ".[dev]"

# 4. Set up environment variables
cp .env.example .env
# Configure GROQ_API_KEY if testing live LLM explainer
```

---

## 2. Code Quality & Formatting Standards

AgentLens enforces strict typing and linting across all source directories:

```bash
# Format code using Ruff
ruff format .

# Check linting and import order
ruff check .

# Run static type checking with Mypy
mypy . --ignore-missing-imports --exclude '(_patch_|_diag_|check_|\.venv|alembic)'
```

All 3 commands must complete with **0 errors** before submitting a pull request.

---

## 3. Running the Test Suite

```bash
# Run unit & injection tests with 75% coverage enforcement
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=75 -v

# Run the automated 20-trace regression runner
python scripts/run_day43_regression.py

# Run all milestone verification scripts
python scripts/verify_day34.py
python scripts/verify_day35.py
python scripts/verify_day36.py
python scripts/verify_day43.py
python scripts/verify_day44.py
```

---

## 4. How to Add a New Detection Rule

1. Define the rule metadata in `analyzers/rule_catalog.py` (`RuleDefinition`).
2. Implement deterministic detection in the relevant analyzer (e.g., `RuleEngine` or `WorkflowValidator`).
3. Add a unit test in `tests/test_rule_engine.py` or a dedicated test file.
4. Execute `python scripts/run_day43_regression.py` to ensure that existing benchmark runs remain stable.

---

## 5. Pull Request Process

1. Create a feature branch: `git checkout -b feat/your-feature-name`.
2. Commit changes with clear, descriptive commit messages.
3. Ensure CI requirements pass locally (`ruff`, `mypy`, `pytest --cov-fail-under=75`).
4. Submit a Pull Request to `main`.
5. GitHub Actions CI will automatically test your PR against:
   - **`lint` job:** `ruff check`, `ruff format --check`, `mypy`.
   - **`test` job:** `pytest` with `--cov-fail-under=75`.
   - **`validation-gate` job:** Automated 20-trace regression run (must maintain $\ge 75\%$ accuracy and 0 per-run regressions).
