"""
dashboard/state.py — Data layer v2
All DB access, pipeline execution, and caching.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from storage.db import DatabaseManager
from normalizer.normalizer import Normalizer
from schema.models import RunTrace, AnalysisBundle, SCHEMA_VERSION
from analyzers.evidence_extraction.extractor import EvidenceExtractor, ExtractedEvidence
from analyzers.detection.information_loss import InformationLossResult, InformationLossRule
from analyzers.arbiter import Arbiter, evidence_from_information_loss
from analyzers.explainer import LLMExplainer


# ── Module-level analysis cache (process lifetime) ────────────────────────────
_analysis_cache: dict[str, "AnalysisState"] = {}

# Estimated cost per token (GPT-4o proxy for display)
COST_PER_TOKEN = 0.000005  # $0.005 per 1K tokens


@dataclass
class StepRow:
    step: int
    agent: str
    status: str
    latency_ms: float
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int


@dataclass
class RunRow:
    run_id: str
    workflow: str
    topic: str
    timestamp: str
    status: str
    latency_ms: float
    tokens_total: int
    step_count: int


@dataclass
class AnalysisState:
    extracted: dict[str, ExtractedEvidence] = field(default_factory=dict)
    loss_result: Optional[InformationLossResult] = None
    bundle: Optional[AnalysisBundle] = None
    error: Optional[str] = None
    done: bool = False


@dataclass
class DiffResult:
    run_a: str
    run_b: str
    steps: list[dict]          # [{agent, lat_a, lat_b, tok_a, tok_b, similarity}]
    first_divergence: str      # agent name where divergence starts
    overall_similarity: float


_db: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager()
        _db.initialize()
    return _db


def _extract_topic(trace_json_str: str) -> str:
    """Pull topic from the first step's input state."""
    try:
        data = json.loads(trace_json_str)
        steps = data.get("steps", [])
        if steps:
            hs = steps[0].get("handoff", {})
            inp = hs.get("input_state", {})
            if isinstance(inp, str):
                inp = json.loads(inp)
            topic = inp.get("topic", "")
            if topic:
                return topic[:60]
        return ""
    except Exception:
        return ""


def _total_tokens(run_id: str) -> int:
    db = get_db()
    return sum(
        (s.get("tokens_total") or 0)
        for s in db.get_steps_for_run(run_id)
    )


def list_runs(limit: int = 50) -> list[RunRow]:
    db = get_db()
    rows = db.list_runs(limit=limit)
    result = []
    for r in rows:
        steps  = db.get_steps_for_run(r["run_id"])
        tokens = sum(s.get("tokens_total", 0) or 0 for s in steps)
        lat    = sum(s.get("latency_ms",   0) or 0 for s in steps)
        topic  = _extract_topic(r.get("trace_json", ""))
        result.append(RunRow(
            run_id    = r["run_id"],
            workflow  = r.get("workflow", "unknown"),
            topic     = topic or r.get("workflow", "unknown"),
            timestamp = (r.get("timestamp", "")[:19] or "").replace("T", " "),
            status    = r.get("status", "unknown"),
            latency_ms= lat,
            tokens_total= tokens,
            step_count= len(steps),
        ))
    return result


def get_steps(run_id: str) -> list[StepRow]:
    db = get_db()
    return [
        StepRow(
            step            = r["step"],
            agent           = r["agent"],
            status          = r.get("status", "unknown"),
            latency_ms      = r.get("latency_ms", 0) or 0,
            tokens_prompt   = r.get("tokens_prompt", 0) or 0,
            tokens_completion= r.get("tokens_completion", 0) or 0,
            tokens_total    = r.get("tokens_total", 0) or 0,
        )
        for r in db.get_steps_for_run(run_id)
    ]


def get_trace_steps(run_id: str) -> list[dict]:
    """Return full step dicts from trace_json (includes handoff/output state)."""
    db = get_db()
    row = db.get_run(run_id)
    if not row or not row.get("trace_json"):
        return []
    data = json.loads(row["trace_json"])
    return data.get("steps", [])


def run_full_analysis(run_id: str) -> AnalysisState:
    """Days 9-12 pipeline. Caches result by run_id."""
    if run_id in _analysis_cache:
        return _analysis_cache[run_id]

    state = AnalysisState()
    try:
        db   = get_db()
        row  = db.get_run(run_id)
        if not row or not row.get("trace_json"):
            state.error = "trace_json not found"
            state.done  = True
            return state

        run  = RunTrace(**json.loads(row["trace_json"]))
        norm = Normalizer().normalize_run(run)

        extractor = EvidenceExtractor()
        for step in norm.steps:
            ev = extractor.extract(step.raw_output, agent=step.agent)
            state.extracted[step.agent] = ev

        r_ev = state.extracted.get("researcher")
        w_ev = state.extracted.get("writer")

        if r_ev and w_ev:
            state.loss_result = InformationLossRule().evaluate(
                researcher_evidence=r_ev,
                writer_evidence=w_ev,
                run_id=run_id,
            )
            ev_rec     = evidence_from_information_loss(state.loss_result)
            all_ev     = [e for e in [ev_rec] if e is not None]
            state.bundle = Arbiter().run(run_id=run_id, evidence=all_ev)
        else:
            state.error = "Missing researcher or writer step"

    except Exception as exc:
        state.error = str(exc)

    state.done = True
    _analysis_cache[run_id] = state
    return state


def run_explanation(bundle: AnalysisBundle) -> AnalysisBundle:
    return LLMExplainer().explain(bundle)


def get_cached(run_id: str) -> Optional[AnalysisState]:
    return _analysis_cache.get(run_id)


def get_metrics_data() -> dict:
    """Aggregate per-agent metrics across all runs for the Metrics view."""
    db   = get_db()
    runs = db.list_runs(limit=200)
    agents: dict[str, dict] = {}

    for r in runs:
        for s in db.get_steps_for_run(r["run_id"]):
            ag  = s.get("agent", "unknown")
            lat = s.get("latency_ms", 0) or 0
            tok = s.get("tokens_total", 0) or 0
            if ag not in agents:
                agents[ag] = {"latencies": [], "tokens": [], "runs": []}
            agents[ag]["latencies"].append(lat)
            agents[ag]["tokens"].append(tok)
            agents[ag]["runs"].append(r["run_id"])

    result = {}
    for ag, data in agents.items():
        lats = data["latencies"]
        toks = data["tokens"]
        result[ag] = {
            "avg_latency_ms": sum(lats) / len(lats) if lats else 0,
            "max_latency_ms": max(lats) if lats else 0,
            "avg_tokens":     sum(toks) / len(toks) if toks else 0,
            "total_tokens":   sum(toks),
            "run_count":      len(lats),
        }
    return result


def compute_diff(run_id_a: str, run_id_b: str) -> DiffResult:
    """Align two runs by agent identity and compute similarity."""
    steps_a = {s["agent"]: s for s in get_trace_steps(run_id_a)}
    steps_b = {s["agent"]: s for s in get_trace_steps(run_id_b)}
    agents  = list(steps_a.keys())

    def _sim(a: str, b: str) -> float:
        if not a or not b: return 0.0
        s1, s2 = set(a.split()), set(b.split())
        if not s1 and not s2: return 1.0
        return len(s1 & s2) / max(len(s1 | s2), 1)

    rows = []
    first_div = ""
    for ag in agents:
        sa  = steps_a.get(ag, {})
        sb  = steps_b.get(ag, {})
        out_a = sa.get("output", "")
        out_b = sb.get("output", "")
        sim = _sim(out_a, out_b)
        if sim < 0.85 and not first_div:
            first_div = ag
        rows.append({
            "agent": ag,
            "lat_a":  sa.get("latency_ms", 0) or 0,
            "lat_b":  sb.get("latency_ms", 0) or 0,
            "tok_a":  sa.get("tokens_total", 0) or 0,
            "tok_b":  sb.get("tokens_total", 0) or 0,
            "sim":    sim,
        })

    overall = sum(r["sim"] for r in rows) / len(rows) if rows else 0
    return DiffResult(
        run_a=run_id_a, run_b=run_id_b,
        steps=rows, first_divergence=first_div or "(none)",
        overall_similarity=overall,
    )


def total_cost_estimate() -> float:
    """Estimate total LLM cost across all runs (display in header)."""
    db   = get_db()
    runs = db.list_runs(limit=500)
    total = 0
    for r in runs:
        total += _total_tokens(r["run_id"])
    return total * COST_PER_TOKEN


def verdict_for_bundle(bundle: Optional[AnalysisBundle]) -> str:
    if bundle is None:
        return "UNKNOWN"
    v = bundle.priority_level.value
    if v == "P5":
        return "PASS"
    lr = _analysis_cache.get(bundle.run_id)
    if lr and lr.loss_result:
        return lr.loss_result.verdict
    return "WARNING"
