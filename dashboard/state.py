"""
dashboard/state.py — Data layer v2
All DB access, pipeline execution, and caching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from analyzers.arbiter import Arbiter, evidence_from_information_loss
from analyzers.detection.information_loss import InformationLossResult, InformationLossRule
from analyzers.evidence_extraction.extractor import EvidenceExtractor, ExtractedEvidence
from analyzers.explainer import LLMExplainer
from normalizer.normalizer import Normalizer
from schema.models import AnalysisBundle, EvidenceRecord, RunTrace
from storage.db import DatabaseManager

# ── Module-level analysis cache (process lifetime) ────────────────────────────
_analysis_cache: dict[str, AnalysisState] = {}

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
    # Day 29: verdict-level for priority badge + stale detection
    verdict_level: str = "UNANALYZED"  # "P1"…"P5" | "UNANALYZED"
    stale_verdict: bool = False  # True when rule engine version changed


@dataclass
class AnalysisState:
    extracted: dict[str, ExtractedEvidence] = field(default_factory=dict)
    loss_result: InformationLossResult | None = None
    bundle: AnalysisBundle | None = None
    error: str | None = None
    done: bool = False


@dataclass
class DiffRow:
    agent: str
    match_status: str  # MATCHED | MISSING_IN_A | MISSING_IN_B
    lat_a: float
    lat_b: float
    lat_delta: float  # lat_b - lat_a  (positive = B slower)
    tok_a: int
    tok_b: int
    tok_delta: int  # tok_b - tok_a
    sim: float  # [0,1] cosine similarity (0 if missing)
    diverged: bool
    method: str  # "cosine" | "jaccard" | "n/a"


@dataclass
class DiffResult:
    run_a: str
    run_b: str
    steps: list[dict]  # kept for backward-compat – mirrors DiffRow fields as dicts
    rows: list[DiffRow] = field(default_factory=list)
    first_divergence: str = "(none)"  # agent name where divergence starts
    overall_similarity: float = 0.0
    matched_count: int = 0
    missing_in_a_count: int = 0
    missing_in_b_count: int = 0


_db: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager()
        _db.initialize()
    return _db


def _extract_topic(trace_json_str: str) -> str:
    """Pull topic from the trace JSON, trying multiple paths."""
    if not trace_json_str:
        return ""
    try:
        data = json.loads(trace_json_str)
        # Path 1: steps[0].handoff.input_state.topic
        for step in data.get("steps", []):
            handoff = step.get("handoff", {})
            if isinstance(handoff, str):
                try:
                    handoff = json.loads(handoff)
                except Exception:
                    continue
            for key in ("input_state", "output_state", "filtered_state"):
                state_val = handoff.get(key, {})
                if isinstance(state_val, str):
                    try:
                        state_val = json.loads(state_val)
                    except Exception:
                        continue
                topic = state_val.get("topic", "")
                if topic:
                    return str(topic)[:60]
        # Path 2: top-level initial_state
        init = data.get("initial_state", {})
        if isinstance(init, str):
            try:
                init = json.loads(init)
            except Exception:
                init = {}
        topic = init.get("topic", "")
        if topic:
            return str(topic)[:60]
    except Exception:
        pass
    return ""


def _total_tokens(run_id: str) -> int:
    db = get_db()
    return sum((s.get("tokens_total") or 0) for s in db.get_steps_for_run(run_id))


_RULE_ENGINE_VERSION = "v1"  # bump this when rule logic changes to trigger stale badges


def list_runs(
    limit: int = 50,
    agent_filter: str | None = None,
    verdict_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "date",  # "date" | "priority" | "latency"
) -> list[RunRow]:
    db = get_db()
    rows = db.list_runs(limit=limit)
    result: list[RunRow] = []

    for r in rows:
        run_id = r["run_id"]
        steps = db.get_steps_for_run(run_id)

        # Agent filter: skip run if no step matches the requested agent
        if agent_filter and agent_filter != "All agents":
            agents_in_run = {s.get("agent", "") for s in steps}
            if agent_filter not in agents_in_run:
                continue

        tokens = sum(s.get("tokens_total", 0) or 0 for s in steps)
        lat = sum(s.get("latency_ms", 0) or 0 for s in steps)

        full = db.get_run(run_id)
        topic = _extract_topic(full.get("trace_json", "") if full else "")

        # Date filters (timestamps are "YYYY-MM-DDTHH:MM:SS…")
        ts = r.get("timestamp", "")
        if date_from and ts and ts[:10] < date_from:
            continue
        if date_to and ts and ts[:10] > date_to:
            continue

        # Attach cached verdict level
        cached = _analysis_cache.get(run_id)
        if cached and cached.bundle:
            verdict_level = cached.bundle.priority_level.value
        else:
            verdict_level = "UNANALYZED"

        # Verdict filter
        if verdict_filter and verdict_filter not in ("All", ""):
            if verdict_level != verdict_filter:
                continue

        run_row = RunRow(
            run_id=run_id,
            workflow=r.get("workflow", "unknown"),
            topic=topic or r.get("workflow", "unknown"),
            timestamp=(r.get("timestamp", "")[:19] or "").replace("T", " "),
            status=r.get("status", "unknown"),
            latency_ms=lat,
            tokens_total=tokens,
            step_count=len(steps),
            verdict_level=verdict_level,
            stale_verdict=False,  # reserved for future rule-version tracking
        )
        result.append(run_row)

    # Sorting
    _P_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5, "UNANALYZED": 9}
    if sort_by == "priority":
        result.sort(key=lambda x: _P_ORDER.get(x.verdict_level, 9))
    elif sort_by == "latency":
        result.sort(key=lambda x: x.latency_ms, reverse=True)
    # "date" is already newest-first from DB query

    return result


def get_steps(run_id: str) -> list[StepRow]:
    db = get_db()
    return [
        StepRow(
            step=r["step"],
            agent=r["agent"],
            status=r.get("status", "unknown"),
            latency_ms=r.get("latency_ms", 0) or 0,
            tokens_prompt=r.get("tokens_prompt", 0) or 0,
            tokens_completion=r.get("tokens_completion", 0) or 0,
            tokens_total=r.get("tokens_total", 0) or 0,
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


def _parse_handoff(handoff_raw: dict | str | None) -> dict:
    """Normalise the handoff field — may be a nested dict or a JSON string."""
    if handoff_raw is None:
        return {}
    if isinstance(handoff_raw, str):
        try:
            handoff_raw = json.loads(handoff_raw)
        except Exception:
            return {}
    return handoff_raw if isinstance(handoff_raw, dict) else {}


def _extract_diff_from_states(input_state: dict, output_state: dict) -> dict[str, list[str]]:
    """
    Re-run the HandoffCapture four-category diff logic on two state dicts.
    Returns {"added": [...], "modified": [...], "dropped": [...], "unchanged": [...]}.
    """
    from capture.handoff import HandoffCapture  # lazy import to avoid circular dep

    diff = HandoffCapture._compute_diff(input_state, output_state)
    return {
        "added": diff.added_keys,
        "modified": diff.modified_keys,
        "dropped": diff.dropped_keys,
        "unchanged": diff.unchanged_keys,
    }


def get_step_handoff_detail(run_id: str, agent: str) -> dict:
    """
    Return a rich detail dict for a single agent step.

    Includes structured input_state / filtered_state / output_state and the
    four-category handoff diff.  Returns an empty dict if the run or agent
    cannot be found.

    Schema
    ------
    {
        "agent": str,
        "step": int,
        "latency_ms": float,
        "tokens_total": int,
        "status": str,
        "input_state": dict,
        "filtered_state": dict,
        "output_state": dict,
        "diff": {"added": [...], "modified": [...], "dropped": [...], "unchanged": [...]},
        "blamed": bool,
    }
    """
    trace_steps = get_trace_steps(run_id)
    step_dict = next((s for s in trace_steps if s.get("agent") == agent), None)
    if not step_dict:
        return {}

    handoff = _parse_handoff(step_dict.get("handoff"))
    input_state: dict = handoff.get("input_state") or {}
    filtered_state: dict = handoff.get("filtered_state") or {}
    output_state: dict = handoff.get("output_state") or {}

    # If output_state is absent, reconstruct from input + filtered (LangGraph merge)
    if not output_state and input_state and filtered_state:
        output_state = {**input_state, **filtered_state}

    diff = _extract_diff_from_states(input_state, output_state)

    # Blamed agent from analysis cache
    cached = _analysis_cache.get(run_id)
    blamed_agent = cached.bundle.primary_agent if (cached and cached.bundle) else None

    return {
        "agent": agent,
        "step": step_dict.get("step", 0),
        "latency_ms": step_dict.get("latency_ms", 0.0) or 0.0,
        "tokens_total": step_dict.get("tokens_total", 0) or 0,
        "status": step_dict.get("status", "unknown"),
        "input_state": input_state,
        "filtered_state": filtered_state,
        "output_state": output_state,
        "diff": diff,
        "blamed": (agent == blamed_agent),
    }


def get_timeline_data(run_id: str) -> list[dict]:
    """
    Return an ordered list of step detail dicts for every agent in the run.

    Each element is the result of get_step_handoff_detail() for that agent.
    Steps are ordered by the ``step`` field in trace_json.
    """
    trace_steps = get_trace_steps(run_id)
    if not trace_steps:
        return []

    # Order by step index, dedupe on agent name (keep first occurrence)
    seen: set[str] = set()
    ordered: list[dict] = []
    for s in sorted(trace_steps, key=lambda x: x.get("step", 0)):
        agent = s.get("agent", "")
        if agent and agent not in seen:
            seen.add(agent)
            detail = get_step_handoff_detail(run_id, agent)
            if detail:
                ordered.append(detail)
    return ordered


def get_unique_agents() -> list[str]:
    """Return sorted distinct agent names found in the steps table."""
    db = get_db()
    runs = db.list_runs(limit=500)
    agents: set[str] = set()
    for r in runs:
        for s in db.get_steps_for_run(r["run_id"]):
            ag = s.get("agent", "")
            if ag:
                agents.add(ag)
    return sorted(agents)


def get_aggregate_stats(runs: list[RunRow]) -> dict:
    """Compute stat card values from a list of RunRows.

    Returns:
        total       — total runs in current filtered view
        analyzed    — runs with a cached verdict
        p1_p2_count — high-severity verdicts (P1 or P2)
        avg_latency — average latency across all runs in ms
        total_tokens
        top_failing_agent — agent name most often blamed, or None
    """
    total = len(runs)
    analyzed = sum(1 for r in runs if r.verdict_level != "UNANALYZED")
    p1_p2_count = sum(1 for r in runs if r.verdict_level in ("P1", "P2"))
    avg_lat = (sum(r.latency_ms for r in runs) / total) if total else 0.0
    total_tok = sum(r.tokens_total for r in runs)

    # Count blamed agents from cache
    agent_blame: dict[str, int] = {}
    for r in runs:
        cached = _analysis_cache.get(r.run_id)
        if cached and cached.bundle and cached.bundle.primary_agent:
            ag = cached.bundle.primary_agent
            agent_blame[ag] = agent_blame.get(ag, 0) + 1
    top_agent = max(agent_blame, key=lambda a: agent_blame[a]) if agent_blame else None

    return {
        "total": total,
        "analyzed": analyzed,
        "p1_p2_count": p1_p2_count,
        "avg_latency": avg_lat,
        "total_tokens": total_tok,
        "top_failing_agent": top_agent,
    }


def persist_rule_matches(run_id: str, bundle: object, db: DatabaseManager | None = None) -> int:
    """
    Store a bundle's rule matches in the rule_matches table (Day 32).

    Rule IDs are normalized (STAT-LAT-* -> STAT-LAT). Never raises: a persistence
    failure must not break analysis. Returns rows written (0 on failure).
    """
    try:
        from analyzers.rule_catalog import normalize_rule_id

        matches = []
        for m in getattr(bundle, "rule_matches", None) or []:
            matches.append(
                {
                    "rule_id": normalize_rule_id(str(m.rule_id)),
                    "rule_version": m.rule_version,
                    "category": m.category,
                    "severity": m.severity,
                    "agent": m.agent,
                    "step": m.step,
                    "description": m.description,
                }
            )
        return (db or get_db()).insert_rule_matches(run_id, matches)
    except Exception:
        return 0


def get_rule_stats(category: str | None = None, db: DatabaseManager | None = None) -> list[dict]:
    """
    Rule Explorer rows: catalog merged with DB fire stats (Day 32).

    Every catalog rule appears (never-fired = 0). Rules seen in the DB but missing
    from the catalog are appended as category 'unknown'. Sorted by times_fired desc,
    then rule_id. ``category`` filters rows (None / 'all' = no filter).
    """
    from analyzers.rule_catalog import RULE_CATALOG

    try:
        stats = {s["rule_id"]: s for s in (db or get_db()).get_rule_stats()}
    except Exception:
        stats = {}

    rows: list[dict] = []
    for rid, info in RULE_CATALOG.items():
        s = stats.pop(rid, None)
        rows.append(
            {
                "rule_id": rid,
                "name": info["name"],
                "category": info["category"],
                "version": (s or {}).get("rule_version") or info["version"],
                "description": info["description"],
                "times_fired": (s or {}).get("times_fired", 0),
                "last_triggered": (s or {}).get("last_triggered"),
                "example_run": (s or {}).get("example_run"),
            }
        )
    for rid, s in stats.items():  # uncatalogued rules still shown
        rows.append(
            {
                "rule_id": rid,
                "name": rid,
                "category": s.get("category") or "unknown",
                "version": s.get("rule_version") or "?",
                "description": "(not in catalog)",
                "times_fired": s["times_fired"],
                "last_triggered": s["last_triggered"],
                "example_run": s["example_run"],
            }
        )

    if category and category != "all":
        rows = [r for r in rows if r["category"] == category]
    rows.sort(key=lambda r: (-r["times_fired"], r["rule_id"]))
    return rows


def get_cached(run_id: str) -> AnalysisState | None:
    """Return the cached AnalysisState for a run, or None if not yet analyzed."""
    return _analysis_cache.get(run_id)


def run_full_analysis(run_id: str, db: DatabaseManager | None = None) -> AnalysisState:
    """Days 9-12 pipeline. Caches result by run_id."""
    if run_id in _analysis_cache:
        return _analysis_cache[run_id]

    state = AnalysisState()
    try:
        db = db or get_db()
        row = db.get_run(run_id)
        if not row or not row.get("trace_json"):
            state.error = "trace_json not found"
            state.done = True
            return state

        run = RunTrace(**json.loads(row["trace_json"]))
        norm = Normalizer().normalize_run(run)

        extractor = EvidenceExtractor()
        for step in norm.steps:
            ev = extractor.extract(step.raw_output, agent=step.agent)
            state.extracted[step.agent] = ev

        r_ev = state.extracted.get("researcher")
        w_ev = state.extracted.get("writer")

        all_ev: list[EvidenceRecord] = []
        if r_ev and w_ev:
            state.loss_result = InformationLossRule().evaluate(
                researcher_evidence=r_ev,
                writer_evidence=w_ev,
                run_id=run_id,
            )
            ev_rec = evidence_from_information_loss(state.loss_result)
            if ev_rec:
                all_ev.append(ev_rec)

        # Day 27/28: Include StatisticalDetector anomalies
        try:
            from analyzers.detection.statistical_detector import StatisticalDetector

            stat_report = StatisticalDetector(db).analyze_run(run_id)
            all_ev.extend(stat_report.anomalies)
        except Exception:
            pass

        state.bundle = Arbiter().run(run_id=run_id, evidence=all_ev)

    except Exception as exc:
        state.error = str(exc)

    state.done = True
    _analysis_cache[run_id] = state

    # Day 32: Persist rule matches for the Rule Explorer
    if state.bundle:
        persist_rule_matches(run_id, state.bundle)

    # Day 31: Fire alert if verdict meets threshold
    try:
        from analyzers.alerter import Alerter

        if state.bundle:
            Alerter().fire(run_id, state.bundle)
    except Exception:
        pass  # alerting must never crash the analysis pipeline

    return state


def run_explanation(bundle: AnalysisBundle) -> AnalysisBundle:
    return LLMExplainer().explain(bundle)


def explain_agent_evidence(agent, evidence, bundle):
    """Focused LLM explanation for one agent's extracted evidence metrics."""
    import os

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    from config.config_loader import get

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not set."

    is_blamed = bundle and (bundle.primary_agent or "").lower() == agent.lower()
    if bundle:
        arbiter_note = (
            f"Arbiter verdict: '{bundle.primary_agent}' caused a {bundle.primary_cause.value} "
            f"failure ({bundle.priority_level.value}). "
            + ("THIS is the blamed agent." if is_blamed else "This agent is not blamed.")
        )
    else:
        arbiter_note = "No arbiter verdict available yet."

    prompt = "\n".join(
        [
            f"AGENT: {agent}",
            f"SOURCES CITED: {evidence.source_count}",
            f"NAMED ENTITIES EXTRACTED: {evidence.entity_count}",
            f"TOOL CALLS MADE: {len(evidence.tool_calls)}",
            "",
            f"ARBITER CONTEXT: {arbiter_note}",
            "",
            "In 3-4 sentences explain:",
            f"1. What do these numbers reveal about what the {agent} agent did in the pipeline?",
            "2. Are these source/entity counts high, low, or normal for this agent role?",
            "3. How does this agent relate to the overall pipeline verdict?",
            "4. What should an engineer inspect first when debugging this agent?",
            "",
            "Be specific and technical. Use hedged language (this is heuristic analysis). No bullet points.",
        ]
    )

    try:
        llm = ChatGroq(
            model=get("llm", "model"),
            temperature=0.0,
            max_tokens=512,
            api_key=api_key,
        )
        resp = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an AI observability analyst. Explain extracted evidence metrics "
                        "for a single agent in a multi-agent research pipeline. "
                        "Be concise, technical, and actionable. Write in flowing prose — no bullet points."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        return str(resp.content)
    except Exception as exc:
        return f"LLM error: {exc}"


def get_metrics_data() -> dict:
    """Aggregate per-agent metrics across all runs for the Metrics view."""
    db = get_db()
    runs = db.list_runs(limit=200)
    agents: dict[str, dict] = {}

    for r in runs:
        for s in db.get_steps_for_run(r["run_id"]):
            ag = s.get("agent", "unknown")
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
            "avg_tokens": sum(toks) / len(toks) if toks else 0,
            "total_tokens": sum(toks),
            "run_count": len(lats),
        }
    return result


def get_failure_timeline(days: int = 14) -> list[dict]:
    """
    Return daily P1/P2 failure counts for the last ``days`` days.

    Each element: {"date": "YYYY-MM-DD", "p1": int, "p2": int, "total": int}
    Sorted ascending by date.
    """
    db = get_db()
    runs = db.list_runs(limit=500)

    # Group by date
    from collections import defaultdict

    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"p1": 0, "p2": 0})

    for r in runs:
        run_id = r["run_id"]
        ts = r.get("timestamp", "") or ""
        date_str = ts[:10] if len(ts) >= 10 else "unknown"

        cached = _analysis_cache.get(run_id)
        if cached and cached.bundle:
            level = str(cached.bundle.priority_level.value)
            if level == "P1":
                buckets[date_str]["p1"] += 1
            elif level == "P2":
                buckets[date_str]["p2"] += 1

    # Build sorted list (last N days only)
    result = []
    for date_str in sorted(buckets.keys()):
        if date_str == "unknown":
            continue
        b = buckets[date_str]
        result.append(
            {
                "date": date_str,
                "p1": b["p1"],
                "p2": b["p2"],
                "total": b["p1"] + b["p2"],
            }
        )

    return result[-days:]


def get_cause_breakdown() -> list[dict]:
    """
    Return top failure causes across all cached analyses.

    Each element: {"cause": str, "count": int}  sorted by count descending.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for cached in _analysis_cache.values():
        if cached and cached.bundle and cached.bundle.primary_cause:
            cause = str(cached.bundle.primary_cause.value)
            counts[cause] += 1

    return [{"cause": c, "count": n} for c, n in counts.most_common()]


def _load_run_trace(run_id: str) -> RunTrace | None:
    """Load a full RunTrace from DB trace_json."""
    db = get_db()
    row = db.get_run(run_id)
    if not row or not row.get("trace_json"):
        return None
    return RunTrace(**json.loads(row["trace_json"]))


def compute_diff(run_id_a: str, run_id_b: str) -> DiffResult:
    """
    Day 26: Align two runs using GraphAligner (Day 24) and score semantic
    similarity using SemanticSimilarityEngine (Day 25).

    Falls back to empty rows if either trace cannot be loaded.
    """
    from diff_engine import align_traces, score_similarity

    trace_a = _load_run_trace(run_id_a)
    trace_b = _load_run_trace(run_id_b)

    if trace_a is None or trace_b is None:
        return DiffResult(
            run_a=run_id_a,
            run_b=run_id_b,
            steps=[],
            rows=[],
            first_divergence="(trace not found)",
            overall_similarity=0.0,
        )

    # Step 1: Graph-based alignment (by agent identity + parent/child topology)
    alignment = align_traces(trace_a, trace_b)

    # Step 2: Semantic similarity for all MATCHED step pairs
    sim_report = score_similarity(alignment)

    # Build a lookup from agent → StepSimilarityScore
    sim_by_agent = {s.agent: s for s in sim_report.scores}

    # Build per-step latency/token lookups from DB (fast, avoids re-parsing JSON)
    db = get_db()

    def _step_metrics(run_id: str) -> dict[str, dict]:
        """Map agent → {lat, tok} from DB step rows."""
        result: dict[str, dict] = {}
        for s in db.get_steps_for_run(run_id):
            ag = s.get("agent", "")
            result[ag] = {
                "lat": float(s.get("latency_ms") or 0),
                "tok": int(s.get("tokens_total") or 0),
            }
        return result

    metrics_a = _step_metrics(run_id_a)
    metrics_b = _step_metrics(run_id_b)

    diff_rows: list[DiffRow] = []
    legacy_steps: list[dict] = []

    for pair in alignment.pairs:
        agent = pair.agent
        status = pair.status.value  # "MATCHED" | "MISSING_IN_A" | "MISSING_IN_B"

        m_a = metrics_a.get(agent, {"lat": 0.0, "tok": 0})
        m_b = metrics_b.get(agent, {"lat": 0.0, "tok": 0})
        lat_a = m_a["lat"] if pair.step_a else 0.0
        lat_b = m_b["lat"] if pair.step_b else 0.0
        tok_a = m_a["tok"] if pair.step_a else 0
        tok_b = m_b["tok"] if pair.step_b else 0

        if status == "MATCHED" and agent in sim_by_agent:
            sc = sim_by_agent[agent]
            sim = sc.similarity
            diverged = sc.diverged
            method = sc.method
        else:
            sim = 0.0
            diverged = status != "MATCHED"
            method = "n/a"

        row = DiffRow(
            agent=agent,
            match_status=status,
            lat_a=lat_a,
            lat_b=lat_b,
            lat_delta=lat_b - lat_a,
            tok_a=tok_a,
            tok_b=tok_b,
            tok_delta=tok_b - tok_a,
            sim=sim,
            diverged=diverged,
            method=method,
        )
        diff_rows.append(row)

        # Build backward-compat legacy dict for existing UI code
        legacy_steps.append(
            {
                "agent": agent,
                "lat_a": lat_a,
                "lat_b": lat_b,
                "tok_a": tok_a,
                "tok_b": tok_b,
                "sim": sim,
                "match_status": status,
                "lat_delta": lat_b - lat_a,
                "tok_delta": tok_b - tok_a,
                "diverged": diverged,
                "method": method,
            }
        )

    return DiffResult(
        run_a=run_id_a,
        run_b=run_id_b,
        steps=legacy_steps,
        rows=diff_rows,
        first_divergence=sim_report.first_divergence_agent or "(none)",
        overall_similarity=sim_report.average_similarity,
        matched_count=alignment.matched_count,
        missing_in_a_count=alignment.missing_in_a_count,
        missing_in_b_count=alignment.missing_in_b_count,
    )


def get_cost_per_token() -> float:
    """Return configured USD cost per token from config.yaml (llm.cost_per_token_usd)."""
    try:
        from config.config_loader import get

        val = get("llm", "cost_per_token_usd", COST_PER_TOKEN)
        return float(val) if val is not None else COST_PER_TOKEN
    except Exception:
        return COST_PER_TOKEN


def get_budget_alert_usd() -> float:
    """Return configured session cost budget alert threshold in USD (llm.budget_alert_usd)."""
    try:
        from config.config_loader import get

        val = get("llm", "budget_alert_usd", 1.00)
        return float(val) if val is not None else 1.00
    except Exception:
        return 1.00


def total_cost_estimate() -> float:
    """
    Estimate total LLM cost across all runs (display in header).
    Logs a WARNING if total estimated cost exceeds llm.budget_alert_usd.
    """
    db = get_db()
    runs = db.list_runs(limit=500)
    total = 0
    for r in runs:
        total += _total_tokens(r["run_id"])
    cost = total * get_cost_per_token()
    budget = get_budget_alert_usd()
    if cost > budget:
        from config.logging_config import get_logger

        get_logger("cost_control").warning(
            f"Session LLM cost (${cost:.4f}) exceeded budget_alert_usd (${budget:.2f})"
        )
    return cost


def get_budget_status() -> dict[str, float | bool]:
    """Return current session cost, budget threshold, and whether budget is exceeded."""
    cost = total_cost_estimate()
    budget = get_budget_alert_usd()
    return {
        "cost_usd": cost,
        "budget_usd": budget,
        "exceeded": cost > budget,
    }


def verdict_for_bundle(bundle: AnalysisBundle | None) -> str:
    if bundle is None:
        return "UNKNOWN"
    v = bundle.priority_level.value
    if v == "P5":
        return "PASS"
    lr = _analysis_cache.get(bundle.run_id)
    if lr and lr.loss_result:
        return lr.loss_result.verdict
    return "WARNING"
