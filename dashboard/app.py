"""
dashboard/app.py — AgentLens Dashboard v2
==========================================
6 views, matching the Run Explorer screenshot design.

Run: python -m dashboard.app
"""

from __future__ import annotations

import json
import asyncio
from nicegui import ui, app, run

import dashboard.state as state
from dashboard.theme import (
    GLOBAL_CSS, BG, BG_SIDEBAR, CARD, BORDER, TEXT, TEXT_MUTED, TEXT_DIM,
    PURPLE, CYAN, GREEN, AMBER, RED, GRAY,
    VERDICT_COLOR, PRIORITY_COLOR, CAUSE_COLOR, STEP_COLOR,
    badge, verdict_badge, priority_badge, cause_badge, rule_badge,
    row_bg, fmt_ms, bar_html,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    ui.add_head_html(GLOBAL_CSS)
    ui.query("body").style(f"background:{BG};color:{TEXT};")


def _logo_html() -> str:
    return f"""
    <div style="width:30px;height:30px;background:linear-gradient(135deg,{PURPLE},{CYAN});
                border-radius:7px;display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:12px;color:#fff;flex-shrink:0;">AL</div>
    <span style="font-size:15px;font-weight:700;color:{TEXT};">AgentLens</span>
    """


def _cost_ticker() -> str:
    try:
        cost = state.total_cost_estimate()
        return f'<span class="al-cost">~${cost:.4f} est.</span>'
    except Exception:
        return ""


def header(pipeline_name: str = "research_report_pipeline"):
    ui.html(f"""
    <div class="al-header">
      {_logo_html()}
      <span style="color:{TEXT_DIM};font-size:18px;padding:0 4px;">|</span>
      <span style="font-size:12px;color:{TEXT_MUTED};font-family:'JetBrains Mono',monospace;">
        {pipeline_name}
      </span>
      <div style="flex:1;"></div>
      {_cost_ticker()}
    </div>
    """)


def nav_bar(active: str, run_id: str = ""):
    """Top navigation tabs. active = one of: runs, metrics, diff, timeline, evidence, explain"""
    global_tabs = [
        ("🔍", "Runs",    "runs",    "/"),
        ("📈", "Metrics", "metrics", "/metrics"),
        ("⚖️",  "Diff",   "diff",    "/diff"),
    ]
    run_tabs = [
        ("⏱",  "Timeline", "timeline", f"/run/{run_id}"),
        ("🔬", "Evidence",  "evidence", f"/run/{run_id}/evidence"),
        ("✦",  "Explain",  "explain",  f"/run/{run_id}/explain"),
    ] if run_id else []

    tabs_html = ""
    for icon, label, key, href in global_tabs:
        cls = "al-nav-tab active" if active == key else "al-nav-tab"
        tabs_html += f'<a href="{href}" class="{cls}">{icon} {label}</a>'

    if run_tabs:
        tabs_html += f'<div class="al-nav-divider"></div>'
        tabs_html += f'<span style="font-size:10px;color:{TEXT_DIM};padding:0 6px;font-family:monospace;">{run_id[:14]}…</span>'
        for icon, label, key, href in run_tabs:
            cls = "al-nav-tab active" if active == key else "al-nav-tab"
            tabs_html += f'<a href="{href}" class="{cls}">{icon} {label}</a>'

    ui.html(f'<div class="al-nav">{tabs_html}</div>')


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Run Explorer  /
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/")
def runs_page():
    inject_css()
    header()
    nav_bar("runs")

    with ui.element("div").classes("al-content"):
        runs = state.list_runs(limit=50)

        # ── Filter bar ────────────────────────────────────────────────────────
        with ui.element("div").style(f"display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;"):
            ui.html(f'<div class="al-section">Run Explorer</div>')
            # (filters are cosmetic for MVP; full filtering = Day 22)
            ui.html(f"""
            <div style="display:flex;gap:8px;">
              <select class="al-select"><option>All agents</option><option>researcher</option><option>writer</option><option>verifier</option></select>
              <select class="al-select"><option>All time</option><option>Last 7 days</option><option>Last 30 days</option></select>
            </div>
            """)

        # ── Stat cards ────────────────────────────────────────────────────────
        analyzed     = [r for r in runs if state.get_cached(r.run_id)]
        warnings     = sum(1 for r in runs if state.get_cached(r.run_id) and
                           state.get_cached(r.run_id).loss_result and
                           state.get_cached(r.run_id).loss_result.verdict != "PASS")
        avg_lat      = (sum(r.latency_ms for r in runs) / len(runs)) if runs else 0
        total_tok    = sum(r.tokens_total for r in runs)

        with ui.element("div").style("display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;"):
            for label, val, color, sub in [
                ("Total Runs",    str(len(runs)),      PURPLE, f"{len(analyzed)} analyzed"),
                ("Avg Latency",   fmt_ms(avg_lat),     CYAN,   "per pipeline run"),
                ("Total Tokens",  f"{total_tok:,}",    AMBER,  f"~${total_tok*0.000005:.3f} est."),
                ("Warnings",      str(warnings),       RED if warnings else GREEN, "from analyzed runs"),
            ]:
                ui.html(f"""
                <div class="al-stat">
                  <div class="al-stat-label">{label}</div>
                  <div class="al-stat-value" style="color:{color};">{val}</div>
                  <div class="al-stat-sub">{sub}</div>
                </div>
                """)

        # ── Runs table ────────────────────────────────────────────────────────
        COLS = "2fr 1.8fr 120px 80px"
        with ui.element("div").classes("al-table"):
            # Header
            ui.html(f"""
            <div class="al-thead" style="grid-template-columns:{COLS};">
              <span>Run / Topic</span>
              <span>Primary Cause</span>
              <span>Verdict</span>
              <span>Latency</span>
            </div>
            """)

            if not runs:
                ui.html(f"""
                <div style="padding:48px;text-align:center;color:{TEXT_MUTED};">
                  <div style="font-size:36px;margin-bottom:12px;">🔬</div>
                  <div>No pipeline runs yet.</div>
                  <div style="font-size:12px;margin-top:8px;">
                    Run: <code style="color:{PURPLE};">python app/main.py --topic "..."</code>
                  </div>
                </div>
                """)

            for r in runs:
                _run_row(r, COLS)


def _run_row(r: state.RunRow, cols: str):
    cached = state.get_cached(r.run_id)
    bundle = cached.bundle if cached else None
    loss   = cached.loss_result if cached else None

    # Determine tint from verdict
    verdict = (loss.verdict if loss else ("PASS" if bundle and bundle.priority_level.value == "P5" else "UNKNOWN"))
    bg      = row_bg(verdict)

    # Primary cause display
    if bundle:
        cause_str  = bundle.primary_cause.value
        agent_str  = bundle.primary_agent or ""
        cause_disp = f'{cause_str.capitalize()} {"in handoff" if "workflow" in cause_str else ""}' \
                     f'<br><span style="font-size:11px;color:{TEXT_MUTED};font-family:monospace;">({agent_str})</span>'
    else:
        cause_disp = f'<span style="color:{TEXT_DIM};">—</span>'

    verdict_disp = verdict_badge(verdict, bundle.grounded if bundle else False) if bundle else \
                   f'<span style="color:{TEXT_DIM};font-size:12px;">Unanalyzed</span>'

    # Row container
    row_el = ui.element("div").style(
        f"background:{bg};"
        f"grid-template-columns:{cols};"
    ).classes("al-trow")

    with row_el:
        with ui.element("div").classes("al-tcell"):
            ui.html(f"""
            <div>
              <div style="font-size:13px;font-weight:500;">{r.topic or r.workflow}</div>
              <div class="al-mono" style="font-size:10px;color:{TEXT_MUTED};margin-top:3px;">{r.run_id}</div>
            </div>
            """)
        with ui.element("div").classes("al-tcell"):
            ui.html(f'<div style="font-size:13px;line-height:1.5;">{cause_disp}</div>')
        with ui.element("div").classes("al-tcell"):
            ui.html(verdict_disp)
        with ui.element("div").classes("al-tcell"):
            ui.html(f'<span style="font-size:13px;">{fmt_ms(r.latency_ms)}</span>')

    # Inline expansion panel (hidden by default)
    expansion = ui.element("div").classes("al-expansion").style("display:none;")

    with expansion:
        if bundle:
            _inline_verdict_panel(bundle, loss, r.run_id)
        else:
            _inline_analyze_panel(r.run_id, expansion, row_el, cols)

    # Click row → toggle expansion
    def toggle(e, exp=expansion):
        exp.style(
            remove="display:none;" if "display:none" in (exp._style or "") else "",
        )
        is_hidden = "display:none" in (exp._props.get("style", "") or exp._style or "")
        exp.style("display:none;" if not is_hidden else "display:block;")
        # simpler: just toggle
        exp.set_visibility(not exp.is_deleted)

    # Use a simpler toggle via JS
    row_el.on("click", lambda e, exp=expansion: _toggle(exp))


def _toggle(el):
    """Toggle display:none on a NiceGUI element."""
    # Use client-side JS for instant response
    ui.run_javascript(f"var el = document.getElementById('{el.id}'); el.style.display = el.style.display === 'none' ? 'block' : 'none';")


def _inline_verdict_panel(bundle, loss, run_id: str):
    p_col = PRIORITY_COLOR.get(bundle.priority_level.value, GRAY)
    c_col = CAUSE_COLOR.get(bundle.primary_cause.value, GRAY)
    conf  = f"{loss.confidence:.0%}" if loss else "—"
    verdict_str = loss.verdict if loss else "UNKNOWN"

    ui.html(f"""
    <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
      <div>
        <div class="al-section" style="margin-bottom:6px;">Priority</div>
        {priority_badge(bundle.priority_level.value)}
      </div>
      <div>
        <div class="al-section" style="margin-bottom:6px;">Cause</div>
        {cause_badge(bundle.primary_cause.value)}
      </div>
      <div>
        <div class="al-section" style="margin-bottom:6px;">Agent</div>
        <span style="font-size:13px;font-weight:500;color:{STEP_COLOR.get(bundle.primary_agent or '', GRAY)};">
          {bundle.primary_agent or 'N/A'}
        </span>
      </div>
      <div>
        <div class="al-section" style="margin-bottom:6px;">Confidence</div>
        <span style="font-size:13px;font-weight:600;">{conf}</span>
      </div>
      <div>
        <div class="al-section" style="margin-bottom:6px;">Grounded</div>
        <span style="color:{"#22c55e" if bundle.grounded else TEXT_MUTED};">
          {"✓ Yes" if bundle.grounded else "✗ No (heuristic)"}
        </span>
      </div>
      <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
        <button class="al-copy" onclick="navigator.clipboard.writeText(`**Run:** {run_id}\\n**Verdict:** {verdict_str} ({bundle.priority_level.value})\\n**Cause:** {bundle.primary_cause.value}\\n**Agent:** {bundle.primary_agent or 'N/A'}\\n**Confidence:** {conf}`)">
          📋 Copy verdict
        </button>
        <a href="/run/{run_id}" style="font-size:12px;color:{PURPLE};text-decoration:none;padding:4px 10px;border:1px solid {PURPLE}44;border-radius:6px;">
          Open trace →
        </a>
        <a href="/run/{run_id}/explain" style="font-size:12px;color:{CYAN};text-decoration:none;padding:4px 10px;border:1px solid {CYAN}44;border-radius:6px;">
          ✦ Explain
        </a>
      </div>
    </div>
    """)


def _inline_analyze_panel(run_id: str, expansion, row_el, cols: str):
    content_area = ui.element("div")

    async def analyze():
        content_area.clear()
        with content_area:
            with ui.element("div").style(f"display:flex;align-items:center;gap:10px;color:{TEXT_MUTED};"):
                ui.spinner(size="xs").style(f"color:{PURPLE};")
                ui.label("Analyzing… (LLM calls, ~8-10s)")

        result = await run.io_bound(state.run_full_analysis, run_id)
        content_area.clear()

        with content_area:
            if result.error or not result.bundle:
                ui.html(f'<span style="color:{RED};">{result.error or "Analysis failed"}</span>')
            else:
                _inline_verdict_panel(result.bundle, result.loss_result, run_id)

    with content_area:
        ui.button("▶  Analyze this run", on_click=analyze).style(
            f"background:{PURPLE}22;color:{PURPLE};border:1px solid {PURPLE}44;"
            f"border-radius:7px;padding:7px 16px;font-size:13px;font-weight:600;"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Trace Timeline  /run/{run_id}
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/run/{run_id}")
def trace_page(run_id: str):
    inject_css()
    header()
    nav_bar("timeline", run_id)

    with ui.element("div").classes("al-content"):
        ui.html(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
          <a href="/" style="color:{TEXT_MUTED};text-decoration:none;font-size:12px;">Runs</a>
          <span style="color:{TEXT_DIM};">›</span>
          <span class="al-mono" style="font-size:12px;">{run_id}</span>
          <span style="color:{TEXT_DIM};">›</span>
          <span style="font-size:12px;color:{TEXT};">Trace Timeline</span>
        </div>
        """)

        steps_db    = state.get_steps(run_id)
        steps_trace = state.get_trace_steps(run_id)
        trace_by_agent = {s.get("agent", ""): s for s in steps_trace}

        if not steps_db:
            ui.html(f'<div style="color:{TEXT_MUTED};">No steps found.</div>')
            return

        max_lat = max((s.latency_ms for s in steps_db), default=1)
        max_tok = max((s.tokens_total for s in steps_db), default=1)

        ui.html(f'<div class="al-section" style="margin-bottom:16px;">Step-by-step pipeline execution</div>')

        # ── Node flow ─────────────────────────────────────────────────────────
        with ui.element("div").style("display:flex;align-items:flex-start;gap:0;overflow-x:auto;padding-bottom:8px;"):
            for i, s in enumerate(steps_db):
                color = STEP_COLOR.get(s.agent, GRAY)
                t_data = trace_by_agent.get(s.agent, {})

                node = ui.element("div").style(
                    f"border-color:{color}44;"
                ).classes("al-node")

                json_panel = ui.element("div").style("display:none;margin-top:12px;")

                with node:
                    ui.html(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                      <span style="font-size:11px;font-weight:600;color:{color};text-transform:uppercase;
                                   letter-spacing:1px;">{s.agent}</span>
                      <span class="al-mono" style="font-size:10px;color:{TEXT_MUTED};">#{s.step}</span>
                    </div>
                    <div style="font-size:20px;font-weight:700;color:{TEXT};margin-bottom:4px;">
                      {fmt_ms(s.latency_ms)}
                    </div>
                    <div style="font-size:11px;color:{TEXT_MUTED};margin-bottom:10px;">
                      {s.tokens_total:,} tokens
                    </div>
                    """)
                    bar_el = ui.html(bar_html(s.latency_ms, max_lat, color))
                    ui.html(f"""
                    <div style="font-size:10px;color:{TEXT_MUTED};margin-top:6px;display:flex;gap:8px;">
                      <span>↑ {s.tokens_prompt:,}</span>
                      <span>↓ {s.tokens_completion:,}</span>
                    </div>
                    <div style="font-size:10px;color:{TEXT_DIM};margin-top:8px;text-align:center;">
                      click to expand
                    </div>
                    """)

                    # JSON tabs
                    with json_panel:
                        tabs_data = {}
                        handoff = t_data.get("handoff", {})
                        if isinstance(handoff, str):
                            try: handoff = json.loads(handoff)
                            except: handoff = {}

                        tabs_data["Input"]    = handoff.get("input_state", {})
                        tabs_data["Filtered"] = handoff.get("filtered_state", {})
                        tabs_data["Output"]   = handoff.get("output_state", {})

                        for tab_name, tab_data in tabs_data.items():
                            ui.html(f'<div style="font-size:10px;color:{TEXT_MUTED};margin:8px 0 4px;font-weight:600;">{tab_name}</div>')
                            content = json.dumps(tab_data, indent=2, default=str)[:800]
                            ui.html(f'<div class="al-json">{content}</div>')

                def make_toggle(n=node, jp=json_panel):
                    def toggle():
                        ui.run_javascript(
                            f"var jp = document.getElementById('{jp.id}'); "
                            f"jp.style.display = jp.style.display === 'none' ? 'block' : 'none';"
                        )
                    n.on("click", toggle)

                make_toggle()

                if i < len(steps_db) - 1:
                    ui.html(f'<div style="font-size:22px;color:{TEXT_DIM};padding:20px 8px;flex-shrink:0;">→</div>')

        # ── Summary card ──────────────────────────────────────────────────────
        ui.html(f'<div class="al-section" style="margin:24px 0 12px;">Run Summary</div>')
        total_lat = sum(s.latency_ms for s in steps_db)
        total_tok = sum(s.tokens_total for s in steps_db)
        with ui.element("div").style(
            f"background:{CARD};border:1px solid {BORDER};border-radius:10px;"
            f"padding:16px 20px;display:flex;gap:32px;flex-wrap:wrap;"
        ):
            for label, val, color in [
                ("Total Latency", fmt_ms(total_lat), CYAN),
                ("Total Tokens",  f"{total_tok:,}",  AMBER),
                ("Step Count",    str(len(steps_db)), PURPLE),
                ("Est. Cost",     f"${total_tok*0.000005:.4f}", GREEN),
            ]:
                ui.html(f"""
                <div>
                  <div class="al-section" style="margin-bottom:5px;">{label}</div>
                  <div style="font-size:18px;font-weight:700;color:{color};">{val}</div>
                </div>
                """)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Evidence View  /run/{run_id}/evidence
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/run/{run_id}/evidence")
def evidence_page(run_id: str):
    inject_css()
    header()
    nav_bar("evidence", run_id)

    with ui.element("div").classes("al-content"):
        ui.html(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
          <a href="/" style="color:{TEXT_MUTED};text-decoration:none;font-size:12px;">Runs</a>
          <span style="color:{TEXT_DIM};">›</span>
          <a href="/run/{run_id}" style="color:{TEXT_MUTED};text-decoration:none;font-size:12px;">{run_id[:16]}…</a>
          <span style="color:{TEXT_DIM};">›</span>
          <span style="font-size:12px;color:{TEXT};">Evidence</span>
        </div>
        """)

        content_area = ui.element("div")
        spinner_area = ui.element("div")

        async def load():
            with spinner_area:
                with ui.element("div").style(f"display:flex;align-items:center;gap:10px;color:{TEXT_MUTED};padding:16px 0;"):
                    ui.spinner(size="sm").style(f"color:{PURPLE};")
                    ui.label("Loading evidence (Days 9–12 pipeline)…")

            result = await run.io_bound(state.run_full_analysis, run_id)
            spinner_area.clear()

            with content_area:
                if result.error or not result.bundle:
                    ui.html(f'<div style="color:{RED};">{result.error or "Analysis failed"}</div>')
                    return

                bundle = result.bundle
                loss   = result.loss_result

                # ── Section: Rule Matches ─────────────────────────────────────
                ui.html(f'<div class="al-section" style="margin-bottom:12px;">Rule Matches ({len(bundle.rule_matches)})</div>')
                if bundle.rule_matches:
                    with ui.element("div").style(
                        f"background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:20px;"
                    ):
                        for rm in bundle.rule_matches:
                            sev_col  = {"HIGH": RED, "MEDIUM": AMBER, "LOW": CYAN}.get(rm.severity.value.upper(), GRAY)
                            p_badge  = priority_badge("P2")
                            r_badge  = rule_badge(rm.rule_id)
                            ag_color = STEP_COLOR.get(rm.agent or "", GRAY)
                            ui.html(f"""
                            <div style="padding:14px 20px;border-bottom:1px solid {BORDER};">
                              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
                                {r_badge}
                                {p_badge}
                                {badge(rm.severity.value, sev_col)}
                                <span style="font-size:12px;color:{ag_color};margin-left:4px;">
                                  agent: {rm.agent or 'unknown'}
                                </span>
                                <span style="margin-left:auto;font-size:11px;color:{TEXT_DIM};">
                                  {rm.category.value}
                                </span>
                              </div>
                              <div style="font-size:13px;color:{TEXT_MUTED};line-height:1.5;">
                                {rm.description[:220]}
                              </div>
                            </div>
                            """)
                else:
                    ui.html(f'<div style="color:{TEXT_MUTED};padding:12px 0;">No rule matches.</div>')

                # ── Section: Extracted Evidence ────────────────────────────────
                ui.html(f'<div class="al-section" style="margin-bottom:12px;margin-top:4px;">Extracted Facts ({len(result.extracted)} agents)</div>')
                with ui.element("div").style(
                    f"background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:20px;"
                ):
                    for agent, ev in result.extracted.items():
                        col = STEP_COLOR.get(agent, GRAY)
                        ui.html(f"""
                        <div style="padding:14px 20px;border-bottom:1px solid {BORDER};
                                    display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                          <div style="min-width:100px;">
                            <span style="font-size:12px;font-weight:600;color:{col};text-transform:capitalize;">{agent}</span>
                          </div>
                          <div style="display:flex;gap:24px;flex:1;">
                            <div>
                              <div class="al-section" style="margin-bottom:4px;">Sources</div>
                              <span style="font-size:18px;font-weight:700;color:{TEXT};">{ev.source_count}</span>
                            </div>
                            <div>
                              <div class="al-section" style="margin-bottom:4px;">Entities</div>
                              <span style="font-size:18px;font-weight:700;color:{TEXT};">{ev.entity_count}</span>
                            </div>
                            <div>
                              <div class="al-section" style="margin-bottom:4px;">Tool Calls</div>
                              <span style="font-size:18px;font-weight:700;color:{TEXT};">{len(ev.tool_calls)}</span>
                            </div>
                          </div>
                          <button class="al-copy" onclick="alert('Explain feature coming in Day 18')">
                            Explain this →
                          </button>
                        </div>
                        """)

                # ── Section: Information Loss Detail ──────────────────────────
                if loss:
                    ui.html(f'<div class="al-section" style="margin-bottom:12px;">Information Loss Delta (Researcher → Writer)</div>')
                    v_col = VERDICT_COLOR.get(loss.verdict, GRAY)
                    with ui.element("div").style(
                        f"background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px 20px;margin-bottom:20px;"
                    ):
                        ui.html(f"""
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                          {verdict_badge(loss.verdict, False)}
                          <span style="font-size:13px;color:{TEXT_MUTED};">
                            confidence <strong style="color:{v_col};">{loss.confidence:.0%}</strong>
                          </span>
                          {rule_badge("information_loss_v1")}
                        </div>
                        """)
                        for diff in [loss.source_diff, loss.entity_diff]:
                            arrow_col = RED if diff.signal == "DROPPED" else AMBER if diff.signal == "ADDED" else GREEN
                            arrow     = "↓" if diff.signal == "DROPPED" else "↑" if diff.signal == "ADDED" else "→"
                            ui.html(f"""
                            <div style="display:flex;justify-content:space-between;align-items:center;
                                        padding:10px 0;border-bottom:1px solid {BORDER};font-size:13px;">
                              <span style="color:{TEXT_MUTED};min-width:120px;">{diff.field_name}</span>
                              <span style="font-family:monospace;">
                                {diff.researcher_value}
                                <span style="color:{arrow_col};font-weight:700;padding:0 8px;">{arrow}</span>
                                {diff.writer_value}
                              </span>
                              <span style="font-size:11px;color:{arrow_col};">
                                {diff.signal} · severity={diff.severity}
                              </span>
                            </div>
                            """)

        ui.timer(0.1, load, once=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Diff Viewer  /diff
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/diff")
def diff_page():
    inject_css()
    header()
    nav_bar("diff")

    with ui.element("div").classes("al-content"):
        ui.html(f'<div class="al-section" style="margin-bottom:16px;">Diff Viewer — compare two runs</div>')

        runs    = state.list_runs(limit=50)
        run_ids = [r.run_id for r in runs]
        if len(run_ids) < 2:
            ui.html(f'<div style="color:{TEXT_MUTED};">Need at least 2 runs. Run the pipeline a few more times.</div>')
            return

        sel_a = ui.select(run_ids, label="Run A", value=run_ids[0]).style(
            f"background:{CARD};color:{TEXT};min-width:240px;"
        )
        ui.html('<div style="height:12px;"></div>')
        sel_b = ui.select(run_ids, label="Run B", value=run_ids[1] if len(run_ids) > 1 else run_ids[0]).style(
            f"background:{CARD};color:{TEXT};min-width:240px;"
        )

        result_area = ui.element("div").style("margin-top:20px;")

        async def compute():
            result_area.clear()
            with result_area:
                with ui.element("div").style(f"display:flex;gap:10px;align-items:center;color:{TEXT_MUTED};"):
                    ui.spinner(size="sm").style(f"color:{PURPLE};")
                    ui.label("Computing diff…")

            diff = await run.io_bound(state.compute_diff, sel_a.value, sel_b.value)
            result_area.clear()

            with result_area:
                # Overall similarity
                sim_pct = diff.overall_similarity * 100
                sim_col = GREEN if sim_pct > 80 else AMBER if sim_pct > 50 else RED
                ui.html(f"""
                <div style="display:flex;gap:20px;margin-bottom:20px;flex-wrap:wrap;">
                  <div class="al-stat">
                    <div class="al-stat-label">Overall Similarity</div>
                    <div class="al-stat-value" style="color:{sim_col};">{sim_pct:.0f}%</div>
                  </div>
                  <div class="al-stat">
                    <div class="al-stat-label">First Divergence</div>
                    <div class="al-stat-value" style="color:{RED};font-size:18px;">{diff.first_divergence}</div>
                  </div>
                </div>
                """)

                # Per-agent comparison
                DCOLS = "100px 1fr 1fr 80px"
                with ui.element("div").style(
                    f"background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;"
                ):
                    ui.html(f"""
                    <div style="display:grid;grid-template-columns:{DCOLS};
                                padding:9px 20px;border-bottom:1px solid {BORDER};
                                font-size:10px;font-weight:600;letter-spacing:1px;
                                text-transform:uppercase;color:{TEXT_MUTED};">
                      <span>Agent</span>
                      <span>Run A (latency / tokens)</span>
                      <span>Run B (latency / tokens)</span>
                      <span>Similarity</span>
                    </div>
                    """)
                    for row in diff.steps:
                        ag    = row["agent"]
                        col   = STEP_COLOR.get(ag, GRAY)
                        sc    = row["sim"] * 100
                        s_col = GREEN if sc > 80 else AMBER if sc > 50 else RED
                        is_div = (ag == diff.first_divergence)
                        div_bg = f"background:{RED}18;" if is_div else ""
                        ui.html(f"""
                        <div style="display:grid;grid-template-columns:{DCOLS};
                                    padding:12px 20px;border-bottom:1px solid {BORDER};{div_bg}">
                          <span style="font-size:12px;font-weight:600;color:{col};">
                            {ag}{" ← diverge" if is_div else ""}
                          </span>
                          <span style="font-size:12px;color:{TEXT_MUTED};">
                            {fmt_ms(row['lat_a'])} / {row['tok_a']:,}
                          </span>
                          <span style="font-size:12px;color:{TEXT_MUTED};">
                            {fmt_ms(row['lat_b'])} / {row['tok_b']:,}
                          </span>
                          <span style="font-size:13px;font-weight:600;color:{s_col};">
                            {sc:.0f}%
                          </span>
                        </div>
                        """)

        ui.html('<div style="height:16px;"></div>')
        ui.button("Compute Diff", on_click=compute).style(
            f"background:{PURPLE};color:#fff;border-radius:7px;padding:9px 20px;font-weight:600;"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Metrics  /metrics
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/metrics")
def metrics_page():
    inject_css()
    header()
    nav_bar("metrics")

    with ui.element("div").classes("al-content"):
        ui.html(f'<div class="al-section" style="margin-bottom:16px;">Aggregate Metrics — no LLM, pure DB reads</div>')

        data = state.get_metrics_data()
        if not data:
            ui.html(f'<div style="color:{TEXT_MUTED};">No metrics yet.</div>')
            return

        agents = list(data.keys())
        colors = [STEP_COLOR.get(ag, GRAY) for ag in agents]

        # ── Stat cards ────────────────────────────────────────────────────────
        with ui.element("div").style("display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap;"):
            for ag in agents:
                d   = data[ag]
                col = STEP_COLOR.get(ag, GRAY)
                ui.html(f"""
                <div class="al-stat">
                  <div class="al-stat-label" style="color:{col};">{ag}</div>
                  <div class="al-stat-value" style="color:{col};">{fmt_ms(d['avg_latency_ms'])}</div>
                  <div class="al-stat-sub">avg latency · {d['run_count']} runs</div>
                </div>
                """)

        # ── Latency bar chart ─────────────────────────────────────────────────
        ui.html(f'<div class="al-section" style="margin-bottom:12px;">Avg Latency per Agent</div>')
        lat_chart = ui.echart({
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis", "backgroundColor": CARD, "borderColor": BORDER, "textStyle": {"color": TEXT}},
            "xAxis": {"type": "category", "data": agents, "axisLabel": {"color": TEXT_MUTED},
                      "axisLine": {"lineStyle": {"color": BORDER}}},
            "yAxis": {"type": "value", "name": "ms", "nameTextStyle": {"color": TEXT_MUTED},
                      "axisLabel": {"color": TEXT_MUTED}, "splitLine": {"lineStyle": {"color": BORDER}}},
            "series": [{
                "type": "bar",
                "data": [{"value": round(data[ag]["avg_latency_ms"]), "itemStyle": {"color": STEP_COLOR.get(ag, GRAY)}} for ag in agents],
                "barMaxWidth": 60,
                "label": {"show": True, "position": "top", "color": TEXT_MUTED, "fontSize": 11,
                          "formatter": "{c}ms"},
            }],
        }).style(f"height:260px;background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px;margin-bottom:16px;")

        # ── Token usage chart ─────────────────────────────────────────────────
        ui.html(f'<div class="al-section" style="margin-bottom:12px;">Total Tokens per Agent</div>')
        ui.echart({
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis", "backgroundColor": CARD, "borderColor": BORDER, "textStyle": {"color": TEXT}},
            "xAxis": {"type": "category", "data": agents, "axisLabel": {"color": TEXT_MUTED},
                      "axisLine": {"lineStyle": {"color": BORDER}}},
            "yAxis": {"type": "value", "name": "tokens", "nameTextStyle": {"color": TEXT_MUTED},
                      "axisLabel": {"color": TEXT_MUTED}, "splitLine": {"lineStyle": {"color": BORDER}}},
            "series": [{
                "type": "bar",
                "data": [{"value": data[ag]["total_tokens"], "itemStyle": {"color": PURPLE + "cc"}} for ag in agents],
                "barMaxWidth": 60,
                "label": {"show": True, "position": "top", "color": TEXT_MUTED, "fontSize": 11,
                          "formatter": "{c}"},
            }],
        }).style(f"height:260px;background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px;")


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Explanation  /run/{run_id}/explain
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/run/{run_id}/explain")
def explain_page(run_id: str):
    inject_css()
    header()
    nav_bar("explain", run_id)

    with ui.element("div").classes("al-content"):
        ui.html(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
          <a href="/" style="color:{TEXT_MUTED};text-decoration:none;font-size:12px;">Runs</a>
          <span style="color:{TEXT_DIM};">›</span>
          <a href="/run/{run_id}" style="color:{TEXT_MUTED};text-decoration:none;font-size:12px;">{run_id[:16]}…</a>
          <span style="color:{TEXT_DIM};">›</span>
          <span style="font-size:12px;color:{TEXT};">Explanation</span>
        </div>
        """)

        # Pipeline flow banner
        days = [("Day 9","Evidence",CYAN), ("Day 10","Info Loss",AMBER),
                ("Day 12","Arbiter",PURPLE), ("Day 13","Explain",GREEN)]
        with ui.element("div").style("display:flex;align-items:center;gap:0;margin-bottom:24px;flex-wrap:wrap;"):
            for i, (day, lbl, col) in enumerate(days):
                ui.html(f"""
                <div style="background:{col}18;border:1px solid {col}44;border-radius:8px;
                            padding:9px 16px;text-align:center;min-width:100px;">
                  <div style="font-size:9px;font-weight:600;color:{col};letter-spacing:1px;">{day}</div>
                  <div style="font-size:12px;font-weight:500;color:{TEXT};margin-top:2px;">{lbl}</div>
                </div>
                """)
                if i < len(days) - 1:
                    ui.html(f'<span style="color:{TEXT_DIM};font-size:16px;padding:0 6px;">→</span>')

        content = ui.element("div")
        spinner = ui.element("div")

        async def load():
            with spinner:
                with ui.element("div").style(
                    f"background:{CARD};border:1px solid {BORDER};border-radius:10px;"
                    f"padding:48px;display:flex;flex-direction:column;align-items:center;gap:12px;"
                ):
                    ui.spinner(size="lg").style(f"color:{PURPLE};")
                    ui.html(f'<div style="color:{TEXT_MUTED};font-size:14px;">Running full analysis pipeline…</div>')
                    ui.html(f'<div style="color:{TEXT_DIM};font-size:12px;">Days 9 → 12 (LLM evidence extraction)</div>')

            analysis = await run.io_bound(state.run_full_analysis, run_id)
            spinner.clear()

            if analysis.error or not analysis.bundle:
                with content:
                    ui.html(f'<div style="color:{RED};">Error: {analysis.error}</div>')
                return

            bundle = await run.io_bound(state.run_explanation, analysis.bundle)

            with content:
                _render_explanation(bundle, analysis)

        ui.timer(0.1, load, once=True)


def _render_explanation(bundle, analysis):
    p_col = PRIORITY_COLOR.get(bundle.priority_level.value, GRAY)
    c_col = CAUSE_COLOR.get(bundle.primary_cause.value, GRAY)
    verdict = (analysis.loss_result.verdict if analysis.loss_result else "UNKNOWN")
    conf    = f"{analysis.loss_result.confidence:.0%}" if analysis.loss_result else "—"

    is_grounded = bundle.grounded
    hedge_note  = "heuristic — hedged language" if not is_grounded else "grounded — confident language"

    # Copy verdict markdown
    md_verdict = (
        f"**Run:** {bundle.run_id}\\n"
        f"**Verdict:** {verdict} ({bundle.priority_level.value})\\n"
        f"**Cause:** {bundle.primary_cause.value}\\n"
        f"**Agent:** {bundle.primary_agent or 'N/A'}\\n"
        f"**Confidence:** {conf}\\n"
        f"**Grounded:** {'Yes' if is_grounded else 'No'}\\n"
        f"**Summary:** {(bundle.summary or '').replace(chr(10), ' ')}\\n"
        f"**Fix:** {(bundle.suggested_fix or '').replace(chr(10), ' ')}"
    )

    # ── Verdict row ───────────────────────────────────────────────────────────
    with ui.element("div").style(
        f"background:{CARD};border:1px solid {BORDER};border-radius:10px;"
        f"padding:16px 20px;margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;"
    ):
        for label, content_html in [
            ("Priority",      priority_badge(bundle.priority_level.value)),
            ("Cause",         cause_badge(bundle.primary_cause.value)),
            ("Agent",         f'<span style="color:{STEP_COLOR.get(bundle.primary_agent or "", GRAY)};font-weight:600;">{bundle.primary_agent or "N/A"}</span>'),
            ("Confidence",    f'<span style="font-size:15px;font-weight:700;">{conf}</span>'),
            ("Grounded",      f'<span style="color:{"#22c55e" if is_grounded else TEXT_MUTED};">{"✓ Yes" if is_grounded else "✗ No"}</span>'),
        ]:
            with ui.element("div").style("flex:1;min-width:100px;"):
                ui.html(f'<div class="al-section" style="margin-bottom:6px;">{label}</div>')
                ui.html(content_html)

    # ── LLM Explanation card ──────────────────────────────────────────────────
    with ui.element("div").style(
        f"background:linear-gradient(135deg,{PURPLE}18,{CYAN}0a);"
        f"border:1px solid {PURPLE}44;border-radius:10px;padding:20px 22px;margin-bottom:16px;"
    ):
        with ui.element("div").style("display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;"):
            ui.html(f"""
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:14px;">✦</span>
              <span class="al-section" style="color:{PURPLE};">LLM Explanation</span>
              <span style="font-size:10px;color:{TEXT_DIM};padding:2px 6px;border:1px solid {BORDER};border-radius:4px;">
                {hedge_note}
              </span>
            </div>
            """)
            ui.html(f"""
            <button class="al-copy" onclick="navigator.clipboard.writeText(`{md_verdict}`).then(()=>this.textContent='✓ Copied!').catch(()=>this.textContent='Error')">
              📋 Copy verdict as markdown
            </button>
            """)

        if bundle.summary:
            ui.html(f"""
            <div style="margin-bottom:16px;">
              <div class="al-section" style="margin-bottom:8px;">Root Cause</div>
              <div style="font-size:15px;line-height:1.75;color:{TEXT};">{bundle.summary}</div>
            </div>
            """)

        if bundle.suggested_fix:
            ui.html(f"""
            <div style="background:{BG_SIDEBAR};border-left:3px solid {GREEN};border-radius:6px;
                        padding:12px 16px;">
              <div class="al-section" style="color:{GREEN};margin-bottom:6px;">Suggested Fix</div>
              <div style="font-size:14px;line-height:1.65;color:{TEXT};">{bundle.suggested_fix}</div>
            </div>
            """)

    # ── Fired rules ───────────────────────────────────────────────────────────
    if bundle.rule_matches:
        with ui.element("div").style(
            f"background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px 20px;margin-bottom:16px;"
        ):
            ui.html(f'<div class="al-section" style="margin-bottom:12px;">Fired Rules ({len(bundle.rule_matches)})</div>')
            for rm in bundle.rule_matches:
                sev_col = {"HIGH": RED, "MEDIUM": AMBER, "LOW": CYAN}.get(rm.severity.value.upper(), GRAY)
                ui.html(f"""
                <div style="padding:10px 0;border-bottom:1px solid {BORDER};">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;">
                    {rule_badge(rm.rule_id)}
                    {badge(rm.severity.value, sev_col)}
                    <span style="font-size:11px;color:{TEXT_MUTED};margin-left:auto;">agent: {rm.agent or 'unknown'}</span>
                  </div>
                  <div style="font-size:12px;color:{TEXT_MUTED};">{rm.description[:200]}</div>
                </div>
                """)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="AgentLens",
        host="127.0.0.1",
        port=8080,
        reload=False,
        dark=True,
        favicon="🔬",
    )
