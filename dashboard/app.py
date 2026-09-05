"""
dashboard/app.py — AgentLens Dashboard v2
==========================================
6 views, matching the Run Explorer screenshot design.

Run: python -m dashboard.app
"""

from __future__ import annotations

import json

from nicegui import app, run, ui

import dashboard.state as state
from dashboard.theme import (
    AMBER,
    BG,
    BG_SIDEBAR,
    BORDER,
    CARD,
    CAUSE_COLOR,
    CYAN,
    GLOBAL_CSS,
    GRAY,
    GREEN,
    PRIORITY_COLOR,
    PURPLE,
    RED,
    STEP_COLOR,
    TEXT,
    TEXT_DIM,
    TEXT_MUTED,
    VERDICT_COLOR,
    badge,
    bar_html,
    cause_badge,
    fmt_ms,
    priority_badge,
    row_bg,
    rule_badge,
    verdict_badge,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared layout helpers
# ─────────────────────────────────────────────────────────────────────────────


def inject_css():
    ui.add_head_html(GLOBAL_CSS)
    ui.query("body").style(f"background:{BG};color:{TEXT};")


def _logo_html() -> str:
    return """
    <img src="/assets/logo.png"
         style="width:30px;height:30px;border-radius:7px;object-fit:cover;flex-shrink:0;"
         alt="AgentLens logo"
         onerror="this.style.display='none';this.nextElementSibling.style.display='flex'" />
    <div style="display:none;width:30px;height:30px;
                background:linear-gradient(135deg,#8b5cf6,#06b6d4);
                border-radius:7px;align-items:center;justify-content:center;
                font-weight:700;font-size:12px;color:#fff;flex-shrink:0;">AL</div>
    <span style="font-size:15px;font-weight:700;color:#e2e8f0;">AgentLens</span>
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
        ("🔍", "Runs", "runs", "/"),
        ("📈", "Metrics", "metrics", "/metrics"),
        ("⚖️", "Diff", "diff", "/diff"),
    ]
    run_tabs = (
        [
            ("⏱", "Timeline", "timeline", f"/run/{run_id}"),
            ("🔬", "Evidence", "evidence", f"/run/{run_id}/evidence"),
            ("✦", "Explain", "explain", f"/run/{run_id}/explain"),
        ]
        if run_id
        else []
    )

    tabs_html = ""
    for icon, label, key, href in global_tabs:
        cls = "al-nav-tab active" if active == key else "al-nav-tab"
        tabs_html += f'<a href="{href}" class="{cls}">{icon} {label}</a>'

    if run_tabs:
        tabs_html += '<div class="al-nav-divider"></div>'
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
        with ui.element("div").style(
            "display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;"
        ):
            ui.html('<div class="al-section">Run Explorer</div>')
            # (filters are cosmetic for MVP; full filtering = Day 22)
            ui.html("""
            <div style="display:flex;gap:8px;">
              <select class="al-select"><option>All agents</option><option>researcher</option><option>writer</option><option>verifier</option></select>
              <select class="al-select"><option>All time</option><option>Last 7 days</option><option>Last 30 days</option></select>
            </div>
            """)

        # ── Stat cards ────────────────────────────────────────────────────────
        analyzed = [r for r in runs if state.get_cached(r.run_id)]
        warnings = sum(
            1
            for r in runs
            if state.get_cached(r.run_id)
            and state.get_cached(r.run_id).loss_result
            and state.get_cached(r.run_id).loss_result.verdict != "PASS"
        )
        avg_lat = (sum(r.latency_ms for r in runs) / len(runs)) if runs else 0
        total_tok = sum(r.tokens_total for r in runs)

        with ui.element("div").style("display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;"):
            for label, val, color, sub in [
                ("Total Runs", str(len(runs)), PURPLE, f"{len(analyzed)} analyzed"),
                ("Avg Latency", fmt_ms(avg_lat), CYAN, "per pipeline run"),
                ("Total Tokens", f"{total_tok:,}", AMBER, f"~${total_tok * 0.000005:.3f} est."),
                ("Warnings", str(warnings), RED if warnings else GREEN, "from analyzed runs"),
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
    loss = cached.loss_result if cached else None

    # Determine tint from verdict
    verdict = (
        loss.verdict
        if loss
        else ("PASS" if bundle and bundle.priority_level.value == "P5" else "UNKNOWN")
    )
    bg = row_bg(verdict)

    # Primary cause display
    if bundle:
        cause_str = bundle.primary_cause.value
        agent_str = bundle.primary_agent or ""
        cause_disp = (
            f"{cause_str.capitalize()} {'in handoff' if 'workflow' in cause_str else ''}"
            f'<br><span style="font-size:11px;color:{TEXT_MUTED};font-family:monospace;">({agent_str})</span>'
        )
    else:
        cause_disp = f'<span style="color:{TEXT_DIM};">—</span>'

    verdict_disp = (
        verdict_badge(verdict, bundle.grounded if bundle else False)
        if bundle
        else f'<span style="color:{TEXT_DIM};font-size:12px;">Unanalyzed</span>'
    )

    # Row container
    row_el = (
        ui.element("div").style(f"background:{bg};grid-template-columns:{cols};").classes("al-trow")
    )

    with row_el:
        with ui.element("div").classes("al-tcell"):
            ui.html(f"""
            <div>
              <div style="font-size:13px;font-weight:500;">{r.topic or r.workflow}</div>
              <div class="al-mono" style="font-size:10px;color:{TEXT_MUTED};margin-top:3px;">{r.run_id}</div>
            </div>
            """)
        # Keep references so we can update after on-demand analysis
        with ui.element("div").classes("al-tcell"):
            cause_el = ui.html(f'<div style="font-size:13px;line-height:1.5;">{cause_disp}</div>')
        with ui.element("div").classes("al-tcell"):
            verdict_el = ui.html(verdict_disp)
        with ui.element("div").classes("al-tcell"):
            ui.html(f'<span style="font-size:13px;">{fmt_ms(r.latency_ms)}</span>')

    # Inline expansion panel (hidden by default)
    expansion = ui.element("div").classes("al-expansion")
    expansion.set_visibility(False)

    with expansion:
        if bundle:
            _inline_verdict_panel(bundle, loss, r.run_id)
        else:
            _inline_analyze_panel(r.run_id, expansion, row_el, cols, cause_el, verdict_el)

    # Click row → toggle expansion
    row_el.on("click", lambda e, exp=expansion: exp.set_visibility(not exp.visible))


def _toggle(el):
    """Toggle display:none on a NiceGUI element."""
    # Use client-side JS for instant response
    ui.run_javascript(
        f"var el = document.getElementById('{el.id}'); el.style.display = el.style.display === 'none' ? 'block' : 'none';"
    )


def _inline_verdict_panel(bundle, loss, run_id: str):
    conf = f"{loss.confidence:.0%}" if loss else "—"
    verdict_str = loss.verdict if loss else "UNKNOWN"

    # Build markdown in Python so it can be passed safely via json.dumps
    import json as _j

    md = (
        f"## AgentLens Verdict\n"
        f"**Run:** `{run_id}`\n"
        f"**Verdict:** {verdict_str} ({bundle.priority_level.value})\n"
        f"**Cause:** {bundle.primary_cause.value}\n"
        f"**Agent:** {bundle.primary_agent or 'N/A'}\n"
        f"**Confidence:** {conf}\n"
        f"**Grounded:** {'Yes' if bundle.grounded else 'No'}"
    )

    with ui.element("div").style("display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;"):
        for label, html_val in [
            ("Priority", priority_badge(bundle.priority_level.value)),
            ("Cause", cause_badge(bundle.primary_cause.value)),
        ]:
            with ui.element("div"):
                ui.html(f'<div class="al-section" style="margin-bottom:6px;">{label}</div>')
                ui.html(html_val)

        ag_color = STEP_COLOR.get(bundle.primary_agent or "", GRAY)
        grnd_color = "#22c55e" if bundle.grounded else TEXT_MUTED
        grnd_txt = "\u2713 Yes" if bundle.grounded else "\u2717 No (heuristic)"

        with ui.element("div"):
            ui.html('<div class="al-section" style="margin-bottom:6px;">Agent</div>')
            ui.html(
                f'<span style="font-size:13px;font-weight:500;color:{ag_color};">{bundle.primary_agent or "N/A"}</span>'
            )

        with ui.element("div"):
            ui.html('<div class="al-section" style="margin-bottom:6px;">Confidence</div>')
            ui.html(f'<span style="font-size:13px;font-weight:600;">{conf}</span>')

        with ui.element("div"):
            ui.html('<div class="al-section" style="margin-bottom:6px;">Grounded</div>')
            ui.html(f'<span style="color:{grnd_color};">{grnd_txt}</span>')

        # Action buttons
        with ui.element("div").style("margin-left:auto;display:flex;gap:8px;align-items:center;"):

            async def _copy(text=md):
                await ui.run_javascript(f"navigator.clipboard.writeText({_j.dumps(text)})")
                ui.notify("\u2713 Copied!", type="positive", position="top", timeout=1500)

            ui.button("\U0001f4cb Copy verdict", on_click=_copy).props("flat dense").style(
                f"color:{TEXT_MUTED};font-size:11px;border:1px solid {BORDER};"
                "border-radius:6px;padding:4px 10px;"
            )
            ui.html(
                f'<a href="/run/{run_id}" style="font-size:12px;color:{PURPLE};text-decoration:none;'
                f'padding:4px 10px;border:1px solid {PURPLE}44;border-radius:6px;">Open trace \u2192</a>'
            )
            ui.html(
                f'<a href="/run/{run_id}/explain" style="font-size:12px;color:{CYAN};text-decoration:none;'
                f'padding:4px 10px;border:1px solid {CYAN}44;border-radius:6px;">\u2726 Explain</a>'
            )


def _inline_analyze_panel(
    run_id: str, expansion, row_el, cols: str, cause_el=None, verdict_el=None
):
    content_area = ui.element("div")

    async def analyze():
        content_area.clear()
        with content_area:
            with ui.element("div").style(
                f"display:flex;align-items:center;gap:10px;color:{TEXT_MUTED};"
            ):
                ui.spinner(size="xs").style(f"color:{PURPLE};")
                ui.label("Analyzing… (LLM calls, ~8-10s)")

        result = await run.io_bound(state.run_full_analysis, run_id)
        content_area.clear()

        with content_area:
            if result.error or not result.bundle:
                ui.html(f'<span style="color:{RED};">{result.error or "Analysis failed"}</span>')
            else:
                _inline_verdict_panel(result.bundle, result.loss_result, run_id)

                # ── Update the row cells so the table reflects the verdict ──
                bundle = result.bundle
                loss = result.loss_result
                verdict = (
                    loss.verdict
                    if loss
                    else ("PASS" if bundle.priority_level.value == "P5" else "UNKNOWN")
                )

                # Update cause cell
                if cause_el is not None:
                    cause_str = bundle.primary_cause.value
                    agent_str = bundle.primary_agent or ""
                    new_cause = (
                        f"{cause_str.capitalize()} {'in handoff' if 'workflow' in cause_str else ''}"
                        f'<br><span style="font-size:11px;color:{TEXT_MUTED};font-family:monospace;">({agent_str})</span>'
                    )
                    cause_el.set_content(
                        f'<div style="font-size:13px;line-height:1.5;">{new_cause}</div>'
                    )

                # Update verdict cell
                if verdict_el is not None:
                    verdict_el.set_content(verdict_badge(verdict, bundle.grounded))

                # Update row background tint
                new_bg = row_bg(verdict)
                row_el.style(f"background:{new_bg};grid-template-columns:{cols};")

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

        steps_db = state.get_steps(run_id)
        steps_trace = state.get_trace_steps(run_id)
        trace_by_agent = {s.get("agent", ""): s for s in steps_trace}

        if not steps_db:
            ui.html(f'<div style="color:{TEXT_MUTED};">No steps found.</div>')
            return

        max_lat = max((s.latency_ms for s in steps_db), default=1)
        _ = max((s.tokens_total for s in steps_db), default=1)  # reserved for token chart

        ui.html(
            '<div class="al-section" style="margin-bottom:16px;">Step-by-step pipeline execution</div>'
        )

        # ── Node flow ─────────────────────────────────────────────────────────
        with ui.element("div").style(
            "display:flex;align-items:flex-start;gap:0;overflow-x:auto;padding-bottom:8px;"
        ):
            for i, s in enumerate(steps_db):
                color = STEP_COLOR.get(s.agent, GRAY)
                t_data = trace_by_agent.get(s.agent, {})

                node = ui.element("div").style(f"border-color:{color}44;").classes("al-node")

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
                    ui.html(bar_html(s.latency_ms, max_lat, color))
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
                            try:
                                handoff = json.loads(handoff)
                            except Exception:
                                handoff = {}

                        tabs_data["Input"] = handoff.get("input_state", {})
                        tabs_data["Filtered"] = handoff.get("filtered_state", {})
                        tabs_data["Output"] = handoff.get("output_state", {})

                        for tab_name, tab_data in tabs_data.items():
                            ui.html(
                                f'<div style="font-size:10px;color:{TEXT_MUTED};margin:8px 0 4px;font-weight:600;">{tab_name}</div>'
                            )
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
                    ui.html(
                        f'<div style="font-size:22px;color:{TEXT_DIM};padding:20px 8px;flex-shrink:0;">→</div>'
                    )

        # ── Summary card ──────────────────────────────────────────────────────
        ui.html('<div class="al-section" style="margin:24px 0 12px;">Run Summary</div>')
        total_lat = sum(s.latency_ms for s in steps_db)
        total_tok = sum(s.tokens_total for s in steps_db)
        with ui.element("div").style(
            f"background:{CARD};border:1px solid {BORDER};border-radius:10px;"
            f"padding:16px 20px;display:flex;gap:32px;flex-wrap:wrap;"
        ):
            for label, val, color in [
                ("Total Latency", fmt_ms(total_lat), CYAN),
                ("Total Tokens", f"{total_tok:,}", AMBER),
                ("Step Count", str(len(steps_db)), PURPLE),
                ("Est. Cost", f"${total_tok * 0.000005:.4f}", GREEN),
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
                with ui.element("div").style(
                    f"display:flex;align-items:center;gap:10px;color:{TEXT_MUTED};padding:16px 0;"
                ):
                    ui.spinner(size="sm").style(f"color:{PURPLE};")
                    ui.label("Loading evidence (Days 9–12 pipeline)…")

            result = await run.io_bound(state.run_full_analysis, run_id)
            spinner_area.clear()

            with content_area:
                if result.error or not result.bundle:
                    ui.html(f'<div style="color:{RED};">{result.error or "Analysis failed"}</div>')
                    return

                bundle = result.bundle
                loss = result.loss_result

                # ── Section: Rule Matches ─────────────────────────────────────
                ui.html(
                    f'<div class="al-section" style="margin-bottom:12px;">Rule Matches ({len(bundle.rule_matches)})</div>'
                )
                if bundle.rule_matches:
                    with ui.element("div").style(
                        f"background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:20px;"
                    ):
                        for rm in bundle.rule_matches:
                            sev_col = {"HIGH": RED, "MEDIUM": AMBER, "LOW": CYAN}.get(
                                rm.severity.value.upper(), GRAY
                            )
                            p_badge = priority_badge("P2")
                            r_badge = rule_badge(rm.rule_id)
                            ag_color = STEP_COLOR.get(rm.agent or "", GRAY)
                            ui.html(f"""
                            <div style="padding:14px 20px;border-bottom:1px solid {BORDER};">
                              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
                                {r_badge}
                                {p_badge}
                                {badge(rm.severity.value, sev_col)}
                                <span style="font-size:12px;color:{ag_color};margin-left:4px;">
                                  agent: {rm.agent or "unknown"}
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
                    ui.html(
                        f'<div style="color:{TEXT_MUTED};padding:12px 0;">No rule matches.</div>'
                    )

                # ── Section: Extracted Evidence ────────────────────────────────
                ui.html(
                    f'<div class="al-section" style="margin-bottom:12px;margin-top:4px;">Extracted Facts ({len(result.extracted)} agents)</div>'
                )
                with ui.element("div").style(
                    f"background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:20px;"
                ):
                    for _ag, _ev in result.extracted.items():
                        _ag_col = STEP_COLOR.get(_ag, GRAY)
                        with ui.element("div").style(
                            f"padding:14px 20px;border-bottom:1px solid {BORDER};"
                            "display:flex;align-items:center;gap:16px;flex-wrap:wrap;"
                        ):
                            ui.html(f"""
                            <div style="min-width:100px;">
                              <span style="font-size:12px;font-weight:600;color:{_ag_col};text-transform:capitalize;">{_ag}</span>
                            </div>
                            <div style="display:flex;gap:24px;flex:1;">
                              <div>
                                <div class="al-section" style="margin-bottom:4px;">Sources</div>
                                <span style="font-size:18px;font-weight:700;color:{TEXT};">{_ev.source_count}</span>
                              </div>
                              <div>
                                <div class="al-section" style="margin-bottom:4px;">Entities</div>
                                <span style="font-size:18px;font-weight:700;color:{TEXT};">{_ev.entity_count}</span>
                              </div>
                              <div>
                                <div class="al-section" style="margin-bottom:4px;">Tool Calls</div>
                                <span style="font-size:18px;font-weight:700;color:{TEXT};">{len(_ev.tool_calls)}</span>
                              </div>
                            </div>
                            """)

                            def _mk_btn(ag=_ag, ev=_ev, bndl=result.bundle):
                                async def _click():
                                    ag_col = STEP_COLOR.get(ag, GRAY)
                                    dlg = ui.dialog()
                                    with (
                                        dlg,
                                        ui.card().style(
                                            f"background:{CARD};border:1px solid {BORDER};"
                                            "min-width:520px;max-width:680px;padding:24px;"
                                        ),
                                    ):
                                        ui.html(f"""
                                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
                                          <span style="color:{ag_col};font-size:14px;">●</span>
                                          <span style="font-size:13px;font-weight:600;color:{TEXT};text-transform:capitalize;">{ag}</span>
                                          <span class="al-section" style="margin-left:8px;">Evidence Explanation</span>
                                        </div>
                                        <div style="display:flex;gap:16px;margin-bottom:16px;">
                                          <div class="al-stat" style="flex:1;">
                                            <div class="al-stat-label">Sources</div>
                                            <div class="al-stat-value" style="color:{CYAN};">{ev.source_count}</div>
                                          </div>
                                          <div class="al-stat" style="flex:1;">
                                            <div class="al-stat-label">Entities</div>
                                            <div class="al-stat-value" style="color:{PURPLE};">{ev.entity_count}</div>
                                          </div>
                                          <div class="al-stat" style="flex:1;">
                                            <div class="al-stat-label">Tool Calls</div>
                                            <div class="al-stat-value">{len(ev.tool_calls)}</div>
                                          </div>
                                        </div>
                                        """)
                                        status_lbl = ui.html(
                                            f'<div style="color:{TEXT_MUTED};padding:8px 0;">'
                                            f'<span style="color:{PURPLE};">⟳</span>  Generating explanation…</div>'
                                        )
                                        with ui.row().style(
                                            "justify-content:flex-end;margin-top:12px;"
                                        ):
                                            ui.button("Close", on_click=dlg.close).props(
                                                "flat"
                                            ).style(f"color:{TEXT_MUTED};")
                                    dlg.open()
                                    text = await run.io_bound(
                                        state.explain_agent_evidence, ag, ev, bndl
                                    )
                                    status_lbl.set_content(f"""
                                    <div style="background:{BG_SIDEBAR};border-left:3px solid {STEP_COLOR.get(ag, GRAY)};
                                                border-radius:6px;padding:14px 16px;">
                                      <div class="al-section" style="color:{STEP_COLOR.get(ag, GRAY)};margin-bottom:8px;">LLM Explanation</div>
                                      <div style="font-size:14px;line-height:1.75;color:{TEXT};">{text}</div>
                                    </div>
                                    """)

                                ui.button("Explain this →", on_click=_click).props(
                                    "flat dense"
                                ).style(
                                    f"color:{TEXT_MUTED};font-size:11px;border:1px solid {BORDER};"
                                    "border-radius:6px;padding:4px 12px;white-space:nowrap;"
                                )

                            _mk_btn()
                # ── Section: Information Loss Detail ──────────────────────────
                if loss:
                    ui.html(
                        '<div class="al-section" style="margin-bottom:12px;">Information Loss Delta (Researcher → Writer)</div>'
                    )
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
                            arrow_col = (
                                RED
                                if diff.signal == "DROPPED"
                                else AMBER
                                if diff.signal == "ADDED"
                                else GREEN
                            )
                            arrow = (
                                "↓"
                                if diff.signal == "DROPPED"
                                else "↑"
                                if diff.signal == "ADDED"
                                else "→"
                            )
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
        ui.html(
            f'<div class="al-section" style="margin-bottom:16px;">Diff Viewer'
            f'<span style="font-size:11px;color:{TEXT_MUTED};font-weight:400;margin-left:10px;">'
            f"Graph-aligned · Semantic similarity (all-MiniLM-L6-v2)</span></div>"
        )

        runs = state.list_runs(limit=50)
        run_ids = [r.run_id for r in runs]
        if len(run_ids) < 2:
            ui.html(
                f'<div style="color:{TEXT_MUTED};">Need at least 2 runs. Run the pipeline a few more times.</div>'
            )
            return

        # ── Run selectors ──────────────────────────────────────────────────
        with ui.element("div").style("display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;"):
            sel_a = ui.select(run_ids, label="Run A (baseline)", value=run_ids[0]).style(
                f"background:{CARD};color:{TEXT};min-width:240px;"
            )
            sel_b = ui.select(
                run_ids, label="Run B (comparison)",
                value=run_ids[1] if len(run_ids) > 1 else run_ids[0],
            ).style(f"background:{CARD};color:{TEXT};min-width:240px;")
            ui.button("Compute Diff", on_click=lambda: compute()).style(
                f"background:{PURPLE};color:#fff;border-radius:7px;"
                f"padding:9px 22px;font-weight:600;align-self:flex-end;"
            )

        result_area = ui.element("div").style("margin-top:24px;")

        async def compute():
            result_area.clear()
            with result_area:
                with ui.element("div").style(
                    f"display:flex;gap:10px;align-items:center;color:{TEXT_MUTED};"
                ):
                    ui.spinner(size="sm").style(f"color:{PURPLE};")
                    ui.label("Aligning runs and computing similarity…")

            diff = await run.io_bound(state.compute_diff, sel_a.value, sel_b.value)
            result_area.clear()

            with result_area:
                # ── Summary stat cards ─────────────────────────────────────
                sim_pct = diff.overall_similarity * 100
                sim_col = GREEN if sim_pct > 80 else AMBER if sim_pct > 50 else RED

                first_div = diff.first_divergence
                first_div_col = RED if first_div not in ("(none)", "(trace not found)") else GREEN

                ui.html(f"""
                <div style="display:flex;gap:14px;margin-bottom:22px;flex-wrap:wrap;">
                  <div class="al-stat">
                    <div class="al-stat-label">Overall Similarity</div>
                    <div class="al-stat-value" style="color:{sim_col};">{sim_pct:.1f}%</div>
                  </div>
                  <div class="al-stat">
                    <div class="al-stat-label">First Divergence</div>
                    <div class="al-stat-value" style="color:{first_div_col};font-size:18px;">{first_div}</div>
                  </div>
                  <div class="al-stat">
                    <div class="al-stat-label">Matched Steps</div>
                    <div class="al-stat-value" style="color:{GREEN};">{diff.matched_count}</div>
                  </div>
                  <div class="al-stat">
                    <div class="al-stat-label">Missing in A</div>
                    <div class="al-stat-value" style="color:{AMBER if diff.missing_in_a_count else TEXT_MUTED};">
                      {diff.missing_in_a_count}
                    </div>
                  </div>
                  <div class="al-stat">
                    <div class="al-stat-label">Missing in B</div>
                    <div class="al-stat-value" style="color:{AMBER if diff.missing_in_b_count else TEXT_MUTED};">
                      {diff.missing_in_b_count}
                    </div>
                  </div>
                </div>
                """)

                if not diff.steps:
                    ui.html(f'<div style="color:{TEXT_MUTED};">No steps to compare.</div>')
                    return

                # ── Per-agent comparison table ─────────────────────────────
                DCOLS = "130px 1fr 1fr 90px 80px 80px 100px"
                with ui.element("div").style(
                    f"background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;"
                ):
                    ui.html(f"""
                    <div style="display:grid;grid-template-columns:{DCOLS};
                                padding:9px 20px;border-bottom:1px solid {BORDER};
                                font-size:10px;font-weight:600;letter-spacing:1px;
                                text-transform:uppercase;color:{TEXT_MUTED};">
                      <span>Agent</span>
                      <span>Run A (lat / tok)</span>
                      <span>Run B (lat / tok)</span>
                      <span>Lat Δ</span>
                      <span>Tok Δ</span>
                      <span>Method</span>
                      <span>Similarity</span>
                    </div>
                    """)
                    for row in diff.steps:
                        ag = row["agent"]
                        col = STEP_COLOR.get(ag, GRAY)
                        sc = row["sim"] * 100
                        status = row.get("match_status", "MATCHED")

                        # Row background: divergence highlight, missing step hint
                        is_div = ag == diff.first_divergence
                        if status == "MISSING_IN_B":
                            row_style = f"background:{AMBER}18;"
                            status_badge = f'<span style="font-size:10px;background:{AMBER}33;color:{AMBER};padding:1px 5px;border-radius:4px;margin-left:4px;">−B</span>'
                        elif status == "MISSING_IN_A":
                            row_style = f"background:{CYAN}15;"
                            status_badge = f'<span style="font-size:10px;background:{CYAN}33;color:{CYAN};padding:1px 5px;border-radius:4px;margin-left:4px;">−A</span>'
                        elif is_div:
                            row_style = f"background:{RED}18;"
                            status_badge = f'<span style="font-size:10px;background:{RED}33;color:{RED};padding:1px 5px;border-radius:4px;margin-left:4px;">↑ div</span>'
                        else:
                            row_style = ""
                            status_badge = ""

                        # Similarity color + bar
                        if status != "MATCHED":
                            s_col = TEXT_MUTED
                            sim_bar = f'<span style="color:{TEXT_MUTED};font-size:12px;">—</span>'
                        else:
                            s_col = GREEN if sc > 80 else AMBER if sc > 50 else RED
                            bar_w = max(4, int(sc))
                            sim_bar = (
                                f'<div style="display:flex;align-items:center;gap:6px;">'
                                f'<div style="width:{bar_w}px;height:6px;border-radius:3px;background:{s_col};"></div>'
                                f'<span style="font-size:13px;font-weight:600;color:{s_col};">{sc:.0f}%</span>'
                                f'</div>'
                            )

                        # Delta display helpers
                        lat_delta = row.get("lat_delta", 0)
                        tok_delta = row.get("tok_delta", 0)
                        lat_d_col = RED if lat_delta > 500 else AMBER if lat_delta > 0 else GREEN if lat_delta < 0 else TEXT_MUTED
                        tok_d_col = RED if tok_delta > 200 else AMBER if tok_delta > 0 else GREEN if tok_delta < 0 else TEXT_MUTED
                        lat_d_str = f'+{fmt_ms(lat_delta)}' if lat_delta > 0 else (f'{fmt_ms(lat_delta)}' if lat_delta < 0 else '—')
                        tok_d_str = f'+{int(tok_delta):,}' if tok_delta > 0 else (f'{int(tok_delta):,}' if tok_delta < 0 else '—')

                        method = row.get("method", "—")

                        ui.html(f"""
                        <div style="display:grid;grid-template-columns:{DCOLS};
                                    padding:12px 20px;border-bottom:1px solid {BORDER};{row_style}">
                          <span style="font-size:12px;font-weight:600;color:{col};">
                            {ag}{status_badge}
                          </span>
                          <span style="font-size:12px;color:{TEXT_MUTED};">
                            {fmt_ms(row['lat_a'])} / {int(row['tok_a']):,}
                          </span>
                          <span style="font-size:12px;color:{TEXT_MUTED};">
                            {fmt_ms(row['lat_b'])} / {int(row['tok_b']):,}
                          </span>
                          <span style="font-size:12px;color:{lat_d_col};">{lat_d_str}</span>
                          <span style="font-size:12px;color:{tok_d_col};">{tok_d_str}</span>
                          <span style="font-size:11px;color:{TEXT_MUTED};">{method}</span>
                          {sim_bar}
                        </div>
                        """)

                # ── Legend ─────────────────────────────────────────────────
                ui.html(f"""
                <div style="margin-top:12px;display:flex;gap:18px;flex-wrap:wrap;font-size:11px;color:{TEXT_MUTED};">
                  <span><span style="color:{GREEN};">■</span> &gt;80% similar</span>
                  <span><span style="color:{AMBER};">■</span> 50–80% (drift)</span>
                  <span><span style="color:{RED};">■</span> &lt;50% (diverged)</span>
                  <span><span style="background:{AMBER}18;padding:0 4px;border-radius:3px;">−B</span> missing in Run B</span>
                  <span><span style="background:{CYAN}15;padding:0 4px;border-radius:3px;">−A</span> missing in Run A</span>
                  <span><span style="background:{RED}18;padding:0 4px;border-radius:3px;">↑ div</span> first divergence</span>
                </div>
                """)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Metrics  /metrics
# ─────────────────────────────────────────────────────────────────────────────


@ui.page("/metrics")
def metrics_page():
    inject_css()
    header()
    nav_bar("metrics")

    with ui.element("div").classes("al-content"):
        ui.html(
            '<div class="al-section" style="margin-bottom:16px;">Aggregate Metrics — no LLM, pure DB reads</div>'
        )

        data = state.get_metrics_data()
        if not data:
            ui.html(f'<div style="color:{TEXT_MUTED};">No metrics yet.</div>')
            return

        agents = list(data.keys())
        # colors reserved for future chart coloring: [STEP_COLOR.get(ag, GRAY) for ag in agents]

        # ── Stat cards ────────────────────────────────────────────────────────
        with ui.element("div").style("display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap;"):
            for ag in agents:
                d = data[ag]
                col = STEP_COLOR.get(ag, GRAY)
                ui.html(f"""
                <div class="al-stat">
                  <div class="al-stat-label" style="color:{col};">{ag}</div>
                  <div class="al-stat-value" style="color:{col};">{fmt_ms(d["avg_latency_ms"])}</div>
                  <div class="al-stat-sub">avg latency · {d["run_count"]} runs</div>
                </div>
                """)

        # ── Latency bar chart ─────────────────────────────────────────────────
        ui.html('<div class="al-section" style="margin-bottom:12px;">Avg Latency per Agent</div>')
        ui.echart(
            {
                "backgroundColor": "transparent",
                "tooltip": {
                    "trigger": "axis",
                    "backgroundColor": CARD,
                    "borderColor": BORDER,
                    "textStyle": {"color": TEXT},
                },
                "xAxis": {
                    "type": "category",
                    "data": agents,
                    "axisLabel": {"color": TEXT_MUTED},
                    "axisLine": {"lineStyle": {"color": BORDER}},
                },
                "yAxis": {
                    "type": "value",
                    "name": "ms",
                    "nameTextStyle": {"color": TEXT_MUTED},
                    "axisLabel": {"color": TEXT_MUTED},
                    "splitLine": {"lineStyle": {"color": BORDER}},
                },
                "series": [
                    {
                        "type": "bar",
                        "data": [
                            {
                                "value": round(data[ag]["avg_latency_ms"]),
                                "itemStyle": {"color": STEP_COLOR.get(ag, GRAY)},
                            }
                            for ag in agents
                        ],
                        "barMaxWidth": 60,
                        "label": {
                            "show": True,
                            "position": "top",
                            "color": TEXT_MUTED,
                            "fontSize": 11,
                            "formatter": "{c}ms",
                        },
                    }
                ],
            }
        ).style(
            f"height:260px;background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px;margin-bottom:16px;"
        )

        # ── Token usage chart ─────────────────────────────────────────────────
        ui.html('<div class="al-section" style="margin-bottom:12px;">Total Tokens per Agent</div>')
        ui.echart(
            {
                "backgroundColor": "transparent",
                "tooltip": {
                    "trigger": "axis",
                    "backgroundColor": CARD,
                    "borderColor": BORDER,
                    "textStyle": {"color": TEXT},
                },
                "xAxis": {
                    "type": "category",
                    "data": agents,
                    "axisLabel": {"color": TEXT_MUTED},
                    "axisLine": {"lineStyle": {"color": BORDER}},
                },
                "yAxis": {
                    "type": "value",
                    "name": "tokens",
                    "nameTextStyle": {"color": TEXT_MUTED},
                    "axisLabel": {"color": TEXT_MUTED},
                    "splitLine": {"lineStyle": {"color": BORDER}},
                },
                "series": [
                    {
                        "type": "bar",
                        "data": [
                            {
                                "value": data[ag]["total_tokens"],
                                "itemStyle": {"color": PURPLE + "cc"},
                            }
                            for ag in agents
                        ],
                        "barMaxWidth": 60,
                        "label": {
                            "show": True,
                            "position": "top",
                            "color": TEXT_MUTED,
                            "fontSize": 11,
                            "formatter": "{c}",
                        },
                    }
                ],
            }
        ).style(
            f"height:260px;background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px;"
        )


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

        # Pipeline analysis flow banner — descriptive steps, no day labels
        steps = [
            ("Evidence", "Extract sources & entities from each agent's output", CYAN),
            ("Info Loss", "Compare researcher → writer handoff for dropped facts", AMBER),
            ("Arbiter", "Score evidence, assign priority (P2/P5), root cause", PURPLE),
            ("Explain", "LLM generates actionable root-cause explanation", GREEN),
        ]
        with ui.element("div").style(
            "display:flex;align-items:center;gap:0;margin-bottom:24px;flex-wrap:wrap;"
        ):
            for i, (lbl, desc, col) in enumerate(steps):
                ui.html(f"""
                <div style="background:{col}18;border:1px solid {col}44;border-radius:8px;
                            padding:10px 16px;text-align:center;min-width:120px;max-width:180px;"
                     title="{desc}">
                  <div style="font-size:12px;font-weight:700;color:{col};">{lbl}</div>
                  <div style="font-size:10px;color:{TEXT_MUTED};margin-top:4px;line-height:1.4;">{desc}</div>
                </div>
                """)
                if i < len(steps) - 1:
                    ui.html(
                        f'<span style="color:{TEXT_DIM};font-size:16px;padding:0 6px;flex-shrink:0;">→</span>'
                    )

        content = ui.element("div")
        spinner = ui.element("div")

        async def load():
            with spinner:
                with ui.element("div").style(
                    f"background:{CARD};border:1px solid {BORDER};border-radius:10px;"
                    f"padding:48px;display:flex;flex-direction:column;align-items:center;gap:12px;"
                ):
                    ui.spinner(size="lg").style(f"color:{PURPLE};")
                    ui.html(
                        f'<div style="color:{TEXT_MUTED};font-size:14px;">Running full analysis pipeline…</div>'
                    )
                    ui.html(
                        f'<div style="color:{TEXT_DIM};font-size:12px;">Days 9 → 12 (LLM evidence extraction)</div>'
                    )

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
    # p_col / c_col reserved for future per-cause color coding
    _ = (
        PRIORITY_COLOR.get(bundle.priority_level.value, GRAY),
        CAUSE_COLOR.get(bundle.primary_cause.value, GRAY),
    )
    verdict = analysis.loss_result.verdict if analysis.loss_result else "UNKNOWN"
    conf = f"{analysis.loss_result.confidence:.0%}" if analysis.loss_result else "—"

    is_grounded = bundle.grounded
    hedge_note = (
        "heuristic — hedged language" if not is_grounded else "grounded — confident language"
    )

    # Copy verdict markdown
    # Build markdown string as a proper Python variable (not embedded in onclick HTML)
    md_verdict = (
        f"## AgentLens Verdict\n"
        f"**Run:** `{bundle.run_id}`\n"
        f"**Verdict:** {verdict} ({bundle.priority_level.value})\n"
        f"**Cause:** {bundle.primary_cause.value}\n"
        f"**Agent:** {bundle.primary_agent or 'N/A'}\n"
        f"**Confidence:** {conf}\n"
        f"**Grounded:** {'Yes' if is_grounded else 'No'}\n\n"
        f"### Root Cause\n{bundle.summary or ''}\n\n"
        f"### Suggested Fix\n{bundle.suggested_fix or ''}"
    )

    # ── Verdict row ───────────────────────────────────────────────────────────
    with ui.element("div").style(
        f"background:{CARD};border:1px solid {BORDER};border-radius:10px;"
        f"padding:16px 20px;margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;"
    ):
        for label, content_html in [
            ("Priority", priority_badge(bundle.priority_level.value)),
            ("Cause", cause_badge(bundle.primary_cause.value)),
            (
                "Agent",
                f'<span style="color:{STEP_COLOR.get(bundle.primary_agent or "", GRAY)};font-weight:600;">{bundle.primary_agent or "N/A"}</span>',
            ),
            ("Confidence", f'<span style="font-size:15px;font-weight:700;">{conf}</span>'),
            (
                "Grounded",
                f'<span style="color:{"#22c55e" if is_grounded else TEXT_MUTED};">{"✓ Yes" if is_grounded else "✗ No"}</span>',
            ),
        ]:
            with ui.element("div").style("flex:1;min-width:100px;"):
                ui.html(f'<div class="al-section" style="margin-bottom:6px;">{label}</div>')
                ui.html(content_html)

    # ── LLM Explanation card ──────────────────────────────────────────────────
    with ui.element("div").style(
        f"background:linear-gradient(135deg,{PURPLE}18,{CYAN}0a);"
        f"border:1px solid {PURPLE}44;border-radius:10px;padding:20px 22px;margin-bottom:16px;"
    ):
        with ui.element("div").style(
            "display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;"
        ):
            ui.html(f"""
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:14px;">✦</span>
              <span class="al-section" style="color:{PURPLE};">LLM Explanation</span>
              <span style="font-size:10px;color:{TEXT_DIM};padding:2px 6px;border:1px solid {BORDER};border-radius:4px;">
                {hedge_note}
              </span>
            </div>
            """)

            # Use NiceGUI button + async JS for clipboard — HTML onclick breaks with special chars
            async def copy_to_clipboard(md=md_verdict):
                import json as _json

                await ui.run_javascript(f"navigator.clipboard.writeText({_json.dumps(md)})")
                ui.notify("✓ Copied to clipboard", type="positive", position="top", timeout=2000)

            ui.button("📋 Copy verdict", on_click=copy_to_clipboard).props("flat dense").style(
                f"color:{TEXT_MUTED};font-size:11px;border:1px solid {BORDER};"
                f"border-radius:6px;padding:4px 10px;"
            )

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
            ui.html(
                f'<div class="al-section" style="margin-bottom:12px;">Fired Rules ({len(bundle.rule_matches)})</div>'
            )
            for rm in bundle.rule_matches:
                sev_col = {"HIGH": RED, "MEDIUM": AMBER, "LOW": CYAN}.get(
                    rm.severity.value.upper(), GRAY
                )
                ui.html(f"""
                <div style="padding:10px 0;border-bottom:1px solid {BORDER};">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;">
                    {rule_badge(rm.rule_id)}
                    {badge(rm.severity.value, sev_col)}
                    <span style="font-size:11px;color:{TEXT_MUTED};margin-left:auto;">agent: {rm.agent or "unknown"}</span>
                  </div>
                  <div style="font-size:12px;color:{TEXT_MUTED};">{rm.description[:200]}</div>
                </div>
                """)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ in {"__main__", "__mp_main__"}:
    import os

    # Serve dashboard/assets/ at /assets so logo.png is reachable
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    app.add_static_files("/assets", assets_dir)

    ui.run(
        title="AgentLens",
        host="127.0.0.1",
        port=8080,
        reload=False,
        dark=True,
        favicon=os.path.join(assets_dir, "logo.png")
        if os.path.exists(os.path.join(assets_dir, "logo.png"))
        else "🔬",
    )
