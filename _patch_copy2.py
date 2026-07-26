"""Patch _inline_verdict_panel — clean version avoiding quote escaping issues."""

src = open("dashboard/app.py", "r", encoding="utf-8").read()

START = "def _inline_verdict_panel(bundle, loss, run_id: str):"
END   = "\ndef _inline_analyze_panel"

i_start = src.find(START)
i_end   = src.find(END, i_start)

if i_start == -1 or i_end == -1:
    print(f"Markers not found: start={i_start}, end={i_end}")
    exit(1)

NEW_FN = (
    'def _inline_verdict_panel(bundle, loss, run_id: str):\n'
    '    conf        = f"{loss.confidence:.0%}" if loss else "\u2014"\n'
    '    verdict_str = loss.verdict if loss else "UNKNOWN"\n'
    '\n'
    '    # Build markdown in Python so it can be passed safely via json.dumps\n'
    '    import json as _j\n'
    '    md = (\n'
    '        f"## AgentLens Verdict\\n"\n'
    '        f"**Run:** `{run_id}`\\n"\n'
    '        f"**Verdict:** {verdict_str} ({bundle.priority_level.value})\\n"\n'
    '        f"**Cause:** {bundle.primary_cause.value}\\n"\n'
    '        f"**Agent:** {bundle.primary_agent or \'N/A\'}\\n"\n'
    '        f"**Confidence:** {conf}\\n"\n'
    '        f"**Grounded:** {\'Yes\' if bundle.grounded else \'No\'}"\n'
    '    )\n'
    '\n'
    '    with ui.element("div").style(\n'
    '        "display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;"\n'
    '    ):\n'
    '        for label, html_val in [\n'
    '            ("Priority",   priority_badge(bundle.priority_level.value)),\n'
    '            ("Cause",      cause_badge(bundle.primary_cause.value)),\n'
    '        ]:\n'
    '            with ui.element("div"):\n'
    '                ui.html(f\'<div class="al-section" style="margin-bottom:6px;">{label}</div>\')\n'
    '                ui.html(html_val)\n'
    '\n'
    '        ag_color = STEP_COLOR.get(bundle.primary_agent or "", GRAY)\n'
    '        grnd_color = "#22c55e" if bundle.grounded else TEXT_MUTED\n'
    '        grnd_txt   = "\\u2713 Yes" if bundle.grounded else "\\u2717 No (heuristic)"\n'
    '\n'
    '        with ui.element("div"):\n'
    '            ui.html(\'<div class="al-section" style="margin-bottom:6px;">Agent</div>\')\n'
    '            ui.html(f\'<span style="font-size:13px;font-weight:500;color:{ag_color};">{bundle.primary_agent or "N/A"}</span>\')\n'
    '\n'
    '        with ui.element("div"):\n'
    '            ui.html(\'<div class="al-section" style="margin-bottom:6px;">Confidence</div>\')\n'
    '            ui.html(f\'<span style="font-size:13px;font-weight:600;">{conf}</span>\')\n'
    '\n'
    '        with ui.element("div"):\n'
    '            ui.html(\'<div class="al-section" style="margin-bottom:6px;">Grounded</div>\')\n'
    '            ui.html(f\'<span style="color:{grnd_color};">{grnd_txt}</span>\')\n'
    '\n'
    '        # Action buttons\n'
    '        with ui.element("div").style("margin-left:auto;display:flex;gap:8px;align-items:center;"):\n'
    '\n'
    '            async def _copy(text=md):\n'
    '                await ui.run_javascript(f"navigator.clipboard.writeText({_j.dumps(text)})")\n'
    '                ui.notify("\\u2713 Copied!", type="positive", position="top", timeout=1500)\n'
    '\n'
    '            ui.button("\\U0001f4cb Copy verdict", on_click=_copy).props("flat dense").style(\n'
    '                f"color:{TEXT_MUTED};font-size:11px;border:1px solid {BORDER};"\n'
    '                "border-radius:6px;padding:4px 10px;"\n'
    '            )\n'
    '            ui.html(\n'
    '                f\'<a href="/run/{run_id}" style="font-size:12px;color:{PURPLE};text-decoration:none;\'\n'
    '                f\'padding:4px 10px;border:1px solid {PURPLE}44;border-radius:6px;">Open trace \\u2192</a>\'\n'
    '            )\n'
    '            ui.html(\n'
    '                f\'<a href="/run/{run_id}/explain" style="font-size:12px;color:{CYAN};text-decoration:none;\'\n'
    '                f\'padding:4px 10px;border:1px solid {CYAN}44;border-radius:6px;">\\u2726 Explain</a>\'\n'
    '            )\n'
    '\n'
)

new_src = src[:i_start] + NEW_FN + src[i_end:]
open("dashboard/app.py", "w", encoding="utf-8").write(new_src)
print(f"app.py patched OK — {len(new_src)} bytes")
