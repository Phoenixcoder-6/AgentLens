"""Patch _inline_verdict_panel in app.py — replace HTML onclick copy button with NiceGUI async handler."""

src = open("dashboard/app.py", "r", encoding="utf-8").read()

# The entire _inline_verdict_panel function body (lines 230-274)
START = "def _inline_verdict_panel(bundle, loss, run_id: str):"
END   = "\ndef _inline_analyze_panel"

i_start = src.find(START)
i_end   = src.find(END, i_start)

if i_start == -1 or i_end == -1:
    print(f"Markers not found: start={i_start}, end={i_end}")
    exit(1)

old = src[i_start:i_end]
print("Replacing function, lines:", old.count("\n"))

NEW_FN = '''def _inline_verdict_panel(bundle, loss, run_id: str):
    conf        = f"{loss.confidence:.0%}" if loss else "—"
    verdict_str = loss.verdict if loss else "UNKNOWN"

    # Build the markdown string in Python (never embed in onclick HTML)
    md = (
        f"## AgentLens Verdict\\n"
        f"**Run:** `{run_id}`\\n"
        f"**Verdict:** {verdict_str} ({bundle.priority_level.value})\\n"
        f"**Cause:** {bundle.primary_cause.value}\\n"
        f"**Agent:** {bundle.primary_agent or 'N/A'}\\n"
        f"**Confidence:** {conf}\\n"
        f"**Grounded:** {'Yes' if bundle.grounded else 'No'}"
    )

    with ui.element("div").style(
        "display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;"
    ):
        # ── Fact chips ────────────────────────────────────────────────────────
        for label, html_val in [
            ("Priority",   priority_badge(bundle.priority_level.value)),
            ("Cause",      cause_badge(bundle.primary_cause.value)),
            ("Agent",      f\'<span style="font-size:13px;font-weight:500;color:{STEP_COLOR.get(bundle.primary_agent or \\\'\\\', GRAY)};">{bundle.primary_agent or "N/A"}</span>\'),
            ("Confidence", f\'<span style="font-size:13px;font-weight:600;">{conf}</span>\'),
            ("Grounded",   f\'<span style="color:{"#22c55e" if bundle.grounded else TEXT_MUTED};">{"✓ Yes" if bundle.grounded else "✗ No (heuristic)"}</span>\'),
        ]:
            with ui.element("div"):
                ui.html(f\'<div class="al-section" style="margin-bottom:6px;">{label}</div>\')
                ui.html(html_val)

        # ── Action buttons ────────────────────────────────────────────────────
        with ui.element("div").style("margin-left:auto;display:flex;gap:8px;align-items:center;"):

            async def _copy(text=md):
                import json as _j
                await ui.run_javascript(f"navigator.clipboard.writeText({_j.dumps(text)})")
                ui.notify("✓ Copied!", type="positive", position="top", timeout=1500)

            ui.button("📋 Copy verdict", on_click=_copy).props("flat dense").style(
                f"color:{TEXT_MUTED};font-size:11px;border:1px solid {BORDER};"
                "border-radius:6px;padding:4px 10px;"
            )
            ui.html(
                f\'<a href="/run/{run_id}" style="font-size:12px;color:{PURPLE};\'
                f\'text-decoration:none;padding:4px 10px;border:1px solid {PURPLE}44;border-radius:6px;">\'
                f\'Open trace →</a>\'
            )
            ui.html(
                f\'<a href="/run/{run_id}/explain" style="font-size:12px;color:{CYAN};\'
                f\'text-decoration:none;padding:4px 10px;border:1px solid {CYAN}44;border-radius:6px;">\'
                f\'✦ Explain</a>\'
            )

'''

new_src = src[:i_start] + NEW_FN + src[i_end:]
open("dashboard/app.py", "w", encoding="utf-8").write(new_src)
print(f"app.py patched OK — size {len(new_src)}")
