"""Patch app.py: replace extracted facts HTML stub block with NiceGUI dialog buttons."""

src = open("dashboard/app.py", "r", encoding="utf-8").read()

# Find the extracted evidence for-loop block by searching for the start/end markers
START = "                # ── Section: Extracted Evidence ────────────────────────────────"
END   = "\n                # ── Section: Information Loss Detail"

i_start = src.find(START)
i_end   = src.find(END, i_start)

if i_start == -1 or i_end == -1:
    print(f"ERROR: markers not found. i_start={i_start}, i_end={i_end}")
    exit(1)

old_block = src[i_start:i_end]
print("old block lines:", old_block.count("\n"))

NEW_BLOCK = """                # ── Section: Extracted Evidence ────────────────────────────────
                ui.html(f'<div class="al-section" style="margin-bottom:12px;margin-top:4px;">Extracted Facts ({len(result.extracted)} agents)</div>')
                with ui.element("div").style(
                    f"background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:20px;"
                ):
                    for _ag, _ev in result.extracted.items():
                        _ag_col = STEP_COLOR.get(_ag, GRAY)
                        with ui.element("div").style(
                            f"padding:14px 20px;border-bottom:1px solid {BORDER};"
                            "display:flex;align-items:center;gap:16px;flex-wrap:wrap;"
                        ):
                            ui.html(f\"\"\"
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
                            \"\"\")

                            def _mk_btn(ag=_ag, ev=_ev, bndl=result.bundle):
                                async def _click():
                                    ag_col = STEP_COLOR.get(ag, GRAY)
                                    dlg = ui.dialog()
                                    with dlg, ui.card().style(
                                        f"background:{CARD};border:1px solid {BORDER};"
                                        "min-width:520px;max-width:680px;padding:24px;"
                                    ):
                                        ui.html(f\"\"\"
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
                                        \"\"\")
                                        status_lbl = ui.html(
                                            f'<div style="color:{TEXT_MUTED};padding:8px 0;">'
                                            f'<span style="color:{PURPLE};">⟳</span>  Generating explanation…</div>'
                                        )
                                        with ui.row().style("justify-content:flex-end;margin-top:12px;"):
                                            ui.button("Close", on_click=dlg.close).props("flat").style(f"color:{TEXT_MUTED};")
                                    dlg.open()
                                    text = await run.io_bound(state.explain_agent_evidence, ag, ev, bndl)
                                    status_lbl.set_content(f\"\"\"
                                    <div style="background:{BG_SIDEBAR};border-left:3px solid {STEP_COLOR.get(ag, GRAY)};
                                                border-radius:6px;padding:14px 16px;">
                                      <div class="al-section" style="color:{STEP_COLOR.get(ag, GRAY)};margin-bottom:8px;">LLM Explanation</div>
                                      <div style="font-size:14px;line-height:1.75;color:{TEXT};">{text}</div>
                                    </div>
                                    \"\"\")

                                ui.button("Explain this →", on_click=_click).props("flat dense").style(
                                    f"color:{TEXT_MUTED};font-size:11px;border:1px solid {BORDER};"
                                    "border-radius:6px;padding:4px 12px;white-space:nowrap;"
                                )

                            _mk_btn()"""

new_src = src[:i_start] + NEW_BLOCK + src[i_end:]
open("dashboard/app.py", "w", encoding="utf-8").write(new_src)
print(f"app.py patched OK — new size {len(new_src)}")
