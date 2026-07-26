"""Patch script: replace 'Explain this' stub HTML in app.py with real NiceGUI dialog buttons."""

src = open("dashboard/app.py", "r", encoding="utf-8").read()

OLD = """                    for agent, ev in result.extracted.items():
                        col = STEP_COLOR.get(agent, GRAY)
                        ui.html(f\\\"\\\"\\\"
                        <div style=\\\"padding:14px 20px;border-bottom:1px solid {BORDER};
                                    display:flex;align-items:center;gap:16px;flex-wrap:wrap;\\\">
                          <div style=\\\"min-width:100px;\\\">
                            <span style=\\\"font-size:12px;font-weight:600;color:{col};text-transform:capitalize;\\\">{agent}</span>
                          </div>
                          <div style=\\\"display:flex;gap:24px;flex:1;\\\">
                            <div>
                              <div class=\\\"al-section\\\" style=\\\"margin-bottom:4px;\\\">Sources</div>
                              <span style=\\\"font-size:18px;font-weight:700;color:{TEXT};\\\">{ev.source_count}</span>
                            </div>
                            <div>
                              <div class=\\\"al-section\\\" style=\\\"margin-bottom:4px;\\\">Entities</div>
                              <span style=\\\"font-size:18px;font-weight:700;color:{TEXT};\\\">{ev.entity_count}</span>
                            </div>
                            <div>
                              <div class=\\\"al-section\\\" style=\\\"margin-bottom:4px;\\\">Tool Calls</div>
                              <span style=\\\"font-size:18px;font-weight:700;color:{TEXT};\\\">{len(ev.tool_calls)}</span>
                            </div>
                          </div>
                          <button class=\\\"al-copy\\\" onclick=\\\"alert('Explain feature coming in Day 18')\\\">
                            Explain this →
                          </button>
                        </div>
                        \\\"\\\"\\\")"""

if OLD.encode() in src.encode():
    print("OLD block found (exact)")
else:
    # Try searching for the distinctive stub line
    stub = "onclick=\\\"alert('Explain feature coming in Day 18')\\\""
    if stub in src:
        print("found stub via simple search")
    else:
        # Try unescaped
        stub2 = "onclick=\"alert('Explain feature coming in Day 18')\""
        if stub2 in src:
            print("found stub (unescaped)")
        else:
            print("Cannot find stub — printing surrounding context")
            idx = src.find("Explain feature coming in Day 18")
            print(repr(src[max(0,idx-200):idx+200]))
