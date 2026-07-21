"""
dashboard/theme.py — AgentLens design system (v2)
Matches the Run Explorer screenshot: near-black bg, dark-tinted rows, badge system.
"""

# ── Base palette ──────────────────────────────────────────────────────────────
BG          = "#0c0c12"
BG_SIDEBAR  = "#09090f"
CARD        = "#111118"
CARD_HOVER  = "#16161f"
BORDER      = "#1e1e2e"
BORDER_SOFT = "#161622"
TEXT        = "#e2e2f0"
TEXT_MUTED  = "#6e6e88"
TEXT_DIM    = "#3e3e55"

# Accents
PURPLE      = "#8b5cf6"
CYAN        = "#06b6d4"
GREEN       = "#22c55e"
AMBER       = "#f59e0b"
RED         = "#ef4444"
GRAY        = "#6b7280"

# ── Row tints (status-coded, matches screenshot) ──────────────────────────────
ROW_TINT = {
    "FAIL":    "rgba(180, 20,  20,  0.18)",   # deep crimson
    "WARNING": "rgba(160, 80,  0,   0.15)",   # dark amber
    "PASS":    "rgba(20,  120, 40,  0.06)",   # very subtle green
    "P5":      "transparent",
    "UNKNOWN": "transparent",
}

# ── Badge colours ─────────────────────────────────────────────────────────────
VERDICT_COLOR = {"PASS": GREEN, "WARNING": AMBER, "FAIL": RED, "UNKNOWN": GRAY}
PRIORITY_COLOR = {"P1": RED, "P2": AMBER, "P3": CYAN, "P4": PURPLE, "P5": GRAY}
CAUSE_COLOR = {
    "reasoning":    AMBER,
    "workflow":     RED,
    "execution":    CYAN,
    "verification": PURPLE,
    "unknown":      GRAY,
}
STEP_COLOR = {
    "researcher": "#3b82f6",
    "writer":     PURPLE,
    "verifier":   CYAN,
}

# ── Global CSS ────────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --bg:       {BG};
  --card:     {CARD};
  --border:   {BORDER};
  --text:     {TEXT};
  --muted:    {TEXT_MUTED};
  --purple:   {PURPLE};
  --cyan:     {CYAN};
  --green:    {GREEN};
  --amber:    {AMBER};
  --red:      {RED};
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body, .q-page, .nicegui-content {{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}}

/* ── Header ── */
.al-header {{
  background: {CARD};
  border-bottom: 1px solid {BORDER};
  height: 54px;
  display: flex;
  align-items: center;
  padding: 0 28px;
  gap: 16px;
  position: sticky;
  top: 0;
  z-index: 100;
}}

/* ── Nav tabs ── */
.al-nav {{
  display: flex;
  align-items: center;
  gap: 2px;
  background: {BG};
  border-bottom: 1px solid {BORDER};
  padding: 0 28px;
  height: 42px;
}}
.al-nav-tab {{
  font-size: 12px;
  font-weight: 500;
  padding: 0 14px;
  height: 42px;
  display: flex;
  align-items: center;
  gap: 6px;
  color: {TEXT_MUTED};
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
  text-decoration: none;
}}
.al-nav-tab:hover {{ color: {TEXT}; }}
.al-nav-tab.active {{ color: {PURPLE}; border-bottom-color: {PURPLE}; }}
.al-nav-divider {{ width: 1px; height: 20px; background: {BORDER}; margin: 0 8px; }}

/* ── Content area ── */
.al-content {{ padding: 24px 28px; max-width: 1280px; margin: 0 auto; }}

/* ── Stat cards ── */
.al-stat {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 16px 20px;
  flex: 1;
  min-width: 130px;
}}
.al-stat-label {{
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: {TEXT_MUTED};
  margin-bottom: 8px;
}}
.al-stat-value {{
  font-size: 26px;
  font-weight: 700;
  line-height: 1;
}}
.al-stat-sub {{
  font-size: 11px;
  color: {TEXT_MUTED};
  margin-top: 4px;
}}

/* ── Table ── */
.al-table {{
  width: 100%;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid {BORDER};
}}
.al-thead {{
  display: grid;
  padding: 9px 20px;
  background: {CARD};
  border-bottom: 1px solid {BORDER};
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: {TEXT_MUTED};
}}
.al-trow {{
  display: grid;
  padding: 0;
  border-bottom: 1px solid {BORDER_SOFT};
  transition: filter 0.12s;
  cursor: pointer;
}}
.al-trow:last-child {{ border-bottom: none; }}
.al-trow:hover {{ filter: brightness(1.12); }}
.al-tcell {{
  padding: 13px 20px;
  display: flex;
  align-items: center;
}}

/* ── Inline expansion panel ── */
.al-expansion {{
  border-top: 1px solid {BORDER};
  padding: 16px 20px 18px;
  background: {BG_SIDEBAR};
}}

/* ── Badge ── */
.al-badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.3px;
}}

/* ── Metric bar ── */
.al-bar-bg {{ background: {BORDER}; border-radius: 3px; overflow: hidden; }}
.al-bar-fill {{ border-radius: 3px; transition: width 0.5s ease; }}

/* ── Mono ── */
.al-mono {{
  font-family: 'JetBrains Mono', monospace !important;
}}

/* ── Section label ── */
.al-section {{
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: {TEXT_MUTED};
}}

/* ── Node (trace timeline) ── */
.al-node {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 14px 18px;
  min-width: 180px;
  cursor: pointer;
  transition: border-color 0.15s;
}}
.al-node:hover {{ border-color: {PURPLE}; }}
.al-node-expanded {{ border-color: {PURPLE}; }}

/* ── JSON viewer ── */
.al-json {{
  background: {BG_SIDEBAR};
  border: 1px solid {BORDER};
  border-radius: 8px;
  padding: 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  line-height: 1.6;
  max-height: 280px;
  overflow-y: auto;
  color: {TEXT_MUTED};
  white-space: pre-wrap;
  word-break: break-all;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {PURPLE}; }}

/* ── Button override ── */
.q-btn {{ border-radius: 7px !important; font-weight: 600 !important; }}
.q-spinner-dots {{ color: {PURPLE} !important; }}

/* ── Dropdown / select ── */
.al-select {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 7px;
  padding: 5px 12px;
  font-size: 12px;
  color: {TEXT};
  cursor: pointer;
  appearance: none;
  min-width: 130px;
}}

/* ── Copy button ── */
.al-copy {{
  font-size: 11px;
  padding: 4px 10px;
  border: 1px solid {BORDER};
  border-radius: 6px;
  background: transparent;
  color: {TEXT_MUTED};
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}}
.al-copy:hover {{ color: {TEXT}; border-color: {TEXT_MUTED}; }}

/* ── Cost ticker ── */
.al-cost {{
  font-size: 11px;
  color: {TEXT_MUTED};
  font-family: 'JetBrains Mono', monospace;
  padding: 3px 8px;
  border: 1px solid {BORDER};
  border-radius: 5px;
}}
</style>
"""


# ── HTML helpers ─────────────────────────────────────────────────────────────

def badge(label: str, color: str, icon: str = "■") -> str:
    return (
        f'<span class="al-badge" style="background:{color}1a;color:{color};border:1px solid {color}44;">'
        f'<span style="font-size:8px;">{icon}</span>{label}</span>'
    )


def verdict_badge(verdict: str, grounded: bool) -> str:
    if verdict == "PASS":
        return badge("Grounded" if grounded else "Pass", GREEN)
    label = "Grounded" if grounded else "Heuristic"
    color = VERDICT_COLOR.get(verdict, GRAY)
    return badge(label, color, "□")


def priority_badge(p: str) -> str:
    return badge(p, PRIORITY_COLOR.get(p, GRAY))


def cause_badge(cause: str) -> str:
    return badge(cause.upper(), CAUSE_COLOR.get(cause, GRAY))


def rule_badge(rule_id: str) -> str:
    """Display a rule ID badge like R-001."""
    short = rule_id.replace("information_loss_", "R-00").replace("_v1", "1")[:6]
    return f'<span class="al-mono al-badge" style="background:{PURPLE}1a;color:{PURPLE};border:1px solid {PURPLE}33;font-size:10px;">{short}</span>'


def row_bg(verdict: str) -> str:
    return ROW_TINT.get(verdict, "transparent")


def fmt_ms(ms: float) -> str:
    return f"{ms/1000:.1f}s" if ms >= 1000 else f"{int(ms)}ms"


def bar_html(value: float, max_val: float, color: str, h: int = 5) -> str:
    pct = min(100, (value / max_val * 100) if max_val else 0)
    return (
        f'<div class="al-bar-bg" style="height:{h}px;">'
        f'<div class="al-bar-fill" style="width:{pct:.1f}%;background:{color};height:{h}px;"></div>'
        f'</div>'
    )
