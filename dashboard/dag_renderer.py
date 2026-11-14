"""
dashboard/dag_renderer.py — Interactive Causal DAG Visualizer (Mermaid.js)
==========================================================================
Day 65: Translates CausalGraph and RunTrace execution DAGs into interactive,
color-coded Mermaid.js flowchart syntax for NiceGUI v2.

Causal State Color Palette:
  - 🔴 ROOT_CAUSE_ORIGIN : Red (#ef4444) / Dark Crimson (#7f1d1d)
  - 🟠 CASCADING_FAILURE : Orange (#f59e0b) / Dark Amber (#78350f)
  - 🟡 PASSTHROUGH_ESCAPED : Yellow (#eab308) / Dark Yellow (#713f12)
  - 🟢 HEALTHY           : Green (#22c55e) / Dark Emerald (#064e3b)
"""

from __future__ import annotations

import re

from analyzers.causal.fault_tree import (
    CausalGraph,
    CausalGraphEngine,
    CausalNodeState,
)
from schema.models import (
    AnalysisBundle,
    BlastRadiusReport,
    RemediationPatch,
    RunTrace,
)


class DAGRenderer:
    """
    Renders CausalGraphs and RunTraces into valid, styled Mermaid.js flowchart diagrams.
    """

    def __init__(self) -> None:
        self.engine = CausalGraphEngine()

    def sanitize_label(self, text: str) -> str:
        """Sanitize text to prevent breaking Mermaid syntax."""
        cleaned = text.replace('"', "#quot;").replace("[", "&#91;").replace("]", "&#93;")
        cleaned = re.sub(r"[\r\n]+", " ", cleaned).strip()
        return cleaned[:60]

    def render_mermaid(
        self,
        causal_graph: CausalGraph,
        orientation: str = "LR",
    ) -> str:
        """
        Generate Mermaid.js flowchart markup from a CausalGraph.

        Args:
            causal_graph: The classified CausalGraph.
            orientation: Flowchart direction ('LR' for horizontal, 'TD' for top-down).

        Returns:
            Mermaid flowchart string with embedded class definitions and node bindings.
        """
        lines: list[str] = [
            f"flowchart {orientation}",
            "    %% AgentLens Causal Status Class Definitions",
            "    classDef root fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#ffffff;",
            "    classDef cascade fill:#78350f,stroke:#f59e0b,stroke-width:2px,color:#ffffff;",
            "    classDef passthrough fill:#713f12,stroke:#eab308,stroke-width:2px,color:#ffffff;",
            "    classDef healthy fill:#064e3b,stroke:#22c55e,stroke-width:2px,color:#ffffff;",
            "",
        ]

        if not causal_graph.nodes:
            lines.append('    empty_node["Empty trace (no execution steps)"]:::healthy')
            return "\n".join(lines)

        # Generate node definitions
        for nid, node in causal_graph.nodes.items():
            css_class = self._get_css_class(node.state)
            icon = self._get_status_icon(node.state)
            safe_name = self.sanitize_label(node.name)

            # Build concise multi-line label
            label_lines = [f"{icon} {safe_name}"]

            if node.state == CausalNodeState.ROOT_CAUSE_ORIGIN:
                rule_tag = node.rule_matches[0].rule_id if node.rule_matches else "Fault Origin"
                label_lines.append(f"(Root: {self.sanitize_label(rule_tag)})")
            elif node.state == CausalNodeState.CASCADING_FAILURE:
                label_lines.append("(Cascading Failure)")
            elif node.state == CausalNodeState.PASSTHROUGH_ESCAPED:
                label_lines.append("(Escaped QC Verifier)")
            else:
                label_lines.append("(Healthy)")

            perf_parts: list[str] = []
            if node.latency_ms > 0:
                perf_parts.append(f"{node.latency_ms:.0f}ms")
            if node.tokens_total > 0:
                perf_parts.append(f"{node.tokens_total:,} tok")
            if perf_parts:
                label_lines.append(" | ".join(perf_parts))

            content = "<br/>".join(label_lines)
            safe_nid = re.sub(r"[^a-zA-Z0-9_]", "_", nid)
            lines.append(f'    {safe_nid}["{content}"]:::{css_class}')

        lines.append("")

        # Generate directed edges
        if causal_graph.edges:
            for edge in causal_graph.edges:
                if len(edge) >= 2:
                    src = re.sub(r"[^a-zA-Z0-9_]", "_", edge[0])
                    dst = re.sub(r"[^a-zA-Z0-9_]", "_", edge[1])
                    lines.append(f"    {src} --> {dst}")
        else:
            # Fallback: sequential linear edges if no explicit edges
            node_ids = list(causal_graph.nodes.keys())
            for i in range(len(node_ids) - 1):
                src = re.sub(r"[^a-zA-Z0-9_]", "_", node_ids[i])
                dst = re.sub(r"[^a-zA-Z0-9_]", "_", node_ids[i + 1])
                lines.append(f"    {src} --> {dst}")

        return "\n".join(lines)

    def render_from_trace(
        self,
        run_trace: RunTrace,
        analysis_bundle: AnalysisBundle | None = None,
        orientation: str = "LR",
    ) -> str:
        """Convenience method to construct CausalGraph and generate Mermaid markup."""
        graph = self.engine.build_graph(run_trace, analysis_bundle=analysis_bundle)
        return self.render_mermaid(graph, orientation=orientation)

    def get_legend_html(self) -> str:
        """Return styled HTML legend explaining causal node color codes."""
        return """
        <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:11px;margin-bottom:12px;">
          <span style="font-weight:600;color:#94a3b8;">Causal Status:</span>
          <span style="display:inline-flex;align-items:center;gap:4px;color:#ef4444;">
            <span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;"></span>
            Root Cause Origin
          </span>
          <span style="display:inline-flex;align-items:center;gap:4px;color:#f59e0b;">
            <span style="display:inline-block;width:10px;height:10px;background:#f59e0b;border-radius:2px;"></span>
            Cascading Failure
          </span>
          <span style="display:inline-flex;align-items:center;gap:4px;color:#eab308;">
            <span style="display:inline-block;width:10px;height:10px;background:#eab308;border-radius:2px;"></span>
            Escaped QC Verifier
          </span>
          <span style="display:inline-flex;align-items:center;gap:4px;color:#22c55e;">
            <span style="display:inline-block;width:10px;height:10px;background:#22c55e;border-radius:2px;"></span>
            Healthy Node
          </span>
        </div>
        """

    def get_blast_radius_html(self, report: BlastRadiusReport) -> str:
        """Generate HTML summary widget for the Blast Radius impact."""
        ratio_pct = f"{report.tainted_steps_ratio:.1%}"
        agents_str = ", ".join(report.impacted_agents) if report.impacted_agents else "None"
        bar_width = min(100, int(report.tainted_steps_ratio * 100))

        return f"""
        <div style="background:#111118;border:1px solid #1e1e2e;border-radius:8px;padding:16px;margin-bottom:16px;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div style="font-size:12px;font-weight:700;color:#f59e0b;text-transform:uppercase;letter-spacing:0.5px;">
              💥 Error Blast Radius & Financial Impact
            </div>
            <div style="font-size:11px;color:#94a3b8;font-family:monospace;">
              Pricing: {report.pricing_model}
            </div>
          </div>

          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:14px;">
            <div style="background:#09090f;border:1px solid #1e1e2e;border-radius:6px;padding:8px 12px;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Wasted Cost</div>
              <div style="font-size:16px;font-weight:700;color:#ef4444;margin-top:2px;">
                ${report.wasted_cost_usd:.4f} <span style="font-size:10px;color:#64748b;">USD</span>
              </div>
            </div>
            <div style="background:#09090f;border:1px solid #1e1e2e;border-radius:6px;padding:8px 12px;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Wasted Tokens</div>
              <div style="font-size:16px;font-weight:700;color:#f59e0b;margin-top:2px;">
                {report.wasted_tokens.total:,} <span style="font-size:10px;color:#64748b;">tok</span>
              </div>
            </div>
            <div style="background:#09090f;border:1px solid #1e1e2e;border-radius:6px;padding:8px 12px;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Wasted Latency</div>
              <div style="font-size:16px;font-weight:700;color:#06b6d4;margin-top:2px;">
                {report.wasted_latency_ms:,.0f} <span style="font-size:10px;color:#64748b;">ms</span>
              </div>
            </div>
            <div style="background:#09090f;border:1px solid #1e1e2e;border-radius:6px;padding:8px 12px;">
              <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Tainted Ratio</div>
              <div style="font-size:16px;font-weight:700;color:#eab308;margin-top:2px;">
                {ratio_pct} <span style="font-size:10px;color:#64748b;">({report.tainted_steps_count}/{report.total_steps})</span>
              </div>
            </div>
          </div>

          <!-- Progress Bar -->
          <div style="background:#1e1e2e;height:6px;border-radius:3px;overflow:hidden;margin-bottom:8px;">
            <div style="background:linear-gradient(90deg,#f59e0b,#ef4444);width:{bar_width}%;height:100%;"></div>
          </div>
          <div style="font-size:11px;color:#64748b;">
            Downstream Impacted Agents: <span style="color:#e2e8f0;font-weight:500;">{agents_str}</span>
          </div>
        </div>
        """

    def get_remediation_panel_html(self, patch: RemediationPatch) -> str:
        """Generate HTML representation of the prompt auto-diff remediation panel."""
        escaped_diff = (
            patch.diff_patch.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

        # Style diff lines
        formatted_lines: list[str] = []
        for line in escaped_diff.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                formatted_lines.append(
                    f'<span style="color:#94a3b8;font-weight:bold;">{line}</span>'
                )
            elif line.startswith("@@"):
                formatted_lines.append(f'<span style="color:#06b6d4;">{line}</span>')
            elif line.startswith("+"):
                formatted_lines.append(
                    f'<span style="color:#22c55e;background:#064e3b33;display:block;">{line}</span>'
                )
            elif line.startswith("-"):
                formatted_lines.append(
                    f'<span style="color:#ef4444;background:#7f1d1d33;display:block;">{line}</span>'
                )
            else:
                formatted_lines.append(f'<span style="color:#cbd5e1;">{line}</span>')

        styled_diff = "\n".join(formatted_lines)
        ref_text = (
            f" | Ref: {patch.historical_reference_run_id}"
            if patch.historical_reference_run_id
            else ""
        )

        return f"""
        <div style="background:#111118;border:1px solid #8b5cf644;border-radius:8px;padding:16px;margin-bottom:16px;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;flex-wrap:wrap;gap:8px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:14px;">🛠️</span>
              <span style="font-size:12px;font-weight:700;color:#8b5cf6;text-transform:uppercase;">
                Automated Prompt Remediation
              </span>
              <span style="font-size:10px;color:#a855f7;background:#8b5cf622;border:1px solid #8b5cf644;border-radius:4px;padding:2px 6px;">
                {patch.constraint_type}
              </span>
            </div>
            <div style="display:flex;align-items:center;gap:12px;font-size:11px;color:#94a3b8;">
              <span>Target: <strong style="color:#e2e8f0;">{patch.agent}</strong></span>
              <span>Confidence: <strong style="color:#22c55e;">{patch.confidence:.0%}</strong></span>
              <span>Risk: <strong style="color:#eab308;">{patch.regression_risk:.0%}</strong>{ref_text}</span>
            </div>
          </div>

          <div style="font-size:12px;color:#94a3b8;margin-bottom:10px;">
            {patch.explanation}
          </div>

          <div style="background:#09090f;border:1px solid #1e1e2e;border-radius:6px;padding:12px;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.5;overflow-x:auto;white-space:pre;">
{styled_diff}
          </div>
        </div>
        """

    def _get_css_class(self, state: CausalNodeState) -> str:
        match state:
            case CausalNodeState.ROOT_CAUSE_ORIGIN:
                return "root"
            case CausalNodeState.CASCADING_FAILURE:
                return "cascade"
            case CausalNodeState.PASSTHROUGH_ESCAPED:
                return "passthrough"
            case _:
                return "healthy"

    def _get_status_icon(self, state: CausalNodeState) -> str:
        match state:
            case CausalNodeState.ROOT_CAUSE_ORIGIN:
                return "🔴"
            case CausalNodeState.CASCADING_FAILURE:
                return "🟠"
            case CausalNodeState.PASSTHROUGH_ESCAPED:
                return "🟡"
            case _:
                return "🟢"
