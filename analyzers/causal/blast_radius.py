"""
analyzers/causal/blast_radius.py — Error Blast Radius & Financial Impact Calculator
====================================================================================
Day 63: Quantifies the computational and financial waste caused by a root-cause error
in multi-agent workflows.

Computes:
  - tainted_steps_ratio: Fraction of steps/spans executed in a corrupted/tainted state.
  - wasted_tokens: Prompt, completion, and total tokens consumed by tainted downstream steps.
  - wasted_cost_usd: Estimated USD financial waste based on token pricing models.
  - wasted_latency_ms: Wall-clock execution time spent computing tainted steps.
  - impacted_agents: Unique list of downstream victim agents tainted by the failure.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from analyzers.causal.fault_tree import CausalGraph, CausalGraphEngine
from schema.models import (
    AnalysisBundle,
    BlastRadiusReport,
    RunTrace,
    TokenUsage,
)


class TokenPricing(BaseModel):
    """
    Token pricing model for calculating USD financial cost.
    Prices are expressed in USD per 1,000,000 tokens (standard provider convention).
    """

    model_name: str = "default"
    prompt_cost_per_million: float = 3.00  # $3.00 / 1M prompt tokens
    completion_cost_per_million: float = 15.00  # $15.00 / 1M completion tokens
    flat_cost_per_token: float | None = None  # Optional uniform per-token rate

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int | None = None,
    ) -> float:
        """Calculate total USD cost for the given token consumption."""
        if self.flat_cost_per_token is not None:
            total = (
                total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
            )
            return round(total * self.flat_cost_per_token, 6)

        prompt_cost = (prompt_tokens / 1_000_000.0) * self.prompt_cost_per_million
        completion_cost = (completion_tokens / 1_000_000.0) * self.completion_cost_per_million
        return round(prompt_cost + completion_cost, 6)

    @classmethod
    def get_preset(cls, name: str) -> TokenPricing:
        """Return a predefined pricing preset."""
        presets: dict[str, dict[str, Any]] = {
            "default": {"prompt_cost_per_million": 3.00, "completion_cost_per_million": 15.00},
            "gpt-4o": {"prompt_cost_per_million": 2.50, "completion_cost_per_million": 10.00},
            "gpt-4o-mini": {"prompt_cost_per_million": 0.15, "completion_cost_per_million": 0.60},
            "claude-3-5-sonnet": {
                "prompt_cost_per_million": 3.00,
                "completion_cost_per_million": 15.00,
            },
            "llama-3-70b-groq": {
                "prompt_cost_per_million": 0.59,
                "completion_cost_per_million": 0.79,
            },
            "flat_config": {"flat_cost_per_token": 0.000005},
        }
        cfg = presets.get(name.lower(), presets["default"])
        return cls(model_name=name, **cfg)

    @classmethod
    def from_config(cls) -> TokenPricing:
        """Build a TokenPricing instance from config/config.yaml if available."""
        try:
            from config.config_loader import get

            rate = get("llm", "cost_per_token_usd")
            if rate is not None and float(rate) > 0:
                return cls(model_name="flat_config", flat_cost_per_token=float(rate))
        except Exception:
            pass
        return cls.get_preset("default")


class BlastRadiusCalculator:
    """
    Computes computational waste, token loss, and USD financial impact caused
    by root-cause errors in multi-agent workflows.
    """

    def __init__(self, default_pricing: TokenPricing | None = None) -> None:
        self.pricing: TokenPricing = default_pricing or TokenPricing.get_preset("default")
        self.graph_engine = CausalGraphEngine()

    def calculate(
        self,
        run_trace: RunTrace,
        causal_graph: CausalGraph | None = None,
        analysis_bundle: AnalysisBundle | None = None,
        pricing: TokenPricing | None = None,
        include_root_cause: bool = False,
    ) -> BlastRadiusReport:
        """
        Calculate the error blast radius and financial waste for a run.

        Args:
            run_trace: The trace containing spans or steps.
            causal_graph: Optional precomputed CausalGraph from Day 62.
            analysis_bundle: Optional Arbiter analysis bundle.
            pricing: Optional custom token pricing override.
            include_root_cause: If True, include the root cause node in wasted compute.
                               Default is False (only downstream collateral waste is counted).

        Returns:
            BlastRadiusReport quantifying waste and impact.
        """
        active_pricing = pricing or self.pricing

        # Ensure causal graph is available
        graph = causal_graph
        if graph is None:
            graph = self.graph_engine.build_graph(
                run_trace,
                analysis_bundle=analysis_bundle,
            )

        total_steps = len(run_trace.spans) if run_trace.spans else len(run_trace.steps)

        # Clean run check: zero root causes
        if not graph.root_causes or total_steps == 0:
            return BlastRadiusReport(
                run_id=run_trace.run_id,
                total_steps=total_steps,
                tainted_steps_count=0,
                tainted_steps_ratio=0.0,
                wasted_tokens=TokenUsage(prompt=0, completion=0, total=0),
                wasted_cost_usd=0.0,
                wasted_latency_ms=0.0,
                impacted_agents=[],
                pricing_model=active_pricing.model_name,
                summary="Clean pipeline run. Zero blast radius and zero financial waste.",
            )

        # Collect tainted nodes
        tainted_node_ids: set[str] = set(graph.cascades + graph.passthroughs)
        if include_root_cause:
            tainted_node_ids.update(graph.root_causes)

        tainted_nodes = [node for nid, node in graph.nodes.items() if nid in tainted_node_ids]
        tainted_count = len(tainted_nodes)
        ratio = min(1.0, round(tainted_count / max(total_steps, 1), 4))

        # Aggregate wasted tokens and latency
        wasted_prompt = sum(n.tokens_prompt for n in tainted_nodes)
        wasted_completion = sum(n.tokens_completion for n in tainted_nodes)
        wasted_total = sum(n.tokens_total for n in tainted_nodes)
        wasted_latency = sum(n.latency_ms for n in tainted_nodes)

        # If individual prompt/completion tokens are 0 but total > 0, estimate 60/40 split
        if wasted_total > 0 and (wasted_prompt + wasted_completion) == 0:
            wasted_prompt = int(wasted_total * 0.6)
            wasted_completion = wasted_total - wasted_prompt

        wasted_cost = active_pricing.calculate_cost(
            prompt_tokens=wasted_prompt,
            completion_tokens=wasted_completion,
            total_tokens=wasted_total,
        )

        impacted_agents: list[str] = []
        for n in tainted_nodes:
            if n.name not in impacted_agents:
                impacted_agents.append(n.name)

        root_cause_agent = None
        primary_root = graph.get_root_cause()
        if primary_root:
            root_cause_agent = primary_root.name

        summary = (
            f"Blast radius: {tainted_count} of {total_steps} steps tainted ({ratio:.1%}). "
            f"Wasted {wasted_total:,} tokens (${wasted_cost:.6f} USD) and {wasted_latency:,.1f}ms "
            f"across impacted agents: {', '.join(impacted_agents) if impacted_agents else 'None'}."
        )

        return BlastRadiusReport(
            run_id=run_trace.run_id,
            root_cause_agent=root_cause_agent,
            root_cause_node_id=graph.primary_root_cause_node,
            tainted_steps_ratio=ratio,
            total_steps=total_steps,
            tainted_steps_count=tainted_count,
            wasted_tokens=TokenUsage(
                prompt=wasted_prompt,
                completion=wasted_completion,
                total=wasted_total,
            ),
            wasted_cost_usd=wasted_cost,
            wasted_latency_ms=round(wasted_latency, 2),
            impacted_agents=impacted_agents,
            pricing_model=active_pricing.model_name,
            summary=summary,
        )
