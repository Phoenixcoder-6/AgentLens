"""
config/topology.py — Config-Driven Multi-Agent Topology Resolver
=================================================================
Day 40a: Decouples detection rules and workflow validation from hardcoded
agent names ("researcher", "writer", "verifier") by resolving agents via
their configured roles (`information_gatherer`, `synthesizer`, `quality_checker`)
and `receives_from` handoff edges in `config/config.yaml` (`pipeline.agents`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import config_loader
from schema.models import AgentStep

ROLE_GATHERER = "information_gatherer"
ROLE_SYNTHESIZER = "synthesizer"
ROLE_CHECKER = "quality_checker"

DEFAULT_AGENTS: list[dict[str, str | None]] = [
    {"id": "researcher", "role": ROLE_GATHERER, "receives_from": None},
    {"id": "writer", "role": ROLE_SYNTHESIZER, "receives_from": "researcher"},
    {"id": "verifier", "role": ROLE_CHECKER, "receives_from": "writer"},
]

_DEFAULT_IDS = ["researcher", "writer", "verifier"]


@dataclass(frozen=True)
class AgentNodeSpec:
    """Topology specification for a single agent node."""

    id: str
    role: str
    receives_from: str | None = None


class PipelineTopology:
    """
    Resolves agent IDs, roles, and handoff edges from `pipeline.agents` in `config.yaml`.
    """

    def __init__(
        self,
        agents: list[dict[str, Any]] | None = None,
        workflow_required_agents: list[str] | None = None,
        pipeline_name: str | None = None,
    ) -> None:
        raw_agents = agents if agents is not None else self._load_raw_agents()
        self.name: str = pipeline_name or str(
            config_loader.get("pipeline", "name", "research_report_pipeline")
            or "research_report_pipeline"
        )

        parsed: list[AgentNodeSpec] = []
        for idx, item in enumerate(raw_agents):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            agent_id = str(item["id"])
            role = str(item.get("role") or self._infer_role(idx, len(raw_agents)))
            receives_from = item.get("receives_from")
            if receives_from is None and idx > 0 and parsed:
                receives_from = parsed[-1].id
            parsed.append(
                AgentNodeSpec(
                    id=agent_id,
                    role=role,
                    receives_from=str(receives_from) if receives_from else None,
                )
            )

        if not parsed:
            parsed = [
                AgentNodeSpec(
                    id=str(d["id"]),
                    role=str(d["role"]),
                    receives_from=d["receives_from"],
                )
                for d in DEFAULT_AGENTS
            ]

        self.nodes: list[AgentNodeSpec] = parsed
        self._workflow_override = workflow_required_agents

    @staticmethod
    def _infer_role(index: int, total: int) -> str:
        if index == 0:
            return ROLE_GATHERER
        if index == 1 or total == 2:
            return ROLE_SYNTHESIZER
        return ROLE_CHECKER

    @staticmethod
    def _load_raw_agents() -> list[dict[str, Any]]:
        try:
            configured = config_loader.get("pipeline", "agents", None)
            if isinstance(configured, list) and configured:
                return configured
        except Exception:
            pass
        return list(DEFAULT_AGENTS)

    def agent_ids_for_role(self, role: str) -> list[str]:
        """Return all configured agent IDs matching the given role."""
        return [node.id for node in self.nodes if node.role == role]

    def primary_agent_for_role(self, role: str) -> str | None:
        """Return the primary (first) agent ID for `role`, or None if role is absent."""
        ids = self.agent_ids_for_role(role)
        return ids[0] if ids else None

    def get_node(self, agent_id: str) -> AgentNodeSpec | None:
        """Look up AgentNodeSpec by agent ID."""
        for node in self.nodes:
            if node.id == agent_id:
                return node
        return None

    def upstream_agent_for(self, agent_id: str) -> str | None:
        """Return the `receives_from` upstream agent ID for `agent_id`."""
        node = self.get_node(agent_id)
        if node and node.receives_from:
            return node.receives_from
        return None

    def find_steps_for_role(self, steps: list[AgentStep], role: str) -> list[AgentStep]:
        """Return all trace steps executed by any agent configured with `role`."""
        target_ids = set(self.agent_ids_for_role(role))
        if not target_ids:
            return []
        return [s for s in steps if s.agent in target_ids]

    def find_step_for_role(self, steps: list[AgentStep], role: str) -> AgentStep | None:
        """Return the last trace step executed by an agent with `role`, or None."""
        matching = self.find_steps_for_role(steps, role)
        return matching[-1] if matching else None

    def find_upstream_step(
        self, steps: list[AgentStep], target_step: AgentStep | None
    ) -> AgentStep | None:
        """
        Find the upstream step that `target_step` receives from according to topology.
        Falls back to the `information_gatherer` step if no explicit upstream match exists.
        """
        if target_step is None:
            return None
        upstream_id = self.upstream_agent_for(target_step.agent)
        if upstream_id:
            upstream_steps = [
                s for s in steps if s.agent == upstream_id and s.step <= target_step.step
            ]
            if not upstream_steps:
                upstream_steps = [s for s in steps if s.agent == upstream_id]
            if upstream_steps:
                return upstream_steps[-1]
        return self.find_step_for_role(steps, ROLE_GATHERER)

    def required_agent_order(self) -> list[str]:
        """
        Return ordered list of required agent IDs for workflow validation.
        Honors `pipeline.agents` topology, while respecting explicit overrides to
        `arbiter.workflow.required_agents` when `pipeline.agents` is at default.
        """
        topo_ids = [node.id for node in self.nodes]
        if self._workflow_override is not None:
            return list(self._workflow_override)

        try:
            wf_cfg = config_loader.get("arbiter", "workflow", {}) or {}
            wf_req = wf_cfg.get("required_agents")
            if (
                isinstance(wf_req, list)
                and wf_req
                and topo_ids == _DEFAULT_IDS
                and wf_req != _DEFAULT_IDS
            ):
                return [str(a) for a in wf_req]
        except Exception:
            pass

        return topo_ids

    def handoff_edges(self) -> list[tuple[str, str]]:
        """
        Return ordered `(from_agent_id, to_agent_id)` handoff pairs.
        Uses `receives_from` when topology nodes are custom, or sequential pairs from
        `required_agent_order()`.
        """
        req = self.required_agent_order()
        topo_ids = [node.id for node in self.nodes]
        if req == topo_ids:
            edges: list[tuple[str, str]] = []
            for node in self.nodes:
                if node.receives_from:
                    edges.append((node.receives_from, node.id))
            if edges:
                return edges

        return [(req[i], req[i + 1]) for i in range(len(req) - 1)]


def get_topology(
    agents: list[dict[str, Any]] | None = None,
    workflow_required_agents: list[str] | None = None,
) -> PipelineTopology:
    """Build and return the active PipelineTopology from config.yaml."""
    return PipelineTopology(
        agents=agents,
        workflow_required_agents=workflow_required_agents,
    )
