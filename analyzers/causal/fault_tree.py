"""
analyzers/causal/fault_tree.py — Causal Fault Propagation Graph Engine
========================================================================
Day 62: Models causal error propagation across multi-agent workflows,
isolating root-cause origins from cascading victim nodes and escaped
quality-control checkpoints.

Classes:
  - CausalNodeState: Enumeration of causal roles (ROOT_CAUSE_ORIGIN,
    CASCADING_FAILURE, PASSTHROUGH_ESCAPED, HEALTHY).
  - CausalNode: Graph node representing an agent, span, or step with causal attribution.
  - CausalChain: Sequence of nodes linking root cause to terminal impact.
  - CausalGraph: Directed graph model containing all nodes, edges, and causal paths.
  - CausalGraphEngine: Engine that analyzes RunTrace DAGs and Evidence/RuleMatches
    to construct the causal fault tree.
"""

from __future__ import annotations

from collections import defaultdict, deque
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from schema.models import (
    AnalysisBundle,
    EvidenceRecord,
    FailureCategory,
    RuleMatch,
    RunTrace,
    StepStatus,
)
from schema.span import TraceSpan


class CausalNodeState(StrEnum):
    """Causal classification of an execution node in a multi-agent workflow."""

    ROOT_CAUSE_ORIGIN = "root_cause_origin"  # First node where an invariant broke or fault originated
    CASCADING_FAILURE = "cascading_failure"  # Downstream node failing as consequence of upstream corrupted state
    PASSTHROUGH_ESCAPED = "passthrough_escaped"  # Quality-control / verifier node that failed to intercept the error
    HEALTHY = "healthy"  # Unaffected execution node running normally


class CausalNode(BaseModel):
    """A node in the causal fault propagation graph."""

    node_id: str = Field(description="Unique identifier for the node (span_id or step_id)")
    name: str = Field(description="Agent or span name (e.g. 'researcher', 'writer')")
    step_index: int | None = Field(default=None, description="Step index if mapped to linear step")
    span_id: str | None = Field(default=None, description="OpenInference span ID if mapped to DAG span")
    parent_id: str | None = Field(default=None, description="Immediate parent node ID in the graph")
    state: CausalNodeState = Field(
        default=CausalNodeState.HEALTHY,
        description="Assigned causal classification",
    )
    failure_category: FailureCategory | None = Field(
        default=None,
        description="Assigned failure category if node is faulty",
    )
    rule_matches: list[RuleMatch] = Field(
        default_factory=list,
        description="Rule engine matches that fired on this specific node",
    )
    evidence_records: list[EvidenceRecord] = Field(
        default_factory=list,
        description="Supporting evidence records attached to this node",
    )
    tainted_by: list[str] = Field(
        default_factory=list,
        description="List of upstream node IDs that propagated corrupted state to this node",
    )
    tainted_keys: list[str] = Field(
        default_factory=list,
        description="Specific state keys or variables corrupted upstream (e.g. ['source_count'])",
    )
    explanation: str = Field(
        default="",
        description="Human-readable causal explanation for why this node received its classification",
    )
    status: StepStatus = Field(
        default=StepStatus.SUCCESS,
        description="Execution status of the node",
    )
    latency_ms: float = Field(default=0.0, description="Latency of this node in milliseconds")
    tokens_total: int = Field(default=0, description="Total tokens consumed by this node")


class CausalChain(BaseModel):
    """An ordered path from a root cause to terminal impact or escaped verifier."""

    chain_id: str = Field(default_factory=lambda: f"chain_{uuid4().hex[:8]}")
    root_node_id: str = Field(description="Node ID where the causal chain originated")
    path: list[str] = Field(description="Ordered list of node IDs along the propagation path")
    escaped_node_id: str | None = Field(
        default=None,
        description="Node ID of escaped quality-control verifier if present",
    )
    impacted_agents: list[str] = Field(
        default_factory=list,
        description="Names of agents impacted along this chain",
    )
    primary_category: FailureCategory = Field(
        default=FailureCategory.UNKNOWN,
        description="Primary failure category associated with this causal chain",
    )


class CausalGraph(BaseModel):
    """Directed graph capturing fault propagation and causal relationships across agents."""

    run_id: str = Field(description="Trace / run identifier")
    nodes: dict[str, CausalNode] = Field(default_factory=dict, description="Lookup of all nodes by node_id")
    edges: list[list[str]] = Field(
        default_factory=list,
        description="Directed execution/dataflow edges [[from_node_id, to_node_id], ...]",
    )
    root_causes: list[str] = Field(default_factory=list, description="Node IDs classified as ROOT_CAUSE_ORIGIN")
    cascades: list[str] = Field(default_factory=list, description="Node IDs classified as CASCADING_FAILURE")
    passthroughs: list[str] = Field(default_factory=list, description="Node IDs classified as PASSTHROUGH_ESCAPED")
    healthy: list[str] = Field(default_factory=list, description="Node IDs classified as HEALTHY")
    causal_chains: list[CausalChain] = Field(default_factory=list, description="Identified causal propagation paths")
    primary_root_cause_node: str | None = Field(
        default=None,
        description="Node ID of the primary root cause origin",
    )
    summary: str = Field(default="", description="High-level narrative summary of the causal diagnosis")

    def get_root_cause(self) -> CausalNode | None:
        """Return the primary root cause node if one exists."""
        if self.primary_root_cause_node and self.primary_root_cause_node in self.nodes:
            return self.nodes[self.primary_root_cause_node]
        if self.root_causes and self.root_causes[0] in self.nodes:
            return self.nodes[self.root_causes[0]]
        return None

    def get_cascading_nodes(self) -> list[CausalNode]:
        """Return all nodes classified as cascading failures."""
        return [self.nodes[nid] for nid in self.cascades if nid in self.nodes]

    def get_passthrough_nodes(self) -> list[CausalNode]:
        """Return all quality-control nodes that failed to intercept upstream errors."""
        return [self.nodes[nid] for nid in self.passthroughs if nid in self.nodes]

    def get_impacted_node_count(self) -> int:
        """Return total count of non-healthy nodes (root causes + cascades + passthroughs)."""
        return len(self.root_causes) + len(self.cascades) + len(self.passthroughs)

    def format_ascii_tree(self) -> str:
        """Return an ASCII representation of the causal tree."""
        if not self.root_causes:
            return f"Run {self.run_id}: All {len(self.nodes)} nodes HEALTHY (No faults detected)."

        lines = [f"Causal Fault Propagation Tree (Run: {self.run_id})"]
        for chain in self.causal_chains:
            chain_labels: list[str] = []
            for nid in chain.path:
                node = self.nodes.get(nid)
                if not node:
                    continue
                if node.state == CausalNodeState.ROOT_CAUSE_ORIGIN:
                    tag = f"[ROOT: {node.name}]"
                elif node.state == CausalNodeState.CASCADING_FAILURE:
                    tag = f"[CASCADE: {node.name}]"
                elif node.state == CausalNodeState.PASSTHROUGH_ESCAPED:
                    tag = f"[ESCAPED: {node.name}]"
                else:
                    tag = f"[{node.name}]"
                chain_labels.append(tag)
            lines.append("  " + " -> ".join(chain_labels))
        return "\n".join(lines)


class CausalGraphEngine:
    """
    Engine that constructs causal fault trees from execution traces.

    Analyzes trace DAG topologies, rule engine firings, and agent handoffs to
    deterministically separate the root cause origin from downstream cascading
    failures and escaped verifiers.
    """

    def __init__(self) -> None:
        self._verifier_keywords: set[str] = {
            "verifier",
            "reviewer",
            "judge",
            "evaluator",
            "checker",
            "validator",
            "quality_control",
        }

    def is_verifier_node(self, name: str) -> bool:
        """Return True if an agent or node behaves as a verifier/quality checkpoint."""
        lower = name.lower()
        return any(kw in lower for kw in self._verifier_keywords)

    def build_graph(
        self,
        run_trace: RunTrace,
        analysis_bundle: AnalysisBundle | None = None,
        rule_matches: list[RuleMatch] | None = None,
        evidence: list[EvidenceRecord] | None = None,
    ) -> CausalGraph:
        """
        Build and classify the causal fault graph for a given RunTrace.

        Args:
            run_trace: The trace containing spans or steps.
            analysis_bundle: Optional precomputed Arbiter verdict bundle.
            rule_matches: Optional explicit list of fired RuleMatches.
            evidence: Optional explicit list of EvidenceRecords.

        Returns:
            CausalGraph with all nodes classified into causal states.
        """
        all_rules: list[RuleMatch] = list(rule_matches or [])
        all_evidence: list[EvidenceRecord] = list(evidence or [])

        if analysis_bundle:
            if not all_rules and analysis_bundle.rule_matches:
                all_rules.extend(analysis_bundle.rule_matches)
            if not all_evidence and analysis_bundle.evidence:
                all_evidence.extend(analysis_bundle.evidence)

        # Step 1: Initialize nodes and directed edges from RunTrace
        nodes: dict[str, CausalNode] = {}
        edges: list[list[str]] = []
        adj: dict[str, list[str]] = defaultdict(list)
        rev_adj: dict[str, list[str]] = defaultdict(list)

        # Check whether trace uses hierarchical spans or linear steps
        if run_trace.spans:
            nodes, edges, adj, rev_adj = self._build_nodes_from_spans(run_trace)
        elif run_trace.steps:
            nodes, edges, adj, rev_adj = self._build_nodes_from_steps(run_trace)
        else:
            # Empty trace
            return CausalGraph(run_id=run_trace.run_id, summary="Empty trace (0 nodes).")

        # Step 2: Associate rule matches and evidence records with nodes
        self._attach_rules_and_evidence(nodes, all_rules, all_evidence)

        # Step 3: Identify faulty nodes and candidate root causes
        candidate_faulty: list[str] = []
        for nid, node in nodes.items():
            has_error = node.status in (StepStatus.ERROR, StepStatus.FAILURE)
            has_rules = len(node.rule_matches) > 0
            if has_error or has_rules:
                candidate_faulty.append(nid)

        if not candidate_faulty:
            # Clean run: all nodes are HEALTHY
            healthy_list = list(nodes.keys())
            for node in nodes.values():
                node.state = CausalNodeState.HEALTHY
                node.explanation = "Node executed successfully with zero rule violations."
            return CausalGraph(
                run_id=run_trace.run_id,
                nodes=nodes,
                edges=edges,
                healthy=healthy_list,
                summary="Clean pipeline run. All nodes executed healthy with zero faults.",
            )

        # Step 4: Determine Root Cause Origin(s)
        # Prioritize the Arbiter's identified primary_agent if available
        primary_agent = analysis_bundle.primary_agent if analysis_bundle else None
        root_cause_nodes: list[str] = self._isolate_root_causes(
            nodes=nodes,
            candidate_faulty=candidate_faulty,
            rev_adj=rev_adj,
            primary_agent=primary_agent,
        )

        for rid in root_cause_nodes:
            r_node = nodes[rid]
            r_node.state = CausalNodeState.ROOT_CAUSE_ORIGIN
            cat = r_node.rule_matches[0].category if r_node.rule_matches else FailureCategory.EXECUTION
            if analysis_bundle and r_node.name == analysis_bundle.primary_agent:
                cat = analysis_bundle.primary_cause
            r_node.failure_category = cat
            rule_names = [rm.rule_id for rm in r_node.rule_matches]
            r_node.explanation = (
                f"Root cause origin: First node where failure occurred ({cat.value}). "
                f"Fired rules: {rule_names if rule_names else 'Execution Error'}."
            )

        # Step 5: Forward BFS to propagate cascading failures and identify escaped verifiers
        cascades: list[str] = []
        passthroughs: list[str] = []
        causal_chains: list[CausalChain] = []

        for root_id in root_cause_nodes:
            chain_path, chain_cascades, chain_escaped = self._propagate_fault(
                root_id=root_id,
                nodes=nodes,
                adj=adj,
            )
            for cid in chain_cascades:
                if cid not in cascades:
                    cascades.append(cid)
            for eid in chain_escaped:
                if eid not in passthroughs:
                    passthroughs.append(eid)

            if len(chain_path) > 1:
                impacted = [nodes[n].name for n in chain_path if n in nodes]
                chain = CausalChain(
                    root_node_id=root_id,
                    path=chain_path,
                    escaped_node_id=chain_escaped[0] if chain_escaped else None,
                    impacted_agents=impacted,
                    primary_category=nodes[root_id].failure_category or FailureCategory.UNKNOWN,
                )
                causal_chains.append(chain)

        # Step 6: Mark remaining nodes as HEALTHY
        healthy_nodes: list[str] = []
        for nid, node in nodes.items():
            if (
                node.state == CausalNodeState.HEALTHY
                and nid not in root_cause_nodes
                and nid not in cascades
                and nid not in passthroughs
            ):
                healthy_nodes.append(nid)
                node.explanation = "Healthy node: unaffected by upstream causal faults."

        primary_root = root_cause_nodes[0] if root_cause_nodes else None
        root_agent = nodes[primary_root].name if primary_root and primary_root in nodes else "Unknown"

        summary = (
            f"Fault propagation analyzed: Primary root cause identified at '{root_agent}' "
            f"({len(root_cause_nodes)} origin(s), {len(cascades)} cascading victim(s), "
            f"{len(passthroughs)} escaped verifier(s))."
        )

        return CausalGraph(
            run_id=run_trace.run_id,
            nodes=nodes,
            edges=edges,
            root_causes=root_cause_nodes,
            cascades=cascades,
            passthroughs=passthroughs,
            healthy=healthy_nodes,
            causal_chains=causal_chains,
            primary_root_cause_node=primary_root,
            summary=summary,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Graph Construction Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_nodes_from_spans(
        self, trace: RunTrace
    ) -> tuple[dict[str, CausalNode], list[list[str]], dict[str, list[str]], dict[str, list[str]]]:
        """Convert TraceSpan DAG into CausalNodes and directed edges."""
        nodes: dict[str, CausalNode] = {}
        edges: list[list[str]] = []
        adj: dict[str, list[str]] = defaultdict(list)
        rev_adj: dict[str, list[str]] = defaultdict(list)

        for span in trace.spans:
            node = CausalNode(
                node_id=span.span_id,
                name=span.name,
                span_id=span.span_id,
                parent_id=span.parent_span_id,
                status=span.status,
                latency_ms=span.latency_ms,
                tokens_total=span.tokens.total,
            )
            nodes[span.span_id] = node

        # Build edges: parent-child invocation / containment edges
        for span in trace.spans:
            if span.parent_span_id and span.parent_span_id in nodes:
                edges.append([span.parent_span_id, span.span_id])
                adj[span.parent_span_id].append(span.span_id)
                rev_adj[span.span_id].append(span.parent_span_id)

        # Cyclic reflection edges for iterative loops (iteration_index k -> k+1)
        spans_by_name: dict[str, list[TraceSpan]] = defaultdict(list)
        for span in trace.spans:
            spans_by_name[span.name].append(span)

        for _name, named_spans in spans_by_name.items():
            if len(named_spans) > 1:
                sorted_spans = sorted(named_spans, key=lambda s: s.iteration_index)
                for i in range(len(sorted_spans) - 1):
                    s_curr, s_next = sorted_spans[i], sorted_spans[i + 1]
                    if s_curr.iteration_index < s_next.iteration_index:
                        if [s_curr.span_id, s_next.span_id] not in edges:
                            edges.append([s_curr.span_id, s_next.span_id])
                            adj[s_curr.span_id].append(s_next.span_id)
                            rev_adj[s_next.span_id].append(s_curr.span_id)

        return nodes, edges, adj, rev_adj

    def _build_nodes_from_steps(
        self, trace: RunTrace
    ) -> tuple[dict[str, CausalNode], list[list[str]], dict[str, list[str]], dict[str, list[str]]]:
        """Convert linear AgentStep sequence into CausalNodes and directed edges."""
        nodes: dict[str, CausalNode] = {}
        edges: list[list[str]] = []
        adj: dict[str, list[str]] = defaultdict(list)
        rev_adj: dict[str, list[str]] = defaultdict(list)

        sorted_steps = sorted(trace.steps, key=lambda s: s.step)
        for s in sorted_steps:
            node_id = f"step_{s.step}"
            node = CausalNode(
                node_id=node_id,
                name=s.agent,
                step_index=s.step,
                status=s.status,
                latency_ms=s.latency_ms,
                tokens_total=s.tokens.total,
            )
            nodes[node_id] = node

        # Sequential execution edges
        for i in range(len(sorted_steps) - 1):
            u_id = f"step_{sorted_steps[i].step}"
            v_id = f"step_{sorted_steps[i + 1].step}"
            edges.append([u_id, v_id])
            adj[u_id].append(v_id)
            rev_adj[v_id].append(u_id)

        return nodes, edges, adj, rev_adj

    def _attach_rules_and_evidence(
        self,
        nodes: dict[str, CausalNode],
        rule_matches: list[RuleMatch],
        evidence: list[EvidenceRecord],
    ) -> None:
        """Map rules and evidence records to the corresponding nodes."""
        for rule in rule_matches:
            target_node: CausalNode | None = None
            if rule.step is not None:
                step_key = f"step_{rule.step}"
                if step_key in nodes:
                    target_node = nodes[step_key]
            if not target_node and rule.agent:
                # Find matching node by agent name
                for n in nodes.values():
                    if n.name == rule.agent:
                        target_node = n
                        break
            if target_node:
                target_node.rule_matches.append(rule)

        for ev in evidence:
            target_node = None
            if ev.step is not None:
                step_key = f"step_{ev.step}"
                if step_key in nodes:
                    target_node = nodes[step_key]
            if not target_node and ev.agent:
                for n in nodes.values():
                    if n.name == ev.agent:
                        target_node = n
                        break
            if target_node:
                target_node.evidence_records.append(ev)

    # ─────────────────────────────────────────────────────────────────────────
    # Root Cause & Fault Propagation Logic
    # ─────────────────────────────────────────────────────────────────────────

    def _isolate_root_causes(
        self,
        nodes: dict[str, CausalNode],
        candidate_faulty: list[str],
        rev_adj: dict[str, list[str]],
        primary_agent: str | None = None,
    ) -> list[str]:
        """Find the earliest un-tainted faulty node(s) in the graph."""
        # If primary_agent was established by Arbiter, locate its earliest faulty node
        if primary_agent:
            agent_faults = [nid for nid in candidate_faulty if nodes[nid].name == primary_agent]
            if agent_faults:
                # Sort by step index or order
                agent_faults.sort(key=lambda nid: (nodes[nid].step_index or 0, nid))
                return [agent_faults[0]]

        # Otherwise, identify nodes with no faulty upstream ancestors
        root_causes: list[str] = []
        for fid in candidate_faulty:
            # Check if any upstream ancestor is also in candidate_faulty
            upstream_faulty = False
            queue: deque[str] = deque(rev_adj.get(fid, []))
            visited: set[str] = set()
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                if curr in candidate_faulty:
                    upstream_faulty = True
                    break
                queue.extend(rev_adj.get(curr, []))

            if not upstream_faulty:
                root_causes.append(fid)

        if not root_causes and candidate_faulty:
            # Fallback to earliest candidate by step index or insertion
            candidate_faulty.sort(key=lambda nid: (nodes[nid].step_index or 0, nid))
            root_causes.append(candidate_faulty[0])

        return root_causes

    def _propagate_fault(
        self,
        root_id: str,
        nodes: dict[str, CausalNode],
        adj: dict[str, list[str]],
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Perform forward BFS from root_id to mark cascading failures and escaped verifiers.

        Returns:
            (chain_path, cascading_node_ids, escaped_node_ids)
        """
        chain_path: list[str] = [root_id]
        cascading_nodes: list[str] = []
        escaped_nodes: list[str] = []

        root_node = nodes[root_id]
        tainted_keys: list[str] = []
        for rm in root_node.rule_matches:
            if "information_loss" in rm.rule_id or "R-IL" in rm.rule_id:
                tainted_keys.extend(["source_count", "sources", "entities", "research_findings"])
            elif "tool_failure" in rm.rule_id or "R-EX" in rm.rule_id:
                tainted_keys.extend(["tool_result", "output"])

        queue: deque[str] = deque(adj.get(root_id, []))
        visited: set[str] = {root_id}

        while queue:
            curr_id = queue.popleft()
            if curr_id in visited:
                continue
            visited.add(curr_id)

            curr_node = nodes.get(curr_id)
            if not curr_node:
                continue

            chain_path.append(curr_id)
            curr_node.tainted_by.append(root_id)
            curr_node.tainted_keys = list(set(curr_node.tainted_keys + tainted_keys))

            # Check if node is a verifier / quality checkpoint
            if self.is_verifier_node(curr_node.name):
                # Did the verifier pass or approve despite corrupted upstream state?
                has_verifier_rule = any(
                    "verifier" in rm.rule_id or "R-VF" in rm.rule_id
                    for rm in curr_node.rule_matches
                )
                if curr_node.status == StepStatus.SUCCESS or has_verifier_rule:
                    curr_node.state = CausalNodeState.PASSTHROUGH_ESCAPED
                    curr_node.failure_category = FailureCategory.VERIFICATION
                    curr_node.explanation = (
                        f"Quality-control escaped: '{curr_node.name}' failed to intercept "
                        f"upstream failure from '{root_node.name}'."
                    )
                    escaped_nodes.append(curr_id)
            else:
                # Downstream worker agent
                curr_node.state = CausalNodeState.CASCADING_FAILURE
                cat = FailureCategory.REASONING
                if curr_node.rule_matches:
                    cat = curr_node.rule_matches[0].category
                elif curr_node.status in (StepStatus.ERROR, StepStatus.FAILURE):
                    cat = FailureCategory.EXECUTION
                curr_node.failure_category = cat
                curr_node.explanation = (
                    f"Cascading failure: Downstream from root cause '{root_node.name}'. "
                    f"Operated on corrupted inputs: {tainted_keys if tainted_keys else 'upstream state'}."
                )
                cascading_nodes.append(curr_id)

            # Continue traversing downstream
            for nxt in adj.get(curr_id, []):
                if nxt not in visited:
                    queue.append(nxt)

        return chain_path, cascading_nodes, escaped_nodes
