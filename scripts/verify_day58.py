"""
scripts/verify_day58.py — Day 58 Verification Gate
===================================================
Phase C Sign-Off: LangGraph & LangChain Auto-Instrumentation (AgentLens v2.0)

Checks:
  1. Callback Handler Surface & Exports (agentlens.integrations.langchain & agentlens top-level).
  2. Zero-Code Linear StateGraph Auto-Instrumentation (2 nodes, 0 manual decorators).
  3. Conditional Routing Branch Capture & Router Span Classification (classifier -> billing/tech).
  4. Cyclic Reflection Loop Tracking with iteration_index (drafter -> critic -> drafter -> critic).
  5. Nested Tool Invocation Interception & Arbiter Rule Detection (tool_failure_v1).
  6. LLM Token Aggregation from on_llm_end into step.tokens & trace.total_tokens.
  7. Graph Node Exception Capture & Database Persistence via save_to_db=True.
  8. Static Analysis (Ruff/Mypy Clean) & Day 43 Frozen Benchmark Gate (16/20 = 80.0%).
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.graph import END, START, StateGraph

import agentlens
from agentlens.integrations.langchain import AgentLensCallbackHandler
from schema.models import FailureCategory, PriorityLevel, StepStatus
from schema.span import SpanType
from storage.db import DatabaseManager


class LinearState(TypedDict, total=False):
    topic: str
    draft: str
    approved: bool
    source_count: int
    entity_count: int


class RoutingState(TypedDict, total=False):
    input_text: str
    intent: str
    output: str


class CyclicState(TypedDict, total=False):
    topic: str
    iteration: int
    draft: str
    approved: bool
    source_count: int


class SimpleState(TypedDict, total=False):
    start: bool
    data: str
    msg: str


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 58 VERIFICATION GATE")
    print("  LangGraph & LangChain Auto-Instrumentation (Phase C Day 58)")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Callback Handler Surface & Exports
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert hasattr(agentlens, "AgentLensCallbackHandler")
        assert agentlens.AgentLensCallbackHandler is AgentLensCallbackHandler

        handler = AgentLensCallbackHandler(workflow="check1_wf")
        assert handler.workflow == "check1_wf"
        assert handler.trace.workflow == "check1_wf"
        assert len(handler.get_steps()) == 0
        assert len(handler.get_spans()) == 0

        print("[PASS] Check 1: Callback Handler Surface & Exports (AgentLensCallbackHandler)")
    except Exception as exc:
        print(f"[FAIL] Check 1: Callback handler export check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Zero-Code Linear StateGraph Auto-Instrumentation
    # ─────────────────────────────────────────────────────────────────────────
    try:

        def researcher(state: LinearState) -> dict[str, Any]:
            return {"draft": f"Research on {state['topic']}", "source_count": 3, "entity_count": 2}

        def reviewer(state: LinearState) -> dict[str, Any]:
            return {"approved": True, "source_count": 3}

        builder_linear = StateGraph(LinearState)
        builder_linear.add_node("researcher", researcher)
        builder_linear.add_node("reviewer", reviewer)
        builder_linear.add_edge(START, "researcher")
        builder_linear.add_edge("researcher", "reviewer")
        builder_linear.add_edge("reviewer", END)
        graph_linear = builder_linear.compile()

        handler = AgentLensCallbackHandler(workflow="linear_research_pipeline")
        result = graph_linear.invoke(
            {"topic": "Quantum Computing"}, config={"callbacks": [handler]}
        )

        assert result["approved"] is True
        trace = handler.get_trace()

        assert len(trace.steps) == 2, f"Expected 2 steps, got {len(trace.steps)}"
        assert len(trace.spans) == 2, f"Expected 2 spans, got {len(trace.spans)}"
        assert trace.steps[0].agent == "researcher"
        assert trace.steps[1].agent == "reviewer"
        assert trace.status == StepStatus.SUCCESS

        bundle = handler.evaluate()
        assert bundle.priority_level == PriorityLevel.P5
        assert bundle.primary_agent is None

        print("[PASS] Check 2: Zero-Code Linear StateGraph Auto-Instrumentation")
    except Exception as exc:
        print(f"[FAIL] Check 2: Linear StateGraph check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Conditional Routing Branch Capture & Router Span Classification
    # ─────────────────────────────────────────────────────────────────────────
    try:

        def classifier(state: RoutingState) -> dict[str, Any]:
            intent = "billing" if "invoice" in state["input_text"].lower() else "technical"
            return {"intent": intent}

        def route_intent(state: RoutingState) -> Literal["billing_agent", "tech_agent"]:
            return "billing_agent" if state["intent"] == "billing" else "tech_agent"

        def billing_agent(state: RoutingState) -> dict[str, Any]:
            return {"output": "Billing resolution"}

        def tech_agent(state: RoutingState) -> dict[str, Any]:
            return {"output": "Technical fix"}

        builder_routing = StateGraph(RoutingState)
        builder_routing.add_node("classifier", classifier)
        builder_routing.add_node("billing_agent", billing_agent)
        builder_routing.add_node("tech_agent", tech_agent)
        builder_routing.add_edge(START, "classifier")
        builder_routing.add_conditional_edges("classifier", route_intent)
        builder_routing.add_edge("billing_agent", END)
        builder_routing.add_edge("tech_agent", END)
        graph_routing = builder_routing.compile()

        # Execute Billing branch
        h_billing = AgentLensCallbackHandler(workflow="routing_pipeline")
        res_b = graph_routing.invoke(
            {"input_text": "Need help with invoice #123"},
            config={"callbacks": [h_billing]},
        )
        assert res_b["output"] == "Billing resolution"
        trace_b = h_billing.get_trace()
        assert len(trace_b.steps) == 2, f"Expected 2 agent steps, got {len(trace_b.steps)}"
        assert [s.agent for s in trace_b.steps] == ["classifier", "billing_agent"]

        # Check router span exists
        router_spans = [s for s in trace_b.spans if s.span_type == SpanType.ROUTER]
        assert len(router_spans) >= 1, "Expected at least 1 router span"
        assert router_spans[0].name == "route_intent"

        print("[PASS] Check 3: Conditional Routing Branch Capture & Router Span Classification")
    except Exception as exc:
        print(f"[FAIL] Check 3: Conditional routing check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Cyclic Reflection Loop Tracking with iteration_index
    # ─────────────────────────────────────────────────────────────────────────
    try:

        def drafter(state: CyclicState) -> dict[str, Any]:
            cur_iter = state.get("iteration", 0) + 1
            return {
                "iteration": cur_iter,
                "draft": f"Draft iteration {cur_iter}",
                "source_count": 2,
            }

        def critic(state: CyclicState) -> dict[str, Any]:
            return {"approved": state["iteration"] >= 2}

        def should_continue(state: CyclicState) -> Literal["drafter", "__end__"]:
            return "__end__" if state["approved"] else "drafter"

        builder_cyclic = StateGraph(CyclicState)
        builder_cyclic.add_node("drafter", drafter)
        builder_cyclic.add_node("critic", critic)
        builder_cyclic.add_edge(START, "drafter")
        builder_cyclic.add_edge("drafter", "critic")
        builder_cyclic.add_conditional_edges(
            "critic", should_continue, {"drafter": "drafter", "__end__": END}
        )
        graph_cyclic = builder_cyclic.compile()

        handler = AgentLensCallbackHandler(workflow="reflection_loop_pipeline")
        res = graph_cyclic.invoke(
            {"topic": "AI Safety", "iteration": 0, "draft": "", "approved": False},
            config={"callbacks": [handler]},
        )
        assert res["approved"] is True
        assert res["iteration"] == 2

        trace = handler.get_trace()
        assert len(trace.steps) == 4, f"Expected 4 steps, got {len(trace.steps)}"
        assert [s.agent for s in trace.steps] == ["drafter", "critic", "drafter", "critic"]

        drafter_spans = [s for s in trace.spans if s.name == "drafter"]
        assert len(drafter_spans) == 2
        assert drafter_spans[0].iteration_index == 0
        assert drafter_spans[1].iteration_index == 1

        print("[PASS] Check 4: Cyclic Reflection Loop Tracking with iteration_index")
    except Exception as exc:
        print(f"[FAIL] Check 4: Cyclic reflection loop check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Nested Tool Invocation Interception & Arbiter Rule Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        handler = AgentLensCallbackHandler(workflow="tool_pipeline")
        root_id = uuid4()
        node_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(serialized={"name": "test_graph"}, inputs={}, run_id=root_id)
        handler.on_chain_start(
            serialized={"name": "financial_analyst"},
            inputs={"query": "fetch balance sheet"},
            run_id=node_id,
            parent_run_id=root_id,
            metadata={"langgraph_node": "financial_analyst"},
        )
        handler.on_tool_start(
            serialized={"name": "fetch_api"},
            input_str="https://api.finance.com/q3",
            run_id=tool_id,
            parent_run_id=node_id,
        )
        handler.on_tool_error(
            error=ConnectionError("API rate limit 429"),
            run_id=tool_id,
            parent_run_id=node_id,
        )
        handler.on_chain_end(
            outputs={"error": "Failed to fetch financials"},
            run_id=node_id,
            parent_run_id=root_id,
        )
        handler.on_chain_end(outputs={}, run_id=root_id)

        trace = handler.get_trace()
        assert len(trace.steps) == 1
        agent_step = trace.steps[0]
        assert len(agent_step.tool_calls) == 1
        assert agent_step.tool_calls[0]["name"] == "fetch_api"
        assert "429" in agent_step.tool_calls[0]["error"]

        bundle = handler.evaluate()
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_agent == "financial_analyst"

        print("[PASS] Check 5: Nested Tool Invocation Interception & Arbiter Rule Detection")
    except Exception as exc:
        print(f"[FAIL] Check 5: Tool invocation check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: LLM Token Aggregation from on_llm_end into step.tokens
    # ─────────────────────────────────────────────────────────────────────────
    try:
        handler = AgentLensCallbackHandler(workflow="llm_pipeline")
        root_id = uuid4()
        node_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(serialized={"name": "test_graph"}, inputs={}, run_id=root_id)
        handler.on_chain_start(
            serialized={"name": "writer"},
            inputs={"draft": "input"},
            run_id=node_id,
            parent_run_id=root_id,
            metadata={"langgraph_node": "writer"},
        )
        handler.on_llm_start(
            serialized={"name": "chat_model"},
            prompts=["Write an essay"],
            run_id=llm_id,
            parent_run_id=node_id,
        )

        class MockLLMResult:
            llm_output = {
                "token_usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 80,
                    "total_tokens": 200,
                }
            }
            generations: list[Any] = []

        handler.on_llm_end(response=MockLLMResult(), run_id=llm_id, parent_run_id=node_id)
        handler.on_chain_end(outputs={"essay": "Done"}, run_id=node_id, parent_run_id=root_id)
        handler.on_chain_end(outputs={}, run_id=root_id)

        trace = handler.get_trace()
        assert len(trace.steps) == 1
        assert trace.steps[0].tokens.prompt == 120
        assert trace.steps[0].tokens.completion == 80
        assert trace.steps[0].tokens.total == 200
        assert trace.total_tokens == 200

        print("[PASS] Check 6: LLM Token Aggregation from on_llm_end into step.tokens")
    except Exception as exc:
        print(f"[FAIL] Check 6: LLM token aggregation check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Graph Node Exception Capture & Database Persistence
    # ─────────────────────────────────────────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            temp_db = Path(tmp_dir) / "langgraph_verify.db"
            agentlens.init(db_path=temp_db)

            def buggy_agent(state: SimpleState) -> dict[str, Any]:
                raise RuntimeError("Node computation fault")

            builder_buggy = StateGraph(SimpleState)
            builder_buggy.add_node("buggy_agent", buggy_agent)
            builder_buggy.add_edge(START, "buggy_agent")
            builder_buggy.add_edge("buggy_agent", END)
            graph_buggy = builder_buggy.compile()

            handler = AgentLensCallbackHandler(
                workflow="verify_error_pipeline",
                save_to_db=True,
            )

            try:
                graph_buggy.invoke({"data": "test"}, config={"callbacks": [handler]})
            except RuntimeError:
                pass

            trace = handler.get_trace()
            assert len(trace.steps) == 1
            assert trace.steps[0].status == StepStatus.ERROR
            assert "RuntimeError" in str(trace.steps[0].error)
            assert trace.status == StepStatus.FAILURE

            # Verify saved into DB
            db = DatabaseManager(db_path=temp_db)
            row = db.get_run(trace.run_id)
            assert row is not None
            assert row["workflow"] == "verify_error_pipeline"
            assert row["status"] == "FAILURE"

            db.close()
            os.environ.pop("DB_PATH", None)

        print("[PASS] Check 7: Graph Node Exception Capture & Database Persistence")
    except Exception as exc:
        print(f"[FAIL] Check 7: Error capture & database persistence check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis & Day 43 Frozen Benchmark Gate
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # 1. Ruff check
        ruff_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "agentlens/integrations/langchain.py",
                "tests/test_langgraph_integration.py",
                "scripts/verify_day58.py",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if ruff_res.returncode != 0:
            raise RuntimeError(f"Ruff check failed: {ruff_res.stdout} {ruff_res.stderr}")

        # 2. Mypy check
        mypy_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "agentlens/integrations/langchain.py",
                "tests/test_langgraph_integration.py",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if mypy_res.returncode != 0:
            raise RuntimeError(f"Mypy check failed: {mypy_res.stdout} {mypy_res.stderr}")

        # 3. Day 43 Benchmark Regression Check
        bench_script = REPO_ROOT / "scripts" / "run_day43_regression.py"
        bench_res = subprocess.run(
            [sys.executable, str(bench_script)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if bench_res.returncode != 0 or "16/20 (80.0%)" not in bench_res.stdout:
            raise RuntimeError(
                f"Day 43 benchmark regression failed:\n{bench_res.stdout}\n{bench_res.stderr}"
            )

        print(
            "[PASS] Check 8: Static Analysis (Ruff/Mypy clean) & Day 43 Benchmark Preserved (16/20 = 80.0%)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 8: Static analysis / regression gate failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    if all_passed:
        print(">>> ALL 8 DAY 58 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> AGENTLENS LANGGRAPH AUTO-INSTRUMENTATION CERTIFIED! <<<")
    else:
        print(">>> SOME DAY 58 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
