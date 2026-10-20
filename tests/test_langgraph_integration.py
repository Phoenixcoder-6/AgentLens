"""
tests/test_langgraph_integration.py — Day 58: LangGraph & LangChain Integration Tests
======================================================================================
Tests:
  1. Standard linear StateGraph execution with zero decorators.
  2. Conditional routing DAG branching (router -> branch_a vs branch_b).
  3. Cyclic reflection loop tracking with iteration_index.
  4. Tool invocation interception and nested tool failure rule detection.
  5. LLM token aggregation from on_llm_end into step.tokens.
  6. Graph node exception capture and failure status propagation.
  7. Automated Arbiter evaluation on auto-instrumented traces.
  8. Database persistence via save_to_db=True.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

import pytest
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


class TestLangGraphIntegration:
    def test_standard_linear_stategraph(self) -> None:
        """Verify standard linear LangGraph executes and traces with zero manual decorators."""

        def researcher(state: LinearState) -> dict[str, Any]:
            return {
                "draft": f"Research draft on {state['topic']}",
                "source_count": 3,
                "entity_count": 2,
            }

        def reviewer(state: LinearState) -> dict[str, Any]:
            return {"approved": True, "source_count": 3}

        builder = StateGraph(LinearState)
        builder.add_node("researcher", researcher)
        builder.add_node("reviewer", reviewer)
        builder.add_edge(START, "researcher")
        builder.add_edge("researcher", "reviewer")
        builder.add_edge("reviewer", END)
        graph = builder.compile()

        handler = AgentLensCallbackHandler(workflow="linear_research_pipeline")
        result = graph.invoke({"topic": "Quantum Computing"}, config={"callbacks": [handler]})

        assert result["approved"] is True
        trace = handler.get_trace()

        assert len(trace.steps) == 2
        assert len(trace.spans) == 2
        assert trace.steps[0].agent == "researcher"
        assert trace.steps[1].agent == "reviewer"
        assert trace.status == StepStatus.SUCCESS

        bundle = handler.evaluate()
        assert bundle.priority_level == PriorityLevel.P5
        assert bundle.primary_agent is None

    def test_conditional_routing_dag_branching(self) -> None:
        """Verify conditional branch routing captures only the executed branch."""

        def classifier(state: RoutingState) -> dict[str, Any]:
            intent = "billing" if "invoice" in state["input_text"].lower() else "technical"
            return {"intent": intent}

        def route_intent(state: RoutingState) -> Literal["billing_agent", "tech_agent"]:
            return "billing_agent" if state["intent"] == "billing" else "tech_agent"

        def billing_agent(state: RoutingState) -> dict[str, Any]:
            return {"output": "Billing resolution"}

        def tech_agent(state: RoutingState) -> dict[str, Any]:
            return {"output": "Technical fix"}

        builder = StateGraph(RoutingState)
        builder.add_node("classifier", classifier)
        builder.add_node("billing_agent", billing_agent)
        builder.add_node("tech_agent", tech_agent)
        builder.add_edge(START, "classifier")
        builder.add_conditional_edges("classifier", route_intent)
        builder.add_edge("billing_agent", END)
        builder.add_edge("tech_agent", END)
        graph = builder.compile()

        # Branch 1: Billing
        handler_billing = AgentLensCallbackHandler(workflow="routing_pipeline")
        res_b = graph.invoke(
            {"input_text": "Need help with invoice #123"}, config={"callbacks": [handler_billing]}
        )
        assert res_b["output"] == "Billing resolution"
        trace_b = handler_billing.get_trace()
        assert len(trace_b.steps) == 2
        assert [s.agent for s in trace_b.steps] == ["classifier", "billing_agent"]

        # Branch 2: Technical
        handler_tech = AgentLensCallbackHandler(workflow="routing_pipeline")
        res_t = graph.invoke(
            {"input_text": "App server crashed with 500"}, config={"callbacks": [handler_tech]}
        )
        assert res_t["output"] == "Technical fix"
        trace_t = handler_tech.get_trace()
        assert len(trace_t.steps) == 2
        assert [s.agent for s in trace_t.steps] == ["classifier", "tech_agent"]

    def test_cyclic_reflection_loop_tracking(self) -> None:
        """Verify multi-iteration cyclic reflection loops capture correct iteration_index."""

        def drafter(state: CyclicState) -> dict[str, Any]:
            cur_iter = state.get("iteration", 0) + 1
            return {
                "iteration": cur_iter,
                "draft": f"Draft iteration {cur_iter}",
                "source_count": 2,
            }

        def critic(state: CyclicState) -> dict[str, Any]:
            approved = state["iteration"] >= 2
            return {"approved": approved}

        def should_continue(state: CyclicState) -> Literal["drafter", "__end__"]:
            return "__end__" if state["approved"] else "drafter"

        builder = StateGraph(CyclicState)
        builder.add_node("drafter", drafter)
        builder.add_node("critic", critic)
        builder.add_edge(START, "drafter")
        builder.add_edge("drafter", "critic")
        builder.add_conditional_edges(
            "critic", should_continue, {"drafter": "drafter", "__end__": END}
        )
        graph = builder.compile()

        handler = AgentLensCallbackHandler(workflow="reflection_loop_pipeline")
        res = graph.invoke(
            {"topic": "AI Safety", "iteration": 0, "draft": "", "approved": False},
            config={"callbacks": [handler]},
        )

        assert res["approved"] is True
        assert res["iteration"] == 2

        trace = handler.get_trace()
        assert len(trace.steps) == 4
        assert [s.agent for s in trace.steps] == ["drafter", "critic", "drafter", "critic"]

        # Verify iteration indices in spans
        drafter_spans = [s for s in trace.spans if s.name == "drafter"]
        assert len(drafter_spans) == 2
        assert drafter_spans[0].iteration_index == 0
        assert drafter_spans[1].iteration_index == 1

    def test_tool_invocation_interception_and_rule_detection(self) -> None:
        """Verify on_tool_start / on_tool_error captures tool crash and triggers tool_failure_v1."""
        handler = AgentLensCallbackHandler(workflow="tool_pipeline")
        root_run_id = uuid4()
        node_run_id = uuid4()
        tool_run_id = uuid4()

        # Simulate root graph start
        handler.on_chain_start(serialized={"name": "test_graph"}, inputs={}, run_id=root_run_id)

        # Simulate agent node start
        handler.on_chain_start(
            serialized={"name": "financial_analyst"},
            inputs={"query": "fetch balance sheet"},
            run_id=node_run_id,
            parent_run_id=root_run_id,
            metadata={"langgraph_node": "financial_analyst"},
        )

        # Simulate nested tool start and error
        handler.on_tool_start(
            serialized={"name": "fetch_api"},
            input_str="https://api.finance.com/q3",
            run_id=tool_run_id,
            parent_run_id=node_run_id,
        )
        handler.on_tool_error(
            error=ConnectionError("API rate limit 429"),
            run_id=tool_run_id,
            parent_run_id=node_run_id,
        )

        # Simulate agent node end
        handler.on_chain_end(
            outputs={"error": "Failed to fetch financials"},
            run_id=node_run_id,
            parent_run_id=root_run_id,
        )
        handler.on_chain_end(outputs={}, run_id=root_run_id)

        trace = handler.get_trace()
        assert len(trace.steps) == 1
        agent_step = trace.steps[0]
        assert len(agent_step.tool_calls) == 1
        assert agent_step.tool_calls[0]["name"] == "fetch_api"
        assert "429" in agent_step.tool_calls[0]["error"]

        # Tool span
        tool_spans = [s for s in trace.spans if s.span_type == SpanType.TOOL]
        assert len(tool_spans) == 1
        assert tool_spans[0].status == StepStatus.ERROR

        # Arbiter evaluation: RuleEngine detects tool_failure_v1
        bundle = handler.evaluate()
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_agent == "financial_analyst"

    def test_llm_token_aggregation(self) -> None:
        """Verify on_llm_start / on_llm_end attributes tokens to parent agent step."""
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

        # Mock LLMResult
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

    def test_graph_node_error_capture(self) -> None:
        """Verify exception in LangGraph node marks step and trace as ERROR."""

        def buggy_node(state: SimpleState) -> dict[str, Any]:
            raise ValueError("Invalid state payload")

        builder = StateGraph(SimpleState)
        builder.add_node("buggy_node", buggy_node)
        builder.add_edge(START, "buggy_node")
        builder.add_edge("buggy_node", END)
        graph = builder.compile()

        handler = AgentLensCallbackHandler(workflow="error_pipeline")

        with pytest.raises(ValueError, match="Invalid state payload"):
            graph.invoke({"data": "test"}, config={"callbacks": [handler]})

        trace = handler.get_trace()
        assert len(trace.steps) == 1
        assert trace.steps[0].status == StepStatus.ERROR
        assert "ValueError" in str(trace.steps[0].error)
        assert trace.status == StepStatus.FAILURE

    def test_database_persistence_via_save_to_db(self, tmp_path: Path) -> None:
        """Verify save_to_db=True persists the trace into SQLite."""
        db_file = tmp_path / "langgraph_test.db"
        agentlens.init(db_path=db_file)

        def simple_agent(state: SimpleState) -> dict[str, Any]:
            return {"msg": "persisted"}

        builder = StateGraph(SimpleState)
        builder.add_node("simple_agent", simple_agent)
        builder.add_edge(START, "simple_agent")
        builder.add_edge("simple_agent", END)
        graph = builder.compile()

        handler = AgentLensCallbackHandler(workflow="persistence_graph", save_to_db=True)
        graph.invoke({"start": True}, config={"callbacks": [handler]})

        trace = handler.get_trace()

        # Read back from database
        db = DatabaseManager(db_path=db_file)
        row = db.get_run(trace.run_id)
        assert row is not None
        assert row["workflow"] == "persistence_graph"
        assert row["run_id"] == trace.run_id

    def test_openinference_dag_span_hierarchy(self) -> None:
        """Verify OpenInference spans preserve parent_span_id hierarchy across nodes and tools."""
        handler = AgentLensCallbackHandler(workflow="dag_pipeline")
        root_id = uuid4()
        node_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start(serialized={"name": "root_graph"}, inputs={}, run_id=root_id)
        handler.on_chain_start(
            serialized={"name": "analyst"},
            inputs={"q": "market data"},
            run_id=node_id,
            parent_run_id=root_id,
            metadata={"langgraph_node": "analyst"},
        )
        handler.on_tool_start(
            serialized={"name": "sql_query"},
            input_str="SELECT * FROM markets",
            run_id=tool_id,
            parent_run_id=node_id,
        )
        handler.on_tool_end(output={"rows": 42}, run_id=tool_id, parent_run_id=node_id)
        handler.on_chain_end(outputs={"analysis": "bullish"}, run_id=node_id, parent_run_id=root_id)
        handler.on_chain_end(outputs={}, run_id=root_id)

        spans = handler.get_spans()
        assert len(spans) == 2
        tool_span = next(s for s in spans if s.span_type == SpanType.TOOL)
        agent_span = next(s for s in spans if s.span_type == SpanType.AGENT)

        assert tool_span.parent_span_id == str(node_id)
        assert agent_span.span_id == str(node_id)
        assert agent_span.parent_span_id == str(root_id)
