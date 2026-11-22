"""
tests/test_swarm_integrations.py — Day 59: CrewAI & AutoGen Interoperability Tests
===================================================================================
Tests:
  1. CrewAI sequential execution with task_callback and step_callback.
  2. CrewAI hierarchical delegation with manager-worker parent-child spans.
  3. CrewAI nested tool failure interception and Arbiter rule detection (tool_failure_v1).
  4. CrewAI database persistence via save_to_db=True.
  5. AutoGen two-agent conversational turn exchange.
  6. AutoGen group chat multi-speaker round-robin with cyclic iteration_index tracking.
  7. AutoGen code execution block extraction and exitcode: 1 error attribution.
  8. AutoGen context manager monitoring and database persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import agentlens
from agentlens.integrations.autogen import AgentLensAutoGenHook
from agentlens.integrations.crewai import AgentLensCrewHook
from schema.models import FailureCategory, PriorityLevel, StepStatus, TokenUsage
from schema.span import SpanType
from storage.db import DatabaseManager

# ─────────────────────────────────────────────────────────────────────────────
# Duck-typed Mocks for CrewAI
# ─────────────────────────────────────────────────────────────────────────────


class MockCrewAgent:
    def __init__(self, role: str, goal: str = "Achieve goal") -> None:
        self.role = role
        self.goal = goal


class MockTaskOutput:
    def __init__(
        self,
        description: str,
        raw: str,
        agent: MockCrewAgent | str,
        json_dict: dict[str, Any] | None = None,
    ) -> None:
        self.description = description
        self.raw = raw
        self.agent = agent
        self.json_dict = json_dict or {}


class MockCrew:
    def __init__(
        self,
        agents: list[MockCrewAgent],
        tasks: list[Any],
        process: str = "sequential",
        manager_agent: MockCrewAgent | None = None,
        step_callback: Any = None,
        task_callback: Any = None,
    ) -> None:
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.manager_agent = manager_agent
        self.step_callback = step_callback
        self.task_callback = task_callback

    def kickoff(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {}
        for idx, task in enumerate(self.tasks):
            # Optional step callback simulation
            if self.step_callback:
                self.step_callback(
                    {"tool": "search_db", "tool_input": "query", "result": "results"}
                )

            out_data = {"task_id": idx, "summary": f"Result for {task.description}"}
            t_out = MockTaskOutput(
                description=task.description,
                raw=f"Completed {task.description}",
                agent=task.agent,
                json_dict=out_data,
            )
            if self.task_callback:
                self.task_callback(t_out)
            result = out_data
        return result


class MockTask:
    def __init__(self, description: str, agent: MockCrewAgent) -> None:
        self.description = description
        self.agent = agent


# ─────────────────────────────────────────────────────────────────────────────
# Duck-typed Mocks for AutoGen
# ─────────────────────────────────────────────────────────────────────────────


class MockAutoGenAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hooks: dict[str, list[Any]] = {}

    def register_hook(self, hookable_method: str, hook_func: Any) -> None:
        self.hooks.setdefault(hookable_method, []).append(hook_func)

    def send(self, message: Any, recipient: MockAutoGenAgent) -> None:
        processed_msg = message
        for h in self.hooks.get("process_message_before_send", []):
            processed_msg = h(self, processed_msg, recipient)


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite
# ─────────────────────────────────────────────────────────────────────────────


class TestCrewAIIntegration:
    def test_crewai_sequential_execution(self) -> None:
        """Verify standard sequential Crew execution records task steps and tool spans."""
        researcher = MockCrewAgent(role="market_researcher")
        writer = MockCrewAgent(role="content_writer")

        task1 = MockTask(description="Research AI market", agent=researcher)
        task2 = MockTask(description="Write summary report", agent=writer)

        hook = AgentLensCrewHook(workflow="market_analysis_swarm")
        crew = MockCrew(
            agents=[researcher, writer],
            tasks=[task1, task2],
            step_callback=hook.step_callback,
            task_callback=hook.task_callback,
        )

        crew.kickoff({"topic": "AI Observability"})
        trace = hook.get_trace()

        assert len(trace.steps) == 2
        assert trace.steps[0].agent == "market_researcher"
        assert trace.steps[1].agent == "content_writer"
        assert trace.status == StepStatus.SUCCESS

        # Tool spans from step_callback
        tool_spans = [s for s in trace.spans if s.span_type == SpanType.TOOL]
        assert len(tool_spans) == 2
        assert tool_spans[0].name == "search_db"

        # Arbiter evaluation
        bundle = hook.evaluate()
        assert bundle.priority_level == PriorityLevel.P5
        assert bundle.primary_agent is None

    def test_crewai_hierarchical_manager_delegation(self) -> None:
        """Verify hierarchical process links worker spans to manager agent span."""
        manager = MockCrewAgent(role="director")
        worker_a = MockCrewAgent(role="engineer_a")
        worker_b = MockCrewAgent(role="engineer_b")

        task1 = MockTask(description="Build backend", agent=worker_a)
        task2 = MockTask(description="Build frontend", agent=worker_b)

        crew = MockCrew(
            agents=[worker_a, worker_b],
            tasks=[task1, task2],
            process="hierarchical",
            manager_agent=manager,
        )

        hook = AgentLensCrewHook(workflow="engineering_hierarchy")
        with hook.monitor_crew(crew):
            crew.kickoff({"project": "Platform v2"})

        spans = hook.get_spans()
        mgr_spans = [s for s in spans if s.name == "director"]
        assert len(mgr_spans) == 1
        mgr_span_id = mgr_spans[0].span_id

        # Worker agent spans must reference manager span
        worker_spans = [s for s in spans if s.name in ("engineer_a", "engineer_b")]
        assert len(worker_spans) == 2
        for ws in worker_spans:
            assert ws.parent_span_id == mgr_span_id

    def test_crewai_tool_error_and_arbiter_detection(self) -> None:
        """Verify nested tool exception marks step as ERROR and triggers tool_failure_v1."""
        hook = AgentLensCrewHook(workflow="failing_crew")
        agent = MockCrewAgent(role="financial_analyst")

        # Record tool start & failure
        t_id = hook.on_tool_start("fetch_stock_quote", "AAPL")
        hook.on_tool_error("fetch_stock_quote", ConnectionError("API rate limit 429"), tool_id=t_id)

        # Record task callback
        task_out = MockTaskOutput(
            description="Fetch Apple Q3 financials",
            raw="Failure: could not reach quote endpoint",
            agent=agent,
        )
        step = hook.task_callback(task_out)

        assert step.status == StepStatus.ERROR
        assert len(step.tool_calls) == 1
        assert "429" in step.tool_calls[0]["error"]

        trace = hook.get_trace()
        assert trace.status == StepStatus.FAILURE

        bundle = hook.evaluate()
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_agent == "financial_analyst"

    def test_crewai_database_persistence(self, tmp_path: Path) -> None:
        """Verify save_to_db=True persists the CrewAI run trace into SQLite."""
        db_file = tmp_path / "crewai_test.db"
        agentlens.init(db_path=db_file)

        hook = AgentLensCrewHook(workflow="persisted_crew", save_to_db=True)
        agent = MockCrewAgent(role="devops_agent")

        task_out = MockTaskOutput(
            description="Deploy microservice",
            raw="Deployed container hash abc1234",
            agent=agent,
        )
        hook.task_callback(task_out, tokens=TokenUsage(prompt=50, completion=25, total=75))

        trace = hook.get_trace()

        # Read back from database
        db = DatabaseManager(db_path=db_file)
        row = db.get_run(trace.run_id)
        assert row is not None
        assert row["workflow"] == "persisted_crew"
        assert row["total_tokens"] == 75


class TestAutoGenIntegration:
    def test_autogen_two_agent_chat(self) -> None:
        """Verify two-agent conversation records alternating speaker turns."""
        hook = AgentLensAutoGenHook(workflow="two_agent_chat")

        user_proxy = MockAutoGenAgent(name="user_proxy")
        assistant = MockAutoGenAgent(name="assistant")

        hook.register(user_proxy)
        hook.register(assistant)

        # Exchange messages
        user_proxy.send("Can you calculate 2 + 2?", assistant)
        assistant.send("The result is 4.", user_proxy)

        trace = hook.get_trace()
        assert len(trace.steps) == 2
        assert trace.steps[0].agent == "user_proxy"
        assert trace.steps[1].agent == "assistant"
        assert trace.steps[0].output == "Can you calculate 2 + 2?"
        assert trace.steps[1].output == "The result is 4."
        assert trace.status == StepStatus.SUCCESS

    def test_autogen_group_chat_cyclic_iterations(self) -> None:
        """Verify cyclic conversation (A -> B -> A -> B) tracks iteration_index correctly."""
        hook = AgentLensAutoGenHook(workflow="round_robin_group_chat")

        # Turn 1: Coder
        hook.on_turn(speaker="coder", message="Drafting script v1", recipient="critic")
        # Turn 2: Critic
        hook.on_turn(speaker="critic", message="Refactor lines 10-15", recipient="coder")
        # Turn 3: Coder (Iteration 1)
        hook.on_turn(speaker="coder", message="Refactored script v2", recipient="critic")
        # Turn 4: Critic (Iteration 1)
        hook.on_turn(speaker="critic", message="Approved! Ready to ship.", recipient="coder")

        trace = hook.get_trace()
        assert len(trace.steps) == 4
        assert [s.agent for s in trace.steps] == ["coder", "critic", "coder", "critic"]

        # Verify iteration indices in spans
        coder_spans = [
            s for s in trace.spans if s.name == "coder" and s.span_type == SpanType.AGENT
        ]
        assert len(coder_spans) == 2
        assert coder_spans[0].iteration_index == 0
        assert coder_spans[1].iteration_index == 1

        critic_spans = [
            s for s in trace.spans if s.name == "critic" and s.span_type == SpanType.AGENT
        ]
        assert len(critic_spans) == 2
        assert critic_spans[0].iteration_index == 0
        assert critic_spans[1].iteration_index == 1

    def test_autogen_code_execution_block_and_error(self) -> None:
        """Verify code blocks in message are parsed into tool spans and exitcode: 1 detected."""
        hook = AgentLensAutoGenHook(workflow="code_execution_chat")

        # Assistant emits code block
        code_msg = "Here is the calculation:\n```python\nimport math\nprint(math.sqrt(-1))\n```"
        step1 = hook.on_turn(speaker="assistant", message=code_msg, recipient="user_proxy")
        assert len(step1.tool_calls) == 1
        assert step1.tool_calls[0]["name"] == "code_exec_python"

        # User proxy replies with execution error
        err_reply = "exitcode: 1 (execution failed)\nCode output: ValueError: math domain error"
        step2 = hook.on_turn(speaker="user_proxy", message=err_reply, recipient="assistant")

        assert step2.status == StepStatus.ERROR
        assert "math domain error" in str(step2.error)

        trace = hook.get_trace()
        assert trace.status == StepStatus.FAILURE

        bundle = hook.evaluate()
        assert bundle.primary_cause == FailureCategory.EXECUTION

    def test_autogen_monitor_context_manager_and_db_persistence(self, tmp_path: Path) -> None:
        """Verify monitor_conversation context manager and save_to_db persistence."""
        db_file = tmp_path / "autogen_test.db"
        agentlens.init(db_path=db_file)

        agent_a = MockAutoGenAgent(name="planner")
        agent_b = MockAutoGenAgent(name="executor")

        hook = AgentLensAutoGenHook(workflow="monitored_chat", save_to_db=True)
        with hook.monitor_conversation([agent_a, agent_b]):
            agent_a.send("Plan: build API service", agent_b)
            agent_b.send("Execution: API service listening on port 8000", agent_a)

        trace = hook.get_trace()
        assert len(trace.steps) == 2

        db = DatabaseManager(db_path=db_file)
        row = db.get_run(trace.run_id)
        assert row is not None
        assert row["workflow"] == "monitored_chat"
        assert row["status"] == "SUCCESS"
