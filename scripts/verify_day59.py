"""
scripts/verify_day59.py — Day 59 Verification Gate
===================================================
Phase C Sign-Off: CrewAI & AutoGen Interoperability (AgentLens v2.0)

Checks:
  1. Swarm Integration Surface & Public SDK Exports (CrewAI & AutoGen hooks).
  2. CrewAI Sequential Task Swarm Auto-Instrumentation.
  3. CrewAI Hierarchical Process Delegation (manager-worker parent spans).
  4. CrewAI Tool Error Interception & Arbiter Root-Cause Attribution (tool_failure_v1).
  5. AutoGen Conversational Turn & Speaker Transition Tracking.
  6. AutoGen Cyclic Iteration Index Accounting (A -> B -> A -> B).
  7. AutoGen Code Execution Block Parsing & Error Detection (exitcode: 1).
  8. Static Analysis (Ruff/Mypy Clean) & Day 43 Frozen Benchmark Gate (16/20 = 80.0%).
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agentlens
from agentlens.integrations.autogen import AgentLensAutoGenHook, AgentLensConversationTracer
from agentlens.integrations.crewai import AgentLensCrewHook
from schema.models import FailureCategory, PriorityLevel, StepStatus, TokenUsage
from schema.span import SpanType
from storage.db import DatabaseManager

# ─────────────────────────────────────────────────────────────────────────────
# Mocks for Verification Gate
# ─────────────────────────────────────────────────────────────────────────────


class MockCrewAgent:
    def __init__(self, role: str) -> None:
        self.role = role


class MockTaskOutput:
    def __init__(self, description: str, raw: str, agent: Any) -> None:
        self.description = description
        self.raw = raw
        self.agent = agent


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
        for task in self.tasks:
            if self.step_callback:
                self.step_callback({"tool": "api_fetch", "tool_input": "q", "result": "res"})
            if self.task_callback:
                self.task_callback(
                    MockTaskOutput(
                        description=task["desc"],
                        raw=f"Output: {task['desc']}",
                        agent=task["agent"],
                    )
                )
        return {"result": "done"}


class MockAutoGenAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hooks: dict[str, list[Any]] = {}

    def register_hook(self, method: str, hook_func: Any) -> None:
        self.hooks.setdefault(method, []).append(hook_func)

    def send(self, message: Any, recipient: MockAutoGenAgent) -> None:
        for h in self.hooks.get("process_message_before_send", []):
            h(self, message, recipient)


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 59 VERIFICATION GATE")
    print("  CrewAI & AutoGen Interoperability (Phase C Day 59)")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Swarm Integration Surface & Public SDK Exports
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert hasattr(agentlens, "AgentLensCrewHook")
        assert hasattr(agentlens, "AgentLensAutoGenHook")
        assert hasattr(agentlens, "AgentLensConversationTracer")

        assert agentlens.AgentLensCrewHook is AgentLensCrewHook
        assert agentlens.AgentLensAutoGenHook is AgentLensAutoGenHook
        assert agentlens.AgentLensConversationTracer is AgentLensConversationTracer

        c_hook = AgentLensCrewHook(workflow="check1_crew")
        assert c_hook.workflow == "check1_crew"
        assert len(c_hook.get_steps()) == 0

        a_hook = AgentLensAutoGenHook(workflow="check1_autogen")
        assert a_hook.workflow == "check1_autogen"
        assert len(a_hook.get_steps()) == 0

        print("[PASS] Check 1: Swarm Integration Surface & Public SDK Exports (CrewAI & AutoGen)")
    except Exception as exc:
        print(f"[FAIL] Check 1: Surface exports check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: CrewAI Sequential Task Swarm Auto-Instrumentation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        researcher = MockCrewAgent(role="market_researcher")
        writer = MockCrewAgent(role="copywriter")

        tasks = [
            {"desc": "Analyze competitor landscape", "agent": researcher},
            {"desc": "Draft launch announcement", "agent": writer},
        ]

        hook_seq = AgentLensCrewHook(workflow="sequential_crew")
        crew = MockCrew(
            agents=[researcher, writer],
            tasks=tasks,
            step_callback=hook_seq.step_callback,
            task_callback=hook_seq.task_callback,
        )

        crew.kickoff({"product": "AgentLens v2"})
        trace = hook_seq.get_trace()

        assert len(trace.steps) == 2, f"Expected 2 steps, got {len(trace.steps)}"
        assert trace.steps[0].agent == "market_researcher"
        assert trace.steps[1].agent == "copywriter"
        assert trace.status == StepStatus.SUCCESS

        tool_spans = [s for s in trace.spans if s.span_type == SpanType.TOOL]
        assert len(tool_spans) == 2
        assert tool_spans[0].name == "api_fetch"

        bundle = hook_seq.evaluate()
        assert bundle.priority_level == PriorityLevel.P5

        print("[PASS] Check 2: CrewAI Sequential Task Swarm Auto-Instrumentation")
    except Exception as exc:
        print(f"[FAIL] Check 2: Sequential CrewAI check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: CrewAI Hierarchical Process Delegation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        manager = MockCrewAgent(role="director")
        analyst = MockCrewAgent(role="analyst")

        crew = MockCrew(
            agents=[analyst],
            tasks=[{"desc": "Audit logs", "agent": analyst}],
            process="hierarchical",
            manager_agent=manager,
        )

        hook_hier = AgentLensCrewHook(workflow="hierarchical_crew")
        with hook_hier.monitor_crew(crew):
            crew.kickoff()

        spans = hook_hier.get_spans()
        mgr_spans = [s for s in spans if s.name == "director"]
        assert len(mgr_spans) == 1
        mgr_id = mgr_spans[0].span_id

        worker_spans = [s for s in spans if s.name == "analyst"]
        assert len(worker_spans) == 1
        assert worker_spans[0].parent_span_id == mgr_id

        print("[PASS] Check 3: CrewAI Hierarchical Process Delegation (Manager-Worker)")
    except Exception as exc:
        print(f"[FAIL] Check 3: Hierarchical CrewAI check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: CrewAI Tool Error Interception & Arbiter Root-Cause Attribution
    # ─────────────────────────────────────────────────────────────────────────
    try:
        hook_err = AgentLensCrewHook(workflow="tool_error_crew")
        agent = MockCrewAgent(role="database_admin")

        hook_err.on_tool_start("exec_sql", "DROP TABLE users")
        hook_err.on_tool_error(
            "exec_sql",
            PermissionError("Access denied: write permissions restricted"),
        )

        task_out = MockTaskOutput(
            description="Run database migration",
            raw="Migration crashed",
            agent=agent,
        )
        step = hook_err.task_callback(task_out)
        assert step.status == StepStatus.ERROR
        assert len(step.tool_calls) == 1
        assert "PermissionError" in step.tool_calls[0]["error"]

        trace = hook_err.get_trace()
        assert trace.status == StepStatus.FAILURE

        bundle = hook_err.evaluate()
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_agent == "database_admin"

        print("[PASS] Check 4: CrewAI Tool Error Interception & Arbiter Root-Cause Attribution")
    except Exception as exc:
        print(f"[FAIL] Check 4: Tool error attribution check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: AutoGen Conversational Turn & Speaker Transition Tracking
    # ─────────────────────────────────────────────────────────────────────────
    try:
        hook_ag = AgentLensAutoGenHook(workflow="autogen_dialogue")
        u_proxy = MockAutoGenAgent("user_proxy")
        a_agent = MockAutoGenAgent("assistant")

        hook_ag.register(u_proxy)
        hook_ag.register(a_agent)

        u_proxy.send("What is the capital of France?", a_agent)
        a_agent.send("The capital of France is Paris.", u_proxy)

        trace = hook_ag.get_trace()
        assert len(trace.steps) == 2
        assert trace.steps[0].agent == "user_proxy"
        assert trace.steps[1].agent == "assistant"
        assert trace.status == StepStatus.SUCCESS

        print("[PASS] Check 5: AutoGen Conversational Turn & Speaker Transition Tracking")
    except Exception as exc:
        print(f"[FAIL] Check 5: AutoGen turn tracking check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: AutoGen Cyclic Iteration Index Accounting
    # ─────────────────────────────────────────────────────────────────────────
    try:
        hook_cyc = AgentLensAutoGenHook(workflow="cyclic_dialogue")

        hook_cyc.on_turn("dev", "Feature implementation draft", recipient="reviewer")
        hook_cyc.on_turn("reviewer", "Needs unit tests", recipient="dev")
        hook_cyc.on_turn("dev", "Added 5 unit tests", recipient="reviewer")
        hook_cyc.on_turn("reviewer", "LGTM, approved!", recipient="dev")

        trace = hook_cyc.get_trace()
        assert len(trace.steps) == 4
        assert [s.agent for s in trace.steps] == ["dev", "reviewer", "dev", "reviewer"]

        dev_spans = [s for s in trace.spans if s.name == "dev" and s.span_type == SpanType.AGENT]
        assert len(dev_spans) == 2
        assert dev_spans[0].iteration_index == 0
        assert dev_spans[1].iteration_index == 1

        rev_spans = [
            s for s in trace.spans if s.name == "reviewer" and s.span_type == SpanType.AGENT
        ]
        assert len(rev_spans) == 2
        assert rev_spans[0].iteration_index == 0
        assert rev_spans[1].iteration_index == 1

        print("[PASS] Check 6: AutoGen Cyclic Iteration Index Accounting (iteration_index)")
    except Exception as exc:
        print(f"[FAIL] Check 6: Cyclic iteration index check failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: AutoGen Code Execution Block Parsing & Error Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            temp_db = Path(tmp_dir) / "autogen_verify.db"
            agentlens.init(db_path=temp_db)

            hook_code = AgentLensAutoGenHook(workflow="code_exec_workflow", save_to_db=True)

            code_text = "Run this snippet:\n```python\nimport sys\nsys.exit(1)\n```"
            s1 = hook_code.on_turn(
                "coder",
                code_text,
                recipient="executor",
                tokens=TokenUsage(prompt=30, completion=20, total=50),
            )
            assert len(s1.tool_calls) == 1
            assert s1.tool_calls[0]["name"] == "code_exec_python"

            err_text = "exitcode: 1 (execution failed)\nSystemExit: 1"
            s2 = hook_code.on_turn("executor", err_text, recipient="coder")
            assert s2.status == StepStatus.ERROR

            trace = hook_code.get_trace()
            assert trace.status == StepStatus.FAILURE

            # Verify SQLite persistence
            db = DatabaseManager(db_path=temp_db)
            row = db.get_run(trace.run_id)
            assert row is not None
            assert row["workflow"] == "code_exec_workflow"
            assert row["status"] == "FAILURE"

            db.close()
            os.environ.pop("DB_PATH", None)

        print("[PASS] Check 7: AutoGen Code Execution Block Parsing & Error Detection")
    except Exception as exc:
        print(f"[FAIL] Check 7: Code execution error check failed: {exc}")
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
                "agentlens/integrations/crewai.py",
                "agentlens/integrations/autogen.py",
                "tests/test_swarm_integrations.py",
                "scripts/verify_day59.py",
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
                "agentlens/integrations/crewai.py",
                "agentlens/integrations/autogen.py",
                "tests/test_swarm_integrations.py",
                "scripts/verify_day59.py",
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
        print(">>> ALL 8 DAY 59 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> AGENTLENS SWARM INTEROPERABILITY CERTIFIED & COMPLETE! <<<")
    else:
        print(">>> SOME DAY 59 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
