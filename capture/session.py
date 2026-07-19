import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional

from schema.models import RunTrace, AgentStep, StepStatus

class CaptureSession:
    """
    Manages the active RunTrace for the current pipeline run.
    Acts as a singleton to collect AgentSteps during execution.
    """
    _current_trace: Optional[RunTrace] = None
    _pending_tokens: Optional[tuple] = None  # (prompt, completion) set by node, read by tracer

    @classmethod
    def set_step_tokens(cls, prompt: int, completion: int) -> None:
        """
        Stage token counts for the step currently being traced.

        Call this inside a pipeline node right after the LLM response:
            response = llm.invoke(messages)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                CaptureSession.set_step_tokens(
                    prompt=response.usage_metadata.get('input_tokens', 0),
                    completion=response.usage_metadata.get('output_tokens', 0),
                )

        The @trace_step decorator reads and clears this slot when it builds
        the AgentStep — ensuring the correct step gets the correct tokens.
        """
        cls._pending_tokens = (prompt, completion)

    @classmethod
    def consume_pending_tokens(cls) -> Optional[tuple]:
        """
        Read and clear the pending token slot.
        Called by @trace_step after the node function returns.
        Returns (prompt, completion) or None if no tokens were staged.
        """
        tokens = cls._pending_tokens
        cls._pending_tokens = None
        return tokens
    
    @classmethod
    def start_trace(cls, workflow: str, run_id: Optional[str] = None) -> RunTrace:
        cls._current_trace = RunTrace(
            run_id=run_id or f"run_{uuid.uuid4().hex[:8]}",
            workflow=workflow,
            timestamp=datetime.now(timezone.utc),
            status=StepStatus.SUCCESS
        )
        return cls._current_trace
        
    @classmethod
    def get_current_trace(cls) -> Optional[RunTrace]:
        return cls._current_trace
        
    @classmethod
    def add_step(cls, step: AgentStep):
        if cls._current_trace:
            cls._current_trace.steps.append(step)


    
    @classmethod
    def end_trace(cls, status: StepStatus = StepStatus.SUCCESS, error: Optional[str] = None) -> Optional[RunTrace]:
        if not cls._current_trace:
            return None
            
        trace = cls._current_trace
        if status != StepStatus.SUCCESS:
            trace.status = status
            
        # Calculate aggregate metrics
        total_latency = 0.0
        total_tokens = 0
        for step in trace.steps:
            total_latency += step.latency_ms
            total_tokens += step.tokens.total
            if step.status in (StepStatus.FAILURE, StepStatus.ERROR):
                trace.status = step.status
                if step.error and not error:
                    error = step.error
                    
        trace.total_latency_ms = total_latency
        trace.total_tokens = total_tokens
        
        # 1. Save full JSON trace to disk
        cls._save_trace_to_disk(trace)
        
        # 2. Normalize and persist to SQLite
        cls._save_trace_to_db(trace)
        
        cls._current_trace = None
        return trace

    @classmethod
    def _save_trace_to_disk(cls, trace: RunTrace):
        traces_dir = "data/traces"
        os.makedirs(traces_dir, exist_ok=True)
        
        trace.trace_path = f"{traces_dir}/{trace.run_id}.json"
        
        with open(trace.trace_path, "w", encoding="utf-8") as f:
            f.write(trace.model_dump_json(indent=2))

    @classmethod
    def _save_trace_to_db(cls, trace: RunTrace) -> None:
        """Normalize the completed RunTrace and write to SQLite."""
        try:
            from normalizer.normalizer import Normalizer
            from storage.db import DatabaseManager
            from storage.writer import StorageWriter

            db = DatabaseManager()
            db.initialize()

            normalized = Normalizer().normalize_run(trace)

            # Read the JSON blob we just saved to disk
            trace_json: Optional[str] = None
            if trace.trace_path and os.path.exists(trace.trace_path):
                with open(trace.trace_path, "r", encoding="utf-8") as f:
                    trace_json = f.read()

            writer = StorageWriter(db)
            writer.write_run(
                run=normalized,
                trace_json=trace_json,
                trace_path=trace.trace_path,
            )
        except Exception as exc:
            # Storage failure must never crash the pipeline
            print(f"[CaptureSession] Warning: storage write failed: {exc}")

