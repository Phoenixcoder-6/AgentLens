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
        
        # Save trace to disk
        cls._save_trace_to_disk(trace)
        
        cls._current_trace = None
        return trace

    @classmethod
    def _save_trace_to_disk(cls, trace: RunTrace):
        traces_dir = "data/traces"
        os.makedirs(traces_dir, exist_ok=True)
        
        trace.trace_path = f"{traces_dir}/{trace.run_id}.json"
        
        with open(trace.trace_path, "w", encoding="utf-8") as f:
            f.write(trace.model_dump_json(indent=2))
