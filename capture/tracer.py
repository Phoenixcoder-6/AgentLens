import time
import functools
import json
from datetime import datetime, timezone
from typing import Any, Callable

from schema.models import AgentStep, StepStatus, HandoffState
from capture.session import CaptureSession

def trace_step(func: Callable) -> Callable:
    """
    Decorator to silently record every agent call as an AgentStep in the active RunTrace.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        trace = CaptureSession.get_current_trace()
        if not trace:
            return func(*args, **kwargs)
            
        agent_name = func.__name__.replace("_node", "")
        step_idx = len(trace.steps) + 1
        
        # Capture input state
        state_in = args[0] if args else kwargs.get('state', {})
        
        start_time = time.perf_counter()
        
        step = AgentStep(
            run_id=trace.run_id,
            step=step_idx,
            agent=agent_name,
            input=json.dumps(state_in, default=str),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Handoff capture: we store the full input state for Day 6
        if isinstance(state_in, dict):
            step.handoff.input_state = state_in.copy()
        
        try:
            # Execute the actual agent
            result = func(*args, **kwargs)
            
            latency = (time.perf_counter() - start_time) * 1000.0
            
            step.output = json.dumps(result, default=str)
            step.latency_ms = latency
            step.status = StepStatus.SUCCESS
            
            # Handoff capture: store the returned output state diff for Day 6
            if isinstance(result, dict):
                step.handoff.output_state = result.copy()
            
            CaptureSession.add_step(step)
            return result
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            step.latency_ms = latency
            step.status = StepStatus.ERROR
            step.error = str(e)
            CaptureSession.add_step(step)
            raise e
            
    return wrapper
