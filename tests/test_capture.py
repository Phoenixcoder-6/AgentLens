import pytest
from datetime import datetime, timezone
import json
import os

from schema.models import RunTrace, AgentStep, StepStatus
from capture.session import CaptureSession
from capture.tracer import trace_step

def test_capture_session_starts_and_ends():
    trace = CaptureSession.start_trace(workflow="test_workflow")
    assert trace.workflow == "test_workflow"
    assert trace.status == StepStatus.SUCCESS
    
    assert CaptureSession.get_current_trace() == trace
    
    completed = CaptureSession.end_trace()
    assert completed is not None
    assert completed.status == StepStatus.SUCCESS
    assert CaptureSession.get_current_trace() is None

def test_trace_step_decorator():
    CaptureSession.start_trace(workflow="test_decorator")
    
    @trace_step
    def dummy_node(state):
        return {"output_key": "output_value"}
        
    result = dummy_node({"input_key": "input_value"})
    assert result == {"output_key": "output_value"}
    
    trace = CaptureSession.end_trace()
    assert trace is not None
    assert len(trace.steps) == 1
    
    step = trace.steps[0]
    assert step.agent == "dummy"
    assert step.status == StepStatus.SUCCESS
    assert step.latency_ms > 0
    assert "input_key" in step.input
    assert "output_key" in step.output
    
def test_trace_step_captures_errors():
    CaptureSession.start_trace(workflow="test_error")
    
    @trace_step
    def failing_node(state):
        raise ValueError("Something broke")
        
    with pytest.raises(ValueError):
        failing_node({})
        
    trace = CaptureSession.end_trace()
    assert trace is not None
    assert len(trace.steps) == 1
    
    step = trace.steps[0]
    assert step.status == StepStatus.ERROR
    assert step.error == "Something broke"
    
    assert trace.status == StepStatus.ERROR
