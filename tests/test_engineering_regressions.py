"""Deterministic deadline and quota boundaries; clocks are injected, not benchmarks."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.schemas import ProjectError
from src import workflow


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def execution(settings, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(workflow.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(workflow.time, "sleep", clock.sleep)
    config = replace(settings, active_budget_seconds=1, request_timeout=1, retry_delay=0.25)
    context = SimpleNamespace(settings=config, audit=Mock())
    state = {"task_id": "deadline-test", "attempts": {}, "model_calls": 0, "usage": {}, "active_seconds": 0}
    return clock, context, state


def test_sleep_overrun_never_dispatches_an_expired_request(execution, monkeypatch):
    clock, context, state = execution
    state["attempts"] = {"summarize": 1}
    monkeypatch.setattr(workflow.time, "sleep", lambda _: clock.sleep(1.1))
    operation = Mock(return_value={})
    result = workflow.run_attempt(state, context, "executor", "summarize", operation, model_call=True)
    operation.assert_not_called()
    assert result["status"] == "failed"
    assert result["error"]["code"] == "BUDGET"
    assert result["model_calls"] == 0


def test_timeout_uses_actual_remaining_time_after_backoff(execution, monkeypatch):
    clock, context, state = execution
    state["attempts"] = {"summarize": 1}
    monkeypatch.setattr(workflow.time, "sleep", lambda _: clock.sleep(0.6))
    operation = Mock(return_value={})
    result = workflow.run_attempt(state, context, "executor", "summarize", operation, model_call=True)
    assert operation.call_args.args[0] == pytest.approx(0.4)
    assert result["status"] == "running"


def test_failed_attempt_cannot_schedule_retry_after_time_budget(execution):
    clock, context, state = execution

    def operation(timeout):
        clock.sleep(1.1)
        raise ProjectError("MODEL_TIMEOUT", "Injected timeout", True)

    result = workflow.run_attempt(state, context, "executor", "summarize", operation, model_call=True)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "MODEL_TIMEOUT"
    assert result["active_seconds"] == pytest.approx(1.1)


def test_failed_last_model_call_cannot_schedule_another_attempt(execution):
    _, context, state = execution
    context.settings = replace(context.settings, max_model_calls=1)
    operation = Mock(side_effect=ProjectError("MODEL_RATE_LIMIT", "Injected limit", True))
    result = workflow.run_attempt(state, context, "executor", "summarize", operation, model_call=True)
    assert result["status"] == "failed"
    assert result["model_calls"] == 1
    assert result["error"]["code"] == "MODEL_RATE_LIMIT"


def test_retryable_failure_with_remaining_budget_still_retries(execution):
    _, context, state = execution
    result = workflow.run_attempt(state, context, "executor", "summarize",
                                  Mock(side_effect=ProjectError("MODEL_TIMEOUT", "Injected", True)), model_call=True)
    assert result["status"] == "running"
    assert result["attempts"]["summarize"] == 1


def test_nonretryable_data_failure_still_stops(execution):
    _, context, state = execution
    result = workflow.run_attempt(state, context, "executor", "aggregate_csv",
                                  Mock(side_effect=ProjectError("CSV_COLUMNS", "Invalid columns")))
    assert result["status"] == "failed"
    assert result["model_calls"] == 0


def test_already_exhausted_call_budget_never_calls_operation(execution):
    _, context, state = execution
    state["model_calls"] = context.settings.max_model_calls
    operation = Mock(return_value={})
    result = workflow.run_attempt(state, context, "executor", "summarize", operation, model_call=True)
    operation.assert_not_called()
    assert result["error"]["code"] == "BUDGET"


def test_request_timeout_remains_capped_by_settings(execution):
    _, context, state = execution
    context.settings = replace(context.settings, request_timeout=0.2)
    operation = Mock(return_value={})
    result = workflow.run_attempt(state, context, "executor", "summarize", operation, model_call=True)
    assert operation.call_args.args[0] == pytest.approx(0.2)
    assert result["model_calls"] == 1
