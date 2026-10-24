"""Identical scenarios on old/new code; the old runner has no request key API."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import inspect
from pathlib import Path
from threading import Event

import pytest

from src.models import OfflineModel
from src.schemas import ProjectError
from src.workflow import AgentRunner


class CountingModel(OfflineModel):
    def __init__(self):
        self.plans = 0
        self.summaries = 0

    def plan(self, *args, **kwargs):
        self.plans += 1
        return super().plan(*args, **kwargs)

    def summarize(self, *args, **kwargs):
        self.summaries += 1
        return super().summarize(*args, **kwargs)


def submit(runner, dataset, key="REQ001", question="分类统计", **kwargs):
    # Capability fallback allows the pre-upgrade implementation to execute normally.
    if "request_id" in inspect.signature(runner.run).parameters:
        kwargs["request_id"] = key
    return runner.run(question, dataset, require_approval=kwargs.pop("require_approval", False), **kwargs)


def test_completed_request_replays_without_second_model_run(settings, registered):
    store, dataset = registered
    model = CountingModel()
    first = submit(AgentRunner(settings, model, store), dataset)
    second = submit(AgentRunner(settings, model, store), dataset)
    assert second == first
    assert (model.plans, model.summaries) == (1, 1)
    assert len(list((settings.runtime_dir / "reports").glob("*.md"))) == 1


def test_same_key_different_question_is_rejected(settings, registered):
    store, dataset = registered
    model = CountingModel()
    submit(AgentRunner(settings, model, store), dataset)
    with pytest.raises(ProjectError, match="请求编号"):
        submit(AgentRunner(settings, model, store), dataset, question="评分分析")
    assert model.plans == 1


def test_same_key_different_configuration_is_rejected(settings, registered):
    store, dataset = registered
    submit(AgentRunner(settings, CountingModel(), store), dataset)
    with pytest.raises(ProjectError):
        submit(AgentRunner(replace(settings, max_evidence_rows=2), CountingModel(), store), dataset)


def test_new_key_is_a_new_business_operation(settings, registered):
    store, dataset = registered
    model = CountingModel()
    first = submit(AgentRunner(settings, model, store), dataset, "REQ001")
    second = submit(AgentRunner(settings, model, store), dataset, "REQ002")
    assert first["task_id"] != second["task_id"]
    assert model.plans == 2


def test_replay_survives_new_store_instance(settings, sample_bytes):
    from src.datasets import DatasetStore
    first_store = DatasetStore(settings)
    dataset = first_store.register(sample_bytes)
    first = submit(AgentRunner(settings, CountingModel(), first_store), dataset)
    new_store = DatasetStore(settings)
    new_store.register(sample_bytes)
    model = CountingModel()
    second = submit(AgentRunner(settings, model, new_store), dataset)
    assert second == first
    assert model.plans == 0


def test_missing_report_is_not_silently_regenerated(settings, registered):
    store, dataset = registered
    first = submit(AgentRunner(settings, CountingModel(), store), dataset)
    Path(first["report_file"]).unlink()
    model = CountingModel()
    with pytest.raises(ProjectError):
        submit(AgentRunner(settings, model, store), dataset)
    assert model.plans == 0


def test_report_tampering_is_detected_on_replay(settings, registered):
    store, dataset = registered
    first = submit(AgentRunner(settings, CountingModel(), store), dataset)
    Path(first["report_file"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ProjectError):
        submit(AgentRunner(settings, CountingModel(), store), dataset)


def test_approval_request_replays_but_does_not_approve(settings, registered):
    store, dataset = registered
    model = CountingModel()
    owner = AgentRunner(settings, model, store)
    first = submit(owner, dataset, require_approval=True)
    duplicate = AgentRunner(settings, model, store)
    second = submit(duplicate, dataset, require_approval=True)
    assert second == first and second["status"] == "awaiting_review"
    with pytest.raises(ProjectError):
        duplicate.resume(True)
    assert owner.resume(True)["status"] == "succeeded"
    assert model.plans == 1


def test_failed_request_is_replayed_not_blindly_retried(settings, registered):
    class Failing(CountingModel):
        def summarize(self, *args, **kwargs):
            self.summaries += 1
            raise ProjectError("INPUT", "Invalid input", False)
    store, dataset = registered
    model = Failing()
    first = submit(AgentRunner(settings, model, store), dataset)
    second = submit(AgentRunner(settings, model, store), dataset)
    assert first == second and first["status"] == "failed"
    assert model.summaries == 1


def test_concurrent_duplicate_is_rejected_before_model(settings, registered):
    entered, release = Event(), Event()
    class Blocking(CountingModel):
        def plan(self, *args, **kwargs):
            entered.set()
            assert release.wait(10)
            return super().plan(*args, **kwargs)
    store, dataset = registered
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(submit, AgentRunner(settings, Blocking(), store), dataset)
        try:
            assert entered.wait(10)
            with pytest.raises(ProjectError):
                submit(AgentRunner(settings, CountingModel(), store), dataset)
        finally:
            release.set()
        assert future.result(timeout=15)["status"] == "succeeded"


def test_question_whitespace_is_normalized(settings, registered):
    store, dataset = registered
    first = submit(AgentRunner(settings, CountingModel(), store), dataset, question="  分类统计  ")
    second = submit(AgentRunner(settings, CountingModel(), store), dataset)
    assert first == second


def test_nonboolean_approval_flag_is_rejected(settings, registered):
    store, dataset = registered
    with pytest.raises(ProjectError):
        submit(AgentRunner(settings, CountingModel(), store), dataset, require_approval="false")
