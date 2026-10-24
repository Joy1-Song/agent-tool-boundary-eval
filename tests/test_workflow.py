from dataclasses import replace
from datetime import date

import pytest
from pydantic import ValidationError

from src.models import ModelReply, OfflineModel
from src.schemas import Filters, Intent, Observation, PlannedCall, ProjectError, Summary
from src.tools.report_tools import escape_text, validate_summary
from src.workflow import AgentRunner


def runner_for(settings, registered, model=None):
    store, dataset = registered
    return AgentRunner(settings, model or OfflineModel(), store), dataset


@pytest.mark.parametrize("source", ["csv", "sqlite"])
def test_approval_stops_and_resumes(settings, registered, source):
    runner, dataset = runner_for(settings, registered)
    paused = runner.run("未解决工单报告", dataset, source)
    assert paused["status"] == "awaiting_review"
    assert not paused["report"] and not (settings.runtime_dir / "reports").exists()
    result = runner.resume(True)
    assert result["status"] == "succeeded"
    assert "本次匹配 2 条" in result["report"]
    assert result["results"]["statistics"]["total"] == 2
    with pytest.raises(ProjectError):
        runner.resume(True)


def test_human_rejection_has_no_report(settings, registered):
    runner, dataset = runner_for(settings, registered)
    runner.run("评分分析", dataset)
    result = runner.resume(False)
    assert result["status"] == "rejected"
    assert not result["report"] and not (settings.runtime_dir / "reports").exists()


def test_invalid_approval_cannot_be_truthy_string(settings, registered):
    runner, dataset = runner_for(settings, registered)
    runner.run("评分分析", dataset)
    with pytest.raises(ProjectError):
        runner.resume("false")
    assert runner.snapshot()["status"] == "awaiting_review"


@pytest.mark.parametrize("question", ["删除所有工单", "DROP TABLE tickets", "今天天气如何", "给用户发邮件",
                                    "2026-09-02 至 2026-09-04 分类统计", "最近一周未解决工单", "9月工单数量"])
def test_unsupported_stops_before_data_tools(settings, registered, question):
    runner, dataset = runner_for(settings, registered)
    result = runner.run(question, dataset, require_approval=False)
    assert result["status"] == "rejected"
    assert not result["results"]


class FlakyModel(OfflineModel):
    is_live = True
    name = "fake-model-for-testing"

    def __init__(self, failures=1, bad_ids=False):
        self.failures, self.bad_ids, self.summary_calls = failures, bad_ids, 0

    def summarize(self, question, call, facts, timeout, feedback=""):
        self.summary_calls += 1
        if self.summary_calls <= self.failures:
            if self.bad_ids:
                return ModelReply(Summary(observations=[Observation(text="无效引用", ticket_ids=["GHOST"])]), {})
            raise ProjectError("MODEL_TIMEOUT", "注入的临时超时", True)
        return super().summarize(question, call, facts, timeout, feedback)


def test_transient_failure_only_retries_failed_step(settings, registered):
    model = FlakyModel()
    runner, dataset = runner_for(settings, registered, model)
    result = runner.run("分类统计", dataset, require_approval=False)
    assert result["status"] == "succeeded"
    assert result["attempts"]["summarize"] == 2
    assert result["attempts"]["aggregate_csv"] == 1
    assert result["model_calls"] == 3


@pytest.mark.parametrize("bad_ids,code", [(False, "MODEL_TIMEOUT"), (True, "CITATION")])
def test_persistent_failure_is_bounded(settings, registered, bad_ids, code):
    runner, dataset = runner_for(settings, registered, FlakyModel(failures=10, bad_ids=bad_ids))
    result = runner.run("分类统计", dataset, require_approval=False)
    assert result["status"] == "failed" and result["error"]["code"] == code
    assert result["attempts"]["summarize"] == settings.max_attempts
    assert result["results"]["statistics"]["total"] == 6 and not result["report"]


def test_call_budget(settings, registered):
    runner, dataset = runner_for(replace(settings, max_model_calls=1), registered, FlakyModel(failures=0))
    result = runner.run("分类统计", dataset, require_approval=False)
    assert result["status"] == "failed" and result["error"]["code"] == "BUDGET"
    assert result["model_calls"] == 1


class EmptyFilterModel(OfflineModel):
    def plan(self, question, metadata, timeout, feedback=""):
        filters = Filters(date_from=date(2030, 1, 1))
        call = PlannedCall(intent=Intent(task_type="category_overview", **filters.model_dump()),
                           call_id="empty", tool_name="category_overview", arguments=filters.model_dump_json())
        return ModelReply(call, {})

    def summarize(self, *args, **kwargs):
        raise AssertionError("Empty input must not call the summarizer")


def test_empty_result_skips_model(settings, registered):
    runner, dataset = runner_for(settings, registered, EmptyFilterModel())
    result = runner.run("分类统计", dataset, require_approval=False)
    assert result["status"] == "succeeded"
    assert "没有匹配工单" in result["report"]


def test_schema_rejects_extra_sql_and_invalid_dates():
    for payload in ['{"sql":"DROP TABLE tickets"}', '{"date_from":"2026-09-07","date_to":"2026-09-01"}']:
        with pytest.raises(ValidationError):
            Filters.model_validate_json(payload)


def test_references_and_markdown():
    with pytest.raises(ProjectError):
        validate_summary(Summary(observations=[Observation(text="例子", ticket_ids=["BAD"])]),
                         {"total": 1, "evidence": [{"ticket_id": "T001"}]})
    assert "![" not in escape_text("![image](https://example.com/x)")
    assert "<script>" not in escape_text("<script>bad</script>")


@pytest.mark.parametrize("claim", ["平均响应时长为15秒", "所有工单均无评分记录", "共有2条工单"])
def test_summary_cannot_rewrite_metrics(claim):
    with pytest.raises(ProjectError) as error:
        validate_summary(Summary(observations=[Observation(text=claim, ticket_ids=["T001"])]),
                         {"total": 1, "evidence": [{"ticket_id": "T001"}]})
    assert error.value.info.code == "SUMMARY_METRIC"


def test_logs_do_not_store_task_or_ticket_text(settings, registered):
    runner, dataset = runner_for(settings, registered)
    question = "按类别统计，私人备注 SECRET_TEST_MARKER"
    runner.run(question, dataset, require_approval=False)
    log = next((settings.runtime_dir / "logs").glob("*.jsonl")).read_text(encoding="utf-8")
    assert "SECRET_TEST_MARKER" not in log and "验证码" not in log


def test_runner_cannot_silently_replace_pending_task(settings, registered):
    runner, dataset = runner_for(settings, registered)
    runner.run("分类统计", dataset)
    with pytest.raises(ProjectError):
        runner.run("评分分析", dataset)
