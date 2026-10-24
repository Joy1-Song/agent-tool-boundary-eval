import pytest

from src.models import QwenModel
from src.schemas import ProjectError
from test_model_adapter import StubClient, call_message


METADATA = {"categories": ["登录", "支付", "课程内容"]}


@pytest.mark.parametrize("question,arguments,tool", [
    ("统计微信渠道支付类别的数量", '{"category":"支付"}', "category_overview"),
    ("按类别统计已解决工单数量", '{}', "category_overview"),
    ("分析评分低于3分的工单", '{}', "rating_analysis"),
    ("分析响应超过20分钟的工单", '{}', "category_overview"),
    ("统计除支付以外其他类别的数量", '{"category":"支付"}', "category_overview"),
    ("分析登录和支付未解决工单", '{"category":"支付"}', "unresolved_report"),
    ("统计支付数量并发送报告给客户", '{"category":"支付"}', "category_overview"),
    ("将未解决工单标记为已解决", '{}', "unresolved_report"),
    ("统计9月份未解决工单", '{"date_from":"2026-09-01","date_to":"2026-09-30"}', "unresolved_report"),
    ("统计最近一周工单数量", '{"date_from":"2026-09-01"}', "category_overview"),
])
def test_valid_tool_must_not_silently_drop_unsupported_conditions(settings, question, arguments, tool):
    model = QwenModel(settings, client=StubClient([call_message(tool, arguments)]))
    with pytest.raises(ProjectError) as error:
        model.plan(question, METADATA, 3)
    assert error.value.info.code == "UNSUPPORTED"


@pytest.mark.parametrize("question,arguments,tool", [
    ("统计支付类别数量", '{"category":"登录"}', "category_overview"),
    ("统计支付类别数量", '{}', "category_overview"),
    ("统计2026-09-01至2026-09-30工单数量", '{}', "category_overview"),
    ("统计2026-09-01至2026-09-30工单数量", '{"date_from":"2026-09-01","date_to":"2026-10-01"}', "category_overview"),
    ("统计2026-09-03当天工单数量", '{"date_from":"2026-09-03"}', "category_overview"),
    ("分析支付类未解决工单", '{"category":"支付"}', "category_overview"),
])
def test_explicit_supported_conditions_must_match_model_arguments(settings, question, arguments, tool):
    model = QwenModel(settings, client=StubClient([call_message(tool, arguments)]))
    with pytest.raises(ProjectError) as error:
        model.plan(question, METADATA, 3)
    assert error.value.info.code == "MODEL_INTENT" and error.value.info.retryable


@pytest.mark.parametrize("question,arguments,tool", [
    ("统计支付类别数量", '{"category":"支付"}', "category_overview"),
    ("统计2026-09-01至2026-09-30工单数量", '{"date_from":"2026-09-01","date_to":"2026-09-30"}', "category_overview"),
    ("统计2026-09-03当天工单数量", '{"date_from":"2026-09-03","date_to":"2026-09-03"}', "category_overview"),
    ("按类别统计全部状态的工单，不限制日期", '{}', "category_overview"),
    ("分析支付类未解决工单", '{"category":"支付"}', "unresolved_report"),
])
def test_supported_requests_are_not_blocked(settings, question, arguments, tool):
    client = StubClient([call_message(tool, arguments)])
    result = QwenModel(settings, client=client).plan(question, METADATA, 3)
    assert result.value.tool_name == tool
    assert len(client.requests) == 1
