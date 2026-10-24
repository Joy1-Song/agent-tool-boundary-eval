"""Conservative checks for explicit conditions, not a general Chinese intent parser."""

import re

from src.schemas import PlannedCall, ProjectError


ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def validate_requested_conditions(question: str, metadata: dict, call: PlannedCall) -> None:
    unsupported = (
        r"(?:微信|电话|邮件|网页|在线|APP|app)渠道|渠道(?:为|是|等于)",
        r"(?<!未)已解决|(?<!未)已处理",
        r"(?:评分|响应|时长).{0,8}(?:低于|高于|超过|小于|大于|至少|最多|[<>≤≥])",
        r"(?:除|排除|不包括|不包含).{0,20}(?:类别|支付|登录|课程内容|以外)",
        r"发送.{0,12}(?:报告|邮件|客户)|发邮件|删除|标记为|执行代码",
        r"最近|上周|本周|昨天|今天|本月|上个月",
    )
    if any(re.search(pattern, question, re.I) for pattern in unsupported):
        raise ProjectError("UNSUPPORTED", "当前工具无法完整表达该条件或操作，请明确支持的类别及绝对日期范围。")
    if re.search(r"\d{1,2}月", question) and not re.search(r"\d{4}[-/年]", question):
        raise ProjectError("UNSUPPORTED", "月份缺少年份，请提供完整起止日期，不能根据数据集推测年份。")
    mentioned = [category for category in metadata.get("categories", []) if category in question]
    all_categories = bool(re.search(r"(?:全部|所有|各)类别", question))
    if len(mentioned) > 1 and not all_categories:
        raise ProjectError("UNSUPPORTED", "当前只支持一个类别或全部类别，不支持多类别集合。")
    if len(mentioned) == 1 and not all_categories and call.intent.category != mentioned[0]:
        raise ProjectError("MODEL_INTENT", "类别参数与用户明确指定的类别不一致，请保留该过滤条件。", True)
    if all_categories and call.intent.category is not None:
        raise ProjectError("MODEL_INTENT", "用户要求所有类别，category 应为 null。", True)
    dates = ISO_DATE.findall(question)
    actual = (str(call.intent.date_from), str(call.intent.date_to))
    if len(dates) == 2 and actual != tuple(dates):
        raise ProjectError("MODEL_INTENT", "日期参数必须与用户明确给出的起止日期一致，不得遗漏或扩展。", True)
    if len(dates) == 1 and re.search(r"当天|当日", question) and actual != (dates[0], dates[0]):
        raise ProjectError("MODEL_INTENT", "用户指定单日，起止日期应均为该日。", True)
    if re.search(r"未解决|未处理", question) and call.tool_name != "unresolved_report":
        raise ProjectError("MODEL_INTENT", "用户要求未解决工单，应使用 unresolved_report 保留状态过滤。", True)
