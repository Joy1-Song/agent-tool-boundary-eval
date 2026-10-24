"""Evidence ID validation and deterministic Markdown report assembly."""

import re

from src.schemas import Intent, ProjectError, Summary


def validate_summary(summary: Summary, facts: dict) -> None:
    allowed = {row["ticket_id"] for row in facts["evidence"]}
    for observation in summary.observations:
        if not set(observation.ticket_ids).issubset(allowed):
            raise ProjectError("CITATION", "摘要引用了输入证据以外的工单 ID，请修复引用。", True)
        if re.search(r"\d|评分|均值|平均|响应时|响应记录|时长|SLA|百分|占比|统计覆盖", observation.text, re.I):
            raise ProjectError("SUMMARY_METRIC", "摘要只归纳原文问题，不得包含数字、评分或时长等统计指标。", True)
    if facts["total"] and not summary.observations:
        raise ProjectError("EMPTY_SUMMARY", "有匹配工单时需要至少一个有依据的摘要要点。", True)
    if not facts["total"] and (summary.observations or summary.recommendations):
        raise ProjectError("EMPTY_EVIDENCE", "没有匹配工单时不能生成无依据的摘要或建议。")


def escape_text(value: str) -> str:
    """Treat model and uploaded text as text, never active Markdown links or HTML."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|<>])", r"\\\1", value.replace("\r", " ").replace("\n", " "))


def render_report(task_id: str, dataset_id: str, intent: Intent, facts: dict, summary: Summary, backend: str) -> str:
    validate_summary(summary, facts)
    labels = {"category_overview": "工单类别分析", "unresolved_report": "未解决工单分析", "rating_analysis": "工单评分分析"}
    mean_rating = "无有效评分" if facts["rating_mean"] is None else f"{facts['rating_mean']:.2f} 分"
    mean_response = "无有效记录" if facts["response_mean"] is None else f"{facts['response_mean']:.2f} 分钟"
    lines = [
        f"# {labels[intent.task_type]}", "", f"任务 ID：`{task_id}`", f"数据 SHA-256：`{dataset_id}`",
        f"执行方式：{escape_text(backend)}，数据访问：{facts['source']}",
        f"筛选范围：类别 {escape_text(intent.category or '全部')}，日期 {intent.date_from or '不限'} 至 {intent.date_to or '不限'}", "",
        "## 统计事实", "", f"原数据 {facts['dataset_rows']} 条，本次匹配 {facts['total']} 条。",
        f"匹配记录中已解决 {facts['resolved']} 条，未解决 {facts['unresolved']} 条。",
        f"平均评分 {mean_rating}，有效评分样本 {facts['rating_count']} 条。",
        f"平均首次响应 {mean_response}，有效响应样本 {facts['response_count']} 条。", "",
        "| 类别 | 工单数 | 有效评分数 | 平均评分 |", "|---|---:|---:|---:|",
    ]
    for row in facts["by_category"]:
        score = "无数据" if row["rating_mean"] is None else f"{row['rating_mean']:.2f}"
        lines.append(f"| {escape_text(row['category'])} | {row['count']} | {row['rating_count']} | {score} |")
    lines += ["", "## 抽样原文归纳", "", f"统计覆盖全部匹配记录，文本归纳只读取 {facts['evidence_count']} 条证据。",
              f"每条正文最多 {facts['evidence_char_limit']} 字符。取样方式：{facts['sampling']}。", ""]
    if not facts["total"]:
        lines.append("当前筛选范围没有匹配工单，没有调用模型生成摘要。")
    for item in summary.observations:
        lines.append(f"- {escape_text(item.text)}（工单：{', '.join(item.ticket_ids)}）")
    if summary.recommendations:
        lines += ["", "## 待验证建议", ""]
        lines.extend(f"- {escape_text(item)}" for item in summary.recommendations)
    lines += ["", "## 依据与限制", "", "引用 ID 已检查属于本次输入，但未自动证明每句话都被原文支持，仍需人工复核。",
              "首次响应时间不等于解决耗时。本报告不是线上业务收益证明。"]
    if backend == "offline-rule":
        lines.append("当前使用离线规则路由与模板摘要，没有调用大语言模型。")
    return "\n".join(lines) + "\n"
