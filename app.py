"""Streamlit is a thin UI. Analysis and approval run through AgentRunner."""

import json
import os
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT, Settings
from src.datasets import DatasetStore
from src.models import OfflineModel, QwenModel
from src.requests import fingerprint
from src.schemas import ProjectError
from src.workflow import AgentRunner


st.set_page_config(page_title="工单分析工作台", layout="wide")
settings = Settings.from_env()
if "store" not in st.session_state:
    st.session_state.store = DatasetStore(settings)
store = st.session_state.store
result = st.session_state.get("result", {})
pending = result.get("status") == "awaiting_review"

with st.sidebar:
    st.header("分析设置")
    data_choice = st.radio("数据集", ["示例工单", "上传 CSV"], disabled=pending)
    uploaded = st.file_uploader("工单 CSV", type=["csv"], disabled=pending) if data_choice == "上传 CSV" else None
    source_label = st.radio("数据访问", ["CSV", "SQLite"], horizontal=True, disabled=pending)
    mode = st.radio("执行方式", ["离线规则", "通义千问"], disabled=pending)
    require_approval = st.checkbox("生成报告前人工确认", value=True, disabled=pending)
    consent = False
    if mode == "通义千问":
        st.caption(f"模型：{settings.model}")
        st.caption("API Key 已配置" if os.getenv("DASHSCOPE_API_KEY") else "API Key 未配置")
        consent = st.checkbox("同意将任务和抽样工单发送到阿里云", disabled=pending)

st.title("工单分析工作台")
st.caption("客服工单与用户反馈")
try:
    content = (PROJECT_ROOT / "data/tickets_sample.csv").read_bytes() if data_choice == "示例工单" else (
        uploaded.getvalue() if uploaded else b""
    )
    if not content:
        st.info("暂无工单文件")
        st.stop()
    dataset_id = store.register(content)
    frame = store.read(dataset_id)
except ProjectError as error:
    st.error(f"{error.info.code}：{error.info.message}")
    st.stop()

if st.session_state.get("last_dataset") != dataset_id and not pending:
    st.session_state.pop("result", None)
    st.session_state.pop("runner", None)
    st.session_state.last_dataset = dataset_id
    result = {}

columns = st.columns(4)
columns[0].metric("工单总数", len(frame))
columns[1].metric("未解决", int((~frame["is_resolved"]).sum()))
columns[2].metric("有效评分", int(frame["rating"].count()))
columns[3].metric("类别", int(frame["category"].nunique()))

analysis_tab, data_tab, trace_tab = st.tabs(["分析任务", "工单数据", "运行记录"])
with data_tab:
    st.caption(f"数据快照：{dataset_id[:16]} · {'人工编写的模拟数据' if data_choice == '示例工单' else '本次上传文件'}")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.bar_chart(frame.groupby("category").size().rename("工单数量"))

with analysis_tab:
    question = st.text_area("分析任务", value="按类别统计工单数量，归纳主要问题并生成报告。", max_chars=1000, disabled=pending)
    execute = st.button("开始分析", icon=":material/play_arrow:", type="primary", disabled=pending or (mode == "通义千问" and not consent))
    if st.button("新建任务", icon=":material/add:", disabled=pending):
        for key in ("submission", "result", "runner"):
            st.session_state.pop(key, None)
        st.rerun()
    if execute:
        request_fingerprint = fingerprint({"question": question.strip(), "dataset": dataset_id,
                                           "source": source_label, "mode": mode, "approval": require_approval,
                                           "model": settings.model, "base_url": settings.base_url})
        submission = st.session_state.get("submission", {})
        if submission.get("fingerprint") != request_fingerprint:
            submission = {"fingerprint": request_fingerprint, "request_id": uuid4().hex}
            st.session_state.submission = submission
        st.session_state.pop("result", None)
        try:
            model = OfflineModel() if mode == "离线规则" else QwenModel(settings)
            runner = AgentRunner(settings, model, store)
            st.session_state.runner = runner
            with st.status("任务执行中", expanded=True) as status:
                runner.audit.on_event = lambda event: status.write(f"{event['node']} / {event['tool']}：{event['status']}")
                try:
                    result = runner.run(question, dataset_id, source="sqlite" if source_label == "SQLite" else "csv",
                                        require_approval=require_approval, request_id=submission["request_id"])
                finally:
                    runner.audit.on_event = None
                status.update(label="等待人工确认" if result["status"] == "awaiting_review" else "执行结束", state="complete")
            st.session_state.result = result
            st.rerun()
        except ProjectError as error:
            st.error(f"{error.info.code}：{error.info.message}")
        except OSError:
            st.error("运行文件或日志无法写入。")

    result = st.session_state.get("result", {})
    if result:
        st.caption(f"任务：{result['task_id'][:12]} · 执行方式：{result['backend']}")
        if result["status"] == "awaiting_review":
            st.warning("等待确认生成本地报告")
            facts = result["results"]["statistics"]
            st.write(f"匹配 {facts['total']} 条，摘要证据 {facts['evidence_count']} 条。")
            for observation in result["results"]["summarize"]["observations"]:
                st.text(f"{observation['text']}  [{', '.join(observation['ticket_ids'])}]")
            left, right = st.columns(2)
            approve = left.button("确认生成", icon=":material/check:", type="primary")
            reject = right.button("拒绝生成", icon=":material/close:")
            if approve or reject:
                try:
                    st.session_state.result = st.session_state.runner.resume(approve)
                    st.rerun()
                except ProjectError as error:
                    st.error(f"{error.info.code}：{error.info.message}")
                except OSError:
                    st.error("运行文件或日志无法写入，请核对原任务状态。")
        elif result["status"] == "succeeded":
            st.success("报告已生成")
            st.download_button("下载报告", result["report"], file_name=f"tickets_{result['task_id'][:12]}.md",
                               mime="text/markdown", icon=":material/download:")
            st.markdown(result["report"], unsafe_allow_html=False)
        else:
            error = result.get("error") or {"code": "UNKNOWN", "message": "任务未完成。"}
            st.error(f"{error['code']}：{error['message']}")

with trace_tab:
    runner = st.session_state.get("runner")
    if runner:
        st.dataframe(pd.DataFrame(runner.audit.events), hide_index=True, width="stretch")
        current = st.session_state.get("result", {})
        st.caption(f"模型请求 {current.get('model_calls', 0)} 次 · 活跃执行 {current.get('active_seconds', 0):.2f} 秒")
        st.json({"attempts": current.get("attempts", {}), "returned_token_usage": current.get("usage", {})}, expanded=False)
        st.download_button("下载步骤日志", json.dumps(runner.audit.events, ensure_ascii=False, indent=2),
                           file_name="run_trace.json", mime="application/json", icon=":material/download:")
    else:
        st.info("暂无运行记录")
