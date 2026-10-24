"""A bounded LangGraph workflow with explicit tool execution and a real approval interrupt."""

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from src.audit import AuditLog
from src.config import Settings
from src.datasets import DatasetStore
from src.models import PROMPT_VERSION, ModelBackend, ModelReply
from src.requests import RequestStore, fingerprint
from src.schemas import ErrorInfo, PlannedCall, ProjectError, Summary, ToolResult
from src.state import ExecutionContext, TaskState
from src.tools.data_tools import aggregate_csv, inspect_csv, validate_filters
from src.tools.report_tools import render_report, validate_summary
from src.tools.sql_tools import query_sql


def run_attempt(state: TaskState, context: ExecutionContext, node: str, tool: str,
                operation: Callable, model_call: bool = False) -> dict:
    """Each physical request counts once. Successful prior steps remain in state."""
    settings = context.settings
    attempt = state["attempts"].get(tool, 0) + 1
    started = time.monotonic()
    calls = state["model_calls"]
    usage = dict(state["usage"])
    try:
        remaining = settings.active_budget_seconds - state["active_seconds"]
        deadline = started + remaining
        if remaining <= 0 or attempt > settings.max_attempts or (model_call and calls >= settings.max_model_calls):
            raise ProjectError("BUDGET", "达到任务耗时、调用次数或尝试次数上限。")
        if attempt > 1:
            delay = settings.retry_delay * (2 ** (attempt - 2))
            if delay >= remaining:
                raise ProjectError("BUDGET", "剩余预算不足以再次尝试。")
            time.sleep(delay)
        # A scheduler can wake us later than the requested backoff interval.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProjectError("BUDGET", "当前步骤开始前已超过任务执行预算。")
        if model_call:
            calls += 1
        payload = operation(min(settings.request_timeout, remaining))
        if isinstance(payload, ModelReply):
            for key, value in payload.usage.items():
                usage[key] = usage.get(key, 0) + value
            data = payload.value.model_dump(mode="json")
        else:
            data = payload
        if time.monotonic() - started > settings.active_budget_seconds - state["active_seconds"]:
            raise ProjectError("BUDGET", "当前步骤完成时已超过任务执行预算。")
        result = ToolResult(success=True, data=data)
    except ProjectError as error:
        result = ToolResult(success=False, error=error.info)
    except OSError:
        result = ToolResult(success=False, error=ErrorInfo(code="IO_ERROR", message="本地文件读写失败。"))
    except Exception:
        result = ToolResult(success=False, error=ErrorInfo(code="INTERNAL", message="执行失败，请根据步骤日志定位问题。"))
    elapsed = time.monotonic() - started
    try:
        context.audit.emit(state["task_id"], node, tool, "success" if result.success else "failed",
                           attempt=attempt, latency_ms=round(elapsed * 1000, 2), model_calls=calls,
                           error_code=None if result.success else result.error.code, prompt_version=PROMPT_VERSION)
    except OSError:
        result = ToolResult(success=False, error=ErrorInfo(code="LOG_IO", message="执行日志无法写入，任务停止。"))
    retryable = (not result.success and result.error.retryable and attempt < settings.max_attempts
                 and state["active_seconds"] + elapsed < settings.active_budget_seconds
                 and (not model_call or calls < settings.max_model_calls))
    status = "running" if result.success or retryable else "failed"
    if not result.success and result.error.code == "UNSUPPORTED":
        status = "rejected"
    return {
        "last_result": result.model_dump(mode="json"), "status": status,
        "attempts": {**state["attempts"], tool: attempt}, "model_calls": calls, "usage": usage,
        "active_seconds": state["active_seconds"] + elapsed,
        "error": None if result.success else result.error.model_dump(),
    }


def planner(state: TaskState, runtime: Runtime[ExecutionContext]) -> dict:
    context = runtime.context

    def operation(timeout):
        frame = context.store.read(state["dataset_id"])
        metadata = inspect_csv(frame)
        metadata["categories"] = metadata["categories"][:30]
        metadata["date_min"] = str(frame["created_at"].min())
        metadata["date_max"] = str(frame["created_at"].max())
        reply = context.model.plan(state["question"], metadata, timeout, (state.get("error") or {}).get("message", ""))
        validate_filters(frame, reply.value.intent)
        return reply

    update = run_attempt(state, context, "planner", "plan", operation, context.model.is_live)
    if update["last_result"]["success"]:
        update["planned_call"] = update["last_result"]["data"]
        update["plan"] = ["inspect_csv", "aggregate_csv" if state["source"] == "csv" else "query_sql", "summarize"]
    return update


def executor(state: TaskState, runtime: Runtime[ExecutionContext]) -> dict:
    context = runtime.context
    tool = state["plan"][state["current_step"]]
    call = PlannedCall.model_validate(state["planned_call"])

    def summarize(timeout):
        facts = state["results"]["statistics"]
        if not facts["total"]:
            return Summary().model_dump()
        reply = context.model.summarize(state["question"], call, facts, timeout,
                                        (state.get("error") or {}).get("message", ""))
        validate_summary(reply.value, facts)
        return reply

    # This registry is the complete executable surface. There is no eval or exec.
    registry = {
        "inspect_csv": lambda timeout: inspect_csv(context.store.read(state["dataset_id"])),
        "aggregate_csv": lambda timeout: aggregate_csv(context.store.read(state["dataset_id"]), call.intent, context.settings),
        "query_sql": lambda timeout: query_sql(context.store.read(state["dataset_id"]), call.intent,
                                                context.settings, state["task_id"]),
        "summarize": summarize,
    }
    uses_model = tool == "summarize" and context.model.is_live and bool(state["results"]["statistics"]["total"])
    return run_attempt(state, context, "executor", tool, registry[tool], uses_model)


def checker(state: TaskState) -> dict:
    if not state["last_result"]["success"]:
        return {}
    tool = state["plan"][state["current_step"]]
    key = "statistics" if tool in {"aggregate_csv", "query_sql"} else tool
    index = state["current_step"] + 1
    status = "running" if index < len(state["plan"]) else ("awaiting_review" if state["require_approval"] else "finalizing")
    # No custom reducer: return a merged dictionary explicitly to retain earlier results.
    return {"results": {**state["results"], key: state["last_result"]["data"]}, "current_step": index, "status": status}


def human_review(state: TaskState, runtime: Runtime[ExecutionContext]) -> dict:
    decision = interrupt({
        "task_id": state["task_id"], "matched_rows": state["results"]["statistics"]["total"],
        "summary": state["results"]["summarize"], "action": "生成本地报告",
    })
    approved = isinstance(decision, dict) and decision.get("approved") is True
    # Code before interrupt can run again on resume; do not put side effects there.
    runtime.context.audit.emit(state["task_id"], "human_review", "approval", "approved" if approved else "rejected")
    return {"status": "finalizing" if approved else "rejected", "error": None if approved else {
        "code": "USER_REJECTED", "message": "用户拒绝生成报告。", "retryable": False,
    }}


def finalizer(state: TaskState, runtime: Runtime[ExecutionContext]) -> dict:
    context = runtime.context

    def operation(timeout):
        call = PlannedCall.model_validate(state["planned_call"])
        summary = Summary.model_validate(state["results"]["summarize"])
        report = render_report(state["task_id"], state["dataset_id"], call.intent,
                               state["results"]["statistics"], summary, state["backend"])
        directory = context.settings.runtime_dir / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{state['task_id']}.md"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(target)
        return {"report": report, "report_file": str(target)}

    update = run_attempt(state, context, "finalizer", "build_report", operation)
    if update["last_result"]["success"]:
        update.update(update["last_result"]["data"])
        update["status"] = "succeeded"
    else:
        update["status"] = "failed"
    return update


def build_graph():
    graph = StateGraph(TaskState, context_schema=ExecutionContext)
    for name, node in [("planner", planner), ("executor", executor), ("checker", checker),
                       ("human_review", human_review), ("finalizer", finalizer)]:
        graph.add_node(name, node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", lambda state: "executor" if state["last_result"]["success"]
                                else ("planner" if state["status"] == "running" else END))
    graph.add_edge("executor", "checker")
    graph.add_conditional_edges("checker", lambda state: {
        "running": "executor", "awaiting_review": "human_review", "finalizing": "finalizer",
    }.get(state["status"], END))
    graph.add_conditional_edges("human_review", lambda state: "finalizer" if state["status"] == "finalizing" else END)
    graph.add_edge("finalizer", END)
    return graph.compile(checkpointer=InMemorySaver())


class AgentRunner:
    """One owned graph run; duplicate request IDs replay a durable local snapshot."""

    def __init__(self, settings: Settings, model: ModelBackend, store: DatasetStore | None = None):
        self.settings = settings
        self.store = store or DatasetStore(settings)
        self.audit = AuditLog(settings.runtime_dir)
        self.context = ExecutionContext(settings, self.store, model, self.audit)
        self.graph = build_graph()
        self.task_id: str | None = None
        self.request_id: str | None = None
        self.requests = RequestStore(settings.runtime_dir)
        self.owner = uuid.uuid4().hex
        self.replayed: dict | None = None

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.task_id}, "recursion_limit": 40}

    def run(self, question: str, dataset_id: str, source: str = "csv", require_approval: bool = True,
            *, request_id: str | None = None) -> dict:
        if self.task_id is not None and (request_id is None or request_id != self.request_id):
            raise ProjectError("RUN_USED", "每个 Runner 只运行一个任务，请新建 Runner。")
        if (not isinstance(question, str) or not question.strip() or len(question) > 1000
                or source not in {"csv", "sqlite"} or type(require_approval) is not bool):
            raise ProjectError("REQUEST", "任务不能为空或超过 1000 字符，数据源必须为 csv 或 sqlite。")
        self.store.read(dataset_id)
        configuration = asdict(self.settings)
        configuration.pop("runtime_dir")
        digest = fingerprint({"version": 1, "question": question.strip(), "dataset": dataset_id,
                              "source": source, "approval": require_approval, "backend": self.context.model.name,
                              "prompt": PROMPT_VERSION, "settings": configuration})
        key = uuid.uuid4().hex if request_id is None else request_id
        task_id, previous = self.requests.claim(key, digest, self.owner)
        self.task_id, self.request_id = task_id, key
        if previous is not None:
            # Reading a saved result never transfers ownership of its in-memory checkpoint.
            if not self.graph.get_state(self.config).values:
                self.replayed = previous
            return previous
        initial = {
            "task_id": self.task_id, "request_id": self.request_id,
            "question": question.strip(), "dataset_id": dataset_id, "source": source,
            "require_approval": require_approval, "backend": self.context.model.name, "status": "running",
            "plan": [], "current_step": 0, "planned_call": {}, "results": {}, "last_result": {},
            "attempts": {}, "model_calls": 0, "usage": {}, "active_seconds": 0.0, "error": None,
            "report": "", "report_file": "",
        }
        return self._invoke(initial)

    def _invoke(self, value) -> dict:
        try:
            self.graph.invoke(value, self.config, context=self.context)
            result = self.snapshot()
            self.requests.save(self.request_id, self.owner, result)
            return result
        except BaseException:
            # Never automatically reclaim an uncertain operation: a side effect may have completed.
            try:
                self.requests.mark_uncertain(self.request_id, self.owner)
            except Exception:
                pass
            raise

    def snapshot(self) -> dict:
        if self.replayed is not None:
            return dict(self.replayed)
        return dict(self.graph.get_state(self.config).values)

    def resume(self, approved: bool) -> dict:
        if (type(approved) is not bool or self.task_id is None or self.replayed is not None
                or self.snapshot().get("status") != "awaiting_review"):
            raise ProjectError("APPROVAL_STATE", "只有正在等待确认的任务可以批准或拒绝。")
        self.requests.begin_resume(self.request_id, self.owner)
        return self._invoke(Command(resume={"approved": approved}))
