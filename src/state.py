"""Only serializable per-task progress belongs in graph state."""

from dataclasses import dataclass
from typing import TypedDict

from src.audit import AuditLog
from src.config import Settings
from src.datasets import DatasetStore
from src.models import ModelBackend


class TaskState(TypedDict, total=False):
    task_id: str
    request_id: str
    question: str
    dataset_id: str
    source: str
    require_approval: bool
    backend: str
    status: str
    plan: list[str]
    current_step: int
    planned_call: dict
    results: dict
    last_result: dict
    attempts: dict[str, int]
    model_calls: int
    usage: dict[str, int]
    active_seconds: float
    error: dict | None
    report: str
    report_file: str


@dataclass(frozen=True)
class ExecutionContext:
    settings: Settings
    store: DatasetStore
    model: ModelBackend
    audit: AuditLog
