"""Explicit contracts for model arguments, tool results and report evidence."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskType = Literal["category_overview", "unresolved_report", "rating_analysis"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filters(Contract):
    category: str | None = Field(default=None, max_length=80)
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class Intent(Filters):
    task_type: TaskType


class PlannedCall(Contract):
    intent: Intent
    call_id: str = Field(min_length=1, max_length=200)
    tool_name: TaskType
    arguments: str = Field(max_length=2000)

    @model_validator(mode="after")
    def same_task(self):
        if self.intent.task_type != self.tool_name:
            raise ValueError("Tool and intent mismatch")
        return self


class Observation(Contract):
    text: str = Field(min_length=1, max_length=400)
    ticket_ids: list[str] = Field(min_length=1, max_length=8)


class Summary(Contract):
    observations: list[Observation] = Field(default_factory=list, max_length=5)
    recommendations: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def bounded_recommendations(self):
        if any(not item.strip() or len(item) > 400 for item in self.recommendations):
            raise ValueError("Recommendations must be nonempty and at most 400 characters")
        return self


class ErrorInfo(Contract):
    code: str
    message: str
    retryable: bool = False


class ToolResult(Contract):
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def consistent_result(self):
        if self.success == (self.error is not None):
            raise ValueError("Success must have no error; failure must have an error")
        return self


class ProjectError(Exception):
    """A sanitized error safe to display. Never wrap a raw provider exception here."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.info = ErrorInfo(code=code, message=message, retryable=retryable)
