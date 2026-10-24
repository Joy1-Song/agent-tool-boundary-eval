"""Central configuration. Paths never depend on the terminal working directory."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    runtime_dir: Path = PROJECT_ROOT / ".runtime"
    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    max_upload_bytes: int = 2 * 1024 * 1024
    max_rows: int = 5000
    max_evidence_rows: int = 12
    max_evidence_chars: int = 400
    max_attempts: int = 3
    max_model_calls: int = 6
    request_timeout: float = 20.0
    active_budget_seconds: float = 90.0
    retry_delay: float = 0.25
    # How a planned tool call is checked against the request: regex | audit | both.
    # Default stays "regex" until the v2 live comparison shows another option is better.
    condition_check: str = "regex"

    def __post_init__(self):
        for name in ("max_attempts", "max_model_calls", "max_rows", "max_evidence_rows", "max_upload_bytes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.request_timeout <= 0 or self.active_budget_seconds <= 0 or self.retry_delay < 0:
            raise ValueError("Invalid timeout or retry delay")
        if self.condition_check not in {"regex", "audit", "both"}:
            raise ValueError("condition_check must be regex, audit or both")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", cls.base_url),
            condition_check=os.getenv("CONDITION_CHECK", "regex"),
        )
