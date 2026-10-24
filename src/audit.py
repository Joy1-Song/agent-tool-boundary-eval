"""Write operational metadata, not API keys, raw prompts or ticket bodies."""

import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, root: Path):
        self.root = root / "logs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.on_event = None

    def emit(self, task_id: str, node: str, tool: str, status: str, **metadata) -> None:
        allowed = {"attempt", "latency_ms", "error_code", "model_calls", "rows", "source", "prompt_version"}
        event = {
            "task_id": task_id,
            "node": node,
            "tool": tool,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **{key: value for key, value in metadata.items() if key in allowed},
        }
        with (self.root / f"{task_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)
