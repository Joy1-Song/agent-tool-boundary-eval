"""Run the same workflow without the web UI: python -m src.cli."""

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT, Settings
from src.datasets import DatasetStore
from src.models import OfflineModel, QwenModel
from src.schemas import ProjectError
from src.workflow import AgentRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Ticket analysis agent")
    parser.add_argument("--file", type=Path, default=PROJECT_ROOT / "data/tickets_sample.csv")
    parser.add_argument("--question", default="按类别统计工单数量并生成报告")
    parser.add_argument("--mode", choices=["offline", "qwen"], default="offline")
    parser.add_argument("--source", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--review", action="store_true", help="Require an interactive confirmation before report output")
    parser.add_argument("--request-id", help="Reuse this ID only when retrying the same operation and parameters")
    arguments = parser.parse_args()
    try:
        settings = Settings.from_env()
        store = DatasetStore(settings)
        dataset = store.register(arguments.file.read_bytes())
        runner = AgentRunner(settings, OfflineModel() if arguments.mode == "offline" else QwenModel(settings), store)
        result = runner.run(arguments.question, dataset, arguments.source, arguments.review, request_id=arguments.request_id)
        if result["status"] == "awaiting_review":
            result = runner.resume(input("确认生成本地报告？[y/N] ").strip().lower() == "y")
        print(json.dumps({key: result[key] for key in ("task_id", "status", "backend", "model_calls", "active_seconds", "error")},
                         ensure_ascii=False, indent=2))
        if result["status"] == "succeeded":
            print(result["report"])
            print(f"报告文件：{result['report_file']}")
            return 0
        return 1
    except (ProjectError, OSError) as error:
        message = error.info.message if isinstance(error, ProjectError) else "无法访问本地文件。"
        print(json.dumps({"status": "failed", "message": message}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
