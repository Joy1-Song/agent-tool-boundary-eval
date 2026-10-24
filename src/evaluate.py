"""Replay explicit task contracts. Offline passes are not model accuracy."""

import argparse
import json
from datetime import datetime, timezone
from statistics import mean

from src.config import PROJECT_ROOT, Settings
from src.datasets import DatasetStore
from src.models import OfflineModel, QwenModel
from src.workflow import AgentRunner


def evaluate(mode: str = "offline", source: str = "csv") -> dict:
    settings = Settings.from_env()
    store = DatasetStore(settings)
    dataset = store.register((PROJECT_ROOT / "data/tickets_sample.csv").read_bytes())
    tasks = json.loads((PROJECT_ROOT / "data/eval_tasks.json").read_text(encoding="utf-8"))
    records = []
    for task in tasks:
        backend = OfflineModel() if mode == "offline" else QwenModel(settings)
        runner = AgentRunner(settings, backend, store)
        result = runner.run(task["question"], dataset, source, require_approval=False)
        actual_total = result["results"].get("statistics", {}).get("total")
        passed = result["status"] == task["status"] and ("total" not in task or actual_total == task["total"])
        records.append({"case_id": task["id"], "passed": passed, "status": result["status"], "total": actual_total,
                        "model_calls": result["model_calls"], "active_seconds": result["active_seconds"]})
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "mode": mode, "source": source,
        "dataset_sha256": dataset, "cases": len(records), "passed": sum(row["passed"] for row in records),
        "mean_active_seconds": mean(row["active_seconds"] for row in records), "records": records,
        "scope": "固定任务的状态和匹配数量验收，不评价摘要语义。离线规则结果不等于真实模型准确率。",
    }
    target = settings.runtime_dir / f"evaluation_{mode}_{source}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "qwen"], default="offline")
    parser.add_argument("--source", choices=["csv", "sqlite"], default="csv")
    args = parser.parse_args()
    result = evaluate(args.mode, args.source)
    print(json.dumps({key: result[key] for key in ["mode", "source", "cases", "passed", "mean_active_seconds", "scope"]},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["cases"] == result["passed"] else 1)
