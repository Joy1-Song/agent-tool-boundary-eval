from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest

from src.models import OfflineModel
from src.requests import RequestStore
from src.schemas import ProjectError
from src.workflow import AgentRunner


def test_only_one_concurrent_claim_acquires_execution(settings):
    store = RequestStore(settings.runtime_dir)
    barrier = Barrier(6)
    def claim(index):
        barrier.wait(timeout=10)
        try:
            store.claim("same-key", "same-digest", str(index))
            return "owner"
        except ProjectError as error:
            return error.info.code
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(claim, range(6)))
    assert results.count("owner") == 1
    assert results.count("REQUEST_IN_PROGRESS") == 5


def test_uncertain_request_cannot_be_automatically_reclaimed(settings):
    store = RequestStore(settings.runtime_dir)
    store.claim("key", "digest", "owner")
    store.mark_uncertain("key", "owner")
    with pytest.raises(ProjectError) as error:
        RequestStore(settings.runtime_dir).claim("key", "digest", "other")
    assert error.value.info.code == "REQUEST_IN_PROGRESS"


def test_report_written_but_result_save_failure_blocks_reexecution(settings, registered, monkeypatch):
    data, dataset = registered
    runner = AgentRunner(settings, OfflineModel(), data)
    def failed_save(*args):
        raise OSError("Injected registry disk failure")
    monkeypatch.setattr(runner.requests, "save", failed_save)
    with pytest.raises(OSError):
        runner.run("分类统计", dataset, require_approval=False, request_id="save-failure")
    reports = list((settings.runtime_dir / "reports").glob("*.md"))
    assert len(reports) == 1
    with pytest.raises(ProjectError) as error:
        AgentRunner(settings, OfflineModel(), data).run("分类统计", dataset, require_approval=False, request_id="save-failure")
    assert error.value.info.code == "REQUEST_IN_PROGRESS"
    assert list((settings.runtime_dir / "reports").glob("*.md")) == reports


def test_same_key_cannot_change_dataset(settings, registered, sample_bytes):
    store, dataset = registered
    first = AgentRunner(settings, OfflineModel(), store)
    first.run("分类统计", dataset, require_approval=False, request_id="dataset-key")
    other = store.register(sample_bytes.replace(b"T001", b"T101"))
    with pytest.raises(ProjectError) as error:
        AgentRunner(settings, OfflineModel(), store).run("分类统计", other, require_approval=False, request_id="dataset-key")
    assert error.value.info.code == "IDEMPOTENCY_CONFLICT"


def test_new_process_replays_completed_result(settings, registered):
    store, dataset = registered
    first = AgentRunner(settings, OfflineModel(), store).run(
        "分类统计", dataset, require_approval=False, request_id="process-key")
    script = '''import json, sys
from pathlib import Path
from src.config import Settings
from src.datasets import DatasetStore
from src.models import OfflineModel
from src.workflow import AgentRunner
class MustNotRun(OfflineModel):
    def plan(self, *a, **k):
        raise AssertionError("Replay called model")
settings = Settings(runtime_dir=Path(sys.argv[1]), retry_delay=0)
store = DatasetStore(settings)
dataset = store.register(Path(sys.argv[2]).read_bytes())
result = AgentRunner(settings, MustNotRun(), store).run(sys.argv[3], dataset, require_approval=False, request_id="process-key")
print(json.dumps({"task_id": result["task_id"], "status": result["status"]}))
'''
    completed = subprocess.run([sys.executable, "-c", script, str(settings.runtime_dir),
                               str(store.root / f"{dataset}.csv"), "分类统计"],
                              cwd=Path(__file__).resolve().parents[1], env=dict(os.environ),
                              capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    response = json.loads(completed.stdout.strip())
    assert response == {"task_id": first["task_id"], "status": "succeeded"}


def test_repeated_approval_cannot_write_twice(settings, registered):
    store, dataset = registered
    runner = AgentRunner(settings, OfflineModel(), store)
    runner.run("分类统计", dataset, request_id="approve-key")
    result = runner.resume(True)
    target = Path(result["report_file"])
    timestamp = target.stat().st_mtime_ns
    with pytest.raises(ProjectError):
        runner.resume(True)
    assert target.stat().st_mtime_ns == timestamp
