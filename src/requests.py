"""Durable local request deduplication, not distributed exactly-once execution."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

from src.schemas import ProjectError


def fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RequestStore:
    def __init__(self, runtime_dir: Path):
        self.root = runtime_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "requests.sqlite3"
        with self.connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, task_id TEXT NOT NULL,
                owner TEXT NOT NULL, status TEXT NOT NULL, result TEXT, report_sha256 TEXT
            )""")

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        except sqlite3.Error as error:
            raise ProjectError("REQUEST_STORE", "请求登记库不可用，请恢复存储后使用原编号重试。") from error
        finally:
            connection.close()

    def claim(self, request_id: str, digest: str, owner: str) -> tuple[str, dict | None]:
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            raise ProjectError("REQUEST_ID", "请求编号只能包含字母、数字、下划线和连字符，最多 128 字符。")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,)).fetchone()
            if row:
                if row["fingerprint"] != digest:
                    raise ProjectError("IDEMPOTENCY_CONFLICT", "请求编号已用于不同参数，请为新的操作使用新编号。")
                if row["status"] in {"running", "uncertain"}:
                    raise ProjectError("REQUEST_IN_PROGRESS", "请求正在执行或结果尚未确认，禁止重复执行；请核对原任务。")
                result = json.loads(row["result"])
                if result["status"] == "succeeded":
                    target = Path(result["report_file"])
                    expected = self.root / "reports" / f"{row['task_id']}.md"
                    try:
                        valid = target.resolve() == expected and hashlib.sha256(target.read_bytes()).hexdigest() == row["report_sha256"]
                    except OSError:
                        valid = False
                    if not valid:
                        raise ProjectError("ARTIFACT_CHANGED", "原报告已缺失或改变，拒绝静默重新执行。")
                return row["task_id"], result
            task_id = uuid.uuid4().hex
            connection.execute("INSERT INTO requests VALUES (?, ?, ?, ?, 'running', NULL, NULL)",
                               (request_id, digest, task_id, owner))
            return task_id, None

    def save(self, request_id: str, owner: str, result: dict) -> None:
        report_hash = None
        if result["status"] == "succeeded":
            report_hash = hashlib.sha256(Path(result["report_file"]).read_bytes()).hexdigest()
        with self.connection() as connection:
            changed = connection.execute(
                "UPDATE requests SET status=?, result=?, report_sha256=? WHERE request_id=? AND owner=? AND status='running'",
                (result["status"], json.dumps(result, ensure_ascii=False), report_hash, request_id, owner),
            ).rowcount
            if changed != 1:
                raise ProjectError("REQUEST_STATE", "请求执行状态不一致，拒绝覆盖既有结果。")

    def begin_resume(self, request_id: str, owner: str) -> None:
        with self.connection() as connection:
            changed = connection.execute(
                "UPDATE requests SET status='running' WHERE request_id=? AND owner=? AND status='awaiting_review'",
                (request_id, owner),
            ).rowcount
            if changed != 1:
                raise ProjectError("APPROVAL_STATE", "只有持有原始检查点的执行实例可以恢复等待确认的任务。")

    def mark_uncertain(self, request_id: str, owner: str) -> None:
        with self.connection() as connection:
            connection.execute("UPDATE requests SET status='uncertain' WHERE request_id=? AND owner=? AND status='running'",
                               (request_id, owner))
