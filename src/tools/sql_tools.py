"""SQLite is a derived snapshot. Model arguments cannot supply SQL statements."""

import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.schemas import Intent, ProjectError
from src.tools.data_tools import aggregate_frame, validate_filters


def create_snapshot(frame: pd.DataFrame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid.uuid4().hex}.db"
    with closing(sqlite3.connect(target)) as connection:
        frame.to_sql("tickets", connection, index=False, if_exists="fail")
        connection.commit()
    return target


def compile_query(intent: Intent) -> tuple[str, list]:
    clauses, parameters = [], []
    if intent.category is not None:
        clauses.append("category = ?")
        parameters.append(intent.category)
    if intent.date_from:
        clauses.append("created_at >= ?")
        parameters.append(intent.date_from.isoformat())
    if intent.date_to:
        clauses.append("created_at <= ?")
        parameters.append(intent.date_to.isoformat())
    if intent.task_type == "unresolved_report":
        clauses.append("is_resolved = ?")
        parameters.append(0)
    sql = "SELECT * FROM tickets"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return sql + " ORDER BY created_at, ticket_id LIMIT ?", parameters


def readonly_query(database: Path, intent: Intent, max_rows: int, timeout: float = 2.0) -> pd.DataFrame:
    sql, parameters = compile_query(intent)
    deadline = time.monotonic() + timeout
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(
                lambda action, table, column, db, trigger: sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and table == "tickets")
                else sqlite3.SQLITE_DENY
            )
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            cursor = connection.execute(sql, [*parameters, max_rows + 1])
            rows = cursor.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                raise ProjectError("ROW_LIMIT", "查询返回行数超过上限，拒绝静默截断统计。")
            return pd.DataFrame.from_records(rows, columns=[item[0] for item in cursor.description])
    except sqlite3.Error as error:
        raise ProjectError("SQL_QUERY", "只读查询失败或超过执行预算。") from error


def query_sql(frame: pd.DataFrame, intent: Intent, settings: Settings, task_id: str) -> dict:
    validate_filters(frame, intent)
    database = create_snapshot(frame, settings.runtime_dir / "databases" / task_id)
    try:
        filtered = readonly_query(database, intent, settings.max_rows)
        return aggregate_frame(filtered, settings, len(frame), "sqlite")
    finally:
        database.unlink(missing_ok=True)
