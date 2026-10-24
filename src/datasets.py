"""Validate uploaded CSV bytes and bind their content hash to a local snapshot."""

import csv
import hashlib
import io
import math
import re
from datetime import date

import pandas as pd

from src.config import Settings
from src.schemas import ProjectError


COLUMNS = ["ticket_id", "content", "category", "channel", "is_resolved", "response_time", "rating", "created_at"]


def parse_csv(content: bytes, settings: Settings) -> pd.DataFrame:
    if not content or len(content) > settings.max_upload_bytes:
        raise ProjectError("FILE_SIZE", "CSV 为空或超过 2 MB 限制。")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ProjectError("ENCODING", "请使用 UTF-8 编码的 CSV。") from error
    if "\x00" in text:
        raise ProjectError("CSV_FORMAT", "CSV 中包含不支持的空字节。")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    records, ids = [], set()
    try:
        fields = reader.fieldnames or []
        if set(fields) != set(COLUMNS) or len(fields) != len(COLUMNS):
            raise ProjectError("COLUMNS", "CSV 必须包含规定的 8 个字段，且不能有重复或额外表头。")
        for row in reader:
            if len(records) >= settings.max_rows:
                raise ProjectError("ROW_LIMIT", "CSV 超过允许的最大工单数。")
            if None in row or any(value is None for value in row.values()):
                raise ProjectError("CSV_FORMAT", f"第 {reader.line_num} 行单元格数量与表头不一致。")
            item = {key: value.strip() for key, value in row.items()}
            for key in ("ticket_id", "content", "category", "channel", "created_at"):
                if not item[key]:
                    raise ProjectError("EMPTY_FIELD", f"第 {reader.line_num} 行的 {key} 不能为空。")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item["ticket_id"]):
                raise ProjectError("TICKET_ID", "工单 ID 只能包含字母、数字、下划线和连字符，最多 64 字符。")
            if item["ticket_id"] in ids:
                raise ProjectError("DUPLICATE_ID", f"第 {reader.line_num} 行出现重复工单 ID。")
            if len(item["content"]) > 2000 or len(item["category"]) > 80 or len(item["channel"]) > 80:
                raise ProjectError("TEXT_LENGTH", "正文最多 2000 字符，类别和渠道最多 80 字符。")
            flag = item["is_resolved"].lower()
            if flag not in {"true", "false"}:
                raise ProjectError("BOOLEAN", "is_resolved 只能为 true 或 false。")
            item["is_resolved"] = flag == "true"
            for key in ("rating", "response_time"):
                try:
                    value = None if item[key] == "" else float(item[key])
                except ValueError as error:
                    raise ProjectError("NUMBER", f"{key} 必须为数字或空值。") from error
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ProjectError("NUMBER", f"{key} 必须为有限的非负数。")
                if key == "rating" and value is not None and (not value.is_integer() or not 1 <= value <= 5):
                    raise ProjectError("RATING", "rating 只能为 1 到 5 的整数或空值。")
                item[key] = value
            try:
                parsed_date = date.fromisoformat(item["created_at"])
                if parsed_date.isoformat() != item["created_at"]:
                    raise ValueError("Noncanonical date")
            except ValueError as error:
                raise ProjectError("DATE", "created_at 必须是有效的 YYYY-MM-DD 日期。") from error
            records.append(item)
            ids.add(item["ticket_id"])
    except csv.Error as error:
        raise ProjectError("CSV_FORMAT", "CSV 引号或行格式不合法。") from error
    if not records:
        raise ProjectError("EMPTY_DATA", "CSV 中没有工单记录。")
    return pd.DataFrame(records, columns=COLUMNS)


class DatasetStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = (settings.runtime_dir / "datasets").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._owned_ids: set[str] = set()

    def register(self, content: bytes) -> str:
        parse_csv(content, self.settings)
        dataset_id = hashlib.sha256(content).hexdigest()
        target = self.root / f"{dataset_id}.csv"
        if target.resolve().parent != self.root:
            raise ProjectError("DATASET_ACCESS", "数据文件位置不合法。")
        if not target.exists():
            # The hash identifies an immutable snapshot; never accept a model-supplied path.
            with target.open("xb") as handle:
                handle.write(content)
        self._owned_ids.add(dataset_id)
        return dataset_id

    def read(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self._owned_ids or not re.fullmatch(r"[a-f0-9]{64}", dataset_id):
            raise ProjectError("DATASET_ACCESS", "本次会话未注册该数据集。")
        target = self.root / f"{dataset_id}.csv"
        if target.resolve().parent != self.root:
            raise ProjectError("DATASET_ACCESS", "数据文件位置不合法。")
        try:
            content = target.read_bytes()
        except OSError as error:
            raise ProjectError("DATASET_MISSING", "数据快照不可读取，请重新上传。") from error
        if hashlib.sha256(content).hexdigest() != dataset_id:
            raise ProjectError("DATASET_CHANGED", "数据快照已改变，拒绝使用旧 ID 分析。")
        return parse_csv(content, self.settings)
