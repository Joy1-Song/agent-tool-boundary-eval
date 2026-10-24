"""Deterministic CSV inspection, filtering and pandas aggregation."""

import pandas as pd

from src.config import Settings
from src.schemas import Intent, ProjectError


def inspect_csv(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "columns": list(frame.columns), "categories": sorted(frame["category"].unique())}


def validate_filters(frame: pd.DataFrame, intent: Intent) -> None:
    if intent.category is not None and intent.category not in set(frame["category"]):
        raise ProjectError("UNKNOWN_CATEGORY", "请求的类别不在数据中，请使用已有类别。")


def filter_frame(frame: pd.DataFrame, intent: Intent) -> pd.DataFrame:
    validate_filters(frame, intent)
    mask = pd.Series(True, index=frame.index)
    if intent.category is not None:
        mask &= frame["category"].eq(intent.category)
    if intent.date_from:
        mask &= frame["created_at"].ge(intent.date_from.isoformat())
    if intent.date_to:
        mask &= frame["created_at"].le(intent.date_to.isoformat())
    if intent.task_type == "unresolved_report":
        mask &= ~frame["is_resolved"].astype(bool)
    return frame.loc[mask].copy()


def aggregate_frame(frame: pd.DataFrame, settings: Settings, dataset_rows: int, source: str) -> dict:
    counts = frame.groupby("category").size().sort_values(ascending=False)
    ratings = frame["rating"].dropna()
    response = frame["response_time"].dropna()
    by_category = []
    for category, group in frame.groupby("category", sort=True):
        values = group["rating"].dropna()
        by_category.append({
            "category": category, "count": len(group), "rating_count": len(values),
            "rating_mean": None if values.empty else round(float(values.mean()), 4),
        })
    sample = frame.sort_values(["created_at", "ticket_id"]).head(settings.max_evidence_rows)
    evidence = [{
        "ticket_id": row.ticket_id, "category": row.category,
        "content": row.content[:settings.max_evidence_chars],
        "truncated": len(row.content) > settings.max_evidence_chars,
    } for row in sample.itertuples()]
    total = len(frame)
    resolved = int(frame["is_resolved"].sum())
    return {
        "source": source, "dataset_rows": dataset_rows, "total": total,
        "resolved": resolved, "unresolved": total - resolved,
        "resolved_rate": None if total == 0 else resolved / total,
        "category_counts": {str(key): int(value) for key, value in counts.items()},
        "rating_count": len(ratings), "rating_mean": None if ratings.empty else float(ratings.mean()),
        "response_count": len(response), "response_mean": None if response.empty else float(response.mean()),
        "by_category": by_category, "evidence": evidence, "evidence_count": len(evidence),
        "evidence_char_limit": settings.max_evidence_chars,
        "sampling": "created_at、ticket_id 升序的前 N 条，不代表随机抽样",
    }


def aggregate_csv(frame: pd.DataFrame, intent: Intent, settings: Settings) -> dict:
    return aggregate_frame(filter_frame(frame, intent), settings, len(frame), "csv")
