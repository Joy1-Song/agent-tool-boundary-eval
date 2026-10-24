import csv
import io
from dataclasses import replace
from datetime import date

import pytest

from src.datasets import DatasetStore, parse_csv
from src.schemas import Intent, ProjectError
from src.tools.data_tools import aggregate_csv
from src.tools.sql_tools import compile_query, create_snapshot, query_sql, readonly_query


def changed(content, key, value):
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    rows, fields = list(reader), reader.fieldnames
    rows[0][key] = value
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


@pytest.mark.parametrize("key,value,code", [
    ("rating", "8", "RATING"), ("rating", "2.5", "RATING"), ("rating", "NaN", "NUMBER"),
    ("response_time", "-1", "NUMBER"), ("response_time", "inf", "NUMBER"),
    ("is_resolved", "maybe", "BOOLEAN"), ("content", "", "EMPTY_FIELD"),
    ("ticket_id", "../x", "TICKET_ID"), ("ticket_id", "T002", "DUPLICATE_ID"),
    ("created_at", "2026-02-30", "DATE"), ("created_at", "20260901", "DATE"),
    ("content", "x" * 2001, "TEXT_LENGTH"),
])
def test_bad_cells(sample_bytes, settings, key, value, code):
    with pytest.raises(ProjectError) as error:
        parse_csv(changed(sample_bytes, key, value), settings)
    assert error.value.info.code == code


def test_csv_quoting_bom_and_false(sample_bytes, settings):
    frame = parse_csv(b"\xef\xbb\xbf" + sample_bytes, settings)
    assert "error, retry" in frame.iloc[-1]["content"]
    assert not frame.iloc[1]["is_resolved"]


@pytest.mark.parametrize("content,code", [(b"", "FILE_SIZE"), (b"\xff", "ENCODING"),
                                         (b"a,b\n1,2\n", "COLUMNS"), (b"\x00", "CSV_FORMAT")])
def test_bad_files(content, code, settings):
    with pytest.raises(ProjectError) as error:
        parse_csv(content, settings)
    assert error.value.info.code == code


def test_header_only(sample_bytes, settings):
    with pytest.raises(ProjectError, match="没有工单"):
        parse_csv(sample_bytes.split(b"\n")[0] + b"\n", settings)


def test_limits(sample_bytes, settings):
    for altered in [replace(settings, max_upload_bytes=10), replace(settings, max_rows=2)]:
        with pytest.raises(ProjectError):
            parse_csv(sample_bytes, altered)


def test_registered_hash_and_isolation(registered, settings):
    store, dataset = registered
    assert len(store.read(dataset)) == 6
    other = DatasetStore(settings)
    with pytest.raises(ProjectError):
        other.read(dataset)
    with pytest.raises(ProjectError):
        store.read("../../secrets")


def test_changed_snapshot_rejected(registered):
    store, dataset = registered
    (store.root / f"{dataset}.csv").write_bytes(b"changed")
    with pytest.raises(ProjectError) as error:
        store.read(dataset)
    assert error.value.info.code == "DATASET_CHANGED"


def test_statistics(frame, settings):
    facts = aggregate_csv(frame, Intent(task_type="category_overview"), settings)
    assert (facts["total"], facts["unresolved"], facts["rating_count"]) == (6, 2, 5)
    assert facts["rating_mean"] == pytest.approx(3.6)
    assert facts["response_mean"] == 12
    assert facts["category_counts"] == {"登录": 3, "支付": 2, "课程内容": 1}


@pytest.mark.parametrize("intent", [
    Intent(task_type="category_overview"), Intent(task_type="unresolved_report"),
    Intent(task_type="rating_analysis", category="支付"),
    Intent(task_type="category_overview", date_from=date(2030, 1, 1)),
])
def test_sql_csv_equivalence(frame, settings, intent):
    csv_result = aggregate_csv(frame, intent, settings)
    sql_result = query_sql(frame, intent, settings, "test")
    assert {k: v for k, v in csv_result.items() if k != "source"} == {k: v for k, v in sql_result.items() if k != "source"}


def test_sql_is_parameterized():
    attack = "x' OR 1=1 --"
    sql, args = compile_query(Intent(task_type="rating_analysis", category=attack))
    assert attack not in sql
    assert args == [attack]


def test_readonly_and_return_limit(frame, settings):
    import sqlite3
    from contextlib import closing
    database = create_snapshot(frame, settings.runtime_dir / "db")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM tickets")
    with pytest.raises(ProjectError):
        readonly_query(database, Intent(task_type="category_overview"), max_rows=1)


def test_sampling_keeps_exact_counts(frame, settings):
    facts = aggregate_csv(frame, Intent(task_type="category_overview"), replace(settings, max_evidence_rows=2))
    assert facts["total"] == 6 and facts["evidence_count"] == 2


def test_unknown_category(frame, settings):
    with pytest.raises(ProjectError, match="类别"):
        aggregate_csv(frame, Intent(task_type="category_overview", category="不存在"), settings)
