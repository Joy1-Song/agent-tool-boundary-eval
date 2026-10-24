"""Offline check: does the regex guard (src/intent_guard.py) generalize beyond the 20 cases it was tuned on?

No model and no API key. For every benchmark case we feed the guard *simulated* planner outputs:
  - supported case, correct call         -> guard should pass   (a block here = false refusal)
  - supported case, one condition dropped -> guard should block (catch rate for silently lost filters)
  - unsupported case, model executes anyway with the conditions it can express -> guard should block

This measures the guard as a safety net in isolation. It is NOT model accuracy; the live comparison
lives in intent_benchmark_v2.py.

Usage (from the repository root):  python experiments/intent_guard_generalization.py  ->  results/guard_offline.json
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
BENCH = PROJECT / "data/intent_benchmark_v2.json"
OUT = ROOT / "results"

sys.path.insert(0, str(PROJECT))
from src.intent_guard import validate_requested_conditions  # noqa: E402
from src.schemas import Intent, PlannedCall, ProjectError  # noqa: E402

ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def make_call(task, category=None, date_from=None, date_to=None) -> PlannedCall:
    intent = Intent(task_type=task, category=category, date_from=date_from, date_to=date_to)
    args = intent.model_dump_json(exclude={"task_type"})
    return PlannedCall(intent=intent, call_id="sim", tool_name=task, arguments=args)


def blocked(question, metadata, call) -> bool:
    try:
        validate_requested_conditions(question, metadata, call)
        return False
    except ProjectError:
        return True


def naive_execution(question, metadata) -> PlannedCall:
    """What a planner that ignores the unsupported part would plausibly call."""
    if re.search(r"未解决|未处理|没解决|没处理|待处理|未结|开着|没搞定|未闭环|unresolved", question, re.I):
        task = "unresolved_report"
    elif re.search(r"评分|评价|打分|星|满意|好评|差评|平均分|rating", question, re.I):
        task = "rating_analysis"
    else:
        task = "category_overview"
    mentioned = [c for c in metadata["categories"] if c in question]
    dates = ISO.findall(question)
    return make_call(task, mentioned[0] if len(mentioned) == 1 else None,
                     dates[0] if len(dates) == 2 else None, dates[1] if len(dates) == 2 else None)


def dropped_variants(expected):
    """Each supported condition removed once (the 'format valid but filter silently lost' failure)."""
    base = {k: expected[k] for k in ("category", "date_from", "date_to")}
    task = expected["task_type"]
    if expected["category"]:
        yield "category", make_call(task, None, base["date_from"], base["date_to"])
    if expected["date_from"]:
        yield "date", make_call(task, base["category"], None, None)
    if task == "unresolved_report":
        yield "status", make_call("category_overview", **base)


def rate(n, d):
    return None if d == 0 else round(n / d, 4)


def main():
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    meta = bench["metadata"]
    rows = []
    for case in bench["cases"]:
        q, exp = case["question"], case["expected"]
        row = {"id": case["id"], "split": case["split"], "tags": case.get("tags", []), "question": q}
        if exp["action"] == "execute":
            correct = make_call(exp["task_type"], exp["category"], exp["date_from"], exp["date_to"])
            row["kind"] = "supported"
            row["false_refusal"] = blocked(q, meta, correct)
            row["drops"] = {name: blocked(q, meta, call) for name, call in dropped_variants(exp)}
        else:
            row["kind"] = "unsupported"
            row["blocked"] = blocked(q, meta, naive_execution(q, meta))
        rows.append(row)

    summary = {}
    for split in ("development", "holdout"):
        part = [r for r in rows if r["split"] == split]
        sup = [r for r in part if r["kind"] == "supported"]
        uns = [r for r in part if r["kind"] == "unsupported"]
        drops = [ok for r in sup for ok in r["drops"].values()]
        summary[split] = {
            "unsupported_cases": len(uns),
            "unsupported_blocked": sum(r["blocked"] for r in uns),
            "unsupported_block_rate": rate(sum(r["blocked"] for r in uns), len(uns)),
            "supported_cases": len(sup),
            "false_refusals": sum(r["false_refusal"] for r in sup),
            "false_refusal_rate": rate(sum(r["false_refusal"] for r in sup), len(sup)),
            "dropped_condition_variants": len(drops),
            "dropped_condition_caught": sum(drops),
            "dropped_condition_catch_rate": rate(sum(drops), len(drops)),
        }
    by_tag = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["split"] == "holdout" and r["kind"] == "unsupported":
            for tag in r["tags"]:
                by_tag[tag][0] += r["blocked"]
                by_tag[tag][1] += 1
    summary["holdout_unsupported_by_tag"] = {t: f"{b}/{n}" for t, (b, n) in sorted(by_tag.items())}
    summary["holdout_missed_unsupported"] = [r["question"] for r in rows
                                             if r["split"] == "holdout" and r["kind"] == "unsupported" and not r["blocked"]]
    summary["holdout_false_refusals"] = [r["question"] for r in rows
                                         if r["split"] == "holdout" and r["kind"] == "supported" and r["false_refusal"]]

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"benchmark": bench["version"], "scope": __doc__.strip().splitlines()[0], "summary": summary, "rows": rows}
    (OUT / "guard_offline.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("development", "holdout")}, ensure_ascii=False, indent=1))
    print("holdout unsupported blocked by tag:", summary["holdout_unsupported_by_tag"])
    return report


if __name__ == "__main__":
    main()
