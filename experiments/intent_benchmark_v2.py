"""Live A/B/C/D comparison on intent_benchmark_v2 (20 development + 100 holdout cases).

Per case: 1 planner call (the project's real QwenModel) + 1 condition-audit call. From those two
responses we score five strategies without extra calls:
  A raw        planner tool call as returned by the model, no checks
  B regex      current production path: planner + intent_guard.py
  C audit      planner + condition audit (condition_audit.py)
  D both       planner + regex guard + condition audit
  E audit-only the audit alone decides the call (diagnostic, ignores the planner)
Single attempt per case (no retry-with-feedback), qwen-plus, temperature=0.

Cost control: requires --approved-live; hard ceiling --max-calls (default 260); every request is
written to a ledger before it is sent and is never replayed. Re-running resumes missing cases.

Usage (from the repository root):
  python experiments/intent_benchmark_v2.py --approved-live --limit 3 --out results/my_run   # smoke test, 6 calls
  python experiments/intent_benchmark_v2.py --approved-live --out results/my_run             # full run, 240 calls
  # round 2 (blind holdout-2, audit-v2 + decision-v2):
  python experiments/intent_benchmark_v2.py --approved-live --bench data/intent_holdout2.json --out results/live_holdout2
  # rebuild a report from saved records (no requests; refuses if src/ or the benchmark changed since the run):
  python experiments/intent_benchmark_v2.py --report --bench data/intent_holdout2.json --out results/live_holdout2
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
BENCH = PROJECT / "data/intent_benchmark_v2.json"
OUT = ROOT / "results/live"
MODEL = "qwen-plus"
STRATEGIES = ("A_raw", "B_regex", "C_audit", "D_both", "E_audit_only")
FIELDS = ("task_type", "category", "date_from", "date_to")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("No DASHSCOPE_API_KEY in the environment; no request sent.")
    return key


class Ledger:
    """Every paid request is recorded before sending. A key is never sent twice."""

    def __init__(self, path: Path, max_calls: int):
        self.path, self.max_calls = path, max_calls
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"requests": []}

    def reserve(self, key: str):
        if any(r["key"] == key for r in self.data["requests"]):
            raise RuntimeError(f"ledger already has {key}; not replaying a paid request")
        if len(self.data["requests"]) >= self.max_calls:
            raise RuntimeError("call budget exhausted")
        self.data["requests"].append({"key": key, "status": "started",
                                      "at": datetime.now(timezone.utc).isoformat()})
        write_json(self.path, self.data)

    def finish(self, key: str, **values):
        next(r for r in self.data["requests"] if r["key"] == key).update(values)
        write_json(self.path, self.data)


class RecordingCompletions:
    def __init__(self, provider, ledger: Ledger):
        self.provider, self.ledger = provider, ledger
        self.key, self.last = None, None

    def create(self, **payload):
        self.ledger.reserve(self.key)
        start = time.perf_counter()
        self.last = None
        try:
            self.last = self.provider.chat.completions.create(**payload)
            usage = getattr(self.last, "usage", None)
            self.ledger.finish(self.key, status="succeeded", latency_ms=round((time.perf_counter() - start) * 1000),
                               prompt_tokens=getattr(usage, "prompt_tokens", None),
                               completion_tokens=getattr(usage, "completion_tokens", None))
            return self.last
        except Exception as error:
            self.ledger.finish(self.key, status="failed", error_type=type(error).__name__)
            raise


def raw_prediction(response, Filters) -> dict:
    if response is None or not response.choices:
        return {"action": "error", "code": "NO_RESPONSE"}
    choice = response.choices[0]
    if choice.finish_reason not in {"stop", "tool_calls"}:
        return {"action": "error", "code": f"FINISH_{choice.finish_reason}"}
    calls = choice.message.tool_calls or []
    if not calls:
        return {"action": "no_tool"}
    if len(calls) != 1:
        return {"action": "error", "code": "MULTIPLE_CALLS"}
    try:
        filters = Filters.model_validate_json(calls[0].function.arguments)
    except Exception:
        return {"action": "error", "code": "BAD_ARGUMENTS"}
    return {"action": "execute", "task_type": calls[0].function.name, **filters.model_dump(mode="json")}


def run_case(case, metadata, model, completions, audit_mod, schemas):
    row = {k: case[k] for k in ("id", "split", "question", "expected", "tags")}
    # 1) planner, through the production QwenModel (regex guard runs inside plan())
    completions.key = f"plan:{case['id']}"
    try:
        reply = model.plan(case["question"], metadata, timeout=20)
        row["B_regex"] = {"action": "execute", **reply.value.intent.model_dump(mode="json")}
    except schemas.ProjectError as error:
        code = error.info.code
        if code in {"MODEL_TIMEOUT", "MODEL_RATE_LIMIT", "MODEL_HTTP", "MODEL_FILTERED", "MODEL_TRUNCATED"}:
            row["B_regex"] = {"action": "error", "code": code}
        else:
            row["B_regex"] = {"action": "no_tool" if code == "UNSUPPORTED" else "blocked", "code": code}
    except RuntimeError as error:  # ledger / budget
        raise SystemExit(str(error))
    row["A_raw"] = raw_prediction(completions.last, schemas.Filters)
    # 2) condition audit
    completions.key = f"audit:{case['id']}"
    try:
        audit, extra = audit_mod.request_audit(SimpleNamespace(chat=SimpleNamespace(completions=completions)),
                                               MODEL, case["question"], metadata)
        row["audit"] = audit.model_dump(mode="json")
        row["audit_raw"] = extra["raw"]
    except RuntimeError as error:
        if "ledger" in str(error) or "budget" in str(error):
            raise SystemExit(str(error))
        row["audit_error"] = f"{type(error).__name__}: {error}"[:300]
    except Exception as error:
        row["audit_error"] = f"{type(error).__name__}: {error}"[:300]
        if getattr(error, "raw_content", None):
            row["audit_raw"] = error.raw_content
    return row


def decide(row, metadata, audit_mod):
    """Fill C/D/E from the saved A/B outputs and audit. Audit failure fails closed."""
    audit = audit_mod.Audit.model_validate(row["audit"]) if "audit" in row else None

    def with_audit(prediction):
        if prediction.get("action") != "execute":
            return prediction if prediction.get("action") == "error" else {"action": "no_tool"}
        if audit is None:
            return {"action": "blocked", "code": "AUDIT_ERROR"}
        planned = {k: prediction.get(k) for k in FIELDS}
        ok, reason = audit_mod.check_call(audit, metadata, planned)
        return prediction if ok else {"action": "blocked", "code": reason}

    row["C_audit"] = with_audit(row["A_raw"])
    row["D_both"] = with_audit(row["B_regex"])
    if audit is None:
        row["E_audit_only"] = {"action": "blocked", "code": "AUDIT_ERROR"}
    else:
        call, reason = audit_mod.expected_call(audit, metadata)
        row["E_audit_only"] = {"action": "execute", **call} if call else {"action": "no_tool", "code": reason}


def score(rows, key):
    sup = [r for r in rows if r["expected"]["action"] == "execute"]
    uns = [r for r in rows if r["expected"]["action"] == "no_tool"]

    def executed(r):
        return r[key].get("action") == "execute"

    def exact(r):
        return executed(r) and all(r[key].get(f) == r["expected"][f] for f in FIELDS)

    errors = sum(r[key].get("action") == "error" for r in rows)
    correct = sum(exact(r) for r in sup) + sum(not executed(r) for r in uns)
    return {
        "cases": len(rows), "decision_correct": correct,
        "supported": len(sup), "supported_exact": sum(exact(r) for r in sup),
        "false_refusal": sum(not executed(r) for r in sup),
        "wrong_parameters": sum(executed(r) and not exact(r) for r in sup),
        "unsupported": len(uns), "unsupported_executed": sum(executed(r) for r in uns),
        "provider_errors": errors,
    }


def build_report(state, audit_mod):
    rows = state["records"]
    meta = state["metadata"]
    for r in rows:
        decide(r, meta, audit_mod)
    report = {"model": MODEL, "benchmark_sha256": state["benchmark_sha256"], "source_sha256": state["source_sha256"],
              "scope": ("Single attempt per case, temperature=0, self-authored synthetic questions. "
                        "Decision-level scores for planning only; no statistics or reports were executed."),
              "metrics": {}, "misses": {}}
    splits = [s for s in ("development", "holdout", "holdout2") if any(r["split"] == s for r in rows)]
    for split in splits:
        part = [r for r in rows if r["split"] == split]
        report["metrics"][split] = {s: score(part, s) for s in STRATEGIES}
    report["audit_versions"] = {"prompt": audit_mod.AUDIT_PROMPT_VERSION,
                                "decision": getattr(audit_mod, "DECISION_VERSION", "decision-v1")}
    for s in STRATEGIES:
        by_tag = defaultdict(lambda: [0, 0])
        misses = []
        for r in rows:
            if r["split"] == "development":
                continue
            ok = score([r], s)["decision_correct"] == 1
            for t in r["tags"]:
                by_tag[t][0] += ok
                by_tag[t][1] += 1
            if not ok:
                misses.append({"id": r["id"], "q": r["question"], "expected": r["expected"]["action"], "got": r[s]})
        report["misses"][s] = {"by_tag": {t: f"{a}/{n}" for t, (a, n) in sorted(by_tag.items())}, "cases": misses}
    report["audit_errors"] = [r["id"] for r in rows if "audit_error" in r]
    return report


def markdown(report) -> str:
    lines = [f"# 意图规划 v2 联网对比（{report['model']}）", "", report["scope"], ""]
    names = {"A_raw": "A 模型原始输出", "B_regex": "B 规则校验（现行）", "C_audit": "C 条件审计",
             "D_both": "D 规则 + 审计", "E_audit_only": "E 仅审计决策（诊断）"}
    labels = {"development": "开发集（原 20 条）", "holdout": "留出集 1（100 条）", "holdout2": "盲测集 2（60 条）"}
    lines.insert(3, f"审计版本：{report.get('audit_versions')}")
    for split in report["metrics"]:
        label = labels.get(split, split)
        lines += [f"## {label}", "", "| 策略 | 决策正确 | 可执行题全对 | 误拒 | 参数错 | 不支持却执行 | 服务错误 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for s in STRATEGIES:
            m = report["metrics"][split][s]
            lines.append(f"| {names[s]} | {m['decision_correct']}/{m['cases']} | {m['supported_exact']}/{m['supported']} | "
                         f"{m['false_refusal']} | {m['wrong_parameters']} | {m['unsupported_executed']}/{m['unsupported']} | "
                         f"{m['provider_errors']} |")
        lines.append("")
    lines += ["## 非开发集按标签（决策正确/题数）", ""]
    tags = sorted(report["misses"]["A_raw"]["by_tag"])
    lines += ["| 标签 | " + " | ".join(STRATEGIES) + " |", "|---|" + "---:|" * len(STRATEGIES)]
    for t in tags:
        lines.append(f"| {t} | " + " | ".join(report["misses"][s]["by_tag"][t] for s in STRATEGIES) + " |")
    lines += ["", f"审计调用失败：{len(report['audit_errors'])} 条 {report['audit_errors']}", "",
              "失败样本明细见 report.json 的 misses 字段，归因请逐条人工阅读后记录。"]
    return "\n".join(lines) + "\n"


def main():
    global BENCH, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-live", action="store_true")
    parser.add_argument("--max-calls", type=int, default=260)
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    parser.add_argument("--report", action="store_true", help="rebuild report from saved records only")
    parser.add_argument("--bench", default=str(BENCH.relative_to(ROOT)), help="benchmark json, relative to the repository root")
    parser.add_argument("--out", default=str(OUT.relative_to(ROOT)), help="output folder, relative to the repository root")
    args = parser.parse_args()
    BENCH, OUT = ROOT / args.bench, ROOT / args.out
    if args.max_calls > 260:
        raise SystemExit("--max-calls above the approved ceiling of 260")

    sys.path.insert(0, str(PROJECT))
    from src import condition_audit as audit_mod
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "records.json"
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    sources = {p.name: sha(p) for p in sorted((PROJECT / "src").glob("*.py"))}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["benchmark_sha256"] != sha(BENCH) or state["source_sha256"] != sources:
            raise SystemExit("Benchmark or source changed since this run started; start a new output folder.")
    else:
        state = {"benchmark_sha256": sha(BENCH), "source_sha256": sources, "metadata": bench["metadata"],
                 "started_at": datetime.now(timezone.utc).isoformat(), "records": []}

    if not args.report:
        if not args.approved_live:
            raise SystemExit("Live run needs --approved-live (up to 2 qwen-plus calls per case).")
        from openai import OpenAI
        from src import schemas
        from src.config import Settings
        from src.models import QwenModel
        done = {r["id"] for r in state["records"]}
        todo = [c for c in bench["cases"] if c["id"] not in done][: args.limit]
        ledger = Ledger(OUT / "ledger.json", args.max_calls)
        provider = OpenAI(api_key=read_key(), base_url=Settings().base_url, max_retries=0, timeout=20)
        completions = RecordingCompletions(provider, ledger)
        model = QwenModel(Settings(model=MODEL), client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
        for i, case in enumerate(todo, 1):
            row = run_case(case, bench["metadata"], model, completions, audit_mod, schemas)
            state["records"].append(row)
            write_json(state_path, state)
            print(f"[{i}/{len(todo)}] {case['id']} A={row['A_raw']['action']} B={row['B_regex']['action']}"
                  f" audit={'ok' if 'audit' in row else 'ERR'}", flush=True)

    report = build_report(state, audit_mod)
    write_json(OUT / "report.json", report)
    (OUT / "report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
