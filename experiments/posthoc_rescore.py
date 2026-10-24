"""Post-hoc rescore of round 1 (holdout-1) with the current decision table (decision-v2). No requests.

Round 1 was run with audit-v1 / decision-v1; its original report stays in results/live/report.md.
decision-v2 was written after reading the round-1 failures, so this rescore is optimistic and NOT a
blind result; the blind check is results/live_holdout2. Audit calls that failed to parse in round 1
still count as failures here (the saved outputs are reused, nothing is re-sent).

Usage (from the repository root):  python experiments/posthoc_rescore.py
  -> results/report_posthoc_decision_v2.md / .json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from intent_benchmark_v2 import build_report, markdown, write_json  # noqa: E402
from src import condition_audit as audit_mod  # noqa: E402

NOTE = ("> **事后重算（post-hoc），不是盲测。** 用第一轮已保存的模型输出，按 decision-v2 判定表重新计算 C/D/E，"
        "没有新增请求。decision-v2 是看过留出集 1 的失败样本后修改的，所以这里的留出集 1 结果偏乐观；"
        "盲测结论以 live_holdout2 为准。审计调用曾解析失败的条目在这里仍按失败计。")


def main():
    state = json.loads((ROOT / "results/live/records.json").read_text(encoding="utf-8"))
    report = build_report(state, audit_mod)
    report["note"] = NOTE
    title, rest = markdown(report).split("\n", 1)
    out = ROOT / "results/report_posthoc_decision_v2"
    write_json(out.with_suffix(".json"), report)
    out.with_suffix(".md").write_text(title.replace("联网对比", "：判定表修正后事后重算") + "\n\n" + NOTE + "\n" + rest,
                                      encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
