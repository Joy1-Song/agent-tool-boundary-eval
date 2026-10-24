"""Draw the README charts from saved results (no model requests).

python scripts/make_charts.py  ->  docs/images/*.png
Reads results/guard_offline.json, results/live/report.json and results/live_holdout2/report.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/images"
IMG.mkdir(parents=True, exist_ok=True)

for path in ("C:/Windows/Fonts/msyh.ttc", "/System/Library/Fonts/PingFang.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, MUTED = "#2a78d6", "#b9b8b1"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10})


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def tidy(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# 1. Regex guard: development set vs holdout set (offline, simulated planner outputs)
guard = load("results/guard_offline.json")["summary"]
metrics = [("不支持的请求被拦下", "unsupported_blocked", "unsupported_cases"),
           ("被丢掉的条件被发现", "dropped_condition_caught", "dropped_condition_variants")]
splits = [("development", "开发集（规则据此编写）", MUTED), ("holdout", "留出集 1（新写 100 条）", ACCENT)]
fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
h = 0.34
for j, (split, label, color) in enumerate(splits):
    for i, (_, num, den) in enumerate(metrics):
        a, b = guard[split][num], guard[split][den]
        y = i + (j - 0.5) * (h + 0.04)
        ax.barh(y, 100 * a / b, height=h, color=color, label=label if i == 0 else None)
        ax.text(100 * a / b + 1.5, y, f"{a}/{b}", va="center", color=INK, fontsize=9.5)
ax.set_yticks(range(len(metrics)), [m[0] for m in metrics], color=INK)
ax.set_xlim(0, 112)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.invert_yaxis()
tidy(ax)
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncol=2)
fig.suptitle("正则规则只认得写规则时见过的说法", x=0.01, ha="left", color=INK, fontsize=11.5)
fig.text(0.01, 0.01, "离线实验：不调模型，用模拟的规划输出测规则校验本身。数据：results/guard_offline.json", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(IMG / "guard_generalization.png")
plt.close(fig)

# 2. Two kinds of error per strategy: round 1 (audit-v1) and the blind set (audit-v2)
rounds = [("results/live/report.json", "holdout", "留出集 1 · 第一轮（审计 v1）"),
          ("results/live_holdout2/report.json", "holdout2", "盲测集 2 · 第二轮（审计 v2）")]
strategies = [("A_raw", "A 模型原始输出"), ("B_regex", "B 正则校验"), ("C_audit", "C 条件审计"), ("D_both", "D 正则 + 审计")]
errors = [("不支持却执行（放过）", "unsupported_executed", "unsupported"), ("误拒正常请求", "false_refusal", "supported")]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), dpi=200, sharey=True)
for r, (rel, split, row_label) in enumerate(rounds):
    m = load(rel)["metrics"][split]
    for c, (col_label, num, den) in enumerate(errors):
        ax = axes[r][c]
        for i, (key, _) in enumerate(strategies):
            a, b = m[key][num], m[key][den]
            ax.barh(i, 100 * a / b, height=0.55, color=ACCENT)
            ax.text(100 * a / b + 1.5, i, f"{a}/{b}", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, 45)
        ax.set_xticks([0, 20, 40], ["0%", "20%", "40%"])
        tidy(ax)
        ax.set_title(f"{row_label}\n{col_label}", loc="left", color=INK, fontsize=9.5)
axes[0][0].set_yticks(range(len(strategies)), [s[1] for s in strategies])
axes[0][0].invert_yaxis()
for row in axes:
    row[0].tick_params(axis="y", labelcolor=INK)
fig.suptitle("两种错误：放过做不到的请求，或拒绝能做的请求", x=0.01, ha="left", color=INK, fontsize=11.5)
fig.text(0.01, 0.01, "qwen-plus，temperature 0，每题一次。数据：results/live/report.json、results/live_holdout2/report.json",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(IMG / "strategy_errors.png")
plt.close(fig)
print("charts ->", IMG)
