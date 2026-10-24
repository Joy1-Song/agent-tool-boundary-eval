# 意图规划 v2 联网对比（qwen-plus）

Single attempt per case, temperature=0, self-authored synthetic questions. Decision-level scores for planning only; no statistics or reports were executed.
审计版本：{'prompt': 'audit-v2', 'decision': 'decision-v2'}

## 盲测集 2（60 条）

| 策略 | 决策正确 | 可执行题全对 | 误拒 | 参数错 | 不支持却执行 | 服务错误 |
|---|---:|---:|---:|---:|---:|---:|
| A 模型原始输出 | 58/60 | 27/27 | 0 | 0 | 2/33 | 0 |
| B 规则校验（现行） | 59/60 | 27/27 | 0 | 0 | 1/33 | 0 |
| C 条件审计 | 59/60 | 26/27 | 1 | 0 | 0/33 | 0 |
| D 规则 + 审计 | 59/60 | 26/27 | 1 | 0 | 0/33 | 0 |
| E 仅审计决策（诊断） | 58/60 | 26/27 | 1 | 0 | 1/33 | 0 |

## 非开发集按标签（决策正确/题数）

| 标签 | A_raw | B_regex | C_audit | D_both | E_audit_only |
|---|---:|---:|---:|---:|---:|
| channel | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| cn_date | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| colloquial | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| compound | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| dot_date | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| english | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| exclusion | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| export | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| external_send | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| full_month | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| injection | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| month_no_year | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 |
| multi_category | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| paraphrase | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| plain | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| rating_threshold | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| relative_date | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| resolved_only | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| response_threshold | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| single_day | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| synonym | 4/4 | 4/4 | 3/4 | 3/4 | 3/4 |
| terse | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| topk | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| typo | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| unknown_category | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| unsupported_analysis | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| unsupported_filter | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| unsupported_groupby | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| vague | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| write_op | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |

审计调用失败：0 条 []

失败样本明细见 report.json 的 misses 字段，归因请逐条人工阅读后记录。
