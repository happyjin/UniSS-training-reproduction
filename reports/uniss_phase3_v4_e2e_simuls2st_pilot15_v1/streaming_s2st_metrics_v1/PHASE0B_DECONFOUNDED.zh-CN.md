# Phase 0.2 结论:去污染后 loss 排序被颠倒

- 时间:2026-08-31
- 方法:四个 100-update canary checkpoint,**同一冻结 fixed-16 selection、同一评测器**,唯一变量是提交策略(`append_only_commit` → local agreement holdback 2)
- 原始产物:`free_running_gates/phase0b_la_*_20260831T193735Z/`

## 1. 污染假说确认

基线 canary(`semend_w0p5`,仅 100 步)换提交策略后:

| 指标 | append_only | local agreement | 变化 |
|---|---:|---:|---:|
| gold 覆盖 | 0.1446 | **0.3569** | **+147%** |
| gold 覆盖 min | 0.000 | 0.0357 | 不再为 0 |
| 提交冲突 | 264 | 168 | −36% |
| cmn→eng chrF | 20.48 | 29.80 | +46% |
| **eng→cmn chrF** | 2.96 | **19.50** | **+559%** |
| free 覆盖 | 0.1194 | 0.2168 | +82% |

**那 11 个 canary 当年全部被封在 0.145–0.159,不是 loss 无效,是提交层压住了上限。** 零训练、只改提交策略,一个 100 步的 checkpoint 就从 0.145 到 0.357。

## 2. 去污染后的四路对照

| canary | 旧 cov | **新 cov** | Δ% | 冲突 | **cmn chrF** | Δ% | eng chrF | free cov | malformed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_semend | 0.1446 | 0.3569 | — | 168 | 29.80 | — | 19.50 | 0.2168 | 34 |
| **decisionrow** | 0.1508 | **0.3841** | **+7.6** | 156 | **35.38** | **+18.7** | **21.65** | **0.2365** | **33** |
| sembinary | 0.1530 | 0.3673 | +2.9 | 160 | 32.81 | +10.1 | 19.50 | **0.1808** | **64** |
| fragend | **0.1593** | 0.3781 | +6.0 | **148** | 29.55 | **−0.9** | 18.50 | 0.2057 | 45 |

极差对比:

| 口径 | 极差 |
|---|---|
| 旧(append_only)gold 覆盖 | 0.1446–0.1593(+10.2%) |
| 新(local agreement)gold 覆盖 | 0.3569–0.3841(+7.6%) |
| **新 cmn→eng chrF** | **29.55–35.38(+19.7%)** |

## 3. 排序被颠倒 —— 这是本轮最重要的结论

**被污染的口径下 `fragend` 最好(0.1593)、`decisionrow` 倒数第二(0.1508)。正确口径下 `decisionrow` 在每一个内容维度上都最好,而 `fragend` 的 chrF 反而比基线低 0.9%。**

- **`decisionrow` 是唯一真正有效的一项**:覆盖 +7.6%、cmn chrF **+18.7%**、eng chrF +11.0%、free 覆盖 +9.1%,且 malformed 最低(33)。它的配置是 `semantic_rollin_end_ce 0.25` + **`semantic_rollin_continue_decision_margin 0.25`** + `semantic_rollin_continue_margin 0.025`,roll-in rate 0.5,tail 12。
- **`sembinary` 结构代价最大**:cmn chrF +10.1% 但 free 覆盖**掉 17%**、malformed **翻倍到 64**。与它"替代整个 margin 家族、更激进"的定位一致。
- **`fragend` 的旧排名是假象**:它的高覆盖来自提交层的偏差,chrF 实际不如基线。

**前一个会话据此选择了错误的 loss 方向。**

## 4. 对我自己上一条判断的修正

我此前说被停掉的 roll-in 续训"靶子选错了"。更准确的说法是:

- 它用的正是 `decisionrow` 的配置(仅少了 0.025 的 `continue_margin`),而这一项现在被证明是**唯一真正改善内容的 loss**;
- 但它确实**不触及开口频率**(掩码选在 `END_SEMANTIC` 行和 gold 为语音 token 的行上),而 Phase 0.1 已证明开口频率无法在推理侧修复。

**两件事同时成立。** 所以 Phase 2 必须同时包含三项,缺一不可:

1. **`semantic_rollin_continue_decision_margin` 保留并可放大** —— 已实测有效(+18.7% chrF);
2. **新写的开口决策 loss**(`WRITE_GENERATE` vs `WAIT`,softplus 校准 + 类别平衡 + roll-in)—— 代码里不存在,Phase 0.1 证明推理侧补不了;
3. **重复惩罚** —— Phase 0.1 证明任何提高开口率的干预都触发重复循环(`emilia_zh_0004122419` 文本长度比 1.70 → 15.40),而现有 loss 一项都不惩罚重复。

## 5. 效应量的边界

四者都是 **100-update** canary,效应量不能外推到 epoch 尺度。本轮成立的是**排序**,而且这个排序第一次是在未被污染的指标上量出来的。参考尺度:同一提交策略下,100 步的基线是 0.357,而 3 个 coverage epoch 的 `iter_0002264` 是 0.495 —— **规模仍然值 +39%,但修提交策略比 22 个 epoch 的训练更值。**
