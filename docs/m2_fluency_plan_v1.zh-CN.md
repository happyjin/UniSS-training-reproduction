# M2 流畅度优化方案 v1：提交卫生 + 口语清洗

依据：[m2_fluency_analysis_v1.zh-CN.md](m2_fluency_analysis_v1.zh-CN.md) 第 5 节的思路一（规则层）和思路二（推理层）。两者都只改推理，不训练，Thinker 和 Talker 的权重都不动。

## 1. 目标

在不降低原有指标（语音 ASR-BLEU、LAAL）的前提下，降低 M2 的句中卡顿和碎片化。

## 2. 改动

### 2.1 提交卫生（思路一，`--hygiene`）

作用在 Thinker 每次读取后准备提交的新内容上。只对非末块生效；源语结束那一次读取照旧全部提交。

1. **悬空结尾**：块末尾是下面这些词时，先不提交，留到下一次读取：
   - 英文：功能词（冠词、介词、连词、助动词、代词等，如 “Choose to”、“But”）；
   - 中文：jieba 切出的单字词（句末语气词 了 吗 呢 吧 啊 等除外，如 “画”、“叶”、“不”、“为”）和虚词、连词（但是、因为、这个、那些……）。
2. **最小块长**：Talker 空闲（队列为空）时，块长不足 `--hyg-min` 个单位就不提交，除非这一块以标点结束。默认中文 3 个字、英文 2 个词。
3. **纯标点块**：不提交。

被扣下的内容不会丢：下一次读取时 Thinker 接着较短的已提交前缀继续生成，会把它们再说出来。

### 2.2 口语清洗（思路二，`--clean`）

作用在交给 Talker 的文本上。Thinker 的上下文仍然是它自己原来的输出，保证翻译行为不变。

1. **删口头禅**：
   - 英文：uh / um / er / erm / ah / hmm / mm；
   - 中文：嗯 / 呃 / 额 / 唔，以及句首的“啊，”。
   - 顺带清理多余的逗号和大小写，例如 “very, uh, interpretive” 改成 “very interpretive”。
2. **紧邻重复只说一次**：
   - 英文：1–3 个词的短语紧挨着重复，例如 “Their, their, their stock price” 改成 “Their stock price”。跨句号的不算（“…copyright. Copyright is…”）。
   - 中文：只处理逗号隔开、并且后面紧跟标点或句末的重复，例如“是的,是的。”改成“是的。”。“电流,电流通过”和“谢谢”不动。
   - 跨块也生效：上一块已经说了“复制,”，这一块开头的“复制,”就不再说。
3. **清洗后没有内容的块不送给 Talker。** SDR 的队列估计按清洗后的字数计算。

规则代码：`experiments/uniss_qwen3_omni_v1/evaluation/si_fluency_rules.py`，带 30 个自测用例，全部来自分析报告里的真实例子。

## 3. 实验设计

用和 M2 完全相同的 Thinker、Talker、读取窗、偏置、SDR 速率和 Talker 种子：
- Thinker：`thinker_op_v3_20261023T213631Z/thinker_merged_i200`，selfwait。
  - 英→中：Δ = 0.5 s，b = 1，SDR 速率 0.191。
  - 中→英：Δ = 1 s，b = 0，SDR 速率 0.339。
- Talker：T1c@1500，`--speed-policy block --history-ban 8`。

| 系统 | 内容 | Thinker 是否重跑 |
|---|---|---|
| M2 | 原系统（已有结果） | — |
| C（只清洗） | M2 的块离线套用 `clean` | 否，直接用 M2 的块 |
| H（只卫生） | `--hygiene` | 是 |
| HC（卫生 + 清洗） | `--hygiene --clean` | 是 |

两个方向都跑 RealSI 全集：英→中 346 句，中→英 431 句。

## 4. 指标和通过判据

**原有指标**，和论文同一口径（`paper/bayes_simul_s2st_v2/scripts/collect_results.py`）：
- 语音 ASR-BLEU：纯语音 ASR 对参考译文。
- LAAL：每个单位取所在块的开始播放时刻。
- 另外报告首次发声和 EndOffset。
- BLEU 差异用配对 bootstrap 检验。

**流畅度指标**：`si_fluency_stats.py` 的全套指标、UTMOS（纯语音）、接缝韵律。

**通过判据**，同时满足：
1. 语音 ASR-BLEU 不比 M2 显著下降（配对 bootstrap，单侧 p > 0.05），或者有提升；
2. LAAL 增加 ≤ 0.3 s；
3. 异常停顿负担和“有 ≥1 s 停顿的句子”至少有一项下降 ≥ 20%，其余流畅度指标不明显变差。

## 5. 产出

- 对比表和结论写入 `docs/m2_fluency_results_v1.zh-CN.md`。
- 有效果的话，做一个 M2 和新系统同句对比的试听页面。
- 每一步代码检查无误后提交，最后推送到 `private` 的 main。
