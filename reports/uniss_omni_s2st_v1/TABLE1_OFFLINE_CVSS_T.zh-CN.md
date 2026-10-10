# 完整离线 CVSS-T 评测 — 按 arXiv 2607.19810 Table 1 的指标

**日期** 2026-10-10 · CVSS-T **test** 全量 **4,897 条 × 2 方向 = 9,794 条**
**被测系统**:Omni Thinker(冻结,零样本)+ Stage 1 Talker(iter 40,000)**级联**
⚠ **不是端到端联合模型** —— Stage 2 尚未开始

## 结果

| | n | ASR-BLEU ↑ | Text-BLEU ↑ | A.PCP ↑ | SIM-O ↑ | UTMOS ↑ |
|---|---|---|---|---|---|---|
| **En→Zh** | 4,896 | **28.70** | 30.45 | 2.774 | 0.880 | 3.862 |
| **Zh→En** | 4,897 | **24.82** | 26.82 | **2.925** | 0.825 | 3.454 |

## 与论文 Table 1 对照

| 类别 | 模型 | 规模 | ASR-BLEU<br>En→Zh / Zh→En | Text-BLEU<br>En→Zh / Zh→En | A.PCP<br>En→Zh / Zh→En |
|---|---|---|---|---|---|
| MLLM | GPT-4o † | — | 25.42 / **31.64** | — | 2.66 / 2.58 |
| | Qwen-Omni2.5 † | 7B | 8.04 / 22.66 | **34.85** / 24.39 | 1.90 / 1.92 |
| | Step-Audio 2 mini | 7B | **32.81** / 25.35 | — | — |
| S2ST | Seamless-m4t-v2-large ‡ | 2.3B | 20.86 / 22.25 | 20.75 / 22.60 | 2.43 / 2.67 |
| | Seamless-expressive ‡ | 1.7B | 23.70 / 15.89 | 25.45 / 16.67 | 2.73 / 2.76 |
| | UniSS (P) ‡ | 1.5B | 30.09 / 23.77 | 30.77 / 24.63 | 2.73 / 2.75 |
| | UniSS (Q) ‡ | 1.5B | 32.04 / 24.72 | 32.95 / 25.51 | 2.71 / 2.75 |
| 论文 | Dec-only | 3B | 27.12 / 23.41 | 31.59 / 24.90 | **2.96** / 2.75 |
| | Thinker–Talker | 3B | 31.12 / — | — | — / 2.64 |
| **本工作** | **Omni + Stage1 级联** | **3B** | **28.70 / 24.82** | **30.45 / 26.82** | **2.774 / 2.925** |

† 引自原论文 · ‡ 论文在其统一协议下重测 · 本行用本项目自己的协议

## 分析

### 1. Zh→En 上我们是 S2ST 类里最好的

ASR-BLEU **24.82** 高于 UniSS (Q) 的 24.72、UniSS (P) 23.77、论文 Dec-only 23.41、
Seamless-m4t 22.25。只低于 GPT-4o(31.64)和 Step-Audio 2 mini(25.35)。

Text-BLEU **26.82** 是该列**全表最高** —— 高于 UniSS (Q) 25.51、Dec-only 24.90、
Qwen-Omni2.5 7B 24.39。这正是换基座的收益:我们原来的 0.5B 是 15.33。

### 2. A.PCP:Zh→En 全表最高

**2.925**,高于 Seamless-expressive 2.76、UniSS 2.75、Dec-only 2.75、论文自己的
Thinker–Talker 2.64、GPT-4o 2.58。

这正是**保留 UniSS 设计的直接结果**:32 个 `bicodec_global` token 绕过语言模型
直通声码器,韵律与音色沿这条路保留。论文把 codec 换成 DualCodec 之后,
它的 Thinker–Talker 在这一列反而只有 2.64。

En→Zh 的 2.774 也高于 UniSS(2.71/2.73)与 GPT-4o(2.66),低于论文 Dec-only 的 2.96。

### 3. Talker 几乎不损失质量

| 方向 | Text-BLEU(上限) | ASR-BLEU | Talker 的代价 |
|---|---|---|---|
| En→Zh | 30.45 | 28.70 | **−1.75** |
| Zh→En | 26.82 | 24.82 | **−2.00** |

Thinker 全程冻结,所以 Text-BLEU 就是上限,差值全部来自 Talker。
不到 2 个 BLEU —— Stage 1 的目标(让全新的 8,192 路码头在我们的码空间收敛)达成。

一致性验证:Text-BLEU 30.45 / 26.82 与此前实测的 Omni 3B **零样本** 30.56 / 26.65
几乎相同,符合"Thinker 未被改动"的预期。

### 4. En→Zh 仍落后 UniSS (Q)

28.70 vs 32.04,差 3.34。拆开看,差距**不在 Talker 而在 Thinker**:
Text-BLEU 30.45 vs 32.95,已经差了 2.5。
Thinker 是零样本的、没做任何适配 —— 这正是 Stage 2 要解决的。

## 必须声明的四点

1. **这是级联,不是端到端。** Omni Thinker 先出完整译文,Talker 再发声。
   论文的 Thinker–Talker 是联合训练的。Stage 2 之后才可比。
2. **协议不同。** 论文带 ‡ 的行是它在自己的统一协议下重测的;本行用的是
   本项目的协议(`evaluation.text_metrics`,去标点 + OpenCC 简体化 + sacrebleu;
   ASR 用 whisper-large-v3)。数字不应逐位对比。
3. **SIM-O 不可比。** 论文用 **WavLM-Large**,本机只有 `wavlm-base-plus-sv` 且无外网。
   我们的 0.880 / 0.825 与论文的 0.40–0.59 是**不同编码器的不同量纲**,
   不代表我们更好。它只能用于本项目检查点之间的排序。
4. **此前的小样本数字偏乐观。** 我在 120 条 dev 子集上报过 34.43 / 34.30,
   全量 test 是 24.82 / 28.70。**那个数字不应被引用** —— 60 条/方向的样本量
   不足以代表 4,897 条。

## 复现

```bash
# 生成（8 卡，两个方向各 4,897 条）
LIMIT=0 OUT=.../table1_full bash scratchpad/table1.sh

# 四项指标
ROOT=.../table1_full bash scratchpad/table1_metrics.sh
#   内部调用 experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh
#   （ASR-BLEU / UTMOS / A.PCP / SLC）与
#   experiments/uniss_omni_s2st_v1/evaluation/sim_o.py
```

逐条结果:`eval_outputs/uniss_omni_s2st_v1/table1_full/{cmn_eng,eng_cmn}/`
汇总:`table1_full/table1_summary.json`
