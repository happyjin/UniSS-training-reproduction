# dev 交叉熵与声学质量反相关 —— 不能用它选检查点

**日期** 2026-10-10 · Stage 1 迁移初始化训练

## 事实

同一次训练的两个检查点,级联 S2ST 在 120 条 CVSS-T dev 上:

| | dev code_ce | zh→en 语音 ASR-BLEU | en→zh 语音 ASR-BLEU |
|---|---|---|---|
| iter 12,000 | **6.857** | 29.69 | 33.50 |
| iter 24,000 | **7.075**(更差 0.22) | **33.78**(+4.09) | **33.92**(+0.42) |

文本 BLEU 两次完全相同(36.36 / 38.16),因为 Thinker 全程冻结 ——
**这 +4.09 全部来自 Talker**,而 Talker 正是 dev CE 所衡量的那部分。

**CE 变差 0.22 的同时,声学质量提升了 4.09 ASR-BLEU。**

## 为什么

BiCodec 语义码**声学冗余**:很多不同的码序列渲染出几乎一样的声音。
交叉熵惩罚"选了另一个等价码",而声码器根本听不出区别。

## 这已经是第三次被同一件事误导

| 当时的判断 | 真相 |
|---|---|
| "top-1 只有 1–3.8%,预测大概率是噪声" | 教师强制音频是**可懂语音**,04 号与参考一字不差 |
| "dev CE 6.86 = 困惑度 950,离可懂两个数量级" | 同一个错误,数字更大 |
| "dev 从 6.855 升到 7.07,该停训或回退配置" | 同期声学指标**提升** 4.09 |

## 由此作废的两个决定

1. **"全局批分桶有害"** —— 唯一证据是 dev CE 上升。回退到 micro 分桶后
   dev 照样从 7.02 升到 7.07,本就证明分桶不是原因;现在更知道那个上升
   根本不代表质量下降。**已恢复全局批分桶**,速度从 484 ms/iter 回到 ~316。
2. **"该在 dev 最低点(iter 12,000)选检查点"** —— 按声学指标应取更晚的。

## 今后怎么做

- **dev CE 只作健康指示**(是否发散、是否 NaN),**不作选择判据**
- 选检查点和验收一律用**声学指标**:级联 ASR-BLEU、UTMOS、AutoPCP
- 级联评测 120 条约需 6 分钟(8 卡),可以每 4,000 步做一次而不是只在最后做

## 复现

```bash
# 导出(CPU 即可)
python -m experiments.uniss_omni_s2st_v1.training.export_talker_from_megatron \
  --checkpoint-dir checkpoints/uniss_omni_s2st_v1/stage1_transfer/iter_0024000 \
  --output .../talker_iter0024000.pt

# 级联 + ASR-BLEU(8 卡,120 条)
#   见 scratchpad/cascade.sh,核心是
#   experiments/uniss_omni_s2st_v1/evaluation/cascade_s2st.py
```
