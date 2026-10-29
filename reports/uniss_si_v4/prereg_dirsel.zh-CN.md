# 预注册：按方向选后验和先验偏置（D1，2026-10-29，在任何 RealSI 新结果之前写定）

起因：`diag/zh2en_diagnosis.zh-CN.md`。中译英的后验提交过早；BMA 集成在中译英更好。RealSI 只用于最终测试，下面的选择只看 dev。

## 候选

每个方向独立选，全部带 SDR（与 A0 相同的队列速率）：
- 后验：s0（A0）、bma3（A1）；A2 训练完成后加入同样的网格。
- 先验偏置：b0 ∈ {0, 1, 2, 3}。b0 越大越保守。

## dev 上的选择规则

- 两套 dev：
  - UniST dev：`data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl`
  - 口语 dev：`data/si_spont_v1/dev_{d}_ref.jsonl`
- 用 LA2 在同一 dev 上的 8 个窗口（0.5–3 s）做对照，计算每个候选的 E1（`paper/bayes_simul_s2st_v4/scripts/endpoints.py`，B=200）。
- 分数 = 两套 dev 的 E1 平均。
- 取分数最高的候选。只有比默认（s0, b0=1）高至少 0.3 BLEU 时才替换默认，与 prereg_v4 的规则一致。

## RealSI 测试（只做一次）

- 选中的配置在 LA2 的 15 个窗口上跑文本端，计算对 LA2 的 E1、E2（B=1000，单侧）。
- 两个方向一起做 Holm 校正。
- 选中的配置若与默认相同，就不再跑 RealSI，直接沿用 v6 的结果。
- 语音端只在文本端通过后再跑，届时另行预注册。

## 不做的事

- 不看 RealSI 的结果来调 b0 或换后验。
- 不在 RealSI 上训练或重新校准。
