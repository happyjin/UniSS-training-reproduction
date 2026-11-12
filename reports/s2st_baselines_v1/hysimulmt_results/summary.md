# Hy-SimulMT-30B-A3B + 论文 talker，RealSI 句级（自动生成，`experiments/baselines_hysimulmt_v1/score_hy.py`）

登记：`paper/baselines_run_plan_hysimulmt_v1.md`（含修订 1：只跑两档原生设置）。协议与论文 QABC 点相同：streaming UniSS talker T1c@1500，三个 talker 种子取均值；speech-only ASR-BLEU；D1 = 段 LAAL（块开始播放时刻，含排队）；D0w = SimulEval 逐词 LAAL（块写出时刻）。BLEU 分母为全部句子（En→Zh 346，Zh→En 431）。

## 全部点

| 方向 | 点 | 原生模式 | ASR-BLEU（±sd） | 文本 BLEU | D1 (s) | D0w (s) | 无输出句 | 末次提交早于句尾 | 块/句 |
|---|---|---|---|---|---|---|---|---|---|
| En→Zh | `hy-sc-w4` | short_chunk | 25.67 (±0.11) | 26.60 | 1.72 | 1.46 | 0 | 25 | 4.41 |
| En→Zh | `hy-std-w4` | standard | 27.96 (±0.05) | 28.85 | 2.05 | 1.90 | 0 | 19 | 3.38 |
| Zh→En | `hy-sc-w4` | short_chunk | 20.04 (±0.32) | 19.82 | 1.57 | 1.31 | 0 | 32 | 4.82 |
| Zh→En | `hy-std-w4` | standard | 21.88 (±0.02) | 21.54 | 1.75 | 1.61 | 0 | 19 | 3.72 |

## 分档（按 D1，论文规则）

| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | Hy-SimulMT + talker |
|---|---|---|---|---|
| En→Zh | m1 | 1.18 | 19.80 | -- |
| En→Zh | m2 | 1.98 | 24.21 | 25.67 (1.72)，`hy-sc-w4` |
| En→Zh | m3 | 2.46 | 25.01 | 27.96 (2.05)，`hy-std-w4` |
| En→Zh | m4 | 2.98 | 25.54 | 27.96 (2.05)，`hy-std-w4` |
| Zh→En | m1 | 1.18 | 14.89 | -- |
| Zh→En | m2 | 1.92 | 19.55 | 21.88 (1.75)，`hy-std-w4` |
| Zh→En | m3 | 2.40 | 21.07 | 21.88 (1.75)，`hy-std-w4` |
| Zh→En | m4 | 2.98 | 22.47 | 21.88 (1.75)，`hy-std-w4` |

## 分档（按 D0w，仅作参考）

| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | Hy-SimulMT + talker |
|---|---|---|---|---|
| En→Zh | m1 | 1.18 | 19.80 | -- |
| En→Zh | m2 | 1.98 | 24.21 | 27.96 (1.90)，`hy-std-w4` |
| En→Zh | m3 | 2.46 | 25.01 | 27.96 (1.90)，`hy-std-w4` |
| En→Zh | m4 | 2.98 | 25.54 | 27.96 (1.90)，`hy-std-w4` |
| Zh→En | m1 | 1.18 | 14.89 | -- |
| Zh→En | m2 | 1.92 | 19.55 | 21.88 (1.61)，`hy-std-w4` |
| Zh→En | m3 | 2.40 | 21.07 | 21.88 (1.61)，`hy-std-w4` |
| Zh→En | m4 | 2.98 | 22.47 | 21.88 (1.61)，`hy-std-w4` |

## 说明（R2，2026-11-12 16:10 布拉格）

- **网格**：按登记修订 1，只跑官方两档原生设置（standard 1600 ms 块；short_chunk 首块 1600 ms、之后 800 ms），`st_wait_chunks` 等全部用官方默认值；只做推理。
  官方代码校验首块固定 1600 ms，所以第一次提交不早于 1.6 s，m1（1.18 s）两个方向都没有点能进（登记时已预期）。
- **与论文 QABC 同表对照**（`tab_realsi_main` 的 QABC 列，括号内为 D1）：
  En→Zh m2 28.26 (1.77) 对 25.67 (1.72)；m3 30.33 (2.25) 对 27.96 (2.05)；m4 30.51 (2.75) 对 27.96 (2.05)。
  Zh→En m2 21.42 (1.70) 对 **21.88 (1.75)**；m3 22.46 (2.18) 对 21.88 (1.75)；m4 22.65 (2.67) 对 21.88 (1.75)。
  Zh→En m2 上 Hy-SimulMT 高 0.46，但它的 D1 比 QABC 那个点高 0.05 s；其余各档 QABC 更高。Hy-SimulMT 只有两个点，曲线不能往更高延迟延伸。
- **打分代码复现检查**：同一套打分代码在论文 QABC 点（b0=1，Δ=1，三个 talker 种子）上得到 En→Zh 30.33 / 2.25 s、Zh→En 22.46 / 2.18 s，
  与 `paper/bayes_simul_s2st_v11/generated/results.json` 完全一致。
- **覆盖**：每个点每个方向都没有无输出的句子，talker 结果 346 / 431 行齐全，ASR 覆盖 100%，Thinker 无报错。
- **末次提交早于句尾**的句子（En→Zh 19 / 25，Zh→En 19 / 32）按登记保留真实提交时刻，没有挪到句尾。
- **英文词中断开**：只有 `hy-sc-w4` Zh→En 有 11 个块按登记规则判为“接着上一块的词”（不加前导空格），其中真正断在词中间的如 “com|pany”；
  另有几处是模型输出本身漏了空格（如 “ah|For”）。按登记照原样处理，文本 BLEU 用的也是模型原样拼接的 `full_text`。
- **Talker 重跑**：`hy-std-w4` Zh→En 种子 20261021 的第一次尝试在卡 7 上因显存不足没起来；`hy-sc-w4` En→Zh 种子 20261021 的第一次尝试在我停掉卡 7 的 worker 时被中断。
  两者都用相同参数重跑，结果在 `realsi_*-r1` 目录（talker_worker 的规则）；失败的目录没有输出、不参与打分。
- 延迟不计算耗时（与论文相同）。Hy-SimulMT 在本机的实际处理速度没有计入。
- 路径：Thinker 输出 `eval_outputs/baselines_hysimulmt_v1/thinker/<点>/{events,blocks,meta}_<方向>.*`；
  talker / ASR `eval_outputs/baselines_hysimulmt_v1/talker/`；代码 `experiments/baselines_hysimulmt_v1/`。
