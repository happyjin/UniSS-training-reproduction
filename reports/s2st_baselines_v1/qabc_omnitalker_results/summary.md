# QABC 提交块 + Qwen3-Omni 原生 Talker（RealSI 流式）结果小结

由 R5 执行者完成（2026-11-12 22:45 布拉格）。此前子代理环境不允许写这份 .md 文件，内容由协调者照 R5 的最终报告整理。数字以 `summary.json` 和 `summary_tables.md` 为准。

**结论**：同样的 QABC 提交块，论文 talker 在 8 个点上的 ASR-BLEU 都高于 Qwen3-Omni 原生 Talker，延迟预算越紧，差距越大。

## 设置
- 登记先于任何合成：`paper/baselines_run_plan_qabc_omnitalker_v1.md`，提交 5eb0f057e。
- 点：`tab_realsi_main` 中 QABC 的 m1–m4，两个方向共 8 个点。用论文 run 存下的提交块文本和 write_ms；三个论文 talker 种子的块逐块相同。
- 原生系统：Qwen3-Omni-30B-A3B-Instruct 自己的 Thinker → Talker → code2wav。vLLM-Omni 0.30.0，配置和解码与 CVSS-T 原生 run 相同，预置音色 Ethan。
- 每块单独一次请求，不带上下文：system 轮放"逐字朗读"指令，user 轮只放块文本。纯标点的块不合成。
- 3 个 talker 种子，共 38,097 个请求，全部有音频。
- 打分与论文相同：排队模型、只对语音部分做 ASR（batch 4）、corpus BLEU、D1、D0w。

## 结果
3 个种子的均值 ± 种子标准差。Δ = Omni − 论文。talker WER 为句级，En→Zh 一栏是 CER。

| 点 | ASR-BLEU 论文 | ASR-BLEU Omni | Δ | D1 论文 / Omni (s) | D0w (s，两边相同) | talker WER 论文 / Omni |
|---|---|---|---|---|---|---|
| En→Zh m1 | 25.33±0.04 | 24.54±0.08 | −0.79 | 1.13 / 1.28 | 1.00 | 5.40 / 5.72 |
| En→Zh m2 | 28.26±0.07 | 27.01±0.05 | −1.26 | 1.77 / 1.85 | 1.67 | 4.73 / 7.40 |
| En→Zh m3 | 30.33±0.07 | 29.57±0.26 | −0.76 | 2.25 / 2.28 | 2.17 | 4.18 / 4.88 |
| En→Zh m4 | 30.51±0.04 | 29.30±0.08 | −1.20 | 2.75 / 2.75 | 2.67 | 3.67 / 6.24 |
| Zh→En m1 | 17.83±0.29 | 14.24±1.02 | −3.59 | 1.14 / 1.41 | 1.04 | 11.93 / 23.68 |
| Zh→En m2 | 21.42±0.26 | 18.36±1.21 | −3.06 | 1.70 / 1.88 | 1.61 | 8.67 / 16.28 |
| Zh→En m3 | 22.46±0.32 | 20.39±1.18 | −2.07 | 2.18 / 2.31 | 2.08 | 9.11 / 13.43 |
| Zh→En m4 | 22.65±0.76 | 21.66±0.23 | −0.99 | 2.67 / 2.72 | 2.57 | 10.14 / 9.57 |

## 分析
- **延迟**：原生 talker 每句语音更长，En→Zh 多 0.05–0.87 s，Zh→En 多 0.68–1.96 s。排队因此使 D1 上升，两个方向的 m1 都超出了 1.18 s 的预算。
- **按 D1 分档**：
  - Omni 在 m1 档没有点。
  - En→Zh m4 档里，Omni 最好的是它的 m3 点，29.57。
  - Zh→En m2、m3、m4 档分别是 18.36、20.39、21.66。
- **错误类型**：Omni 的 talker WER 以插入为主（Zh→En m1 插入 13.1%，论文 talker 0.7%），论文 talker 以删除为主。Zh→En m4 是唯一 Omni WER 略低的点，但它的 ASR-BLEU 仍低 0.99。
- **文本忠实度**：Omni Thinker 与送入文本完全一致的块占 99.1–99.9%，没有低于登记的"系统性偏离"线（95%），三个种子的 Thinker 文本相同。但每个点仍有 2–11 块严重偏离，单块语音长达 40–80 s，例如：
  - 回答问题；
  - 续写（"Sure, here's an article on nature…"）；
  - 自我介绍；
  - 把中文翻成英文。
- **talker 跑偏**：Zh→En 有 2–4% 的块拖长或跑偏，占音频时长 1–9%。这是 Zh→En 种子标准差达到约 1.0–1.2 的主要原因。
- **块级 WER**（只算 Omni，种子 0，仅作诊断）：En→Zh CER 6.3–9.0，Zh→En WER 18–41。孤立的短片段上 ASR 本身也更不准，而论文 talker 没有对应的数可比。
- **检查**：
  - 用同一份代码重算论文 talker，8 个点的 ASR-BLEU 和 D1 都与 tier 表一致；
  - D0w 和提交文本的 BLEU 两边逐位相同；
  - 排队违规 0 次，ASR 覆盖率 100%。
- **局限**：
  - 没有做显著性检验（未登记）。
  - 原生 talker 每块独立调用：没有上下文，不能克隆源说话人音色，也不能控制语速。
  - 两边都不计合成耗时。

## 文件
- 结果：本目录（tracked 副本），以及 `paper/baselines_results/qabc_omnitalker/`。
- 代码：`experiments/s2st_baselines_v1/qabc_omnitalker/`
- 运行输出：`eval_outputs/s2st_baselines_v1/qabc_omnitalker_realsi/`
- 日志：`logs/s2st_baselines_v1/qabc_omnitalker/`
- 提交：5eb0f057e（登记）、f0c6366c5、02146b92a、c8f1f7f02、f89a43a8f
