# X-Translator（Qwen3-ASR 1.7B + LMT-60-8B + X-Voice 0.4B），RealSI 句级（自动生成，`experiments/baselines_xtranslator_v1/summarize_xt.py`）

生成于 2026-11-12 22:55（布拉格时间）。登记：`paper/baselines_run_plan_xtranslator_v1.md`（含修订）。
官方权重、官方代码，只做推理。X-Translator 用自己的 TTS（X-Voice）出语音；源时钟是模拟的，不计算耗时。
speech-only ASR-BLEU 用 Whisper-large-v3（英）/ Paraformer（中）。D1 是段 LAAL，按块开始播放时刻计，含排队；D0w 是 SimulEval 逐词 LAAL，按块写出时刻计。
打分代码就是 R1 的 `score_seamless.score()`。BLEU 分母为全部句子（En→Zh 346，Zh→En 431）。每个点每个方向只跑一次。

## 全部点

| 方向 | 点 | 设置 | ASR-BLEU | 文本 BLEU | D1 (s) | D0w (s) | StartOffset p50 (s) | 块/句 | 句末才写出的单位占比 | 无输出句 | TTS 失败段 | ASR 覆盖 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| En→Zh | `xt-default` | release default (main) | 25.19 | 26.25 | 3.05 | 2.69 | 2.4 | 3.66 | 39.8% | 0 | 2 | 1.0 |
| En→Zh | `xt-c1280` | paraformer_chunk_size 1280 (0.08 s requests, 0.64 s endpoint) | 23.76 | 24.96 | 2.53 | 2.21 | 2.18 | 3.86 | 35.3% | 0 | 2 | 1.0 |
| En→Zh | `xt-c5120` | paraformer_chunk_size 5120 (0.32 s requests, 2.56 s endpoint) | 23.14 | 24.26 | 3.39 | 2.89 | 2.56 | 4.27 | 40.2% | 1 | 1 | 1.0 |
| En→Zh | `xt-a1.6` | ASR server chunk 1.6 s (paper App. C; secondary) | 25.62 | 26.6 | 3.37 | 3.06 | 3.2 | 3.46 | 44.9% | 0 | 1 | 1.0 |
| Zh→En | `xt-default` | release default (main) | 16.23 | 16.37 | 2.27 | 1.91 | 1.76 | 4.38 | 32.3% | 0 | 0 | 1.0 |
| Zh→En | `xt-c1280` | paraformer_chunk_size 1280 (0.08 s requests, 0.64 s endpoint) | 15.14 | 15.2 | 2.09 | 1.78 | 1.44 | 4.39 | 30.8% | 0 | 0 | 1.0 |
| Zh→En | `xt-c5120` | paraformer_chunk_size 5120 (0.32 s requests, 2.56 s endpoint) | 15.89 | 16.21 | 2.44 | 2.00 | 1.92 | 4.71 | 32.3% | 0 | 0 | 1.0 |
| Zh→En | `xt-a1.6` | ASR server chunk 1.6 s (paper App. C; secondary) | 16.36 | 16.49 | 2.67 | 2.32 | 1.76 | 4.08 | 36.6% | 0 | 0 | 1.0 |

## 分档（按 D1，论文规则）

| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | X-Translator |
|---|---|---|---|---|
| En→Zh | m1 | 1.18 | 19.80 | -- |
| En→Zh | m2 | 1.98 | 24.21 | -- |
| En→Zh | m3 | 2.46 | 25.01 | -- |
| En→Zh | m4 | 2.98 | 25.54 | 23.76 (2.53)，`xt-c1280` |
| Zh→En | m1 | 1.18 | 14.89 | -- |
| Zh→En | m2 | 1.92 | 19.55 | -- |
| Zh→En | m3 | 2.40 | 21.07 | 16.23 (2.27)，`xt-default` |
| Zh→En | m4 | 2.98 | 22.47 | 16.36 (2.67)，`xt-a1.6` |

## 分档（按 D0w，仅作参考）

| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | X-Translator |
|---|---|---|---|---|
| En→Zh | m1 | 1.18 | 19.80 | -- |
| En→Zh | m2 | 1.98 | 24.21 | -- |
| En→Zh | m3 | 2.46 | 25.01 | 23.76 (2.21)，`xt-c1280` |
| En→Zh | m4 | 2.98 | 25.54 | 25.19 (2.69)，`xt-default` |
| Zh→En | m1 | 1.18 | 14.89 | -- |
| Zh→En | m2 | 1.92 | 19.55 | 16.23 (1.91)，`xt-default` |
| Zh→En | m3 | 2.40 | 21.07 | 16.36 (2.32)，`xt-a1.6` |
| Zh→En | m4 | 2.98 | 22.47 | 16.36 (2.32)，`xt-a1.6` |

## 说明（R4，2026-11-12 23:58 布拉格时间）

- **网格**：按登记跑了 4 个点（主点 `xt-default`；config 暴露的旋钮 `paraformer_chunk_size` 取 1280 和 5120；次要点 `xt-a1.6` 用论文附录 C 写的 1.6 s ASR 块）。只做推理，结果出来后没有加点、改点。
- **分档结论**：
  - En→Zh：主点 D1 是 3.05 s，超过 m4 的 2.98 s，所以 m1–m3 都没有点；m4 只有 `xt-c1280` 进档（23.76，D1 2.53 s）。
  - Zh→En：m1 和 m2 没有点；m3 是主点 16.23（2.27 s），m4 是 `xt-a1.6` 的 16.36（2.67 s）。
- **与论文 QABC 同表对照**（`tab_realsi_main` 的 QABC 列，括号内为 D1，数字取自 R2 的 summary）：
  - En→Zh：m4 是 30.51 (2.75) 对 23.76 (2.53)。
  - Zh→En：m3 是 22.46 (2.18) 对 16.23 (2.27)；m4 是 22.65 (2.67) 对 16.36 (2.67)。
  - 两个方向都是 QABC 更高，差 6 BLEU 以上。
- **延迟的来源**：
  - 承诺层要识别文本里有两个标点才提交，而且只提交到倒数第二个标点。所以 31–45% 的输出单位要等句末 finalize 才写出。
  - 每个标点段单独翻译，不带上下文。块/句是 3.5–4.7。
  - 说话人提示器在单说话人音频里也会误报说话人切换，强制提前提交：En→Zh 共 50 次，Zh→En 共 95 次。这是发布默认行为，原样保留。
  - X-Voice 的语速使播放队列越排越长，EndOffset 中位数 3.2–6.1 s。
- **覆盖与失败**：
  - 每个点 ASR 覆盖都是 100%，没有报错行。
  - TTS 失败的段每点 0–2 个，都是“啊”这类译文，X-Voice 的 IPA 序列为空、合成不出声音；原系统里也是没声音，按登记计为无声。
  - `xt-c5120` En→Zh 有 1 句（0.8 s 的 en2zh-09-sport_001）没有任何提交，按空串计入 BLEU。
- **文本 BLEU** 用的是 LMT 对各段输出的拼接。各点的 ASR-BLEU 比文本 BLEU 低 0.06–1.2。
- **不计算耗时**，与论文相同。本机实测每句处理约 7–13 s（与别的作业共卡），没有计入。
- **路径**：
  - 运行输出：`eval_outputs/baselines_xtranslator_v1/<点>/realsi_<方向>/`；
  - 代码：`experiments/baselines_xtranslator_v1/`；
  - 服务日志：`eval_outputs/baselines_xtranslator_v1/servers/`。
