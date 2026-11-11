# S1 预登记：talker 输出里的静音占比，以及压缩静音后的延迟上界（2026-11-11，只用 CPU）

**登记时间**：2026-11-11，在任何 S1 数字算出之前提交并推送（`git push private HEAD:main`）。
**用户批准**：约 07:30 UTC（与 E2′ 同一条指令）。
**数据**：只用 A′ 的 dev talker 输出，两个 TTS 种子：
- `eval_outputs/mist_w3po_v1/speech/dev_crf_v2p_s3000_finalpost_s8_c2g06/talker`
- `eval_outputs/mist_w3po_v1/speech/dev_crf_v2p_s3000_finalpost_s8_c2g06_tts2/talker`

测试集不碰。

## 1. 测量（每个块内部）

- **块的音频**：`results.jsonl` 每条流的 `speech_path`（`<id>_speech.wav`，16 kHz）。它是各块音频按顺序直接拼接的结果，块 k 占 [Σ_{j<k} dur_j, Σ_{j≤k} dur_j)，样本级对齐。只量有块、无 error 的流。
- **帧**：10 ms 一帧（160 个样本），不重叠；帧能量 = 20·log10(RMS + 1e-10)。
- **块峰值**：块内各帧能量的最大值。
- **静音帧**：能量 < 块峰值 − 40 dB。
- **静音段**：连续静音帧组成的最长段，长度 ≥ 150 ms（≥ 15 帧）。按位置分类：
  - **开头**：从块的第一帧开始；
  - **结尾**：到块的最后一帧结束；
  - **内部**：其余。
  - 整块都是静音时单独计为"全静音块"，不归入上面三类，单独报告块数和时长。
- **占比**：各类静音段的总时长 / 全部块的总时长（= 总语音时间 Σ dur_ms）。
  - 两个种子分别报，再报均值；
  - 另报每类静音段的个数、段长中位数和 p90、含该类静音的块所占比例。

## 2. 重新计时（同一音频内容下的上界）

- **压缩规则**：每个块里每个静音段（开头、结尾、内部，以及全静音块）都压到 ≤ 150 ms。块的新时长 = dur − Σ_段 max(0, 段长 − 150 ms)。块里的其他内容不变。
- **重算延迟**：把新时长写进一份新的 `results.jsonl`（原 `dur_ms` 另存为 `dur_ms_orig`；新目录，不覆盖原文件），用当前的 `speech_latency_lit.py --win eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl` 不改一行代码地重算：
  - 在线队列上，块按新时长播放；
  - 块内各词按新时长均分（与原口径相同）。
- **基线**：用同一版 `speech_latency_lit` 对原 `results.jsonl` 重算，与压缩后的结果严格配对。
- **指标**：
  - **主指标**：每条流语音 LAAL（按单元，在线，all）；
  - EndOffset（按单元，在线，all）；
  - 补充：单通道语音 LAAL / EndOffset。
- **性质**：这是上界。音频内容按假设不变，ASR-BLEU 没有测；真实压缩可能改变 ASR 结果，所以这里不报任何质量数字。

## 3. GO 线（看数之前定下）

- **GO** ⇔ 每条流语音 LAAL 的降幅（原 − 压缩后），两种子均值 ≥ 0.15 s。
- 两个种子的降幅另外分别列出。
- **GO 之后**：只在报告里提出一个推理时压缩静音的实验方案，交用户批准，不自行运行。原因是压缩可能改变 ASR-BLEU，而 ASR-BLEU 不得下降是硬约束。
- **NOT GO**：如实报告占比和上界，不再跟进。

## 4. 实现

- `experiments/mist_w3po_v1/talker/silence_share.py`：
  - `measure`：输出静音段和占比的 json；
  - `retime`：写压缩后的 `results.jsonl`；
  - 都拒绝覆盖已有输出。
- 配单元测试（合成信号，已知静音位置）。
- 结果写 `talker_silence_s1_results.zh-CN.md`，并追加到 LOG。
