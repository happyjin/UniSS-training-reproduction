# Hy-SimulMT-30B-A3B + Qwen3-TTS-12Hz-0.6B-Base，RealSI 句级（自动生成，`experiments/baselines_hysimulmt_qwen3tts_v1/score.py`）

登记：`paper/baselines_run_plan_hysimulmt_qwen3tts_v1.md`。只替换语音生成器：Hy-SimulMT 的提交块（R2，未重跑）、排队模型、speech-only ASR-BLEU、D1（段 LAAL，块开始播放时刻，含排队）、D0w（块写出时刻）、分档规则都与 R2 的“+ 论文 talker”相同。Qwen3-TTS 0.6B Base，x-vector 克隆源说话人（参考片段同论文 talker），官方默认采样解码，三个种子取均值。BLEU 分母为全部句子（En→Zh 346，Zh→En 431）。

## 全部点（并列：+ 论文 talker，R2）

| 方向 | 点 | 原生模式 | ASR-BLEU Qwen3-TTS（±sd） | ASR-BLEU 论文 talker（±sd） | 文本 BLEU | D1 Qwen3-TTS (s) | D1 论文 talker (s) | D0w (s) | 每句语音 (s) | 句尾后延续 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| En→Zh | `hy-sc-w4` | short_chunk | 25.72 (±0.03) | 25.67 (±0.11) | 26.60 | 1.90 | 1.72 | 1.46 | 9.11 | 4.27 |
| En→Zh | `hy-std-w4` | standard | 27.93 (±0.05) | 27.96 (±0.05) | 28.85 | 2.17 | 2.05 | 1.90 | 8.23 | 4.21 |
| Zh→En | `hy-sc-w4` | short_chunk | 18.97 (±1.40) | 20.04 (±0.32) | 19.82 | 2.08 | 1.57 | 1.31 | 11.86 | 7.34 |
| Zh→En | `hy-std-w4` | standard | 20.11 (±1.62) | 21.88 (±0.02) | 21.54 | 2.11 | 1.75 | 1.61 | 10.29 | 6.14 |

## 诊断：到上限的块（描述性，不进分档）

每个种子里，凡是有一块到生成上限（没有 codec EOS）的句子算“受影响句”；其余句子上两个系统的 ASR-BLEU 和 D1 对比（论文 talker 取同序号种子的运行，句子集合相同），三个种子取均值。

| 方向 | 点 | 到上限的块（三种子合计） | 受影响句 / 全部 | 其余句 ASR-BLEU Qwen3-TTS | 其余句 ASR-BLEU 论文 talker | 其余句 D1 Qwen3-TTS (s) | 其余句 D1 论文 talker (s) |
|---|---|---|---|---|---|---|---|
| En→Zh | `hy-sc-w4` | 30 | 10.0 / 346 | 25.86 | 25.80 | 1.86 | 1.71 |
| En→Zh | `hy-std-w4` | 13 | 4.3 / 346 | 28.06 | 28.10 | 2.14 | 2.06 |
| Zh→En | `hy-sc-w4` | 157 | 41.7 / 431 | 20.57 | 20.24 | 1.91 | 1.56 |
| Zh→En | `hy-std-w4` | 82 | 24.7 / 431 | 22.27 | 22.09 | 1.97 | 1.74 |

## 分档（按 D1，论文规则）

| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | Hy-SimulMT + Qwen3-TTS | Hy-SimulMT + 论文 talker |
|---|---|---|---|---|---|
| En→Zh | m1 | 1.18 | 19.80 | -- | -- |
| En→Zh | m2 | 1.98 | 24.21 | 25.72 (1.90)，`hy-sc-w4` | 25.67 (1.72)，`hy-sc-w4` |
| En→Zh | m3 | 2.46 | 25.01 | 27.93 (2.17)，`hy-std-w4` | 27.96 (2.05)，`hy-std-w4` |
| En→Zh | m4 | 2.98 | 25.54 | 27.93 (2.17)，`hy-std-w4` | 27.96 (2.05)，`hy-std-w4` |
| Zh→En | m1 | 1.18 | 14.89 | -- | -- |
| Zh→En | m2 | 1.92 | 19.55 | -- | 21.88 (1.75)，`hy-std-w4` |
| Zh→En | m3 | 2.40 | 21.07 | 20.11 (2.11)，`hy-std-w4` | 21.88 (1.75)，`hy-std-w4` |
| Zh→En | m4 | 2.98 | 22.47 | 20.11 (2.11)，`hy-std-w4` | 21.88 (1.75)，`hy-std-w4` |

## 分档（按 D0w，仅作参考）

| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | Hy-SimulMT + Qwen3-TTS | Hy-SimulMT + 论文 talker |
|---|---|---|---|---|---|
| En→Zh | m1 | 1.18 | 19.80 | -- | -- |
| En→Zh | m2 | 1.98 | 24.21 | 27.93 (1.90)，`hy-std-w4` | 27.96 (1.90)，`hy-std-w4` |
| En→Zh | m3 | 2.46 | 25.01 | 27.93 (1.90)，`hy-std-w4` | 27.96 (1.90)，`hy-std-w4` |
| En→Zh | m4 | 2.98 | 25.54 | 27.93 (1.90)，`hy-std-w4` | 27.96 (1.90)，`hy-std-w4` |
| Zh→En | m1 | 1.18 | 14.89 | -- | -- |
| Zh→En | m2 | 1.92 | 19.55 | 20.11 (1.61)，`hy-std-w4` | 21.88 (1.61)，`hy-std-w4` |
| Zh→En | m3 | 2.40 | 21.07 | 20.11 (1.61)，`hy-std-w4` | 21.88 (1.61)，`hy-std-w4` |
| Zh→En | m4 | 2.98 | 22.47 | 20.11 (1.61)，`hy-std-w4` | 21.88 (1.61)，`hy-std-w4` |

## 说明（R3，2026-11-12 21:13 布拉格）

- **En→Zh**：ASR-BLEU 与论文 talker 几乎一样：std 27.93 对 27.96，sc 25.72 对 25.67，差别在种子波动以内。
  但 D1 高 0.12–0.18 s：std 2.17 对 2.05 s，sc 1.90 对 1.72 s。按 D1 分档，两者落在同样的档位、用同样的点：
  m2 是 25.72 (1.90 s)，m3 / m4 是 27.93 (2.17 s)。
- **Zh→En**：ASR-BLEU 低 1.1–1.8：std 20.11 ± 1.62 对 21.88，sc 18.97 ± 1.40 对 20.04。D1 高 0.36–0.51 s：std 2.11 对 1.75 s，
  sc 2.08 对 1.57 s。std 点的 D1 是 2.11 s，超过 m2 预算 1.92 s，所以 m2 没有点（论文 talker 在 m2 有 21.88）。m3 / m4 是 20.11。
- **原因一：不停的块**（见上面的诊断表）。Qwen3-TTS 有一部分块一直不输出 codec EOS，只能等到生成上限才停，
  产生至少 14.4 s 的乱语。触发条件是极短文本（1–3 个字或词，如“而”“嗯。”“So,”“uh,”）加上真实会议录音的 x-vector。
  用 LJSpeech 音色时，同样的短文本都能正常停止。
  受影响的句子：Zh→En 占 6–10%（std 24.7 句、sc 41.7 句 / 431），En→Zh 占 1–3%。short_chunk 的短块多，受影响更多。
  Whisper 把这些乱语转写成几百词的重复或幻觉文本（Zh→En 系统长度 7.8k–9.1k 词，参考 7.1k），所以 Zh→En 三个种子差得多：
  std 18.93 / 19.44 / 21.95，sc 17.53 / 19.06 / 20.32。
  去掉受影响的句子后，两个系统的 ASR-BLEU 持平或 Qwen3-TTS 略高：Zh→En std 22.27 对 22.09，sc 20.57 对 20.24；En→Zh 差 ≤ 0.06。
- **原因二：语速**。Qwen3-TTS 没有语速控制，论文 talker 有按块的语速策略。即使只看没受影响的句子，Qwen3-TTS 每句语音也更长：
  En→Zh std 8.05 对 6.91 s，Zh→En std 8.78 对 6.04 s。排队因此更长，所以这些句子上 D1 仍高 0.08–0.35 s，句尾后延续也更长。
- **修订 1**（登记 §5）：每块生成上限 min(1800, max(180, 18 × 字符数)) 帧，取自 SimulInfer 自己的 qwen-tts 后端。
  到上限的块保留截断后的音频，照常进入时间轴和 ASR。三个种子合计到上限的块：En→Zh std 13 / 3507、sc 30 / 4575；
  Zh→En std 82 / 4809、sc 157 / 6225。如果按 generation_config 不设上限（8192 帧），这类块每块约 655 s，Zh→En 的 D1 会到几十秒量级。
  采样参数始终是 generation_config 的默认值。
- **检查**：每个运行 346 / 431 行齐全，ASR 覆盖 100%，0 错误。文本 BLEU 与 R2 一致（脚本里断言）。
  D0w 与 R2 完全相同（只由提交时刻决定），说明用的是同一批提交块。只有标点的 4 块没有合成。
- **种子**：三个，与论文的惯例一样；每批 32 块的组成固定，批级 torch 种子。延迟不计合成耗时，与论文相同；
  HF qwen-tts 在 H200 上 batch 32 时 RTF 约 0.1，没有计入延迟。
- **路径**：代码在 `experiments/baselines_hysimulmt_qwen3tts_v1/`；运行在 `eval_outputs/baselines_hysimulmt_qwen3tts_v1/a1/runs/`；
  日志在 `logs/baselines_hysimulmt_qwen3tts/`。第一次不设上限的尝试保留在 `eval_outputs/baselines_hysimulmt_qwen3tts_v1/{jobs.jsonl,claims,tts}`，
  没有用于结果。被替换的作业和运行在 `a1/tts/*/*/done_superseded/` 和 `a1/runs_superseded/`。
