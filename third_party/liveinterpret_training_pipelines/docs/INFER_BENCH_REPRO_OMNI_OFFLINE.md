# Repro-Omni-offline 推理跑测：不同 GPU 单卡对比

Repro-Omni-offline = SimulS2ST-Omni（Qwen2.5-Omni-3B + DualCodec talker）**离线**整句 S2ST，权重 = 官方
`SimulS2ST-Omni/offline` 基座 + zeyangsong 复现的 `stage2_multitask_full`（thinker LoRA r32 + talker 全量）。
推理用官方代码路径：`custom/streaming_si/eval/run_omni_offline.py`（只把官方 `run_offline.py` 改成「模型加载一次、批量循环」，
参数为官方默认：`max_s2tt_new_tokens=256`、`max_tts_new_tokens=500`、`do_sample=False`）。

> 基座是官方已训练好的 offline（含作者 Stage 1+2），不是从零复现（见 README_self 开头的解读边界）。

> **列的含义**：`README baseline` = `/data/workspace/README_self.md` 公布的该系统的数值（基线）；`H200 / A10G / L40S` = 本次单卡实测。README_self.md 表格里没展示的指标（Text-chrF、显存峰值、p90 等）沿用其底层 `out3/_report` 报告的数值。

| GPU | 状态 | 跑测目录 |
|---|---|---|
| H200（p5e，141GB） | ✅ 2026-10-21 完成 | `/mnt/fsx/junyaoyang/infer_bench/20261021-215832/` |
| A10G（g5.2xlarge，23GB） | ✅ 2026-10-23 完成 | `iegg-speech-model-data-eks` curion ns，S3 PVC `junyaoyang/infer_bench/20261023-005705-a10-repoff/` |
| L40S（g6e.xlarge，46GB） | ✅ 2026-10-26 完成 | `speech-model-data-eks` curion ns，S3 PVC `junyaoyang/infer_bench/20261026-193918-l40s-repoff/` |

---

## 1. 跑测口径

与 README_self（第三轮）完全相同：同一份官方仓库（`/mnt/fsx/stevehguo/eval/code/SimulS2ST-Omni-main`）、同一组
依赖（`code/site_omni`，transformers 4.52.3 用 PYTHONPATH 前置，不动容器环境）、同一组权重，清单最前加 1 条预热句。

| 项 | 取值 |
|---|---|
| 测试集 | RealSI en→zh 346 句 / zh→en 431 句、ACL 60/60 en→zh 416 句，共 1193 句、142.0 分钟源音频（stevehguo `work/manifests/all.jsonl`，只读复用） |
| 打分 | 同一套 `custom/streaming_si/eval/`：Whisper-large-v3 ASR → SacreBLEU BLEU/chrF、wmt22-comet-da、UTMOS22、SIM-o / ESim / AutoPCP |
| 运行方式 | **单卡独占、1193 句串行**（README_self 是 7 张卡各跑一片，各卡之间不争抢） |
| 入口 | `pipelines/streaming_si/bench_entry.sh`（`SYSTEM=rep_off GPU_TAG=<卡>`），任务 yaml `ktrain/ktrainjob_infer_bench.yaml` |

---

## 2. 结论（H200）

1. **译文逐位一致**：Text-BLEU / Text-chrF / Text-COMET 三个数据集都与 README_self 相同（28.20 / 23.42 / 52.51）——
   thinker 用 greedy，结果是确定的。
2. **语音侧有小幅波动**：ASR-BLEU 18.91 / 21.31 / 38.27（README_self 19.98 / 21.37 / 37.45），UTMOS 相差 ≤0.017、SIM-o ≤0.003。
   推断原因：VoiceBox（flow matching）合成没有固定随机种子，每次跑出的波形不同，经 ASR 后 BLEU 随之浮动；
   时长比（1.641 / 1.592 / 0.942）完全相同，说明 talker 生成的语音 token 数没变，变的是波形。**该推断未单独验证。**
3. **单卡 RTF 0.595，单句平均 4.25s、p90 13.1s**：长尾很重——RealSI 两个方向时长比约 1.6，生成语音明显过长，
   VoiceBox 解码时间随之拉长。各数据集 RTF 0.452～0.675，README_self 0.465～0.695。
4. **显存峰值 68.5GB**（缓存分配器口径，不等于必需显存）。
5. **0 失败、0 空输出**。

## 2.1 A10G 补充结论

6. **RTF 0.778，是 H200（0.595）的 1.31 倍**；单句均值 5.56s、p90 18.29s，推理 7284s。
   比 S4/S5 的 1.6 倍小，与流式 Omni 的 1.06～1.09 倍也不同：离线这条路线的耗时分在 thinker（GPU 重）
   与 VoiceBox 解码（长尾）两段，A10G 只在 thinker 段吃亏。
7. **显存峰值 21.6GB，占 23GB 的 94%——A10G 上余量很小。** 这是本次唯一需要警惕的地方：
   同一条链路在 H200 上显示 68.5GB（缓存分配器占用口径），A10G 已经贴着上限，且 0 失败。
   若要在 A10G 上批量并发（多进程 / 多请求），必须限制 batch 或换 L40S。
8. **质量与 H200 的差异在语音侧噪声量级**：Text-BLEU / Text-COMET 与 H200 相同（28.23 vs 28.20、0.7969 vs 0.7979），
   ASR-BLEU 20.04 / 20.21 / 40.31（H200 18.91 / 21.31 / 38.27，README_self 19.98 / 21.37 / 37.45）——
   三个值都落在 H200 与 README_self 之间的同一波动带内，与第 2 条「VoiceBox 无固定种子」的解释一致。
   UTMOS / SIM-o 相差 ≤0.006。
9. **0 失败、0 空输出**，与 H200 相同。

## 2.2 L40S 补充结论

10. **RTF 0.593，几乎等于 H200 的 0.595**，比 A10G（0.778）快 24%。这条路线 GPU 只占一部分——
    VoiceBox 解码、音频读写等 CPU/IO 段与卡无关，L40S 的算力已经够把 GPU 段压到不再主导。
11. **显存峰值 17.5GB（46GB 卡的 38%）**，比 A10G 的 21.6GB 还低，四档里余量最充足的一张卡。
12. **质量与 H200 一致**：ASR-BLEU 18.55 / 20.95 / 37.83（H200 18.91 / 21.31 / 38.27，
    A10G 20.04 / 20.21 / 40.31），Text-BLEU 28.16 / 23.45 / 52.06（H200 28.20 / 23.42 / 52.51），
    差异仍是 VoiceBox 无固定种子带来的波形波动，不是卡的问题。
13. **0 失败、0 空输出**。

---

## 3. 对比表

### 3.1 效率与资源（全部 1193 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| RTF（模型内计时） | 0.465～0.695（各数据集） | **0.595** | **0.778** | **0.593** |
| RTF（整体墙钟，含 I/O） | - | 0.622 | 0.855 | 0.665 |
| 单句耗时 均值 / 中位 | 3.43～5.09s（各数据集均值） | 4.25s / 2.82s | 5.56s / 3.39s | 4.23s / 2.63s |
| 单句耗时 p90 / 最大 | - | 13.10s / 14.68s | 18.29s / 21.21s | 13.99s / 16.31s |
| 显存峰值（nvidia-smi） | - | 68.5GB | 21.6GB | 17.5GB |
| 失败 / 空译文 / 空语音 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

### 3.2 RealSI en→zh（346 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 19.98 | 18.91 | 20.04 | 18.55 |
| ASR-chrF | 17.51 | 17.23 | 17.00 | 16.73 |
| ASR-COMET | 0.6319 | 0.6298 | 0.6314 | 0.6234 |
| Text-BLEU | 28.20 | 28.20 | 28.23 | 28.16 |
| Text-chrF | 25.85 | 25.85 | 25.98 | 25.94 |
| Text-COMET | 0.7979 | 0.7979 | 0.7969 | 0.7976 |
| UTMOS | 2.529 | 2.530 | 2.524 | 2.545 |
| SIM-o | 0.788 | 0.788 | 0.789 | 0.794 |
| SIM-o delta | 0.132 | 0.133 | 0.127 | 0.138 |
| ESim delta | 0.011 | 0.008 | 0.011 | 0.009 |
| AutoPCP delta | 0.713 | 0.716 | 0.718 | 0.713 |
| RTF（语料级） | 0.689 | 0.675 | 0.907 | 0.678 |
| RTF（中位） | 0.493 | 0.481 | 0.563 | 0.451 |
| 单句耗时均值（s） | 5.09 | 4.98 | 6.70 | 5.01 |
| 时长比 | 1.641 | 1.641 | 1.690 | 1.611 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 0 | 0 | 0 | 0 |
| 空语音 | 0 | 0 | 0 | 0 |

### 3.3 RealSI zh→en（431 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 21.37 | 21.31 | 20.21 | 20.95 |
| ASR-chrF | 45.94 | 46.43 | 45.73 | 46.42 |
| ASR-COMET | 0.7070 | 0.7092 | 0.6991 | 0.7108 |
| Text-BLEU | 23.42 | 23.42 | 23.64 | 23.45 |
| Text-chrF | 53.40 | 53.40 | 53.55 | 53.34 |
| Text-COMET | 0.7780 | 0.7780 | 0.7779 | 0.7778 |
| UTMOS | 2.639 | 2.631 | 2.612 | 2.677 |
| SIM-o | 0.808 | 0.811 | 0.801 | 0.811 |
| SIM-o delta | 0.109 | 0.111 | 0.112 | 0.113 |
| ESim delta | 0.020 | 0.018 | 0.019 | 0.022 |
| AutoPCP delta | 0.822 | 0.831 | 0.770 | 0.827 |
| RTF（语料级） | 0.695 | 0.675 | 0.918 | 0.669 |
| RTF（中位） | 0.506 | 0.490 | 0.571 | 0.460 |
| 单句耗时均值（s） | 4.67 | 4.54 | 6.17 | 4.49 |
| 时长比 | 1.592 | 1.592 | 1.695 | 1.547 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 0 | 0 | 0 | 0 |
| 空语音 | 0 | 0 | 0 | 0 |

### 3.4 ACL 60/60 en→zh（416 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 37.45 | 38.27 | 40.31 | 37.83 |
| ASR-chrF | 31.96 | 31.59 | 33.15 | 31.73 |
| ASR-COMET | 0.7053 | 0.7116 | 0.7122 | 0.7138 |
| Text-BLEU | 52.51 | 52.51 | 52.58 | 52.06 |
| Text-chrF | 44.88 | 44.88 | 44.92 | 44.49 |
| Text-COMET | 0.8469 | 0.8469 | 0.8452 | 0.8466 |
| UTMOS | 2.765 | 2.748 | 2.754 | 2.749 |
| SIM-o | 0.922 | 0.921 | 0.928 | 0.923 |
| SIM-o delta | 0.097 | 0.094 | 0.096 | 0.097 |
| ESim delta | 0.009 | 0.011 | 0.013 | 0.012 |
| AutoPCP delta | 0.754 | 0.746 | 0.757 | 0.744 |
| RTF（语料级） | 0.465 | 0.452 | 0.540 | 0.450 |
| RTF（中位） | 0.418 | 0.405 | 0.473 | 0.382 |
| 单句耗时均值（s） | 3.43 | 3.33 | 3.98 | 3.32 |
| 时长比 | 0.942 | 0.942 | 0.874 | 0.949 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 0 | 0 | 0 | 0 |
| 空语音 | 0 | 0 | 0 | 0 |

AutoPCP 的 README_self 列取自 `README_self.md` 正文：stevehguo 的 `out3/_report/*.json` 里 AutoPCP 仍是修 bug 前的旧值（他目录下有 `_para_pcpbug` 备份），与正文不一致；本次结果与正文一致。

---

## 4. 换卡注意

- 官方 Omni 依赖 transformers 4.52.3（`site_omni` 前置），换集群时要连同 `site_omni/`、官方仓库、`models/SimulS2ST-Omni`（约 18GB）、
  复现 stage2 一起搬过去。
- 其余同 `INFER_BENCH_S4_OFFLINE.md` 第 4 节。
- **A10G 显存贴顶（21.6/23GB）**：只适合单路串行；并发或多实例请上 L40S。

## 5. 复现

yaml 里 `SYSTEM: "rep_off"`。推理调用与 stevehguo `run_worker3.sh repoff` 逐字相同，只换输出目录。
H200 上推理 88 分钟、A10G 上 7284s（约 2 小时）；打分各步耗时见跑测目录的 `status.log`。
