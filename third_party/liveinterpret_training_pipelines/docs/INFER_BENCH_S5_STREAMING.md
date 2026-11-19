# S5-streaming 推理跑测：不同 GPU 单卡对比

S5-streaming = 自研流式同传：S4 `checkpoint-24980` 底座 + S5 adapter（`s5_simul_default_640`），
640ms 分块 WAIT / WRITE，每段 WRITE 立即生成语音 token，整句末尾一次过声码器。
推理入口 `custom/streaming_si/inference/infer_s5_streaming.py`（thinker 增量缓存 + CUDA Graph、talker CUDA Graph）。

每种 GPU 一列，基线列取自 `README_self.md` 的 S5-streaming 列。

> **列的含义**：`README baseline` = `/data/workspace/README_self.md` 公布的该系统的数值（基线）；`H200 / A10G / L40S` = 本次单卡实测。README_self.md 表格里没展示的指标（Text-chrF、显存峰值、p90 等）沿用其底层 `out3/_report` 报告的数值。

同系列：`INFER_BENCH_S4_OFFLINE.md`、
`INFER_BENCH_REPRO_OMNI_OFFLINE.md`、`INFER_BENCH_REPRO_OMNI_STREAMING.md`。

| GPU | 状态 | 跑测目录 |
|---|---|---|
| H200（p5e，141GB） | ✅ 2026-10-21 完成 | `/mnt/fsx/junyaoyang/infer_bench/20261021-215810/` |
| A10G（g5.2xlarge，23GB） | ✅ 2026-10-23 完成 | `iegg-speech-model-data-eks` curion ns，S3 PVC `junyaoyang/infer_bench/20261023-200224-a10-s5/`（latency 段续跑产物在 `...-a10-s5-lat/`，已写回前者） |
| L40S（g6e.xlarge，46GB） | ✅ 2026-10-26 完成 | `speech-model-data-eks` curion ns，S3 PVC `junyaoyang/infer_bench/20261026-193907-l40s-s5/` |

---

## 1. 跑测口径

与 README_self 完全相同：模型 `s4_ck` + `s5_adapter`（FSX 只读），`infer_s5_streaming.py` 全部默认参数
（640ms 分块、talker top_k 25 / top_p 0.8、flow ODE 5 步、源语音声学 prompt、seed 0），清单最前加 1 条预热句、打分时剔除。

| 项 | 取值 |
|---|---|
| 测试集 | RealSI en→zh 346 句 / zh→en 431 句、ACL 60/60 en→zh 416 句，共 1193 句、142.0 分钟源音频（stevehguo `work/manifests/all.jsonl`，只读复用） |
| 打分 | 同一套 `custom/streaming_si/eval/`：Whisper-large-v3 ASR → SacreBLEU BLEU/chrF、wmt22-comet-da、UTMOS22、SIM-o / ESim / AutoPCP、延迟（SimulEval 语音口径 LAAL，CA 与非 CA） |
| 运行方式 | **单卡独占、1193 句串行**（README_self 是 7 张卡各跑一片，各卡之间不争抢） |
| 入口 | `pipelines/streaming_si/bench_entry.sh`（`SYSTEM=s5 GPU_TAG=<卡>`），任务 yaml `ktrain/ktrainjob_infer_bench.yaml` |

延迟口径：每句的分段输出由 `stream_segments.py` 整理（WRITE 步发出时刻 = 已读源音频；CA 再加上计算墙钟，
声码器耗时按 token 比例摊到各段），`latency_eval.py` 用 Whisper 词级时间戳对齐算 LAAL / LAAL-EOW / Start/EndOffset。

---

## 2. 结论（H200）

1. **质量与延迟都与 README_self 逐位一致**：三个数据集的 ASR/Text BLEU、chrF、COMET，UTMOS，SIM-o / ESim / AutoPCP，
   非 CA 的 LAAL 全部相同（LAAL zh→en 2.425 vs 2.426 为四舍五入）。S5 推理是确定的。
2. **CA 延迟比 README_self 低 0.007～0.048s**：单卡独占，计算耗时略短；LAAL (CA) 3.00～3.32s，StartOffset (CA) 1.22～1.73s。
3. **单卡 RTF 0.356，单句平均 2.54s、p90 3.80s**（各数据集 RTF 0.351～0.359，README_self 0.344～0.353）。
   比 S4-offline（0.300）慢约 19%：流式每个 640ms 分块都要跑一次 thinker 决策。
4. **显存峰值 33.2GB**（nvidia-smi 口径），比 S4 全量跑测的 74.7GB 低得多：S5 每句序列短，缓存分配器没有被长句撑大。
5. **稳定性与 README_self 相同**：2 条失败（RealSI zh→en 两条 7.5s 句子，talker StaticCache 上限 2048 被超出）、
   13 条空译文（极短句全程 WAIT）。都是模型 / 推理代码的既有问题，不是跑测环境引起的。

## 2.1 A10G 补充结论

6. **RTF 0.590，是 H200（0.356）的 1.66 倍**；单句均值 4.21s、p90 6.50s，推理 5870s。
   与 S4-offline 的 A10G/H200 比（0.478/0.300 ≈ 1.6 倍）一致，比例由算力和显存带宽决定。
7. **CA 延迟随计算变慢而抬升**：LAAL (CA) 3.45/3.48/3.66s（H200 3.11/3.00/3.32），EndOffset (CA) 5.03/5.16/5.59s
   （H200 3.29/3.20/4.18）。非 CA 的 LAAL 仍与 H200 一致（en→zh 2.513 vs 2.51、ACL 2.694 vs 2.70），
   zh→en 2.515 vs 2.42 偏高 0.09，与该方向 ASR 质量下降同源（见第 9 条）。
8. **显存峰值 16.3GB，为 23GB 的 71%**，无 OOM、失败 0 条。A10G 单卡跑 S5 有余量；
   S5 的 16.3GB 低于 H200 的 33.2GB，同样是「缓存分配器占用」口径差异，不是模型变小。
9. **质量：en→zh 与 ACL 与 H200 基本一致，zh→en 的合成语音有偏差。** ASR-BLEU en→zh 21.26 vs 21.22、
   ACL 38.41 vs 38.07，COMET/UTMOS/SIM-o/ESim/AutoPCP 全部在 0.01 量级内（AutoPCP 与 README_self 正文一致）。
   zh→en 则是 ASR-BLEU 14.96 vs 16.16（−1.2）、时长比 0.889 vs 0.808（合成语音长约 10%），
   但 **Text-BLEU 反而更高**（17.30 vs 16.44）——模型译文没问题，是 talker 在这些样本上采样出更长的语音、
   导致 Whisper 转写变差。S5 的 talker 是 top_k/top_p 采样，A10G 的 bf16 内核与 H200 舍入不同，
   采样起点一偏就整句分叉（S4 的 thinker 是 greedy，所以 S4 没有这个现象）。
10. **H200 上那 2 条失败（talker StaticCache 2048 溢出）在 A10G 上没有复现**，同样源于采样分叉：
   这两句在 A10G 上生成的 token 数没到上限。
11. **latency（LAAL）一步在 A10G 上会 OOM，需把 whisper 批大小降到 2。** `latency_eval.py` 用
   `return_timestamps='word'`，transformers 为此开 `output_attentions`，32 层 encoder 自注意力会同时留在显存里：
   8 条 × 32 层 × 1500² × 20 头 × 2B ≈ 688 MiB/层 × 32 ≈ 22GB，23GB 的 A10G 加上权重就爆
   （`CUDA out of memory ... Process has 21.79 GiB in use`）。`bench_entry.sh` 已按卡型给默认值
   （a10→`LAT_BATCH=2`、l40s→4、其它→8）；batch 2 时显存约 10GB，1193 句约 55 分钟。
   只用已有推理结果补跑这一段：`RESUME_FROM=lat`（注意 S3 挂载不能覆盖已有对象，PUBLISH_TO 要指到新目录）。

## 2.2 L40S 补充结论

12. **RTF 0.441，介于 H200（0.356）与 A10G（0.590）之间**，是 H200 的 1.24 倍、A10G 的 0.75 倍。
    单句均值 3.15s、p90 4.70s，推理 4603s。
13. **质量与 H200 更接近，zh→en 没有 A10G 那种下滑**：ASR-BLEU 20.68 / 16.11 / 37.53
    （H200 21.22 / 16.16 / 38.07，A10G 21.26 / 14.96 / 38.41），非 CA 的 LAAL 2.53 / 2.46 / 2.71
    与 H200（2.51 / 2.42 / 2.70）差 ≤0.04。zh→en 在 L40S 上是 16.11、几乎等于 H200，说明 A10G 那次的
    14.96 确实是采样分叉的一次偶发结果，不是算力不足导致的系统性退化。
14. **空译文 11 条（H200 / A10G 都是 13 条）**，差异来自少数极短句在这次采样里没有全程 WAIT；
    仍然 0 失败、0 空语音。
15. **显存峰值 17.2GB（46GB 卡的 37%）**，比 A10G 的 16.3GB 略高；与 H200 的 33.2GB 一样是缓存分配器口径，
    不代表必需显存。

---

## 3. 对比表

### 3.1 效率与资源（全部 1193 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| RTF（模型内计时） | 0.344～0.353（各数据集） | **0.356** | **0.590** | **0.441** |
| RTF（整体墙钟，含 I/O） | - | 0.403 | 0.689 | 0.540 |
| 单句耗时 均值 / 中位 | 2.38～2.58s（各数据集均值） | 2.54s / 2.30s | 4.21s / 3.80s | 3.15s / 2.90s |
| 单句耗时 p90 / 最大 | - | 3.80s / 17.20s | 6.50s / 17.20s | 4.70s / 15.80s |
| 显存峰值（nvidia-smi） | - | 33.2GB | 16.3GB | 17.2GB |
| 失败 / 空译文 / 空语音 | 2 / 13 / 2 | 2 / 13 / 2 | 0 / 13 / 0 | 0 / 11 / 0 |

### 3.2 RealSI en→zh（346 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 21.22 | 21.22 | 21.26 | 20.68 |
| ASR-chrF | 19.13 | 19.13 | 18.83 | 18.38 |
| ASR-COMET | 0.6196 | 0.6196 | 0.6227 | 0.6199 |
| Text-BLEU | 22.96 | 22.96 | 22.98 | 22.97 |
| Text-chrF | 20.45 | 20.45 | 20.50 | 20.41 |
| Text-COMET | 0.6634 | 0.6634 | 0.6666 | 0.6648 |
| UTMOS | 3.314 | 3.314 | 3.295 | 3.312 |
| SIM-o | 0.805 | 0.805 | 0.805 | 0.803 |
| SIM-o delta | 0.156 | 0.156 | 0.154 | 0.152 |
| ESim delta | 0.012 | 0.012 | 0.013 | 0.010 |
| AutoPCP delta | 0.946 | 0.946 | 0.946 | 0.957 |
| LAAL | 2.51 | 2.51 | 2.51 | 2.53 |
| LAAL (CA) | 3.12 | 3.11 | 3.45 | 3.13 |
| LAAL-EOW (CA) | 3.51 | 3.50 | 3.89 | 3.52 |
| StartOffset (CA) | 1.54 | 1.53 | 1.76 | 1.54 |
| EndOffset (CA) | 3.32 | 3.29 | 5.03 | 4.05 |
| RTF（语料级） | 0.344 | 0.351 | 0.580 | 0.439 |
| RTF（中位） | 0.342 | 0.351 | 0.581 | 0.432 |
| 单句耗时均值（s） | 2.54 | 2.60 | 4.29 | 3.24 |
| 时长比 | 0.816 | 0.816 | 0.855 | 0.890 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 4 | 4 | 5 | 4 |
| 空语音 | 0 | 0 | 0 | 0 |

### 3.3 RealSI zh→en（431 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 16.16 | 16.16 | 14.96 | 16.11 |
| ASR-chrF | 41.00 | 41.00 | 41.13 | 41.31 |
| ASR-COMET | 0.6614 | 0.6614 | 0.6606 | 0.6627 |
| Text-BLEU | 16.44 | 16.44 | 17.30 | 16.49 |
| Text-chrF | 41.39 | 41.39 | 42.19 | 41.65 |
| Text-COMET | 0.6607 | 0.6607 | 0.6642 | 0.6627 |
| UTMOS | 3.713 | 3.712 | 3.702 | 3.711 |
| SIM-o | 0.894 | 0.894 | 0.895 | 0.895 |
| SIM-o delta | 0.145 | 0.144 | 0.145 | 0.149 |
| ESim delta | 0.020 | 0.020 | 0.021 | 0.020 |
| AutoPCP delta | 0.793 | 0.793 | 0.810 | 0.811 |
| LAAL | 2.43 | 2.42 | 2.52 | 2.46 |
| LAAL (CA) | 3.01 | 3.00 | 3.48 | 3.06 |
| LAAL-EOW (CA) | 3.85 | 3.83 | 4.49 | 3.93 |
| StartOffset (CA) | 1.23 | 1.22 | 1.47 | 1.23 |
| EndOffset (CA) | 3.25 | 3.20 | 5.16 | 3.45 |
| RTF（语料级） | 0.353 | 0.359 | 0.605 | 0.446 |
| RTF（中位） | 0.363 | 0.376 | 0.621 | 0.457 |
| 单句耗时均值（s） | 2.38 | 2.42 | 4.06 | 3.00 |
| 时长比 | 0.808 | 0.808 | 0.889 | 0.821 |
| 失败 | 2 | 2 | 0 | 0 |
| 空译文 | 3 | 3 | 2 | 1 |
| 空语音 | 2 | 2 | 0 | 0 |

### 3.4 ACL 60/60 en→zh（416 句）

| 指标 | README baseline | H200 | A10G | L40S |
|---|---|---|---|---|
| ASR-BLEU | 38.07 | 38.07 | 38.41 | 37.53 |
| ASR-chrF | 31.77 | 31.77 | 31.92 | 30.97 |
| ASR-COMET | 0.7004 | 0.7004 | 0.7018 | 0.6912 |
| Text-BLEU | 39.72 | 39.72 | 39.85 | 39.03 |
| Text-chrF | 33.60 | 33.60 | 33.68 | 33.10 |
| Text-COMET | 0.7395 | 0.7395 | 0.7388 | 0.7345 |
| UTMOS | 3.385 | 3.385 | 3.354 | 3.378 |
| SIM-o | 0.936 | 0.936 | 0.938 | 0.936 |
| SIM-o delta | 0.100 | 0.100 | 0.100 | 0.104 |
| ESim delta | 0.013 | 0.013 | 0.013 | 0.013 |
| AutoPCP delta | 0.846 | 0.846 | 0.847 | 0.819 |
| LAAL | 2.70 | 2.70 | 2.69 | 2.71 |
| LAAL (CA) | 3.33 | 3.32 | 3.66 | 3.35 |
| LAAL-EOW (CA) | 3.75 | 3.73 | 4.11 | 3.78 |
| StartOffset (CA) | 1.74 | 1.73 | 1.91 | 1.72 |
| EndOffset (CA) | 4.23 | 4.18 | 5.59 | 4.36 |
| RTF（语料级） | 0.350 | 0.356 | 0.583 | 0.437 |
| RTF（中位） | 0.347 | 0.354 | 0.579 | 0.431 |
| 单句耗时均值（s） | 2.58 | 2.62 | 4.30 | 3.22 |
| 时长比 | 0.875 | 0.875 | 0.870 | 0.876 |
| 失败 | 0 | 0 | 0 | 0 |
| 空译文 | 6 | 6 | 6 | 6 |
| 空语音 | 0 | 0 | 0 | 0 |

AutoPCP 的 README_self 列取自 `README_self.md` 正文：stevehguo 的 `out3/_report/*.json` 里 AutoPCP 仍是修 bug 前的旧值（他目录下有 `_para_pcpbug` 备份），与正文不一致；本次结果与正文一致。

chrF 中英两个方向不可横比；ESim 区分度弱，不作结论依据。

---

## 4. 换卡注意

同 `INFER_BENCH_S4_OFFLINE.md` 第 4 节（A10G sm_86 不在镜像 kernel 架构列表里；A10G / L40S 在
`iegg-speech-model-data-eks`，需 kubectl 起 Pod、模型与数据先搬运）。S5 的显存峰值 33GB，L40S（48GB）可直接跑；
A10G（24GB）已实测：S5 推理/ASR/UTMOS/副语言都在 16.3GB 以内，**只有 latency 一步要用 `LAT_BATCH=2`**（见 2.1 第 11 条）。

## 5. 复现

```bash
# yaml 里 SYSTEM: "s5"，其余同 S4（见 INFER_BENCH_S4_OFFLINE.md 第 5 节）
# A10G / L40S 额外：LAT_BATCH 会自动按 GPU_TAG 取 2 / 4（也可显式覆盖）
# 只补跑 latency（复用已有推理结果）：RESUME_FROM=lat OUT=<本地已有目录>
```

步骤与 S4 相同，多两步：第 4 步先用 `stream_segments.py s5` 整理分段，第 5 步多跑 `latency_eval.py`（H200 上约 40 分钟，
Whisper 词级对齐是最慢的一步；A10G 上 `LAT_BATCH=2` 约 55 分钟）。H200 上推理 57 分钟、A10G 上 98 分钟；
打分各步耗时见跑测目录的 `status.log`。
