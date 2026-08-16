# Stage 00 结果报告：Phase3 v4 基线、缓存与真流式前端审计

## 1. 结论

Stage 00 已完成，基础设施 gate **有条件通过**，允许开始 Stage A 的数据审计和实现。

这里的“有条件通过”只表示以下底层条件已经建立：Phase3 原生权重可精确重导出、共享因果 WhisperVQ 前端无未来泄漏、缓存与严格同因果参考一致、BiCodec 分块覆盖无 gap/overlap、固定 pilot15 离线锚点已成功生成。它不表示 Phase3 greedy 音频生成本身没有质量问题。

后续阶段必须遵守三个约束：

1. Qwen cached runtime 暂时固定为 `FP32 + eager attention`；BF16 runtime gate 未通过，不能宣称 BF16 cache 数值等价。
2. 所有保持率必须和本报告的固定 pilot15、greedy、相同模式结果比较，不能与 full198 dev 或论文表格混用。
3. Stage A/C 必须显式监控 EOS、semantic 长度和空 semantic，因为 Phase3 基线已经暴露这些问题。

## 2. 运行身份与隔离

| 项目 | 值 |
|---|---|
| 实验 | `uniss_phase3_v4_quality_first_true_streaming_pilot15_v1` |
| Stage 00 前端/缓存审计 run | `20260816T024658Z` |
| 离线基线 run | `20260816T031129Z` |
| 评估代码 commit | `277f0a4` |
| 原生 Phase3 | `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075` |
| canonical HF | `checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf` |
| fresh HF re-export | `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_native_reexport_20260816T024658Z` |
| 原始结果目录 | `eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_phase3_offline_20260816T031129Z` |

所有新代码、数据索引、日志、报告、checkpoint 和评估音频都写入本实验专属目录。未覆盖或修改任何历史实验结果。

## 3. 固定 validation

固定 validation 从已有 pilot15 formal validation 中按方向和时长分层抽取，随机种子为 `20260816`。

| 集合 | 样本数 | SHA256 |
|---|---:|---|
| text | 256 | `f96581f6dbe621b92e9a8e7652185be8e90f9d217b74e8e4212c00ed87ce59d7` |
| audio | 64 | `53ca77bbaa2fe970c1080aa1ba2aa50316dba64326033705611ef8a1d9dd28e7` |
| formal source manifest | 13,469 population records | `2981204ad0db9ca08bbc5ec206203ed1c8f7a410ad7a211cd632661e14d91483` |

text256 恰好包含 8 个 `方向 × 时长` strata，每个 stratum 32 条：

- `cmn→eng` 和 `eng→cmn`；
- `<4 s`、`4–8 s`、`8–15 s`、`>=15 s`。

text/audio ID 均唯一，8 个 worker part 无重复、无遗漏。

## 4. Gate 结果

| 子门 | 结果 | 核心证据 |
|---|---|---|
| canonical artifact audit | 通过 | native iteration=9075；HF 24 层、hidden=896、heads=14/KV=2、vocab=180480；WhisperVQ pre-VQ=16 层 |
| real-PCM causality | 通过 | 改动第 3 个 block 后，过去 6 个 token/hidden 完全不变 |
| Whisper cached/recomputed parity | 通过 | 真实 PCM 87/87 GLM exact；hidden/quantized 最大误差 0 |
| 30.4 s acoustic reset | 通过 | 380/380 GLM exact；hidden 最大误差 0；恰好发生 1 次 reset |
| Qwen BF16 cache | **未通过** | runtime top-1 最终可达 100%，但最小 cosine `0.99945366 < 0.9999` |
| Qwen FP32 eager cache | 通过 | 所有模式 runtime top-1 100%；cache length 100%；最小 cosine `0.99999982` |
| BiCodec streaming coverage | 通过 | 626 semantic、200,320 samples；3 种 irregular partition 均 0 gap/overlap；speaker change 被拒绝 |
| native→fresh HF export | 通过 | 290/290 tensor bit-exact；最大权重误差 0；固定 prompt logits cosine 1.0 |
| fixed Phase3 offline baseline | 通过 | 512 text rows + 256 audio rows 完整聚合；无分片遗漏 |

### 4.1 共享因果 WhisperVQ 前端

配置：

- 16 kHz PCM；
- 160 ms block；
- 0 ms right context；
- `STFT center=False`；
- arrived-block local normalization；
- WhisperVQ 前 16 层 block-causal KV cache；
- 29.92 s 最大 acoustic segment 后 reset。

真实样本 `NCSSD_R_EN_0000000083`（6.94 s）结果：

| 指标 | 数值 |
|---|---:|
| cached GLM token | 87 |
| recomputed/cached exact | 87/87 |
| hidden maximum absolute error | 0 |
| cached step p50 | 9.14 ms / 160 ms block |
| cached step p95 | 10.39 ms / 160 ms block |
| cached RTF | 0.05876 |

单次整段 block-mask 与逐块重计算因 CUDA GEMM reduction geometry 有约 `2.72e-5` 最大 hidden 误差，但 GLM token 仍 87/87 相同。正式 gate 使用“逐 block 无 persistent KV 重计算”作为严格同因果参考，因此没有把不同 reduction geometry 误判为 cache 错误。

### 4.2 Qwen cache 的失败尝试与最终约束

| 版本 | dtype / attention | top-1 | 最小 cosine | 结论 |
|---|---|---:|---:|---|
| 初始 BF16 | BF16 / 默认 | 95.9%–100% | 最低约 0.99897 | 失败 |
| append-boundary BF16 | BF16 | 部分不一致 | `<0.9999` | 失败 |
| runtime BF16 v3 | BF16 / eager | 100% | `0.99945366` | cosine 硬门失败 |
| runtime FP32 v4 | FP32 / eager | 100% | `0.99999982` | 通过 |

因此 Stage A 训练仍可使用 BF16 的完整 causal forward，但所有用于 commit/rollback 判定的 cached validation/runtime 在另行证明前必须使用 FP32 eager。不得把 BF16 v3 写成通过。

## 5. Phase3 matching offline baseline

### 5.1 解码设置

```text
temperature=0
do_sample=false
max_new_tokens=1500
repetition_penalty=1.1
seed=20260816
dtype=BF16
8 GPUs / 8 deterministic workers
```

text256 执行 `quality + performance`；audio64 执行 `quality + performance + direct_s2st + tts`，并保存模型音频、source audio 和 reference audio。

### 5.2 文本翻译与 ASR

| Mode | 方向 | Text-BLEU | 样本数 |
|---|---|---:|---:|
| Quality | ZH→EN | 33.2298 | 128 |
| Quality | EN→ZH | 52.4212 | 128 |
| Performance | ZH→EN | 28.8797 | 128 |
| Performance | EN→ZH | 46.7853 | 128 |

| Quality ASR 源语言 | 指标 | 错误率 | edits / reference units |
|---|---|---:|---:|
| 中文（ZH→EN） | CER | 5.4661% | 285 / 5,214 |
| 英文（EN→ZH） | WER | 5.6072% | 175 / 3,121 |

audio64 子集上的翻译锚点：

| Mode | ZH→EN BLEU | EN→ZH BLEU |
|---|---:|---:|
| Quality | 36.7778 | 51.4297 |
| Performance | 26.2079 | 44.0350 |

### 5.3 SLC

`SLC-0.2` 表示生成/源音频时长比落在 `[0.8, 1.2]`，`SLC-0.4` 表示落在 `[0.6, 1.4]`。

| Mode | 方向 | N | mean duration ratio | SLC-0.2 | SLC-0.4 |
|---|---|---:|---:|---:|---:|
| Quality | ZH→EN | 31 | 2.4651 | 38.71% | 41.94% |
| Quality | EN→ZH | 32 | 1.5479 | 56.25% | 81.25% |
| Performance | ZH→EN | 32 | 2.3814 | 43.75% | 50.00% |
| Performance | EN→ZH | 32 | 1.5721 | 65.62% | 81.25% |
| Direct S2ST | ZH→EN | 31 | 3.0609 | 12.90% | 19.35% |
| Direct S2ST | EN→ZH | 32 | 3.2719 | 3.12% | 12.50% |
| TTS | ZH→EN | 32 | 1.0392 | 62.50% | 90.62% |
| TTS | EN→ZH | 32 | 1.4893 | 46.88% | 84.38% |

Direct S2ST 和部分 Quality/Performance 的时长明显失控，说明后续 Stage C 必须保留 duration/EOS 诊断，不能只看文本 BLEU。

### 5.4 生成健康度

| 子集 | rows | error | missing semantic | missing EOS | generation p50 | generation p95 |
|---|---:|---:|---:|---:|---:|---:|
| text256 两模式 | 512 | 0 | 0 | 125 | 11.19 s | 30.14 s |
| audio64 四模式 | 256 | 2 | 2 | 92 | 12.75 s | 30.40 s |

唯一空 semantic 样本为 `emilia_zh_0006179615`：

- `direct_s2st`：直接 EOS，无 translation、无 semantic；
- `quality`：生成了 translation，但直接 EOS、无 semantic。

这两行没有被删除；SLC 明确记录为 skipped `missing_audio_path`。其余 254/256 行成功生成可解码音频。

missing-EOS 主要表现为 semantic 生成达到 `max_new_tokens=1500`，部分短源音频生成接近 29–30 秒目标音频。该问题来自固定 greedy Phase3 baseline 的生成行为，不是分片、解析或 BiCodec 解码错误。

## 6. GPU 与耗时

整个离线基线从 `03:11:29 UTC` 启动，约在 `03:45:17 UTC` 完成聚合，wall time 约 33 分 49 秒。

对 GPU 有实际模型驻留的监控点统计：

| 指标 | 数值 |
|---|---:|
| active GPU utility mean | 19.50% |
| active power mean | 131.42 W |
| observed power max | 152.70 W |
| observed memory max | 8,537 MiB |

低利用率原因是每卡单样本、逐 token 自回归 `model.generate` 加 BiCodec 解码，并非 Megatron 训练 data loader 或 batch 配置。本次优先保持固定样本与确定性；后续正式训练仍必须按 8 卡、18000 pack、GBS=128 的真实训练负载单独验收 utility/power，不能用本评估功率预测训练功率。

## 7. Gate 判定与 Stage A 入口条件

Stage 00 基础设施 gate 判定为 `passed=true`，同时冻结以下约束：

- `qwen_cached_runtime_dtype=float32`；
- `qwen_attention_implementation=eager`；
- `whisper_block_ms=160`；
- `whisper_right_context_ms=0`；
- `stft_center=false`；
- fixed text/audio manifest hash 不得改变；
- Stage A 只能从 native `iter_0009075` fresh fine-tune；
- Stage A 不得读取完整 source `bicodec_global` 作为早期未来信息；
- Stage A validation 必须同时报告 WER/CER、pre-final commit、final-only/blank collapse 和 cached/full parity。

Stage A 可以开始的工作仅包括：数据 provenance/对齐审计、event 构造、native Megatron compound wrapper、单元测试、单卡 real-checkpoint smoke 和 8 卡 20–50 step smoke。正式训练只能在 Stage A 自己的数据与 smoke gate 通过后开始。

## 8. 原始证据路径

- 前端：`reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_baseline/20260816T024658Z/frontend_real_pcm.json`
- Qwen BF16：`reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_baseline/20260816T024658Z/qwen_hf_cache_parity_v3.json`
- Qwen FP32：`reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_baseline/20260816T024658Z/qwen_hf_cache_parity_fp32_v4.json`
- BiCodec：`reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_baseline/20260816T024658Z/bicodec_streaming_coverage.json`
- native/HF：`reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_baseline/20260816T024658Z/native_hf_reexport_parity.json`
- baseline summary：`eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_phase3_offline_20260816T031129Z/baseline_summary.json`
- merged text：`eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_phase3_offline_20260816T031129Z/merged/text_results.jsonl`
- merged audio：`eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_phase3_offline_20260816T031129Z/merged/audio_results.jsonl`
- GPU CSV：`logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_baseline/gpu_20260816T031129Z.csv`
