# Stage A 正式训练与自由运行 ASR 质量门报告

## 1. 最终结论

本次 Stage A 正式训练**数值稳定并完整结束**，但最终自由运行质量门**未通过**。因此：

- 保留 `iter_0000381` 作为失败诊断 checkpoint，不把它标记为 selected；
- 已写出不可变 `GATE_FAILED.json`；
- **没有创建 `SELECTED_CHECKPOINT.json`**；
- **Stage B incremental MT 不启动**，避免把上游 ASR 错误继续传播到翻译和 TTS。

训练本身不是崩溃或 NaN 失败：381/381 iteration、3 个严格全局 shuffle coverage epoch、48,768 个 consumed samples 全部完成，skipped/NaN iteration 均为 0。失败发生在训练后的真实 free-running 内容质量，而不是 teacher-forced loss。

最终正式诊断覆盖 334 个固定 validation 样本、4 个 chunk，共 1,336 次评估：

| 指标 | 结果 |
|---|---:|
| streaming ASR 加权 WER/CER | **27.1432%** |
| causal-full ASR 加权 WER/CER | **14.4546%** |
| teacher-forced token accuracy | 92.0737% |
| CTC blank ratio | 87.2990% |
| AR 空文本行 | 0 / 1,336 |
| streaming final-only 行 | 0 / 972 |
| streaming pre-final 有内容行 | 972 / 972 |
| CTC 样本级全 blank 行 | **15 / 1,336** |
| AR event 未正常 stop 行 | **27 / 1,336** |

最关键的判断是：模型已经学会 streaming event grammar，也确实在 source EOS 前输出内容，但输出内容仍不够准确，尤其英文 streaming WER 高达 35.34%。这不能作为 Stage B 的可靠 committed source text。

## 2. 运行身份与隔离

| 项目 | 值 |
|---|---|
| 实验 | `uniss_phase3_v4_quality_first_true_streaming_pilot15_v1` |
| formal run | `stage_a_formal8_20260816T224100Z` |
| 框架 | native Megatron，单机 8×H200 |
| 初始化 | Phase3 v4 native `iter_0009075` 权重，重新建立 Stage A optimizer/scheduler |
| 最终 iteration | `381` |
| 最终 checkpoint | `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381` |
| 最终 HF 导出 | `checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf` |
| TensorBoard | `http://10.1.6.203:6101/` |
| 训练日志 | `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/train.log` |
| GPU 日志 | `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/train.gpu.csv` |
| 正式 diagnosis | `reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_checkpoint_diagnosis/iter381_formal334_4chunk_20260816T232000Z/diagnosis.json` |
| diagnosis SHA256 | `5c06ac98f76a30dd7a24dd2c5da787fe8c48c2128a934d991bdfec0e851eed43` |
| evaluator commit | `f61219fefc4d689a5a5004d2232b579115bcb850` |

全部新 checkpoint、日志、诊断和 gate 文件都位于本实验专属路径；没有覆盖历史 Phase1/2/3、streaming、GRPO、CVSS-T 或 Gradio 结果。

## 3. 数据、shuffle 与训练几何

Stage A 使用固定 15-shard 数据的版本化 pack：

| 项目 | 数值 |
|---|---:|
| source records | 1,325,243 |
| train packs | 16,195 |
| streaming ASR acoustics | 794,291 |
| causal-full ASR acoustics | 266,161 |
| offline ASR replay samples | 198,675 |
| Phase3 replay samples | 66,116 |
| sequence length | 18,000 |
| pack fill ratio | 99.0793% |
| global shuffle seed | 20,260,816 |
| coverage epochs | 3 |
| samples / coverage epoch | 16,256 |
| total consumed samples | 48,768 |

训练 sampler 按 coverage epoch 生成严格全局 permutation；不是按 shard 顺序读取，也不是每个 rank 各自局部 shuffle。正式 geometry：

```text
TP / PP             = 1 / 1
micro batch         = 1
global batch        = 128
sequence length     = 18000
precision           = BF16
train iterations    = 381
warmup iterations   = 12
save interval       = 100
validation interval = 50
validation          = 21 iters × 8 samples = 168 effective samples
```

multi-chunk curriculum 是一个连续 run：

| coverage 进度 | 主 chunk |
|---|---|
| 0%–10% | 1280 ms |
| 10%–30% | 1280 / 960 ms |
| 30%–60% | 960 / 640 ms |
| 60%–85% | 640 / 320 ms |
| 85%–100% | 320 / 160 ms |

## 4. 实际训练 objective

代码声明的权重为：

```text
1.00 × AR-ASR CE
0.30 × source CTC
0.20 × same-prefix offline teacher KL
0.10 × hidden multi-chunk consistency
0.10 × cache/full consistency
1.00 × offline ASR replay CE
1.00 × exact Phase3 replay CE
```

但本次日志暴露了两个必须明确记录的实现事实：

1. `offline_teacher_kl` 从头到尾为 `0`。当前 Stage A dataset/collator 没有提供 `teacher_batch / teacher_positions / teacher_indices / teacher_probabilities / teacher_mask`，所以声明的 `0.20 × KL` 实际未参与训练。这不是“KL 已经优化到零”，而是 denominator 不存在时返回 anchored zero。
2. `cache_full_consistency` 从头到尾为 `0`。当前实现有意把它作为 external parity gate，而不是每一步重复计算；因此最终 checkpoint 必须另跑 checkpoint-level cached/full parity 才能关闭该门。本次只完成自由运行内容诊断，尚未完成该 checkpoint-level runtime gate。

有效训练信号主要来自 AR-ASR、source CTC、hidden multi-chunk consistency、offline ASR replay 和 Phase3 replay。hidden consistency 在主 chunk 与 reference chunk 相同时按设计为零，在两个 chunk 不同时为非零。

## 5. 训练与 validation 曲线

| iter | val AR-ASR | val source CTC | val offline ASR replay | val Phase3 replay | val CTC blank ratio |
|---:|---:|---:|---:|---:|---:|
| 50 | 2.1791 | 5.4414 | 0.3045 | 3.9521 | 97.9156% |
| 100 | 1.3869 | 3.8448 | 0.2939 | 3.9197 | 99.9748% |
| 150 | 0.4785 | 3.3418 | 0.3041 | 3.9319 | 99.6404% |
| 200 | 0.4069 | 2.7721 | 0.2934 | 3.8995 | 94.0684% |
| 250 | **0.3653** | 2.8175 | 0.2867 | 3.9300 | 91.3864% |
| 300 | 0.3708 | **2.5017** | 0.2890 | 3.9265 | 88.0923% |
| 350 | 0.3901 | 2.5746 | **0.2776** | 3.9587 | **87.9190%** |
| 381 | 0.3807 | 2.6206 | 0.2828 | 3.9372 | 89.2493% |

正向结果：

- AR-ASR validation CE 从 2.179 降到约 0.38；
- source CTC 从 5.44 降到约 2.62，早期接近 100% blank 后恢复；
- offline ASR replay 保持约 0.28–0.30；
- Phase3 replay 保持约 3.90–3.96，没有出现明显灾难性漂移；
- 0 skipped、0 NaN，最终 grad norm 2.313。

不足：teacher-forced CE 已很低，但 free-running streaming 错误仍高，说明主要问题已经从“不会拟合标签”转为 exposure error、event 增量拼接和短 chunk 内容鲁棒性。

## 6. GPU 与耗时

训练从 `2026-08-16 22:18:00 UTC` 开始，到 `23:05:20 UTC` 完成，纯训练 wall time 约 47 分 20 秒。按显存大于 10 GiB 的 active 监控点统计：

| 指标 | 数值 |
|---|---:|
| active GPU utility mean | 65.36% |
| utility ≥95% 监控点 | 27.84% |
| active power mean | 355.42 W |
| observed power max | 462.30 W |
| max memory | 92,862 MiB |

训练吞吐在稳定区间通常为约 110–150 TFLOP/s/GPU。没有达到 700 W 的原因是 compound Whisper+Qwen 路径包含变长 pack、CPU/audio dispatch、交替 consistency second-forward 和 validation/save 间隙；不能用无关 synthetic load 抬高功率来伪造训练利用率。

正式 8-GPU free-running evaluation 从约 `23:16 UTC` 到 `23:22 UTC`，约 6 分钟。评估逐 event 自回归生成，每卡单 worker，GPU 约 12%–20%、约 120–140 W；这是解码形态，不应与 Megatron 训练功率直接比较。

## 7. 正式 free-running 评估协议

正式 validation pack 有 167 个 pack。与训练/validation loader 一致，每个 pack 通过 deterministic rotated selection 最多取 2 个 acoustics：

```text
334 unique samples
  243 streaming_asr
   91 causal_full_asr
× 4 chunks: 160 / 320 / 640 / 1280 ms
= 1,336 evaluations
```

8 个 worker 使用 `occurrence % 8` 做不相交 partition；merge 时验证 worker 0–7 完整、`task + sample_id + chunk_ms` 无重复。之前误把全部 sidecar 当 formal validation 的 `iter381_full_validation_4chunk_20260816T231000Z` 已停止且不参与本报告。

每条评估同时执行：

- causal Whisper acoustic forward；
- CTC greedy collapse；
- Qwen teacher-forced token accuracy；
- 对每个 WRITE event 的 free-running AR-ASR；
- 英文 event-boundary-safe joining；
- 英文 WER、中文 CER；
- event stop、空输出、sample-level CTC blank 和 pre-final content 检查。

## 8. 分任务、语言结果

| 任务 | 语言 | evaluations | WER/CER | teacher accuracy | CTC blank ratio |
|---|---|---:|---:|---:|---:|
| causal-full | 中文 | 144 | 15.8787% | 83.7957% | 96.7071% |
| causal-full | 英文 | 220 | 12.7760% | 89.5572% | 79.8996% |
| streaming | 中文 | 456 | 21.0112% | 94.4558% | 96.8011% |
| streaming | 英文 | 516 | **35.3399%** | 90.8995% | 80.0457% |

streaming 与 causal-full 的加权差距为 12.69 个绝对百分点。相同 causal acoustic encoder 在整句一次生成时明显好于逐 WRITE event 拼接，说明当前最大损失不只来自声学前端，也来自增量解码/commit 训练与 free-running 历史不匹配。

## 9. 分 chunk 结果

### 9.1 Streaming ASR

| chunk | 中文 CER | 英文 WER | 中文 teacher acc. | 英文 teacher acc. |
|---:|---:|---:|---:|---:|
| 160 ms | 23.1910% | **38.2436%** | 93.8703% | 90.2987% |
| 320 ms | 22.1011% | 35.1275% | 94.4405% | 90.9563% |
| 640 ms | 19.5883% | 34.2776% | 94.6645% | 91.1187% |
| 1280 ms | **19.1644%** | **33.7110%** | 94.8478% | 91.2242% |

### 9.2 Causal-full ASR

| chunk | 中文 CER | 英文 WER |
|---:|---:|---:|
| 160 ms | 16.7707% | 15.0368% |
| 320 ms | 15.7003% | 12.4080% |
| 640 ms | **14.7190%** | 12.1977% |
| 1280 ms | 16.3247% | **11.4616%** |

更大的 chunk 总体更好，尤其 streaming 英文从 160 ms 的 38.24% 改善到 1280 ms 的 33.71%，但改善幅度不足以接近 offline 门。这说明短 chunk curriculum 产生了可用信号，却没有消除内容退化。

## 10. Streaming event 与 collapse 健康度

| 检查 | 结果 | 判定 |
|---|---:|---|
| AR 空文本 | 0 / 1,336 | 通过 |
| streaming final-only | 0 / 972 | 通过 |
| streaming pre-final 有内容 | 972 / 972 | 通过 |
| mean WRITE structure rate | 99.7704% | 接近通过，但非 100% |
| all events reached stop | 97.9790% | **未通过**；27 行失败 |
| CTC 样本级全 blank | 15 / 1,336 | **未通过** |

需要区分两个结论：

1. CTC 不是全局塌缩；绝大多数样本仍有 nonblank token。
2. 硬门要求 sample-level collapse 为 0，而不是只要求“并非所有样本都 blank”。15 个全 blank 行全部来自中文 streaming 分支，所以该硬门仍失败。

## 11. 与 offline Phase3 anchor 对比

Stage00 固定 pilot15 Quality ASR anchor：

| 语言 | offline | 允许上限（相对 +15%） | Stage A streaming | 相对退化 | 结论 |
|---|---:|---:|---:|---:|---|
| 中文 CER | 5.4661% | 6.2860% | 21.0112% | +284.39% | **失败** |
| 英文 WER | 5.6072% | 6.4483% | 35.3399% | +530.26% | **失败** |

即使只看 causal-full，中文 15.8787%、英文 12.7760% 也分别相对 offline 退化约 190.49% 和 127.85%，仍不达门。

协议限制必须如实说明：Stage00 offline anchor 与本次 Stage A 334 条 formal selection 都来自固定 pilot15，但样本 ID 不完全相同。因此上表足以说明当前差距很大，却不能替代严格的 matching-sample offline rerun。严格 pass 仍要求对同一 334 条源音频运行 Phase3 offline ASR；该缺失本身也阻止创建通过 gate。

## 12. iter 200 / 300 / 381 趋势

iter200 和 iter300 之前只在 4 个固定英文样本、960/1280 ms 上做快速诊断：

| checkpoint | fixed4 weighted error | teacher accuracy |
|---|---:|---:|
| iter200 | 43.9024% | 89.7351% |
| iter300 | 42.6829% | 90.3974% |

iter381 的正式 334-sample 结果不能直接与 fixed4 数字等价比较。只取 formal selection 中确实重叠的两个 1280 ms 样本：

| checkpoint | edits / units | error |
|---|---:|---:|
| iter200 | 7 / 17 | 41.1765% |
| iter300 | 8 / 17 | 47.0588% |
| iter381 | 6 / 17 | 35.2941% |

这支持 iter381 比早期 checkpoint 有局部改善，但样本过少，不能把它解读为已通过全量 checkpoint selection。当前没有任何 checkpoint 获得 `SELECTED_CHECKPOINT.json`。

## 13. 失败原因分析

### 13.1 Teacher forcing 与 free running 的明显间隙

teacher-forced token accuracy 已达到 92.07%，但 streaming WER/CER 仍为 27.14%。teacher forcing 每一步看到正确历史；真实 streaming event generation 看到自己的历史。一处 event 边界、单词或 stop token 错误会进入后续上下文并累计。因此低 CE/高 teacher accuracy 不能替代 free-running 质量门。

### 13.2 Streaming event 路径是主要新增损失

causal-full 为 14.45%，streaming 为 27.14%。两者使用同一 checkpoint 和 causal acoustic representation，差值说明逐 event transcript delta、英文跨 event 空格/词边界、早期 commit 内容以及生成历史误差仍是主要问题。

### 13.3 计划中的 same-prefix teacher KL 实际未启用

数据 loader 没有提供 teacher top-k posterior 字段，导致 `offline_teacher_kl` denominator 始终为 0。模型缺少原计划用于约束 causal student posterior 的 same-prefix teacher 信号，这是本次 formal run 与计划之间最重要的实现缺口之一。

### 13.4 Checkpoint-level cached runtime 门未关闭

`cache_full_consistency` 在训练内部是 anchored zero，依赖 external parity gate。Stage00 已证明初始化前端基础设施的因果性，但 Stage A 已更新前端/Qwen 参数，最终 checkpoint 仍需要单独验证 cached/full parity、future perturbation 和 committed rollback。本次自由运行 evaluator 没有直接测量这三项。

### 13.5 CTC 辅助对齐仍偏 blank

整体 blank ratio 87.30%，中文 streaming 约 96.80%，并有 15 个 sample-level 全 blank 行。CTC loss 已下降，但它还没有形成足够稳定的短 chunk 对齐辅助信号。

### 13.6 不是“再机械增加 epoch”就能保证解决

AR-ASR validation CE 在 iter250 附近已到最低，iter250–381 基本平台化；source CTC 在 iter300 后也没有继续单调改善。直接继续同一 objective 更多 epoch，可能继续降低 teacher-forced loss，却未必缩小 free-running gap。应先修复 inactive loss 和 rollout 训练分布，再重训。

## 14. Gate 判定

| 硬门 | 状态 | 证据 |
|---|---|---|
| causal frontend 基础设施 | Stage00 通过 | real PCM future perturbation、cached reference parity 已通过 |
| matching offline 相对退化 ≤15% | **失败/未完整证明** | 当前中文 21.01%、英文 35.34%；同 ID offline 尚未重跑 |
| pre-final source commit | 通过 | 972/972 streaming evaluations 在最后 event 前已有内容 |
| AR empty/final-only | 通过 | 0 empty、0 final-only |
| sample-level CTC all blank = 0 | **失败** | 15 行 |
| every event stop | **失败** | 27 行至少一个 event 未 stop |
| checkpoint cached/full parity | **未评估** | external checkpoint gate 尚未执行 |
| committed rollback = 0 | **未评估** | 当前 evaluator 不执行持久化 cached commit/rollback |

最终 gate：`passed=false`。

## 15. 下一版修复顺序

下一次 Stage A 重训前按以下顺序修复；不要先启动 Stage B：

1. **建立同 334 样本的 Phase3 offline ASR baseline。** 固定同一音频、normalization、WER/CER 和 SHA256，消除比较集不匹配。
2. **真正接入 same-prefix teacher posterior。** 预计算或在线构造当前 prefix 可支持位置的 top-k teacher distribution；dataset/collator 必须携带五个 teacher 字段。若一个 validation interval 内 denominator 始终为 0，训练应 fail-fast。
3. **增加 checkpoint-level cached runtime gate。** 对 final checkpoint 测 full-causal/cached parity、future perturbation、cache growth、commit rollback；不得复用只验证初始化权重的 Stage00 结论。
4. **用 free-running event rollout 训练，而不只 teacher forcing。** 以受控 scheduled sampling/DAgger 风格把模型自己的前一个 transcript delta 放回下一 event context，同时保留 Phase3/offline replay 防止内容退化。
5. **短 chunk 与英文 streaming 定向补强。** 在不改变 validation 分布的前提下，训练 sampler 增加 160/320 ms、英文跨 event word-boundary 和长 event-count 样本；分别报告语言、chunk 和 event-count strata。
6. **修复 sample-level CTC blank。** 先检查中文 byte target/frame feasibility 与 blank bias；采用受审计的 blank-logit regularization或更均衡的 CTC sample weighting，但不能通过删除失败样本来“过门”。
7. 修复后重新做 15-shard Stage A formal run；只有所有内容、collapse、causality、rollback 门同时通过，才创建 selected artifact 并进入 Stage B。

## 16. 证据文件

- `RUN_MANIFEST.json`：正式训练身份和 geometry；
- `STAGE_A_FINAL_SUMMARY.json`：可复现的分组统计；
- `GATE_FAILED.json`：失败门、阻塞的下一阶段和所有输入 SHA256；
- `diagnosis.json`：1,336 条逐样本诊断；
- `diagnosis.md`：逐样本可读表；
- `parts/part_00.json` 到 `part_07.json`：8-GPU disjoint worker 原始结果；
- Stage00 baseline：`eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_phase3_offline_20260816T031129Z/baseline_summary.json`。
