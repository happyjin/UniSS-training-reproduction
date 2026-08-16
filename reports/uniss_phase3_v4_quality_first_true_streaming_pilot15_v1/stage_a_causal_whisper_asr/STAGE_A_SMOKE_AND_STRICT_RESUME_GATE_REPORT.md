# Stage A Smoke 与 Strict-Resume Gate 报告

日期：2026-08-16（UTC）

## 结论

Stage A 原生 Megatron 训练入口已通过进入 full-pack/full-train 前的工程 Gate：

- Phase3 v4 Megatron checkpoint 能以受审计的非严格 handoff 加载新增 Stage A 模块；
- 8 卡 smoke 完成 32/32 iterations，训练、validation 和 checkpoint 均正常；
- `raise_all` strict resume 能从 iteration 5 恢复模型、optimizer、scheduler、RNG 和 rerun state；
- 恢复后的 iteration 6 从 consumed samples 96 继续，首个单步 loss 与未中断 run 完全相同；
- TensorBoard 原始单步 scalar 证明 curriculum 在 iteration 6/7/8/9 分别为
  `960/1280/960/1280 ms`，不存在 1120 ms 的真实训练档位；
- skipped iterations 为 0，NaN iterations 为 0；
- 终止 codec 边界和 8 卡 validation padding 两个已知阻塞均已修复。

Gate 状态：**PASSED**。

## 1. 使用的 run

### 1.1 完整 8 卡 smoke

- Run：`stage_a_smoke8_20260816T210719Z_valfix`
- Log：`logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_smoke/stage_a_smoke8_20260816T210719Z_valfix/train.log`
- Checkpoint：`checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_smoke/stage_a_smoke8_20260816T210719Z_valfix`
- TensorBoard：`runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_smoke/stage_a_smoke8_20260816T210719Z_valfix/tensorboard`

结果：

- 32/32 iterations 完成；
- iteration 10/20/30/32 validation 全部完成；
- iteration 5/10/15/20/25/30/32 checkpoint 全部保存；
- AR-ASR：约 `8.49 -> 2.34`；
- source CTC：约 `19.10 -> 11.71`；
- Phase3 replay 保持约 `4.1–4.2`；
- 最终 validation：AR-ASR `2.8345`、source CTC `12.8838`、offline ASR replay `0.3057`、Phase3 replay `4.1779`。

### 1.2 strict resume

- Run：`stage_a_strict_resume8_20260816T211131Z_from_step5`
- Load：`stage_a_smoke8_20260816T210058Z_boundaryfix/iter_0000005`
- Log：`logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_strict_resume/stage_a_strict_resume8_20260816T211131Z_from_step5/train.log`
- Checkpoint：`checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_strict_resume/stage_a_strict_resume8_20260816T211131Z_from_step5`
- TensorBoard：`runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_strict_resume/stage_a_strict_resume8_20260816T211131Z_from_step5/tensorboard`

恢复证据：

- `--dist-ckpt-strictness raise_all`；
- checkpoint iteration 5 成功载入；
- optimizer/scheduler checkpoint 值成功载入；
- 未使用 finetune、`--no-load-optim` 或 `--no-load-rng`；
- iteration 6 consumed samples 为 96；
- iteration 32 checkpoint 与最终 validation 均成功。

## 2. 连续性对比

以下数据取自 TensorBoard 的原始逐 iteration scalar，而不是控制台区间平均值。

| Iteration | 指标 | 未中断 run | strict resume | 相对差异 |
|---:|---|---:|---:|---:|
| 6 | curriculum progress | 0.15625 | 0.15625 | 0% |
| 6 | chunk ms | 960 | 960 | 0% |
| 6 | AR-ASR | 6.118981 | 6.118981 | 0% |
| 6 | source CTC | 17.855675 | 17.855675 | 0% |
| 6 | Phase3 replay | 4.214127 | 4.214127 | 0% |
| 7 | curriculum progress | 0.18750 | 0.18750 | 0% |
| 7 | chunk ms | 1280 | 1280 | 0% |
| 7 | AR-ASR | 5.767034 | 5.767386 | 0.0061% |
| 7 | source CTC | 17.761440 | 17.761789 | 0.0020% |
| 7 | Phase3 replay | 4.180301 | 4.179262 | 0.0249% |
| 8 | curriculum progress | 0.21875 | 0.21875 | 0% |
| 8 | chunk ms | 960 | 960 | 0% |
| 9 | curriculum progress | 0.25000 | 0.25000 | 0% |
| 9 | chunk ms | 1280 | 1280 | 0% |

首个恢复 step 完全一致；后续 BF16/dropout 训练的微小数值漂移处于正常范围。上述对比区间内：

- AR-ASR 最大相对差异约 `0.0533%`；
- source CTC 最大相对差异约 `0.0020%`；
- Phase3 replay 最大相对差异约 `0.0258%`；
- offline ASR replay 最大相对差异约 `0.4727%`。

## 3. 1120 ms 控制台值的取证结论

strict resume 控制台的 iteration 7 曾显示：

- curriculum progress `0.171875`；
- chunk `1120 ms`。

这不是模型在同一个 optimizer update 中混用了两个 curriculum 档位。Megatron 的 `training_log()` 在当前进程的第一个 iteration 会打印但不清空统计累加器（`should_reset = not is_first_iteration`），因此第二行控制台日志是前两个 iterations 的均值：

- `(0.15625 + 0.18750) / 2 = 0.171875`；
- `(960 + 1280) / 2 = 1120 ms`；
- `(6.118981 + 5.767386) / 2 = 5.943183`，与控制台 AR-ASR 完全对应。

TensorBoard 在写入时直接使用当前 `loss_dict`，所以其 iteration 7 原始 scalar 正确记录为 progress `0.1875`、chunk `1280 ms`、AR-ASR `5.767386`。因此正式代码保留原生 Megatron 的 consumed-sample curriculum 定位，没有增加逐 microbatch broadcast 或其他额外同步。

## 4. 已修复的阻塞

### 4.1 codec 终止边界

当 PCM 长度恰好为 80 ms 整数倍且 offline GLM 恰好多一个终止 token 时，使用最后一个已可见 causal state 填充该终止槽；其他长度差异仍立即报错。新增诊断为 `causal_glm_terminal_extensions`。

### 4.2 8 卡 validation batch 对齐

`PaddedStageAValidationDataset` 将 validation 循环 padding 到完整 DP microbatch，同时保证每条原始 validation pack 至少覆盖一次，避免小验证集在 DP=8 时 active samples 为 0。

## 5. 自动化验证

最终代码状态：

```text
44 passed
```

正式 full training 前仍须完成：

1. 构建 seq=18000 的正式 train/valid packs；
2. 按 pack count 计算三轮严格覆盖的 `train_iters`；
3. 生成独立 formal run manifest；
4. 从 Phase3 v4 iteration 9075 做首次 non-strict handoff；
5. 后续中断恢复一律使用 strict resume。

