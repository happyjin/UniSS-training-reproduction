# 字段口径对照

## deepspeed 链路（`global_step/max_steps` 字段存在）

HF Trainer logging 一条记录典型字段：

```
{'loss': 17.01062775, 'grad_norm': 540.50186182, 'learning_rate': 4e-08,
 'eos_boundary_top1_acc': 0.01538086, 'premature_eos_top1_rate': 0.00998205,
 'token_acc': 0.00209844, 'token_acoustic_token': 0.00438948,
 'token_text_acc': 0.7252187, 'epoch': 0.0,
 'global_step/max_steps': '1/9364', 'elapsed_time': '3m 36s',
 'remaining_time': '23d 9h 7m 9s', 'memory(GiB)': 66.67,
 'train_speed(s/it)': 215.745963}
```

- `token_acc` / `token_text_acc` / `token_acoustic_token` 是 **deepspeed 链路专属**字段。
- `elapsed_time` / `remaining_time` 是字符串（`"1h 45m 18s"` / `"3d 8h 24m 51s"` 这种格式），
  直接照抄进表格即可，不需要重新格式化。
- `memory(GiB)` 是训练循环里某一时刻的显存快照（不是峰值，是打点时刻的值）。
- `train_speed(s/it)` 是 **HF Trainer 自己维护的累计均值**（从训练开始到当前 step 的平均
  每步耗时，含第 1 步 warmup 拖累），**不等于**瞬时/稳态 step 耗时——稳态瞬时值只能从
  `mfu_log.jsonl` 里算。

## megatron 链路（`iteration` 字段存在）

```
{'loss': 16.64839554, 'codec_loss': 16.63004494, 'codec_token_acc': 0.00249266,
 'codec_residual_loss': 8.4946928, 'eos_boundary_top1_acc': 0.01831055,
 'premature_eos_top1_rate': 0.01013158, 'grad_norm': 470.6991272,
 'learning_rate': 4e-08, 'iteration': '1/9363', 'elapsed_time': '2m 26s',
 'remaining_time': '15d 18h 37m 11s', 'memory(GiB)': 75.83,
 'train_speed(s/it)': 145.591888}
```

- `codec_loss` / `codec_token_acc` / `codec_residual_loss` 是 **megatron 链路专属**字段，
  **无** `token_text_acc` / `token_acoustic_token`。
- `codec_token_acc` 语义上接近 deepspeed 链路的 `token_acc`，但**不是同一套 loss 计算实现**，
  写文档时填入 `token_acc` 那一行要显式标注"非严格同口径，仅供趋势参考"，不要暗示两者可
  精确对比。
- megatron 侧总步数（`iteration` 分母）常比 deepspeed 侧（`global_step/max_steps` 分母）
  少 1（如 9363 vs 9364），这是两套框架对"2 epochs 派生总步数"的取整方式不同，属预期口径
  差异，不是 bug。

## mfu_log.jsonl（默认不采集/不使用，见 SKILL.md §0；仅用户明确要求时才用）

每行一个 JSON object：

```json
{"step": 1, "tokens": 349647, "step_sec": 122.768052, "dp": 16,
 "params": 1933463808, "flop_factor": 8.0, "tflops": 2.753, "mfu": 0.278,
 "tokens_per_sec": 2848.029}
```

- `step_sec` 是**这一步的瞬时耗时**（跟 HF Trainer 的累计 `train_speed(s/it)` 不同）。
- `mfu` 已经是百分数（`0.278` 表示 0.278%）。
- 文件是**追加写入**的：如果同一个 `OUTPUT_DIR` 被多次提交，文件里会有多段"从 step 1
  开始"的历史，`parse_metrics.py` 已自动处理（只取最后一次 `step==1` 之后的记录）。
- **默认策略（对齐 `training-optimization` skill §0）：不拉取、不解析、不填写这份数据**。
  MFU 依赖的理论 FLOPs 分母是项目自己手算/估算的公式，容易在具体实现里悄悄算错而不
  报错，比 s/it 这种直接测量的墙钟耗时更容易误导结论。表格里稳态 MFU/TFLOPs/
  tokens_per_sec 单元格**统一填"不采用"**（不是"未采集"——即使拉到了数据也不填，除非
  用户明确要求）。不要拿 HF Trainer 的累计 `train_speed(s/it)` 硬凑成"稳态值"填进去。

## 百分比字段格式化

以下字段是 0~1 的小数，写入 Markdown 表格前需要 `×100` 并格式化成 `"NN.NN%"`：

`eos_boundary_top1_acc` / `premature_eos_top1_rate` / `token_acc` / `token_acoustic_token` /
`token_text_acc` / `codec_token_acc`

`loss` / `codec_loss` / `codec_residual_loss` / `grad_norm` 保留 2-5 位小数（跟随文档里
已有列的小数位数保持一致，不要突然改变精度）。`learning_rate` 用科学计数法
`f"{lr:.3e}"`（如 `4.000e-08`）。
