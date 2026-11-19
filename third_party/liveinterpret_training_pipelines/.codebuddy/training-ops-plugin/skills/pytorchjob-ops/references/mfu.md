# 估算训练 MFU（Model FLOPs Utilization）

> 触发关键词：`MFU` `mfu` `算力利用率` `FLOPs 利用率` `吞吐` `TFLOP` `喂数瓶颈` `data-starved` `GPU 利用率低`

用于评估 PyTorchJob 训练是否吃满 GPU 算力。**MFU = 实际达到的每卡 FLOP/s ÷ 单卡理论峰值 FLOP/s**，是判断"训练效率高不高、是不是被 dataloader 卡住"的核心指标。

---

## 0. 为什么需要本方法（关键背景）

ms-swift 封装的 Megatron 训练日志里**没有** Megatron 原生的 `elapsed time per iteration (ms):` / TFLOP 字段——进度只走 tqdm + wandb 上报，stdout 里逐 step 耗时通常拿不到。

因此本方法**不依赖逐 step 日志**，而是用**带时间戳的 checkpoint 保存日志**反推每步平均耗时：

```
2026-07-30T11:43:46Z ... Successfully saved Megatron model weights in .../checkpoint-500
2026-07-30T11:53:23Z ... Successfully saved Megatron model weights in .../checkpoint-1000
```

相邻两条相差 500 step、耗时 577s → **1.15 s/step**（含 ckpt 保存 + 周期 eval 开销，属耗时上界）。

---

## 1. 计算公式

| 量 | 公式 |
|---|---|
| `step_sec` | Δt(相邻 checkpoint 保存) / `SAVE_STEPS` |
| `tokens_per_step` | `GLOBAL_BATCH_SIZE × SEQ_LEN` |
| `flops_per_step` | `FACTOR × PARAM_COUNT × tokens_per_step`（`FACTOR=6`：前向+反向） |
| `DP`（数据并行度） | `TOTAL_GPUS / (TP × PP × CP)` |
| `per_gpu_flops_s` | `flops_per_step / step_sec / DP` |
| **`MFU`** | `per_gpu_flops_s / GPU_PEAK_FLOPS` |

**关键常量**：
- `FACTOR=6`：dense Transformer 前向 2N + 反向 4N 的经验系数（N=参数量）。
- `GPU_PEAK_FLOPS`：H200 bf16 稠密 ≈ `989 TFLOP/s`（A100 bf16 ≈ 312、H100 bf16 ≈ 989）。
- `PARAM_COUNT`：dense 用总参数；MoE 用**激活**参数量。

**⚠️ 上界说明**：`6N` 假设全参前反向。若训练 `freeze` 了主干（如只训 bridge/talker/aligner），真实可训练 FLOPs 更小，**实际 MFU 会低于本方法的上界值**。因此本方法输出的是 **MFU 上界**，主要用于判断喂数瓶颈：**若实测 MFU 远低于上界 + GPU 利用率波动大 + 显存占用低 → 基本可判定 GPU data-starved（dataloader 喂数慢）**。

---

## 2. 执行流程

### Step 1 — 直接用脚本（推荐）

```bash
K8S_NS=<NAMESPACE> bash <skill>/scripts/estimate_mfu.sh <JOB_NAME> <NAMESPACE>
```

脚本会自动：
1. 拉带 `--timestamps` 的 master 日志。
2. 从日志解析 `SEQ_LEN`(max_length)、`TP/PP/CP`、`TOTAL_GPUS`(pod数×每pod GPU) → 算 `DP`。
3. 从 `checkpoint-N` 标记推断 `SAVE_STEPS`，从相邻 `Successfully saved Megatron` 时间戳算 `step_sec`。
4. 计算并输出结构化 KEY=VALUE：`STEP_SEC / TOKENS_PER_STEP / PER_GPU_TFLOPS / MFU_UPPER_BOUND_PCT / SAMPLES_PER_SEC / DP`。

### Step 2 — 按需覆盖参数

日志解析不到或需精确值时，用环境变量覆盖（脚本内有完整列表）：

```bash
GLOBAL_BATCH_SIZE=1024 PARAM_COUNT=3000000000 GPU_PEAK_FLOPS=989000000000000 \
TP=1 TOTAL_GPUS=16 \
bash <skill>/scripts/estimate_mfu.sh <JOB_NAME> <NAMESPACE>
```

- `GLOBAL_BATCH_SIZE`：**最关键**，务必用真实生效值（= MBS × DP × grad_accum）。
- `STEP_SEC_OVERRIDE`：若已从别处拿到更精确的纯训练 step 时间，可直接覆盖。
- `SAVE_STEPS`：checkpoint 间隔步数（默认从日志推断，常见 = `save_steps`）。

### Step 3 — 手工核算（脚本不可用时）

```bash
# 1) 取相邻两次 checkpoint 保存时间戳
kubectl -n <NS> logs <JOB>-master-0 -c pytorch --timestamps | grep 'Successfully saved Megatron' | tail -6
# 2) Δt / SAVE_STEPS = step_sec
# 3) MFU = 6 × N × (GBS × seq) / step_sec / DP / GPU_PEAK
```

---

## 3. 输出模板

```markdown
## MFU 估算 — <JOB_NAME>

| 项 | 值 |
|---|---|
| **每步耗时** | <STEP_SEC> s/step（含 ckpt/eval，上界） |
| **有效 batch** | GBS=<GBS>，seq=<SEQ_LEN> → <TOKENS_PER_STEP> tokens/step |
| **并行** | TP=<TP> PP=<PP> CP=<CP> → DP=<DP>（<TOTAL_GPUS> GPU） |
| **单卡算力** | <PER_GPU_TFLOPS> TFLOP/s（峰值 <PEAK> TFLOP/s） |
| **MFU（上界）** | **<MFU_UPPER_BOUND_PCT>%** |
| **吞吐** | <SAMPLES_PER_SEC> samples/s |

结论：<data-starved 判定 / 是否需优化 dataloader 或 batch>
```

---

## 4. 结果研判

| 现象 | 判断 | 处置 |
|---|---|---|
| MFU 上界 < 30% + GPU 利用率波动大 + 显存占用低（<30%） | **GPU data-starved**（喂数瓶颈） | 加大 `DATALOADER_NUM_WORKERS` / `PREFETCH_FACTOR`；增大 `MICRO_BATCH_SIZE`；检查数据 IO（S3/NVMe） |
| MFU 40%~55% | dense bf16 训练的健康区间 | 一般无需干预 |
| MFU 上界很高但显存也满 | 已算力受限 | 只能靠并行/kernel/fp8 等，不是喂数问题 |

> H200/H100 bf16 dense SFT 常见 MFU 落在 **35%~55%**；freeze 主干的 bridge-alignment 由于可训练 FLOPs 少，"上界 MFU" 本身就偏低，需结合 GPU 利用率曲线一起看。

---

## 5. 注意事项

1. **只读操作**，不改动任务。
2. 至少需 **2 条** checkpoint 保存日志才能反推 step_sec；任务刚起、还没存过 2 次 checkpoint 时会提示数据不足。
3. `step_sec` 由 checkpoint 间隔算得，**包含 ckpt 保存 + 周期 eval 开销**，因此是耗时上界（→ MFU 下界偏保守）；要纯训练 MFU 可用 `STEP_SEC_OVERRIDE` 传入更精确值。
4. `GLOBAL_BATCH_SIZE` 必须是**真实生效值**，不要用脚本默认；改过 batch 配置后务必核对。
5. MoE 模型用**激活参数量**作 `PARAM_COUNT`，否则会高估。
