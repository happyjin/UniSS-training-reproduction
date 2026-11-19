# 场景 B — 查询训练任务进度

> 触发关键词：`进度` `progress` `step` `loss` `token_acc` `还要多久` `ETA` `跑到哪` `跑完了吗`

---

## 流程总览

本场景只做一件事：调用 `query_progress.sh` 获取结构化进度数据，然后分析并汇报给用户。所有 `anyscale` CLI 调用和日志解析都由脚本完成，本文件只描述智能分析逻辑。

```
用户输入
  → Step 1: 提取 PRODJOB_ID
  → Step 2: 调用 query_progress.sh 获取 KEY=VALUE 数据
  → Step 3: 分析输出，提取关键信息
  → Step 4: 按 JOB_STATE 分支处理
  → Step 5: 格式化输出，汇报给用户
```

---

## Step 1 — 提取 PRODJOB_ID

从用户输入中识别目标任务，支持两种格式：

- **完整 URL**：如 `https://console.anyscale.com/jobs/prodjob_abc123def456`，提取其中的 `prodjob_` 开头 ID
- **直接使用 ID**：用户直接提供 `prodjob_abc123def456`

识别规则：
- 找到 1 个 ID → 直接使用
- 找到多个 ID → 列出所有 ID，询问用户选择
- 未找到 → 提示用户提供 prodjob_id

---

## Step 2 — 调用进度查询脚本

运行以下命令获取进度数据：

```bash
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/query_progress.sh <PRODJOB_ID>
```

脚本会输出结构化的 KEY=VALUE 数据，核心字段如下：

| KEY | 含义 | 示例值 |
|---|---|---|
| `JOB_STATE` | 任务状态 | `RUNNING` / `FAILED` / `SUCCEEDED` 等 |
| `STEP_PROGRESS` | 当前步数进度 | `1200/10000 (12.0%)` |
| `STEP_TIME` | 单步耗时 | `0.85s/it` 或 `1.18it/s` |
| `LATEST_LOSS` | 最近几次 loss | `step 1198: 0.823 \| step 1199: 0.811 \| ...` |
| `TOKEN_ACC` | token 准确率 | `0.8745` |
| `GRAD_NORM` | 梯度范数 | `1.234` |
| `LR_LATEST` | 当前学习率 | `5.0e-05` |
| `SAVE_CHECKPOINTS` | 已保存的 checkpoint 列表 | `checkpoint-500,checkpoint-1000` |
| `HEALTH` | 健康状态 | `OK` / `WARN` / `FAIL` |
| `HEALTH_NOTES` | 健康状态备注 | `benign deprecation warnings present` |
| `FINAL_CHECKPOINT_S3` | 最终 checkpoint S3 路径（仅 SUCCEEDED） | `s3://...` |
| `LOG_LINES` | 日志总行数 | `15420` |

---

## Step 3 — 分析 KEY=VALUE 输出

拿到脚本输出后，按以下逻辑提取并分析关键信息。

### 3.1 解析 STEP_PROGRESS

从 `STEP_PROGRESS` 中解析出：
```
CURRENT_STEP = STEP_PROGRESS 中 "/" 前的数字
MAX_STEPS    = STEP_PROGRESS 中 "/" 后的数字（去除百分比部分）
PROGRESS_PCT = 括号内的百分比数值
```

若 `STEP_PROGRESS` 为 `N/A`，说明训练尚未开始输出 step 日志，属于 `STARTING` 或刚启动的正常状态。

### 3.2 解析 LATEST_LOSS

`LATEST_LOSS` 格式为 `step N: X | step N+1: Y | ...`，从中提取：
- **最新 loss 值**：最后一个 `step X: Y` 中的 Y
- **loss 趋势**：对比最近 3 条 loss，判断是否下降/发散/震荡

loss 健康判断：
- 正常：loss 整体呈下降趋势，或在小范围内波动
- 警告：loss 突然 spike（单步上涨超过 50%）
- 异常：loss 变为 NaN 或 INF（此时 `HEALTH` 应为 `FAIL`）

### 3.3 解析 STEP_TIME

`STEP_TIME` 可能有两种格式：
- `0.85s/it` → 每步 0.85 秒
- `1.18it/s` → 每秒 1.18 步，换算为 `1/1.18 ≈ 0.85s/it`

统一换算为**秒每步**，用于 ETA 计算。

step time 异常判断：
- 正常：step time 稳定，波动在 ±20% 以内
- 警告：step time 突然增长超过 50%（可能是数据加载瓶颈或通信异常）

---

## Step 4 — 按 JOB_STATE 分支处理

### JOB_STATE = STARTING

集群正在启动，日志可能极少或为空。

**汇报内容**：
- 任务状态：`STARTING（集群启动中）`
- 建议：等待 3-5 分钟后再次查询

### JOB_STATE = RUNNING

任务正在训练，需汇报进度 + ETA + 健康状态。

**汇报内容**（详见 Step 5 模板）：
- 当前 step / 总 step（进度百分比）
- 最新 loss + token_acc
- ETA（预计剩余时间）
- 健康状态摘要

### JOB_STATE = SUCCEEDED

任务已正常完成。

**汇报内容**：
- 任务状态：`SUCCEEDED（已完成）`
- 最终 step 数（从 `STEP_PROGRESS` 解析）
- 最终 loss / token_acc
- 最终 checkpoint S3 路径（`FINAL_CHECKPOINT_S3`）
- 已保存的 checkpoint 列表（`SAVE_CHECKPOINTS`）

### JOB_STATE = FAILED / OUT_OF_RETRIES / TERMINATED

任务异常退出，**停止进度查询，路由到诊断流程**：

> 读取 `.codebuddy/training-ops-plugin/skills/anyscale-job-ops/references/diagnose.md`，按照其中的诊断流程排查失败原因。

同时将以下信息传递给诊断流程：
- `PRODJOB_ID`
- `HEALTH_NOTES`（如果非空）
- `LOG_LINES` 数量（判断日志是否完整）

---

## Step 5 — 格式化输出

按以下模板整理并汇报给用户。

### RUNNING 状态报告模板

```
## 训练进度 — `<PRODJOB_ID>`

| 指标 | 当前值 |
|---|---|
| **状态** | 🟢 RUNNING |
| **Step** | <CURRENT_STEP> / <MAX_STEPS>（<PROGRESS_PCT>%） |
| **最新 Loss** | <LATEST_LOSS_VAL> |
| **token_acc** | <TOKEN_ACC> |
| **单步耗时** | <STEP_TIME> |
| **梯度范数** | <GRAD_NORM> |
| **学习率** | <LR_LATEST> |
| **健康状态** | <HEALTH> (<HEALTH_NOTES>) |
| **Checkpoints** | <SAVE_CHECKPOINTS> 或"暂无" |

**预计剩余时间**：约 <ETA_HOURS> 小时（剩 <REMAINING_STEPS> 步 × <STEP_TIME_SECONDS>s/步）
```

### SUCCEEDED 状态报告模板

```
## 训练完成 — `<PRODJOB_ID>`

| 指标 | 值 |
|---|---|
| **状态** | ✅ SUCCEEDED |
| **完成 Step** | <MAX_STEPS> |
| **最终 Loss** | <LATEST_LOSS_VAL> |
| **最终 token_acc** | <TOKEN_ACC> |
| **Final Checkpoint** | <FINAL_CHECKPOINT_S3> |
| **中间 Checkpoints** | <SAVE_CHECKPOINTS> |
```

### STARTING 状态报告模板

```
## 训练任务 — `<PRODJOB_ID>`

任务状态：**STARTING（集群启动中，请稍候）**

集群资源正在调度，通常需 3-5 分钟。启动完成后会自动开始训练并输出 step 日志。
建议稍后再次查询进度。
```

---

## ETA 计算逻辑

ETA 计算依赖 `STEP_PROGRESS` 和 `STEP_TIME` 两个字段。

```
解析：
  CURRENT_STEP = 从 STEP_PROGRESS 提取当前步数
  MAX_STEPS    = 从 STEP_PROGRESS 提取总步数
  STEP_SEC     = 将 STEP_TIME 统一换算为秒/步

计算：
  REMAINING_STEPS = MAX_STEPS - CURRENT_STEP
  ETA_SECONDS     = REMAINING_STEPS × STEP_SEC
  ETA_HOURS       = ETA_SECONDS / 3600
```

汇报时：
- `ETA_HOURS < 1` → 显示「约 X 分钟」
- `ETA_HOURS < 24` → 显示「约 X 小时 Y 分钟」
- `ETA_HOURS ≥ 24` → 显示「约 X 天 Y 小时」

**注意**：ETA 是基于当前 step time 的静态估算，实际剩余时间会随 step time 波动而变化。若 `HEALTH=WARN` 且 step time 呈上升趋势，需在报告中注明「ETA 可能偏乐观」。

---

## 健康状态判断逻辑

健康状态主要由脚本输出的 `HEALTH` 字段决定，但可结合 `LATEST_LOSS` 趋势做补充判断。

| HEALTH | 含义 | 汇报建议 |
|---|---|---|
| `OK` | 一切正常 | 直接汇报进度 |
| `WARN` | 有警告但训练继续 | 汇报进度 + 注明警告内容（`HEALTH_NOTES`） |
| `FAIL` | 训练已异常退出 | 停止进度查询，路由到诊断流程 |

补充判断（基于 `LATEST_LOSS` 趋势）：
- 若最近 5 个 step 的 loss 持续上升且未恢复 → 在报告中额外注明「⚠️ loss 持续上升，建议关注」
- 若 `TOKEN_ACC` 持续下降 → 在报告中额外注明「⚠️ token_acc 下降，可能存在数据或模型问题」

---

## 注意事项

1. 本场景是只读操作，不会对运行中的任务产生任何影响。
2. 若 `STEP_PROGRESS` 为 `N/A` 且 `JOB_STATE=RUNNING`，可能是日志尚未输出 step 信息，建议等待 1-2 分钟后重新查询。
3. 若 `LOG_LINES` 极少（如少于 10 行），可能是日志拉取不完整，可在报告中注明「日志行数较少，结果可能不完整」。
4. 查询 `STARTING` 状态的任务时，日志为空属于正常现象，无需报错。
