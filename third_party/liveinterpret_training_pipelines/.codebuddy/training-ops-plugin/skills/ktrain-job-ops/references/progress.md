# 查询 KtrainJob 训练进度

> 触发词：`进度` `progress` `step` `loss` `ETA` `还要多久` `跑到哪` `跑完了吗`

## Step 1 — 确定任务名与命名空间

- `<JOB_NAME>`：从用户输入提取；未提供时先问用户，或列出候选让用户选：
  ```bash
  ktrain job list --namespace <NAMESPACE> --all
  ```
- `<NAMESPACE>`：用户输入 > 任务 yaml `metadata.namespace` > 提交时的汇报记录。两者都不许缺省猜测。

## Step 2 — 调用进度脚本

```bash
bash <SKILL_DIR>/scripts/query_job.sh <JOB_NAME> <NAMESPACE>
```

脚本执行：`ktrain job status` 取状态 → `ktrain job logs` 落盘全量日志到 `/tmp/ktrain-progress/<JOB_NAME>/` → 解析 step / loss / token_acc / grad_norm / lr / checkpoint / 异常信号，stdout 输出 KEY=VALUE（`JOB_STATE` / `STEP_PROGRESS` / `STEP_TIME` / `LATEST_LOSS` / `TOKEN_ACC` / `HEALTH` / `LOG_LINES` / `OUT_BUNDLE`）。

## Step 3 — 按 JOB_STATE 分支

| 状态 | 动作 |
|---|---|
| `Pending` / `Creating` | 汇报调度中；转 `node_health.md` 查节点 |
| `Running` | 按模板汇报 step / loss / ETA |
| `Succeeded` | 汇报最终 step、最终 loss、checkpoint 列表 |
| `Failed` / `Error` | 停止进度汇报，转 `diagnose.md` |

ETA 计算：

```
REMAINING_STEPS = MAX_STEPS - CURRENT_STEP
ETA = REMAINING_STEPS × STEP_TIME
```

`STEP_TIME` 为 `s/it` 时直接乘；为 `it/s` 时取倒数再乘。`STEP_PROGRESS=N/A`（日志未打出 step 行）时不估 ETA，只汇报状态。

## 输出模板（Running）

```markdown
## 训练进度 — `<JOB_NAME>`

| 指标 | 当前值 |
|---|---|
| **状态** | Running |
| **Step** | <CURRENT_STEP> / <MAX_STEPS>（<PCT>%） |
| **最新 Loss** | <LATEST_LOSS> |
| **token_acc** | <TOKEN_ACC> |
| **单步耗时** | <STEP_TIME> |
| **健康** | <HEALTH>（<HEALTH_NOTES>） |

**预计剩余**：约 <ETA>
```

## 注意事项

- 全程只读，不影响任务。
- 需要机器可读状态时：`ktrain -o json job status <JOB_NAME> --namespace <NAMESPACE>`（全局 `-o json` 对 list / status / info / show 生效）。
- `LOG_LINES` 很小（<10）说明任务刚启动，等几分钟后重查，不要报"无日志"。
- `HEALTH != OK` 时即使状态 Running 也要在汇报中给出 `HEALTH_NOTES` 内容。
