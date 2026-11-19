# 查询 PyTorchJob 训练进度

> 触发关键词：`进度` `progress` `step` `loss` `还要多久` `ETA` `跑到哪`

---

## 流程

### Step 1 — 定位 pod

PyTorchJob 名：`<JOB_NAME>`（从 `kubectl -n <NAMESPACE> get pytorchjob` 读取或用户提供）。
副本角色为 `master-0` / `worker-0`（Kubeflow PyTorchJob 约定）。

```bash
kubectl -n <NAMESPACE> get pods -l training.kubeflow.org/job-name=<JOB_NAME> -o wide
```

### Step 2 — 调用进度脚本

```bash
K8S_NS=<NAMESPACE> bash <skill>/scripts/query_job.sh <JOB_NAME> <NAMESPACE>
```

脚本会：
1. 查 pod 状态（master-0 / worker-0 的 Ready / Restarts / 节点）。
2. `kubectl logs -n <NAMESPACE> <JOB_NAME>-master-0 -c pytorch` 拉日志并解析 step / loss / token_acc。
3. 输出结构化 KEY=VALUE（`JOB_STATE` / `POD_STATUS` / `MASTER_NODE` / `RESTARTS` / `STEP_PROGRESS` / `LATEST_LOSS` / `TOKEN_ACC` / `HEALTH` / `LOG_LINES`）。

### Step 3 — 按状态分支

| 状态 | 汇报 |
|---|---|
| pod `Running` 0 重启 | 汇报 step 进度 + loss + ETA |
| pod `Pending` | 查节点调度（见 node_health.md） |
| pod `Error` / `OOMKilled` / Restarts 累加 | 停止进度，路由到 diagnose.md |
| job `Succeeded` | 汇报完成 step、最终 loss、checkpoint |

### ETA 计算

```
REMAINING_STEPS = MAX_STEPS - CURRENT_STEP
ETA = REMAINING_STEPS × STEP_SEC
```

---

## 输出模板（Running）

```
## 训练进度 — <JOB_NAME>

| 指标 | 当前值 |
|---|---|
| **Pod 状态** | master-0 Running (Restarts=0) @ <node> |
| **Step** | <CURRENT> / <MAX>（<PCT>%） |
| **最新 Loss** | <LATEST_LOSS> |
| **token_acc** | <TOKEN_ACC> |
| **健康** | <HEALTH> |

**预计剩余**：约 <ETA>
```

---

## 注意事项

1. 只读操作。
2. 日志极少可能是刚启动；GPU 节点日志可能较大，必要时重定向到文件本地分析（见 diagnose.md 纪律）。
3. 关注 `Restarts` 是否累加——worker 一旦 OOM 崩溃会重启并触发分布式通信（如 NCCL）级联失败。
