# KtrainJob 节点健康 / 资源排查

> 触发词：`节点` `node` `GPU` `资源` `Pending` `白名单` `cordon` `health`

## Step 1 — 确定命名空间与白名单

- `<NAMESPACE>`：用户输入 > 任务 yaml `metadata.namespace`。
- `<NODE_WHITELIST>`：任务 yaml render-extra affinity 中 `kubernetes.io/hostname In` 的 values。用户未指定任务时做全集群排查，跳过白名单比对。

## Step 2 — 连通性检查

```bash
ktrain mode
```

确认当前 ktrain context（mode / cluster）与目标集群一致。需要切换集群时用 `aws eks update-kubeconfig --name <集群> --region <区域>`（集群与区域由用户提供，不猜），然后 `ktrain auth kubeconfig` 刷新 ktrain 的 kubeconfig 引用。

## Step 3 — 调用排查脚本

```bash
bash <SKILL_DIR>/scripts/inspect_nodes.sh <NAMESPACE>
```

脚本执行：`ktrain mode`（当前 context 模式 / 集群）→ `ktrain gpu`（GPU 节点总览）→ `ktrain job list --namespace <NAMESPACE> --all`（检测重复 / 残留任务）→ `ktrain fs list`（文件系统）。stdout 输出 KEY=VALUE（`KTRAIN_MODE` / `CLUSTER` / `GPU_NODES_TOTAL` / `JOB_COUNT` / `FSX_LIST` / `OUT_BUNDLE`）。

需要实时 GPU 指标（利用率 / 显存 / 温度，经 nvidia-smi）时：`ktrain gpu --detail`（可加 `-n <NAMESPACE>` 过滤 Pod）。

## Step 4 — 逐节点细查（白名单或异常节点）

`ktrain gpu` 已列出每节点 GPU 分配；以下命令用于 ktrain 不覆盖的节点级诊断（`kubectl get nodes` / `kubectl describe node`），仅在 `ktrain gpu` 输出异常或需要节点级详情时执行：

```bash
kubectl get nodes -o custom-columns='NODE:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,UNSCHED:.spec.unschedulable,GPU:.status.allocatable.nvidia\.com/gpu' --no-headers
kubectl describe node <NODE>   # 仅对 NotReady / cordon 的节点
```

有白名单时逐节点核对：存在、Ready、未 cordon、GPU 可分配数符合任务 yaml `spec.resources.gpu`。

## Step 5 — 队列 / 配额（集群启用 Kueue 时）

```bash
ktrain queue list                 # LocalQueue 列表；未启用 Kueue 时返回 No queues found
ktrain queue show <QUEUE>         # 单队列配额详情（-o json 机器可读）
ktrain queue workloads <QUEUE>    # 队列内 Workload（排查任务排队 / 配额阻塞）
ktrain queue flavors              # ResourceFlavor 列表
```

任务 Pending 且节点资源充足时，优先查队列配额是否阻塞。队列的 create / update / delete / enable / disable 等为 admin 操作，不属常规排查手段。

## 输出模板

```markdown
## 节点 / 资源排查

### 白名单节点（来自任务 yaml）
| 节点 | Ready | 可调度 | GPU 可分配 |
|---|---|---|---|
| <NODE> | <True/False> | <是/否> | <N> |

### 集群
- GPU 节点总数 / Ready：<N> / <M>
- Pending pods：<P>（<原因>）
- 命名空间内任务数：<JOB_COUNT>（有重复 / 残留任务时列出）

### 结论
<健康 / 异常点及受影响节点>
```

## 禁止事项

- Pending pods 排查用 `ktrain job status` 看任务调度状态（ktrain CLI 不提供全局 Pending Pod 列表）；需要查全集群 Pending Pod 时用 `kubectl get pods -A --field-selector=status.phase=Pending`。
- 禁止 cordon / uncordon / 删节点等变更操作，除非用户明确指示。
- 禁止靠改任务 yaml 白名单绕开异常节点来「解决」资源问题；节点异常就报节点异常。
