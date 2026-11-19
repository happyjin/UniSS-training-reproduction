# PyTorchJob 节点健康 / 资源排查

> 触发词：`节点` `node` `白名单` `health` `cordon` `资源` `GPU` `Pending`

本场景做节点层健康 / 资源 / 调度诊断。具体节点白名单**来自项目 `pytorchjob.yaml` 的 affinity 配置**，本 skill 不硬编码。

---

## 节点白名单（由项目固定，不可随意更改）

`pytorchjob.yaml` 的 Master/Worker 副本通常通过 `affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution`（`kubernetes.io/hostname` **`In`**）把 job 固定到一组已知良好节点：

- 白名单节点由项目 yaml 定义，agent 提交前解析 `<NODE_WHITELIST>`。
- **白名单固定不动**：不要靠换节点规避训练侧问题（如 OOM）。若节点本身有问题（NotReady / cordon），白名单自然排除它；若需调整白名单，改 yaml 而非临时调度。
- 调度失败时先区分：是**节点不健康**，还是**白名单容量不足 / 资源被重复 job 占用**。

---

## 流程

### Step 1 — 连通性

```bash
kubectl version 2>&1 | head -2
kubectl -n <NAMESPACE> get nodes 2>/dev/null || kubectl get nodes -L node.kubernetes.io/instance-type
```

### Step 2 — 调用诊断脚本

```bash
K8S_NS=<NAMESPACE> bash <skill>/scripts/inspect_nodes.sh <NAMESPACE>
```

脚本输出结构化字段：`NODES_TOTAL` / `NODES_READY` / `GPU_HEALTHY` / `PENDING_PODS` / `DUP_JOBS`（重复 job 检测）。

### Step 3 — 手动逐节点（如需）

```bash
# 所有训练 GPU 节点（按 GPU 可分配数识别，避免硬编码机型）
kubectl get nodes -o custom-columns='NODE:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,GPU:.status.allocatable.nvidia\.com/gpu' --no-headers
# 逐节点 GPU 可分配 / Ready
for n in $(kubectl get nodes -o name); do
  kubectl get ${n} -o jsonpath='{.metadata.name} Ready={.status.conditions[?(@.type=="Ready")].status} GPU={.status.allocatable.nvidia\.com/gpu}{"\n"}'
done
# Pending pods
kubectl get pods -A --no-headers | awk '$4=="Pending"'
```

### Step 4 — 重复 job 检测（重要）

外部自动化可能删了又建带后缀的重复 PyTorchJob（与主 job 争抢 GPU）：

```bash
kubectl -n <NAMESPACE> get pytorchjob
# 若有重复，清理重复实例（保留主 job）
kubectl -n <NAMESPACE> delete pytorchjob <重复 JOB_NAME>
```

---

## 输出模板

```markdown
## 节点 / 资源排查（PyTorchJob）

### 白名单节点（项目固定）
| 角色 | 节点 | Ready | GPU 可分配 |
|---|---|---|---|
| master | <NODE_WHITELIST master> | <Ready> | <%per-node GPU> |
| worker | <NODE_WHITELIST worker> | <Ready> | <%per-node GPU> |

### 集群
- 训练 GPU 节点总数：<N>，Ready：<M>，GPU 健康：<K>
- Pending pods：<P>（<原因>）
- 重复 job：<有/无>

### 建议
<...>
```

---

## 注意事项

1. 白名单固定不动，不要靠换节点规避训练侧问题。
2. `kubectl version --short` 已废弃，禁止用。
3. 可选即时 `kubectl cordon <坏节点>`，但白名单已约束调度范围；cordon 通常用于隔离故障节点供排查。
4. 重复 job 需排查外部 controller / 自动重试根源，而非仅删除实例。
