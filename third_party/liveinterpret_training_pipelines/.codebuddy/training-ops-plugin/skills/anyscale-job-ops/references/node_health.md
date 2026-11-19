# 节点健康诊断

> 触发词：`节点健康` `node health` `资源调度` `资源不足` `卡在调度` `Insufficient` `节点状态` `GPU 故障`

本场景**仅做节点层健康诊断**，不拉取任务日志、不查任务运行状态、不分析训练指标。

---

## 流程总览

```
用户输入
  → Step 1: 确定目标集群（从 prodjob 或 compute config 反查）
  → Step 2: 验证 kubectl 连通性
  → Step 3: 列出所有 GPU 节点，逐节点采集健康指标
  → Step 4: 检查 Pending pods 及调度失败原因
  → Step 5: 交叉对比 compute config 黑名单与可用节点
  → Step 6: 汇总输出结构化诊断报告
```

所有步骤使用 `kubectl` + `anyscale` CLI，不读本地仓库文件。

---

## Step 1 — 确定目标集群

从用户输入中提取目标信息，按以下优先级确定集群：

**方式 A：用户提供了 prodjob_id**
```bash
# 获取任务详情（含 compute config）
anyscale job status --id <PRODJOB_ID> --verbose 2>&1
```
从输出中提取 `compute_config:` 字段（如 `tokyo-h100-gpu:1`）。

如需进一步查询 compute config 的云信息：
```bash
anyscale compute-config get -n <COMPUTE_CONFIG_NAME> 2>&1 | grep -E 'cloud:|cloud_resource:|region'
```
从 `cloud` 字段确认目标集群。

**方式 B：用户提供了 compute config 名称**
```bash
anyscale compute-config get -n <COMPUTE_CONFIG_NAME> 2>&1 | grep -E 'cloud:|cloud_resource:|instance_type:|min_nodes|max_nodes|blacklist|NotIn'
```

**方式 C：用户未提供标识符**
直接查当前 kubectl 上下文对应的集群：
```bash
kubectl config current-context
```

---

## Step 2 — 验证 kubectl 连通性

```bash
kubectl version 2>&1
```

**注意**：不要使用 `kubectl version --short`（新版本 kubectl 已废弃该 flag，会误判为不通）。

如果 `kubectl version` 失败：
- 检查 kubeconfig 是否存在：`ls -la ~/.kube/config`
- 检查当前 context：`kubectl config current-context`
- 列出所有 context：`kubectl config get-contexts`
- 如果集群在 ap-northeast-1，尝试：`aws eks update-kubeconfig --region ap-northeast-1 --name <cluster-name>`

kubectl 不通 → 停止诊断，输出 `FAIL_REASON=kubectl-not-connected`。

---

## Step 3 — 列出所有 GPU 节点并采集逐节点健康指标

> **⚠️ 强制约束**：如果 Step 1/Step 5 已获取 compute config 的白名单/黑名单，则 **必须只诊断白名单中的节点（或黑名单外的节点）**，不得诊断所有集群节点。

### 3.0 确定诊断范围（白名单/黑名单）

**在执行 3.1 之前，必须先确定诊断范围**：

```bash
# 情况 1: 有白名单（operator: In）→ 只诊断白名单中的节点
WHITELIST_NODES=("node-1" "node-2")

# 情况 2: 有黑名单（operator: NotIn）→ 诊断除黑名单外的所有节点  
BLACKLIST_NODES=("node-3" "node-4")

# 情况 3: 无约束 → 诊断所有 GPU 节点
```

**⚠️ 如果无法确定白名单/黑名单（如未提供 prodjob_id 或 compute config），则诊断所有节点，但必须在输出中注明"未获取 compute config，诊断所有节点"。

### 3.1 列出目标节点

根据集群中实际的实例类型筛选。常见 GPU 实例类型：

| 实例类型 | GPU 型号 | 说明 |
|---|---|---|
| `ml.p5.48xlarge` | 8×H100-80GB | H100 节点 |
| `p5.48xlarge` | 8×H100-80GB | H100 节点（旧标签） |

```bash
kubectl get nodes -L node.kubernetes.io/instance-type --no-headers 2>&1 | grep -iE 'p5\.48xlarge|h100'
```

如果无匹配节点，尝试不加过滤列出所有节点：
```bash
kubectl get nodes -L node.kubernetes.io/instance-type --no-headers 2>&1 | grep -i 'gpu\|h100\|a100\|p5\|p4d\|g5'
```

无 GPU 节点 → 输出 `P5_NODES_TOTAL=0`，结束。

### 3.2 逐节点采集健康指标

对 3.1 列出的**每个** GPU 节点，采集以下指标。所有字段除了 Running pods 用 `kubectl describe node` 外，均用 `kubectl get node -o jsonpath` 获取：

```bash
# 对每个节点 NODE_NAME，逐一执行：

# (a) Ready 状态
kubectl get node "${NODE}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}'

# (b) 是否被标记为不可调度
kubectl get node "${NODE}" -o jsonpath='{.spec.unschedulable}'

# (c) 可分配 GPU 数量
kubectl get node "${NODE}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'

# (d) 可分配 EFA 数量
kubectl get node "${NODE}" -o jsonpath='{.status.allocatable.vpc\.amazonaws\.com/efa}'

# (e) SageMaker 节点健康状态
kubectl get node "${NODE}" -o jsonpath='{.metadata.labels.sagemaker\.amazonaws\.com/node-health-status}'

# (f) SageMaker 故障原因
kubectl get node "${NODE}" -o jsonpath='{.metadata.labels.sagemaker\.amazonaws\.com/fault-reasons}'

# (g) 可用区
kubectl get node "${NODE}" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}'

# (h) Taints（完整 JSON）
kubectl get node "${NODE}" -o jsonpath='{.spec.taints}'

# (i) 节点上正在运行的 Ray/Anyscale workloads（仅关键 pod）
kubectl describe node "${NODE}" 2>/dev/null | grep -E '^\s+Namespace|anyscale|ray' | head -20
```

### 3.3 判断每节点健康状态

对每个节点，按以下规则分类：

| 条件 | 状态 | 说明 |
|---|---|---|
| `Ready=True` + `GPU=8` + `Unschedulable != true` | 🟢 健康 | 可调度 |
| `Ready=True` + `0 < GPU < 8` | 🟡 GPU 降级 | N 张 GPU 不可用，需排查 |
| `Ready=True` + `GPU=0` 或 `GPU` 不存在 | 🔴 GPU 全部故障 | 设备插件异常或硬件全坏 |
| `Ready != True` | 🔴 节点 NotReady | kubelet 异常或节点宕机 |
| `Unschedulable=true` | 🔴 已标记不可调度 | SageMaker HMA 可能已 taint |
| `Fault=TaskHung` | ⚠️ 任务挂起 | 节点上曾有任务 hang |
| `Fault=NvidiaGpuXidRestartApp` | ⚠️ Xid 重启 | GPU 曾触发 Xid 错误 → 设备插件重启过 |

---

## Step 4 — 检查 Pending pods 及调度失败原因

### 4.1 列出所有 Pending pods

```bash
kubectl get pods -A --no-headers 2>&1 | awk '$4 == "Pending"'
```

记录 Pending pod 数量和命名空间。

### 4.2 查看每个 Pending pod 的调度失败原因

对每个 Pending pod（重点关注 `anyscale-operator` 命名空间下的 `k-*` pod），describe 并只看 Events 段：

```bash
# NS=命名空间, NAME=pod名
kubectl describe pod -n "${NS}" "${NAME}" 2>/dev/null | sed -n '/^Events:/,$p' | head -30
```

也可以用更高效的方式一次性收集：

```bash
for ns in $(kubectl get pods -A --no-headers 2>/dev/null | awk '$4=="Pending"{print $1}' | sort -u); do
  for pod in $(kubectl get pods -n "$ns" --no-headers 2>/dev/null | awk '$3=="Pending"{print $1}'); do
    echo "=== ${ns}/${pod} ==="
    kubectl describe pod -n "$ns" "$pod" 2>/dev/null | sed -n '/^Events:/,$p' | head -20
    echo ""
  done
done
```

### 4.3 解析 FailedScheduling 事件

从 Events 中提取关键信息：

| Event 关键词 | 含义 | 对应操作 |
|---|---|---|
| `Insufficient nvidia.com/gpu` | GPU 不足 | 统计哪些节点 GPU < 8 |
| `Insufficient cpu` | CPU 不足 | 检查节点上已有 pod 的 CPU requests |
| `Insufficient memory` | 内存不足 | 检查节点上已有 pod 的 memory requests |
| `Insufficient vpc.amazonaws.com/efa` | EFA 不足 | 检查节点 EFA allocatable |
| `didn't match Pod's node affinity/selector` | 节点选择器不匹配 | 交叉对比 compute config 中的 nodeAffinity / 黑名单 |
| `No preemption victims found` | 无法驱逐低优先级 pod | 所有节点均被高优先级 pod 占满 |

---

## Step 5 — 交叉对比 compute config 与可用节点

### 5.1 获取 compute config 的完整配置

如果用户提供了 compute config 名称或 prodjob_id（从 Step 1 获取），**必须**拉取完整配置并解析节点约束：

```bash
anyscale compute-config get -n <COMPUTE_CONFIG_NAME> --verbose 2>&1
```

**关键：必须解析 nodeAffinity 中的所有约束**

从配置中提取以下信息：

```bash
# 1. 提取 worker 节点的 nodeAffinity（白名单/黑名单）
anyscale compute-config get -n <COMPUTE_CONFIG_NAME> 2>&1 | grep -A 30 'nodeAffinity'

# 2. 提取 min/max worker 节点数
anyscale compute-config get -n <COMPUTE_CONFIG_NAME> 2>&1 | grep -E 'min_nodes|max_nodes'

# 3. 提取 worker 实例类型和名称
anyscale compute-config get -n <COMPUTE_CONFIG_NAME> 2>&1 | grep -E 'instance_type|name:'
```

### 5.2 解析节点约束（白名单优先于黑名单）

**⚠️ 关键：必须先检查白名单（In），再检查黑名单（NotIn）**

从 `nodeAffinity` 中解析：

| operator | 含义 | 处理方式 |
|---|---|---|
| `In` | **白名单**：只有列表中的节点可用 | **只诊断白名单中的节点** |
| `NotIn` | **黑名单**：列表中的节点不可用 | 诊断除黑名单外的所有节点 |
| 两者都存在 | 白名单优先 | 先应用白名单，再应用黑名单 |

### 5.3 计算有效可用节点数

**必须按以下顺序过滤节点**：

1. **应用白名单**（如有）：只保留白名单中的节点
2. **应用黑名单**（如有）：从剩余节点中移除黑名单中的节点
3. **检查 GPU 健康**：过滤出 `allocatable GPU = 8` 的节点
4. **检查 Ready 状态**：过滤出 `Ready=True` 的节点

### 5.4 调用诊断脚本（传递黑白名单参数）

**关键**：解析完黑白名单后，必须调用 `inspect_p5_nodes.sh` 脚本并传递正确的参数：

```bash
# 情况 1: 只有白名单（In operator）
WHITELIST="<node-a>,<node-b>"
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --whitelist "${WHITELIST}"

# 情况 2: 只有黑名单（NotIn operator）
BLACKLIST="<node-a>,<node-b>"
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --blacklist "${BLACKLIST}"

# 情况 3: 既有白名单又有黑名单
WHITELIST="<node-a>,<node-b>"
BLACKLIST="<node-c>"
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --whitelist "${WHITELIST}" --blacklist "${BLACKLIST}"

# 情况 4: 无约束（既无白名单也无黑名单）
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh

# 如果需要自动恢复 GPU 降级节点，添加 --recover 参数
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --whitelist "${WHITELIST}" --recover
```

**注意**：
- 脚本会自动处理所有 4 种黑白名单组合情况
- 白名单优先于黑名单
- 如果二者都不指定，脚本会检查所有 GPU 节点
- 解析黑白名单的逻辑**必须**在 Step 5.1-5.2 完成，然后在 Step 5.4 传递给脚本

### 5.5 判断根因

| 条件 | 根因 |
|---|---|
| 有效可用节点数 < min_nodes | **资源不足**：白名单/黑名单 + GPU 降级导致可用节点不够 |
| 有效可用节点数 ≥ min_nodes 但仍有 Pending | 检查 pod 的 resource requests 是否超过节点剩余容量 |
| 白名单中的节点全部不健康 | **白名单配置问题**：所有指定的节点都不可用，需手动更换白名单 |

---

## Step 6 — 输出结构化诊断报告

按以下模板输出，不得自由发挥：

```markdown
## 节点健康诊断

**集群**: <cluster-name>（<region>），共 <N> 个 <instance-type> <GPU 型号> 节点

### 逐节点状态

| 节点 | GPU 可分配 | Ready | 白名单 | 黑名单 | Fault | 备注 |
|---|---|---|---|---|---|
| <node-1> | **8/8** ✅ | True | ✅/❌/N/A | ✅/❌/N/A | <fault> | |
| <node-2> | **2/8** ⚠️ | True | ✅/❌/N/A | ✅/❌/N/A | <fault> | 6 GPU 故障 |
| ... | | | | | |

**白名单**: [节点列表] （如未配置则显示 "未设置（所有节点可用）"）
**黑名单**: [节点列表] （如未配置则显示 "未设置"）

### Pending Pods

| Pod | 命名空间 | 失败原因 |
|---|---|---|
| <pod-name> | <ns> | `FailedScheduling: 2 Insufficient nvidia.com/gpu, ...` |

### 根因分析

- **白名单指定节点数**: <N> 个
- **黑名单排除节点数**: <N> 个
- **有效可用节点数**: <healthy_count> / <total_count>
- **瓶颈**: <具体瓶颈描述>

### 建议操作

1. <操作 1>
2. <操作 2>
```

---

## 快速诊断命令集合（一次性采集）

以下命令块可一次性采集所有诊断所需信息，逐条复制执行即可：

```bash
# === 0. kubectl 连通性 ===
kubectl version 2>&1 | head -2

# === 1. 所有 p5/H100 节点一览 ===
kubectl get nodes -L node.kubernetes.io/instance-type --no-headers 2>&1 | grep -iE 'p5|h100|gpu'

# === 2. 逐节点健康（替换 NODE_NAME 循环） ===
for NODE in $(kubectl get nodes -L node.kubernetes.io/instance-type --no-headers 2>/dev/null | grep -iE 'p5\.48xlarge|h100' | awk '{print $1}'); do
  echo "=== $NODE ==="
  kubectl get node "$NODE" -o jsonpath='Ready: {.status.conditions[?(@.type=="Ready")].status}, Unschedulable: {.spec.unschedulable}, GPU: {.status.allocatable.nvidia\.com/gpu}, EFA: {.status.allocatable.vpc\.amazonaws\.com/efa}, Health: {.metadata.labels.sagemaker\.amazonaws\.com/node-health-status}, Fault: {.metadata.labels.sagemaker\.amazonaws\.com/fault-reasons}, AZ: {.metadata.labels.topology\.kubernetes\.io/zone}' 2>/dev/null
  echo ""
  echo "Taints:"
  kubectl get node "$NODE" -o jsonpath='{.spec.taints}' 2>/dev/null
  echo ""
  echo "Anyscale pods:"
  kubectl describe node "$NODE" 2>/dev/null | grep -E '^\s+Namespace|anyscale' | head -12
  echo "---"
done

# === 3. Pending pods 及调度失败原因 ===
kubectl get pods -A --no-headers 2>/dev/null | awk '$4=="Pending"'
echo ""
for ns in $(kubectl get pods -A --no-headers 2>/dev/null | awk '$4=="Pending"{print $1}' | sort -u); do
  for pod in $(kubectl get pods -n "$ns" --no-headers 2>/dev/null | awk '$3=="Pending"{print $1}'); do
    echo "=== ${ns}/${pod} ==="
    kubectl describe pod -n "$ns" "$pod" 2>/dev/null | sed -n '/^Events:/,$p' | head -20
    echo ""
  done
done

# === 4. compute config 黑名单（需 COMPUTE_CONFIG_NAME） ===
# anyscale compute-config get -n <COMPUTE_CONFIG_NAME> 2>&1 | grep -A 30 'nodeAffinity'
```

---

## Step 7 — 恢复 GPU 降级节点（可选）

当 Step 3 发现节点 GPU allocatable < capacity（如 cap=8, alloc=2），通常是 device-plugin 因 Xid 错误将 GPU 标记为 unhealthy。

**适用场景**：
- 节点出现 XID 94 → device-plugin 把 GPU mark 为 unhealthy → `allocatable < capacity`
- 节点 Ready 且其他系统指标正常

**不适用场景**（不要重启 / 重启了也没用）：
| 场景 | 处理 |
|---|---|
| 重启完不到 1 小时又跌回 `alloc < cap` | **不要再重启**，硬件持续故障，走 AWS Support 换节点 |
| 重启后 alloc 没恢复 | 同上，硬件级问题 |
| 节点上有不认识的 host 进程占着 GPU | 先确认是不是同事在跑，再决定 |
| 节点本身 NotReady / 不可达 | 不是 device-plugin 的问题，找运维 |

### 7.1 使用脚本自动恢复（推荐）

**推荐使用 `inspect_p5_nodes.sh` 脚本的 `--recover` 模式**，它会自动发现 GPU 降级节点并恢复：

```bash
# 基本用法：诊断并自动恢复所有 GPU 降级节点
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --recover

# 结合黑白名单：只恢复特定节点
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --whitelist "${WHITELIST}" --recover
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --blacklist "${BLACKLIST}" --recover

# 结合 compute config：对比期望 vs 实际节点数
bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh <COMPUTE_CONFIG_NAME> --recover
```

> ⚠️ 重启 device-plugin 属于变更操作，执行前需确认（除非用户明确指令恢复）。

### 7.2 手动恢复（仅用于脚本无法使用的场景）

```bash
# 1. 查找 GPU 降级节点
for n in $(kubectl get nodes -o name | grep hyperpod | sed 's|node/||'); do
  cap=$(kubectl get node $n -o jsonpath='{.status.capacity.nvidia\.com/gpu}')
  alloc=$(kubectl get node $n -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
  [ "$cap" = "$alloc" ] || echo "❌ $n  cap=$cap alloc=$alloc"
done

# 2. 找到该节点的 device-plugin pod
NODE=hyperpod-i-XXXXXXXX
POD=$(kubectl -n kube-system get pods -o wide | grep nvidia-device-plugin | grep $NODE | awk '{print $1}')
echo "要重启的 pod: $POD"

# 3. 重启 device-plugin pod
kubectl -n kube-system delete pod $POD

# 4. 验证恢复
sleep 15
kubectl get node $NODE -o jsonpath='cap={.status.capacity.nvidia\.com/gpu} alloc={.status.allocatable.nvidia\.com/gpu}{"\n"}'
```

`cap = alloc` 就是恢复成功。

### 7.3 反复故障的判定

如果一台节点的某张 GPU 在 24 小时内**报 ≥ 2 次 XID 94**，按"反复硬件故障"对待，走 AWS Support 替节点流程。

### 7.4 背景：为什么重启 device-plugin 能恢复

XID 94 是 NVIDIA 的 "ECC Contained Memory Error"——单个 SM 出 ECC 错误后被自动隔离，**GPU 本身不需要重启**。但 device-plugin 的 health check 会缓存这次 unhealthy 状态，把这张 GPU 从 K8s allocatable 列表里去掉。重启 device-plugin 让它重新 query NVML，看到 GPU 健康就会重新注册。

---

## 注意事项

1. **只做节点层诊断**：不拉取任务日志、不查 job status、不分析训练指标。如果用户同时需要任务诊断，应由主路由分流到 `diagnose.md`。
2. **kubectl 版本兼容**：`kubectl version --short` 在 1.27+ 已废弃，禁止使用。改用 `kubectl version`。
3. **GPU allocatable 为 0**：通常意味着 nvidia-device-plugin 未正常运行或 GPU 全部故障，需检查 `kube-system` 下的 device-plugin pod：
   ```bash
   kubectl logs -n kube-system -l app=nvidia-device-plugin --tail=100
   ```
4. **Fault=NvidiaGpuXidRestartApp**：说明 GPU 曾触发 Xid 错误导致设备插件重启，GPU allocatable 恢复了但历史错误仍在标签上。不影响当前调度，但需关注是否频繁反复。
5. **Fault=TaskHung**：节点曾出现任务卡死，GPU 可能已降级。如果 GPU allocatable < 8，该节点应加入黑名单。
6. **Pending pod 的 Events 可能为空**：如果 pod 刚创建，scheduler 还未评估，等待 5-10 秒后重新 describe。
7. **恢复操作需用户确认**：重启 device-plugin 会改变节点状态，执行前需用户确认（除非用户明确要求恢复）。
