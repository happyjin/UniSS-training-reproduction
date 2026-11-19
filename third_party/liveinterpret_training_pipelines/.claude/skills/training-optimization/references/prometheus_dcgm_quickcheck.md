# Prometheus/DCGM 快速预判（nsys 下钻前的免费一层）

**定位**：插在"四项核心指标诊断"之后、"要不要下钻 nsys"之前。如果集群里有
Prometheus + DCGM exporter（很多 GPU 集群标配），可以在**秒级、零改动、
不用等到指定 step、不用改任何任务配置**的情况下，直接对正在跑的任务（哪怕
不是自己提交的）做一次方向性判断——比 nsys 轻量得多，通常能替代"要不要抓
nsys"这个决策里最耗时的"猜方向"环节。

**这一层不是 nsys 的替代品**：它只能告诉你瓶颈属于哪一类（launch-bound /
计算-bound / 显存带宽-bound / 通信-bound），**不能告诉你具体是哪个 kernel**
——需要精确定位到算子层面时，仍然要走 nsys。集群没有 Prometheus/DCGM 时，
直接跳过这一层，走原有的"指标诊断→nsys下钻"流程，不强求。

## 0. 判断集群是否具备这个能力

```bash
export KUBECONFIG=<你的kubeconfig路径>
kubectl get svc -A | grep -iE "prometheus|dcgm"
```

看到类似 `kube-prometheus-stack-prometheus`（9090端口）的 ClusterIP service
即可用；DCGM 数据本身由某个 `*dcgm-exporter*` DaemonSet 采集并被 Prometheus
抓取，不需要单独确认（Prometheus 里查得到 `DCGM_FI_*` 系列指标名就说明链路通）。
没有则跳过本节。

记录下这一步找到的 Prometheus service 的**命名空间**和**service 名**（下面
第 1 节的 port-forward、第 5 节的脚本都要用到），不同集群这两项通常不同，
不要照抄别的项目/别的集群里用过的值。

## 1. 访问方式

### 先决条件：任务必须已提交并在运行

⚠️ **重要**：Prometheus 只能查询**正在运行或最近运行过**的任务数据。如果任务还没提交过（如 yaml 中 `metadata.name: __JOB_NAME__` 为模板），查不到任何 DCGM 数据。

**操作顺序**：
1. 先用 `ktrain-job-ops` 或其他工具**提交任务**，确保 Pod 启动运行
2. 等待任务运行至少 2-3 个 step，确保 DCGM exporter 有时间采集数据
3. 才能用本节方法查询 Prometheus 中的硬件指标

### 本地访问 Prometheus

Prometheus service 通常只有 ClusterIP，本地要 port-forward：

```bash
export KUBECONFIG=<你的kubeconfig路径>
kubectl port-forward -n <第0步找到的Prometheus所在namespace> svc/<第0步找到的service名> 19090:9090 &
```

用 HTTP API 查询（`query` 是某一时刻的瞬时值——**不要直接用它下结论**，见
第 2 节；`query_range` 才是本节真正要用的）：

```bash
# 瞬时值（仅用于快速摸底，不作为结论依据）
curl -s "http://localhost:19090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL"

# 区间查询（结论依据，见第2节的窗口选择）
curl -s "http://localhost:19090/api/v1/query_range?query=<METRIC>%7BHostname%3D%22<NODE>%22%7D&start=<UNIX_START>&end=<UNIX_END>&step=15"
```

（PromQL 的花括号/引号在 URL 里要做 percent-encode：`{`→`%7B`、`"`→`%22`、
`}`→`%7D`；也可以用 `--data-urlencode` 让 curl 自动处理，见下方脚本化写法。）

### 查询时需要的参数

| 参数 | 含义 | 如何获取 |
|---|---|---|
| **Hostname** | Pod 所在节点名 | `kubectl describe pod <pod_name> -n <你的任务所在namespace>` 中的 `Node:` 字段 |
| **GPU ID** (可选) | 具体哪张卡（0-7）| DCGM 指标中会有 `gpu` 标签，不指定则查所有卡的平均或聚合 |
| **START / END** (Unix时间戳) | 查询时间窗口 | `date +%s` 获取当前时间戳；往前推，如最近15分钟：`END=$(date +%s); START=$((END-900))` |


## 2. ⭐ 窗口大小怎么选（不要用固定 5 分钟）

**原则：窗口至少要覆盖 10-20 个当前稳态 step 的时长**，跟
`metric_interpretation.md` 里 s/it 稳态窗口的逻辑同源，只是单位从 step 换成
了时间：

| 已知信息 | 窗口取法 |
|---|---|
| 知道稳态 s/it（比如从训练日志读到） | `窗口 = max(5分钟, 15~20 × 稳态s/it)`。10s/it 的任务 5 分钟够（约30步）；60s/it 的任务要拉到 15-20 分钟 |
| 不知道稳态 s/it（比如查别人的任务、没有日志权限） | 先查是否有周期性事件（checkpoint/eval 间隔），窗口要**明显大于**这个周期；完全没信息时默认从 15 分钟起步，不要低于 10 分钟 |
| 任务刚启动不久 | 避开冷启动那一段，起点不能从任务开始时刻算（跟 s/it 排除 warmup 同一逻辑） |

**必须交叉验证**：同时查两个不同长度的窗口（如 5 分钟 vs 15 分钟），对比
`mean`/`max`：

- 两次结果同量级（差异不超过 30-50%）→ 可信，正常往下判断
- 两次结果差好几倍/方向不一致 → 说明这段时间本身不稳定（大概率踩在
  checkpoint/eval/数据切片边界，或者任务刚好在这几分钟内发生了状态切换），
  **不能直接采信任一窗口的数字**，如实报告"当前观测窗口不稳定，建议避开
  明显的周期性事件重新采样"，不要硬凑结论。

## 3. 核心指标清单

| 指标 | 含义 | 用途 |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | 传统"利用率"（有任意 kernel 在跑就算忙） | 只做参考，**不能单独下结论**——高利用率不代表真的在算东西，见第4节案例 |
| `DCGM_FI_PROF_SM_ACTIVE` | 有 SM 在工作的时间占比 | 比 GPU_UTIL 准一点，但仍需配合 OCCUPANCY 一起看 |
| `DCGM_FI_PROF_SM_OCCUPANCY` | 活跃周期里 warp 占用率 | **kernel-launch bound 的直接信号**：GPU_UTIL 高但这个持续 <5% = kernel 太碎太小 |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core（matmul）活跃占比 | 比自己算的 MFU 更可信的"类 MFU"信号——硬件计数器直出，不依赖手算 FLOPs 公式（呼应 `metric_interpretation.md` §0 对 MFU 计算风险的警告） |
| `DCGM_FI_PROF_DRAM_ACTIVE` | 显存带宽利用率 | 排查是不是 memory-bandwidth bound |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | 卡间 NVLink 带宽 | 单机多卡通信是否跑满 |
| `DCGM_FI_PROF_PCIE_TX_BYTES` / `PCIE_RX_BYTES` | PCIe 收发带宽 | 主机-设备/跨节点通信是否是瓶颈，不用等 nsys 的 NCCL 统计 |
| `DCGM_FI_DEV_XID_ERRORS` | GPU 硬件错误计数 | 排除"是不是硬件本身有问题"，跟代码无关的干扰项 |
| `DCGM_FI_DEV_GPU_TEMP` / `DCGM_FI_DEV_POWER_USAGE` / `*_VIOLATION`（`THERMAL_VIOLATION`/`POWER_VIOLATION`等） | 温度/功耗及各类降频原因计数 | 排查是不是被动降频拖慢速度，跟训练代码无关 |

## 4. 判断分支（结合窗口内 mean 值）

| 观测组合 | 判断 | 下一步 |
|---|---|---|
| GPU_UTIL 高/接近100%，但 SM_ACTIVE / SM_OCCUPANCY / TENSOR_ACTIVE 全部 <5% | **kernel-launch bound**（已用真实案例验证：GPU_UTIL 100% 时 SM_OCCUPANCY 可以低到 0.04%，TENSOR_ACTIVE=0%） | 值得抓 nsys——但目的从"猜方向"变成"定位具体是哪些算子"，抓之前心里已经有预期 |
| GPU_UTIL 低 + SM_ACTIVE/OCCUPANCY 也低 | 真的在空闲等（数据/通信/同步），不是 kernel 太碎 | 不必抓 nsys，先查 dataloader 并发度/NCCL 配置（走 `bottleneck_playbook.md` 数据瓶颈/通信瓶颈类方向） |
| GPU_UTIL 高 + SM_ACTIVE/OCCUPANCY 也高（如 >50%）但 TENSOR_ACTIVE 低 | 计算密集但没吃到 Tensor Core（可能是 fp32 未走 TF32/BF16 路径、算子本身非 matmul 密集、tile 大小没对齐） | 查精度设置/算子实现，方向跟 kernel-launch bound 不同，别混着调 |
| DRAM_ACTIVE 高，SM_ACTIVE/TENSOR_ACTIVE 低 | memory-bandwidth bound | 走"融合算子减少显存往返读写"方向，跟调 batch size 无关 |
| NVLINK/PCIE 带宽持续跑满（接近硬件峰值带宽） | 真正的通信瓶颈，直接坐实，不用等 nsys 的 NCCL 耗时占比 | 走 `bottleneck_playbook.md` 通信瓶颈类调整方向 |
| `XID_ERRORS` 非零 / 频繁 `*_VIOLATION` / 温度接近降频阈值 | 硬件/环境层面的干扰，不是代码问题 | 先处理硬件问题（换节点/报修/降低功耗墙），不要在这个状态下调训练参数，数据不可信 |

## 5. 脚本化查询模板（Python，避免手写 curl 转义）

### 前置：先用 kubectl 确认任务确实存在、Pod 在跑，再找到对应节点

跟第 1 节"先决条件"呼应——查询前务必先确认任务是真实存在且在运行的，不要
凭猜测/凭 yaml 里的 job 名直接查，否则 Prometheus 无数据时会误判成"任务卡住/
GPU 空闲"。

```bash
export KUBECONFIG=<你的kubeconfig路径>
# 确认任务/Pod 是否存在、状态如何（不存在或非 Running 就不用往下查了）
kubectl get pod -n <你的任务所在namespace> -l <适配你项目的label selector> -o wide
# 或者如果用的是 KtrainJob/其他 CRD，直接查对应资源
kubectl get <你的任务CRD类型> -n <namespace> <job_name> -o yaml | grep -i "pod\|node\|status"
```

从输出的 `NODE` 列获得节点名（不同集群命名规则不同，不要假设格式）。

### 脚本模板

```python
import subprocess, json, time

END = int(time.time())
START = END - 900  # 15分钟窗口，按第2节规则调整
NODE = "<上一步获得的节点名>"
METRICS = ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_SM_ACTIVE",
           "DCGM_FI_PROF_SM_OCCUPANCY", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
           "DCGM_FI_PROF_DRAM_ACTIVE"]

print(f"查询节点: {NODE}")
print(f"时间窗口: {time.ctime(START)} ~ {time.ctime(END)}\n")

for metric in METRICS:
    query = f'{metric}{{Hostname="{NODE}"}}'
    out = subprocess.run(
        ["curl", "-s", "http://localhost:19090/api/v1/query_range",
         "--data-urlencode", f"query={query}",
         "--data-urlencode", f"start={START}",
         "--data-urlencode", f"end={END}",
         "--data-urlencode", "step=15"],
        capture_output=True, text=True).stdout
    data = json.loads(out)
    if not data["data"]["result"]:
        print(f"{metric}: 无数据（任务可能未运行或刚启动——回到第1节先确认任务状态）")
        continue
    for r in data["data"]["result"]:
        vals = [float(v[1]) for v in r["values"]]
        if not vals:
            continue
        gpu = r["metric"].get("gpu", "all")
        mean_val = sum(vals) / len(vals)
        max_val = max(vals)
        print(f"{metric:40s} gpu{gpu}: mean={mean_val:7.2f}% max={max_val:7.2f}% samples={len(vals)}")
```

## 6. 输出规范（融入主流程的 `[核心指标]`/`[诊断]` 段落）

### 任务已提交并在运行时

```
[Prometheus快速预判]  窗口=<Xmin>（交叉验证：<Y min>窗口一致/不一致）
                      GPU_UTIL均值=<..%> / SM_ACTIVE均值=<..%> /
                      SM_OCCUPANCY均值=<..%> / TENSOR_ACTIVE均值=<..%>
                      判断：<launch-bound|数据/通信等待|计算未吃满TensorCore|
                            显存带宽bound|通信bound|硬件干扰>
```

### 任务还未提交或刚提交还未启动时

```
[Prometheus快速预判]  无法执行 — 任务未提交或未启动（Prometheus 无历史数据）
                      → 提交任务后，运行 2-3 步再查询 DCGM 指标
                      → 查询步骤见第 1 节和第 5 节
```

### 集群不具备 Prometheus/DCGM 时

```
[Prometheus快速预判]  集群不具备该能力（跳过）
```

不要省略这一行（让读者知道这层预判是"没做"而不是"做了但没写"）。
