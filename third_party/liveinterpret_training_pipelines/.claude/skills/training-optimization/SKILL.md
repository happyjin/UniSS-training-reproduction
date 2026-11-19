---
name: training-optimization
description: This skill should be used to diagnose the health and efficiency of a machine learning training run from its basic runtime metrics (loss trend, step time / s-per-iter, remaining-time ETA, GPU memory), and to translate that diagnosis into concrete training-parameter adjustments (batch size, parallelism/sharding stage, dataloader concurrency, gradient accumulation, kernel fusion, etc.). It should be used for requests like "这个任务跑得健不健康", "为什么 s/it 这么慢", "还要跑多久正常吗", "显存是不是要炸了", "帮我把这个训练调快/调稳", "对比一下这两次实验的效率/scaling efficiency", or "根据 nsys 报告调一下参数". When the metric-level diagnosis is inconclusive or points to a GPU/kernel-level bottleneck, this skill escalates to a dedicated Nsight Systems (nsys) analysis skill (if one is available in the environment) before finalizing recommendations, then produces a ready-to-review parameter-change variant of the job's training configuration for the user to confirm before submission.
---

# training-optimization

本 skill 提供**训练任务效率诊断 → （按需）nsys 下钻 → 参数调整方案**的标准流程。目标是把"s/it、remaining_time、显存"这类基础运行指标（loss 作为正确性维度附带检查；MFU/TFLOP-s 仅在计算方法已验证时作辅助参考，见 `references/metric_interpretation.md` §0）转成**可执行的训练参数调整建议**，而不是停留在"看起来还行/不太行"的模糊结论。

**本 skill 本身不含任何具体框架的字段名（不绑定某个 yaml/CLI 的具体 key）**——具体参数名要在诊断阶段从用户当前项目的任务定义/启动脚本里现查，方法论是通用的。

---

## 0. 何时使用 / 何时不用

**用**：训练效率、速度、显存、多机扩展性方面的诊断与调参。

**不用**：
- 纯模型质量/收敛正确性问题（loss 数值是否合理、指标是否符合预期）——这类问题只在"loss 出现异常"时作为附带发现提及，不是本 skill 的主战场。
- 提交/停止/查日志等任务运维本身——那是 job-ops 类 skill（如 `ktrain-job-ops`/`anyscale-job-ops`/`pytorchjob-ops`）的职责，本 skill 只**调用**它们取数据、并在最后请它们提交生成的变体任务。
- GPU kernel/NCCL 层面的详细分析——那是 `nsys-analyze` 的职责，本 skill 只负责判断"要不要下钻到这一层"并把结论folding回参数建议。

## 1. 标准流程

```
确定数据来源 → 拉取四项核心指标 → 逐项诊断
   → （若集群有 Prometheus/DCGM）先做一次轻量预判 → 判断是否需要 nsys 下钻
   → （按需）调用 nsys-analyze → 汇总根因 → 生成参数调整方案
   → 生成任务变体（diff 呈现）→ 请用户确认 → 确认后交给对应 job-ops skill 提交
```

### Step 1 — 确定数据来源

两类来源，按用户表述自动选（不确定时都尝试，交叉验证）：

1. **实时任务**：用户提到具体任务名/"这个任务"/"刚提交的"/"正在跑的" → 找当前环境里可用的 job-ops 类 skill（`ktrain-job-ops` / `anyscale-job-ops` / `pytorchjob-ops` 等，视平台而定），用其"进度查询/日志"能力拉最新状态与日志。
2. **历史文档**：用户提到"这份报告"/"对比一下 A 和 B"/给了具体文档路径 → 直接读文档里整理好的指标表（如某种 `*_METRICS.md`/`*_RESULTS.md`），不必重新连集群。
3. 两者都可用时，优先以**实时日志**为准（更新鲜），历史文档作为基线/对照。

详见 `references/data_sources.md`。

### Step 2 — 拉取并规整四项核心指标

不管来源是日志还是文档，统一整理成这四类。**优先级：s/it > remaining_time
（含 scaling efficiency） > 显存 > loss**（loss 是正确性维度，异常必须报，但
跟效率判断是两件事）。MFU/TFLOP-s **不是四项核心指标之一**，只在其计算方法
已验证覆盖当前模型结构时作为辅助参考，未验证时不作为结论依据（见
`references/metric_interpretation.md` §0）。

| 指标 | 关注点 |
|---|---|
| **s/it（step time）** ⭐ | 冷启动值 vs 稳态值要分开看，只用稳态窗口做判断——最直接、最不容易算错的效率信号 |
| **remaining_time / ETA** ⭐ | 是否由稳态速率外推（不是被冷启动污染的累计均值）；多机场景要能算 scaling efficiency |
| **显存** ⭐ | 峰值/稳态值相对显卡容量的余量；增长模式是"爬升后持平"（正常）还是"持续增长"（疑似泄漏） |
| **loss 轨迹** | 是否单调下降、有无 NaN/Inf/突刺、是否早早 plateau |

计算口径、稳态窗口取法、scaling efficiency 公式，见 `references/metric_interpretation.md`（**必读，包含本 skill 的核心量化方法**）。

### Step 3 — 逐项诊断，给出健康度结论

对照 `references/metric_interpretation.md` 的判定阈值/模式，给出：

- 每个维度（收敛/速度/显存/扩展性）：正常 / 亚健康 / 异常，并附具体数字证据
- 一句话总诊断（例如："速度正常，显存有 15% 余量，双节点 scaling efficiency 91%，健康；" 或 "s/it 比同配置基线慢 2 倍，怀疑数据读取瓶颈，建议 nsys 下钻确认"）

### Step 3.5 — （若集群有 Prometheus/DCGM）先做一次轻量预判

在决定要不要抓 nsys 之前，先看集群是否有 Prometheus + DCGM exporter
（`kubectl get svc -A | grep -i prometheus`）。有的话，用
`references/prometheus_dcgm_quickcheck.md` 的方法秒级查一次
`SM_ACTIVE`/`SM_OCCUPANCY`/`TENSOR_ACTIVE`/`DRAM_ACTIVE` 等硬件计数器指标——
**零改动、不用等到指定 step、不用是自己提交的任务**，通常能直接把"要不要
下钻 nsys、下钻了大概会看到什么"提前判断出来，比单纯凭 s/it 猜方向可信得多
（已用真实案例验证：GPU_UTIL 显示 100% 时，SM_OCCUPANCY 可以低到 0.04%，
这种"看起来很忙、实际几乎没算"的模式是 kernel-launch bound 的直接证据）。

集群没有这个能力时，注明"跳过，集群不具备 Prometheus/DCGM"，直接进
Step 4，不强求。

### Step 4 — 判断是否需要 nsys 下钻

命中 `references/nsys_handoff.md` 里任一升级信号（如：s/it 显著劣于可比基线且指标层面查不出原因、疑似 GPU 空闲/kernel launch bound、需要精确算 MFU/TFLOP-s）→ 调用 `nsys-analyze` skill（若环境中已加载/可用）分析对应的 `.nsys-rep`；若还没有 profile 数据，先说明需要采集一次（采集方式让用户项目自己的训练框架决定，本 skill 不规定具体开关名），采集后再下钻。

nsys 结论（launch-bound / 通讯-bound / 计算-bound / idle 高）要**折算回 Step 5 的参数调整类别**，不要只停在"nsys 说这里慢"。nsys 报告里的 MFU/TFLOP-s 同样受 §0 的计算方法风险影响，交叉验证方式是看 kernel 总耗时/idle 占比这类直接测量值，而不是单看 MFU 数字本身。

### Step 5 — 归因 → 参数调整方案

用 `references/bottleneck_playbook.md` 的对照表，把 Step 3/4 的根因映射到**参数调整类别**（不是具体 key，具体 key 要现查用户项目的任务定义/启动脚本）。每条建议必须包含：

1. 根因（引用 Step 3/4 的具体证据）
2. 调整方向（调大/调小/切换到哪种模式）
3. 预期效果（量级估计，不夸大）
4. 风险/副作用（例如调大 batch 可能涨显存、调小可能拖慢收敛）

### Step 6 — 生成任务变体，交给用户确认

**不直接改用户现有的任务定义文件**，生成一个新的变体（复制一份并只改 Step 5 建议的那一两个参数，文件名/头部注释清楚标注"改了什么、为什么改、对照哪个基线"——一次只验证一个变量，除非用户明确要求组合改动）。

按 `references/variant_workflow.md` 的模板：
1. 生成变体文件，展示与原文件的 diff
2. 汇报预期效果和验证方式（要看哪个指标、大概等多久能看出结果）
3. **明确询问用户是否要提交**（本 skill 不自动提交）
4. 用户确认后，调用对应的 job-ops skill 完成提交，并告知用户如何跟踪进度

## 2. 参考文件清单

| 文件 | 用途 |
|---|---|
| `references/data_sources.md` | 如何选择/组合实时日志与历史文档两类数据源 |
| `references/metric_interpretation.md` | ⭐ 四项核心指标的量化方法：稳态窗口、scaling efficiency 公式、显存余量阈值、loss 异常模式 |
| `references/bottleneck_playbook.md` | ⭐ 根因 → 参数调整类别对照表（数据瓶颈/显存瓶颈/通讯瓶颈/launch瓶颈/扩展性瓶颈） |
| `references/nsys_handoff.md` | 何时该升级到 nsys-analyze、如何把 nsys 结论折算回参数建议 |
| `references/prometheus_dcgm_quickcheck.md` | ⭐ nsys 下钻前的轻量预判：Prometheus/DCGM 硬件计数器指标（SM_ACTIVE/OCCUPANCY/TENSOR_ACTIVE等）、窗口选择规则、判断分支表 |
| `references/variant_workflow.md` | 任务变体生成规范、diff 呈现、确认后提交的流程 |

## 3. 输出规范

```
[数据来源]      实时任务 <job_name> / 历史文档 <path>（或两者对照）
[核心指标]      稳态s_it（窗口说明）/ remaining_time（外推方式）/ 显存（峰值+余量）/ loss@<step>（MFU 若未验证计算方法则不列或注明"仅供参考"）
[诊断]          速度：正常|亚健康|异常；扩展性：<efficiency>%（若适用）；显存：正常|紧张|异常；收敛：正常|异常
[Prometheus预判] 若集群支持——窗口/交叉验证结果/关键指标均值/判断类别（若不支持，注明"跳过"）
[nsys 结论]     若触发下钻——launch/通讯/计算/idle 判断 + 关键数字（若未触发，注明"未触发，原因"）
[根因]          最可能的1-2个瓶颈，附具体证据数字
[参数调整方案]  参数类别 + 调整方向 + 预期效果 + 风险
[任务变体]      变体文件路径 + diff + 是否提交（等待用户确认）
```
