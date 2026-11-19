---
name: anyscale-job-ops
description: Anyscale prodjob（Ray 训练/ETL 任务）全生命周期管理：提交 / 进度查询 / 失败诊断 / 节点健康 / 终止任务。支持 Anyscale Console URL 触发。（training-ops 插件）
---

# anyscale-job-ops

本 skill 提供 Anyscale prodjob 的标准诊断流程。根据用户输入自动路由到对应参考文件，按需加载，避免无关内容占用上下文。

---

## 0. 意图识别路由

读取用户输入，按以下顺序匹配，**首个命中即确定场景**：

| 命中关键词 | 场景 | 加载文件 |
|---|---|---|
| `提交任务` `提交 job` `submit job` `跑训练` `跑 ETL` `启动任务` `启动 job` `重新提交` `重跑` `anyscale` `prodjob` | 提交任务 | `references/submit.md` |
| `节点健康` `node health` `资源调度` `资源不足` `卡在调度` `Insufficient` `节点状态` `GPU 故障` `GPU 降级` `核实节点` `恢复节点` `重启device-plugin` | 节点健康诊断 + 可选恢复 | `references/node_health.md` |
| `进度` `progress` `step` `loss` `token_acc` `还要多久` `ETA` `跑到哪` `跑完了吗` | 查询进度 | `references/progress.md` |
| `失败` `FAILED` `挂了` `起不来` `排查` `报错` `error` `traceback` | 失败诊断 | `references/diagnose.md` + 按需追加 `references/failure_patterns.md` |
| `终止` `kill` `cancel` `terminate` `停掉` `停止任务` | 终止任务 | `references/terminate.md` |

> **路由优先级说明**：节点健康诊断排在失败诊断之前。因为用户说「卡在调度」「资源不足」时，根因通常是节点层问题，此时应先做节点健康诊断而非拉日志排查。如节点健康诊断未发现问题，再由用户手动触发失败诊断。

### 仅提供 prodjob_id 但意图不明

向用户确认意图后再加载对应文件：

```
选项：查看进度 / 节点健康 / 排查失败 / 终止任务
```



## 1. 触发方式

### 关键词触发
用户消息包含上述路由表中的关键词时自动触发本 skill。

### URL 触发
用户粘贴 Anyscale Console URL 时触发，格式如下：

```
https://console.anyscale.com/.../jobs/prodjob_xxx
```

提取 `prodjob_xxx` 作为任务标识符，**默认路由到进度查询流程**（用户粘贴 URL 大多是想查进度）。

**判断依据**（按优先级）：
1. 用户消息中同时包含失败关键词（`FAILED` / `失败` / `挂了` / `报错`）→ 路由到失败诊断
2. URL 的 hash 段包含 `job-tab=metrics` 或用户明确说「进度」→ 路由到进度查询
3. 用户消息中同时包含终止关键词（`终止` / `kill` / `停掉`）→ 路由到终止任务
4. 以上都不命中 → 默认路由到**进度查询**

---

## 2. 执行约束

### 禁止检索本地仓库文件
本 skill 的所有诊断信息必须通过以下方式远程获取，不得读取本地仓库中的配置文件、脚本、日志：
- `anyscale` CLI 命令
- Anyscale REST API
- `kubectl`（针对 Anyscale 管理的 k8s 集群）

### 变更操作需确认
以下操作属于变更类，执行前必须向用户二次确认：
- 提交新任务（`anyscale job submit`）
- 终止任务（`anyscale job terminate`）
- 修改集群配置
- 重启 device-plugin 恢复节点

只读操作（查询进度、拉取日志、诊断失败、节点健康诊断）无需确认。

**例外：用户意图明确时跳过确认。**
当用户的输入是一个明确的执行指令（而非咨询或探索性提问）时，
直接执行，不再询问确认。典型场景：
- 用户明确给出了 Job YAML 路径，如 `提交任务 anyscale/train/job.yaml`
- 用户明确给出了 prodjob_id，如 `终止任务 prodjob_abc123`
- 用户的指令中没有疑问词（如何、能不能、是否可以等）

反之，当用户问 `我想提交训练任务，该怎么做？` 或 `帮我看看这个配置能不能提交`
这类咨询/探索性问题时，仍需先展示计划再请求确认。

### 任务名
任务名格式为三段式：`<模型名>-<user>-<4位随机字符>`，自动生成，提交结果中展示即可。
用户可通过 `--model-name`（改模型名）或 `--name`（完全自定义）调整，但**禁止主动询问或要求用户确认任务名**。

### 镜像/计算配置版本
不得硬编码镜像 tag、compute config 版本、节点数量。从远程 `job.yaml` 或 Anyscale API 现场解析当前值。

---

## 3. 参考文件清单

| 文件 | 用途 |
|---|---|
| `references/submit.md` | 提交流程步骤、参数说明、dry-run 方式 |
| `references/node_health.md` | 节点健康诊断：kubectl 连通性、GPU 可分配数、故障标签、黑名单交叉对比、Pending pod 调度失败原因、GPU 降级节点恢复（重启 device-plugin） |
| `references/progress.md` | 进度查询方式、ETA 计算方法、健康状态判断 |
| `references/diagnose.md` | 诊断流程步骤、日志收集、失败阶段定位 |
| `references/failure_patterns.md` | 常见失败模式速查表（错误关键词 → 根因 → 修复建议） |
| `references/terminate.md` | 终止流程步骤、确认模板 |

脚本位于 `.codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/`：
`submit_helper.sh` `query_progress.sh` `inspect_p5_nodes.sh` `collect_logs.sh` `locate_failure.sh` `baseline_check.sh`

---

## 4. 输出规范

### 提交任务
必须按 `references/submit.md` 中 Step 6 的模板汇报，重点包括：
- **任务名**（`<模型名>-<user>-<4位随机>` 三段式）
- **Prodjob ID**
- **Console 链接**
- **初始状态**

### 诊断结果（节点健康 / 进度查询 / 失败诊断 / 终止任务）
按以下结构返回给用户：

```
## 诊断摘要
- 任务 ID（如有）：
- 状态：
- 失败阶段（如有）：

## 详细信息
（按对应 reference 中的模板填充）

## 建议操作
（修复建议或下一步操作）
```

**节点健康诊断**必须按 `references/node_health.md` 中 Step 6 的模板汇报，重点包括：
- 集群名称和区域
- 逐节点 GPU 状态表（GPU 可分配、Ready、Fault、黑名单）
- Pending pods 及 FailedScheduling 原因
- 根因分析（可用节点数 vs 需求）
- 具体建议操作

如果执行了节点恢复操作（Step 7），还需汇报：
- 恢复前 vs 恢复后的 GPU allocatable 对比
- 恢复成功的节点列表
- 恢复失败的节点及原因（如有）

不得自由发挥格式，严格使用对应 reference 中定义的输出模板。
