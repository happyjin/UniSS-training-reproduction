---
name: pytorchjob-ops
description: 纯 kubectl PyTorchJob 运维（Kubeflow PyTorchJob，无 anyscale / ray / ktrain 运行时）：任务提交 / 状态进度查询 / 日志拉取与诊断 / 节点健康与资源管理 / 代码同步与清理。（training-ops 插件）
---

# pytorchjob-ops

本 skill 提供以**纯 `kubectl`** 提交并运维 Kubeflow `PyTorchJob` 的标准流程，不依赖 anyscale / ray / ktrain 任何运行时。适用于「直接用 kubectl 管理 PyTorchJob」的 Kubernetes 训练平台。

本 skill 是**通用**的：集群名、命名空间、镜像、节点白名单、对象存储 bucket、job 名等**均不在 skill 内硬编码**，而是来自项目自身的 `pytorch_job/pytorchjob.yaml`（单一事实源，所有配置已直接写进该 yaml）。提交与数据就绪门控逻辑在本 skill 的 `scripts/`：`submit.sh` 读取训练 yaml，并由 `derive_sync_jobs.py` 从中派生出数据同步所需的 4 个串行 sync Job（model/tokenizer/manifest/audio）。

根据用户输入自动路由到对应参考文件，按需加载。

---

## 0. 意图识别路由

| 命中关键词 | 场景 | 加载文件 |
|---|---|---|
| `提交` `submit` `跑训练` `启动` `重新提交` `重跑` `pytorchjob` `apply` | 提交任务 | `references/submit.md` |
| `节点` `node` `白名单` `health` `cordon` `资源` `GPU` `Pending` | 节点健康 / 资源排查 | `references/node_health.md` |
| `进度` `progress` `step` `loss` `还要多久` `ETA` `跑到哪` | 查询进度 | `references/progress.md` |
| `MFU` `mfu` `算力利用率` `FLOPs 利用率` `吞吐` `TFLOP` `喂数瓶颈` `data-starved` `GPU 利用率低` | 估算 MFU / 效率诊断 | `references/mfu.md` |
| `失败` `FAILED` `挂了` `排查` `报错` `error` `traceback` `日志` | 失败诊断 / 日志查询 | `references/diagnose.md` |
| `同步代码` `sync_code` `代码` `更新代码` | 文件操作（代码同步到对象存储 CSI） | `references/file_ops.md` |
| `终止` `kill` `cancel` `删` `清理` `delete` `teardown` | 终止 / 清理任务 | `references/terminate.md` |

---

## 1. 通用约束（重要）

- **配置唯一真源 = `pytorch_job/pytorchjob.yaml`**：镜像、命名空间、节点白名单、模型 / tokenizer / manifest / 原音频的 S3 路径、S3 CSI PVC 名、NVMe hostPath 全部写在该 yaml 中（仅 `__RUN_ID__` / `__JOB_NAME__` 为运行时占位符）。skill 不假定具体值，提交前先读取它。
- **提交与同步逻辑在 skill 内**：`scripts/submit.sh` 统一编排——同步代码到 S3、读 yaml 派生 4 个串行同步任务（model/tokenizer/manifest/audio）、节点健康预检、四任务严格串行 `aws s3 sync --size-only`（`completions=N` 门控确认全部节点落盘）后 apply 训练 yaml。同步 Job 由 `scripts/derive_sync_jobs.py` 把训练 Master.template 深拷贝为 `batch/v1 Job` **推导生成**（不在业务仓库里维护独立的同步任务配置）。
- **工件 S3 源（去掉 `s3://<bucket>/` 头即为 `/nvme/<s3-key>`，与 NVMe hostPath 1:1）**：
  - 模型 + tokenizer：`s3://iegg-speech-model-models/origin_models/...`（模型桶，PVC `speech-h200-s3-models-pvc`）
  - manifest + 原音频：`s3://iegg-speech-model-data-.../data/...`（数据桶，PVC `speech-h200-s3-pvc`）
  - 训练入口 `TRAIN_SCRIPT` 是 **S3 路径**（容器内经 S3 CSI 直接读，无需拷贝到本地）。
- **容器路径 = `/nvme/<s3-key>`**：NVMe hostPath 挂载在 `/nvme`，无隐式路径映射潜规则；WDS 音频经 `/curated_audio -> /nvme/data` 符号链接命中。
- **节点白名单固定**：白名单由训练 yaml 的 `affinity` 固定，不可随意更改；不要靠换节点规避训练侧问题（如 OOM）。
- **Pod 可能无云凭证**：只读输入走 CSI 驱动（S3 CSI），**不要在 pod 里用云 CLI 查权限**。所有诊断信息通过 `kubectl` / 云 CLI（在本地/跳板执行）远程获取。
- **外部自动化可能重建重复 job**：某些平台存在外部 controller 删了又建带后缀的重复 PyTorchJob 抢 GPU，需排查并清理（见 node_health.md / terminate.md）。
- 所有诊断信息通过 `kubectl` / 云 CLI 远程获取，不读本地仓库文件作为诊断依据。
- 变更类操作（提交、删除、cordon、覆盖式 `sync_code`）执行前必须二次确认，除非用户给出明确执行指令（含具体 job 名 / 清单路径、无疑问词）。

---

## 2. 参考文件清单

| 文件 | 用途 |
|---|---|
| `references/submit.md` | 提交流程：读 `pytorchjob.yaml`、节点健康预检、四任务串行同步（model/tokenizer/manifest/audio，completions=N 门控）、apply PyTorchJob |
| `references/node_health.md` | 节点健康与白名单排查、GPU 资源、Pending 诊断、重复 job 检测 |
| `references/progress.md` | 进度查询：pod 状态、日志 step/loss/ETA 解析 |
| `references/mfu.md` | MFU 估算：用 checkpoint 保存时间戳反推 step_sec，算算力利用率、判定 data-starved |
| `references/diagnose.md` | 失败诊断：全量日志本地分析纪律、通用失败类别、修复方向 |
| `references/file_ops.md` | 代码同步到对象存储 CSI + 模型/tokenizer/manifest/音频同步到节点 NVMe |
| `references/terminate.md` | 终止 / 清理任务（delete PyTorchJob / pod） |

脚本位于本 skill 的 `scripts/`（`pytorchjob-ops/scripts/`）：
- `submit.sh` — 提交编排（读 yaml → 派生同步 → 数据门控 → apply 训练 yaml）
- `derive_sync_jobs.py` — 从 `pytorchjob.yaml` 推导 sync Job yaml
- `sync_code.sh` — repo 同步到 data bucket（S3 CSI）
- `sync_artifact.sh` — 单工件 `aws s3 sync --size-only` 到节点 NVMe
- `query_job.sh` `inspect_nodes.sh` `collect_logs.sh` `estimate_mfu.sh` — 查询 / 诊断

脚本均**参数化**，不内置具体集群 / job 名（除 `submit.sh` 会从训练 yaml 解析所需值）。

---

## 3. 输出规范

诊断 / 查询结果严格按对应 reference 中定义的输出模板返回。提交结果重点汇报：PyTorchJob 名、ns、目标节点（白名单）、镜像、监控命令——这些值均取自训练 yaml，不臆造。
