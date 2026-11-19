---
name: ktrain-job-ops
description: ktrain 平台（KtrainJob CRD，ktrain.oit-svc.com/v1alpha2，operator 渲染为 PyTorchJob）训练任务运维：任务提交 / 进度查询 / 失败诊断 / 节点健康排查 / 代码与工件同步 / 停止与删除；任务定义管理（validate / manifest / export / schema / explain）、dev workspace、队列与存储操作。
---

# ktrain-job-ops

用 ktrain CLI 运维 KtrainJob 训练任务。任务操作（提交 / 停止 / 删除 / 状态 / 日志 / 列表 / 导出 / 校验 / 渲染）、节点 / GPU / 队列排查、文件系统与存储操作一律用 `ktrain` CLI；只有 `ktrain` CLI 不覆盖的功能（查 Pod 详情、查节点详情、删 Pod、PVC→PV→bucket 解析、恢复已停止任务）才退到 `kubectl`。本 skill 不含任何环境值（集群名、区域、命名空间、bucket、节点名、本地路径），全部从传入的任务 yaml 或显式参数获取。

`SKILL_DIR` = 本 SKILL.md 所在目录。脚本调用写法：`bash <SKILL_DIR>/scripts/<脚本>`。

## CLI 版本基线

- 以 ktrain CLI **1.7.10** 为准（`ktrain --version` 核查）；低于此版本先 `ktrain update` 升级。
- 1.7.x 关键能力：
  - `ktrain job stop`：停止任务（旧版 CLI 没有，只能 kubectl patch）。
  - `ktrain job export` / `job manifest` / `job schema` / `ktrain explain`：任务定义导出重提、渲染预览、字段查询。
  - `ktrain workspace` + `ktrain ssh-keys`：code-server dev 开发机与远程 SSH。
  - `ktrain queue`：Kueue 队列 / 配额管理（list / show / workloads / flavors / use；其余 admin）。
  - `ktrain storage sync` / `s3-sync` / `s3-cp` / `presign-upload` / `presign-download`：存储快路径。
  - 全局 `--cluster/-c`（env `KTRAIN_CLUSTER`）与 `-o json`（env `KTRAIN_OUTPUT`；list / status / info / show 机器可读）。
  - `ktrain job logs --offset`：已结束任务持久化日志按行翻页。

## 任务 yaml 契约（skill 的唯一输入接口）

| 配置项 | 在任务 yaml 中的位置 |
|---|---|
| 命名空间 | `metadata.namespace`（必填，缺失即报错） |
| 镜像 | `spec.image` |
| 节点数 N | `spec.numNodes`，必须等于白名单长度 |
| 节点白名单 | `metadata.annotations["ktrain.oit-svc.com/render-extra"]` JSON → `affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution` 中 `kubernetes.io/hostname In` 的 values |
| 卷 / 挂载 / 调度 | render-extra JSON → `extra_volumes` / `extra_volume_mounts` / `node_selector` / `tolerations` / `service_account_name` |
| 训练资源 | `spec.resources`（平铺 `cpu`/`memory`/`gpu`） |
| 工件声明 | `spec.env` 中以 `s3://` / `s3mnt://` / `fsx://` 开头的值（空格分隔可多值） |
| NVMe 挂载点 | `spec.env.NVME_MOUNT`（存在 `s3://` / `s3mnt://` 工件时必填） |
| FSX 挂载点 | `spec.env.FSX_MOUNT`（存在 `fsx://` 工件时必填） |
| s3mnt 源挂载点 | `spec.env.S3ALL_MOUNT`（存在 `s3mnt://` 工件时：未设置且 render-extra 中只有一个 PVC 卷则自动取该卷挂载点，多个 PVC 卷时必须显式设置） |
| 代码前缀 | `spec.env.CODE_PREFIX`（代码同步必填） |
| 代码 CSI 卷 | render-extra `extra_volumes` 中的 PVC 卷；仅一个时自动采用，多个时必须用 `--pvc` 指定 |
| 运行时占位符 | `__RUN_ID__` / `__JOB_NAME__`，由 submit.sh 渲染 |

工件 URI scheme（sync Job 按此派生）：

- `s3://<bucket>/<key>` → 提交前经 `aws s3 sync --size-only` 同步到每个白名单节点的 `<NVME_MOUNT>/<key>`
- `s3mnt://<key>` → 源已在 S3 CSI 挂载点上，从 `<S3ALL_MOUNT>/<key>` 本地增量同步（rsync/cp/tar）到 `<NVME_MOUNT>/<key>`
- `fsx://<path>` → 不产生同步任务，训练经 FSX CSI 直读 `<FSX_MOUNT>/<path>`

## 意图识别路由

| 命中关键词 | 场景 | 加载文件 |
|---|---|---|
| `提交` `submit` `跑训练` `启动任务` `重新提交` `重跑` | 提交任务 | `references/submit.md` |
| `进度` `progress` `step` `loss` `ETA` `还要多久` `跑到哪` `跑完了吗` | 进度查询 | `references/progress.md` |
| `失败` `FAILED` `挂了` `报错` `error` `traceback` `日志` `排查` `诊断` | 失败诊断 | `references/diagnose.md` |
| `节点` `node` `GPU` `资源` `Pending` `白名单` `cordon` `health` `队列` `queue` `配额` `Kueue` | 节点 / 资源 / 队列排查 | `references/node_health.md` |
| `同步代码` `sync_code` `更新代码` `同步模型` `同步工件` `上传` `下载` `存储` `storage` | 代码 / 工件同步与存储 | `references/file_ops.md` |
| `停止` `stop` `停掉` `暂停` `释放资源` `kill` `cancel` `清理` `删除任务` | 停止 / 删除任务 | `references/terminate.md` |
| `导出` `export` `克隆` `校验` `validate` `渲染` `manifest` `schema` `字段` `explain` | 任务定义管理 | `references/job_yaml.md` |
| `开发机` `workspace` `code-server` `远程开发` `web IDE` `ssh-keys` | dev workspace | `references/workspace.md` |

## 职责边界

本 skill 只做：提交、进度查询、失败诊断、节点 / 队列排查、代码 / 工件同步、停止 / 删除、任务定义管理（validate / manifest / export / schema / explain）、dev workspace 与存储操作。训练超参、训练脚本、pipeline 代码的修改不属于本 skill，诊断给出根因与证据后交还用户处理。

## 通用禁令

- 禁止硬编码任何环境值；命名空间、节点白名单、镜像、bucket、挂载点一律从任务 yaml 解析，集群上下文一律用当前 ktrain / kubectl context（或显式 `--cluster` / `ktrain -c` 参数）。
- 任务操作（提交 / 停止 / 删除 / 状态 / 日志 / 列表 / 导出 / 校验 / 渲染）、节点 / GPU / 队列排查、文件系统与存储操作一律用 `ktrain` CLI；`ktrain` CLI 不覆盖的（查 Pod / 节点详情、删 Pod、PVC 解析、恢复已停止任务的 `stop=false` patch）才用 `kubectl`。
- 同步 Job（`batch/v1 Job`，非 KtrainJob）的 apply / delete / 查终态用 `kubectl`（ktrain CLI 不管理此资源）。
- 所有诊断信息通过 `ktrain` / `kubectl` / `aws` 远程获取，禁止拿本地仓库文件当诊断依据。
- 用户指令清晰明确（含具体任务名 / yaml 路径等关键参数）时直接执行，不二次确认；关键参数缺失或有歧义时先问清楚再执行。
- 禁止靠改节点白名单规避训练侧故障。
- 禁止在 Pod 内用云 CLI 查对象存储权限（Pod 可能无云凭证）；权限问题查 CSI / SA。

## 脚本清单（`<SKILL_DIR>/scripts/`）

| 脚本 | 用途 | 用法 |
|---|---|---|
| `submit.sh` | 提交编排：同步代码 → 派生并串行执行同步 Job → 节点预检 → `ktrain job submit` | `submit.sh --yaml <任务yaml> --repo <本地代码库根> --launcher <入口脚本本地路径> [--job-name-prefix P] [--cluster C --region R] [--pvc PVC] [--aws-creds-secret S] [--deadline-sec N] [--task-deadline 任务名=秒]...` |
| `derive_sync_jobs.py` | 从任务 yaml 派生同步 Job yaml + `plan.env` | 由 submit.sh 调用，不直接执行 |
| `sync_code.sh` | 本地代码库覆盖式同步到代码 bucket 的 `CODE_PREFIX` | `sync_code.sh --yaml <任务yaml> --repo <本地代码库根> [--pvc PVC] [--bucket B] [--dest s3://...] [--cluster C --region R] [--exclude PATTERN]...` |
| `sync_artifact.sh` | 同步 Pod 内的单节点工件同步 | 由同步 Job 调用，不直接执行 |
| `query_job.sh` | 进度查询（status + 日志解析 step/loss/ETA） | `query_job.sh <JOB_NAME> <NAMESPACE>` |
| `collect_logs.sh` | 全量日志与状态下载到本地 | `collect_logs.sh <JOB_NAME> <NAMESPACE>` |
| `inspect_nodes.sh` | 集群 GPU / 节点 / 任务 / 文件系统排查 | `inspect_nodes.sh <NAMESPACE>` |

## 输出规范

诊断 / 查询结果严格按对应 reference 的输出模板返回；提交结果汇报任务名、命名空间、目标节点、镜像、监控命令，取值一律来自任务 yaml 与脚本输出，不臆造。
