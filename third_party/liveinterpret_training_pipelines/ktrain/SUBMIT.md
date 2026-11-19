# KtrainJob 提交指南（submit.sh 全流程）

> 本文记录在本地（macOS）通过 `submit.sh` 把训练任务提交到 ktrain 集群（EKS `speech-h200-eks`，
> HyperPod H200）的完整流程：前置环境、命令参数、各阶段日志含义、提交后监控，以及实际踩过的坑。
> 任务配置本身（yaml 写法）见 [README.md](README.md) 与 `pipelines/<family>/README.md`。

---

## 1. 前置条件（本机一次性准备）

| 依赖 | 安装 / 获取 | 验证 |
|------|------------|------|
| ktrain CLI | 仓库 `.venv`：`source .venv/bin/activate`（代理模式，集群 `speech-h200-hyperpod`） | `ktrain --version`、`ktrain auth status` |
| kubectl | `brew install kubectl` | `command -v kubectl` |
| kubeconfig | `aws eks update-kubeconfig --region ap-southeast-3 --name speech-h200-eks`；或代理模式 `ktrain auth kubeconfig` | `kubectl config current-context` 应为 `speech-h200-eks` |
| AWS 凭证 | `~/.aws` 已配置即可（同步数据、查 PVC 用） | `aws sts get-caller-identity` |

权限检查（`submit.sh` 需要在 `speech-h200` 命名空间创建 ConfigMap / Job / Pod、查 PVC 与节点）：

```bash
kubectl -n speech-h200 auth can-i create configmaps   # yes
kubectl -n speech-h200 auth can-i create jobs.batch   # yes
```

> 注意：必须在**同一个终端**里激活 `.venv` 并确认 `ktrain`、`kubectl` 都能找到再提交。
> `submit.sh` 的同步与提交步骤全部依赖这两个命令。

---

## 2. 提交命令

```bash
cd <repo 根目录>
source .venv/bin/activate
bash .codebuddy/training-ops-plugin/skills/ktrain-job-ops/scripts/submit.sh \
    --yaml ktrain/<任务yaml> \
    --repo . --launcher ktrain/launch_train.sh \
    --aws-creds-secret aws-local-creds \
    --job-name-prefix <前缀> \
    --task-deadline '<同步任务名或env名>=<秒>'
```

### 参数含义

| 参数 | 必填 | 含义 |
|------|------|------|
| `--yaml` | 是 | KtrainJob 任务定义。脚本从中解析命名空间、节点白名单、`s3://` 数据源（派生同步 Job）、`CODE_PREFIX`、环境变量 |
| `--repo` | 是 | 本地代码库根目录，上传到代码 bucket 的 `CODE_PREFIX` 下（`.git` / `.codebuddy` / `__pycache__` / 权重文件自动排除） |
| `--launcher` | 是 | 训练容器入口脚本，做成 ConfigMap 挂到容器 `/opt/launcher/`，即 yaml `command` 指向的文件 |
| `--job-name-prefix` | 否 | 任务名前缀，最终任务名 `<前缀>-<RUN_ID>`；`RUN_ID` 每次随机 4 位，重提交不重名。缺省用 yaml 文件名 |
| `--aws-creds-secret` | 否 | 集群里存 AWS 凭证的 Secret，同步 Job 的 Pod 用它访问 S3 |
| `--task-deadline` | 否 | 指定同步任务的等待超时（秒），可传多个；左边匹配任务名或对应环境变量名。未匹配的用默认 3600s |
| `--cluster` + `--region` | 否 | 必须成对。当前 kubectl context 不是目标集群时，先 `aws eks update-kubeconfig` 切换 |
| `--pvc` | 否 | yaml `render-extra` 有多个 PVC 卷时，指定代码同步用哪一个 |
| `--deadline-sec` | 否 | 所有同步任务的默认超时（默认 3600s） |

> **任务名占位符**：若 yaml 的 `metadata.name` 是 `__JOB_NAME__`，提交时替换为
> `<前缀>-<RUN_ID>`；若写死了具体名字（如 `my-train-job`），`--job-name-prefix`
> 不生效，且**每次重提交前必须先删旧任务**（`ktrain job delete <名字>`），否则重名冲突。

---

## 3. 提交流程（6 步，串行，任一步失败即中止，不会提交训练）

全程输出同时写本地 `/tmp/submit-ktrain-<RUN_ID>.log`（首行打印路径）。

| 步骤 | 做什么 | 成功日志标志 |
|------|--------|--------------|
| 准备 | 从 yaml 解析命名空间，生成 `RUN_ID` | `[submit] namespace=... prefix=... run_id=xxxx` |
| 1. 同步代码 | 从代码 PVC 查出代码 bucket，**先清空** `s3://<bucket>/<CODE_PREFIX>/` 再上传本地仓库 | `[sync] done -> s3://...` |
| 2. 派生同步任务 | 建 ConfigMap `artifact-sync-script-<RUN_ID>`；扫描 yaml 中所有 `s3://` 值，每个生成一个同步 Job | `artifacts derived: N`、`sync tasks (serial): ...` |
| 3. 预检节点 | 白名单节点全部 Ready 且未被 cordon | `pre-flight OK: all N node(s) Ready & schedulable` |
| 4. 同步数据 | 逐个执行同步 Job：每个在全部白名单节点各起一个 Pod，`aws s3 sync`（默认 32 并发）拉到节点 `/nvme/<s3-key>`；本地每 10s 轮询，要求所有 Pod 成功，之后采日志并删 Job | `step 4: === sync task [name] ===` → `<<< sync job ... complete on all N node(s)` |
| 5. 渲染 yaml | 替换 `__RUN_ID__` / `__JOB_NAME__`；把 `launch_train.sh` 做成 ConfigMap 注入 `render-extra`，挂到 `/opt/launcher` | `step 5: render ... -> <任务名>`、`injected launcher ConfigMap volume ...` |
| 6. 校验提交 | `ktrain job validate`，通过后 `ktrain job submit --yes` | `Job '<任务名>' submitted.` |

要点：

- **数据同步是增量的**：节点 `/nvme` 上已有的文件跳过，同一组节点第二次提交时第 4 步很快。
- **同步任务超时**：大工件（如几百 GB 音频）用 `--task-deadline` 单独给长超时，格式
  `<任务名或env名>=<秒>`（如 `si-s1-data=21600`，6 小时）。
- **代码不拷到节点**：训练容器经 S3 CSI（`/s3all`）直接读代码 bucket，`launch_train.sh` 按
  `CODE_PREFIX` 拼出 `TRAIN_SCRIPT` 的实际路径执行。
- **改了代码 / 超参脚本必须重跑 `submit.sh`**：它会重新上传代码（`launch_train.sh` 走 ConfigMap 也会刷新）。

---

## 4. 提交后监控

```bash
ktrain job status <任务名>
ktrain job logs <任务名> --follow
kubectl -n speech-h200 get pods -l training.kubeflow.org/job-name=<任务名> -o wide -w
ktrain job delete <任务名>      # 停止 / 清理
```

训练 Pod 内的日志标记（按出现顺序）：

1. `[launch] NVME_MOUNT=... MODEL_PATH=/nvme/...`：`launch_train.sh` 已把 s3 地址换成节点本地路径，并转换分布式变量（`WORLD_SIZE/RANK` → `NNODES/NODE_RANK/MASTER_*`）。
2. `[streaming_si/ktrain] stage=... nodes=... node_rank=...`：`ktrain_entry.sh`（streaming_si 专用入口）按 `SI_STAGE` 分发。
3. `[streaming_si] <阶段>: model_type=... dataset=...`、`bs=<每卡>x<NPROC>x<NNODES>`：swift sft 启动；末位是节点数，确认多机都加入了。

---

## 5. 踩过的坑（报错 → 原因 → 解决）

### 5.1 `bash: /opt/launcher/launch_train.sh: No such file or directory`

**原因**：直接用 `ktrain job submit -f` 提交。`/opt/launcher` 不在镜像里，是 `submit.sh`
第 5 步把 launcher 脚本做成 ConfigMap 临时挂载的；直接提交还跳过了代码同步与数据同步。
**解决**：始终通过 `submit.sh` 提交，不要直接 `ktrain job submit -f`。

### 5.2 `sync_code.sh: kubectl: command not found`

**原因**：提交终端的 PATH 里没有 kubectl（未安装，或 shell 没加载 Homebrew PATH）。
**解决**：`brew install kubectl`；必要时 `eval "$(/opt/homebrew/bin/brew shellenv)"`。
`sync_code.sh` 需要用 kubectl 从代码 PVC 解析 S3 bucket 名。

### 5.3 `sync_code.sh: REGION_ARGS[@]: unbound variable`

**原因**：macOS 自带 bash 3.2 在 `set -u` 下展开空数组 `"${ARR[@]}"` 会报 unbound
（bash 4.4+ 不会）。未传 `--cluster/--region` 时 `REGION_ARGS` 为空。
**解决**：`sync_code.sh` 已修复（空数组改用 `${ARR[@]+"${ARR[@]}"}` 安全展开写法）。其他
脚本若在 bash 3.2 下报同样错误，按同样写法改。

### 5.4 `AttributeError: 'TrainJobAnnotations' object has no attribute 'notifications'`

**原因**：ktrain CLI 1.8.6 的 `ktrain/core/job_admission.py` 读取 `annotations.notifications`
（通知策略注解 `ktrain.oit-svc.com/notifications`），但其依赖声明固定的 `ktrain-core==1.4.5`
的 `TrainJobAnnotations` 模型没有该字段——1.8.6 版本自身的打包缺陷，与任务 yaml 无关，
任何 yaml 都会在 validate 崩溃（`submit` 走同一检查，跳过 validate 也没用）。
**解决（本机补丁）**：把 `job_admission.py` 中该处改为
`getattr(job.metadata.annotations, "notifications", None)` 再判断（原文件备份为
`job_admission.py.orig`）。补丁只改本机 `.venv`，`ktrain update` / 重装会覆盖，届时先检查
`python -c "from ktrain_core.models.train_job import TrainJobAnnotations as A; print('notifications' in A.model_fields)"`
为 `True` 即已修复，否则重打补丁。建议把该版本组合反馈给平台方。

### 5.5 任务重名冲突

**原因**：yaml 里 `metadata.name` 写死了具体名字，不是 `__JOB_NAME__` 占位符。
**解决**：要么改回占位符（每次提交自动生成不重名的任务名），要么每次重提交前
`ktrain job delete <名字>`。

---

## 6. 相关文档

- 任务 yaml 配置、镜像构建、NCCL/EFA、挂载设计：[README.md](README.md)
- streaming_si 的 ktrain 用法（入口脚本、S4 数据准备、两段式提交）：`pipelines/streaming_si/README.md` §3.4
- 集群与节点排错：`.codebuddy/training-ops-plugin/skills/ktrain-job-ops/references/`
