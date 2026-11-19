# 提交 KtrainJob

> 触发词：`提交` `submit` `跑训练` `启动任务` `重新提交` `重跑`

## Step 0 — 确定三个输入路径

1. **任务 yaml**（`--yaml`）：用户给出；未给出时在仓库内搜索 `*ktrainjob*.yaml` 并请用户确认用哪一个。
2. **本地代码库根**（`--repo`）：待同步代码的本地根目录，通常取当前 workspace 根。
3. **入口脚本本地路径**（`--launcher`）：读任务 yaml 的 `spec.command`，取其中引用的入口脚本文件名（如 `.../launch_train.sh` → `launch_train.sh`），在任务 yaml 同目录或仓库内定位同名文件；定位不到则问用户。

从任务 yaml 预解析并记住以下值（后续汇报与命令都要用）：`<NAMESPACE>`=`metadata.namespace`、`<TRAINING_IMAGE>`=`spec.image`、`<NODE_WHITELIST>`=render-extra affinity 白名单、`<NUM_NODES>`=`spec.numNodes`。

三个输入路径齐备（用户指令已给出或可唯一确定）时直接执行，不二次确认。

## Step 1 — 执行 submit.sh

```bash
bash <SKILL_DIR>/scripts/submit.sh \
  --yaml <任务yaml> --repo <本地代码库根> --launcher <入口脚本本地路径> \
  [--job-name-prefix <前缀>] [--cluster <EKS集群> --region <区域>] \
  [--pvc <代码PVC>] [--aws-creds-secret <secret名>] \
  [--deadline-sec <秒>] [--task-deadline <任务名>=<秒>]...
```

参数规则：

- `--job-name-prefix`：缺省取任务 yaml 文件名（去扩展名、sanitize 为 DNS-1123），最终任务名 = `<前缀>-<RUN_ID>`。
- `--cluster` / `--region`：仅当当前 kubectl context 不是目标集群时才传，二者必须同时给；脚本会执行一次 `aws eks update-kubeconfig`。不传则直接用当前 context。
- `--pvc`：render-extra 中有多个 PVC 卷时必传，指定代码 CSI 卷对应的 claimName；只有一个 PVC 卷时自动识别，无需传。
- `--aws-creds-secret`：同步 Pod 访问 S3 用的凭证 secret（envFrom 注入）。集群已配 IRSA / 节点实例 Profile 时可不传。
- `--deadline-sec`：每个同步任务的统一超时（默认 3600 秒）。
- `--task-deadline 任务名=秒`：按任务单独放宽超时（大体积工件必配，例如 `--task-deadline artifact-audio-shards=86400`）。任务名 = 派生时由 env 名 sanitize 得到，submit.sh 会在派生阶段打印实际任务名列表。
- 提交后脚本 stdout 会打印派生的工件 / 任务清单，**核对与预期一致**；若工件数为 0 或缺预期工件，停止并检查任务 yaml 的 `spec.env`，不要继续。

## submit.sh 内部步骤（脚本自动完成，agent 不手工执行）

1. 解析任务 yaml 得到命名空间；可选刷新 kubecontext（`--cluster` + `--region` 时）。
2. `sync_code.sh`：代码库覆盖式同步到代码 bucket 的 `CODE_PREFIX`（先清空该前缀再全量上传）。
3. 建同步脚本 ConfigMap `artifact-sync-script-<RUN_ID>`（内容 = `sync_artifact.sh`）。
4. `derive_sync_jobs.py`：扫描任务 yaml `spec.env`，把每个含 `s3://` / `s3mnt://` / `fsx://` 值的 env 派生为一个任务，生成 `sync_<任务名>.yaml` + `plan.env`；`fsx://` 工件只记录不同步。
5. 节点预检：白名单每个节点必须 `Ready` 且未 cordon，任一异常即中断，不提交训练。
6. 同步任务严格串行：逐个 `kubectl apply` 同步 Job → 轮询终态 → 核对 `status.succeeded == N`（N = 白名单节点数）→ 采日志 → 删 Job。任一失败即中断，不提交训练。
7. 渲染任务 yaml（替换 `__RUN_ID__` / `__JOB_NAME__`），建入口脚本 ConfigMap `launch-train-script-<RUN_ID>` 并注入 render-extra 的 `extra_volumes` / `extra_volume_mounts`（卷名 `launcher`，挂载 `/opt/launcher`）。
8. `ktrain job validate -f` 校验后 `ktrain job submit -f ... --yes` 提交，打印监控命令。

同步 Job 结构（`batch/v1 Job`，非 KtrainJob）：`completions=N` / `parallelism=N` / `completionMode=Indexed` + `podAntiAffinity(kubernetes.io/hostname)`，N 个 Pod 一一落到 N 个白名单节点，全部成功才判 Complete。每个 Pod 按工件 kind 执行同步：`s3` → `aws s3 sync <源>/ <目标>/ --size-only`；`s3mnt` / `mnt` → 本地增量同步（`rsync -a --size-only`，镜像无 rsync 时退化 `cp -a -u`，再退化 `tar` 全量）。返回码即成败。

## 队列 / 拓扑约定（ktrain 1.7.x）

- 队列拓扑在任务 yaml `metadata.annotations["ktrain.oit-svc.com/topology"]` 声明：`single-node` / `single-su` / `single-pod`（`single-rack` 为兼容别名）；缺省 = best-effort locality。
- 新提交不要显式设置 `ktrain.oit-svc.com/scheduling-mode`；集群启用队列管理时 yaml 必须含 `kueue.x-k8s.io/queue-name` label，ktrain 与 operator 依据集群能力自动选 `queue` / `queue+gang`；旧 yaml 里的显式 `direct` / `queue` / `queue+gang` 仍支持但会被校验，不会静默降级。
- 提交前需预览 operator 渲染结果时用 `ktrain job manifest -f <yaml>`（不 apply；详见 `job_yaml.md`）。

## Step 2 — 汇报（模板）

```markdown
## 提交结果（KtrainJob）

| 项 | 值 |
|---|---|
| **任务名** | <JOB_NAME> |
| **命名空间** | <NAMESPACE> |
| **目标节点（白名单）** | <NODE_WHITELIST> |
| **镜像** | <TRAINING_IMAGE> |
| **同步任务** | <已执行的同步任务及耗时；fsx 工件标注"直读未同步"> |
| **状态** | <Created / Running> |

监控：
- `ktrain job status <JOB_NAME> --namespace <NAMESPACE>`
- `ktrain job logs <JOB_NAME> --namespace <NAMESPACE> --follow`

停止（ktrain CLI ≥ 1.7 已覆盖；详见 `terminate.md`）：
- `ktrain job stop <JOB_NAME> --namespace <NAMESPACE> --force`
```

## 禁止事项

- 禁止跳过 submit.sh 手工拼步骤提交；流程已固化在脚本里。
- 禁止用 `kubectl apply` 直接提交 KtrainJob；同步 Job 之外的 KtrainJob 一律 `ktrain job submit`。
- 禁止改节点白名单、节点数来绕过同步失败或训练故障。
- 禁止镜像用 `:latest` 之类浮动 tag（取值来自任务 yaml，若 yaml 里是浮动 tag，先让用户改 yaml）。
- 改动训练代码 / pipeline 脚本后必须重新走提交（submit.sh 内含代码同步），否则 Pod 内是旧代码。
