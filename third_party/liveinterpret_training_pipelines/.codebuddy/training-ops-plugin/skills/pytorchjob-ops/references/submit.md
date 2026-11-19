# 提交 PyTorchJob（纯 kubectl）

> 触发词：`提交` `submit` `跑训练` `启动` `重新提交` `重跑` `pytorchjob` `apply`

本流程适用于「不依赖 anyscale / ray / ktrain 运行时，直接用 `kubectl` 提交 Kubeflow `PyTorchJob`」的平台。

任务配置的唯一真源是项目 `pytorch_job/pytorchjob.yaml`（镜像、命名空间、节点白名单、模型 / tokenizer / manifest / 原音频路径、S3 CSI PVC、NVMe hostPath 全部写在该 yaml 中，仅 `__RUN_ID__` / `__JOB_NAME__` 为运行时占位符）。skill 不硬编码这些值。

提交与数据就绪门控逻辑在本 skill 的 `scripts/`：

- `scripts/submit.sh` — 编排全流程。
- `scripts/derive_sync_jobs.py` — 从训练 yaml 解析出工件列表，并**推导生成**各工件的 sync Job（独立简洁 yaml），不在业务仓库维护同步任务配置。
- `scripts/sync_code.sh` / `sync_artifact.sh` — 实际的同步动作。`sync_artifact.sh` 对每个工件跑一次 `aws s3 sync --size-only`（按 key 名 + 文件大小增量比对、幂等、自带成败返回码），**没有也不做任何额外的校验 / 就绪标记 / dry-run 机制**（`.synced` / `find` / verify 均已移除）。

---

## 0. 提交前：从项目读取上下文

agent 读取 `pytorch_job/pytorchjob.yaml`，解析出（后续命令用占位符指代）：

| 占位符 | 含义 | 来源（yaml 内） |
|---|---|---|
| `<CLUSTER>` | k8s/EKS 集群名 | `submit.sh` 内常量 `speech-h200-eks` |
| `<REGION>` | 云区域 | `submit.sh` 内常量 `ap-southeast-3` |
| `<NAMESPACE>` | 命名空间 | yaml `metadata.namespace` |
| `<TRAINING_IMAGE>` | 训练镜像（含 tag） | yaml `containers[].image` |
| `<NODE>` | 固定节点白名单 | yaml `affinity` 的 `values` |
| `<SERVICE_ACCOUNT>` | Pod 使用的 SA | yaml `serviceAccountName` |
| `<DATA_PROFILE>` | 数据集（由 yaml env `ARTIFACT_AUDIO_SHARDS` 等体现） | yaml `env` |

工件 S3 源（去掉 `s3://<bucket>/` 头即为 `/nvme/<s3-key>`，与 NVMe hostPath 1:1）：
- **模型 + tokenizer**：`s3://iegg-speech-model-models/origin_models/...`（模型桶；节点本地 NVMe 直接读，未挂 CSI）
- **manifest + 原音频**：`s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/...`（数据桶 PVC `speech-h200-s3-pvc`）
- **代码**：repo 经 `sync_code.sh` 同步到 data bucket，训练入口 `TRAIN_SCRIPT` 即该 bucket 下的 S3 路径，容器经 S3 CSI 直接读。

---

## 1. 提交流程（`scripts/submit.sh` 自动编排）

```bash
bash .codebuddy/skills/pytorchjob-ops/scripts/submit.sh
```

`submit.sh` 内部完成节点健康预检 + 串行数据同步门控后提交训练：

1. 刷 kubecontext（`aws eks update-kubeconfig`）；建 ns + 准入 label（`ktrain.io/managed=true`）。
2. 确保 SA 与模型桶 PVC（缺失则创建；数据桶 PVC 须预置，否则 FATAL）。
3. **同步代码到 S3**：`scripts/sync_code.sh` 把 repo 覆盖式同步到 data bucket；`TRAIN_SCRIPT` 即该 bucket 下的 S3 路径。
4. 建同步脚本 ConfigMap（`artifact-sync-script-<run-id>`，含 `sync_artifact.sh`）；`derive_sync_jobs.py` 读训练 yaml，按 `TASK_DEFS` 推导出**四个串行同步任务**并生成 `sync_<task>.yaml` + `plan.env`：
   - `model`    ← `MODEL_PATH`
   - `tokenizer` ← `QWEN3_TTS_TOKENIZER`（`QWEN3_TTS_TOKENIZER_BASE`）
   - `manifest` ← `BRIDGE_ALIGNMENT_DATA`
   - `audio`    ← `ARTIFACT_AUDIO_SHARDS`
   - env 缺失/为空的任务会被列入 `SKIPPED_TASKS` 显式打印（绝静默丢弃）。
5. **节点预检**：`precheck_nodes` 检查白名单每个节点 `Ready` 且未 cordon，异常立即 FATAL（避免 Pod 一直 Pending 到 deadline）。
6. **四任务严格串行同步**：对每个任务 `run_job`：apply `sync_<task>.yaml` → 轮询终态 → 显式核对 `status.succeeded == N`（N = 白名单节点数）→ 采日志 → 删 Job。任一任务失败立即 FATAL，**不提交训练**。
7. **提交训练**：渲染训练 yaml（`__RUN_ID__` / `__JOB_NAME__`，Job 名 `qwen3-omnispeak-bridge-align-h200-<RUN_ID>`）并 apply，同时建 `launch-train-script` ConfigMap（来自 `pytorch_job/launch_train.sh`）；打印监控命令。

### 同步 Job 架构（数据就绪门控的核心）

每个同步任务 = 一个 `batch/v1 Job`（**不是** PyTorchJob），由 `derive_sync_jobs.py` 把训练 Master.template 深拷贝后改造而来：
- `completions=N` / `parallelism=N` / `completionMode=Indexed` + `podAntiAffinity(hostname)` → N 个 Pod 落 N 个不同节点，**「全部 N 个 Pod 成功才 Complete」是硬条件**；
- 自定义 label `sync-job=<name>`（不依赖 controller 注入的 `job-name`，跨 k8s 版本稳定），用于逐 Pod 采日志；
- 超时从 yaml 派生、经 buffer 放大（`plan.env` 的 `WAIT_TIMEOUT_<task>`，已严格 > 各自 `activeDeadlineSeconds`）：model 3600s / tokenizer 1800s / manifest 3600s / audio 21600s。

这与旧「PyTorchJob 在 Master 成功即假成功、清掉仍在同步的 Worker」的 bug 彻底隔离：submit.sh 用 `assert_job_complete` 核对 `succeeded == N`，不足 N 视为失败。

### 就绪判定完全交给 `aws s3 sync` 自身

每个 sync Pod 内 `sync_artifact.sh` 对每个工件对执行**一次**：

```bash
aws s3 sync "${SRC%/}/" "${DST}/" --size-only --no-progress --only-show-errors
```

`aws s3 sync` 原生按「相对 key 名 + 文件大小」做增量比对（官方 help：默认忽略 same-sized items，除非本地更新）：
- 本地缺失、或本地大小与源对象不一致 → 重新下载并覆盖本地文件；
- 本地已一致 → 跳过（幂等，已一致时重跑输出 0 行）。

必须 `--size-only`：NVMe 本地文件 mtime 是下载时刻，与 S3 对象 last-modified 不同，按默认 size+mtime 比较每次都会判「不一致」而全量重传。`aws s3 sync` 的返回码即成败，失败即 FATAL。**不写任何就绪标记文件、不遍历本地磁盘计数、不做任何 dry-run 预检/后检**——这些机制相对 `aws s3 sync` 自带的比对都是多余的。

---

## 2. 容器路径约定（去掉 S3 头）

所有工件在容器内的路径 = `/nvme/<s3-key>`（NVMe hostPath 挂载在 `/nvme`，与 S3 key 1:1），例如：
- 模型：`/nvme/origin_models/custom_models/qwen2_5_3b_omnispeak_a`
- tokenizer：`/nvme/origin_models/Qwen3-TTS-Tokenizer-12Hz`
- manifest：`/nvme/data/audio/manifests/manifest_version=common_voice_bidirectional_ms_swift_codec_20260727_v1/task=speech_to_speech_translation/split=train`
- 音频 shards：`/nvme/data/audio/processed/shards/dataset=common_voice_scripted_speech/process_version=speech_chain_reference_v5`（等）

WDS 音频通过 `wds://curated_audio/...` 解析：容器启动时建符号链接 `/curated_audio -> /nvme/data`，使 `wds://curated_audio/audio/...` 1:1 命中 `/nvme/data/audio/...`。

---

## 3. 提交结果汇报模板

```markdown
## 提交结果（PyTorchJob）

| 项 | 值 |
|---|---|
| **PyTorchJob** | <JOB_NAME> |
| **集群 / ns** | <CLUSTER> / <NAMESPACE> |
| **目标节点（白名单）** | <NODE> |
| **镜像** | <TRAINING_IMAGE> |
| **状态** | <Created / Running> |

监控：
- `kubectl -n <NAMESPACE> get pods -l training.kubeflow.org/job-name=<JOB_NAME> -o wide -w`
- `kubectl -n <NAMESPACE> logs -f <JOB_NAME>-master-0 -c pytorch`

清理：
- `kubectl -n <NAMESPACE> delete pytorchjob <JOB_NAME>`
```

---

## 4. 注意事项（通用）

- **镜像 tag 不要用 `:latest`**：固定可复现的 tag。
- **节点白名单固定不动**：由训练 yaml 的 affinity 固定，不要靠换节点规避训练侧问题（如 OOM）。
- **提交前务必先 sync 代码**（`submit.sh` 自动做），否则 pod 内是旧代码。
- **数据就绪门控 = 四任务串行 `aws s3 sync --size-only`**：模型/tokenizer/manifest/音频在白名单所有节点上经 `completions=N` 门控确认全部落盘后才提交训练；重复提交若数据已一致则 `aws s3 sync` 幂等跳过、几乎瞬时完成。不依赖任何额外的 check/verify 环节。
- 改动训练超参 / pipeline 脚本后必须重新 `sync_code` 才能生效；改动训练 yaml 直接重新 `submit.sh` 即可。
- 变更类操作（提交、终止、覆盖式 sync）执行前需二次确认，除非用户给出明确的执行指令（含具体 job 名 / 清单路径、无疑问词）。
