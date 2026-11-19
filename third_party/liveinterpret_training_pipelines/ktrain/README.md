# KtrainJob on EKS (H200) — ktrain CLI 训练运维指南

在 `ktrain/` 目录下，用 `ktrain job submit` 提交 KtrainJob（`ktrain.oit-svc.com/v1alpha2`）到
EKS 集群 `speech-h200-eks`，在 H200（`ml.p5e.48xlarge`，8×GPU/节点）节点运行
`pipelines/bridge_alignment/train_speech_to_speech_translation.sh`。
**与 `pytorch_job/` 目录的区别**：pytorch_job 用纯 `kubectl apply` 原生 PyTorchJob，本目录用
ktrain CLI + KtrainJob CRD（operator 渲染为 PyTorchJob），其余镜像 / 节点 / 数据 / 代码完全一致。

- 集群：`arn:aws:eks:ap-southeast-3:124355679795:cluster/speech-h200-eks`（新加坡）
- 机型：`ml.p5e.48xlarge`（H200，Hopper sm_90），8 GPU + 32 EFA/节点
- 命名空间：`speech-h200`
- 镜像：`124355679795.dkr.ecr.ap-southeast-3.amazonaws.com/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9`
- **配置唯一事实源**：`ktrain/ktrainjob_sts_translation.yaml` 与 `ktrain/ktrainjob_sts_zeyang_custom_data_train.yaml`（每个任务一个 yaml，模型 / tokenizer / manifest / 原音频的 S3 / FSX 路径、镜像、节点白名单、S3 CSI PVC、NVMe hostPath 全部写在对应 yaml 的 env + render-extra annotation 中）
- **提交 / 同步逻辑**：`.codebuddy/training-ops-plugin/skills/ktrain-job-ops/scripts/`（读训练 yaml → 派生同步 Job → 数据门控 → ktrain job submit）

---

## 1. 训练环境：镜像构建与推送

### 1.1 镜像来源

容器镜像基于 `ktrain/image/Dockerfile` 构建，关键依赖与仓库 `custom/`、`pipelines/bridge_alignment/`
存在**强版本耦合**。当前使用的 tag 为 **`p5-cuda12.8-torch2.8-te-v9`**（与 `pytorch_job/` 目录共用同一镜像）。

| 组件 | 版本 | 说明 |
|------|------|------|
| base | `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` | 有 nvcc；需补 `python3.12-dev`、`cmake`、`rsync` |
| torch | 2.8.0+cu128 | — |
| **transformer_engine** | 2.16.0 | mcore-bridge `import` 时硬依赖；需 `--no-build-isolation` 源码编译，`MAX_JOBS=6` 防 OOM |
| megatron-core | 0.17.0 | — |
| **mcore-bridge** | 1.5.2 | 与 ms-swift 4.4.0 配套 |
| **ms-swift** | 4.4.0 | 仓库绑定版本 |
| transformers | 4.57.3 | — |
| flash_attn | 2.8.3.post1 (prebuilt wheel) | sm_89/sm_90 都覆盖 |

### 1.2 构建与推送

```bash
REG=124355679795.dkr.ecr.ap-southeast-3.amazonaws.com
aws ecr get-login-password --region ap-southeast-3 | sudo docker login --username AWS --password-stdin "$REG"
sudo docker build -f ktrain/image/Dockerfile -t "$REG/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9" ktrain/image
sudo docker push "$REG/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9"
```

> **重要约束**：ECR 没有 `latest` tag；禁止用 `:latest`。镜像与 `pytorch_job/` 目录共用。

---

## 2. 任务配置（`ktrain/ktrainjob_sts_*.yaml`）

KtrainJob 的 spec 字段比原生 PyTorchJob 精简，节点白名单 / initContainer / 卷挂载 / NCCL env 等
通过 `metadata.annotations["ktrain.oit-svc.com/render-extra"]` 注入（JSON），由 ktrain operator 渲染到
底层 PyTorchJob。yaml 的 env 以 `s3://<bucket>/<key>` 形式给出模型 / tokenizer / manifest / 音频来源；
容器入口 `ktrain/launch_train.sh` 在启动时把这些 `s3://` 路径转成 NVMe 挂载路径
`/nvme/<key>`（两桶共享 NVMe hostPath）。提交脚本（`scripts/submit.sh`）读取 yaml，并由
`scripts/derive_sync_jobs.py` 从中推导出数据同步所需的 check / sync Job（不在业务仓库维护同步任务配置）。

### 2.1 S3 路径（用户给定的源）

| 工件 | S3 路径 | 容器路径（去 S3 头 = `/nvme/<s3-key>`） |
|------|---------|------------------------------------------|
| 模型 | `s3://iegg-speech-model-models/origin_models/custom_models/` | `/nvme/origin_models/custom_models` |
| tokenizer（1） | `s3://iegg-speech-model-models/origin_models/Qwen3-TTS-12Hz-1.7B-Base/` | `/nvme/origin_models/Qwen3-TTS-12Hz-1.7B-Base` |
| tokenizer（2） | `s3://iegg-speech-model-models/origin_models/Qwen3-TTS-Tokenizer-12Hz/` | `/nvme/origin_models/Qwen3-TTS-Tokenizer-12Hz` |
| 代码 | `s3://iegg-speech-model-data-.../deferzhou/liveinterpret_training_pipelines/` | 经 S3 CSI `/s3all` 直接读；`TRAIN_SCRIPT` 为 repo 相对路径，由 `launch_train.sh` 按 `CODE_PREFIX` 拼出 `/s3all/<CODE_PREFIX>/<TRAIN_SCRIPT>` |

**数据集（manifest 与音频分开）：**
- manifest：`s3://iegg-speech-model-data-.../data/audio/manifests/manifest_version=common_voice_bidirectional_ms_swift_codec_20260727_v1/task=speech_to_speech_translation/split=train/`
- 音频 shards：
  - `s3://iegg-speech-model-data-.../data/audio/processed/shards/dataset=common_voice_scripted_speech/process_version=speech_chain_reference_zh_en_v2/`
  - `s3://iegg-speech-model-data-.../data/audio/processed/shards/dataset=common_voice_scripted_speech/process_version=speech_chain_reference_v5/`

### 2.2 容器路径约定（去掉 S3 头，无隐式映射）

NVMe hostPath `/opt/dlami/nvme` 挂载在容器 `/nvme`。`launch_train.sh` 把 yaml env 里的
`s3://<bucket>/<key>` 转成 `/nvme/<key>`（S3 key 即 bucket 后的路径），与 S3 1:1，无隐式映射潜规则。
最终容器内的工件路径：

| 工件 | 容器路径 |
|------|----------|
| 模型 | `/nvme/origin_models/custom_models/qwen2_5_3b_omnispeak_a` |
| tokenizer | `/nvme/origin_models/Qwen3-TTS-Tokenizer-12Hz` |
| manifest（wds） | `/nvme/data/audio/manifests/manifest_version=common_voice_bidirectional_ms_swift_codec_20260727_v1/task=speech_to_speech_translation/split=train` |
| 音频 shards | `/nvme/data/audio/processed/shards/dataset=common_voice_scripted_speech/process_version=<...>` |

**WDS 音频**：manifest 内 `wds://curated_audio/audio/...` 由 `custom/wds_audio.py` 解析为
`/curated_audio/audio/...`；容器启动建符号链接 `/curated_audio -> /nvme/data`，使其 1:1 命中
`/nvme/data/audio/...`，无需任何音频路径变量。

### 2.3 任务 yaml

| yaml | 对应 pytorch_job | 数据源 | 节点数 | 后端 |
|------|-----------------|--------|--------|------|
| `ktrainjob_sts_translation.yaml` | `pytorchjob.yaml` | common_voice / S3（manifest + 音频 shards 同步到 NVMe） | 1 | megatron |
| `ktrainjob_sts_zeyang_custom_data_train.yaml` | `pytorchjob_unist_2node.yaml` | unist / manifest 走 FSX（`fsx://` 直读）+ 音频 shards 走 S3（ap-northeast-1 跨区同步到 NVMe） | 2 | megatron（adapter / reply_add） |

### 2.4 通常需要修改的配置项

直接在对应任务 yaml（`ktrainjob_sts_translation.yaml` 或 `ktrainjob_sts_zeyang_custom_data_train.yaml`）中修改，
提交时 `submit.sh` 会读出这些 env 并派生同步 Job：

| 配置项 | 位置（yaml 内） | 说明 |
|--------|------------------|------|
| 节点白名单 | `render-extra` annotation 的 `affinity.nodeAffinity...values` | 固定不可随意改；`numNodes` 必须等于白名单节点数（`derive_sync_jobs.py` 硬校验） |
| 镜像 tag | `spec.image` | 禁止 `:latest` |
| 模型 / tokenizer / manifest / 音频路径 | `spec.env` 的 `MODEL_PATH` / `QWEN3_TTS_TOKENIZER` / `BRIDGE_ALIGNMENT_DATA` / `ARTIFACT_AUDIO_SHARDS` | 改 S3/FSX 源即改此处 |
| 训练入口 | `spec.env.TRAIN_SCRIPT` + `spec.env.CODE_PREFIX` | repo 相对路径；容器经 S3 CSI 按 `CODE_PREFIX` 拼出实际路径执行 |
| 训练后端 | `spec.env.TRAIN_BACKEND`（`megatron` / `deepspeed`） | 超参不在 yaml 设，见 `common_args.sh` 对应分支 |
| Worker 副本数 | `spec.numNodes`（`sts_translation`=1，`sts_zeyang_custom_data_train`=2） | 双节点须同步扩白名单节点数 |

### 2.5 训练超参（MBS / GBS / MAX_LENGTH / SCENE）的事实源

批量超参**不在任务 yaml 设默认值**，默认事实源是训练脚本
`pipelines/bridge_alignment/common_args.sh` 的 `SCENE` 预设：
- `SCENE="${SCENE:-dense}"` → `dense) _TP=1; _EP=1; _GBS=1024; _MBS=8`（当前默认）
- 另含 `moe) _TP=8; _EP=1; _GBS=16; _MBS=8`

任务 yaml 的 env 块**仅用于临时覆盖**（如临时覆盖 MBS=8 试跑），不写默认值。
`common_args.sh` 注释明确「可用 env 覆盖任意单项」。

---

## 3. 任务提交（skill `scripts/submit.sh` 编排）

> 提交命令、参数含义、各阶段日志含义与本机环境准备，详见 [SUBMIT.md](SUBMIT.md)。

```bash
bash .codebuddy/training-ops-plugin/skills/ktrain-job-ops/scripts/submit.sh
```

提交逻辑全部在 skill 内（不留在业务仓库），流程含**数据就绪门控**：

1. 刷 kubecontext；建 ns + 准入 label；确保 SA 与模型桶 PVC（缺失则创建）。
2. **同步代码到 S3**：`scripts/sync_code.sh` 强制覆盖式同步 repo 到 data bucket（前缀取自 `CODE_PREFIX`）。
3. 建 sync/check 脚本 ConfigMap；`derive_sync_jobs.py` 读训练 yaml，推导出：
   - `check_job.yaml`（在白名单节点检查 NVMe 上各工件是否已存在）；
   - 每个工件的 `sync_<name>.yaml`（从对应 S3 CSI PVC 并行 rsync 到节点 NVMe，已存在则跳过）。
4. **节点数据检查**：apply `check_job.yaml`，输出 `OK:<name>` / `MISSING:<name>`。
5. **并行同步缺失工件**：对 `MISSING` 的工件分别 apply 对应 `sync_<name>.yaml`，等完成。
6. **校验后提交**：再跑一次 check 确认全部存在；用 `ktrain job submit -f <rendered.yaml>` 提交训练，同时建 `launch-train-script` ConfigMap（来自 `ktrain/launch_train.sh`）。仍有缺失则中止。

### 监控与清理

```bash
ktrain job status <JOB_NAME>
ktrain job logs <JOB_NAME> --follow
kubectl -n speech-h200 get pods -l training.kubeflow.org/job-name=<JOB_NAME> -o wide -w
ktrain job delete <JOB_NAME>
```

---

## 4. 数据与挂载设计（凭证免注入）

`speech-h200-eks` 上 Pod 无 AWS 凭证（IMDS hop-limit=1、无 IRSA、S3 CSI 驱动角色按 bucket 授权），
因此只读输入走 CSI 驱动，Pod 自身不需要凭证。

| 内容 | S3 CSI 源 | 容器挂载 | 用途 |
|------|-----------|----------|------|
| 代码 | data bucket `deferzhou/liveinterpret_training_pipelines`（前缀 `CODE_PREFIX`） | `/s3all`（PVC `speech-h200-s3-pvc`，readOnly） | `TRAIN_SCRIPT` 相对路径，由 `launch_train.sh` 按 `CODE_PREFIX` 拼出实际路径执行 |
| 模型 + tokenizer | model bucket `origin_models/` | 同步后 `/nvme/origin_models/...` | 训练加载 |
| manifest | data bucket `data/audio/manifests/...` | 同步后 `/nvme/data/audio/manifests/...` | `BRIDGE_ALIGNMENT_DATA` |
| 原音频 shards | data bucket `data/audio/processed/shards/...` | 同步后 `/nvme/data/audio/processed/shards/...`；经 `/curated_audio -> /nvme/data` 符号链接命中 WDS URI | WDS 音频读取 |
| FSX（unist 任务） | `speech-h200-fsx-pvc` | `/fsx`（readOnly） | unist manifest + 音频直读（fsx://） |

流程：提交时把 **模型/tokenizer/manifest/音频** 从 S3 CSI 并行同步到节点 NVMe（`/opt/dlami/nvme`，
hostPath，跨重启持久）；**代码** 不拷贝，训练容器直接经 S3 CSI 读 S3 路径入口。

---

## 5. NCCL / EFA

H200 有 EFA，镜像 aws-ofi-nccl 插件名 `libnccl-net-ofi.so`，NCCL 找 `libnccl-net.so` → 不建符号链接会回退
TCP Socket。`launch_train.sh` 自动建 `/opt/amazon/ofi-nccl/lib/libnccl-net.so -> libnccl-net-ofi.so`。
`/dev/shm` 用 emptyDir Memory 64Gi（否则 NCCL SHM `No space left on device`）。

---

## 6. 排错 FAQ（与 pytorch_job 目录共用的已知坑）

详见 `pytorch_job/README.md` §6-§7，镜像 / 训练代码 / 数据 / NCCL 等**完全一致**，区别仅在提交方式：

- **提交被拒**：ns 没 `ktrain.io/managed=true`，`submit.sh` 自动打。
- **Pod 卡 Pending**：查节点健康；卡在故障节点的 pod `kubectl delete pod --force --grace-period=0`。
- **OOM 后 pod 卡 Terminating 占 GPU**：`kubectl -n speech-h200 delete pod <job>-master-0 --force --grace-period=0`。
- **同步 Job 失败 / 工件 MISSING**：检查对应 PVC 是否 Bound，节点 IAM 角色能否读对应 bucket。
- **WDS 音频 FileNotFoundError**：确认 `/curated_audio -> /nvme/data` 符号链接存在，且音频 shards 已同步。
- **改动不生效**：`launch_train.sh` 走 ConfigMap（重跑 `submit.sh` 刷新）；超参脚本改动需 `sync_code` 重新同步到 S3。

---

## 7. 目录结构

```
ktrain/
├── README.md                    # 本文件
├── launch_train.sh              # 容器入口（ConfigMap 注入），纯接线、不拷数据；
│                                #   把 s3://、s3mnt://、fsx:// 工件 URI 转容器内真实路径，建 WDS/unist 软链。
├── ktrainjob_sts_translation.yaml          # 主任务（common_voice / S3 数据，单节点）
├── ktrainjob_sts_zeyang_custom_data_train.yaml  # unist / FSX+跨区S3 数据，双节点（adapter / reply_add）
└── image/
    ├── Dockerfile               # 训练镜像（与 pytorch_job 目录共用）
    └── build_image.sh           # 构建与推送辅助脚本
```

与 `pytorch_job/` 目录结构对齐，保持简洁，方便上手。
