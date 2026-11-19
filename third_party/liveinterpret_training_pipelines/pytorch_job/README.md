# PyTorchJob on EKS (H200) — 纯 kubectl 训练运维指南

在 `pytorch_job/` 目录下，用纯 `kubectl` 提交 Kubeflow `PyTorchJob`（`kubeflow.org/v1`）到 EKS 集群
`speech-h200-eks`，在 H200（`ml.p5e.48xlarge`，8×GPU/节点）节点运行
`pipelines/bridge_alignment/train_speech_to_speech_translation_megatron.sh`。**不依赖 anyscale / ray / ktrain 任何运行时**。

- 集群：`arn:aws:eks:ap-southeast-3:124355679795:cluster/speech-h200-eks`（新加坡）
- 机型：`ml.p5e.48xlarge`（H200，Hopper sm_90），8 GPU + 32 EFA/节点
- 命名空间：`speech-h200`（已打 `ktrain.io/managed=true` 准入 label）
- 镜像：`124355679795.dkr.ecr.ap-southeast-3.amazonaws.com/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9`
- **配置唯一事实源**：`pytorch_job/pytorchjob.yaml`（模型 / tokenizer / manifest / 原音频的 S3 路径、镜像、节点白名单、S3 CSI PVC、NVMe hostPath 全部写在该 yaml 中）
- **提交 / 同步逻辑**：`.codebuddy/skills/pytorchjob-ops/scripts/`（读训练 yaml → 派生同步 Job → 数据门控 → 提交）

---

## 1. 训练环境：镜像构建与推送

### 1.1 镜像来源

容器镜像基于 `ktrain/image/Dockerfile` 构建，关键依赖与仓库 `custom/`、`pipelines/bridge_alignment/`
存在**强版本耦合**（见 §7）。当前使用的 tag 为 **`p5-cuda12.8-torch2.8-te-v9`**。

| 组件 | 版本 | 说明 |
|------|------|------|
| base | `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` | 有 nvcc；需补 `python3.12-dev`(Python.h)、`cmake`、`rsync` |
| torch | 2.8.0+cu128 | — |
| **transformer_engine** | 2.16.0 | mcore-bridge `import` 时硬依赖；需 `--no-build-isolation` 源码编译，`MAX_JOBS=6` 防 OOM |
| megatron-core | 0.17.0 | — |
| **mcore-bridge** | 1.4.1 | 与 ms-swift 4.4.0.dev0 配套 |
| **ms-swift** | **4.4.0.dev0**（`git@c7da97c9`，2026-06-25） | **仓库绑定版本**，必须 pin 到该 commit |
| transformers | 4.57.6 | — |
| flash_attn | 2.8.3.post1 (prebuilt wheel) | sm_89/sm_90 都覆盖 |

### 1.2 构建与推送

```bash
REG=124355679795.dkr.ecr.ap-southeast-3.amazonaws.com
aws ecr get-login-password --region ap-southeast-3 | sudo docker login --username AWS --password-stdin "$REG"
sudo docker build -f ktrain/image/Dockerfile -t "$REG/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te" ktrain/image
sudo docker tag "$REG/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te" \
                 "$REG/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9"
sudo docker push "$REG/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9"
```

> **重要约束**：ECR 没有 `latest` tag；旧 `l4-cuda12.8-torch2.8` 缺 TE，**不可用**。**不要用 `:latest`**。

---

## 2. 任务配置（`pytorch_job/pytorchjob.yaml`）

所有工件 S3 源、镜像、节点白名单、S3 CSI PVC、NVMe hostPath **直接写在
`pytorch_job/pytorchjob.yaml`**，单一事实源。yaml 的 env 以 `s3://<bucket>/<key>` 形式给出模型 / tokenizer /
manifest / 音频来源；容器入口 `pytorch_job/launch_train.sh` 在启动时把这些 `s3://` 路径转成 NVMe 挂载路径
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

> 旧 `bridge_legacy` 数据（`data/curated/task=speech2speech/`、`data/curated/task=tts/`、`data/raw/`）目前未纳入自动同步，如需使用请扩展 `pytorchjob.yaml` 的 env（如增加 `ARTIFACT_LEGACY_KEYS`）。
>
> 注：`pytorchjob.yaml` 的 env 中，模型 / tokenizer / manifest / 音频以 `s3://` URI 给出，由 `launch_train.sh` 转成 `/nvme/<key>`；`TRAIN_SCRIPT` 与 `OUTPUT_ROOT` 为 `/s3all/...`（S3 CSI 直接读，不转换）。

### 2.2 容器路径约定（去掉 S3 头，无隐式映射）

NVMe hostPath `/opt/dlami/nvme` 挂载在容器 `/nvme`。`launch_train.sh` 把 yaml env 里的
`s3://<bucket>/<key>` 转成 `/nvme/<key>`（S3 key 即 bucket 后的路径），与 S3 1:1，无隐式映射潜规则。
最终容器内的工件路径：

| 工件 | 容器路径 |
|------|----------|
| 模型 | `/nvme/origin_models/custom_models/qwen3_omnispeak_d` |
| tokenizer | `/nvme/origin_models/Qwen3-TTS-Tokenizer-12Hz` |
| manifest（wds） | `/nvme/data/audio/manifests/manifest_version=common_voice_bidirectional_ms_swift_codec_20260727_v1/task=speech_to_speech_translation/split=train` |
| 音频 shards | `/nvme/data/audio/processed/shards/dataset=common_voice_scripted_speech/process_version=<...>` |

**WDS 音频**：manifest 内 `wds://curated_audio/audio/...` 由 `custom/wds_audio.py` 解析为
`/curated_audio/audio/...`；容器启动建符号链接 `/curated_audio -> /nvme/data`，使其 1:1 命中
`/nvme/data/audio/...`，无需任何音频路径变量。

### 2.3 通常需要修改的配置项

直接在 `pytorch_job/pytorchjob.yaml` 中修改（提交时 `submit.sh` 会读出这些 env 并派生同步 Job）：

| 配置项 | 位置（yaml 内） | 说明 |
|--------|------------------|------|
| 节点白名单 | `affinity.nodeAffinity` 的 `values` | 固定不可随意改 |
| 镜像 tag | `containers[].image` | 禁止 `:latest` |
| 模型 / tokenizer / manifest / 音频路径 | `env` 的 `MODEL_PATH` / `QWEN3_TTS_TOKENIZER` / `BRIDGE_ALIGNMENT_DATA` / `ARTIFACT_AUDIO_SHARDS` | 改 S3 源即改此处 |
| 训练入口 | `env.TRAIN_SCRIPT` + `env.CODE_PREFIX` | repo 相对路径（如 `pipelines/.../train_xxx.sh`）；容器经 S3 CSI 按 `CODE_PREFIX` 拼出实际路径执行 |
| Worker 副本数 | `Worker.replicas`（当前 0，单节点） | — |

### 2.4 训练超参（MBS / GBS / MAX_LENGTH / SCENE）的事实源

批量超参**不在 `pytorchjob.yaml` 设默认值**，默认事实源是训练脚本
`pipelines/bridge_alignment/common_args_megatron.sh` 的 `SCENE` 预设：

- `SCENE="${SCENE:-dense}"` → `dense) _TP=1; _EP=1; _GBS=1024; _MBS=16`（当前默认，双节点 16 卡 DP=16）
- `MAX_LENGTH` 默认值也在 `common_args`（当前 8192）

`pytorch_job/pytorchjob.yaml` 的 env 块**仅用于临时覆盖**（如临时覆盖 MBS=8 试跑），
不写默认值。`common_args` 注释明确「可用 env 覆盖任意单项」。`pytorchjob.yaml` 只放硬件 / 工件 / env 接线，
不要把 yaml 当 batch 超参事实源（否则会与 `common_args` 的 `SCENE` 预设打架，且 submit 只透传 env）。

---

## 3. 任务提交（skill `scripts/submit.sh` 编排）

```bash
bash .codebuddy/skills/pytorchjob-ops/scripts/submit.sh
```

提交逻辑全部在 skill 内（不留在业务仓库），流程含**数据就绪门控**：

1. 刷 kubecontext；建 ns + 准入 label；确保 SA 与模型桶 PVC（缺失则创建；数据桶 PVC 须预置）。
2. **同步代码到 S3**：`scripts/sync_code.sh` 强制覆盖式同步 repo 到 data bucket（前缀取自 `CODE_PREFIX`，即 `deferzhou/liveinterpret_training_pipelines`）；训练入口 `TRAIN_SCRIPT` 为该 repo 的相对路径，由 `launch_train.sh` 拼出 `/s3all/<CODE_PREFIX>/<TRAIN_SCRIPT>` 执行。
3. 建 sync/check 脚本 ConfigMap；`derive_sync_jobs.py` 读训练 yaml，推导出：
   - `check_job.yaml`（在白名单节点检查 NVMe 上各工件是否已存在）；
   - 每个工件的 `sync_<name>.yaml`（从对应 S3 CSI PVC 并行 rsync 到节点 NVMe，已存在则跳过）。
4. **节点数据检查**：apply `check_job.yaml`，输出 `OK:<name>` / `MISSING:<name>`。
5. **并行同步缺失工件**：对 `MISSING` 的工件分别 apply 对应 `sync_<name>.yaml`，等完成。
6. **校验后提交**：再跑一次 check 确认全部存在；渲染训练 yaml（`__RUN_ID__` / `__JOB_NAME__`）并 apply，同时建 `launch-train-script` ConfigMap（来自 `pytorch_job/launch_train.sh`）。仍有缺失则中止。

> 若节点数据已全部就绪，步骤 4 直接判定「全部存在」，跳过步骤 5，直接提交训练。

相关文件：
- `pytorch_job/pytorchjob.yaml` — 任务配置唯一事实源。
- `pytorch_job/launch_train.sh` — 容器入口（经 ConfigMap 注入），纯接线、不拷数据；仅把 `s3://` 工件路径转 `/nvme`。
- `.codebuddy/skills/pytorchjob-ops/scripts/submit.sh` — 提交编排。
- `.codebuddy/skills/pytorchjob-ops/scripts/derive_sync_jobs.py` — 从训练 yaml 推导 check/sync Job。
- `.codebuddy/skills/pytorchjob-ops/scripts/sync_code.sh` / `sync_artifact.sh` / `check_artifacts.sh` — 同步 / 检查动作。

### 监控与清理

```bash
kubectl -n speech-h200 get pods -l training.kubeflow.org/job-name=<JOB_NAME> -o wide -w
kubectl -n speech-h200 logs -f <JOB_NAME>-master-0 -c pytorch
kubectl -n speech-h200 delete pytorchjob <JOB_NAME>
```

> **诊断纪律**：一律**下载全量日志到本地**分析，禁止 cherry-pick、禁止先射箭再画靶。

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

流程：提交时把 **模型/tokenizer/manifest/音频** 从 S3 CSI 并行同步到节点 NVMe（`/opt/dlami/nvme`，
hostPath，跨重启持久）；**代码** 不拷贝，训练容器直接经 S3 CSI 读 S3 路径入口。同步 Job 由
`derive_sync_jobs.py` 解析 yaml env 中的 `s3://<bucket>/<key>`（按 key 前缀 `origin_models/*` → 模型桶、`data/*` → 数据桶）自动选择对应 PVC。

---

## 5. NCCL / EFA

H200 有 EFA，镜像 aws-ofi-nccl 插件名 `libnccl-net-ofi.so`，NCCL 找 `libnccl-net.so` → 不建符号链接会回退
TCP Socket。`launch_train.sh` 自动建 `/opt/amazon/ofi-nccl/lib/libnccl-net.so -> libnccl-net-ofi.so`。
`/dev/shm` 用 emptyDir Memory 64Gi（否则 NCCL SHM `No space left on device`）。

---

## 6. 排错 FAQ

- **提交被 webhook 拒绝**：ns 没 `ktrain.io/managed=true`，`submit.sh` 自动打。
- **Pod 卡 Pending**：查节点健康；卡在故障节点的 pod `kubectl delete pod --force --grace-period=0`。
- **OOM 后 pod 卡 `Terminating` 占 GPU，节点 `NotReady` / submit 预检失败**：训练 OOM 死的 pod 其 cuda 进程僵死，`kubectl delete pytorchjob` 删了 CRD 但容器内进程不响应 SIGKILL，pod 长期 `Terminating` 占用 8 张 GPU，导致节点 GPU 资源未释放、新任务调度不到该节点、submit 预检判 `NotReady`。**排查必须看 pod 级状态**（`kubectl get pods -A | grep <job>`），不能只看 `kubectl get pytorchjob`（空会误判已释放）。处置：`kubectl -n speech-h200 delete pod <job>-master-0 --force --grace-period=0` 强制清掉，GPU 释放后节点恢复。
- **节点白名单固定（不可改）**：`pytorchjob.yaml` affinity `In` 白名单；此前 worker 崩溃是训练代码 OOM（未切分 talker+code-predictor 长序列激活爆炸），NCCL `Connection refused` 是次生级联，非死锁、与节点/网络无关。
- **重复 `-209` job 抢 GPU**：外部自动化重建带 `-209` 后缀重复 PyTorchJob，先 `kubectl -n speech-h200 get pytorchjob` 确认再清理。
- **同步 Job 失败 / 工件 MISSING**：检查对应 PVC 是否 Bound（`speech-h200-s3-pvc` / `speech-h200-s3-models-pvc`），节点 IAM 角色能否读对应 bucket；S3 CSI 只读，`rsync --size-only` 增量。
- **`Unable to locate credentials`（pod 内 aws）**：Pod 无凭证，只读走 CSI；不要在 pod 用 awscli。
- **WDS 音频 `FileNotFoundError`**：确认 `/curated_audio -> /nvme/data` 符号链接存在，且音频 shards 已同步到 `/nvme/data/audio/processed/shards/...`。
- **`meta tensor; no data`**：先查模型是否已同步到 NVMe；若已同步，是 ms-swift meta 上下文问题，见 §7 #16。
- **NCCL `No space left on device`**：/dev/shm 没挂 emptyDir Memory。
- **NCCL 跨节点 abort**：EFA 符号链接没建。
- **`remaining_argv: --run_name`**：ms-swift 版本不对。
- **改动不生效**：`launch_train.sh` 走 ConfigMap（重跑 `submit.sh` 刷新）；超参脚本改动需 `sync_code` 重新同步到 S3。
- **kubectl 无权**：IAM 用户没加 EKS access entry（见 §3 前置）。

---

## 7. 版本耦合与已知坑（精简）

仓库与 ms-swift **强版本耦合**：必须 pin 到 ms-swift 4.4.0.dev0（`git@c7da97c9`）+ transformers 4.57.6 +
mcore-bridge 1.4.1（镜像 `-v5`）。已修复的关键问题速查：

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 4 | `Cannot copy out of meta tensor` | safetensors 在 S3 CSI 上 mmap 失败 | 模型同步到 NVMe 本地加载 |
| 5 | NCCL `No space left on device` | /dev/shm 太小 | emptyDir Memory 64Gi |
| 6 | NCCL 跨节点 Socket abort | EFA 插件名不匹配 | `libnccl-net.so` 符号链接 |
| 9/10 | encode 崩 `Boolean value of Tensor` / `CUDA re-init in fork` | 对 tensor 误用 `or`；codec 在 fork 子进程走 GPU | 显式 None 判断；`QWEN3_TTS_TOKENIZER_DEVICE=cpu` |
| 11 | `padding_free=True` | megatron 默认破坏逐样本对齐 | `--padding_free false` |
| 12 | `No dot product attention backend` | flash 拒绝自定义 mask | `ATTENTION_BACKEND=unfused` |
| 13 | `Dynamo failed to run FX node`（MoE） | torch.compile 追踪失败 | `TORCH_COMPILE_DISABLE=1` |
| 14 | `IndexError` [62],[42] | thinker/talker 分离 `labels` 长度不匹配 | `_patched_forward_step` 替换 `labels` 为 `-100` |
| 15 | worker OOM（长 codec 序列） | talker+code-predictor 无激活重算（激活 ∝ `codec_len~13600` 序列长，不被 TP 除） | 两处 replicated 前向加 `torch.utils.checkpoint` 激活重算（已修复）；修复后 MBS 边界由单卡显存决定，旧 `common_args` 注释「MBS≥8@4096 OOM」已过时，实测边界见 §8 |
| 16 | `Cannot copy out of meta tensor`（模型已在 NVMe，非 #4） | ms-swift 用 `with torch.device('meta')` 包裹 `from_pretrained`，模型建在 meta 上，transformers 走 `_load_state_dict_into_meta_model` 对 meta 张量做 `param = param[...]` | `modeling_qwen_omnispeak.py::from_pretrained` 内层 `with torch.device(device_map)` 覆盖外层 meta 上下文 + `low_cpu_mem_usage=False`，且 `device_map` 保持原值 |

### 7.1 #16 定位过程（对照实验，供同类问题复用）

同一处报错换了 4 次提交才定位，因为 transformers 有两条互斥的失败路径，改一个参数只是从一条跳到另一条：

| job | `device_map` | `low_cpu_mem_usage` | 结果 |
|-----|-------------|---------------------|------|
| `sbp2` | `'cuda:N'`（原值） | `True`（默认） | ❌ `Cannot copy out of meta` @ `_load_state_dict_into_meta_model:748` |
| `cml2` | `None` | `False` | ❌ meta anti-pattern @ `modeling_utils:4776` |
| `9hbq` | `'cuda:N'` | `False` | ❌ 仍崩 `:748`（外层 meta 上下文压过 `low_cpu_mem_usage`） |
| `eki0` | `None` + `set_default_device('cpu')` | `False` | ❌ 4776 anti-pattern；且 `finally` 里 `set_default_device` 回填触发 `AssertionError`（外层是 `DeviceContext`，不能回传） |
| `3ust` | `'cuda:N'`（原值） | `False` + 内层 `with torch.device('cuda:N')` | ✅ 模型加载通过 |

关键结论（读容器内 transformers 源码得出，勿凭经验猜）：

1. 4776 的 anti-pattern 守卫**只在 `device_map is None` 时**才检查上下文设备 → `device_map` 必须保留 ms-swift 传入的 `'cuda:N'`。
2. `low_cpu_mem_usage` / `device_map` 都压不住**外层 `with torch.device('meta')` 上下文**；只有内层 `with torch.device(...)` 能覆盖它。
3. `torch.set_default_device()` 与上下文管理器不可混用：`get_default_device()` 在上下文内返回 `DeviceContext`，回传即 assert 失败。
4. 排查手段：用训练镜像起一次性 pod（`kubectl run ... -- sh -c "sed -n 'A,Bp' .../modeling_utils.py"`）读**运行时版本**的源码，比对报错行号，避免与本地/文档版本错位。

模型加载通过后，sidecar `qwen3_tts_extras.safetensors` 的 `talker.text_projection.*` 需先 `.to(model_device)` 再 `load_state_dict(..., assign=True)`，否则会把这几个参数搬回 CPU。

### 7.2 观察（非故障）：Map 100% 后的 `Flattening the indices` 是正常数据准备，勿误判 hang

现象：`Map (num_proc=32): 100%` 之后长时间无"新阶段"提示，随后出现 `Flattening the indices: x%`（8 个 rank 的进度条混叠），数字跳得慢，极易被 NCCL 连接日志刷屏掩盖，看着像卡死。

根因：ms-swift 在 Megatron 非 streaming（`--packing false --padding_free false`）路径下，每个 rank 各自对**全量** 252 万条音频样本做 `flatten_indices`（`train_test_split` 前必须物化）。8 ranks × 2 节点 = 16 份重复解码，单节点约 14–15 分钟。

如何区分"在跑"还是"真卡死"（ucdc 实测）：

- 真推进：`logs` 末尾 `Flattening the indices` 时间戳 `[mm:ss]` 持续增大；rank 主进程 `Rsl` 运行态、CPU 高；`/proc/diskstats` 的 nvme 读扇区持续累加；`/proc/<pid>/task/*/stat` 无 D（磁盘等待）线程。
- 真卡死：时间戳与 % 长时间不变；某 rank 在 barrier 等其它 rank；`ps` 显示某进程 D 态堆积或 CPU 0%。

处置：无需干预，flatten 到 100% 后会做 `train_test_split`（split_dataset_ratio=0.01）再进入训练。若要加速，可评估开启 streaming / packing 让每 rank 只取分片，避免 16 份全量重复解。

---

## 8. 训练超参调优（实测显存边界，H200 141 GiB/卡）

OOM 根因见 §7 #15：replicated 的 talker+code-predictor 处理 `codec_len~13600`（远超 `MAX_LENGTH`）
的长序列，激活显存 ∝ 序列长度且不被 TP 除；已加 activation checkpointing 修复，但 **MBS 仍受单卡显存硬约束**。

> 注意：`common_args` 旧注释「MBS≥8 @ MAX_LENGTH=4096 OOM、MBS=4 安全」是**修复前**的经验，已过时。
> 修复后 OOM 由 `codec_len` 序列长度主导，MBS 边界需按当前 `MAX_LENGTH` 实测。

`SCENE=dense`（默认，双节点 16 卡，DP=16，GBS=1024）下实测：

| MBS | MAX_LENGTH | 单卡显存 | 占比 | 结果 | step 时间 | 吞吐 |
|-----|-----------|---------|------|------|----------|------|
| 4   | 4096      | ~39 GiB | 28% | 安全（基线） | 8.9 s/it | ~115 samples/s |
| 16  | 8192      | ~92 GiB | 65% | **安全**（已验证越过首 step，无 OOM） | 4.4 s/it | ~233 samples/s |
| 20  | 8192      | ~108 GiB（估） | 77% | 预计安全，推荐起步 | — | — |
| 24  | 8192      | ~123 GiB（估） | 87% | 偏紧，可试 | — | — |
| 32  | 8192      | 139.5 GiB | OOM | **首 step forward OOM**（master rank7：Tried to allocate 378 MiB, 229 MiB free） | — | — |

调优结论：
- `MBS=16 @ MAX_LENGTH=8192` 是当前平衡最优：吞吐较 `MBS=4@4096` 翻倍、显存 65%、真实 MFU（按可训参数）~55%。
- 显存仍余 ~49 GiB，可继续加 MBS 到 **20（稳）/ 24（激进）** 吃满；MBS≥28 风险高（MBS=32 已实测 OOM）。
- `MAX_LENGTH` 不宜再加：8192→16384 让 attention 激活 ∝seq²（4×），除非把 MBS 砍回 4–8，吞吐反降。
- `packing` 当前 `false`；开启可能再提吞吐，但需验证与 talker codec loss 的 mask 兼容（见 §7 #11/#12 的自定义 mask 约束）。

---

## 9. MFU 评估（skill `scripts/estimate_mfu.sh`）

```bash
# 基于 checkpoint 时间戳（需 iter≥SAVE_STEPS=500，首次 ckpt 约 1.5–2h 后才有）
bash .codebuddy/skills/pytorchjob-ops/scripts/estimate_mfu.sh <JOB_NAME> speech-h200

# 训练早期无 ckpt 时，用实测 train_speed(s/it) 覆盖（日志 'train_speed(s/it)' 稳定值）
STEP_SEC_OVERRIDE=4.4 K8S_NS=speech-h200 \
  bash .codebuddy/skills/pytorchjob-ops/scripts/estimate_mfu.sh <JOB_NAME> speech-h200
```

要点：
- 公式：`step_sec` → `tokens_per_step = GBS × SEQ` → `flops = 6 × N × tok` → `per_gpu_flops = flops / step_sec / DP` → `MFU = per_gpu_flops / GPU_PEAK`。
- **freeze 主干时 MFU 上界会失真（>100%）**：脚本默认按全参 3B 算 `6×N`，但实际 `freeze_llm/freeze_vit=true` 只训 aligner+talker，真实每 step FLOPs 远小于 `6×3B`，故上界虚高（如 z5ht 算出 217%）。应按**可训参数量**理解真实 MFU：反推等效可训 ~0.76B 时真实 MFU 上界约 55%（H200 稠密训练合理区间），说明 GPU 算力利用健康、非 data-starved。
- MFU 远低于上界且 step 时间被拉长 = GPU data-starved（dataloader 喂数瓶颈）；本任务 step 稳定、显存吃满，非此情形。MFU 评估后若要继续优化，优先吃满显存余量（§8 的 MBS 上调），而非怀疑喂数。
