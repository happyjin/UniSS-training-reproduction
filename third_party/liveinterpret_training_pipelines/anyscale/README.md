# Anyscale 训练模块 — 使用指引与故障排查

> 本文档覆盖 Anyscale 训练 Job 的提交、监控、参数修改与故障排查。
> 架构背景：Anyscale cloud `anyscale-poc-tokyo-cloud`（EKS `anyscale-poc-v2-eks`，region `ap-northeast-1`），训练入口 `anyscale/train/dispatch_train.py`。

---

## 目录

1. [架构总览](#1-架构总览)
2. [前置条件](#2-前置条件)
3. [提交训练 Job](#3-提交训练-job)
4. [修改训练参数](#4-修改训练参数)
5. [监控与诊断](#5-监控与诊断)
6. [取回 Checkpoint](#6-取回-checkpoint)
7. [终止 Job](#7-终止-job)
8. [常见故障速查](#8-常见故障速查)
9. [历史踩坑记录](#9-历史踩坑记录)
10. [核心文件速查](#10-核心文件速查)
11. [参数总表](#11-参数总表)
12. [优先级与覆盖规则](#12-优先级与覆盖规则)

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  本地 laptop                                                │
│  aws cli 已登录 / anyscale cli 已登录 / kubectl 已配         │
└──────────────────┬──────────────────────────────────────────┘
                   │ 仅注入 AWS 凭证 + TRAIN_SCRIPT + SUBMIT_USER
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Anyscale Job (anyscale-poc-tokyo-cloud)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Image: anyscale/image/qwen3-omnispeak-d:N            │   │
│  │ Compute: tokyo-h100-gpu:N (集群规模来自 yaml)         │   │
│  │ Working dir: 本仓库 (excludes models/data/.git)      │   │
│  │ Entrypoint (head pod, m5.8xlarge CPU):               │   │
│  │   python3 anyscale/train/dispatch_train.py            │   │
│  │     ├─ phase 1: 每 worker 拷贝 custom/ → /home/ray/  │   │
│  │     └─ phase 2: 选 worker[0] 跑 TRAIN_SCRIPT         │   │
│  │        └─ bash pipelines/.../train_*.sh               │   │
│  │           └─ swift sft --use_ray                      │   │
│  │              └─ N 个 ray actor（GPU 数由 yaml 自动派生）│   │
│  │                 └─ DeepSpeed ZeRO-2 + EFA RDMA        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 核心文件

| 文件 | 职责 |
|---|---|
| `anyscale/train/job.yaml` | Job 定义（image/compute_config/env_vars/entrypoint） |
| `anyscale/train/dispatch_train.py` | 唯一 Python 入口：phase 1 worker prepare + phase 2 swift dispatch |
| `pipelines/<family>/train_*.sh` | 训练任务脚本（TRAIN_SCRIPT 指向的目标） |
| `pipelines/<family>/common_args.sh` | 训练超参默认值 + swift sft 调用封装 |

### 入口链路

`dispatch_train.py` 分两阶段运行在 head pod：

- **Phase 1 — worker prepare**：将 `custom/` 拷贝到每个 worker 的 `/home/ray/default/custom/`
- **Phase 2 — swift dispatch**：Ray `NodeAffinitySchedulingStrategy` 将 `TRAIN_SCRIPT` 钉到 worker[0] 执行。ms-swift `--use_ray` 自行在 worker 上调度 GPU actor

### 环境变量转发

`dispatch_train.py` 使用 **blocklist 方式**过滤环境变量：

- **Hop 1**（head → ray actor）：过滤 shell 内部变量（`HOME/PATH/PWD/SHELL`）+ `SHLVL/_/...`
- **Hop 2**（ray actor → swift 子进程）：追加剥离 `RAY_*`（保留白名单 `RAY_DEVICE_GROUPS/RAY_WORLD_SIZE/...`）+ 防御性 pop torchrun/torchelastic 残留 key
- 白名单变量（`_RAY_ALLOWLIST`）在 `dispatch_train.py:76-82` 定义

---

## 2. 前置条件

### 2.1 工具安装

```bash
# aws cli v2
aws --version   # ≥ 2.x

# anyscale cli
pip install anyscale   # 或 uv tool install anyscale，≥ 0.26

# kubectl
kubectl version --client   # ≥ 1.28
```

### 2.2 凭证登录

```bash
# AWS — 用 IAM 用户 long-lived key
aws configure --profile default
# 验证：
aws sts get-caller-identity

# Anyscale — 一次性 device flow login
anyscale login
# 验证：
anyscale cloud list
# 期望看到: anyscale-poc-tokyo-cloud

# kubectl — 连 EKS 集群
aws eks update-kubeconfig --name anyscale-poc-v2-eks --region ap-northeast-1
# 验证：
kubectl get nodes -o wide | head
```

### 2.3 检查云端资产

```bash
# 镜像（应看到 :18 或更高）
anyscale image list -n qwen3-omnispeak-d | head

# Compute config（应看到 tokyo-h100:17 或更高）
anyscale compute-config get -n tokyo-h100

```

---

## 3. 提交训练 Job

### 3.1 一键提交

```bash



```

> `TRAIN_SCRIPT` 直接指向 `pipelines/<family>/train_*.sh`，由 `dispatch_train.py` 在 worker[0] 上执行。不再有统一入口 `pipelines/train.sh`。

成功输出示例：

```
[submit] anyscale job submit -f anyscale/train/job.yaml
[submit] job name: qwen3-omnispeak-d-deferzhou-a1b2
Job 'qwen3-omnispeak-d-deferzhou-a1b2' submitted, ID: 'prodjob_XXXXXXXXXXXXXXXXXX'.
View the job in the UI: https://console.anyscale.com/jobs/prodjob_XXXXXXXXXXXXXXXXXX
```

记下 `prodjob_XXX`，后续监控/终止都用它。

### 3.2 提交脚本做了什么

1. 从 `~/.aws/credentials` 读取 AWS 凭证
2. 自动生成三段式 Job 名：`<模型名>-<user>-<4位随机码>`（可由 `SUBMIT_NAME_OVERRIDE` 覆盖）
3. 通过 `--env` 仅注入：`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `TRAIN_SCRIPT` / `SUBMIT_USER` / `SUBMIT_JOB_NAME`
4. 上传 `working_dir`（整个仓库，按 `job.yaml` 的 `excludes:` 过滤）
5. 调用 `anyscale job submit -f anyscale/train/job.yaml`

> **环境变量注入已极致精简**：`submit_job.sh` 只传 AWS 凭证 + TRAIN_SCRIPT + 归档元数据。集群/框架/NCCL/EFA/Wandb 全部写在 `job.yaml` 的 `env_vars` 中，路径变量（`SOURCE_ROOT` 等）也在 `job.yaml` 中。训练超参由各 pipeline 脚本（`common_args.sh` + `train_*.sh` 的 `:=` 兜底）管理。

### 3.3 如需重 build 镜像

只有改了 `anyscale/Dockerfile` 或 `anyscale/requirements.txt` 才需要：

```bash
anyscale image build -n qwen3-omnispeak-d -f anyscale/Dockerfile
# 输出: Image built successfully with URI: anyscale/image/qwen3-omnispeak-d:N
# 然后改 anyscale/train/job.yaml 的 image_uri 行
```

### 3.4 如需注册新 compute config

只有改了 `anyscale/config/compute_config_h100.yaml` 才需要：

```bash
anyscale compute-config create -n tokyo-h100 -f anyscale/config/compute_config_h100.yaml
# 输出: Created compute config: tokyo-h100:N
# 然后改 anyscale/train/job.yaml 的 compute_config 行
```

---

## 4. 修改训练参数

训练参数按以下层级分布（后到覆盖前到）：

| 层级 | 位置 | 内容 |
|---|---|---|
| 1（默认值） | `pipelines/<family>/common_args.sh` | 超参 `:=` 兜底值（`NUM_TRAIN_EPOCHS/LEARNING_RATE/...`） |
| 2（任务默认） | `pipelines/<family>/train_*.sh` | 任务级覆盖（`MODEL_PATH/MODEL_TYPE` + 超参） |
| 3（集群基座） | `anyscale/train/job.yaml` `env_vars:` | SOURCE_ROOT / NCCL/EFA / USE_RAY / Wandb |
| 4（单次实验） | shell `export VAR=val` 后提交 | 临时覆盖任意变量 |

> `job.yaml` 不再包含 `MODEL_PATH/MODEL_TYPE` — 这些由 `train_*.sh` 管理。`anyscale/env.sh`、`anyscale/env.paths.sh`、`pipelines/train.sh` 等已移除，路径变量直接写在 `job.yaml`。

### 4.1 最常见的改法

#### A. 单次实验改超参（最常用）

不改文件，直接 export 后提交：


#### B. 换模型 / 任务变体

改对应 `train_*.sh` 的 `MODEL_PATH` / `MODEL_TYPE`，同时改 `job.yaml` 的 `TRAIN_SCRIPT` 指向该脚本。

#### C. 改集群规模

**唯一来源**：`anyscale/config/compute_config_h100.yaml`。改完推送新 compute config：

```bash
vim anyscale/config/compute_config_h100.yaml
anyscale compute-config create -n tokyo-h100-gpu -f anyscale/config/compute_config_h100.yaml
# bump job.yaml 的 compute_config pin 到新 :N
```

---

## 5. 监控与诊断

### 5.1 Anyscale UI（最方便）

访问 `https://console.anyscale.com/jobs/<prodjob_id>`，查看 4 个标签：Overview / Application logs / Driver logs / System logs

### 5.2 命令行实时 follow

```bash
# 全部日志实时滚动
anyscale job logs --id <prodjob_id> --follow

# 看 step 进度
anyscale job logs --id <prodjob_id> 2>&1 \
  | grep -oE "Train: +[0-9]+%[|][^|]*[|] +[0-9]+/4410 \[[^]]+\]" | tail -5

# 看 loss 时间序列
anyscale job logs --id <prodjob_id> 2>&1 \
  | grep -oE "'loss': [0-9.]+, 'grad_norm': [0-9.]+, 'learning_rate': [0-9.e-]+, 'token_acc': [0-9.]+" \
  | tail -20

# Job 状态（一次性）
anyscale job status --id <prodjob_id>
```

### 5.3 TensorBoard

`job.yaml` 里 `REPORT_TO: tensorboard`，event 文件落在 worker[0] 的 `/tmp/qwen3_omnispeak_output/<task>/runs/`。

**训练完成后**：从 S3 拉取 `tensorboard --logdir ./runs`

**训练中实时看**（需要 port-forward）：

```bash
# 找到 driver actor 所在 worker pod
kubectl get pods -n anyscale-operator -o wide | grep '4/4' | head -1

# 在 pod 里启动 tensorboard
kubectl exec -n anyscale-operator <pod-name> -c ray -- bash -c \
  "nohup /home/ray/anaconda3/bin/tensorboard --logdir /tmp/qwen3_omnispeak_output --bind_all --port 6006 > /tmp/tb.log 2>&1 &"

# 本地 port-forward
kubectl port-forward -n anyscale-operator <pod-name> 6006:6006
# 浏览器开 http://localhost:6006
```

### 5.4 直接进 Pod 抓现场（debug 用）

```bash
# 列 pod
kubectl get pods -n anyscale-operator -o wide | grep -v anyscale-operator-

# 进 worker 看 GPU
kubectl exec -n anyscale-operator <worker-pod> -c ray -- nvidia-smi

# py-spy 看某 actor 在干什么
kubectl exec -n anyscale-operator <worker-pod> -c ray -- bash -c \
  "/home/ray/anaconda3/bin/py-spy dump --pid \$(pgrep -f 'SwiftSft.run' | head -1)"

# 看 actor 环境变量（debug NCCL/EFA 时用）
kubectl exec -n anyscale-operator <worker-pod> -c ray -- bash -c \
  "cat /proc/\$(pgrep -f 'SwiftSft.run' | head -1)/environ | tr '\0' '\n' | grep -E '^(NCCL|FI_|MASTER_|RAY_)' | sort"
```

---

## 6. 取回 Checkpoint

训练完成后，`dispatch_train.py` 会自动把最终 checkpoint 上传到 S3，按
`<user>/<job>/<task>/<run>` 四级组织，便于区分提交人、job 与训练任务：

```
s3://anyscale-poc-v2-bucket-124355679795-ap-northeast-1/
  qwen3_omnispeak/
    checkpoints/
      <user>/                ← 提交人（submit_job.sh 注入 SUBMIT_USER）
        <job-name>/          ← job 名（<model>-<user>-<rand> 或 SUBMIT_NAME_OVERRIDE）
          <task-name>/       ← 训练任务（joint / speech_to_speech_translation / ...）
            <run-name>/      ← v*-<timestamp> 版本目录
              checkpoint-500/
              checkpoint-1000/
              ...
              args.json
              logging.jsonl
              val_dataset.jsonl
              runs/          ← tensorboard event 文件
```

> 未经 `submit_job.sh` 提交（本地 dev / 手动 submit）时，`SUBMIT_USER` /
> `SUBMIT_JOB_NAME` 缺省会落到 `unknown-user` / `unknown-job`。

下载到本地：

```bash
aws s3 cp --recursive \
  s3://anyscale-poc-v2-bucket-124355679795-ap-northeast-1/qwen3_omnispeak/checkpoints/<user>/<job-name>/<task-name>/<run-name>/ \
  ./output/<run-name>/
```

> **注意**：`dispatch_train.py` 在训练 rc=0 时会自动调用 `archive_checkpoint_to_s3()`，将最终 checkpoint 同步到 `FINAL_CHECKPOINT_S3_PREFIX` 指定的 S3 路径。确保该环境变量已设置。

---

## 7. 终止 Job

```bash
anyscale job terminate --id <prodjob_id>
# 或按名字（取最新一个）：
anyscale job terminate -n qwen3-omnispeak-d-bridge-align-joint
```

终止后集群异步释放（pod 进入 Terminating，节点约 2 分钟内空闲）。

---

## 8. 常见故障速查

| 症状 | 大概率原因 | 解法 |
|---|---|---|
| Job 提交后很久还 STARTING，pod 全 Pending | head pod ephemeral-storage 不够 / 没空闲节点 | `kubectl describe pod <head-pod>` 看 events |
| step time > 60s 且 GPU 利用率极低 | NCCL 没用 EFA，回退 TCP socket | 检查 `anyscale/config/compute_config_h100.yaml` 是否有 `vpc.amazonaws.com/efa: 32` |
| 所有 actor 卡住不动，**GPU 0%**（永久 hang） | swift 触发 torchrun 包装（NNODES env 污染 → TORCHELASTIC TCPStore） | 检查 `common_args.sh` USE_RAY guard + `dispatch_train.py` clean_env (详见 §9.1) |
| 所有 actor 卡住不动，**GPU 100% flat + 显存 flat + CPU 100-200%** | Swift rank 0/非0 初始化路径分叉 → barrier↔new_group 循环死锁 | `py-spy dump`：rank 0 在 `new_group` / 其余在 `barrier` = §9.7 已知 bug |
| `ActorUnschedulableError` | ray internal env 污染到 swift actor | 检查 `dispatch_train.py` 的 `clean_env` 白名单是否完整 |
| `bf16/gpu` 检查失败 | swift driver 跑在了 head pod（无 GPU） | 检查 `dispatch_train.py` 是否用 `NodeAffinitySchedulingStrategy` 把 driver 钉到 worker |
| `FileNotFoundError: nvcc` | deepspeed 找不到 nvcc | 检查 Dockerfile 的 nvcc symlink（Layer 4b） |
| `NCCL WARN NET/OFI Failed to find CUDA Runtime library` | aws-ofi-nccl plugin 找不到 cudart | 检查 Dockerfile Layer 4c 的 libcudart.so symlink |
| `PLUGIN_PATH` AssertionError | `custom/` 目录不在预期位置 | `dispatch_train.py` phase 1 应已将 `custom/` 拷到 `/home/ray/default/custom/` |
| 训练完成后 checkpoint 没归档到 S3 | `FINAL_CHECKPOINT_S3_PREFIX` 未设置 | shell export 该变量或在 `job.yaml` 添加 |

---

## 9. 历史踩坑记录

每条给出 **症状 → 根因 → 解法** 链。

### 9.1 ms-swift `--use_ray` 模式下所有 actor 卡死 TCPStore hang

**症状**：跑很久 0% GPU，所有 SwiftSft actor 都 stuck 在 `_create_c10d_store rendezvous.py:190`

**根因**（多层叠加）：
1. `common_args.sh` 在 `USE_RAY=true` 时仍 export `NNODES/NODE_RANK/MASTER_*`
2. 让 swift `use_torchrun()` 返回 True，把 `swift sft` 包装在 `python -m torch.distributed.run`
3. torchrun 写 `TORCHELASTIC_USE_AGENT_STORE=True` 进 os.environ
4. swift `_create_workers` 用 `os.environ.copy()` 复制污染 env 给所有 actor
5. 所有 rank 都被强制 is_master=False，全 connect 不存在的 master

**解法**：
1. `common_args.sh` 加 `USE_RAY` guard 跳过 export NNODES/NODE_RANK/MASTER_*
2. `dispatch_train.py` clean_env 防御层 pop 掉所有 torchrun/horchelastic 残留 key

### 9.2 NCCL fallback 到 TCP socket（没走 EFA）

**症状**：NCCL 跨节点 channel 全是 `via NET/Socket/0`，step time 远超预期

**根因分两阶段**：

**阶段 1**：aws-ofi-nccl plugin 加载成功但 init 失败——`dlopen("libcudart.so")` 失败（cu13 wheel 只提供 `libcudart.so.13`，没有不带版本号的 symlink）

**解法**：Dockerfile Layer 4c 加 `ln libcudart.so.13 → libcudart.so` + `ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:...`

**阶段 2**：plugin init 通过但找不到 EFA device——pod spec 没 request `vpc.amazonaws.com/efa`

**解法**：`compute_config_h100.yaml` worker container resources 加 `vpc.amazonaws.com/efa: '32'`

### 9.3 head pod 200Gi ephemeral-storage 不可调度

**症状**：head pod 永远 Pending，`Insufficient ephemeral-storage`

**根因**：候选非-GPU 节点 `allocatable.ephemeral-storage = 89 GiB`，head 配置 200Gi 过高

**解法**：`compute_config_h100.yaml` head ephemeral-storage 降到 80Gi

### 9.4 head pod 无 GPU 但 transformers 检查 bf16 崩溃

**症状**：driver 直接在 head 跑时，`TrainingArguments(bf16=True).__post_init__` 调 `torch.cuda.is_bf16_supported()` raise

**解法**：`dispatch_train.py` 用 `NodeAffinitySchedulingStrategy` 把 swift sft driver 派到 worker[0]

### 9.5 `RAY_WORLD_SIZE` 硬编码导致集群扩容后只用部分 GPU

**症状**：把 compute_config 从 2 节点扩到 4 节点，但训练只跑了 16 个 actor

**根因**：旧版脚本硬编码 `RAY_WORLD_SIZE=${RAY_WORLD_SIZE:-8}`，yaml 改了但下游用旧值

**解法（v16+ 重构）**：`dispatch_train.py` 在 job 启动时从 compute config yaml 自动派生 `RAY_DEVICE_GROUPS/RAY_WORLD_SIZE/RAY_NPROC_PER_NODE`，删除所有硬编码

**结论**：以后改集群规模只改 `anyscale/config/compute_config_h100.yaml` 一个文件

### 9.6 P5 节点触发 NVIDIA Xid 94 → GPU Unhealthy

**症状**：提交训练任务时只有部分 worker pod 起来，其余 Pending

**根因**：某 P5 节点触发 Xid 94（contained SM ECC error），nvidia-device-plugin 把 7 张 GPU 标记为 Unhealthy

**解法**：在 `compute_config_h100.yaml` 中用 `nodeAffinity` NotIn 黑名单排除该节点；节点修复后把 yaml 改回 4/4 并推新 compute config

### 9.7 NCCL barrier↔new_group 循环死锁 & L4 Head Node 尝试

> 详见 **[anyscale/deadlock_analysis.md](deadlock_analysis.md)** — 含原理图解、诊断决策树、修复方案演进、commit 记录。

---

## 10. 核心文件速查

| 文件 | 谁动 | 何时动 |
|---|---|---|
| `anyscale/train/job.yaml` | 换任务/集群 | 改 `image_uri / compute_config / TRAIN_SCRIPT / SOURCE_ROOT` |
| `anyscale/train/dispatch_train.py` | 偶尔 | 改 swift driver 派发 / env blocklist / phase 1 prepare |
| `anyscale/config/compute_config_h100.yaml` | 后端 | 改节点数 / EFA / volume mount |
| `pipelines/<family>/train_*.sh` | 换模型/任务 | 任务身份（MODEL_PATH/MODEL_TYPE）+ 超参默认 |
| `pipelines/<family>/common_args.sh` | 偶尔 | 超参兜底值 + swift sft 调用封装 |
| `custom/plugin.py` | 偶尔 | 模型注册 + NCCL 死锁修复 |

---

## 11. 参数总表

### 11.1 集群基座（job.yaml env_vars）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SOURCE_ROOT` | `/local/.../liveinterpret_training_pipelines` | NVMe 数据/模型根 |
| `TRAIN_SCRIPT` | `pipelines/bridge_alignment/train_joint.sh` | 训练任务入口 |
| `USE_RAY` | `true` | Ray 模式开关 |
| `NCCL_DEBUG` | `INFO` | NCCL 日志级别 |
| `NCCL_SOCKET_IFNAME` | `^lo,docker0` | NCCL 网络接口过滤 |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | EFA RDMA 使能 |
| `FI_EFA_FORK_SAFE` | `1` | EFA fork 安全 |

完整列表见 `anyscale/train/job.yaml:77-109`。

### 11.2 训练超参（pipelines）

| 变量 | common_args.sh 兜底 | train_*.sh 覆盖示例 | 覆盖方式 |
|---|---|---|---|
| `MODEL_PATH` | `models/qwen3_omnispeak_d` | `models/qwen2_5_3b_omnispeak_d` | 改 train_*.sh |
| `MODEL_TYPE` | `qwen3_omnispeak_d` | `qwen2_5_3b_omnispeak_d` | 改 train_*.sh |
| `NUM_TRAIN_EPOCHS` | `1` | `5` | shell export |
| `LEARNING_RATE` | `1e-4` | `1e-4` | shell export |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | `8` | shell export |
| `GRADIENT_ACCUMULATION_STEPS` | `16` | `4` | shell export |
| `MAX_LENGTH` | `4096` | `4096` | shell export |
| `DEEPSPEED` | `zero2` | — | shell export |
| `ATTN_IMPL` | `flash_attention_3` | — | shell export |
| `RESIDUAL_LOSS` | — | `true` | shell export |

> common_args.sh 用 `:=` 提供兜底；train_*.sh 用 `:=` 提供任务级默认。shell export 可覆盖任意层。

### 11.3 Ray 派生变量

| 变量 | 来源 | 说明 |
|---|---|---|
| `RAY_DEVICE_GROUPS` | `dispatch_train.py` 从 compute config yaml 自动派生 | GPU 分配 JSON |
| `RAY_WORLD_SIZE` | 同上 | 总 GPU 数 |
| `RAY_NPROC_PER_NODE` | 同上 | 每节点 GPU 数 |

---

## 12. 优先级与覆盖规则

```
                    ┌──────────────────────────────────┐
                    │  common_args.sh 兜底              │  ← `:=` 弱默认值
                    └────────────────┬─────────────────┘
                                     │ 被 train_*.sh 覆盖
                                     ▼
                    ┌──────────────────────────────────┐
                    │  train_*.sh 任务默认              │  ← MODEL_PATH/MODEL_TYPE + 超参
                    └────────────────┬─────────────────┘
                                     │ 被 job.yaml 覆盖（同名 KEY）
                                     ▼
                    ┌──────────────────────────────────┐
                    │  job.yaml env_vars                │  ← 集群基座（NCCL/EFA/Wandb/SOURCE_ROOT）
                    └────────────────┬─────────────────┘
                                     │ 被 shell export 覆盖
                                     ▼
                    ┌──────────────────────────────────┐
                    │  shell export VAR=val            │  ← 单次实验入口
                    └──────────────────────────────────┘
```

**本地直跑**：`bash pipelines/bridge_alignment/train_joint.sh`，无需 anyscale 文件。

### 加新参数的 checklist

1. 在 `common_args.sh` 加 `: "${NEW_VAR:=default}"`
2. 如需任务级覆盖，在 `train_*.sh` 加 `: "${NEW_VAR:=task_default}"`
3. `dispatch_train.py` 的 blocklist 自动放行非 shell/Ray 内部变量，无需修改

---

## 附录：命名空间速查

| 名 | 值 |
|---|---|
| Anyscale cloud | `anyscale-poc-tokyo-cloud` |
| EKS cluster | `anyscale-poc-v2-eks` (region `ap-northeast-1`) |
| K8s namespace | `anyscale-operator` |
| S3 bucket | `anyscale-poc-v2-bucket-124355679795-ap-northeast-1` |
| Image | `anyscale/image/qwen3-omnispeak-d:N` |
| Compute config | `tokyo-h100:N` |

## 附录： AI助手命令示例(不消耗token额度)


```
claude-internal --permission-mode bypassPermissions -p "提交任务anyscale/train/job.yaml"

claude-internal --permission-mode bypassPermissions -p "prodjob_inbveyburwxwubg5eziplp21hl 进度"

claude-internal --permission-mode bypassPermissions -p "prodjob_atcarkqklq4argl84qinb6hrgl 失败了"

claude-internal --permission-mode bypassPermissions -p "停掉 prodjob_dutc9z5tlqf8zq2367bi36wgg4"
```

上述命令，也可以直接贴anyscale任务链接来替代任务id

