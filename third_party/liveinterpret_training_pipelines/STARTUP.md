# Startup Guide — liveinterpret_training_pipelines

> 本文档帮助你在新机器上快速恢复训练/评测环境。
> 最后更新：2026-07-17

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA H100 80G ×N（训练 3B 模型至少 2 卡，推荐 8 卡） |
| OS | Linux x86_64（Amazon Linux 2023 / Ubuntu 22.04+） |
| CUDA | 13.0（驱动 ≥ 580） |
| 存储 | ≥ 1TB NVMe（模型 ~30G + 数据 27G + checkpoint ~650G + 缓存 28G） |
| 网络 | 能访问 AWS S3（拉全部数据+代码）；HuggingFace 可选（代码也有 S3 备份） |

需预装的系统工具：
```bash
# AWS CLI（用于从 S3 恢复数据）
pip install awscli && aws configure   # 或使用 IAM role（HyperPod 自带）
# Miniconda（用于创建 Python 环境）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
source /opt/miniconda3/etc/profile.d/conda.sh
```
其余系统依赖（ffmpeg/sox 等）随 conda 环境安装，无需额外 apt。

---

## 2. 获取代码

**方式 A：从 S3 恢复（推荐，自包含无需 HF 访问）**
```bash
BUCKET=s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup/liveinterpret_training_pipelines
mkdir liveinterpret_training_pipelines && cd liveinterpret_training_pipelines
aws s3 sync $BUCKET/custom/ custom/
aws s3 sync $BUCKET/pipelines/ pipelines/
aws s3 sync $BUCKET/inference/ inference/
aws s3 sync $BUCKET/scripts/ scripts/
aws s3 sync $BUCKET/dataset_maker/ dataset_maker/
aws s3 sync $BUCKET/docs/ docs/
aws s3 sync $BUCKET/anyscale/ anyscale/
aws s3 sync $BUCKET/issue/ issue/
aws s3 cp $BUCKET/.gitignore .gitignore
aws s3 cp $BUCKET/CLAUDE.md CLAUDE.md
aws s3 cp $BUCKET/README.md README.md
aws s3 cp $BUCKET/requirements.txt requirements.txt
aws s3 cp $BUCKET/environment.yml environment.yml
aws s3 cp $BUCKET/STARTUP.md STARTUP.md
```

**方式 B：从 HuggingFace 克隆（含 git 历史）**
```bash
git clone https://huggingface.co/Poookeman/liveinterpret_training_pipelines
cd liveinterpret_training_pipelines
git checkout feature/talker-full-add   # 当前开发分支
```

> 两种方式二选一。S3 方式恢复的代码与 git 最新 commit (`5dcadca`) 一致。

---

## 3. 恢复数据（从 S3）

```bash
BUCKET=s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup/liveinterpret_training_pipelines

# 3.1 模型权重（全部从 S3 恢复，共 ~30G）
aws s3 sync $BUCKET/models/qwen2_5_3b_omnispeak_a/ models/qwen2_5_3b_omnispeak_a/        # 3B 基座（训练起点）
aws s3 sync $BUCKET/models/qwen2_5_3b_omnispeak_d/ models/qwen2_5_3b_omnispeak_d/        # 3B 变体 d
aws s3 sync $BUCKET/models/Qwen3-TTS-12Hz-1.7B-Base/ models/Qwen3-TTS-12Hz-1.7B-Base/    # speaker encoder
aws s3 sync $BUCKET/models/Qwen3-TTS-Tokenizer-12Hz/ models/Qwen3-TTS-Tokenizer-12Hz/    # codec tokenizer

# 3.2 训练数据
aws s3 sync $BUCKET/datasets/ datasets/

# 3.3 实验结果（checkpoint + 日志 + 评测产物）
aws s3 sync $BUCKET/output/ output/ --no-follow-symlinks
aws s3 sync $BUCKET/logs/ logs/
aws s3 sync $BUCKET/exp_results/ exp_results/
aws s3 sync $BUCKET/inference_outputs/ inference_outputs/

# 3.4 codec 缓存（28G，可选但强烈建议——跳过后首次数据加载会很慢）
aws s3 sync $BUCKET/.cache/bridge_alignment_codec/ .cache/bridge_alignment_codec/

# 3.5 中间数据（可选）
aws s3 sync $BUCKET/data_maker/ data_maker/
```

> **注意**：`output/checkpoints/` 里有 2 个软链指向 `output/` 内的实体目录。S3 上传时未跟随软链，恢复后需重建：
> ```bash
> cd output/qwen_omnispeak_output/checkpoints
> ln -s ../../upstream_full_add-full-3b-lr1e-5-bs16x8-s2s/v0-20260714-115902 batch4.3/full_add-full-3b-lr1e-5-bs16x8-s2s
> ln -s ../../batch4.4/upstream_full_add-thk_full-3b-lr1e-6-bs16x8-s2s-stg2/v1-20260715-153147 batch4.4/full_add-thk_full-3b-lr1e-6-bs16x8-s2s-stg2
> ```

---

## 4. 创建 Conda 环境

> 前提：已安装 Miniconda（见 §1）。如使用 SageMaker HyperPod，conda 通常已预装在 `/opt/dlami/nvme/zeyangsong/miniconda3/`。

```bash
conda env create -f environment.yml    # 从导出的 YAML 创建
conda activate liveinterpret
```

或手动创建（如果 environment.yml 不兼容）：
```bash
conda create -n liveinterpret python=3.11 -y
conda activate liveinterpret
pip install -r requirements.txt
# Flash-Attention 需单独安装（需 --no-build-isolation）：
pip install flash-attn==2.8.3.post1 --no-build-isolation
```

验证：
```bash
python -c "import swift; print('swift OK')"
swift sft --help | head -1   # 应输出 swift sft 用法
```

---

## 5. 关键路径与配置

| 路径 / 变量 | 说明 |
|---|---|
| `models/qwen2_5_3b_omnispeak_a/` | 3B 基座模型（4.3/4.4 训练起点） |
| `models/Qwen3-TTS-12Hz-1.7B-Base/` | Speaker encoder（speaker prompt 用） |
| `models/Qwen3-TTS-Tokenizer-12Hz/` | Codec tokenizer（音频→离散 token） |
| `datasets/jsonl_zeyang/speech_to_speech_translation_v2.tgt30s.local.dedup.s2sprompt.jsonl` | S2S 训练数据 |
| `custom/` | 核心 model code（modeling/configuration/template/plugin） |
| `pipelines/upstream_talker/` | 训练脚本 |
| `output/qwen_omnispeak_output/checkpoints/` | 已训好的 checkpoint（按批次分组） |

脚本自动探测的关键路径（无需手动设）：
- `QWEN3_TTS_SPEAKER_ENCODER`：脚本先查 `models/`，再查 `project/`
- `QWEN3_TTS_TOKENIZER`：脚本查 `models/Qwen3-TTS-Tokenizer-12Hz`
- `BRIDGE_ALIGNMENT_CODEC_CACHE`：默认 `.cache/bridge_alignment_codec/`

---

## 6. 验证环境（Smoke Test）

跑 10 步验证 pipeline 能跑通（wandb 关闭，输出到临时目录）：

```bash
# 4.3 pipeline smoke test（2 GPU）
CUDA_VISIBLE_DEVICES=0,1 CONDA_ENV=liveinterpret \
  TALKER_METHOD=full_add UNFREEZE_TALKER_LAYERS=28 \
  UNFREEZE_MTP=1 UNFREEZE_CODEC_HEAD=1 RESIDUAL_LOSS_WEIGHT=0.3 \
  REPORT_TO=none BATCH_TAG=smoke_test \
  OUTPUT_DIR=output/qwen_omnispeak_output/smoke_test/batch4.3 \
  PER_DEVICE_TRAIN_BATCH_SIZE=2 MAX_LENGTH=2048 \
  MASTER_PORT=29510 \
  bash pipelines/upstream_talker/train_s2s_upstream.sh --max_steps 10

# 4.4 stage2 smoke test（需先建 clean checkpoint 软链，见下文）
CUDA_VISIBLE_DEVICES=2,3 CONDA_ENV=liveinterpret \
  TALKER_METHOD=full_add UNFREEZE_TALKER_LAYERS=28 \
  UNFREEZE_MTP=1 UNFREEZE_CODEC_HEAD=1 \
  UNFREEZE_THINKER=1 THINKER_CE_LOSS=1 \
  PREV_CHECKPOINT=/path/to/ckpt_clean \
  REPORT_TO=none BATCH_TAG=smoke_test \
  OUTPUT_DIR=output/qwen_omnispeak_output/smoke_test/batch4.4 \
  PER_DEVICE_TRAIN_BATCH_SIZE=2 MAX_LENGTH=2048 \
  MASTER_PORT=29511 \
  bash pipelines/upstream_talker/train_s2s_upstream_stage2.sh --max_steps 10
```

> **Stage2 clean checkpoint**：swift checkpoint 目录含 `.py` 文件会导致 `trust_remote_code` 冲突。需建软链目录只含 model 文件（排除 .py / trainer 文件）：
> ```bash
> SRC=output/.../checkpoint-15805
> DST=/tmp/ckpt_clean && mkdir -p $DST
> for f in config.json *.safetensors model.safetensors.index.json \
>          tokenizer.json tokenizer_config.json vocab.json merges.txt \
>          special_tokens_map.json added_tokens.json chat_template.jinja \
>          preprocessor_config.json video_preprocessor_config.json; do
>     ln -s "$(pwd)/$SRC/$f" "$DST/$f"
> done
> ```

---

## 7. 启动训练

### 4.3 Stage1（talker-only，从 base 模型起步）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CONDA_ENV=liveinterpret \
  TALKER_METHOD=full_add UNFREEZE_TALKER_LAYERS=28 \
  UNFREEZE_MTP=1 UNFREEZE_CODEC_HEAD=1 \
  RESIDUAL_LOSS_WEIGHT=0.3 LEARNING_RATE=1e-5 \
  PER_DEVICE_TRAIN_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 \
  NUM_TRAIN_EPOCHS=5 BATCH_TAG=batch4.3 \
  WANDB_RUN_GROUP="talker input ablation" \
  bash pipelines/upstream_talker/train_s2s_upstream.sh
```

### 4.4 Stage2（解冻 thinker + 文本 CE，从 stage1 checkpoint 续训）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CONDA_ENV=liveinterpret \
  TALKER_METHOD=full_add UNFREEZE_TALKER_LAYERS=28 \
  UNFREEZE_MTP=1 UNFREEZE_CODEC_HEAD=1 \
  UNFREEZE_THINKER=1 THINKER_CE_LOSS=1 THINKER_LOSS_WEIGHT=1.0 \
  RESIDUAL_LOSS_WEIGHT=0.3 LEARNING_RATE=1e-6 \
  PER_DEVICE_TRAIN_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 \
  NUM_TRAIN_EPOCHS=5 BATCH_TAG=batch4.4 \
  PREV_CHECKPOINT=/path/to/stage1_ckpt_clean \
  WANDB_RUN_GROUP="talker input ablation" \
  bash pipelines/upstream_talker/train_s2s_upstream_stage2.sh
```

关键环境变量：
- `TALKER_METHOD`：`full_add` / `reply_add`（切 talker 输入布局）
- `UNFREEZE_TALKER_LAYERS`：0/2/8/28（解冻 talker 层数）
- `UNFREEZE_MTP` / `UNFREEZE_CODEC_HEAD`：是否解冻残差 MTP / 主码头
- `UNFREEZE_THINKER` / `THINKER_CE_LOSS`：stage2 解冻 thinker + 文本 CE
- `REPORT_TO`：`wandb`（默认）/ `none`（关闭）
- `BATCH_TAG`：日志归档批次号
- `--max_steps N`：附加参数，限制训练步数（smoke test 用）

---

## 8. 已有实验产出

详见 `docs/EXPERIMENTS.md`。核心结果：

| 批次 | run | token_acc(主码) | 说明 |
|---|---|---|---|
| 4.3 | full-add | 0.474 | stage1，talker-only |
| 4.3 | reply-add | 0.437 | stage1 |
| 4.3 | bridge-add | 0.490 | stage1 |
| 4.4 | full-add-stg2 | 0.517 | stage2，解冻 thinker |
| 4.4 | reply-add-stg2 | 0.471 | stage2 |

Checkpoint 位置：`output/qwen_omnispeak_output/checkpoints/batch4.{3,4}/<RUN_NAME>/`

---

## 9. wandb 配置（可选）

本地 wandb server：`http://43.173.190.13:8080`

```bash
export WANDB_API_KEY=<your_key>    # 见 ~/.netrc
export WANDB_BASE_URL=http://43.173.190.13:8080
# 如走代理，需放行 wandb server：
export NO_PROXY=43.173.190.13,localhost,127.0.0.1
```

不需要 wandb 时设 `REPORT_TO=none`。

---

## 10. 常见问题

| 问题 | 解决 |
|---|---|
| `swift: command not found` | `conda activate liveinterpret` |
| `Could not locate speaker-encoder` | 确认 `models/Qwen3-TTS-12Hz-1.7B-Base/` 存在，或 `export QWEN3_TTS_SPEAKER_ENCODER=/path/to/it` |
| OOM（bs16 全解冻 3B） | 降 `PER_DEVICE_TRAIN_BATCH_SIZE=8` + `GRADIENT_ACCUMULATION_STEPS=2`（有效 batch 不变） |
| wandb 卡死 | 设 `REPORT_TO=none` 或配 `NO_PROXY` 放行 wandb server |
| Stage2 加载 checkpoint 报 `KeyError` | 用 clean checkpoint 软链目录（见 §6），排除 `.py` 文件 |
| DDP 端口冲突 | 改 `MASTER_PORT=2950X` |
