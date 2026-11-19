# Live Interpret Training Pipelines (`liveinterpret_training_pipelines`)

`liveinterpret_training_pipelines` 是一个独立的训练管线项目，基于
[ms-swift](https://github.com/modelscope/ms-swift) 提供 `qwen_omnispeak`
模型的多阶段训练能力，覆盖：

- **Stage 0 (预处理)**：离线生成 codec labels、codec input ids、Thinker hidden cache 等，降低训练期重复计算。
- **Stage 1 (Bridge Alignment)**：训练 DualRoute Bridge + 残差 Code Predictor + Talker 底 4 层，端到端把 Thinker 隐状态对齐到 Talker 的 codec 输入空间（主码 + 残差码联合监督）。
- **Stage 2 (Continue Pretrain)**：Thinker 上层 LoRA + Qwen3-TTS top-16 上层 LoRA 联合训练，提升自然度、说话人一致性与语义保真。
- **Stage 3 (SFT)**：codec-only speech-to-speech 有监督微调，分两个子阶段——Stage-1 从非实时翻译过渡到实时翻译（`normal_translation` / `bilingual_translation` / `multi_speaker_translation`），Stage-2 纯实时同传策略学习（`simultaneous_policy_learning`）。通过流式 READ/WRITE 注意力掩码与静默 codec 监督，隐式控制翻译时延与语义边界，不引入额外 loss 头。

所有阶段共享一套自洽的 `qwen_omnispeak` 模型注册、Qwen3-TTS codec
tokenizer 与混合监督模板代码，通过 ms-swift `--external_plugins` 接入。

qwen_omnispeak 模型原始checkpoint存放在`s3://iegg-speech-model-models/origin_models/custom_models/qwen_omnispeak_*`，*为`a/b/c/d/e/f` 之一。

### 训练入口与后端

每个阶段在 `pipelines/<stage>/` 下提供**按任务拆分的 `train_*.sh` 脚本**（如
`train_speech_to_speech_translation.sh`、`train_normal_translation.sh`），不再有统一的
`pipelines/train.sh` 入口。后端通过环境变量 `TRAIN_BACKEND`（默认 `deepspeed`）选择，
由脚本内部 `common_args.sh` 分发：

| 后端 | 并行策略 | 适用场景 | 选择方式 |
| --- | --- | --- | --- |
| deepspeed（ms-swift） | DP + ZeRO-2 | 单机 / 多机（Ray） | `TRAIN_BACKEND=deepspeed`（默认） |
| megatron | TP/EP/DP + ZeRO-1 | 大规模高性能训练 | `TRAIN_BACKEND=megatron`（或 sft 的 `train_*_megatron.sh` 变体） |
| swift + Ray | DP + ZeRO（Ray 调度） | Anyscale 多机 | 见 `anyscale/` 目录 |

`bridge_alignment` 还用 `TRAIN_TARGET`（默认 `adapter`）区分训练目标（`adapter` / `bridge`）。
其余超参（MBS / GBS / MAX_LENGTH / SCENE 等）默认事实源在各阶段 `common_args.sh` 的 `SCENE` 预设，
不在任务 yaml 设默认值（集群提交见 `ktrain/` / `pytorch_job/` README）。

用法：

```bash
# bridge_alignment：单脚本 + TRAIN_BACKEND 选后端
TRAIN_BACKEND=deepspeed  bash pipelines/bridge_alignment/train_speech_to_speech_translation.sh
TRAIN_BACKEND=megatron    bash pipelines/bridge_alignment/train_speech_to_speech_translation.sh

# sft：megatron 走独立脚本变体
bash pipelines/sft/train_normal_translation.sh
bash pipelines/sft/train_normal_translation_megatron.sh
```

> Megatron 方案通过非侵入式 mcore-bridge 适配把 thinker 映射为 Megatron
> `GPTModel`（TP/EP 切分），并原生计算 talker codec loss（与 DeepSpeed 路径
> 数值一致，已 CPU 验证）。详见各阶段 README 与「Megatron talker codec loss」一节。

## 目录结构

推理与文档导航：

- [Omni offline/streaming 与 S4/S5 入口及调用关系](docs/INFERENCE_ENTRYPOINTS.md)
- [文档索引](docs/README.md)
- [仓库整理记录](docs/REPOSITORY_MAINTENANCE.md)

```
liveinterpret_training_pipelines/
├── README.md                      # 本文件
├── requirements.txt               # 项目依赖
├── custom/                        # 跨阶段共享代码（ms-swift plugin + qwen_omnispeak 注册 + WDS 音频）
│   ├── plugin.py                  # ms-swift external_plugins 入口
│   ├── dataset.py                 # 各阶段 RowPreprocessor 与数据集注册
│   ├── template.py                # 各阶段 Template 重载与混合 collator
│   ├── codec_tokenizer.py         # Qwen3-TTS-Tokenizer-12Hz 包装与磁盘缓存
│   ├── wds_audio.py               # WDS manifest 中 wds:// URI 解析
│   └── qwen_omnispeak/           # qwen_omnispeak 模型注册 / 单文件 modeling
│       ├── register_qwen_omnispeak.py
│       ├── configuration_qwen_omnispeak.py
│       ├── modeling_qwen_omnispeak.py
│       ├── export_qwen_omnispeak.py
│       ├── mcore_adaptation.py         # 非侵入式 mcore-bridge 注册（Megatron 方案）
│       ├── mcore_talker_loss.py        # Megatron 原生 talker codec loss
│       ├── test_qwen_omnispeak.py
│       ├── test_mcore_adaptation.py
│       ├── test_mcore_talker_loss.py   # codec loss 与 DeepSpeed 路径数值一致性
│       └── README.md
├── pipelines/
│   ├── bridge_alignment/          # Stage 1：Bridge Alignment 训练
│   │   ├── common_args.sh              # 双后端（deepspeed + megatron）共用参数 + SCENE 预设
│   │   ├── init_env.sh                 # 环境初始化（路径 / conda / 数据发现）
│   │   ├── data_format.jsonl
│   │   ├── train_joint.sh
│   │   ├── train_tts.sh
│   │   ├── train_speech_to_speech_conversation.sh
│   │   ├── train_speech_to_speech_translation.sh   # 主入口（TRAIN_BACKEND 选后端，TRAIN_TARGET 选目标）
│   │   ├── run_reply_add_wds_eos_history.sh
│   │   ├── PERF_TUNING.md
│   │   └── README.md
│   ├── continue_pretrain/         # Stage 2：Continue Pretrain
│   │   ├── common_args.sh
│   │   ├── common_args_megatron.sh
│   │   ├── init_env.sh
│   │   ├── data_format.jsonl
│   │   ├── train_*.sh（按任务拆分；megatron 走 _megatron.sh 变体）
│   │   └── README.md
│   ├── sft/                       # Stage 3：SFT（各子阶段 train_*_*.sh，含 _megatron.sh）
│   │   ├── common_args.sh
│   │   ├── common_args_megatron.sh
│   │   ├── data_format.jsonl
│   │   ├── train_normal_translation.sh / _megatron.sh
│   │   ├── train_bilingual_translation.sh / _megatron.sh
│   │   ├── train_multi_speaker_translation.sh / _megatron.sh
│   │   ├── train_joint_stage1.sh / _megatron.sh
│   │   ├── train_simultaneous_policy_learning.sh / _megatron.sh
│   │   ├── Normal-Translation.md ...（各子阶段设计文档）
│   │   └── README.md
│   ├── pretrain/                  # 占位（仅 data_format.jsonl）
│   └── rl/                        # 占位（仅 data_format.jsonl）
├── dataset_maker/                 # 数据集制作 ETL（S3 音频本地化 / 切分 / shards 生成）
├── inference/                     # 推理脚本
├── test_script/                   # 测试脚本
├── docs/                          # 设计 / 架构文档
├── anyscale/                      # 云端训练（Anyscale 托管 Ray on EKS）
├── ktrain/                        # KtrainJob 提交到 H200 EKS（ktrain CLI + CRD）
│   ├── README.md                  # 提交 / 同步 / 监控 / 排错（详见该 README）
│   ├── launch_train.sh            # 容器入口：s3:// / s3mnt:// / fsx:// URI 转容器内路径 + 软链
│   ├── ktrainjob_sts_translation.yaml          # 单节点 / common_voice S3
│   ├── ktrainjob_sts_zeyang_custom_data_train.yaml  # 双节点 / unist FSX + 跨区 S3
│   └── image/ (Dockerfile + build_image.sh)
└── pytorch_job/                   # 纯 kubectl PyTorchJob 提交到 H200 EKS
```

> 各阶段 `train_*.sh` 通过 `SCRIPT_DIR` 自定位仓库根目录，可在仓库任意位置以绝对 /
> 相对路径执行；`PLUGIN_PATH`、`MODEL_PATH`、`OUTPUT_ROOT` 等默认基于 repo root 推导，
> 无需手动设置。集群提交（ktrain / pytorch_job）通过 yaml 的 `TRAIN_SCRIPT` 指定这些脚本的
> 相对路径，由对应 `launch_train.sh` 拼出 `/s3all/<CODE_PREFIX>/<TRAIN_SCRIPT>` 执行。

## 设计要点

- **共享插件**：唯一的 `--external_plugins` 入口是
  `liveinterpret_training_pipelines/custom/plugin.py`，注册六个 qwen_omnispeak 变体、
  各阶段数据集与 template（bridge_alignment 仅 codec 监督，sft 为 codec-only 流式同传）。
- **混合监督模板**：`Qwen3OmniSpeakJointTrainingTemplate` 在 batch 内允许文本-only
  与 codec-supervised 样本共存，通过 `has_codec_supervision` per-row gating 仅对
  有 codec 监督的样本反传 codec loss。
- **目标语音 tokenizer**：`Qwen3/Qwen3-TTS-Tokenizer-12Hz`，可通过
  `QWEN3_TTS_TOKENIZER` 覆盖。编码结果按 fingerprint 写入磁盘缓存。
- **可训练 thinker 路径**：`self.training=True` 时不再 `@torch.no_grad()`，
  使联合训练能端到端反传 thinker 梯度；推理保留原行为。
- **S2S 推理 streaming**：Thinker/Talker/predictor KV-cache、predictor/decoder compile、
  H200 ablation 与推荐配置见 [`inference/README.md`](inference/README.md)「S2S 流式推理加速开关」。
- **支持 full / LoRA / DeepSpeed / 分布式 / Ray / Megatron**：见各阶段 README。

## Megatron talker codec loss

在不修改 `mcore-bridge` / `ms-swift` 源码的前提下，让 talker codec loss
走 Megatron 原生并行：

- `mcore_adaptation.py`：thinker 注册为 Megatron `GPTModel`（TP/EP/DP/ZeRO-1）；
  talker 等子模块通过 `module_mapping` 复制式加载。
- `mcore_talker_loss.py`：在 `accept_hidden_layer` 用 forward hook 捕获隐藏态，
  计算主码 CE + 残差 CE，通过 monkey-patch `MegatronTrainer.forward_step` 注入。
- **数值一致性**：`test_mcore_talker_loss.py` 已验证 codec loss 与 DeepSpeed 路径
  `|Δ|=0`。

> **约束**：`PP=1`、`CP=1`、`sequence_parallel=false`（脚本已默认）。
> 不满足会明确报错。`QWEN_OMNISPEAK_MCORE_CODEC_LOSS=false` 退回纯文本训练。

## 环境准备

### 依赖安装

`requirements.txt` 已声明本项目独立运行所需的最小依赖（包含 ms-swift
框架本体）：

```bash
pip install -r liveinterpret_training_pipelines/requirements.txt
```

如果你已经在 ms-swift 仓库内部使用，则该项目的依赖被 ms-swift 自身
覆盖，重复安装是幂等的。

### 可选组件

- **Ray**：`pip install ray`
- **DeepSpeed**：`pip install deepspeed`（如需 ZeRO-2/3）
- **Flash-Attention**：`pip install flash-attn --no-build-isolation`
- **Qwen3-TTS Tokenizer**：`pip install qwen-tts && python -c "from qwen_tts import Qwen3TTSTokenizer; print('ok')"`
- **Megatron**：`megatron-core`、`mcore-bridge` 在 `requirements.txt` 中；
  `transformer_engine` 需从源码编译：`pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable --no-build-isolation`

### 模型与 transformers

`qwen_omnispeak` 复用 `transformers.models.qwen3_omni_moe`，需要
`transformers>=4.57.dev0`。变体目录默认指向
`models/custom_models/qwen_omnispeak_b`，可通过 `MODEL_PATH` /
`MODEL_TYPE` 切换。

## 快速开始

各阶段入口是 `pipelines/<stage>/train_*.sh`（不再有统一 `train.sh`）。后端用
`TRAIN_BACKEND`（默认 `deepspeed`）选择；`bridge_alignment` 另用 `TRAIN_TARGET`
（`adapter` / `bridge`）选训练目标。

### 单卡（仅冒烟）

```bash
NPROC_PER_NODE=1 TUNER_TYPE=full \
bash pipelines/bridge_alignment/train_speech_to_speech_translation.sh
```

### 单机多卡（DeepSpeed / 默认后端）

```bash
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TUNER_TYPE=full DEEPSPEED=zero2 \
bash pipelines/bridge_alignment/train_joint.sh
```

### 多机多卡 / Ray 集群

本地多机一般经 **Ray** 调度（见 `anyscale/`）；云端多机多卡走 **Anyscale**（Ray on EKS）
或 **ktrain / PyTorchJob**（H200 EKS，纯 kubectl / ktrain CLI，见下节「集群训练」）。
`bridge_alignment` 脚本内已按 `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` / `MASTER_ADDR` /
`MASTER_PORT` 组装分布式环境，无需单独的 `train.sh` 透传。

```bash
# head 节点：ray start --head --port=6379；worker：ray start --address='<head-ip>:6379'
# 仅在 head 节点运行（Ray 聚合多机）：
USE_RAY=true RAY_WORLD_SIZE=16 RAY_NPROC_PER_NODE=8 \
bash pipelines/bridge_alignment/train_joint.sh
```

> 详见 <https://swift.readthedocs.io/zh-cn/latest/Instruction/Ray.html>。

### Megatron 大规模训练

```bash
# TP=2, EP=2（thinker 为 MoE），ZeRO-1 由 DistributedOptimizer 提供
TRAIN_BACKEND=megatron TP=2 PP=1 CP=1 EP=2 NPROC_PER_NODE=8 \
bash pipelines/bridge_alignment/train_speech_to_speech_translation.sh
```

> 含 codec loss 的任务需 `PP=1`、`CP=1`、`sequence_parallel=false`（`common_args.sh` 已默认设好）。

### LoRA

```bash
TUNER_TYPE=lora \
LORA_TARGET_MODULES=all-linear \
LORA_RANK=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 \
FREEZE_VIT=true \
bash pipelines/sft/train_normal_translation.sh
```

### TRAIN_SCRIPT（集群提交入口）

`ktrain` / `PyTorchJob` / `Anyscale` 通过 yaml 的 `TRAIN_SCRIPT` 指定要执行的 `train_*.sh`
repo 相对路径，容器内由 `launch_train.sh` 按 `CODE_PREFIX` 拼出
`/s3all/<CODE_PREFIX>/<TRAIN_SCRIPT>` 执行。示例取值：

```bash
TRAIN_SCRIPT=pipelines/bridge_alignment/train_speech_to_speech_translation.sh
TRAIN_SCRIPT=pipelines/sft/train_normal_translation_megatron.sh
```

> 集群提交时 `TRAIN_SCRIPT` 由 yaml env 注入（见 `ktrain/README.md` / `pytorch_job/README.md`）；
> 本地直接运行对应 `train_*.sh` 即可，无需设 `TRAIN_SCRIPT`。

### 任务映射

| 管线 family | 任务 task | 说明 |
|---|---|---|
| `bridge_alignment` | `joint` | 联合训练（TTS + S2SConv + S2STrans） |
| `bridge_alignment` | `tts` | 纯 TTS 训练 |
| `bridge_alignment` | `speech_to_speech_conversation` | 语音到语音对话 |
| `bridge_alignment` | `speech_to_speech_translation` | 语音到语音翻译 |
| `continue_pretrain` | `joint` | 继续预训练联合任务 |
| `continue_pretrain` | `speech_to_speech_translation` | S2ST 继续预训练 |
| `continue_pretrain` | `speech_to_text_translation` | S2TT 继续预训练 |
| `continue_pretrain` | `multi_speaker_segmentation` | 多说话人分割 |
| `sft` | `normal_translation` | 常规翻译微调 |
| `sft` | `bilingual_translation` | 双语翻译微调 |
| `sft` | `multi_speaker_translation` | 多说话人翻译微调 |
| `sft` | `simultaneous_policy_learning` | 同传策略学习 |
| `sft` | `joint_stage1` | SFT Stage1 联合训练 |

> 上表为任务目录（family / task），对应 `pipelines/<family>/train_<task>.sh`
> （sft 的 Megatron 变体为 `train_<task>_megatron.sh`）。不再通过 `train.sh` 的命令行参数选择。

### streaming_si（Qwen3-0.6B 流式同传，迁移自 streaming-si-0.6b）

完整文档（文件清单、各阶段参数、数据准备、S4 细节、验证方法、差异与限制、常见报错）见
[`pipelines/streaming_si/README.md`](pipelines/streaming_si/README.md)。

Voxtral 音频塔 + Thinker + CosyVoice speech-token Talker，代码在 `custom/streaming_si/`，与 qwen_omnispeak 互不影响。
有两种模型变体（`variants.py`），共用 template、数据集和下表全部训练脚本：

| model_type | Thinker | Talker | 参数量（thinker / talker） |
|---|---|---|---|
| `streaming_si` | Qwen3-0.6B | 0.1B：8 层 Qwen3 风格 decoder | 0.60B / 0.14B |
| `streaming_si_4b` | Qwen3.5-4B（24 层 Gated DeltaNet + 8 层门控全注意力） | 0.6B：Qwen3-0.6B 结构（28 层） | 4.21B / 0.47B |

变体由导出时的 `--variant` 决定并写入 config（`thinker_arch`），训练脚本按 `MODEL_PATH` 自动识别 model_type。
transformers 4.57 缺少的两个结构由兼容层提供，均与官方实现逐元素对拍（max|Δ|<1e-6）：
`voxtral_compat.py`（Voxtral realtime 编码器，对 transformers 5.8）、`qwen3_5_compat.py`（Qwen3.5 文本塔，对 5.17）。

阶段按执行顺序编号（与源项目编号的对应见 `pipelines/streaming_si/README.md` 第 3 节）：

| 阶段 | 任务 | 入口 | 可训参数 | MODEL_PATH |
|---|---|---|---|---|
| S1 | TTS | `train_tts.sh s1` | talker（cond=text_only） | `export_streaming_si.py` 导出的基座目录 |
| S2 | 语音→文本翻译 | `train_st.sh` | audio_proj + thinker | 同上（可与 S1 并行） |
| S3-w | TTS | `train_tts.sh s3-w` | thinker_proj + fusion（桥接预热） | `merge_stage_checkpoints.py`（S2 thinker + S1 talker） |
| S3 | TTS | `train_tts.sh s3` | talker（cond=full） | S3-w 的 checkpoint |
| S4-w | 语音→语音翻译 | `train_s2st.sh s4-w` | thinker_proj + fusion（跨池桥接预热） | S3 的 checkpoint |
| S4 | 语音→语音翻译 | `train_s2st.sh s4` | audio_proj + thinker + talker | S4-w 的 checkpoint |
| S5 | 流式读写时机 | `train_simul.sh` | LoRA（thinker/talker）+ 4 行流式 token + thinker_proj/fusion | S4 的 checkpoint |

```bash
python custom/streaming_si/scripts/export_streaming_si.py --thinker_dir <Qwen3-0.6B> \
    --voxtral_dir <Voxtral-Mini-4B-Realtime-2602> --out_dir models/streaming_si_base
MODEL_PATH=models/streaming_si_base SI_S1_DATA='<s1_emilia_fsq_v3>/*.parquet' \
    bash pipelines/streaming_si/train_tts.sh s1
# 4B 变体：换基座与 --variant，后续各阶段命令完全相同
python custom/streaming_si/scripts/export_streaming_si.py --variant streaming_si_4b --thinker_dir <Qwen3.5-4B> \
    --voxtral_dir <Voxtral-Mini-4B-Realtime-2602> [--talker_init_dir <Qwen3-0.6B>] --out_dir models/streaming_si_4b_base
# CPU 连通性冒烟：python custom/streaming_si/scripts/make_smoke_fixture.py ... 后加 SCENE=smoke
```

S5（chunk-as-turn 流式同传，即源项目 S4，相关模块与变量沿用 `s4` 命名）要点：

- 数据准备：`dataset_maker/streaming_si/build_s4_val_split.py`（按源文本分组切 val）与
  `build_record_index.py`（NIR × 句长分层索引），用法见 `train_simul.sh` 头部注释。
- 样本编码、分层采样、prefix-drop 调度复用源项目的 `S4StreamingDataset` / `si_sampler` /
  `s4_context_views`（原样迁入，源项目 80 个相关单测在本仓库通过）；swift 数据集只存指针行
  （`s4_pipeline.py`），采样在加载时一次性物化，`--resume_from_checkpoint` 精确续训。
- `--tuner_type streaming_si_s4`（`s4_peft.py`），只保存 adapter，格式与源一致（含 `s4_metadata.json`）；
  `SI_S4_TRAIN_SCOPE=joint|thinker|talker`。验证集以 `eval_dec_macro_f1` 选 best。
- 实验臂：`S4_ARM=prefix_drop|prefix_drop_control S4_P_DROP=<比例>`（约束同源：thinker scope、speech_w=0、m=1）。
- 与源项目的差异：多 epoch 时分层抽样结果在各 epoch 间复用（源为每个 epoch 重新抽）；验证只报两方向合并后的
  ALL 口径（没有分方向指标，也没有 prefix-drop 的 full/crop 配对验证）；学习率调度与源相差一个步位。
- 对拍：`custom/streaming_si/tests/test_s4_pipeline.py`（采样计划 / 逐样本编码 / collate / loss / macro-F1，需源仓库）。

`streaming_si_4b` 变体注意事项：

- Qwen3.5 官方发布为多模态模型：导出只取文本塔（`model.language_model.*`），丢弃视觉塔与 MTP；
  `<|audio_pad|>` 使用词表中已有的 248076，S4 流式 token 追加在其后（248077 起，词表空位 243 个）。
- S4 LoRA 在线性注意力层挂 `in_proj_qkv / in_proj_z / out_proj`（全注意力层仍是 q/k/v/o），共 396 个目标。
- 兼容层不支持线性注意力的增量缓存（只覆盖训练前向），流式推理需另行实现。
- GPU 训练请安装 `flash-linear-attention` 与 `causal-conv1d`，否则线性注意力走纯 torch 实现，速度很慢。
- 各阶段 batch size 等默认值沿用 0.6B 变体，未针对 4B 调过，上集群前需先探测显存。

### simuls2st_omni（Qwen2.5-Omni-3B + DualCodec talker 三阶段，迁移自 SimulS2ST-Omni-stage123）

完整文档（接入方式、与源项目的差异、对拍结果、ktrain 用法、推理）见
[`pipelines/simuls2st_omni/README.md`](pipelines/simuls2st_omni/README.md)。

源训练 / 推理代码原样迁入 `custom/simuls2st_omni/`，`integration/` 把它接到 swift：model_type / template /
tuner 都叫 `simuls2st_omni`，数据集按 `SIMULS2ST_CONFIG`（阶段配置 yaml）注册，未设置时跳过。

| 阶段 | 入口 | 训练范围 |
|---|---|---|
| Stage 1 | `pipelines/simuls2st_omni/train_stage1.sh` | thinker 冻结，talker 全参（Emilia DualCodec TTS） |
| Stage 2 | `train_stage2.sh` | thinker LoRA r32 + talker 全参，5 类任务按步混合 |
| Stage 3 | `train_stage3.sh` | 合并 Stage 2 后挂 thinker / talker LoRA r8，流式轨迹 |

- 训练参数的唯一事实源是 `pipelines/simuls2st_omni/configs/stage{1,2,3}.yaml`（与源项目逐字相同）。
- 单卡与源项目逐步一致；多卡修正了源项目梯度不同步的问题，并对 Stage 2/3 开启 `ddp_find_unused_parameters`。
- 不要加 `--deepspeed`（ZeRO 会更新本该跳过的参数）。
- checkpoint 为源格式（`talker_components.pt` + `thinker_lora/` / `talker_lora/`），源项目推理脚本与桶里的源 checkpoint 可互通。

## Anyscale 云端训练

`anyscale/` 提供把上述训练管线一键提交到 Anyscale 托管集群（Ray on EKS，
cloud `anyscale-poc-tokyo-cloud`，region `ap-northeast-1`）的能力。本地只需装好
`aws` / `anyscale` / `kubectl` 三件套并登录，即可提交、监控、取回 checkpoint。

### 三种后端编排

| 子目录 | 编排方式 | 并行策略 | 适用场景 |
| --- | --- | --- | --- |
| `anyscale/train/` | Swift + Ray | DP + DeepSpeed ZeRO-2 | 主路径，自动节点发现 |
| `anyscale/ray_megatron/` | Ray 调度 + Megatron | TP/EP/DP + ZeRO-1 | 大规模 Megatron，自动节点发现 |
| `anyscale/torch_megatron/` | 纯 torchrun + Megatron | TP/EP/DP + ZeRO-1 | 无 Ray 依赖，手写 hostfile |


提交后记下 `prodjob_XXX`，用 `anyscale job logs/status/terminate --id <prodjob_id>`
监控与终止。训练完成后 `dispatch_train.py` 自动把最终 checkpoint 归档到 S3，按
`<user>/<job>/<task>/<run>` 四级组织。

### 参数层级

集群上的参数优先级为（后者覆盖前者）：`pipelines/<family>` 默认（兜底） →
`job.yaml` 的 `env_vars`（任务身份 `MODEL_PATH/MODEL_TYPE` + 集群基座 NCCL/EFA/Wandb）
→ shell `export`（单次实验）。`job.yaml` 的 `entrypoint` 直接运行
`python3 anyscale/train/dispatch_train.py`（head pod 编排，训练实跑在 worker）。改集群
规模只改 `anyscale/config/compute_config_h100.yaml` 的 `min_nodes/max_nodes` 一处。

> 完整的提交流程、监控/诊断、checkpoint 取回、故障速查与历史踩坑记录见
> [`anyscale/README.md`](anyscale/README.md)。

## 集群训练（H200 EKS）：ktrain / PyTorchJob

除 `anyscale/`（Ray on EKS，ap-northeast-1）外，本仓库还提供两条直接提交到
**H200 EKS 集群 `speech-h200-eks`（新加坡，ap-southeast-3）**的训练路径，二者共用同一镜像 /
节点 / 数据 / 代码，仅提交方式不同：

| 目录 | 提交方式 | 说明 |
| --- | --- | --- |
| `ktrain/` | `ktrain` CLI + KtrainJob CRD（operator 渲染为 PyTorchJob） | 见 [`ktrain/README.md`](ktrain/README.md) |
| `pytorch_job/` | 纯 `kubectl apply` 原生 PyTorchJob | 见 [`pytorch_job/README.md`](pytorch_job/README.md) |

两条路径的共同约定：
- 训练超参（MBS/GBS/MAX_LENGTH/SCENE 等）**唯一事实源是训练脚本**（`pipelines/bridge_alignment/common_args.sh` 等），任务 yaml 只放硬件 / 工件 S3(FSX) 路径 / 环境接线，不覆盖训练超参。
- 工件（模型 / tokenizer / manifest / 音频）在提交时由 skill 脚本并行同步到节点 NVMe（`/opt/dlami/nvme` hostPath，跨重启持久）；代码经 S3 CSI 直接读（`/s3all`），由 `launch_train.sh` 按 `TRAIN_SCRIPT` + `CODE_PREFIX` 拼出执行路径。
- WDS 音频经 `/curated_audio -> /nvme/data` 符号链接命中；H200 EFA 需 `libnccl-net.so` 符号链接 + `/dev/shm` emptyDir 64Gi（均在 `launch_train.sh` 内处理）。

> 提交 / 同步 / 监控 / 排错完整流程与已知坑见各自 README，不要凭仓库脚本臆测命令。

## 文档索引

- 阶段 README：[`pipelines/bridge_alignment/README.md`](pipelines/bridge_alignment/README.md) |
  [`pipelines/continue_pretrain/README.md`](pipelines/continue_pretrain/README.md) |
  [`pipelines/sft/README.md`](pipelines/sft/README.md) |
  [`pipelines/streaming_si/README.md`](pipelines/streaming_si/README.md) |
  [`pipelines/simuls2st_omni/README.md`](pipelines/simuls2st_omni/README.md)
- 模型文档：[`custom/qwen_omnispeak/README.md`](custom/qwen_omnispeak/README.md) |
  [`custom/qwen_omnispeak/QWEN_OMNISPEAK_MODEL_DESIGN.md`](custom/qwen_omnispeak/QWEN_OMNISPEAK_MODEL_DESIGN.md)
- 云端训练：[`anyscale/README.md`](anyscale/README.md)（提交 / 监控 / 故障排查） |
  [`anyscale/ray_megatron/README.md`](anyscale/ray_megatron/README.md) |
  [`anyscale/torch_megatron/README.md`](anyscale/torch_megatron/README.md)
- H200 EKS 集群训练：[`ktrain/README.md`](ktrain/README.md)（ktrain CLI + KtrainJob） |
  [`pytorch_job/README.md`](pytorch_job/README.md)（纯 kubectl PyTorchJob）

## 常见环境变量

| 变量 | 默认值 | 适用阶段 | 说明 |
| --- | --- | --- | --- |
| `MODEL_PATH` | `<repo>/models/qwen_omnispeak_b` | 全部 | 本地导出的 qwen_omnispeak 目录 |
| `MODEL_TYPE` | `qwen_omnispeak_b` | 全部 | `a/b/c/d/e/f` 之一 |
| `PLUGIN_PATH` | `<repo>/liveinterpret_training_pipelines/custom/plugin.py` | 全部 | ms-swift external_plugins |
| `QWEN3_TTS_TOKENIZER` | `Qwen3/Qwen3-TTS-Tokenizer-12Hz` | 涉及语音生成 | 目标音频 tokenizer |
| `BRIDGE_ALIGNMENT_DATA` | `<stage>/data_format.jsonl` | 对应阶段 | 训练 JSONL |
| `BRIDGE_ALIGNMENT_CODEC_CACHE` | `<repo>/.cache/...` | 对应阶段 | codec 编码磁盘缓存 |
| `TUNER_TYPE` | `full` | continue_pretrain | `full` / `lora` |
| `FREEZE_PARAMETERS` | Code2Wav 正则 | continue_pretrain | 强制冻结的参数正则 |
| `TRAINABLE_PARAMETERS` | (空) | continue_pretrain | 额外可训练前缀，逗号分隔 |
| `DEEPSPEED` | `zero2` | Swift | ms-swift 内置 ds 配置 |
| `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` | `1`/`1`/`0`/`127.0.0.1`/`29500` | 全部 | 分布式启动 |
| `USE_RAY` / `RAY_EXP_NAME` / `RAY_DEVICE_GROUPS` | `false`/空/空 | Swift + Ray | Ray 透传（`train.sh` 自动生成 device_groups） |
| `RAY_WORLD_SIZE` / `RAY_NPROC_PER_NODE` | `8` / `8` | Swift + Ray | 集群 GPU 总数 / 每节点 GPU 数 |
| `TP` / `PP` / `CP` / `EP` / `ETP` | `2`/`1`/`1`/`1`/`1` | Megatron | 张量/流水线/上下文/专家/专家张量并行度 |
| `SEQUENCE_PARALLEL` | `true`（含 codec 任务为 `false`） | Megatron | Megatron 序列并行 |
| `QWEN_OMNISPEAK_MCORE_CODEC_LOSS` | `true` | Megatron | 关闭则仅训练 thinker 文本 |
| `MCORE_HIDDEN_LAYER` | （取自 `accept_hidden_layer`） | Megatron | 覆盖 talker 捕获的 thinker 层 |
| `RESIDUAL_LOSS` / `RESIDUAL_LOSS_WEIGHT` | config 默认 | 涉及语音生成 | 残差 code-predictor 损失开关/权重 |

## 兼容性与注意事项

- **`--remove_unused_columns false`** 是必需的，让 `target_audio`、
  `speaker_spans`、`task` 等字段能进入模板编码阶段；模板 collator 会
  在 forward 之前丢弃原始 JSONL 字段。
- **ZeRO-3 与音频多模态分支** 在某些 transformers 版本下会因 rank 间
  音频流不一致挂起，默认保持 `zero2`，需要 ZeRO-3 时请先在小规模上验证。
- **本地导出目录的 modeling 文件**
  （`models/custom_models/qwen_omnispeak_{a,b,c,d,e,f}/modeling_qwen_omnispeak.py`）
  与 `custom/qwen_omnispeak/modeling_qwen_omnispeak.py` 必须保持一致，
  因为 `MODEL_PATH` 加载本地模型时使用的是导出目录里的 trust_remote_code
  代码。任何 `forward`/`run_thinker` 改动都需要同步到对应导出目录。

## 许可

本项目代码遵循 ms-swift 仓库许可证。第三方依赖（transformers / ray /
qwen-tts 等）以其各自的许可证为准。
