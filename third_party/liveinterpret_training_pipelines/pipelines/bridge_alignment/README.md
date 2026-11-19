# Qwen-OmniSpeak Bridge Alignment Training

本目录提供三族 Thinker（Qwen3-Omni / Qwen2.5-Omni-7B / Qwen2.5-Omni-3B）× 变体 A-J 的 bridge 对齐训练入口。训练时冻结 thinker，放开 `model.bridge`、残差 code predictor（`model.talker.code_predictor`）以及 talker 最下面 4 层（`model.talker.model.layers.0~3`）一起微调；目标语音使用 `Qwen3/Qwen3-TTS-Tokenizer-12Hz` 编码为离散 codec token（每帧 16 级码本：1 级主码 + 15 级残差码，12.5Hz）。

## 三族 Thinker 与对应模型目录

| MODEL_TYPE 前缀 | 对应 Thinker | hidden state 策略 | Megatron 建议 EP |
|----------------|-------------|-------------------|-----------------|
| `qwen3_omnispeak_{a..j}` | Qwen3-Omni-30B-A3B（MoE） | 第 24 层 hidden | `EP>1`（如 `EP=8`） |
| `qwen2_5_7b_omnispeak_{a..j}` | Qwen2.5-Omni-7B（dense） | 最后一层 hidden | `EP=1`（默认） |
| `qwen2_5_3b_omnispeak_{a..j}` | Qwen2.5-Omni-3B（dense） | 最后一层 hidden | `EP=1`（默认） |

导出目录位于 `models/custom_models/` 下，命名与 MODEL_TYPE 一致，例如 `qwen2_5_7b_omnispeak_b`。

监督信号包含两部分：

- **主码（码本第 0 级）**：经 thinker→bridge→talker→`codec_head` 自回归预测，CE 监督。
- **残差码（码本第 1~15 级，共 15 组）**：由 `code_predictor` 在**码本组维度上自回归**预测（teacher forcing 喂入真值低阶码 + roll-out），与上游 Qwen3-TTS 训练方式一致。

### 控制 token 前缀（贴近 talker 预训练分布）

codec 序列前会插入一段 talker 预训练时熟悉的"模式开关"控制 token（不参与监督，标签为 `-100`），让 frozen/半放开的 talker 进入正确的条件状态：

- 识别到目标语言（`target_lang`）：`[codec_think, codec_think_bos, language_id, codec_think_eos, codec_pad, codec_bos]`
- 未识别语言（auto）：`[codec_nothink, codec_think_bos, codec_think_eos, codec_pad, codec_bos]`

其中 `language_id` 由 `target_lang`（支持 `en`/`english`/`zh`/`chinese` 等别名）映射到 talker 的 `codec_language_id`；base talker 的 `spk_id` 为空，故不加 speaker 位。控制 token id 与 `codec_language_id` 映射在模型 config 中定义（默认取自 Qwen3-TTS-12Hz-1.7B-Base）。

> 同传演进预留：`build_block_causal_mask` 提供可选的 `codec_to_sem_visible` 参数（默认 `None`＝当前"codec 看全部语义"的离线行为）。后续做"一边听一边说"时，传入单调的逐帧可见语义长度即可切换为 chunk-causal（read-write）流式 mask，无需改动其它逻辑。

## 数据格式

默认读取 `data_format.jsonl`，也可以通过 `BRIDGE_ALIGNMENT_DATA=/path/to/train.jsonl` 指定。每行至少包含：

- `task`: `tts`、`speech_to_speech_conversation`、`speech_to_speech_translation`
  或 `text_to_speech_translation`
- `messages`: ms-swift 标准消息；语音输入使用 `<audio>` 并配合 `audios`
- `target_audio`: 目标语音路径
- `target_text`: 目标语音文本转写，默认会追加为 assistant 文本，供 thinker 提供语义上下文
- `task_prompt`: 可选任务指令，会合并到 system message
- `chat_template_kwargs`: 可选，例如 `{"sampling_rate": 24000}`

如需不把 `target_text` 追加到输入消息，设置：

```bash
export BRIDGE_ALIGNMENT_APPEND_TARGET_TEXT=false
```

## 单任务训练

TEMPLATE 固定为 `qwen_omnispeak`（bridge alignment 阶段）；三族 Thinker 均使用相同的 template 名称，模板内部会根据 `config.thinker_family` 自动切换 Qwen3 / Qwen2.5 的多模态编码路径。

默认模型为 `models/custom_models/qwen_omnispeak_b`，切换族和变体只需覆盖 `MODEL_PATH` 和 `MODEL_TYPE`：

```bash
# Qwen2.5-Omni-7B 变体 G (Q-Former Bridge)
MODEL_TYPE=qwen2_5_7b_omnispeak_g \
MODEL_PATH=/path/to/project/models/qwen2_5_7b_omnispeak_g \
BRIDGE_ALIGNMENT_DATA=/data/bridge/train.jsonl \
bash pipelines/bridge_alignment/train_tts.sh

# Qwen2.5-Omni-3B 变体 B
MODEL_TYPE=qwen2_5_3b_omnispeak_b \
MODEL_PATH=/path/to/project/models/qwen2_5_3b_omnispeak_b \
bash pipelines/bridge_alignment/train_joint.sh

# Qwen3-Omni 变体 A（默认）
MODEL_PATH=/path/to/project/models/qwen3_omnispeak_a \
MODEL_TYPE=qwen3_omnispeak_a \
bash pipelines/bridge_alignment/train_joint.sh
```

## 三任务联合训练

```bash
BRIDGE_ALIGNMENT_DATA=/data/bridge/train.jsonl \
bash pipelines/bridge_alignment/train_joint.sh
```

后台训练：

```
LOG="logs/train_joint_$(date +%Y%m%d_%H%M%S).log" && nohup bash train_joint.sh > "$LOG" 2>&1 & echo "PID=$!" && echo "LOG=$LOG"
```

## 多机多卡：三种并行方案

每个任务（`tts` / `speech_to_speech_conversation` / `speech_to_speech_translation` / `joint`）都通过 `TRAIN_BACKEND` 环境变量切换后端：
新增的 `text_to_speech_translation` 任务当前先提供 DeepSpeed Stage-1 入口：

| 方案 | 并行策略 | 适用场景 | 环境变量 |
| --- | --- | --- | --- |
| swift sft + DeepSpeed ZeRO2/3 | DP + ZeRO | 单机多卡 / 多机多卡 | `TRAIN_BACKEND=deepspeed`（默认） |
| swift sft + Ray + DeepSpeed | DP + ZeRO | 多机多卡（Ray 调度） | `TRAIN_BACKEND=deepspeed` + `USE_RAY=true` |
| megatron sft | TP/PP/CP/EP + ZeRO-1 | 大规模高性能训练 | `TRAIN_BACKEND=megatron` |

任务 ↔ 脚本对照（统一入口，不再有独立 `_megatron.sh`）：

| 任务 | 统一脚本 |
| --- | --- |
| tts | `train_tts.sh` |
| s2s 对话 | `train_speech_to_speech_conversation.sh` |
| s2s 翻译 | `train_speech_to_speech_translation.sh` |
| 三任务联合 | `train_joint.sh` |
| text→speech 翻译 | `train_text_to_speech_translation.sh`（仅 DeepSpeed） |

### 方案 1：swift sft + DeepSpeed ZeRO2/3（DP + ZeRO）

复用 ms-swift / torchrun 分布式启动环境变量。

单机多卡：

```bash
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BRIDGE_ALIGNMENT_DATA=/data/bridge/train.jsonl \
bash pipelines/bridge_alignment/train_joint.sh
```

多机多卡（每个节点各跑一次，按 `NODE_RANK` 区分）：

```bash
# 主节点
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NPROC_PER_NODE=8 \
bash pipelines/bridge_alignment/train_joint.sh

# 从节点
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NPROC_PER_NODE=8 \
bash pipelines/bridge_alignment/train_joint.sh
```

默认 `DEEPSPEED=zero2`。Qwen Omni 音频链路在 ZeRO-3 下可能出现 rank 间音频分支不一致导致挂起，除非确认环境已修复，否则建议保持 `zero2`。

### 方案 2：swift sft + Ray + DeepSpeed（DP + ZeRO，多机）

并行策略与方案 1 相同（DP + DeepSpeed ZeRO），区别仅在于用 Ray 集群统一调度，无需在每个节点手动设置 `NODE_RANK`。

先拉起 Ray 集群：

```bash
# 主节点
ray start --head --port=6379
# 每个 worker 节点
ray start --address='<head-ip>:6379'
```

然后在主节点运行一次即可，Ray 会把 worker 分发到整个集群：

```bash
RAY_WORLD_SIZE=16 \      # 集群内训练 GPU 总数
RAY_NPROC_PER_NODE=8 \   # 每个物理节点的 GPU 数（放置粒度）
BRIDGE_ALIGNMENT_DATA=/data/bridge/train.jsonl \
bash pipelines/bridge_alignment/train_joint_ray.sh
```

底层通过 ms-swift 原生 `--use_ray / --ray_exp_name / --device_groups` 实现，脚本会按
`RAY_WORLD_SIZE` / `RAY_NPROC_PER_NODE` 自动生成 `--device_groups` JSON；也可直接用
`RAY_DEVICE_GROUPS='{...}'` 覆盖。参考
<https://swift.readthedocs.io/zh-cn/latest/Instruction/Ray.html>。

### 方案 3：megatron sft（TP/EP/DP + ZeRO-1，大规模）

使用 `custom/qwen_omnispeak/mcore_adaptation.py`
的非侵入式 mcore-bridge 注册，把 thinker 映射成 Megatron `GPTModel`（按 TP/EP 切分），
audio/visual/bridge/**talker** 作为复制子模块加载；ZeRO-1 由 Megatron
`DistributedOptimizer` 提供。

talker codec loss（主码 CE + 残差 code-predictor CE）已原生接入 Megatron 训练：
`mcore_talker_loss.py` 在 thinker 第 `accept_hidden_layer` 层捕获隐藏态，驱动
bridge→talker→codec heads，计算与 DeepSpeed 路径**完全一致**的 codec loss
（已由 `test_mcore_talker_loss.py` 在 CPU 上验证 6 个变体 `|Δ|=0`）。bridge 对齐
默认冻结 thinker（`FREEZE_LLM=true`），训练 bridge + talker。

> **不同 Thinker 族的 EP 设置**：
> - Qwen3-Omni-30B（MoE thinker）→ 建议 `EP>1`（如 `EP=8`），启用专家并行以降低显存。
> - Qwen2.5-Omni-7B / 3B（dense thinker）→ 保持默认 `EP=1`，无需 MoE 参数。

```bash
# Qwen2.5-Omni-7B 变体 B，dense thinker，单机 8 卡（EP=1 默认）
TRAIN_BACKEND=megatron \
MODEL_TYPE=qwen2_5_7b_omnispeak_b \
MODEL_PATH=models/custom_models/qwen2_5_7b_omnispeak_b \
TP=2 PP=1 CP=1 \
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash pipelines/bridge_alignment/train_joint.sh

# Qwen3-Omni 变体 G (Q-Former Bridge)，MoE thinker，单机 8 卡（EP=2 建议开启）
TRAIN_BACKEND=megatron \
MODEL_TYPE=qwen3_omnispeak_g \
MODEL_PATH=models/custom_models/qwen3_omnispeak_g \
TP=2 PP=1 CP=1 EP=2 \
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash pipelines/bridge_alignment/train_joint.sh

# 多机，每节点各跑一次
TRAIN_BACKEND=megatron \
TP=4 EP=4 \
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NPROC_PER_NODE=8 \
bash pipelines/bridge_alignment/train_joint.sh
```

可调并行/超参环境变量：`TP` / `EP` / `ETP`、`MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE`、
`LR` / `MIN_LR` / `LR_WARMUP_FRACTION`、`FREEZE_LLM` / `FREEZE_VIT` / `FREEZE_ALIGNER`、
`RESIDUAL_LOSS` / `RESIDUAL_LOSS_WEIGHT`、`RECOMPUTE_*`、`ATTENTION_BACKEND`。

> **codec loss 并行约束**：talker codec loss 需要 `PP=1`、`CP=1`、
> `sequence_parallel=false`、且不开 packing/padding-free（脚本已默认设好）。
> 此时 thinker 仍享有完整 TP/EP/DP/ZeRO-1，talker 为复制式训练。不满足时会
> **明确报错**（不会静默算错）。若只想训练 thinker（文本）而跳过 codec loss，
> 设 `QWEN_OMNISPEAK_MCORE_CODEC_LOSS=false`。捕获层可用 `MCORE_HIDDEN_LAYER` 覆盖。

## 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `models/custom_models/qwen_omnispeak_b` | 导出变体目录，三族 A-J 均可填入 |
| `MODEL_TYPE` | `qwen_omnispeak_b` | 注册名，格式：`{qwen3\|qwen2_5_7b\|qwen2_5_3b}_omnispeak_{a..j}` |
| `TEMPLATE` | `qwen_omnispeak` | **bridge alignment 固定此值**，三族通用，无需修改 |
| `BRIDGE_ALIGNMENT_DATA` | `data_format.jsonl` | 训练 JSONL |
| `QWEN3_TTS_TOKENIZER` | `Qwen3/Qwen3-TTS-Tokenizer-12Hz` | 目标语音 tokenizer |
| `QWEN3_TTS_TOKENIZER_DEVICE` | `cuda:${LOCAL_RANK}` 或 `cpu` | tokenizer 编码设备 |
| `BRIDGE_ALIGNMENT_CODEC_CACHE` | `.cache/bridge_alignment_codec` | 目标语音编码缓存目录 |
| `OUTPUT_ROOT` | `output/qwen_omnispeak_output` | 输出根目录 |
| `DEEPSPEED` | `zero2` | ms-swift DeepSpeed 配置 |
| `LEARNING_RATE` | `1e-4` | bridge 学习率 |
| `MAX_LENGTH` | `4096` | thinker 输入最大长度 |
| `USE_RAY` | `false` | 设为 `true` 用 Ray 启动（`--use_ray`） |
| `RAY_WORLD_SIZE` / `RAY_NPROC_PER_NODE` | `8` / `8` | （`*_ray.sh`）集群训练 GPU 总数 / 每节点 GPU 数 |
| `RAY_DEVICE_GROUPS` | （自动） | 透传给 `--device_groups` 的 JSON；`*_ray.sh` 自动生成 |
| `TP` / `PP` / `CP` / `EP` / `ETP` | `2`/`1`/`1`/`1`/`1` | （`TRAIN_BACKEND=megatron`）并行度；dense thinker 保持 `EP=1`，MoE thinker 建议 `EP>1` |
| `SEQUENCE_PARALLEL` | `false` | （`TRAIN_BACKEND=megatron`）bridge alignment 固定 `false`（codec loss 约束） |
| `MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` | `1` / `16` | （`TRAIN_BACKEND=megatron`）Megatron batch 大小 |
| `LR` / `MIN_LR` / `LR_WARMUP_FRACTION` | `1e-4` / `1e-5` / `0.03` | （`TRAIN_BACKEND=megatron`）Megatron 学习率调度 |
| `FREEZE_LLM` / `FREEZE_VIT` / `FREEZE_ALIGNER` | `true`/`true`/`false` | （`TRAIN_BACKEND=megatron`）thinker / 音视频编码器 / bridge 冻结开关 |

也可以在脚本后追加任意 `swift sft` 参数，例如：

```bash
bash pipelines/bridge_alignment/train_joint.sh \
  --num_train_epochs 2 \
  --save_steps 200
```

## 冻结策略

`common_args.sh` 使用：

```bash
--tuner_type full \
--freeze_parameters_ratio 1.0 \
--trainable_parameters model.bridge model.talker.code_predictor \
--trainable_parameters_regex 'model\.talker\.model\.layers\.[0-3]\.'
```

ms-swift 会先冻结全部参数，再按前缀/正则重新激活下列模块（不注入 LoRA，直接训练原始参数）：

- `model.bridge`：DualRoute Bridge，Stage 1 最核心的可训练模块；
- `model.talker.code_predictor`：残差 code predictor，学习 15 级残差码；
- `model.talker.model.layers.0~3`：talker **最下面 4 层**，让 talker 适应
  `[语义块][codec块] + block-causal mask` 这种新的输入排布。

> 该正则与变体无关：a/b/c/d/e/f/g/h/i/j 导出后 talker 层均重新编号为 `0..N-1`，
> `layers.0~3` 在十个变体里都对应各自切片中最靠近输入的 4 层，因此十个变体通用。
> 如显存吃紧，可把正则收窄到 `[0-2]`（底 3 层）或 `[0-1]`。

注意：放开 talker 底层 + code_predictor 后可训练参数量明显大于"仅训 bridge"，
注意显存；默认 `DEEPSPEED=zero2`。

## 残差损失开关 / 权重

残差（code predictor）损失可开关并配置权重，总损失为 `主码CE + weight × 残差CE`。优先级：环境变量 > 模型 config（默认 `enabled=true`、`weight=1.0`）。

通过训练脚本变量（推荐，无需改 config / 重新导出）：

```bash
# 只训主码、关闭残差（最快验证 loss 是否继续下探）
RESIDUAL_LOSS=false bash pipelines/bridge_alignment/train_joint.sh

# 保留残差但降权
RESIDUAL_LOSS_WEIGHT=0.5 bash pipelines/bridge_alignment/train_joint.sh
```

底层等价于设置环境变量 `QWEN_OMNISPEAK_RESIDUAL_LOSS` / `QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT`，或在模型 config 中设置 `residual_loss_enabled` / `residual_loss_weight`。关闭残差时会跳过 code predictor 的前向，省显存与算力。

## 插件入口

训练脚本只加载一个插件：

```bash
--external_plugins custom/plugin.py
```

该插件会注册：

- `qwen_omnispeak_{a..j}`、`qwen3_omnispeak_{a..j}`、`qwen2_5_7b_omnispeak_{a..j}`、`qwen2_5_3b_omnispeak_{a..j}` 共 40 个模型类型
- `bridge_alignment_tts`
- `bridge_alignment_speech_to_speech_conversation`
- `bridge_alignment_speech_to_speech_translation`
- `bridge_alignment_text_to_speech_translation`
- `bridge_alignment_all`
- `qwen_omnispeak` bridge alignment 模板覆盖（三族通用）

## 依赖

除 ms-swift 现有依赖外，需要安装 Qwen3-TTS tokenizer 包并确保可以导入：

```bash
python -c "from qwen_tts import Qwen3TTSTokenizer; print('ok')"
```

如需预下载 tokenizer：

```bash
huggingface-cli download Qwen3/Qwen3-TTS-Tokenizer-12Hz
```
