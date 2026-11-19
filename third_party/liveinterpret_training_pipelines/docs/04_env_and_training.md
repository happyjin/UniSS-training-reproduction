# 04 · 环境变量、入口脚本用法与训练超参

主入口：`pipelines/bridge_alignment/train_speech_to_speech_translation_upstream.sh`

---

## 1. 入口脚本设计

- **默认方法 `TALKER_METHOD=reply_add`**（本分支主打）。
- 与 public 原版 `train_speech_to_speech_translation.sh`（sem_block 拼接路线）**并列**，
  两者都 `source common_args.sh`，但本脚本**不调用** `run_bridge_alignment()` 封装函数，
  而是自己直接拼 `swift sft`（原因见下节）。
- 默认 `MODEL_TYPE=qwen2_5_3b_omnispeak_a`（variant A，`use_upstream_talker_input=True`）。

### 1.1 为什么 source common_args.sh 但不用它的封装函数

`common_args.sh` 有两部分：

1. **通用基建**（前半段）：环境默认值、GPU 数量探测（`NPROC_PER_NODE`）、
   attn 探测（FA3→FA2→sdpa 级联回退）、deepspeed 守护、wandb、分布式启动变量
   （`NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT`）。
2. **bridge 专用封装** `run_bridge_alignment()`（后半段）：把
   `--trainable_parameters model.bridge` + talker 前 2 层 regex 等**写死**后执行 `swift sft`。

原版脚本两部分都用；本脚本**只用通用基建**，因为 reply/full-add 要训的是
`model.thinker_adapter`（+ 可选层），参数组合与 bridge 完全不同，还要注入
`TALKER_METHOD`/ICL 等专属开关。硬套封装函数会污染原版 bridge 逻辑，故约定
"无 wrapper，直接调 `swift sft`"，两条路线互不干扰。

---

## 2. 环境变量总表（在哪消费 / 默认 / 作用）

### 2.1 方法与布局

| 环境变量 | 默认 | 消费位置 | 作用 |
|---|---|---|---|
| `TALKER_METHOD` | `reply_add` | 入口脚本 | `reply_add` / `full_add`；据此 export `QWEN_OMNISPEAK_FULL_ADD_LAYOUT`。 |
| `QWEN_OMNISPEAK_FULL_ADD_LAYOUT` | `false` | `modeling`、`template` | text 通道布局：reply-add(false) / full-add(true)。 |
| `QWEN_OMNISPEAK_USE_ICL` | `false` | `modeling`、`template` | 开启 ICL 参考前缀（默认走 x-vector）。 |
| `QWEN_OMNISPEAK_USE_SPEAKER_PROMPT` | `true`（入口脚本设） | `modeling`、`template` | 是否注入 x-vector speaker prompt。优先用清单里的 `source_speaker_embedding`（离线 ECAPA）；无则从 source wav 在线提取。Mel + SPEAKER 且无该列会在 launch / encode 报错。 |
| `LANCE_USE_SPEAKER_EMBEDDING` | 未设（schema 探测） | `lance_manifest` / Shape B loader | 是否写 `lance://…/speaker` 指针；与 mel/raw 正交。 |
| `LANCE_PREFETCH_SPEAKER` | `1` | `dataset` map | map 阶段批预取 speaker x-vector（2048 floats，默认开）。 |
| `LANCE_TRAIN_SHAPE` | `A` | Lance launcher | `A` 读取既有 pointer JSONL；`B` 从 Lance 原子生成本地 pointer JSONL（兼容当前 ms-swift 不可靠加载 HF `.py` script 的限制）。 |
| `LANCE_USE_PRECOMPUTED_MEL` | 未设（schema 探测） | manifest 生成 | 是否生成 `audios=[] + source_input_features=lance://…/features` 的自描述 mel 行。 |
| `LANCE_PREFETCH_FEATURES` | `0` | dataset map | 是否在 map 阶段批量展开 mel；默认保持 URI lazy，在 template encode 时读取，避免放大 HF cache。 |
| `QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT` | 未设（用 config） | `configuration` | 运行时覆盖 config 总开关，便于多布局共享同一 base ckpt。 |

Megatron 与 DeepSpeed 共用上述 Lance / codec / mel 数据契约。Megatron 的
`mcore_adaptation` 会在 audio token 存在但 `input_features` 缺失或数量不一致时
直接报错，防止离线 mel 静默退化为零音频。音色条件由共享助手
`apply_speaker_prompt_to_codec_emb` 在 `embed_codec` 后注入 speaker slot；
`mcore_talker_loss` 会 pop 并消费 `speaker_embedding` /
`reference_audio_wav` / `speaker_slot_index`，与 DeepSpeed forward 对齐。
优先使用离线 `source_speaker_embedding`；否则回落到在线 ECAPA。

### 2.2 Loss

| 环境变量 | 默认 | 消费位置 | 作用 |
|---|---|---|---|
| `RESIDUAL_LOSS` / `QWEN_OMNISPEAK_RESIDUAL_LOSS` | `true` | `modeling` | 残差(code-predictor) loss 开关。 |
| `RESIDUAL_LOSS_WEIGHT` / `QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT` | `0.3` | `modeling` | 残差 CE 权重。 |
| `QWEN_OMNISPEAK_TEXT_LOSS` / `..._WEIGHT` | `true` / `1.0` | `modeling` | thinker 文本 CE 开关与权重（thinker 冻结时仅监控）。 |
| `CODEC_EOS_LOSS_WEIGHT` / `QWEN_OMNISPEAK_CODEC_EOS_LOSS_WEIGHT` | `1.0` | 入口脚本 / `modeling` | main-codec EOS 标签的 CE 权重；新版配方设为 `10`。 |
| `CODEC_HISTORY_CORRUPTION_PROB` / `QWEN_OMNISPEAK_CODEC_HISTORY_CORRUPTION_PROB` | `0.0` | 入口脚本 / `modeling` | 仅训练时扰动 teacher-forced codec 历史的概率；新版配方设为 `0.05`，eval/inference 不扰动。 |

### 2.3 可训练范围（入口脚本层）

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `TALKER_TRAINING_SCOPE` | `layers` | `layers`（按层解冻）/ `whole`（整个 talker）。 |
| `UNFREEZE_TALKER_LAYERS` | `0` | `0/2/8/28`，解冻 talker 前若干层。 |
| `UNFREEZE_MTP` | `0` | 解冻残差 code predictor。 |
| `UNFREEZE_CODEC_HEAD` | `0` | 解冻主 codec 头。 |
| `UNFREEZE_THINKER` | `0` | 解冻 thinker（阶段 2）。 |

### 2.4 数据与 DataLoader 配置（透传给 `swift sft`）

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `BRIDGE_ALIGNMENT_DATA` | 自动探测 | 训练数据路径，支持 glob（如 `.../converted/*.parquet`）。 |
| `GROUP_BY_LENGTH` | `false` | 按序列长度分桶采样，减少 padding。 |
| `DATALOADER_NUM_WORKERS` | `0`（脚本默认；实跑可设 16） | DataLoader 工作进程数。 |
| `DATALOADER_PERSISTENT_WORKERS` | `false` | 仅 num_workers>0 时生效。 |
| `WDS_AUDIO_MODE` | `auto` | `off` / `auto` / `required`；新版纯 WDS 配方使用 `required`。 |
| `BRIDGE_ALIGNMENT_NORMALIZE_S2S_PROMPT` | `true` | 按 `target_lang` 生成标准 S2S system prompt，覆盖旧 manifest prompt。 |

---

## 3. 用法示例

```bash
# reply_add（默认），adapter-only，冻结 thinker
bash pipelines/bridge_alignment/train_speech_to_speech_translation_upstream.sh

# full_add + 解冻 talker 前 8 层
TALKER_METHOD=full_add UNFREEZE_TALKER_LAYERS=8 \
  bash pipelines/bridge_alignment/train_speech_to_speech_translation_upstream.sh

# 指定数据(glob 多分片) + 大 batch 多卡
BRIDGE_ALIGNMENT_DATA="/path/converted/*.parquet" \
NPROC_PER_NODE=8 PER_DEVICE_TRAIN_BATCH_SIZE=96 LEARNING_RATE=1e-5 \
DATALOADER_NUM_WORKERS=16 DATALOADER_PERSISTENT_WORKERS=true \
  bash pipelines/bridge_alignment/train_speech_to_speech_translation_upstream.sh

# indexed-WDS + EOS/history 配方（必填路径均由调用方提供）
MODEL_PATH=/path/to/checkpoint \
WDS_MANIFEST_DIR=/path/to/manifests \
PYTHON_BIN=/path/to/training-env/python \
  bash pipelines/bridge_alignment/run_reply_add_wds_eos_history.sh
```

多机多卡：每节点设 `NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT`
（`common_args.sh` 以 `:=` 尊重传入值；`USE_RAY=true` 时须保持这些 unset）。

---

## 4. 默认超参（入口脚本内 `: "${VAR:=...}"`，均可环境变量覆盖）

| 超参 | 默认值 | 传递到 |
|---|---|---|
| `TALKER_METHOD` | `reply_add` | talker 布局 |
| `MODEL_TYPE` | `qwen2_5_3b_omnispeak_a` | variant A |
| `NUM_TRAIN_EPOCHS` | `5` | `--num_train_epochs` |
| `LEARNING_RATE` | `1e-4` | `--learning_rate` |
| `WARMUP_RATIO` | `0.03` | `--warmup_ratio` |
| `MAX_LENGTH` | `4096` | `--max_length` |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `16` | `--per_device_train_batch_size` |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | `--gradient_accumulation_steps` |
| `RESIDUAL_LOSS_WEIGHT` | `0.3` | 残差 loss 权重 |
| `CODEC_EOS_LOSS_WEIGHT` | `1.0` | EOS loss 权重（新版配方覆盖为 `10`） |
| `CODEC_HISTORY_CORRUPTION_PROB` | `0.0` | 历史 token 扰动率（新版配方覆盖为 `0.05`） |
| `GROUP_BY_LENGTH` | `false` | 长度分桶 |
| `UNFREEZE_TALKER_LAYERS` | `0` | 仅训 adapter |
| `UNFREEZE_THINKER` | `0` | 冻结 thinker |

---

## 5. 实际运行配置与吞吐（batch4.5，reply-add）

在 **8× H200（143GB/卡）** 上跑 common_voice S2S（967,290 样本）的实测：

| 项 | 值 |
|---|---|
| 方法 / 布局 | reply_add，adapter-only，冻结 thinker |
| Batch size | 96/GPU × 8 GPU = **有效 768** |
| Learning rate / Epochs | `1e-5` / 5 |
| 总步数 | 6235 steps |
| DataLoader workers | 16（persistent） |
| **稳态迭代速度** | **约 3.9 – 4.4 s/it** |
| 显存峰值 | ~96.5 GB / 143 GB |

> 迭代速度随 batch 内序列长度波动（未开 `group_by_length`）。
> 训练日志中偶发 `[WARNING] Max/Min value of input waveform signal is ±1.0x`（x-vector mel
> 范围检查）、`sox not found`、`onnxruntime .../sys/class/drm` 等均为无害告警，不影响训练。

常规 train/eval 会记录 `token_text_acc`、`token_acc`、
`token_acoustic_token`、`eos_boundary_top1_acc` 和
`premature_eos_top1_rate`。public 配方不内置逐帧 AR rollout；AR 音频质量与停止行为应在
训练 checkpoint 产出后由独立 inference/evaluation 工具评估。
