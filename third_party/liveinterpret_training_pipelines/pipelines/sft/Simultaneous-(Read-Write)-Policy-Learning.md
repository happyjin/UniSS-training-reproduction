# 同声传译（Read-Write Policy）学习方案说明

> Stage-3 SFT · `simultaneous_policy_learning`
> 模型：Qwen-OmniSpeak（Thinker + Dual-Route Bridge + Qwen3-TTS Talker + Code2Wav）
> 模板：`qwen_omnispeak_simultaneous_policy`
> 入口：`pipelines/sft/train_simultaneous_policy_learning.sh`

本文档完整描述**流式语音到语音翻译（同声传译）**任务的训练方案，涵盖任务定义、
数据构建、输入/输出张量构造、读写（Read-Write）策略的监督机制、损失函数、训练流程、
关键超参与开关，以及实现细节与自检结论。

---

## 1. 任务说明

### 1.1 目标

训练一个**全双工同声传译**模型：源语音持续到达的同时，模型在**每一个时刻**决定
是继续 **READ（听）** 还是 **WRITE（开口翻译）**，并以流式方式直接输出目标语言的
**语音（codec → 波形）**。

> **本任务不预测目标文本**：模型直接由源语音映射到目标 codec 流，**不生成/不监督
> 文本**。因此整条管线**没有 Thinker 文本 CE（Loss_text）、没有文本 token 准确率
> （token_text_acc）**，监督是**纯 codec**（与 bridge alignment 一致）。Thinker 仍参与
> 前向（产出 hidden 供 bridge→talker 使用），但其文本输出不参与 loss。

与离线（offline）语音翻译的本质区别：

| | 离线 S2S 翻译 | 同声传译（本方案） |
| --- | --- | --- |
| 源语音可见性 | 全部可见 | 只能看到**已到达**的源前缀 |
| 输出时机 | 听完再译 | 边听边译，需学“何时开口” |
| 监督信号 | codec CE | codec CE + **流式注意力掩码** + **静默帧监督** |

### 1.2 核心思想：用“掩码 + 静默”隐式监督读写策略

本方案**不引入任何新的 loss / 策略头 / 分类器**。读写策略通过两个机制隐式注入：

1. **WRITE（何时开口）** —— 在目标 codec 流的**真实时间轴**上，把“该听、还不该说”
   的时间段用**真实静默 codec 帧**填满，并施加 CE 监督。模型因此被显式教会：
   *该听时输出静默、该说时输出译文*。
2. **READ（能看到多少源）** —— 通过单调非减的 `codec_to_sem_visible` 流式注意力
   掩码，限制每一帧目标 codec 只能注意到“截至该帧真实时刻已到达”的源音频前缀。

两者在**同一条真实时钟**下统一：开口前的静默帧只能看到极少源 → 学“先听后说”；
随时间推进源逐帧解锁 → 学“边听边翻”。

---

## 2. 模型结构与数据流（前置背景）

```
                       ┌────────────────────────────────────────────┐
  <audio> + task_prompt│  Thinker (Qwen-Omni)                         │
  ───────────────────► │  → thinker_hidden_states / text_embeds       │  (文本输出不参与 loss)
                       └───────────────┬──────────────────────────────┘
                                       │
                            Dual-Route MLP Bridge
                                       │  sem_block  (sem_len == #thinker tokens)
                                       ▼
   control_prefix + main codec frames  ┌──────────────────────────────┐
   ───────────────────────────────────►│  Talker (Qwen3-TTS backbone)  │
                                        │  full_emb = [sem_block ; codec]│
   codec_to_sem_visible（流式掩码）────►│  build_block_causal_mask(...)  │
                                        └──────────────┬────────────────┘
                                                       │ codec_hidden
                                          ┌────────────┴─────────────┐
                                          ▼                          ▼
                                   codec_head (main)         code_predictor (residual)
                                          │                          │
                                     main CE loss              residual CE loss
                                          │
                                          ▼ (推理时)
                                   Code2Wav（冻结）──► 目标波形
```

要点：
- 在 dual-route MLP bridge 下 `sem_len == #thinker tokens`，因此**源音频占位 token
  可直接在 `input_ids` 中定位**（首索引 `a0`，数量 `M`）。这是构造读写掩码的关键。
- Talker 把 `[sem_block ; codec]` 拼成一条序列做注意力；`codec → sem` 这一注意力块
  就是“目标帧看源/Prompt”的通道，正是 `codec_to_sem_visible` 作用的地方。
- **本任务不预测文本**：Thinker 仅作为编码器为 bridge 提供 hidden，其文本 logits 不参与
  任何 loss（无 Loss_text / 无 token_text_acc）。
- Code2Wav（vocoder）在本阶段**冻结**，只训练 thinker + bridge + talker backbone。

---

## 3. 数据构建

### 3.1 数据 Schema（`ms-swift-speech-simul-v1`）

每条样本是一行 JSON，关键字段见 `simultaneous_policy_learning.json`（带注释版）与
`data_format.jsonl`（训练用）。核心结构：

```jsonc
{
  "schema_version": "ms-swift-speech-simul-v1",
  "task": "simultaneous_policy_learning",
  "task_prompt": "Streaming speech translation: ...",
  "messages": [
    {"role": "user", "content": "<audio>"}
    // 注意：本任务不预测文本，messages 中**不含** assistant 文本轮。
  ],
  "audios": ["<源语音 .wav（整段）>"],
  "target_text": "<目标语言全文译文（仅作元数据/参考，不参与训练）>",
  "target_audios": [           // 逐 chunk 的目标语音（已 TTS 切片）
    "chunk_0000.wav", "chunk_0001.wav", ...
  ],
  "source_lang": "en",
  "target_lang": "zh",
  "alignment": [               // 逐 chunk 的对齐 + 时间轴（核心监督来源）
    {
      "chunk_id": 0,
      "source_text": "...",
      "target_text": "...",
      "source_start_ms": 0,      // 源音频该 chunk 起点
      "source_end_ms":   2260,   // 源音频该 chunk 终点
      "target_start_ms": 2260,   // 目标音频该 chunk 起点（开口时刻）★
      "target_end_ms":   6772,   // 目标音频该 chunk 终点 ★
      "target_audio":    "chunk_0000.wav",
      "extra_lag_ms":    0,       // 该 chunk 译文相对源的额外延迟
      "open_delay_ms":   2260
    },
    ...
  ],
  "chat_template_kwargs": {"sampling_rate": 24000}
}
```

### 3.2 标注策略（数据侧约定，训练侧完全信任）

逐 chunk 的开口时刻由标注阶段按读写策略给定，典型保守策略：

```
tgt_start[i] = max(src_end[i], tgt_end[i-1])
```

即：第 `i` 段译文的开口时间不早于①该段源说完（`src_end[i]`，保证语义完整）、
②上一段译文说完（`tgt_end[i-1]`，避免 TTS 互相重叠）。

> **训练侧约定**：`target_start_ms / target_end_ms` 被**完全信任**，作为真实时间轴的
> 唯一依据；训练代码不会重算或改写时间戳（只在浮点取整造成重叠时做 `max(start, cursor)`
> 的防重叠保护）。正式训练数据必须保证时间戳自洽、单调。

### 3.3 数据集注册与预处理

- 数据集注册：`custom/dataset.py` → `register_simultaneous_policy_datasets`，
  注册名 `simultaneous_policy_learning` / `simultaneous_policy_all`。
- 预处理器：`SimultaneousPolicyPreprocessor`
  - 过滤 `task == "simultaneous_policy_learning"`；
  - 合并 `task_prompt` 到 user 轮；
  - **不补 assistant 文本轮**（本任务不预测文本，纯 codec 监督，无 thinker 文本 CE）；
  - 透传 `alignment` / `target_audios` 给模板；
  - `SIMULTANEOUS_POLICY_STRICT`：缺字段时是否抛错（默认 `false`，跳过坏样本）。

### 3.4 Codec 缓存

每个 chunk 音频经 Qwen3-TTS Tokenizer 编码为 codec codes 后落盘缓存（按文件指纹），
路径由 `SIMULTANEOUS_POLICY_CODEC_CACHE` / `CONTINUE_PRETRAIN_CODEC_CACHE` 决定，
两个 pipeline 可共享缓存。

---

## 4. 输入 / 输出张量构造

实现位置：`custom/template.py::QwenOmniSpeakSimultaneousPolicyTemplate` 与
`custom/codec_tokenizer.py::Qwen3TTSCodecTokenizer`。

### 4.1 文本（Thinker）侧

- 模板沿用 `<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n` 结构；
- `messages` 里**只有** user 轮含 `<audio>` 占位，**没有 assistant 文本轮**；
- 因此 Thinker 仅作为编码器为 bridge 提供 hidden（`run_thinker=True`），**不产生文本
  监督**：collator 不保留文本 LM labels，而是把 `labels` 设为 main codec 的右移序列
  （供 trainer 计算 **codec token 准确率**），与 bridge alignment 一致。

### 4.2 目标 Codec 流：真实时间轴铺设 + 静默填充

核心方法：`encode_target_audios_concat(...)`。**不做无缝拼接**（无缝拼接会把任务退化为
离线 TTS），而是把每个 chunk 的 codec 帧放到其 `target_start_ms` 对应的帧位置，
所有空白（开头听音窗 + chunk 间隙）用**真实静默 codec 帧**填满：

```
frame 0 ...... start[0] ...... start[0]+T0 ... start[1] ......
[ 静默(边听边等) ][  chunk_0 codec  ][ 静默 ][  chunk_1 codec  ] ...

start[i] = round(target_start_ms[i] / frame_ms)
```

- **帧率自校准** `frame_ms()`：直接从活的 tokenizer 读
  `encode_downsample_rate / input_sample_rate`，实测 `1920 / 24000 = 80.0ms`
  （即 12.5Hz），避免硬编码文档中互相矛盾的 12Hz / 13fps / 480×。
  失败时回退 `QWEN3_TTS_FRAME_MS` 或常量 `80.0`。
- **静默帧生成** `_steady_silence_band()`：对一段数字静默波形过一次 codec，裁掉首尾
  瞬态，缓存**稳态静默帧**；`_silence_frames(n)` 平铺到任意间隙长度。解码端可还原为
  真静音，比直接用 `codec_pad` 更忠实于声学。
- **防重叠**：`start_frame = max(round(target_start/frame_ms), cursor)`，绝不让某段在
  上一段说完前开口（应对取整误差），但不改写时间戳。
- **回退**：当时间戳缺失/数量不匹配时，回退为旧的时间连续拼接（安全网，不影响正式训练）。

### 4.3 Codec 训练张量布局（`_build_training_tensors`）

设 `k = len(control_prefix)`，`T = num_main_frames`（静默 + 语音）：

```
codec_input_ids : [ p0, p1, ..., p_{k-1}=BOS,  m0, m1, ..., m_{T-1} ]      len = k+T
codec_labels    : [ -100 × (k-1)            ,  m0, m1, ..., m_{T-1}, EOS]   len = k+T
```

- **control_prefix**（mode-switch 控制 token，永不监督，label=-100）：
  - 识别到目标语言：`[think, think_bos, language_id, think_eos, pad, bos]`
  - 否则：`[nothink, think_bos, think_eos, pad, bos]`
  - 末位必须是 `codec_bos`。
- BOS 位预测第一个 main code，main 帧 teacher-forcing；
- `codec_residual_labels`（`[k+T, num_residual_groups]`）逐帧与 `codec_labels` 对齐，
  control_prefix / EOS 行为 -100，越界/无效残差填 -100。

### 4.4 读写掩码 `codec_to_sem_visible`（READ 半边）

方法：`_build_streaming_schedule(...)`。在**真实时钟**上逐帧计算每帧目标 codec 可见的
源 token 数：

```
主帧 j 的真实时刻  t_ms(j) = j * frame_ms + extra_lag_ms
已到达源 token 数  arrived(j) = round( M * clamp(t_ms(j)/source_duration_ms, 0, 1) )
可见前缀长度       vis[j] = clamp(a0 + arrived(j),  max = min(a0+M, sem_len))
```

- `a0`：源音频占位 token 首索引（保证音频前的 task prompt 永远可见）；
- `M`：源音频占位 token 数；`source_duration_ms`：优先读源 wav 真实时长，回退取
  `max(source_end_ms)`；
- control_prefix 位（在首个主帧之前）共享首主帧的可见度；
- 最后对整条 `vis` 做 `cummax`，强制**单调非减**（接口硬性要求）。

> **设计取舍（接口决定）**：单前缀接口只能表达“可见多少个前导 semantic token”。由于源
> 音频 token 在序列中**先于** assistant 目标文本 token，无法同时“门控源音频”又“揭示
> 已提交的译文文本”。因此**流式训练时 assistant 译文文本 token 不进入可见前缀**，每帧
> codec 依赖“可见源前缀 + 自身因果 codec 历史”。这与 Stage-3 学“何时开口”的目标一致。

### 4.5 模型侧掩码构造（`build_block_causal_mask`）

`codec_to_sem_visible` 传入 `modeling_qwen_omnispeak.py::build_block_causal_mask`：

- `sem ↔ sem`：块内双向可见；
- `codec → codec`：因果（下三角）；
- `codec → sem`：`frame t` 可见 `sem[j]` 当且仅当 `j < vis[t]`（这就是读写门控）；
- `sem → codec`：禁止（防未来泄漏）；
- `codec_to_sem_visible=None` 时退化为**离线全可见**，与 Stage-1/2 完全一致。

### 4.6 Data Collator（批内对齐）

- codec 张量按 `codec_pad_id` / `-100` padding（`pad_1d` / `pad_2d_time`）；
- `codec_to_sem_visible` 在父 collator pad codec **之前**收集，再 pad 到 `codec_input_ids`
  的宽度；缺失调度的行回退为**全可见**（`1<<30`），绝不置 0（否则会把该行 codec 致盲）；
- **codec-only**：collator 移除 `has_codec_supervision`，并把 `labels` 覆盖为 main codec
  的右移序列。这样 `model.forward` 走纯 codec 分支（`text_labels → None`，无 thinker 文本
  CE / 无 `token_text_acc`），trainer 对 `main_codec_logits` 统计 **codec token 准确率**；
  Megatron 路径同理进入 `bridge_mode`，把无意义的 thinker 文本 loss 置零。

---

## 5. 损失函数

实现位置：`modeling_qwen_omnispeak.py` 的 forward。本任务为**纯 codec 监督**，
总损失为两项加权和（**无文本 CE**）：

### 5.1 Main Codec CE（核心）

```
L_codec_main = CrossEntropy( main_codec_logits.reshape(-1, V),
                             codec_labels.reshape(-1), ignore_index=-100 )
```

- 监督目标 codec 主码序列（**含静默帧**）。静默帧的 CE 监督即“何时该闭嘴/何时开口”的
  WRITE 学习信号；
- 在 `codec_to_sem_visible` 流式掩码下，每帧只看到已到达的源 → READ 半边随之学到。

### 5.2 Residual Codec CE（声学细节，可选加权）

```
L_codec_res = CrossEntropy( residual_codec_logits.reshape(-1, V_res),
                            codec_residual_labels.reshape(-1), ignore_index=-100 )
```

- 由 code_predictor 沿 codebook-group 维 teacher-forced 自回归预测；
- 仅当残差 loss 开启且残差 labels 存在时计算。

### 5.3 总损失

```
L = L_codec_main + residual_loss_weight · L_codec_res
```

（**不含 Thinker 文本 CE**：本任务不预测文本。各项仅在其 labels 非全 -100 时计入；
全 -100 时给出带梯度的 0 loss 保证 trainer 步可定义。）

监控指标：`token_acc`（main codec）、`token_acoustic_token`（residual）。
**不再有 `token_text_acc`**（文本预测已移除）。

---

## 6. 训练流程方案

### 6.1 训练阶段定位

本任务是 **Stage-3 SFT**，建议在已完成 bridge alignment（Stage-1）与 continue-pretrain
（Stage-2）的检查点上继续训练。可训练范围：

- **可训练**：thinker + dual-route bridge + talker backbone（含 codec_head / code_predictor）；
- **冻结**：Code2Wav / vocoder（由 `FREEZE_PARAMETERS` 正则匹配）。

### 6.2 端到端流程

```
JSONL 样本
  └─ SimultaneousPolicyPreprocessor      # 过滤/合并 prompt/透传 alignment（不补 assistant 文本）
       └─ Template._encode               # 音频编码 + codec 真实时间轴铺设 + 静默填充 + 读写掩码
            └─ Data Collator             # codec/调度 padding 对齐 + codec-only labels
                 └─ model.forward        # main/residual codec CE（流式掩码下，无文本 CE）
                      └─ Trainer (swift sft, DeepSpeed zero2/3)
```

### 6.3 启动方式

```bash
# 单机 / 多卡（DeepSpeed zero2 自动探测可见 GPU）
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_b
export MODEL_TYPE=qwen3_omnispeak_b
export SIMULTANEOUS_POLICY_DATA=/abs/path/to/your.jsonl
export QWEN3_TTS_TOKENIZER=/path/to/project/models/Qwen3-TTS-Tokenizer-12Hz
bash pipelines/sft/train_simultaneous_policy_learning.sh

# 多机（Ray，先起集群）
USE_RAY=true bash pipelines/sft/train_simultaneous_policy_learning.sh
```

### 6.4 关键训练超参（默认值，可 env 覆盖）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `TUNER_TYPE` | `full` | `full` 或 `lora` |
| `DEEPSPEED` | `zero2` | zero2 / zero3 |
| `TORCH_DTYPE` | `bfloat16` | |
| `ATTN_IMPL` | `flash_attention_3` | 不可用自动按 `flash_attention_3 → flash_attention_2 → sdpa` 回退 |
| `LEARNING_RATE` | `1e-5` | |
| `NUM_TRAIN_EPOCHS` | `1` | |
| `MAX_LENGTH` | `4096` | |
| `GRADIENT_ACCUMULATION_STEPS` | `16` | |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | |
| `WARMUP_RATIO` | `0.03` | |
| `GRADIENT_CHECKPOINTING` | `true` | |

### 6.5 Megatron 训练（变体 3：`megatron sft`，TP/EP/DP + ZeRO-1）

大规模训练可直接走 Megatron-Core，与 DeepSpeed 路径**优化同一目标**，无需改任何监督逻辑。

**链路（非侵入式 mcore-bridge 适配）**：

```
thinker  ──► Megatron GPTModel（TP/EP 分片）          → 仅作编码器（文本 loss 被 bridge_mode 置零）
audio_tower / visual / bridge / talker
         ──► 作为 mcore "visual" 的复制子模块加载      → DP 同步 + 标准 visual 路径存取
forward hook 抓取 thinker 第 accept_hidden_layer 层隐状态
         ──► bridge → talker → codec_head / code_predictor
              （复用 modeling 的 build_block_causal_mask / 残差设置，保证数值对齐）
              → main codec CE + residual codec CE
```

> **codec-only**：本任务的 codec batch **不带 `has_codec_supervision`**，`mcore_talker_loss.py`
> 据此进入 `bridge_mode`，把无意义的 thinker 文本 CE 置零，只优化 codec loss——与 DeepSpeed
> 路径“无文本 loss”的行为一致。

**读写策略如何在 Megatron 下生效**：同传模板产出的 codec batch **额外携带
`codec_to_sem_visible`**。`mcore_talker_loss.py` 在 `forward_step` 中先把它从
`data` 里 pop 出（避免误传入 thinker 的 `model(**data)`），再传入
`compute_codec_loss → build_block_causal_mask(..., codec_to_sem_visible=...)`，
与 DeepSpeed forward **逐位一致**（`test_mcore_talker_loss.py::
test_streaming_parity_all_variants` 实测 `|Δ|=0`，且验证"流式 vs 离线"loss 确有差异）。
因此 **WRITE**（真实时间轴 codec CE，含静默帧）与 **READ**（流式注意力掩码）两半监督
在 Megatron 下与 DeepSpeed 完全等价。

**并行约束（`_check_parallel_guards` 快速失败）**：codec / 流式路径要求
`PP=1`、`CP=1`、`sequence_parallel=false`、不开 packing/padding-free；
**thinker 仍享受完整 TP/EP/DP/ZeRO-1**，仅复制的 talker codec 路径受限。

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_b
export MODEL_TYPE=qwen3_omnispeak_b
export SIMULTANEOUS_POLICY_DATA=/abs/path/to/your.jsonl
export QWEN3_TTS_TOKENIZER=/path/to/project/models/Qwen3-TTS-Tokenizer-12Hz
export TP=2 PP=1 CP=1 EP=1
export SEQUENCE_PARALLEL=false
bash pipelines/sft/train_simultaneous_policy_learning_megatron.sh
```

| Megatron 专属 Env | 默认 | 说明 |
| --- | --- | --- |
| `TP` / `PP` / `CP` / `EP` / `ETP` | `2/1/1/1/1` | thinker 模型并行；codec 路径要求 `PP=CP=1` |
| `SEQUENCE_PARALLEL` | `false` | codec + 流式掩码要求关闭 |
| `MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` | `1 / 16` | |
| `QWEN_OMNISPEAK_MCORE_CODEC_LOSS` | `true` | `false` → 关闭 codec loss（本任务即无监督，不建议） |
| `QWEN_OMNISPEAK_MCORE_HIDDEN_LAYER` | 配置 | 覆盖抓取隐状态的层（默认 `accept_hidden_layer`） |
| `RESIDUAL_LOSS` / `RESIDUAL_LOSS_WEIGHT` | 配置 | 残差 codec CE 开关与权重 |
| `SIMULTANEOUS_POLICY_STREAMING` | `true` | `false` → 离线全可见（消融） |

> 注：`megatron sft` 需要模型已转换为 mcore 权重格式（首次运行由 mcore-bridge 自动从
> HF/导出目录转换并缓存）。Code2Wav 在本阶段不参与 codec loss（只用 talker.model /
> codec_head / code_predictor），推理时再由 Code2Wav 还原波形。

---

## 7. 读写 / 时延开关（消融与时延-质量权衡）

| Env | 默认 | 作用 |
| --- | --- | --- |
| `SIMULTANEOUS_POLICY_STREAMING` | `true` | `false` → 离线全可见 codec 注意力（baseline 消融） |
| `SIMULTANEOUS_POLICY_EXTRA_LAG_MS` | `0` | 在数据驱动调度之上叠加的全局 wait-k 式时延；`0` = 纯数据驱动策略 |
| `SIMULTANEOUS_POLICY_SOURCE_DURATION_FROM_FILE` | `true` | `false` → 用 `max(source_end_ms)` 代替读 wav 取时长 |
| `SIMULTANEOUS_POLICY_STRICT` | `false` | 坏样本是否抛错 |
| `QWEN3_TTS_FRAME_MS` / `QWEN3_TTS_INPUT_SR` / `QWEN3_TTS_SILENCE_SECONDS` | 自校准 | 帧率/采样率/静默探针时长兜底 |

> 数据已通过 `alignment` 编码了读写策略，wait-k 是**可选**且默认关闭的，仅用于时延-质量消融。

---

## 8. 实现自检（与标注意图自洽性验证）

以真实 codec 模型实测一条多 chunk 样本（开头 2000ms 听音窗，frame_ms=80）：

- `frame_ms = 80.0` 自校准 ✓
- 时间轴铺设 `[静默25, 说16, 静默6, 说16, 静默5, 说16] = 84 帧`，开头静默
  `= round(2000/80) = 25` ✓
- `codec_input_ids` 长度 `= k+T = 5+84 = 89`，residual `(89, 15)` 对齐 ✓
- 开头静默帧 == 稳态静默 band ✓；语音帧与静默帧 id 不同 ✓
- 调度单调非减 ✓；control_prefix + 开头静默 `vis = a0`（源到达 0）✓
- chunk0 开口帧 `vis` 对应“已听到 ≈ 25 个源 token”，与 `target_start=2000ms` 对应的
  源到达量一致 ✓ —— **开口时刻看到的源恰为 `src_end` 之前的内容，与标注策略
  `tgt_start = max(src_end, tgt_end[i-1])` 自洽。**

结论：监督目标即 **“该听时输出静默、该说时输出译文，全程边听边翻”**，与
`target_start_ms / target_end_ms` 的标注意图完全一致。

---

## 9. 相关文件索引

| 文件 | 作用 |
| --- | --- |
| `custom/dataset.py` | 数据集注册 + `SimultaneousPolicyPreprocessor` |
| `custom/template.py` | `QwenOmniSpeakSimultaneousPolicyTemplate` + `_build_streaming_schedule` |
| `custom/codec_tokenizer.py` | `encode_target_audios_concat` / 帧率自校准 / 静默帧 |
| `custom/qwen_omnispeak/modeling_qwen_omnispeak.py` | `build_block_causal_mask` + 三路 loss |
| `custom/qwen_omnispeak/mcore_adaptation.py` | mcore-bridge 模型注册（thinker→GPTModel，talker 复制子模块） |
| `custom/qwen_omnispeak/mcore_talker_loss.py` | Megatron talker codec loss + `codec_to_sem_visible` 流式掩码接入 |
| `pipelines/sft/common_args.sh` | 共享参数 + `run_simultaneous_policy()`（DeepSpeed） |
| `pipelines/sft/train_simultaneous_policy_learning.sh` | DeepSpeed 一键入口；设置 `USE_RAY=true` 时作为 Ray 多机入口 |
| `pipelines/sft/common_args_megatron.sh` | 共享参数 + `run_simultaneous_policy_megatron()`（Megatron） |
| `pipelines/sft/train_simultaneous_policy_learning_megatron.sh` | `megatron sft` 一键入口 |
| `pipelines/sft/simultaneous_policy_learning.json` | 带注释的单条样本（schema 参考） |
| `pipelines/sft/data_format.jsonl` | 训练 JSONL（占位路径，正式训练前替换） |

---

## 10. 注意事项

1. **数据路径**：`data_format.jsonl` 自带占位绝对路径（`/Users/…`），正式训练前需把
   `audios` / `target_audios`（或 `alignment[i].target_audio`）指向可达文件，或依赖 codec
   tokenizer 的 S3/本地镜像解析（`S3_AUDIO_LOCAL_ROOT`、`AUDIO_LOCAL_ROOT` 等）。
2. **时间戳质量**：训练完全信任 `target_start_ms / target_end_ms`，要求单调、自洽、无异常
   尾部空洞；异常时间戳会直接污染时间轴铺设与读写掩码。
3. **修改模型代码**：`trust_remote_code` 从**导出目录** `models/custom_models/<variant>/`
   加载 `modeling_qwen_omnispeak.py`，而非 `custom/qwen_omnispeak/`。若改了源 modeling 文件，
   需把导出副本同步并清空模块缓存（见本目录 `README.md` §"Editing the model"）。
4. **Code2Wav 冻结**：本阶段只训 thinker+bridge+talker backbone，vocoder 冻结；推理时由
   Code2Wav 把 codec 还原为波形。
