# 普通流式语音翻译（normal_translation）训练方案说明

> Stage-3 SFT · `normal_translation`
> 模型：Qwen-OmniSpeak（Thinker + Dual-Route Bridge + Qwen3-TTS Talker + Code2Wav）
> 模板：`qwen_omnispeak_normal_translation`
> 入口：`pipelines/sft/train_normal_translation.sh`（DeepSpeed）/ `train_normal_translation_megatron.sh`（Megatron）

本文档完整描述 **普通流式语音到语音翻译（normal_translation）** 任务的训练方案，涵盖任务
定义、数据构建（含 **2~5s 随机翻译时延的数据侧说明** 与 **共享时间轴样本校验规则**）、输入/
输出张量构造、读写（Read-Write）策略的监督机制、损失函数、训练流程（DeepSpeed / Ray /
Megatron）、关键超参与开关，以及实现细节与自检结论。

> **与同声传译（`simultaneous_policy_learning`）的关系**：本任务在**训练机制和样本时间轴校验上与同传完全
> 等价**（codec-only 监督 + 真实时间轴静默铺设 + 单调 `codec_to_sem_visible` 读写掩码 +
> 任务无关的 main/residual codec CE）。差异仅有两点：①使用**独立任务命名/注册/模板/脚本/文档**；
> ②所有开关使用**独立的 `NORMAL_TRANSLATION_*` 环境变量前缀**，与同传管线互不串扰。
> 因此实现采用「最大化复用 + 子类化」：模板 `QwenOmniSpeakNormalTranslationTemplate` 继承同传模板，仅覆写 `env_prefix`。

---

## 1. 任务说明

### 1.1 目标

训练一个 **语音流同时输入输出** 的翻译模型：源语音从 0 时刻持续到达，模型在**每一个
时刻**决定是继续 **READ（听）** 还是 **WRITE（开口翻译）**，并以流式方式直接输出目标
语言的**语音（codec → 波形）**。

> **本任务不预测目标文本**：模型直接由源语音映射到目标 codec 流，**不生成/不监督文本**。
> 整条管线**没有 Thinker 文本 CE（Loss_text）、没有文本 token 准确率（token_text_acc）**，
> 监督是**纯 codec**（与 bridge alignment 一致）。Thinker 仍参与前向（产出 hidden 供
> bridge→talker 使用），但其文本输出不参与 loss。

与离线（offline）语音翻译的本质区别：

| | 离线 S2S 翻译 | 普通流式翻译（本方案） |
| --- | --- | --- |
| 源语音可见性 | 全部可见 | 只能看到**已到达**的源前缀 |
| 输出时机 | 听完再译 | 边听边译，需学“何时开口” |
| 监督信号 | codec CE | codec CE + **流式注意力掩码** + **静默帧监督** |

### 1.2 核心思想：用“掩码 + 静默”隐式监督读写策略

本方案**不引入任何新的 loss / 策略头 / 分类器**。读写策略通过两个机制隐式注入：

1. **WRITE（何时开口）** —— 在目标 codec 流的**真实时间轴**上，把“该听、还不该说”的
   时间段用**真实静默 codec 帧**填满，并施加 CE 监督。模型因此被显式教会：*该听时输出
   静默、该说时输出译文*。
2. **READ（能看到多少源）** —— 通过单调非减的 `codec_to_sem_visible` 流式注意力掩码，
   限制每一帧目标 codec 只能注意到“截至该帧真实时刻已到达”的源音频前缀。

两者在**同一条真实时钟**下统一：开口前的静默帧只能看到极少源 → 学“先听后说”；随时间
推进源逐帧解锁 → 学“边听边翻”。源音频起点固定为 0，`0 ~ target_start_ms[0]` 的开头听音
窗用静默填充。

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

- 在 dual-route MLP bridge 下 `sem_len == #thinker tokens`，因此**源音频占位 token 可直接
  在 `input_ids` 中定位**（首索引 `a0`，数量 `M`）。这是构造读写掩码的关键。
- Talker 把 `[sem_block ; codec]` 拼成一条序列做注意力；`codec → sem` 这一注意力块就是
  “目标帧看源/Prompt”的通道，正是 `codec_to_sem_visible` 作用的地方。
- **本任务不预测文本**：Thinker 仅作为编码器为 bridge 提供 hidden，其文本 logits 不参与
  任何 loss。
- Code2Wav（vocoder）在本阶段**冻结**，只训练 thinker + bridge + talker backbone。

---

## 3. 数据构建

### 3.1 数据 Schema（`ms-swift-speech-simul-v1`）

每条样本是一行 JSON，关键字段见 `normal_translation_sample.json`（带注释版）与
`normal_translation_data_format.jsonl`（训练用占位）。核心结构：

```jsonc
{
  "schema_version": "ms-swift-speech-simul-v1",
  "task": "normal_translation",   // 也兼容数据生成侧的 "speech2speech_translation"
  "task_prompt": "Streaming speech translation: ...",
  "messages": [
    {"role": "user", "content": "<audio>"}
    // 注意：本任务不预测文本，messages 中**不含** assistant 文本轮。
  ],
  "audios": ["<源语音 .wav（整段，从 0 时刻开始）>"],
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
      "target_start_ms": 3000,   // 目标音频该 chunk 起点（开口时刻）★
      "target_end_ms":   5238,   // 目标音频该 chunk 终点 ★
      "target_audio":    "chunk_0000.wav",
      "extra_lag_ms":    0,
      "open_delay_ms":   3000
    },
    ...
  ],
  "chat_template_kwargs": {"sampling_rate": 24000}
}
```

> **任务字符串识别**：预处理器接受 `task ∈ {"normal_translation", "speech2speech_translation"}`
> 两种取值（数据生成侧常用后者，注意它与 continue_pretrain 的下划线形式
> `speech_to_speech_translation` 不同，互不冲突）。

### 3.2 翻译时延（数据侧策略，训练侧完全信任）

> **翻译时延 2~5s 随机分布采样属于数据准备阶段的策略**：在构造每条样本时，数据侧按
> `U(2s, 5s)` 随机采样一个听译延迟来决定译员“滞后听译”的开口时机，并据此写出各 chunk 的
> `target_start_ms / target_end_ms`。**该随机时延与训练管线无关**——训练侧不重采样、不
> 重算时延，只**严格遵循** `alignment` 中已写定的 `target_start_ms / target_end_ms` 作为
> 真实时间轴的唯一依据（仅在浮点取整造成重叠时做 `max(start, cursor)` 的防重叠保护，且不
> 改写时间戳）。

### 3.3 样本校验与舍弃规则（四个 SFT 任务共享）

预处理器在**按 `chunk_id` 升序排序后**，对每条样本施加**三条共享校验规则**，任一不满足即
**舍弃该样本**（`preprocess` 返回 `None`）：

1. **源 chunk 时序不重叠**：对每个 `i ≥ 1`，要求
   ```
   source_end_ms[i-1] < source_start_ms[i]
   ```

2. **目标 chunk 时序不重叠**：对每个 `i ≥ 1`，要求
   ```
   target_end_ms[i-1] < target_start_ms[i]
   ```
   即上一个 chunk 的结束时间必须早于下一个 chunk 的开始时间（单 chunk 样本天然跳过此检查）。

3. **目标总时长上界**：要求
   ```
   max(target_end_ms) < 源音频时长
                        + 0.5 * (last.source_end_ms - last.source_start_ms)
                        + 3000ms
   ```
   其中 `last` 为排序后最后一个 chunk。**源音频时长**优先用 `soundfile` 读取 `audios[0]` 的
   真实 wav 时长（受 `NORMAL_TRANSLATION_SOURCE_DURATION_FROM_FILE` 控制，默认 `true`，可覆盖
   尾部静默），读取失败/关闭时回退为 `max(source_end_ms)`。

附加约定：

- **排序优先**：先按 `chunk_id` 排序再校验，避免 JSONL 中 chunk 乱序导致误判。
- **缺失时间戳**：当 chunk 缺少 `source_start_ms / source_end_ms / target_start_ms / target_end_ms`
  （或源时长无法确定）时，按 `NORMAL_TRANSLATION_STRICT` 决定行为：默认
  （`false`）**跳过该样本**；置 `true` 时**抛错**以便排查脏数据。
- 校验放在**预处理阶段**（行级 filter）而非模板 `_encode`：这是 ms-swift 丢弃坏样本的标准
  位置，且校验仅依赖时间戳与源时长，**无需 codec 编码**，开销极低。

### 3.4 数据集注册与预处理

- 数据集注册：`custom/dataset.py` → `register_normal_translation_datasets`，注册名
  `normal_translation`（默认数据路径读 `NORMAL_TRANSLATION_DATA`）。
- 预处理器：`NormalTranslationPreprocessor`（继承 `SimultaneousPolicyPreprocessor`）
  - 过滤 `task ∈ {normal_translation, speech2speech_translation}`；
  - 按 `chunk_id` 排序后执行 §3.3 的三条共享校验舍弃规则；
  - 合并 `task_prompt` 到 user 轮；
  - **不补 assistant 文本轮**（纯 codec 监督，无 thinker 文本 CE）；
  - 透传 `alignment` / `target_audios` 给模板。

### 3.5 Codec 缓存

每个 chunk 音频经 Qwen3-TTS Tokenizer 编码为 codec codes 后落盘缓存（按文件指纹），路径由
`NORMAL_TRANSLATION_CODEC_CACHE` / `CONTINUE_PRETRAIN_CODEC_CACHE` 决定，与同传 /
continue_pretrain 管线**共享缓存键**，避免重复编码。

---

## 4. 输入 / 输出张量构造

实现位置：`custom/template.py::QwenOmniSpeakNormalTranslationTemplate`（继承
`QwenOmniSpeakSimultaneousPolicyTemplate`）与 `custom/codec_tokenizer.py::Qwen3TTSCodecTokenizer`。

### 4.1 文本（Thinker）侧

- 模板沿用 `<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n` 结构；
- `messages` 里**只有** user 轮含 `<audio>` 占位，**没有 assistant 文本轮**；
- 因此 Thinker 仅作为编码器为 bridge 提供 hidden（`run_thinker=True`），**不产生文本监督**：
  collator 不保留文本 LM labels，而是把 `labels` 设为 main codec 的右移序列（供 trainer 计算
  **codec token 准确率**），与 bridge alignment 一致。

### 4.2 目标 Codec 流：真实时间轴铺设 + 静默填充

核心方法：`encode_target_audios_concat(...)`。**不做无缝拼接**（无缝拼接会把任务退化为
离线 TTS），而是把每个 chunk 的 codec 帧放到其 `target_start_ms` 对应的帧位置，所有空白
（开头 `0 ~ target_start_ms[0]` 听音窗 + chunk 间隙）用**真实静默 codec 帧**填满：

```
frame 0 ...... start[0] ...... start[0]+T0 ... start[1] ......
[ 静默(边听边等) ][  chunk_0 codec  ][ 静默 ][  chunk_1 codec  ] ...

start[i] = round(target_start_ms[i] / frame_ms)
```

- **帧率自校准** `frame_ms()`：直接从活的 tokenizer 读 `encode_downsample_rate /
  input_sample_rate`，实测 `1920 / 24000 = 80.0ms`（即 12.5Hz），失败时回退
  `QWEN3_TTS_FRAME_MS` 或常量 `80.0`。
- **静默帧生成** `_steady_silence_band()`：对一段数字静默波形过一次 codec，裁掉首尾瞬态，
  缓存**稳态静默帧**；`_silence_frames(n)` 平铺到任意间隙长度。解码端可还原为真静音。
- **防重叠**：`start_frame = max(round(target_start/frame_ms), cursor)`，绝不让某段在上一段
  说完前开口（应对取整误差），但不改写时间戳。
- **回退**：当时间戳缺失/数量不匹配时，回退为旧的时间连续拼接（安全网）。

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

方法：`_build_streaming_schedule(...)`。在**真实时钟**上逐帧计算每帧目标 codec 可见的源
token 数：

```
主帧 j 的真实时刻  t_ms(j) = j * frame_ms + extra_lag_ms
已到达源 token 数  arrived(j) = round( M * clamp(t_ms(j)/source_duration_ms, 0, 1) )
可见前缀长度       vis[j] = clamp(a0 + arrived(j),  max = min(a0+M, sem_len))
```

- `a0`：源音频占位 token 首索引（保证音频前的 task prompt 永远可见）；`M`：源音频占位
  token 数；`source_duration_ms`：优先读源 wav 真实时长（`NORMAL_TRANSLATION_SOURCE_DURATION_FROM_FILE`），
  回退 `max(source_end_ms)`；
- control_prefix 位（在首个主帧之前）共享首主帧的可见度；
- 最后对整条 `vis` 做 `cummax`，强制**单调非减**（接口硬性要求）。
- `extra_lag_ms` 由 `NORMAL_TRANSLATION_EXTRA_LAG_MS` 提供，作为全局 wait-k 式时延偏移
  （默认 0 = 纯数据驱动策略）。

> **设计取舍（接口决定）**：单前缀接口只能表达“可见多少个前导 semantic token”。由于本任务
> 不预测文本（codec-only），不存在“已提交译文文本”需要揭示，掩码**纯粹门控源音频**，与
> Stage-3 学“何时开口”的目标一致。

### 4.5 模型侧掩码构造（`build_block_causal_mask`）

`codec_to_sem_visible` 传入 `modeling_qwen_omnispeak.py::build_block_causal_mask`：

- `sem ↔ sem`：块内双向可见；
- `codec → codec`：因果（下三角）；
- `codec → sem`：`frame t` 可见 `sem[j]` 当且仅当 `j < vis[t]`（读写门控）；
- `sem → codec`：禁止（防未来泄漏）；
- `codec_to_sem_visible=None` 时退化为**离线全可见**。

### 4.6 Data Collator（批内对齐）

- codec 张量按 `codec_pad_id` / `-100` padding（`pad_1d` / `pad_2d_time`）；
- `codec_to_sem_visible` 在父 collator pad codec **之前**收集，再 pad 到 `codec_input_ids`
  的宽度；缺失调度的行回退为**全可见**（`1<<30`），绝不置 0；
- **codec-only**：collator 移除 `has_codec_supervision`，并把 `labels` 覆盖为 main codec 的
  右移序列。这样 `model.forward` 走纯 codec 分支（无 thinker 文本 CE / 无 `token_text_acc`），
  trainer 对 `main_codec_logits` 统计 **codec token 准确率**；Megatron 路径同理进入
  `bridge_mode`，把无意义的 thinker 文本 loss 置零。

---

## 5. 损失函数

实现位置：`modeling_qwen_omnispeak.py` 的 forward。本任务为**纯 codec 监督**，总损失为两项
加权和（**无文本 CE**）：

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

### 5.3 总损失

```
L = L_codec_main + residual_loss_weight · L_codec_res
```

（**不含 Thinker 文本 CE**：本任务不预测文本。）监控指标：`token_acc`（main codec）、
`token_acoustic_token`（residual）。**不再有 `token_text_acc`**。

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
  └─ NormalTranslationPreprocessor       # 排序 + 三条共享校验舍弃规则 / 合并 prompt / 透传 alignment（不补 assistant 文本）
       └─ Template._encode               # 音频编码 + codec 真实时间轴铺设 + 静默填充 + 读写掩码
            └─ Data Collator             # codec/调度 padding 对齐 + codec-only labels
                 └─ model.forward        # main/residual codec CE（流式掩码下，无文本 CE）
                      └─ Trainer (swift sft, DeepSpeed zero2/3 / megatron sft)
```

### 6.3 启动方式（DeepSpeed / Ray）

```bash
# 单机 / 多卡（DeepSpeed zero2 自动探测可见 GPU）
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export NORMAL_TRANSLATION_DATA=/abs/path/to/your.jsonl
export QWEN3_TTS_TOKENIZER=/path/to/project/models/Qwen3-TTS-Tokenizer-12Hz
bash pipelines/sft/train_normal_translation.sh

# 多机（Ray，先起集群）
USE_RAY=true bash pipelines/sft/train_normal_translation.sh
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
forward hook 抓取 thinker 隐状态
         ──► bridge → talker → codec_head / code_predictor → main + residual codec CE
```

> **Megatron 零改动验证项（重要）**：`mcore_talker_loss.py` 的 codec-loss 路径是
> **任务无关**的——它从 batch `pop('codec_to_sem_visible')` 并传入
> `build_block_causal_mask(...)`，且在缺少 `has_codec_supervision` 时进入 `bridge_mode`
> 置零文本 loss。normal_translation 模板产出的 **codec batch + schedule 结构与同传完全
> 相同**，因此 **Megatron 路径无需任何任务特定代码改动**。该等价性可借用同传已有的
> `test_mcore_talker_loss.py::test_streaming_parity_all_variants`（实测 `|Δ|=0`）的测法，
> 以 normal_translation 数据复跑验证——这是本管线交付时的显式验证项。

**并行约束（`_check_parallel_guards` 快速失败）**：codec / 流式路径要求 `PP=1`、`CP=1`、
`sequence_parallel=false`、不开 packing/padding-free；**thinker 仍享受完整 TP/EP/DP/ZeRO-1**，
仅复制的 talker codec 路径受限。

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export NORMAL_TRANSLATION_DATA=/abs/path/to/your.jsonl
export QWEN3_TTS_TOKENIZER=/path/to/project/models/Qwen3-TTS-Tokenizer-12Hz
export TP=2 PP=1 CP=1 EP=1
export SEQUENCE_PARALLEL=false
bash pipelines/sft/train_normal_translation_megatron.sh
```

| Megatron 专属 Env | 默认 | 说明 |
| --- | --- | --- |
| `TP` / `PP` / `CP` / `EP` / `ETP` | `2/1/1/1/1` | thinker 模型并行；codec 路径要求 `PP=CP=1` |
| `SEQUENCE_PARALLEL` | `false` | codec + 流式掩码要求关闭 |
| `MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` | `1 / 16` | |
| `QWEN_OMNISPEAK_MCORE_CODEC_LOSS` | `true` | `false` → 关闭 codec loss（本任务即无监督，不建议） |
| `QWEN_OMNISPEAK_MCORE_HIDDEN_LAYER` | 配置 | 覆盖抓取隐状态的层 |
| `RESIDUAL_LOSS` / `RESIDUAL_LOSS_WEIGHT` | 配置 | 残差 codec CE 开关与权重 |
| `NORMAL_TRANSLATION_STREAMING` | `true` | `false` → 离线全可见（消融） |

> 注：`megatron sft` 需要模型已转换为 mcore 权重格式（首次运行由 mcore-bridge 自动转换并缓存）。

---

## 7. 读写 / 时延开关（消融与时延-质量权衡）

| Env | 默认 | 作用 |
| --- | --- | --- |
| `NORMAL_TRANSLATION_STREAMING` | `true` | `false` → 离线全可见 codec 注意力（baseline 消融） |
| `NORMAL_TRANSLATION_EXTRA_LAG_MS` | `0` | 在数据驱动调度之上叠加的全局 wait-k 式时延；`0` = 纯数据驱动策略 |
| `NORMAL_TRANSLATION_SOURCE_DURATION_FROM_FILE` | `true` | `false` → 用 `max(source_end_ms)` 代替读 wav 取时长 |
| `NORMAL_TRANSLATION_STRICT` | `false` | 坏样本/缺字段是否抛错（默认跳过） |
| `QWEN3_TTS_FRAME_MS` / `QWEN3_TTS_INPUT_SR` / `QWEN3_TTS_SILENCE_SECONDS` | 自校准 | 帧率/采样率/静默探针时长兜底 |

> 数据已通过 `alignment`（含 2~5s 随机时延写定的时间戳）编码了读写策略，wait-k 是**可选**且
> 默认关闭的，仅用于时延-质量消融。所有开关均使用 `NORMAL_TRANSLATION_*` 前缀，与同传管线
> 的 `SIMULTANEOUS_POLICY_*` 完全隔离。

---

## 8. 实现要点（防回归）

- **非侵入式**：不修改 `modeling_qwen_omnispeak.py`、`mcore_talker_loss.py`、
  `mcore_adaptation.py`、`codec_tokenizer.py`，也不改变现有同传/continue_pretrain 预处理器与
  模板的默认行为（同传默认 env 前缀保持 `SIMULTANEOUS_POLICY_*`，回归安全）。
- **子类化复用**：`QwenOmniSpeakNormalTranslationTemplate` 仅覆写类属性
  `env_prefix='NORMAL_TRANSLATION'`；张量构造/掩码/collator 全部继承自同传模板。
- **校验在预处理**：三条共享舍弃规则在行级 filter 完成（返回 `None` 即丢弃），不做 codec 编码，
  开销低。
- **codec 缓存共享**：复用 `CONTINUE_PRETRAIN_CODEC_CACHE` 解析键，与同传/continue_pretrain
  共享已编码 codes。

---

## 9. 相关文件索引

| 文件 | 作用 |
| --- | --- |
| `custom/dataset.py` | `register_normal_translation_datasets` + `NormalTranslationPreprocessor`（三条共享校验舍弃规则 + 源时长读取） |
| `custom/template.py` | `QwenOmniSpeakNormalTranslationTemplate`（继承同传模板，`env_prefix='NORMAL_TRANSLATION'`）+ `register_normal_translation_template` |
| `custom/codec_tokenizer.py` | `encode_target_audios_concat` / 帧率自校准 / 静默帧（复用，未改动） |
| `custom/plugin.py` | 注册 normal_translation 数据集与模板 |
| `custom/qwen_omnispeak/modeling_qwen_omnispeak.py` | `build_block_causal_mask` + 三路 loss（复用，未改动） |
| `custom/qwen_omnispeak/mcore_talker_loss.py` | Megatron talker codec loss + `codec_to_sem_visible` 接入（任务无关，未改动） |
| `pipelines/sft/common_args.sh` | 共享参数 + `run_normal_translation()`（DeepSpeed，与 `run_simultaneous_policy()` 共用） |
| `pipelines/sft/train_normal_translation.sh` | DeepSpeed 一键入口；设置 `USE_RAY=true` 时作为 Ray 多机入口 |
| `pipelines/sft/common_args_megatron.sh` | 共享参数 + `run_normal_translation_megatron()`（Megatron，与 `run_simultaneous_policy_megatron()` 共用） |
| `pipelines/sft/train_normal_translation_megatron.sh` | `megatron sft` 一键入口 |
| `pipelines/sft/normal_translation_sample.json` | 带注释的单条样本（schema 参考，多 chunk 示范校验规则） |
| `pipelines/sft/normal_translation_data_format.jsonl` | 训练 JSONL（占位路径，正式训练前替换） |

---

## 10. 注意事项

1. **数据路径**：`normal_translation_data_format.jsonl` 自带占位绝对路径（`/Users/…`），正式
   训练前需把 `audios` / `target_audios`（或 `alignment[i].target_audio`）指向可达文件，或依赖
   codec tokenizer 的 S3/本地镜像解析（`S3_AUDIO_LOCAL_ROOT`、`AUDIO_LOCAL_ROOT` 等）。
2. **时间戳质量**：训练完全信任 `source_start_ms / source_end_ms / target_start_ms / target_end_ms`，要求单调、自洽；不满足
   §3.3 三条共享校验规则的样本会在预处理阶段被直接舍弃。
3. **修改模型代码**：`trust_remote_code` 从**导出目录** `models/custom_models/<variant>/` 加载
   `modeling_qwen_omnispeak.py`。若改了源 modeling 文件，需把导出副本同步并清空模块缓存
   （见本目录 `README.md` §“Editing the model”）。
4. **Code2Wav 冻结**：本阶段只训 thinker+bridge+talker backbone，vocoder 冻结；推理时由
   Code2Wav 把 codec 还原为波形。
