## Qwen-OmniSpeak 模型说明

### 背景
由于Qwen3-Omni Talker的音频tokenizer没有开源，导致无法进行端到端的语音训练，因此改用 Qwen Omni Thinker + Qwen3-TTS Talker 的组合方案（命名为 Qwen-OmniSpeak）。

当前 Thinker 可配置为：

- `Qwen/Qwen3-Omni-30B-A3B-Instruct` / 兼容 Qwen3-Omni：沿用第 24 层 hidden states 输入 Talker。
- `Qwen/Qwen2.5-Omni-7B`：使用最后一层 hidden states 输入 Talker。
- `Qwen/Qwen2.5-Omni-3B`：使用最后一层 hidden states 输入 Talker。

所有 Thinker 均支持 A-J 十个 Qwen-OmniSpeak 变种（Bridge 类型与 Talker 层裁剪方式保持不变）。

**为什么选用Qwen3-TTS作为 Talker？**
- Qwen3-TTS的文本vocab和 Qwen3-Omni 的 text vocab 完全对应，便于语义上的对齐。
- Qwen3-TTS的帧率只有12Hz，满足实时性要求。
- Qwen3-TTS支持 10 种语言，且表现优异。

---

#### 一、Qwen3-Omni Thinker（多模态理解，30B-A3B MoE）

##### 输入端（多路）

| 输入模态 | 原始形状 | 经编码器后 | 进入 Thinker 主干的形状 |
|---------|---------|-----------|----------------------|
| **文本** | `input_ids: [B, T_text]` (vocab=152064) | embed → `[B, T_text, 2048]` | `[B, T_text, 2048]` |
| **音频** | `mel: [B, 128, T_mel]` (128 mel bins) | Audio Encoder (d_model=1280, 32 层) → `output_dim=2048` | `[B, T_audio, 2048]` |
| **视觉** | `pixels: [B, C=3, H, W]` (768×768, patch=16) | ViT (hidden=1152, 27 层, deepstack@[8,16,24]) → `out_hidden_size=2048` | `[B, T_vision, 2048]` |

所有模态在进 Thinker 之前都已对齐到 **hidden_size = 2048**，按特殊 token (`<\|audio_start\|>`, `<\|vision_start\|>` 等) 拼接成统一序列。

##### Thinker 主干（MoE LLM）

| 项 | 值 |
|---|---|
| 输入 | `[B, T_total, 2048]` |
| 层数 | 48 |
| 专家配置 | 128 experts × top-8 (moe_intermediate_size=768) |
| GQA | 32 Q heads / 4 KV heads, head_dim=128 |
| 输出 hidden | `[B, T_total, 2048]` |
| 输出 logits | `[B, T_total, 152064]` |
| 关键中间产物 | **第 24 层 hidden state**（供 Talker 接收） |

#### 二、Qwen3-Omni Talker（语音生成主干，MoE）

Talker 由两个串行子模块组成：**Talker 主干（MoE LLM）** → **Code Predictor（残差码头）** → 再下游接 Code2Wav。

##### Talker 主干

| 项 | 形状 / 值 |
|---|---|
| **输入 1：Thinker hidden** | `[B, T, 2048]`（来自 Thinker 第 24 层，`accept_hidden_layer=24`） |
| **输入 2：codec token ids** | `[B, T]`（vocab=3072，含 `<codec_bos>=2149`, `<codec_eos>=2150`, speaker_id 等） |
| **输入投影** | `2048 → 1024`（thinker_hidden_size → talker hidden_size） |
| **主干 hidden_size** | **1024** |
| **层数 / 专家** | 20 层，128 experts × top-6 (moe_intermediate_size=384, shared=768) |
| **GQA** | 16 Q heads / 2 KV heads, head_dim=128 |
| **主干 hidden 输出** | `[B, T, 1024]` |
| **主干 logits 输出（主码 group 0）** | `[B, T, 3072]`（codec vocab） |

##### Code Predictor（残差码本头，5 层 Transformer）

| 项 | 形状 / 值 |
|---|---|
| 输入 | Talker 主干 hidden `[B, T, 1024]` |
| hidden_size | 1024 |
| 层数 | 5（全 full_attention） |
| 输出 logits | **`[B, T, 16, 2048]`**（16 组残差码本，每组 vocab=2048） |
| 离散输出 | `codes: [B, T, 16]`（int，每位 ∈ [0, 2047]） |

##### 下游：Code2Wav（声码器，非 Talker 严格意义内）

| 项 | 形状 / 值 |
|---|---|
| 输入 | `codes: [B, T, 16]` |
| 上采样率 | [8, 5, 4, 3] = **480×** |
| 输出波形 | `[B, T × 480]`（约 24kHz 采样率，T 单位为 12Hz frame，13 frame/s） |

---

#### 三、Qwen3-TTS（独立 TTS 模型，非 Omni 一部分）

以 **Qwen3-TTS-12Hz-1.7B-CustomVoice** 为参考（与 Omni Talker 同一家族，但参数独立）：

##### 顶层接口（对外暴露）

| 项 | 形状 / 值 |
|---|---|
| **唯一对外输入** | `text: str`（字符串，内部自己 tokenize） |
| 内部 tokenizer | Qwen2Tokenizer（**与 Qwen3-Omni 文本 vocab 完全一致**, vocab=151,936） |
| 可选输入 | `speaker_id`、参考音频（custom voice） |
| 输出 | 24kHz waveform `[T × 480]` |

##### Talker 主干（核心 1.7B Dense LLM，**非 MoE**）

| 项 | 形状 / 值 |
|---|---|
| **输入：text token ids** | `[B, T_text]`（text_vocab_size=151,936） |
| **输入：codec token ids** | `[B, T_codec]`（vocab=3072） |
| **输入 embedding 维度** | `[B, T, 2048]` |
| **hidden_size** | **2048** |
| **intermediate_size** | 6144 |
| **层数** | **28**（dense，非 MoE） |
| **GQA** | 16 Q heads / 8 KV heads, head_dim=128 |
| **主干 hidden 输出** | `[B, T, 2048]` |
| **主码 logits 输出** | `[B, T, 3072]`（codec vocab） |
| max_position_embeddings | 32,768 |

##### Code Predictor（残差码本头）

| 项 | 形状 / 值 |
|---|---|
| 输入 | Talker 主干 hidden `[B, T, 2048]` |
| 内部 hidden_size | **1024**（投影：2048 → 1024） |
| 层数 | 5（全 full_attention） |
| 输出 logits | **`[B, T, 16, 2048]`** |
| 离散输出 | `codes: [B, T, 16]` |

##### Codec Decoder（Qwen3-TTS-Tokenizer-12Hz, 开源）

| 项 | 形状 / 值 |
|---|---|
| 输入 | `codes: [B, T, 16]` |
| 帧率 | 12 Hz |
| 输出 | 24kHz waveform |

---

#### 三方对齐速查表（最关键的拼接信息）

| 维度 | Qwen3-Omni Thinker | Qwen3-Omni Talker 主干 | Qwen3-TTS Talker 主干 |
|---|:---:|:---:|:---:|
| 架构 | MoE (128×top8) | MoE (128×top6) | **Dense** |
| hidden_size | **2048** | **1024** | **2048** |
| 层数 | 48 | 20 | 28 |
| text vocab_size | **152064** | — | text_vocab=**151936**（实际 BPE 完全相同，差额是 padding 槽位）|
| codec vocab_size | — | **3072** | **3072** |
| num_code_groups | — | 16 | 16 |
| 残差码 vocab | — | 2048/组 | 2048/组 |
| 接收的外部 hidden | — | `2048→1024` (from Thinker L24) | 无（自带 text embedding） |
| Codec Decoder | Code2Wav（**未独立开源 wav→code**） | 同 Talker | **Tokenizer-12Hz（开源双向）** |

因此下面给出 `Qwen3-Omni Thinker + Qwen3-TTS Talker` 组合方案的完整模型定义、训练路线与设计动机。当前版本的核心决策有两条：

- **Talker 主干采用 `Qwen3-TTS` 1.7B backbone 的上层 16 层**
- **Talker 输入采用同传优先的双路输入结构**：
  - 语义路：`text embedding + hidden states from middle layer`
  - 自回归语音路：`padding + codec embedding`

### 1. 方案概述

`Qwen-OmniSpeak` 的目标是把 `Qwen3-Omni` 的多模态理解与推理能力，和 `Qwen3-TTS` 的高质量语音生成能力拼成一个可训练、可评测、可流式推理的统一模型，并优先支持**端到端同传**场景下的“边听边说”。

整体架构如下：
```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      Qwen-OmniSpeak                                          │
│                                  (Dual-Route Talker Design)                                   │
│                                                                                               │
│  Audio / Vision / Text Input                                                                  │
│            │                                                                                  │
│            ▼                                                                                  │
│  ┌──────────────────────────────┐                                                             │
│  │   Thinker (Qwen3-Omni 30B)   │   MoE, 48 layers, hidden = 2048                             │
│  │        multimodal LM         │                                                             │
│  └───────────────┬──────────────┘                                                             │
│                  │                                                                            │
│                  ├── (a) text token ids / text embeddings [B, T_text] ───────┐                │
│                  │                                                           │                │
│                  └── (b) hidden states @ middle layer (e.g. L24)             │                │
│                      [B, T_sem, 2048] ───────────────────────────────────────┤                │
│                                                                              ▼                │
│                                                                  ┌────────────────────────┐   │
│                                                                  │    DualRoute Bridge    │   │
│                                                                  │   (MLP / Cross-Attn)   │   │
│                                                                  │      2048  →  2048     │   │
│                                                                  │   ★ Stage1 主要训练     │   │
│                                                                  └───────────┬────────────┘   │
│                                                                              │                │
│                                                                              │                │
│                                  Semantic Route                              │                │
│                    ┌─────────────────────────────────────────────────────────┘                │
│                    │                                                                          │
│                    ▼                                                                          │
│     [semantic tokens / slots] = [projected text embeddings ; projected thinker hidden]        │
│                    │                                                                          │
│                    │                                                                          │
│                    │                     Autoregressive Codec Route                           │
│                    │         ┌───────────────────────────────────────────────────────┐        │
│                    │         │                                                       │        │
│                    │         │  Training: target_audio ──> Qwen3-TTS Tokenizer       │        │
│                    │         │            └────────────> codec ids [B, T_codec, 16]  │        │
│                    │         │                           └─ shift-right              │        │
│                    │         │                           └─ codec embedding lookup   │        │
│                    │         │                                                       │        │
│                    │         │  Inference: previous generated codec ids              │        │
│                    │         │             └──────────> codec embedding lookup       │        │
│                    │         └───────────────────────────────┬───────────────────────┘        │
│                    │                                         │                                │
│                    │                                         ▼                                │
│                    │                     [codec slots] = [padding ; shifted codec embeddings] │
│                    │                                         │                                │
│                    └──────────────────────────────┬──────────┘                                │
│                                                   │                                           │
│                                                   ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                 Qwen3-TTS Talker (1.7B Dense, hidden = 2048, top 16 layers)              │ │
│  │                                                                                          │ │
│  │  Dual-lane / dual-slot input layout:                                                     │ │
│  │   - Semantic lane : text embeddings + thinker hidden states                              │ │
│  │   - Codec lane    : padding + codec embeddings                                           │ │
│  │                                                                                          │ │
│  │  Attention / mask rule:                                                                  │ │
│  │   - semantic lane 对 codec lane 全可见                                                    │ │
│  │   - codec lane 严格自回归因果 mask                                                         │ │
│  │   - 支持 streaming chunk 增量刷新 semantic route                                           │ │
│  │                                                                                          │ │
│  │  Output: main codec logits [B, T_codec, 3072]                                            │ │
│  └───────────────────────────────────────┬──────────────────────────────────────────────────┘ │
│                                          ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │      Code Predictor (5 layers, hidden = 1024)  →  residual codes [B, T_codec, 16]        │ │
│  └───────────────────────────────────────┬──────────────────────────────────────────────────┘ │
│                                          ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │          Qwen3-TTS-Tokenizer-12Hz Decoder  →  24kHz Waveform Output                      │ │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

职责分工如下：

- **`Thinker`**：负责音频/文本/视觉输入理解、多模态推理、目标文本生成，以及为 Talker 提供语义表示。
- **`DualRoute Bridge`**：负责把 Thinker 的 `text embeddings` 与中层 `hidden states` 投影到 Talker 可消费的条件空间，并构建双路输入。
- **`Talker`**：负责在双路输入条件下生成 codec 主码序列。
- **`Code Predictor`**：负责补全 16 组残差码本。
- **`Decoder`**：负责把离散 codec 序列解码成最终波形。

整体数据流为：

1. 输入音频/文本/图像进入 `Qwen3-Omni Thinker`
2. Thinker 输出目标文本、文本 embedding 与中层 hidden states
3. `DualRoute Bridge` 构造语义路输入：`text embedding + middle-layer hidden states`
4. 训练或推理端构造 codec 自回归路输入：`padding + codec embedding`
5. 两路输入按块组织后送入 `Qwen3-TTS` 上层 16 层主干
6. `Code Predictor` 预测余下残差码
7. `Qwen3-TTS Tokenizer Decoder` 解码波形

### 2. 关键维度与模块边界

#### 2.1 Thinker 侧

- 基础模型：`Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Thinker 文本 hidden size：`2048`
- Thinker 层数：`48`
- 官方 Omni Talker 参考接入层：`accept_hidden_layer = 24`

#### 2.2 Qwen3-TTS 侧

- 基础模型：`Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Talker backbone 层数：`28`
- Talker backbone hidden size：`2048`
- intermediate size：`6144`
- num attention heads：`16`
- num key/value heads：`8`
- codec vocab size：`3072`
- `code_predictor`：`5` 层、`1024` hidden、每组 vocab `2048`
- code groups：`16`

#### 2.3 本方案中的 Talker 定义

本方案不直接把完整 `Qwen3-TTS` 28 层 backbone 整体作为 Talker，而是定义：

- **Talker 主干 = `Qwen3-TTS` backbone 的上层 16 层**
- **下层 12 层不作为首版默认生成主干参与前向**
- **Talker 输入 = 双路输入**：
  - 语义路：`projected text embeddings + projected middle-layer hidden states`
  - 自回归语音路：`padding + shifted codec embeddings`
- `code_predictor` 和 `decoder` 仍保持独立模块，不并入 Talker 主干层数统计

### 3. 为什么仍然选择上层 16 层

### 3.1 不直接用完整 28 层的原因

直接使用完整 28 层虽然最接近原始 `Qwen3-TTS` 预训练结构，但在本任务里并不是最优默认选择。

原因有三点：

- **上游已有强语义模型**：`Qwen3-Omni Thinker` 已经完成大部分多模态理解、推理和目标文本生成，Talker 不再需要承担完整的“从文本重新建立全部语义表征”的职责。
- **训练成本过高**：完整 28 层会显著增加自回归训练成本、显存开销与推理时延，尤其是在联合微调和流式推理时更明显。
- **工程复杂度更高**：完整主干意味着更高的冻结/LoRA 管理成本，也更难做双路输入相关的增量实验和消融。

### 3.3 为什么是 16 层

把 `28` 层裁成 **上层 16 层** 是一个效果、稳定性与成本之间较平衡的点：

- **保留了超过半数的高层生成深度**，仍有足够 capacity 完成 codec 主码建模
- **更贴近输出端**，更适合承接 speaker/style/prosody 这类高层生成控制
- **相比完整 28 层显著降本**，更利于双路联合训练和流式部署
- **相比 top-8 / top-12 更稳**，能减轻 Bridge 必须同时完成“语义分流 + 分布校正”的负担

简化理解就是：

- `Thinker` 负责“理解与说什么”
- `DualRoute Bridge` 负责“把理解结果分发成可流式消费的双路条件”
- `Talker top-16` 负责“怎么把条件稳定地变成高质量 codec 序列”

### 4. 为什么同传场景要采用双路输入

### 4.1 prefix-only 方案为什么不够

如果只是把 Thinker hidden 通过 `Bridge` 压成一个 prefix，再让 Talker 靠 codec 自回归往下生成，那么这个结构更适合离线 TTS 或段级 speech-to-speech 任务，但对**边听边说的端到端同传**往往不够。

原因在于同传同时需要两类能力：

- **语义增量更新**：随着新的源语音 chunk 到来，Talker 需要不断刷新“现在该说什么”
- **语音自回归连续性**：Talker 还需要维持已经说出的语音轨迹，保证韵律、音色和流式稳定性

因此，同传问题天然要求把以下两件事分开建模：

- `What to say`
- `How to continue saying it`

### 4.2 双路输入各自负责什么

本方案中，Talker 的双路输入分别承担：

- **语义路**：负责“说什么”
  - 输入来自 `text embeddings + hidden states from middle layer`
  - 文本 embedding 负责明确文本级语义条件
  - 中层 hidden 负责保留源语音相关的多模态增量语义
- **自回归语音路**：负责“怎么继续说”
  - 输入来自 `padding + codec embedding`
  - 已生成 codec embedding 维持连续语音轨迹
  - padding 预留未来待发射的语音位置，便于 chunk 化发射与 teacher forcing

### 4.3 为什么不完全照搬官方 Omni Talker

本方案采用的是**Omni-like 双路输入思想**，但不会机械照抄官方 `Qwen3-Omni Talker` 的整套实现，原因是：

- 当前主干换成了 `Qwen3-TTS top-16`，输入分布与原生 Omni Talker 不完全相同
- 需要尽量少改动 `Qwen3-TTS` 的高层 codec LM 能力
- 更稳的落地方式是：**保留双路输入能力，同时把两路输入组织成 `Qwen3-TTS top-16` 更容易消化的 block-causal 入口格式**

因此，本方案的目标是：

- **保留双路输入能力等价性**
- **不强求和官方 Omni Talker 完全同构**

### 5. 模型结构定义

### 5.1 Thinker

输入可以是：

- `messages`
- `audios`
- 可选 `images/videos`

Thinker 的输出包括：

- `target_text` 对应文本 token 或生成序列
- `thinker_text_embeds`
- 指定层的 `thinker_hidden_states`
- 可选 `alignment / chunk boundary / latency` 辅助信号

默认建议：

- 第一版优先读取 **第 24 层** hidden 作为主语义 hidden
- 文本 embedding 使用 Thinker 文本侧 embedding 输出或与 `target_text_ids` 对应的 embedding
- 如果后续发现 top-16 Talker 更偏依赖高层语义，可增加 `L24/L32/L40` 的消融

### 5.2 DualRoute Bridge

`DualRoute Bridge` 是本方案的核心新模块，负责：

- `text_projection`: 2048 → 2048
- `hidden_projection`: 2048 → 2048
- 从 Thinker 分布对齐到 Talker 分布
- 融合 `speaker/reference/target_text` 条件
- 构造双路输入需要的 `semantic_route_mask`
- 支持流式 chunk 级别的增量更新与缓存

建议实现两个版本：

- **V1: DualRoute MLP Bridge**
  - 结构简单、稳定
  - 包含 `text_projection`、`hidden_projection`、`RMSNorm/MLP entry adapter`
  - 适合先跑通 Stage1
- **V2: DualRoute Q-Former Bridge**
  - 更适合变长 hidden 压缩
  - 更适合在同传中做 chunk 级语义压缩与对齐
  - 适合后续提升效果与流式表现

### 5.3 语义路设计

语义路输入的目标是构造一个**持续更新的条件语义块**，输入内容由两部分组成：

- `projected thinker_text_embeds`
- `projected thinker_hidden_states`

语义路的推荐构造方式：

- 先基于 `alignment`、chunk 边界或规则模板构建 `semantic_route_mask`
- 对每个语义位置：
  - 若该位置更适合表达文本级译文条件，使用 `text_projection(thinker_text_embeds)`
  - 若该位置更适合表达源语音增量语义，使用 `hidden_projection(thinker_hidden_states)`
- 最终得到统一 hidden size 为 `2048` 的 `semantic_inputs_embeds`

这一路输入的设计动机是：

- 文本 embedding 更擅长稳定表示译文 token 条件
- 中层 hidden 更擅长携带“还在听、还在变”的多模态语义证据
- 二者混合比单用 prefix 更适合同传

### 5.4 codec 自回归路设计

codec 路输入的目标是构造一个**持续滚动的语音生成块**，其输入形式为：

- `padding + codec embedding`

具体定义为：

- 训练时：
  - codec 序列前插入 talker 预训练熟悉的**控制 token 前缀**（`think/nothink`、`think_bos/eos`、可选 `language_id`、`codec_pad`、`codec_bos`），不参与监督（标签 -100），详见 §10.6
  - 由 `codec_labels` 右移一位得到主码 teacher forcing 输入（前缀末尾的 `codec_bos` 位预测第 0 帧主码）
  - padding 使用 `codec_pad_id = 2148`（上游真实 pad；**不要用 0**，0 是合法码本值，详见 §10.5）
  - `codec_input_ids` 经 `codec_embedding` 映射为 `codec_inputs_embeds`
- 推理时：
  - 初始位置为上述控制前缀（以 `codec_bos` 结尾）
  - 每生成一个新的 codec token，就替换对应 pad 位置
  - 维持一个随时间增长的 codec 自回归缓存

这一路输入的设计动机是：

- 显式保留“已经说出去的语音历史”
- 让 Talker 在生成过程中持续感知自己的已发射语音轨迹
- 兼容 teacher forcing、chunk 化解码与 streaming codec 生成

### 5.5 Talker 主干与双路融合方式

Talker 主干采用 `Qwen3-TTS` 的 **上层 16 层**。推荐的双路融合方式是：

- 先构造 `semantic_inputs_embeds`
- 再构造 `codec_inputs_embeds`
- 在入口处按块组织为：
  - `[semantic block][codec block]`
- 使用 **block-causal attention mask** 控制信息流

推荐的 mask 规则：

1. `codec block` 的每个位置都可以看见当前窗口内全部 `semantic block`
2. `codec block` 只能看见过去的 `codec block`，不能看未来 codec
3. `semantic block` 在流式场景中只包含当前已提交 chunk 的语义信息，不泄漏未来 chunk
4. 首版不强求完全复刻 Omni 的 token-level 内部拓扑，但保留“双路输入 + 语义对 codec 全可见 + codec 自回归因果约束”这三个核心性质

> 流式扩展点（v2）：`build_block_causal_mask` 提供可选参数 `codec_to_sem_visible`，默认 `None`＝上述“codec 看全部语义”的离线行为；传入单调非降的逐帧可见语义长度即可切换为 chunk-causal（read-write）流式 mask，无需改动其它逻辑。详见 §10.8。

这种组织方式的优点是：

- 对 `Qwen3-TTS top-16` 更友好
- 仍然保留 Omni-like 的双路输入能力
- 工程上更容易在 `inputs_embeds + custom attention mask` 入口处落地

### 5.6 Code Predictor 与 Decoder

> v2 更新：首版“`code_predictor` 默认冻结”的策略已调整。`code_predictor` 现按上游 Qwen3-TTS 的**码本组维度自回归 roll-out**（teacher forcing）实现并**参与 Stage-1 训练**，学习 15 级残差码；详见 §10.3 / §10.7。

- `code_predictor` 在 Stage-1 与 bridge、talker 底 4 层一起训练（残差码 CE 辅助主码 CE）
- `decoder` 首期仅用于评测和抽样试听，不进入每 step 主训练循环
- 残差预测必须用 `forward_teacher_forced`（组维度自回归），不可用早期的并行预测路径

### 6. 数据设计

延续仓库现有 schema：

- `messages`
- `audios`
- `target_text`
- `target_audio`
- `reference_audios`
- `speaker_profile`
- `alignment`
- `latency_budget_ms`

新增派生字段或离线缓存：

- `target_text_ids`
- `codec_labels`
- `codec_input_ids`
- `semantic_route_mask`
- `semantic_chunk_ranges`
- 可选 `thinker_hidden_cache`
- 可选 `thinker_text_embeds_cache`

这样可以把高成本步骤从训练主循环挪走：

- `target_audio -> codec labels`
- `Thinker 前向 -> hidden states / text embeddings`
- `alignment -> semantic_route_mask`

### 7. 训练策略

### 7.1 Stage0：预处理

离线生成：

- `target_audio -> codec labels`
- `codec_labels -> codec_input_ids`
- 可选 `Thinker hidden cache`
- 可选 `Thinker text embeds cache`
- `alignment/chunk -> semantic_route_mask`

目标：

- 降低训练期重复计算
- 保证大规模语料可流式加载
- 让双路输入构造尽量前移到预处理或轻量在线步骤

### 7.2 Stage1：Thinker-Talker对齐训练（Bridge + 残差头 + talker 底层）

> v2 更新：Stage-1 从“仅训 bridge”扩展为同时训练 bridge、残差 code predictor 与 talker 最下面 4 层，以适应 `[语义块][codec块] + block-causal` 这一 talker 未见过的输入排布。详见 §10.7。

可训练模块：

- `model.bridge`（`text_projection` / `hidden_projection` / `entry_adapter`）
- `model.talker.code_predictor`（15 级残差码，组维度自回归）
- `model.talker.model.layers.0~3`（talker 最下面 4 层）

冻结模块：

- `Thinker`
- `Qwen3-TTS top-16 Talker` 的其余层（仅放开底 4 层）
- `decoder`

实现方式：`--freeze_parameters_ratio 1.0` + `--trainable_parameters model.bridge model.talker.code_predictor` + `--trainable_parameters_regex 'model\.talker\.model\.layers\.[0-3]\.'`。

目标：

- 先学会把 Thinker 语义稳定映射到双路 Talker 入口空间
- 让 talker 底层适应新的拼接输入排布 + 控制 token 前缀
- 验证双路输入格式本身是否成立

损失：

- `L_codec`（主码 CE）
- 残差码 CE（`code_predictor`，组维度自回归 teacher forcing）
- 可选 `L_route_consistency`

训练任务：
- TTS语音合成
- 端到端语音对话
- 端到端语音翻译

### 7.3 Stage2：Thinker-Talker联合训练

可训练模块：

- `DualRoute Bridge`
- `Thinker` 上层 LoRA
- `Qwen3-TTS top-16` 上层 LoRA

目标：

- 提升自然度、speaker/style 一致性与语义保真
- 提升流式同传中的稳定性与增量修正能力

联合损失建议：

- `L_codec`
- `L_text`
- 可选 `L_distill`
- 可选 `L_latency`

其中：

- `L_distill` 可用于让 `top-16` 学习 `full-28` teacher 的主码分布
- `L_latency` 可用于优化发射时机、等待策略或 chunk 级别延迟偏好

训练任务：
- 语音到文本翻译任务
- 语音到语音翻译任务
- 多说话人区分任务*

### 7.4 Stage3：有监督同传微调

目标：

- 提升自然度
- 提升时延/稳定性/风格一致性
- 处理 voice clone、speech translation 与 simultaneous interpretation 场景中的主观体验差异

方法：

- DPO / pairwise preference learning
- latency-aware preference optimization

任务：
- 实时同传读写策略学习
- 语音翻译与跨语言语音克隆联合训练
- 多说话人同传*
- 语音文本同步翻译任务*

### 8. 推理策略

推理阶段流程：

1. Thinker 读入输入语音/文本/多模态上下文
2. 增量输出文本 embedding 与中层 hidden
3. `DualRoute Bridge` 刷新 `semantic block`
4. `codec route` 基于已生成 codec 构造 `padding + codec embedding`
5. top-16 Talker 主干在 block-causal mask 下继续生成 codec 主码
6. `code predictor` 补残差码
7. `decoder` 解码音频

流式推理时：

- Thinker 做 chunk 化输入
- 语义路按 chunk 增量更新
- codec 路保持自回归缓存
- Talker 按 codec chunk 逐步输出
- Decoder 按分块回放音频
- 可选引入 `wait-k` 或 learned emission policy 控制发射时机

### 9. 首版实验矩阵

默认建议至少做五组实验：

- **A**：`dual-route` + `full-28 train` baseline
- **B**：`dual-route` + `full-28 train top-16`
- **C**：`dual-route` + `top-16` 默认方案
- **D**：`dual-route top-16 + Q-Former Bridge`
- **E**：`dual-route top-12` 压缩版

其中：

- **A** 用于量化双路输入的收益
- **B** 是首版默认目标架构
- **C** 用于验证更强 Bridge 是否提升同传效果
- **D** 用于验证还能压到什么程度

---

## 10. 实现修订与关键建模决策（v2，本轮排查与重构）

本章记录在首版跑通后，针对“bridge_alignment 的 loss 卡在 ~12 且 token 预测准确率几乎为 0”这一现象做的彻底排查，以及由此引出的一系列建模/实现决策。结论已全部落地到代码，正文 5.x / 7.x 的相关描述也据此做了同步修订。

### 10.1 问题现象与三个独立根因

现象：语音到语音 / 文本到语音任务，loss 下降到 ~12 后不再下降，训练集/验证集 token_acc 都≈0，疑似没训起来。

排查后定位到**三个相互独立、叠加成假象**的问题：

1. **token_acc 指标比错了张量（解释 acc≈0）**：模型 forward 返回的 `logits` 是 Thinker 的**文本 logits**（vocab=152064），而 bridge_alignment 把 `labels` 设成了 **codec 标签**。swift 的 `_compute_acc` 直接拿 `logits.argmax(-1)` 和 `labels` 比，两者形状/语义完全不对应 → 大多数 step 直接跳过、偶尔蒙中 → acc≈0。**该指标当时根本不能反映 codec 是否学会。**
2. **残差码预测器 forward 结构性错误（解释 loss 卡 12）**：`code_predictor` 把 15 组残差**用同一隐藏向量并行**预测，既没用 `codec_embedding`，也没有码本组维度的 residual roll-out，等价于随机猜 → 残差 CE 恒等于 `ln(2048)=7.62`。主码 CE 当时已从 8 降到 ~4.4，但残差把总 loss 顶死在 `4.4 + 7.6 ≈ 12`。
3. **`codec_pad_id=0` 与合法码本值冲突**：上游真实 pad 是 **2148**，而 0 是码本 0 的合法取值；一旦走到“用 `!=pad` 推 mask”的路径就会误伤主码为 0 的真实帧。

### 10.2 codec 码本结构澄清

经 tokenizer 缓存实测：每帧 **16 级码本，12.5Hz**。

- **主码 = 码本第 0 级**：语义码（semantic codebook），vocab=4096（实测值域落在 `[0, 2045]`）。走 Thinker→Bridge→Talker→`codec_head` 自回归预测。
- **残差码 = 码本第 1~15 级**：共 15 组声学码（acoustic codebooks），每组 vocab=2048。由 `code_predictor` 预测。

对应关系：`num_code_groups=15`（残差组数）= 16 级 − 1 级主码。数据拆分、teacher forcing 右移对齐、码本值域均已验证正确。

### 10.3 残差 Code Predictor：并行预测 → 码本组维度自回归 roll-out

按上游 Qwen3-TTS（`forward_sub_talker_finetune` + `code_predictor.forward_finetune`）重写为 `forward_teacher_forced`：

对每个 codec 帧，构造长度 `num_code_groups + 1` 的**组序列**：

```
[ talker_hidden , E_main(main_code) , E_res[0](res_0) , ... , E_res[G-2](res_{G-2}) ]
  pos 0           pos 1               pos 2                     pos G
```

在**组维度上跑因果 transformer**（组位置 `i` 只能看 `0..i-1` 加 talker_hidden），`lm_head[i]` 在位置 `i+1` 预测残差组 `i`，teacher forcing 喂入真值低阶码。

验证（CPU 小模型 + 真实 codec 缓存）：输出形状 `[B,T,15,2048]`；**组维度因果性精确成立**（改动残差组 g，组 ≤g 预测变化为 0、组 >g 显著变化）；未训练时残差 CE = `ln(2048)=7.62`（即原先卡死的随机基线，现在可被训练拉低）。

### 10.4 token 准确率指标修复

- 模型在**纯 codec 模式**（无 thinker 文本监督）下，把 `outputs.logits` 改为 `main_codec_logits`；文本监督模式仍返回文本 logits。
- bridge_alignment collator 把 `labels` 构造成**右移对齐**的主码流 `[-100, m0, m1, ..., m_{T-1}]`，使 swift 的 next-token shift 后恰好 `pred(t) ↔ codec_labels[t]`，padding/EOS/控制位均为 -100 被忽略 → token_acc 反映真实主码准确率。

### 10.5 codec_pad_id 0 → 2148

config 默认值、模板默认值、十个已导出变体的 `config.json` 全部改为 2148（上游 Qwen3-TTS 真实 codec pad）。

### 10.6 talker 预训练分布对齐：控制 token 前缀（决策：选项 A）

通过通读 Qwen3-TTS talker 源码，确认其预训练时 codec 序列前有一串“模式开关”控制 token，且文本与 codec 是**逐位叠加 + trailing_text_hidden 持续注入**。据此评估了两条路线：

- **选项 B（彻底照搬原生“逐位叠加”范式）——否决**：原生叠加 + trailing_text_hidden 依赖**文本-语音单调对齐**（同一句话字↔音，文本短则用 tts_pad 补齐到 codec 长度）。而本项目是**翻译/同传**（源↔目标跨语言、非单调、长度无对应），强行逐位叠加会注入错误的对齐归纳偏置，且语义 hidden 与 12.5Hz 目标帧率无法对齐。原生 trailing_text_hidden 在翻译里也没有语义。
- **选项 A（保留 `[语义块][codec块]` 拼接范式，仅补控制 token 前缀）——采纳**：控制 token 是**任务无关的“模式开关”**，与对齐范式正交，让 frozen/半放开的 talker 进入其熟悉的条件状态。

落地的控制前缀（base talker `spk_id` 为空，不加 speaker 位；控制位标签全为 -100、不监督）：

```
识别到目标语言:  [codec_think, codec_think_bos, language_id, codec_think_eos, codec_pad, codec_bos]
未识别(auto)  :  [codec_nothink, codec_think_bos, codec_think_eos, codec_pad, codec_bos]
```

`language_id` 由数据的 `target_lang`（支持 `en/english/zh/chinese` 等别名）映射到 talker 的 `codec_language_id`。控制 token id 与 `codec_language_id` 映射在模型 config 中定义（默认取自 Qwen3-TTS-12Hz-1.7B-Base：think=2154 / nothink=2155 / think_bos=2156 / think_eos=2157）。

### 10.7 冻结策略更新（Stage-1 不再“仅 bridge”）

为让 talker 适应 `[语义块][codec块] + block-causal` 这种它预训练时没见过的输入排布，Stage-1 可训练模块扩展为：

- `model.bridge`
- `model.talker.code_predictor`（训练 15 级残差码）
- `model.talker.model.layers.0~3`（talker **最下面 4 层**）

实现：`--trainable_parameters model.bridge model.talker.code_predictor` + `--trainable_parameters_regex 'model\.talker\.model\.layers\.[0-3]\.'`。该正则与变体无关：a/b/c/d/e/f/g/h/i/j 导出后层均重编号为 `0..N-1`，`layers.0~3` 即各自切片中最靠近输入的 4 层，十变体通用。损失 = 主码 CE + `residual_loss_weight` × 残差 CE。

**残差损失开关与权重**：残差 CE 可开关、可配权重，优先级 env > config（默认 `enabled=true`、`weight=1.0`）：

- config 字段：`residual_loss_enabled` / `residual_loss_weight`
- 环境变量：`QWEN_OMNISPEAK_RESIDUAL_LOSS`（`true/false`）/ `QWEN_OMNISPEAK_RESIDUAL_LOSS_WEIGHT`（浮点）
- 训练脚本快捷变量：`RESIDUAL_LOSS=false` / `RESIDUAL_LOSS_WEIGHT=0.5`
- `weight<=0` 视为关闭；关闭时跳过 code predictor 前向以省显存/算力

这给出一条稳妥的课程化路径：可先 `RESIDUAL_LOSS=false` 只训主码（验证主码能否继续下探到 ~4 以下），再开启或逐步加大残差权重。

> 这一条对正文 5.6 / 7.2“code_predictor 首期默认冻结、Stage-1 仅训 bridge”的描述做了修订：残差 code_predictor 现在参与 Stage-1 训练。

### 10.8 同传演进预留：流式 attention mask 接口

`build_block_causal_mask` 新增可选参数 `codec_to_sem_visible`（`[codec_len]` 或 `[B, codec_len]`）：给出每个 codec 帧可见的**前缀语义长度**。

- `None`（默认）= 当前“codec 看全部语义”的离线行为，与原实现完全等价（Stage-1 使用）；
- 传入**单调非降**的可见长度 = 把 `codec→sem` 块变成 chunk-causal（read-write）流式 mask，用于“一边听一边说”的同传，无需改动其它任何逻辑。

这是为后续阶段埋好的扩展点：Stage-1 学“语义→codec 的映射”（what），后续阶段用该接口 + read-write 数据组织学“何时说”（when）。

### 10.9 ST + S2S 同步双输出

语音→文本翻译 = Thinker 文本 logits；语音→语音翻译 = Talker codec（主码 + 残差）。两者是**独立的 head**（Talker 消费 Thinker 的 hidden/text），天然支持同步双输出，与控制前缀 / 拼接范式正交。

### 10.10 导出 / 加载注意事项

- 本轮改动均为**代码逻辑 + 配置标量**，未改动任何权重张量，因此**无需重新跑加载 30B 权重的完整 export**。
- 但 `trust_remote_code` 加载用的是**导出目录内的 `.py` 副本**（由 `config.json` 的 `auto_map` 指向），所以更新 `modeling_*.py` / `configuration_*.py` 后必须同步到 a/b/c/d/e/f 对应导出目录，并清理 `HF_MODULES_CACHE` 缓存，否则会用到旧副本。
- 控制 token id 在加载旧 `config.json` 时由 config 类默认值自动填充，`config.json` 无需手改（除已更新的 `codec_pad_id`）。
- codec 磁盘缓存存的是**原始 codes**（控制前缀是缓存之后才拼接的），故本轮改动**不影响已有缓存的有效性**，无需清缓存重编码。
