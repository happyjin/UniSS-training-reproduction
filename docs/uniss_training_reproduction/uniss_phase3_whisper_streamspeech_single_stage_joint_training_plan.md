# UniSS 最佳 Phase3 基础上的 Whisper-StreamSpeech 单阶段联合多任务训练方案

> 方案日期：2026-08-05
>
> 目标：保留当前最佳 UniSS Phase3 的成功架构和能力，在此基础上移植 StreamSpeech 的联合多任务训练、CTC 对齐策略、NAR Text-to-Unit 与 multi-chunk 训练，将模型改造成 simultaneous speech-to-speech translation。
>
> 基础 checkpoint：`checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075`
>
> 约束：正式训练只有一个 stage、一个 optimizer、一条 checkpoint 序列；不使用 Emformer 替换 WhisperVQ；不覆盖现有 Phase1/2/3、Student v2/v3 或 StreamSpeech-inspired 历史实验。
>
> 对照论文：*StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning*，ACL 2024，[arXiv:2406.03049](https://arxiv.org/pdf/2406.03049)
> 对照代码：[ictnlp/StreamSpeech](https://github.com/ictnlp/StreamSpeech)

---

## 1. 最终决策摘要

本方案不是重新训练一个脱离 UniSS Phase3 的 StreamSpeech，也不是继续此前的 Stage02→03→03b→04→06→08 多阶段链路，而是：

> **以最佳 Phase3 的 WhisperVQ → GLM token → Qwen → BiCodec 架构为主干，在同一次端到端训练中加入 StreamSpeech 的四任务 objective、CTC-guided prefix policy、NAR BiCodec Unit CTC 和 multi-chunk。**

正式模型保持以下成功能力：

- 原 WhisperVQ/GLM 16,384 codebook source-speech 接口；
- Phase3 Qwen2.5-0.5B 的翻译、Quality/Performance 协议和文本生成能力；
- 原 180k 扩展词表与任务、语言、速度、GLM、BiCodec token；
- BiCodec semantic unit 与 32 个 `bicodec_global` 音色条件；
- 原 Phase3 checkpoint 的 offline S2ST 能力和独立 fallback 路径。

新增或改造的 simultaneous 模块：

1. WhisperVQ 不换架构，只增加可变 chunk-causal attention mask；
2. 在 WhisperVQ pre-VQ hidden 上增加 source ASR CTC；
3. 在相同 hidden 上增加 target NAR-S2TT CTC；
4. 用两个 CTC posterior 构造 `g(i)`，限制 Phase3 Qwen 第 `i` 个目标文本 token 只能看到允许的 source prefix；
5. 从 Phase3 Qwen 的目标翻译 hidden 并行预测 `target_bicodec`，实现 NAR BiCodec Unit CTC；
6. 每个 batch 随机选择 chunk size，使一个 checkpoint 同时支持多种延迟；
7. 推理时使用 WhisperVQ 累积前缀重编码作为正确性基线，并进一步实现相同 WhisperVQ 权重的 chunk cache，避免长期重复计算。

推荐主损失为：

\[
\boxed{
L_{total}=
1L_{BiCodec\text{-}CTC}
+8L_{AR\text{-}S2TT}^{g(i)}
+4L_{ASR\text{-}CTC}
+4L_{NAR\text{-}S2TT\text{-}CTC}
+0.5\mathbf{1}_{replay}L_{P3\text{-}old\text{-}exact}
}
\]

前四项严格采用 StreamSpeech 官方相对权重 `1/8/4/4`。训练中有 `20%` 的 microbatch 按原 Phase3 数据格式、token 顺序、Quality/Performance 协议和 causal CE 掩码执行 exact replay，并以 `0.5` 权重加入；这是保护现有 Phase3 offline 能力和 fallback 的 UniSS 扩展，不是原论文 loss。

最重要的设计原则是：

> **不能在每个 joint sample 上把原 Phase3 全序列 NLL 与 StreamSpeech 四项 loss 简单、等权相加。主样本将 Phase3 文本 CE 改造成 policy-conditioned AR-S2TT，将主要 semantic 监督改造成 NAR Unit CTC；另用独立抽样的 20% exact replay microbatch 防止原 Phase3 能力被新目标覆盖。**

---

## 2. 本方案严格依据的证据

### 2.1 StreamSpeech 论文的正式 objective

论文第 3.2 节定义：

\[
L=L_{S2UT}+L_{AR-S2TT}+L_{ASR}+L_{NAR-S2TT}.
\]

官方配置和 Appendix H 实际使用：

| Objective | 官方权重 |
|---|---:|
| primary speech-to-unit CTC / `L_S2UT` | 1.0 |
| AR target-text translation / `L_AR-S2TT` | 8.0 |
| source ASR CTC / `L_ASR` | 4.0 |
| target NAR-S2TT CTC / `L_NAR-S2TT` | 4.0 |

本地官方仓库快照中的证据：

- `researches/ctc_unity/models/streamspeech_model.py`；
- `researches/ctc_unity/criterions/speech_to_speech_ctc_asr_st_criterion.py`；
- `configs/fr-en/config_mtl_asr_st_ctcst.yaml`；
- `researches/ctc_unity/train_scripts/train.simul-s2st.sh`。

本地快照目录：

```text
/opt/dlami/nvme/jasonleeeli/research_sources/streamspeech/
```

### 2.2 官方 multi-chunk

论文写作：

\[
C\sim U(1,|X|),
\]

其中 `C=|X|` 表示 offline。

官方 criterion 的离散实现是：

```python
chunk_size = random.choice([8, 16, 24, 32, 99999])
```

每个 encoder feature 约 40 ms，对应：

```text
320 / 640 / 960 / 1280 ms / offline
```

### 2.3 当前审计报告的核心缺口

`uniss_emformer_stages_vs_streamspeech_original_training_audit.md` 已确认当前历史实验缺少：

- policy-conditioned `g(i)` AR training；
- NAR Text-to-Unit CTC；
- multi-chunk；
- 四任务在同一 optimizer update 中联合训练。

本方案直接补齐这四项，但按照本次约束不采用 Emformer，源前端继续使用 Phase3 原 WhisperVQ。

---

## 3. 当前最佳 Phase3 是什么

### 3.1 Checkpoint

Megatron checkpoint：

```text
checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075
```

已导出的 Hugging Face checkpoint：

```text
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
```

### 3.2 Phase3 Qwen 规格

| 项目 | 当前值 |
|---|---:|
| backbone | Qwen2.5-0.5B compatible |
| layers | 24 |
| hidden size | 896 |
| FFN hidden | 4864 |
| attention heads | 14 |
| KV heads | 2 |
| vocabulary | checkpoint padding 后 180,480；有效 UniSS token 约 180,407 |
| sequence length | 18,000 |
| historical micro batch | 2 |
| historical global batch | 128 |

### 3.3 WhisperVQ source tokenizer 规格

当前 `glm4_tokenizer` 配置：

| 项目 | 当前值 |
|---|---:|
| architecture | WhisperVQ encoder |
| hidden size | 1280 |
| configured encoder layers | 32 |
| 量化前实际加载层数 | 前 16 层 |
| attention heads | 20 |
| FFN hidden | 5120 |
| pooling position | layer 16 |
| pooling kernel | 4 |
| GLM codebook size | 16,384 |

Phase3 的 source speech 实际不是直接输入 waveform，而是：

```text
waveform
  → WhisperVQ
  → source_glm: [0, 16383]
  → 映射为 UniSS GLM special token
  → Phase3 Qwen
```

### 3.4 Phase3 当前训练目标

Phase3 使用全部 UniST 的 Quality 和 Performance 样本：

```text
Quality:
source speech tokens
→ transcription
→ translation
→ target BiCodec semantic tokens

Performance:
source speech tokens
→ translation
→ target BiCodec semantic tokens
```

所有输出本质上使用一个 next-token CE/NLL：

\[
L_{P3-old}=-\sum_t\log p(z_t\mid z_{<t},source\_glm,prompt).
\]

这个 loss 成功训练出了 offline Phase3，但不包含：

- 当前 source prefix 能否支持 WRITE 的显式对齐；
- target token 对 source prefix 的 `g(i)` 限制；
- NAR target-unit generation；
- 多 chunk size 鲁棒性。

---

## 4. 为什么不使用 Emformer

本方案明确不采用：

```text
WhisperVQ → Emformer Student → Phase3
```

原因不是 Emformer 结构不可行，而是本实验要回答更严格的问题：

> 在不替换最佳 Phase3 原语音表示架构的情况下，仅移植 StreamSpeech 的训练方法，能否把 Phase3 改造成 simultaneous S2ST？

因此源侧保持：

```text
原 WhisperVQ 权重
+ 原 GLM codebook
+ 原 Phase3 GLM token embedding
```

Student v2 只复用以下思想和代码：

- `chunk_right_attention_mask()`；
- chunk + bounded right context；
- exact-prefix 监督；
- stable-prefix append-only commit；
- future perturbation 因果验证；
- cache/full parity 验证。

不复用 Student v2 的 Causal Emformer 网络。

对应现有代码基础：

```text
training/simul_uniss/subsecond_v2/streaming_whispervq_teacher.py
web_demo/streaming_s2st_r2_v1/engine/prefix_frontend.py
uniss/streaming/stable_prefix.py
```

---

## 5. 单阶段目标架构

### 5.1 完整计算图

```text
source waveform X
  │
  ├─ Whisper feature extractor
  │
  └─ 原 WhisperVQ encoder Wθ
       │  每个 batch 随机使用一个 multi-chunk mask C
       │
       ├─ pre-VQ hidden H
       │    ├─ Source ASR CTC head ───────────────→ transcription A
       │    └─ Target NAR-S2TT CTC head ─────────→ translation Y
       │
       └─ 原 16,384 GLM codebook quantization
            │
            └─ source_glm / straight-through Phase3 embedding
                 │
                 └─ 最佳 Phase3 Qwen Qφ
                      │  使用 CTC 构造的 g(i) source-prefix mask
                      │
                      ├─ AR target text logits ───→ translation Y
                      │
                      └─ target text hidden Dtext
                           │
                           └─ 2L T2U encoder
                                │
                                └─ 上采样 r
                                     │
                                     └─ 2L causal NAR Unit CTC decoder
                                          │
                                          └─ BiCodec unit IDs U
                                               │
                                               └─ 冻结 BiCodec decoder
                                                    + bicodec_global
                                                    → target waveform S
```

### 5.2 仍然属于“原 Phase3”的部分

- WhisperVQ 网络结构与 codebook 不换；
- source GLM token 范围不换；
- Phase3 Qwen 24 层结构不换；
- Qwen embedding/LM head 和控制协议不换；
- BiCodec semantic vocabulary 不换；
- `bicodec_global` 的 32 token 音色条件不换；
- 原 Phase3 checkpoint 可以继续单独运行和复现。

### 5.3 新增的参数模块

| 新模块 | 建议结构 | 初始化 |
|---|---|---|
| ASR CTC heads | EN/ZH 各一个 linear CTC head | Xavier/random |
| NAR-S2TT CTC heads | EN/ZH 各一个 linear CTC head | Xavier/random |
| Phase3 source STE bridge | `1280 → 896` projection | 由 codebook/Qwen embedding regression 初始化 |
| T2U input projection | `896 → 512` | Xavier/random |
| T2U encoder | 2-layer causal Transformer, dim 512 | random |
| Unit CTC decoder | 2-layer causal Transformer, dim 512 | random |
| Unit projection | `512 → 8193` | random；8192 units + blank |

BiCodec waveform decoder保持冻结，不加入 optimizer。

### 5.4 Phase3 输出协议如何从旧 AR semantic 改成 two-pass

当前 Phase3 Quality/Performance target 通常是：

```text
translation text
→ <end_content>
→ <start_semantic>
→ AR target_bicodec tokens
→ <end_semantic>
```

新的 simultaneous 主路径改为：

```text
Pass 1：Phase3 Qwen只生成当前被 g(i) 支持的 translation text
        并在当前 text prefix上保留对应 hidden Dtext

Pass 2：T2U + Unit CTC从 Dtext并行生成对应 target_bicodec
```

因此主路径在目标文本到达当前可提交边界后直接调用 T2U，不等待 Qwen 自回归输出 `<start_semantic>` 和长 semantic 序列。原 Phase3 AR semantic 分支仍保留在参数和词表中，但只用于：

- offline compatibility regression；
- `L_P3-old-exact` replay；
- NAR Unit CTC失败时的显式 fallback；
- 与旧 Phase3 做严格 A/B。

这一步保留 Phase3 的翻译主干和原 checkpoint兼容性，同时把最耗时的长语音 unit 生成替换为 StreamSpeech 式 NAR second pass。

---

## 6. WhisperVQ 如何改成适合 simultaneous 的形式

### 6.1 不改变网络结构，只改变 attention 可见范围

现有 `StreamingWhisperVQTeacher` 已经验证可以给原 WhisperVQ 替换 attention mask：

```text
允许：全部合法历史 + 当前 chunk + 有限 right context
禁止：更远未来音频
```

其核心函数是：

```python
chunk_right_attention_mask(
    attention_mask,
    chunk_frames=...,
    right_context_frames=...,
)
```

正式训练应将固定 `160 ms + 80 ms` 扩展为 multi-chunk：

```text
C ∈ {320, 640, 960, 1280 ms, offline}
right_context = 80 ms
```

为了严格对齐官方首版，不在正式训练集合中加入 160 ms。160 ms 可作为训练后压力测试点。如果后续确认 320 ms 无法满足目标延迟，再单独做含 160 ms 的消融，但不与本次主实验混淆。

### 6.2 每个 batch 如何选择 chunk

每个 optimizer micro-batch 开始时：

```python
chunk_id = random.choice([320, 640, 960, 1280, INF])
```

然后在 data-parallel group 内广播相同 `chunk_id`，确保：

- 一个 batch 内 mask geometry 一致；
- 各 rank 日志可复现；
- loss/throughput 对应同一个 chunk 条件；
- TensorBoard 可以按 chunk 分桶统计。

五种 chunk 等概率时，offline batch 比例为 20%。这同时为 Phase3 提供 full-context 质量锚点。

### 6.3 `offline` 的准确含义

`C=offline` 不能只表示“最后一次才触发 WRITE”，而应让 WhisperVQ encoder attention 看到整条有效 utterance，与当前 offline Phase3 source tokenizer 条件尽量一致。

有限 `C` 时使用 chunk-causal mask；`C=offline` 时恢复原 WhisperVQ full/block attention mask。

### 6.4 训练是 mask-parallel，不等于推理已经有 cache

训练时可以在一整条音频上一次 forward，通过 mask 模拟每个 chunk 的合法上下文。这在数学上满足流式可见性，但仍是并行训练。

推理分两种实现：

#### 正确性基线：累积 prefix re-encode

```text
收到 0–320 ms
→ WhisperVQ 编码 prefix

收到 320–640 ms
→ WhisperVQ 重新编码 0–640 ms

收到 640–960 ms
→ WhisperVQ 重新编码 0–960 ms
```

优点：

- 完全复用原 WhisperVQ forward；
- 最容易验证训练/推理 mask 一致；
- 不会因 cache 实现 bug 改变结果。

缺点：

- 历史反复计算；
- 长音频计算量随时长增长；
- 属于 simultaneous I/O，但计算上是 pseudo-streaming。

#### 正式运行时：相同 WhisperVQ 权重的 chunk cache

不换模型结构，只给 Whisper encoder 实现状态缓存：

- conv1/conv2 保存边界 overlap；
- 每层 self-attention 保存已提交位置的 K/V；
- 当前 chunk 加 80 ms lookahead；
- lookahead hidden 暂不提交，在下一 tick 重新计算；
- absolute position embedding 使用全局 frame offset；
- pooling layer 保存不足一个 kernel 的残留帧；
- codebook 只量化新 committed hidden；
- cache/full masked forward 必须达到数值一致性 gate。

这种实现保持 WhisperVQ 参数和层结构不变，比替换为 Emformer更符合本实验目的。

---

## 7. WhisperVQ 离散量化与端到端梯度

### 7.1 问题

Phase3 当前读取 hard `source_glm` token。WhisperVQ 的 nearest-codebook 操作：

```text
pre-VQ hidden
→ argmin distance
→ integer GLM ID
```

`argmin` 不可微。如果直接把整数 token 喂给 Qwen：

- ASR/NAR CTC loss 可以更新 Whisper；
- Qwen AR-S2TT loss不能通过 hard token 回传到 Whisper。

这不满足原文“所有模块 joint optimized”的原则。

### 7.2 推荐的 straight-through Phase3 embedding

前向仍使用 Phase3 完全熟悉的 hard GLM token embedding：

```python
glm_id = nearest_codebook(pre_vq_hidden)
e_hard = qwen_embedding[GLM_OFFSET + glm_id]
e_cont = source_bridge(pre_vq_hidden)       # 1280 -> 896
e_source = e_cont + (e_hard - e_cont).detach()
```

其性质：

- forward：`e_source == e_hard`，与原 Phase3 source embedding 完全一致；
- backward：梯度通过 `e_cont` 回到 bridge 和 WhisperVQ；
- inference：仍可以只输出 hard GLM ID，接口不变。

这是在不更改 Phase3 Qwen 架构的情况下完成端到端 joint training 的关键。

### 7.3 Codebook 处理

第一版建议：

- 16,384 GLM codebook 冻结；
- Qwen GLM token embedding 可训练，但使用很低 LR；
- 不做 codebook EMA 更新；
- 记录 active-code 数、code entropy、与原 source_glm agreement；
- 禁止 codebook collapse。

---

## 8. 文本 tokenizer 必须统一

### 8.1 为什么现有独立 8k CTC tokenizer 有风险

StreamSpeech 官方的 `target_unigram` AR decoder 与 `ctc_target_unigram` 使用相同 target tokenizer，因此第 `i` 个 CTC target token与第 `i` 个 AR target token天然一致。

当前历史 Stage01 使用独立 EN/ZH 8k SentencePiece，而最佳 Phase3 Qwen 使用 Qwen tokenizer。如果直接组合：

```text
NAR-S2TT CTC 的第 i 个 token
≠
Qwen 翻译的第 i 个 token
```

此时 `g(i)` 无法直接作用到 Qwen 目标位置，必须增加脆弱的字符边界映射。

### 8.2 推荐方案：Qwen text token 的 compact CTC vocabulary

对 full198 `transcription` 和 `translation` 使用 Phase3 同一 Qwen tokenizer，然后建立 compact ID 映射：

```text
Qwen text token ID
↔ compact CTC class ID
```

只收录 EN/ZH 文本实际使用的 base-text token，排除：

- GLM token；
- BiCodec token；
- task/control token；
- padding 和非文本特殊 token。

这样：

- CTC head 不需要输出完整 180k vocabulary；
- NAR-S2TT CTC token position 与 Qwen AR token position一致；
- `g(i)` 可以直接限制 Phase3 第 `i` 个翻译 token；
- source/target CTC 仍可使用语言独立的 heads。

建议保留四个 CTC heads：

```text
asr_eng
asr_cmn
nar_s2tt_eng
nar_s2tt_cmn
```

---

## 9. CTC-guided `g(i)` 如何接入 Phase3 Qwen

### 9.1 两个 CTC posterior

从相同 WhisperVQ hidden `H` 得到：

\[
P_{ASR}(a\mid H),\qquad P_{NAR}(y\mid H).
\]

计算当前 source prefix `X≤j` 支持的期望 token 数：

\[
N_j=\sum_{m=1}^{j}
\left[
1-p(blank_m)-\sum_vp(v_m)p(v_{m-1})
\right].
\]

分别得到：

- `N_asr(j)`：已经识别出多少 source token；
- `N_nar(j)`：当前音频支持多少 target token。

### 9.2 Source boundary

目标 token `y_i` 可见的最早 source boundary：

\[
g(i)=
\arg\min_{j:N_{asr}(j-1)<N_{asr}(j)}
\{j\mid N_{nar}(j)\ge i\}.
\]

含义：

1. source CTC 刚刚确认了新 source 内容；
2. target CTC 判断现有内容足以支持第 `i` 个目标 token；
3. 两个条件同时成立时，Phase3 才能训练/推理该 token。

### 9.3 与官方 GitHub 一致的梯度边界

官方 `streamspeech_model.py` 对 CTC count 路径使用 `.detach()`，随后通过 round/threshold 构造 hard mask。

本方案同样采用：

```python
with torch.no_grad():
    g = build_g_from_ctc_posteriors(asr_probs, target_probs)
```

因此：

- `g(i)` 不做 policy gradient；
- CTC heads 通过各自 CTC loss 获得梯度；
- Qwen 通过 masked AR CE 获得梯度；
- 避免离散 mask 的不稳定伪梯度。

### 9.4 Decoder-only Qwen 的 attention mask

Phase3 是 decoder-only Qwen，不是原文 encoder-decoder。因此需要构造 block attention mask。

输入布局保持原 Phase3 protocol：

```text
[task/mode/lang]
[bicodec_global]
[source GLM embeddings H1 ... HT]
[WRITE/control]
[target text y1 ... yM]
```

对目标 query `y_i`：

```text
允许：system/control、bicodec_global、H≤g(i)、y<i
禁止：H>g(i)、y>i
```

即使 `H>g(i)` 在物理序列中位于 `y_i` 前面，也必须显式屏蔽。

Megatron实现建议：

- 使用 `AttnMaskType.arbitrary`；
- mask shape 为 `[batch, 1, query, key]`；
- 优先验证 Transformer Engine fused arbitrary-mask 路径；
- 如果当前 TE 版本不支持正确 fused mask，正确性版本使用 unfused attention；
- 不允许为了 FlashAttention 吞吐量而退化成“所有 target token看完整 source”。

`g(i)` mask 是本方案最重要的正确性条件之一。

---

## 10. NAR BiCodec Unit CTC

### 10.1 输入与目标

输入不是 source speech hidden，而是 Phase3 Qwen 已生成目标翻译的 hidden：

```text
Dtext = hidden states at target translation positions
```

目标为：

```text
U = target_bicodec
```

`bicodec_global` 不作为 CTC target。它继续作为冻结 BiCodec decoder 的说话人/全局条件。

### 10.2 推荐结构

参考官方 StreamSpeech：

```text
Dtext: 896
→ linear projection: 896 → 512
→ 2-layer causal T2U Transformer encoder
→ deterministic upsampling r
→ 2-layer causal Unit CTC Transformer decoder
→ linear: 512 → 8193
```

其中：

```text
8192 = BiCodec semantic unit vocabulary
+1 = CTC blank
```

### 10.3 因果约束

第 `k` 个 upsampled unit position只能依赖：

\[
D^{text}_{\le \lceil k/r\rceil}.
\]

这保证当前只生成了目标文本 prefix 时，T2U 不能偷看未来目标文本。

### 10.4 UniST 上采样率不能直接使用 25

对现有 Phase3 processed JSONL 做了约 99k 条按 shard 分层的 preliminary audit：

| target language | `unit/text` p50 | raw ratio p95 | CTC minimum-path ratio p95 |
|---|---:|---:|---:|
| Chinese | 约 16.70 | 约 25.54 | 约 25.93 |
| English | 约 16.48 | 约 29.40 | 约 30.33 |

`CTC minimum-path` 已考虑相邻重复 unit 之间需要 blank 的额外位置。

StreamSpeech 原文建议：

```text
r ≈ unit/text length ratio × 2–3
```

按当前 p50 估计，候选范围约为：

```text
r ≈ 40–50
```

第一推荐候选为：

```text
r = 48
```

正式启动前必须全量扫描 full198，并在 `{40, 48, 56, 64}` 中选择满足以下条件的最小值：

```text
CTC infeasible rate < 0.1%
```

不可使用 `zero_infinity=True` 静默掩盖大量无效样本。所有无效样本必须计数、按方向统计并写入数据审计报告。

---

## 11. Loss 应如何设计

### 11.1 所有 loss 必须先独立归一化

直接把 sum-reduction CTC 与 token-mean CE 相加会导致权重没有可比意义。

建议：

- `L_ASR-CTC`：按有效 source text token 数归一化；
- `L_NAR-S2TT-CTC`：按有效 target text token 数归一化；
- `L_AR-S2TT`：按 target translation token 数归一化；
- `L_BiCodec-CTC`：按有效 target unit 数归一化；
- DP all-reduce 前记录 numerator 和 denominator，按全局有效 token 计算 mean。

### 11.2 四个原文主 loss

#### Source ASR CTC

\[
L_{ASR}=CTC(Head_{src}(H),A).
\]

作用：

- 学 source speech → source text 对齐；
- 判断 source 是否出现新的稳定内容；
- 为 READ/WRITE policy 提供 source boundary。

推荐权重：

```text
4.0
```

#### Target NAR-S2TT CTC

\[
L_{NAR-S2TT}=CTC(Head_{tgt}(H),Y).
\]

作用：

- 学 source speech → target token 单调软对齐；
- 判断当前音频 prefix 能支持几个目标 token；
- 主要服务 policy，不单独承担最终流畅翻译。

推荐权重：

```text
4.0
```

#### Policy-conditioned Phase3 AR-S2TT

\[
L_{AR-S2TT}^{g}
=-
\frac{1}{|Y|}
\sum_i
\log p_{P3}(y_i\mid X_{\le g(i)},Y_{<i}).
\]

这里的 decoder 就是最佳 Phase3 Qwen，不新增另一个独立 4-layer translation decoder。

推荐：

- 使用 Phase3 当前 target text token；
- label smoothing `0.1`，与官方 StreamSpeech 对齐；
- 只对真实翻译文本 span 计算；
- 不把 semantic span混入该 loss。

推荐权重：

```text
8.0
```

#### NAR BiCodec Unit CTC

\[
L_{BiCodec-CTC}
=CTC(UnitHead(T2U(D^{text})),target\_bicodec).
\]

推荐权重：

```text
1.0
```

### 11.3 Phase3 原 loss 是否一起加

结论：

> **不应该把完整旧 `L_P3-old` 再以 1.0 或更高权重直接加到四项主 loss 上。**

原因：

1. 旧 Phase3 translation CE 与 `8L_AR-S2TT` 重复计算；
2. 旧 Phase3 semantic AR CE 与新的 NAR Unit CTC 同时争夺主要生成路径；
3. 旧 full-context Phase3 CE 会鼓励 Qwen 看完整 source，抵消 `g(i)` prefix training；
4. Phase3 semantic 序列很长，未正确归一化时会重新主导总梯度；
5. 最终又会回到“offline teacher forcing 训练、prefix streaming 推理”的 mismatch。

正确做法是拆解旧 Phase3 loss：

| 旧 Phase3 supervision | 新训练中的去向 |
|---|---|
| source transcription tokens | `4 L_ASR-CTC` |
| target translation tokens | `8 L_AR-S2TT^g` |
| target semantic tokens | `1 L_BiCodec-CTC` |
| source→target token capacity/alignment | 新增 `4 L_NAR-S2TT-CTC` |
| task/mode/end/control protocol | 20% microbatch 的 `L_P3-old-exact` |
| 旧 AR semantic fallback | 20% microbatch 的 `L_P3-old-exact`，同时保留独立原 checkpoint fallback |

### 11.4 推荐 Phase3 exact replay

为了防止最佳 Phase3 的控制协议、Quality/Performance 行为和旧 AR semantic fallback 遗忘，训练采样器令 `20%` microbatch直接读取原 Phase3 packed/indexed 数据，并完整复现原训练路径：

\[
L_{P3-old-exact}
=-\frac{1}{|M_{P3}|}
\sum_{t\in M_{P3}}
\log p_\theta(z_t\mid z_{<t},source\_glm,prompt).
\]

其中 `source_glm`、prompt、task/mode token、Quality/Performance 分支、文本/semantic token顺序以及 loss mask `M_P3` 都与成功的 Phase3 训练脚本保持完全一致。它不是近似的 control-token anchor，而是真正的旧任务 replay。

推荐采样比例和权重：

```text
p_replay = 0.20
lambda_p3_replay = 0.50
```

replay microbatch仍在同一个 dataloader、同一个 optimizer、同一条 checkpoint序列中完成，因此不是额外训练 stage。以采样概率计，其平均标称贡献约为 `0.20 × 0.50 = 0.10`；同时每项 loss 都必须先按自身有效 token 数归一化，防止较长的 semantic 序列重新主导梯度。

默认实现建议在 replay microbatch 上只计算 `L_P3-old-exact`，在 joint microbatch 上计算 StreamSpeech 四项主 loss；这样不会在同一个样本上重复计算 translation/semantic 监督，也不会让 full-context CE直接抵消 `g(i)`。如实现采用混合样本 global batch，则必须分别累计 numerator/denominator，再按 sample type施加 mask。

如果要做严格 paper-objective 对照，可设置：

```text
p_replay = 0
lambda_p3_replay = 0
```

但本项目正式 Phase3-preserving run 推荐保留 `20% / 0.50`，并将纯 `1/8/4/4` 作为消融实验。

### 11.5 最终推荐公式

\[
\boxed{
L_{total}=
1.0\bar L_{BiCodec-CTC}
+8.0\bar L_{AR-S2TT}^{g(i)}
+4.0\bar L_{ASR-CTC}
+4.0\bar L_{NAR-S2TT-CTC}
+0.50\mathbf{1}[sample=replay]\bar L_{P3-old-exact}
}
\]

所有带横线的 loss 都是按各自有效 target 数量归一化后的全局 mean。`sample=replay` 由确定性、可恢复的分布式采样器产生，目标比例为 20%。

---

## 12. 哪些参数训练，哪些参数冻结

正式 run 从 step 1 开始所有四项主任务都激活，不做先冻后解冻的阶段切换。

| 模块 | 初始化 | 是否训练 | 推荐 peak LR |
|---|---|:---:|---:|
| Whisper feature conv | 原 WhisperVQ | 是，低 LR | `5e-6` |
| WhisperVQ 前 8 层 | 原 WhisperVQ | 是，低 LR | `5e-6` |
| WhisperVQ 后 8 层 | 原 WhisperVQ | 是 | `1e-5` |
| GLM codebook | 原 16,384 codebook | 否 | 0 |
| source STE bridge | 新建 | 是 | `5e-5` |
| ASR/NAR CTC heads | 新建 | 是 | `1e-4` |
| Phase3 Qwen embeddings | iter9075 | 是，极低 LR | `1e-6` |
| Phase3 Qwen 24 layers | iter9075 | 是，低 LR | `2e-6` |
| Phase3 LM head | iter9075 | 是，极低 LR | `1e-6` |
| T2U encoder | 新建 | 是 | `1e-4` |
| Unit CTC decoder/head | 新建 | 是 | `1e-4` |
| BiCodec decoder | 原 checkpoint | 否 | 0 |

这种配置仍然是 joint training：所有可训练模块从第一步同时进入 optimizer，只是 pretrained 模块使用更小的 LR multiplier。

不建议第一版使用 LoRA 替代 Qwen 全参数低 LR 更新，因为：

- `g(i)` 改变的是 Qwen 的主要条件分布；
- full parameter 的小 LR 更接近原文 all-module joint optimization；
- Qwen 0.5B 在 8×H200 上可承受全参数训练。

---

## 13. 数据准备

### 13.1 使用 full198，不重新制造标注

正式模型使用与最佳 Phase3 相同 full198 范围的 UniST train，dev/test 严格排除训练。

每条 joint record 需要：

```json
{
  "id": "...",
  "source_audio": "...",
  "source_glm": [/* 原 Phase3 teacher/fallback */],
  "transcription": "...",
  "translation": "...",
  "target_bicodec": [/* 0..8191 */],
  "bicodec_global": [/* exactly 32 */],
  "src_lang": "eng|cmn",
  "tgt_lang": "cmn|eng"
}
```

当前公开 UniST 主要提供 tokenized fields。源 waveform 使用已经按现有流程由：

```text
source_bicodec + bicodec_global
```

重建的 16 kHz source audio sidecar，不修改原 parquet。

### 13.2 一条数据同时计算四项主 loss

与原 StreamSpeech 一致，一条记录同时提供：

| 任务 | 输入 | Target |
|---|---|---|
| ASR CTC | Whisper hidden | `transcription` |
| NAR-S2TT CTC | Whisper hidden | `translation` |
| AR-S2TT | Whisper/GLM prefix + Qwen history | `translation` |
| S2UT Unit CTC | Qwen translation hidden | `target_bicodec` |

不是把一条 raw record复制成四个彼此独立的训练 task，也不是先训练某个 task 再训练下一个 task。

### 13.3 方向平衡

每个 global batch 建议严格：

```text
EN→ZH : ZH→EN = 1 : 1
```

不能依赖文件顺序或 rank-local shard 的自然分布。应使用全局 deterministic shuffle 和 direction-balanced sampler。

### 13.4 不继续复用旧 packed Phase3 JSONL 作为唯一数据输入

现有 `packed_train.jsonl` 适合纯 Qwen next-token SFT，但 joint model还需要：

- waveform/features；
- CTC label lengths；
- target unit lengths；
- multi-chunk metadata；
- Qwen text position与 CTC position映射。

因此需要新建 joint indexed manifest。旧 packed Phase3 数据只用于 `P3-old-exact` replay 或回归验证。

---

## 14. 基于 Megatron 的实现方式

### 14.1 仍然使用同一个 Megatron 框架

保持当前：

```text
third_party/Megatron-LM
torchrun
Megatron distributed checkpoint
TensorBoard
bf16
8 GPU
```

但不能继续直接使用薄封装 `training/pretrain_uniss_megatron.py`，因为它只调用标准 GPT next-token loss。

需要新增隔离入口，例如：

```text
training/phase3_whisper_streamspeech_joint/
├── dataset.py
├── tokenizer_maps.py
├── whisper_multichunk.py
├── ctc_heads.py
├── policy_mask.py
├── phase3_ste_bridge.py
├── nar_bicodec_ctc.py
├── model.py
├── losses.py
├── pretrain_joint_megatron.py
└── tests/
```

实验脚本放在：

```text
experiments/uniss_phase3_whisper_streamspeech_joint_v1/
├── README.md
├── configs/
├── data/
├── scripts/
├── evaluation/
└── reports/
```

不修改或覆盖：

```text
training/pretrain_uniss_megatron.py
scripts/train_phase3_qwen0p5b.sh
checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/
experiments/uniss_streamspeech_ctc_v1/
```

### 14.2 Compound model

新 Megatron `model_provider` 返回一个 compound module：

```python
Phase3WhisperStreamSpeechJointModel(
    whisper_vq,
    qwen_phase3,
    asr_ctc_heads,
    nar_s2tt_ctc_heads,
    source_ste_bridge,
    t2u_encoder,
    unit_ctc_decoder,
)
```

第一版并行策略建议：

```text
data parallel = 8
tensor parallel = 1
pipeline parallel = 1
```

理由：

- Qwen 0.5B + WhisperVQ + T2U 可以放入 H200；
- arbitrary `g(i)` mask 与 compound loss 在 TP/PP=1 最容易保证正确；
- 保持与成功 Phase3 类似的 8 rank data-parallel 行为；
- 等 correctness gate 通过后再评估 TP2/DP4。

### 14.3 Megatron forward/loss contract

每次 forward 返回：

```python
{
    "asr_ctc_numerator": ...,
    "asr_ctc_tokens": ...,
    "nar_ctc_numerator": ...,
    "nar_ctc_tokens": ...,
    "ar_ce_numerator": ...,
    "ar_tokens": ...,
    "unit_ctc_numerator": ...,
    "unit_tokens": ...,
    "p3_replay_numerator": ...,
    "p3_replay_tokens": ...,
    "is_p3_replay": ...,
    "metrics": {...},
}
```

loss function先跨 data-parallel ranks 汇总 numerator/denominator，再对 joint/replay 样本分别应用 `1/8/4/4` 与 `0.5` 权重。

### 14.4 Checkpoint loading

新 checkpoint namespace 必须独立，例如：

```text
checkpoints/uniss_phase3_whisper_streamspeech_joint_v1/
```

初始化时分别加载：

1. Phase3 Qwen：`iter_0009075`；
2. WhisperVQ：`pretrained_models/UniSS/glm4_tokenizer`；
3. BiCodec：只读加载，不写入 optimizer state；
4. 新 heads：随机初始化。

新 checkpoint 保存 compound state、optimizer、RNG、data cursor、chunk sampler state和 tokenizer map hash。

---

## 15. 推荐训练超参数

### 15.1 Batch

首选延续成功 Phase3 吞吐配置：

```text
GPU = 8
micro batch = 2 utterances/GPU
global batch = 128 utterances
gradient accumulation = 128 / (8×2) = 8
```

这里的一个样本是一条 joint utterance，不是一个 18k packed block中的多个不相关任务。

如果 `g(i)` arbitrary mask + Whisper 导致显存不足：

```text
micro batch = 1
global batch = 128
gradient accumulation = 16
```

不能为了提高显存利用率把同一条长音频切成互相泄漏未来的独立样本。

### 15.2 Optimizer 与 scheduler

建议：

```text
optimizer = AdamW
beta1 = 0.9
beta2 = 0.98
eps = 1e-8
clip_grad = 0.5
precision = bf16
new-module dropout = 0.1
scheduler = inverse square root
warmup updates = 4000
```

使用 `base_lr=1e-4` 和 parameter-group LR multiplier：

| group | multiplier | effective peak LR |
|---|---:|---:|
| new CTC/T2U | 1.0 | `1e-4` |
| STE bridge | 0.5 | `5e-5` |
| Whisper top half | 0.1 | `1e-5` |
| Whisper bottom half/conv | 0.05 | `5e-6` |
| Qwen blocks | 0.02 | `2e-6` |
| Qwen embedding/LM head | 0.01 | `1e-6` |

这一设计同时考虑：

- 官方 StreamSpeech 对新模块使用较高 LR；
- Phase3 checkpoint 已经成功，不能用官方从零训练的 `1e-3` 直接更新 Qwen；
- Whisper 需要适应 chunk mask，但不能快速遗忘原 GLM codebook geometry。

### 15.3 Weight decay

建议：

```text
Qwen matrix weights: 0.1
Whisper/T2U/head matrix weights: 0.01
norm/bias/embedding/codebook: 0
```

### 15.4 训练长度

先构建 full198 去重后的 joint record 数 `N`，正式固定训练：

\[
updates=2\times\left\lceil\frac{N}{128}\right\rceil.
\]

也就是两个完整 global-shuffle passes。训练阶段只有一个，第二个 pass 不是新 stage，也不重新加载 checkpoint或改变 loss。

保存与验证：

```text
log interval = 10
save interval = 250 or 500
validation interval = 250 or 500
```

---

## 16. 单一正式训练流程

### 16.1 Preflight 不属于训练 stage

正式启动前只做无状态检查：

1. 验证 Phase3 iter9075 可以精确加载；
2. 验证 WhisperVQ full-context token 与原 tokenizer一致；
3. 验证每种 chunk mask 都不访问禁止未来；
4. 建立 Qwen compact CTC vocabulary；
5. 全量统计 `target_bicodec/text` 和 CTC feasibility；
6. 固定 `r`；
7. 验证 hard GLM forward 与原 Phase3 embedding一致；
8. 验证 STE backward 能到达 Whisper top layer；
9. 验证 `g(i)` mask确实屏蔽 source future；
10. 在 8 GPU 上跑 1–10 update smoke，确认所有 loss finite。

这些检查不产生可用于正式续训的模型 checkpoint，因此不构成多阶段训练。

### 16.2 正式 step 1 开始

从第一步开始：

```text
随机 multi-chunk
+ ASR CTC
+ NAR-S2TT CTC
+ g(i)-conditioned Phase3 AR translation
+ NAR BiCodec Unit CTC
+ offline-only small Phase3 anchor
```

全部同时计算并一次 backward：

```python
optimizer.zero_grad()
loss = weighted_joint_loss(batch)
loss.backward()
clip_grad_norm_(...)
optimizer.step()
scheduler.step()
```

训练过程中不做：

- Stage A/B/C checkpoint handoff；
- 先冻 Whisper 后解冻；
- 先训 CTC 后训 Qwen；
- 先固定 320 ms 后换 640 ms；
- 训练到某个 iteration 再更换 objective。

### 16.3 每步 TensorBoard

必须分别记录，不能只看 total loss：

```text
loss/total
loss/asr_ctc_eng
loss/asr_ctc_cmn
loss/nar_s2tt_ctc_eng
loss/nar_s2tt_ctc_cmn
loss/ar_s2tt_policy_eng
loss/ar_s2tt_policy_cmn
loss/bicodec_unit_ctc_eng
loss/bicodec_unit_ctc_cmn
loss/p3_replay_exact
sampler/p3_replay_ratio

ctc/asr_blank_rate
ctc/target_blank_rate
ctc/asr_expected_count
ctc/target_expected_count
ctc/infeasible_samples

policy/mean_g_ratio
policy/first_supported_token_ms
policy/source_event_coverage

whisper/glm_code_entropy
whisper/glm_teacher_agreement
whisper/chunk_id

grad/whisper
grad/qwen
grad/ctc_heads
grad/t2u

throughput/samples_per_second
throughput/audio_seconds_per_second
memory/allocated
power/gpu_watts
```

并按：

```text
320 / 640 / 960 / 1280 / offline
```

分别聚合 validation。

---

## 17. Checkpoint 选择不能只看 total loss

每个候选 checkpoint需要统一运行：

### 17.1 Text endpoint

- source ASR：WER/CER；
- target NAR CTC：unigram precision/recall、CTC edit rate；
- AR translation：BLEU、chrF；
- prefix-conditioned translation BLEU；
- full-context translation BLEU。

### 17.2 Unit/audio endpoint

- BiCodec unit error rate；
- unit BLEU/edit distance；
- Speech-BLEU；
- BLASER/AutoPCP（链路可用时）；
- speaker similarity/SLC；
- UTMOS；
- semantic collapse/repetition rate；
- BiCodec decode failure rate。

### 17.3 Streaming

- AL；
- AP；
- DAL；
- LAAL；
- ATD；
- StartOffset；
- EndOffset；
- computation-aware `_CA` versions；
- NumChunks；
- Discontinuity Sum/Ave/Num；
- RTF；
- first text WRITE；
- first audible audio；
- fallback/reject/rollback/conflict rate。

### 17.4 Phase3 保真 gate

推荐最低 gate：

```text
offline Phase3 text BLEU drop <= 1.0
offline Speech-BLEU drop <= 1.0
两方向均不得单边崩坏
CTC infeasible rate < 0.1%
NaN/skipped iteration = 0
GLM code collapse = 0
BiCodec decode failure < 0.1%
```

此外 streaming checkpoint必须在至少一个有限 chunk 点形成相对原 offline Phase3 可接受的质量—延迟 Pareto，不能只凭 first WRITE 很早选模。

---

## 18. 正式 simultaneous 推理流程

### 18.1 初始化

```text
load joint checkpoint
load frozen GLM codebook
load frozen BiCodec decoder
initialize Whisper chunk state
initialize ASR/target CTC state
initialize Phase3 Qwen KV cache
initialize T2U Unit CTC cache
initialize audio crossfade/playback buffer
```

### 18.2 每收到一个 source chunk

1. WhisperVQ 用当前 chunk、历史 cache 和 80 ms lookahead生成新的 committed hidden；
2. 量化为新的 hard source GLM token；
3. source ASR CTC 更新 source count；
4. target NAR-S2TT CTC 更新 target count；
5. 如果 source count 出现新事件且 target count 支持更多 token，则 WRITE；
6. Phase3 Qwen 从 KV cache继续生成新的目标文本 token；
7. T2U NAR CTC 对新增目标文本 hidden并行生成对应 BiCodec units；
8. CTC collapse 后保留未提交的新 unit；
9. 将 semantic units 与固定 `bicodec_global` 送入 BiCodec decoder；
10. 输出新音频 chunk并记录真实播放时间。

否则 READ 下一个 source chunk。

### 18.3 句尾 flush

源音频结束后：

- flush Whisper remaining lookahead；
- CTC target count允许完成剩余文本；
- Qwen 生成 `<end_content>`；
- T2U 生成剩余 units；
- BiCodec 解码最后音频；
- 不允许静默丢失句尾。

### 18.4 Offline 模式

同一个 checkpoint设置：

```text
C = offline
```

即可运行 full-context S2ST，用来与当前 Phase3 和论文 offline 表格对比，不另训 offline checkpoint。

---

## 19. 关键风险与修复顺序

### 19.1 Qwen token 与 CTC token 不一致

风险：`g(i)` 错位，模型在错误 source boundary 生成目标词。

修复：三个文本任务统一使用 Qwen text tokenization，CTC 使用 compact remap。

### 19.2 Whisper chunk训练但推理仍偷看未来

风险：mask frame单位、pooling前后比例或 right-context commit boundary错误。

修复：future perturbation必须为 0；每个 committed output只允许使用规定 lookahead。

### 19.3 Hard GLM token阻断梯度

风险：Qwen loss只更新 Qwen，不更新 Whisper。

修复：使用 forward-hard/backward-continuous STE source embedding，并单测 gradient reachability。

### 19.4 Arbitrary mask退化成 full source

风险：某个 fused attention backend忽略自定义 mask，训练再次变成 offline。

修复：构造未来 source perturbation；如果 `y_i` logits变化则 mask gate失败。正确性优先于 fused throughput。

### 19.5 Unit CTC 被无效长度静默置零

风险：`r` 太小，`zero_infinity=True` 导致大量样本没有 Unit loss。

修复：全量 feasibility audit，TensorBoard记录 invalid rate，正式目标 `<0.1%`。

### 19.6 Phase3 灾难性遗忘

风险：new heads 高 LR 梯度破坏 Qwen offline 能力。

修复：Qwen LR `1–2e-6`、offline multi-chunk比例20%、20% exact Phase3 replay（权重 `0.5`）、双向 offline gate。

### 19.7 伪流式计算随时间增长

风险：prefix re-encode 在5分钟音频上反复计算历史。

修复：先以 prefix re-encode验证结果，再给同一 WhisperVQ实现 K/V cache和 conv/pooling overlap；以 cache/full parity作为上线条件。

### 19.8 Speaker similarity下降

风险：NAR Unit CTC只学 semantic units，忽略音色。

修复：`bicodec_global` 始终从 source speaker条件传给 BiCodec decoder；第一版不将其塞入普通 CTC target。

---

## 20. 与此前多阶段路线的区别

| 项目 | 历史 Stage 路线 | 本方案 |
|---|---|---|
| source frontend | Emformer Student | 原 Phase3 WhisperVQ |
| training topology | 多阶段冻结/解冻/拼 checkpoint | 一个 stage，一次联合训练 |
| AR translation | 独立小 decoder或 offline Phase3 NLL | 最佳 Phase3 Qwen + `g(i)` mask |
| CTC policy | 后训练/推理阶段接入 | 训练时直接约束 Phase3 AR |
| Unit generation | Phase3 AR semantic | NAR BiCodec Unit CTC主路径 |
| multi-chunk | 无 | 每 batch 随机采样 |
| Qwen | 大多冻结或后续 LoRA | 从 iter9075 初始化，低 LR joint update |
| BiCodec | 增量解码后端 | 同样冻结，但接收 NAR CTC units |
| checkpoint | 多目录 handoff | 单一 compound checkpoint |

---

## 21. 建议的实现顺序

以下是代码开发顺序，不是训练 stages：

1. 新建隔离实验目录和 compound checkpoint schema；
2. 建立 full198 joint indexed manifest；
3. 建立 Qwen compact CTC token maps；
4. 将现有 WhisperVQ bounded mask扩展为 multi-chunk；
5. 实现四个 CTC text heads；
6. 实现 `g(i)` 和 decoder-only block mask；
7. 实现 hard-forward STE source embedding；
8. 实现 2L T2U + 2L Unit CTC；
9. 实现四项 joint normalized loss与原 Phase3 exact replay loss；
10. 接入 Megatron DP、checkpoint和 TensorBoard；
11. 做 1–10 update smoke；
12. 从最佳 Phase3 + 原 WhisperVQ 初始化一个正式 single-stage full198 run；
13. 使用同一 checkpoint跑 multi-chunk dev/test；
14. 在 prefix re-encode结果正确后实现 Whisper cache；
15. 接入在线 Gradio 前必须通过 cache/full parity和正式质量—延迟 gate。

---

## 22. 最终回答

### 是否应该把 Phase3 loss 和 StreamSpeech loss 一起训练？

应该联合训练，但不是简单相加完整旧 Phase3 loss。

推荐做法是：

```text
Phase3 translation CE
→ 改造成 8× policy-conditioned AR-S2TT

Phase3 semantic AR supervision
→ 主路径改造成 1× NAR BiCodec Unit CTC

Phase3 transcription supervision
→ 改造成 4× source ASR CTC

新增 source→target alignment
→ 4× target NAR-S2TT CTC

旧 Phase3 完整 protocol/fallback
→ 20% microbatch exact replay，权重 0.5
```

### 是否能模仿原文 multi-chunk？

可以，而且应该移植。WhisperVQ 不换架构，只在每个 batch使用：

```text
320 / 640 / 960 / 1280 ms / offline
```

随机 chunk-causal attention mask。一个 checkpoint即可在推理时切换延迟。

### Whisper 是否是真流式？

- 只做累积 prefix re-encode：输出是 simultaneous，但计算是 pseudo-streaming；
- 增加相同 WhisperVQ 权重的 layer K/V cache、conv overlap和 pooling state后：可以成为有界增量计算；
- 两者使用相同模型和 checkpoint，cache版本必须与 masked full forward数值一致。

### 最终推荐

> 以 full198 Phase3 iter9075 作为 Qwen 主干，以原 WhisperVQ 作为唯一 source frontend，不引入 Emformer；新增 CTC alignment heads、`g(i)` attention mask和 NAR BiCodec Unit CTC，用 `1/8/4/4` joint objective 加 `20%` 原 Phase3 exact replay（权重 `0.5`），在 Megatron 中进行一次连续的 full198 joint multi-task training。这是最符合“保留 Phase3 成功基础，同时移植 StreamSpeech 原始训练方法”的方案。
