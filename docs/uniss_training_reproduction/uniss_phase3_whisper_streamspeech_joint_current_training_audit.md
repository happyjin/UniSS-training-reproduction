# UniSS Phase3 + Whisper-StreamSpeech 单阶段联合训练当前状态与实现审计

> 审计时间：2026-08-06 UTC  
> 当前代码提交：`87bdcd2`  
> 实验名称：`uniss_phase3_whisper_streamspeech_joint_v1`  
> 目的：回答当前是否已经在训练、是否基于最佳 Phase3、与原 Phase3 和官方 StreamSpeech 的差异、loss、Whisper 流式改造、checkpoint 初始化和可训练参数。

## 1. 结论先行

### 1.1 当前是不是在训练

**当前还没有进入模型参数训练。**

当前 8 张 GPU 执行的是 full198 的 **Stage-A source audio reconstruction**：把 UniST 中 source BiCodec 表示恢复成 WhisperVQ 联合训练所需的 source waveform。该过程只有 BiCodec 前向解码和数据写盘，没有 optimizer、backward 或参数更新，因此不能把当前 GPU 占用称为 Megatron 训练。

审计时状态：

| 项目 | 状态 |
|---|---|
| Stage-A 完成度 | `82/198` shards |
| 活跃 worker | 16，2 worker/GPU |
| Stage-A 报错 | 未发现 |
| full198 replay offset index | 已完成，`1,161,587` records |
| `STAGE_A_SOURCE_COMPLETE.json` | 尚未生成 |
| full198 joint manifest | 尚未生成 |
| 正式 Megatron 训练 | 尚未启动 |

自动 watcher 会按以下顺序继续：

```text
198 个 Stage-A shard 全部完成
  → assemble Stage-A manifest
  → 抽样验证 source waveform
  → 构造 full198 joint train/dev manifest
  → 自动启动 8-GPU Megatron 单阶段联合训练
```

对应 watcher：

```text
tmux: uniss_phase3_joint_full198_pipeline_v3
log:  logs/uniss_phase3_whisper_streamspeech_joint_v1/full198_pipeline_v3.log
```

### 1.2 是否是在最佳 Phase3 上修改

**是，但不是覆盖旧 Phase3，也不是从旧 Phase3 optimizer 状态继续跑。**

准确描述是：

1. 从最佳 full198 Phase3 `iter_0009075` 导出的 Hugging Face Qwen 权重初始化新的联合模型；
2. 从原 `glm4_tokenizer` 加载 WhisperVQ；
3. 新增 STE bridge、四个 text CTC heads 和 NAR BiCodec Unit CTC 模块；
4. 用一个新的 Megatron optimizer 从新实验 iteration 0 开始联合训练；
5. 使用很低的 Qwen 学习率和 20% Phase3 exact replay，降低旧 offline 能力遗忘风险。

Phase3 来源链是明确可验证的：

```text
checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075
  → export
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
  → AutoModelForCausalLM.from_pretrained(...)
新 Phase3WhisperStreamSpeechJointModel
```

`export_manifest.json` 记录的 `source_checkpoint` 正是上述 `iter_0009075`。

### 1.3 是否完全按照官方 StreamSpeech 实现

**不是逐行或逐架构复现，而是保留最佳 UniSS Phase3 架构后，移植 StreamSpeech 最核心的训练方法。**

已经移植的核心思想：

- 单阶段 multi-task joint training；
- `1/8/4/4` 四任务相对 loss 权重；
- source ASR CTC；
- target NAR-S2TT CTC；
- CTC posterior 构造 `g(i)`，限制目标 token 的 source-prefix 可见范围；
- NAR text-to-unit CTC；
- `320/640/960/1280 ms/offline` multi-chunk training。

没有照搬的部分：

- 不使用官方 12-layer Conformer encoder，而保留 Phase3 的 WhisperVQ；
- 不使用官方 4-layer MT Transformer decoder，而保留 Phase3 的 Qwen2.5-0.5B；
- 不使用 1000 类离散 unit 和 CodeHiFiGAN，而使用 8192 类 BiCodec semantic unit 与 BiCodec decoder；
- 使用 Megatron-Core，不使用 Fairseq；
- 额外加入官方论文没有的 20% Phase3 exact replay；
- 当前只完成训练时 chunk mask，尚未完成该新实验专属的 Whisper K/V cache 在线推理实现。

所以最准确的项目名称是：

> **Phase3-preserving StreamSpeech-inspired single-stage joint training**

而不是“官方 StreamSpeech 原模型复现”。

## 2. 独立文件夹和输出路径

本实验新建了独立目录，不覆盖此前 Phase1/2/3、Emformer、Student v2/v3 或其他 streaming 实验。

### 2.1 实验入口

```text
/opt/dlami/nvme/jasonleeeli/projects/UniSS/
└── experiments/uniss_phase3_whisper_streamspeech_joint_v1/
    ├── experiment.env
    ├── README.md
    ├── scripts/
    └── tests/
```

### 2.2 新训练实现

```text
/opt/dlami/nvme/jasonleeeli/projects/UniSS/
└── training/phase3_whisper_streamspeech_joint/
    ├── pretrain_joint_megatron.py
    ├── model.py
    ├── whisper_frontend.py
    ├── whisper_multichunk.py
    ├── policy_mask.py
    ├── phase3_ste_bridge.py
    ├── phase3_batch.py
    ├── ctc_heads.py
    ├── nar_bicodec_ctc.py
    ├── dataset.py
    └── losses.py
```

### 2.3 独立数据、checkpoint、日志和 TensorBoard

```text
data/processed/phase3_whisper_streamspeech_joint_v1/
checkpoints/uniss_phase3_whisper_streamspeech_joint_v1/
logs/uniss_phase3_whisper_streamspeech_joint_v1/
runs/uniss_phase3_whisper_streamspeech_joint_v1/
```

正式 run 默认名称：

```text
phase3_whisper_streamspeech_joint_full198_v1
```

TensorBoard：

```text
http://127.0.0.1:6031/
```

## 3. 为什么 Stage-A 是必要的

原 Phase3 训练不是直接读取 waveform，而是读取预先生成的 `source_glm` token：

```text
source_glm + prompt → Qwen → text + target BiCodec token
```

因此原 Phase3 的训练图里没有 WhisperVQ 前向过程，Qwen loss 也无法更新 WhisperVQ。

新的联合训练需要在同一 forward 中计算：

```text
waveform
  → WhisperVQ hidden
  → ASR CTC / target CTC
  → g(i)
  → policy-conditioned Qwen
  → target text hidden
  → NAR BiCodec Unit CTC
```

UniST 当前 source 侧主要保存的是 BiCodec 表示，所以先用 Stage-A 恢复 source waveform。Stage-A 的输出是新实验的训练输入，不会修改原始 parquet 或旧 Phase3 packed JSONL。

## 4. 正式模型结构

正式联合模型的计算图如下：

```text
source waveform
  │
  ├─ GPU log-Mel
  │
  └─ original WhisperVQ encoder
       │  每个 joint microbatch 选择一个 chunk mask
       │  C ∈ {320, 640, 960, 1280 ms, offline}
       │  bounded right context = 80 ms
       │
       ├─ source ASR CTC head ────────────────→ transcription
       ├─ target NAR-S2TT CTC head ──────────→ translation
       │          │
       │          └─ detached CTC counts → g(i)
       │
       └─ hard GLM code + straight-through bridge
                  │
                  └─ Phase3 Qwen2.5-0.5B
                       │  g(i)-conditioned source attention
                       ├─ AR target text CE
                       └─ target-text hidden
                              │
                              └─ 2L T2U + 2L Unit decoder
                                   └─ BiCodec semantic Unit CTC
```

另有一条互斥 replay 路径：

```text
old Phase3 packed sequence
  → same Phase3 Qwen
  → exact old causal CE
```

每个 microbatch 只走 joint 或 replay 中的一种，不会在同一个样本上同时叠加两套 conflicting objective。

## 5. 当前训练 loss

### 5.1 Joint microbatch：80%

联合样本计算四项 loss：

\[
L_{joint}=
1L_{BiCodec\text{-}CTC}
+8L_{AR\text{-}S2TT}^{g(i)}
+4L_{ASR\text{-}CTC}
+4L_{NAR\text{-}S2TT\text{-}CTC}.
\]

四项 loss 在跨 data-parallel rank 汇总前，分别按自身有效 target token 数归一化，避免长 BiCodec 序列仅因为长度更长而压过文本任务。

#### `L_BiCodec-CTC`，权重 1

输入是 Qwen 对目标翻译文本产生的 hidden，目标是 `target_bicodec` semantic unit。

作用：

- 把目标文本并行映射成语音 semantic unit；
- 避免旧 Phase3 必须逐 token 自回归生成很长的 semantic 序列；
- 对应 StreamSpeech 的 primary speech-to-unit CTC，但词表换成 UniSS BiCodec 8192 类 semantic unit。

#### `L_AR-S2TT^{g(i)}`，权重 8

这是目标翻译文本的 next-token CE，但与原 Phase3 full-source CE 不同。

目标文本第 `i` 个 token 只能看到 `g(i)` 允许的 source speech prefix，而不能默认看到整句 source。它负责保留 Qwen 强翻译能力，同时让训练条件接近 simultaneous inference。

#### `L_ASR-CTC`，权重 4

从 WhisperVQ pre-VQ hidden 预测 source transcription。

作用：

- 让 speech hidden 保留 source 内容和单调时间位置；
- 给 `g(i)` 提供 source token count；
- 使 WhisperVQ 在 chunk mask 下仍能形成可用的增量语音表示。

#### `L_NAR-S2TT-CTC`，权重 4

从相同 WhisperVQ hidden 直接预测 target translation token。

作用：

- 判断当前 source prefix 已经支持多少 target token；
- 与 ASR CTC 一起构造 `g(i)`；
- 为 Qwen 的 WRITE 边界提供显式对齐监督。

### 5.2 Exact Phase3 replay microbatch：20%

Replay 样本只计算：

\[
L_{replay}=0.5L_{P3\text{-}old\text{-}exact}.
\]

它使用原 full198 Phase3：

```text
data/megatron/phase3_unist198/packed_train.jsonl
```

并保留旧 packed causal attention、旧 token 顺序、旧 Quality/Performance 协议和 loss mask。目标是防止联合训练把最佳 Phase3 的 offline 翻译与旧 AR semantic fallback 能力完全覆盖。

由于 replay 采样概率是 20%，其平均标称贡献约为：

\[
0.20\times0.50=0.10.
\]

完整的期望训练目标更准确地写成：

\[
\mathbb E[L]
=0.8\,\mathbb E[L_{joint}]
+0.2\,\mathbb E[0.5L_{P3\text{-}old\text{-}exact}].
\]

### 5.3 当前仅监控、不直接加入 total loss 的量

代码还记录：

- `bridge/commitment_mse`；
- `whisper/quantize_loss`；
- `ctc/asr_infeasible`；
- `ctc/nar_infeasible`；
- `ctc/unit_infeasible`。

前两个目前是诊断指标，没有额外权重加入 `total loss`。CTC infeasible 指标用于发现 encoder 输出长度不足以容纳 target CTC path 的样本。

## 6. `g(i)` 如何让 Phase3 学习 simultaneous translation

设目标翻译为：

```text
"我 明天 去 北京"
```

CTC heads 根据当前 speech hidden 估计 source 和 target 已经累计出现多少 token。假设得到：

```text
g(我)=source frame 8
g(明天)=source frame 15
g(去)=source frame 19
g(北京)=source frame 27
```

训练 Qwen 预测“明天”时，它只能读取到 source frame 15，不能读取后面的 frame 16–末尾；预测“北京”时才允许读取到 frame 27。

这样 Qwen 学到的是：

```text
当前 prefix 足够时产生 target token
```

而不是旧 Phase3 的：

```text
先看到整句 source，再离线生成整句 target
```

`g(i)` 由 detached CTC posterior 构造，hard boundary 本身不反向传播；CTC heads 通过各自 CTC loss 学习，Qwen 通过受限 attention 下的 AR CE 学习。

## 7. WhisperVQ 是如何改成 streaming 的

### 7.1 没有更换 WhisperVQ 网络和 codebook

当前没有使用 Emformer，也没有把 Whisper 换成 Zipformer。加载的是原 Phase3 数据制作时使用的 WhisperVQ encoder 和 16,384 GLM codebook。

变更主要发生在 attention 可见范围：

```text
原始 offline：一个 query 可以利用整条有效 utterance

当前有限 chunk：一个 query 只能利用
  历史 chunk
  + 当前 chunk
  + 80 ms bounded right context
```

每个 joint microbatch 从下列集合确定性采样：

```text
320 ms / 640 ms / 960 ms / 1280 ms / offline
```

`offline` 仍保留，是为了让同一个 checkpoint 保持 full-context 能力并降低 streaming 微调导致的退化。

### 7.2 当前属于训练时 multi-chunk mask，不等于完整在线 runtime

当前训练调用仍把一条完整 waveform 传给 GPU，然后通过 attention mask 禁止模型使用未允许的未来帧。这是 **mask-parallel chunk training**。

它已经解决的是“模型在训练时不能任意偷看整句未来”的问题；尚未完全解决的是“在线运行时如何只计算新 chunk，而不重复计算历史 prefix”。

因此当前状态应这样表述：

| 能力 | 当前状态 |
|---|---|
| Whisper attention 的 chunk 限制 | 已实现 |
| multi-chunk 联合训练 | 已实现并通过 smoke |
| 80 ms bounded right context | 已实现 |
| full-prefix re-encode 推理基线 | 计划可实现，当前新实验目录尚无正式评估 runner |
| Whisper layer K/V cache | 尚未完成 |
| conv/pooling overlap state | 尚未完成 |
| cache/full 数值一致性 gate | 尚未执行 |
| 完整在线 Gradio | 尚未为该 checkpoint 实现 |

所以目前不能声称已经完成严格、计算复杂度恒定的生产级 streaming Whisper。更准确的说法是：

> 已实现支持 simultaneous 训练的 bounded multi-chunk WhisperVQ；正式在线增量 cache 仍是训练后的推理工程步骤。

### 7.3 理论等待时间如何理解

有限 chunk 下，一个 frame 最坏需要等待当前 chunk 剩余部分，再加 80 ms right context。以 320 ms chunk 为例，纯 encoder 策略的等待量不是固定 80 ms，而是大致位于：

```text
当前 chunk 剩余时间 + 80 ms
```

最终端到端延迟还会叠加：

- feature/conv 局部上下文；
- CTC boundary 稳定时间；
- Qwen 目标 token 生成；
- NAR Unit CTC commit；
- BiCodec 首包解码与音频缓冲。

因此不能仅凭 `chunk=320 ms` 就提前宣称端到端延迟已经小于 400 ms，必须在正式 checkpoint 上做流式生成和 latency evaluation。

## 8. 加载哪些 checkpoint

### 8.1 Qwen

加载：

```text
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
```

该目录由最佳 Megatron Phase3：

```text
checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075
```

导出。Qwen 结构保持：24 layers、hidden 896、FFN 4864、14 attention heads、2 KV heads、约 180k 扩展词表。

### 8.2 WhisperVQ

加载：

```text
pretrained_models/UniSS/glm4_tokenizer
```

需要注意：最佳 Phase3 checkpoint 本身主要是 Qwen 权重。原 Phase3 使用预计算 `source_glm`，所以 WhisperVQ 不是从 Phase3 Megatron optimizer 中恢复，而是从原 WhisperVQ tokenizer checkpoint 独立加载。

### 8.3 BiCodec

BiCodec 在当前 Stage-A 用于恢复 source waveform；正式 joint forward 的目标是预计算的 `target_bicodec` IDs。

BiCodec waveform decoder不进入当前 optimizer，也不会在本次 Megatron 训练中更新。正式推理时它作为冻结 vocoder/decoder，把预测的 semantic unit 和 speaker/global condition合成为语音。

## 9. 训练哪些参数、冻结哪些参数

8-GPU smoke v4 输出的实际 trainable parameter group 如下：

| 参数组 | 参数量 | 是否训练 | 最大有效 LR | 最小有效 LR |
|---|---:|---|---:|---:|
| 新 CTC/T2U/Unit heads | 26,697,407 | 是 | `1e-4` | `1e-5` |
| STE bridge | 1,147,776 | 是 | `5e-5` | `5e-6` |
| WhisperVQ 后半层 | 157,409,280 | 是 | `1e-5` | `1e-6` |
| WhisperVQ 前半层与 conv | 163,298,560 | 是 | `5e-6` | `5e-7` |
| Phase3 Qwen body | 357,898,112 | 是 | `2e-6` | `2e-7` |
| Phase3 Qwen input/output embedding | 161,710,080 | 是 | `1e-6` | `1e-7` |

总 trainable parameter 数约：

```text
868,161,215 ≈ 868.2M
```

冻结部分：

| 模块 | 状态 | 原因 |
|---|---|---|
| WhisperVQ 16,384 codebook | 冻结，约 20.97M 参数 | 防止 GLM code identity 漂移 |
| BiCodec waveform decoder | 不进入 optimizer | 保持旧音频生成接口和 speaker condition |
| feature extractor Hann window/mel filters | buffer，无可训练参数 | 固定声学预处理 |

### 9.1 是否只训练新增模块

不是。当前方案会：

- 高 LR 训练新增 CTC/T2U 模块；
- 中等 LR 训练 STE bridge；
- 低 LR 微调 WhisperVQ；
- 极低 LR 全参数微调 Phase3 Qwen。

这比“冻结 Phase3，只训练 adapter”更有适配能力，但灾难性遗忘风险也更高，所以必须依赖：

- Qwen `1e-6–2e-6` 量级 LR；
- 20% exact Phase3 replay；
- 20% 概率的 offline chunk；
- offline Phase3 回归评估 gate。

### 9.2 是否继承 Phase3 optimizer

不继承。

Megatron 参数中 `--load` 为 `None`，因为 Qwen 权重在 composite model 内通过 Hugging Face `from_pretrained` 加载。新实验：

- iteration 从 0 开始；
- optimizer state 从零初始化；
- scheduler 从零初始化；
- RNG/采样状态使用新实验 seed。

这属于“以 Phase3 权重初始化的新联合微调”，不是“从 Phase3 iteration 9075 原地 resume 到 9076”。

## 10. 与原 Phase3 训练的详细对比

| 项目 | 最佳 Phase3 v4 | 当前 joint v1 |
|---|---|---|
| 初始化 | Phase2 v4 checkpoint | Phase3 iter9075 HF Qwen + 原 WhisperVQ |
| 主输入 | 预计算 `source_glm` token | source waveform，经 trainable WhisperVQ |
| 模型主干 | Qwen2.5-0.5B | 同一个 Phase3 Qwen + WhisperVQ + 新 heads |
| 主 loss | packed next-token CE | 四项 StreamSpeech-style joint loss |
| 旧 Phase3 CE | 100% 主任务 | 20% exact replay，权重 0.5 |
| source ASR CTC | 无 | 有，权重 4 |
| target NAR-S2TT CTC | 无 | 有，权重 4 |
| NAR Unit CTC | 无 | 有，权重 1 |
| target text AR CE | full-source | `g(i)` prefix-conditioned，权重 8 |
| multi-chunk | 无 | 320/640/960/1280/offline |
| Whisper 是否训练 | 不在 Phase3 图中 | 除 codebook 外低 LR 训练 |
| Qwen 是否训练 | 全参数 | 全参数极低 LR |
| micro batch | 2 | 1，因 waveform/可变长度 compound batch |
| global batch | 128 | 128 |
| sequence length | 18000 | 18000，主要用于 exact replay 兼容 |
| train iterations | 9075 | 9075 |
| Phase3 LR | `1e-5 → 1e-6` | Qwen body `2e-6 → 2e-7`；I/O `1e-6 → 1e-7` |
| warmup | 200 | 4000 |
| scheduler | cosine | inverse-square-root |
| weight decay | 0.1 | 0.01 |
| Adam beta | `0.9/0.95` | `0.9/0.98` |
| clip grad | 0.5 | 0.5 |
| precision | BF16 | BF16 |
| GPU | 8 | 8 |
| dataloader | cyclic global shuffle | cyclic、direction-balanced、joint/replay 同步采样 |

### 10.1 哪些地方“和 Phase3 一样”

- 使用最佳 Phase3 Qwen 权重；
- Qwen 层数、hidden、FFN、heads、RoPE 和扩展词表一致；
- 继续使用 UniSS GLM/BiCodec token 体系；
- sequence length 保持 18000；
- global batch 保持 128；
- 8-GPU BF16 Megatron-Core；
- 正式长度保持 9075 iterations；
- exact replay 保留旧 Phase3 packed sequence 和 causal loss mask。

### 10.2 哪些地方“不一样”

最根本差异不是超参数，而是训练图已经从：

```text
pretokenized SFT
```

变成：

```text
waveform → WhisperVQ → CTC alignment → prefix-conditioned Qwen
         → NAR text-to-unit
```

因此不能直接用旧 Phase3 的 `micro-batch=2`、统一 `1e-5` LR 和单 CE loss 原样训练。当前 compound model 更大、显存更高，且新增模块与 pretrained 模块需要不同 LR。

## 11. 与官方 StreamSpeech 的详细对比

| 项目 | 官方 StreamSpeech | 当前 UniSS joint v1 |
|---|---|---|
| 框架 | Fairseq | Megatron-Core + Transformers compound model |
| speech encoder | 12-layer Conformer，dim 256，unidirectional/chunk conv | 原 WhisperVQ，hidden 1280，chunk attention mask |
| AR translation | 4-layer Transformer，dim 512 | 最佳 Phase3 Qwen 24-layer，hidden 896 |
| unit vocab | 1000 discrete units | 8192 BiCodec semantic units |
| vocoder | CodeHiFiGAN | BiCodec decoder |
| T2U | 2-pass synthesizer + CTC unit decoder | Qwen text hidden + 2L T2U + 2L Unit CTC |
| loss weights | `1/8/4/4` | 相同 `1/8/4/4` |
| ASR CTC | 有 | 有，中/英独立 heads |
| target NAR-S2TT CTC | 有 | 有，中/英独立 heads |
| CTC-guided streaming mask | 有 | 有，移植到 decoder-only Qwen attention |
| multi-chunk | `8/16/24/32/99999` feature steps | `320/640/960/1280/offline` ms |
| old-model replay | 无 | 20% Phase3 exact replay，权重 0.5 |
| 训练语言/数据 | 官方 CVSS-C 等方向 | UniST full198 双向中英 |
| online cache | 官方模型/agent路径 | 当前新实验尚未补齐 Whisper cache runtime |

### 11.1 最接近原文的部分

- 四任务在同一个 optimizer update 流中联合训练；
- 四项 loss 的相对权重 `1/8/4/4`；
- ASR CTC 与 target CTC posterior 构造 prefix boundary；
- AR translation 在 `g(i)` 下训练；
- multi-chunk 随机训练；
- target text hidden 到 unit CTC 的 two-pass 思路。

### 11.2 主要创新/适配部分

- 将官方 Conformer/Transformer 换成现有最佳 WhisperVQ/Qwen，而不是舍弃 Phase3；
- 将 `g(i)` 改造成 decoder-only Qwen 的 arbitrary attention mask；
- 用 STE 把 hard GLM forward 与 Whisper 梯度连接；
- 用 BiCodec 8192 semantic unit 替代官方 1000 unit；
- 用 exact replay 保护已经验证过的 Phase3 offline 能力；
- 在 full198 双向中英 UniST 上训练。

## 12. 当前 smoke 是否证明脚本能训练

8-GPU smoke v4 已完成两个 Megatron update、validation 和 checkpoint save：

```text
iteration 1/2: 完成，无 skipped/nan iteration
iteration 2/2: 完成，无 skipped/nan iteration
checkpoint iter 1: 保存成功
checkpoint iter 2: 保存成功
final validation: 完成
```

Smoke 中五项 loss 均有有限值，且四类 CTC infeasible 指标没有出现系统性失败。它证明：

- 8 卡 Megatron forward/backward 可以完成；
- BF16 与 gradient checkpointing 可运行；
- joint/replay 同步采样可运行；
- 参数组 LR 生效；
- validation 和 checkpoint 写盘可运行。

但两个 iteration 只证明工程链路，不证明最终质量、延迟或 Phase3 保真。正式训练后仍必须评估：

1. offline BLEU/ASR-BLEU、AutoPCP、SLC、speaker similarity；
2. streaming BLEU/quality-latency curve；
3. AL/AP/DAL、first token/first audio latency；
4. 各 chunk size 的结果；
5. Phase3 old protocol replay 回归；
6. cache/full parity 和 future perturbation causality。

## 13. 当前实现仍需注意的风险

### 13.1 正式训练尚未启动

当前 TensorBoard 端口存在，但 full198 正式 run 的新曲线要等 Stage-A 和 joint manifest 完成后才出现。

### 13.2 Smoke 初始 CTC loss 高是正常现象，但需要下降

CTC/T2U heads 是随机初始化，初始 ASR/NAR/unit CTC loss 明显高于已经预训练的 Qwen text CE 是预期现象。正式训练需要观察：

- CTC loss 是否持续下降；
- infeasible rate 是否接近 0；
- grad norm 是否在 warmup 后稳定；
- Qwen replay loss 是否恶化；
- offline dev 质量是否保持。

### 13.3 当前 Whisper 是 bounded mask，不是完成的在线 cache

如果训练后直接用 full utterance masked forward 做评估，可以验证算法质量与延迟策略；要部署实时网页，还需实现和验证增量 cache，不能把训练 mask 本身当成完整 streaming runtime。

### 13.4 Phase3 replay 是保护机制，不是绝对保证

20% replay 和低 Qwen LR能降低遗忘风险，但是否真正保住 Phase3 性能只能由 offline dev/test 对照决定。checkpoint 选择不能只看 total loss。

## 14. 对用户问题的逐条回答

### 当前是训练吗

不是。当前是 full198 Stage-A waveform reconstruction；完成后自动进入 8-GPU Megatron 正式训练。

### 当前训练计划是在 Phase3 上修改的吗

是。Qwen 从最佳 Phase3 full198 iter9075 初始化，但新建独立 composite model、optimizer、checkpoint 和目录，不覆盖旧 Phase3。

### 当前训练和 StreamSpeech 有什么差别

训练 objective 和 CTC/multi-chunk 思想来自 StreamSpeech；架构仍是 UniSS 的 WhisperVQ + Phase3 Qwen + BiCodec，框架是 Megatron，并额外加入 Phase3 replay。

### 训练 loss 是什么

80% joint microbatch 使用 `1×BiCodec Unit CTC + 8×policy AR-S2TT + 4×ASR CTC + 4×target NAR-S2TT CTC`；20% replay microbatch使用 `0.5×old Phase3 exact CE`。

### Whisper 如何改成 streaming

不更换模型，通过 `320/640/960/1280/offline` 随机 multi-chunk attention mask和 80 ms right context限制未来信息。当前是训练时 mask-parallel streaming 模拟，增量 K/V cache runtime尚未完成。

### 是否按照 StreamSpeech 和最好 Phase3 模型实现

是“最好 Phase3 主干 + StreamSpeech 核心训练方法”的混合实现；不是官方 StreamSpeech Conformer/Fairseq 模型的逐架构复现。

### 是否加载训练好的 Phase3

是，加载 `qwen0p5b_phase3_unist198_iter_0009075_hf`。但只加载模型权重，不继承原 optimizer/scheduler iteration。

### 训练哪些参数

训练新 CTC/T2U、STE bridge、除 codebook 外的 WhisperVQ、以及全部 Phase3 Qwen；不同模块使用分层学习率。冻结 Whisper codebook和 BiCodec waveform decoder。

### 训练和 Phase3 一样吗

部分基础配置相同：8 GPU、BF16、GBS 128、seq 18000、9075 iterations、相同 Qwen/词表。数据入口、microbatch、loss、Whisper、attention、optimizer schedule和参数组均不同，因此不是原 Phase3 训练的简单重复。

## 15. 关键源码索引

| 内容 | 路径 |
|---|---|
| 正式启动参数 | `experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/run_megatron_8gpu.sh` |
| full198 自动接续 | `experiments/uniss_phase3_whisper_streamspeech_joint_v1/scripts/wait_and_train_full198.sh` |
| Phase3/Whisper checkpoint 路径 | `experiments/uniss_phase3_whisper_streamspeech_joint_v1/experiment.env` |
| Megatron entrypoint | `training/phase3_whisper_streamspeech_joint/pretrain_joint_megatron.py` |
| compound model和 loss | `training/phase3_whisper_streamspeech_joint/model.py` |
| Whisper multi-chunk frontend | `training/phase3_whisper_streamspeech_joint/whisper_frontend.py` |
| chunk attention mask | `training/phase3_whisper_streamspeech_joint/whisper_multichunk.py` |
| CTC-guided `g(i)` | `training/phase3_whisper_streamspeech_joint/policy_mask.py` |
| Phase3 attention batch | `training/phase3_whisper_streamspeech_joint/phase3_batch.py` |
| STE bridge | `training/phase3_whisper_streamspeech_joint/phase3_ste_bridge.py` |
| NAR BiCodec Unit CTC | `training/phase3_whisper_streamspeech_joint/nar_bicodec_ctc.py` |
| 原始详细方案 | `docs/uniss_training_reproduction/uniss_phase3_whisper_streamspeech_single_stage_joint_training_plan.md` |

