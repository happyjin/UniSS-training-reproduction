# UniSS Causal Audio Student v2：完整实现、训练、推理与结果说明

> 文档用途：向项目同事完整解释 Student v2 为什么要做、如何构建数据、模型如何训练和推理、目前取得了什么结果、为什么还需要 Stage-B-v3 修复。
>
> 实验范围：UniST 15-shard simultaneous/streaming 验证路线。
>
> 状态日期：2026-08-03。
>
> 重要结论：Student v2 的流式结构、因果性、cache 一致性和实时计算速度已经通过验证，但 teacher GLM token 一致率仅约 29.3%，因此它是可运行的真流式前端原型，还不是最终质量模型。

## 1. 一句话说明 Student v2

Student v2 是一个约 121.1M 参数的 causal Emformer 语音编码器。它每 160 ms 接收一次新音频，最多使用 80 ms lookahead，并通过流式 cache 连续输出与原 UniSS WhisperVQ 相同 codebook 空间中的 GLM speech token。

训练时，冻结 WhisperVQ 作为 teacher；推理时，teacher 被完全移除，只保留 Student v2、冻结的 UniSS Phase3 Qwen 和 BiCodec。

它解决的核心问题是：

> 原 WhisperVQ 更适合整段或 prefix re-encode，直接用于麦克风真流式推理时会反复计算历史音频；Student v2 改为有状态的因果编码器，使历史状态可以缓存，每次只处理新到达的音频。

## 2. Student v2 在 UniSS 系统中的位置

原 offline UniSS 推理：

```text
完整源语音
  → WhisperVQ
  → source GLM tokens
  → Phase3 Qwen
  → 翻译文本/目标 semantic tokens
  → BiCodec
  → 翻译语音
```

Student v2 streaming 推理：

```text
实时16 kHz音频流
  → Causal Audio Student v2
  → 增量 source GLM tokens
  → Phase3 Qwen / WAIT-WRITE控制
  → 增量目标 semantic tokens
  → BiCodec缓冲与解码
  → 翻译语音
```

Student v2 只替换源语音 tokenizer/frontend，不修改已经训练好的 Phase1、Phase2、Phase3 和 BiCodec checkpoint。因此旧实验和原始 offline 路线仍可独立复现。

## 3. 为什么不能直接把原 WhisperVQ 当成最终流式前端

WhisperVQ 的权重本身可以在有限音频 prefix 上运行，但如果每到 160 ms 都把从句首到当前时刻的全部音频重新编码一次，会出现三个问题：

1. 历史音频反复计算，音频越长，累计开销越大。
2. 不同 prefix 的 token 可能发生 revision，需要额外稳定化策略。
3. 很难为长会话维护固定计算量和固定显存。

Student v2 使用 Emformer 的流式 state/cache：

- 新 chunk 只计算一次；
- 历史信息保存在有限左上下文和网络 state 中；
- 推理成本近似随新增音频线性增长；
- cache 推理可以与一次性 causal forward 做严格一致性测试。

## 4. 数据基础

Student v2 使用 Stage A 正式处理后的 UniST 15-shard 数据，不是 full198。

| 数据 | 记录数 | 用途 |
|---|---:|---|
| formal train manifest | 1,325,243 | clone 预训练与源音频读取 |
| formal valid manifest | 13,469 | clone/prefix validation |
| clone train sidecar | 1,325,243 | 全量 streaming teacher 蒸馏 |
| clone valid sidecar | 13,469 | clone validation |
| prefix80 train sidecar | 100,000 | exact-prefix 80 ms 微调 |
| prefix80 valid sidecar | 13,469 | prefix validation |

源 manifest：

```text
data/processed/simul_uniss_subsecond_v2/
formal_15shard_v1/stage_a_formal/formal_train_manifest.jsonl

data/processed/simul_uniss_subsecond_v2/
formal_15shard_v1/stage_a_formal/formal_valid_manifest.jsonl
```

每条源记录主要包含：

- `source_audio`：源语言音频；
- `src_lang`、`tgt_lang`：翻译方向；
- `transcription`：源语音转写；
- `translation`：目标翻译文本；
- `teacher_source_glm`：teacher GLM token；
- `teacher_source_glm_end_ms`：GLM token 的时间代理；
- `source_words`：源文本词级时间信息；
- `target_support`：当前源 prefix 对目标内容的支持关系。

## 5. Stage-A-v3：构建 Student v2 的 teacher sidecar

Student v2 训练前先生成两套独立监督数据：streaming clone sidecar 和 exact-prefix80 sidecar。

### 5.1 Streaming clone sidecar

路径：

```text
data/processed/simul_uniss_subsecond_v2/
stage_a_v3_clone_15shard_v1/
```

生成规模：

- 1,325,243 条记录；
- 93,171,767 个 target GLM token；
- 8 张 GPU 并行生成。

Teacher 是原 WhisperVQ 权重的只读副本，但 attention mask 被替换为受限流式 mask：

```text
chunk = 160 ms
right context = 80 ms
允许注意：全部有限历史 + 当前chunk + 80 ms右上下文
禁止注意：更远未来音频
```

clone sidecar 保存：

- streaming target GLM token；
- 完整 teacher reference token；
- token stability；
- 真实 pre-VQ hidden，维度 1280；
- codebook top-k 邻居及距离；
- 原 source manifest 的 index 和 byte offset。

这一阶段提供广覆盖的表示蒸馏监督，是 Student v2 学会 WhisperVQ codebook 几何的主要来源。

### 5.2 Exact-prefix80 sidecar

路径：

```text
data/processed/simul_uniss_subsecond_v2/
stage_a_v3_prefix80_100k_v1/
```

生成规模：

- 100,000 条记录；
- 5,927,281 个 target GLM token；
- 8 张 GPU 并行生成。

对每条音频构造：

```text
commit time = 160, 320, 480, ... ms
visible audio end = commit time + 80 ms
```

例如 commit time 为 640 ms 时，teacher 最多看到前 720 ms 音频，但只提交由前 640 ms 时间线支持的 token。

稳定性标签的含义是：某 token 在当前以及后续若干 prefix 中保持相同，并且能够形成 append-only 输出。

### 5.3 v2 prefix sidecar 的已知缺陷

Student v2 的 prefix80 sidecar 只保存了 token 和 stability，没有保存真实 pre-VQ hidden。因此 prefix 微调日志中的：

```text
hidden_l1 = 0
hidden_cosine = 0
```

不是 hidden 已经完美对齐，而是这一阶段根本没有 hidden target。这是后续 v3 修复的主要原因之一。

此外，这 100k 数据是从原 manifest 前部选取，不是严格的 `50k EN→ZH + 50k ZH→EN` 平衡随机样本。

## 6. Student v2 模型架构

Student v2 checkpoint 中共有：

```text
121,113,859 trainable parameters
```

WhisperVQ 的 16,384×1,280 codebook 作为冻结 buffer 使用，不计入可训练参数。

### 6.1 输入特征

```text
16 kHz waveform
  → 25 ms window / 10 ms hop MelSpectrogram
  → 128维 log-Mel
  → 连续4帧stack
  → 每40 ms一个 Student frame
  → Linear + LayerNorm + GELU
```

### 6.2 Causal Emformer Encoder

| 参数 | 数值 |
|---|---:|
| hidden size | 768 |
| layers | 16 |
| attention heads | 12 |
| FFN dimension | 3072 |
| dropout | 0.1 |
| segment frames | 4 |
| 每帧时间 | 40 ms |
| segment/chunk | 4×40 = 160 ms |
| right-context frames | 2 |
| lookahead | 2×40 = 80 ms |
| left-context frames | 50 |
| 显式左上下文 | 50×40 = 2000 ms |

Emformer 同时维护内部 state，因此在线推理时不需要从句首重新计算。

### 6.3 GLM token时间分辨率

Emformer 每 40 ms 输出一个 hidden frame。随后每两个 frame 池化一次：

```text
2 × 40 ms = 80 ms / GLM latent
```

因此 Student v2 最多每 80 ms 产生一个新的 1280 维 GLM latent。

### 6.4 四个输出头

| 输出头 | 输出 | 作用 |
|---|---|---|
| GLM latent head | 1280维向量 | 对齐 WhisperVQ pre-VQ/codebook 空间 |
| source CTC head | 8193类 | 约束当前音频已支持的源语言内容 |
| target capacity head | 标量 | 估计当前源证据支持多少目标内容 |
| stability head | 标量 | 预测当前 GLM token 是否稳定可提交 |

GLM latent 通过欧氏距离在冻结的 16,384 项 codebook 中寻找最近邻，得到 Phase3 可直接读取的 source GLM token ID。

## 7. Student v2 的损失函数

总体训练目标可以写为：

```text
L_total =
    1.0 L_hidden_L1
  + 0.5 L_hidden_cosine
  + 1.0 L_codebook_CE
  + 0.5 L_codebook_margin
  + 0.05 L_full_context
  + auxiliary_scale × (
        0.1 L_source_CTC
      + 0.1 L_capacity
      + 0.1 L_stability
      + 0.05 L_chunk_consistency
    )
```

各损失含义：

### 7.1 Hidden L1 与 cosine

让 Student 的 1280 维 latent 接近 teacher 进入 VQ codebook 前的 hidden。

这比只优化离散 token 更平滑：即使预测还没有落入完全相同的 VQ cell，latent 也可以先接近正确的声学语义区域。

### 7.2 Full-codebook CE

对全部 16,384 个 codebook 项计算距离，以负距离作为 logits，直接优化正确 teacher token 的概率。

### 7.3 Codebook margin

要求正确 codebook 向量的距离至少比最近错误向量小一个 margin，减少落入相邻错误 cell 的情况。

### 7.4 Full-context cosine

以完整 reference token 对应的冻结 codebook embedding 作为弱约束，防止因果 token 完全偏离原 UniSS 语义空间。

### 7.5 Source CTC

辅助 Student 保留当前 source prefix 的语言内容，而不是只模仿声学量化编号。

### 7.6 Target capacity

预测当前已听到的源语音能够支持多少目标翻译内容，为后续 WAIT/WRITE 提供证据量特征。

### 7.7 Stability

预测 token 在未来 prefix 中是否会继续保持不变。

### 7.8 Chunk consistency

改变 chunk 切分边界，对相同已提交时间区域运行另一次 forward，约束两次 latent 输出一致，减少 chunk boundary artifact。

## 8. 两阶段训练过程

Student v2 使用 8 张 H200、DDP、BF16 训练。

训练脚本中的 `batch_size=32` 是每个 DDP rank 的 DataLoader batch；8 卡、无梯度累积时，有效全局 batch 约为 256。

### 8.1 阶段一：clone pretraining

| 配置 | 数值 |
|---|---:|
| 初始化 | 随机初始化 Student v2 |
| 数据 | 1,325,243 条 clone sidecar |
| steps | 20,000 |
| learning rate | `1e-4` |
| minimum LR | `1e-6` |
| warmup | 5% |
| optimizer | AdamW |
| weight decay | 0.01 |
| per-GPU batch | 32 |
| effective global batch | 256 |
| evaluation interval | 500 steps |
| validation batches | 8 |

课程学习：

1. 前 5,000 steps：只重点学习 representation/codebook；
2. 接下来 5,000 steps：逐渐把 CTC、capacity、stability、consistency 权重从0升到1；
3. 后 10,000 steps：全部目标联合优化。

最终 clone checkpoint：

```text
checkpoints/simul_uniss_subsecond_v2/
stage_b_v2_clone_pretrain_15shard_v1/best.pt
```

最佳 checkpoint 位于约 step 19,500，训练记录中的最佳 target agreement 约 23.0%。

### 8.2 阶段二：prefix80 fine-tuning

| 配置 | 数值 |
|---|---:|
| 初始化 | clone `best.pt` |
| 数据 | 100,000 条 exact-prefix80 sidecar |
| steps | 10,000 |
| learning rate | `2e-5` |
| minimum LR | `1e-6` |
| warmup | 5% |
| per-GPU batch | 32 |
| effective global batch | 256 |
| evaluation interval | 500 steps |

最终 checkpoint：

```text
checkpoints/simul_uniss_subsecond_v2/
stage_b_v2_prefix80_finetune_100k_v1/best.pt
```

该 checkpoint 为 step 10,000，训练过程中的最佳 validation target agreement 约 37.94%。

实际运行时间：

- clone 20k steps：约 47 分钟；
- prefix80 10k steps：约 23.5 分钟。

## 9. Student v2 在线推理过程

### 9.1 单个chunk如何处理

假设客户端已经收到 0–240 ms 音频：

```text
committed audio = 0–160 ms
lookahead only = 160–240 ms
```

Student 可以利用 160–240 ms 辅助判断边界，但只输出由前 160 ms committed timeline 支持的状态。

下一步收到 240–400 ms 音频后：

```text
new committed region = 160–320 ms
new lookahead region = 320–400 ms
history = cached Emformer state
```

模型不重新编码 0–160 ms，而是从 cache 继续计算。

### 9.2 为什么这是因果推理

验证时对未来音频做随机扰动，如果已提交位置的输出发生变化，说明模型偷看未来。

Student v2 的结果：

```text
future_perturbation_max_abs = 0.0
```

说明已经提交的 hidden 不受被禁止未来区域影响。

### 9.3 为什么80 ms不等于80 ms开始播放译文

80 ms 是 GLM latent/token 的输出时间分辨率，不是最终端到端首包延迟。

最终译音开始时间还包括：

- Student 需要积累的首个160 ms chunk和80 ms lookahead；
- stability/commit判断；
- Phase3 WAIT/WRITE策略；
- Qwen目标token生成；
- BiCodec最小可解码目标token数量；
- 播放器buffer和网络时间。

所以 Student v2 证明了前端可以低于实时连续运行，但没有单独证明最终翻译音频一定在1秒内开始。

## 10. 正式验证结果

验证报告：

```text
reports/simul_uniss_subsecond_v2/
stage_b_v2_prefix80_validation.json
```

| 指标 | 结果 | 解读 |
|---|---:|---|
| samples | 128 | 正式 causal validation 样本数 |
| audio seconds | 789.16 s | 被测音频总时长 |
| compute seconds | 77.75 s | Student active计算时间 |
| active RTF | 0.0985 | 约为实时预算的十分之一，速度通过 |
| long-session RTF | 0.0971 | 长会话下速度没有明显退化 |
| future perturbation max | 0 | 因果性通过 |
| cache/full committed parity | 1.0 | cache token结果完全一致 |
| cache max abs | `2.86e-6` | 数值误差很小 |
| first self-stable coverage | 100% | 每条样本最终都能产生自身稳定token |
| first self-stable p50 | 320 ms | 自身输出稳定中位时间 |
| first self-stable p95 | 480 ms | 自身输出稳定95分位时间 |
| first correct-stable coverage | 31.25% | 只有31.25%样本出现与target一致且稳定的首token |
| target position agreement | 29.29% | 主要质量指标，未通过 |
| target edit agreement | 29.24% | 编辑距离口径，同样偏低 |
| committed target accuracy | 36.75% | committed位置的target正确率 |
| structural pass | true | 结构、cache、因果、速度通过 |
| quality pass | false | teacher兼容质量未通过 |
| final gate | failed | 不能直接进入正式后续质量路线 |

## 11. Phase3敏感性评估

将 Student v2 输出的 source GLM token 接入冻结的 full198 Phase3 checkpoint：

```text
checkpoints/exported_hf/
qwen0p5b_phase3_unist198_iter_0009075_hf
```

8条小样本 sensitivity 测试结果：

| 输入source GLM stream | ZH→EN BLEU | EN→ZH BLEU |
|---|---:|---:|
| released UniST GLM | 26.61 | 33.45 |
| streaming clone 160×80 | 22.46 | 22.95 |
| Student v2 prefix80 | 15.32 | 21.13 |

注意：ZH→EN 只有3条、EN→ZH只有5条，这不是正式可发表BLEU，只能作为敏感性检查。

结论是：Phase3 能读取 Student v2 token并完成翻译，接口兼容成立；但 Student v2 的 source token误差会明显传递到下游翻译。

## 12. Student v2到底成功了什么

Student v2 已经成功证明：

1. 可以用约121M的因果模型替换完整WhisperVQ在线前端；
2. 160 ms chunk + 80 ms lookahead可以稳定执行；
3. cache推理和full causal forward一致；
4. 不偷看禁止的未来音频；
5. active RTF约0.1，计算速度足够实时；
6. 输出token使用原16384项WhisperVQ codebook，Phase3接口无需修改；
7. 当前在线Gradio可以使用该Student运行真实录音/上传推理。

## 13. Student v2没有成功什么

它没有达到最终质量门：

- exact target agreement约29.3%，离研究目标90%很远；
- correct-stable coverage仅31.25%；
- 下游Phase3 BLEU较released/clone stream下降；
- 因此不能把“320 ms first self-stable”解释成“320 ms高质量同传结果”。

最关键的区别是：

> self-stable只表示模型不再修改自己的预测，不代表预测是正确的。

一个模型可以非常稳定地输出错误token，所以必须同时检查 correct-stable coverage、teacher agreement和下游Phase3质量。

## 14. v2质量未通过的主要原因

### 14.1 Prefix阶段缺少hidden监督

clone阶段有真实 pre-VQ hidden，prefix80阶段没有。第二阶段本应适应真实prefix分布，却只能依赖离散16384分类和辅助任务，优化难度很高。

### 14.2 100k样本方向不平衡

从manifest前部连续选择100k，可能使 EN→ZH 和 ZH→EN 分布偏斜；模型在一个方向上的进步可能掩盖另一个方向的退化。

### 14.3 Prefix数据替换而不是混合clone数据

第二阶段只读取prefix80 sidecar，没有保留clone hidden supervision，可能发生 representation forgetting。

### 14.4 离散codebook分类过难

codebook有16,384个cell。没有连续hidden target时，Student必须直接命中teacher的精确cell；即使落在声学近邻cell，exact agreement仍记为错误。

### 14.5 Checkpoint选择没有直接约束双向Phase3质量

v2主要依据validation agreement保存best checkpoint，没有对 top candidates 做严格双向平衡的冻结Phase3 BLEU联合选择。

## 15. 当前Stage-B-v3如何修复Student v2

Stage-B-v3不是推翻Student v2架构，而是在保留其已通过的因果结构和checkpoint基础上修复监督。

| Student v2问题 | Stage-B-v3修复 |
|---|---|
| prefix无pre-VQ hidden | exact-prefix重新导出真实1280维pre-VQ hidden |
| 前100k方向偏置 | 随机确定性选择50k EN→ZH + 50k ZH→EN |
| prefix替换clone | exact-prefix hidden与clone hidden按1:1混合 |
| 可能遗忘clone表示 | 每个prefix样本配同source index的clone样本 |
| 只看总体agreement | 分方向、分监督类型统计agreement与调和均值 |
| 缺少下游选模 | top-3 checkpoint跑双向冻结Phase3 BLEU联合选择 |

v3从以下v2 checkpoint初始化：

```text
checkpoints/simul_uniss_subsecond_v2/
stage_b_v2_prefix80_finetune_100k_v1/best.pt
```

因此v2不是废弃实验，而是v3的初始化基础和对照基线。

## 16. 完整复现命令

### 16.1 环境

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
source /opt/dlami/nvme/jasonleeeli/env_recovery/\
uniss-train-20260721/activate_uniss.sh
```

所有cache和临时文件应位于：

```bash
export TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp
export XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache/xdg
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
```

### 16.2 配置

```text
configs/experiments/simul_uniss_subsecond_v2/
stage_b_v2_causal_15shard_v1.env
```

### 16.3 完整自动流程

原始自动流水线：

```bash
bash scripts/simul_uniss_subsecond_v2/\
run_stage_b_v2_repair_pipeline.sh
```

它按顺序执行：

1. 等待GPU空闲；
2. 生成clone validation sidecar；
3. 生成prefix80 validation sidecar；
4. 生成全量15-shard clone train sidecar；
5. 生成100k prefix80 train sidecar；
6. 8卡clone预训练；
7. 8卡prefix80微调；
8. clone causal validation；
9. prefix80 causal validation；
10. 冻结Phase3 sensitivity评估。

### 16.4 单独训练

```bash
# clone pretraining
bash scripts/simul_uniss_subsecond_v2/\
train_stage_b_v2_causal.sh clone

# prefix80 fine-tuning
bash scripts/simul_uniss_subsecond_v2/\
train_stage_b_v2_causal.sh prefix80
```

### 16.5 单独验证

```bash
MODE=clone GPU=0 \
bash scripts/simul_uniss_subsecond_v2/\
validate_stage_b_v2_causal.sh

MODE=prefix80 GPU=0 \
bash scripts/simul_uniss_subsecond_v2/\
validate_stage_b_v2_causal.sh
```

## 17. 关键代码和产物索引

| 类型 | 路径 |
|---|---|
| Student模型 | `training/simul_uniss/subsecond_v2/stage_b_latent_model.py` |
| v2训练损失 | `training/simul_uniss/subsecond_v2/train_stage_b_v2.py` |
| v2数据集 | `training/simul_uniss/subsecond_v2/stage_b_v2_data.py` |
| streaming teacher | `training/simul_uniss/subsecond_v2/streaming_whispervq_teacher.py` |
| sidecar生成 | `training/simul_uniss/subsecond_v2/prepare_stage_a_v3_sidecar.py` |
| causal验证 | `training/simul_uniss/subsecond_v2/validate_stage_b_latent.py` |
| Phase3敏感性 | `training/simul_uniss/subsecond_v2/evaluate_phase3_token_streams.py` |
| 正式配置 | `configs/experiments/simul_uniss_subsecond_v2/stage_b_v2_causal_15shard_v1.env` |
| clone checkpoint | `checkpoints/simul_uniss_subsecond_v2/stage_b_v2_clone_pretrain_15shard_v1/` |
| 最终v2 checkpoint | `checkpoints/simul_uniss_subsecond_v2/stage_b_v2_prefix80_finetune_100k_v1/` |
| v2验证报告 | `reports/simul_uniss_subsecond_v2/stage_b_v2_prefix80_validation.json` |
| Phase3报告 | `reports/simul_uniss_subsecond_v2/stage_b_v2_prefix80_phase3_sensitivity.json` |

## 18. 给同事讲解的三分钟版本

可以直接使用下面这段话：

> UniSS原本使用WhisperVQ把源语音编码成GLM token，但它更适合整段推理或不断做prefix re-encode。我们为了做真流式同传，训练了一个约1.21亿参数的Causal Audio Student v2。它使用16层Emformer，每160毫秒处理一个chunk，允许80毫秒lookahead，并维护约2秒显式历史和内部cache。模型每80毫秒产生一个1280维latent，然后使用原WhisperVQ冻结的16384项codebook量化，所以后面的Phase3 Qwen不需要改接口。
>
> 训练分两步。第一步在15-shard共132万条数据上，用streaming版冻结WhisperVQ做2万步hidden和codebook蒸馏；第二步在10万条真实prefix数据上做1万步微调。模型除了GLM latent，还训练source CTC、target capacity和stability三个辅助head。
>
> 目前它的因果性和工程结构已经通过：future perturbation为0，cache和full causal token完全一致，active RTF约0.098，说明计算速度约为实时预算的十分之一；自身稳定token的p50/p95约320/480毫秒。但teacher exact token agreement只有约29.3%，correct-stable coverage只有31.25%，所以稳定不等于正确，质量门没有通过。
>
> 我们现在做v3修复，不改Student架构，而是重新生成双向平衡的50k+50k exact-prefix真实pre-VQ hidden，并与原clone hidden按1:1混训，再用双向Phase3 BLEU选checkpoint。v2是成功的流式架构原型和v3初始化基线，不是最终质量版本。

## 19. 常见问题

### Q1：Student v2还使用Whisper吗？

训练数据生成时使用冻结WhisperVQ做teacher；正式Student推理不需要运行原WhisperVQ。

### Q2：它是真流式还是伪流式？

Student本身是真因果、带cache的流式encoder，不会每次从句首重算，也不会使用被禁止的远未来音频。完整系统的Phase3和codec是否能在1秒内开始播放，需要另外看WAIT/WRITE与audio buffer指标。

### Q3：320 ms是不是最终同传延迟？

不是。320 ms是Student自身首个稳定GLM事件的中位数；最终译音还要经过commit、Qwen、BiCodec和播放器。

### Q4：为什么速度通过但质量没通过？

速度和因果性属于结构问题，已经解决；teacher token兼容性属于监督和优化问题，v2 prefix阶段缺少hidden、数据方向不平衡且发生clone监督替换。

### Q5：为什么不直接改Phase3适配Student错误token？

可以作为后续路线，但先提高Student与原codebook的兼容性，可以继续复用已经训练好的full198 Phase3，保持旧实验可复现，也能更清楚定位误差来源。

### Q6：Student v2现在能做demo吗？

可以。它可以展示真流式前端、cache更新和左右声道试听，但必须说明当前teacher agreement质量门未通过，不能将demo主观效果作为最终模型结论。

## 20. 最终结论

Student v2完成了从“不断重编码WhisperVQ prefix”到“有状态因果音频Student”的关键架构迁移：

- 真因果：通过；
- cache一致性：通过；
- 计算实时性：通过；
- 与原Phase3接口兼容：通过；
- teacher GLM token质量：未通过；
- 最终端到端亚秒高质量同传：尚未证明。

因此最准确的项目表述是：

> Student v2证明了UniSS源语音前端可以被改造成高效真流式架构；Stage-B-v3正在解决该架构在teacher表示兼容性和双向翻译质量上的剩余问题。
