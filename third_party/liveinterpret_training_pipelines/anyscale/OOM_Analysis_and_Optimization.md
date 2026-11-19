# Qwen-OmniSpeak 分布式训练 OOM 与死锁原因梳理

> 本文只做现象归纳、原理解释和原因定位，不给解决方案。  
> 当前最重要的判断：**OOM 不是 codec 本身导致的；codec 只是第一个显式申请临时显存、从而暴露"常态显存占用过高、没有余量"的位置。**

---

## 0. 结论摘要

### 0.1 已确认的事实

1. 当前至少有两个 OOM 现场：
   - **现场 A：codec tokenizer 临时显存申请 OOM**。发生在目标音频编码阶段，通常是 tokenizer 的 Conv1d / Mimi 编码临时 workspace 需要额外显存，但 GPU 已被训练常态占用挤满。
   - **现场 B：`gradient_checkpointing=true` 后，反向/重计算/同步附近尝试申请约 35–37 GiB OOM**。日志表面看像"梯度同步时申请巨大显存"，但已有诊断证伪它不是 ZeRO-2 优化器状态未分片。
2. ZeRO-2 的关键分布式状态在诊断任务中是正常的：
   - `WORLD_SIZE=16` 正确；
   - `dist.get_world_size()` 返回 16；
   - DeepSpeed `dp_size=16`；
   - freeze 生效，仅约 `213.9M` 参数可训练；
   - ZeRO-2 optimizer state 按 16 卡分片后每卡只有百 MB 级。
3. 因此，**常态显存 50+ GiB（GC=true）/ 70+ GiB（GC=false）是异常现象**。按当前训练范围估算，模型权重 + optimizer + 梯度的稳定占用不应该达到这个量级。
4. 当前代码中存在两个会放大训练常态显存的模型侧机制：
   - `custom/qwen_omnispeak/modeling_qwen_omnispeak.py` 会拼接 `semantic block + codec embedding`，并为 talker 构造 dense 4D attention mask `[B,1,T,T]`；`T = sem_len + codec_len`，随目标音频长度平方增长。
   - `Qwen3TtsTalkerModel.forward()` 和 `Qwen3CodePredictorTransformer.forward()` 是手写 layer loop，没有在 forward 中使用 `self.gradient_checkpointing` / `_gradient_checkpointing_func`，因此 `--gradient_checkpointing true` 很可能没有覆盖这两段最关键的自定义计算路径。
5. 用户补充确认：单机与分布式使用同一份 `qwen2_5_3b_omnispeak_d` 模型目录和同一份 `bridge_alignment_joint.jsonl` 数据文件。因此，单机不 OOM 与分布式 OOM 的差异不应再优先归因于"模型/数据文件不同"，而应聚焦在启动方式、rank 数据切分/样本顺序、tokenizer 缓存状态、分布式 runtime 与各 rank 时序差异。
6. ZeRO-3 卡死发生在训练第一个 step 前、没有 loss，且死锁时显存仅约 17%。结合 `anyscale/deadlock_analysis.md` 的 py-spy 证据，可确认这是已定位的 `barrier↔new_group` 循环死锁：rank 0 已进入 `deepspeed.initialize()` 的 `new_group`，其余 rank 仍卡在 ms-swift `safe_ddp_context` 的无 tag `dist.barrier`，NCCL FIFO 队列错位导致永久等待；ZeRO-3 因额外的参数分片 collective 放大了该错位窗口。
7. CPU tokenizer / DataLoader 迁移后的卡死不是一次性立即死锁，而是先表现为训练中周期性卡顿，每次几分钟到十几分钟，最后才演化成数小时无进度的硬死锁；这更符合 rank 间 straggler 逐步累积、最终 collective 顺序错位的过程。

### 0.2 当前不应再使用的旧判断

以下判断已经被后续诊断推翻，本文不再采用：

| 旧判断 | 当前结论 |
|---|---|
| `36.74 GiB = Thinker 全量 Adam optimizer state，ZeRO-2 没分片` | 已证伪。`world_size=16`、`dp_size=16`、trainable 参数量均正常；optimizer state 不可能是 37 GiB/卡。 |
| `多节点 NCCL 初始化慢导致 get_world_size() 返回 1` | 已证伪。诊断日志中 `dist.get_world_size()` 始终返回 16。 |
| `freeze_parameters_ratio=1.0` 未生效 | 已证伪。诊断日志显示可训练参数仅约 `213.9M`。 |
| OOM 根因是 codec | 不成立。codec 是触发临时申请的位置，根因是常态显存过高导致没有余量。 |

---

## 1. 当前训练配置与关键代码路径

### 1.1 Anyscale 分布式入口

Anyscale job 使用：

- `anyscale/train/job.yaml`
  - `TRAIN_SCRIPT: pipelines/bridge_alignment/train_joint.sh`
  - `USE_RAY: true`
  - `compute_config: tokyo-h100-gpu:17`
- `anyscale/train/dispatch_train.py`
  - head 进程只负责调度；实际 `swift sft` driver 被 pin 到 worker；训练 actor 由 ms-swift Ray 逻辑创建。
  - 会从实际 alive worker / GPU 数推导 `RAY_DEVICE_GROUPS`，避免声明拓扑与真实拓扑不一致。

### 1.2 训练脚本

- `pipelines/bridge_alignment/train_joint.sh`
  - `MODEL_TYPE=qwen2_5_3b_omnispeak_d`
  - `ATTN_IMPL=flash_attention_2`
  - `PER_DEVICE_TRAIN_BATCH_SIZE=1`
  - `PER_DEVICE_EVAL_BATCH_SIZE=1`
  - `GRADIENT_ACCUMULATION_STEPS=1`
  - `DATALOADER_NUM_WORKERS=0`
  - `RESIDUAL_LOSS=true`
- `pipelines/bridge_alignment/common_args.sh`
  - `DEEPSPEED=zero2`
  - `--freeze_parameters_ratio 1.0`
  - `--trainable_parameters model.bridge`
  - `--trainable_parameters_regex 'model\.talker\.model\.layers\.[0-1]\.'`
  - `--gradient_checkpointing true`
  - 当 `DEEPSPEED=zero3` 且没有显式 `ALLOW_BRIDGE_ZERO3=true` 时，会回退到 `zero2`，说明当前代码已有"ZeRO-3 易 hang"的经验性防护。

### 1.3 codec tokenizer 设备

- `custom/codec_tokenizer.py`
  - 默认设备逻辑：若 `QWEN3_TTS_TOKENIZER_DEVICE` 未设置且 CUDA 可用，则使用 `cuda:${LOCAL_RANK}`。
  - 因此默认每个训练 rank 都会在本 rank GPU 上加载并运行一个 Qwen3-TTS tokenizer。

### 1.4 模型 forward 的显存关键路径

- `custom/qwen_omnispeak/modeling_qwen_omnispeak.py`
  - `QwenOmniSpeakModel.run_thinker()`：当 thinker 参数都冻结时，走 `@torch.no_grad()` 包裹的 `_run_thinker_no_grad()`。
  - `QwenOmniSpeakForConditionalGeneration.forward()`：
    1. 运行 thinker，取 `thinker_hidden_states` / `thinker_text_embeds`；
    2. 运行 bridge 生成 `sem_block`；
    3. 将 `sem_block` 与 `codec_emb` 拼接为 `full_emb`；
    4. 调 `build_block_causal_mask()` 构造 `[B,1,T,T]` mask；
    5. 调 `self.talker.model(inputs_embeds=full_emb, attention_mask=attn_bias)`；
    6. 若 residual loss 开启，额外跑 `code_predictor.forward_teacher_forced()`。

---

## 2. OOM 现场整理

### 2.1 现场 A：codec tokenizer 临时显存申请 OOM

现象：

- OOM 栈常落在 Qwen3-TTS tokenizer / Mimi encoder / Conv1d 相关位置。
- 日志中可见 GPU 已经有很高常态占用，例如 70+ GiB 已占用后，再申请若干 GiB 临时 workspace 失败。

解释：

1. tokenizer 在 GPU 上编码目标音频时需要临时 workspace。
2. 这部分 workspace 是"额外申请"，不是长期持有的模型训练显存。
3. 如果训练常态只占 20–30 GiB，codec 临时申请通常有足够余量。
4. 当前 OOM 是因为进入 tokenizer 前 GPU 已经被模型权重、激活、attention workspace、PyTorch cache 等占到 50–70+ GiB，剩余空间不足。
5. 因此不能把根因归咎于 codec；codec 是触发 OOM 的位置，不是造成常态显存异常的根因。

需要特别注意：

- `custom/template.py` 不再对 codec 序列做截断，完整 codec 进入 talker forward。

### 2.2 现场 B：GC=true 后反向附近申请 35–37 GiB OOM

典型日志：

```text
Tried to allocate 36.74 GiB
GPU 80.00 GiB total capacity
55.98 GiB already allocated
24.02 GiB free
```

早期误判：

- `36.74 GiB ≈ 3.3B params × 12 bytes`，看起来像 Thinker text backbone 的 fp32 Adam 三份状态。

后续诊断已证伪该判断：

- `WORLD_SIZE` 正确为 16；
- `dist.get_world_size()` 全部调用返回 16；
- DeepSpeed ZeRO-2 `dp_size=16`；
- 可训练参数约 `213.9M`；
- optimizer state 分片后每卡约百 MB 级，不可能产生 37 GiB 单卡申请。

当前解释：

- 这 35–37 GiB 更符合反向期间的临时激活 / attention workspace / checkpoint recompute 峰值，而不是 optimizer state。
- `gradient_checkpointing=true` 会减少一部分 forward 常驻激活，但 backward 时需要重算 forward；如果重算路径中存在 dense attention 或没有被 checkpoint 覆盖的自定义模块，就可能出现大额临时申请。
- 从 `52 GiB already allocated + 35 GiB temporary` 的形态看，问题不是单个临时申请"过大"这么简单，而是临时申请发生前常态已经过高。

### 2.3 GC=true 与 GC=false 的差异

观察：

| 配置 | 常态显存观察 | 含义 |
|---|---:|---|
| `gradient_checkpointing=true` | 50+ GiB | 有一部分激活被释放，但仍然远高于合理静态占用。 |
| `gradient_checkpointing=false` | 70+ GiB | 关闭 GC 后，更多 forward 激活常驻，说明显存主要差异来自激活而不是权重/optimizer。 |

解释：

- 权重、ZeRO-2 optimizer state、梯度 buffer 不会因为 GC 开关产生 20 GiB 级差异。
- GC 开关带来的大幅差异说明：**主要显存差距来自 forward/backward 激活与 attention workspace**。
- 当前自定义 talker / code predictor forward 没有显式执行 checkpoint 逻辑，因此 GC 对最重的自定义路径覆盖程度存疑。

---

## 3. 常态显存为什么异常高

### 3.1 理论稳定占用不应达到 50–70 GiB

在当前训练范围下：

- ZeRO-2 不切参数，因此每卡持有完整 bf16 模型权重；
- 但 optimizer state 和梯度只针对可训练参数，且 ZeRO-2 会分片；
- 诊断日志显示可训练参数约 `213.9M`，不是全量模型。

粗略量级：

| 组成 | 量级 | 说明 |
|---|---:|---|
| bf16 模型权重 | 10–15 GB 级 | 每卡完整持有。 |
| 可训练梯度 | < 1 GB | 仅 bridge + talker 前 2 层。 |
| Adam optimizer state | 百 MB 到数百 MB/卡 | ZeRO-2 按 world size 分片。 |
| CUDA context / allocator / NCCL | 数 GB | 正常固定开销。 |
| tokenizer 常驻 | 约 GB 级 | 默认在 `cuda:${LOCAL_RANK}`。 |

因此，如果只看权重和优化器，常态不应是 50–70 GiB。多出来的几十 GiB 只能来自：

1. 训练激活；
2. attention workspace；
3. checkpoint recompute 临时峰值；
4. PyTorch allocator reserved/cache；
5. tokenizer 临时 workspace 与训练显存峰值重叠。

### 3.2 talker 长序列 dense attention 是主要放大器

当前模型在 codec route 中做了以下操作：

```text
sem_block = bridge(thinker outputs)
codec_emb = talker.model.embed_codec(codec_input_ids)
full_emb = concat([sem_block, codec_emb], dim=1)
T = sem_len + codec_len
attn_bias = build_block_causal_mask(..., shape=[B,1,T,T])
talker_out = talker.model(inputs_embeds=full_emb, attention_mask=attn_bias)
```

`build_block_causal_mask()` 明确物化 dense mask：

- `allowed = torch.zeros(B, T, T, dtype=torch.bool, device=device)`
- `bias = bias.unsqueeze(1)`，得到 `[B,1,T,T]`

这里的关键不是 mask tensor 本身有多大，而是：

- 4D dense additive mask 进入 attention 后，attention 计算需要处理完整 `T × T` 关系；
- `T` 随 codec frame 增长，codec frame 又随目标音频时长增长；
- 所以显存随目标音频长度近似平方增长；
- 如果 attention backend 不能走最省显存的 flash 路径，就会产生更大的 score / softmax / backward workspace。

### 3.3 codec 序列截断（已移除）

`custom/template.py` 不再对 codec 序列做截断（原 `MAX_CODEC_FRAMES` 已删除）。完整 codec 序列直接进入 talker forward。

### 3.4 自定义 talker / code predictor 可能未被 GC 覆盖

当前代码事实：

- `Qwen3TtsTalkerModel.supports_gradient_checkpointing = True`；
- 但 `Qwen3TtsTalkerModel.forward()` 是手写：
  - 计算 `position_embeddings`；
  - `for layer in self.layers: out = layer(...)`；
  - 没有检查 `self.gradient_checkpointing`；
  - 没有调用 `_gradient_checkpointing_func`；
- `Qwen3CodePredictorTransformer.forward()` 同样是手写 layer loop，也没有显式 checkpoint 分支。

含义：

- 即使命令行传了 `--gradient_checkpointing true`，这两段自定义 forward 也未必像标准 HF `Qwen3Model.forward()` 那样被 checkpoint。
- 而这两段正好位于长 codec 序列路径上，是当前 OOM 最相关的计算区域。
- 这能解释为什么 GC=true 比 GC=false 低一些，但仍然高到 50+ GiB：GC 可能覆盖了部分上游或标准模块，但没有覆盖最重的自定义路径。

### 3.5 residual code predictor 的额外显存

`train_joint.sh` 默认：

- `RESIDUAL_LOSS=true`
- `RESIDUAL_LOSS_WEIGHT=0.3`

因此 forward 中会运行：

```text
self.talker.code_predictor.forward_teacher_forced(...)
```

该路径会把每个 codec frame 展开成 `G + 1` 个 codebook-group token，再跑 5 层 transformer：

- 主要复杂度随 `codec_len` 线性增长；
- 不像 talker self-attention 那样随 `T²` 增长；
- 但它仍然会增加常态激活和临时 workspace，尤其在 codec_len 很长时不能忽略。

---

## 4. 分布式训练与单机训练常态显存差距

### 4.1 已确认：模型目录与数据文件一致

用户补充确认，单机和分布式训练使用同一份模型目录与同一份数据文件。现场核查命令分别为：

```bash
kubectl debug node/hyperpod-i-088f8cefb5f164975 -it --image=busybox:1.36 --quiet -- chroot /host ls /opt/dlami/nvme/speech-language-model/liveinterpret_training_pipelines/models/qwen2_5_3b_omnispeak_d
```

```bash
kubectl debug node/hyperpod-i-088f8cefb5f164975 -it --image=busybox:1.36 --quiet -- chroot /host ls /opt/dlami/nvme/speech-language-model/liveinterpret_training_pipelines/datasets/jsonl/bridge_alignment_joint.jsonl
```

这条信息会显著收窄排查范围：

- 不应再把单机/分布式差异优先解释为"模型文件不同"；
- 不应再把单机/分布式差异优先解释为"训练数据文件不是同一份"；
- 仍需区分"同一份数据文件"和"训练时每个 rank 实际遇到的样本序列完全相同"这两个概念。

### 4.2 单机与分布式仍可能出现显存差距的变量

即使模型目录和 JSONL 文件相同，以下变量仍会导致显存曲线不同：

| 变量 | 为什么会影响常态显存 |
|---|---|
| rank 数据切分 | 分布式每个 rank 看到的是 shard 后的样本流；某个 rank 提前遇到长音频样本，就可能先出现高峰。 |
| sampler 顺序 | seed、shuffle、drop_last、dataset split 会影响长样本出现的位置。 |
| tokenizer cache 命中率 | 同一份数据在不同运行中，codec cache 是否已预热会改变 tokenizer 阶段耗时与显存峰值重叠位置。 |
| Ray / ms-swift actor 时序 | 多 actor 独立启动，数据加载、模型加载、tokenizer 初始化、forward 进入时机可能不同。 |
| 分布式 collective | 单机 direct run 可以没有 Ray actor / DeepSpeed 多 rank collective；分布式 run 需要所有 rank 按同样顺序进入 collective。 |
| allocator 历史状态 | 同一模型/数据下，之前的临时申请、reserved memory、碎片化状态会影响后续可分配大块显存。 |

因此，现在更准确的判断是：

```text
同模型 + 同数据文件
  ≠ 同一时刻、同一 rank、同一输入长度、同一 allocator 状态
  ≠ 单机 direct 与 Ray/DeepSpeed 分布式显存曲线完全一致
```

### 4.3 分布式本身的显存增量不应是 30–50 GiB

在 ZeRO-2 + 纯 DP 下：

- 参数每卡完整复制，单机/多机都一样；
- optimizer state 和梯度分片，卡数越多每卡越小；
- NCCL / reduce bucket / contiguous gradients 会有额外 buffer，但通常是 GB 内或少数 GB 级，且只跟可训练参数规模强相关；
- 当前可训练参数只有约 `213.9M`，不支持"分布式同步本身多占 30–50 GiB"的解释。

因此，更合理的解释是：

```text
同一份数据中存在长音频 / 长 codec 序列
→ 分布式 shard 或样本顺序让某些 rank 在特定时刻遇到高显存样本
→ 模型侧长序列激活和 attention workspace 过大
→ 常态显存高
→ codec 或 backward 临时申请时无余量
→ OOM
```

而不是：

```text
codec 本身有问题
或 Ray 本身固定多占几十 GiB GPU 显存
或 ZeRO-2 分片失效
或单机/分布式使用了不同模型、不同数据文件
```

### 4.4 仍需复现确认的对比项

为严格证明"分布式 vs 单机"的显存差距，需要同一组样本、同一 env、同一代码、同一模型下对比：

| 对比项 | 需要记录 |
|---|---|
| 数据样本 | `target_audio` 路径、音频时长、codec 原始帧数、截断后帧数、样本在 sampler 中的位置。 |
| 模型输入 | `sem_len`、`codec_len`、`T=sem_len+codec_len`。 |
| 显存阶段 | tokenizer 前后、model forward 前后、talker 前后、code predictor 前后、backward 前后、optimizer step 前后。 |
| 分布式状态 | `WORLD_SIZE`、`RANK`、`LOCAL_RANK`、DeepSpeed ZeRO stage、trainable 参数量。 |
| 数据路径状态 | 模型目录、JSONL 路径、codec cache 路径、cache 命中/未命中。 |
| allocator 状态 | `allocated`、`reserved`、`max_allocated`、是否存在大块连续显存不足。 |

这些是复现实验需要采集的事实，不是解决方案。

---

## 5. 为什么"梯度同步时"会看到 37 GiB 申请

### 5.1 它不像真正的梯度同步 buffer

当前可训练参数约 `213.9M`：

- bf16 gradient 全量约 `213.9M × 2 bytes ≈ 0.43 GB`；
- ZeRO-2 reduce-scatter / contiguous gradient buffer 即便有重叠，也不应达到 37 GiB；
- optimizer state 分片后更不可能达到 37 GiB。

因此，"梯度同步时申请 37 GiB"更可能是观察时机与栈位置造成的表象：

- backward 进入 checkpoint recompute；
- recompute 再次执行长序列 forward / attention；
- attention 或中间激活申请大 workspace；
- 由于这发生在 backward/step 附近，看起来像"梯度同步申请显存"。

### 5.2 为什么 GC=true 反而出现大额临时申请

`gradient_checkpointing=true` 的机制是：

- forward 少存激活，降低常驻显存；
- backward 需要重算部分 forward，增加临时峰值。

如果重算区域包含：

- 长 `codec_len`；
- dense `T×T` attention；
- 未被 checkpoint 正确覆盖的自定义 layer loop；
- residual code predictor；

就可能出现：

```text
常驻显存已经 50+ GiB
+ backward/recompute 临时 workspace 35+ GiB
> 80 GiB H100 容量
```

这比"ZeRO-2 分片失效"更符合已采集证据。

---

## 6. ZeRO-3 卡死原因梳理

### 6.1 ZeRO-3 与 ZeRO-2 的关键区别

| 维度 | ZeRO-2 | ZeRO-3 |
|---|---|---|
| 参数 | 每卡完整持有 | 参数分片，各 rank 只持有部分参数 |
| forward 前 | 不需要为每层 all-gather 全量参数 | 需要按模块 all-gather 参数 |
| 对 rank 同步要求 | 主要在梯度/optimizer 阶段 | forward/backward 中更频繁要求所有 rank 进入相同 collective |
| 对 straggler 敏感性 | 较低 | 更高 |

### 6.2 当前工程里 ZeRO-3 易卡住的原因（核心机制已由 py-spy 确认）

已有代码和文档显示，当前工程有多处会造成 rank 进入 collective 的时机不一致：

1. ms-swift dataset load / shuffle / preprocess 中有 `safe_ddp_context`，内部使用无 tag 的 `dist.barrier()`（源码 `swift/utils/torch_utils.py`，见 `deadlock_analysis.md §2.4`）。
2. Anyscale + Ray 下，每个 rank 是独立 actor，数据加载、样本处理、tokenizer 编码耗时可能不同。
3. ZeRO-3 在模型初始化和 forward 中需要更多 `new_group` / `all_gather` / 参数 materialization collective。
4. 只要某些 rank 已进入 ZeRO-3 参数 all-gather，而另一些 rank 仍在 dataset barrier、tokenizer、或其他阶段，就会形成等待环。

**这不是推测，而是已在 `anyscale/deadlock_analysis.md` 中用 py-spy 栈帧确认的 `barrier↔new_group` 循环死锁**。该文档记录了两次复现：

- `prodjob_pldld9x4v874xdsdzkbyxd1i6k`（qwen2_5_3b_omnispeak_d，多节点 16 GPU）；
- `prodjob_z25yehia4xk6t75i9k1mwy1f2x`（单节点 8 GPU）。

两次 py-spy 栈帧一致：

```text
rank 0:   _new_process_group_helper  (deepspeed.initialize → _configure_distributed_model
          → _get_data_parallel_group → new_group)  → 已进入 DeepSpeed init，等待其他 rank
rank 1-N: safe_ddp_context.__exit__ → dist.barrier  (shuffle_dataset / load_dataset)
          → 仍在 dataset barrier，等待 rank 0
=> 双方等待的 collective 不一致（new_group vs barrier），无法匹配，永久卡住
```

关键：`dist.barrier()` 底层是在 default process group 上对 dummy tensor 做 all_reduce，且**无 tag**；NCCL 用 FIFO 队列匹配，无法区分不同 `safe_ddp_context` 的 barrier。当 rank 0 先跑完 load_dataset 的多层 barrier、进入 `deepspeed.initialize()` 的 `new_group()` 时，其余 rank 还停在 barrier，队列错位即永久死锁。

**这一机制本身并非 ZeRO-3 独有**（上述复现里 ZeRO-2 也发生过），但 ZeRO-3 会显著放大它：ZeRO-3 在初始化和首个 forward 前需要额外的参数分片 / `all_gather` / `new_group`，使 rank 间时序错位窗口更大、更容易稳定触发。这与用户观察一致：ZeRO-2 通常能跑起来（之后才 OOM），而 ZeRO-3 更稳定地卡在训练循环开始之前。

### 6.3 tokenizer 与 ZeRO-3 的额外风险

`custom/codec_tokenizer.py` 中已有专门逻辑：加载辅助 Qwen3-TTS tokenizer 时，会临时 detach `HfDeepSpeedConfig` 并禁用 `zero.Init`。

代码注释说明原因：

- tokenizer 不是训练模型的一部分；
- 若 ZeRO-3 全局 hook 影响 tokenizer `from_pretrained()`，可能把 tokenizer 权重也 partition；
- 这会导致 tokenizer encode 时出现权重形状异常。

这说明当前工程里 ZeRO-3 不只是通信更复杂，还可能影响"训练模型之外的辅助模型加载"。这不一定是本次卡死的唯一原因，但它是 ZeRO-3 路线的已知风险点。

### 6.4 当前对 ZeRO-3 的原因定性

可以确认：

- ZeRO-3 卡死不应解释为"显存不够"本身；`deadlock_analysis.md` 记录死锁时 GPU 显存仅约 17%（~13.7 GB/进程，仅模型参数已加载，无 activation/optimizer state），进一步印证卡点在训练循环之前、与显存无关；
- 用户补充确认：ZeRO-3 卡死在训练第一个 step 前，没有 loss 日志输出；这与 `deadlock_analysis.md` 确认的死锁位置一致——卡在 `SwiftSft.run()` → `trainer.train()` → `deepspeed.initialize()` → `new_group()`，即"创建 DeepSpeed engine 的进程子组"阶段，尚未进入 DataLoader 创建和 training loop；
- 因此"卡在第一个 step 前、无 loss"不是新问题，而是本工程已用 py-spy 定位的 `barrier↔new_group` 循环死锁的典型表现；
- 具体 collective 已确认：rank 0 在 `new_group`（`_new_process_group_helper`），其余 rank 在 `safe_ddp_context` 的 `dist.barrier`；根因是无 tag barrier 在 NCCL FIFO 队列中与 `new_group` 错位匹配；
- 仍需复现确认的只是"ZeRO-3 相比 ZeRO-2 具体多引入哪些 collective、是否叠加 tokenizer ZeroInit 副作用"，而死锁的核心机制无需再复现即可确认。

> 注意：`deadlock_analysis.md §3g` 另有一条独立发现——早期 `custom/debug_hooks.py` 对 `dist.init_process_group` / `dist.barrier` 的 patch 会额外引入死锁。这是调试代码引入的次生原因，与 `barrier↔new_group` 的核心机制并列，排查 ZeRO-3 时需要区分当前运行是否加载了该调试 hook。

---

## 7. CPU tokenizer / DataLoader 路线为什么会死锁

### 7.1 现场现象

历史诊断记录中，CPU tokenizer 相关任务出现：

```text
节点 0: 7/8 rank 在 to_device → _post_encode → pre_forward_hook
       1/8 rank 在 dist.barrier() 或 CPU tokenizer (Mimi Conv1d)
节点 1: 8/8 rank 在 to_device → _post_encode → pre_forward_hook
```

也就是：

- 大多数 rank 已完成 CPU tokenizer；
- 少数 rank 仍在 CPU tokenizer 或 barrier；
- 先到的 rank 已经进入 model forward 前的 hook；
- hook / DeepSpeed engine 内部触发 collective；
- collective 需要全部 rank，但落后的 rank 尚未进入同一个 collective。

### 7.2 死锁链路

```text
CPU tokenizer 每个 rank 处理不同样本
  → 音频时长不同，CPU Mimi Conv1d 耗时不同
  → rank 间进度差扩大
  → 15 个 rank 先进入 pre_forward_hook / DeepSpeed forward
  → 触发需要全体 rank 的同步或参数相关 collective
  → 第 16 个 rank 仍在 tokenizer 或 barrier
  → 15 个 rank 等第 16 个 rank，第 16 个 rank又未进入同一 collective
  → 死锁
```

关键点：

- 问题不是"CPU tokenizer 慢"这么简单；
- 真正的问题是 CPU tokenizer 放大了 rank 间 straggler 差异；
- 分布式训练的 forward/backward collective 要求所有 rank 以一致顺序进入；
- 一旦某些 rank 进入 forward collective，而另一些 rank 还停在数据处理阶段，就会出现 collective 顺序错位。

### 7.3 为什么简单 barrier 没有消除死锁

历史尝试中，加入 `dist.barrier()` 后仍然死锁。原因是：

- barrier 只能同步"放置 barrier 的那个点"；
- 过了 barrier 后，各 rank 仍会继续执行 `_post_encode`、`to_device`、`pre_forward_hook`；
- 如果后续阶段仍存在耗时差异或不同 rank 进入不同 collective，死锁仍会发生；
- barrier 本身也是一个 collective，如果与后续 `new_group` / all-gather 顺序错位，反而也可能成为等待环的一部分。

### 7.4 与 ZeRO-3 死锁的共同本质

CPU tokenizer 死锁和 ZeRO-3 死锁的共同点：

```text
不是单个算子有 bug，
而是多个 rank 进入 collective 的顺序不一致（无 tag barrier + NCCL FIFO 队列错位）。
```

关键区别在于**发生阶段**：

| 路线 | 放大 rank 差异的来源 | 发生阶段 | 卡住位置 | 时间形态 |
|---|---|---|---|---|
| ZeRO-3 | 参数分片导致 init/forward collective 更频繁、更早出现 | 训练循环开始**之前**（DeepSpeed 初始化） | `deepspeed.initialize` → `new_group` vs `safe_ddp_context.barrier` | 一开始就卡死，无 loss 日志 |
| CPU tokenizer | 不同音频在 CPU 上 encode 耗时差异巨大，逐 batch 累积 | 训练循环**运行中**（每个 step 的数据准备 → forward） | `pre_forward_hook` / `to_device` vs 落后 rank 仍在 CPU tokenizer / barrier | 先周期性卡顿（几分钟到十几分钟），最后演化为永久死锁 |

这解释了用户观察到的两种不同时间形态：

- ZeRO-3 的死锁发生在 `deepspeed.initialize()`，属于**一次性、启动阶段**的 collective 错位，所以从一开始就没有 loss；
- CPU tokenizer 的死锁发生在**训练循环内**，每个 step 都要重新对齐一次 forward/backward collective。CPU 编码耗时随样本长度剧烈波动，某个 step 出现长音频时该 rank 落后，其余 rank 在 forward 前 hook 等待，就出现一次"卡几分钟到十几分钟"的卡顿；等落后 rank 追上后又能继续。随着长样本反复出现、straggler 差距不断累积，最终某一步彻底错位到无法匹配的 collective，演化为数小时的硬死锁。

---

## 8. 历史尝试路线与定位结论

> 本节只记录尝试后暴露出的事实，不评价是否应该继续采用。

### 8.1 ZeRO-3 切分模型

观察：

- 可以缓解参数常驻显存；
- 但卡死在训练第一个 step 前，没有 loss 日志输出；
- 当前代码中已默认禁止 bridge alignment 直接使用 ZeRO-3，除非设置 `ALLOW_BRIDGE_ZERO3=true`。

定位结论：

- 卡死更符合分布式 collective 顺序错位，而非单纯 OOM（死锁时显存仅约 17%）；
- 核心机制已在 `deadlock_analysis.md` 用 py-spy 确认为 `barrier↔new_group` 循环死锁：rank 0 进入 `deepspeed.initialize()` 的 `new_group`，其余 rank 仍卡在 `safe_ddp_context` 的无 tag `barrier`；
- ZeRO-3 引入更多参数 all-gather / new_group，使已有 rank 不同步问题更容易、更稳定地暴露；
- 仍待复现的仅是 ZeRO-3 相对 ZeRO-2 的额外 collective 清单与 tokenizer ZeroInit 是否叠加影响，死锁本身机制无需再复现。

### 8.2 tokenizer / dataloader 迁移到 CPU 并发

观察：

- GPU codec 临时 OOM 缓解；
- 但训练一段时间后开始周期性卡顿，每次几分钟到十几分钟；
- 最后演化成训练或 eval 阶段硬死锁，数小时不再有进度；
- py-spy 显示多数 rank 已到 forward pre-hook，少数 rank 仍在 tokenizer / barrier。

定位结论：

- CPU 路线把 GPU 临时显存压力转移到 CPU，但放大了 rank 之间的数据处理耗时差；
- 死锁来自 rank 间进入 collective 的顺序不一致。

### 8.3 压缩 codec 序列（已移除）

~~原 `MAX_CODEC_FRAMES` 截断已删除~~。当前不做 codec frames 限制，完整 codec 序列进入 talker forward。以下为历史记录：

- 进入模型 forward 的 codec 序列变短；
- talker attention 激活下降；
- 但 tokenizer 仍对完整目标音频做 encode，所以 tokenizer 临时显存峰值不受截断影响；
- 若 dataloader / tokenizer 进程 OOM 后被重启，主训练可能不中断，但吞吐会显著下降。

定位结论：

- 该路线说明模型侧长 codec 序列确实是显存重要变量；
- 它不能证明 codec 是根因；
- 它也不能解释或消除训练常态显存异常高的全部来源。

---

## 9. 当前问题链路图

```text
完整/长目标音频样本
  → codec_len 增大
  → sem_len + codec_len = T 增大
  → talker dense attention 处理 T×T
  → 自定义 talker/code predictor 未明确 checkpoint
  → forward/backward 常态显存升至 50–70+ GiB
  → 进入 codec tokenizer 或 backward recompute 时需要额外临时 workspace
  → 80GB H100 无余量
  → OOM
```

分布式死锁链路：

```text
Ray/ms-swift 多 rank 独立执行
  → 数据加载 / tokenizer / DeepSpeed init 耗时不一致
  → 某些 rank 先进入 DeepSpeed collective
  → 另一些 rank 仍在 barrier/tokenizer/数据阶段
  → collective 顺序不一致
  → NCCL/Gloo 等待无法匹配
  → 死锁
```

---

## 10. 需要复现才能最终确认的事项

以下事项需要复现现场采集证据后才能定论：

1. **分布式 vs 单机常态显存差距的严格对比**
   - 必须固定同一条或同一批样本；
   - 记录音频时长、原始 codec 帧数、截断后帧数、`sem_len`、`codec_len`；
   - 分别采集单机 direct、单机 Ray、两节点 Ray 的每阶段显存。
2. **37 GiB 申请的准确调用栈**
   - 需要定位它发生在 checkpoint recompute、attention workspace、还是 code predictor；
   - 当前已能排除 optimizer state 未分片，但还需要精确到具体模块。
3. **ZeRO-3 卡死的核心机制已确认，仅剩细化项**
   - 核心已定位：`barrier↔new_group` 循环死锁（`deadlock_analysis.md` 有 py-spy 栈帧与两个复现任务）；
   - 仍需复现细化的是：ZeRO-3 相对 ZeRO-2 具体多引入哪些 `all_gather` / `new_group`，以及 tokenizer 的 `zero.Init` detach 逻辑是否在 ZeRO-3 下叠加新的卡点。
4. **CPU tokenizer 死锁的 rank 时序**
   - 需要记录每个 rank 的 tokenizer 开始/结束时间、样本音频时长、进入 `pre_forward_hook` 的时间；
   - 目标是证明是否由单个或少数长样本 rank 拖慢导致 collective 顺序错位。
5. **GC 覆盖范围**
   - 需要确认 `Qwen3TtsTalkerModel`、`Qwen3CodePredictorTransformer` 的 layer 是否真的被 checkpoint；
   - 仅从当前源码看，forward 内没有显式 checkpoint 分支，因此覆盖范围存疑。

---

## 11. 参考文件

- `GPU_MEMORY.md`
- `anyscale/OOM_Analysis_and_Optimization.md`
- `anyscale/deadlock_analysis.md`
- `anyscale/train/job.yaml`
- `anyscale/train/dispatch_train.py`
- `pipelines/bridge_alignment/train_joint.sh`
- `pipelines/bridge_alignment/common_args.sh`
- `pipelines/bridge_alignment/common_args_megatron.sh`
- `custom/template.py`
- `custom/codec_tokenizer.py`
- `custom/qwen_omnispeak/modeling_qwen_omnispeak.py`
