# 06 · Megatron vs DeepSpeed 训练一致性：adapter/bridge 结构差异分析、设计与完善

> 记录日期：2026-08-07
> **状态更新（2026-08-18）**：本文 §2 的"当前实现差异"表与 §3 的六处缺口均为 **MR 57 修复前的历史快照**。当前 adapter 核心链（upstream 分支、adapter 构建与随机初始化注入、4 个 upstream 字段消费、freeze API 统一）已在 Megatron 侧接通并有真实训练验证（4 任务 forward/backward/optimizer step 正常）。**数值 parity 尚未闭环**：仍存在的确认差异（speaker 注入、Qwen2.5 `last` hidden 的 final norm 位置、loss 全局聚合口径等）与验收标准统一记录在 `docs/CUSTOM_BACKEND_PARITY_REVIEW_SPEC.md`，请以该文档为准。本文中基于 synthetic fixture 的"逐位 parity（|Δ|=0）"仅覆盖 legacy bridge 子图，不代表 adapter 链路已证明一致。
> 关联：`docs/01_architecture.md`（bridge 拼接 vs adapter 叠加的结构对比）、`PERFORMANCE_OPTIMIZATION_PLAN.md`（训练健康度对比表）
> 主线：发现两后端收敛差异 → 定位根因（训练目标不同）→ 分析结构差异 → 设计并完善 megatron adapter。

---

[TOC]

---

## 1. 背景:换后端后训练表现完全不同



`PERFORMANCE_OPTIMIZATION_PLAN.md` §4.3「训练健康度对比表」，相同有效 batch（4096）、相同步数：

| 指标 | megatron（otty） | deepspeed（bxa0） |
|---|---|---|
| loss | 20.98 → **8.95** | ~23.5（几乎不动） |
| codec_token_acc | 0.05% → **25.9%** | ~0.21%（不动） |
| eos_boundary_top1_acc | 0.15% → **59.8%** | ~7.6% |

**关键疑问**：`custom/` 只是自定义了模型结构/配置，两后端从**同一 checkpoint**（`qwen2_5_3b_omnispeak_a`）、**同一份模型代码**、**同一 template / dataset / codec tokenizer** 出发——输入完全相同，差异只可能来自"训的模型结构 / 训练目标不同"。下面逐环节核对两后端的实际实现。

---

## 2. 两后端当前实现差异

> **历史快照（2026-08-07，MR 57 修复前）**：下表记录的是当时 Megatron adapter 侧的六处缺口，已由 MR 57 修复（见 §3/§4 与文首状态更新）。保留原文供追溯，**不要据此判断当前实现**；当前状态见 `docs/CUSTOM_BACKEND_PARITY_REVIEW_SPEC.md`。

> 结构层面的 bridge/adapter 差异见 `docs/01_architecture.md`；此处聚焦**两后端对同一结构的实现完整度**。

| 环节 | DeepSpeed | Megatron | 后果 |
|---|---|---|---|
| forward 结构 | 完整 forward，读 `use_upstream_talker_input` 切 bridge/adapter | `mcore_talker_loss.compute_codec_loss` **硬编码 bridge 拼接**（`cat([sem_block, codec_emb])`） | adapter 模式**静默退化成 bridge** |
| `thinker_adapter` 子模块 | 构建并使用（叠加） | `prepare_model` 未构建、`module_mapping` 无 key、`compute_codec_loss` 未使用 | 参数不在训练图，无法加载/保存 |
| 模型加载 | `load_state_dict(strict=False)`：checkpoint 缺 adapter 权重 → **随机初始化**，正常启动 | mcore-bridge `_set_module` **严格加载**（`assert missing_keys==0`）：缺 adapter 权重 → **断言失败** | 声明 adapter 即启动崩溃 |
| batch 字段消费 | template 注入 `thinker_text_labels`/`target_text_ids`/`codec_bos_index`/`full_add_prompt_len`，白名单保活 | `_patched_forward_step` 只 pop `CODEC_KEYS` 等，**对 4 个 upstream 字段零消费** | upstream 监督数据丢失 |
| freeze 语义 | `--freeze_parameters_ratio 1.0` + `--trainable_parameters model.thinker_adapter` | `freeze_llm/vit/aligner` 分类冻结 | 无法表达"只训 adapter" |
| 可训练参数集 | **仅 thinker_adapter**（~25M） | **bridge + talker + codec 头**（2069M，otty args dump 实证） | **训练的是两套不同模型** |
| **梯度链完整性**（零标签容错） | **talker forward 与 loss 计算分离**：`codec_input_ids` 存在即执行 talker forward，loss 为零时用 `ref.sum()*0.0` 兜底（`main_codec_logits` 通过 `QwenOmniSpeakOutput` 暴露给训练器，指标计算自然触发梯度） | **talker forward 与 loss 计算耦合**：`codec_loss` 被包在 `if bool(num_tokens > 0)` 分支里，`num_tokens==0` 时 `_weighted_codec_cross_entropy` 未被调用 → `main_codec_logits` 的梯度链断裂（`main_codec_logits` 是局部变量，未返回给训练器） | **overlap_grad_reduce 不兼容**：Megatron 0.17+ 要求每个 trainable param 在每个 micro-batch 都登记 grad-ready，`num_tokens==0` 时 talker 参数无梯度 → `param_and_grad_buffer.py:253 assert` 失败 |

**补充：bridge 为何在两后端一致**
megatron 已为 bridge 做了完整 mcore 适配，与 deepspeed **逐位一致**（`bridge→cat→block_causal→talker→codec_head`），`test_mcore_talker_loss.py` 验证数值 parity（`|Δ|=0`）。因此差异全部集中在 **adapter 路径**——上表前六行即 megatron 在 adapter 侧的六处缺口。

**结论**：两后端修复前训练的是**两套不同的模型结构 + 不同参数集**，收敛差异大是必然结果，而非后端本身的性能差异。

---

## 3. 解决方案

### 3.1 设计原则

> 用户要求：两后端用**同一语义参数**控制走哪个结构，不因后端框架写特殊逻辑，训练逻辑保持"单一事实源"。

统一的两把钥匙（均由 `TRAIN_TARGET` 生成）：
- **forward 结构**：`use_upstream_talker_input`（bridge / adapter）。
- **freeze 语义**：`--freeze_parameters_ratio 1.0` + `--trainable_parameters`。

### 3.2 修复实现差异对照（DeepSpeed 为基准，Megatron 补齐）

| 环节 | DeepSpeed（基准，无需改） | Megatron（修复后实现） |
|---|---|---|
| forward 结构 | 读 `use_upstream_talker_input` 走 upstream 叠加 | `compute_codec_loss` 增加 `use_upstream_input` 分支，复刻 deepspeed `_build_upstream_text_channel`（reply 抽取→adapter→text_projection→散射→full-add）；bridge 分支保持不变 |
| `thinker_adapter` 子模块 | modeling 构建 | `prepare_model` 构建 `Qwen3TTSThinkerAdapter` + `module_mapping` 增加 `'model.thinker_adapter'` |
| 模型加载 | `strict=False` 缺权重随机初始化 | **sidecar-lazy 注入**（见 §3.3） |
| batch 字段 | template 注入 4 个 upstream 字段 | `_patched_forward_step` 按开关 pop 4 字段并透传给 `compute_codec_loss` |
| freeze / 可训练 | `freeze_parameters_ratio` + `trainable_parameters` | 弃用 `freeze_llm/vit/aligner`，改用与 deepspeed 同一 API；`TRAIN_TARGET` 经 `_trainable_prefix`（deepspeed=`model`、megatron=`visual`）派生可训参数 |

> **命名空间差异（修复时需对齐）**：deepspeed 在 HF 命名空间 `model.*`；megatron 把 bridge/talker/thinker_adapter 挂在 `mm.visual` 下，前缀为 `visual.*`。megatron thinker 解冻特例为 `language_model`（mcore GPTModel 命名空间），非 `visual.thinker`。
>
> **freeze 语义为何能统一**：megatron 底层 `prepare_mcore_model`（`swift/megatron/utils/utils.py`）本就支持 `--freeze_parameters_ratio + --trainable_parameters`（与 deepspeed 同一 API），`MegatronSftArguments` 含全部四个字段、`megatron sft --help` 参数同名同语义——因此只需弃用 `freeze_llm/vit/aligner` 分类冻结即可对齐，无需改 megatron 源码。

### 3.3 模型加载语义对齐（sidecar-lazy 注入）

- **目标**：让 megatron 对齐 deepspeed `strict=False` → 随机初始化语义，改动最小、不改 mcore-bridge 源码。
- **做法**：在 `_convert_pre_process` 中，于 `super()` 之前调用 `_inject_thinker_adapter_init`——仅当 `use_upstream_talker_input=True` 且 checkpoint 确无 `model.thinker_adapter.*` 时，用与 `prepare_model` 相同超参（trunc_normal std=0.02、bias=0、LayerNorm 1/0）构造临时 adapter 并随机初始化，以 `model.thinker_adapter.` 前缀 + `LazyTensor` 注入 `state_dict`，使严格加载的 `_set_module` 找齐 6 个 key。
- **效果**：与 deepspeed 完全一致——adapter 从随机初始化起步训练，经标准 visual 保存路径回写；与既有 `_inject_sidecar_lazy`（text_projection）同模式。

### 3.4 `overlap_grad_reduce` 兼容性分析（Megatron 0.17+ 梯度链约束）

> 背景：`PERF_TUNING.md` 二章指出 `overlap_grad_reduce` 暂锁 false，根因是 Megatron 0.17.0 引入的「首 batch 每个 trainable param 必须登记 grad-ready」不变量。本节深入分析该约束与 codec_loss 实现的冲突根因，并给出修复方案。

#### 3.4.1 Megatron 0.17+ 的严格约束

Megatron 0.17.0 在 `_ParamAndGradBucketGroup.reset()` 中引入不变量：

```python
assert len(self.per_param_grad_ready_counts) == len(self.params)
```

**语义**：第一个 batch 结束时，bucket group 中**所有** trainable param 都必须被 backward hook 标记过至少一次 grad-ready。该集合固化为 `golden_per_param_grad_ready_counts`，作为后续所有 batch 的基准——一旦某个 batch 中某个 param 的 grad-ready 计数与 golden 不一致（缺少或多余），就触发 assert 失败。

**为何需要这个约束**：`overlap_grad_reduce=True` 在反向传播过程中逐层发起 AllReduce（不等整个 backward 结束）。为了精确追踪每个参数的梯度是否已完成 AllReduce，Megatron 需要确保每个 param 在每个 micro-batch 都有确定的梯度状态。

**Megatron 0.18.0（main 分支）仍然保留此约束**，未提供 `allow_unused` 或类似豁免机制。

#### 3.4.2 冲突根因：DeepSpeed vs Megatron 的梯度链差异

| 维度 | DeepSpeed（`modeling_qwen_omnispeak.py`） | Megatron（`mcore_talker_loss.py`） |
|---|---|---|
| **talker forward 触发** | `codec_input_ids is not None` → 执行 talker forward | 同左 |
| **loss 计算触发** | `codec_labels is not None` 且 `(effective_labels != -100).any()` → 计算 loss | 同左 |
| **零标签兜底** | `loss = ref.sum() * 0.0`（`main_codec_logits` 通过 `QwenOmniSpeakOutput` 返回给训练器，指标计算自然触发梯度） | `codec_loss = None`（`main_codec_logits` 是 `compute_codec_loss` 的局部变量，未返回给训练器） |
| **梯度链** | **始终完整**：talker forward 执行 → `main_codec_logits` 有值 → 指标计算（`argmax`+`sum`）或 `ref.sum()*0.0` 触发梯度 → talker 参数有梯度登记 | **num_tokens==0 时断裂**：`_weighted_codec_cross_entropy` 未被调用 → `main_codec_logits` 无梯度 → talker 参数无梯度登记 |

**关键差异**：DeepSpeed 把 talker forward 和 loss 计算**分离**了——forward 只要 `codec_input_ids` 存在就执行，loss 为零时用 `ref.sum()*0.0` 兜底。Megatron 把 loss 计算包在 `if bool(num_tokens > 0)` 分支里——`num_tokens==0` 时 `_weighted_codec_cross_entropy` 未被调用，虽然 talker forward 已执行，但 `main_codec_logits` 没有连接到任何 loss，梯度链断裂。

#### 3.4.3 修复方案

修复只需删掉 `if bool(num_tokens > 0)` 分支，**始终调用 `_weighted_codec_cross_entropy`**：

```python
# 修复前（梯度链断裂）：
codec_loss: Optional[torch.Tensor] = None
if bool(num_tokens > 0):
    codec_loss = _weighted_codec_cross_entropy(...)

# 修复后（梯度链完整）：
codec_loss = _weighted_codec_cross_entropy(
    main_codec_logits,
    effective_labels,
    eos_id=eos_id,
    eos_weight=_codec_eos_loss_weight(hf_config),
)
```

`_weighted_codec_cross_entropy` 内部已处理零标签情况（`valid_count==0` 时返回 `flat_logits.sum() * 0.0`）——**零 loss 但有梯度**。这样：

1. `main_codec_logits.sum()` 触发梯度 → `codec_head`/`talker.model()`/`bridge` 参数有梯度登记
2. 乘以 0.0 → loss 为零 → 不影响收敛
3. 所有 DP rank 执行相同 forward 路径 → 梯度集合对称 → 无死锁

**在当前 bridge-alignment 数据集下**，所有 batch 都有 `codec_input_ids` 且 `num_tokens > 0`（`_encode` 中 `raise ValueError` 过滤了无 target_audio 的样本），所以 `num_tokens==0` 几乎不会触发。但修复后即使遇到零标签样本（如 lazy_tokenize error row），也能保证梯度链完整，兼容 `overlap_grad_reduce`。

#### 3.4.4 修复后的预期收益

开启 `overlap_grad_reduce=true` 后，梯度 AllReduce 与反向计算重叠：
- 当前 nsys 诊断：NVTX AllReduce 占稳态时间 **94.8%**（通信是绝对瓶颈）
- 开启 overlap 后预计降至 **40-50%**，训练速度可再提升 **2-3×**
- 配合已生效的 `overlap_param_gather` + `align_param_gather`（patch），通信-计算重叠完整

---

## 4. 详细代码变动

> MR 57（`feature/deferzhou/adapter_megatron` → `master`），共 5 个代码文件。`+` 为新增行、`-` 为删除行。

### 4.1 `custom/qwen_omnispeak/mcore_adaptation.py`（+95 −3）

| 改动 | 说明 |
|---|---|
| import `Qwen3TTSThinkerAdapter` | 顶层 import（含 `Optional` 类型） |
| `module_mapping` 增加 `'model.thinker_adapter': 'thinker_adapter'` | 让 adapter 权重经标准 visual 路径加载/保存、纳入 DP 训练图 |
| 新增静态方法 `_infer_thinker_hidden_size` | 从 `thinker_config.text_config.hidden_size` 推断（回退顶层 `hidden_size`），与 deepspeed 同名方法一致 |
| `prepare_model` 构建 `thinker_adapter` | 仅 `use_upstream_talker_input=True` 时构建；超参 `thinker_h`/`text_h`/`adapter_hidden_size`/`adapter_dropout` 与 deepspeed 对齐 |
| 新增 `_inject_thinker_adapter_init` | 落地 §3.3：CPU 实例化 adapter（`_init_weights`：trunc_normal std=0.02、bias=0、LayerNorm 1/0），仅当 `use_upstream_talker_input=True` 且 checkpoint 无 `model.thinker_adapter.*` 时，以 `LazyTensor` 注入 `hf_state_dict` |
| `_convert_pre_process` 调用注入 | 在 `super()._convert_pre_process(...)` **之前**、且独立于 `_inject_sidecar_lazy` 单独判断 `to_mcore` 时调用 |

### 4.2 `custom/qwen_omnispeak/mcore_talker_loss.py`（+294）

| 改动 | 说明 |
|---|---|
| 新增模块级 `_build_upstream_text_channel` | megatron 镜像 deepspeed `modeling._build_upstream_text_channel`：reply 抽取（`text_labels != -100`）→ `adapter(reply_hidden+reply_embed)` → `text_projection` → 散射到 `codec_len`（`codec_bos_index + k`）→ full-add；注释要求与 deepspeed 保持逐位一致 |
| `compute_codec_loss` 增加 upstream 分支 | 新增 `use_upstream_input`/`text_labels`/`codec_bos_index`/`target_text_ids`/`full_add_prompt_len` 参数；upstream 走 `full_emb = codec_emb + text_channel`（`sem_len=0`，`codec_hidden` 取全序列）；bridge 分支完全不变 |
| `_patched_forward_step` 消费 4 字段 | 从 `mm.visual.hf_config.use_upstream_talker_input` 判定开关，在 `model(**data)` **之前** pop `thinker_text_labels`/`codec_bos_index`/`target_text_ids`/`full_add_prompt_len` 并透传给 `compute_codec_loss`（避免泄漏为 GPTModel 的未知 kwarg） |

### 4.3 `custom/qwen_omnispeak/test_mcore_adaptation.py`（+1）

断言 `module_mapping['model.thinker_adapter'] == 'thinker_adapter'`。

### 4.4 `pipelines/bridge_alignment/train_speech_to_speech_translation.sh`（+51 −43）

可训参数逻辑由"deepspeed / megatron 两套分支"重构为**两后端共用**：

- `_trainable_prefix` 区分命名空间：deepspeed=`model`、megatron=`visual`。
- 按 `TRAIN_TARGET` 生成 `_trainable_params` / `_trainable_regex`：adapter → `{prefix}.thinker_adapter`；bridge → `{prefix}.bridge`。
- `UNFREEZE_TALKER_LAYERS`（0/2/8/28）派生 talker 层 regex；`TALKER_TRAINING_SCOPE`、`UNFREEZE_THINKER`/`UNFREEZE_MTP`/`UNFREEZE_CODEC_HEAD` 复用同一前缀拼接。
- `UNFREEZE_THINKER` 特例：megatron 的 thinker 是 `language_model`（mcore GPTModel 命名空间），非 `visual.thinker`。

### 4.5 `pipelines/bridge_alignment/common_args.sh`（+25）

| 改动 | 说明 |
|---|---|
| SCENE dense `_MBS` 8 → **16** | 注释记录实测依据：MBS=32 在 step0 thinker forward（qwen3 q_proj）OOM（138.71/139.80 GiB，rank1）——thinker 虽冻结但前向 attention 激活仍随 MBS 线性增长；MBS=8 正常、16 为折中；整除 `4096 % (16×8)=0`、`num_microbatches=32` |
| 冻结策略 `FREEZE_LLM/VIT/ALIGNER` → `FREEZE_PARAMETERS_RATIO`（默认 1.0） | 弃用分类冻结（无法表达"只训 adapter"） |
| `run_bridge_alignment_megatron` 删 `--freeze_llm/vit/aligner` | 改组装 `trainable_args`：`--freeze_parameters_ratio 1.0` +（可选）`--trainable_parameters ${_trainable_params}` +（可选）`--trainable_parameters_regex ${_trainable_regex}`，透传给 `megatron sft` |


---

## 附：关键文件与代码位置

| 文件 | 关键位置 |
|---|---|
| `custom/qwen_omnispeak/mcore_adaptation.py` | `prepare_model`(230-274)、`module_mapping`(178-184)、`_infer_thinker_hidden_size`(215-228)、`_inject_sidecar_lazy`(84-120)、`_inject_thinker_adapter_init`(122-166)、`_convert_pre_process`(60-82) |
| `custom/qwen_omnispeak/mcore_talker_loss.py` | `_build_upstream_text_channel`(255-420)、`compute_codec_loss`(426-665，含 `num_tokens>0` 分支分析)、`_combined_loss_func`(723-769)、`_patched_forward_step`(772-...) |
| `custom/qwen_omnispeak/modeling_qwen_omnispeak.py` | `Qwen3TTSThinkerAdapter`(199-228)、`_build_upstream_text_channel`(1420-1628)、upstream forward 分支(1877-1955) |
| `pipelines/bridge_alignment/train_speech_to_speech_translation.sh` | 可训参数逻辑(89-136) |
| `pipelines/bridge_alignment/common_args.sh` | megatron 分支(330-338)、`run_bridge_alignment_megatron`(486-...) |
