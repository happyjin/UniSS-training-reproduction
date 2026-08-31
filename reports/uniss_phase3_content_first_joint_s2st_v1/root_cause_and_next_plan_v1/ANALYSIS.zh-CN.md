# UniSS 流式 S2ST：content-first 方案为何仍然失败 —— 根因分析与下一步详细方案

- 撰写时间：2026-08-31
- 分析对象：`uniss_phase3_content_first_joint_s2st_v1`（content-first 联合 SFT + 两轮 coverage GRPO）
- 对照对象：`uniss_phasea_coverage_constrained_grpo_v3`、`uniss_phasea_commit_complete_sft_rl_v4`（此前最优血脉）
- 证据来源：Codex 会话 `01a04876-dd46-75f1-9c57-a250cd095b39`、训练日志、`METRICS.json`、实验源码
- 声明范围：全部数字均为 **train-seen**（固定 15-shard 内的 64 条长 episode），不构成泛化结论

---

## 0. 一句话结论

**这一版不是"改进幅度不够"，而是方向性回退：content-first SFT 训出来的流式模型在所有内容指标上比它想改进的旧模型差 3–6 倍；随后的 GRPO 在数学上不可能修复它。**

两个独立的结构性原因：

1. **SFT 侧**：真正决定翻译内容的损失项 `runtime_text_content` 在整个 epoch 中只拿到 **435,315 / 40,057,425 = 1.09%** 的监督 token；语音 token 头（`microblock_token_accuracy`）全程停在 **1.5%** 准确率，等于没学会发音。同时从 offline Phase-3 v4 重新起步、只训 717 步，等于放弃了旧血脉已经积累的流式能力。
2. **RL 侧**：reward 的三个关键项在当前工作点上**梯度恒为零或饱和** —— 延迟项被 `complete >= 0.75` 的门控永久关闭、failure 项被 `min(4.0, ...)` 的上限永久饱和、coverage 惩罚在 3% 附近几乎是常数；而 group advantage 归一化的方差下限只有 `1e-4`，把 4 个几乎相同候选之间的噪声放大成满幅优势。GRPO 因此在拟合噪声，而不是在优化目标。

---

## 1. 当前状态的量化事实

### 1.1 本轮三臂结果（64 episodes × 4 candidates = 256 候选／臂）

| 指标 | pre_GRPO | Round 1 | Round 2 | 方向 |
|---|---:|---:|---:|---|
| reward mean | -10.482 | **-10.376** | -10.564 | ↑ 越大越好 |
| ASR teacher similarity | 0.0479 | 0.0484 | 0.0479 | ↑ |
| MT teacher similarity | 0.0258 | 0.0287 | **0.0300** | ↑ |
| translation_length_ratio | 0.0678 | 0.0719 | **0.0741** | ↑ |
| target_coverage | 3.06% | 3.12% | **3.24%** | ↑ |
| spoken_target_coverage | 3.01% | 3.10% | **3.21%** | ↑ |
| first_write p50 | 11.84 s | **9.60 s** | 10.24 s | ↓ |
| max internal silence mean | 28.46 s | 28.39 s | **27.41 s** | ↓ |
| audio_writes mean | 4.14 | 4.40 | **4.61** | ↑ |
| tts_failures mean | 0.137 | 0.059 | **0.039** | ↓ |
| RTF | 7.80 | 7.80 | 7.85 | ↓ |

两轮 GRPO 的净效果：coverage +0.18 个百分点，first_write p50 −1.6 s。**在 60–70 秒的长音频上，覆盖 3.2% 意味着基本没有翻译内容。**

### 1.2 与此前最优血脉的对比（同样 64×4 几何、同一套指标）

| 指标 | 旧血脉 `coverage_constrained_grpo_v3`<br>(PostR3 / FinalPostRound3) | 本轮 content-first<br>Round 2 | 倍数 |
|---|---:|---:|---:|
| ASR teacher similarity | 0.2611 | 0.0479 | **5.5× 差** |
| MT teacher similarity | 0.1782 | 0.0300 | **5.9× 差** |
| translation_length_ratio | 0.3390 | 0.0741 | **4.6× 差** |
| target_coverage | 17.09% | 3.24% | **5.3× 差** |
| first_write mean | 4.45 s | 19.63 s | **4.4× 差** |
| max internal silence mean | 18.89 s | 27.41 s | **1.5× 差** |
| audio_writes mean | 18.14 | 4.61 | **3.9× 差** |

单条最佳 demo 层面差距更大：旧血脉存在 first_write 1.92–2.88 s、最大空白 4.0–7.2 s、coverage 31%–48% 的样本；本轮最佳样本 coverage 仅 3.8%。

**这是本文最重要的一张表：本轮"解决方案"生产出的模型，比它要修复的模型差一个数量级。**

---

## 2. 根因分析

### 2.1 SFT 侧：五个结构性缺陷（全部有训练日志证据）

证据文件：`logs/uniss_phase3_content_first_joint_s2st_v1_formal1e_v1.log`（717 条 iteration 记录，全部逐条解析）

#### (A) 关键损失项被 token 饥饿：只有约 1% 的监督预算

| 量 | 全程合计 | 每步均值 |
|---|---:|---:|
| `supervised_tokens` | 40,057,425 | 55,868 |
| `runtime_text_supervised_tokens`（增量 MT 文本） | **435,315** | **607** |
| 占比 | **1.09%**（逐步比值均值 1.05%） | 单步最高仅 1.37% |

`CONTENT_FIRST_WEIGHTS`（[pretrain_content_first_megatron.py:40-58](../../../experiments/uniss_phase3_content_first_joint_s2st_v1/training/pretrain_content_first_megatron.py#L40-L58)）把 `runtime_text_content` 的权重设为 5.00，是全部 17 项里最高的。但**权重乘的是一个只有 1% token 的项**。全 epoch 只有 43 万个目标语言 token 参与增量翻译学习——这个量级连微调一个 0.36B 模型的翻译头都不够，更不用说从 offline 基座新建 144 张量的流式适配器。

同时 `phase3_replay`（权重 1.50）在 717 步里 loss 从 1.361 → 1.443，**没有下降**：replay 消耗了大量梯度预算，却既没有改善 replay 本身，也没有帮助流式任务。

> **结论：这不是"权重没调好"，是数据构造层面的监督信号缺失。加权重解决不了 1% 的 token 占比。**

#### (B) 语音 token 头等于没有训练

| 指标 | 前 10 步均值 | 后 50 步均值 |
|---|---:|---:|
| `microblock_token_accuracy` | 0.0126 | **0.0153** |
| `microblock_semantic_content` (loss) | 7.426 | 4.886 |
| iter 717 验证 PPL | — | **2,151.9** |

一个 epoch 之后语音 token 的下一步准确率只有 **1.5%**。这直接解释了为什么 `audio_writes` 只有 4.1 次／条、音频听起来是碎片：**模型不知道该发什么声。** 权重只给了 `microblock_semantic_content: 2.00`，且这一项与 `runtime_text_content` 抢同一批极少的 WRITE 事件。

#### (C) 三个已加权的损失项从未被触发过一次

| 损失项 | 设定权重 | 717 步实测 min/max/mean |
|---|---:|---|
| `real_prefix_kd` | 0.50 | **0 / 0 / 0** |
| `prefix_stability` | 0.50 | **0 / 0 / 0** |
| `speaker_consistency` | 0.25 | **0 / 0 / 0** |
| `deadline_forced_fraction`（诊断） | — | **0 / 0 / 0** |

方案文档里承诺的"前缀一致性约束（已提交前缀不可撤销）"和"跨片段 speaker 状态保持"**在训练中从未产生任何梯度**。`phrase_oracle_sessions(minimum_tokens=4)` 的数据视图没有为这些项生成目标。

`deadline_forced_fraction` 恒为 0 意味着**训练里没有任何 deadline 强制 WRITE 的样本，但推理运行时有 4 秒强制 WRITE**——训练与推理的动作分布不一致。

#### (D) teacher-forced 指标本身就不合格，谈不上"内容达标"

iter 717 验证集：

| 项 | loss | PPL |
|---|---:|---:|
| `runtime_text_content` | 3.847 | **46.9** |
| `interleaved_trajectory` | 5.315 | **203.4** |
| `ar_semantic_microblock` | 5.886 | **360.0** |
| `microblock_semantic_content` | 7.674 | **2,151.9** |

`runtime_text_token_accuracy` 后 50 步均值 **0.2206**：即使给它完美的 teacher 前缀，下一个译文 token 也只有 22% 猜对。自由运行时误差指数放大，coverage 落到 3% 完全是可预测的结果，不是意外。

**流水线在"content checkpoint 评估合格后才进入 GRPO"这一关卡上没有真正设门槛**——它只检查了训练不崩（无 NaN/OOM/skip），没有检查内容能力是否达标。

#### (E) 自由运行行为指标在训练过程中变差

| 诊断 | 前 10 步 | 后 50 步 | 变化 |
|---|---:|---:|---|
| `event_rollout_grammar_valid_fraction` | 0.656 | 0.655 | 持平（35% 非法） |
| `event_rollout_false_write_fraction` | 0.000 | **0.195** | 显著变差 |
| `event_rollout_all_wait_fraction` | 0.000 | **0.067** | 变差 |
| `event_rollout_first_divergence`（事件数） | 0.794 | **1.191** | 变差 |
| `safe_commit_f1` | 0.383 | **0.235** | 显著变差 |
| `runtime_action_accuracy` | 0.338 | 0.345 | 持平 |
| `runtime_predicted_eos_fraction` | — | 0.0168 | vs target 0.0078 → **过预测 2.15×** |
| `runtime_eos_precision / recall` | — | 0.149 / 0.321 | 极差 |
| `frontend_residual_rms` | 7.3e-5 | 0.0301 | 因果前端修正幅度仅 3% |

这是典型的"**只优化 teacher-forced CE，自由运行行为反而退化**"：token CE 在降（2.71 → 2.07），但 `first_divergence` 只有约 1 个事件、`safe_commit_f1` 从 0.38 掉到 0.24、EOS 过预测 2 倍。EOS 过预测直接对应报告里的 "premature END 5.28 次／episode"，也直接对应 `translation_length_ratio = 0.074`——**模型学会了尽早说完。**

### 2.2 RL 侧：GRPO 在当前工作点上数学上不可能起作用

#### (A) 可训练参数不包含内容能力

[pretrain_content_first_grpo.py:95-140](../../../experiments/uniss_phase3_content_first_joint_s2st_v1/training/pretrain_content_first_grpo.py#L95-L140)：144 个 `true_subsecond_lora` + 77 个 objective 张量 + Qwen 主干 + Whisper 前端全部冻结，唯一可训练的是顶 8 层（layer 16–23）rank-16 的 `quality_grpo_lora.policy`，lr 5e-6，99 步。

设计意图（不破坏已有能力）是对的，但前提是"已有能力"存在。当 ASR 相似度只有 0.048、语音 token 准确率 1.5% 时，**冻结内容 = 冻结失败**。RL 只能重排一个空集合的发送时机。

#### (B) 延迟奖励的门控永久关闭

[reward.py:89-96](../../../experiments/uniss_phasea_coverage_constrained_grpo_v3/training/reward.py#L89-L96)：

```python
eligible_for_latency = (complete >= 0.75 and mt >= mt_target and asr >= asr_target)
latency_term = (-0.15*first_penalty - 0.25*silence_penalty) if eligible_for_latency else 0.0
```

当前 `complete` 均值 0.031、p95 0.074，**256 个候选中 0 个达到 0.75**。因此 `latency_term ≡ 0`：**对 first_write 与 internal silence 的梯度恒等于零**。而这两项恰恰是本轮想要改善的指标。Round 1 观察到的 first_write 提前 2 秒，是其它项的副作用，不是被优化出来的。

这个门控当初是为了防止"早说两个词骗低延迟"的 reward hacking，方向正确，但**把门槛设在 0.75 而模型只有 0.03，等于把奖励函数的这一半直接删掉**。

#### (C) failure 惩罚永久饱和

[reward.py:79-83](../../../experiments/uniss_phasea_coverage_constrained_grpo_v3/training/reward.py#L79-L83)：

```python
failures = min(4.0, premature_end_count + tts_failure_count + 2.0*invalid_semantic_fraction)
```

`premature_end_count` 均值 **5.28**，单独一项就已超过上限 4.0。于是对几乎每个候选，`failures = 4.0`、惩罚恒为 `-6.0`，**导数为零**。

**最直接导致译文过短的失败模式（premature END），在 reward 里完全没有梯度。**

#### (D) coverage 惩罚在当前区间近似常数

`terminal_coverage_penalty = 8.0 * (0.80 - complete)²`（[reward.py:85](../../../experiments/uniss_phasea_coverage_constrained_grpo_v3/training/reward.py#L85)）

| complete | penalty |
|---:|---:|
| 0.024 (p50) | 4.82 |
| 0.031 (mean) | 4.74 |
| 0.074 (p95) | 4.22 |

在 −10.5 的总 reward 里，组内候选之间这一项的全部差异只有约 **0.6**。反向拆解 reward：`-4.74`(coverage) `-6.00`(failures 饱和) `+~1.4`(quality 正项) ≈ `-9.3`，其余由 asr/mt shortfall 补齐 —— **reward 的 ~90% 幅度来自两个几乎不随策略变化的常数项。**

#### (E) advantage 归一化把噪声放大成满幅信号

[event_credit.py:92-99](../../../experiments/uniss_phasea_coverage_constrained_grpo_v3/training/event_credit.py#L92-L99)：

```python
def normalize(values, epsilon=1e-4):
    scale = max(float(epsilon), variance**0.5)
    return [(v - mean)/scale for v in values]
```

方差下限只有 `1e-4`（标准差 0.01）。一组 4 个候选如果 raw return 差异只有 0.01 量级（当前正是如此），归一化后 advantage 依然是 ±1.2~1.5 的满幅值。**退化组与信息组被同等对待。**

叠加 [event_credit.py:101-142](../../../experiments/uniss_phasea_coverage_constrained_grpo_v3/training/event_credit.py#L101-L142) 的设计：`mt` / `asr` / `tts` 三个 family 的 raw return **只由 terminal 指标决定**，同一候选内所有 MT token 共享一个完全相同的 advantage，没有任何 per-event credit。于是"这个词该不该现在译"这个问题从未收到局部信号。

实测后果：Round 2 的 `control action clipped fraction = 40.4%`、`overall clipped fraction = 15.0%`——策略被大幅推动；而 Round 2 的 reward（−10.564）**比 Round 1（−10.376）更差**。这正是"在噪声上做大步更新"的签名。

### 2.3 运行时与工程侧：两个必须验证的隐患

#### (A) 跨血脉推理桥（未验证，优先级最高）

本轮为了让 content-first checkpoint 跑在旧 cascade 上，临时加了兼容桥（commit `ce748644`、`84e06bd2`、`696117fc`、`e5977b65`），核心是 [model_loader.py:173-190](../../../experiments/uniss_phase3_content_first_joint_s2st_v1/runtime/model_loader.py#L173-L190)：

```python
codes = _nearest_codes(objective, hidden)          # L2 argmin 重量化
adapted = objective.frontend_adapter(objective.codebook(codes.unsqueeze(0)))
```

而训练侧（[joint_model.py:177-178](../../../experiments/uniss_phase3_true_subsecond_deadline_full198_v1/training/joint_model.py#L177-L178)、[native_kv_backend.py:125-128](../../../experiments/uniss_phase3_event_rollout_joint_full198_v1/native_kv_backend.py#L125-L128)）消费的是**数据管线直接给出的离散 `frontend_ids`**。

已核对：GLM4 参考量化器 `vector_quantize`（`uniss/speech_tokenizer/glm4/modeling_whisper.py:68-78`）用的也是平方 L2 argmin，度量一致。**但"桥拿到的 hidden 是否恰好是参考量化器量化的那一层、那一次 pooling 后的 pre-VQ hidden"尚未验证。** 如果不是，模型在推理时看到的离散码与训练时不同，ASR 相似度 0.048 就可能主要是这个 bug 而不是能力缺失。

**这必须在做任何新训练之前先测。见 §4.1。**

#### (B) 自动流水线的"合格门"是伪门

`run_automatic_pipeline.sh` 的串联条件只校验进程成功与 checkpoint 存在，不校验内容质量。因此一个 coverage 3% 的 SFT checkpoint 被自动送进了两轮 GRPO 与三次 256 候选 rollout，消耗了约 20 小时 8 卡算力去优化一个不该进入 RL 的模型。

---

## 3. 为什么"content-first"这个方案本身解决不了问题

方案的诊断是对的（"能早发声但没有内容可说"），但落地时把结论执行反了：

| 方案承诺 | 实际执行 |
|---|---|
| 保留 Phase-3 多任务能力做锚 | `phase3_replay` loss 全程不降（1.361→1.443），锚没有起作用，只消耗预算 |
| 加强内容、弱化动作 | 内容项确实提权，但内容项的 token 占比只有 1.09%，提权无效 |
| 稳定短语提交监督 | `safe_commit_f1` 从 0.383 掉到 0.235；`prefix_stability` loss 恒为 0 |
| 声学监督让文字真正变成声音 | `microblock_token_accuracy` 1.5%，语音头未学会 |
| 先内容达标、再少量 RL | 没有内容达标门槛，3% coverage 直接进入 RL |
| RL 只优化提交时机 | 时机相关的 reward 项被门控关闭，梯度为零 |
| 从 Phase-3 最佳 checkpoint 出发 | 从 **offline** Phase-3 v4 iter_9075 出发，丢掉了旧血脉已训出的流式能力（ASR 0.26） |

**更根本的一点：**从 offline 基座重新长出"因果 ASR + 增量 MT + 语音 token 生成"三件能力，用 717 步 / 91,776 样本 / 43 万目标文本 token 是不可能的。旧血脉之所以能到 ASR 0.26 / coverage 0.17，是多个阶段累计训练的结果。本轮等于把进度条清零后跑了 1/10 的路程，再用一个梯度为零的 RL 去补。

---

## 4. 详细解决方案

原则：**先证伪最便宜的假设，再决定是否重训；任何重训必须以旧血脉最优 checkpoint 为起点；内容不达标不进 RL。**

### 4.1 第 0 阶段：24 小时内、几乎零算力的证伪实验（必须先做）

#### 实验 0-A：推理桥的离散码一致性测试（最高优先级）

目的：判断 ASR 0.048 是能力问题还是推理路径 bug。

做法：
1. 取 4 条训练用长音频，用**数据管线**产出的 `frontend_ids`（金标）；
2. 同一音频走 `TrainableSharedCausalWhisperVQ` + `_nearest_codes` 桥，产出 `codes`；
3. 报告逐帧 code 一致率、以及 `frontend_adapter(codebook(ids))` 与 `frontend_adapter(codebook(codes))` 的余弦相似度。

判据：
- 一致率 ≥ 99% → 桥没问题，ASR 0.048 是真实能力缺失，进入 4.2；
- 一致率 < 95% → **桥是主因**，先修桥并用修好的桥重跑一次 pre_GRPO rollout（约 1.2 小时），再重估全部结论。

同时补一个更强的判据：**teacher-forced 一致性**。用同一 checkpoint、同一条音频，分别走(1) 训练 forward（喂金标 `frontend_ids` 与金标前缀）与(2) 推理 cascade（喂桥 + 金标前缀），比较下一 token logits 的 top-1 一致率与 KL。这个测试能一次性排除所有前端／位置／dtype／batch 轴类的路径不一致。

> 交付物：`reports/.../root_cause_and_next_plan_v1/BRIDGE_PARITY.json`

#### 实验 0-B：旧血脉最优 checkpoint 在新评测器下的复测

目的：确认 §1.2 的对比不是评测口径差异造成的。

做法：用**本轮 v2 的评测器与 reward 代码**，对 `uniss_phasea_commit_complete_sft_rl_v4` / `coverage_constrained_grpo_v3` 的最优 checkpoint 重跑同一批 64×4。约 1.2 小时。

判据：若旧 checkpoint 在新评测器下仍是 ASR≈0.26 / coverage≈0.17，则"旧优于新"成立，**下一版必须以旧 checkpoint 为起点**。

#### 实验 0-C：SFT checkpoint 的 teacher-forced 生成上限

目的：把"模型不会译"与"自由运行发散"分开。

做法：对 8 条长 episode，强制喂入金标 ASR 前缀与金标已提交译文前缀，只让模型预测下一段增量译文（scheduled sampling = 0），统计 coverage。

判据：
- teacher-forced coverage 也 < 20% → 能力缺失，必须重训（4.2）；
- teacher-forced coverage > 60% 而自由运行 3% → 是**曝光偏差／EOS 过预测**，可用 4.3 的低成本修复解决，不必全量重训。

**这三个实验合计 < 3 小时 8 卡，但决定了后面是花 20 小时还是 200 小时。**

### 4.2 第 1 阶段：数据构造修复（无论 0 阶段结论如何都要做）

这是本轮失败的第一性原因，必须在下一次训练前修掉。

| 问题 | 修复 | 验收 |
|---|---|---|
| 增量 MT 只占约 1% 监督 token | 重构 event 视图：**每个事件都产出"当前应有的目标语言完整前缀"**作为 CE 目标，而不只对 WRITE 事件的新增 delta 监督。目标 token 占比提到 **≥ 25%** | `runtime_text_supervised_tokens / supervised_tokens ≥ 0.25` |
| 语音 token 头未学会 | 对**每个已提交短语**监督完整目标 codec 序列（含 speaker reference 状态），而非只监督 microblock 首槽；语音项权重从 2.0 提到与文本项同量级 | 训练 500 步内 `microblock_token_accuracy ≥ 0.25` |
| `prefix_stability` / `real_prefix_kd` / `speaker_consistency` 恒为 0 | 在 `phrase_oracle_sessions` 里显式生成这三类目标；**加断言：任何权重非零的损失项，若前 50 步内 loss 恒为 0 则训练直接失败退出** | 三项 loss > 0 |
| 训练无 deadline 强制 WRITE，推理有 | 训练数据中按推理同样的 4 秒 deadline 规则注入强制 WRITE 事件，并监督其后的内容 | `deadline_forced_fraction > 0` |
| EOS 过预测 2.15× | EOS 位置改用 **focal loss + 正类降权**（当前 predicted 0.0168 vs target 0.0078）；并增加"源音频未结束时 EOS 为硬负例"的显式监督 | `predicted_eos_fraction / target_eos_fraction ∈ [0.8, 1.2]`，precision ≥ 0.6 |
| `phase3_replay` 不下降、白耗预算 | replay fraction 从 0.35 降到 **0.10**，并只保留与 S2ST 直接相关的 offline ASR/MT 任务 | replay loss 在训练中单调下降 |

### 4.3 第 2 阶段：训练配方修复

#### (A) 起点：换回旧血脉最优 checkpoint

```
❌ --load checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4   (offline, iter 9075)
✅ --load <commit_complete_v4 / coverage_constrained_v3 最优 checkpoint>
```

若 0-B 证实旧 checkpoint 更强，则**必须**从它继续；`pretrain_content_first_megatron.py:97-100` 目前硬编码只允许 Phase3 v4 或自身，需要放开并加入旧血脉 root 的白名单 + fingerprint 校验。

#### (B) 训练量：717 步远远不够

| 项 | 本轮 | 建议 |
|---|---:|---:|
| updates | 717 | **≥ 4,000**（先跑 1,500 做中期门槛） |
| epochs over 15-shard | 1 | 3–5 |
| 有效目标文本 token | 0.44 M | **≥ 25 M** |

#### (C) 必须引入自由运行在环（本轮最大的方法论缺口）

本轮完全是 teacher-forced CE + 一个只读诊断的 `event_rollout_*`。证据显示 teacher-forced CE 在降而自由运行行为在退化。必须让自由运行进入**损失**，而不只是进入日志：

1. **Scheduled sampling / DAgger**：从 iter 500 起，以 p 从 0 线性升到 0.5 的概率用模型自己上一步的输出替换 teacher 前缀，对**金标前缀**做 CE。这直接压 `first_divergence`。
2. **每 250 步一次小规模自由运行评估**（8 条 episode，约 4 分钟），把 coverage / first_write / grammar_valid 写入 TensorBoard，并作为选 checkpoint 的**唯一依据**（不再用 val loss 选）。
3. **硬门槛（gate）**：自由运行 coverage 未达标就**不允许**进入 GRPO。建议门槛见 §4.5。

#### (D) 参数范围：不要只训新 LoRA

本轮 `frontend_residual_rms` 只有 0.030，说明因果前端几乎没动。建议：
- 解冻并训练：因果前端最后 4 层 + `frontend_adapter` + `frontend_projection`（lr 5e-6）；
- 解冻并训练：`true_subsecond_lora` 全部 144 张量（lr 1e-5）+ 语音输出头（lr 5e-5）；
- 冻结：BiCodec / 声码器（保音质）。

### 4.4 第 3 阶段：Reward 与 GRPO 修复（在内容达标后才用）

即使内容达标，当前 reward 仍需修四处，否则同样的零梯度问题会重演：

| # | 问题 | 修复 |
|---|---|---|
| 1 | `eligible_for_latency = complete >= 0.75` 永久关闭 | 改为**连续门控**：`latency_term *= clamp(complete / 0.75, 0, 1)²`。低覆盖时权重小但非零，梯度始终存在，同时仍然无法通过"早说两个词"取胜 |
| 2 | `failures = min(4.0, ...)` 在 premature_end≈5.3 时恒饱和 | 上限提到 12.0，或把 `premature_end_count` 单独拆成 `-0.5 * premature_end`（不设上限，改用 tanh 软饱和） |
| 3 | `terminal_coverage_penalty` 在低 coverage 区近似常数 | 换成对 coverage 单调、低区斜率更大的形式，例如 `-6.0 * (1 - complete)` 线性项 + 小二次项 |
| 4 | `normalize(epsilon=1e-4)` 放大退化组噪声 | 把方差下限提到 **组内 raw return 绝对尺度的 5%**（例如 `scale = max(0.05*abs(mean)+0.02, std)`）；并**丢弃 std 低于阈值的退化组**，不产生训练样本 |
| 5 | `mt`/`asr`/`tts` family advantage 是候选级常数 | 为 MT 引入 per-event credit：用该事件的 `target_coverage_delta` 与前缀稳定性构造局部回报，再做 reward-to-go（`control` family 已经这样做了，照搬） |

另外：GRPO 的可训练范围应从"顶 8 层 rank-16"扩到**包含增量 MT commit adapter 与语音提交头**，否则 RL 依然只能改时机。

### 4.5 阶段门槛（gate）—— 不达标不推进

| 阶段 | 门槛（train-seen 64×4 自由运行） | 未达标怎么做 |
|---|---|---|
| SFT 中期（iter 1500） | teacher-forced `runtime_text_token_accuracy ≥ 0.55`；`microblock_token_accuracy ≥ 0.25` | 停训，回 4.2 查数据构造 |
| SFT 结束 | 自由运行 `target_coverage ≥ 0.40`；`grammar_valid ≥ 0.95`；`premature_end ≤ 1.0/episode` | 停训，**不进 GRPO** |
| GRPO 进入前 | `spoken_target_coverage / target_coverage ≥ 0.95` | 修 TTS 提交链路 |
| GRPO 每轮后 | reward 组内 std ≥ 0.3；退化组比例 ≤ 20% | reward 或 rollout 温度需调整 |
| 最终 | coverage ≥ 0.80；first_write(source-time) ≤ 1.5 s；max internal silence ≤ 3 s | — |

### 4.6 明确不要做的事

- ❌ 不要在当前 content-first checkpoint 上加第 3、4 轮 GRPO。两轮已证明每轮只动 0.1 个百分点，且 Round 2 比 Round 1 更差。
- ❌ 不要为了让流水线跑通而关闭 handoff 审计。审计两次都正确地拦住了真实不兼容。
- ❌ 不要继续用 val loss 选 checkpoint。本轮 val loss 一路下降而自由运行行为退化。
- ❌ 不要在 coverage < 40% 时讨论 wall-clock 延迟、KV cache、异步 TTS。当前 RTF 7.85，且 `first_write` 是源时间轴口径，不是用户等待时间——在没有内容之前优化延迟没有意义。
- ❌ 不要再新建一条并行血脉。当前已有 19 个 rollout / 4,240 个候选目录，跨血脉兼容桥本身已经成为一类 bug 来源。

---

## 5. 建议的执行顺序与预估

| 步骤 | 内容 | 8 卡耗时 | 决策价值 |
|---|---|---:|---|
| 1 | 实验 0-A 桥一致性 + logits parity | 0.5 h | **决定是否需要重训** |
| 2 | 实验 0-C teacher-forced 上限 | 0.5 h | 区分能力缺失 vs 曝光偏差 |
| 3 | 实验 0-B 旧 checkpoint 新评测器复测 | 1.2 h | 确定重训起点 |
| 4 | 4.2 数据构造修复 + 单元测试 + 零 loss 断言 | 0（CPU） | 消除第一性原因 |
| 5 | 2-step smoke + 500 步中期门槛训练 | 3 h | 早停判断 |
| 6 | 正式 SFT（≥4000 步，含 scheduled sampling 与周期自由运行评估） | 20–24 h | — |
| 7 | SFT 门槛评估（64×4） | 1.2 h | 决定是否进 GRPO |
| 8 | 4.4 reward 修复 + 2 轮 GRPO | 6 h | 只在门槛通过后 |

**关键：第 1–3 步必须先做完再决定第 6 步的规模。本轮的教训正是跳过了"这个 checkpoint 值不值得进入下一阶段"的判断。**

---

## 6. 附录：关键数字速查

### 6.1 SFT 训练诊断（717 步全程解析）

| 指标 | min | max | mean | 前 10 步 | 后 50 步 |
|---|---:|---:|---:|---:|---:|
| supervised_tokens | 13,629 | 93,372 | 55,868 | 70,672 | 46,135 |
| runtime_text_supervised_tokens | 0 | 1,203 | 607 | 801 | 480 |
| runtime_text_token_accuracy | 0 | 0.403 | 0.196 | 0.055 | 0.221 |
| microblock_token_accuracy | 0 | 0.0337 | 0.0147 | 0.0126 | 0.0153 |
| runtime_text_content (loss) | 0 | 7.269 | 2.711 | 5.512 | 2.068 |
| microblock_semantic_content (loss) | 0 | 9.941 | 5.048 | 7.426 | 4.886 |
| phase3_replay (loss) | 0 | 4.189 | 1.481 | 1.361 | 1.443 |
| real_prefix_kd (loss) | 0 | **0** | **0** | 0 | 0 |
| prefix_stability (loss) | 0 | **0** | **0** | 0 | 0 |
| speaker_consistency (loss) | 0 | **0** | **0** | 0 | 0 |
| deadline_forced_fraction | 0 | **0** | **0** | 0 | 0 |
| safe_commit_f1 | 0 | 0.584 | 0.294 | 0.383 | 0.235 |
| event_rollout_grammar_valid_fraction | 0 | 1.0 | 0.650 | 0.656 | 0.655 |
| event_rollout_false_write_fraction | 0 | 0.516 | 0.117 | 0.000 | 0.195 |
| event_rollout_first_divergence | 0 | 3.680 | 0.955 | 0.794 | 1.191 |
| runtime_predicted_eos_fraction | 0 | 0.0391 | 0.0171 | 0 | 0.0168 |
| runtime_target_eos_fraction | 0 | 0.0203 | 0.0097 | 0.0129 | 0.0078 |
| frontend_residual_rms | 0 | 0.0462 | 0.0249 | 7.3e-5 | 0.0301 |

### 6.2 关键源码位置

| 内容 | 位置 |
|---|---|
| content-first 损失权重（17 项） | `experiments/uniss_phase3_content_first_joint_s2st_v1/training/pretrain_content_first_megatron.py:40-58` |
| 起点 checkpoint 白名单（硬编码 Phase3 v4） | 同上 `:97-100` |
| 短语最小 token = 4 | 同上 `:39` |
| GRPO 可训练范围（顶 8 层 policy LoRA） | `experiments/uniss_phase3_content_first_joint_s2st_v1/training/pretrain_content_first_grpo.py:95-140` |
| 延迟奖励门控 `complete >= 0.75` | `experiments/uniss_phasea_coverage_constrained_grpo_v3/training/reward.py:89-96` |
| failure 惩罚上限 `min(4.0, ...)` | 同上 `:79-83` |
| coverage 二次惩罚 | 同上 `:85` |
| advantage 归一化 `epsilon=1e-4` | `experiments/uniss_phasea_coverage_constrained_grpo_v3/training/event_credit.py:92-99` |
| family 级 advantage（terminal-only） | 同上 `:101-142` |
| 推理兼容桥（重量化） | `experiments/uniss_phase3_content_first_joint_s2st_v1/runtime/model_loader.py:155-215` |
| 训练侧前端残差路径 | `experiments/uniss_phase3_true_subsecond_deadline_full198_v1/training/joint_model.py:162-192` |
| 参考 VQ（平方 L2 argmin） | `uniss/speech_tokenizer/glm4/modeling_whisper.py:68-88` |

### 6.3 数据与算力口径

- 训练：15-shard event cache，`--train-iters 717`、`--global-batch-size 128` → 91,776 样本，1 epoch
- 峰值显存约 128–132 GB/卡，吞吐 35–200 TFLOP/s/GPU（长度异构导致波动）
- rollout：64 episodes（32 中→英 + 32 英→中）× 4 候选 = 256；66,924 traces
- 三次 256 候选 rollout + 2 轮 GRPO ≈ 20 h 8 卡
- 全部结果为 train-seen，无泛化声明

### 6.4 会话与产出路径

- Codex 会话 JSONL：`/opt/dlami/nvme/jasonleeeli/.codex/sessions/2026/08/28/rollout-2026-08-28T13-02-24-01a04876-dd46-75f1-9c57-a250cd095b39.jsonl`（367 MB，922 条用户消息，血脉起点 2026-07-17）
- 本轮官方报告：`reports/uniss_phase3_content_first_joint_s2st_v1/coverage_grpo_final_v2/REPORT.zh-CN.md`
- 本轮指标：同目录 `METRICS.json`
- 对照血脉指标：`reports/uniss_phasea_coverage_constrained_grpo_v3/final_v1/METRICS.json`
- SFT 训练日志：`logs/uniss_phase3_content_first_joint_s2st_v1_formal1e_v1.log`
