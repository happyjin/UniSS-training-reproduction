# Phase3 Joint V6 失败的深层重读与流式接口重新设计

> 日期：2026-08-08
> 输入材料：[`uniss_phase3_whisper_streamspeech_joint_v6_full198_stage_b_failure_analysis.md`](./uniss_phase3_whisper_streamspeech_joint_v6_full198_stage_b_failure_analysis.md)、codex session `019f6fc5`（2026-08-06 至 08-08 全部对话）、[`stage_b_latent_15shard_h200_execution_report.md`](./stage_b_latent_15shard_h200_execution_report.md)、`reports/uniss_phase3_whisper_streamspeech_joint_v6/`
> 关系：本文不推翻已有失败分析，它的六个根因都成立。本文回答一个它没有问的问题——**这六个根因是否来自同一个前提，以及那个前提是否本来就不成立。**

---

## 0. 结论先行

已有失败分析把 V6 Stage B 判为「训练工程没崩，但研究目标没达成」，并给出六个根因。这个判断是准确的。但把 codex 会话里 v1→v4→v6 的完整轨迹连起来看，加上你自己两份更早的测量报告，会得到一个更根本的结论：

**六个根因不是六个独立的 bug，它们是同一个设计前提的六个必然推论。那个前提是：**

> 保住冻结的 Phase3 后端 = 让流式前端产出与离线 WhisperVQ **相同的 GLM token ID**。

这个前提可以被你**自己已经测过的两组数据**直接证否，而且是从两个相反的方向：

**方向一：目标不可达。** 你的 causal-teacher ceiling 审计（`stage_b_latent_15shard_h200_execution_report.md` §10）测得，即使给 **640 ms** 前瞻，与全上下文 teacher 的 agreement 也只有 **0.6814**，达不到 0.70 的续训门；而一个「全 teacher 正确」的 token 要到中位数约 **2.9 秒** 才稳定。报告自己的原话是：

> "It is therefore **mathematically inconsistent** to require an 80 ms-lookahead student to reproduce 90% of these full-context token IDs while also claiming subsecond latency."

**方向二：即使可达也没用，因为 agreement 不预测下游质量，在你的数据里甚至是反向的。** 同一份报告 §11 的冻结 Phase3 敏感性表：

| 源 GLM 流 | 与 released 的 agreement | EN→ZH Text-BLEU | ZH→EN Text-BLEU |
| --- | ---: | ---: | ---: |
| released（原始） | 1.0 | 33.45 | 26.61 |
| 重建音频 + 全上下文 WhisperVQ | **0.40476** | 25.75 | 19.37 |
| prefix-causal 80 ms | **0.2632** | **31.22** | **25.21** |

**agreement 更低的那个流（0.2632 vs 0.40476），Text-BLEU 反而高 5.5 / 5.8 点。** 因为真正伤害下游的是 BiCodec 重建带来的声学损伤，不是因果性限制。而 Stage B 的整个 loss 设计是在优化前者、忽略后者。

所以 V6 Stage B 的正确读法不是「模型漂移了」，而是：

> **优化器是对的，指标是错的。** 训练把表示推向了对四个真实任务更好的位置（ASR/NAR/AR/BiCodec validation loss 全在下降），而一个既不可达、又不预测下游质量、还没进反向传播的诊断量（teacher agreement / bridge commitment）把训练判为失败并终止了它。

这不代表 V6 checkpoint 可用——code perplexity 从 59.65 掉到 50.36 是真实的坍塌风险，必须警惕。但它意味着**修复方向不是把 agreement 修上去，而是把接口从「离散 token 模仿」换成学术界通用的连续接口。**

---

## 1. 从 codex 会话重建的完整轨迹

三天的会话记录（2026-08-06 01:49 → 08-08 04:44，287 条消息）显示这不是一次失败，而是**同一个病在两种形态下发作两次**。

### 1.1 第一次发作：v4 数值爆炸（08-06）

| Iteration | 加权 val loss | ASR CTC | NAR CTC | Bridge MSE | Grad norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 110.60 | 8.20 | 8.34 | 4.49 | 1,180 |
| 700 | **108.72**（最优） | **8.00** | **8.27** | 8.53 | 425 |
| 1000 | 109.25 | 8.28 | 8.39 | 29.96 | 1,233 |
| 2200 | 139.66 | 13.12 | 11.28 | 713.97 | 明显升高 |
| 3400 | 1,342.91 | 176.94 | 148.44 | 9,952.82 | 约 150 万 |
| 4100 | 3,007.61 | 310.00 | 431.55 | 25,558.75 | 约 1800 万 |

当时定位的四个原因：

1. `bridge/commitment_mse` 与 `whisper/quantize_loss` 被 `detach()`，**只记录不进反向传播**——表示可以自由漂移而不受惩罚；
2. STE bridge 前反向不一致：前向用最近码字的冻结 embedding，反向经过一个**随机初始化、可训练但前向用不到**的 `continuous_projection`，逐渐变成放大梯度的错误 Jacobian；
3. Whisper 内部 EMA codebook 仍在更新并每 100 次 forward 重启 inactive codes——到 iter 4140 触发约 530 次，其中 316 次单次重启超过 10,000 个码字，常见 15,800–16,100 / 16,384，**接近整体坍塌**；同时 bridge 持有的是初始化时复制的**另一份**冻结 codebook，形成两个互相漂移的参照系；
4. LR warmup 4000 iterations，但模型 700 就最优、1200 后回退，学习率还在一路涨。

### 1.2 修复：v6（08-07）

- 两个对齐 loss 加进总 loss；
- bridge 换成 `topk_soft`，**删掉那个可训练 projection**；
- 学习率大幅下调：heads `1e-5`、Whisper pre-VQ `1e-7`、Qwen `1e-8`、bridge 乘子 `0`；
- 加入 `bridge_commitment` 绝对安全门 `0.10`。

这些修复都对症，而且**确实生效了**：v6 full198 Stage B 全程无 NaN、无 skipped iteration、无 OOM，四个任务 loss 全部单调下降。

### 1.3 第二次发作：v6 full198 Stage B 缓慢语义漂移（08-07 → 08-08）

跑到 5060/9075（55.76%）被安全门终止（`value=0.101447` / `limit=0.100000`）。固定 320 ms validation：

| Stage B iteration | Teacher agreement ↑ | Bridge commitment ↓ | Teacher commitment ↓ | Code perplexity ↑ | Hidden RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 700 | 15.89% | 0.02016 | 0.03810 | 59.65 | 0.6941 |
| 3000 | 9.28% | 0.02967 | 0.05567 | 53.69 | 0.7390 |
| 5000 | **5.45%** | 0.04007 | 0.07487 | 50.36 | 0.7599 |

同期四个任务 val loss：BiCodec `-3.44%`、AR `-13.06%`、ASR `-3.29%`、NAR `-5.14%`，**全部改善**。

### 1.4 这个轨迹本身说明什么

v4 是「表示逃逸 + 梯度爆炸」，v6 是「表示逃逸，但被限速了」。修复消除了爆炸，**没有消除逃逸的动机**。因为逃逸的动机不来自数值问题，而来自：任务 loss 想去的地方，和冻结 GLM codebook 所在的地方，不是同一个地方。

只要接口还是「必须落在那个 codebook 的最近邻上」，这个张力就永远在。v4 用几千倍的 loss 爆掉，v6 用一万个 iteration 慢慢漂——**同一个病的两种病程**。

---

## 2. 六个根因其实是一个前提的六个推论

已有报告的六个根因，逐条对应回那个前提：

| 报告的根因 | 为什么它只在「token 模仿」前提下才是问题 |
| --- | --- |
| 1. teacher token 与训练音频不闭环 | 只有当你要求 **exact token ID 匹配**时，`source_glm` 是从哪条音频链算出来的才致命。如果接口是连续 hidden，重建音频只是一个域偏移 |
| 2. 流式 student 被要求复现有完整未来的 offline teacher | 只有当 teacher 是**另一个全上下文模型**时才存在。FAST 的做法是让 teacher 和 student 是**同一个模型**、只差 m 帧 oracle 未来 |
| 3. 没有独立的 streaming repair bridge | 需要「repair」是因为你在把 hidden 硬掰进一个外来的 codebook 几何。连续接口不需要修 |
| 4. 优化目标 / 诊断指标 / 安全门三者不一致 | `teacher agreement` 是 argmin 后的 exact match，**不可微**，天然无法作为优化目标；`bridge_commitment` 权重是 `0` 却被用作停止判据 |
| 5. exact-position agreement 对帧偏移过于敏感 | 量化边界 + 16384 分类 top-1 的固有脆弱性，连续 hidden 距离没有这个问题 |
| 6. guard 把不同 chunk 的 rank 混在一起 | 监控设计缺陷，但它监控的对象本身就是根因 3 的产物 |

六条里有五条在换成连续接口后**自动消失**，只有第 6 条是独立的工程 bug。

### 2.1 停止判据有三层问题叠加

V6 训练是被 `bridge_commitment > 0.10` 终止的。这个判据同时满足：

1. **它不在优化目标里**（权重 `0`）——你在用一个没人负责的量做停止判据；
2. **它被跨 chunk 混合测量**——同一 step 的 8 个 rank 分别在 320/640/960/1280/offline 上，guard 先求均值再比阈值，不同 chunk 的正常分布本来就不同；
3. **它是一个不可达目标的代理**——它保护的是「留在 GLM codebook 几何里」，而 §0 的天花板审计说这个几何在亚秒延迟下根本到不了。

三层叠起来，这个 gate 的信噪比接近于零。

---

## 3. 学术界为什么没有人这么做

我把最近的 simultaneous S2ST / streaming speech-LLM 系统按「流式前端如何接到生成后端」列出来：

| 系统 | 前端 → 后端接口 | 是否模仿离线 tokenizer 的离散 ID |
| --- | --- | :---: |
| **UniSS Phase3-joint V6** | streaming Whisper hidden → 冻结 GLM codebook 最近邻 → Qwen embedding | **是** |
| StreamSpeech (ACL 2024) | 共享 Conformer hidden 直接进四个任务 head | 否 |
| SimulS2S-LLM (ACL 2025) | CIF 连续边界感知 prompt → LLM | 否 |
| InfiniSST (ACL 2025) | 2 层 1-D conv + linear → LLM embedding | 否 |
| SimulS2ST-Omni (2026) | encoder hidden 直接进 Thinker LLM | 否 |
| Hibiki (ICML 2025) | Mimi **流式原生** codec，双流并行 | 否（tokenizer 本身就是流式的） |
| FAST (EMNLP 2023) | CIF 边界检测 + **同模型** hidden 蒸馏 | 否 |
| Verdini et al. (Interspeech 2025) 五种 adapter | Base / Conv / CIF / CTC / WLQ-former，**全部连续** | 否 |

**没有一个系统训练流式编码器去复现某个离线 tokenizer 的离散 token ID。** 这不是巧合。离散 ID 匹配把三种彼此独立的误差（数据链错配、因果性缺失、量化边界抖动）压进一个不可微的 0/1 指标，既无法优化，也无法归因。

UniSS 之所以走上这条路，动机是完全合理的：**不想重训那个已经很强的 Phase3 后端**（SimulS2ST-Omni 论文把 UniSS-Q 的 CVSS-T 32.04 / 24.72 引为 SOTA 基线）。但学术界对同一个诉求给的答案不一样——**训一个连续 adapter，必要时给 LLM 加小 LoRA，而不是把编码器硬掰进一个外来的离散几何。**

---

## 4. 适合当前框架的思路

按与「保住 Phase3 + 要流式」这个具体约束的契合度排序。

### 4.1 FAST：这就是你的问题的标准解法（最对口）

> Fu et al., *Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference*, EMNLP 2023，[PDF](https://aclanthology.org/2023.emnlp-main.1033.pdf)（厦门大学 + 阿里 DAMO）

**Motivation** 与你完全一致：用**单个离线模型 + wait-k** 支持多种延迟，比训多个在线模型简单得多；但直接这么做有 mismatch——论文实测「**流式输入末端抽出的语音表示，与完整语句抽出的表示显著不同**」。这句话就是你 Stage A offline agreement 只有 22.96% 的普适版本。

**两个机制**：

- **FAI（Future-Aware Inference）**：推理时在流式输入后面接上 `m` 个**可训练的 mask embedding** 作为未来上下文占位。模型学会为「还没听到的部分」生成一个合理的表示，从而让**已听到那些帧**的表示更准确。
- **FAD（Future-Aware Distillation）**：
  - teacher = 流式前缀 + `m` 个 **oracle** 语音帧（真实未来音频）
  - student = 流式前缀 + `m` 个 **mask** token
  - **student 从 teacher 初始化**（同一个模型）
  - 最小化两者输出之间的蒸馏 loss

**为什么这恰好修掉你的根因 1/2/5**：

| | UniSS Stage B | FAST |
| --- | --- | --- |
| teacher 是什么 | **另一条编码链**上的 offline WhisperVQ 的 **discrete token ID** | **同一个模型**在 prefix + m 帧 oracle 未来上的 **连续 hidden** |
| 监督形式 | exact-position top-1，16384 分类 | hidden 蒸馏，连续、可微、无量化 |
| teacher 是否闭环 | **否**（offline agreement 22.96%） | **构造上必然闭环**（同模型同音频） |
| gap 由什么构成 | 数据链错配 + 因果性 + 帧偏移 + 量化边界，四者混在一起 | **只有因果性一项** |
| 推理时未来怎么办 | 没有占位，直接截断 | m 个可训练 mask embedding 显式建模 |

最后一行是重点：你的 student 在 320 ms 时被要求「凭空知道后文」，而 FAST 给了它一个专门的、可学的地方去放那个猜测。

**落到 UniSS**：在 `phase3_whisper_streamspeech_joint` 里，把 `teacher_CE + teacher_commitment` 换成 FAD 的 hidden 蒸馏——teacher 是同一个 Whisper 前端喂 `prefix + m 帧真实未来`，student 是喂 `prefix + m 个 mask embedding`。`source_glm` 完全不参与，根因 1 直接消失，也不需要先花几天重建 `source_glm_reencoded`。

### 4.2 CIF 适配器：把接口从「离散 token」换成「文本 token 粒度的连续 prompt」

> Dong & Xu, *CIF*, ICASSP 2020；用法见 SimulS2S-LLM (ACL 2025) 与 Verdini et al., *How to Connect Speech Foundation Models and LLMs? What Matters and What Does Not*, Interspeech 2025，[PDF](https://www.isca-archive.org/interspeech_2025/verdini25_interspeech.pdf)

**Motivation**：Speech LLM 的语音 prompt 是**帧粒度**的，而 LLM 训练时见的是**文本 token 粒度**。CIF 对每帧输出一个积分权重 $\alpha_t$，累加到阈值 1.0 就"发放"一次，把对应帧加权平均成一个 hidden——**发放次数天然对应 token 数**。这样流式 prompt 的增长语义和 LLM 训练时一致。

Verdini 的受控消融给了选型依据（5 种 adapter，同一套设置）：

- CIF-based：Whisper 上平均压缩率约 **25**；辅助 loss = 转写 CTC + quantity loss，权重各 `0.1`
- CTC-based（合并连续相同预测再平均）：Whisper 上约 **13**
- Conv-based：固定压缩率 4，无辅助 loss
- WLQ-former：窗口级 Q-Former

**对 UniSS 的意义**：你的 GLM 是 12.5 Hz 固定帧率，一个 4 秒句子给 Qwen 50 个 speech token。换成 CIF 后，prompt 长度由**内容**决定而不是时长决定，且：

- CIF 只在**源侧**发放，**不依赖目标侧单调对齐**——这一点很关键，它绕开了中英重排问题；
- 发放计数本身就是天然的 read/write 信号，可以直接驱动 wait-k，不需要再训一个 policy；
- quantity loss 用源文本长度做弱监督，你的 `transcription` 字段现成。

代价是 Qwen 需要用 LoRA 适应新的 prompt 分布——但这比让 Whisper 去够一个够不着的 codebook 便宜得多。

### 4.3 换成流式原生的 tokenizer（Mimi 路线）

> Défossez et al., *Moshi / Mimi*；[HF](https://huggingface.co/kyutai/mimi) · [代码](https://github.com/kyutai-labs/moshi)

**Motivation**：与其把一个**非因果**的 tokenizer（WhisperVQ 的 `encoder_causal_attention=false`、`quantize_causal_encoder=false`）强行改造成流式，不如用一个**从设计之初就是流式**的。

Mimi 的参数和 UniSS 惊人地接近：24 kHz 输入、**12.5 Hz**（和你的 GLM 完全一样）、1.1 kbps、**完全流式，延迟 80 ms（=帧长）**，且第一层 codebook 通过蒸馏对齐 WavLM 语义表示——语义/声学在一个模型里。它在这些指标下仍优于非流式的 SpeechTokenizer（50 Hz / 4 kbps）和 SemantiCodec（50 Hz / 1.3 kbps）。

**对 UniSS 的意义**：这条路要求重训 Phase3 的源侧接口（把 `source_glm` 换成 Mimi token 并重新对齐 Qwen 词表），代价大。但它把「流式 tokenizer」这个问题**一次性删除**，而不是每个 V 版本重打一次。如果 V7 的 FAST 方案仍然撞墙，这是下一个落点。

### 4.4 SimulS2ST-Omni 的轨迹监督与 Thinker–Talker（数据侧立刻可做）

细节见上一份文档 [`streamspeech_failure_diagnosis_and_alternative_simul_s2st_ideas.md`](./streamspeech_failure_diagnosis_and_alternative_simul_s2st_ideas.md) §2.1。这里只强调与本次失败最相关的两点：

- **它的 Dec-only 消融就是 UniSS 的架构**（把 code token 追加进 LLM 词表、单头输出），受控对比下在**所有**延迟档都输给双流 Thinker–Talker，低延迟端差距最大；
- **NIR 单调性过滤**的消融：不做过滤时最低延迟档 BLEU 从 21.14 崩到 4.59（En→Zh）、11.98 崩到 3.56（Zh→En）。

你已经有 `source_words` / `target_words` 时间戳（`subsecond_v2/prepare_a45.py`），缺 SimAlign 跨语言对齐和 NIR 过滤。**这部分数据工作对下面任何一条路线都有用**，可以和 V7 训练并行开始。

### 4.5 InfiniSST 的 Λ 形 KV cache（治 RTF）

细节同上文档 §2.3。要点：只保留 system instruction 的 KV + 最近 `w=1000` token 的 KV，**存储前去掉 RoPE、拼接后重新施加**，实现无限长度外推，CA 延迟降 0.5–1 秒。你的 Stage12 RTF 是 2.23 / 7.49，这是对症的。

---

## 5. 对已有修复计划的调整建议

已有报告 §8 的五步修复（重建自洽 teacher → 加 repair adapter → soft alignment → 两阶段训练 → 修 guard）方向都对，但**第一步的性价比需要重新评估**。

### 5.1 「重建自洽 teacher」能买到什么，买不到什么

按报告 §8.1，要对每条数据重跑 `BiCodec decode → FLAC → frozen offline WhisperVQ`，质量门是 offline agreement ≥99%。

**能买到**：offline agreement 从 22.96% → ~99%，根因 1 消除。

**买不到**：320 ms 的 agreement。因为剩下的 gap 由天花板决定，而天花板你已经测过了：

| Lookahead | 与全上下文 teacher 的 agreement |
| ---: | ---: |
| 80 ms | 0.2632 |
| 160 ms | 0.3933 |
| 320 ms | 0.5465 |
| 640 ms | 0.6814 |

即使 teacher 完全闭环，**320 ms 的上限也就是 0.5465 附近，640 ms 是 0.6814**，都够不到 0.70 续训门，更够不到 0.90 最终目标。而「全 teacher 正确的 token」中位数要 **2.9 秒**才稳定。

所以：**重建 teacher 是必要的清洁工作，但它不能让 Stage B 的既定目标变得可达。** 如果做，应该明确它的用途是「消除一个混淆变量、让后续归因干净」，而不是「修复 agreement」。同时应该**把 agreement 从质量门降级为诊断项**，改用报告 §9 里已经列出但没被当作主门的那些：frozen-Phase3 Text-BLEU / COMET。

### 5.2 建议替换的顺序

| 报告原计划 | 建议调整 |
| --- | --- |
| ① 重建 `source_glm_reencoded`，≥99% 闭环门 | **降级为可选**。先做 §6 的 D1 判定它值不值 |
| ② 加 residual repair adapter | **保留，但改成 FAST 的 FAI mask embedding**——它不只是"修复"，还显式建模未来 |
| ③ soft alignment（hidden MSE + top-k KL + CE + commit） | **保留 hidden 项，删掉 CE 和 commit**。teacher 改成 FAD 的同模型 oracle-future hidden |
| ④ 两阶段训练 + loss ramp | **保留**，这个设计是对的 |
| ⑤ 修 guard（per-chunk baseline、rank 广播 chunk、记录触发样本） | **保留且优先**，这是纯工程 bug，且不修的话下次还是定位不了 |
| — | **新增**：把主质量门从 `teacher agreement` 换成 `frozen-Phase3 Text-BLEU 下降 ≤ 2 点` |

---

## 6. 三个决策实验

在启动 V7 之前，这三个实验能把「该不该重建 teacher」「该不该继续 token 模仿」「V6 是不是真的坏了」一次性判掉。都不需要大规模训练。

**D1：V6 的 checkpoint 到底坏了没有？（1 天，纯推理）**

对 Stage A、iter 750/1500/3000/5000 这几个 checkpoint，跑冻结 Phase3 的固定 32 条双向 Text-BLEU 探针（和 Stage07 同协议）。

- 若 BLEU 随 iteration **基本持平或上升**，而 agreement 从 15.89% 掉到 5.45% → **证实 §0 的判断**：agreement 是坏指标，V6 其实在进步，应立刻把主门换成 BLEU 并考虑续训；
- 若 BLEU 随 agreement 一起掉 → 报告的 drift 判断成立，按 §5.2 修复。

**这是本文档里最重要的一个实验。** 它只需要跑已有的 checkpoint，成本极低，但决定后面所有事。

**D2：重建 teacher 值不值？（2 天）**

在 15-shard 上取 2000 条做 `BiCodec decode → FLAC → frozen offline WhisperVQ`，只算 offline agreement。

- 若真能到 ≥99% → 说明 pipeline 可闭环，但**仍要结合天花板表决定**：闭环后 320 ms 上限约 0.5465，问自己这个数字是否值得几天的全量重建；
- 若到不了 99%（比如只有 60–80%）→ 说明预处理链里还有别的不一致，此时**更应该直接放弃 token 模仿**，走 FAST 的同模型蒸馏。

**D3：FAST 的 FAD 能不能替代当前 teacher？（3–4 天，小规模）**

在 15-shard 上做一个最小实现：teacher = 同一 Whisper 前端喂 `prefix + m 帧真实未来`（m 取 320 ms 对应帧数），student = 喂 `prefix + m 个可训练 mask embedding`，loss = hidden MSE/cosine。冻结其余一切，只训 mask embedding + 最后一层。

- 指标看：student hidden 与 **同模型 full-context hidden** 的 cosine（不是与 `source_glm` 的 agreement）；
- 以及冻结 Phase3 的 Text-BLEU。

若 cosine 明显上升且 BLEU 不掉，说明 FAST 路线成立，V7 就按它做，`source_glm` 彻底退出训练目标。

---

## 7. 参考

**直接对口本次失败**
- Fu et al. *Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference* (FAST). EMNLP 2023. [PDF](https://aclanthology.org/2023.emnlp-main.1033.pdf)
- Verdini et al. *How to Connect Speech Foundation Models and Large Language Models? What Matters and What Does Not*. Interspeech 2025. [PDF](https://www.isca-archive.org/interspeech_2025/verdini25_interspeech.pdf)
- Dong & Xu. *CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition*. ICASSP 2020.

**流式 S2ST 系统**
- Deng et al. *SimulS2S-LLM*. ACL 2025. [arXiv:2504.15509](https://arxiv.org/abs/2504.15509)
- He et al. *SimulS2ST-Omni*. 2026. [arXiv:2607.19810](https://arxiv.org/html/2607.19810)
- Labiausse et al. *High-Fidelity Simultaneous Speech-To-Speech Translation* (Hibiki). ICML 2025. [项目页](https://hibiki-s2st.github.io/)
- Ouyang et al. *InfiniSST*. ACL 2025 Findings. [ACL](https://aclanthology.org/2025.findings-acl.157/)
- Zhang et al. *StreamSpeech*. ACL 2024. [arXiv:2406.03049](https://arxiv.org/abs/2406.03049)

**流式 tokenizer**
- Défossez et al. *Moshi: a speech-text foundation model for real-time dialogue*（含 Mimi codec）. 2024. [HF](https://huggingface.co/kyutai/mimi)

**本地证据**
- [`uniss_phase3_whisper_streamspeech_joint_v6_full198_stage_b_failure_analysis.md`](./uniss_phase3_whisper_streamspeech_joint_v6_full198_stage_b_failure_analysis.md)
- [`stage_b_latent_15shard_h200_execution_report.md`](./stage_b_latent_15shard_h200_execution_report.md) §10 天花板审计、§11 下游敏感性
- [`streamspeech_failure_diagnosis_and_alternative_simul_s2st_ideas.md`](./streamspeech_failure_diagnosis_and_alternative_simul_s2st_ideas.md)
- `reports/uniss_phase3_whisper_streamspeech_joint_v6/fixed_chunk_stage_a_v2_vs_stage_b_v3_v1/report.md`
