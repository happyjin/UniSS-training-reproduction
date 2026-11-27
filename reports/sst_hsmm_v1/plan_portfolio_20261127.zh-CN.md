# 读写时机：并行验证的方法组合方案（2026-11-27 CET，未启动，等你批准）

目的：不再"一个方法失败 → 再想一个"的串行流程。先把问题拆成可测量的瓶颈，再为每个瓶颈准备机制不同的方法，用同一套框架和同一套登记规则并行筛选，只把通过筛选的方法送测试。

## 0. 之前的思路算几个方法？
上一份分析（overnight_20261127_analysis）里列了 1 个诊断、3 个修补、1 个新方向和 1 个兜底：
- 第 0 步是诊断，不是方法。
- 1a / 1b / 1c 是同一个方法（预判 LoRA）的三个修补。
- 2 是真正不同的方向（减少要猜的内容）。
- 兜底的"音频侧"只是方向，没有具体方法。

所以实际只有 **2 个独立方法家族**，不够并行验证。本方案扩展为 **4 个家族、9 个方法**，每个都借自本领域或其他领域的顶会工作。

## 1. 问题（已测量，p3hk 测试集）
时延 = 内容稳定滞后 + 确认等待。

| | 内容稳定滞后 | 确认等待 | 未提交尾部每读被改写的比例 |
|---|---|---|---|
| en2zh OBSI / RealSI | 1.08 / 1.48 s | 0.54 / 0.64 s | 69% / 55% |
| zh2en OBSI / RealSI | 1.33 / 1.30 s | 0.48 / 0.60 s | 74% / 59% |

四个失败机制（夜间 8 个方案的诊断，见 overnight_20261127_analysis）：
- **F1 域外过度自信：** 在 UniST 上学到的"猜中率 / 证书可靠性"到 OBSI / RealSI 上偏高。例：S1 的后验 0.79，实际存活率 0.65。
- **F2 先验驱动的内容：** 模型分不清"截断"和"说完"（续写泄漏）。同一底座的两个模型会在同一个错误猜测上一致（S8）。
- **F3 无信息抖动：** 即使是基础 drafter，每读也要改写 55–74% 的未提交尾部。其中有多少是新音频带来的、有多少只是重翻译的非确定性，目前未知。这部分抖动直接推高"确认等待"和"稳定滞后"。
- **F4 语序依赖（en2zh 的核心）：** 目标单元依赖时间上更晚的源词，预判要猜内容本身。

提交层本身已接近信息极限（论文 anatomy），所以新方法必须改变以下之一：drafter 产生什么（F2 / F3）、证据在域外是否可信（F1）、或者要猜什么（F4）。

## 2. 本领域的现状（检索）
- **策略模型 / 端到端：** InfiniSST（多轮对话式 SimulST，IWSLT 2025 低时延赛道最好）、StreamUni（用中间转写决定生成和截断）。Seed LiveInterpret 2.0 和 Hibiki-Zero 用 RL（含 GRPO）优化时延–质量，前者报告 RL 让时延降低 20% 以上。
  这些都是重新训练策略模型，与我们"冻结 drafter + 提交层"的设定不同；我们自己的 RL 尝试因流水线对浮点扰动敏感而失败。
- **时间对齐训练：** Hibiki（ICML 2025）用对齐的合成数据让模型"只积累足够的上下文"；DSM（Kyutai 2025）把对齐挪到预处理，用延迟条件在推理时控制时延。
- **重翻译稳定化：** Arivazhagan et al. 2020 用 biased beam search + 输出遮蔽大幅降低擦除；SSBD（2025）把上一读输出当推测草稿，一次前向验证、带偏置接受，从第一个分歧处续写（加速最多 1.7×，同时减少闪烁）。
- **预判与幻觉：** TAF（NAACL 2025）用 LLM 预测多个源语言续写再投票；多篇工作报告 LLM 在部分输入上会补全幻觉，且预判加重幻觉。
- **口译员策略：** He et al. 2016 发现断句、被动化降低时延；Zhang et al. 2026 给 LLM 加 Sentence_Cut / Drop 等动作，但训练参考由 GPT-4o 生成（属于师生蒸馏，不能照搬）。
- **截断检测：** Simul-Whisper 用 integrate-and-fire 检测块边界上被截断的词，丢掉不可靠的末词，等下一块。

**结论：** 本领域还没有人系统处理"不可撤回的提交在域外的校准"（F1），也没有人把重翻译的无信息抖动（F3）和提交证书联系起来。这两点是可以做出 ICLR 级 novelty 的空白。

## 3. 方法组合（9 个方法，4 个家族）

| ID | 针对 | 机制 | 借鉴（领域 / 会议） | 训练？ | novelty | 主要风险 |
|---|---|---|---|---|---|---|
| A1 | F3 | 推测式偏置重起草：上一读草稿作为推测草稿，一次前向验证，带偏置接受，从分歧处续写 | 推测解码（ICML 2023）；SSBD（2025）；biased beam（IWSLT 2020） | 否（重拟合 HSMM） | 低–中 | 锁住错误；跨读一致失去信息量 |
| A2 | F3 + 质量 | 双向草稿选择：K 个候选中，按"后向一致（与上一读未提交尾部一致）+ 前向对比（贴近当前读的候选、远离上一读音频下的候选）"选择 | 机器人动作分块 BID（ICLR 2025）+ MBR | 否（重拟合） | **高** | 算力 1.7×（同 S7） |
| B1 | F2 | 反事实音频证书：用上一读的音频给当前草稿打分，单元 log-ratio = 它有多少来自新音频、多少来自先验，作为 HSMM 第 5 个发射 | 多模态幻觉对比解码 VCD（CVPR 2024）、CAD（NAACL 2024） | 否（重拟合） | **高** | 与 SAP 证书相关性高 |
| B2 | F3（识别） | 边界对齐的读：每读的音频截断点退到最近的低能量 / 词边界，不在半个词中间截断 | Simul-Whisper 截断检测（Interspeech 2024） | 否（重拟合） | 低 | 多等 0.1–0.2 s |
| C1 | F1 | 影子终稿 + 自适应共形提交：每句结束后额外做一次不带已提交前缀的"影子"整句解码，得到无参考的提交错误标签；用 ACI 在线调 θ，保证长期提交错误率 ≤ α | ACI（NeurIPS 2021）、共形风险控制（ICLR 2024） | 否 | **高** | 测试时在线自适应需你批准；影子译文的同义改写会算成错误 |
| C2 | F1 | HSMM 的 MAP 在线自适应：UniST 拟合作为先验，用测试流上无参考的存活信号和 C1 的影子标签做 MAP-EM 更新发射 | 语音识别说话人自适应 MAP（IEEE TSAP 1994）；测试时自适应 Tent（ICLR 2021） | 否 | 中 | 同 C1 的批准问题；实现量较大 |
| D1 | F4 | 可提交度感知的 MBR：候选效用 = 共识质量 + 已被语音支持（SAP）、可立即提交的前缀长度，在质量相当时偏向按源语序组织的译法 | 口译策略（NAACL 2016）+ MBR | 否（重拟合） | 中 | ASR-BLEU 以离线译文为参考，会惩罚顺句译法 |
| D2 | F4 | 断句动作：HSMM 发现长时间不稳定区、且前面子句已稳定时，提供一个以句号断开的候选（受约束续写），由 D1 的效用决定是否采用 | Sentence_Cut（IWSLT 2026，但不用 GPT-4o 参考）；口译策略 | 否 | 中 | 同上 |
| E1 | F2 + F1 | 观测视野条件化的 P2F：训练时给 partial / complete 标记，推理时非最后一读用 partial；HSMM 用与部署一致的 rollout 拟合（修 S3 的两个漏洞） | 扩散强制的逐 token 噪声条件（NeurIPS 2024）、DSM 延迟条件（2025） | 是（P2F 重训，同一双向模型） | 中 | en2zh 可能仍为负（F4） |

**组合关系：** A1/A2 改草稿，B1 加证书，C1/C2 改阈值或发射，D1/D2 改候选效用，E1 改 drafter。这些层彼此正交，筛选通过的可以叠加。论文主线最可能是 **B1 + C1 + A2**：一个反事实证据、一个分布漂移下有保证的提交、一个稳定的草稿选择。

### 每个方法的要点
- **A1：**
  - 实现：对上一读未提交尾部用 prompt_logprobs 打分，接受到第一个 p < τ_b 的位置，从那里续写，每读 2 次调用。
  - 关键：HSMM 必须在同样起草方式的 rollout 上重拟合（S3 的教训）。
  - 筛选信号：基础 drafter 的 instab（55–74%）是否明显下降，以及 BLEU 是否不降。
- **A2：**
  - 实现：在 hsc20（S7 MBR）上加两个效用项；缓存上一读的样本作为"弱策略"。
  - 假设：S7 的 +0.33–0.42 BLEU 来自晚提交和共识，后向一致项能去掉 S7 的 +0.12–0.14 s 抖动代价。
- **B1：**
  - 实现：每读多一个 prompt_logprobs 打分（当前草稿 + 上一读音频），代码路径复用 S8 的第 5 发射（hsmm_p2f → 通用版）。
  - 为什么可能在域外成立：它测的是"内容是否依赖刚听到的音频"，而不是"在 UniST 上猜中的频率"，所以不依赖领域的可预测性。这正是 F1 / F2 的共同根源。
- **B2：** 能量 / VAD 找最近的低能量帧（≤ 200 ms 回退）。这是唯一攻"识别滞后"的方法，先看第 0 步诊断里识别滞后占多少再决定。
- **C1：**
  - 新问题：提交是不可撤回的，所以被提交单元的对错永远观测不到（自我删失）。影子终稿给出反事实标签，这一点本领域没人做过。
  - 实现：最后一读多一个不带前缀的提示；用 chrF 对齐判定提交单元是否与影子译文一致；ACI 每句更新 θ，目标 α 取 edev 上 p3hk 的提交错误率。
- **C2：** 只有在 C1 显示"θ 漂移明显、按 cell 不同"时才做（C1 是它的简化版）。
- **D1 / D2：** 和 A2 共用 MBR 框架，只换效用函数，实现成本低。风险在评测：需要同时报告 ASR-BLEU 和对参考语序不敏感的指标（如 COMET），由你决定是否接受这个风险。
- **E1：** 只有 zh2en 前景明确（S3 已显著更早、BLEU 不降）；en2zh 靠 D 家族。训练约 1.5 小时（8 卡），可与 C1 / D1 的 CPU 部分并行。

## 4. 并行验证流程
- **阶段 0（CPU，约 2 小时，可与阶段 1 的实现并行）：**
  - (a) 用强制对齐把 1.1–1.5 s 的稳定滞后拆成识别 / 重排 / 无信息抖动三份，决定 B2 和 D 家族的优先级；
  - (b) 搭通用框架 hsc22：起草规则、额外证书、θ 控制器、候选效用都做成开关，A1–D2 共用一份代码，每个方法 bash -n + DRY_RUN；
  - (c) 一次性登记全部方法的设置和判定标准（组合 prereg，change 55），然后才看任何数字。
- **阶段 1（GPU 约 4 小时，筛选）：** 每个方法只跑一个设置。
  - 需要重拟合的先做 rollout（约 10 分钟）再拟合（约 5 分钟）。
  - 在 edev 上跑与 p3hk 相同的 θ 点，加一个与 p3hk 等时延的点。
  - 如果你批准，再在一个开放的域外 dev 上重复（见第 7 节第 2 项）。夜间方案在 edev 上全部"有效"、到测试全部无效，单靠 edev 筛选不可靠。
  - 8 张卡按 2 个方法一批轮转。
- **筛选判定（预先登记）：** 相对同条件的 p3hk，满足 (ΔBLEU ≥ −0.1 且 ΔD2 ≤ −0.10 s) 或 (ΔBLEU ≥ +0.3 且 ΔD2 ≤ +0.05 s)，且两个 dev 集都满足。
- **阶段 2（GPU 约 3–4 小时）：** 通过筛选的前 3 个方法和它们的组合上测试（每个约 50 分钟）。所有指标相对 B_b，并与 p3hk / p3hk2 / N 对比。只对有效方法做配对 bootstrap。
- **阶段 3：** 消融 + CI + 写论文（只在有效时）。
- **总计：** 约 1 天墙钟时间完成全部 9 个方法的筛选和前 3 个的测试，而不是 9 个串行夜晚。

## 5. ICLR 叙事（如果 B1 / C1 / A2 有效）
"不可撤回的流式决策在分布漂移下的校准提交"，三个贡献：
1. 指出重翻译 SimulS2ST 的提交反馈是自我删失的，提出影子终稿给出反事实标签，在此之上做有保证的风险控制（C1）；
2. 反事实音频截断证书，区分"证据驱动"与"先验驱动"的内容（B1，借自多模态幻觉）；
3. 双向草稿选择消除无信息抖动（A2，借自机器人动作分块）。

再配上时延分解的 anatomy（第 0 步）作为分析贡献。

## 6. 排除的方法与原因
- 少重排伪参考（Chen et al. 2021）、GPT-4o 动作参考（Zhang et al. 2026）：属于师生蒸馏。
- 以更多音频下的输出为目标的自一致训练：也是蒸馏。
- token 级 RL / ES：已证实流水线对浮点扰动混沌，信号被淹没。
- K 样本一致性 / 语义熵作为证书：change 38-2 已测，AUC 只 +0.010–0.015。
- prefix-to-prefix / 对齐前缀训练：等价于"用 SAP verifier 当翻译器"，已证实慢 0.6–0.8 s。
- 硬一致上限（S4）和 chrF-MBR（S6 / S7）：已测。
- TAF 式语音版源语言未来粒子（K 个未来 × 翻译）：机制对 F4 最直接，但算力是 K×，且需要"音频 + 预测文本"的新提示格式。放到第二轮，看 D 家族的结果再定。

## 7. 需要你决定
1. **是否允许测试时在线自适应（C1 / C2）：** 只用模型自己的影子译文，不用参考，严格因果。这与"RealSI 只测不调"的规则有交集，需要你明确同意。若不同意，C1 改为只在 dev 上校准 α 的离线版，novelty 下降。
2. **是否允许一个开放的域外 dev 集用于筛选：** 例如 CoVoST dev 的语流，只用于筛选、不训练，与 OBSI / RealSI 不重叠，用前先查重。这违反"所有方法用同一 UniST 数据"的约定，只在筛选阶段使用。
3. **D 家族的评测风险：** 顺句译法可能被 ASR-BLEU 惩罚。要不要加对参考语序不敏感的辅助指标（如 COMET）一起看，还是严格只看 ASR-BLEU？
4. **预算：** 阶段 0–2 约 1 天 GPU 和中等 token 量（框架一次实现，之后只看事件）。是否全部 9 个方法都跑，还是先跑 A2 / B1 / C1 / E1 这 4 个最有希望的？

## 参考文献
- Arivazhagan et al. 2020. Re-translation versus Streaming for Simultaneous Translation. IWSLT. https://aclanthology.org/2020.iwslt-1.27
- Zeng et al. 2025. Self-Speculative Biased Decoding for Faster Re-Translation. https://arxiv.org/abs/2509.21740
- Leviathan, Kalman, Matias 2023. Fast Inference from Transformers via Speculative Decoding. ICML. https://arxiv.org/abs/2211.17192
- Liu et al. 2025. Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling. ICLR. https://arxiv.org/abs/2408.17355
- Black, Galliker, Levine 2025. Real-Time Execution of Action Chunking Flow Policies. NeurIPS. https://arxiv.org/abs/2506.07339
- Zhao et al. 2023. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT, temporal ensembling). RSS. https://arxiv.org/abs/2304.13705
- Leng et al. 2024. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding. CVPR. https://arxiv.org/abs/2311.16922
- Shi et al. 2024. Trusting Your Evidence: Hallucinate Less with Context-aware Decoding. NAACL. https://arxiv.org/abs/2305.14739
- Wang et al. 2024. Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection. Interspeech. https://arxiv.org/abs/2406.10052
- Gibbs, Candès 2021. Adaptive Conformal Inference Under Distribution Shift. NeurIPS. https://arxiv.org/abs/2106.00170
- Angelopoulos et al. 2024. Conformal Risk Control. ICLR. https://arxiv.org/abs/2208.02814
- Gauvain, Lee 1994. Maximum a posteriori estimation for multivariate Gaussian mixture observations of Markov chains. IEEE TSAP 2(2). https://ieeexplore.ieee.org/document/279278
- Wang et al. 2021. Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR. https://arxiv.org/abs/2006.10726
- He, Boyd-Graber, Daumé III 2016. Interpretese vs. Translationese. NAACL. https://aclanthology.org/N16-1111/
- Zhang, Yang, Nakamura 2026. Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-Like Strategies. IWSLT. https://arxiv.org/abs/2601.11002
- Chen et al. 2021. Improving Simultaneous Translation by Incorporating Pseudo-References with Fewer Reorderings. EMNLP. https://aclanthology.org/2021.emnlp-main.473
- Chen et al. 2024. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. NeurIPS. https://arxiv.org/abs/2407.01392
- Zeghidour, Défossez et al. 2025. Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling. https://arxiv.org/abs/2509.08753
- Labiausse et al. 2025. High-Fidelity Simultaneous Speech-To-Speech Translation (Hibiki). ICML. https://arxiv.org/abs/2502.03382
- Kyutai 2026. Simultaneous Speech-to-Speech Translation Without Aligned Data (Hibiki-Zero). https://arxiv.org/abs/2602.11072
- Seed Team 2025. Seed LiveInterpret 2.0. https://arxiv.org/abs/2507.17527
- Ouyang et al. 2025. Anticipating Future with Large Language Model for Simultaneous Machine Translation (TAF). NAACL. https://arxiv.org/abs/2410.22499
- InfiniSST 2025. https://arxiv.org/abs/2503.02969 ；StreamUni 2025. https://arxiv.org/abs/2507.07803
- Farquhar et al. 2024. Detecting hallucinations in large language models using semantic entropy. Nature. https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750
