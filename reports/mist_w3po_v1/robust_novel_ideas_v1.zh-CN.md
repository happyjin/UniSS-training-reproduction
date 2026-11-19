# 鲁棒多说话人同传：同时提升 BLEU 与音质、LAAL 不升的 novel idea（文献调研 v1）

2026-11-19 布拉格。4 个文献检索代理并行调研（推理时鲁棒性 / 视觉"为任务服务的复原" / 语音大模型与语音翻译鲁棒性 / 生成侧引导与解耦），结合 Idea 1–7 的实验证据整理。

**注意：** 部分文献只读到摘要或二手总结（文中标 †），立项前必须核实原文的公式与数字。

---

## 0. 出发点：我们已经知道什么

| 证据 | 数字 | 含义 |
|---|---|---|
| 噪声只伤害英→中的**翻译文本** | 英→中文本 BLEU：干净 38.57，ID-2 36.67，OOD-1 37.07；中→英：18.49 / 18.53 / 18.73 | Thinker 侧有约 **+1.5–1.9 BLEU** 的上限空间，只在英→中 |
| 音色发闷会降低 **ASR-BLEU** | OOD-1 中→英：B 17.74/15.60 → 音色去噪后 19.1（+2.45） | Talker 侧的音质改善能直接转化成 ASR-BLEU |
| 英→中的"文本→ASR-BLEU"有约 3.2 分的固定损失 | 与噪声条件无关 | 中文合成可懂度是另一个独立瓶颈 |
| 已测方法的"BLEU 下降"在种子噪声以内 | B 自己两种子差 0.05–2.14；方法差多为 −0.02 到 −0.28；逐窗口 chrF 无显著变差 | 门槛必须用多种子 + 逐窗口检验 |
| 音色 token 改变 Talker 语速 → LAAL 漂移 | 时长 ±5%，LAAL −0.034 到 +0.12 s | LAAL 可以通过控制语速保持不变甚至降低 |
| 一条流里的背景被所有说话人共享 | ICC 0.26（Idea 5） | 多说话人同传独有的结构，可以利用 |
| 话轮之间存在纯背景窗口 | 非语音窗口与流残差余弦约 0.2 | 可以在线、因果地拿到"纯噪声参照" |
| 码空间误差与音质反向 | 码误差 7.36 < 7.95 < 9.58，PESQ 排序正相反 | **所有门槛只能用真实输出指标**（PESQ、ASR-BLEU、相似度、LAAL） |
| 后验均值压扁音色分布，但不能解释逐条音质 | Idea 7：M1 成立、M2 4/9 | 失真–感知 / 最优传输路线已有反面证据 |
| SEEN（ACL 2026）已做"音频大模型编码器内投影掉噪声子空间" | — | Thinker 上直接做 SEEN 没有新意 |
| 现有最强同传系统都没有抗噪评测 | Hibiki（ICML'25）、Hibiki-Zero（ICML'26）、SimulS2S-LLM（ACL'25）、Seed LiveInterpret 2.0 | **抗噪同传 + 评测集本身是空白** |

**目标：** BLEU 和音质同时提升，LAAL 不变或更低（小幅上升可接受）；有顶会级新意；每个方法只跑一个设置，先做便宜的预检。

---

## 1. 统一主线：流背景锚（Stream Background Anchor）

四个方向的代理各自独立得出同一个核心：**用每条流自己话轮之间的纯噪声间隙，在线估计一个所有说话人共享的背景表示，再把它同时用在 Thinker 和 Talker 上。**
- 它利用的是多说话人流式同传独有的两个结构（共享背景、话轮间隙），离线单说话人系统没有这两样；
- 它直接回应 SEE 论文自述的局限："需要事先采集部署环境的纯噪声录音"；
- 已有的 Idea 5 证实了背景共享（ICC 0.26），提供了动机证据。

下面的 idea A、C、D 是这条主线的三个落点，可以组成一篇论文；B 负责保住 LAAL。

---

## 2. 候选 idea（按"新意 × 能 work 的可能性"排序）

### A. 流内噪声负条件引导 Talker（Noise-Negative Voice Guidance）★ 首选

- **动机：** 音色去噪（C）已经把音质拉高，但仍残留"发闷"的成分，而且残差和身份纠缠，在 token 空间很难再分离（Idea 4/5/6/7）。与其继续"估得更准"，不如在生成时**主动推离"带噪音色"这个方向**。
- **参考：**
  - MM-Sonate †（arXiv 2601.01568, 2026）：用"噪声的说话人嵌入"代替无条件分支做 CFG，WER 3.17% → 2.06%；
  - Stay on topic with Classifier-Free Guidance（ICML 2024, arXiv 2306.17806）：CFG 用于自回归语言模型；
  - Dynamic Negative Guidance（ICLR 2025, arXiv 2410.14398）；
  - Koel-TTS（EMNLP 2025, arXiv 2502.05236）。
- **做法：** Talker 每步跑两条分支，batch=2 一起算：
  - 正分支用去噪后的 32 个音色 token g_clean（C 的输出）；
  - 负分支用本流的负参照 g_neg，二选一：原始带噪提示的 token，或日志器标为非语音的间隙编码出的 token；
  - logits = l(g_clean) + w·[l(g_clean) − l(g_neg)]，w 只取一个固定值；BiCodec 解码器仍用 g_clean。
  - 两个分支都是真实音色码，Talker 不需要重新训练。
- **新意：** MM-Sonate 用的是通用白噪声，模型是非自回归 DiT；Koel-TTS 是去掉上下文，不是对比。**没找到用同一场景的背景作负条件、用于自回归编解码 LM 或 S2ST 的工作。**
- **预期：** 音质 ↑；在声音发闷的地方 ASR-BLEU ↑。每步多一次 batch 内前向，解码阶段受显存带宽限制，单步约 +10–30% 耗时。引导可能改变语速，所以要和 B 一起用。
- **风险：** 语音里 CFG 技巧常常失灵（arXiv 2509.19668 †），可能出现过度引导的伪影。
- **预检：**
  1. 先算 l(g_clean) 与 l(g_neg) 的逐步 KL，确认两个分支确实有差别；
  2. 在 dev 上用一个固定 w（=1），比较 C 和 C + 引导的 PESQ、ASR-BLEU、tokens/s。门槛：PESQ ↑ 且 ASR-BLEU 不降、相似度 ≥ −0.02。

### B. 与音色无关的语速钳制（Voice-Invariant Rate Clamp）★ 保 LAAL

- **动机：** 我们实测到音色 token 会让 Talker 时长变化 ±5%，LAAL 漂移最多 +0.12 s。这是所有音色方法共同的延迟来源。
- **参考：**
  - Controlling Speaking Rate in Autoregressive TTS via Activation Steering †（SLT 2026, arXiv 2609.33810）：在中层找到"语速轴"，把投影钳到目标值，身份和自然度不受影响；
  - VoXtream2 †（arXiv 2603.13518）。
- **做法：** 用时间拉伸过的 Talker 数据学出语速轴。推理时，把隐状态在这个轴上的投影钳到"这段文本在 B 下的期望语速"，使音色 token 和 A 的引导都改变不了语速。
- **新意：** 语速操控本身已有。新的是"让语速对音色条件不变"，并且以同传 LAAL 为目标。
- **预期：** 不需要额外前向。消除 ±0.1 s 的漂移，可能让 LAAL 略降。
- **风险：** 加速可能损失可懂度。
- **预检：** 用线性探针从隐状态预测 tokens/s，看预测值随音色码的变化有多大；钳制后在 dev 上看 LAAL 和 ASR-BLEU。

### C. 背景对比解码 Thinker（只在英→中打开）

- **动机：** 噪声让英→中翻译文本掉 1.5–1.9 BLEU，这只能在 Thinker 上解决。SEEN 改的是编码器内部，我们改解码。
- **参考：**
  - VCD（CVPR 2024, arXiv 2311.16922）；
  - Context-Aware Decoding（NAACL 2024, arXiv 2305.14739）；
  - Audio-Aware Decoding †（arXiv 2506.07233）；
  - Whisper-CD †（arXiv 2603.06193）；
  - Contrastive Decoding（ACL 2023, arXiv 2210.15097）。
- **做法：** 负分支用同一个流式窗口，但把语音换成本流里等长的纯背景间隙；和主分支 batch 在一起。logits = l_full − α·l_bg，只在主分支认为合理的 token 上生效（自适应可信截断）。用噪声能量分数门控，只在噪声大的窗口、只在英→中方向打开；α 只取一个值。
- **新意：** 已有的音频对比解码针对的是幻觉，负条件用高斯噪声、静音或空白。**没找到用于语音翻译、并且用流内真实背景作负条件的工作。**
- **预期：** 英→中 BLEU 收回一部分噪声损失。负分支并行计算，不额外读入音频，LAAL 基本不变。
- **风险：** 减去先验可能伤害流畅度。
- **预检：** 离线用已有的带噪英→中 dev 窗口，α=0.5 固定，比较翻译文本 BLEU。门槛：收回至少一半的噪声损失（≥ +0.8）。

### D. 只在话轮间隙里更新的在线测试时自适应

- **动机：** 背景在一条流里持续存在，并且被说话人共享；而话轮间隙不在出译文的关键路径上，可以拿来做免标签的自适应。
- **参考：**
  - EATA（ICML 2022, arXiv 2204.02610）；
  - SAR（ICLR 2023, arXiv 2302.12400）；
  - Test-Time Training on Video Streams（JMLR 2025, arXiv 2307.05014）；
  - E-BATS †（NeurIPS 2025, arXiv 2506.07078）；
  - DSUTA（arXiv 2406.11064）；
  - SLM-TTA †（arXiv 2512.24739）。
- **做法：**
  - 每遇到一个间隙，就更新 Thinker 音频编码器输入端的一个很小的偏移量，免标签损失为：把间隙的特征对齐到训练时静音段的统计量；
  - 这个偏移量在整条流里用 EMA 共享给所有说话人，同时也加在音色编码器之前；
  - 用 EATA 式过滤和定期重置防止漂移。
- **新意：** SLM-TTA 覆盖了语音翻译，但是按批次、不是流式；E-BATS 和 DSUTA 只做 ASR。**没有工作用话轮间隙作监督信号，也没有工作针对同传。**
- **预期：** BLEU ↑（英→中）；音色编码器的输入更干净，所以音质也 ↑。更新在关键路径之外，LAAL 不变。
- **风险：** 自适应漂移。
- **预检：** 回放 3 条带噪 dev 流，看间隙特征是否向静音统计靠拢、英→中文本 BLEU 是否变化。

### E. 干净特权的流式自蒸馏，包括"何时写出"的决策

- **动机：** 带噪时 Thinker 不仅会翻错，还可能更晚写出。用"听干净音频的自己"当老师，同时教内容和时机。
- **参考：**
  - EchoDistill †（arXiv 2605.23954，QA 任务 +1.6–3.6 pp）；
  - StableToken（arXiv 2509.22220）；
  - 语音翻译一致性正则（NAACL 2024）。
- **做法：**
  - 用开源干净英→中训练数据，混入开源噪声（MUSAN、人群声、音乐）；
  - 老师 = 冻结的 Thinker 在干净窗口上的输出；
  - 学生用 LoRA 在带噪窗口上学习，蒸馏两样东西：每窗口的译文前缀（token 级 KL），以及提交/停止决策。
  - 只用开源训练数据。
- **新意：** 干净老师蒸馏本身已有（偏 QA 和 ASR）。**没有人用于同传，也没有人蒸馏提交决策。**
- **预期：** BLEU ↑（英→中）；LAAL 持平或略降。对音质没有直接作用。
- **风险：** 合成噪声和真实电竞/会议噪声不匹配。需要训练，成本中等。
- **预检：** 在 dev 上算干净 vs 带噪窗口的 Thinker 输出 KL，看它是否和逐句 BLEU 下降相关，并且英→中大于中→英。

### F. 按声学可靠度分配等待时间 + 用语速抵消等待

- **动机：** 噪声大的窗口多等一会再写，能提高正确率；多花的延迟用 B 的语速控制补回来。
- **参考：**
  - REINA †（arXiv 2604.09916）；
  - DiG-SST（AAAI）；
  - EASiST（arXiv 2504.11809）；
  - Dub-S2ST（EMNLP Findings 2025, arXiv 2505.20899）。
- **做法：** 训练一个小头预测"干净–带噪输出差异"作为可靠度分数：可靠度低就多读，高就早写；总延迟预算由 Talker 的语速吸收。
- **新意：** 还没有发表的读写策略以声学可靠度或信噪比为条件。
- **风险：** 再分配不当反而让整体 LAAL 上升。
- **预检：** 保持 BLEU 不变，把最噪 20% 和最干净 20% 窗口之间的等待时间对调，看效果。

### G. 对下游任务友好的伪干净目标（一个前端同时服务 Thinker 和音色编码器）

- **参考：** Rethinking Image Restoration for Object Detection（NeurIPS 2022）；Image-Adaptive YOLO（AAAI 2022, arXiv 2112.08088）；FeatEnHancer（ICCV 2023, arXiv 2308.03594）；When De-noising Hurts †（arXiv 2512.17562）。
- **做法：** 训练目标不是干净音频，而是"干净音频 + 几步让 Thinker 翻译损失和音色损失都下降的定向扰动"。用这个目标训练一个因果前端，同时输入给 Thinker 和音色编码器。
- **新意：** 语音领域还没有用"定向攻击"构造伪目标的做法，也没有用在音频大模型或 TTS 上。
- **风险：** 前端会带来频谱伪影，而 SEEN 表明大模型对这类伪影敏感；因果前端还会增加约 20–40 ms 延迟。
- **预检：** 对约 200 个 dev 片段构造伪目标，直接喂给 Thinker。如果 BLEU 不能超过干净时的 BLEU，就停止。

### H. 背景"噪声寄存器"前缀（低优先级）

- **参考：** Vision Transformers Need Registers（ICLR 2024）；Vision Transformers Don't Need Trained Registers †（NeurIPS 2025）；Online Register †（arXiv 2602.23702）。
- **做法：** 把编码后的话轮间隙作为 KV 缓存前缀放在 Thinker 上下文里，让注意力用它"解释掉"噪声。
- **风险：** 不训练的话，模型可能直接忽略这个前缀。
- **预检：** 看前缀分到的注意力质量，以及带/不带前缀的 dev BLEU。

---

## 3. 不推荐的方向

| 方向 | 不推荐的原因 |
|---|---|
| 后验均值 + 整流流 / 最优传输修正（PMRF, ICLR 2025；Freirich et al., NeurIPS 2021） | 我们的 Idea 7 已有反面证据（M2 4/9） |
| Thinker 上直接做 SEEN、网络内特征去噪（CVPR 2019） | SEEN 已经做过，新意不足 |
| AugMix 一致性 + WiSE-FT 插值 | 语音领域已经很拥挤（EchoDistill、Whisper 蒸馏、NAACL 2024），只适合当组件 |
| 先转写再翻译（CoT-ST） | 会增加 LAAL，与目标冲突 |
| 用翻译奖励训练增强前端 | 新意有，但重，并且增加 20–40 ms 延迟；可并入 G |
| 用"差模型"引导 Talker（autoguidance, NeurIPS 2024） | ETTA 报告在音频上不稳定；作为 A 的备选 |

---

## 4. 建议的论文故事与执行顺序

**论文主线：** 背景感知的抗噪多说话人同传。
- 贡献 1：OpenBroadcast-SI 抗噪同传评测集。现有最强系统都没有抗噪评测，这本身是空白。
- 贡献 2：流背景锚。从话轮间隙在线估计、所有说话人共享的背景表示，同时用于 Talker 的负条件引导（A）、Thinker 的对比解码（C）和间隙自适应（D）。
- 贡献 3：语速钳制（B），使音色和鲁棒性改动不增加 LAAL。
- 分析：码误差和音质反向、音色影响语速与延迟、少量会议做 BLEU 门槛不可靠等。

**执行顺序（先做便宜的预检，每个只跑一个设置，门槛只用真实输出指标）：**
1. **A + B 预检：** dev 上 Talker 引导 + 语速钳制，同时看 PESQ、ASR-BLEU、相似度、LAAL。GPU 用量小。
2. **C 预检：** 离线在带噪英→中 dev 窗口上做对比解码，看翻译文本 BLEU。GPU 用量小。
3. **D 预检：** 回放 3 条带噪 dev 流，验证间隙自适应的方向是否正确。
4. 通过预检的方法组合起来，用多种子、扩展到更多场会议的 OOD-3 跑完整系统测试。

**前置条件：**
- 需要一个 **OpenBroadcast-SI 的 dev 版本**（不碰 test），用来做方法选择；
- OOD-3 要扩到更多场会议，否则"不伤害"的结论站不住。

**新意的边界（要如实写进论文）：** 负条件 CFG、对比解码、测试时自适应、语速操控都是已有的机制。我们的新意在于：
1. 利用多说话人流式同传独有的"共享背景 + 话轮间隙"在线构造负参照和自适应信号；
2. 用同一个背景表示同时改善翻译（Thinker）和合成（Talker）；
3. 把延迟作为约束显式处理。

立项前要对 A、C、D 再做一次全文级的文献核查，尤其是 MM-Sonate、Whisper-CD、SLM-TTA、EchoDistill。

---

## 参考文献（按出现顺序）

- SEE / SEEN，ACL 2026：https://arxiv.org/abs/2601.07331
- Hibiki，ICML 2025：https://arxiv.org/abs/2502.03382；Hibiki-Zero，ICML 2026：https://arxiv.org/abs/2602.11072；SimulS2S-LLM，ACL 2025：https://aclanthology.org/2025.acl-long.817；Seed LiveInterpret 2.0：https://arxiv.org/abs/2507.17527
- MM-Sonate †：https://arxiv.org/abs/2601.01568；Koel-TTS：https://arxiv.org/abs/2502.05236；Stay on topic with CFG，ICML 2024：https://arxiv.org/abs/2306.17806；Dynamic Negative Guidance，ICLR 2025：https://arxiv.org/abs/2410.14398
- Rate steering in AR TTS †，SLT 2026：https://arxiv.org/abs/2609.33810；VoXtream2 †：https://arxiv.org/abs/2603.13518
- VCD，CVPR 2024：https://arxiv.org/abs/2311.16922；Context-Aware Decoding，NAACL 2024：https://arxiv.org/abs/2305.14739；Audio-Aware Decoding †：https://arxiv.org/abs/2506.07233；Whisper-CD †：https://arxiv.org/abs/2603.06193；Contrastive Decoding，ACL 2023：https://arxiv.org/abs/2210.15097
- EATA，ICML 2022：https://arxiv.org/abs/2204.02610；SAR，ICLR 2023：https://arxiv.org/abs/2302.12400；TTT on Video Streams，JMLR 2025：https://arxiv.org/abs/2307.05014；E-BATS †，NeurIPS 2025：https://arxiv.org/abs/2506.07078；DSUTA：https://arxiv.org/abs/2406.11064；SLM-TTA †：https://arxiv.org/abs/2512.24739
- EchoDistill †：https://arxiv.org/abs/2605.23954；StableToken：https://arxiv.org/abs/2509.22220
- REINA †：https://arxiv.org/abs/2604.09916；EASiST：https://arxiv.org/abs/2504.11809；Dub-S2ST，EMNLP Findings 2025：https://arxiv.org/abs/2505.20899
- Rethinking Image Restoration for Object Detection，NeurIPS 2022：https://proceedings.neurips.cc/paper_files/paper/2022/file/1cac8326ce3fbe79171db9754211530c-Paper-Conference.pdf；IA-YOLO，AAAI 2022：https://arxiv.org/abs/2112.08088；FeatEnHancer，ICCV 2023：https://arxiv.org/abs/2308.03594；When De-noising Hurts †：https://arxiv.org/abs/2512.17562
- Vision Transformers Need Registers，ICLR 2024：https://arxiv.org/abs/2309.16588；Don't Need Trained Registers †，NeurIPS 2025：https://arxiv.org/abs/2506.08010；Online Register †：https://arxiv.org/abs/2602.23702
- PMRF，ICLR 2025：https://arxiv.org/abs/2410.00418；Distortion-Perception in Wasserstein Space，NeurIPS 2021：https://arxiv.org/abs/2107.02555；Autoguidance，NeurIPS 2024：https://arxiv.org/abs/2406.02507；Selective CFG in speech †：https://arxiv.org/abs/2509.19668
- 带噪提示 TTS 相关：LauraTTS token 去噪 https://arxiv.org/abs/2505.13830；Noise-robust zero-shot TTS https://arxiv.org/abs/2406.05699

---

## 5. 与已测 6 个版本的组合：统一框架"背景锚定的零噪声外推"（v1.1 追加，2026-11-19）

### 5.1 哪些旧版本能接进来

| 已测版本 | 在新框架里的角色 | 能否提升新意 |
|---|---|---|
| Idea 1 C（去噪编码器） | 提供正条件 g⁺（音质最好的音色 token） | 组件，本身不新 |
| Idea 1 Z（ZNE 外推） | **提供理论框架**：把引导解释为在输出 logit 空间做 Richardson 零噪声外推，权重由噪声水平算出 | **是**：H1 在编码器空间失败，这里换到 Talker/Thinker 的输出空间 |
| Idea 2 冻结版（收缩后验） | 提供生成模型 z = λs + (1−λ)(c+b) 和每窗口的 λ（信噪比），用来**算外推权重** | **是**：λ 从"估音色"变成"定引导强度" |
| Idea 5（层级共模背景，未上完整测试） | **估计背景锚** b̂：闭式 Schur 补联合后验，用话轮间隙和其他说话人 | **是**：作为音色估计器它太弱（−4.9%），作为负条件来源正好对口 |
| Idea 6 C_post / E2E | 可选：逐窗口可靠度 α_k 作为 Thinker 侧的门控信号 | 弱，E2E 已证实过拟合 |
| Idea 2 端到端版 | 不接：KL 目标和音质错位 | — |

### 5.2 推导

**(1) 背景感知的生成模型（Idea 2 + Idea 5）。** 第 k 个窗口、说话人 j 的音色编码：

$$z_k = \lambda_k s_j + (1-\lambda_k)\,(c + b) + \varepsilon_k,\qquad s_j \sim \mathcal N(\mu_0,\Sigma_0),\; b \sim \mathcal N(0,\Sigma_b),\; \varepsilon_k \sim \mathcal N(0, R/\pi_k).$$

话轮间隙窗口 λ_k = 0，直接观测 c + b。所有说话人的 s_j 与 b 的联合后验是闭式的（Idea 5 已实现的 Schur 补）：

$$\Big(\Sigma_b^{-1} + \sum_j C_j G - \sum_j B_j^2\, G M_j^{-1} G\Big)\,\hat b \;=\; G\sum_j y^{b}_j - \sum_j B_j\, G M_j^{-1}\big(\Sigma_0^{-1}\mu_0 + G y^{s}_j\big).$$

正条件 g⁺ = Q(C 编码器输出 或 ŝ_j)，负条件 g⁻ = Q(c + b̂)，Q 是 FSQ 量化。

**(2) 引导 = 零噪声外推（Idea 1 Z 的理论搬到输出空间）。** 设条件里的背景占比为 σ ∈ [0,1]（σ = 1 − λ），Talker 每步的 logit 是 σ 的光滑函数：

$$\ell(\sigma) = \ell(0) + a\,\sigma + O(\sigma^2).$$

在两个已知噪声水平 σ⁺（去噪后的正条件）和 σ⁻（负条件；纯背景时 σ⁻ = 1）上各算一次，一阶 Richardson 外推到 σ = 0：

$$\hat\ell(0) = \frac{\sigma^- \ell(\sigma^+) - \sigma^+ \ell(\sigma^-)}{\sigma^- - \sigma^+} = \ell(\sigma^+) + w\,\big[\ell(\sigma^+) - \ell(\sigma^-)\big],\qquad w = \frac{\sigma^+}{\sigma^- - \sigma^+}.$$

这正是负条件 CFG 的形式，但 **w 不用调，而是由噪声水平算出**：
- σ⁺ 由 Idea 2 的 λ 头给出原始噪声水平，再乘以去噪编码器在 dev 上测得的残差比 ρ（只在 dev 上测一次，固定）；
- 干净说话人 σ⁺ → 0，w → 0，自动不引导；噪声大的说话人引导更强。

**偏差–方差：** 外推的方差约为 (1 + w)² + w² 倍单次方差，σ⁺ 越接近 σ⁻ 越不稳定。可以加一个原始带噪提示（σ_raw）作为第三个点做二阶外推，并用它检验线性假设。

**(3) Thinker 侧同理（候选 C）。** 窗口 k 的 logit：

$$\hat\ell_T = \ell_T(x_k) + w_k\big[\ell_T(x_k) - \ell_T(x^{bg}_k)\big],\qquad w_k = \frac{1-\lambda_k}{1-(1-\lambda_k)} = \frac{1-\lambda_k}{\lambda_k},$$

其中 x^bg 是本流等长的话轮间隙。干净窗口 λ_k → 1 时 w_k → 0，所以中→英这类不受噪声影响的情形**自动关闭**，不需要人为门控。

**(4) 语速钳制保 LAAL（候选 B）。** 设 Talker 时长被放大 (1+ε)，单元 i 内第 m 个词的发声时刻为 t_m(1+ε)，则

$$\Delta\mathrm{LAAL} \approx \varepsilon\cdot \mathbb E_i\Big[\tfrac{1}{|i|}\sum_m t_m\Big],$$

与我们实测的"时长 +5% → LAAL +0.05–0.12 s"同量级。钳制：在中层隐状态上，沿语速方向 r（线性探针学得）把投影固定到文本期望值 r̂(text)：

$$h \leftarrow h - \big(r^\top h - \hat r(\text{text})\big)\,r,$$

使 ε ≈ 0，从而 ΔLAAL ≈ 0，且对引导造成的语速变化同样有效。

**(5) 可选的训练版本（加分项）：**
- 引导蒸馏：把两次前向蒸馏成一次，参考 On Distillation of Guided Diffusion Models（CVPR 2023），消除额外计算；
- 对比式训练 C：让 g⁺ 与 g⁻ 在 Talker 输出分布上的 KL 更大，用 InfoNCE 形式推导，增强引导信号。

### 5.3 新意

单独的负条件 CFG、对比解码、测试时自适应都已有。组合后的新意在于：
1. **引导权重有闭式来源：** 引导 = 零噪声外推，w 由贝叶斯模型估计的噪声水平决定，不调参，并给出偏差–方差分析；
2. **负条件有贝叶斯来源：** 背景锚由多说话人共享背景的层级模型从话轮间隙在线估计；
3. **一套公式同时用于翻译（Thinker）和合成（Talker）；**
4. **延迟有显式约束（语速钳制）。**

另外，之前的负结果（Idea 1 的 ZNE、Idea 5 的共模背景）在这里变成了有作用的组件，论文叙事更完整。

### 5.4 预检（先验证核心假设：logit 对 σ 近似线性）

- **线性检验：** 在 dev 上，用本流间隙噪声把干净提示混成 3–4 个已知 σ，看 Talker 每步 logit 随 σ 的变化能否被一次函数解释。门槛：R² ≥ 0.8。Thinker 侧同样检验。
- **线性成立才做外推：** 在 dev 上比较 C 与 C + ZNE 引导（w 由公式算出，不扫参）的 PESQ、ASR-BLEU、相似度、LAAL（同时打开语速钳制）。
- 线性不成立：外推框架的理论依据不足，退回普通负条件 CFG（固定 w = 1），或停止。

**需要如实说明：** Idea 1 的 H1（编码器空间的 ZNE）没有得到支持；这里假设的是输出空间的线性，必须先过线性检验。

---

## 6. 输入侧方法（进入大模型之前 / 编码器内部处理）与组合（v1.2 追加，2026-11-19）

### 6.1 文献里的输入侧方法

| 类别 | 代表工作 | 做法 | 对我们的启示 |
|---|---|---|---|
| 编码器内噪声子空间投影 | SEE/SEEN（ACL 2026, arXiv 2601.07331） | 干净/纯噪声样本的编码器激活做 SVD，取与语义方向近正交的噪声方向，推理时减去激活在其上的分量（免训练） | 直接可用作 Thinker 的**基线**；需要事先的纯噪声录音（论文自述局限） |
| 增强前端 + 观测叠加 | Iwamoto et al., "How bad are artifacts?" Interspeech 2022（arXiv 2201.06685）；When De-noising Hurts †（arXiv 2512.17562） | 增强输出带伪影会伤害识别；把增强信号与原始带噪信号按比例混合（observation adding）可缓解 | 混合比例可由 Idea 2 的 λ 逐窗口闭式给出 |
| 测试时自适应编码器 | SLM-TTA †、E-BATS †（NeurIPS 2025）、EATA（ICML 2022） | 推理时更新少量参数或输入偏移 | 即 K3，用话轮间隙作信号 |
| 下游任务驱动的增强 | Rethinking Image Restoration for Object Detection（NeurIPS 2022）；IA-YOLO（AAAI 2022） | 增强目标由下游损失决定 | 即 G，训练较重，放第二波 |
| 寄存器 / 前缀吸收噪声 | Registers（ICLR 2024 / NeurIPS 2025 †） | 额外 token 吸收背景激活 | 即 H，低优先级 |
| 经典噪声估计 | VAD 间隙估计噪声谱 + 谱减 / Wiener | 用非语音段估计噪声 | 新意低，但"间隙估计 + 跨说话人共享"的思路可迁移到表示空间 |

另外要注意：我们已测的 Idea 1 C、Idea 2、Idea 6 本身就是 **Talker 的"输入侧"处理**（在音色提示进入 Talker 之前去噪），与 SEEN 处理 Thinker 输入是对称的两半。

### 6.2 SEEN 在我们场景里会有效吗

- **可能有效，但只在英→中，且需要实测：** 噪声让英→中翻译文本掉 1.5–1.9 BLEU（有空间），中→英几乎不掉（没有空间）。
- **不确定因素：**
  - SEEN 测的是 −10 dB 下的 GSR（问答 / 转写的相对稳健性），没有翻译和 BLEU；我们的背景强度分布更温和。
  - 它用整句均值池化估计子空间，流式短窗口下的稳定性未知。
  - 我们的 Idea 4 在音色码空间发现噪声残差不低秩、并与身份纠缠（重叠是随机水平的 6.6 倍）；SEEN 的"与语义方向余弦 < 0.1 才保留"的过滤正是针对这种纠缠，但 Thinker 编码器空间是否低秩要实测。
- 结论：把**原版 SEEN 作为基线**测一次（K6）；我们的新意放在"在线、贝叶斯、软投影、跨说话人共享、同时用于 Thinker 和音色编码器"的版本上（K7）。

### 6.3 推导：在线软 SEEN（K7）

**(a) 子空间内的 Wiener 增益代替硬投影。** 对帧激活 a，噪声子空间基 Q = [q_1,…,q_r]。沿 q_i 的分量 x_i = q_iᵀa = s_i + n_i，设干净语音方差 σ²_{s,i}（干净集上估计）、噪声方差 σ²_{n,i}。s_i 的 MMSE 估计为 ŝ_i = σ²_{s,i}/(σ²_{s,i}+σ²_{n,i}) · x_i，因此应减去的部分是

$$\tilde a = a - \sum_i \gamma_i\,(q_i^\top a)\,q_i,\qquad \gamma_i = \frac{\sigma^2_{n,i}}{\sigma^2_{n,i}+\sigma^2_{s,i}}.$$

SEEN 是 γ_i ≡ 1 的特例（硬投影）。

**(b) 每条流的噪声协方差用逆 Wishart 收缩在线估计（Idea 5 的层级思想）。** 全局噪声协方差 Σ_g（训练噪声上估计）作先验，Σ_stream ~ IW(ν, (ν−d−1)Σ_g)。话轮间隙累计 n 帧、散度矩阵 S 后，后验均值

$$\hat\Sigma_{stream} = \frac{S + (\nu-d-1)\,\Sigma_g}{n + \nu - d - 1}.$$

间隙越长越相信本流，越短越靠全局先验；对所有说话人共享。Q 取 Σ̂_stream 的主特征向量，并按 SEEN 的规则过滤掉与语义方向余弦 ≥ 0.1 的方向；σ²_{n,i} = q_iᵀ Σ̂_stream q_i。

**(c) 两个使用者共用一个背景锚：** 同一个 Σ̂_stream 的思想分别用于 Thinker 编码器层（K7）和 BiCodec 说话人编码器的中间特征（K7v）。

### 6.4 推导：λ 自适应的观测叠加（K8）

带噪输入 x_n = s + n（噪声相对能量 ∝ (1−λ)/λ），增强输入 x_e = s + e（伪影相对能量 σ²_a，在训练数据上测一次后固定）。两者误差独立时，线性组合 αx_e + (1−α)x_n 的误差能量最小解是逆方差加权：

$$\alpha^* = \frac{\sigma_n^2}{\sigma_n^2 + \sigma_a^2},\qquad \sigma_n^2 = \frac{1-\lambda}{\lambda}.$$

干净窗口 λ → 1 时 α* → 0，自动用原始输入，避免"去噪伤害识别"。增强器用流程里已有的 HDemucs。

### 6.5 推导：用 SEE 估计零噪声外推的 σ（K9）

SEE 本身就是"激活能量落在噪声子空间的比例"，可以直接当作噪声水平 σ 的估计：w = SEE/(1−SEE)，与第 5.2 节的零噪声外推同一形式。和 Idea 2 的 λ（音色码空间的估计）比较，看哪个更能预测翻译质量的下降。

### 6.6 合并后的组合表（第一波 K1–K9）

| 编号 | 组合（已测版本 + 新机制） | 作用位置 | 目标 | 新意 | 执行者 |
|---|---|---|---|---|---|
| K1 | Idea 1 Z + Idea 2 λ + Idea 5 背景锚 + 负条件引导 | Talker 输出 | 音质、ASR-BLEU | 引导 = 零噪声外推，权重闭式；负条件取本流背景 | Talker 组 |
| K2 | Idea 2 λ + 背景对比解码 | Thinker 输出 | 英→中 BLEU | 权重由 λ 闭式给出，干净窗口自动关闭 | Thinker 组 |
| K3 | Idea 5 背景锚 + 话轮间隙在线校正（一阶） | Thinker 输入 | BLEU | 间隙作免标签自适应信号 | Thinker 组 |
| K4 | 语速钳制（可配任何音色方法） | Talker 隐状态 | LAAL 不升 | 语速对音色不变，以同传延迟为目标 | Talker 组 |
| K5 | 可靠度信号（干净/带噪分歧） | Thinker | E/F 的前提 | 蒸馏"何时写出"的决策 | Thinker 组 |
| **K6** | **原版 SEEN（离线子空间）** | Thinker 输入 | 英→中 BLEU | 无（**已发表基线**） | Thinker 组 |
| **K7** | **Idea 5（逆 Wishart 收缩）+ Idea 2 思路 + SEEN → 在线软 SEEN** | Thinker 编码器 | 英→中 BLEU | 在线、贝叶斯收缩、Wiener 软投影、跨说话人共享 | Thinker 组 |
| **K7v** | **同上，用在 BiCodec 说话人编码器 + Idea 1 C** | 音色编码器 | 音质 | 同一背景锚同时服务翻译与合成 | Talker 组 |
| **K8** | **Idea 2 λ + 观测叠加（HDemucs 已有）** | Thinker 输入 | 英→中 BLEU，不被去噪伪影伤害 | 混合比例由逆方差加权闭式给出 | Thinker 组 |
| **K9** | **SEE 分数 vs Idea 2 λ 作为噪声水平 σ** | 可靠度估计 | 为 K2/K5/F 选 σ 来源 | 把 SEE 用作零噪声外推的 σ | Thinker 组 |

> **v1.3 变更（2026-11-19）：** 用户不采用"老师教学生"的蒸馏思路。候选 E 取消，K5 取消；第 5.2 节 (5) 里的"引导蒸馏"同样不再考虑。F 的可靠度信号改由 K9 提供。
