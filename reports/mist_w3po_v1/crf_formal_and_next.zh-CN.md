# 当前主系统中的 CRF：严格形式，以及能同时提高 ASR-BLEU、降低 LAAL 的 CRF 类方案（2026-11-10）

本文的方法依据只取公开文献，不引用审稿中的论文：
- 半马尔可夫 CRF：Sarawagi & Cohen 2004；
- 退火 EM：Ueda & Nakano 1998；
- 后验正则化：Ganchev et al. 2010；
- 其余见 §5。

数字全部来自 dev。

---

## 1. 主系统 A′ 用了 CRF 吗？

用了，在 Thinker 的训练里。A′ 的 Thinker 是 **SerialCRF v2′**（`crf/serial_v2p_train.py`，训练目标 `serial_v2p`）。
- 它在训练时对"每个块在哪一次读写出、同一次读内先写谁"这两个隐变量，用 CRF 后验精确求边际。
- 推理时没有 CRF 解码：模型照常逐 token 生成，CRF 只改变训练信号。

v2′ 对 C0-fp 的效果：
- 文本 +0.29（P=0.012），ov +0.79（P=0.015）；
- 语音因此提高（同一 talker 下，与 C0-fp 持平或更好）；
- 代价是文本提交平均晚约 0.2 s。原因是它的目标里没有延迟项（§1.3）。

### 1.1 严格模型（v2′）

**符号**
- 窗口 $x$：音频和 diarizer 后验。读 $r=1..R$，时刻为 $t_r$；$t_R$ 是窗口的最终读。
- 目标（参考译文）切成条目（块）集合 $I$。条目 $i$ 有：说话人 $k_i$、标签读 $d_i$（源端听完的那次读）、token 序列 $y_i$。

**隐变量 $z=(A_{1:R},\,o_{1:R})$**
- $A_r\subseteq I$：到第 $r$ 次读为止已写出的条目集合。约束：
  - 每个说话人内按序写出，即 $A_r$ 对每个说话人是前缀闭集；
  - 最多推迟一次读：$\{i:d_i\le r-1\}\subseteq A_r\subseteq\{i:d_i\le r\}$；
  - 最终读的条目不可推迟：$A_R=I$。
- $o_r\in\{\text{canon},\text{newfirst}\}$：读内顺序。只有当"上一读推迟下来的条目"和"其他说话人的新条目"都非空时，才是真选择。

**势函数**（神经网络给出，就是 LLM 自己的对数概率）

$$\psi_{r,o}(A'\!\to\!A)=\log p_\theta\big(W_r^{o}(A',A)\oplus e_r\ \big|\ x_{\le t_r},\,c,\,\kappa(A')\big)$$

- $W_r^{o}$：把 $A\setminus A'$ 按顺序 $o$ 拼成的 token 串；
- $e_r$：`<wait>`；最终读时为 `<eos>`；
- $\kappa(A')$：把已写集合按规范顺序串行化的历史。它只依赖集合，这正是按读分解能精确成立的原因。

**CRF 后验**（在给定 $x$ 和参考 $S$ 的条件下，对 $z$ 全局归一化的线性链 CRF，状态 = 已写集合）

$$q_\theta(z\mid x,S)=\frac{\exp\sum_{r=1}^{R}\psi_{r,o_r}(A_{r-1}\!\to\!A_r)}{Z(x,S)}$$

$$\alpha_r(A)=\log\!\!\sum_{A'\to A\ \text{可行}}\exp\!\Big(\alpha_{r-1}(A')+\log\sum_{o}e^{\psi_{r,o}(A'\to A)}\Big),\qquad \log Z=\alpha_R(I)$$

- 前向-后向给出每条边的边际 $q(r,A'\!\to\!A,o)$，精确计算，不做采样。
- 生成模型 $p_\theta$ 本身是局部归一化的 LLM；"CRF"指的是隐结构的后验。这和 HMT、CTC 的隐对齐是同一类做法。

**训练**（退火 EM，$\tau=0.5$ 预登记）

$$q_\tau(z)\propto\exp\Big(\tau\sum_r\psi_{r,o_r}\Big),\qquad
\mathcal L(\theta)=-\frac1D\sum_{r}\sum_{A'\to A,\,o}q_\tau(r,A'\!\to\!A,o)\Big[\sum_t h_t\log p_\theta(W_t\mid\cdot)+\log p_\theta(e_r\mid\cdot)\Big]$$

- $q_\tau$ 不回传梯度（M 步）；$h_t$ 是 v8 的分层 token 权重；$D$ 与 C0 用同一种归一化。
- 在标签路径上（$q=1$），损失和梯度与 C0 逐位相同（已测试）。

### 1.2 它解决了什么

- 多说话人同读时，"先写对方、漏掉插话"的串行化失败：ov +0.79。
- 不需要人为规定读内顺序和推迟。

### 1.3 它没有解决什么（也就是剩下的问题）

| 没覆盖的部分 | 现状（A′） |
|---|---|
| 目标里没有任何延迟项 | 文本提交晚约 0.2 s；模型滞后 0.80 s |
| 窗口什么时候关，是固定规则（gap 0.6） | 检测器等待 0.87 s；缩短 gap 会切断话语（gap 0.5 时 ov −0.21） |
| 写出之后的语音时间线 | 最后一块的说话时长 1.74 s，排队 0.50 s；单通道 EndOffset 6.92 s |
| 漏译 | 文本 >1 s 的漏译单元，补回上界 +3.9（`text_to_speech_gap_plan` §6） |

两条教训决定了新方案必须怎么设计：
- **S0**：训练时把截止整体提前，ov −4.62，重复率 6.4% → 9.5%。
- **规则式静音提交**：文本 −1.55。

强行让模型"更早写"会伤质量。所以提前只能发生在模型的似然显示"此刻已经能译完整"的地方，这正是隐变量 CRF 能提供的信号。

---

## 2. 文献：CRF / 隐结构同传做到了哪里

| 工作 | 隐结构 | 延迟定义 | 多说话人 | 输出 |
|---|---|---|---|---|
| HMT（Zhang & Feng, ICLR 2023） | 每个目标词的开始时刻（HMM），对时刻求边际 | 文本位置 | 否 | 文本 |
| Seg2Seg（Zhang & Feng, NeurIPS 2023） | 源 / 目标的隐分段（聚合与发出时刻），期望训练 | 源 / 目标位置 | 否 | 文本 |
| EMMA / Seamless（2023） | 单调注意力的期望对齐 | 期望延迟正则 | 否 | 语音 |
| Hibiki-Zero（Labiausse et al., arXiv 2602.11072） | 无隐结构；GRPO，过程奖励 | 强化学习奖励 | 否 | 语音 |
| DIAL-S2ST（arXiv 2610.32214） | 无；带前瞻 teacher 的自蒸馏 | 源端前瞻 | 否 | 语音 |
| DiariST（2023）、JSTAR（2024） | 序列化输出训练（SOT）；无隐变量边际 | 文本 | 是 | 文本 |
| **本项目 v2′** | 写出读 × 读内顺序，集合状态 CRF | 无 | **是** | 文本 + 语音 |

**空白**：
- 没有工作把"窗口怎么切"或"输出语音的播放时间线"作为隐结构求边际；
- 多说话人加语音输出的同传，还没有隐结构训练的先例（据本轮检索）。

---

## 3. 方案（按"能同时提高 ASR-BLEU、降低 LAAL"的把握排序）

### 方案一：窗口切分隐变量 CRF（Latent-Window CRF，LW-CRF）——推荐先做

**要解决的问题**
- 检测器等待 0.87 s，来自固定 gap 规则。
- 缩短 gap 会切断话语，伤文本；规则式静音提交更伤文本（−1.55）。
- 把"什么时候可以关窗口、给出最终译文"变成模型学到的决策，目标是在不切断话语的前提下尽早收尾。

**为什么能两个指标一起改善**（dev 证据）
- C2 已经显示：窗口更早关闭时，延迟下降（−0.29 s），官方语音分上升（+0.51 到 +0.80）。
- 唯一的代价来自切断话语：切断率 2.5% → 3.6% 时，ov −0.21。
- LW-CRF 只在模型的似然认为"此处已完整"的停顿处收尾，所以可以把收尾等待压到约 0.3 s，同时不切断话语。

**严格模型**（会议级半马尔可夫 CRF）
- 会议 $m$ 的候选边界 $\mathcal B$：所有持续 ≥ $h$ 帧的停顿起点（diarizer 规则，$h=4$，即 0.32 s），再加会议首尾。
- 切分 $z=(0=\beta_0<\beta_1<\dots<\beta_M=T)$，$\beta_j\in\mathcal B$；窗口 $w_j=[\beta_{j-1},\beta_j)$。
- **可行性**：边界不得落在任何参考话语内部，所以每个窗口的参考目标 $S_w$ 都是完整话语，按时间归属。
- **窗口势函数**：v2′ 格在这个窗口上的对数边际似然，减去延迟代价：

$$\phi(w)=\log Z_{\text{v2'}}(x_w,S_w;\,\mathrm{ctx}(w))\;-\;\lambda\,c(w),\qquad c(w)=\sum_{u\in w}\big(\beta_j+h\cdot80\,\mathrm{ms}+1.04\,\mathrm{s}-b_u\big)^{+}\cdot\mathbb 1[u\ \text{的末块在最终读}]$$

- **后验与前向**（长度上限为 $L$ 个候选边界）：

$$q(z)\propto\exp\sum_j\phi(w_j),\qquad \alpha(\beta)=\log\!\!\sum_{\beta'<\beta,\ \beta'\in\mathcal B}\!\!\exp\big(\alpha(\beta')+\phi([\beta',\beta))\big)$$

- **学习**：
  1. 收尾决策：在每个候选停顿 $p$ 上（线上在 $p+h\cdot80\,\text{ms}+1.04\,\text{s}$ 时刻可判定），目标为后验边界概率 $\pi_p=P_q(p\in z)$，用软交叉熵训练 Thinker 的"收尾 / 继续"决策：

     $$\mathcal L_{\rm close}=-\sum_p\big[\pi_p\log p_\theta(\text{close}\mid\cdot)+(1-\pi_p)\log p_\theta(\text{cont}\mid\cdot)\big]$$

     说话人在句中停顿时 $\pi_p=0$（不可行），所以模型会学到"句中停顿不收尾"，规则式静音提交的失败正是在这里。
  2. 窗口内的翻译：用 $q$ 加权的 v2′ 损失，与 v2′ 相同。
  3. $\lambda$ 用对偶二分确定，使 $\mathbb E_q[c]$ 等于目标值（预登记，不扫参）。
- **推理**：每个 ≥ 0.32 s 的停顿处，Thinker 做一次"收尾?"判断；判为收尾就执行最终读并关窗口，否则继续。这取代固定 gap 规则和规则式静音提交。

**新颖性**
- Seg2Seg 的隐分段是单流的源 / 目标 token 聚合。
- LW-CRF 的隐变量是多说话人会议上的**窗口切分**：候选边界来自 diarizer 停顿，势函数是窗口级的 v2′ 边际似然（嵌套的两层精确动态规划），收尾的语义是"最终读 + 全窗口后验"。
- 本轮检索没有找到同类工作。

**预检（GPU 约 1.5 小时，单一设置）**
- **oracle 窗口上界**：用参考话语边界作为切分（只在 ≥ 0.32 s 的话语间停顿处收尾，从不切断话语），收尾时刻 = 停顿 + 0.32 s + 1.04 s，其余与 A′ 完全相同，跑一次解码加 talker。
- **GO 条件**（初版）：文本对 A′ 过门槛，且语音 EndOffset 降 ≥ 0.25 s。
- **GO 条件（2026-11-10 20:40 UTC 修订，在 GPU 预检任何结果出来之前提交）**：用户决定以**语音 LAAL** 为主延迟指标。GO 条件改为：文本对 A′ 过门槛，且每条流的语音 LAAL 降 ≥ 0.25 s；EndOffset 照报，不作判定。
- 不通过说明"不切断话语、尽早收尾"本身没有收益，方案一停止。

### 方案二：播放感知的调度 CRF（Playback-Aware Scheduling CRF，PASP）

**要解决的问题**
- 写出之后的语音时间线：最后一块的说话时长 1.74 s，排队 0.50 s。
- 单通道（一个译员声音）下，跨说话人排队让 EndOffset 达到 6.92 s。

**严格模型**（输出侧，给定已写出的文本）
- 每条输出通道有空闲时刻 $F$，按 Δ = 80 ms 离散化，作为增广状态。
- 待说的块集合为 $Q$。每个决策点选择：
  - 顺序 $\pi$；
  - 每块的语速档 $v_k$（talker 已有，见 `si_infer --speed-shift`）；
  - 是否把同一说话人的碎片与下一块合并 $g_k$。
- **(max, +) 递推**：

$$s_k=\max(\text{avail}_k,\,F),\qquad F\leftarrow s_k+\hat\delta(k,v_k,g_k),\qquad e_k=F$$

- **得分**：

$$S(\pi,v,g)=\sum_k\Big[\underbrace{\varphi_{\rm fid}(k,v_k,g_k)}_{\text{该块被读对的对数概率}}-\lambda\,\underbrace{\ell(e_k)}_{\text{LAAL / EndOffset 代价}}\Big]$$

  - $\varphi_{\rm fid}$ 是对数线性模型，特征为长度、语速、是否碎片、是否重叠，用 talker 日志（独立的 CTC 正确率）按 CRF 最大似然拟合；
  - $\hat\delta$ 是时长模型。
- **推理**：在（待说子集, $F$）上做精确动态规划，取 Viterbi；新块到达时重新规划（滚动时域）。
- **两个指标**：
  - 质量：$\varphi_{\rm fid}$ 不让难读的块说太快，并合并碎片 → ASR-BLEU 不降或上升；
  - 延迟：排队和时长由动态规划显式优化 → LAAL 下降。单通道下收益最大。

**新颖性**：多说话人单通道同传的输出调度，以前没有被研究过。不过它是推理期的结构化调度，不训练 LLM，单独成文偏工程。更适合作为方案一的配套，并引入"单通道"这一新的评测设定。

**预检（不用 GPU）**：
- 用 A′ 现有的文本和 talker 时长，离线重放上述动态规划。
- **GO 条件**（预登记）：单通道 EndOffset 降 ≥ 1.0 s，或每条流降 ≥ 0.3 s，同时被模型判为"读错风险高"的块比例不上升。
- **结果（20:30 UTC，最短块优先重放，A′ 的输出）**：
  - 单通道 EndOffset 6.93 → 6.20 s（−0.73），按预登记的 EndOffset 条件**未通过**；
  - 单通道语音 LAAL 7.07 → 5.84 s（−1.23）；
  - 每条流不变（重放只影响单通道）。
  - 用户随后（20:40 UTC）把主指标定为 LAAL。按 LAAL 计这一项降幅很大，但这个判断是**看到结果之后**做的，属于事后，所以在报告里单独标明。

### 方案三：CRF 动态 oracle 补漏译（复活 §14，改覆盖判定）

**要解决的问题**：漏译。文本 >1 s 的漏译单元补回上界 +3.9，是目前最大的文本质量空间，会直接带动 ASR-BLEU。

**模型**：§14 的残差格精确后验作为 on-policy oracle（DAgger 式）。数学部分已经通过预检 O3（误差 10⁻¹⁴ 量级）。

**卡在哪里**：覆盖判定（"模型是否已经写了某个条目"）的人工一致率只有 0.865，低于 0.95 的门限。

**改法（需要重新预登记）**：用 Thinker 自己的强制解码似然判断覆盖，即"该条目的参考译文在已写文本条件下是否几乎不增加信息"，取代词面规则；或改用跨语言句向量对齐。

**两个指标**：漏译补回 → 文本和 ASR-BLEU 上升；补写发生在下一读，而不是永远不写，对延迟中性。

---

## 4. 统一的论文叙事

**多说话人语音同传中，三个时间尺度上的隐结构精确边际。**

| 尺度 | 隐结构 | 方案 | 状态 |
|---|---|---|---|
| 会议级 | 窗口切分（何时收尾） | LW-CRF | 待预检 |
| 读级 | 写出时机 × 说话人串行化 | v2′ | 已完成：ov +0.79 |
| 输出级 | 播放顺序 / 语速 / 合并 | PASP | 待预检 |

- 三者都用精确动态规划求后验，都用后验加权的交叉熵训练（不需要强化学习），可以组合。
- 现有的结果 v2′ 是第一个实例。

---

## 5. 不建议做的

| 方案 | 原因 |
|---|---|
| HMT 式延迟正则（§10.3 一′） | 与 HMT 区别小；S0 显示整体提前会伤 ov |
| 半马尔可夫 CRF 做帧级端点检测 | 与正在审的投稿建模过于接近；而且是老方法。LW-CRF 的隐变量在窗口层，势函数是 LLM 的窗口边际似然，不是帧级分类 |
| 推理期 `<wait>` 偏置、继续缩短 gap、继续加快语速 | 只是调节旋钮，或收益已递减 |

---

## 6. 建议顺序

1. LW-CRF 的 oracle 窗口预检（GPU 约 1.5 小时）+ PASP 的离线重放预检（不用 GPU），并行做。
2. 通过的方案写一页设计、预登记，经用户批准后实现。LW-CRF 约 4–6 天，包括窗口势函数的批量前向、会议级动态规划、收尾决策训练、一次 8 卡训练。
3. 方案三在新的覆盖判定下重做 O1；通过才跑 O2。

## 参考文献（公开）

- Sarawagi & Cohen. Semi-Markov Conditional Random Fields for Information Extraction. NeurIPS 2004.
- Zhang & Feng. Hidden Markov Transformer for Simultaneous Machine Translation. ICLR 2023.
- Zhang & Feng. Unified Segment-to-Segment Framework for Simultaneous Sequence Generation. NeurIPS 2023. https://arxiv.org/abs/2310.17940
- Labiausse et al. Simultaneous Speech-to-Speech Translation Without Aligned Data (Hibiki-Zero). arXiv 2602.11072. https://arxiv.org/abs/2602.11072
- Choi, Ahn, Chung. DIAL-S2ST. arXiv 2610.32214. https://arxiv.org/abs/2610.32214
- Yang et al. DiariST: Streaming Speech Translation with Speaker Diarization. arXiv 2309.08007. https://arxiv.org/abs/2309.08007
- JSTAR: Transcribing and Translating, Fast and Slow. arXiv 2412.15415. https://arxiv.org/abs/2412.15415
- Ueda & Nakano. Deterministic Annealing EM Algorithm. Neural Networks 1998.
- Ganchev et al. Posterior Regularization for Structured Latent Variable Models. JMLR 2010.
- Ross et al. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011.
- Papi et al. Length-Adaptive Average Lagging (LAAL). 2022. https://arxiv.org/abs/2206.05807

## 7. 预检结果（2026-11-10 21:35 UTC；dev；主延迟指标为语音 LAAL）

| 预检 | 质量（对 A′，两种子） | 语音 LAAL（每条流） | 单通道 LAAL | EndOffset | 预登记规则 | 判定 |
|---|---|---|---|---|---|---|
| **LW-CRF oracle 窗口**（参考话语边界，不切断话语，停顿 0.32 s 收尾；4359 窗口） | 文本 28.17 → **28.41（+0.24，P=0.038）**，ov −0.06（不显著），语音 23.26 → **24.12（+0.86，P=0.000）**，三项全过 | 4.15 → 3.98（**−0.17**） | 7.07 → 6.60（−0.47） | 3.91 → 3.68（−0.23） | 文本过门槛且每条流 LAAL 降 ≥ 0.25 s（20:40 UTC 登记） | **NOT GO**（−0.17 < 0.25） |
| LW-CRF 完美收尾上界（CPU，窗口不变） | 不变 | −0.15 | −0.14 | −0.18 | （参考项） | — |
| **PASP 单通道最短块优先重放**（CPU） | 不变 | 不变 | 7.07 → 5.84（**−1.23**） | 单通道 −0.73 | 单通道 EndOffset 降 ≥ 1.0（事先登记） | **未过**（按 LAAL 是事后判断） |
| PASP 时长可预测性 | — | — | — | — | R² ≥ 0.8 | **通过**（0.87） |
| **CRF 动态 oracle，覆盖 v3** | — | — | — | — | 人工一致率 ≥ 0.95 | **STOP**（≤ 0.949，`crf_dynamic_oracle_precheck.zh-CN.md` §8） |

- **LW-CRF 的 oracle 质量上升，延迟只降一点。**
  - oracle 窗口比 gap 0.6 更细（4359 对 3618），而且从不切断话语。文本显著上升（+0.24），语音上升（+0.86）。
  - 但语音里有多少是"短流"的口径效应，需要看段级分。缺口分解已在 CPU 上运行，输出前缀 `speech_gap_v1/v2p_orcl032_s*`。
  - 每条流的 LAAL 只降 0.17 s，没有达到登记的 0.25 s。学出来的收尾策略不会超过 oracle，所以按规则，LW-CRF 不作为降延迟方法继续。
- **PASP**：只在单通道下有效，而且用的是最简单的规则。是否把"单通道"定为论文主设定、以 PASP 为核心，需要用户决定。
