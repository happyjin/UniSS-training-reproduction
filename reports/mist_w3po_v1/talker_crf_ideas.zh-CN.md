# 以 talker 为对象的 CRF / 隐结构方案：缩小文本到语音的差距，同时降低语音 LAAL（2026-11-10）

**任务目标**（用户 2026-11-10 21:50 UTC）：
- 提高 talker 的 ASR-BLEU，缩小同一译文的文本分与语音分之间的差距；
- 同时降低语音 LAAL；
- 方法沿用此前查过的 CRF / 隐结构路线，新颖性要达到 ICLR 一类顶会的水平。

**引用与数据范围**：
- 只引用公开文献，不引用审稿中的论文。
- 半马尔可夫 CRF 以 Sarawagi & Cohen 2004 为依据。
- 数字全部来自 dev；测试集没有动过。

---

## 0. 结论

| 排序 | 方案 | 训练对象 | 提高 talker ASR-BLEU | 降低语音 LAAL | 新颖性 | 建议 |
|---|---|---|---|---|---|---|
| **1** | **播放格 EM（Playback-Lattice EM，PL-EM）**：在"候选 × 语速 × 播放队列状态"的格上精确求后验，作为 talker 的训练目标。可以看作结构化的 best-of-N 蒸馏 | talker | 是：把多次采样里读得准的那条学进单次采样 | 是：一块说太长会拖后面所有块，这种耦合代价被精确地分摊到每一块 | **高** | **主线** |
| 2 | 对齐 CRF talker：文本到语音码的单调对齐作为半马尔可夫隐变量，并加时长预算 | talker | 是：针对漏读和重复 | 部分：按预算分配时长 | 中（对齐损失已有工作，新意在流式、多说话人、预算） | 作为 1 的组件或消融 |
| 3 | 跨接口联合格：在 v2′ 的格里加入 talker 侧的"可说性"势函数 | Thinker | 间接：让 Thinker 写出 talker 更容易读准的文本 | 部分 | 高 | 高风险，放在 1 之后 |
| ~~4~~ | ~~可说性切块 CRF~~ | 外挂的接口小模型 | 小（上界约 +0.1） | 是 | 中 | **放弃**（2026-11-10 22:10 UTC，用户指出：不是端到端，降低和系统的融合；还会改变 talker 输入的分布）|

**统一的方法论**（对应用户担心的"偏工程"）：只有一个方法，即"在隐结构格上精确求后验，并用后验加权的交叉熵训练"。
- v2′ 是它在 Thinker 上的第一个实例：隐结构是串行化的读格；
- PL-EM 是它在 talker 上的实例：隐结构是播放格，状态带有输出队列的空闲时刻。
- 论文的主贡献是这一方法在语音输出侧的扩展，以及由此得到的、可以精确分摊的非可加延迟代价，而不是几个技巧的拼装。

---

## 1. 问题：数据告诉我们 talker 侧还差什么（A′，dev）

| 项 | 数值 | 来源 |
|---|---|---|
| 文本 / 语音（官方口径） | 28.17 / 23.27，差距 4.91 | `gap_decomp_v2p_dev.zh-CN.md` |
| 差距的构成 | 大小写和标点 2.23，打分粒度 1.32，**真实读错 1.35** | 同上 |
| 同一归一化、按段打分时的真实差距 | **1.48**；talker 词错率 5.1%（替换 2.5%，漏读 2.2%） | 同上 |
| 只把漏读或替换换回来的上界（repo 口径） | 漏读 +0.43，替换 +0.45 | 同上 |
| best-of-k | k=2 有效；k=4 不再提升（+0.03），而 talker 计算量与 k 成正比 | `c2_and_a_points_dev.zh-CN.md` |
| 语音 LAAL / 文本 LAAL | 4.15 / 3.47：**有 0.68 s 发生在 talker 侧** | `speech_latency_lit.py` |
| 语音 EndOffset 的分项 | 最后一块的时长 1.74 s，排队 0.50 s，检测器 0.87 s，模型滞后 0.80 s | 同上 |
| 单通道语音 LAAL | 7.07 s；只按"最短块优先"重排就降到 5.84 s | `crf_formal_and_next.zh-CN.md` §7 |
| 规则式调语速 | 语速 +2：LAAL −0.13，+3：再 −0.01，收益递减 | `c2_and_a_points_dev.zh-CN.md` |

**对目标的含义**：
1. 能靠 talker 补回的质量上限约 1.5 BLEU（归一化、按段）。现在只能靠"多采样再挑"拿到其中一部分，而且计算量翻倍。
2. talker 侧的延迟（0.68 s 的 LAAL 差，单通道下更多）来自**每块说多长、排在谁后面**。块与块之间通过播放队列耦合，一块的代价取决于前面所有块，**不能逐块单独优化**。
3. 所以 talker 要学的是：在给定的队列状态下，用多快的语速、说成什么样，既读得准又不拖累后面。这正是一个**带耦合代价的序列结构化决策**，适合用格加精确后验来训练。

---

## 2. 方案一（主线）：播放格 EM（PL-EM）

### 2.1 动机

- **现有的 talker 优化都是逐条或逐块独立打分。**
  - 偏好或强化学习（Align2Speak、F5R-TTS、DMOSpeech 2、DiffRO）对每条输出单独给奖励，奖励只有读得准、像不像，**没有延迟**。
  - best-of-N 蒸馏（BOND、ReST-EM）的奖励按样本独立，可以逐项相加。
- **同传 talker 的延迟代价不能逐项相加。** 第 k 块的结束时刻 = max（放出时刻，上一块结束时刻）+ 本块时长。前面一块慢了，后面每一块都晚。
  - 逐块贪心（我们的 b1）会选"这一块读得最准"的候选，却不管它让后面晚了多少。
  - 强化学习方法（Hibiki-Zero 的 GRPO）用采样来估计这种耦合，方差大，信用分配不精确。
- **把队列状态离散化（80 ms 一格）之后，耦合代价可以写进有限状态的格里，用前向-后向精确计算。** 这和 v2′ 把串行化历史写成"已写集合"状态，从而得到精确后验，是同一个思路。

### 2.2 严格模型

**对象**：一条输出通道上的块序列 $n=1..N$。
- 多通道时，一条通道是"窗口 × 说话人"流；单通道时，是一个会议片段上所有说话人的块。
- 块 $n$ 的文本是 $x_n$，可以放出的时刻是 $a_n$（在线协议下的提交时刻）。

**候选**：对每块，用当前 talker 在 $V$ 个语速档 $v\in\mathcal V$ 下各采 $K$ 个样本，得到 $c\in\mathcal C_n$，$|\mathcal C_n|=VK$。每个候选带有：
- 语音码 $y_{n,c}$；
- 时长 $\delta_{n,c}$；
- 读对分 $f_{n,c}$：独立 CTC ASR（wav2vec2，不是评测用的 Whisper）对 $x_n$ 的对数似然，按词数归一化；
- 说话人相似度 $s_{n,c}$：WavLM-SV；
- talker 自身的对数概率 $\log p_\theta(y_{n,c}\mid\cdot)$。

**状态**：通道空闲时刻 $F$ 相对 $a_n$ 的积压 $b_n=\max(0,F-a_n)$，按 Δ = 80 ms 离散，上界为 $B$。

**转移**（(max, +) 递推，在离散状态上是确定性的）：

$$s_{n}=a_n+b_n,\qquad e_{n,c}=s_n+\delta_{n,c},\qquad b_{n+1}=\Big\lceil\max(0,\,e_{n,c}-a_{n+1})\Big\rceil_\Delta$$

**边势函数**：

$$\psi_n(b_n,c)=\tau\log p_\theta(y_{n,c}\mid\cdot)+\beta f_{n,c}+\gamma s_{n,c}-\lambda\,\ell_n(s_n,\delta_{n,c})$$

- $\ell_n$ 是这一块对语音 LAAL 的贡献：块内各词按时长均分时刻，减去对应参考单元的起点和理想进度，与 `speech_latency_lit.py` 的定义一致。
- 它只依赖 $(b_n,c)$，所以整条路径的 LAAL 可以沿格分解。

**后验（以积压为状态的线性链 CRF）**：

$$q(c_{1:N})=\frac{\exp\sum_n\psi_n(b_n,c_n)}{Z},\qquad \alpha_{n+1}(b')=\log\sum_{b,c:\,b\to b'}\exp\big(\alpha_n(b)+\psi_n(b,c)\big)$$

- 前向-后向给出每块每个候选的边际 $q(n,c)$，以及每块的语速边际 $q(n,v)$，计算是精确的。
- 复杂度为 $O(N\cdot B/\Delta\cdot VK)$。$B$ = 8 s 时只有 100 个状态，可以在 CPU 上直接算。

**学习**（与 v2′ 相同的 EM 配方）：
1. **E 步**：在训练集流上采候选，按上式精确求 $q$；$q$ 不回传梯度。
2. **M 步**：用后验加权的交叉熵训练 talker：
   $$\mathcal L(\theta)=-\sum_n\sum_c q(n,c)\,\log p_\theta(y_{n,c}\mid x_n,\text{voice},\text{src},v_{n,c})$$
   **语速决策并入 talker**（2026-11-10 修订，不另加外挂头）：输入里加一个积压条件 token（当前积压 $b_n$ 按 0.5 s 分档），talker 先生成语速 token $v$，再生成语音码。每块的语速边际 $q(n,v)$ 直接作为这个语速 token 的软标签：
   $$\mathcal L_v=-\sum_n\sum_v q(n,v)\log p_\theta(v\mid b_n,x_n,\text{voice},\text{src})$$
   这样取代现在规则式的 queue 语速策略，语速和语音在同一个模型里一起学。
3. **λ 用对偶二分**：使 $\mathbb E_q[\text{LAAL}]$ 等于预登记的预算（例如 A′ 的语音 LAAL − 0.3 s），这是后验正则化（Ganchev et al. 2010），**不调权重**。β、γ 预登记；τ = 0.5（退火 EM，与 v2′ 相同）。
4. **推理**：talker 只采一次（计算量是 b1 的一半），语速 token 由 talker 自己生成，不需要外部规则或外挂模块。

### 2.3 新颖性（相对已检索文献）

| 工作 | 做法 | 与 PL-EM 的区别 |
|---|---|---|
| BOND（ICLR 2025）、ReST-EM（2023）、reward-weighted regression | 把 best-of-N 或过滤后的样本蒸馏回模型；奖励按样本独立 | PL-EM 是**结构化的 best-of-N**：候选之间通过队列耦合，奖励（延迟）不可逐项相加，用增广状态的精确动态规划求后验 |
| Align2Speak（arXiv 2509.21718）、F5R-TTS、DiffRO（arXiv 2507.05911）、"No Verifiable Reward for Prosody"（arXiv 2509.18531） | TTS 的强化学习或偏好优化，奖励是 CER / WER 加说话人相似度 | 都没有延迟，也不是流式；PL-EM 的延迟项在播放队列上，并同时监控说话人相似度和 UTMOS，防止韵律塌缩 |
| DMOSpeech 2 | 用 GRPO 优化每条话语的时长策略，奖励为 WER 加说话人相似度 | 离线 TTS，没有队列和流式；PL-EM 的语速由后验边际精确监督 |
| Hibiki-Zero（arXiv 2602.11072） | 端到端同传，用 GRPO 加过程奖励降延迟 | 单说话人；用采样估计；PL-EM 是精确后验，并以播放队列为状态，含多说话人单通道 |
| DIAL-S2ST（arXiv 2610.32214） | 用带前瞻的 teacher 自蒸馏控制延迟 | 不建模输出时长和积压 |
| 可微动态规划（Mensch & Blondel, ICML 2018） | 把动态规划中的 max 平滑成可微层 | PL-EM 把 (max, +) 积压**离散成状态**，用和-积前向-后向做精确推断，不需要平滑近似 |
| v2′（本项目） | Thinker 侧的读格 EM | PL-EM 是同一方法在输出语音时间线上的扩展 |

**可主张的贡献**：
1. 指出同传语音输出的延迟在块之间不可逐项相加，并量化它（talker 侧占 0.68 s，单通道更大）。
2. 结构化的 best-of-N 蒸馏：以积压为状态的精确格后验，加后验正则化的延迟预算。
3. 一个方法同时作用于 Thinker（v2′）和 talker（PL-EM）。
4. 多说话人单通道的设定。

本轮检索没有找到"带耦合（非可加）代价的结构化 best-of-N 蒸馏"，也没有找到"以播放队列为状态训练同传 talker"的工作。正式写作前还要系统核查。

### 2.4 为什么有把握见效

- **质量**：在 c1 之上，k=2 的逐块选择已经在 dev 上实测 +0.62（C0-fp）和 +0.70（v2′），两者都显著。PL-EM 蒸馏的是比逐块 k=2 更好的结构化选择，所以单次采样至少应接近 k=2 的效果。
- **延迟**：仅靠重排就让单通道 LAAL −1.23。PL-EM 还能选语速和候选长度，并把选择学进 talker 和语速头。
- **训练框架现成**：
  - talker 训练代码 `si_model.py` / `pretrain_si.py`，T1c 就是用它训出来的；
  - CTC 打分服务；
  - 多候选、多语速采样，`si_infer --select-k`、`--speed-shift`。

### 2.5 预检（先证伪，单一设置，在任何结果之前登记）

| 项 | 做法 | GO 条件 |
|---|---|---|
| P1 格选择上界（GPU 约 3–4 小时，4 卡） | 在 dev 的一个子集（5 个会议）上，每块采 K=4 个样本 × 2 个语速档（A′ 的 +2，以及 +4）。在格上求最优路径（Viterbi，λ 按上式定），与 A′（逐块 b1、k=2）比较 | 每条流的语音 LAAL 降 ≥ 0.25 s，且语音 ASR-BLEU ≥ A′ − 0.2；**或者**在 LAAL 相等时，归一化、按段的语音 ASR-BLEU ≥ +0.4。按会议配对 bootstrap |
| P2 精确性（CPU） | 格的前向-后向结果与暴力枚举一致（容差 1e-9） | 通过 |
| P3 可蒸馏性（CPU） | 后验边际的熵，以及 talker 自身概率与后验的 KL：后验不能塌缩到 talker 本来就选的样本上 | 均匀样本上 KL ≥ 0.1 |

### 2.6 成本与风险

- **成本**：
  - 预检：代码约 0.5 天，GPU 约 4 小时；
  - 正式训练：代码约 2 天；训练集子集采样约 4–6 小时，训练约 4–6 小时，评测约 1.5 小时（每项都是 4 卡）。
- **风险**：
  - 只用读对奖励会使韵律塌缩（arXiv 2509.18531）：加说话人相似度项，并报 UTMOS；
  - 候选多样性不足：先用 P3 检查；
  - 时长与积压的离散误差：可以缩小 Δ。


### 2.7 训练哪些参数：全参数还是 LoRA（2026-11-10 修订）

**历史做法**：
- **talker 一直是全参数训练**。
  - T1c（`si_t1c_20261021T181440Z`，第 1500 步，`full_iter0001500`）：从 R2（Plan 1 之后的 UniSS）出发，全参数继续训练。Megatron 入口 `pretrain_si.py`，模型 `si_model.py` 里是 `talker.requires_grad_(True)`。训练时用两个冻结副本做 KL 约束：
    - 离线 teacher（读完整个源和译文，权重 w_future）；
    - 原始 UniSS anchor（权重 w_anchor）。
  - 单说话人同传的阶段 2（`reports/uniss_omni_s2st_v1/stage2/`）：Thinker 用 LoRA，talker 全量 SFT。当时因为共用一个学习率而回退，修正结论是**分 param group**：LoRA 1e-4，talker 3e-5。
- **Thinker 一直是 LoRA**：v8、C0、v2′ 都是注意力层 rank 32、alpha 64，另加 `sd_proj` 和新增的嵌入、输出行。

**选择：talker 全参数，并对冻结的 T1c 加 KL 锚定。** 理由：
1. 与 T1c 的训练方式和代码完全一致：`si_model.py` 已有 teacher / anchor KL，Megatron 入口现成。LoRA 要为 talker 新写注入和导出代码，没有先例。
2. talker 只有约 1.9B 参数（Qwen2 结构，28 层，hidden 1536），4 张 H200 全参数训练没有显存压力。
3. PL-EM 要改变的是时长、语速和读准这类全局行为：
   - 新增的积压条件 token 和语速 token 的嵌入行必须训练；
   - 时长和韵律分布在所有层里。
   全参数容量更合适；用 KL 锚定控制漂移，比靠 LoRA 的低秩限制更直接、可控。
4. 吸取阶段 2 的教训，学习率单独设为 3e-5，并退火；步数和 T1c 同量级（≤ 1500）。

**训练的参数**：talker 全部权重，包括新增的积压条件 token、语速 token 的嵌入和输出行。

**冻结的参数**：Thinker；作为 KL 锚的 T1c 副本；CTC 打分器、WavLM、语音解码器。

**备选**：如果预检或早期训练显示全参数漂移过大（UTMOS 或说话人相似度下降），改用 talker LoRA 作消融。需要先写 LoRA 注入代码，单独登记。

---

## 3. 方案二：对齐 CRF talker（作为方案一的组件或消融）

**动机**：真实读错里漏读 2.2%、替换 2.5%（只换回漏读的上界是 +0.43）。当前 talker 是只有解码器的码流语言模型，文本与语音之间没有显式对齐，所以会漏读、重复，偶尔失控。

**模型**：
- 块文本单元 $u_{1:M}$（词）与语音码帧 $1..T$ 之间，有一个单调的半马尔可夫隐对齐 $z$：每个单元占一段连续帧，长度为 $d_m$。
- 势函数为
  $$\phi(z)=\sum_m\big[\textstyle\sum_{t\in\text{seg}_m}e_\theta(t,u_m)+\omega_m(d_m)\big]$$
  其中 $e_\theta$ 来自 talker 隐状态上的一个轻量对齐头，$\omega_m$ 是时长势。
- 用精确的前向-后向求配分函数。
- 训练：辅助的边际似然 $-\log Z_{\text{cover}}$，要求每个单元都被覆盖（不漏读）。时长总和用预算约束做后验正则：$\mathbb E_q[\sum d_m]=$ 目标时长，目标来自 PL-EM 的语速。
- 推理：对齐后验显示所有单元都已覆盖之前，不允许发出结束 token。

**相关工作**：
- Neekhara et al. 2024（arXiv 2406.17957）：T5-TTS 加 CTC 对齐损失和注意力先验，是编码器-解码器架构的交叉注意力；
- Nankaku et al. 2021（arXiv 2108.13985）：HSMM 结构化注意力；
- Mehta et al. 2022：neural HMM TTS；
- Glow-TTS：单调对齐搜索（MAS）。

**新颖性**：中。新意在于只有解码器的流式多说话人 talker，以及时长预算正则。单独成文偏弱，适合作为 PL-EM 的保真组件（消融）。

---

## 4. 方案三：跨接口联合格（高风险、高新颖）

**动机**：差距的一部分可能来自 Thinker 写出 talker 难读的文本（长句、罕见词形、碎片）。如果 Thinker 在写的时候就考虑可说性，可以从源头缩小差距。

**模型**：
- 在 v2′ 的读格边势上加一项 talker 侧势：
  $$\psi^{\rm joint}_r=\psi^{\rm v2'}_r+\eta\,\mathbb E_{y\sim p_{\rm talker}}[f(y;W_r)]-\lambda\,\ell^{\rm play}_r$$
  - 第二项是写出的块被读对的期望，用采样或 PL-EM 的读对分估计；
  - 第三项是它在播放格上的延迟。
- 联合后验仍然精确：talker 侧势只依赖本读写出的块和积压状态，积压作为增广状态进入 v2′ 的集合状态。
- Thinker 用后验加权的交叉熵训练。
- **这是跨离散接口的级联联合训练，用精确的格 EM 代替强化学习。**

**风险**：
- 可说性不等于准确，文本 BLEU 可能下降，由门槛把关；
- 势函数要在每条边上调用 talker，代价高。只在方案一成立之后再考虑。

---

## 5. 方案四：可说性切块 CRF（已放弃）

**放弃原因**（用户 2026-11-10 指出）：它是夹在 Thinker 和 talker 之间的外挂模块，不参与两者的训练，不是端到端，降低和系统的融合；它改变交给 talker 的块边界，使 talker 的输入偏离训练分布；而且在 A′ 上上界只有约 +0.1。以下保留原记录。

- **做法**：在每个说话人的已写文本流上做半马尔可夫切分，决定交给 talker 的块边界。势函数是可读性加播放延迟，在 talker 日志上按 CRF 最大似然拟合。
- **相关工作**：Prosodic Boundary-Aware Streaming（arXiv 2603.06444，每 k 个词插一个边界标记）、SpeakStream（arXiv 2505.19206）、incremental TTS（Ma et al. 2020）。
- **不建议的原因**：在 A′ 上，把碎片全部换回只提高约 +0.1，空间太小。

---

## 6. 指标与门槛（所有方案相同）

- **质量**：
  - 文本 tcpBLEU all / ov 不得下降。talker 方案不改 Thinker，文本按构造不变。
  - 语音 ASR-BLEU 同时报官方口径和归一化、按段口径。**差距 = 文本 − 语音，两者用同一口径**（需要用户确认）。
- **延迟**：语音 LAAL 是主指标（每条流、单通道），EndOffset 和文本 LAAL 作补充；发表前补计算感知延迟。
- **自然度和说话人**：UTMOS、WavLM-SV 相似度不得下降。
- **门槛**：点估计 ≥ 参考 − 0.2，按会议配对的 bootstrap 满足 P(差 ≤ 0) < 0.95，两个 TTS 种子。只用 dev 选点；方案冻结后，经用户批准，在测试集上评一次。

## 7. 建议顺序

1. PL-EM 预检 P1–P3：先在任何结果之前登记，然后 4 卡、约 4 小时。另外 4 卡可以同时为方案二准备对齐头的数据。
2. P1 通过后：写 PL-EM 训练代码，在训练集上采样、训练、评测，4 卡。
3. PL-EM 成立后，再考虑方案二作为组件、方案三作为扩展。

## 参考文献（公开；写作前逐条核对原文）

- Sarawagi & Cohen. Semi-Markov Conditional Random Fields for Information Extraction. NeurIPS 2004.
- Ganchev et al. Posterior Regularization for Structured Latent Variable Models. JMLR 2010.
- Ueda & Nakano. Deterministic Annealing EM Algorithm. Neural Networks 1998.
- Mensch & Blondel. Differentiable Dynamic Programming for Structured Prediction and Attention. ICML 2018. https://arxiv.org/abs/1802.03676
- Sessa et al. BOND: Aligning LLMs with Best-of-N Distillation. ICLR 2025. https://arxiv.org/abs/2407.14622
- Singh et al. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (ReST-EM). TMLR 2024. https://arxiv.org/abs/2312.06585
- Peters & Schaal. Reinforcement Learning by Reward-Weighted Regression for Operational Space Control. ICML 2007.
- DMOSpeech 2（用 GRPO 优化时长策略，奖励为 WER 和说话人相似度；写作前核对原文）。
- F5R-TTS（GRPO，奖励为 WER 和说话人相似度；写作前核对原文）。
- Align2Speak: Improving TTS for Low Resource Languages via ASR-Guided Online Preference Optimization. arXiv 2509.21718. https://arxiv.org/abs/2509.21718
- No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS. arXiv 2509.18531. https://arxiv.org/abs/2509.18531
- Differentiable Reward Optimization for LLM based TTS system (DiffRO). arXiv 2507.05911. https://arxiv.org/abs/2507.05911
- Neekhara et al. Improving Robustness of LLM-based Speech Synthesis by Learning Monotonic Alignment. Interspeech 2024. https://arxiv.org/abs/2406.17957
- Nankaku et al. Neural Sequence-to-Sequence Speech Synthesis Using a Hidden Semi-Markov Model Based Structured Attention Mechanism. arXiv 2108.13985. https://arxiv.org/abs/2108.13985
- Mehta et al. Neural HMMs are all you need (for high-quality attention-free TTS). ICASSP 2022. https://arxiv.org/abs/2108.13320
- Labiausse et al. Simultaneous Speech-to-Speech Translation Without Aligned Data (Hibiki-Zero). arXiv 2602.11072. https://arxiv.org/abs/2602.11072
- Choi, Ahn, Chung. DIAL-S2ST: Learning Latency Control for Simultaneous Speech-to-Speech Translation. arXiv 2610.32214. https://arxiv.org/abs/2610.32214
- Prosodic Boundary-Aware Streaming Generation for LLM-Based TTS with Streaming Text Input. arXiv 2603.06444. https://arxiv.org/abs/2603.06444
- SpeakStream: Streaming Text-to-Speech with Interleaved Data. arXiv 2505.19206. https://arxiv.org/abs/2505.19206
- Zhang & Feng. Hidden Markov Transformer for Simultaneous Machine Translation. ICLR 2023.
- Zhang & Feng. Unified Segment-to-Segment Framework for Simultaneous Sequence Generation. NeurIPS 2023. https://arxiv.org/abs/2310.17940
- Papi et al. Length-Adaptive Average Lagging (LAAL). 2022. https://arxiv.org/abs/2206.05807

---

## 8. 三个保留方案：严格形式、是否 CRF、是否端到端、新颖性（2026-11-10 22:20 UTC）

"端到端"分两层含义：**目标端到端**（训练目标直接是系统最终的指标，即语音 ASR-BLEU 和语音 LAAL）和**梯度端到端**（梯度穿过结构层，直接进入模型参数）。

| 方案 | 严格模型 | 是 CRF 吗 | 目标端到端 | 梯度端到端 | 新颖性 |
|---|---|---|---|---|---|
| **1. PL-EM** | 以积压为状态的线性链 CRF：$q(c_{1:N})\propto\exp\sum_n\psi_n(b_n,c_n)$，(max,+) 转移，配分函数精确；后验正则的延迟预算 λ 用对偶二分确定 | **是**：采样候选格上的全局归一化后验，与 v2′ 是同一类 | 是：读对加语音 LAAL | **talker 内部是**：用后验加权的交叉熵训练 talker（含语速 token）。E 步的后验不回传梯度（EM），与 v2′ 相同。Thinker 冻结 | **高**：结构化的 best-of-N 蒸馏，耦合代价不可逐项相加但能精确算；以播放队列为状态训练同传 talker；多说话人单通道 |
| 2. 对齐 CRF talker | 文本单元与语音码帧的单调半马尔可夫对齐：$p(z)\propto\exp\phi(z)$，覆盖约束加时长预算的后验正则 | **是**：条件半马尔可夫 CRF | 部分：直接针对漏读，时长服从预算 | **是**：辅助损失 $-\log Z_{\rm cover}$ 经可微的前向算法直接传到 talker 的隐状态，是三个里梯度最端到端的 | 中：单调对齐损失、HSMM 注意力已有；新意在只有解码器的流式多说话人 talker 和时长预算。作组件 |
| 3. 跨接口联合格 | v2′ 的集合状态格加积压状态，边势再加 talker 侧的读对期望和播放延迟 | **是**：v2′ CRF 的增广 | 是：Thinker 直接以语音侧的读对和延迟为目标 | 不完全：Thinker 侧用 EM；talker 只作为势函数，梯度不穿过离散文本接口 | 高：用精确格 EM 代替强化学习，跨离散接口联合优化级联系统；风险和代价高 |

**三者合起来的系统级端到端**：交替执行方案 1（固定 Thinker，训练 talker）和方案 3（固定 talker，训练 Thinker），就是对整个级联系统的坐标上升：
- 每一步都是精确的格 EM；
- 目标都是系统最终的语音 ASR-BLEU 和 LAAL；
- 不需要梯度穿过离散接口，也不需要强化学习。

这可以作为论文的完整叙事，前提是方案 1 单独先成立。
