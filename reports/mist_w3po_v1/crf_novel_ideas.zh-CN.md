# 结合 CRF 的高新颖度方案：多说话人同传（MIST）

> 日期：2026-11-09
> 目的：排除工程性的做法，只保留可能撑起顶会主贡献、而且能写成严格数学模型的方案。
> 依据：
> - 本地证据：GRPO 七个变体的结果，`LOG.zh-CN.md`；语音噪声诊断，`diag_speech_drop.json`；ECF 预检；方案文档 `docs/mist_grpo_plan.zh-CN.md`。
> - 新做了 14 次文献检索（2026-11-09，第 8 节列出来源）。检索不是穷尽的，写论文前要系统核查。
> - 上一版方案在 `semicrf_borrowing_plans.zh-CN.md`。
>
> **保密说明。** 启发本文的论文是审稿中的保密稿件，下面不引用、也不使用它尚未发表的内容。所有方法依据都是公开文献：semi-CRF（Sarawagi & Cohen 2004）、全局归一化（Lafferty 2001；Andor 2016；Variani 2022）、结构化注意力（Kim 2017）等。

---

## 0. 结论

| 方案 | 一句话 | 解决的问题 | 严格 CRF | 训练期 / 测试期 | 端到端 | 新颖度 | 建议 |
|---|---|---|---|---|---|---|---|
| **一 ScheduleCRF** | 用精确推断求出多说话人"何时、以多大块提交"的最优调度后验，作为 SFT 和蒸馏的教师 | 启发式截止标签的噪声；GRPO 采样不到就学不到（全组漏掉）；延迟与质量的权衡；碎片 | 是：每个说话人一条半马尔可夫链，可加跨说话人耦合 | 训练期；推理架构不变 | 教师是 CRF，学生是 LLM，用 EM 交替 | **高** | **主推，先做** |
| **二 TurnCRF 层** | 在语音 LLM 内部加一层流式、可处理重叠的神经 semi-CRF 说话人轮次层，用它的边际替代冻结 diarizer 的后验输入，整体端到端训练 | 错标、collapse、diarizer 错误向下传（P10）、短插话检测 | 是：联合标签 semi-CRF，前缀上的截尾时长 | 训练期加测试期 | **是**，通过结构化注意力 | 中到高 | 第二个做，和一合起来讲 |
| 三 全局归一化策略头 | 把 `<wait>`/说话人的控制决策从逐步 softmax 改成全局归一化的线性链 CRF，对抗流式场景的标签偏置 | 标签偏置导致提交时机不稳 | 是 | 训练期加测试期 | 是 | 中到高（同传上没检到） | 有风险：贪心解码下收益有限，需要延迟承诺；放到最后 |
| 四 CRF 引导探索 | 从一的后验里精确采样调度，作为引导 rollout 混进 GRPO 组 | 全组漏掉导致零对比、零梯度 | 用 CRF 采样；本身是策略梯度 | 训练期 | — | 中（离策略引导已有 LUFFY） | 一做成以后再考虑 |

**旧方案审核**（`semicrf_borrowing_plans.zh-CN.md`）：
- A、C、E 属于评测、规则和工程，**排除**。
- B 单独做新颖度低：贝叶斯 HSMM 做说话人分离已有 HDP-HSMM，HMM 平滑已有 VBx。只作为方案二的非端到端消融保留。
- D 是硬 EM 近似，被方案一完全取代：方案一是精确推断，有变分刻画，还考虑了分块。
- D′ 并入方案三。

**一和二可以合成一篇论文**："把多说话人同传形式化为结构化隐变量问题：谁在说（TurnCRF）× 何时、以多大块提交（ScheduleCRF）"。

---

## 1. 文献检索结论：空白在哪里

| 方向 | 已有工作（公开） | 有没有覆盖我们的场景 |
|---|---|---|
| 同传的隐变量时机 | HMT（Zhang & Feng, ICLR 2023）：把"何时开始翻译"当作隐状态，对候选时刻求边际 | 单说话人、编码器-解码器结构，没有分块和说话人，也不是用 CRF 后验做教师 |
| 同传的 oracle 或教师策略 | Zheng et al. ACL 2019（受限 dynamic oracle，模仿学习）；EMNLP 2021 用完整 NMT 构造读写 oracle；LSG（2025）让 LLM 在低延迟基线上提出改进策略 | 都是单说话人、规则或贪心的 oracle，没有精确的全局后验，也没有延迟约束的对偶解 |
| 期望延迟可微 | MILk 的 Differentiable Average Lagging（Arivazhagan et al., ACL 2019） | 期望是对单调注意力求的，不是对调度格的精确边际 |
| 流式中的标签偏置与全局归一化 | GNAT（Variani et al., NeurIPS 2022），流式 ASR，与非流式的差距缩小一半以上；Andor et al. ACL 2016；Goyal et al. NAACL 2019；CTC-CRF（Xiang & Ou, ICASSP 2019） | **检索没有发现同传或语音翻译中的全局归一化** |
| 多说话人流式语音翻译 | DiariST（ICASSP 2024，transducer + t-SOT + t-vector）；t-SOT 系列（Kanda 2022）；Sortformer（ICML 2025，按到达时间排序解决置换） | 输出顺序和时机都是固定规则（按时间戳或到达时间），没有对"串行化和提交时机"建模或求边际 |
| 串行化顺序的歧义 | 按时间戳排序（t-SOT，要求强制对齐）、用 prompt 固定说话人顺序（Moriya 2024）、学习一个串行化模块（Shi et al.） | 都是挑一个顺序，没有对顺序求边际（O(S!) 被认为太贵） |
| 神经 semi-CRF / HSMM 用于说话人分离 | HDP-HSMM（Johnson & Willsky，贝叶斯，非神经）；EEND 系后处理多为聚类或中值滤波 | **没有检到神经、流式、可处理重叠、端到端接入 LLM 的 semi-CRF 说话人轮次层** |
| CRF 作为网络内部的层 | Structured Attention Networks（Kim et al., ICLR 2017）；Torch-Struct（Rush 2020，有 GPU 版 SemiMarkovCRF） | 通用工具，没有用在说话人轮次或同传上 |
| 离策略引导的 GRPO | LUFFY（NeurIPS 2025）：混合策略 GRPO 加正则化重要性采样 | 引导来自外部模型的轨迹，不是结构化后验采样 |

**空白。** 多说话人、流式、基于 LLM 的同传里，"谁、何时、多大块"这一结构从来没有被当作**全局归一化的结构化隐变量**来精确推断和学习。

---

## 2. 方案一：ScheduleCRF——精确推断最优多说话人提交调度，用来训练流式 LLM

### 2.1 动机

我们已有三条证据：
1. `<wait>` 策略由**启发式截止时间标签**监督（`docs/mist_grpo_plan.zh-CN.md` 第 34 行）。在哪一次读提交、切成几块，都是人为定的，并不考虑"模型在这个时刻有没有足够的信息把这块译好"。
2. GRPO 七个变体都推不动提交时机和短插话：
   - 66.6% 的短插话漏译是 8 条 rollout 全部漏掉，组内没有对比，梯度为零；
   - 组内最好的那条奖励 0.55，平均只有 0.41，但 100 步后 dev R 只涨了 0.008。
   - 根本原因是**策略梯度只能学到采样到的东西**。
3. W³PO 的延迟奖励经 z 归一化后导致切碎：用标量奖励做延迟和质量的权衡不稳定。

**核心思想。** 不再用采样去找好的调度，而是在调度格上**精确**求出"质量减延迟"的熵正则最优分布（一个 CRF 后验），再把它蒸馏给流式 LLM。

- 所有调度都被考虑到，包括策略从来采样不到的"在这一刻输出插话"，所以全组漏掉导致零梯度的问题从结构上消失了。
- 延迟只作为一个约束出现，由对偶变量精确满足，不需要调权重。

### 2.2 严格模型

**符号**
- 窗口内有 R 次读，时间 $t_1<\dots<t_R$（$t_R$ 是窗口末的最终读）。
- 说话人 $k\in\mathcal K$。
- 说话人 k 的目标流 $Y_k=(y_{k,1},\dots,y_{k,N_k})$：k 的各参考单元译文按时间顺序拼接，用 token 表示。
- 候选块边界 $\mathcal B_k\subseteq\{0,\dots,N_k\}$：包含 0、$N_k$、所有单元结尾，以及单元内源语到目标语对齐单调的词边界。
- 就绪时间 $\tau_k(\beta)$：目标前缀 $Y_k^{<\beta}$ 对齐到的源语词全部被听到的时刻。单元结尾处就是 $b_u$。

**调度（隐变量）**

$$z_k=\big((\beta_{m-1},\beta_m,r_m)\big)_{m=1}^{M_k},\qquad 0=\beta_0<\beta_1<\dots<\beta_{M_k}=N_k,\ \ \beta_m\in\mathcal B_k$$

$$r_1<r_2<\dots<r_{M_k},\qquad t_{r_m}\ge\tau_k(\beta_m),\qquad t_{r_m}\le\tau_k(\beta_m)+D_{\max}\ \text{或}\ r_m=R$$

意思是：说话人 k 的译文被切成 $M_k$ 块，第 m 块 $Y_k[\beta_{m-1}:\beta_m)$ 在读 $r_m$ 提交。不允许重复或遗漏，每一次读每个说话人最多提交一块。

**块势函数**

$$\psi_k(\beta,\beta',r)=\underbrace{\sum_{j=\beta}^{\beta'-1}\log p_{\bar\theta}\big(y_{k,j}\mid x_{\le t_r},\,Y_k^{<j}\big)}_{\ell_k(\beta,\beta'\mid r)\ \text{质量：只听到 }t_r\text{ 之前时模型能否译好这块}}\;-\;\lambda\,\underbrace{\big(t_r-\tau_k(\beta')\big)}_{\text{该块的滞后}}\;+\;\underbrace{g(\beta'-\beta)}_{\text{语音保真先验}}$$

- $\bar\theta$ 是当前模型的冻结副本。
- 在同一个 (k, r) 下，所有块的 $\ell$ 可以由**一次**教师强制前向得到：对 $Y_k$ 逐 token 的 log 概率做前缀和，再取区间差。所以每个窗口只需要 $|\mathcal K|\cdot R$ 次前向。
- 近似处理：上下文中省略块之间的控制 token 和其他说话人的块，这一点需要写明。
- 不同切分覆盖的是同一串 token，所以 $\sum\ell$ 之间可以直接比较，没有长度偏置。
- $g(n)=\log\big(1-\widehat{\mathrm{WER}}_{\text{TTS}\to\text{ASR}}(n)\big)$：长度为 n 的块合成再识别后的保真度，在**训练集**的 TTS 输出上估计（不用 dev）。它把"语音门槛"直接写进调度目标，针对的是 ≤2 词片段 WER 约 40% 的问题。

**联合分布（CRF）**

$$q_\lambda(z\mid x,Y)=\frac{1}{Z_\lambda(x,Y)}\exp\Big(\sum_{k}\sum_{m}\psi_k(\beta_{m-1},\beta_m,r_m)\;+\;\sum_{r}\kappa\big(W_r(z)\big)\Big)$$

- $\kappa$ 是可选的跨说话人耦合（第二版）：$\kappa(W)=-\mu\,[W-\nu(t_{r+1}-t_r)]_+$。$W_r$ 是读 r 上所有说话人提交的词数，$\nu$ 是 talker 的实测语速。它表示输出语音的承载能力：同一时刻提交太多，语音就会积压。
- 没有 $\kappa$ 时，分布按说话人分解，$Z_\lambda=\prod_k Z_k$。
- 有 $\kappa$ 时是因子化的格，在联合状态 $(\beta^{(1)},\dots,\beta^{(K)})$ 上做精确动态规划。窗口内同时活跃的说话人不超过 3 个，可以承受。

**精确推断（半马尔可夫前向）**

$$\alpha_k(\beta',r)=\log\!\!\sum_{\beta<\beta',\ \beta\in\mathcal B_k}\ \sum_{r'<r}\exp\{\alpha_k(\beta,r')+\psi_k(\beta,\beta',r)\},\qquad \alpha_k(0,0)=0,\qquad \log Z_k=\log\sum_r e^{\alpha_k(N_k,r)}$$

- 对 $r'$ 用前缀 logsumexp，复杂度是 $O(|\mathcal B_k|^2R)$。
- 前向-后向给出块边际 $\gamma_k(\beta,\beta',r)$。
- 前向滤波加后向采样（FFBS）可以从 $q$ 中**精确采样**。
- 期望延迟 $\mathbb E_q[\mathrm{Lat}]=\sum\gamma\cdot(t_r-\tau)$ 可以精确计算。

**变分刻画：为什么说它"最优"**

由 Gibbs 变分原理，

$$q_\lambda=\arg\max_{q}\ \mathbb E_q\Big[\sum\ell+\sum g\Big]-\lambda\,\mathbb E_q[\mathrm{Lat}]+\mathcal H(q)$$

所以 $q_\lambda$ 是"质量最大，同时满足延迟约束 $\mathbb E_q[\mathrm{Lat}]\le L_{\text{ref}}$"的**熵正则最优调度分布**，$\lambda$ 就是这个约束的拉格朗日乘子。这与"控制即推断"（Levine 2018）的观点一致。

$\lambda^\*$ 的确定方法：在训练集上用二分法，使 $\mathbb E_{q_\lambda}[\mathrm{Lat}]$ 等于 v8 的实测平均滞后。这是**等延迟约束**，不是超参搜索，所以结果可以表述为"在同等延迟下质量更高"。

### 2.3 学习：把 CRF 教师蒸馏给流式 LLM（EM）

学生是 MIST 原样的 LLM（`<spk_k>`、`<wait>` 交错格式），**推理架构不变**。

$$\mathcal L(\theta)=\mathbb E_{(x,Y)}\ \mathbb E_{z\sim q_{\lambda^\*}}\big[-\log p_\theta(s(z)\mid x)\big]=\mathbb E\big[\mathrm{KL}(q_{\lambda^\*}\,\|\,p_\theta)\big]+\text{const}$$

- $s(z)$ 是调度 z 对应的交错 token 序列：每次读内各说话人的块按 $\tau$ 排序，再接 `<wait>`。
- 用 FFBS 采 M = 4 个样本做蒙特卡洛。
- EM：每轮训练后令 $\bar\theta\leftarrow\theta$，重新计算 $\ell$ 和 $q$，共 2 轮（预登记）。
- 这是对隐变量 z 的 EM。启发式标签只相当于 $q$ 的一个点估计，而且没考虑模型能力。

### 2.4 它为什么能解决当前问题

| 问题 | 机制 |
|---|---|
| 启发式截止标签的噪声 | 标签换成"模型在当下能否译好"加延迟约束的精确后验 |
| GRPO 全组漏掉、零梯度 | 不靠采样：格上包含所有调度，插话在其最佳读的提交概率由 $\ell$ 和延迟直接决定 |
| 延迟与质量权衡不稳（W³PO 切碎） | 延迟是约束，由对偶变量精确满足；不存在 z 归一化放大的问题 |
| 碎片、短片段语音差 | $g$ 项加上 $\ell$（信息不足的小块，对数概率低） |
| 语音门槛的 TTS 噪声 | 训练信号不经过 TTS 抽样，没有这类噪声 |
| **不能保证解决的** | 推理时的暴露偏差导致的重复（学生仍是自回归的）；diarizer 根本没检测到的插话（感知问题，交给方案二） |

### 2.5 预检（证伪优先，代价小）

1. 在 500 个训练窗口上计算 MAP 调度 $\hat z$，与现有启发式标签比较：
   - 提交读不同的单元占比；
   - 块长分布的变化；
   - ≤1 s 单元的提交读差异。
2. 若不同的占比 < 10%，说明方案一和现状差不多，**停止**。
3. 成本：只需要 $|\mathcal K|\cdot R$ 次前向乘以 500 个窗口，约 1 GPU 小时。动态规划代码和单元测试可以先在 CPU 上写好。

### 2.6 实验协议（单一设置，不扫参）

- $D_{\max}=4$ s，$\lambda=\lambda^\*$（等延迟），M = 4，EM 2 轮，$g$ 由训练集估计，第一版不用 $\kappa$。
- 从 v8 做 SFT 续训，步数和学习率与 v8 的 SFT 一致。
- dev 上用 guarded 协议跑正式门槛，同时看结构指标（区域 F1、时长 CDF、碎片率）和双种子语音。
- 对照：同步数的普通 SFT 续训，排除"多训几步"的混淆。

### 2.7 新颖性与最接近的工作

- **HMT**：对"开始翻译时刻"求边际，单说话人、单 token 粒度，没有延迟约束的对偶，也不是教师蒸馏。
- **dynamic oracle 和 EMNLP 2021 的 oracle**：都是贪心或规则，没有全局后验。
- **MILk 的 DAL**：期望延迟，但定义在单调注意力上。
- **我们的新意**：
  1. 多说话人、可分块的半马尔可夫调度格，及其跨说话人的语音承载耦合；
  2. 精确的熵正则最优调度和等延迟对偶；
  3. 用 CRF 后验教师加 EM 训练语音 LLM 的交错输出；
  4. 把 TTS 保真度写进调度目标。
- 检索中没有看到相同的工作，需要系统核查。

### 2.8 风险

- $\ell$ 的近似（省略控制 token 和其他说话人的上下文）可能让后验偏离真实可译性。预检和 EM 第二轮能部分发现这个问题。
- 学生学不会软目标（容量或格式限制）。
- 语音门槛仍然需要实测。

---

## 3. 方案二：TurnCRF——语音 LLM 内部的流式、可处理重叠的神经 semi-CRF 说话人轮次层（端到端）

### 3.1 动机

- MIST 的说话人信息完全来自**冻结 diarizer 的逐帧后验** $P\in[0,1]^{T\times 8}$（80 ms 一帧，`post/lowlat`），经说话人投影器输入 Thinker。diarizer 的错误直接向下传（方案文档 P10），而它是逐帧决策的，没有轮次、时长和边界结构。
- w3dia、w3dia2、EA 三次在输出端加说话人信用，都只是在"敢拆分"和"标得准"之间来回挪，**从输出端修说话人这条路已经证伪**。
- 合理的做法是：让说话人结构成为模型内部可学习的结构化隐变量，用翻译损失端到端地塑造它。

### 3.2 严格模型

**标签空间**（允许重叠，最多 2 人同时说）：

$$\mathcal C=\{\varnothing\}\cup\{k\}_{k=1}^{8}\cup\{\{k,k'\}\}_{k<k'},\qquad |\mathcal C|=37$$

**势函数（残差初始化）**，$e_t$ 是 Thinker 音频编码的因果状态：

$$u_t^{c}=\log\!\Big(\prod_{k\in c}P_{t,k}\prod_{k\notin c}(1-P_{t,k})\Big)+\big[\mathrm{MLP}_\phi(e_t)\big]_c,\qquad (\ell_t^c,r_t^c)=\mathrm{MLP}'_\phi(e_t)$$

MLP 初始化为 0。初始时内容势就是 diarizer 在"独立假设"下的联合标签概率，所以起点不比 diarizer 差。

**区段与切分分数**

$$s(i,j,c)=\frac{1}{j-i+1}\sum_{t=i}^{j}u_t^c+\ell_i^c+r_j^c+d^{\,c}_{j-i+1},\qquad S(z)=a_{c_1}+\sum_m s(i_m,j_m,c_m)+\sum_{m\ge2}A_{c_{m-1},c_m}+b_{c_M}$$

**流式：前缀分布加右截尾。** 在时刻 τ 只观测到 $x_{\le\tau}$，最后一段尚未结束：

$$\tilde s(i,\tau,c)=\frac{1}{\tau-i+1}\sum_{t=i}^{\tau}u_t^c+\ell_i^c+\bar d^{\,c}_{\tau-i+1},\qquad \bar d^{\,c}_q=\log\sum_{q'\ge q}e^{d^c_{q'}}$$

$$p_\tau(z\mid x_{\le\tau})\propto\exp\Big(a_{c_1}+\sum_{m<M}s(i_m,j_m,c_m)+\tilde s(i_M,\tau,c_M)+\sum_{m\ge2}A_{c_{m-1},c_m}\Big)$$

最后一段用生存势，表示真实长度至少为 q，也没有结束分。这是 HSMM 右截尾在神经 semi-CRF 中的对应，保证训练（随机截断前缀）与流式推理一致。

**与 LLM 的耦合：结构化注意力，可微**

$$\pi_t^{k}=\sum_{c\ni k}\ \sum_{i\le t\le j}\gamma^{c}_{i,j}\qquad(\gamma\text{ 由前缀前向-后向得到})$$

用 $\pi_t$ 替换 $P_t$，作为说话人投影器的输入。接口形状不变（T×8），梯度经前向-后向回传到 $\phi$，参照 Kim et al. 2017。

**损失**

$$\mathcal L=\mathcal L_{\text{LLM}}(\theta,\phi)+\mathbb E_{\tau}\Big[\tfrac{1}{\tau}\big(\log Z_\tau-\log Z_{\tau,y}\big)\Big]$$

- 第二项是带隐内部边界的边际似然（参考说话人和时间经匈牙利匹配映射到槽），对随机前缀 τ 取平均。
- 两项权重都固定为 1。

**复杂度。** $O(T\cdot L\cdot|\mathcal C|^2)$，T ≈ 375（30 s）、L = 50、|C| = 37，每窗口约 $2.6\times10^7$ 次运算。可以用 Torch-Struct 的 `SemiMarkovCRF` 在 GPU 上算。

### 3.3 解决的问题

- **错标和 collapse**：转移分保证相邻轮次一致；时长势防止槽抖动。
- **短插话检测**：短区段有专门的时长势和边界势，并且由翻译损失驱动。
- **P10（diarizer 冻结、错误下传）**：说话人结构变成可学习的，而且以 diarizer 为起点。

### 3.4 预检与协议

1. **非端到端消融（即旧方案 B，CPU）。** 先只训练 φ 的 CRF 部分（输入只有 P），看 dev 上的轮次 IoU-F1 和 ≤1 s 轮次召回是否高于原始 P。若没有提升，端到端版的前提就不成立。
2. **端到端版。** 从 v8 续训 SFT（同步数、同学习率），判定看门槛、错标、collapse 和短插话召回。
3. **论文对照表：** 输出层 ∈ {逐帧 P（现状）、线性链 CRF、semi-CRF} × {冻结、端到端}。

### 3.5 新颖性与风险

- **新颖性。** 检索没有发现"神经、流式（右截尾）、可处理重叠的 semi-CRF 说话人轮次层，经结构化注意力端到端接入语音 LLM"。
  - 贝叶斯 HDP-HSMM 是非神经、离线的；
  - EEND、Sortformer 都没有时长和区段结构；
  - DiariST 用的是 t-vector，同样没有区段结构。
- **风险。**
  - φ 只在 AMI 训练集上学，可能过拟合，残差初始化能缓解；
  - 槽是 diarizer 的局部编号，跨窗口的一致性仍然依赖 diarizer；
  - 计算开销。

---

## 4. 方案三：全局归一化的控制策略头（对抗流式标签偏置）

### 4.1 动机

- 流式决策只能看到部分输入。局部归一化（每一步 softmax）的模型存在标签偏置：每一步的概率都被迫归一，没法表达"这一步还不确定，等后面的证据"。
- Andor et al.（2016）证明，在前瞻有限时，全局归一化模型族严格比局部归一化的表达力强。
- GNAT（NeurIPS 2022）在流式 ASR 上用全局归一化，把与非流式的差距缩小了一半以上。
- 检索**没有发现**同传或语音翻译中的同类工作。

### 4.2 严格模型

- 控制动作：$a_r\in\mathcal A=\{\text{wait}\}\cup\{\text{write}(S):S\subseteq\mathcal K,|S|\le2\}$。
- 文本：$y$ 在给定 a 时仍由 LLM 局部归一化生成。

$$p_\theta(a\mid x)=\frac{\exp\sum_{r=1}^{R}\phi_\theta(a_{r-1},a_r,h_r)}{Z_\theta(x)},\qquad Z_\theta(x)=\sum_{a'\in\mathcal A^R}\exp\sum_r\phi_\theta(a'_{r-1},a'_r,h_r)$$

- $Z$ 由线性链前向精确计算，复杂度 $O(R|\mathcal A|^2)$。
- 监督：$a^\*$ 取方案一的 FFBS 样本，损失为 $-\log p_\theta(a^\*\mid x)$。
- $h_r$ 只用 $x_{\le t_r}$，所以势函数是因果的。

### 4.3 关键的诚实结论

- 如果推理是**不可撤回的贪心**决策，第 r 步的 argmax 只依赖 $\phi(a_{r-1},\cdot,h_r)$，这和局部 softmax 的 argmax 是同一种形式。全局归一化只改变训练出来的势函数，表达力上的优势体现不出来。
- 要发挥这个优势，需要**延迟承诺**：保留宽度为 W 的束，读 r 的动作到读 r+Δ 才最终确定。代价是控制决策多出 Δ 次读的延迟。
- 所以这个方案理论上有吸引力，但工程风险和延迟代价都大。建议放在一和二之后，作为理论分析章节或扩展。

---

## 5. 方案四：CRF 引导的结构化探索（在一之上接 RL）

### 5.1 动机

GRPO 失败的根因是"组内没有对比"（66.6% 全组漏掉）。方案一的后验 $q$ 能够精确采样到"在正确时刻输出插话"的调度。把这类调度混进 rollout 组，零对比的问题就消失了。

### 5.2 形式

- 组 $\mathcal G=\mathcal G_{\text{on}}\cup\mathcal G_{q}$：$|\mathcal G_{\text{on}}|=6$ 条是策略自己的 rollout。
- 另外 $|\mathcal G_q|=2$ 条的控制动作 $a\sim q$（FFBS），文本由 $\pi_\theta(\cdot\mid a)$ 生成。
- 优势在整个组上归一化。对引导轨迹的控制 token，使用 LUFFY 式的策略塑形重要性权重 $f(w)=w/(w+\gamma)$，其中 $w=\pi_\theta(a)/q(a)$。

$$J(\theta)=\frac{1}{|\mathcal G|}\Big[\sum_{i\in\mathcal G_{\text{on}}}\sum_t \min\big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t})A_i\big)+\sum_{i\in\mathcal G_q}\sum_{t\in\text{ctrl}}f(w_{i,t})\,A_i\Big]-\beta\,\mathrm{KL}$$

### 5.3 评估

- 新颖度中等：离策略引导已有 LUFFY，这里的新意是"引导来自结构化后验的精确采样"。
- 只在方案一有效之后，作为进一步提升来做。

---

## 6. 论文叙事与执行顺序

**题目方向（示意）：** *Interpretation as Structured Latent Scheduling: Who-When Lattices for Multi-Speaker Simultaneous Speech Translation*

- **主贡献一，ScheduleCRF。** 多说话人、可分块的提交调度格；精确的熵正则最优后验和等延迟对偶；EM 教师蒸馏。解决"采样学不到"与"启发式标签"两个问题。
- **主贡献二，TurnCRF。** 流式（右截尾）、可处理重叠的神经 semi-CRF 说话人轮次层，经结构化注意力端到端接入语音 LLM。解决说话人结构问题。
- **分析：**
  - GRPO 在强 SFT 模型上失效的原因（零对比、TTS 噪声）——已有数据；
  - 语音门槛的噪声分析；
  - 结构指标（区段 F1、时长 CDF）。

**执行顺序**

1. 方案一：写动态规划、FFBS 和对偶求解的代码与单元测试（CPU），然后做 500 窗口的预检（约 1 GPU 小时）。
2. 方案二：先做非端到端消融（CPU），再做端到端版（GPU）。
3. 一、二都有效后，合在一起训练；可选再接方案四。
4. 方案三放到最后，作为理论扩展。

**现在能做的（没有 GPU）：** 一的格、前向-后向、FFBS、λ\* 二分求解及单元测试；二的前缀截尾 semi-CRF 和单元测试；一的预检脚本（等 GPU 后运行）。

---

## 7. 统一约束（照旧）

- 只用 AMI 训练集训练；RealSI、ACL 60/60、AliMeeting 只做测试；dev 只做选择。
- 正式门槛：文本 all、ov、语音三项，点估计 ≥ ref − 0.2，配对 bootstrap P(diff≤0) < 0.95。
- 每个方案只跑一个预登记设置，不扫参数。
- 代码只做增量修改，默认行为逐位不变，配单元测试。

---

## 8. 参考文献（公开；写论文前逐条核对原文）

- Lafferty, McCallum & Pereira. Conditional Random Fields. ICML 2001.
- Sarawagi & Cohen. Semi-Markov Conditional Random Fields for Information Extraction. NeurIPS 2004.
- Andor et al. Globally Normalized Transition-Based Neural Networks. ACL 2016.
- Goyal, Dyer & Berg-Kirkpatrick. An Empirical Investigation of Global and Local Normalization for Recurrent Neural Sequence Models. NAACL 2019. https://arxiv.org/abs/1904.06834
- Variani et al. Global Normalization for Streaming Speech Recognition in a Modular Framework (GNAT). NeurIPS 2022. https://arxiv.org/abs/2205.13674
- Xiang & Ou. CRF-based Single-stage Acoustic Modeling with CTC Topology (CTC-CRF). ICASSP 2019; CAT toolkit https://arxiv.org/abs/1911.08747
- Kim, Denton, Hoang & Rush. Structured Attention Networks. ICLR 2017. https://arxiv.org/abs/1702.00887
- Rush. Torch-Struct: Deep Structured Prediction Library. ACL 2020 Demo. https://arxiv.org/abs/2002.00876
- Zhang & Feng. Hidden Markov Transformer for Simultaneous Machine Translation. ICLR 2023. https://arxiv.org/abs/2303.00257
- Arivazhagan et al. Monotonic Infinite Lookback Attention for SiMT (DAL). ACL 2019. https://arxiv.org/abs/1906.05218
- Zheng et al. Simultaneous Translation with Flexible Policy via Restricted Imitation Learning. ACL 2019. https://arxiv.org/abs/1906.01135
- Zhang et al. StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning. ACL 2024. https://arxiv.org/abs/2406.03049
- Yang, Kanda et al. DiariST: Streaming Speech Translation with Speaker Diarization. ICASSP 2024. https://arxiv.org/abs/2309.08007
- Kanda et al. Streaming Speaker-Attributed ASR with Token-Level Speaker Embeddings. Interspeech 2022. https://arxiv.org/abs/2203.16685
- Park et al. Sortformer. ICML 2025. https://arxiv.org/abs/2409.06656
- Moriya et al. Alignment-Free Training for Transducer-based Multi-Talker ASR. 2024. https://arxiv.org/abs/2409.20301
- Johnson & Willsky. The Hierarchical Dirichlet Process Hidden Semi-Markov Model. UAI 2010. https://arxiv.org/abs/1203.3485
- Fujita et al. End-to-End Neural Speaker Diarization (EEND). Interspeech / ASRU 2019.
- Yan et al. Learning to Reason under Off-Policy Guidance (LUFFY). NeurIPS 2025. https://arxiv.org/abs/2504.14945
- Levine. Reinforcement Learning and Control as Probabilistic Inference. arXiv 1805.00909, 2018.
- Ganchev et al. Posterior Regularization for Structured Latent Variable Models. JMLR 2010.

---

## 9. 补充：是否端到端、是否基于 v8 续训

v8 由三部分组成：Qwen3-Omni-30B-A3B Thinker（冻结）、48 层 q/k/v/o 上的 LoRA（r = 32）、说话人投影器（输入是冻结 diarizer 的 8 槽后验）。音频编码器和 diarizer 都不训练。

四个方案都**从 v8 checkpoint 起步续训**，不从头训练，训练数据仍是 AMI 训练集窗口。

| 方案 | CRF 在哪里 | 梯度是否穿过 CRF | 推理时有没有 CRF | 是否端到端 | 对 v8 架构的改动 |
|---|---|---|---|---|---|
| 一 ScheduleCRF | 训练时离线计算的**教师**；势函数来自 v8 的冻结副本 | 否：用 EM 交替，CRF 生成目标，LLM 用 SFT 学 | 无 | **不是**端到端 CRF，是"结构化教师 + v8 续训" | **无改动**，只换 SFT 目标 |
| 二 TurnCRF | 模型**内部**的一层，位于 diarizer 后验与说话人投影器之间 | **是**：翻译损失经前向-后向回传到 CRF 参数 φ 和 LoRA | 有（流式前缀推断） | **是**：CRF 层与 LoRA 联合训练；但 diarizer 和音频编码器仍冻结，不是从波形开始的全端到端 | 增加一个小层，接口形状 T×8 不变；残差初始化，起点接近 v8 |
| 三 全局归一化策略头 | 模型内部的控制头，接在 Thinker 隐状态上 | 是 | 有：由它决定 wait 或 write，需要延迟承诺 | 是 | 改动解码器：控制决策不再由 `<wait>` token 产生 |
| 四 CRF 引导探索 | 用来采样引导轨迹 | 否 | 无 | 不是 | 无，在 v8 或方案一的 checkpoint 上做 RL |

**取舍**
- 方案一风险最低：推理不变，可以直接和 v8 对照。但它的 CRF 只在训练时出现，审稿人可能不认为它是"端到端 CRF"。它的卖点应该放在精确的最优调度后验和"采样学不到"问题的消除上。
- 如果需要"端到端 CRF"作为卖点，就要靠方案二（改动小）或方案三（改动大）。推荐**一加二的组合**：一提供结构化的训练目标，二提供模型内部端到端的结构层。

---

## 10. 补充（2026-11-09）：HMT 的启发与方案一的端到端修订（方案一′）

### 10.1 HMT 做了什么

来源：Zhang & Feng, ICLR 2023，arXiv 2303.00257；代码 github.com/ictnlp/HMT。代码基于 fairseq，MIT 许可，Python 3.6，PyTorch 1.7。

- **隐状态。** 每个目标 token $y_i$ 有 K 个候选状态，第 k 个状态对应"读了 $t_{i,k}=\min\{L+(i-1)+(k-1),|x|\}$ 个源 token 后开始翻译"。L 是下界（wait-L），上界是 wait-(L+K−1)。
- **发射概率。** $p(y_i\mid x_{\le t_{i,k}},y_{<i},z_i=k)$。训练时把 decoder 输入复制 K 份，一次前向算出全部状态。$s_{i,k}$ 在自注意力里可以看到之前 token 的**所有**状态（只要 $t_{j,k'}\le t_{i,k}$），所以 $y_i$ 的上下文与之前选了哪个状态无关，边际似然是精确的。
- **转移概率。** 先算置信度 $c_{i,k}=\sigma([\bar h_{\le t_{i,k}};s_{i,k}]W^S)$，并令 $c_{i,K}=1$。然后
  $p(z_i\mid z_{i-1})=c_{i,z_i}\prod_{l:\,t_{i-1,z_{i-1}}\le t_{i,l}<t_{i,z_i}}(1-c_{i,l})$。
- **损失。** $\mathcal L=-\log\sum_z\prod_i p(y_i\mid\cdot,z_i)p(z_i\mid z_{i-1})+\lambda_{lat}\sum_z p(z)\,\mathrm{AL}(z)+\lambda_{state}\cdot\big(-\tfrac1K\sum_{i,k}\log p(y_i\mid\cdot,z_i{=}k)\big)$，两个 λ 默认都是 1（论文在 0–2 之间扫过）。
- **推理。** 依次检查各状态：$c_{i,k}\ge0.5$ 就写，否则读一个源 token。$c_{i,K}=1$ 保证最晚在第 K 个状态写出。
- **结果。**
  - 数据：IWSLT15 En→Vi、WMT15 De→En（文本同传）。
  - 边际似然 29.29 BLEU（AL 3.46），只取最大状态 24.02（AL 1.77）。两者**延迟不同**，不能直接读成"边际化 +5 BLEU"。
  - 去掉状态损失：29.29 → 27.74。
- **代码位置。**
  - `fairseq/criterions/label_smoothed_cross_entropy.py`：`HMM_loss`、`Latency_loss`、`CE_loss`。
  - `fairseq/sim_sequence_generator.py`：`SimSequenceGenerator._generate`。
  - 代码与论文有出入：转移概率在代码里做了 logsumexp 重归一化；延迟损失用了 abs。复现时以论文公式为准。

### 10.2 关键核对：v8 的解码天然满足 HMT 式的精确分解

`experiments/mist_w3po_v1/rl/rollout.py` 的文档字符串和第 133 行显示：第 r 次读的提示是"音频到 $t_r$ + 跨窗上下文 c + 已提交的 token"。已提交部分去掉了 `<wait>` 和 `<eos>`，不记录每段是在哪次读提交的。读取周期是 1 s（`core/engine.py` 的 `read_times`）。

**符号**
- S：窗口的目标流，即 G-STAR 格式串行化的序列，说话人顺序与 SFT 标签相同。
- 调度 $z=(\beta_1\le\dots\le\beta_R=|S|)$，$\beta_r$ 是第 r 次读之后已经提交的 token 数，允许空块。
- $e_r$ 是第 r 次读的结束符：r < R 时为 `<wait>`，r = R 时为 `<eos>`。

$$p_\theta(S,z\mid x)=\prod_{r=1}^{R}p_\theta\big(S[\beta_{r-1}{:}\beta_r)\oplus e_r\ \big|\ x_{\le t_r},c,S^{<\beta_{r-1}}\big)$$

这**正是** v8 实际解码时生成 (S, z) 的概率。每次读的条件只依赖 $(r,\beta_{r-1})$，不依赖更早的提交时刻，所以不需要 HMT 的 K 倍复制，也**不改架构**。块势函数为

$$\psi_r(\beta,\beta')=\sum_{j=\beta}^{\beta'-1}\log p_\theta(S_j\mid x_{\le t_r},c,S^{<j})+\log p_\theta(e_r\mid x_{\le t_r},c,S^{<\beta'})$$

- 在因果注意力下，对每次读做一次"音频到 $t_r$ + c + 完整 S"的教师强制前向，就能拿到所有 $\psi_r$。
- 可行约束：
  - 段的起止时间 token 不能晚于 $t_r$；
  - 单元内的文本 token 要在 $t_r\ge a_u$ 时才能提交（有词对齐时可以收紧）；
  - $t_r\le b_u+D_{\max}$，或者 r = R。
- 和 §2 的近似相比，这里**没有省略控制 token 的上下文**。

### 10.3 方案一′：受延迟约束的隐调度边际似然（端到端，严格 CRF）

- 给定 S，后验 $q_\theta(z\mid x,S)=p_\theta(S,z\mid x)/p_\theta(S\mid x)$ 是以 $\beta_r$ 为状态的线性链 CRF，等价于 S 上的 semi-Markov 切分。它是全局归一化的，配分函数可以精确计算。
- 前向递推：$\alpha_r(\beta')=\log\sum_{\beta\le\beta'}\exp\{\alpha_{r-1}(\beta)+\psi_r(\beta,\beta')\}$，复杂度 $O(R\cdot|S|\cdot W)$，W 是块长上限。

**延迟约束（posterior regularization，Ganchev 2010）。** 定义 $\mathrm{Lat}(z)=\frac1{|S|}\sum_r\sum_{\beta_{r-1}\le j<\beta_r}(t_r-\tau_j)$，它可以按块分解。

$$J(\theta)=\log\sum_z p_\theta(S,z\mid x)\,e^{-\lambda^\*\mathrm{Lat}(z)}=\max_q\Big\{\mathbb E_q[\log p_\theta(S,z\mid x)]+\mathcal H(q)-\lambda^\*\mathbb E_q[\mathrm{Lat}]\Big\}$$

- 达到最大值的 q 是 $q_{\lambda^\*}\propto p_\theta(S,z\mid x)e^{-\lambda^\*\mathrm{Lat}}$。
- $\lambda^\*$ 用二分法确定，使训练集上的 $\mathbb E_q[\mathrm{Lat}]$ 等于启发式标签的平均滞后（v8 学的就是这些标签；同一个 S，口径完全一致）。这是**等延迟约束，不是调权重**。

**总损失**

$$\mathcal L=-J(\theta)+\mathcal L_{state},\qquad \mathcal L_{state}=-\frac1{|\mathcal A|}\sum_{(r,j)\in\mathcal A}\log p_\theta(S_j\mid x_{\le t_r},c,S^{<j})$$

- $\mathcal L_{state}$ 照搬 HMT 的状态损失，权重 1，预登记。$\mathcal A$ 是可行的 (读, token) 对。它防止后验塌到单个读上，导致其他读的发射没有被训练到。

**梯度的实现（Fisher 恒等式）**

$$\nabla J=\mathbb E_{q_{\lambda^\*}}\big[\nabla\log p_\theta(S,z\mid x)\big]=\sum_r\sum_{\beta\le\beta'}\gamma_r(\beta,\beta')\,\nabla\psi_r(\beta,\beta')$$

拆到 token 上：
- 第 r 次读的文本 token j，权重 $P_q(\beta_{r-1}\le j<\beta_r)$；
- 第 r 次读在位置 β′ 的结束符，权重 $P_q(\beta_r=\beta')$。

做法：
1. 用无梯度前向算 ψ，再做前向-后向，得到权重；
2. 对逐读的行做**加权交叉熵**。行格式与 v8 的 SFT 读取行相同，各行之间独立，可以梯度累积。

代价约等于 v8 的 SFT 加一次无梯度前向。推理完全不变：测试时仍由 v8 自己的 `<wait>` 概率决定读或写，和 HMT 推理时用 c 一样。

**与 §2 的关系。** §2 的做法是冻结 $\bar\theta$、FFBS 采 4 个样本做 SFT、EM 两轮。如果每一步都刷新 $\bar\theta=\theta$、用精确期望代替采样，并且在势函数中加入模型自己的 `<wait>` 概率，得到的就是 ∇J。所以"蒸馏"是 J 的离线近似，一′是精确的在线版本。**建议用一′取代 §2.3。**

- §2 中不含模型 `<wait>` 先验的版本是"与当前习惯无关的最优调度"。一′含先验，是自洽的 EM。
- 第一版先不加 g 和 κ；块长分布由预检监控。

**限制与扩展（一′-b）。** 上面的 S 固定了说话人的串行化顺序。如果让顺序也成为隐变量（例如在 A 的长段中途插入 B 的插话），就需要多说话人的联合格；而已提交历史的 token 顺序依赖调度，分解不再精确。可以在推理时把历史按源时间规范化排序来近似：v8 本来每次读都重新编码，这样做没有额外代价。**多说话人方面的新意主要在这一步。**

### 10.4 与 HMT 的区别

| | HMT | 一′ |
|---|---|---|
| 任务 | 单说话人文本同传 | 多说话人同传，语音 LLM 输出文本和语音 |
| 粒度 | 逐 token 的开始时刻（HMM） | 按块提交（semi-Markov，空块允许） |
| 精确性从哪里来 | encoder-decoder，K 倍状态复制，自注意力看所有状态 | v8 每次读重新编码、历史不含读标记：天然精确，**无额外状态、无新参数** |
| 转移 / 置信 | 新加的置信头 $c_{i,k}$ | 复用 v8 的 `<wait>` token 概率 |
| 延迟 | 惩罚项，权重扫 0–2 | 等延迟约束，用对偶精确求解 |
| 多说话人 | 无 | 串行化顺序作为隐变量（一′-b） |

**诚实评估。** "对时机求边际"这一核心思想属于 HMT，必须作为最接近的工作引用。审稿人可能会把一′看成"HMT 搬到语音 LLM"。真正的新意需要落在一′-b（多说话人的交错顺序）、等延迟对偶，以及对强 SFT 模型的分析上。

### 10.5 S0 教训与预检

**S0-all 的情况（2026-11-07）**
- 它**不是**从 v8 续训，而是从与 v8 相同的起点（`thinker_gstar_o2_20261029b` 合并权重）重训 3000 步，同时改了 5 项。其中 S0-1 把截止时间随机提前 U(−0.5, 1.0) s。
- 结果：ov −4.62（P≤0 0.9998），重复率 6.4 → 9.5%，ov 提交 p50 从 1.66 s 提前到 0.96 s。
- 留一消融 no1–no5 没有跑完（作业被挂起，之后节点被回收）。所以没法知道是哪一项拖了后腿。

**一′与 S0 的区别**
- 一′只在 ψ 显示"此刻已经能译好"的地方提前提交，平均延迟被约束为与标签相同，**不可能整体提前**。
- 但 S0 失败的方式就是一′最需要防的风险。

**预检**：在 500 个训练窗口上，用 v8（无梯度）计算 $q_{\lambda^\*}$，再与启发式标签比较：
1. 提交读不同的单元占比，< 10% 则停止；
2. ov 区提交时间的平均偏移，如果明显提前（例如 > 0.3 s），就是 S0 风险，停止或先改设计；
3. ≤ 2 词块的占比，明显上升则加入 g。

**协议**：同 §2.6。另设一个对照：从 v8 出发、用原启发式标签、相同步数的普通续训 SFT。

### 10.6 一′与方案二

- **相互独立。** 两者的变量不同（一′是输出"何时提交"，二是输入"谁在说"），参数也不同（一′只用 LoRA，二是 φ 加 LoRA），都从 v8 起步。可以作为两个**不同的方法**并行验证，不违反"不扫参"的规则；之后做 2×2 组合。
- **"严格 CRF"和"端到端"是两个轴。**
  - 严格 CRF 指模型形式：全局归一化，配分函数精确。
  - 端到端指梯度是否穿过 CRF。
  - §2 的方案一在形式上是严格 CRF，只是不端到端；一′两者都满足；方案二也两者都满足。
- **不冲突。** 它们是两个不同隐变量上的因子，通过 LLM 耦合。
  - 唯一的实际影响：一′的势函数依赖说话人输入，方案二改变输入之后势函数需要重新计算。一′的势函数本来就是在线计算的，会自动更新。
- **论文。** 先做哪个只是执行顺序，不决定论文叙事。两个独立结构叠在一起，读起来像系统论文。建议以一′（含一′-b）为唯一主贡献，方案二单独成文，或作为同一原理（"语音 LLM 内部用精确动态规划对隐结构求边际"）的第二个实例。

---

## 11. 当前方案清单与单独验证协议（2026-11-09，取代 §0 的表）

| 方案 | 状态 | 能否单独验证 | 先做的低成本证伪 | 正式验证 | 依赖 |
|---|---|---|---|---|---|
| **一′ 隐调度边际似然**（§10.3，取代 §2.3 的教师版） | 主推 | 能 | ① CPU：动态规划 + 加权损失的单元测试，用 HMT 的 `HMM_loss` 在退化情形对拍；② 500 个训练窗口的后验预检，约 1 GPU 小时，停止条件见 §10.5 | 从 v8 续训，4 卡 | 无 |
| 一′-b 交错顺序也作为隐变量 | 一′ 的扩展 | 只能在一′有效之后，和一′同步数对比 | 同上 | 同上 | 一′ |
| **二 TurnCRF** | 主推，可与一′并行 | 能 | CPU 非端到端版：dev 上的轮次 IoU-F1 和 ≤1 s 轮次召回都要高于原始 P，否则停止 | 从 v8 续训，4 卡 | 无 |
| 三 全局归一化策略头 | 暂缓 | 原则上能 | — | 要改解码（延迟承诺） | 用一′的后验样本做监督 |
| 四 CRF 引导 GRPO | 暂缓 | 能，验证方式同以前的 GRPO 小测 | — | 50/100 步，对照 dia 100 | 一′的后验 q |
| §2 的教师版方案一 | 被一′取代 | 不单独跑 | — | 只作为论文消融（离线教师 vs 在线精确） | — |

**一′和二共用的验证协议**（与 GRPO 小测一致的地方：从 v8 起步、单一预登记设置、dev 上的正式门槛）：
- **共用对照 C0**：从 v8 出发，用原启发式标签做普通续训 SFT，步数、学习率、数据都与两个方法相同。它是对照，不是超参取值，不违反"不扫参"。
- **步数**：与 v8 的 SFT 相同（3000 步）。中途在第 1000 步评测一次，与 GRPO 评测第 50/100 步同理。
- **门槛**：对 v8 跑正式门槛（文本 all、ov、语音；点估计 ≥ ref − 0.2；按会议配对 bootstrap P(diff≤0) < 0.95）。语音用两个 TTS 种子（`evaluation/gate_2seed.py`）。dev R 只作方向信号。
- **有效**：通过门槛，并且在目标指标上优于同步数的 C0。
  - 一′：ov tcpBLEU、≤1 s 单元召回、重复率；同时 ov 提交 p50 和 AL 与 v8 相差不超过 ±0.1 s，即延迟相等。
  - 二：窗口错标率、collapse、短插话召回。
- **卡**：C0 和一′先跑；二在 CPU 阶段通过后再上卡。三个运行可以并行（每个 4 卡），总计约 12 卡次 × 3–4 小时。
