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

---

## 12. 方案二为什么没通过：诊断（2026-11-09）

只读诊断：`experiments/mist_w3po_v1/crf/diag_turn_short.py`，输出在 `eval_outputs/mist_w3po_v1/crf/turn_v1_dev_diag/diag.json`。数据为 dev 上 2,252 个 ≤1 s 参考轮次，其中 55.6% 处在别人说话期间（重叠）。

| 系统 | 召回 IoU≥0.5 | 召回（任意重叠） | 孤立短轮次召回 | 重叠短轮次召回 | 预测短轮次数 | 短轮次精确率 | 完全无参考的假短轮次 | 软值 max≥0.5 |
|---|---|---|---|---|---|---|---|---|
| raw P | 0.873 | 0.941 | 0.920 | 0.836 | 4858 | 0.457 | 50.1% | 0.954 |
| med5 | 0.862 | 0.911 | 0.910 | 0.824 | 3962 | 0.552 | 41.3% | — |
| semi 离线 | 0.616 | 0.671 | 0.855 | **0.426** | 1867 | 0.780 | 15.7% | 0.724 |
| semi 流式 | 0.663 | 0.727 | 0.874 | **0.494** | 2499 | 0.640 | 28.1% | 0.762 |

训练集参考：≤1 s 的轮次占轮次数的 32.7%，但只占帧数的 7.4%。

semi 学到的参数：
- 内容权重 a 为 ∅ 0.63、单人 0.26、双人 0.26。
- 双人标签的 start / end 分别是 −1.9 / −2.6。
- 11 种转移势全为负（−0.9 到 −3.1）。
- 短时长势为负，偏好长段。

**原因（不是代码错误：测试对暴力枚举都精确一致）**

1. **训练目标与目标指标错位。**
   - 最大似然按帧计分，短轮次只占 7.4% 的帧。最优解就是"牺牲短轮次换取长段平滑"：内容权重压到 0.26，所有切换都罚分。
   - 召回在"任意重叠"标准下也从 0.94 掉到 0.67，所以主要是**整段被抹掉**，不是边界对不准。
2. **只看 P，信息上限决定了只能在精确率和召回之间挪动。**
   - raw P 的预测短轮次有一半完全不对应任何参考（假"毛刺"）。只看 P 时，真插话和毛刺长得一样，去掉毛刺必然连真插话一起去掉。
   - 实测：短轮次精确率从 0.46 升到 0.78，召回从 0.87 降到 0.62。（事后看短轮次 F1：raw 0.60、semi 离线 0.69、semi 流式 0.65。这是事后指标，不能替代预登记规则。）
3. **联合标签的结构惩罚集中打在重叠插话上。**
   - 一次重叠插话需要两次转移（单人 → 双人 → 单人），两次都是负分，再加上双人的低内容权重和负时长势。
   - 结果：重叠短轮次召回从 0.84 掉到 0.43，孤立短轮次只从 0.92 掉到 0.85。
   - 而重叠插话（backchannel）正是方案二要救的对象。
4. **semi 时长相对线性链没有增益。** 学到的时长势同样偏好长段；P 本身已经平滑，显式时长没有带来新信息。

**方案本身的问题（比训练目标更根本）**

- **瓶颈不在感知。**
  - raw P 已经能找到 94% 的 ≤1 s 轮次（IoU≥0.5 时为 87%）。
  - 但 v8 的 dev 短单元翻译召回只有 ≤0.5 s 单人 57.6%、重叠 70.5%。两者口径不同（轮次 vs 单元），只能作粗比。
  - 所以短插话的漏译主要发生在 LLM 的"何时、是否翻译"，也就是方案一′的范围，而不是 diarizer 的检测。
- **已有的上界证据。**
  - 用 oracle diarizer 替换后，En→Zh 并没有变好（42.32 对 42.81；`docs/mist_grpo_plan.zh-CN.md` P10）。改善说话人输入对翻译质量的上限很小。
  - 方案二可能的收益只在说话人归属一侧（本次轮次 F1 +0.06、帧错误 −26%），代价是插话。
- **端到端版本会继承这个问题。** 辅助损失仍然是按帧的最大似然。短插话在翻译损失里也只占极少 token，梯度同样偏向长段平滑。音频特征能补一些信息，但瓶颈不在这里。

**结论**：方案二失败有两层原因：一是训练目标错位，二是只看 P 的信息上限。更根本的是它瞄准的不是当前的瓶颈。建议放弃方案二，资源集中到方案一′。

如果将来仍想做说话人结构，需要作为新方法重新预登记，至少要：
- 按轮次（而不是按帧）计分，或对短轮次加权；
- 让重叠进出有独立参数；
- 使用音频特征；
- 以"在 oracle 条件下说话人归属指标仍有提升空间"为前提先做核查。

---

## 13. 方案 1′-b：SerialCRF——同读多说话人提交的"序列化"隐变量 CRF（2026-11-09，设计稿，未实现）

**一句话。** 同一次读里如果有多个说话人的单元都该写，它们按什么顺序串行化，是一个隐变量。我们把"已提交历史"规范成一个集合（与写出顺序无关），这样就能对所有合法顺序**精确求和**，得到一个严格的 CRF；再用确定性退火的后验正则化，把模型从"只会标签顺序"的不动点上拉出来。

**回答两个问题。**
- **是严格的 CRF。** 它是给定 (x, S) 时，在"序列化排列"这个结构化隐变量上的全局归一化分布，配分函数精确可算（§13.2）。
- **是端到端的。** 势函数就是 LLM 自己的条件对数概率；损失是 CRF 的（退火）对数配分函数；梯度经 logsumexp 直接回传到 LoRA。没有冻结的教师，也没有额外模块；推理时不需要 CRF 解码（§13.4）。和 v8 一样，diarizer 与音频编码器冻结，所以不是从波形开始的全端到端。

### 13.1 证据与机制假说

- **证据**（dev，v8 正式闸门参考的在线解码；`sched_precheck_v8.zh-CN.md` §7–§8）：
  - ≤1 s 的重叠单元漏掉 295 个，其中"先写对方、插话再也没写"占 36–51%，74% 发生在插话应写的那一次读。
  - 对方同时段恰好在应写读首次写出时，漏译率 28%，其他情况 16–18%。
  - 教师强制重打分：最终读上 43% 的重叠短单元（≤0.5 s 时 61%），模型更想先写对方的待写段。
- **机制假说 H（序列化暴露偏差）。**
  - 训练目标流 S 按"(提交读, 源起点)"排序，是时间单调的。所以"起点更晚的单元已经写了、起点更早且已到期的插话还没写"这种状态，在训练里**从来没出现过**。
  - 模型一旦按自己的偏好先写了对方的段，就不知道还要补写插话，于是写 `<wait>`，插话永久漏掉。
  - H 需要在预检里验证（§13.6 的 P2）。

### 13.2 严格模型

**符号**
- 单元（参考段）u：说话人 k_u，源起点 a_u，token 序列 y_u（G-STAR 段：说话人 token、时间 token、文本、时间 token）。
- 读 r = 1..R，提交读 ρ(u)。v1 固定为启发式标签，依据是 §10/预检：v8 的调度后验与标签 96% 一致。
- $B_r=\{u:\rho(u)=r\}$。单元跨读切成块时，把块当作条目。
- 合法顺序集合 $\Pi(B_r)$ 是 $B_r$ 在以下约束下的全部线性扩展：
  - 同一单元的块保持原顺序；
  - 上一读留下的未闭合段的续写块必须排第一（G-STAR 中，写出说话人 token 会把当前段闭合）。

**集合状态历史（关键）**
- $h_r=\kappa\big(\bigcup_{r'<r}B_{r'}\big)$，κ 是把"已提交条目的集合"按（ρ, a_u, 块序号）规范串行化的确定性函数。它只依赖集合，**不依赖之前各读的顺序 π_{<r}**。
- 推理时引擎在每次读按 κ 重排已提交的段，作为默认关闭的开关。v8 本来每次读都重新编码整个提示，所以没有额外开销。
- 在标签路径上，κ 的输出恰好就是现有 S 的前缀，所以训练行与 v8 / C0 逐位一致。

**势函数**

$$\psi_r(\pi)=\log p_\theta\big(B_r^{\pi}\oplus e_r\ \big|\ x_{\le t_r},c,h_r\big),\qquad \pi\in\Pi(B_r)$$

$B_r^{\pi}$ 是按 π 拼接的条目 token。$e_r$ 是 `<wait>`，最后一次读为 `<eos>`。

**联合分布与后验（CRF）**

$$p_\theta(S,\pi\mid x)=\prod_{r=1}^{R}e^{\psi_r(\pi_r)},\qquad q_\theta(\pi\mid x,S)=\prod_{r=1}^{R}\frac{e^{\psi_r(\pi_r)}}{Z_r},\qquad Z_r=\sum_{\pi'\in\Pi(B_r)}e^{\psi_r(\pi')}$$

- 这是 $\Pi(B_1)\times\dots\times\Pi(B_R)$ 上的全局归一化 Gibbs 分布，$\log Z=\sum_r\log Z_r$ 精确可算。
- 按读分解是集合状态历史带来的。若历史保留写出顺序，相邻读之间会耦合，就需要在"(集合, 末尾顺序)"状态上做动态规划。
- $|B_r|\le4$ 时最多 24 项，直接枚举；$|B_r|>4$ 时只用标签顺序，并报告这种读的占比。
- $p_\theta(S,\pi\mid x)$ 正是 v8（带集合状态历史）实际生成"这些条目、按这些顺序"的概率。与 §10.2 同理，分解是精确的，不是近似。

**v2（联合调度 × 序列化，供论文完整性）**
- 隐变量 z = (ρ, π)，状态是"已提交集合" A（各说话人的下闭集）。
- 递推：

$$\alpha_r(A)=\log\!\!\sum_{\substack{A'\subseteq A\\ A\setminus A'\text{ 在 }r\text{ 可行}}}\!\!\exp\Big\{\alpha_{r-1}(A')+\log\!\!\sum_{\pi\in\Pi(A\setminus A')}\!\!e^{\psi_r(\pi\mid A')}\Big\}$$

- 这是集合状态上的 semi-Markov CRF。状态数上界为 $\prod_k(n_k+1)$；同时活跃的说话人 ≤ 3，可以承受。
- 它把一′（单一顺序下的链）推广到多说话人，并覆盖"跨读补写"（第 r 读先写对方，第 r+1 读再补插话）。v1 不做它，因为预检显示调度后验约等于标签。

### 13.3 学习（端到端）

**纯边际似然**

$$J_{\rm ML}(\theta)=\sum_r\log Z_r=\sum_r\log P_\theta\big(\text{第 }r\text{ 读恰好写出整个到期集合 }B_r\text{（任意顺序）}\big)$$

- 它就是"不漏、不重复、不论顺序"的集合似然。
- 梯度（Fisher 恒等式）：$\nabla J_{\rm ML}=\sum_r\mathbb E_{q_\theta(\pi_r)}[\nabla\psi_r(\pi_r)]$，即对每个顺序做后验加权的交叉熵。
- **一′ 的教训。** 如果 $q_\theta$ 集中在标签顺序上（v8 就是在它上面训出来的），$J_{\rm ML}$ 约等于 C0，又会卡在不动点。

**补救：确定性退火的后验正则化**（Ueda & Nakano 1998 的 DAEM；Ganchev et al. 2010 的 PR）

$$q_r^\*=\arg\min_q\ \mathrm{KL}\big(q\,\|\,q_\theta(\cdot\mid r)\big)\quad\text{s.t.}\quad \mathcal H(q)\ge\eta\log|\Pi(B_r)|,\qquad q_r^\*\propto q_\theta(\cdot\mid r)^{1/(1+\mu_r)}$$

- 对偶变量 $\mu_r\ge0$ 在每一读上用一维二分精确求出（因为是枚举）。η = 0.5 预登记，不扫参。
- 损失：$\mathcal L=-\frac1D\sum_r\sum_{\pi}q_r^\*(\pi)\,\psi_r(\pi)$，$q^\*$ 不回传梯度，D 与 C0 用同一种 token 归一化。这是一个精确 E 步（退火）加 M 步的 EM。
- **含义。** 每个多条目读的训练信号至少覆盖一半熵的其他顺序，包括模型自己偏好的起手。模型因此学会"从非标签的起手出发，也要把到期集合写完"，正对着"先写对方、漏掉插话"这一失败。
- **其余读完全不变。** $|B_r|\le1$ 的读（绝大多数）损失与 C0 逐位相同，所以改动只作用在多说话人同读的情形，对单人区和 ASR-BLEU 的风险小。
- **消融（不并行跑，论文用）：**
  - η = 0，即纯边际似然；
  - η = 1，即均匀顺序，类似 order-agnostic CE 的平均版。
- **推理。** v8 原样解码，加集合状态历史开关。不做 CRF 解码，顺序由模型自己决定。

### 13.4 是否端到端

| 项 | SerialCRF |
|---|---|
| CRF 在哪里 | 训练目标：LLM 自身概率构成的排列 CRF 的（退火）对数配分函数 |
| 梯度是否穿过 CRF | 是。经 logsumexp 和精确后验回传到 LoRA、投影器、新 token 行 |
| 教师 / 冻结副本 | 无（与 §2 的教师版不同） |
| 推理时有没有 CRF | 无。只加集合状态历史开关（重排已提交段），模型自己选顺序 |
| 架构改动 | 无新参数 |
| 是否波形级端到端 | 否。diarizer 与音频编码器冻结，同 v8 |

### 13.5 能解决什么，不能解决什么

- **能：** 重叠插话的"同读竞争→漏译"（占重叠短单元漏译的 36–51%）；顺序导致的漏写。
- **v2 才能：** 跨读补写。
- **不能：** 孤立插话的错标。主因是最终读的 P 掩码，由"最终读因果 P"另行修复，正在测试。
- **效果上限要诚实：**
  - dev 上 ≤1 s 的重叠漏译共 295 个，顺序相关约 105–150 个，最多能让 ≤1 s 单元的总召回提高约 4–5 个百分点；
  - 对 ov tcpBLEU 的提升会小于这个数。
- **风险：**
  - 集合状态历史是推理协议改动：标签路径上与训练一致，其他顺序下模型看到的是重排后的历史。
  - Talker 按写出顺序合成，顺序变化可能轻微改变语音的时间分布。
  - 不动点：靠退火 PR 处理，由预检 P1 量化。

### 13.6 预检（GPU，8 卡约 10–20 分钟；停止规则预登记）

- **范围。** 训练 500 窗口和 dev 300 窗口（与一′ 的预检相同）中 $|B_r|\ge2$ 的读。对所有 π，在集合状态历史下计算 $\psi_r(\pi)$。
- **P0 覆盖率。** dev 上"顺序→漏译"的重叠短单元，其应写读落在 $|B_r|\ge2$ 的读里的占比。< 30% 则**停止**，因为 v1 覆盖不到。
- **P1 不动点。** $q_\theta$ 在非标签顺序上的质量 ≥ 0.1 的读占比。同时报告 η = 0.5 时 $q^\*$ 在非标签顺序上的平均质量。只报告，不作停止条件。
- **P2 机制检验 H。** 设 f = 模型偏好的第一个条目。比较以下条件"完成概率"在 f ≠ 标签首项和 f = 标签首项时的中位数：

$$P_{\rm complete}(f)=\frac{\sum_{\pi:\pi_1=f}e^{\psi_r(\pi)}}{p_\theta(\text{首个条目}=f\mid x_{\le t_r},c,h_r)}$$

  若 f ≠ 标签首项时的中位数 ≥ 0.8，则 H 不成立（不是顺序暴露偏差），**停止**。
- **P3 历史开关本身的影响。** 集合状态历史开或关时，v8 在标签路径上的 ψ 应逐位相同（单元测试）；在模型自选顺序下的差异另报。

### 13.7 新颖性与最接近的工作（公开文献，写论文前要系统核查）

| 已有工作 | 做法 | 与本方案的区别 |
|---|---|---|
| SOT（Kanda et al., Interspeech 2020）；t-SOT（Kanda et al., 2022） | 固定 FIFO 或按时间戳的串行化 | 不对顺序建模，也不求边际 |
| PIT（Yu et al. 2017；Kolbæk et al. 2017；Seki et al. 2018 端到端多说话人 ASR） | 对排列取 **min**（硬分配） | 我们取精确的**和**（软），且是流式、LLM、带提交调度 |
| OAXE（Du, Tu & Jiang, ICML 2021） | 非自回归翻译，用匈牙利算法取最佳顺序 | 非流式、取 max |
| InDIGO（Gu, Liu & Cho, TACL 2019）；VOI（Li et al., ICLR 2021） | 隐生成顺序，近似搜索或变分推断 | 单序列、近似；我们靠集合状态历史得到**精确**可解 |
| DAEM（Ueda & Nakano 1998）；PR（Ganchev et al. 2010） | 通用工具 | 首次用于摆脱强 SFT 同传模型的序列化不动点 |

**新意：**
1. 指出并量化多说话人流式同传中的"序列化暴露偏差"：同读竞争导致插话永久漏译；
2. 用集合状态历史，使串行化顺序在流式语音 LLM 里能被精确求和，得到严格的排列 CRF，v2 再与提交调度组合成集合状态上的 semi-Markov CRF；
3. 退火后验正则化，摆脱 SFT 不动点（一′ 预检给出了不动点的实证）。

### 13.8 实验协议（单一预登记设置）

- 从 v8 续训，与 C0 同步数、同学习率、同数据（`train_crf_sft.py` 新增 `--objective serial`，增量实现，默认不变）。η = 0.5。
- dev 正式闸门：文本 all/ov，语音两个种子，最终读因果 P 的设置随测试结论统一。
- 目标指标（对 C0 和 v8）：重叠 ≤1 s 和 ≤0.5 s 召回、"顺序→漏译"率（检查 A 的脚本）、重复率、ov 延迟。
- 只有预检 P0、P2 都通过才实现训练代码；正式运行仍需用户命令。

### 13.9 预检后的修正（2026-11-09 20:30 UTC；`serial_crf_precheck.zh-CN.md`，7bd1cf93、88573d69）

- **κ 的排序键要改。**
  - 原文的 (ρ, a_u, 块序号) 有 6–7% 的读与 S 前缀不同，因为 S 在同一提交读内按块的**源终点**排序。
  - 改为 (ρ, 源终点, 源起点) 后，dev 和训练集的每一次读都与 S 前缀逐 token 一致。这个键同样只依赖集合，§13.2 的精确性结论不变。
- **预登记的 P2 有设计缺陷。** $P_{\rm complete}$ 包含剩余条目逐 token 的内容概率，标签起手时中位数也只有 $10^{-8}$–$10^{-5}$，0.8 的停止线永远达不到，没有判别力。这是设计时的错误，看到结果后才发现。
- **结果**
  - 按预登记规则判 GO：P0 = 73.5%；P1 显示顺序这一维没有不动点，非标签顺序的平均质量 0.25–0.27。
  - 但**事后**的结构诊断（不含内容概率，不作停止规则）**不支持机制 H**：
    - 写完首项后打开剩余到期条目的概率：非标签起手 0.83，标签起手 0.82；
    - 2 条目读的结构完成概率：dev 0.70 对 0.73，训练集 0.45 对 0.43。
  - 只有 dev 上含 ≤1 s 重叠单元的读有差距（0.29 对 0.57，n = 15 / 25），训练集不复现。
- **结论。** 先写哪一段，不影响模型继续写完剩余的到期段。所以同读竞争导致漏译的原因**不是顺序**。更可能是：同读有多段到期时，模型无论按什么顺序都常常只写一段就 `<wait>`（完成概率约 0.7）；之后进入"有到期段未写"的状态，而这种状态在训练标签里从来没有（标签从不延迟），于是不再补写。这是**跨读的补写暴露偏差**，有待检验。
- **处理。** 暂不实现 SerialCRF 的训练代码。如果要继续：先用约 10 分钟 8 卡做"补写概率"诊断，即在第 r+1 读、历史中缺一个已到期条目时，模型打开它的概率。若补写概率低，对应的方案是 §13.2 的 v2（集合状态上的调度 CRF，加退火 PR），或 DAgger 式补写训练行。

### 13.10 v2 预检的预登记（2026-11-09 21:40 UTC，在任何 v2 结果之前写下）

**v2 的格（预检版）**
- 条目 = 一′/SerialCRF 中的块。同一说话人的条目保持原序。
- 提交读允许**松弛 1 读**：$\rho(u)\in\{\rho_{\rm lab}(u),\ \rho_{\rm lab}(u)+1\}$，最终读的条目不能推迟。
- 状态是"已提交集合" A。历史 $h_r=\kappa(A)$，κ 用修正后的键 (ρ_lab, 源终点, 源起点)。读内按 κ 的规范顺序写出（SerialCRF 已表明读内顺序不影响写完）。
- 势函数：$\psi_r(A'\!\to\!A)=\log p_\theta\big(\kappa\text{-序的 }(A\setminus A')\oplus e_r\mid x_{\le t_r},c,\kappa(A')\big)$。$\log Z$ 用集合状态上的前向递推精确计算，并在小窗口上与暴力枚举对拍。
- 打分协议用修正后的基线：最终读提供完整后验（训练窗口没有关闭检测时刻，一律给整窗 P）。其余与以前的预检相同：500 个训练窗口、300 个 dev 窗口，dev 为主要证据。
- 退火：$q_\tau\propto p_\theta^{\tau}$，预登记 τ = 0.5，同时报告 τ = 1。

**停止规则（只用结构概率，或解码事实）**
- **S1 覆盖率（CPU，基于 v8 正式参考解码）。** dev 漏掉的重叠 ≤1 s 单元中，"应写读不是最终读，且该读写了别的条目却没写它"（即发生了推迟）的占比。< 30% → **停止**。
- **S2 信号量。** τ = 0.5 时，在多条目读的条目上，"推迟 1 读后补写"路径的后验质量均值（dev）。< 0.05 → **停止**：v2 约等于 C0，没有训练信号。
- **S3 精确性。** 标签路径上的 κ 历史与 S 前缀逐 token 一致；格的 log Z 与暴力枚举一致。不通过 → 先修代码，不能上 GPU。
- **只报告：**
  - 推迟状态下，下一读 m 排在规范顺序首位时的结构开段概率（只看说话人 token，以及说话人 + 时间 token）；
  - τ = 1 时的推迟-补写质量；
  - 新协议与旧协议下 S2 的差别（如果顺便算了）。

### 13.11 v2′：在 v2 的格里加入读内顺序隐变量（2026-11-10 04:50 UTC 预登记；`serial_crf_v2p_precheck.zh-CN.md`）

**动机。** v2 的读内顺序固定为规范顺序，被推迟的 m 在下一读总排第一，这正是 v8 已经会补写的情形。在线推断出的失败链是"下一读先写新到期的条目，再在 m 之前 `<wait>`"（`serial_crf_precheck.zh-CN.md` §12.5–§12.6）。"新条目已写、m 仍待写"这种上下文不在 v2 的格里，所以 v2 训练学不到"写完新条目还要补写 m"。

**模型（其余与 §13.10 的 v2 完全相同）。**
- 每读一个二值隐变量 $o_r\in\{\text{canon},\text{newfirst}\}$。
  - canon 是 v2 的顺序。
  - newfirst 先写 $N_r$：$D_r$ 中说话人与任何待补写条目都不同的条目，按 kap 序。然后按规范顺序写其余条目。每个说话人内的标签顺序不变。
  - 只有待补写集合与 $N_r$ 都非空时才是真选择，否则只算一次。
- 边势函数：

$$\psi_r(D_{r-1}\!\to\!D_r)=\log\sum_{o}p_\theta\big(W_r^{o}\oplus e_r\mid x_{\le t_r},\kappa(A_{r-1})\big)$$

- 退火在联合隐变量 (ρ, o) 上：$q_\tau\propto\exp(\tau\sum_r\psi_{r,o_r})$。前向的边项为 $\log\sum_o e^{\tau\psi_{r,o}}$，τ = 1 时就是上式。
- **精确性。** κ 只依赖集合，o_r 只进入第 r 读自己的项，前向递推不变，只是每条边多一个 logsumexp。
- **合法性。** 打分器给一读内所有条目打同一个写出时刻，评测对读内顺序不变。
- **标签路径。** 所有 D_r = B_r 且 o_r = canon，仍然逐 token 等于 S（v8 / C0 的目标）。

**含义。**
- 后验可以把质量放到"先写新条目、再补写 m"的路径上，训练行于是包含"写完 N_r 之后接着打开 m"这一步。这正是在线失败链断掉的地方。
- 如果 teacher forcing 下 N_r 之后的 p(`<wait>`) 本来就很小（S2′b < 0.02），说明丢弃只在自由生成的状态分布下出现，格训练够不着，应改用 on-policy 方法（DAgger 式补写行，或在自由生成的轨迹上做调度后验）。

**预检（预登记的停止规则）。**
- S3′：标签路径 = S；格与 (ρ × 顺序) 暴力枚举一致；反例必须被抓到。
- S1′：在线 CHAIN 占 ≤1 s ov50 漏译的比例，< 0.15 → 停止。
- S2′a：有真选择的读上 q_{0.5}(newfirst) 的均值，< 0.05 → 停止。
- S2′b：newfirst 上下文中写完 N_r 后 teacher forcing 的 p(e_r) 均值，< 0.02 → 停止。

结果见预检报告。
