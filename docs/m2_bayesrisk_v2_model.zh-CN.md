# 同传 Bayesian 提交策略 v2：为什么 v1 没有超过 v3，以及严格的改进模型

前置文档：
- 模型 v1：[m2_test_time_bayes_model.zh-CN.md](m2_test_time_bayes_model.zh-CN.md)
- 结果 v1：[m2_bayesrisk_results_v1.zh-CN.md](m2_bayesrisk_results_v1.zh-CN.md)
- 基线 v3：[m2_fluency_bayes_v3cfg.zh-CN.md](m2_fluency_bayes_v3cfg.zh-CN.md)

记号沿用模型 v1：
- 第 $k$ 次读取的草稿为 $\mathbf d_k$，长度为 $K_k$；
- 稳定长度为 $S_k$；
- 统计量为 $\mathbf T_k$；
- 播放状态为 $\xi_k=(q_k,\beta_k,g_k)$；
- 动作 $a$ 表示提交草稿的前 $a$ 个单位。

## 0. 结论

**v1 没有超过 v3，原因有四个。前两个是结构性的，决定了 v1 的上限。**

1. **信息几乎没有增加。** 拟合后验大幅改善了校准（ECE 0.069→0.006），但区分能力只提高了 AUC +0.01～0.014。
   - 提交规则本质上是对存活概率取阈值，而 v3 扫描 bias 时已经走遍了整条阈值曲线。
   - 只改校准、不改排序，策略族就和 v3 几乎相同（§1 命题 1）。
   - 所以 en→zh 的 BLEU 打平。
2. **损失函数与评价指标不一致，而且决策是短视的。** v1 的损失只计“这次提交之后的下一个窗口”，看不到一个后果：把确定的那一段说出去之后，如果切点落在短语中间，后面要等的那段就成了一次突兀停顿（§1 命题 2）。
   - v1 又把 v3 的提交卫生从硬约束放松成了有限代价。
   - 于是 Bayes 动作在损失中的盲区里变得最优：短语中间的切点从 25% 升到 40–44%，突兀停顿变成 v3 的 1.4–3 倍。
3. **按场自适应可提升的空间很小。** 扣除种子噪声后，场级 oracle 的净上限只有 +0.35 / +0.5 BLEU。
   - en→zh 的场间差异 $\hat\Psi$ 只有 0.01，在线后验几乎不动；
   - 同一句多次读取的离散度高达 6–7，每场的有效样本很少。
4. **dev 上选配置时看不到这类问题。** dev 的时间轴只统计 ≥1 s 的空隙，统计不到单字块和短语中间的停顿，因此 dev 选出了偏碎片化的配置。

**v2 的四处改动：**
- 提交卫生恢复为硬约束；
- 损失中加入**切点条件下的停顿代价**，并用“下一次提交时间”的 hazard 模型做两步前瞻；
- 用 Lagrange 对偶把“突兀停顿不超过 v3 的 κ 倍”写成**约束**，而不是偏好；
- 只有当多草稿一致性这类统计量在 dev 上**确实提升 AUC** 时，才把它加入信息集。

可以证明 v3 和 v1 都是 v2 的特例（§2.8），所以在 dev 上 v2 的最优成员至少不差于两者。RealSI 只测一次。

## 1. 为什么 v1 没有超过 v3

### 1.1 命题 1：只改校准，不会扩大可达的前沿

设按单位计的损失（模型 v1 §11）为
$$
R(a)=C_\mathrm{err}\,\mathbb E[(a-S)^+]+C_\mathrm{lat}D\,\mathbb E[(\min(S,K)-a)^+],
$$
它的 Bayes 动作是对边际存活概率取阈值：
$$
a^\star=\max\{a:\ \hat P(S\ge a)>\tau(\xi)\},\qquad \tau(\xi)=\frac{1}{1+C_\mathrm{lat}D(\xi)} .
$$

**命题 1。** 设拟合后验是 ABC 存活概率的严格单调变换，即对所有 $a$ 有 $\hat P(S\ge a)=g\big(P_\mathrm{ABC}(S\ge a)\big)$，且 $g$ 严格递增。那么
$$
\hat P(S\ge a)>\tau(\xi)\iff P_\mathrm{ABC}(S\ge a)>g^{-1}(\tau(\xi)),
$$
所以 v1 的每个策略都对应 ABC 存活概率上的某条阈值曲线 $\tilde\tau(\xi)=g^{-1}(\tau(\xi))$。

v3 扫描 bias $b$ 得到的是阈值曲线族 $\{\tau_b(\xi)\}$，它同样是 $D(\xi)$ 的单调函数（SDR）。两者只在“阈值随播放状态如何变化”上有差别，在“哪个单位更可能被改”的排序上完全相同。

**推论。** 在同一 ROC 曲线上，延迟–错误前沿的改善只能来自两个途径：
- 排序变好，即 ROC/AUC 提升；
- 阈值随状态变化的方式更好。

v1 的拟合后验并非严格的单调变换（它是 8 个统计量的 logistic 组合），但 AUC 只提高了 0.01–0.014，说明它与单调变换相差很小。这正好解释了 en→zh 的 BLEU 为什么打平。

zh→en PO 的 +1.22 来自在线后验把每场的 bias 下调（1.14 → 0.84–1.28），即第二个途径“阈值更好”，而不是排序变好（结果 v1 §3.4）。

### 1.2 命题 2：短视损失看不到切点之后的停顿

模型 v1 的静音项只计算下一个窗口里的静音 $\varsigma_k(a)=\max(0,\Delta-p_k-\mathrm{dur}(a))$。这里假设下一次读取时一定能接着说（A5）。

实际情况是：提交到切点 $a$ 之后，听众能听到的下一段语音，要等**策略下一次提交新内容**的时刻 $\tau_k(a)$ 才开始。真实的停顿长度是
$$
G_k(a)=\max\big(0,\ (\tau_k(a)-k)\Delta-p_k-\mathrm{dur}(a)\big).
$$

**命题 2。** 设 $\mathrm{seam}(a)$ 是切点类型，取值为：词内 W、虚词后 P-fw、短语中间 M、小句边界 C、句末 E。听感代价是
$$
\mathbf 1[\mathrm{seam}(a)\in\{W,\text{P-fw},M\}]\ \varphi(G_k(a)) .
$$
它依赖 $\tau_k(a)$ 的分布，而 v1 的 $R_k(a)$ 对任何 $a$ 都把这一项当作 0，或者只计入 $\varsigma_k(a)$。

所以只要一个短语中间的切点在内容项上更优，v1 就会选它，不论之后的停顿有多长。当碎块代价有限时（zh→en P 配置为 0，A 配置为 0.5），这正是 RealSI 上观察到的现象：
- 短语中间切点从 25% 升到 40–44%；
- “草稿已有、被规则扣下”的停顿从 28% 升到 46%。

### 1.3 场级后验：可提升的空间很小

- 场级 oracle（每场挑最好的 bias）扣除种子噪声后，净上限只有 +0.35 / +0.5 BLEU；逐句 oracle 有 +2.6 / +3.1。
- 读取之间的离散度为 6–7，每句的有效读取数大约只有名义值的 1/6。
- en→zh 的 $\hat\Psi=0.01$，在线后验几乎不动。

因此可提升的空间主要在**读取层面**，即决定在哪里切、等不等，而不在场级参数上。

### 1.4 dev 上选配置看不到问题

dev 的模拟时间轴只统计 ≥1 s 的空隙。v1 真正变差的指标（单字块、短语中间停顿、突兀停顿）在 dev 上完全不可见，于是代价参数朝着碎片化的方向被选了出来。

## 2. v2 的严格模型

### 2.1 随机变量（在 v1 基础上新增）

| 符号 | 含义 |
|---|---|
| $\mathrm{seam}(a)\in\{W,\text{P-fw},M,C,E\}$ | 提交到 $a$ 时的切点类型，由草稿文本和标点确定，是已知量 |
| $\tau_k(a)$ | 在策略 $\pi$ 下，提交到 $a$ 之后**下一次提交新内容**的读取序号；隐变量，依赖 $\pi$ |
| $J_k(a)=\tau_k(a)-k\in\{1,\dots,J_\max\}$ | 下一次提交前要等的读取次数 |
| $\sigma_k\in\{0,1\}$ | 源语当前是否在说话（VAD）；突兀停顿只在源语说话时计 |

### 2.2 动作集合（硬约束）

$$
\mathcal A_k=\Big\{a\in\{0\}\cup\mathcal B_k:\ a=0\ \text{或}\ n(a)\ge m\ \text{或}\ \mathrm{seam}(a)\in\{C,E\}\ \text{或}\ k=k_\mathrm{deadline}\Big\}.
$$
- $\mathcal B_k$ 是 token 边界集合（v1 §11 修正 2）。
- $n(a)\ge m$ 就是 v3 的提交卫生（至少 3 字或 4 词）。这对应模型 v1 命题 2 中 $C_\mathrm{frag}\to\infty$ 的极限。
- 小句边界、句末和 deadline 是例外，允许提交短块。

### 2.3 观测模型

**存活链（不变）：**
$$
P(S_k\ge u{+}1\mid S_k\ge u,\mathbf T_{k,u},\theta_s)=\sigma(\theta_s^\top\mathbf T_{k,u}) .
$$

**下一次提交时间：离散时间 hazard 模型。** 对 $j=1,\dots,J_\max$，
$$
P\big(J_k(a)=j\mid J_k(a)\ge j,\ \mathbf x_{k,a,j}\big)=\sigma(\eta^\top\mathbf x_{k,a,j}),
\qquad
P(J_k(a)=j)=\sigma(\eta^\top\mathbf x_{k,a,j})\prod_{i<j}\big(1-\sigma(\eta^\top\mathbf x_{k,a,i})\big).
$$
- 特征 $\mathbf x_{k,a,j}$ 包括：
  - 第 $a{+}1$ 个单位的存活 logit；
  - 被扣下单位的个数 $K_k-a$；
  - 切点类型；
  - 源语是否在说话 $\sigma_k$；
  - 距离上一个源语停顿的时长；
  - 步数 $j$ 的独热编码。
- 截断：$J_\max$ 处剩余的概率质量全部放到 $J_\max$。

**拟合方式。** $\tau$ 依赖策略本身，所以采用策略迭代：
1. 用当前策略在 dev 上跑一遍，记录每次提交后的 $J$；
2. 用 logistic 极大似然拟合 $\eta$（同句的离散度用 v1 §11 第 4 点的方式修正）；
3. 用新的 $\eta$ 更新策略，重复 2–3 轮，直到 dev 上 $J$ 的分布稳定（KL < 0.01）。

这和 fitted policy evaluation 是同一个结构。不动点的存在性不做保证，用收敛与否做经验检查。

### 2.4 损失

$$
\ell(a;S_k,J_k,\xi_k)=
\underbrace{C_\mathrm{err}(a-S_k)^+ + C_\mathrm{lat}D_k(\min(S_k,K_k)-a)^+}_{\text{内容：按单位计（v1 §11）}}
+\underbrace{C_\mathrm{sil}\,\varphi\big(G_k(a)\big)}_{\text{听众静音段，按段计}}
+\underbrace{C_\mathrm{seam}(\mathrm{seam}(a))\ \sigma_k\ \mathbf 1\big[G_k(a)\ge g_0\big]}_{\text{突兀停顿：切点差且确实停了}}
+C_\mathrm{talk}\pi_T(a).
$$
- $G_k(a)=\max\big(0,J_k(a)\Delta-p_k-\mathrm{dur}(a)\big)$，其中 $\varphi$ 和 $g_0$ 与 v1 相同。
- $C_\mathrm{seam}(C)=C_\mathrm{seam}(E)=0$，$C_\mathrm{seam}(M)>0$，$C_\mathrm{seam}(W)=C_\mathrm{seam}(\text{P-fw})=\infty$（后两者已由动作集合排除）。
- **关键区别：** 切点差本身不计代价，只有“切点差并且之后真的停了 ≥ $g_0$”才计。如果后续内容能立即接上，在短语中间切开并不影响听感。这一项正是 RealSI 上“突兀停顿”指标的期望值。

**后验期望损失。** $S_k$ 与 $J_k$ 在给定统计量的条件下取为独立（**A8**，§4 给出检验方法），于是
$$
R_k(a)=R^\mathrm{cont}_k(a)+\sum_{j=1}^{J_\max}P(J_k(a)=j)\Big[C_\mathrm{sil}\varphi\big(G_{k,j}(a)\big)+C_\mathrm{seam}(\mathrm{seam}(a))\,\sigma_k\,\mathbf 1[G_{k,j}(a)\ge g_0]\Big]+C_\mathrm{talk}\pi_T(a),
$$
其中 $R^\mathrm{cont}_k(a)=C_\mathrm{err}\mathbb E[(a-S)^+]+C_\mathrm{lat}D_k\mathbb E[(\min(S,K)-a)^+]$，与 v1 相同。

### 2.5 把流畅度门槛写成约束（Lagrange 对偶）

偏好比 $C_\mathrm{seam}/C_\mathrm{err}$ 本身不由数据决定。我们真正的要求是一个门槛：**内容损失尽量小，同时突兀停顿率不超过 v3 的 $\kappa$ 倍**（$\kappa=1.2$）。写成约束优化：
$$
\min_\pi\ \mathbb E_\pi\Big[\sum_k \ell^\mathrm{cont}_k\Big]\quad\text{s.t.}\quad
\mathbb E_\pi[\text{突兀停顿数}]\le\kappa\,\mathbb E_{\pi_\mathrm{v3}}[\text{突兀停顿数}] .
$$
Lagrange 函数为
$$
\mathcal L(\pi,\lambda)=\mathbb E_\pi\Big[\sum_k\ell^\mathrm{cont}_k+\lambda\,\mathbf 1[\text{突兀停顿}_k]\Big]-\lambda\,\kappa\,\bar A_\mathrm{v3}.
$$

对固定的 $\lambda$，逐次读取的 Bayes 动作就是 §2.4 中取 $C_\mathrm{seam}(M)=\lambda$ 时的 $\arg\min_a R_k(a)$。$\lambda$ 由 dev 上的对偶上升确定：
$$
\lambda\leftarrow\Big[\lambda+\gamma\big(\hat A_\pi^\mathrm{dev}-\kappa\hat A_\mathrm{v3}^\mathrm{dev}\big)\Big]_+ .
$$

这样 $C_\mathrm{seam}$ 就从“拍脑袋定的偏好”变成了“满足门槛所需的影子价格”，可以写进论文，而且可复现。

### 2.6 两步前瞻（取代 A5）

一步短视的 $\arg\min$ 改为带近似后续价值的滚动时域：
$$
a_k^\star=\arg\min_{a\in\mathcal A_k}\Big\{R_k(a)+\mathbb E\big[\hat V_{k+J}\mid a\big]\Big\},
\qquad
\hat V_{k+J}\approx C_\mathrm{lat}\,\mathbb E\big[D_{k+J}\big]\cdot\mathbb E\big[(K_k-a)\big],
$$
也就是“被扣下的单位在下次提交前还要继续积累滞后”。

如果第一版实现中 $\hat V$ 的作用可以忽略（与一步决策在 dev 上的一致率 > 98%），就保留一步决策。这时 §2.4 的 $J$ 分布已经承担了“看得见后果”的作用。

### 2.7 增加信息：只接受真正提升排序的统计量

由命题 1，只有提升 AUC 的统计量才可能推动前沿。候选统计量：
- **多草稿一致性**：用温度采样得到 $R$ 条草稿 $\mathbf d^{(r)}_k$，计算
  $$e^{(4)}_{k,u}=\frac1R\sum_r\mathbf 1\big[\mathrm{lcp}(\mathbf d_k,\mathbf d^{(r)}_k)>u\big].$$
  它是 Thinker 自身预测分布下 $P(S_k>u)$ 的 Monte Carlo 估计。

  注意（手册 T5.4 的结论同样适用）：额外采样**不是**对真实未来音频的观测，$I(S_k;\mathbf d^{(r)}\mid o_k)$ 只来自模型自身的不确定性。它能降低单条贪心草稿带来的估计误差，但不会带来新的外部信息。
- 草稿第 $u$ 个位置的 top-1 / top-2 概率差，以及熵。

**接受门槛（dev，2 折交叉验证）：** AUC 至少提高 0.02，并且 NLL 下降。达不到门槛就不加入，同时停止在这个方向（例如 en→zh）上继续投入。采样 $R$ 条草稿的计算量要计入延迟。

### 2.8 v3 和 v1 都是 v2 的特例

- **v3：** 取 $\theta_s\equiv m_0$（ABC），$C_\mathrm{sil}=0$，$\lambda=0$，动作集合用 §2.2 的硬约束，并用 SDR 的 $D$。
- **v1：** 取 §2.2 中 $m=1$（不加卫生约束），$J_k\equiv1$（短视），$\lambda=C_\mathrm{seam}$ 取有限常数。

v2 的参数空间包含这两点，所以 dev 上的最优 v2 成员在 dev 目标下至少不差于两者（这是样本内结论，测试集仍要单独检验）。

## 3. 执行方案

全部只用训练域 dev（UniST 805 句）选择和拟合；RealSI 只测一次；ACL 不用。

| 步骤 | 内容 | 产物 | 门槛（不过门槛就停） | 大致成本 |
|---|---|---|---|---|
| 0 | **dev 时间轴补上听感指标**：用 SQ Talker 的实测时长表和 s1b1.2q 播放规则，在 dev 上模拟连续时间轴，输出单字块、切点类型分布、突兀停顿/分钟 | `si_bayesrisk_dev_eval.py --timeline full` | 在 dev 上，v3 与 v1 PO 的突兀停顿比值，与 RealSI 上同向且大于 1.5 倍（复现盲区） | 只需 CPU，几小时 |
| 1 | **硬卫生约束 + 切点条件停顿代价**：实现 §2.2、§2.4，以及 $J$ 的 hazard 模型和 2–3 轮策略迭代 | `si_bayesrisk.py --br-v2`，`fit_next_commit_hazard.py` | dev 上 hazard 模型的 AUC > 0.7；突兀停顿 ≤ v3 的 1.2 倍 | 每轮 dev 解码约 1 小时（8 卡） |
| 2 | **约束 Bayes**：按 §2.5 的对偶上升确定 $\lambda$，再扫 bias 得到延迟–BLEU 曲线 | `lambda_dual.json`、dev 前沿 | 同等延迟下 dev BLEU ≥ v3，且突兀停顿 ≤ 1.2 倍 | 约 5 轮 × 1 小时 |
| 3 | **信息增益（可选）**：多草稿一致性等统计量，按 §2.7 的门槛决定是否接受 | dev AUC/NLL 表 | AUC 至少 +0.02；未达到就跳过这一步 | 草稿采样 R=4，读取开销约 ×4 |
| 4 | **两步前瞻（可选）**：§2.6 | 与一步决策的一致率 | 一致率 < 98% 且 dev 前沿改善时才保留 | 很小 |
| 5 | **RealSI 测一次**：zh→en 优先（v1 PO 已有 +1.22 的信号）；en→zh 只有步骤 3 通过才测 | 与 v3 曲线的配对 bootstrap、`matched.md`、突兀停顿 | **同等 LAAL 下 ASR-BLEU ≥ v3（配对 CI 下界 > −0.3），且突兀停顿 ≤ v3 的 1.2 倍，且 UTMOS 不显著下降** | 约 2 小时 |

**预期。**
- zh→en：v1 PO 的增益来自“扣得更准”（在线后验下调了 bias），碎片化来自放松了卫生约束。§2.2 和 §2.4 把两者分开处理，目标是在保留约 +0.5～+1.0 BLEU 的同时，把突兀停顿压回 v3 附近。
- en→zh：$\hat\Psi\approx0$，而且 AUC 增益很小。除非步骤 3 通过，否则预期只能与 v3 打平，不再继续投入。

## 4. 新增近似与检验

| 编号 | 近似 | 检验方式 |
|---|---|---|
| A8 | 给定统计量时 $S_k$ 与 $J_k$ 条件独立 | 在 dev 上比较 $P(J\mid S\ \text{的分位数})$ 的分层差异；若差异明显，就把 $\mathbb E[(a-S)^+]$ 的分位数加入 hazard 特征 |
| A9 | $\tau$ 依赖策略，用策略迭代求不动点 | 观察 $J$ 分布逐轮的 KL；不收敛时取最后两轮的平均 |
| A10 | Talker 时长使用实测表（替代 A6 的均值） | 与 RealSI 上实测的 dur 比较，误差的 p90 < 0.3 s |
| A11 | 对偶上升在 dev 上收敛 | 约束违约量与 $\lambda$ 的轨迹单调收敛 |

## 5. 严格的先验、似然与后验（v2 完整概率模型）

本节把 §2–§3 中需要从数据推断的部分，全部写成“先验 × 似然 → 后验 → 后验预测”的形式。不是推断问题的部分，在 §5.8 单独列出。

### 5.1 数据与参数

**数据（全部来自训练域 dev；测试时只用自标注，见 §5.6）：**

| 数据 | 内容 |
|---|---|
| $\mathcal D_S$ | 每次读取 $(n,k)$ 的统计量 $\{\mathbf T_{k,u}\}_{u<K_k}$ 和稳定长度 $S_k$。$S_k$ 用 Thinker 听完整句后的续写自动得到，不需要参考译文 |
| $\mathcal D_J$ | 在策略 $\pi_\psi$ 下，每次提交 $(k,a)$ 的协变量 $\mathbf x_{k,a}$、观测到的等待次数 $J^\mathrm{obs}$ 和删失标记 $c\in\{0,1\}$（$c=1$ 表示句子在下一次提交前结束） |
| $\mathcal D_r$ | 每个提交块的 Talker 实测时长 $r$ 和块特征 $\mathbf z$（字数、末字类型、语速档位） |

**参数：**

| 参数 | 含义 | 维度 |
|---|---|---|
| $\omega$；$\delta_s$；$b_n$ | 存活链：群体系数；场 $s$ 的偏差；句 $n$ 的随机截距 | $p$；$p$；1 |
| $\gamma\in\{0,1\}^{p'}$ | 候选新统计量（§2.7）是否进入模型 | $p'$ |
| $\nu$；$\epsilon_s$；$c_n$ | 下一次提交的 hazard：群体系数；场偏差；句随机截距 | $q$；$q$；1 |
| $\beta,\sigma_r^2$ | Talker 时长的对数线性模型 | $d$；1 |
| $\Psi,\Phi,\tau_b^2,\tau_c^2$ | 方差超参数 | — |

### 5.2 先验

各块之间相互独立：
$$
p(\Theta)=p(\omega,\gamma)\,\prod_s\mathcal N(\delta_s;0,\Psi)\prod_n\mathcal N(b_n;0,\tau_b^2)\;\cdot\;\mathcal N(\nu;n_0,V_0)\prod_s\mathcal N(\epsilon_s;0,\Phi)\prod_n\mathcal N(c_n;0,\tau_c^2)\;\cdot\;p(\beta,\sigma_r^2).
$$

**1. 存活链：群体系数，对新统计量用 spike-and-slab 先验。** 把 $\omega$ 分成已有的 $p$ 维 $\omega_\mathrm{base}$ 和候选的 $p'$ 维 $\omega_\mathrm{new}$：
$$
\omega_\mathrm{base}\sim\mathcal N(m_0,S_0),\quad m_0=(1,0,\dots,0)\ \ (\text{ABC logit 的系数为 1，即先验均值就是 ABC}),
$$
$$
\omega_{\mathrm{new},j}\mid\gamma_j\sim\gamma_j\,\mathcal N(0,s^2)+(1-\gamma_j)\,\delta_0,\qquad \gamma_j\sim\mathrm{Bernoulli}(\tfrac12).
$$

**2. 下一次提交的 hazard。** $n_0$ 取“常数 hazard 等于 dev 上 v3 的平均提交率”对应的 logit，其余分量取 0；$V_0=\mathrm{diag}(2.5^2)$。

**3. Talker 时长：共轭的 Normal–Inverse-Gamma 先验。**
$$
\sigma_r^2\sim\mathrm{IG}(a_0,b_0),\qquad \beta\mid\sigma_r^2\sim\mathcal N(\beta_0,\sigma_r^2\Lambda_0^{-1}).
$$

**4. 方差超参数。** 用 type-II 极大似然（经验 Bayes，即 v1 的 A3）。也可以改为全 Bayes：给每个标准差一个 half-Cauchy(0, 1) 先验，再用 NUTS 采样。两种做法的差异在 §4 的 A2 检查中报告。

### 5.3 似然

**(a) 存活链。** 用句随机截距 $b_n$ 取代 v1 里的“离散度修正”：v1 §11 第 4 点那种做法属于 generalized Bayes，而这里是严格的层级模型。
$$
P(S_k\ge u{+}1\mid S_k\ge u,\ \mathbf T_{k,u},\theta_s,b_n)=\sigma\big(\theta_s^\top\mathbf T_{k,u}+b_n\big),\qquad \theta_s=\omega+\delta_s,
$$
$$
\mathcal L^S_k(\theta_s,b_n)=\Big[1-\sigma(\theta_s^\top\mathbf T_{k,S_k}+b_n)\Big]^{\mathbf 1[S_k<K_k]}\prod_{u<S_k}\sigma(\theta_s^\top\mathbf T_{k,u}+b_n).
$$
句 $n$ 的边际似然把 $b_n$ 积掉（一维积分，用 Gauss–Hermite 求积，节点数增加时收敛到精确值）：
$$
\mathcal L^S_n(\theta_s)=\int\prod_{k\in n}\mathcal L^S_k(\theta_s,b)\ \mathcal N(b;0,\tau_b^2)\,db .
$$
同一句的各次读取共用 $b_n$，因此在边际上正相关。这正是 v1 中离散度 6–7 的来源，现在它由模型本身表达，而不是事后修正。

**(b) 下一次提交时间：带右删失的离散时间生存似然。** 协变量全部在**提交时刻 $k$ 已知**：
$$
\mathbf x_{k,a,j}=\big(\mathbf f(\mathbf T_k,a,\mathrm{seam}(a),\sigma_k,\xi_k,\psi),\ \mathbf e_j\big),
$$
其中 $\mathbf e_j$ 是步数 $j$ 的独热编码，$\psi$ 是当前策略的参数（阈值、$\lambda$），不含任何未来的统计量。设 $h_j=\sigma(\theta^{J\top}_s\mathbf x_{k,a,j}+c_n)$，$\theta^J_s=\nu+\epsilon_s$，则
$$
\mathcal L^J_{k,a}(\theta^J_s,c_n)=\Big[h_{J^\mathrm{obs}}\Big]^{1-c}\prod_{i<J^\mathrm{obs}}(1-h_i)\cdot\Big[1-h_{J^\mathrm{obs}}\Big]^{c}.
$$
- $c=1$ 时，最后一个因子表示“到句末仍未提交”的删失；
- 句级的 $c_n$ 与 (a) 一样用求积积掉。

**为什么这个似然可以用来学一个依赖策略的量。** $J$ 依赖于策略，但策略 $\pi_\psi$ 的每个动作只是观测量（$\mathbf T,\xi$）和已知参数 $\psi$ 的确定函数，而且 $\psi$ 已经放进了协变量。因此动作的选择机制是可忽略的（sequential ignorability）：条件似然 $p(J\mid\mathbf x,\theta^J)$ 不需要对策略的选择机制建模，就可以直接用于推断。

策略迭代（§2.3）的每一轮，都在新的 $\psi$ 下补充数据。用旧轮数据预测新策略下的 $J$，靠的是协变量 $\psi$ 外推；外推范围只限于 dev 上实际出现过的 $\psi$ 区间。

**(c) Talker 时长。**
$$
\log r\mid\mathbf z,\beta,\sigma_r^2\sim\mathcal N(\mathbf z^\top\beta,\sigma_r^2).
$$

**A8（条件独立）的精确含义：** 给定 $(\mathbf T_k,\mathbf x_{k,a},\mathbf z)$ 和参数，$S_k$、$J_k(a)$、$r$ 三者条件独立。§4 中的检验针对的就是这一条。

### 5.4 后验

先验各块独立，而三类似然各自只含互不重叠的参数，所以**联合后验精确地分解为三块**：
$$
p(\Theta\mid\mathcal D)=
\underbrace{p(\omega,\gamma,\{\delta_s\}\mid\mathcal D_S)}_{\text{存活链}}\;
\underbrace{p(\nu,\{\epsilon_s\}\mid\mathcal D_J)}_{\text{下一次提交}}\;
\underbrace{p(\beta,\sigma_r^2\mid\mathcal D_r)}_{\text{Talker 时长}},
$$
其中
$$
p(\omega,\gamma,\{\delta_s\}\mid\mathcal D_S)\propto p(\omega,\gamma)\prod_s\Big[\mathcal N(\delta_s;0,\Psi)\prod_{n\in s}\mathcal L^S_n(\omega+\delta_s)\Big],
$$
$$
p(\nu,\{\epsilon_s\}\mid\mathcal D_J)\propto \mathcal N(\nu;n_0,V_0)\prod_s\Big[\mathcal N(\epsilon_s;0,\Phi)\prod_{n\in s}\int\prod_{(k,a)\in n}\mathcal L^J_{k,a}(\nu+\epsilon_s,c)\,\mathcal N(c;0,\tau_c^2)\,dc\Big].
$$

**Talker 时长有闭式后验（共轭）。** 设设计矩阵为 $Z$，$y=\log r$：
$$
\Lambda_N=\Lambda_0+Z^\top Z,\quad \beta_N=\Lambda_N^{-1}(\Lambda_0\beta_0+Z^\top y),\quad a_N=a_0+\tfrac{N}{2},\quad b_N=b_0+\tfrac12\big(y^\top y+\beta_0^\top\Lambda_0\beta_0-\beta_N^\top\Lambda_N\beta_N\big).
$$

**前两块没有闭式解：**
- 用 Laplace 近似（A2），并在小样本上与 NUTS 对比；
- spike-and-slab 部分对 $2^{p'}$ 个模型（$p'\le 3$）逐一做 Laplace 近似，得到边际似然 $p(\mathcal D_S\mid\gamma)$，然后
  $$
  P(\gamma\mid\mathcal D_S)=\frac{p(\mathcal D_S\mid\gamma)\,p(\gamma)}{\sum_{\gamma'}p(\mathcal D_S\mid\gamma')\,p(\gamma')},\qquad
  P(\gamma_j{=}1\mid\mathcal D_S)=\sum_{\gamma:\gamma_j=1}P(\gamma\mid\mathcal D_S).
  $$
  这就把 §2.7 的“AUC 门槛”换成了严格的 Bayes 模型选择：后验包含概率大于 0.5 才保留该统计量。AUC 仍然报告，但只作为辅助检查。

### 5.5 后验预测（决策需要的三个量）

对测试场 $s$，先验取 $\theta_s\sim\mathcal N(\mu_S,\Sigma_S+\hat\Psi)$、$\theta^J_s\sim\mathcal N(\mu_J,\Sigma_J+\hat\Phi)$。当前句还没有自标注，所以 $b_n,c_n$ 按它们的先验积分。

**存活概率：**
$$
\hat P_k(S_k\ge a)=\sum_\gamma P(\gamma\mid\mathcal D_S)\,\mathbb E_{\theta_s,b}\Big[\prod_{u<a}\sigma(\theta_s^\top\mathbf T_{k,u}+b)\Big].
$$

**下一次提交的等待次数：**
$$
\hat P_k(J=j\mid a)=\mathbb E_{\theta^J_s,c}\Big[h_j\prod_{i<j}(1-h_i)\Big].
$$

**Talker 时长：** 后验预测是 Student-t 分布，
$$
\log\mathrm{dur}(a)\sim t_{2a_N}\Big(\mathbf z_a^\top\beta_N,\ \tfrac{b_N}{a_N}\big(1+\mathbf z_a^\top\Lambda_N^{-1}\mathbf z_a\big)\Big).
$$

三个期望都用 Monte Carlo 计算（A7，$M=64$）。注意都是“对乘积求期望”，不能先逐项求期望再相乘。

**期望损失。** 由 A8 和后验的精确分解，三个量在后验预测下相互独立，因此
$$
\mathbb E\big[\mathbf 1[G_k(a)\ge g_0]\big]=\sum_{j}\hat P_k(J=j\mid a)\ P\Big(\mathrm{dur}(a)\le j\Delta-p_k-g_0\Big),
$$
$$
\mathbb E\big[\varphi(G_k(a))\big]=\sum_j\hat P_k(J=j\mid a)\ \mathbb E_{\mathrm{dur}}\Big[\varphi\big((j\Delta-p_k-\mathrm{dur}(a))^+\big)\Big].
$$
代入 §2.4，即得 $R_k(a)$。

### 5.6 测试场的在线后验（Bayes 滤波）

每句结束后，用 Thinker 的整句续写得到该句各次读取的 $S_k$，从播放日志得到每次提交实际的 $J$（包括删失）。两者都不需要参考译文。然后更新两块场参数：
$$
p(\theta_s,\theta^J_s\mid\mathcal Y_s^{(\le n)})\propto p(\theta_s\mid\mathcal Y_s^{(<n)})\,\mathcal L^S_n(\theta_s)\ \cdot\ p(\theta^J_s\mid\mathcal Y_s^{(<n)})\,\mathcal L^J_n(\theta^J_s)
\approx\mathcal N(\mu^{(n)}_{s},\Sigma^{(n)}_{s})\,\mathcal N(\mu^{J(n)}_{s},\Sigma^{J(n)}_{s}).
$$
- 这一步用 Laplace 假设密度滤波（A2）。
- 句随机截距已经在 $\mathcal L_n$ 中积掉，因此**不再需要** v1 那种乘以 $1/\text{离散度}$ 的似然权重，滤波是严格的 Bayes 更新。
- 标签只用于之后的句子，不存在使用未来信息的问题。
- 在线更新还要求测试场 $\psi$ 的取值落在训练覆盖的范围内，这一点由 §5.3(b) 的外推说明保证。

### 5.7 Bayes 动作

$$
a^\star_k(\lambda)=\arg\min_{a\in\mathcal A_k}\ R_k(a;\lambda),
$$
其中 $R_k(a;\lambda)$ 是 §2.4 的期望损失在 §5.5 后验预测下的值，并取 $C_\mathrm{seam}(M)=\lambda$。

### 5.8 哪些部分不是 Bayes 推断

| 部分 | 性质 | 怎么确定 |
|---|---|---|
| $C_\mathrm{err},C_\mathrm{lat},C_\mathrm{sil},g_0$ | 偏好，只有比值有意义 | 沿用 v1，在训练域 dev 上确定 |
| $\lambda$（取 $C_\mathrm{seam}(M)=\lambda$） | 约束优化的对偶变量，不是随机变量 | §2.5 的对偶上升；约束中的期望可以用 §5.5 的后验预测在 dev 上计算，也可以用经验频率，两者都报告 |
| 动作集合 $\mathcal A_k$ | 硬约束 | 规则（§2.2） |
| 是否付出采样 $R$ 条草稿的计算 | 决策 | 后验包含概率 > 0.5，**并且**在 dev 上 Bayes 风险的下降超过计算带来的延迟代价 |
| RealSI 上的判定 | 评测，不属于模型 | 配对 bootstrap（频率派）。也可以同时报告 Bayesian bootstrap 下的 $P(\Delta\text{BLEU}>0\mid\text{RealSI})$，两者都只算一次 |

### 5.9 与 v1 的对应

| v1 | v2 |
|---|---|
| 离散度修正（似然乘以 $1/\phi$，generalized Bayes） | 句随机截距 $b_n$，严格的层级模型 |
| 下一个窗口一定能接上（$J\equiv1$） | $J$ 的生存似然和后验预测 |
| Talker 时长取均值（A6） | 共轭后验，后验预测为 Student-t |
| 新统计量用 AUC 门槛 | spike-and-slab，看后验包含概率 |
| 碎块/切点代价是固定常数 | 硬约束 + 对偶变量 $\lambda$ |
