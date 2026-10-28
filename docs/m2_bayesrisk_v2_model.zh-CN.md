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
