# 同传 test-time Bayesian：分层模型（场 / 句 / 读取）的先验、似然、后验，以及步骤 0 的检验

- 前置：[m2_test_time_bayes_model.zh-CN.md](m2_test_time_bayes_model.zh-CN.md)（v1 模型）、[m2_bayesrisk_v2_model.zh-CN.md](m2_bayesrisk_v2_model.zh-CN.md)（v2 模型）
- 代码：`experiments/uniss_qwen3_omni_v1/data_clean/fit_sentence_filter.py`、`history_features_auc.py`
- 结果：`reports/uniss_m2_bayesrisk_sf/step0.json`、`step0b.json`、`step0b_frontier.json`

所有部件都只在测试时做推断。Thinker、ABC、Talker 都冻结。超参数只在训练域 dev（UniST 805 句）上用经验 Bayes 确定。

## 1. 记号

- 场（演讲）$s$，句 $n$，读取 $k=1,\dots,K_n$。
- 第 $k$ 次读取时，已提交前缀为 $\mathbf c_k$，草稿在已提交之后的单位为 $\mathbf g_k=(g_{k,0},\dots,g_{k,K_k-1})$。
- 单位 $u$ 的统计量为 $\mathbf x_{k,u}\in\mathbb R^8$：截距、ABC logit、forced、非 wait 概率占比、与上一次读取一致、源语静音、位置、句首。
- **决策量** $S_k$：草稿中最终不会被改的长度。它要到句末才知道。
- **测试时可观测量** $V_{k,u}$：第 $k$ 次读取草稿的单位 $u$，在第 $k{+}1$ 次读取的草稿里是否不变。读到第 $k{+}1$ 次时就能看到。

## 2. 三层模型

### 2.1 先验

$$
\begin{aligned}
\theta&\sim\mathcal N(m_0,S_0),\quad m_0=\text{ABC}\ (\text{ABC logit 的系数为 1，其余为 0})&&\text{全局：最终稳定性的系数}\\
\delta_s&\sim\mathcal N(0,\Psi)&&\text{场级偏移，}\theta_s=\theta+\delta_s\\
b_n&\sim\mathcal N(0,\tau^2)&&\text{句级“易变程度”}
\end{aligned}
$$

### 2.2 似然

**最终稳定性**（决策量）：
$$
P(S_k\ge u{+}1\mid S_k\ge u,\mathbf x_{k,u},\theta_s,b_n)=\sigma(\theta_s^\top\mathbf x_{k,u}+b_n)
$$
所以一次读取的似然是一条生存链：
$$
\mathcal L^{S}_k(\theta_s,b_n)=\big[1-\sigma(\theta_s^\top\mathbf x_{k,S_k}+b_n)\big]^{\mathbf 1[S_k<K_k]}\prod_{u<S_k}\sigma(\theta_s^\top\mathbf x_{k,u}+b_n).
$$

**相邻两次读取间的不变性**（测试时可观测）：
$$
P(V_{k,u}=1\mid V_{k,u-1}=1,\mathbf x_{k,u},b_n)=\sigma(\eta^\top\mathbf x_{k,u}+\gamma\,b_n)
$$
它的似然 $\mathcal L^{V}_k(b_n)$ 是同样的生存链：
- 从第 $k$ 次读取后才提交的单位开始；
- 遇到第一个变动的单位记 0；
- 若下一次草稿更短，则删失。

### 2.3 后验与更新：先验会更新，而且分层更新

| 层级 | 何时更新 | 用到的数据 | 公式 |
|---|---|---|---|
| 全局 $\theta,\eta,\gamma,\tau$ | 测试前，只做一次 | 训练域 dev 的 $S$ 和 $V$ | $\theta,\eta,\gamma$ 取 MAP；$\tau$ 用积掉 $b$ 后的边际似然最大化（经验 Bayes） |
| 场级 $\delta_s$ | 每句结束后 | 本句各读取的 $S_k$（句末才可得） | $p(\theta_s\mid\mathcal D_{s,\le n})\propto p(\theta_s\mid\mathcal D_{s,<n})\prod_k\mathcal L^S_k(\theta_s)$（v1 的 PO，Laplace 近似） |
| 句级 $b_n$ | 每读一次 | 之前各次读取的 $V$ | $p(b_n\mid O_{n,k+1})\propto p(b_n\mid O_{n,k})\,\mathcal L^V_k(b_n)$，$p(b_n\mid O_{n,1})=\mathcal N(0,\tau^2)$ |

**每一层上一步的后验，就是下一步的先验**：
- 句内，第 $k$ 次读取后的 $p(b_n\mid O_{n,k+1})$ 是第 $k{+}1$ 次读取的先验；
- 场内，第 $n$ 句后的 $p(\theta_s\mid\cdot)$ 是第 $n{+}1$ 句的先验；
- 新的一句开始时，$b_n$ 回到 $\mathcal N(0,\tau^2)$，因为各句可交换。

$b_n$ 只有一维，用 24 点 Gauss–Hermite 网格精确滤波。

### 2.4 后验预测与 Bayes 动作

$$
P(S_k\ge a\mid\mathbf x_k,O_{n,k})=\int\prod_{u<a}\sigma(\theta_s^\top\mathbf x_{k,u}+b)\;p(b\mid O_{n,k})\,db
$$
$$
a_k^\star=\arg\min_{a\in\mathcal A_k}\ C_\mathrm{err}\,\mathbb E[(a-S_k)^+]+C_\mathrm{lat}D\,\mathbb E[(\min(S_k,K_k)-a)^+]
$$
这等价于在硬动作集合 $\mathcal A_k$ 中，提交满足 $P(S_k\ge a)>1/(1+C_\mathrm{lat}D)$ 的最长前缀。

取 $\tau=0$、$\Psi=0$、$\theta=m_0$ 时就是 ABC + SDR，即 v3。所以 v3 是这个模型的特例。

## 3. 步骤 0：test-time 证据能不能提高排序（门槛：AUC +0.02 且 NLL 下降）

**数据**：dev 上 ABC 先验 + 硬卫生的解码记录（`bayesrisk_v2/dev_it0b`，每个方向 6 个 bias），按句做 2 折交叉验证。
- en→zh：2,448 次运行，13.6 万行；
- zh→en：2,382 次运行，6.3 万行。

### 3.1 句级滤波（§2.3 第三行）

| 方向 | ABC 先验 | 拟合 $\theta$（v1） | 加句效应，不滤波 | **句级滤波（test-time）** | 句效应 oracle（知道本句所有最终标签） | $\tau$ | $\gamma$ |
|---|---|---|---|---|---|---|---|
| en→zh AUC / NLL | 0.826 / 0.489 | 0.850 / 0.307 | 0.846 / 0.309 | **0.849 / 0.307** | 0.857 / 0.302 | 0.50 | 0.62 |
| zh→en AUC / NLL | 0.802 / 0.451 | 0.826 / 0.375 | 0.826 / 0.376 | **0.826 / 0.375** | 0.830 / 0.372 | 0.45 | 0.89 |

**未通过门槛**：滤波带来的 AUC 变化为 −0.0015 / −0.0006。就算句效应完全已知（oracle），也只多 +0.007 / +0.004。
- 句与句之间确实有差异（$\tau\approx0.5$），但这个差异几乎不改变句内各单位的排序。
- v1 观察到的读取间离散度 6–7，来自同一个可疑单位在连续多次读取中被重复计分，而不是句级的易变程度。

### 3.2 单位级历史（这个单位在之前各次读取中的表现）

新增三个统计量，都在读取时可观测：
- **连续存活次数**：之前连续多少次读取在前缀直到该位置上与当前草稿完全一致；
- **改动次数**：之前起草过该位置、并且在它之前或在它上面与当前草稿不一致的读取次数；
- **见过次数**：之前起草过该位置的读取次数。

| 方向 | 拟合 T | T + 连续存活 | T + 全部历史 | 决策前沿：同等“扣下的稳定单位”时“说错的单位” |
|---|---|---|---|---|
| en→zh AUC / NLL | 0.850 / 0.307 | 0.855 / 0.304 | 0.854 / 0.304 | 减少约 3–4% |
| zh→en AUC / NLL | 0.826 / 0.375 | 0.832 / 0.370 | 0.834 / 0.367 | −1% ～ +2%，没有改善 |

**未通过门槛**：AUC 只有 +0.004 / +0.008。连续存活次数的系数显著（zh→en 为 1.28 ± 0.16），但它携带的信息大部分已经包含在 ABC 和“与上一次读取一致”里了。

## 4. 结论：test-time 后验已经接近可观测信息的上限

- 决策量 $S_k$ 的不确定性，来自**还没听到的源语音频**。测试时能观测到的东西，包括这句之前的读取和这个单位的历史，对它几乎没有额外信息。
- 后验这一侧（先验、似然、滤波）再怎么改，排序最多提高约 0.01 AUC。按 v2 模型的命题 1，这推不动延迟–质量前沿。
- 这也是 v1/v2 在 v3 上没有效果的根本原因：v3 的 ABC 已经用掉了测试时可得的大部分信息。

**test-time Bayesian 还能起作用的地方，是决策和损失这一侧，而不是对 $S$ 的后验：**
1. **先测上限**：在 dev 上跑“oracle 提交”，即事先知道 $S_k$、每次正好提交稳定部分。如果它在同等 LAAL 下的 BLEU 也只比 v3 高很少，那么任何“改提交时机”的 Bayesian 都不值得再做。
2. **改说什么，而不只是什么时候说**：对 Thinker 采样 $R$ 条续写，作为对译文的后验预测样本，用期望效用（MBR）选出要提交的前缀。这是 test-time 对译文内容的 Bayes 决策，是唯一能直接改变内容的杠杆。
3. **损失里加入 Talker**：块长度 → Talker 读错率和 UTMOS 的观测模型。它针对的是 zh→en 切得过碎、UTMOS 下降的问题。
