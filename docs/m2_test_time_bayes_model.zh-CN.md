# test-time Bayesian 提交策略的严格数学模型

本文把 [m2_test_time_bayes_plan.zh-CN.md](m2_test_time_bayes_plan.zh-CN.md) 的方案 A–D 写成一个完整的 Bayesian 决策模型。模型由四部分组成：
1. 随机变量；
2. 似然与先验；
3. 后验与后验预测；
4. 损失与 Bayes 动作。

所有近似都单独编号（A1–A7），并给出检验方法。论文中 v3 用的 ABC+SDR 是这个模型的一个特例（§5 命题 1）。

## 0. 结论

- **严格成立的部分**：
  - 后验由明确的似然和先验经 Bayes 公式得到；
  - 动作是后验期望损失的最小值点；
  - 按场的自适应是层级模型中场参数的后验更新。
  
  这三部分在模型假设下都是精确的。
- **需要近似的部分**：
  - 真实后验 $p(S_k\mid o_k)$ 要对未来音频积分，无法直接计算，用 ABC 的摊销后验作为统计量（A1）；
  - Gaussian/Laplace 近似（A2、A3）；
  - 读取之间条件独立（A4）；
  - 短视的一步决策，而不是求解完整的 Bellman 方程（A5）；
  - Talker 时长取均值（A6）；
  - 后验预测的 Monte Carlo 计算（A7）。
- **损失中的代价比**（$C_\mathrm{frag}/C_\mathrm{err}$ 等）表达的是偏好，不由 Bayes 推断决定。它们在训练域留出集上设定。
- **严格化后需要修正上一版方案的五处**，见 §8。

## 1. 随机变量与已知量

| 符号 | 含义 | 已知 / 隐 |
|---|---|---|
| $s$，$n$，$k$ | 演讲（场）、句子、句内第 $k$ 次读取 | — |
| $t_k=\min(k\Delta,T)$ | 第 $k$ 次读取时已听到的源语时长 | 已知 |
| $o_k=(\mathbf x_{1:t_k},\mathbf c_k)$ | 已听到的音频和已提交的文本 | 观测 |
| $\mathbf d_k=f(o_k)\in\mathcal V^{K_k}$ | 当前草稿（贪心续写） | 观测的函数 |
| $Z_k=\mathbf x_{t_k:T}$ | 还没听到的音频 | **隐** |
| $S_k=\mathrm{lcp}\big(\mathbf d_k,f(\mathbf x_{1:T},\mathbf c_k)\big)$ | 稳定长度：草稿中整句续写也会说的前缀长度 | 隐，是 $Z_k$ 的确定函数 |
| $\xi_k=(q_k,\beta_k,g_k)$ | 播放状态：Talker 队列、播放缓冲、听众当前已连续静音的时长 | 已知，由历史动作确定 |
| $\theta_s\in\mathbb R^p$ | 场 $s$ 的校准参数 | **隐** |
| $a_k\in\{0,\dots,K_k\}$ | 动作：提交草稿的前 $a_k$ 个单位 | 决策 |

$S_k$ 的定义只用到 Thinker 自己听完整句后的续写，**不需要参考译文**。因此测试时可以在每句结束后得到 $S_k$ 的真实值，作为自标注。

对 $u=0,\dots,K_k-1$，定义嵌套事件 $\{S_k\ge u{+}1\}\subseteq\{S_k\ge u\}$。由链式法则（精确）：
$$
P(S_k\ge a\mid\cdot)=\prod_{u<a}\rho_k(u),\qquad
P(S_k=s\mid\cdot)=\big(1-\rho_k(s)\big)\prod_{u<s}\rho_k(u),
$$
其中 $\rho_k(u)=P(S_k\ge u{+}1\mid S_k\ge u,\cdot)$，并约定 $\rho_k(K_k)=0$。

## 2. 似然与先验

**真实后验**是
$$
p(S_k=s\mid o_k)=\int\mathbf 1\big[\mathrm{lcp}(\mathbf d_k,f(\mathbf x_{1:t_k}\oplus z,\mathbf c_k))=s\big]\,p(z\mid\mathbf x_{1:t_k})\,dz .
$$
它要对未来音频积分，无法直接计算。

**A1（统计量）**：不以完整的 $o_k$ 为条件，而以它的一个统计量为条件：
$$
\mathbf T_{k,u}=\phi(o_k,u)\in\mathbb R^p .
$$
各分量如下：
- $\mathrm{logit}\,\rho_{\theta,k}(u)$：ABC 的摊销后验，$=\mathrm{logit}(1-h_\theta(u\mid o_k))$；
- $\log\pi_k(u)$：草稿第 $u{+}1$ 个单位在非 wait 概率中的占比；
- $e^{(2)}_{k,u}$：该单位是否已出现在上一次读取草稿的同一位置（LocalAgreement 的证据）；
- $e^{(3)}_k$：读取前 200 ms 内源语是否静音（意群边界）；
- 位置、方向、截距。

以统计量为条件的后验 $P(\cdot\mid\mathbf T)$ 是一个**严格的 Bayes 后验**，只是信息集比 $o_k$ 粗。基于它的 Bayes 动作，是所有只依赖 $\mathbf T$ 的策略中最优的（在模型类正确时）。
- 当 $\mathbf T$ 只包含 ABC 的 logit 时，就退化为现在的 v3；
- 增加分量只会扩大信息集，不会损失信息。

**似然**（条件 Bernoulli，一个 logistic 模型）：
$$
P\big(S_k\ge u{+}1\mid S_k\ge u,\ \mathbf T_{k,u},\ \theta_s\big)=\sigma\big(\theta_s^\top\mathbf T_{k,u}\big).
$$
于是一次读取的标签 $S_k$ 的似然恰好是
$$
\mathcal L_k(\theta_s)=\Big[1-\sigma(\theta_s^\top\mathbf T_{k,S_k})\Big]^{\mathbf 1[S_k<K_k]}\prod_{u<S_k}\sigma(\theta_s^\top\mathbf T_{k,u}),
$$
它就是 §1 的链式法则，没有额外近似。

**层级先验**：
$$
\theta_s=\omega+\delta_s,\qquad \omega\sim\mathcal N(m_0,S_0),\qquad \delta_s\sim\mathcal N(0,\Psi).
$$
- $m_0=(1,0,\dots,0)$：对 ABC logit 的系数为 1，其余为 0。数据不提供信息时，模型就是原来的 ABC。
- $S_0$：先验不确定度。
- $\Psi$：场与场之间的差异。

**A4（条件独立）**：给定 $\theta_s$，不同读取的标签相互独立。同一句的各次读取共用音频，实际上正相关，所以这个假设会让后验偏自信。§7 给出检查方法和修正。

## 3. 后验与后验预测

**训练域**：$\mathcal D_\mathrm{tr}$ 是 UniST 留出集的读取（`crc32(id) % 50 == 0` 的验证句），每次读取带有 $\mathbf T$ 和 $S_k$，并按录音分组 $g$。
$$
p(\omega,\{\delta_g\}\mid\mathcal D_\mathrm{tr},\Psi)\ \propto\ \mathcal N(\omega;m_0,S_0)\prod_g\Big[\mathcal N(\delta_g;0,\Psi)\prod_{k\in g}\mathcal L_k(\omega+\delta_g)\Big].
$$
- **A2（Laplace）**：$p(\omega\mid\mathcal D_\mathrm{tr})\approx\mathcal N(\mu,\Sigma)$。
- **A3（经验 Bayes）**：$\hat\Psi=\arg\max_\Psi p(\mathcal D_\mathrm{tr}\mid\Psi)$，边际似然同样用 Laplace 近似计算。

**测试场 $s$ 的先验**：$\omega$ 已由大量训练数据确定，测试场的少量标签对它几乎没有影响。忽略这部分影响后，
$$
\theta_s\sim\mathcal N\big(\mu,\ \Sigma+\hat\Psi\big).
$$

**在线后验（Bayes 滤波）**：句 $n$ 结束后，用整句续写得到该句各次读取的 $S_k$（无参考），然后更新
$$
p\big(\theta_s\mid\mathcal Y_s^{(\le n)}\big)\ \propto\ p\big(\theta_s\mid\mathcal Y_s^{(<n)}\big)\prod_{k\in n}\mathcal L_k(\theta_s)\ \approx\ \mathcal N\big(\mu_s^{(n)},\Sigma_s^{(n)}\big)\quad\text{（Laplace 假设密度滤波，A2）}.
$$
标签只影响**之后**的句子，因此不存在用未来信息的问题。

**后验预测**：决策需要的是 $\theta_s$ 积分后的概率。
$$
\hat P_k(S_k\ge a)=\mathbb E_{\theta_s\sim\mathcal N(\mu_s,\Sigma_s)}\Big[\prod_{u<a}\sigma(\theta_s^\top\mathbf T_{k,u})\Big]\ \approx\ \frac1M\sum_{m=1}^M\prod_{u<a}\sigma\big(\theta^{(m)\top}\mathbf T_{k,u}\big).
$$
- **A7**：用 Monte Carlo 计算，$M=64$ 即可。
- 注意乘积的期望 $\neq$ 期望的乘积，所以不能逐单位代入预测概率再相乘。

## 4. 损失

给定动作 $a$、真实稳定长度 $S_k$ 和播放状态 $\xi_k$，本次读取的损失为
$$
\ell(a;S_k,\xi_k)=
\underbrace{C_\mathrm{err}\,\mathbf 1[S_k<a]}_{\text{说出会被改的内容}}
+\underbrace{C_\mathrm{lat}\,D_k\,N_\mathrm{wait}(a,S_k)}_{\text{内容滞后}}
+\underbrace{C_\mathrm{sil}\,\ell_\mathrm{sil}(a,\xi_k)}_{\text{听众静音}}
+\underbrace{C_\mathrm{frag}\,\mathbf 1[1\le n(a)<m]\,\mathbf 1[p_k=0]}_{\text{空闲时的碎块}}
+\underbrace{C_\mathrm{seam}\,\mathbf 1[\mathrm{seam}(d_{k,a})\in\{W,P\text{-}fw\}]}_{\text{切在词内或虚词后}}
+\underbrace{C_\mathrm{talk}\,\pi_T(a)}_{\text{Talker 读错}}
+\underbrace{\infty\cdot\mathbf 1[\text{含复读 n-gram}]}_{\text{复读}} .
$$

**各项的定义**：
- **内容滞后**：$N_\mathrm{wait}(a,S)=\big(\min(S,K_k-1)-a+1\big)_+$，即被扣下、且前面的内容都稳定的位置数（包括紧接其后、等着被改正的那个位置）。每个这样的位置让听众多等
  $$D_k=\max(0,\Delta-p_k),\qquad p_k=\frac{q_k}{s(q_k)}+\beta_k,$$
  其中 $p_k$ 是听众还能听多久的译音：$s(q)$ 是 `--speed-policy queue` 的加速倍率，$\beta_k$ 是缓冲中尚未播放的部分。
- **听众静音**：设本次提交的语音时长为 $\mathrm{dur}(a)=\bar r\,n(a)/s(q_k)$（**A6**：Talker 时长取均值 $\bar r$）。下一个窗口内听众会听到的静音是 $\varsigma_k(a)=\max(0,\Delta-p_k-\mathrm{dur}(a))$。静音段的代价为 $\varphi(\text{长度})$，取
  $$\varphi(g)=(g-g_0)_++\lambda\,(g-g_0)_+^2,$$
  其中 $g_0\approx0.5$ s，即不长于 $g_0$ 的停顿不计代价。按“每段静音计一次代价”的会计方式：
  $$\ell_\mathrm{sil}(a,\xi_k)=\begin{cases}\varphi\big(g_k+\varsigma_k(0)\big)-\varphi(g_k), & n(a)=0\ \ (\text{当前静音段继续延长})\\[2pt] \varphi\big(\varsigma_k(a)\big), & n(a)>0\ \ (\text{当前静音段结束，下一段可能开始})\end{cases}$$
  当前已有的 $\varphi(g_k)$ 是沉没成本，两个分支都不计入。
- **Talker 读错**：$\pi_T(a)$ 是 Talker 读错这一块的概率，由已测得的读错率表给出（按块长和末字类型分档）。它与 $S_k$ 无关，所以放在损失里，而不是作为证据（见 §8 修正 2）。
- **复读**：含复读 n-gram 的提交代价为无穷大，即硬约束。等价于把这类单位的可提交集合设为空。
- **清洗**：口语清洗是动作之后的确定性文本变换，损失在变换后的文本上计算，不进入决策。

$C_\cdot$ 各项是偏好参数，只有比值有意义。

## 5. Bayes 动作

后验期望损失为（期望对 §3 的后验预测求取）：
$$
R_k(a)=\mathbb E\big[\ell(a;S_k,\xi_k)\mid\mathbf T_k,\mathcal Y_s\big]
= C_\mathrm{err}\Big(1-\hat P_k(S_k\ge a)\Big)
+ C_\mathrm{lat}D_k\sum_{u=a}^{K_k-1}\hat P_k(S_k\ge u)
+ C_\mathrm{sil}\ell_\mathrm{sil}(a,\xi_k)+C_\mathrm{frag}(\cdot)+C_\mathrm{seam}(\cdot)+C_\mathrm{talk}\pi_T(a),
$$
其中用到 $\mathbb E[\mathbf 1[S<a]]=1-P(S\ge a)$ 和 $\mathbb E[N_\mathrm{wait}]=\sum_{u=a}^{K-1}P(S\ge u)$。

Bayes 动作是
$$
a_k^\star=\arg\min_{a\in\{0,\dots,K_k\}}R_k(a).
$$
每次读取只需枚举 $K_k{+}1$ 个候选，计算量可以忽略。

**命题 1（v3 是特例）。** 设：
- $C_\mathrm{sil}=C_\mathrm{frag}=C_\mathrm{seam}=C_\mathrm{talk}=0$；
- 后验退化为点质量 $\theta_s\equiv m_0$，此时 $\hat\rho_k(u)=\rho_{\theta,k}(u)$；
- $p_k$ 取 SDR 的队列 $q_k$。

则
$$
R_k(a{+}1)-R_k(a)=\hat P_k(S_k\ge a)\Big[C_\mathrm{err}\big(1-\rho_{\theta,k}(a)\big)-C_\mathrm{lat}D_k\Big].
$$
它的符号与论文 Prop. 1 的逐单位阈值 $\rho_{\theta,k}(a)>\tau_k=1-C_\mathrm{lat}D_k/C_\mathrm{err}$ 相同，代入 $D_k$ 即得 SDR 的 $\tau_k$。

如果 $\rho$ 在第一次不过阈值之后不再回升，全局 $\arg\min$ 就等于论文的“最长全部过阈值前缀”。否则全局 $\arg\min$ 可能越过一个低置信单位、去拿后面的高置信内容。这是论文中的贪心规则相对严格 Bayes 动作的一处差别。

**命题 2（硬规则是代价取无穷的极限）。** v3 的提交卫生“Talker 空闲时块必须 $\ge m$ 个单位”，等于取 $C_\mathrm{frag}\to\infty$。

所以方案 A 并没有引入新机制，只是把规则从“无穷大的代价”改成“有限、可校准的代价”。这样短块在等待代价很高（例如长静音）时可以被放行，也就消除了“后验说该说、规则不让说”的冲突。

## 6. 序贯最优与短视近似

严格的最优策略要求解部分可观测的序贯决策问题，状态是（信念，$\xi$），Bellman 方程为
$$
V_k(\mathbf T_k,\xi_k)=\min_a\Big\{R_k(a)+\mathbb E\big[V_{k+1}(\mathbf T_{k+1},\xi_{k+1})\mid\mathbf T_k,\xi_k,a\big]\Big\}.
$$

**A5（短视）**：假设动作只通过 §4 中显式计入的“下一窗口”项影响未来，于是 $V_{k+1}$ 与 $a$ 无关，最优动作退化为 $\arg\min_a R_k(a)$。论文的 Prop. 1 用的就是同一假设。

可选的改进是一步信息价值：若
$$
\min_a R_k(a)\;-\;\mathbb E_{\mathbf T_{k+1}}\big[\min_{a'}R_{k+1}(a')\big]\;>\;C_\mathrm{lat}D_k+C_\mathrm{sil}\Delta\ell_\mathrm{sil},
$$
则等待。这需要预测下一次读取的统计量，留作第二阶段。

## 7. 近似清单与检验

| 编号 | 近似 | 怎么检查 |
|---|---|---|
| A1 | 以统计量 $\mathbf T$ 代替 $o_k$ | 在训练域留出集上比较 AUC 和负对数似然：只用 ABC 对比用全部 $\mathbf T$ |
| A2 | Laplace / 假设密度滤波 | 与 MCMC（NUTS）在小样本上对比 $\mu,\Sigma$ |
| A3 | 经验 Bayes 估计 $\Psi$ | 留一录音交叉验证的预测对数似然 |
| A4 | 读取间条件独立 | 在训练域上比较同句读取残差的相关性；若偏自信，就把每句的似然乘以 $1/\bar n_\mathrm{eff}$（用有效样本数计似然），或给每句加一个随机效应 |
| A5 | 短视决策 | 与一步信息价值的版本在留出集上对比 |
| A6 | Talker 时长取均值 | 用实测时长代替均值重算，比较两种决策的一致率 |
| A7 | Monte Carlo 计算后验预测 | 比较 $M=64$ 与 $M=1024$ 的结果 |

## 8. 严格化后对上一版方案的修正

1. **证据融合（方案 C）的温度权重 $w_j$ 改为 logistic 系数。** 原写法 $\mathrm{logit}\,\rho+\sum_j w_j\ell_j$ 是 generalized Bayes，$w_j$ 是人为加的温度，不严格。严格版本把所有统计量放进 §2 的 logistic 似然，系数有 Gaussian 先验，由训练域数据给出后验，对系数积分后得到预测。方案 C 和 D 由此合并为同一个层级模型：$\omega$ 是群体参数，$\delta_s$ 是场的偏差。
2. **Talker 读错从“证据”移到“损失”。** Talker 是否读对与 $S_k$（文本是否会改）无关，把它当作 $S_k$ 的证据属于模型错误。它应作为 $C_\mathrm{talk}\pi_T(a)$ 计入损失。
3. **静音代价按“听众的静音段”计，而不是按单位计。** 上一版把 $\varphi$ 的增量乘到每个扣下的单位上，混淆了“内容滞后”（按单位计，对应 LAAL）和“听感静音”（按段计）。§4 把两者分开，静音段只计一次，并且依赖本次提交的时长 $\mathrm{dur}(a)$。
4. **取全局 $\arg\min$，不用逐单位的贪心前缀**（§5 命题 1）。
5. **复读保护改为无穷大的损失**，而不是把 $\rho$ 设为 0。复读的单位不一定“会被改”，但无论如何都不该说出口，所以它是偏好约束，不是概率判断。

## 9. 用这个模型解释 Beta 自适应为什么无效

把 Beta 方案放进同一框架，相当于：
- **似然只有截距**：$P(\text{这次读取有修改}\mid\pi_s)=\pi_s$，丢掉了 $\mathbf T$；
- **事件选错**：用的是“这次读取有任何修改”，而损失里的事件是逐单位的 $\{S_k<a\}$；
- **偏置不是由损失推出的 Bayes 动作**：$b_s=\kappa(\mathrm{logit}\,\mathbb E\pi_s-\mathrm{logit}\,\varepsilon)$ 是在把修改率拉向目标 $\varepsilon$，属于约束控制，不是最小化期望损失。

在严格模型里，场效应只通过 $\delta_s$ 进入后验，阈值始终由损失决定，所以：
- 如果训练域上场与场之间的差异 $\hat\Psi$ 很小，测试场的后验 $\delta_s$ 就停在 0 附近，决策基本不变；
- 这与诊断结果一致：按场挑偏置，扣掉种子噪声后的净上限只有 +0.35 / +0.5 BLEU，可提升的空间主要在读取层面（$\mathbf T$ 的系数和损失中的静音、碎块项）。

## 10. 实现对应

在 `si_thinker_stream.py` 中新增 `--policy bayesrisk`（决策在 `si_bayesrisk.py`），参数全部默认关闭，v3 的默认行为不变：
- `--br-model <json>`：训练域拟合的 $\mu,\Sigma,\hat\Psi$ 和似然权重；不给时用先验（即 ABC）；
- `--br-loss {event,unit}`、`--br-cost-sil`、`--br-g0`、`--br-lambda`、`--br-cost-frag`、`--br-frag-min`、`--br-cost-seam`；
- `--br-playback`（播放感知的滞后 $D$）、`--br-tau-max`、`--br-mc`；
- `--br-online`、`--br-like-w`（在线 Laplace 更新及其似然权重）、`--br-dump`（导出训练用的链式标签）。

拟合脚本放在 `data_clean/`，只读 UniST 训练域留出集。RealSI 和 ACL 只用于测试。

## 11. 实现后的修正（结果见 [m2_bayesrisk_results_v1.zh-CN.md](m2_bayesrisk_results_v1.zh-CN.md)）

1. **按事件计的损失在全局 argmin 下会退化。** §4 中“一次修改只计一次 $C_\mathrm{err}$”的写法，会让含可疑单位的长草稿的期望损失被摊薄，全局最优于是整段提交。改为按单位计：
   $$R(a)=C_\mathrm{err}\,\mathbb E[(a-S)^+]+C_\mathrm{lat}D\,\mathbb E[(\min(S,K)-a)^+],\qquad C_\mathrm{lat}=\frac{1-\sigma(b)}{\sigma(b)\,\Delta}.$$
   - 其 argmin 是对边际存活概率取阈值：$a^\*=\max\{a: P(S\ge a)>1/(1+C_\mathrm{lat}D)\}$。
   - 因为 $P(S\ge a)$ 随 $a$ 单调，这个阈值是唯一的。
   - 在先验下它恰好是 ABC，命题 1 不变。
2. **动作集合 $\mathcal A$ 只包含 token 边界。** 在多字 token 中间提交，会让 Thinker 在不完整的 token 后续写，产生零宽字符和断词。这是模型之外的约束（Thinker 的 prompt 必须以完整 token 结尾），不是概率判断。
3. **新增统计量 `forced`。** token 内部的单位没有等待标记的 hazard，ABC logit 只能取上限 15。为此加一个指示量 `forced`，其先验均值取 15，使“只用先验”时与 ABC 逐字节一致。
4. **近似 A4 的处理。** 同一句多次读取的 Pearson 离散度为 6–7：$\Sigma$ 乘以离散度，在线更新的似然权重取其倒数。
5. **RealSI 结果指出损失的缺口。** 碎块和切点代价有限时，Bayes 动作会在短语中间提交，突兀停顿变为 v3 的 1.4–3 倍。按命题 2，应把提交卫生作为硬约束（$C_\mathrm{frag}\to\infty$），并让切点代价随切点类型变化。
