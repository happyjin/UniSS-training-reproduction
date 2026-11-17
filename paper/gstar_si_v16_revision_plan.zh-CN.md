# MIST 论文 v16 修改方案（基于 gstar_si_v15，第二轮）

2026-11-17（布拉格时间）。v15：`paper/gstar_si_v15/`（commit 88bd508ad）。不做新实验。
不改动：abstract、introduction、Figure 1（同 v15）。

**第二轮的变化**
- 用户决定 R3′ 不放进正文。
- 正文写出 Proposition 2，并先用一句话交代它的用处。
- 压缩 Problem Formulation，以及每个小节开头的动机段。
- 新增全部数学符号的审查结果（§2）。
- 实验部分与 Method 的对应关系见 §4。
- 承诺公开评测脚本。

---

## 1. 已定决策

| 问题 | 决定 |
|---|---|
| R3′（语速匹配规则）放进正文 ablation | **不放**（用户）。正文措辞要克制：不说“收益主要来自 CRF”，只说 PA Talker 在每个集上进一步缩短语音、在 Zh→En 上提高质量，这些都是事实。完整析因表是否放进附录 G 见 §7 |
| 恢复数学推导，正文写出 Proposition 2 | **放**，要讲清动机和用处，简短，衔接顺 |
| 实验部分合并为 “4 Experiments”，最后一节是 Ablation Analysis | **是** |
| 公开 cpASR-BLEU / tcpASR-BLEU 评测脚本 | **承诺**。匿名审稿期写 “we will release the scorer” |

---

## 2. 数学符号逐个审查（新问题）

### 2.1 Eq. (9) 中各量的类型

$\min_q \mathrm{KL}(q\Vert p_0)\ \text{s.t.}\ \mathbb{E}_q[\Phi_\mathcal{S}]\le bW$：

- $q$、$p_0$ 是定义在训练语料全部播放计划上的**概率分布**（有限集合上的分布），不是向量或矩阵。
- $\mathbb{E}_q[\Phi_\mathcal{S}]$、$b$（秒/单位）、$W$（目标单位总数）**都是标量**。所以 $bW$ 是一个标量预算，单位是“单位·秒”。
- 问题在 $W$：它和每个块的 $W_i$ 容易混淆。v16 改成 $W_\mathcal{S}=\sum_{\mathcal{U}\in\mathcal{S}}\sum_i W_i$，与 $\Phi_\mathcal{S}$ 的写法对应。

### 2.2 v15 Method 中的问题和改法

| 问题 | v15 写法 | v16 改法 |
|---|---|---|
| **字母冲突** | $q$ 既表示 PR 分布，又表示语速 token（$q_i$、$q(c)$） | 语速 token 改用 $\tau$（$\tau_i$、$\tau^{\mathrm{pace}}_i$、$\tau_0$、$\bar\tau$），$q$ 只留给分布 |
| 字母冲突 | $h_i$ 是 backlog，$\hat{\bm h}_k$ 是 ASR 转写 | 转写改为 $\bm{o}_k$，参考改为 $\bm{g}_\varrho$ |
| 字母冲突 | $g_\psi$ 是投影器，$g_i$ 是 backlog 转移 | 投影器改为 $\mathrm{Proj}_\psi$ |
| 字母冲突 | $W$ 与 $W_i$ | $W_\mathcal{S}$ |
| 字母冲突 | $\mathrm{H}$（交叉熵）与 $\mathcal{H}$（格点）、$\bar H$ | 交叉熵改为 $\mathrm{CE}(\cdot,\cdot)$ |
| 字母冲突 | §3.2 里 $j$ 同时作块下标（$\bm{s}_j$）和 chunk 下标（$e_j$） | chunk 下标改为 $l$ |
| **缺维度** | $\bm{x}$ 只写了 “T seconds” | $\bm{x}\in\R^{L}$（$L$ 个采样点，时长 $T$ 秒） |
| 缺维度 | $\bm{y}^{(n)}$、$\bm{y}$ 没有空间 | $\bm{y}^{(n)}\in\mathcal{V}^{|\bm{y}^{(n)}|}$，$\mathcal{V}$ 为词表（含新增的说话人 token 和时间 token） |
| 缺维度 | $\bm{v}_k$ | $\bm{v}_k\in\R^{L_k}$（一段音频） |
| **该用向量/矩阵却写成标量函数** | $\alpha_i(h)$、$\beta_i(h)$ | 向量 $\bm{\alpha}_i,\bm{\beta}_i\in\R_{\ge0}^{|\mathcal{H}|}$，每个 backlog 值对应一维 |
| 同上 | $\mu_i(h,c)$ | 矩阵 $\bm{\mu}_i\in[0,1]^{|\mathcal{H}|\times|C_i|}$，行对应 backlog，列对应候选 |
| 同上 | $w_i(c)$ | 概率向量 $\bm{w}_i\in\Delta^{|C_i|}$（单纯形，各分量之和为 1） |
| 同上 | $\rho_i(q)$ | 概率向量 $\bm{\rho}_i\in\Delta^{|\mathcal{T}|}$，$\mathcal{T}$ 为语速 token 集合 |
| 同上 | CRF 的观测 $\mathcal{U}$ 只用文字描述 | 写成 $\mathcal{U}=(\bm{\nu},\bm{W},C_{1:I})$，其中 $\bm{\nu}\in\R^{I}$ 是释放时刻，$\bm{W}\in\mathbb{N}^{I}$ 是各块单位数。这样 “conditional” 的含义就明确了 |

### 2.3 下标 $i$ 的标量要不要改成向量？

**大多数确实是标量**，例如 $h_i$、$\nu_i$、$W_i$、$D(c)$、$\lambda$、$b$、$\eta$、$\kappa$。链式模型的要点就是沿 $i$ 递推，所以逐块标量写法清楚，也是 HMM/CRF 论文的标准写法，保留。

**本身是向量或矩阵的量**（消息、边缘分布、候选权重、语速后验）改成粗体，并写明维度。前向–后向也改成矩阵–向量形式：对每个候选 $c$，定义 0/1 转移矩阵 $\bm{T}_i^{(c)}\in\{0,1\}^{|\mathcal{H}|\times|\mathcal{H}|}$，其元素为 $[\bm{T}_i^{(c)}]_{h,h'}=\mathds{1}[g_i(h,c)=h']$，每行只有一个 1；再定义势函数向量 $\bm{\phi}_i^{(c)}\in\R^{|\mathcal{H}|}$。于是

$$\bm{\alpha}_{i+1}=\sum_{c}\bm{T}_i^{(c)\top}\big(\bm{\alpha}_i\odot e^{\bm{\phi}_i^{(c)}}\big),\qquad
\bm{\beta}_i=\sum_{c}e^{\bm{\phi}_i^{(c)}}\odot\big(\bm{T}_i^{(c)}\bm{\beta}_{i+1}\big),$$

$\bm{\mu}_i$ 的第 $c$ 列等于 $\bm{\alpha}_i\odot e^{\bm{\phi}_i^{(c)}}\odot\bm{T}_i^{(c)}\bm{\beta}_{i+1}/Z_\lambda$。由于每行只有一个 1，每个候选的计算量是 $O(|\mathcal{H}|)$，复杂度 $O(I|\mathcal{H}|\max_i|C_i|)$ 一眼可见。篇幅与 v15 相当。

**统一记号规则（全文）**：
- 标量用普通斜体；
- 向量用粗体小写（希腊字母也用 `\bm`）；
- 矩阵用粗体大写；
- 集合用 `\mathcal`；
- 分布用 $p,q,\pi$；
- 函数写成 $\mathrm{Proj}_\psi$、$g_i$ 这类形式；
- 每个符号第一次出现时，在同一句里给出类型和维度。

附录里的符号同样按这些规则统一：$\varphi_l$、$\sigma^\star$、$\bm{V}^{(m)}$ 等。

---

## 3. Method 的修改

### 3.1 压缩 Problem Formulation 和各小节开头

**3.1 Problem Formulation**：从约 12 行压到约 6 行，不写动机句，只给定义：
> *Given a single-channel recording $\bm{x}\in\R^L$ ($T$ s), \sys{} outputs segments $(k_n,a_n,b_n,\bm{y}^{(n)})$ … committed at reads $t_1<\dots<t_R$ from $\bm{x}_{\le t_r}$ only and never revised; each is played in the stream of speaker $k_n$, and a target unit's heard time is its playback time there (computed, §3.3).*

“为什么要同时解决‘谁说了什么’和‘何时听到’”这层意思，已经由 §3 开头的那句话交代了。

**其余各节开头只用一两句话交代问题，意思不变**：

| 小节 | v16 开头 |
|---|---|
| 3.2 | *Acoustic tokens carry no speaker identity across a meeting and are shared by overlapping speakers, so a speaker-blind translator merges concurrent contributions (Figure 1). \sys{} therefore conditions the Thinker on a speaker stream and has it name the speaker first.* |
| 3.3 | *Committed text is heard only after it is spoken, and one speaker's blocks are spoken in turn, so a stream that receives more speech than its source time builds a queue; in B, two thirds of the delay arise there (Figure 1).* |
| 3.4 | *A rendering also delays every later block of its stream; playback alignment (PA) prices this by scoring whole playback plans with a linear-chain CRF.* |
| 3.5 | *The CRF needs future release times, so it runs only in training, where its posterior becomes the Talker's training target.* |
| 3.6 | *Inspired by cpWER and tcpWER, we propose cpASR-BLEU and tcpASR-BLEU to score translated speech for content and speaker attribution, without and with time constraints.* |

每节结尾的承接句也压成一句。

### 3.2 恢复 3.4 的数学推导，并在正文写出 Proposition 2

在 3.4 的 *A latency budget instead of a tuned price* 一段中，按以下顺序写：

1. **动机，一句**：*Hand-tuning $\lambda$ would need a sweep per direction and its scale has no meaning; a budget in seconds per unit is interpretable, and Proposition 2 shows it fixes $\lambda$ uniquely while keeping the same CRF and inference.*
2. Eq. (9)（$W$ 改成 $W_\mathcal{S}$）。
3. 拉格朗日函数的极小点是 Gibbs 形式：$q_\lambda\propto p_0\,e^{-\lambda\Phi_\mathcal{S}}=\prod_{\mathcal{U}}p_\lambda(\cdot\mid\mathcal{U})$。原因是 $\Phi_\mathcal{S}$ 可以按块分解，所以这一项只是在每个节点势上减去 $\lambda L_i$。这正好解释了为什么延迟的价格会出现在 Eq. (6) 里。
4. 对偶函数与导数：$\mathcal{D}(\lambda)=-\sum\log Z_\lambda/Z_0-\lambda bW_\mathcal{S}$，$\mathcal{D}'=\mathbb{E}_{p_\lambda}[\Phi_\mathcal{S}]-bW_\mathcal{S}$，$\mathcal{D}''=-\mathrm{Var}\le0$。
5. **Proposition 2（陈述）**：Eq. (9) 的解是共享 $\lambda^\star$ 的 CRF；若 $p_0$ 已满足预算，则 $\lambda^\star=0$，否则 $\lambda^\star$ 是 $\mathbb{E}_{p_\lambda}[\Phi_\mathcal{S}]=bW_\mathcal{S}$ 的唯一根，可用二分求出。证明见附录 D。
6. 一句话说明**用处**：同一次前向–后向就能算出约束后的后验；求 $\lambda$ 只需一维二分，并保证收敛。

此外：
- 消息改为 §2.3 的向量形式。
- 3.4 加一句建模近似：候选是采样的，backlog 做了离散化，“精确”指在该状态空间内精确。
- 3.5 的 “one M-step of posterior-regularised EM” 改成 “analogous to …”。
- Proposition 1 和 3 只在正文行内引用。

### 3.3 3.6 只保留我们提出的指标

- LAAL（Papi et al. 2022）和 EndOffset（SimulEval）移到 Setup 的 Metrics 段，并加引用。
- 3.6 写 cpASR-BLEU 和 tcpASR-BLEU 各一个公式，转写和参考使用新记号 $\bm{o}_k$、$\bm{g}_\varrho$。
- 末尾加一句：*We will release the scorer.*

---

## 4. 实验部分（按 SEE 的组织方式，与 Method 对应）

是的，v16 按 SEE 的方式组织：先 Setup，再主结果，然后每节分析回答一个关于方法的论点，最后一节是 Ablation Analysis。

| v16 小节 | 证据 | 对应的 Method | 能否证明 |
|---|---|---|---|
| 4.1 Experimental Setup | — | — | — |
| 4.2 Main Results | Table 1 | 整个系统（3.2–3.5），并用 3.6 的指标打分 | 能。所有集合上 cp/tcpASR-BLEU 都最高；同数据级联与 B 的对比说明“先翻译、后归属说话人”不够 |
| 4.3 Where the Listener-Side Delay Is Removed | 时延分解表 | 3.3 的队列模型 Eq. (3)、(4) 和语速 Eq. (5) | 能。延迟的减少全部来自排队等待和最后一块，load 降到约 1，正是 Eq. (3) 预测的机制 |
| 4.4 Out-of-Domain Generalisation and Robustness | 稳健性表 | 3.6 的指标是否可靠，以及 3.3–3.5 能否泛化 | 能。第二识别器和 COMET 结论一致；锁定的 OOD 会议上方向相同 |
| 4.5 Ablation Analysis，*Speaker tokens* 段 | 重叠说话表 | 3.2 的 Eq. (1)、(2) | AMI 上能（重叠单元得分和并发召回 CR 都上升）；AliMeeting 只有 CR 上升，照实写 |
| 4.5 Ablation Analysis，*Playback-layer components* 段 | ablation 表 | 3.3 Eq. (5)、3.5 的 PA Talker、Eq. (10)、按保真度选择 | 能证明每个组件都有贡献。措辞克制：PA Talker 在每个集上进一步缩短语音，在 Zh→En 上提高质量 |

**三处对应不足，v16 的处理**：
1. **时延预算（3.4 的 Eq. 9 / Prop. 2）没有单独的实验**：预算只跑了一个值。v16 在 4.5 加一句：λ* 由预算自动确定（0.50 / 0.70），两个方向都没有调参。Limitations 已经写了“一个预算值”。不做新实验。
2. **3.6 的新指标缺少有效性证据**（SEE 用整节验证它的指标）。可以用已有数据补一句：在 AliMeeting 上，我们的 tcpASR-BLEU 给四个系统的排序（MIST > B > X-Translator > SeamlessStreaming）与 DiariST 官方的说话人归属 BLEU（SAt，附录 Table 17）**完全一致**。这句放在 4.4，引用附录。另外，级联系统用自己猜的标签时 text tcpBLEU 只有 10.9 / 24.4 / 11.4（附录 F），说明这类指标确实会惩罚错误的说话人归属，也可以加一句。
3. **R3′ 不放正文**（用户决定）。相应地，正文不写会被 R3′ 反驳的结论。

**还要不要从附录再搬实验进正文？** 不需要。正文已有 5 个结果图表，加上两张方法图共 7 个浮动体，与 SimulS2ST-Omni 相当。

---

## 5. 结尾三部分压到半页以内（同第一轮）

- Conclusion：约 90 词。
- Limitations：约 250 词，3 段（Evaluation / Method / System）。
- Ethics：约 60 词。
- 耗时实测数字移到附录 C。

按 ACL 规定，只有 Conclusion 计入 8 页正文；Limitations 和 Ethics 不计入，所以压缩它们不会给正文腾出位置。

---

## 6. 正文能不能放下（第 2 问）

| 变化 | 行数（估计，正文） |
|---|---|
| 恢复 Gibbs 形式、对偶、Proposition 2 及其动机和用处 | +16 |
| 向量形式的消息（转移矩阵定义） | +2 |
| 3.6 指标公式和 “release” | +3 |
| 4.4 指标一致性的一句、4.5 预算的一句 | +3 |
| **合计增加** | **约 +24** |
| 压缩 Problem Formulation | −6 |
| 压缩 6 个小节的开头和结尾 | −8 |
| LAAL/EndOffset 移到 Setup | −2 |
| 第 4、5 章合并，小节合并 | −6 |
| Figure 2 缩到 0.85 倍宽 | 约 −6 |
| caption 再精简 | −3 |
| **合计减少** | **约 −31** |

预计能放下，还剩约 7 行余量。

如果仍然超页，按以下顺序处理，不动方法：
1. 稳健性表的 COMET 行移到附录；
2. Related Work 精简两三行（它不在受保护范围内）。

---

## 7. 附录修改（同第一轮，再加一个待定项）

- **D**：补充 CRF 的正式因子分解定义、建模假设（采样候选、80 ms 格点、上限 8 s、向上取整）、对数空间实现、二分细节（λ ∈ [0, 50]，二分 40 次）和与暴力枚举的对照验证。
- **E**：cpASR-BLEU / tcpASR-BLEU 的完整形式化定义（记号、单元构造、匹配、归属规则、插入、平局、汇总、分词、TTS 合并方式），外加 Algorithm 2 和一个小例子，并写明会公开评测脚本。
- **C**：接收原 Computation 段的耗时数字。
- **G（待你定）**：是否把完整析因表（R0–R5，含 R3、R3′）放进附录。正文不放。建议放进附录：不占正文页数，审稿人问到时有据可查，可以避免“选择性报告”的质疑。如果你不同意，就完全不放。

## 8. 执行步骤（确认后执行）

1. 新建 `paper/gstar_si_v16/`，从 v15 复制过去，v15 不做任何修改。
2. 统一符号（§2）。
3. 修改 Method（§3）。
4. 合并实验部分并改写（§4）。
5. 修改附录（§7）。
6. 压缩结尾三部分（§5）。
7. 编译，检查以下各项：
   - 正文 ≤ 8 页；
   - 没有未定义引用；
   - 没有 overfull box；
   - 全文 grep 不到 distil；
   - 外部系统没有 EndOffset；
   - 只读每段首句时能读通；
   - 符号表里每个符号都在正文有定义和维度。
8. 打包 zip，commit 后 push。
