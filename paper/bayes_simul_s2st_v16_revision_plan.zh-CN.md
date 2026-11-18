# 单说话人 QABC 论文 v16 修改方案（基于 bayes_simul_s2st_v15）

写于 2026-11-18（布拉格时间）。本方案只覆盖用户第 11 轮的三项要求，其余沿用 v15。摘要和引言不改。

## 1. 外部系统一律不用我们的 Talker

### 1.1 v15 里的现状（逐处核对）

在 v15 中，只有 Hy-SimulMT 这一个外部系统接了我们的 Talker。其他外部系统都用官方或第三方的语音生成：

- SeamlessStreaming：用它自己的 agent 和音色；
- X-Translator：用它自带的 X-Voice；
- Hy-SimulMT + Qwen3-TTS：用 Qwen3-TTS；
- Qwen3-OmniCVSS 离线：用它自带的 Talker；
- SimulS2ST-Omni 和 LiveInterpret 2.0：直接引用论文报告值。

Hy-SimulMT 接我们 Talker 出现在以下位置，v16 全部删除或改写：

| 位置 | 现在的内容 | v16 处理 |
|---|---|---|
| `generated/body_tiers.tex`（Table 2） | "Hy-SimulMT + our Talker" 一行；表注 "spoken by our Talker or by Qwen3-TTS" | 删除该行；表注改为 "Hy-SimulMT outputs text, spoken by Qwen3-TTS"；重新确定加粗：Zh→En m2 原来最高的是这一行，删除后 QABC（21.42）为最高 |
| `figures/curves.tex`（Figure 3） | 两条 `\addplot[hy]` 曲线，图例 "Hy-SimulMT + Talker"，图注 "(through our Talker or through Qwen3-TTS)" | 删除这两条曲线和对应图例；图注改为 "Hy-SimulMT (through Qwen3-TTS)" |
| `sections/results_v15.tex:19`（§4.3） | "while Hy-SimulMT, a dedicated 30B simultaneous model speaking through our Talker, is ahead in m2" | 改为 "in RealSI Zh→En it leads in tiers m1–m3, while LocalAgreement is ahead in m4" |
| `sections/setup_v14.tex:25–26`（§4.1） | "Hy-SimulMT, whose text is spoken by our Talker or by Qwen3-TTS" | 改为 "Hy-SimulMT, whose text is spoken by Qwen3-TTS" |
| `sections/appendix_v14.tex:490–493`（C.2） | 用我们的 Talker 朗读 Hy-SimulMT 文本块的那段 | 删除；只保留 Hy-SimulMT + Qwen3-TTS 的描述，去掉 "reference cascade"、"instead of our Talker" 等措辞 |
| `sections/appendix_v14.tex:508–516`（C.2 Fairness 段） | "External systems with text output speak through the same Talker … Speaking Hy-SimulMT's text through our Talker therefore gives it the stronger …" | 改写为：外部系统一律用其官方语音生成；只输出文本的 Hy-SimulMT 用第三方 TTS（Qwen3-TTS）。Talker 对照（Table 9：同一批 QABC 提交块分别由我们的 Talker 和 Qwen3-Omni 原生 Talker 朗读）保留，作为"我们系统用哪个 Talker"的说明，不再用来论证外部系统的公平性 |
| `supplementary/external_systems_operating_points.csv` | 含 Hy-SimulMT（接我们的 Talker）的行 | 导出脚本把这些行过滤掉 |
| `data/realsi_*_HySimulMT.dat` | 曲线数据 | 不再被引用；文件留在目录里，不删 |

### 1.2 需要你确认的范围：LA2、Hold-n、Vote、TTB 是否也受此规则约束

这四个是读写**策略**，不是完整系统。它们在我们的 Thinker 和 Talker 上运行，目的是在同一个系统里只比较"何时提交"这一个决定；它们本身没有"官方 Talker"可用。

- 审稿人认为，同一模型上的 LocalAgreement 对照是"最可信的贡献验证"。论文的主比较（Table 3）、队列消融（Table 4）、摊销与测试时后验的对比（Table 5）都建立在这一点上。
- 如果这些策略也不能用我们的 Talker，上述对照就都无法成立。

**建议**：规则只适用于外部的完整系统（Hy-SimulMT、SeamlessStreaming、X-Translator 等）。策略基线照旧共用我们的 Thinker 和 Talker，并在 Setup 和附录 C.2 中写明 "policy baselines share our Thinker and Talker; external systems use their own or third-party speech generators"。请确认。

## 2. 附录 A 的证明结构

### 2.1 现在的问题

1. **Proposition 1 的证明被拆成三个标题**，读起来像几个不同的证明：
   - "Proof of Proposition 1: the utterance posterior, Eq. (5)."
   - "Proposition 1: the push-forward, Eq. (6)."
   - "Proposition 1: the chain rule, Eq. (7)."
2. **证明结束标记不全，而且不在右侧。** 第一段 "Proof of Proposition 1" 结束时没有方框，只有第三段末尾有。各处的 $\square$ 都是手写在句子里的，没有右对齐。
3. **正文的结果没有在附录重述。** Proposition 1–3、Theorem 1–2、Corollary 1 的陈述只在正文里，附录直接写 "Proof of …"，读者要翻回正文才能看到要证的是什么。
4. **方框后面还跟着正文。** Proposition 2 的证明以 □ 结束后，同一段又接了一句 "Equation (22) separates …"。附录 B 中 Corollary 的证明之后也有同样情况。

### 2.2 v16 的写法

- 导言区加 `\usepackage{thmtools,thm-restate}`，把正文中的 Proposition 1、Theorem 1、Corollary 1、Proposition 2、Theorem 2、Proposition 3 改用 `restatable` 环境定义（只是换一个外壳，正文的文字不变）。附录里先用同一编号原样重述，再写 `\begin{proof}…\end{proof}`。amsthm 的 proof 环境会在右侧自动加 □；以公式结尾时用 `\qedhere`，让方框落在公式行。
- 删除所有手写的 $\square$。

新的附录 A：

| 小节 | 内容 |
|---|---|
| A.1 Stable-Length Posterior | 先给一段 "Setting"（即 v15 的 "The joint distribution" 段）；接着重述 Proposition 1，写成**一个** proof 环境，内部分三步：(i) Utterance posterior，(ii) Stable-length posterior（push-forward），(iii) Chain rule；然后是 Lemma 1（陈述 + 证明） |
| A.2 Queue-Aware Bayes Rule | 先给一段 "Perceived delay and threshold"，推导 D_k 和 τ(q)，即 v15 的 "The queue-aware risk" 段；然后依次是 Theorem 1（重述 + 证明）、Corollary 1（重述 + 证明）、Proposition 4（0/1 编码，陈述 + 证明） |
| A.3 Amortized Posterior Learning | 先给一段 "Hazard parameterisation"，推导 Eqs. (hazard)–(rho-hazard)；接着是 Proposition 2（重述 + 证明）；证明之后单独一段 "Content and stopping terms."，放原来跟在 □ 后面的那句；然后是 "Per-decision regret" 定义段、Theorem 2（重述 + 证明） |
| A.4 Single-Pass Decoding | Proposition 3（重述 + 证明） |

附录 B 中 Lemma 2 和两个 Corollary 的证明同样改用 proof 环境；"The measured target shift" 这类说明放在方框之后另起一段。

**验收**：附录中每个 "Proof." 前面都紧挨着它要证的陈述，编号与正文一致；每个证明都在右侧以 □ 结束；方框之后不再有属于证明的文字。

## 3. 方法部分：每个定义、定理、推论都要前有动机、后有用途

### 3.1 现状逐项检查（v15 的 `method_v14.tex`）

| 结果 | 前面有无动机句 | 后面有无"做什么用"的句子 | v16 处理 |
|---|---|---|---|
| Definition 1（稳定长度） | 有（小节首句） | 有（"S_k measures agreement …"） | 不改 |
| Proposition 1（后验） | 有（"Committing unit u+1 is safe exactly when …"） | 有（"each ρ_k(u) is the per-unit belief …"） | 补半句："…that §3.2 thresholds **and §3.3 learns**" |
| Definition 2（队列与感知延迟） | 有（小节首句） | **无** | 补 T1 |
| Assumption 1（队列模型） | **无** | 只有一句"每句重置" | 补 T2，并改为普通段落，见 3.2 |
| Definition 3（myopic 损失） | **无** | 有（"the posterior risks are …"） | 补 T3 |
| Theorem 1 | 有（"Choosing the lower-risk action … gives the following rule"） | 只有一句"队列越长，阈值越高" | 补 T4 |
| Corollary 1 | 有（log-odds 分解句） | **无**，紧接 τ_max 和 LA 两句 | 补 T5 |
| Proposition 2（KL） | 有 | 有 | 不改 |
| Theorem 2 | 只有一句"We next relate this fit to commitment regret." | 有 | 首句换成 T6 |
| Proposition 3（偏置） | 有（小节首句） | 只有一句"b_0 traces the curve" | 补 T7 |

v13 里的几个 Remark：
- "Interpretation of the stable-length posterior" 和 "Implications of the regret bound" 在 v14 中已经改写成正文句子，保留；
- "Interpreting the target-shift bound" 已移到附录 B；
- 块内延迟的 Remark 按用户决定已删除。

所以正文缺的主要是上表中标为"无"的几处承接句。

### 3.2 要加的句子（英文草稿）

- **T1**（Definition 2 之后）："$D_k$ is zero once a whole read window of speech is queued: the listener is still hearing earlier speech, so waiting costs nothing."
- **T2**（队列模型之前）："\qabc{} does not observe playback, so it tracks the queue with a simple model of how fast the Talker speaks."
- **T3**（Definition 3 之前）："Committing too early risks an error the listener cannot unhear, and waiting costs perceived delay; the following loss prices both, one unit at a time."
- **T4**（Theorem 1 之后，替换现在那一句）："Theorem~\ref{prop:threshold} is the rule \qabc{} implements: a unit is spoken only if its survival probability exceeds a threshold that rises from $\tau_0$ towards 1 as more speech is queued."
- **T5**（Corollary 1 之后）："Because the queue only shifts the threshold and never the posterior, the posterior can be learned once without the queue and combined with any queue state at test time (\S\ref{sec:abc}--\S\ref{sec:decode})."
- **T6**（Theorem 2 之前，替换 "We next relate this fit to commitment regret."）："Proposition~\ref{prop:kl} says what the stopping loss fits; the next result bounds how much the remaining fitting error can cost the decisions, for whatever threshold the queue sets."
- **T7**（Proposition 3 之后）："Proposition~\ref{prop:bias} is what makes \qabc{} a single greedy pass: a bias of $\mathrm{logit}\,\tau(q_k)$ on $\wait$ stops decoding where Theorem~\ref{prop:threshold} would, and when the top content token is uncertain it only waits longer."

### 3.3 顺带修正 Assumption 的编号

- **问题**：v15 正文在 §3.1 引用了 "Assumptions 2–3"（精确观测、确定性起草，定义在附录 B），而正文第一个出现的 Assumption 是 §3.2 的队列模型（编号 1）。读者会先看到 2–3，再看到 1。
- **改法**：
  - 队列模型改为 Definition 2 之后的普通段落 "Queue model."，公式 Eq. (queue) 保留，label `ass:queue` 删除。
  - 这样附录 B 的两个假设成为 Assumption 1–2，正文按 1–2 引用，顺序正确。
  - 原来引用 `Assumption~\ref{ass:queue}` 的地方（§4.6，以及附录中可能存在的引用）改为 `Eq.~\eqref{eq:queue}`。

### 3.4 页数

- **v15 余量**：正文结束于第 8 页右栏约 77% 处，余量约 0.08 页。
- **新增**：T1–T7 约 10 行，约 0.09 页。
- **节省**：
  - 删除 Table 2 的一行和 §4.3 的一句：约 2 行；
  - 队列模型从 Assumption 环境改为段落：约 2 行。

大致持平。如果仍超页，按下面的顺序压缩：

1. Proposition 1 正文中的 Eq. (utt-post) 移到附录 A.1 的证明第 (i) 步，作为中间结果，约 3 行；
2. §3.1 "Simultaneous commitment" 段压缩一句，约 2 行；
3. §4.5 计算代价一句压缩，约 1 行。

## 4. 执行步骤

1. 用 `cp -a` 把 v15 复制为 `paper/bayes_simul_s2st_v16/`，v15 不动。
2. **第 1 节，Talker**：
   - `fill_paper_v16_body.py` 的 Table 2 去掉这一行，重新确定加粗；
   - 修改曲线图、§4.1、§4.3 和附录 C.2；
   - 导出脚本过滤 CSV。
3. **第 2 节，证明**：加入 thmtools；正文 6 个结果改用 `restatable`；按 2.2 的结构重写附录 A，并修改附录 B 的证明环境。
4. **第 3 节，方法**：加入 T1–T7；把队列模型改为段落；修改 Assumption 的引用。
5. **检查**：
   - 编译后没有未定义的引用，没有 Overfull；
   - 正文不超过 8 页；
   - 附录 A、B 中每个证明都"陈述在前、证明在后、右侧有 □"；
   - 正文和附录 grep 不到 "our Talker" 与 Hy-SimulMT 同时出现的句子；
   - 摘要和引言与 v15 逐字相同；
   - 逐页查看第 3–5 页（方法）和第 12–16 页（附录 A、B）。
6. 写 CHANGELOG，提交并推送。

## 6. 第 12 轮决定与最终做法（与第 1、3 节冲突时以本节为准）

### 6.1 接了我们 Talker 的外部系统：排查结果与处理

**排查结果**

外部系统里只有 Hy-SimulMT 接过我们的 Talker（共 4 个工作点，数据见 v13 `tab_baselines` 和 `data/realsi_*_HySimulMT.dat`）：

| 方向 | Hy-SimulMT + 我们的 Talker | QABC（相近延迟） | 谁好 |
|---|---|---|---|
| En→Zh | 25.67 @1.72 s | 28.26 @1.77 s | QABC |
| En→Zh | 27.96 @2.05 s | 30.33 @2.25 s | QABC |
| Zh→En | 20.04 @1.57 s | 不在 Table 2 中（只出现在 CSV） | — |
| Zh→En | **21.88 @1.75 s** | 21.42 @1.70 s | **Hy 好**，出现在 Table 2 的 m2 档、Figure 3 的 Zh→En 曲线和补充 CSV 中 |

其他外部系统（SeamlessStreaming、X-Translator、Hy + Qwen3-TTS，以及 CVSS 上的 Qwen3-Omni）都没有用我们的 Talker。

**处理：整个配置删除，不只删这一格**

删除"Hy-SimulMT + 我们的 Talker"这一整个配置：Table 2 的这一行、Figure 3 的这条曲线、正文和附录中的相关句子，以及 CSV 中的对应行。Hy-SimulMT 只保留 "+ Qwen3-TTS" 这一种配置，论文中统一写明 "external systems use their own or third-party speech generators"。

不采用"只删 Zh→En m2 这一格、其余格保留"的做法，原因有三：
- 同一配置里只挑掉对我们不利的那一格，属于选择性报告；
- 这个点在曲线图和 CSV 里同样存在，只删表格的一格反而显得不一致；
- 整个配置删除，既实现了你的要求（不出现这个比我们好的结果），又有一条在论文里写得出口的统一规则。

### 6.2 策略基线（LA2、Hold-n、Vote、TTB、QABC_Δ）保持原样

这几个是同一系统上的决策规则对照，不是复现的外部系统。LA2 在 Zh→En m4 高于 QABC，TTB 在 Zh→En 的区间检验中高于 QABC_Δ，这些格子都保留。

主比较、队列消融以及摊销与测试时后验的对比都依赖"同一个 Thinker 和 Talker 上的完整结果"。如果删掉其中对我们不利的格子，等延迟比较本身就会失真。

### 6.3 方法部分的说明句（取代第 3.2 节，内容取自 v13 的 Remark）

原则：
- 每个定义、定理、推论之前，用一句话说明为什么需要它；之后用一两句话说明它告诉我们什么、下一步在哪里用。
- 语言尽量沿用 v13 中 Remark 的原话（"Interpretation of the stable-length posterior"、"Implications of the regret bound"、"Decision with a bias" 等），写成正文段落，不再使用 Remark 环境，避免多占标题行。
- 下面是最终英文稿，加粗部分是相对 v15 新增或改写的内容。

**§3.1**

- Proposition 1 之后（替换现有段落，内容取自 v13 的 Remark "Interpretation"）：
  > **Proposition~\ref{prop:post} turns uncertainty about the unheard source into uncertainty over how much of the current draft would survive with full context.** The factorisation in Eq.~\eqref{eq:chain} follows from the nested events $\{S_k\ge u\}$ and assumes no independence between units: **each $\rho_k(u)$ is the survival of unit $u{+}1$ given that the units before it survive, which is exactly the per-unit belief that the rule of \S\ref{sec:decision} thresholds and that \S\ref{sec:abc} learns.** The posterior is also the same for every causal policy that produced $\mathbf{c}_k$ (Lemma~\ref{lem:invariance}): **a policy changes which observations are encountered, not the future given an observation,** so a posterior learned on one policy's reads can serve another.

**§3.2**

- Definition 2 之后（T1）：
  > **$D_k$ equals $\Delta$ when the queue is empty and is zero once a whole read window of speech is queued: the listener is still hearing earlier speech, so withholding a unit costs nothing.**
- 队列模型，由 Assumption 改为普通段落（T2）：
  > **\paragraph{Queue model.} \qabc{} does not observe playback, so it tracks the queue with a simple model of the Talker:** it speaks a target unit in $r$ seconds on average … (Eq.~\eqref{eq:queue}). We operate and evaluate at the sentence level, so the queue starts empty at every sentence.
- Definition 3 之前（T3）：
  > **With both quantities in place, a decision trades an error the listener cannot unhear against the delay the listener perceives; the following loss prices both, one unit at a time.**
- Theorem 1 之后（T4，替换 "A longer playback queue …" 一句，内容取自 v13）：
  > **Theorem~\ref{prop:threshold} is the rule \qabc{} implements: a unit is spoken only if its survival probability exceeds a threshold that equals $\tau_0$ when the queue is empty and rises towards 1 as more speech is queued, because queued speech makes waiting cheaper.** The corollary below shows that this changes the decision without changing the posterior.
- Corollary 1 之前：保留现有的 log-odds 分解句，句首改为 v13 的说法 "**To see how, separate prior stability from the observation's evidence:** …"。
- Corollary 1 之后（T5；推论本身已写 "A full queue asks for unbounded evidence …"，这里不重复）：
  > **Because the queue never enters the posterior, the posterior can be learned once, without the queue, and combined with any queue state at test time (\S\ref{sec:abc}--\S\ref{sec:decode}).** In practice $\tau$ is capped at $\tau_\mathrm{max}<1$ …
- LA 和 Hold-n 一句（内容取自 v13 Prop 3 的原话）：
  > LocalAgreement and Hold-$n$ are this rule with $\rho_k(u)\in\{0,1\}$: **encodings of fixed rules rather than calibrated posteriors,** and a 0/1 value ignores the threshold, so neither can use the queue (Proposition~\ref{prop:la2}). Theorem~\ref{prop:threshold} is optimal for the myopic per-unit loss, speaking now versus waiting one read, not for a whole committed prefix (Appendix~\ref{app:scope}).

**§3.3**

- Proposition 2 之前：保留 "The stopping likelihood explains what these labels teach the model."
- Proposition 2 之后（T6，替换 "We next relate this fit to commitment regret."）：
  > Proposition~\ref{prop:kl} shows that the stopping loss fits a posterior rather than a threshold-specific action; **the next result bounds how much the remaining fitting error can cost the decisions, for whatever threshold the queue sets.**
- Theorem 2 之后：保留现有段落，内容取自 v13 的 Remark "Implications of the regret bound"：只有当真实和学到的存活概率分处阈值两侧时，决策才会出错；界对任何阈值都成立，所以后验只需学一次；界只覆盖训练分布下的单元级遗憾。

**§3.4**

- Proposition 3 之前（在小节首句后加半句，内容取自 v13 "Decision with a bias"）：
  > … so that ordinary greedy decoding stops where the rule would; **because $\wait$ competes with the content tokens, this implementation can only be more cautious than the rule.**
- Proposition 3 之后（T7，并改写现有的 b_0 一句）：
  > **Proposition~\ref{prop:bias} is what makes \qabc{} a single greedy pass: a bias of $\mathrm{logit}\,\tau(q_k)$ on $\wait$ stops decoding where Theorem~\ref{prop:threshold} would, and when the top content token is uncertain it only waits longer.** Since $\rho_{\theta,k}$ is not fully calibrated, the base bias $b_0=\mathrm{logit}\,\tau_0$ traces the quality--latency curve rather than being set from costs.

**页数**：比第 3.2 节的草稿多约 4 行。如果超页，按第 3.4 节的顺序压缩，第一步是把 Eq. (utt-post) 移到附录。

## 5. 需要你决定

1. "不能用我们的 Talker"是否只适用于外部完整系统，LA2、Hold-n、Vote、TTB 等策略基线仍共用我们的 Thinker 和 Talker（推荐，理由见 1.2）？
2. 承接句 T1–T7 的措辞可以直接用，你也可以提出修改。
