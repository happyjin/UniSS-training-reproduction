# MIST 论文 v16 修改方案（基于 gstar_si_v15，待确认）

2026-11-17（布拉格时间）。v15：`paper/gstar_si_v15/`（commit 88bd508ad）。不做新实验；新增的正文内容只用已有 JSON 和已有表格。
不改动：abstract、introduction、Figure 1（同 v15）。

---

## 1. 结尾三部分压缩到半页以内（第 1 问）

| | SimulS2ST-Omni | SEE | v15 |
|---|---|---|---|
| Conclusion | 约 0.45 栏 | 约 0.3 栏 | 69 词，约 0.2 栏 |
| Limitations | 约 0.5 栏，3 段 | 约 0.65 栏，3 段 | **510 词，7 段，约 1.3 栏** |
| Ethics | 约 0.25 栏 | 无 | 80 词，约 0.2 栏 |
| 合计 | 约半页 | 约半页 | **约 0.85 页** |

v16 目标：三部分合计约 1 栏（约 420 词）。
- **Conclusion**：约 90 词，三句。第一句总结问题，第二句总结方法（说话人优先解码、播放队列 CRF、两个新指标），第三句总结证据（三个测试集、OOD 和锁定测试）。可加半句 future work。
- **Limitations**：约 250 词，压成 3 段。
  1. **Evaluation**：只有自动指标、没有听测；En→Zh 参考是机器翻译；Zh→En 源语音是合成的；AliMeeting 测试集在开发中被多次打分，只有 8 场锁定会议是盲测；diarizer 见过 AMI/AliMeeting 的训练数据。
  2. **Method**：候选是采样得到的，backlog 做了离散化；势函数权重是固定的；Talker 的训练只沿 base-rate 的 backlog 轨迹进行，只做了一轮；只用一个随机种子、一个预算值；更快的语音带来更多停顿。
  3. **System**：已提交的输出不再修改，所以错的说话人标签也会被念出来；没有单人发言段的说话人拿不到 voice prompt；每个说话人一个声道，重叠时会同时听到多个声音；时延不含计算时间。
- **移到附录**：Computation 段里的实测耗时数字移到附录 C；外部系统的运行细节已在附录 F，Limitations 中只留一句。
- **Ethics**：约 60 词，三句：说明有声音克隆风险，需要征得同意并标注合成语音；Zh→En 的 AMI 语音是克隆合成的；数据集许可证。

**注意（影响第 3 问）**：按 ACL 规定，Limitations 和 Ethics **不计入 8 页正文**，计入的只有 Conclusion，而它已经很短。所以压缩这三部分主要是为了好读，**不会给正文腾出空间**。

---

## 2. 实验章节的结构（第 2 问）

| | 小节 | 组织逻辑 |
|---|---|---|
| SimulS2ST-Omni | 4.1 Experimental Protocol / 4.2 Offline S2ST / 4.3 Sentence-Level Simultaneous / 4.4 Long-form Simultaneous / **4.5 Ablation Studies**（段落：Data quality and sampling strategy；Dataset size and augmented tasks） | 按**评测场景由易到难**排列，最后一节集中放消融 |
| SEE | 4.1 Setups / 4.2 Quantifying Noise Intensity with SEE / 4.3 Effectiveness Across Noise Types / 4.4 SEE on Denoising / 4.5 Suppressing Noise with SEEN / **4.6 Ablation Analysis**（段落：Layer Selection；Sensitivity to Hyperparameters） | 按**论点**排列：指标有效 → 能泛化 → 能诊断 → 能指导方法；最后一节集中放消融 |

两篇的共同点：
1. Setup 是实验章的第一小节，不单独成章。
2. 前面几节是主结果和分析，**最后一个 Ablation 小节里用加粗段落标题**分开不同的消融。

SEE 按论点排列，这和我们的情况更接近，因为我们的贡献本身就是论点：说话人归属、播放队列、指标、泛化。

**v16 实验章**：
- **4 Experiments**
  - 4.1 Experimental Setup
  - 4.2 Main Results（Table 1）
  - 4.3 Where the Listener-Side Delay Is Removed（时延分解 + 语音性质表）
  - 4.4 Out-of-Domain Generalisation and Robustness（稳健性表）
  - **4.5 Ablation Analysis**，下分三段：
    - *Speaker tokens under overlap*（重叠说话表）
    - *Playback-layer components*（ablation 表）
    - *CRF-trained Talker vs. a rate-matched rule*（见下）

v15 的第 4、5 章合并成一章。

**附录里的实验还要不要放进正文？**
- 正文已有 5 个结果图表，8 页的空间基本用满了。**建议只加一行，不加新表**：把 v11 的“语速匹配规则”行（R3′）加进 ablation 表。
- 原因：这是 v11 的**预注册主检验**（PA Talker 对比“语速调到同样快、同样做 best-of-2 的规则层”），结果对 CRF 不利：
  - MIST − R3′ 的 tcpASR-BLEU 差值为 +0.11 / +0.05 / +0.23，**只有 AliMeeting 显著**（p_Holm = 0.007）；
  - 两者 LAAL 相同（差值 ≤ 0.003 s）。
- 当时（LOG 第 3291 行）决定在摘要里写“同样快的规则层复现大部分收益”。但从 v12 起，这个结果在正文和附录里都消失了。
- 现在的累积 ablation 顺序是 R2→PA Talker→best-of-2，会让读者以为收益主要来自 CRF。审稿人很可能追问“是不是只要说快一点就行”。
- 如实写法：规则层的语速需要在 dev 上手调（shift 8）才能达到 MIST 的语速，PA 则由预算自动确定工作点；在相同语速下，CRF 训练的 Talker 只在 OOD 的 AliMeeting 上显著更好。这样写也正好支撑 OOD 的叙事。
- 质量–LAAL 图（附录 Figure 5）：只有页面还有空余时才放回正文，否则留在附录。

---

## 3. 数学部分：v15 相对 v14 删了什么、要不要恢复（第 3、4 问）

**Proposition**：从 v12 起，三个 Proposition（精确推断、时延预算、后验加权训练）**一直放在附录**，正文只引用。v15 并没有把它们从正文移走。v15 真正删掉的是**正文里的推导**：

| v14 正文里有、v15 正文删掉的 | 是否属于我的方法核心 | v16 建议 |
|---|---|---|
| 前向/后向消息 α_i、β_i 的定义式（v14 Eq. 6） | 中等：能精确说明消息的含义 | **恢复**为一个紧凑公式 |
| 拉格朗日函数 → Gibbs 形式 q_λ ∝ p_0·e^{−λΦ_S} → 恰好等于共享 λ 的 CRF（即“延迟的价格为何出现在势函数里”） | **是**，时延预算这个贡献的核心 | **恢复**，并把 **Proposition 2 的陈述放进正文**，证明留在附录 |
| 对偶函数 D(λ) 及其导数 D′ = E[Φ_S] − bW、D″ = −Var ≤ 0（v14 Eq. 10–11） | **是**，二分法成立的依据 | **恢复**为一个公式 |
| w_i 的两种等价写法（μ 的比值 = β 的形式） | 较小 | 恢复为一行 |
| 加权交叉熵的公式和 Ω、读时刻的掩码公式 P^(r)、cpASR-BLEU 的集合公式 | 不是（或已用文字说明） | 正文保持文字；cpASR/tcpASR 的完整公式放进附录（见第 4 节） |
| 符号表、具体数值 | 不是 | 留在附录 B |

**Eq. (9) 的求解过程（第 4 问）**：
- 正文只有一段结论性的文字。
- 附录 D 的 Proposition 2 有完整证明：Lagrangian → 唯一的 Gibbs 极小点 → 等于共享 λ 的 CRF → 对偶 D(λ) 及其一二阶导 → 强对偶和互补松弛 → 二分。
- **缺的是数值细节**。代码 `experiments/mist_w3po_v1/talker/playback_lattice.py::solve_lambda` 的做法是：λ ∈ [0, 50]；若 E_{p0} 已满足预算则取 λ = 0；否则二分 40 次，取满足预算的最小 λ。v16 把这一点写进附录 D 和 Algorithm 1。

**多出来的空间放在哪里**：第 1 节已说明，压缩 Limitations 和 Ethics 不会给正文腾出位置。v16 正文要增加约 20 行：
- 恢复上表三项公式和 Proposition 2 的陈述，约 +15 行；
- R3′ 行和对应的说明句，约 +3 行；
- §3.6 加上指标公式，约 +3 行。

需要从下面这些地方腾出：
- Setup 并入实验章，少一个章标题（约 −3 行）；
- 实验小节合并（约 −4 行）；
- LAAL/EndOffset 从 Method 移到 Setup（约 −2 行）；
- Figure 2 缩小到 0.85 倍宽（约 −6 行当量）；
- 各表 caption 再精简（约 −3 行）。

**建议全部用在 Method**，因为恢复的是核心推导；实验部分的增量只有 R3′ 那一行。如果还超页，把稳健性表的 COMET 行移到附录。

---

## 4. Speaker-Attributed Speech Metrics 一节（第 5 问）

- **LAAL 不是我们提出的**（Papi et al., 2022）。**EndOffset 也不是**，它是 SimulEval（Ma et al., 2020）的语音时延指标，我们只是按单元计算。所以两者都**移出 Method**，放到 Setup 的 Metrics 段，并分别引用 Papi 2022 和 ma2020simuleval（bib 里已有）。
- Method 的 §3.6 只保留我们提出的两个指标，标题改为 **“cpASR-BLEU and tcpASR-BLEU”**。第一段写明：受说话人归属识别指标 cpWER（Watanabe et al., 2020）和 tcpWER（von Neumann et al., 2023）启发，我们为语音翻译提出这两个指标。正文各给一个公式：
  - cpASR-BLEU：在最优一对一说话人匹配 σ* 下计算 corpus BLEU；
  - tcpASR-BLEU：加 collar 的单元分配规则。
- **附录现状**：附录 E 只有文字描述，**没有公式，也没有算法**，有些细节读者无法复现，例如：
  - chrF 的参数；
  - “重叠最多”怎么度量、平局怎么处理；
  - 犹豫词和窗口外语音怎么排除；
  - 各会议怎么汇总成 corpus BLEU；
  - TTS 两次运行怎么合并。
- **v16 的附录 E 改成完整的形式化定义**，读者照着就能实现：
  1. 记号：会议 m；输出流 k 的转写片段，带时间 (s, e) 和文本；参考话语，带说话人 ϱ、时间和是否计分。
  2. 参考单元的构造：同一说话人间隔小于 1 s 的话语串成一个单元（公式）。
  3. 说话人匹配：用 sacreBLEU 默认 chrF 打分，Hungarian 算法求 σ*（公式）。
  4. 片段归属规则：先把片段的时间区间两边各加 2 s collar，找它重叠时长最长的计分话语。
     - 若该话语属于 σ* 匹配到的说话人，片段归入这个话语所在的单元。
     - 若不属于，就在匹配说话人自己的单元里找重叠最长的那个，归入它。
     - 两步都找不到时，片段算插入。平局按时间顺序取第一个。
  5. 两个指标的定义：cpASR-BLEU 和 tcpASR-BLEU 各写成一个 corpus-BLEU 公式，并写明分词器（13a / zh）、各会议的统计量合并方式、TTS 两次运行的合并方式。
  6. 一个 **Algorithm 2** 伪代码框，外加一个小例子：一句话被分给了错误的说话人，分别算两个指标各被扣多少。

  **需要你决定：是否在论文里承诺公开评测脚本**（建议承诺，便于别人使用新指标）。

---

## 5. CRF 是否是严格的数学模型、推导在哪里（第 6 问）

**结论：是严格的线性链 CRF**，推导和证明都在附录 D。
- Eq. (7) 是给定观测 𝒰（各块文本、释放时间、候选集合）时，在状态序列 z 上的 Gibbs 分布。
- 它能分解成节点因子 exp φ_i(z_i) 与成对因子 𝟙[h_{i+1} = g_i(h_i, c_i)] 的乘积：成对因子的对数势只取 0 或 −∞，也就是确定性的 FIFO 约束。归一化常数是 Z_λ。这正是 Lafferty et al. (2001) 定义的线性链结构。
- 前向–后向是精确的：Prop. 1 有证明；实现在对数空间；单元测试与暴力枚举结果一致（test_playback_lattice.py）。
- 后验正则的解仍然是 CRF：Prop. 2。
- 训练目标的分解：Prop. 3。

**需要如实说明的近似**（v16 写进 3.4 的一句话和附录 D 的“Modelling assumptions”段）：
1. 势函数权重 η_ℓ、η_f 是固定的，不是用条件似然学出来的。这个 CRF 只用于推断，不做参数学习。
2. 状态空间是近似的：候选是有限次采样；backlog 被离散到 80 ms 格点、上限 8 s，并且向上取整。代码注释记录了这一点：在很长的链上，向上取整平均会累积 +0.28 s 的漂移。“精确”只是指在这个状态空间内精确。
3. 把训练称为“一步后验正则 EM”只是类比：E 步里的“模型”是 base Talker 的似然加上一个外部识别器给的保真度分，不完全是 PR-EM 的定义。正文改成 “analogous to one M-step of posterior-regularised EM”。
4. Talker 只在 base-rate 的 backlog 轨迹上取条件分布训练，没有对 backlog 做边缘化。Limitations 中已写。

v16 附录 D 还要补充：
- 用因子分解写出 CRF 的正式定义（一个公式）；
- 对数空间实现和复杂度；
- 二分的数值细节；
- 与暴力枚举的对照验证。

---

## 6. v16 执行步骤（确认后执行）

1. 新建 `paper/gstar_si_v16/`，从 v15 复制过去。v15 不做任何修改。
2. **Method**：
   - 3.4 恢复消息定义、Gibbs 形式和对偶推导，正文写出 Proposition 2 的陈述；加一句建模近似；“M-step”改为类比措辞。
   - 3.6 改为只讲 cpASR-BLEU 和 tcpASR-BLEU，第一段写明受 cpWER/tcpWER 启发，各给一个公式；LAAL/EndOffset 移到 Setup，引用 Papi 2022 和 SimulEval。
3. **实验**：第 4、5 章合并成 “4 Experiments”（4.1–4.5）；4.5 Ablation Analysis 分三段；ablation 表加入 R3′ 行（数据来自 factorial_test_v1.json，含 p_Holm）。
4. **附录**：
   - D 补充 CRF 的正式定义、建模假设和二分细节；
   - E 改成 cpASR-BLEU / tcpASR-BLEU 的完整形式化定义，加 Algorithm 2 和一个小例子；
   - C 接收原 Computation 段的耗时数字；
   - G 放入完整的 factorial 表（R0–R5，含 R3、R3′）。
5. **结尾三部分**：压到约 420 词，约半页。
6. 编译（Overleaf 用的 pdflatex），检查：正文 ≤ 8 页；没有未定义引用和溢出；全文 grep 不到 distil；正文中没有外部系统的 EndOffset。然后打包 zip，commit 后 push。

## 7. 需要你确认

1. **R3′（语速匹配规则）放进正文 ablation 表并如实说明。** 建议放；这是预注册的主检验，审稿人一定会问。
2. **数学恢复范围按第 3 节**：消息定义 + Gibbs 形式 + 对偶公式 + 正文写出 Proposition 2，其余 Proposition 只引用。建议是。
3. **实验章合并成 “4 Experiments”，最后一节是 Ablation Analysis。** 建议是。
4. **是否在论文中承诺公开 cpASR-BLEU / tcpASR-BLEU 评测脚本。** 建议承诺。
