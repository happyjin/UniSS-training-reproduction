# MIST 论文 v15 修改方案（基于 gstar_si_v14_overleaf_v1，待确认）

2026-11-17（布拉格时间）。源文件：`paper/gstar_si_v14_overleaf_v1/`（S3 zip 解压），编译结果 `paper/gstar_si_v14_overleaf_v1/main.pdf`。
**不改动**：abstract、introduction、Figure 1（`figures/motivation.tex` 和 PDF 图，caption 也不改）。
**不做新实验**：所有正文新增的图表都直接取自 v14 已有的表格数字。

---

## 0. 已完成

- `aws s3 cp …/gstar_si_v14_overleaf_v1.zip paper/` → 解压到新目录 `paper/gstar_si_v14_overleaf_v1/`（没有覆盖任何已有文件）。
- 编译用 TinyTeX 的 pdflatex + bibtex，和 Overleaf 默认引擎相同，编译通过：22 页，没有未定义的引用或文献。
- 编译后发现**正文超页**：Conclusion 从第 9 页开始，而 ACL long 正文（含结论）上限是 8 页。v15 要压回 8 页以内。

---

## 1. 两篇参考论文的写法

| | SimulS2ST-Omni（arXiv 2607.19810v2） | SEE（ACL 2026 long） | 我们 v14_overleaf_v1 |
|---|---|---|---|
| Method 篇幅 | 约 1.7 页（p3–p5 上） | 约 1.7 页（p3–p5） | **约 3.7 页**（p3 下–p7 上） |
| Method 小节 | 3.1 Problem Setup and Formulation / 3.2 Trajectory Construction / 3.3 Backbone Choices | 3.1 Noise Subspace Identification（开头一个 Notation 段）/ SEE Calculation / SEEN Construction | 6 个小节，13 个公式 |
| 公式数 | 5 | 12 | 13，另有附录公式 3 个 |
| 符号 | 首次出现时定义；维度用字母 | `A_ℓ(x) ∈ R^{T(x)×d_ℓ}`，紧接一句“T(x) 是时间步数，d_ℓ 是隐层宽度”；阈值只写符号 α、δ | 维度写成具体数（`R^{16000T}`、d=2048、S=8、K=5、251 个时间 token……），另外还在附录列了 Table 3 符号表 |
| 超参数位置 | 附录 B **Training Implementation Details and Hyperparameters**（Table 8） | 附录 A.4 SEE Intervention and Hyperparameters | 散落在正文：权重 0.5/5、λ\*=0.50/0.70、8 个候选、101 个格点、35 个 rate token、κ=(1,0.5)…… |
| 正文结果图表数 | **7**（Table 1、Fig 3、Fig 4、Table 3、Fig 5、Table 4、Table 5） | **10**（Fig 3–6、Table 1–6） | **2**（Table 1、Table 2） |
| 实验结构 | 4.1 Protocol → 4.2 Offline → 4.3 句级同传 → 4.4 长篇同传 → 4.5 Ablation | 4.1 Setups → 4.2–4.5 每节回答一个关于方法的问题 → 4.6 Ablation | 5.1 系统对比 → 5.2 时延（含 ablation） |
| 附录 | A 数据构造 / B 训练细节与超参 / C 推理解码 / D 评测协议 / E 补充结果 / F 计算感知时延 / G 种子与 ASR 敏感性 / H 人评 | A 实验设置（模型、任务、噪声合成、超参、baseline）/ B–H 补充分析、效率、有效性 | Notation / Proofs / PA 训练 / Thinker / Data / Metrics / Further results / Overlap / Analysis plans |

### 两篇如何分析结果（对应第 4 点）

1. **每节先用一句话说明这组实验检验什么。** 例如 Omni 写“This reflects the model's basic … allowing a clean comparison between the Thinker–Talker and Dec-only architectures”。
2. **用一个定性结论描述图表。** 例如“strictly dominates … across all latency tiers”“GSR degrades consistently as SEE increases”。
3. **数字只挑能支撑论点的一个，或者用区间、平均值。**
   - SEE 写“R ∈ [−0.96, −1.00]”“by 6.7% on average”。
   - Omni 写“+4.0 ASR-BLEU gain notably outpaces its +2.14 Text-BLEU gain”，用两个数的差距说明 two-stream 让语音路径更忠实。
4. **紧接一句 “This confirms / indicates that ⟨方法里的某个设计⟩ …”**，把数字解释回方法机制。
5. **主动讲代价。** 例如 Omni 承认 Dec-only 的声学质量更好，并称之为 “intelligence-versus-quality trade-off”。
6. **p 值、逐集数字、置信区间只放在表格或附录里。** 正文的 p 值只出现在 Setup 里的一句统计说明中。

v14 的结果段正好相反：
- 逐集三元组，例如 “+1.55 / +0.24 / +0.80”“0.18 / 0.15 / 0.23 s”；
- 段中夹着 p 值和 “exploratory” 标注；
- 读者要自己推断这些数字说明了方法的哪一部分有效。

---

## 2. v14_overleaf_v1 的问题诊断

1. **Method 太长，而且混入了实现细节。** 它占 3.7 页，是正文超页的主要原因。真正的创新（说话人优先的同传解码、播放队列 CRF + 后验正则 + 蒸馏、cp/tcpASR-BLEU）被下面这些内容淹没了：
   - G-STAR 式投影器的结构；
   - Hungarian 槽位映射；
   - deadline chunk 的切分规则；
   - voice prompt 的抽取细节；
   - online window 和 chunk protocol；
   - 消息定义和对偶函数的求导。
2. **符号冲突。** 把符号表删掉以后，这些冲突必须在正文里修好：
   - `α_n, β_n`（时间 token 下标，Eq.1）与 `α_i, β_i`（前向/后向消息，Eq.6–8）冲突；
   - `ω(y_u)`（token 权重）与 `π_𝛚`（Talker 参数 𝛚）冲突；
   - `x` 先表示录音，PA 一节里又写 “x now denotes the stream”；
   - `ρ` 在附录里表示参考说话人，在 Eq.13 里表示 rate 后验；
   - `π_n`（附录里的播放起点）与 `π_𝛚`（Talker）冲突；
   - `W_i / W`（目标单位数）与附录里的 `W^{(r)}`（卷积权重）冲突；
   - `c` 既表示候选，也表示 deadline chunk `c_j`。
3. **正文只有 2 个结果图表。** 方法里每个组件的证据（重叠说话、时延分解、语音性质、第二识别器和 COMET、未用过的会议）都放在附录 G–H，读者无法在正文里把“方法 → 证据”对上。
4. **小问题：**
   - Table 7、Table 9 的 Set 列是空的，没有写集名；
   - Table 8 还留着 `δ_w = 0` 的行。写对齐（WA）已经移出正文，这个符号在正文里没有定义；
   - Ablation 的 R0 En→Zh 是 31.64，Table 1 的 B 是 31.65，两者不一致；
   - Figure 2 里写死了 S=8、K=5，节号引用（§3.1/§3.2/§3.5）在 v15 里会变。

---

## 3. v15 修改方案

### 3.1 Method 新结构（第 5 点）：目标约 2 页（含 Figure 2），公式约 10 个

小节和引言里的三条贡献一一对应：
- 3.2 对应贡献 1；
- 3.3–3.4 对应贡献 2；
- 3.5 对应贡献 3。

| 小节 | 正文保留 | 移到附录 |
|---|---|---|
| **§3 开头**（2–3 句）+ Figure 2 | 总览：Thinker 写出带说话人和时间戳的译文；每个说话人有一条语音流，由 Talker 播放；播放层由 CRF 蒸馏训练，测试时不跑 CRF | — |
| **3.1 Problem Formulation** | 输入是单通道录音 x；输出是片段序列 `(k_n, a_n, b_n, y^{(n)})`；同传约束是第 r 次读时只看到 `x_{≤t_r}`；每个说话人一条输出语音流；定义“被听到的时间”（承接 Figure 1 的动机）。维度只用字母 | 采样率、帧长等数值 |
| **3.2 Speaker-Attributed Simultaneous Translation** | ① 冻结的 Nemotron 3 Diarization 给出后验 `P ∈ [0,1]^{F×S}`，每 K 帧经投影器 `g_ψ` 变成一个说话人 token，与声学 token 交错（**Eq: E**，一句话注明沿用 G-STAR）；② speaker-first、带时间戳的序列化（**Eq: y**）；③ 加权交叉熵，只说明“加重 speaker/time token”；④ 同传读写：每 Δ 读一次，说话人流只能用已定稿的帧（延迟 γ），每个 chunk 在其源语音结束 ε 之后到期，未到期就写 `⟨wait⟩`（一个公式或行内）。这一套就是 B 和 MIST 共用的 Thinker | 投影器结构（Eq.14）、零初始化；Hungarian 槽位映射；token 权重数值；LoRA 层数；chunk 切分与对齐规则；训练行构造（50% 空读、整窗/单人行混入）；Δ、γ、ε、时间 token 数、20 s 窗长 |
| **3.3 Playback Alignment over Speaker Queues** | ① 说话人语音流与播放队列：backlog 递推（**Eq: backlog**），被听到的时延（**Eq: L_i**），一句话说明“长渲染要付两次代价”；② 随 backlog 提速的基准语速（一句话）；③ CRF：势函数（**Eq: φ**，权重写成符号 `η_ℓ, η_f`）和链式分布（**Eq: CRF**）；④ 精确推断：前向-后向递推 + 边缘概率合成一个公式，并给出复杂度；⑤ 时延预算：后验正则约束（**Eq: PR**），然后一句话给结论：“解就是 λ\* 下的 CRF；期望时延随 λ 单调，所以用二分求 λ\*（Prop. 2）” | 消息的定义式（Eq.6）；对偶函数及其一二阶导（Eq.10–11）；候选数、两档语速、格点 δ/H̄、101 个值、权重 0.5/5、CTC 截断 10、预算 b 的取值、λ\*=0.50/0.70 |
| **3.4 Distilling Playback Alignment into the Talker** | 条件后验权重 `w_i`（**Eq: w**，强调 β_{i+1} 把未来排队代价带进来）；蒸馏目标 `J`（**Eq: J**，系数用 κ 符号）；一句 Prop. 3 的含义（等于把 Talker 在候选上的偏好拟合到 CRF 后验，即一步 M-step）；解码：PA Talker 读实际队列的 backlog token，采样少量候选，按 f 选；测试时不跑 CRF；一句话定义 playback layer = read-level merging + 基准语速 + PA Talker | backlog bin 数和宽度、rate token 数、κ 数值、候选筛选阈值 0.01、训练数据量、步数、checkpoint 规则、Algorithm 1 |
| **3.5 Speaker-Attributed Speech Metrics**（从 Setup 移过来，作为贡献 3） | cpASR-BLEU：每个说话人的 ASR 转写与参考按会议拼接，用 chrF 做 Hungarian 一对一匹配，未匹配的一方对空串打分（一个公式）；tcpASR-BLEU：加上 tcpBLEU 的时间约束（collar 用符号）。说明为什么需要这两个指标：给错说话人的句子会同时损失 precision 和 recall | tcpBLEU 的单元构造细节、collar 数值、ASR 模型、过滤规则 |

另外两项：
- 原 §3.6 Running on a whole meeting（online window、chunk protocol）整体移到附录 C；正文只在 Setup 里留一句“所有系统共用同一 online detector window”。
- “MIST (offline) / MIST-Simul” 这两个名字在正文 Method 里不再出现，只在重叠实验表的 caption 里说明是同配方、按方向分别训练的模型。

### 3.2 符号规则（第 1 点）

- 删除 Method 第一句 “Every symbol … Table 3”，同时删除附录的 Notation 表（参考的两篇论文都没有符号表）。
- 每个符号在第一次出现的句子里定义含义，维度只用字母，例如：
  - `A ∈ R^{N×d}` 后面紧接 “d is the Thinker's hidden size”；
  - `P ∈ [0,1]^{F×S}` 后面紧接 “over F frames and S arrival-ordered speaker slots”；
  - 候选集 `|C_i| = C`；
  - backlog 格点 `H = {0, δ, …, H̄}`。
- 所有具体数值统一放进附录 B 的 **Training Implementation Details and Hyperparameters** 大表。分组为 Thinker / Diarizer / 同传读写 / Talker 与语速 / Playback alignment / 蒸馏。
- 修好第 2 节第 2 条列出的符号冲突，建议的改法：
  - 时间 token 下标改成 `\hat a_n, \hat b_n`，α/β 只给前向/后向消息用；
  - Talker 参数改成 `ξ`；
  - PA 一节里的“一条语音流”改用 `\mathcal{U}`；
  - rate 后验 ρ 保留，附录里的参考说话人改用 `r`；
  - 附录里的播放起点改成 `o_n`；
  - deadline chunk 改称 `u_j`。

改写示例（3.2 第一段）：

> 改前：*maps the recording to acoustic tokens A=[a_1;…;a_N]∈R^{N×d}: row i is the token for the 80 ms of audio centred at t̄_i (so N≈T/0.08), and its d=2048 columns are the Thinker's hidden dimensions … S=8 speaker slots … Every K=5 rows …*
>
> 改后：*The frozen audio encoder maps x to acoustic tokens A ∈ R^{N×d}, where N is the number of tokens and d the Thinker's hidden size. The frozen diarizer returns speaker-activity posteriors P ∈ [0,1]^{F×S} over F frames and S arrival-ordered speaker slots; a projector g_ψ maps every K frames to one speaker token s_j ∈ R^d, which follows the K acoustic tokens it describes.*（数值见 Appendix B。）

### 3.3 Diarizer 引用（第 2 点）

代码里用的是 `pretrained_models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo`（`reports/gstar_si_v9/dump_sdm_posteriors.py`）：10 ms 后验均值池化到 80 ms，8 个说话人，在线档位 1.04 s。这些和论文里的 Nemotron 3 Diarization 一致：8 个说话人，有 30.4 / 1.04 / 0.32 s 三个延迟档位，构建在 Streaming Sortformer 之上。

新增 bib 条目：
```bibtex
@article{medennikov2026nemotron,
  title   = {{Nemotron 3 Diarization}: Advancing the Accuracy--Efficiency Frontier in Streaming Speaker Diarization},
  author  = {Medennikov, Ivan and Park, Taejin and Timofeeva, Tatiana and Huang, He and Polok, Alexander and
             Kostandian, Sofia and Wang, Weiqing and Gao, Dongji and Balam, Jagadeesh and Ginsburg, Boris},
  journal = {arXiv preprint arXiv:2611.12246},
  year    = {2026}
}
```

需要修改的位置：

| 位置 | 现在 | v15 |
|---|---|---|
| `sections/method.tex` 第 26 行 | A frozen Streaming Sortformer~\citep{medennikov2025streaming} | 冻结的 Nemotron 3 Diarization~\citep{medennikov2026nemotron}（基于 Streaming Sortformer 的到达顺序说话人缓存，S 个槽位） |
| `sections/setup.tex` 第 17 行 | the diarizer is Streaming Sortformer~\citep{medennikov2025streaming} | Nemotron 3 Diarization~\citep{medennikov2026nemotron}；离线 30.4 s 和在线 1.04 s 档位的说明放到附录 B |
| `figures/arch.tex` | “Streaming diarizer, S=8 slots” | “Simultaneous diarizer (Nemotron 3), S slots”；caption 同步修改 |
| 附录：Causality、Chunk protocol、Table hyper（“Diarizer preset: offline / low latency (1.04 s)”） | 只写了 “low-latency diarizer” | 写明 Nemotron 3 Diarization 的 30.4 s / 1.04 s 档位，以及 10 ms→80 ms 均值池化 |
| `sections/related.tex` | streaming diarizers …~\citep{park2024sortformer,medennikov2025streaming} | 保留原引用（这条脉络是对的），再加上 `medennikov2026nemotron` |
| `references.bib` | — | 新增上面的条目；`medennikov2025streaming` 保留，供 Related Work 使用 |

**需要补充一条披露。** Nemotron 3 Diarization 论文 §2.1 写明，它的训练数据包括 AMI 和 AliMeeting 的真实录音，以及用二者 headset 音频模拟的混合语音。它评测时用的是 16 个 AMI 会议和 20 个 AliMeeting 会议（与我们的测试集相同），因此大概率只用了训练划分，但论文没有明说是哪个划分。我们 Setup 里写的 “no AliMeeting data is used in training” 应改成：“我们的模型没有使用 AliMeeting 训练；现成 diarizer 的训练数据包含 AMI 和 AliMeeting 的训练录音”。Limitations 里也加一句。

### 3.4 实验部分重构（第 3、4 点）

**§4 Experimental Setup（约 0.5 页）**
- Datasets：一段。数据统计表移到附录 A。
- Implementation：一句话，指向附录 B。
- Compared Systems：SeamlessStreaming、X-Translator、同数据级联、B、MIST，各一句。
- Metrics：tcpBLEU、cpASR-/tcpASR-BLEU（指向 §3.5）、LAAL、EndOffset。
- Statistics：两句话，交代显著性检验方法和 Holm 校正。

**§5 Results**：每节回答一个关于方法的问题，并和 Method 小节对应。正文结果图表从 2 个增加到 **6 个**（可选第 7 个）。

| 正文图表 | 来源 | 对应的方法声明 | 正文要说的结论（数字尽量少） |
|---|---|---|---|
| **Table 1** 系统对比（现有） | 正文 Table 1 | 整体系统（§3.2–3.4） | MIST 在三个集上的 cp/tcpASR-BLEU 都最高。外部系统起步更早，但只有 MIST 质量的约 1/3 到 3/4，说明它们的低时延来自丢掉或错配的内容 |
| **Figure 3（新）** 质量–时延平面：x = EndOffset，y = tcpASR-BLEU，三个子图对应三个测试集；点为各系统，B→R1→R2→R3→MIST 画成箭头路径 | Table 1 + Table 2 的现有数字，用 pgfplots 画（与 Overleaf 兼容，不需要图片文件） | §3.3–3.4 同时改善质量和时延 | MIST 在每个集上都在 B 的左上方；路径显示 R2、R3 主要向左移，R4 主要向上移 |
| **Table 3（从附录 Table 14 提上来，压缩）** 只保留在线的两行：speaker-blind + causal relabel 对比 MIST Thinker（同配方，按方向分别训练）；列为 all / overlapped tcpBLEU 和 CR | 附录 Table 14（+ Table 15 的显著性标记） | §3.2 说话人 token 与 speaker-first 解码（贡献 1，对应 Figure 1 的 (1)） | 加入说话人 token 后，重叠单元的得分和并发说话人召回（CR）都大幅提高，事后重新标注恢复不了。**必须如实写**：AliMeeting 上 all-unit 更低、overlapped 持平（预注册检验未通过），只有 CR 提高 |
| **Table 4（从附录 Table 6 + Table 8 合并上来）** B 与 MIST × 三个测试集；列为 queue wait、last block、EndOffset、load、speaking rate、UTMOS、SIM、pauses/min | 附录 Table 6 的 test 行 + Table 8 的 B/MIST 行（去掉 δ_w 行） | §3.3 的队列模型（Eq. backlog）和 Figure 1 的 (2) | 时延的减少几乎都来自队列和最后一块的时长；load 从高于 1 降到约 1；SIM 不变（仍是说话人自己的声音），UTMOS 上升；代价是停顿变多。Figure 1 caption 引用的 `tab:decomp` 仍然有效 |
| **Table 5** Ablation（现有） | 正文 Table 2 | §3.3 基准语速、§3.4 PA Talker、best-of-C 选择 | 各组件分别作用在不同的轴上（见下面的改写示例） |
| **Table 6（从附录 Table 10 + 11 合并上来）** MIST−B 在三个测试集上的：主识别器 ASR-BLEU Δ、第二识别器 Δ、语音 COMET Δ；另加一行在 8 个未用过的 AliMeeting 会议上的锁定测试（ASR-BLEU、LAAL、EndOffset 的 Δ） | 附录 Table 10、11 | §3.5 指标的有效性 + 泛化 | 增益不是识别器造成的假象，在从未用过的会议上方向相同 |
| （可选）Table 7：DiariST 官方协议 | 附录 Table 13 | 与已发表系统对比 | 建议仍留在附录，5.1 里加一句话引用 |

小节标题建议：
- 5.1 Comparison with Multi-Speaker S2ST Systems（Table 1、Figure 3）
- 5.2 Speaker Tokens under Overlapping Speech（Table 3）
- 5.3 Where the Listener-Side Delay Is Removed（Table 4）
- 5.4 Ablation of the Playback Layer（Table 5）
- 5.5 Robustness of the Gains（Table 6）

**写法规则：**
- 每段按“问题 → 图表上的定性结论 → 必要时一个数或一个区间 → 对应哪个公式或组件 → 代价”的顺序写。
- 不写逐集三元组。
- 正文不写 p 值，表格里用 \* 标记。
- “exploratory” 只在 Setup 里说明一次。

改写示例 1（5.4，原 “What each component adds”）：

> 改前：*Merging (R1) changes little; the backlog-dependent base rate (R2) gives the largest step, +1.55 / +0.24 / +0.80 tcpASR-BLEU; the CRF-distilled PA Talker (R3) adds +0.29 / +0.02 / +0.33 (exploratory p 0.002 / 0.82 / 0.001) and ends speech 0.18 / 0.15 / 0.23 s earlier, because …*
>
> 改后：*Most of the gain comes from the backlog-dependent rate (R2), which brings the speech load back to the source's pace and removes most of the queue (Table 4). The CRF-distilled Talker (R3) shortens speech further on every set and raises quality on both Zh→En sets, where queues are longest, while En→Zh quality is unchanged. This is the pattern the backward messages of Eq. (w) predict: a shorter rendering is rewarded only where a later block would otherwise wait. Fidelity-based selection (R4) adds a consistent quality gain without changing latency, so delay and intelligibility are improved by different components.*

改写示例 2（5.1）：

> *MIST obtains the highest cpASR-BLEU and tcpASR-BLEU on all three sets (Table 1). The public systems speak earlier on Zh→En but retain only a third to three quarters of MIST's tcpASR-BLEU, so their lower delay is bought with dropped or misattributed content. The same-data cascade isolates the order of attribution: translating speaker-blind and assigning speakers afterwards is level with B in speech but starts later, whereas MIST lies above and to the left of B on every set (Figure 3).*

### 3.5 streaming → simultaneous（第 6 点）

规则（需要你确认）：
- **改成 simultaneous**：标题（“Speaker-Attributed Streaming Meeting Interpretation” → “… Simultaneous Meeting Interpretation”）、Method/Setup/Results/Conclusion/Limitations/附录中所有指**我们的任务、模型、模式和时延**的 streaming。例如 §3.3 标题 “Streaming interpretation”、“in streaming mode”、“streaming latencies”、表格 caption 里的 “streaming systems”、DiariST 表里的 mode 列。
- **diarizer**：与你在 abstract 里的写法保持一致，用 “simultaneous diarizer”。
- **保持不变**：专有名词（SeamlessStreaming、Streaming Sortformer、bib 标题），以及 Related Work 里描述别人工作的 streaming ASR/diarization（例如 t-SOT “also in streaming”）。这些是准确的术语。
- 统计：正文和附录约 30 处，其中 Method 6、Setup 6、Results 2、Conclusion 2、Figure 2 5、附录 9、表格约 12。
- Introduction 里还有 2 处 “streaming”（“in streaming SimulS2ST”“frozen streaming diarizer”）。它们属于你不让动的部分，我**不会改**；需要改的话请告诉我。

### 3.6 新附录结构（参照 SimulS2ST-Omni 和 SEE）

- **A. Data Construction and Statistics**：原 Data 一节、tab_data、中文 AMI 重配音、AliMeeting 参考译文。
- **B. Training Implementation Details and Hyperparameters**：
  - B.1 说话人 token 与序列化：投影器公式、槽位映射、prompt；
  - B.2 同传 Thinker：deadline chunk、训练行、LR、LoRA、batch、GPU；
  - B.3 Talker、voice prompt、基准语速；
  - B.4 Playback alignment：候选、保真度识别器、格点、势函数权重、预算 b、λ\*、backlog bin、rate token、κ、数据量、步数、checkpoint 规则，以及 Algorithm 1；
  - 外加一张合并的超参数总表，收齐正文删掉的所有数值。
- **C. Inference and Online Processing**：online window、chunk protocol、read-level merging、因果性检查、解码设置。
- **D. Playback Alignment: Derivations and Proofs**：消息定义、Prop. 1–3、对偶函数与导数、PA lattice 图。
- **E. Evaluation Protocols**：tcpBLEU、cpASR-BLEU 细节、两种 speech LAAL、识别器、统计检验、预注册分析计划。
- **F. Baseline Systems**：SeamlessStreaming、X-Translator（含增益校准和不加增益的原始结果）、同数据级联。
- **G. Supplementary Results**：完整显著性表（补上 Set 列）、完整语音表与时延区间、DiariST 官方协议、重叠实验的 12 个预注册检验、dev 集的时延分解。

### 3.7 页数预算（ACL long，正文 ≤ 8 页）

| 部分 | v14_overleaf_v1 | v15 目标 |
|---|---|---|
| Title + Abstract + Intro + Related（不动） | 约 2.45 页 | 约 2.45 页 |
| Method（含 Figure 2） | 约 3.7 页 | **约 2.0–2.3 页** |
| Setup | 约 0.8 页 | 约 0.5 页 |
| Results（6 个图表） | 约 1.2 页（2 个图表） | **约 2.3–2.6 页** |
| Conclusion | 约 0.2 页 | 约 0.2 页 |
| 合计 | 约 8.3 页（超页） | **约 7.5–8.0 页** |

结构上与两篇参考论文一致：Method 大约在第 5 页结束，实验约占第 5–8 页。

### 3.8 顺带修复

- 正文删掉 Table 8 的 `δ_w=0` 行，附录完整表里也删掉（WA 不在本文里）。
- Table 7、9 补上 Set 列。
- 统一 31.64 / 31.65：以 Table 1 的汇总口径为准，核对原始 JSON 后再统一。
- Figure 2 的数值改成字母，节号引用随新结构更新。
- 根据新的 Figure 3 和 Table 3–6，同步调整正文对附录的引用。

---

## 4. 需要你确认的问题（每题后附我的建议）

1. **Method 小节**：采用 3.1–3.5 五个小节，把指标作为 §3.5 放在 Method 里？另一种做法是单独设一节 “§4 Speaker-Attributed Speech Evaluation”。**建议放在 Method 里**，这样小节与三条贡献一一对应，也更省页面。
2. **符号表**：附录里的 Notation 表整个删掉？**建议删掉**，两篇参考论文都没有符号表。
3. **正文图表**：采用 Table 1、新的 Figure 3、Table 3（重叠）、Table 4（时延分解 + 语音性质）、Table 5（ablation）、Table 6（稳健性）共 6 个？DiariST 官方协议表留在附录？另外，PA lattice 小图要不要放进 §3.3？**建议**：前 6 个放进正文；PA lattice 小图如果页面够就放进 §3.3，不够就留在附录。
4. **Figure 3 的横轴用哪个时延**：用 EndOffset（与 abstract 和 Figure 1 一致，B→MIST 的差距明显）还是 LAAL？此前（11-10）定过“主时延指标是 speech LAAL”，但你这一版 abstract 突出的是 completion delay（EndOffset）。**建议横轴用 EndOffset**，LAAL 留在 Table 1。
5. **streaming → simultaneous 的范围**：按 §3.5 的规则执行，包括改标题、diarizer 写成 “simultaneous diarizer”、专有名词和他人的 streaming ASR 不改、Introduction 的 2 处不改？
6. **Diarizer 训练数据披露**：在 Setup 和 Limitations 里加一句“Nemotron 3 Diarization 的训练数据包含 AMI 和 AliMeeting 的训练录音”？**建议加**，审稿人很可能查到。

## 5. 确认后的执行步骤

1. 新建 `paper/gstar_si_v15/`，从 `gstar_si_v14_overleaf_v1` 复制过去。v14_overleaf_v1 不做任何修改。
2. 依次修改：
   - Method 重写；
   - 附录 B 超参数表；
   - bib 和 diarizer；
   - Setup 和 Results 重写；
   - 新的 Figure 3 和 Table 3/4/6，数字逐项核对来源表；
   - streaming → simultaneous 替换；
   - 附录重排。
3. 用 TinyTeX pdflatex 编译（Overleaf 兼容），检查：正文 ≤ 8 页，没有未定义的引用、`??` 或 overfull 表格。
4. 打包 `paper/gstar_si_v15_overleaf.zip`（文件放在 zip 根目录），commit 后 push 到 private。是否上传 S3 由你决定。
