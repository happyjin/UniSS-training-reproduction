# MIST 论文 v15 修改方案（基于 gstar_si_v14_overleaf_v1，第二轮，待确认）

2026-11-17（布拉格时间）。源文件：`paper/gstar_si_v14_overleaf_v1/`（S3 zip 解压），编译结果 `paper/gstar_si_v14_overleaf_v1/main.pdf`。
**不改动**：abstract、introduction、Figure 1（`figures/motivation.tex` 和 PDF 图，caption 也不改）。
**不做新实验**：正文新增的图表全部取自 v14 已有表格的数字。

**第二轮修订（回应 5 个问题）**
- 重新命名“蒸馏”一节，避免被误解为 teacher–student 蒸馏（§3.1、§3.2）。
- 每个 Method 小节都用第一段交代要解决的问题，不出现 “Motivation:” 字样；同时补上小节之间的承接句（§3.3）。
- Setup 压缩到约半页，其余内容移到附录（§3.6）。
- Method 不再追求压到 2 页，只把别人的方法和实现细节移到附录，自己的方法写完整（§3.1、§3.9）。
- 把 AliMeeting 定位为 out-of-domain 测试，用来展示泛化能力（§3.5）。
- CRF 明确出现在小节标题里（§3.1）。

---

## 0. 已完成

- 从 S3 下载 zip，解压到新目录 `paper/gstar_si_v14_overleaf_v1/`，没有覆盖任何已有文件。
- 用 TinyTeX 的 pdflatex + bibtex 编译通过（与 Overleaf 默认引擎相同）：22 页，没有未定义的引用或文献。
- 发现**正文超页**：Conclusion 从第 9 页开始，而 ACL long 的正文（含结论）上限是 8 页。v15 必须压回 8 页以内。

---

## 1. 两篇参考论文的写法（第一轮结论，未变）

| | SimulS2ST-Omni（arXiv 2607.19810v2） | SEE（ACL 2026 long） | 我们 v14_overleaf_v1 |
|---|---|---|---|
| Method | 约 1.7 页，3 小节，5 个公式 | 约 1.7 页，3 小节，12 个公式 | 约 3.7 页，6 小节，13 个公式，另有附录符号表 |
| 符号 | 首次出现即定义，维度用字母 | 例如 `A_ℓ(x) ∈ R^{T(x)×d_ℓ}`，紧接一句解释 T(x)、d_ℓ | 维度写死数值，另列 Table 3 符号表 |
| 超参数 | 附录 B *Training Implementation Details and Hyperparameters*（Table 8） | 附录 A.4 | 散落在正文 |
| Setup 篇幅 | 4.1 Experimental Protocol 约 0.6 页 | 4.1 Setups 约 0.5 页 | §4 约 0.8 页，外加 Table 1 |
| 正文结果图表 | 7 个 | 10 个 | 2 个 |
| 结果写法 | 每节一个问题 → 定性结论 → 至多 1 个关键数 → “This confirms that …”落回方法设计 → 承认代价 | 同上，多用区间或平均值（“R ∈ [−0.96, −1.00]”“6.7% on average”） | 逐集三元组和 p 值堆叠 |

---

## 2. v14_overleaf_v1 的问题诊断（未变）

1. Method 3.7 页，是正文超页的主因；里面混入了 G-STAR 投影器、Hungarian 映射、chunk 切分、voice prompt、online window、消息定义、对偶求导等实现细节。
2. 符号冲突：
   - `α_n, β_n`（时间 token 下标）与前向/后向消息 `α_i, β_i` 冲突；
   - token 权重 `ω` 与 Talker 参数 `𝛚` 冲突；
   - PA 一节里的 `x` 改成表示一条流，与录音 `x` 冲突；
   - `ρ`、`π`、`W`、`c` 在正文和附录中各有两种含义。
3. 正文只有 2 个结果图表。
4. 小问题：
   - Table 7、Table 9 的 Set 列为空；
   - Table 8 留着已移出正文的 `δ_w=0` 行；
   - R0 En→Zh 是 31.64，Table 1 的 B 是 31.65，不一致；
   - Figure 2 里写死了 S=8、K=5。

---

## 3. v15 方案

### 3.1 Method 新结构（第 2、4、5 点）

**原则**：
- 不追求固定页数。**我的方法完整保留在正文，包括动机、直觉、公式和解释**；
- 只把**别人的方法、工程细节、具体数值和证明**移到附录；
- 每个小节用第一段说明它要解决的问题，并且承接上一小节留下的问题；
- 小节内每段的第一句都要接住上一段；先讲为什么，再讲怎么做。

**小节之间的逻辑链**：
- 3.1 定义两个问题：谁说了什么，听众何时听到。
- 3.2 解决“谁说了什么”，但文本一旦定稿，何时被听到仍未决定。
- 3.3 引出每个说话人的播放队列；按积压调语速只能事后反应，看不到某个渲染选择会拖慢之后所有块。
- 3.4 用 CRF 对整条播放序列建模；但 CRF 需要未来块的释放时间，线上拿不到。
- 3.5 只在训练时用 CRF 的后验来教 Talker。
- 3.6 回到“听到的是什么”，给出评测语音的指标。

| 小节（标题） | 第一段：用问题引出本节（示意句，不写 “Motivation:”） | 正文保留（我的方法） | 移到附录（别人的方法、细节、数值） |
|---|---|---|---|
| **§3 开头** + Figure 2 | *For every contribution in a meeting, MIST must decide who said what and when the listener hears its translation.* 后接两三句介绍各小节的分工 | 总览 | — |
| **3.1 Problem Formulation** | *A meeting translation helps a listener only if it is attributed to the right speaker and heard soon after that speaker spoke.* | 输入 x；输出片段 `(k_n, a_n, b_n, y^{(n)})`；同传约束：第 r 次读时只看到 `x_{≤t_r}`；每个说话人一条输出语音流；定义每个目标单位“被听到的时间”，从而区分**文本定稿时间**和**听到时间**两种时延（承接 Figure 1） | 采样率、帧长 |
| **3.2 Speaker-Attributed Simultaneous Translation** | *The acoustic tokens of a speech LLM describe what is said but not who says it: they keep no identity across a meeting, and overlapping speakers share the same tokens. A translator that sees only these tokens merges concurrent contributions or guesses speakers afterwards, and a contribution it never translated cannot be attributed later (Figure 1).* | ① **Speaker tokens**：冻结的 simultaneous diarizer（Nemotron 3）输出后验 P，投影成说话人 token，与声学 token 交错（Eq. E，注明沿用 G-STAR）。② **Speaker-first, time-stamped output**：Eq. y；说明为什么先写说话人（译文以说话人为条件）、为什么带时间 token（释放时刻和语音流的排布都要用）；加权交叉熵只写“加重 speaker/time token，因为一个错标签会让整句归错人”。③ **Reading and writing simultaneously**：每 Δ 读一次；说话人流只用已定稿的帧（延迟 γ）；从已提交前缀继续写到 `⟨wait⟩`；deadline 目标（chunk 在其源语音结束 ε 后到期）教模型何时写。最后一句：B 和 MIST 共用这个 Thinker，所以后文的差异全部来自语音层。结尾承接句：*Committing a segment fixes what is said and by whom, but not when the listener hears it.* | 投影器结构（G-STAR 式卷积 + MLP、零初始化）；Hungarian 槽位映射的细节；LoRA 配置；chunk 切分与对齐规则；训练行构造；所有数值 |
| **3.3 Speaker Streams and Playback Queues** | *Committed text reaches the listener only after the Talker speaks it in the speaker's voice, and the blocks of one speaker are spoken one after another. Whenever a stream receives more speech than the source time it covers, later blocks wait behind earlier ones; in the deadline baseline, two thirds of the completion delay arises after the text is committed (Figure 1).* | ① **Voice-prompted streams**：每个说话人一条流，voice prompt 取自该槽位到目前为止的单人语音段。② **Queue and heard delay**：backlog 递推（Eq.）、听到时延 `L_i`（Eq.），以及“长渲染付两次代价”。③ **Backlog-aware speaking rate**：**升格为完整一段，并给出符号公式**（基准语速 = 源语音语速 + 随 backlog 增加的档位，有上限）；说明为什么这样设计（让语音 load 回到约 1，队列就不会累积）。它是 ablation 中贡献最大的一步，v14 只用一句话带过。再加一句 read-level merging。结尾承接句：*A rate rule reacts to the backlog that already exists, but it cannot see that a rendering chosen now delays every later block of the stream.* | Talker / BiCodec 内部结构、voice prompt 的阈值、语速档位的数值 |
| **3.4 Playback Alignment: A Latency-Constrained CRF**（CRF 写进标题，与 abstract 的 “playback alignment, a linear-chain CRF” 一致） | *To charge each rendering for the delay it imposes on later speech, playback alignment scores whole playback plans of a speaker stream instead of single blocks.* | ① **Playback plans as a linear-chain CRF**：候选渲染；状态 (h, c)；势函数 = 质量项 − λ·时延（Eq.）；CRF（Eq.）；确定性转移就是 FIFO 队列。② **Exact inference**：backlog 是过去的充分统计量，所以可以在 backlog 格点上做前向–后向（递推 + 边缘概率合成一个 Eq.），复杂度 O(I·\|H\|·C)。③ **A latency budget instead of a tuned price**：后验正则约束（Eq.）；它的解就是 λ\* 处的 CRF；期望时延随 λ 单调，所以用二分求 λ\*；为什么用预算而不是调 λ（秒/单位，可解释，两个方向各自自动确定）。结尾承接句：*The posterior tells which renderings balance quality against present and future delay, but computing it needs the release times of future blocks.* | 消息定义式、Prop. 1–2 的证明、对偶函数及其导数、候选数、格点、势函数权重、CTC 截断、预算 b、λ\* 的数值 |
| **3.5 Training the Talker on the CRF Posterior**（见 §3.2 的命名说明） | *Future release times are unknown while a meeting is running, so the CRF cannot be run at test time. We use its posterior only during training, as a target that teaches the Talker which renderings the CRF prefers given the backlog it currently faces.* | ① **Candidate weights**：`w_i`（Eq.），由本块质量和 β_{i+1} 带来的未来排队代价共同决定。② **Objective**：`J`（Eq.）= 后验加权似然 + rate 交叉熵 + 对 base Talker 的 KL；用 Prop. 3 说明它等于把 Talker 在候选上的分布拟合到 CRF 后验，即后验正则 EM 的一步 M-step；**并加一句明确说明这里的 “distil” 不涉及 teacher 网络**（见 §3.2）。③ **Decoding**：Talker 读当前实际队列的 backlog token，采样少量候选，按保真度 f 选一个；测试时不跑 CRF；给出 playback layer 的定义。结尾承接句：*Whether these choices help a listener can be judged only on the speech itself, attributed to speakers.* | backlog bin、rate token 数、κ 数值、候选筛选阈值、训练数据量、步数、checkpoint 规则、Algorithm 1 |
| **3.6 Speaker-Attributed Speech Metrics** | *Text BLEU and text-emission latency cannot show whether a spoken translation reaches the right speaker's stream, or when it is heard; speaker-attributed recognition has cpWER and tcpWER, but translated speech has no counterpart.* | cpASR-BLEU（一个 Eq.：按说话人拼接转写和参考，用 chrF 做 Hungarian 一对一匹配，未匹配的一方对空串打分）；tcpASR-BLEU（加时间约束）；语音时延基于 3.1 定义的“听到时间”计算（LAAL、EndOffset） | tcpBLEU 的单元构造、collar 数值、ASR 模型、过滤规则 |

**篇幅**：Method 估计约 2.6–2.9 页（含 Figure 2），公式约 11 个。比 v14 的 3.7 页短，但少掉的主要是附录内容，不是我的方法。
**命名**：“MIST (offline) / MIST-Simul” 不再出现在 Method 里；原 §3.6 Running on a whole meeting 移到附录 C。

### 3.2 “蒸馏”的命名（第 1 点）

- v14 的 “From the posterior to a Talker” **不是 CRF 本身**。CRF 只在 3.4 里：它是一个定义在播放计划（backlog × 候选渲染）上的分布，只在训练时计算。3.5 是**用 CRF 后验给候选加权，微调 Talker**。Talker 是自回归的语音 token 模型，测试时只运行它，不跑 CRF。
- 这里**没有 teacher 网络**。“老师”是 CRF 的后验，它用到了未来块的释放时间，而这个信息只在训练时有。因此 “distillation” 一词容易让读者联想到 Hinton 式的大模型 → 小模型压缩，甚至追问 teacher 模型在哪里。
- **处理方式**：
  - 小节标题用 **“Training the Talker on the CRF Posterior”**，不用 “Distilling”。
  - abstract 和 intro 里已经写了 “distilled into the speech generator”，这部分不能改；所以 3.5 第一次使用该词时给出定义：
    > *We call this distilling the posterior into the Talker: the Talker is fine-tuned to reproduce the CRF's preferences among candidate renderings. No teacher network is involved; the target is a structured posterior that uses future release times, which only training provides.*
  - 技术上最准确的说法是 “posterior-weighted maximum likelihood, one M-step of posterior-regularised EM”（Prop. 3）。
  - 建议补引 Hu et al. (2016) *Harnessing Deep Neural Networks with Logic Rules*（ACL）。它就是把后验正则得到的后验“蒸馏”进网络的先例，能让 “distil” 一词站得住。目前 bib 里没有这篇，需要新增。

### 3.3 段落衔接的写法规则（第 2、4 点）

- 每个小节的第一段说明“上一节留下了什么问题、为什么需要本节”，不写 “Motivation:” 这类标签；示意句见 §3.1 表格。
- 每个小节的最后一句交代本节解决不了、留给下一节的问题。
- 段首可以保留加粗的 run-in 标题（两篇参考论文都这样用），但每段的第一句必须接住上一段。
- 先讲为什么，再讲怎么做；每个符号在引入它的那一句里给出含义；不提前引用尚未解释的概念。
- 写完后逐段检查：只读每段第一句，连起来能否读成一个连贯的故事。

### 3.4 符号规则（第 1 点，未变）

- 删除 “Every symbol … Table 3” 这句和附录 Notation 表；每个符号在首次出现时定义，维度只用字母。
- 所有数值汇总到附录 B 的 *Training Implementation Details and Hyperparameters* 大表。
- 修复符号冲突：
  - 时间 token 下标改成 `\hat a_n, \hat b_n`；
  - Talker 参数改成 `ξ`；
  - 一条语音流用 `\mathcal{U}`；
  - 附录里的参考说话人改成 `r`；
  - 播放起点改成 `o_n`；
  - deadline chunk 改成 `u_j`。

### 3.5 Diarizer 引用与 out-of-domain 定位（第 2 点 + 新问题）

**引用**：新增 `medennikov2026nemotron`（Nemotron 3 Diarization，arXiv:2611.12246，作者 Medennikov, Park, Timofeeva, Huang, Polok, Kostandian, Wang, Gao, Balam, Ginsburg）。需要改的位置：
- `method.tex` 第 26 行；
- `setup.tex` 第 17 行；
- `figures/arch.tex` 的节点和 caption；
- 附录的 Causality、Chunk protocol、超参数表（写明 30.4 s / 1.04 s 档位，以及 10 ms→80 ms 均值池化）；
- `related.tex` 在原有引用之外加上这一篇。

**“no AliMeeting data is used in training” 改成 out-of-domain，可以，而且是更强的说法**：
- AliMeeting 是真实的普通话会议，录音设置、说话人和话题都与 AMI 不同；而我们 Zh→En 训练用的中文源语音是**合成的（重新配音的 AMI）**。所以它同时是跨域和从合成到真实的测试。
- LOG 第 636 行写明：“参数只在 AMI dev 上选定；RealSI / ACL / AliMeeting 从不用于调参。”

建议写法：
- **Setup**：
  > *AliMeeting serves as an out-of-domain test: real Mandarin meetings, whereas our Chinese training source is re-voiced AMI. None of the components we train (Thinker, speaker projector, Talker, PA Talker) sees AliMeeting audio or text, and every choice was made on AMI dev.*

  紧接着如实披露：
  > *The off-the-shelf diarizer was trained with AMI and AliMeeting training recordings (Medennikov et al., 2026).*

  因此“out-of-domain”的说法限定在**我们训练的部分**。
- **Table 1** 的表头写成 “AliMeeting Zh→En (out-of-domain)”。
- **Results** 按泛化来解读：
  - 播放层在 AliMeeting 上收益最大：队列等待从 1.33 s 降到 0.33 s，EndOffset 降低最多。原因是队列模型只依赖语音时长，与领域无关。
  - 最干净的泛化证据是 8 场此前从未解码过的 AliMeeting 会议上的锁定测试（Table 6）。
  - **如实写**：说话人 token 在 AliMeeting 上只部分迁移（并发说话人召回提高，重叠单元的分数持平）；AliMeeting 测试集在开发中被多次打分，Limitations 已说明这一点（非盲）。
- **Conclusion** 可加一个分句：“including out-of-domain real Mandarin meetings”。Conclusion 不在“不动”范围内。

### 3.6 Setup 压缩到约半页（第 3 点）

可以借鉴。两篇参考论文的 Setup 都约半页，只交代“做了什么”，细节全部指向附录。

| 保留在正文（约 0.5 页） | 移到附录 |
|---|---|
| **Data**（约 6 行）：AMI En→Zh；重新配音的 Zh→En（合成源语音，参考译文是人工转写）；AliMeeting out-of-domain 测试（真实普通话）；各 16 / 16 / 20 场会议 | MT 系统与 QE 选择、重新配音流程、窗口统计表 → 附录 A |
| **Compared systems**（约 6 行）：SeamlessStreaming、X-Translator、同数据级联、B、MIST 各一句；外部系统 zero-shot 运行、用 causal relabelling 归属说话人 | X-Translator 的增益校准、级联的构造、relabelling 规则 → 附录 F（Table 1 脚注里保留 “input level set on dev”） |
| **Implementation**（约 2 行）：模型名称；训练和解码细节见附录 B/C | H200 和 GPU 数、PA Talker 用的 15 场训练会议、分方向的 Thinker 等 → 附录 B |
| **Metrics and statistics**（约 5 行）：文本 tcpBLEU；语音 cp/tcpASR-BLEU（指向 §3.6），识别器名称一句话；LAAL、EndOffset（chunk protocol）；按会议做配对检验，确认性对比做 Holm 校正 | sign-flip / bootstrap 细节、Holm 分组、预注册分析计划、chunk protocol 定义 → 附录 C/E |

### 3.7 实验与结果（第 3、4 点，未变，只加了 OOD 解读）

正文结果图表从 2 个增加到 6 个：
1. Table 1：系统对比（已有，AliMeeting 列标 OOD）。
2. **Figure 3**：质量–时延平面（新，用 Table 1 和 Table 2 的数字，pgfplots 绘制）。
3. **Table 3**：重叠说话（从附录 Table 14 提上来，压缩）。
4. **Table 4**：时延分解 + 语音性质（由附录 Table 6 + Table 8 合并）。
5. Table 5：Ablation（已有）。
6. **Table 6**：稳健性（由附录 Table 10 + 11 合并，含锁定测试）。

结果小节：
- 5.1 Comparison with Multi-Speaker S2ST Systems
- 5.2 Speaker Tokens under Overlapping Speech
- 5.3 Where the Listener-Side Delay Is Removed
- 5.4 Ablation of the Playback Layer
- 5.5 Robustness and Out-of-Domain Generalization

每个结果小节的第一句也是问题句，并且对应 Method 的某个小节。写法规则和改前/改后示例同第一轮：每个论点至多一个关键数，不写逐集三元组，正文不写 p 值。

### 3.8 streaming → simultaneous（第 6 点，未变）

- 我们的任务、模型、模式和时延，以及标题，都改成 simultaneous；diarizer 用 abstract 里的 “simultaneous diarizer”。
- 专有名词（SeamlessStreaming、Streaming Sortformer）和他人的 streaming ASR/diarization 不改。
- Introduction 里的 2 处不改。

### 3.9 页数预算（第 4 点调整后）

| 部分 | v14_overleaf_v1 | v15 目标 |
|---|---|---|
| Title + Abstract + Intro + Related（不动） | 约 2.35 页 | 约 2.35 页 |
| Method（含 Figure 2） | 约 3.7 页 | **约 2.6–2.9 页** |
| Setup | 约 0.8 页 | **约 0.5 页** |
| Results（6 个图表） | 约 1.2 页 | **约 2.1–2.3 页** |
| Conclusion | 约 0.2 页 | 约 0.2 页 |
| 合计 | 约 8.3 页（超页） | 约 7.8–8.2 页 |

预算比较紧。**超页时按以下顺序处理，不删我的方法内容**：
1. 缩短表格和图的 caption；
2. Figure 2 压低高度；Figure 3 改成单栏；
3. Table 6 移回附录，正文只留一句结论和引用；
4. Related Work 精简两三行（它不在“不动”范围内）。

### 3.10 附录结构与顺带修复（未变）

**附录结构**：
- A. Data Construction and Statistics
- B. Training Implementation Details and Hyperparameters（含数值总表）
- C. Inference and Online Processing
- D. Playback Alignment: Derivations and Proofs（含 PA lattice 图）
- E. Evaluation Protocols
- F. Baseline Systems
- G. Supplementary Results

**顺带修复**：
- 删除 `δ_w=0` 行；
- Table 7、9 补上 Set 列；
- 统一 31.64 / 31.65：先核对原始 JSON；
- Figure 2 的数值改成字母，节号引用随新结构更新。

---

## 4. 需要你确认的问题（每题后附我的建议）

1. **3.4、3.5 的标题**：用 “Playback Alignment: A Latency-Constrained CRF” 和 “Training the Talker on the CRF Posterior”？3.5 正文第一次出现 “distil” 时给出定义（与 abstract、intro 保持一致）。另一种选择是 3.5 叫 “Distilling the CRF Posterior into the Talker”。**建议用前者**。
2. **补引 Hu et al. (2016)**，作为把后验正则的后验蒸馏进网络的先例？**建议引**。
3. **Method 分成 6 个小节**（3.1–3.6，指标放在 3.6），backlog-aware 语速升格为完整一段？**建议是**。
4. **AliMeeting 的 OOD 写法**：按 §3.5 写（限定为“我们训练的组件”，披露 diarizer 的训练数据，强调锁定测试）？**建议是**。
5. **第一轮留下的问题**：
   - 删除 Notation 表——建议删；
   - 正文放 6 个图表——建议是；PA lattice 图在页面够时放进 3.4；
   - Figure 3 横轴用 EndOffset 还是 LAAL——建议 EndOffset；
   - streaming 的替换范围按 §3.8——建议是。

## 5. 确认后的执行步骤

1. 新建 `paper/gstar_si_v15/`，从 v14_overleaf_v1 复制过去。v14_overleaf_v1 不做任何修改。
2. 依次修改：
   - 重写 Method，先写各小节的第一段和最后一段，确认逻辑链通顺后再填中间内容；
   - 附录 B 数值总表；
   - bib 和 diarizer；
   - 压缩 Setup；
   - 写 Results 和新图表，数字逐项核对来源；
   - streaming → simultaneous 替换；
   - 重排附录。
3. 用 TinyTeX pdflatex 编译（Overleaf 兼容），检查：正文 ≤ 8 页；没有未定义的引用或 `??`；只读每段首句时能连成一个故事。
4. 打包 `paper/gstar_si_v15_overleaf.zip`，commit 后 push 到 private。是否上传 S3 由你决定。
