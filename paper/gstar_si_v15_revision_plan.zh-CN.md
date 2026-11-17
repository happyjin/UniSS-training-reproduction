# MIST 论文 v15 修改方案（基于 gstar_si_v14_overleaf_v1，第三轮，决策已定）

2026-11-17（布拉格时间）。源文件：`paper/gstar_si_v14_overleaf_v1/`（S3 zip 解压），编译结果 `paper/gstar_si_v14_overleaf_v1/main.pdf`。

**不改动**：abstract、introduction、Figure 1。**唯一例外**：其中 “distil” 的说法要替换（§3.2 列出逐句改动）。
**不做新实验**：正文新增的图表都直接取自 v14 已有表格的数字。

**第三轮修订**
- “distilled” 的说法在 abstract、intro 和全文统一替换（§3.2）。
- 检查后发现 “listener / heard” 的措辞容易被读成做过主观听测，已修正，并定了一条写作规则（§3.3）。
- Table 1：
  - 列名改为 cpASR-BLEU / tcpASR-BLEU；
  - 删掉所有 EndOff. 列；
  - 去掉单元格里的星号（§3.7）。
- 定了 EndOffset 的使用规则。因此 Figure 3 的横轴从 EndOffset 改为 LAAL（§3.7）。
- 之前所有待确认的问题都按建议采纳（§4）。

---

## 0. 已完成

- 从 S3 下载 zip，解压到新目录 `paper/gstar_si_v14_overleaf_v1/`。
- pdflatex + bibtex 编译通过，22 页，没有未定义的引用。
- **正文超页**：Conclusion 从第 9 页开始，超过 ACL long 的 8 页上限。v15 必须压回 8 页。

## 1. 两篇参考论文的写法（结论不变）

| | SimulS2ST-Omni | SEE | v14_overleaf_v1 |
|---|---|---|---|
| Method | 约 1.7 页，3 小节 | 约 1.7 页，3 小节 | 约 3.7 页，6 小节，另有符号表 |
| Setup | 约 0.6 页 | 约 0.5 页 | 约 0.8 页 |
| 超参数 | 附录 B *Training Implementation Details and Hyperparameters* | 附录 A.4 | 散落在正文 |
| 正文结果图表 | 7 个 | 10 个 | 2 个 |
| 结果写法 | 一个问题 → 定性结论 → 至多 1 个关键数 → “This confirms that …”落回方法设计 → 说明代价 | 同上，多用区间或平均值 | 逐集三元组加 p 值 |

## 2. v14_overleaf_v1 的问题诊断（不变）

1. Method 太长，混入了实现细节。
2. 符号冲突（α/β、ω、x、ρ、π、W、c）。
3. 正文只有 2 个结果图表。
4. 小问题：Table 7/9 的 Set 列为空；残留 `δ_w=0` 行；31.64 与 31.65 不一致；Figure 2 写死 S=8、K=5。

---

## 3. v15 方案

### 3.1 Method 新结构：6 个小节，自己的方法写完整，其余移到附录

**原则**：
- 我的方法完整保留在正文，包括动机、直觉、公式和解释；
- 别人的方法、工程细节、数值和证明移到附录；
- 每个小节用第一段说明要解决的问题，不写 “Motivation:”；
- 每个小节用最后一句交代本节解决不了、留给下一节的问题；
- 每段第一句接住上一段。

| 小节 | 第一段（示意） | 正文保留 | 移到附录 |
|---|---|---|---|
| §3 开头 + Figure 2 | *For every contribution in a meeting, MIST must decide who said what and when its translation is played to the listener.* | 总览 | — |
| **3.1 Problem Formulation** | *A meeting translation is useful only if it is attributed to the right speaker and played soon after that speaker spoke.* | 输入、输出片段、同传约束、每个说话人一条输出流；定义**播放时间**（heard time）。它是根据播放排期**计算**出来的，不是对听众测得的（见 §3.3）。由此区分文本定稿时延和播放时延 | 采样率、帧长 |
| **3.2 Speaker-Attributed Simultaneous Translation** | *Acoustic tokens describe what is said but not who says it …*（全文见第二轮） | 说话人 token（Nemotron 3 + G-STAR 式交错）；speaker-first、带时间戳的输出；同传读写与 deadline 目标；B 和 MIST 共用这个 Thinker | 投影器结构、Hungarian 映射、LoRA、chunk 切分、训练行构造、数值 |
| **3.3 Speaker Streams and Playback Queues** | *Committed text reaches the listener only after the Talker speaks it …* | voice-prompted 语音流；backlog 递推和播放时延公式；**随积压调整的语速**（完整一段，给出公式）；read-level merging | Talker/BiCodec 内部结构、阈值、语速档位数值 |
| **3.4 Playback Alignment: A Latency-Constrained CRF** | *To charge each rendering for the delay it imposes on later speech, playback alignment scores whole playback plans …* | 作为线性链 CRF 的播放计划；精确前向–后向；用时延预算代替手调 λ（后验正则 + 二分） | 消息定义、证明、对偶推导、数值 |
| **3.5 Training the Talker on the CRF Posterior** | *Future release times are unknown while a meeting is running, so the CRF cannot run at test time; we use its posterior only during training, as the target that teaches the Talker which renderings the CRF prefers given its current backlog.* | 候选权重 `w_i`；训练目标 `J`，并用 Prop. 3 说明它等于把 Talker 在候选上的分布拟合到 CRF 后验，即后验正则 EM 的一步 M-step；一句说明 *the target is not produced by a teacher network but by the CRF posterior, which uses future release times available only in training*；补引 Hu et al. (2016) 作为“用后验正则的目标训练网络”的先例（不使用 distil 一词）；解码。结尾句改为：*It remains to score the synthesised speech automatically, for content, speaker attribution and playback delay.* | backlog bin、κ、数据量、步数、checkpoint 规则、Algorithm 1 |
| **3.6 Speaker-Attributed Speech Metrics** | *Text BLEU and text-emission latency do not show whether a translation is spoken into the right speaker's stream, or when it is played. We therefore score the synthesised speech automatically: its ASR transcripts for content and attribution, and its playback times for delay. Speaker-attributed recognition has cpWER and tcpWER, but translated speech has no counterpart.* | cpASR-BLEU（一个 Eq.）、tcpASR-BLEU；基于播放时间计算的 LAAL 和 EndOffset | tcpBLEU 的单元构造、collar、ASR 模型、过滤规则 |

Method 预计约 2.6–2.9 页、约 11 个公式。原 §3.6 Running on a whole meeting 移到附录 C。

### 3.2 统一替换 “distil”（第一条要求）

替换后全文不再出现 distil。统一说法是 “train the Talker on the CRF posterior”，以及 “PA Talker”。

| 位置 | 原文 | v15 |
|---|---|---|
| `abstract.tex`（只改这一处） | … yield a latency-constrained distribution **that is distilled into the speech generator**, allowing … | … yield a latency-constrained distribution **that serves as the training target of the speech generator**, allowing … |
| `introduction.tex` 第 19 行（只改这一处） | **By distilling its latency-constrained posterior into the Talker**, PA makes speech generation aware of … | **By training the Talker on its latency-constrained posterior**, PA makes speech generation aware of … |
| `introduction.tex` 第 24 行（贡献 2） | … with a latency-constrained CRF **distilled into the speech generator** to improve … | … with a latency-constrained CRF **whose posterior trains the speech generator** to improve … |
| Figure 1 caption（`figures/motivation.tex`） | … a backlog-aware Talker **trained by playback-queue CRF distillation**; no CRF runs online. | … a backlog-aware Talker **trained on the posterior of a playback-queue CRF**; no CRF runs online. |
| **Figure 1 图片里的文字** | “CRF-distilled Talker reduces waiting.” | 建议改成 “CRF-trained Talker reduces waiting.”。**图是 PDF，我没有它的源文件，需要你在作图工具里改后重新导出** |
| Method 3.5 | we therefore distil its posterior into the Talker … | 按 §3.1 重写 |
| Results | the CRF-distilled PA Talker (R3) | the PA Talker (R3), trained on the CRF posterior |
| Conclusion | A linear-chain CRF over this queue, distilled into the speech generator, … | A linear-chain CRF over this queue, whose posterior trains the speech generator, … |
| Limitations | the posterior is distilled in one round | the Talker is trained on the posterior in one round |
| 附录 | Proposition “Posterior distillation”；Algorithm 和 ablation 说明里的 “CRF-distilled Talker” | “Posterior-weighted training”；“PA Talker”；内部标签 `eq:distill`、`prop:distill` 改成 `eq:patrain`、`prop:patrain` |

### 3.3 “听众实际听到的语音”这一条：检查结果

**确实写得不准确。** 我在上一轮回复里写的“最后回到听众实际听到的语音，给出评测指标”，本意是指评测**合成的语音输出本身**。具体做法是：
- 用 ASR 转写计算 cpASR-BLEU 和 tcpASR-BLEU；
- 用播放排期计算 LAAL 和 EndOffset。

这些都是自动评测，**没有任何主观听测**。这与 Limitations 中 “No human evaluation” 一致。中文表述应改为“最后对合成语音做自动评测（ASR 转写 + 播放时间）”。

方案草稿里也有两句英文会让人误以为做过听测，已经改掉：
- 3.5 的结尾句，原来是 *Whether these choices help a listener can be judged only on the speech itself* → 改成 *It remains to score the synthesised speech automatically …*。
- 3.6 的第一段，改为明确写出 “score the synthesised speech automatically: ASR transcripts … playback times …”。

**写作规则（全文适用）**：
- 正文里的 “heard / listener-side delay” 一律指**根据播放排期计算出的播放时间**。abstract 和 intro 是你写的，沿用这个词。
- 3.1 第一次出现时给出定义：*the time at which a target unit is played in its speaker's output stream, computed from the playback schedule*。
- 不写任何暗示人类主观判断的句子，例如 “listeners prefer”“easier to follow”。
- UTMOS 写明是自动 MOS 预测器。

### 3.4 符号规则（不变）

- 删除 Notation 表；符号在首次出现时定义，维度只用字母；数值汇总到附录 B 的大表。
- 修复符号冲突：
  - 时间下标用 `\hat a_n, \hat b_n`；
  - Talker 参数用 `ξ`；
  - 一条语音流用 `\mathcal{U}`；
  - 附录里的参考说话人用 `r`；
  - 播放起点用 `o_n`；
  - deadline chunk 用 `u_j`。

### 3.5 Diarizer 引用与 out-of-domain（不变）

- 新增 `medennikov2026nemotron`（Nemotron 3 Diarization，arXiv:2611.12246）。需要改的位置：method、setup、Figure 2、附录（Causality、Chunk protocol、超参数表）、Related Work 补引。
- AliMeeting 定位为 **out-of-domain 测试**：
  - 它是真实普通话会议，而我们中文训练源是合成的重新配音 AMI；
  - 我们训练的 Thinker、投影器、Talker、PA Talker 都没有见过 AliMeeting 的音频或文本，所有选择只在 AMI dev 上做（LOG 第 636 行可证）。
- 需要如实写明三点：
  - 现成的 diarizer 训练时用过 AMI 和 AliMeeting 的训练录音；
  - 最干净的泛化证据是 8 场锁定测试会议；
  - 说话人 token 在 AliMeeting 上只部分迁移。
- Table 1 表头写成 “AliMeeting Zh→En (out-of-domain)”。

### 3.6 Setup 压缩到约半页（不变）

正文只保留四段：Data、Compared systems、Implementation（两行）、Metrics and statistics。其余移到附录 A、B、C、E、F。

### 3.7 实验与结果

**Table 1 的修改（第二条要求）**

| 项目 | 现在 | v15 |
|---|---|---|
| 列名 | cpASR / tcpASR | **cpASR-BLEU / tcpASR-BLEU**（你消息里写的 “BLUE” 应为 BLEU） |
| 时延列 | LAAL + EndOff. | **只保留 LAAL**（speech LAAL，块起点口径），删除全部 3 列 EndOff. |
| 显著性 | 大部分单元格带 `*` | **去掉所有星号**。改在 caption 里写一句：*Every cpASR-BLEU and tcpASR-BLEU difference between MIST and each other system is significant (paired test over meetings, Holm-corrected, p < 0.001; Appendix G).* 完整的检验结果放在附录 |
| 加粗 | 每列最好的值 | 保持每列加粗最好值，这是如实的做法：Zh→En 两个集上 X-Translator 的 LAAL 最低，仍然加粗它 |
| 表头 | AliMeeting Zh→En | AliMeeting Zh→En (out-of-domain) |

列数从 12 个数字列减到 9 个，表格会更宽松。

**EndOffset 的使用规则**：
- **正文中凡是包含外部系统的比较（Table 1、Figure 3、5.1 的文字），只用 LAAL。** 5.1 的写法：外部系统在 Zh→En 上 LAAL 更低，但 tcpASR-BLEU 只有 MIST 的 1/3 到 3/4。同数据级联的 LAAL 比 MIST 高 0.4–0.6 s。
- **只和自己的方法比较时（B vs MIST、ablation、时延分解、稳健性、锁定测试），继续用 EndOffset 加 LAAL**。abstract 和 Figure 1 里的 EndOffset 都是 B vs MIST，符合这条规则。
- **Figure 3 的横轴改为 LAAL**（上一轮的建议是 EndOffset，与新规则冲突，已改）。图中点为 Table 1 的各系统，并标出 B→MIST 的箭头。
- **附录的处理（建议）**：外部系统的 EndOffset 只保留在附录 G 的完整结果表里，正文不引用。如果完全删除，审稿人追问时会显得像选择性报告；而 LAAL 是 SimulS2ST 的标准时延指标，所以主表用 LAAL 站得住。

**其他正文表格也统一去掉星号**：Ablation、重叠说话、稳健性三张表都改在 caption 里用一句话说明显著性，完整的 p 值留在附录 G。

**正文结果图表（6 个）**：
1. Table 1：系统对比（只用 LAAL）。
2. Figure 3：质量–LAAL 平面。
3. Table 3：重叠说话（来自附录 Table 14）。
4. Table 4：时延分解 + 语音性质，B vs MIST，含 EndOffset（来自附录 Table 6 + 8）。
5. Table 5：Ablation（tcpASR-BLEU + EndOffset）。
6. Table 6：稳健性（第二识别器、COMET、8 场锁定测试，B vs MIST）。

**结果小节**：
- 5.1 Comparison with Multi-Speaker S2ST Systems
- 5.2 Speaker Tokens under Overlapping Speech
- 5.3 Where the Listener-Side Delay Is Removed
- 5.4 Ablation of the Playback Layer
- 5.5 Robustness and Out-of-Domain Generalization

**写法**：每个论点至多一个关键数，不写逐集三元组，正文不写 p 值。

**名称统一**：附录里笼统写的 “ASR-BLEU”，实际指的是 tcpASR-BLEU（例如 Table 7 的 +2.28 就是 Table 1 中 tcpASR-BLEU 的差值）。v15 全文统一写成 tcpASR-BLEU 或 cpASR-BLEU。

### 3.8 streaming → simultaneous（不变）

- 我们的任务、模型、模式、时延以及标题都改成 simultaneous；diarizer 写成 “simultaneous diarizer”。
- 专有名词和他人的 streaming ASR/diarization 不改。
- Introduction 里的 2 处 streaming 不改。

### 3.9 页数预算（不变）

- 目标：Method 约 2.6–2.9 页，Setup 约 0.5 页，Results 约 2.1–2.3 页，合计约 7.8–8.2 页。Table 1 少了 3 列，会稍微宽松一些。
- 超页时按以下顺序处理，不删方法内容：
  1. 缩短 caption；
  2. 压低 Figure 2；
  3. Table 6 移回附录；
  4. 精简 Related Work。

### 3.10 附录结构与顺带修复（不变）

- 附录结构：A Data / B Training Implementation Details and Hyperparameters / C Inference / D Derivations and Proofs / E Evaluation Protocols / F Baseline Systems / G Supplementary Results（完整显著性检验、外部系统的完整时延）。
- 顺带修复：删除 `δ_w=0` 行；Table 7/9 补上 Set 列；统一 31.64/31.65（先核对 JSON）；Figure 2 的数值改成字母。

---

## 4. 已定决策（按建议采纳）

1. 3.4 标题为 “Playback Alignment: A Latency-Constrained CRF”，3.5 标题为 “Training the Talker on the CRF Posterior”；全文不再使用 distil，包括 abstract、intro 中的 3 句和 Figure 1 caption。
2. 补引 Hu et al. (2016)，措辞是“用后验正则的目标训练网络”。
3. Method 分 6 个小节，指标放在 3.6；随积压调整的语速写成完整一段。
4. AliMeeting 按 out-of-domain 写，并附 §3.5 中的三条披露。
5. 删除 Notation 表；正文放 6 个图表；PA lattice 图在页面够时放进 3.4。
6. Figure 3 横轴用 LAAL（按新的 EndOffset 规则修改）。
7. streaming 的替换范围按 §3.8。
8. Table 1 按 §3.7 修改；其他正文表格也去掉星号。

**还需要你做的一件事**：Figure 1 **图片**中的 “CRF-distilled Talker reduces waiting.” 需要你在源文件里改，例如改成 “CRF-trained Talker reduces waiting.”，然后导出新的 PDF。caption 我来改。

## 5. 执行步骤

1. 新建 `paper/gstar_si_v15/`，从 v14_overleaf_v1 复制过去。v14_overleaf_v1 不做任何修改。
2. 依次修改：
   - abstract、intro、Figure 1 caption 中的 distil（只改 §3.2 列出的句子）；
   - 重写 Method：先写各小节的首段和尾句，检查逻辑链，再填中间内容；
   - 附录 B 数值总表；
   - bib 和 diarizer；
   - 压缩 Setup；
   - Table 1 改版、新 Figure 3、Table 3/4/6，数字逐项核对来源；
   - Results 重写；
   - streaming 替换；
   - 重排附录。
3. 用 TinyTeX pdflatex 编译，检查：
   - 正文 ≤ 8 页；
   - 没有未定义的引用或 `??`；
   - 全文 `grep -i distil` 为 0；
   - 正文中没有外部系统的 EndOffset；
   - 只读每段首句时能连成一个故事。
4. 打包 `paper/gstar_si_v15_overleaf.zip`，commit 后 push 到 private。是否上传 S3 由你决定。
