# 单说话人 QABC 论文 v15 修改方案（基于 bayes_simul_s2st_v14）

写于 2026-11-18（布拉格时间）。本方案只覆盖用户第 9 轮提出的四项要求；其余内容沿用 v14，见 `bayes_simul_s2st_v14_revision_plan.zh-CN.md`。

约束：摘要和引言的文字仍然不动。新增的动机图从 `main.tex` 中、在 `\input{sections/introduction}` 之前插入，不需要改引言文件（见第 5 节）。

## 1. Table 1 的离线结果：恢复成 v13 的完整版并放在正文

**现状**

v14 正文其实有 Table 1（第 6 页，§4.2 Offline S2ST），但做成了压缩版，和 v13 相比少了不少内容：

| | v13 Table 1 | v14 Table 1 |
|---|---|---|
| 行 | 15 行：3-stage 和 2-stage cascade、GPT-4o、Qwen2.5-Omni、Step-Audio 2 mini、Qwen3-Omni（自带 Talker）、Seamless-m4t-medium、Seamless-m4t-v2-large、Seamless-expressive、UniSS (P)、UniSS (Q)、SimulS2ST-Omni 的 Dec-only 和 Thinker–Talker、我们的 Thinker+Talker、加 QABC adapter | 8 行 |
| 列 | ASR-BLEU、Text-BLEU、A.PCP、SIM-O，每项两个方向 | 只有 ASR-BLEU 和 Text-BLEU |
| 版式 | 跨栏 `table*` | 单栏 |

**v15 的改法**

- 正文 §4.2 直接 `\input{generated/tab_cvss}`。这个文件就是 v13 的原表，在 v14 目录里还保留着；表体逐字不改，作为跨栏 `table*`，与 SimulS2ST-Omni 的 Table 1 一样。
- 表注从 v13 的 8 行压到约 4 行，以节省篇幅。三个来源脚注（†、‡、¶）和 SIM-O 不同尺度的说明（§）都保留。
- v14 的压缩版 `body_cvss.tex` 不再 input，并从生成脚本里删掉对应部分。
- 如实说明一点：A.PCP 这一列 En→Zh，我们（2.79）低于 SimulS2ST-Omni 的 Dec-only（2.96）和两个 cascade（2.80、2.87）。v13 原表就是这样，按"恢复原表"处理，不删这一列。

## 2. 正文实验表格的排版

**现状（v14 PDF）**

- 第 7 页没有一行正文，全是浮动体：图 2、Table 2、Table 3、Table 4，下方空白 12–15%。
- 第 8 页上半页是 Table 5、6。
- 4.3、4.4 的文字在第 6 页，对应的表要到第 7、8 页才出现。

**原因**

1. **同类浮动体按编号顺序排版。** LaTeX 规定同一类浮动体必须按编号顺序出现。跨栏的 Table 2 只能放到下一页顶部，所以排在它后面的单栏 Table 3–6 都被拖到了它之后。
2. **一页里浮动体太多，被当成浮动体页。** 第 7 页的浮动体加起来超过了"每页至少留给正文的比例"（`\textfraction`，默认 0.2），于是整页只放浮动体。
3. **表格插入的位置太靠后。** 在源文件里，每张表都 `\input` 在对应小节的开头，但与文字之间隔着两张跨栏浮动体。

**v15 的排法**（目标：每张表与首次引用它的文字在同一页或下一页；第 5–8 页每页至少一半是正文）

| 浮动体 | 版式 | 在源文件中的位置 | 目标页 |
|---|---|---|---|
| Figure 1 动机图（新增，见第 5 节） | 单栏 | `main.tex` 中，引言之前 | 第 1 页右栏顶部 |
| Figure 2 方法流程图（原 Figure 1） | 跨栏，高度缩约 25% | §3 节首 | 第 3 页顶部 |
| Table 1 离线（完整版） | 跨栏 | §4.1 Setup 末尾（提前 input，好让它落在 §4.2 所在页的顶部） | 第 6 页顶部 |
| Figure 3 质量–延迟曲线 | 跨栏 | §4.2 末尾 | 第 7 页顶部 |
| Table 2 分档 | 跨栏 | §4.3 第一段之后 | 第 7 页顶部，与 Figure 3 同页；同一页最多放两个跨栏浮动体（`dbltopnumber` = 2） |
| Table 3 主比较 | 单栏 `[t]` | 紧跟引用它的段落 | 第 7 页，Table 2 下方的左栏 |
| Table 4 队列 | 单栏 `[t]` | §4.4 段首 | 第 7 页右栏，或第 8 页 |
| Table 5、6 | 单栏 `[t]` | 各自小节段首 | 第 8 页左右两栏顶部 |

正文的浮动参数写在 `main.tex` 导言区（只调浮动参数，不改模板的间距）：

```
\setcounter{topnumber}{3}  \setcounter{dbltopnumber}{2}
\renewcommand{\topfraction}{0.85}      \renewcommand{\dbltopfraction}{0.8}
\renewcommand{\textfraction}{0.12}     \renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}
```

**验收**

- 用空白检查脚本核对，第 5–8 页每栏空白不超过 12%；
- 不出现只有浮动体、没有正文的页；
- 每张表与首次引用它的文字最多相隔一页；
- 逐页渲染第 1–8 页，人工看一遍。

## 3. Table 9 是什么，要不要删

- **内容**：附录 C.2 中 TTB 基线的先验网络所用的 17 维特征逐行列表（对数概率、top-2 差、熵、位置……）。TTB 是对照用的测试时后验，不是我们的方法。
- **建议删除这张表，改成文字**：C.2 正文已经按三组概括了这些特征（置信度统计、位置统计、上下文）。只需在那句后面用括号把 17 维逐一列全（约 3 行），复现所需的信息一点不少，附录还能少一个浮动体。
- 同时删除 v14 附录里两处对 `tab:features` 的引用：附录 C.2 一处，`appendix_train` 的 D.4 一处。

## 4. 附录最后一张表嵌入正文

**现状**：附录最后一页，左栏依次是 Table 15、G 节、Table 16、H 节；右栏只有 Table 17 孤零零在顶部，下面 83% 是空白。

**v15 的改法**

- 导言区加 `\usepackage{float}`。
- 附录 F、G、H 的三张小表（`app_calib`、`app_timeaxes`、`app_robust`）以及附录 E 的推理设置表改用 `[H]`，即在源文件的位置原样排出，不再浮动。每张表都紧跟引用它的那一段之后，嵌在文字中。
- 附录 H 只有一段文字，把 Table 17 放到这段的第一句之后。这样最后一页是"文字—表—文字"连续排下来。
- 验收：最后一页不出现单独占一栏的表；附录其余各页每栏空白不超过 15%。

## 5. Figure 1：动机图（仿照 SEE 的 Figure 1）

### 5.1 SEE 的 Figure 1 是怎么组织的

- **位置与尺寸**：单栏图，放在第 1 页右栏顶部，在引言旁边；图高约占半栏。
- **标题条**："Bridging the Gap in LALM Noise Quantification"。
- **三栏并排，用虚线分隔**：
  1. Baseline：直接输入噪声；
  2. 传统降噪；
  3. SEE & SEEN，即本文方法。
- **每栏的纵向结构相同**：输入（波形）→ 处理（embedding，加 SEE 指示）→ 模型输出（对话气泡，答错标 ✗，答对标 ✓）。
- **底部结论框**分三种颜色：红框 "Model overwhelmed by noise."、橙框 "Acoustic denoising alone is insufficient."、绿框 "Our approach targets cleaner representations."
- **图注**："Motivation and overview of ……"，一句话交代问题、已有方法的不足，以及本文如何解决。
- 读者看图就能明白三件事：问题是什么，现有做法为什么不够，本文方法为什么有效。

### 5.2 我们的 Figure 1：三栏，每栏一种"何时开口"的策略

- **标题条**："When should a simultaneous interpreter speak?"
- **例子**：一个示意性的 Zh→En 句子。源语音用英文逐词注释，不需要中文字体。
  - 源语音：[next week] [we] … [not] [go] [to Beijing]，否定词出现得晚。
  - 读到第 k 次时，只听到了 [next week] [we]。
  - 此时的草稿："Next week we will go"。
  - 完整上下文下的译文："Next week we will not go to Beijing"。
  - 例子和概率值都是示意，图注中写明 "illustrative"。

| 栏 | 标题 | 画面（从上到下） | 底部结论框 |
|---|---|---|---|
| (1) | Speak as soon as drafted（每次立即提交整份草稿） | 源语音时间轴，只听到前半句 → 草稿 "Next week we will go" 被整块送进 Talker → 听者已经听到 "go" → 后来的音频带来 [not]，完整译文变成 "will **not** go"，用 ✗ 标出 "go" 无法撤回 | 红框："Premature commitment: spoken words cannot be revised." |
| (2) | Wait for agreement（LocalAgreement） | 前后两份草稿逐词比较，只提交一致的前缀 "Next week we" → 听者时间轴：每个单元都要等下一读确认，Talker 空闲、听者那里出现静音段（灰色空白） → 一致性只有 0/1，看不到"还在播放的语音" | 橙框："A 0/1 agreement signal ignores what the listener is still hearing: needless silence." |
| (3) | QABC (ours) | 草稿每个单元上方画一根存活概率条 ρ（示意值：Next 0.97、week 0.95、we 0.90、will 0.72、go 0.31），水平虚线为阈值 τ(q) → 旁边是播放队列指示条：队列空时 τ 低（τ₀），提交 "Next week we will"；队列里还有语音时 τ 升高，暂缓提交，而听者仍在听前面的语音，没有静音 → 一次贪心解码完成 | 绿框："Commit what will survive; wait while queued speech hides the wait — in one decoding pass." |

**与正文各部分的对应**

- ρ 对应 §3.1 的稳定长度后验；
- 阈值 τ(q) 随队列变化，对应 §3.2 的 Theorem 1 和 Corollary 1；
- "一次解码"对应 §3.3–3.4。

图中出现的符号（ρ、τ(q)、q）与正文一致。

### 5.3 制作方式

- 用 TikZ 画，新文件 `figures/motivation.tex`。和现有的 `pipeline.tex`、`curves.tex` 一样是矢量图，可以直接改字。
- 配色沿用 `main.tex` 里已定义的颜色（`frozen`、`trained`、`decision`），三个结论框用红、橙、绿。
- 波形用简单的锯齿线；听者用耳朵或扬声器的小图形；静音段用灰色空白块。不使用 SEE 的插画素材，只借鉴它的版式。
- **尺寸**：单栏宽，高度约 7 cm（含图注约占 0.16 页），字号 `\scriptsize` 或 `\tiny`，三栏各约 2.6 cm 宽。如果画出来字太小，就改成跨栏 `figure*`、高约 4.5 cm，放在第 2 页顶部，约占 0.24 页。
- **图注草稿**：

  > Motivation (illustrative example). (1) Committing each draft at once speaks words that later audio changes, and spoken words cannot be revised. (2) LocalAgreement waits until consecutive drafts agree, a 0/1 proxy of stability that ignores the speech still queued for playback, so the listener may hear silence. (3) QABC estimates how likely each draft unit is to survive and commits it when this probability exceeds a threshold set by the playback queue: it waits while queued speech hides the wait and speaks when the listener would otherwise hear silence.

- **插入方式**：在 `main.tex` 中、`\input{sections/introduction}` 之前加一行 `\input{figures/motivation}`，用 `[t]` 放置，引言文件不动。
- **是否在引言里引用**：SEE 在引言中写了 "(Figure 1)"。是否在引言的缺口句末加 "(Figure~\ref{fig:motivation})" 由你决定（见第 8 节）；这会改动引言一处，只加一个括号。
- **编号变化**：动机图为 Figure 1，方法流程图变为 Figure 2，质量–延迟曲线变为 Figure 3。正文都用 label 引用，编号会自动更新。

## 6. 页数预算（正文严格不超过 8 页）

v14 的 Limitations 结束在第 8 页右栏约 77% 处，正文只剩约 0.08 页余量。v15 新增：

- 动机图：约 0.16 页；
- 完整 Table 1：约 +0.1 页；
- 合计约 0.26 页，另需约 0.05 页安全余量。

按以下顺序压缩，压够即停：

| 顺序 | 措施 | 预计节省 |
|---|---|---|
| 1 | 按第 2 节重排浮动体，回收第 6–8 页 12–17% 的空白 | 约 0.10–0.15 页 |
| 2 | 方法流程图压低约 25%：去掉图例块，收紧行距 | 约 0.08 页 |
| 3 | Related Work 从 4 段并成 3 段，去掉与引言重复的句子 | 约 0.08 页 |
| 4 | 结果部分精简：4.3 的 ACL60/60 段并入上一段；4.5 的计算代价只留一句 | 约 0.05 页 |
| 5 | 若仍超页：Table 4 下半部分的两行对照改为正文中一句话（数字照写） | 约 0.04 页 |

方法部分的命题、定理和定义一律不动。

## 7. 执行步骤

1. 用 `cp -a` 把 v14 复制为 `paper/bayes_simul_s2st_v15/`，v14 不动。
2. Table 1：恢复 v13 原表并精简表注；生成脚本删去压缩版 `body_cvss` 的部分。
3. 画 `figures/motivation.tex`，在 `main.tex` 中插入。
4. 按第 2 节调整浮动参数和各表的插入位置；按第 6 节压缩篇幅。
5. 附录：删除 Table 9，特征改为文字；F/G/H 和 E 的小表改用 `[H]`。
6. 检查：
   - 正文不超过 8 页；
   - 没有未定义的引用，没有 Overfull；
   - 每栏空白：正文不超过 12%，附录不超过 15%；
   - 逐页渲染第 1–8 页和附录最后两页，人工查看；
   - 摘要和引言与 v14 逐字相同（除非你同意在引言里加图的引用）。
7. 记录写入 CHANGELOG，脚本提交并推送。

## 8. 需要你决定的事

1. **Table 1**：恢复 v13 完整版，所有行和列都保留，跨栏表（推荐）；或者只把缺的行补回、仍做单栏。
2. **动机图的尺寸**：单栏、放第 1 页右栏，和 SEE 一样（推荐）；或者跨栏、放第 2 页顶部，字更大，但多占约 0.08 页。
3. **引言里是否加 "(Figure~\ref{fig:motivation})"**：加，只改动引言一处（推荐）；或者不加，图只靠图注独立说明。
4. **动机图的例子**：用示意性的 Zh→En 句子，源语音用英文注释（推荐，不需要中文字体）；或者用 En→Zh 例子，图中需要排中文，要加中文字体支持。
