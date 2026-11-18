# 单说话人论文（QABC）外部基线：论文版本改动记录

`paper/` 不进 git，论文目录的改动记在这里；填表脚本和检查脚本在 `experiments/s2st_baselines_v1/`。

## v13（2026-11-13 00:04 布拉格时间）

- 新目录 `paper/bayes_simul_s2st_v13_baselines/`，由 v12 `cp -a` 得到。v12 和更早的版本都没有改动。
  v12 当时的 `generated/ sections/ figures/ references.bib` 快照放在 `eval_outputs/s2st_baselines_v1/paper_base_v13/`，作为填表输入，所以脚本可以重跑。
- 填表：`fill_paper_v13.py`，约定与 `fill_paper_v12.py` 相同（论文的分档规则；tab_realsi_main 用三种子均值，tab_tiers / tab_tiers_main 用种子 0；加粗 = 我们跑过的系统里本行 ASR-BLEU 最高）。
  - X-Translator（R4，主点 xt-default，单次运行）：加入 tab_realsi_main（表 11）、tab_tiers（表 8）、tab_tiers_main（表 55，只加这一列）和 tab_baselines（表 16，全部 4 个点）。
  - Hy-SimulMT + Qwen3-TTS（R3，参考级联，三种子；每块生成上限 min(1800, max(180, 18×字符数)) 帧）：加入表 11、表 8、表 16。表 55 是单栏表，没有加。
  - QABC 提交块 + Qwen3-Omni 原生 Talker（R5）：附录 C 新增表 17（`generated/tab_omnitalker.tex`）和一段说明。
  - 正文：setup 段加了一句（列出新增的基线），results 段 RealSI 部分加了两句（X-Translator 只进 En→Zh m4、Zh→En m3–m4，比 QABC 低 6.2–6.8；论文 talker 在 8 个点都高于 Omni 原生 talker，+0.8–3.6）。
  - 参考文献：`zhao2026xtranslator`（arXiv 2607.17544，条目取自 `paper/gstar_si_v9/references.bib`）；`qwen2026qwen3tts`（HF 模型卡 Qwen/Qwen3-TTS-12Hz-0.6B-Base，写法与 `hy2026simulmt` 相同）。
- 编译：tectonic，与 v12 相同。没有未定义的引用或文献，没有 Overfull。页数：v12 32 页，v13 33 页；正文仍在第 9 页参考文献之前结束。
- 检查：`check_paper_v13.py` 重读结果 JSON（含执行者自己的 tiers.json 和 summary_tables.md），核对 .tex 中的每个新数字、正文里的范围、引用和编译日志，157 项全部通过。
- 没有执行的事项（等用户决定）：v12 的页数长度；把 RealSI 对比移到正文。

### v13 补充：曲线图（2026-11-13 00:06 布拉格时间）

- 按协调者要求，`figures/curves.tex`（图 2，第 17 页）加入 X-Translator 和 Hy-SimulMT + Qwen3-TTS，做法同 v12 的 `fill_curves`：
  注册点按 D1 取上包络（均值），数据写到 `data/realsi_<方向>_{XTranslator,HySimulMTQwen3TTS}.dat`，加两个图例项，图注补一句。
- 所有点都在 y 轴范围内，所以不加角注；X-Translator En→Zh 的最后一个包络点（25.62，3.37 s）超出 3.2 s，在图框处截断，图注写明。
  图仍可读，所以保留了全部点，没有退到只画主点。
- 重新编译：仍是 33 页，没有未定义的引用，没有 Overfull。`check_paper_v13.py` 加了曲线数据、addplot、图例和图注的检查，共 172 项，全部通过。

## v14（2026-11-18 14:50 布拉格时间）

按 `paper/bayes_simul_s2st_v14_revision_plan.zh-CN.md` 第 13–20 节修改，第 17–20 节记录了用户的各项决定。

- **新目录** `paper/bayes_simul_s2st_v14/`，由 v13 用 `cp -a` 得到，v13 本身没有改动。
- **模板**：换成 ICML 2027（`icml2027.sty/.bst`），新增必填的 AI Use Statement（`sections/aiuse.tex`）。
- **摘要和引言**：按用户要求没有改，与 v13 逐字相同。
- **正文**：方法、实验设置、结果和结论都写成新文件（`method_v14`、`setup_v14`、`results_v14`、`conclusion_v14`），v13 的旧文件留在目录里，但已不再被 input。
  - 方法：合并为一节，分四个子节。
  - 实验：分 4.1–4.6 六个子节。
  - 结论后加了 Limitations 段。
  - 正文共 8 个浮动体：图 1、图 2 和 Table 1–6。
  - 正文在 8 页以内，全文共 24 页。
- **图**：
  - 图 1 删去 FR 节点，并把 q_θ 统一为 p_θ。
  - 图 2 从附录移入正文，删去 TTB 和 QABC_Δ 两条曲线，图注也删去 envelope、registered 和对已删表格的引用。
- **正文表格**由 `experiments/s2st_baselines_v1/fill_paper_v14_body.py` 生成：
  - 每个原始数值都与 v13 的 generated 表逐一核对过；
  - 五个 seed 的均值 ± sd、consensus 两次训练和四个 checkpoint 的均值 ± sd 由脚本计算；
  - "Ahead 5/5" 取自 v13 的 `tab_v10_seeds_main`（En→Zh 最差的 seed 为 +0.003）。
- **附录**分为 A–H 八节：
  - A–C、F–H 在 `sections/appendix_v14.tex`；
  - D、E 在 `sections/appendix_train.tex`，由 `appendix_v14` 在 C 和 F 之间 input；
  - D、E 中的数值来自启动脚本、TensorBoard 参数和训练日志，部分只读查询了 S3 备份。
- **附录新增的小表**：`generated/app_{omnitalker,calib,timeaxes,robust,scope}.tex`。
- **补充材料 CSV**：v13 的 Table 14–16（全部工作点）由 `experiments/s2st_baselines_v1/export_v14_supplement.py` 导出到 `supplementary/`，投稿时需要一并上传。
- **删除的内容**（理由见方案第 14–19 节）：
  - Holm 和 p 值；附录中所有 p 值；逐 seed 的单值；
  - 一条时钟连续解码的全部内容；
  - 二值门、FR、播放缓冲；
  - 按词计时、无 envelope 和 talk 重采样的 p 列；
  - 决策规则对比、测试时 refine；
  - Zh→En 诊断、power、按方向选择；
  - "extensions that did not help"、早期错误表；
  - v13 的 Table 14/15/16/23/24/26/28/31/32/33。
- **ACL60/60 主数字**：保持 +2.63（用户决定）。
- **改正的错误**：
  1. v13 写 TTB 的先验头"预测存活"，实际训练标签是"是否落在 8 次采样的一致前缀内"，附录 C.2 已按实际改写；
  2. 训练数据：40k 句中 805 句留作 dev，其余 39k 句产生 235k 次读，正文和附录 C.1 已改；
  3. Prop 4 证明末尾的 "does not compete" 一句，以及 Lemma 2 证明的标题；
  4. 附录 C.1 的表述收窄为 "No test set is used for training the adapter or for choosing its checkpoint"；
  5. v13 "约 25 分钟"的训练时间找不到依据，正文已不写，附录 D 改用日志实测值：前 200 步每个 seed 用时 11–19 分钟，seed 0 跑满 800 步用时 42 分钟。
- **核实后仍找不到、因而没有写的值**：T1c 选第 1,500 步的依据、标签生成的 GPU 小时、TTB 先验头的实际启动命令、vLLM 版本。
- **编译检查**：tectonic 编译通过，没有未定义的引用，没有 Overfull；正文每栏空白不超过 18%，附录除最后一页外不超过 17%。

## v15（2026-11-18，布拉格时间）

按 `paper/bayes_simul_s2st_v15_revision_plan.zh-CN.md` 和用户第 10 轮的决定修改。

- **目录**：新建 `paper/bayes_simul_s2st_v15/`，由 v14 用 `cp -a` 复制得到，v14 不动。
- **摘要与引言**：摘要不变；引言只在第一段末尾加了 "(Figure~\ref{fig:motivation})"（用户同意）。
- **Figure 1：新增动机图** `figures/motivation.tex`（TikZ，单栏，位于第 1 页右栏顶部），版式参照 SEE 的 Figure 1：
  - 三栏分别为：每次立即提交整份草稿、LocalAgreement、QABC；每栏底部是红、橙、绿三色结论框。
  - 例子是示意性的 Zh→En 句子，源语音用英文逐词注释，图注写明 "illustrative"。
  - 插入点在 `main.tex` 中、引言之前。新增 TikZ 库 `patterns` 和宏包 `float`。
  - 原有的流程图和曲线图编号依次顺延为 Figure 2、Figure 3。
- **Table 1：恢复 v13 的完整离线表**（`table*`，共 15 行，含 ASR-BLEU、Text-BLEU、A.PCP、SIM-O 四组列）。表体逐字不改，只精简了表注，由 `experiments/s2st_baselines_v1/fill_paper_v15_body.py` 生成。
- **Table 2：转置成单栏表**：
  - 行为各系统，列为两个方向各四个延迟档；格子里只写分数，各点的延迟见补充 CSV。
  - ACL60/60 作为表下方的小块，只有 Omni、LA2、QABC 三行。
  - Vote 在 3 s 内只有一个工作点（En→Zh m4，28.32），因此不单列一行，改在表注中写明。
  - 所有数值都与 v13 源表逐一核对。
- **正文排版**：
  - 导言区设置浮动参数：`topnumber 3`、`dbltopnumber 2`、`topfraction 0.85`、`dbltopfraction 0.8`、`textfraction 0.12`、`floatpagefraction 0.85`。
  - 调整各浮动体的插入位置：Table 1 放在 §4.1 开头；曲线图放在 §4.2 末尾；Table 2–6 各自放在对应小节的开头。
  - 第 7 页不再是纯浮动体页。第 5–8 页每栏空白 1–16%。
- **篇幅压缩**：
  - 流程图从 0.82 倍版心宽缩到 0.72 倍；
  - 结果部分重写为 `results_v15.tex`，各段更紧凑；
  - 结论压成两句（`conclusion_v15.tex`）；
  - Related Work 删去两处冗余的句子成分。
  - 方法部分不动。正文结束于第 8 页（Impact Statement 也在第 8 页）。
- **附录**：
  - 删除 v14 的 Table 9（TTB 的 17 维特征表），改成附录 C.2 正文中的一句完整列举；附录 D.4 的引用随之改为指向附录 C.2。
  - 最后三张小表（`app_calib`、`app_timeaxes`、`app_robust`）改用 `[H]`；H 节的表嵌在该段两句之间。现在最后一页是"文字—表—文字"连续排版。
  - 附录 E 的推理设置表和 F 节的两张大表试过 `[H]`，会在第 22–23 页留下 20–25% 的空白，所以仍用普通浮动。
- **编译**：tectonic 编译通过，没有未定义的引用，也没有 Overfull；全文 24 页，正文严格在 8 页以内。

## v16（2026-11-18，布拉格时间）

按 `paper/bayes_simul_s2st_v16_revision_plan.zh-CN.md`（以第 6 节为准）修改。

- **目录**：新建 `paper/bayes_simul_s2st_v16/`，由 v15 用 `cp -a` 复制得到，v15 不动。
- **摘要与引言**：与 v15 逐字相同。
- **外部系统不再使用我们的 Talker**
  - 外部系统中只有 Hy-SimulMT 接过我们的 Talker，共 4 个点，其中 Zh→En 约 1.75 s 处 21.88 高于 QABC。
  - 整个"Hy-SimulMT + 我们的 Talker"配置删除，包括：Table 2 的这一行、Figure 3 的曲线和图例、§4.1 和 §4.3 中的相关句子、附录 C.2 中的描述和 Fairness 段，以及补充 CSV（由新脚本 `experiments/s2st_baselines_v1/export_v16_supplement.py` 过滤）。
  - Hy-SimulMT 只保留 + Qwen3-TTS 这一种配置。Table 2 中 Zh→En m2 的最高分改为 QABC（21.42）；§4.3 改为 "m1–m3 领先，m4 LocalAgreement 领先"。
  - 策略基线（LA2、Hold-n、Vote、TTB）照旧共用我们的 Thinker 和 Talker，所有格子都保留。
- **方法部分**：依据 v13 的 Remark 原文，为每个定义、定理、推论补写动机句和用途句（方案 6.3 的 T1–T7）：
  - Proposition 1 之后改写为一段解释；
  - Definition 2 之后、Definition 3 之前、Theorem 1 之后、Corollary 1 之后、Proposition 2 之后、Proposition 3 前后各补一句。
  - 队列模型从 Assumption 环境改为普通段落 "Queue model."。这样附录中的两个假设成为 Assumption 1–2，正文按顺序引用；原来对 `ass:queue` 的引用都改为引用 Eq. (queue)。
- **附录 A、B 的证明**
  - 导言区加入 thmtools / thm-restate；正文的 Proposition 1–3、Theorem 1–2、Corollary 1 改为 `restatable`。
  - 附录中每个结果先按同一编号重述，再写 `\begin{proof}…\end{proof}`，右侧自动出现 □。手写的 $\square$ 全部删除，共 11 个证明。
  - Proposition 1 合成一个证明，分 (i)(ii)(iii) 三步。原来属于命题陈述的 Eq. (utt-post) 移到第 (i) 步，作为中间结果。
  - 方框之后的说明另起一段，包括 "Content and stopping terms" 和 Lemma 1、Lemma 2 之后的说明句。
- **修正一个从 v13 起就存在的排版问题**：摘要和引言的源文件直接写了 Unicode "→"，Times 字体没有这个字形，PDF 中箭头被丢掉（显示成 "EnZh"）。在导言区用 newunicodechar / DeclareUnicodeCharacter 把它映射为 `\rightarrow`，摘要和引言的源文件不动。
- **篇幅与排版**
  - 正文压缩：流程图缩到 0.6 倍版心宽；动机图的图注和结论框收紧；Table 2 表注精简；§3.1 "Simultaneous commitment" 段、Theorem 2 之后一句、结果部分两处和 Limitations 都做了精简。定义、命题、定理和新增的说明句都没有删。
  - Table 1 改在 §3.4 开头插入，使其落在第 6 页顶部。浮动体分布：Table 1 在第 6 页，Table 2–4 在第 7 页，Figure 3 和 Table 5–6 在第 8 页。
  - 附录 G、H 的开头句并入段落，避免第 25 页只剩两行。
- **编译**：tectonic 编译通过，没有未定义的引用，也没有 Overfull；全文 24 页，正文结束于第 8 页（第 9 页从 Impact Statement 开始），每栏空白不超过 16%（参考文献末页除外）。
