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
