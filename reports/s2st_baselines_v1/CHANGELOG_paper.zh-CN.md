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
