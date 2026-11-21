# 鲁棒线 S1「见证代数」预登记（2026-11-21 09:25 CET，写在任何数字之前）

只做推理，不训练。属于 `robust_bottleneck_v6_20261121.zh-CN.md` 第 2 优先级（中→英流式提交，只做推理期、不加延迟的提交规则），不是新的 Thinker 训练方案。来源：`robust_next_ideas_v2_20261120.zh-CN.md` S1。

## 1. 动机
- 带噪时，中→英流式约 1.9 text BLEU 的损失来自「噪声 × 提交」交互：前缀不稳定，提交了后来被推翻的内容。
- CV-LA（LA2 前缀与 DeepFilterNet3 增强视角草稿的公共前缀）收回一部分，但延迟 +0.3–1.3 s。等延迟对照（`robust_combo_plan_20261120.zh-CN.md` 末尾）表明：它的价值是「随噪声自适应的保守度」，不是跨视角信息本身。
- 假设：增强视角的一致性可以**代替**时间一致性（等下一次读），而不是和它取交集。这样带噪时仍然自适应地保守，干净时不用多等一个读窗。

## 2. 规则（每条一个设置，不扫参）
记第 t 次非末次读的 v0 草稿为 d_t（已提交前缀 + 贪心续写），v1 草稿 d_t^{v1} 是同一前缀的增强视角（`wave15/views`，与 v0 同一个 vLLM batch、同一提示）。已提交 n 个单元，提交 d_t[n:cut]（cut ≤ n 时不提交）。末次读提交 v0 的完整草稿（同 LA2），不需要 v1。

| 规则 | cut | 说明 |
|---|---|---|
| LA1 | \|d_t\| | 每次读提交当前整份草稿（参照） |
| LA2 | LCP(d_{t−1}, d_t) | 现有底线，= si_thinker_stream la2 w1；第一次读不提交 |
| LA3 | LCP(d_{t−2}, d_{t−1}, d_t) | 参照；凑齐三份草稿前不提交 |
| CV-LA | min(LA2, LCP(d_t, d_t^{v1})) | 已有运行 `wave15/eval_cvla_base`，不重跑 |
| **AoV** | LCP(d_t, d_t^{v1}) | 只用视角见证，不等下一次读 |
| **EITHER** | max(LA2 的 cut, AoV 的 cut) | 任一见证即可；第一次读只有视角见证 |

- 每条规则有自己的提交轨迹（d_{t−1}、d_{t−2} 是该规则自己的前几次草稿）。
- 底座 Thinker `pretrained_models/Qwen3-Omni-30B-A3B-Instruct`，不加适配器，贪心，max_tokens 160，seed 20261021，提示与 si_thinker_stream la2 w1 相同。
- 代码：`experiments/mist_w3po_v1/voicecond/s1w_si_thinker.py`（复制 combo_si_thinker.py 的读循环，原文件不改）。LA2 用新脚本重跑一遍，使曲线上的点来自同一代码路径；原 wave2_S1 LA2 只作批次噪声参照（LA2_S1，不进曲线）。

## 3. 数据与指标（只报点估计）
- S1 dev（`eval_outputs/mist_w3po_v1/wave2_S1/data`）：英→中 1274 条、中→英 2012 条；单元格 = 方向 × 集合 a / b × 干净 / 带噪。
- **text BLEU：** `wave2_s1_score.Bleu`（论文规范化，sacrebleu zh / 13a），单元格内语料级；假设 = 各块文本拼接。
- **D1（文本级 segment LAAL，主延迟）：** 每个提交单元一个延迟 = 所在块的写出时刻（不超过 |X|），`si_metrics.laal`，|Y*| = 参考单元数（同 `wave5_viewmarg.d1`）；对句子取平均。
- **D2：** 同一组延迟走 `laal_defs.laal_unitwise`（D2 的公式）。说明：文本级没有朗读时刻，且写出时刻不超过 |X|，所以 D2 与 D1 在数值上应当相同（laal_defs 已核对 `si_laal ≡ laal_unitwise`）；照报，不另作判据。
- **先提交后被推翻占比：** 照 `wave15_commit.py`：有非末次提交的句子上，1 − LCS(非末次提交, 参考) / |非末次提交| 的平均；另报非末次提交占比。
- 打分：`s1w_score.py`；编排：`s1w_orch.sh`（等 K1 第 2 步结束：`k1/step2/DONE` 或 `FAILED` 存在且无 k1_ 进程；卡 1、2、3、7 每卡一个 vLLM Thinker，--gpu-mem 0.85；占位进程按 PID 停，≤120 s；自带 watcher POLL=1）。

## 4. 判定（写定）
- **LA 族曲线：** 每个单元格，LA1、LA2、LA3 三点按 D1 排序，折线连接。某规则的曲线值 = 在包住它 D1 的两个相邻 LA 点之间对 BLEU 线性插值；D1 落在 LA 范围外时用最近一段外推，并标记「外推」。margin = 该规则 BLEU − 曲线值。
- **AoV 或 EITHER 记为「有效」，需同时满足：**
  1. 中→英两个带噪单元格（a、b）margin 都 ≥ +0.5；
  2. 中→英两个干净单元格（a、b）margin 都 ≥ −0.3。
- 英→中只报告，不进判定。CV-LA 的 margin 同样计算，作参照。
- 每个单元格打印曲线表（LA1、LA2、LA3、CV-LA、AoV、EITHER 的 D1、BLEU、margin、推翻占比）。
- 批次噪声参照：新 LA2 与 LA2_S1 的 BLEU 差只报告（历史复跑差约 0.3）。

## 5. 预期与风险
- AoV 第一次读就能提交，延迟应低于 LA2；风险是两个视角一起幻觉（尤其干净时 v1 ≈ v0，AoV 退化为 LA1）。
- EITHER 的延迟不会高于 LA2，可能接近 LA1，同样有干净时退化为 LA1 的风险。
- 如果两条都不有效，结论是：增强视角只能作为附加的保守度（CV-LA），不能代替时间见证。

## 结果
（跑完后只追加）
