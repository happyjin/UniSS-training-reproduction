# 鲁棒第十三波（诊断）：ERM 的干净中→英掉分从哪里来（2026-11-20 13:10 CET，写在任何数字之前）

## 已知（第五、十二波）
- 中→英 (a) 干净：ERM α = 1 为 −3.63 [−6.39, −1.14]，ERM-WiSE α = 0.5 为 +0.34。英→中干净在所有检查点都是正的。
- 第十二波：掉分在第 500 步（2.3 个 epoch）就已出现（−2.18），之后不单调。"越训越过拟合"不成立。
- 训练集：中→英 7004 条、22 个说话人，约 10.5 个 epoch；英→中 118983 条、9157 个说话人，约 0.6 个 epoch。每步两个方向的样本数相同（各 32 条）。

## 四个候选原因与对应测量（都不训练新模型）
- **H1 提交策略：** 掉分来自 LA2 在部分音频上的提交，而不是整句翻译质量。全句训练让模型在部分音频上更倾向"补全"成训练集里见过的句式，再被两次草稿认同后提交。
  - 测量：oracle 提交（`--policy oracle --windows 1`，第三波 D-GPU 的同一设置；整句听完后的最终 draft 即离线翻译）。看离线 BLEU 和提交损失（oracle − LA2）。
- **H2 翻译质量：** 整句离线翻译本身也掉分。测量同 H1。
- **H3 双向干扰：** 掉分来自和英→中共享同一个 LoRA。
  - 测量：第十波的单方向 D_zh2en（同样的 zh2en 样本、同样步数，不含英→中）按 α = 1 合并，跑 LA2。
- **H4 窄数据记忆：** 掉分来自对 22 个说话人、7004 条句子的记忆，训练集上变好而 dev 上变差。
  - 测量：按 sha1("20261120|w13|train|i") 从中→英训练集选 600 条干净训练音频，比较底座与 ERM（α = 1）的 LA2 BLEU。只作诊断，不用于训练或选择。

## 方法（固定）
- 系统：
  - base：底座；LA2 用 S1 已有输出，oracle 用第三波已有输出（若与 S1 的条目一致）；
  - erm：α = 1，即 `wave5/M/merged_erm`；
  - erm_wise05：`merged_erm_wise05`；
  - dzh：D_zh2en，α = 1，用 `wave5_m_merge.py` 合并第十波的 `run_zh2en`。
- 单元：(a)，中→英为主，英→中为对照，干净和带噪。text BLEU 用第五波的口径和 bootstrap；底座 3 次运行的噪声底只用于 LA2。
- 运行：
  - oracle：erm、erm_wise05 × 两个方向；
  - LA2：dzh × zh2en；
  - 训练子集 LA2：base、erm × zh2en 干净。

## 判读（写定）
- **H1 成立：** zh2en 干净上，erm − base 的离线 BLEU（oracle 最终 draft）≥ −1.0，且 CI 上界 ≥ 0；同时 erm 的提交损失比 base 多 ≥ 2.0。
- **H2 成立：** erm − base 的离线 BLEU ≤ −2.0，且 CI 上界 < 0。
- **H3 成立：** dzh − base 的 LA2 干净 zh2en ≥ −0.3（不掉分），而 erm 掉分。若 dzh 也掉分（点估计 ≤ −2.0），H3 不成立。
- **H4 成立：** 训练子集上 erm − base ≥ +3.0，且 CI 下界 > 0（训练集明显变好，而 dev 变差）。
- 几条可以同时成立。全部只用于理解原因和写论文的分析部分，不改主系统。
- 辅助：WiSE 在离线和提交两部分各自的变化，说明为什么减半能把干净掉分拿回来。

## 实现说明（2026-11-20 13:35 CET，写在任何数字之前）
- **离线翻译怎么取：** oracle 输出（`blocks_<dir>_w1_oraclecommit.jsonl`）每行带 `drafts`，最后一个 draft 的 t ≥ 音频时长，就是整句听完后的最终 draft。
  - 第三波 base 的 oracle 输出已核对：zh2en 2012 行、en2zh 1274 行，每行都有 t ≥ dur 的最后 draft，且 blocks 拼接 == 最后 draft（逐行相同）。所以"离线 BLEU"= oracle 输出的 text BLEU，不必另跑 `--windows` 大于最长音频的离线运行。
  - 注意：这个最终 draft 以已提交前缀为条件；oracle 提交的前缀本身就是同一 prompt 下整句音频贪心续写的前缀（temperature 0），因此它和无前缀的离线翻译只差分词 / 批次带来的数值差异。照预登记把它当作离线翻译。
- **base 的 oracle：** 复用第三波 `wave3_combos/D/oracle_<dir>`。已核对：它用的 items 文件就是 S1 的 `wave2_S1/data/items_<dir>.jsonl`（条目集合与 S1 的 LA2 输出完全相同）；第十二波 (a) 单元的 600 条 items 是它的逐字段相同子集；`si_thinker_stream.py` 自 10-29 起没有改动，口径一致；底座即默认 `--thinker`。
- **新运行一律只跑 (a) 单元：** items 用 `wave12/data/items_<dir>_a.jsonl`（每方向 600 行 = 300 对 × 干净/带噪），窗口 1 s，gpu-mem 0.85。
  - oracle：`--policy oracle`，`--thinker` 分别为 `wave5/M/merged_erm`、`merged_erm_wise05`，两个方向，共 4 个运行。
  - LA2：`--policy la2`，`--thinker wave13/merged_dzh`，zh2en，1 个运行。
  - dzh 合并：`wave5_m_merge.py --adapters wave10/run_zh2en/adapters_final.pt`（rank 16、alpha 32，scale 2.0，即 α = 1），输出 `wave13/merged_dzh`。
- **训练子集（H4）：** 对 `wave5/M/data/rows_zh2en.jsonl` 的行号 i（0 起），按 int(sha1("20261120|w13|train|i")) 升序取前 600 行（i 用行号，与第六波 `h(f"...|{i}")` 的约定一致）。
  - 音频为 `clips/zh2en/<id>.wav`（干净），dur_ms 从 wav 读，参考译文为行的 `ref`，id 为 `t_clean__<id>`，写成 `wave13/data/items_zh2en_train600.jsonl`（字段同 S1 items）。
  - 跑 base（默认 Thinker）和 erm（`merged_erm`）两个 LA2 运行，窗口 1 s。只作诊断，不用于训练或选择。
  - 这个子集没有底座的 3 次运行，所以不算噪声底。
- **打分（`wave13_score.py`）：** 第五波口径，`wave2_s1_score.Bleu`（normalize_for_bleu，sacrebleu zh / 13a），单元为句子（(a) 的 stream），配对 bootstrap 2000 次，seed 0，95% CI 取 2.5 / 97.5 百分位。
  - 离线 / oracle：erm − base、erm_wise05 − base，base 为第三波的单次运行。
  - LA2 的 base 一律用 3 次底座均值（S1、base_rerun1、base_rerun2；第五波 erm − base_mean = −3.63 的口径），同时给出相对 S1 单次运行的数作敏感性，并报告噪声底。
  - 提交损失 = oracle BLEU − LA2 BLEU（同一系统，同一单元）。"erm 的提交损失比 base 多"= (erm_or − erm_la2) − (base_or − base_la2_mean)，在同一组 bootstrap 重采样上算 CI；用 S1 的版本作敏感性。
  - 自检：erm / erm_wise05 的 LA2 和 erm − base_mean 必须与 `wave5/M/summary.json` 相同，否则不出结果；base 的 oracle zh2en/a 干净必须等于第三波 `oracle.json` 的 35.385（两位小数）。
- **判读的落地：**
  - H1、H2 都看 zh2en/a/clean。H1 的"多 ≥ 2.0"看点估计。
  - H3："erm 掉分"用第五波的 −3.63。若 dzh − base_mean 的点估计在 (−2.0, −0.3) 之间，记为"不确定"（规则两条都不满足）。
  - H4：训练子集上 erm − base 的点估计 ≥ +3.0，且 CI 下界 > 0。
  - 其余单元（en2zh 对照、带噪）只列表，不参与判读。
- **卡：** 卡 0–1（guard 占位）。用自己的 watcher `wave13_noplaceholder.sh`（PIDS `logs/mist_w3po_v1/wave13/w13_workers.pids`，STOP `w13_noplaceholder.stop`）。编排 `wave13_eval_orch.sh` 在两张卡上串行跑 7 个运行。卡 2–3 在本波不需要。不碰卡 4–7。
