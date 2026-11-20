# 鲁棒第十七波 b：感知层 LoRA 改用 CoVoST 加噪数据重训 预登记

写于 2026-11-20 18:10 CET，在本波任何冒烟、训练和评测数字之前。方法本身见主线方案 4.3 节（`reports/mist_w3po_v1/robust_mainline_plan_20261120.zh-CN.md`）和第十七波预登记 `robust_wave17_enclora_prereg.zh-CN.md` 的实现说明，本文件只写与第十七波不同的地方。

## 1. 动机
- 第十四波 UniST-ERM-WiSE 两条主判定都失败（commit 48a3c2c83）。相对 erm_wise05，12 个单元全部显著变差；zh2en/a/clean 相对底座 −4.49。
- 推测原因有两点：
  - UniST pilot 的源音频是 BiCodec 重建的；
  - 译文是合成的，与 CoVoST 人工参考风格错配。
- 第十七波用的是同一份 UniST 数据。对只训练音频编码器的方法，"源音频是 codec 重建的"是最大的混淆：编码器可能学到的是 codec 伪迹，而不是噪声鲁棒性。所以第十七波的结论，无论通过还是失败，都无法归因到"感知层适配"本身。
- 第十七波 b 只换数据，用第五波 ERM 的 CoVoST 加噪数据，即主系统 ERM-WiSE 用的同一份数据。这样得到的是一个干净的对照：同一份数据，LoRA 加在编码器上（本波）还是加在 LLM 文本层上（erm_wise05）。
- 这不是重跑失败的方法：第十七波原波按预登记照常评测、照常报告，由 uniss-2b 的执行者负责，本波不碰它。

## 2. 与第十七波的唯一差别：数据
- `DATA = eval_outputs/mist_w3po_v1/wave5/M/data`，是第五波 `wave5_m_train.py` 的数据：中→英 7004 行，英→中 118983 行，clips 已在盘上。
- 不变的部分：
  - 8 个环境：{zh2en, en2zh} × {clean, music, crowd, sfx}；
  - sampler 的 sha1 键 "20261119|w5m|row|s|j"、"20261119|w5m|mix|s|j"，噪声池（Idea 2 背景索引的 train 划分），BVR 分布，限幅器；
  - LoRA 位置：音频编码器 32 层的 q/k/v/out_proj，rank 16，alpha 32，LLM 冻结；
  - 2300 步，全局批量 64，AdamW 1e-4，余弦调度，种子 20261119，world 2（每个 rank 32 个槽位）。
- 因为数据和 sampler 都与第五波相同，本波与第五波 ERM 每一步看到的样本完全相同，只差 LoRA 加在哪里。
- 最终系统：enc_cv_wise05，即 alpha 减半（WiSE 0.5，与 erm_wise05 同一做法）。enc_cv（α = 1）作为辅助一并评测。目录名沿用第十七波的 `eval_enc_wise05` / `eval_enc`，放在 `eval_outputs/mist_w3po_v1/wave17b/` 下。

## 3. 判定（沿用主线方案 4.3 的两条主判定，参照换成 CoVoST 的 erm_wise05）
1. **enc_cv_wise05 相对底座，用统一规则：**
   - 每个方向 × {a, b} × {干净, 带噪} 单元，text BLEU（相对三次底座运行的均值）和 ASR-BLEU（相对 S1）都满足点估计 ≥ −0.3 且 CI 上界 ≥ 0；
   - 至少一个带噪单元（方向 × {a, b, b_no_crosstalk}）的 text BLEU 增益 CI 下界 > 0，且点估计超过噪声底。
2. **enc_cv_wise05 相对 erm_wise05：**
   - 至少一个单元（方向 × {a, b, b_no_crosstalk} × {干净, 带噪}）text 差的 CI 下界 > 0，且超过噪声底；
   - 没有任何方向 × {a, b} × 条件 × {text, ASR-BLEU} 单元的点估计 < −0.3 且 CI 上界 < 0。
- **结论：** 两条都过 = PASS，进组合（替换或叠加在 erm_wise05 上另行预登记）；只过第 1 条 = "对 ERM-WiSE 无增量"；第 1 条不过 = FAIL。
- 只评一个设置，不扫 α、rank、步数。

## 4. 机理检验（辅助，写定的预测，同 4.3）
- **预测 1：** enc_cv（α = 1）在中→英 (a) 干净上不显著掉分，即 text 相对底座均值的 CI 上界 ≥ 0。第五波文本层 ERM（α = 1）在这里显著掉分，第十三波诊断为 LLM 层记住了窄的 CoVoST 中→英数据。
- **预测 2（编码器不记忆译文）：** 在第十三波的 600 条 CoVoST 中→英训练子集上（`wave13/data/items_zh2en_train600.jsonl`，与第十三波 H4 同一批 clip），enc_cv − base 的提升远小于第十三波文本层 ERM 的 +13.31。
  - "记忆"定义为点估计 ≥ 3.0 且 CI 下界 > 0；
  - base 和 erm 复用第十三波的 `la2_base_train600`、`la2_erm_train600`，erm − base 必须逐位复现 +13.31（自检）。
- **预测 3：** 英→中带噪的提升主要来自离线翻译，而不是提交（oracle 分解）。
  - enc_cv 跑 `--policy oracle`，用第十二波的 (a) 条目，两个方向都跑；
  - 底座的 oracle 用 `wave3_combos/D/oracle_<dir>`，同第十七波。

## 5. 卡和链条
- 只用卡 6–7。要等第十七波原波的评测和机理作业交还卡之后才上，判断标准是：wave17 的 watcher STOP，或者 CHAIN_DONE / CHAIN_FAILED，并且卡 6–7 上连续 3 次检查（60 s 一次）都只有占位进程或没有进程。
- 卡 4–5 归 uniss-2b 的 ncwi_erm，本波不用。
- 冒烟也只在卡 6–7、并且在卡交还之后跑：先试 `--mb 32`（6 步），失败再试 16。
- 训练卡上不叠任何别的作业。
- 链条 `wave17b_chain.sh` 一次做完：等卡 → watcher → 冒烟 → 训练 → WiSE 副本 → 评测 enc_wise05、enc（合并用 `wave17_merge.py`，不改动，带逐位自检）→ 机理作业 → touch STOP → `wave17b_mech_score.py` → `wave17b_score.py`（自检：erm_wise05 相对底座与 `wave5/M/summary_erm_wise05.json` 逐位一致，已在现有文件上验证：180 个数 + 规则 1 判定）→ CHAIN_DONE。

## 6. 文件
- 训练：`wave17b_enc_train.py`。它是 `wave17_enc_train.py` 的副本，只改了 DATA 一行；另外在 adapters_final.json 里记一个 "data" 字段，WiSE 副本生成时会断言这个字段。
- 启动 / 链条 / 评测 / watcher：`wave17b_run.sh`、`wave17b_chain.sh`、`wave17b_eval.sh`、`wave17b_eval_orch.sh`、`wave17b_noplaceholder.sh`。
  - 自己的 PIDS：`logs/mist_w3po_v1/wave17b/w17b_workers.pids`；
  - 自己的 STOP：`w17b_noplaceholder.stop`；
  - torchrun 端口 29667。
- 打分：`wave17b_score.py`（去掉 unist_wise05，规则 2 的参照为 erm_wise05）、`wave17b_mech_score.py`（训练子集用第十三波的 CoVoST 600 条，n_sampled 按第五波 sampler 重算，600 条全部被抽到过）。
- 检查：都只做了 bash -n、DRY_RUN、Python 语法检查，以及在现有文件上的打分自检，没有试跑启动脚本。

## 启动记录（只追加）
- **2026-11-20 18:10:40 CET** 启动 `wave17b_chain.sh`，pid 3766228，日志 `logs/mist_w3po_v1/wave17b/chain.log`。启动时卡 6–7 仍在跑第十七波原波（训练将结束，随后是评测和机理作业），链条在等它们交还卡。
- **预计（CET）：** 第十七波原波约 21:00–22:00 交还卡 6–7；然后冒烟约 5 min，训练 2300 步约 2.5–2.7 h（按第十七波约 4 s/步；CoVoST 句子更短，可能更快），评测 enc_wise05 / enc 与机理作业约 2–2.5 h。结果约 2026-11-21 01:30–03:00。
