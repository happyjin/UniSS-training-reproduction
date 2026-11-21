# 鲁棒音色线：TTS 种子 3、4 测量扩展（预登记）

写于 2026-11-21 13:10 CET，早于本文件任何数字。**只是测量扩展，不是新方法**：B、V1、AbS、AbS-C 的设置与 token 表都不变，不重跑任何搜索。`robust_b_abs_prereg.zh-CN.md`（含"确认运行"）与 `robust_abs_c_prereg.zh-CN.md` 的原判定（都为 not_established，AbS 的"ASR 下降得到确认"）**保持不变**，本文件结果只作附加测量。

## 1. 动机
两个 TTS 种子下，AbS / AbS-C 收回 B 噪声身份损失的 ECAPA2 比例为 82% / 76%（V1 47%），但对 V1 的单格 ASR-BLEU 差（−0.4..−0.7）在两种方法里出现在不同格子，且同一格在两个种子间最多移动约 0.9（V1 自己单格最多移动 0.81）。"每格 ≥ V1 − 0.3"的规则可能主要被 TTS 采样噪声主导。再加两个种子，用 4 种子均值、种子间 SD 与 4 种子合并 bootstrap 判断 ASR 是否非劣。

## 2. 运行（GPU，只用卡 4–7）
- 臂与 token 表（全部已有，不改）：
  - B：`experiments.uniss_qwen3_omni_v1.evaluation.si_infer`（无覆盖）；
  - V1：`idea1_si_infer_vo --voice-override eval_outputs/mist_w3po_v1/wave3_combos/V/tokens/tokens_v1_<d>.jsonl`（与 b_abs 所用相同）；
  - AbS：`--voice-override eval_outputs/mist_w3po_v1/b_abs/tokens/tokens_abs_<d>.jsonl`；
  - AbS-C：`--voice-override eval_outputs/mist_w3po_v1/abs_c/tokens/tokens_absc_<d>.jsonl`。
- 其余参数与 `b_abs_seed2_orch.sh` / `abs_c_orch.sh` 完全相同：同一 Thinker 文本块（`wave2_S1/thinker_<d>/blocks_<d>_w1.jsonl`）、T1c Talker（`si_t1c_20261021T181440Z/autoext/export_1500/uniss`）、`--window 1 --text-policy file --speed-policy block --gpu-mem 0.1`、`SI_KV_CACHE_GIB=7.73`、4 分片；只把 `--seed` 改为 **20261023（种子 3）** 与 **20261024（种子 4）**。
- 布局：每个种子一波，en2zh 分片 0,1 → 卡 4、2,3 → 卡 6；zh2en 分片 0,1 → 卡 5、2,3 → 卡 7；每卡 4 臂 × 2 分片 = 8 个 Talker 同时（固定 KV，跳过显存 profiling），同卡相邻启动间隔约 45 s；种子 3 全部结束后再启动种子 4。
- 每个 (臂, 方向, 种子) 一条指标链（共 16 条，每卡 4 条，每卡 2 条 zh2en + 2 条 en2zh）：speech-only ASR（`asr_speech.jsonl`）→ TitaNet（`wave3_v_spksim`）+ ECAPA2（`a_sv_ecapa2_spksim`）（GPU）+ SQUIM PESQ（`wave2_s1_score.py pesq`，CPU）。
- 脚本 `experiments/mist_w3po_v1/voicecond/voice_seeds34_orch.sh`（由 `b_abs_seed2_orch.sh` / `abs_c_orch.sh` 拼成；setsid nohup，fds 8/9 关闭；先 `bash -n` 与 `DRY_RUN=1`）。每个 GPU 任务前按 PID 停该卡占位（argv 复查），最多等 120 s 直到卡上没有 gpu_load 进程；自己的 watcher 实例（`b_abs_noplaceholder.sh`，POLL=1，PIDS / STOP 在本次日志目录）。不碰 2726951 / 52045 / 1222453。拒绝覆盖已有输出。
- 输出 `eval_outputs/mist_w3po_v1/voice_seeds34/`（`talker_<arm>_<d>_s<seed>`，`metrics/`），日志 `logs/mist_w3po_v1/voice_seeds34/`。评分 `experiments/mist_w3po_v1/voicecond/voice_seeds34_score.py` → `voice_seeds34/summary.json`。
- 种子 1（20261021）、种子 2（20261022）用已有输出（路径同 `abs_c_score.paths`）。
- 一次运行，不扫参，不因结果重跑；任何 Talker / 指标失败 → FAILED，原样报告。

## 3. 报告
格子 = 方向 × {a, b, b_noxt} × {带噪, 干净}；主格 = en2zh / zh2en × {a, b} × {带噪, 干净}（8 格），b_noxt 只报告。条目集合 = 4 个种子 × 4 个臂都有输出的条目交集。
1. 每格、每臂（B / V1 / AbS / AbS-C）：4 种子均值与种子间 SD（样本 SD，n−1 = 3）：语料 ASR-BLEU、SQUIM PESQ 中位数；以及 AbS − V1、AbS-C − V1 的每种子差的 4 种子均值与 SD。
2. 收回 B 噪声损失比例（每格 = 方向 × set；ECAPA2 与 TitaNet，相似度中位数）：每个种子用**该种子自己的 B** 带噪 / 干净中位数作分母和基线，再取 4 种子均值与 SD；另报 4 格（主格）平均。
3. 4 种子合并配对 bootstrap 95% CI：ASR-BLEU AbS − V1、AbS-C − V1；单位 = stream，2000 次重采样，种子 0；每个单位把 4 个种子的 BLEU 充分统计量相加（= 每条目对种子取平均的语料统计量，乘常数不改变 BLEU），与 `b_abs_seed2_score.pooled_asr` 相同算法。主格与 b_noxt 都算。
4. 只报告：种子 3、4 各自的格值（语料 ASR-BLEU、PESQ、相似度中位数），以及种子 1–4 的对照（种子 1、2 应复现已有表）。

## 4. 事先写定的判读（不改原判定）
对每个方法 M ∈ {AbS, AbS-C}：若在**全部 8 个主格**中，每格满足
**4 种子均值 (M − V1) ASR-BLEU ≥ −0.3** 或 **4 种子合并 CI 下界 ≥ −0.5**，
则记 M 为"**ASR 上非劣**"（相对 V1，在本测量口径下）；否则记"**ASR 非劣未得到支持**"，并列出不满足的格子。
- 身份与 PESQ 只报告（4 种子均值 ± SD），不入本判读。
- 差值用未取整的数计算。
