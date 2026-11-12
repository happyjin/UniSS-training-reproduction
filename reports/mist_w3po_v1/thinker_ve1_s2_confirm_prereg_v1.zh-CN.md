# VE1-s2：VE1 换训练种子的确认实验（预登记，2026-11-12 布拉格约 06:30；在任何 VE1-s2 数据、训练和结果之前提交）

- **提出者**：协调者（代用户），布拉格约 06:25。
- **照实说明**：这个确认实验是在**看过 VE1 的探索性 dev 全集结果之后**设计的（`thinker_crf_t_results_v1` 的"VE1：dev 全集（探索性）"一节）。VE1 的登记速查是 NOT GO（dev5 文本 ov −0.84），这个判定不变。探索性全集结果：
  - 文本 28.07 / 23.70 / 29.99；
  - VE1rl + PL-EM：语音 23.64（对 A′rl + PL-EM −0.06，P 0.64），每条流语音 LAAL 3.756 s（A′rl + PL-EM 4.050）。
- **要回答的问题**：VE1rl + PL-EM 的收益（语音不降、语音 LAAL 降约 0.3 s）是 VE1 这个配方可复现的效果，还是那一次训练碰巧得到的。

## 1. 设定（唯一）

- **训练**：与 VE1（`thinker_v2p_evidence_ve1_prereg_v1`，`start_ve1.sh`）完全相同，**只换训练种子**：`--seed 20261112`（VE1 用的是默认值 20261109）。
  - 相同的部分：
    - `run_crf_v2p.sh` 的全部参数（`--tau 0.5 --final-read-full-post 1`、3000 步、每步约 64 个读行、学习率 1e-4 → 1e-5 余弦、v8 初始化、LoRA r32 / alpha 64、`--feat-per-row 1 --fast-moe-bwd 1 --prefetch 1 --empty-cache 0 --tok-budget 32000 --mem-cap-gib 120`）；
    - `--objective latent_write --lw-mode v2p`；
    - 数据 `crf/sched_t5_merge3_w10_v1`（T5 的证据时刻标签，merge3 粒度，1.0 s 读）；
    - DART 参数不变（扰动比例 0.5，p_item 0.3，kmax 3，DART 抽样种子 20261111）。
  - 训练种子决定的东西：窗口进入各步的顺序（`plan_steps`）、后验扰动的抽样、`torch.manual_seed`（新参数的初始化）。
  - **DART 清单要重建**：清单绑定训练计划（`summary.json` 的 plan 种子），所以用同一个 `dart_aug --manifest --lw-mode v2p --sched crf/sched_t5_merge3_w10_v1`，只加 `--plan-seed 20261112`，输出新目录 `crf/ve1s2_dart_merge3_w10_v1`。训练器加载清单时会核对：计划种子、每一条都在本计划内、每个（步, 窗口）的抽样都与清单一致。不一致就退出。
  - `train_crf_sft.py` 本来就有 `--seed` 参数，不改代码。
- **卡**：0–3 号卡（slotA），4 卡，约 3.2 小时。按 takeover 语义接手占卡程序。
- **推理与评测**（与 VE1 探索性评测的 PL-EM 那条链相同）：
  - dev 全集解码：c2g06 窗口，1.0 s 读，8 分片，`--cap135`，只用 0–3 号卡（`--exclude 4 5 6 7`）；
  - 读级块合并（`merge_read_segments.py`，不改）；
  - **PL-EM talker**（`--talker plem`，与 A′rl + PL-EM 相同），两个 TTS 种子；
  - 不跑 c1+b1+s2 talker 和逐段消融。

## 2. 主终点（dev 全集；三条全部满足才算"确认"）

`gate_2seed`（两种子合并，按会议配对 bootstrap），门槛线与以往相同：差 ≥ −0.2，且 P(差 ≤ 0) < 0.95。

1. **语音**（VE1-s2rl + PL-EM）对 **A′rl + PL-EM**（`dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem` 与 `_tts2`）：在线内。
2. **文本 all 和文本 ov** 对 **A′**（`dev_crf_v2p_s3000_finalpost_s8_c2g06` 与 `_tts2`）：两项都在线内。
3. **每条流语音 LAAL**（`speech_latency_lit` 的 `speech_laal_online` all，两种子平均）≤ **3.85 s**（与 VE1 / D1 预登记的全集延迟线相同）。

- **判读**：三条都满足 → "VE1 配方的收益在第二个训练种子上复现"；任一条不满足 → "没有复现"，并写明是哪一条。不调任何参数，不换种子重试。

## 3. 只作描述（不是门槛）

- **dev5 速查**：从全集解码截出 dev5 的 1092 个窗口（`t_dev5_cut.sh`，只用 CPU），报告 GO 线 1–3 的各项、停顿分类、送到 talker 的块。不作门槛。
- 另外两个参考的语音门槛（对 A′、对 A′ + PL-EM）；语音 ov / 1spk；EndOffset；单通道 LAAL；排队；最后一块；单元终点分解。
- **VE1 与 VE1-s2 两次训练**：文本 all / ov / 1spk、文本 LAAL、语音、每条流语音 LAAL、EndOffset 的均值和差（VE1-s2 − VE1）。
- ≥ 10 s 拆分：文本用 `early_commit_diag --light`（对 A′），语音 LAAL 用 `laal_breakdown`（按单元时长）。

## 4. 不做

- 只用 dev，不碰任何测试集；不调参数，不扫种子；不用 COMET / MetricX；不用 QABC 方法。

## 5. 代码与启动（登记之后做，只做加法；都只用 `bash -n` 和 DRY_RUN 检查）

- `scripts/start_ve1s2.sh`：先建 DART 清单（只用 CPU，约 5 s），再等 slotA 上只剩占卡进程，noclobber 占槽，按 `takeover_train.sh` 接手，核对 4 个 rank。
- `scripts/ve1s2_after_train.sh`：等 CRF_RUN_EXIT（rc ≠ 0 或没有 step_03000/DONE 就停），释放 slotA，然后解码 → 合并 → PL-EM 语音链（三个参考的门槛）→ dev5 截取。

## 结果

（2026-11-12 10:46 布拉格写入。详细数字见 `thinker_crf_t_results_v1.zh-CN.md` 的"VE1-s2"一节。）

- **训练**：`ve1s2_20261112T052308Z`，布拉格 06:23–09:36，rc=0，RANKS_IN_SYNC ×6。
- **评测**：dev 解码、读级块合并、PL-EM 语音链（两个 TTS 种子）、dev5 截取、听感都已完成，没有失败作业。
- **已知偏差**：合并工具报 pair_lost = 2（IB4010_on_0067、TS3004c_on_0052）。最终读那一段的写出时刻比对应的块晚一次读，段和块配不上对。
  - 协调者决定照用合并输出续跑，工具、输出和命令都不改，用续跑脚本 `ve1s2_after_merge.sh`。
  - 合并后能配对的窗口是 3599 / 3618，与 VE1rl 相同。
- **主终点**：
  1. 语音对 A′rl + PL-EM：23.24 vs 23.69，−0.45，CI [−0.81, −0.06]，P 0.988 → **不满足**。
  2. 文本对 A′：all −0.32（CI [−0.74, 0.05]，P 0.952），ov −0.42（CI [−1.03, 0.19]，P 0.917）→ **不满足**。
  3. 每条流语音 LAAL 3.622 ≤ 3.85 → 满足。
- **判读**：**没有复现**（主终点 1、2 不满足）。VE1 的"语音不降"在第二个训练种子上没有复现，"语音 LAAL 降"复现了。不调参数，不换种子重试。
- **描述**：
  - VE1-s2 − VE1：文本 all / ov −0.21 / −0.45，语音 −0.40（配对 P 0.987），每条流语音 LAAL −0.134 s，EndOffset −0.126 s。
  - 两次训练的均值：语音 23.44（对 A′rl + PL-EM −0.25），语音 LAAL 3.689（−0.36 s）。
