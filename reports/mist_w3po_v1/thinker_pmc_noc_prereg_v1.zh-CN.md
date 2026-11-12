# PMC-noC：去掉 PMC 内容项的消融对照（预登记，2026-11-12 布拉格约 11:05）

- **登记时机**：在 PMC-1 训练期间、看到任何 PMC 评测结果之前提交并 push。
  - PMC-1：`pmc_20261112T072924Z`，4–7 号卡，布拉格 08:29 开训，预计约 12:50 训完，速查约 13:45。
  - 登记时只看过 PMC-1 训练日志里的启动核对项：前 10 步的 rank 数、端口、λ 是否绑定、有没有 NaN，以及训练进度（步数）。没有看过任何 dev 解码、文本、语音或延迟结果。
- **提出者**：协调者（用户批准），Thinker 执行者 v5 写成登记。
- **照实说明**：这个对照是在看过 VE1、VE1-s2、RC5 的结果之后设计的。
  - VE1-s2（只换训练种子）对 VE1：语音 −0.40（配对 P 0.987），文本 ov −0.45。
  - 也就是说，同一配方、只换训练种子，语音就能差 0.4。下面的判读线把这一点写进去了。
- **要回答的问题**：PMC 的收益（如果有），是来自新的内容项（q^pol 加权的内容交叉熵），还是来自写出窗前移一读加上 λ* 期望延迟预算。
- **范围**：只有一种设置，不扫参，不重试，不换种子。只用 AMI train 和 dev；RealSI、ACL 60/60、AliMeeting、AMI test 不碰；不用 COMET / MetricX；不用 QABC（与 PMC 登记相同）。

## 1. 设定（唯一）

- **名称**：PMC-noC。
- **与 PMC-1 完全相同**（`thinker_pmc_prereg_v1.zh-CN.md` §1）：
  - `--objective latent_write --lw-mode pmc`；
  - 写出窗 [max(0, e − 1), min(R − 1, e + 1)]；
  - λ* 每步求解，把 q_λ 的期望延迟卡在 e 路径（证据标签路径）上；
  - 数据 `crf/sched_t5_merge3_w10_v1`（merge3 证据标签），1.0 s 读；
  - DART 与清单原样复用 `crf/pmc_dart_merge3_w10_v1`（模式 pmc，计划种子 20261109；PMC-noC 的计划与 PMC-1 相同，训练器加载时逐条核对）；
  - `run_crf_v2p.sh` 的全部参数，3000 步，学习率 1e-4 → 1e-5，训练种子 20261109。
- **唯一的区别**：PMC 内容项的权重设为 0，即 M 步只用 q_λ 加权（`serial_v2p_train.set_coefs`），与 VE1 的 M 步完全一样。
  - PMC-1 的内容系数：c_tok[t] = h_t (Σ_{b>t} cend[b] + Σ_{b>t} cend_pol[b]) / D；
  - PMC-noC：c_tok[t] = h_t Σ_{b>t} cend[b] / D；c_end 两者都只按 q_λ。
  - q^pol 照样计算，只用来记录 `Eq_lat_pol`、`exp_early_pol` 等训练动态，不进入损失。
- **所以 PMC-noC = "VE1 的 M 步 + PMC 的写出窗 + λ* 预算"**，PMC-1 − PMC-noC 只差内容项。
- **卡**：0–3 号卡（slotA），4 卡。按 takeover 语义接手占卡程序，不留空档。

## 2. 代码与 CPU 预检（登记之后做；只做加法，默认行为不变；任何一项不过就不训练，报告协调者）

- **代码**：
  - `crf/latent_write_train.py`：`set_coefs_pmc` 加参数 `w_pol`（默认 1.0；w_pol = 0 时直接等于 `set_coefs`；1.0 时与现在逐位相同）；`run_step` 加参数 `pmc_w`（默认 1.0）；pmc_w ≠ 1 时 metrics 多一个字段 `pmc_w`。
  - `crf/train_crf_sft.py`：新参数 `--pmc-content-weight`（默认 1.0，只在 `--lw-mode pmc` 下允许改，必须 ≥ 0）。
  - 被改的模块都是写新文件后 `command mv -f` 替换。在跑的 PMC-1 已经加载了旧模块，不受影响。
- **单元测试**（新文件 `crf/test_pmc_noc.py`）：
  - N1：参数解析。默认 1.0；`--lw-mode pmc --pmc-content-weight 0` 解析为 0.0；在其他模式下给非 1 的值会被拒绝；负值被拒绝。
  - N2：系数。在合成的多说话人窗口和真实 VE1 窗口上：w_pol = 0 的系数与 `V2T.set_coefs` 逐位相同；不给 w_pol 与显式给 1.0 逐位相同；w_pol = 0.5 等于 set_coefs + 0.5 · h · suf(cend_pol) / D（误差 ≤ 1e-12）。
  - N3：训练一步（小模型，CPU，与 test_pmc 的 P5 相同的设置）。pmc_w = 0 时：
    - 损失有限；Σ c_tok × D = Σ h（VE1 的 M 步：每个 token 只计一次），Σ c_end × D = Σ R；
    - λ*、q_λ 的 Eq_lat、q^pol 的统计与 pmc_w = 1 的同一步逐位相同（内容项不影响后验）；
    - metrics 里有 `pmc_w` = 0。
    - 不给 pmc_w 的一步与 pmc_w = 1.0 的一步，损失和梯度逐位相同。
- **回归**：`test_pmc`（P1 / P2 / P5）、`test_latent_write_crf`、`test_train_crf_sft`、`test_dart_aug` 全过。
- **CPU 预检**（`crf/pmc_noc_precheck.py` → `crf/pmc_noc_precheck_v1.json`）：
  - 在 PMC-1 训练计划的 30 个抽样步（真实窗口，dummy 打分）上：
    - 计划和格（每块的写出窗）与 `--lw-mode pmc` 的逐位相同；
    - w_pol = 0 的系数与 `V2T.set_coefs`（VE1 的 M 步，用同一个 pmc 后验）逐位相同；
    - 报告 PMC 内容项在 PMC-1 内容权重里所占的比例（描述）。
  - 训练器参数 `--lw-mode pmc --pmc-content-weight 0 --dart crf/pmc_dart_merge3_w10_v1` 能解析；清单通过 `load_manifest` 核对。

## 3. 评测（与 PMC-1 的训后流程相同的形状）

- `pmc_after_train.sh` 照搬，换名字和目录：
  - dev 全集解码 `dev_pmcnoc_s3000`（c2g06 窗口，1.0 s 读，8 分片，`--cap135 --exclude 4 5 6 7`）；
  - dev5 截取和 PMC 的 GO 线（只作描述）；
  - 读级块合并；**合并一步与 PMC 一样适用协调者的预先授权**：只因 pair_lost 停住、合并输出完整、能配对的窗口 ≥ 3589 / 3618 时，用续跑脚本继续；否则报协调者；
  - PL-EM talker，两个 TTS 种子；门槛参考：A′、A′rl + PL-EM、VE1rl + PL-EM（都只作描述）；
  - 听感：UTMOS / SIM、talker WER、句中停顿、单词块占比。
- 训练或评测有失败就停，报协调者，不改设定重跑。

## 4. 比较与判读（登记，结果之前写死）

**主比较**：PMC − PMC-noC（候选 PMC-1，参考 PMC-noC），两边都是读级块 + PL-EM、两个 TTS 种子。
- 用 `gate_2seed`（两种子合并，按会议配对 bootstrap）得到文本 all、文本 ov、语音的配对差和 P(差 ≤ 0)。
- 每条流语音 LAAL（`speech_laal_online` all，两种子平均）的差。

**判读 A：内容项对质量的贡献**（看语音）：
1. 语音差 > 0 且 P(差 ≤ 0) < 0.05，并且差 ≥ 0.40 → **"内容项有贡献，而且超出了已观察到的训练种子波动"**。0.40 是 VE1 与 VE1-s2 之间的语音差。
2. 语音差 > 0 且 P < 0.05，但差 < 0.40 → **"方向支持内容项，但幅度在已观察到的训练种子波动之内，不能归因"**。
3. 其他情况（差 ≤ 0，或 P ≥ 0.05）→ **"没有证据表明内容项有贡献"**。这时 PMC 相对 VE1 的变化与"来自写出窗前移 + λ*"相容。
- 文本 all / ov 的差和 P 一起报告，但不进入判读 A 的结论。如果文本 all 与语音方向相反（一正一负，且各自 P < 0.05 或 > 0.95），要特别写明。

**判读 B：内容项对延迟的影响**（看每条流语音 LAAL）：
- |PMC − PMC-noC| ≤ 0.05 s → **"延迟由写出窗 + λ* 决定，内容项不改变延迟"**；
- PMC 低 0.05 s 以上 → "内容项也降低了延迟"；
- PMC 高 0.05 s 以上 → "内容项提高了延迟"。

**局限（照实写在结果里）**：
- 每边只训一次。bootstrap 的 P 只覆盖测试集抽样，不覆盖训练随机性；判读 A 第 1、2 档的区分就是为此设的。
- 两个模型用同一个训练种子和计划，但损失不同，训练轨迹会分开。所以配对差里仍含有训练噪声。

**只作描述（不是门槛）**：
- PMC-noC 自己的门槛：语音对 A′rl + PL-EM，文本对 A′，每条流语音 LAAL 与 3.756 s 比较。
- PMC-noC 对 VE1rl + PL-EM 的配对差。这个差 = 写出窗前移 + λ* 相对 VE1 的效果，同样含训练噪声。
- dev5 速查的各项；EndOffset、单通道、排队、最后一块；文本 LAAL；≥ 10 s 拆分；听感（UTMOS 对 PMC 的同流配对差等）。
- 训练动态：λ 绑定比例、E_q[Lat] / 词、exp_early_q / exp_early_pol。
- 如果 PMC-1 的评测没有完成，就只报 PMC-noC 自己的结果，主比较不做。

## 5. 算力
- 训练：0–3 号卡，4 卡。PMC-1 实测约 5.2 s/步，约 4.3–5 小时；PMC-noC 的计算量与 PMC-1 相同（q^pol 照算，只是系数不同）。
- 评测：解码约 30 分钟，PL-EM 语音链约 30–50 分钟，听感约 10 分钟，都在 0–3 号卡。

## 结果

（训练和评测完成后写在这里。）
