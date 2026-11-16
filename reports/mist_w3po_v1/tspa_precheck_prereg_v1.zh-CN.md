# TS-PA（双边播放对齐）CPU 预检 P1 / P2：预登记 v1

- 写于 2026-11-16，布拉格时间下午，**早于任何 P1 / P2 数字**；提交后才写代码和运行。
- 方法设计见 `crf_effectiveness_vs_hmt.zh-CN.md` §8.4。
- 只用 CPU，只用 dev 和 train 数据；AliMeeting、ACL、RealSI 和所有测试集都不碰。
- 一个设置，不扫参。

## 共同定义

- **空隙**：同一条流（窗口 × 输出说话人）中相邻块 n、n+1 之间的静音，`start_{n+1} − (start_n + dur_n)`。≥ 500 ms 记一次停顿，与 `talker_block_pauses.py` 的定义相同。
- **句内**：块 n 的文本去掉末尾的引号和括号后，最后一个字符不在 `.?!。？！` 中。
  - 这里比 `talker_block_pauses.sent_end` 多认中文标点；那个函数不改。
- **每分钟**：按产出语音的分钟数归一（Σ 时长 / 60 s）。

## P1：格上界（真实到达时刻，事后可知）

**数据**：
- (a) **Zh→En**：`speech/plcrf_dev5_cands/talker/cands_*.jsonl`（dev5，A′ 文本，每块 8 个候选），窗口 `windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl`。
- (b) **En→Zh**：`speech/train16_en2zh_v2p_20261112T204539Z_s02000/talker_cands/cands_*.jsonl`，窗口 `windows_en2zh_c2g06_train16_v1/windows_train_online.jsonl`，`--lang cmn --units char`。没有 En→Zh 的 dev 候选，这是唯一可用的 En→Zh 候选集。
- (c) 仅作描述：Zh→En train16 候选（`train16_crf_v2p_s3000_finalpost_c2g06`），窗口 `windows_zh2en_c2g06_train16_v1/windows_train_online.jsonl`。

**格**：`playback_lattice.Stream` 原样使用，`talker_rows.stream_of` 构造，availability 规则不变。

**PL-EM（参照，即现在的方法）**：
- 预算 b = 规则路径的语料延迟 − 0.3 s；
- λ\* 用 `PL.solve_lambda` 求出；
- 后验 q_PL = CRF(λ\*)。

**TS-PA**：
- 对 n < N−1，定义
  - `G_n(s, c) = a_{n+1} − (a_n + 80·s + D_c)`（ms，s 是积压格点）；
  - `P_n(s, c) = 1[G_n ≥ 500]`；
  - ω_n：句内为 1，句末为 0.25。
- 势：`ψ_n(s, c) = feat_c − λ·l_n(s, c) − κ·ω_n·P_n(s, c)`。
- 约束：
  - `E_q[Σl]/ΣW ≤ b`（与 PL-EM 相同的 b）；
  - `E_q[Σ ω P] ≤ 0.75 · E_{q_PL}[Σ ω P]`。
- 求解：嵌套二分。κ\*(λ) 取满足空隙约束的最小 κ；λ 取使 `latency(λ, κ*(λ)) ≤ b` 的最小 λ。上限 λ ≤ 200、κ ≤ 200，二分 40 次。
- 可行 = 两个约束都在容差内满足（延迟 2e-3 s，空隙相对 1e-3），且 λ、κ 都没有撞上限。

**报告**（q_PL、q_TS、规则路径、两个 CRF 的 Viterbi 路径各一行）：
- 停顿/分钟（全部、句内），后验取期望；
- 延迟代理 Σl/ΣW (s)；
- 平均 CTC 保真度 f、平均 ℓ̄；
- 语速（单位/秒）。

另外报告：q_PL 的 Viterbi 路径上，有多少比例的停顿是在该路径的积压状态下，块 n 的任何候选都填不上的（"填不上的占比"）。

**P1 的 GO 条件**（(a)、(b) 各自满足下面全部四条）：
1. TS-PA 可行；
2. TS-PA Viterbi 路径的句内停顿/分钟 ≤ 0.75 × PL-EM Viterbi 路径的；
3. TS-PA Viterbi 路径的延迟代理 ≤ PL-EM Viterbi 路径 + 0.10 s；
4. `E_{q_TS}[f] ≥ E_{q_PL}[f] − 0.05`（每块平均，nats/字符）。

**正确性检查**：新的空隙势在小流上（≤ 4 块，每块 ≤ 3 个候选）的前向–后向结果，与暴力枚举一致（log Z、E[ΣωP]、E[Σl]，容差 1e-9）。单元测试不过就不运行 P1。

## P2：到达的可预测性（因果）

**数据**：dev 全集上 MIST 的实际输出，即 B 的文本 + 读级合并 + PA Talker：
- Zh→En：`speech/jdev_bz_joint_v8h6k_20261113T073658Z_s06000__rlplem`；
- En→Zh：`speech/jdev_be_joint_v8h6k_20261113T073658Z_s06000__rlplem`；
- 都是 TTS 种子 1，`talker/results.jsonl`。种子 2（`_tts2`）只作复现描述。

**样本**：每条流里有后继块的块 n。
**标签**：`y_n = 1[start_{n+1} − (start_n + dur_n) ≥ 500 ms]`（所有时间都取 talker 记录的布局）。

**特征**：都在块 n 交给 Talker 的时刻 `t_n = write_ms_n` 计算，因果。diarizer 帧 f 只有当 `t_f ≤ t_n − 1040 ms` 时可见，对应论文 Eq. causal 的界 γ。
- 基线组（PA Talker 现在已经能看到的信息）：
  - backlog = start_n − write_ms_n；
  - 块 n 的单位数 W_n；
  - 句末标志。
- 预测组（新增）：
  - 是否在最终读写出（final）；
  - 窗口关闭是否已被检测到（t_n ≥ close_detect_ms）；
  - debt = max(0, (t_n − 1040) − 1000·end_s_n)：块 n 的源端终点之后，已经可见的源时间（秒）；
  - pend = 说话人槽位 k（`__S{k}` 对应槽位 k−1）在 (end_s_n, t_n − 1040] 区间的 Σ P × 0.08 s（块 n 之后又说了多少，以秒计）；
  - act1 = 该槽位在 [t_n − 2040, t_n − 1040] 的平均 P。

**模型**：
- 标准化后做 L2 逻辑回归（sklearn，C = 1，class_weight = None）；
- 按会议分 5 折（GroupKFold），用折外概率计算 AUC；
- 基线组和全部特征各拟合一次。

**P2 的 GO 条件**（Zh→En、En→Zh 各自满足）：
- 全部特征的 AUC ≥ 0.75；
- 且全部特征的 AUC − 基线组的 AUC ≥ 0.03（新增的预测信号要带来 Talker 现在看不到的信息）。

## 总体判定与之后的步骤

- P1 与 P2 都 GO：写 TS-PA 的 GPU 阶段登记（补采慢两档的候选、训练、dev 评测、R3′+gap 对照、门槛见 §8.4），**交用户决定是否上 GPU**。
- 任何一项 NOT GO：不上 GPU，不改阈值重算，把结果和原因写进报告。
