# 鲁棒 K3：在 V1 去噪音色编码器的目标里加入说话人可辨性（InfoNCE）预登记

2026-11-21 07:25 CET（布拉格）写，早于 K3 的任何训练、预检或评测数字。依据 `robust_review_v5_20261121.zh-CN.md` 第 1.2 节与第 4 节 K3；用户 07:20 CET 决定 K1–K3 同时开始，K3 用卡 0、6。

## 1. 动机
- V1（voicecond Idea 1 的去噪编码器对照臂，`voicecond_idea1/m2/ctrl/encoder.pt`，sha1 `fce60570…`）在测试上把带噪时的说话人相似度损失只收回 13–59%（中→英最差），干净输入上还略降相似度（中→英 RealSI 0.458 → 0.440）。
- 原因假设：V1 的损失 ‖φ(y) − u*‖² 是 MMSE 回归，估计向平均音色收缩，"干净但更像普通人"。
- K3 直接针对这一点：在同一个目标里加入跨说话人的判别项，要求带噪码离自己的干净码比离同一批里其他说话人的干净码更近，抑制向平均收缩。

## 2. 文献
- 判别式说话人嵌入：AAM-softmax（ArcFace 式，Deng et al. CVPR 2019；用于说话人验证）、GE2E（Wan et al. ICASSP 2018）；
- InfoNCE / 对比预测编码（van den Oord et al. 2018）；DINO 说话人编码器（arXiv 2311.09770）。
- 新意：中低（组件常见）；价值在于直接针对已测得的 MMSE 收缩问题，并在 Talker 端用身份指标检验。

## 3. 方法（一个设置，其余与 V1 完全相同）
- 训练脚本 `experiments/mist_w3po_v1/voicecond/k3_discrim_train.py`：逐行复制 `idea1_e2e_train.py --mode ctrl`，只加一项损失。
- 与 V1 相同：数据（`data/open_voicecond_v1/idea1/train_en` + `train_zh`，各 20000 对，LibriSpeech train-clean-100 251 人 + AISHELL-1 400 人）、初始化（冻结 BiCodec 编码器）、可训练模块（ECAPA-TDNN + Perceiver + project_in，BN / dropout eval 模式）、u* = φ₀(s)（冻结编码器）、AdamW lr 5e−5、200 步预热 + 余弦到 0、6000 步、每批 32 对、权重衰减 0、裁剪 1.0、fp32、种子 20261119（含 DataLoader 生成器）、只存最终检查点、ID-1 dev 监控只记录。
- 损失：

$$
L=\|\phi(y)-u^\star\|^2+1.0\,\|\phi(s)-u^\star\|^2+1.0\,L_{\rm NCE},\qquad
L_{\rm NCE}=-\frac1B\sum_i\log\frac{\exp(\cos(\phi(y_i),u^\star_i)/0.1)}{\sum_{j\in\mathcal N(i)}\exp(\cos(\phi(y_i),u^\star_j)/0.1)}
$$

  - cos 在展平的 192 维 u（32×6，未绑定）上计算；u*_j 是冻结目标，不回传梯度；
  - 𝒩(i) = {i} ∪ {j：说话人(j) ≠ 说话人(i)}。说话人键 = 语言 + manifest 的 `speaker`。同一批可能出现同一说话人两次（651 人、每批 32 对，期望约 0.8 次/批），**同说话人的其他条目从负例中屏蔽**（不作负例，也不作正例）；
  - 温度 0.1、权重 1.0，固定，不扫。
- **不是蒸馏：** 目标仍是冻结编码器在干净语音上的码 u*（与 V1 相同）加说话人标签（数据自带）；没有教师模型的软输出、伪标签或学生-教师结构。
- **与 voicecond 线的重叠：** K3 是 voicecond Idea 1 对照臂的修改版，与 voicecond 线（Idea 1 / Idea 7 的"收缩"讨论，以及 K2 的最优传输修正）重叠。用户 07:20 CET 命令 K1–K3 先开始，结果归哪篇论文待用户决定。

## 4. 预检（dev 码，便宜，Talker 评测之前）
- 数据：ID-1 diag dev（`diag_en` 500 对 40 人、`diag_zh` 500 对 107 人，split = dev），每对有干净 s、带噪 y、说话人标签。
- 编码器：φ₀（冻结）、V1、K3。对每种语言分别报告：
  1. **散布比**：tr(Σ_between(φ(y))) / tr(Σ_between(u*))，Σ_between = 说话人均值码（192 维）的协方差；另报干净输入 φ(s) 的同一比值；
  2. **码级余弦**：cos(φ(y), u*) 与 cos(φ(s), u*) 的中位数；
  3. 记录：带噪码到干净说话人中心的最近中心说话人识别率（只作记录）。
- **停止规则（固定）：** 若 K3 在两种语言上带噪散布比都不高于 V1（机制没有起作用），停止，不做 Talker 评测，报告预检。否则继续。

## 5. 评测与判断规则（S1 dev，与 wave3 V 组完全相同）
- 底座：S1 单说话人 LA2（`wave2_S1`），同一 Thinker 文本块；T1c Talker（`si_t1c_20261021T181440Z/export_1500`），种子 20261021，4 分片；音色 token 由 `idea1_si_infer_vo.py --voice-override` 注入，提示文件与 sha1 与 B / V1 相同（`wave3_combos/V/tokens/prompts`）。
- K3 token：对 V1 表中每条提示，`Q(φ_K3(y))`（与 V1 的 `Pc.tokens(Pc.phi_wave(y))` 同一计算，只换编码器）。
- 复用：B = `wave2_S1/talker_<d>_s20261021`，V1 = `wave3_combos/V/v1/talker_<d>`，两者的 SQUIM / TitaNet 指标 = `wave3_combos/V/metrics/{B,v1}_<d>_*`。只新跑 K3。
- 指标：ASR-BLEU（speech-only 协议，`asr_speech.jsonl`）、SQUIM PESQ 中位数、TitaNet 相似度中位数（对干净源），同 `wave3_v_score.py`；点估计，不做 bootstrap。
- 格子：方向 {en2zh, zh2en} × 集合 {a, b} × 条件 {noisy, clean}；另报 b_noxt（只报告）。
- **判断规则（K3 对 V1，每个方向 × 集合 {a, b} 都要满足；任何一格不满足即 K3 不成立）：**
  - 带噪相似度：按条配对的中位数差（K3 − V1）> 0；
  - 干净相似度：按条配对的中位数差（K3 − V1）≥ 0；
  - ASR-BLEU：带噪、干净两种条件下 K3 ≥ V1 − 0.3；
  - PESQ：带噪、干净两种条件下按条配对的中位数差（K3 − V1）≥ −0.1。
- 另报 K3 对 B 的同样数字。

## 6. 运行
- `experiments/mist_w3po_v1/voicecond/k3_orch.sh`（setsid nohup 脱离，fds 8/9 关闭）：训练（卡 0）→ 预检（卡 0）→ 停止规则 → token（卡 0）→ Talker（en2zh 4 分片卡 0、zh2en 4 分片卡 6，间隔 45 s）→ ASR → SQUIM / TitaNet → 打分 → `k3/DONE`（失败写 `k3/FAILED`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `k3_noplaceholder.sh`（POLL 1 s）。不碰 guard 2726951、gpu_watch 52045、sync 1222453。
- 输出 `eval_outputs/mist_w3po_v1/k3/`，日志 `logs/mist_w3po_v1/k3/`。

## 结果
