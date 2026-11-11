# talker 两个新实验的预登记：实验一"无格"消融（BoN-SFT）、实验二 单通道 PL-EM v2（2026-11-11）

**登记时间**：2026-11-11 06:25 UTC，在任何训练行、预检、训练或评测结果出来之前提交并推送（`git push private HEAD:main`）。
**用户批准**：约 05:50 UTC，两个实验都批准，要求立即开始。下面每个数值都是**唯一设置**，不扫超参。

前置文件：`talker_crf_prereg_v1.zh-CN.md`（含 §5 修订）、`talker_crf_results_v1.zh-CN.md`。下文"PL-EM"指 v1 的方案一（导出 `eval_outputs/mist_w3po_v1/talker_crf/talker_plem_20261111T001759Z/export`），"PL-EM c1+b1+s2"指它在 A′ 推理设置下的 dev 结果（语音 23.62，每条流语音 LAAL 4.071，单通道 LAAL 5.64）。

**用户规则**（不放宽）：预检结果不影响是否训练；ASR-BLEU 与文本 BLEU 都不得下降；只用 train 训练、dev 选择，RealSI / ACL 60/60 / AliMeeting 不碰；不用 COMET / MetricX；不扫超参。

---

## 0. 两个实验共同的部分（与 PL-EM 完全相同）

- **训练候选**：v1 的 train 采样（`eval_outputs/mist_w3po_v1/speech/train16_crf_v2p_s3000_finalpost_c2g06/talker_cands/cands_*.jsonl`）：15 个 train 会议、8640 条流、19077 块，每块 8 个候选（A′ 语速 ×4、再快 2 档 ×4），每个候选带 talker 累积对数概率和 CTC NLL。不重新采样。
- **历史**：A′ 规则路径（每块在 A′ 语速的前 2 个候选里取 CTC NLL 最小者，与采样时的条件一致）。
- **输入格式**：`--backlog-token`（S2T 头的语速 token = block pace，后接积压 token：0.5 s 一档，8 档，≥ 3.5 s 为第 8 档，用 8 个保留 id）；TTS 头的语速 token = 候选自己的语速。KL 锚 T1c 读原格式序列。
- **行**：每块候选各一行，CE 权重 = 该块的 q，q < 0.01（Q_MIN）的候选不入行，剩余的归一化；语速软标签用该块完整的 q（阈值之前）在 35 个语速 token 上聚合；每块一次。
- **训练**：`run_talker_crf.sh MODE=plem` 原样（`pretrain_talker_crf.py --crf-mode plem`）：从 T1c 导出版本起训，talker 全参数；LR 3e-5（warmup 50，cosine 到 3e-6）；1200 步；0–3 号卡，GBS 4（每卡每步一组行，每组 ≤ 24000 个填充后 token）；weight decay 0.01，clip 1.0，bf16；对冻结 T1c 的 KL 权重 0.5（只在语音码位置）；收尾 token 的 CE 权重 3；语速损失权重 1.0；种子 20261110。两个实验只换行文件。
- **起点、卡、步数**都与 PL-EM 相同。行文件写成 `mode = plem`（训练入口原样读取），另记 `variant = bon / meeting`。

## 1. 实验一："无格"消融（BoN-SFT）

**问题**：PL-EM 的收益是来自播放格和延迟预算，还是只是 best-of-N 微调？

- **唯一改动是权重**：每块独立，在它的 8 个候选上
  q(c) ∝ exp(τ·lp̄_c + β·f_c)，τ = 0.5，β = 5，f = −min(CTC NLL 每字符, 10)，lp̄ = talker 每 token 平均对数概率（与 PL-EM 的 `feat` 相同）。
  - 没有延迟项、没有 λ、没有后向消息，块与块独立；
  - 语速软标签由这个 q 得到；
  - 保留 Q_MIN = 0.01 的行阈值。
- **实现**：`talker_rows.py --mode bon`（新增，只做加法）。默认模式的输出逐位不变：单元测试对合成数据做哈希对比；另外在真实 train 候选上重建 PL-EM 的行，与已有的 `rows/plem.pt` 逐项相等（已验证）。
- **推理**（与 PL-EM c1+b1+s2 完全相同）：`--speed-policy queue --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 --speed-shift 2 --backlog-token --select-k 2`，CTC 打分服务，A′ 的文本，全 dev，两个 TTS 种子（默认 20261021 / 20261109）。
- **主指标**：语音官方 ASR-BLEU 和每条流语音 LAAL，**对照 PL-EM c1+b1+s2**（同一文本、同一种子，配对）。

**解读规则**（看结果之前定下）：Δ = BoN − PL-EM（两种子均值）。
- 质量："BoN 与 PL-EM 持平" ⇔ BoN 以 PL-EM 为参考过门槛（Δ语音 ≥ −0.2，且按会议配对 bootstrap P(Δ ≤ 0) < 0.95）。不过 ⇒ 格对质量有贡献。
- 延迟："BoN 与 PL-EM 持平" ⇔ 每条流语音 LAAL 的 Δ ≤ +0.05 s（与 v1 预检"LAAL 不变差"的容差相同）。Δ > +0.05 s 且两个种子同号 ⇒ 格和延迟预算对延迟有贡献。
- 两项都持平 ⇒ 结论写成"PL-EM 在每条流设定下的收益可由 best-of-N 蒸馏解释"，如实报告。
- 补充（不判定）：单通道 LAAL、EndOffset、归一化按段语音分与差距、WER、UTMOS / SIM。

## 2. 实验二：单通道 PL-EM v2（会议共用一条输出通道）

**设定**：一个会议里所有说话人的译文语音在**一条**输出通道上播放（一个同传声音）。输入音频本来就是单通道混合。

### 2.1 格（每个会议一条链）

- **块的顺序**：会议内所有说话人的块，按会议绝对时间的可放出时刻 a_n 先进先出（FIFO）。a_n = 窗口起点 + max(write_ms, open_detect_ms)（写在窗口末尾或之后的块再取 ≥ close_detect_ms），即 `speech_latency_lit.py` 的在线协议；同时刻按流 id、再按块序（与 `speech_latency_lit` 单通道排程的排序相同）。代码在 `talker/shared_channel.py`，训练行、预检、推理三处共用。
- **状态**：共享通道的积压 b_n，Δ = 80 ms（与 PL-EM 相同）。
- **转移**：b_{n+1} = round_80ms(max(0, a_n + b_n + δ_c − a_{n+1}))，截到上限。
- **延迟项**：ℓ_n(b, c) = W_n·b + δ_c·(W_n + 1)/2（秒·词）：块内 W_n 个词按时长均分，每个词说完的时刻相对 a_n 的延迟之和，即单通道语音 LAAL（`speech_laal_online_1ch`，FIFO）分子中随选择变化的部分；形式与 PL-EM 相同，只是 b 换成共享通道的积压。
- **势函数**：ψ_n(b, c) = τ·lp̄ + β·f − λ·ℓ_n(b, c)，τ、β、f 与 PL-EM 相同。
- **λ**：同样的对偶二分（λ ∈ [0, 50]，40 次），使后验下按词平均的延迟等于预算；**预算 = A′ 规则路径在同一格上的单通道语料延迟 − 1.0 s**。
- **条件后验**：q(c | b_n = A′ 路径上的实际状态)，与 PL-EM 相同。

**两个落地细节**（登记前只看了 A′ 参考路径的积压分布这一项数据事实，没有看任何实验结果）：
1. **积压上限 32 s**（PL-EM 是 8 s）。A′ 路径的共享积压在 train 上：均值 3.22 s，p90 7.92 s，p99 16.78 s，最大 31.40 s，9.8% 的块超过 8 s（dev5：17.1%）。8 s 上限会把这些状态合并，格就看不到大积压下的差别。
2. **取整改为最近点**（PL-EM 是向上取整）。一个会议的链平均约 1272 块，向上取整每步多出平均 40 ms，在连续排队时会累积：A′ 路径上量化状态相对真实积压的平均漂移，向上取整为 +0.28 s（train）/ +0.49 s（dev5），最近点为 +0.09 s / +0.17 s。
   - 相应地，条件后验的"实际状态"取 A′ 路径的**精确**共享积压（由真实时长逐块算出）再取最近格点，不用会漂移的链上状态。
   - 默认的每条流模式（向上取整、8 s）不变。

### 2.2 训练行

与 PL-EM 相同（历史 = A′ 规则路径；CE 权重 = 条件后验；Q_MIN；语速软标签由 q 得到；KL 锚原格式），只改两处：
- 权重来自会议格；
- **每块的积压 token（含历史块）= A′ 路径上该块的共享通道积压**（真实时长、20 ms 一个码），按 0.5 s 分档。

实现：`talker_rows.py --mode plem --scope meeting`（默认 `--scope stream` 逐位不变）。精确性检验（P2 同款）：一个小会议上的前向-后向、Viterbi、条件后验与暴力枚举一致（容差 1e-9）；1 ms 格点下链上状态与共享通道的真实积压逐块相等（`test_playback_lattice.py`）。

### 2.3 推理一致性

- **现状**：`si_infer --backlog-token` 的积压按每条流算（该流之前的块，write_ms 时间线），不是共享通道。
- **新增**：`--backlog-scope meeting --backlog-win <dev 窗口>`（默认 `stream` 不变）：
  - 每个分片放整个会议（按块数做最长处理时间优先分配到 8 个分片）；
  - 每个会议的块按 FIFO 顺序逐块生成；
  - 积压 token = 共享通道上由已合成的时长（码数 × 20 ms）算出的积压，分档同上；
  - 各会议之间并行（同一轮里每个会议各推进一块）；
  - 每块的种子公式与默认模式相同（`seed + 1000·条目序号 + 块序号`）。
  - 单元测试：用假 talker 跑完整的会议循环，每块的积压 token 与按 `speech_latency_lit` 单通道排程独立算出的积压一致。
- **c1 语速规则保持每条流**（A′ 的规则不变）。
  - 理由：训练候选是在每条流 c1 语速下采样的；共享积压的适应由 talker 自己完成。
  - 这项选择由执行者在登记前做出，已记入 LOG。
- **推理标志**：与 PL-EM c1+b1+s2 相同，再加 `--backlog-scope meeting --backlog-win eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl`。

### 2.4 预检（CPU 选择 + 渲染；dev 5 个会议，v1 的预检候选）

- **三条路径**，候选相同，严格配对：
  - A′ 规则路径（采样本身）；
  - 每条流的 PL-EM 格 Viterbi 路径（v1 的 `sel_lattice.json` 和渲染）；
  - 会议格 Viterbi 路径（新）。
- 新路径的处理：渲染后走现有的 ASR 和打分流程。
- 三条路径的延迟都用当前的 `speech_latency_lit --win <dev5 窗口>` 重算（FIFO 单通道）。
- **GO** ⇔ 同时满足：
  - 单通道语音 LAAL（会议格）≤ 单通道语音 LAAL（每条流格）− 0.5 s；
  - 官方语音 ASR-BLEU（会议格）≥ A′ 路径 − 0.2。
- **无论 GO 与否都训练**，如实报告。另报每条流 LAAL、EndOffset、归一化按段语音分、WER，以及对 A′ 路径的单种子门槛。

### 2.5 主指标与解读规则

- **主延迟指标**：单通道语音 LAAL（FIFO）。另报每条流语音 LAAL、EndOffset（每条流和单通道）。
- **门槛**：对 A′ 只看语音项（文本按构造与 A′ 完全相同，这一点在报告里写明）。另外与 PL-EM c1+b1+s2 配对比较。
- **解读规则**（看结果之前定下）："单通道建模有效"需同时满足：
  - 对 A′ 过门槛（语音）；
  - 单通道语音 LAAL（两种子均值）比 PL-EM c1+b1+s2 低 ≥ 0.5 s，且两个种子同号。
  - 若对 PL-EM 的配对语音分下降超过 0.2 或配对 P(差 ≤ 0) ≥ 0.95，如实写明延迟换来了质量损失。

## 3. GPU 安排（用户指令）

- **训练**：只用 0–3 号卡。先训实验一（行便宜），实验二的行（会议格，CPU）同时在后台构建，准备好后接着训实验二。启动脚本 `scripts/start_talker_crf_v2.sh`。
- **评测**：只用 4–7 号卡，一个编排器 `scripts/talker_crf_v2_orch.py` 管实验二预检和两个 dev 评测。
  - 启动时排除 0–3 号卡；
  - 每卡 ≤ 135 GiB；
  - vLLM 每对卡每 60 s 最多启动一个；
  - CTC 打分服务用自己的端口 18791–18794（v1 用 18771–18774）。
- **占卡任务**：用卡之前按 PID 停掉该卡的占卡任务（取自 nvidia-smi / ps 的字段，不按模式匹配）；空闲的卡由守护脚本 `keep_cards_busy.sh` 补回占卡任务。全部结束后确认 8 张卡都在跑占卡任务。

## 4. 评测协议（与 v1 §4 相同）

- **数据**：全 dev，A′ 的文本，两个 TTS 种子。
- **链路**：`speechonly_part → asr_transcribe（每片 2 个子分片）→ asr_merge → asr_hyp → score_run --asr`。
- **延迟**：`speech_latency_lit.py --win <c2g06 dev 窗口>`。
- **差距**：`speech_gap_decomp.py --win`，官方口径和归一化按段口径都报。
- **质量**：`talker_quality.py` 报 UTMOS / WavLM 说话人相似度；另报 talker WER 和强制截断块数。
- **门槛**（`gate_2seed.py`，规则不变）：点估计 ≥ 参考 − 0.2，且按会议配对 bootstrap P(差 ≤ 0) < 0.95；项目为文本 all、文本 ov、语音（两种子合并）。talker 只换语音，文本项按构造相同，判定看语音项。
- **对 PL-EM 的配对比较**：同样用 `gate_2seed`，以 PL-EM c1+b1+s2 两种子为参考，报语音分的差和 P；其余列报两种子均值之差，并分别列出两个种子。
- **汇总表**：`talker/eval_summary_v2.py` → `eval_outputs/mist_w3po_v1/talker_crf/eval/summary_v3.{md,json}`。
- **报告**：结果写 `talker_crf_results_v2.zh-CN.md`；同时更新 `HANDOFF_20261111.zh-CN.md` §0 和 `LOG.zh-CN.md`。
