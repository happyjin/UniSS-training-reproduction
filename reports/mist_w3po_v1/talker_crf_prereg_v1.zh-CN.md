# talker 两方案（PL-EM、对齐 CRF talker）的预登记：预检通过线、训练设置、评测协议（2026-11-10 23:05 UTC）

**登记时间**：2026-11-10 23:05 UTC，在任何预检、训练或评测结果出来之前提交。下面所有数值设置都是**唯一设置**，不扫超参。

**用户指令**（2026-11-10 22:35 UTC）："今晚就开始按照少消耗 token 方式，做这两个方案，从预检到评测做完所有；即使预检不过，还是仍然训练这两个方案，一直做到评测完。"
因此：**用户要求预检结果不影响是否训练**。预检照常登记、照常如实报告 GO / NOT GO，但两个方案无论结果如何都训练并评测。

方案定义见 `talker_crf_ideas.zh-CN.md` §2（PL-EM）、§3（对齐 CRF talker）、§8。下面只写落地时必须定下来的细节。

---

## 0. 共同部分

- **主系统 A′**：v2′ step_03000 finalpost + C2 gap 0.6 窗口 + talker T1c，c1（queue 语速策略）+ b1（k=2，CTC 选择）+ s2（语速 +2）。dev 输出 `decode/dev_crf_v2p_s3000_finalpost_s8_c2g06{,_tts2}`。
- **talker 起点和 KL 锚**：T1c（`eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss`）。T1c 的 Megatron checkpoint 不在本机，只有导出版本，所以从导出版本起训。
- **训练入口**：沿用 T1c / Plan 2 的 Megatron 写法（TP=PP=1，HuggingFaceModule，只存 talker）。新入口 `experiments/mist_w3po_v1/talker/pretrain_talker_crf.py` 复用 `si_model.SIModel._logits / _kl`、`pretrain_e2.keep_lr_groups_apart`、`export_e2.py`。
- **训练参数**（两方案相同）：talker 全参数；Thinker 不加载（冻结）；学习率 3e-5（warmup 50，cosine 到 3e-6）；**1200 步**；4 卡，每卡每步一组行（每组 ≤ 24000 个填充后 token），GBS 4；weight decay 0.01，clip 1.0，bf16，梯度重算；对冻结 T1c 的 KL 权重 0.5（只在语音码位置，与 T1c / selfmon 相同）；收尾 token 的 CE 权重 3（selfmon 默认）。
- **候选采样**（预检和训练数据共用一套代码）：`si_infer.py` 新增 `--cand-speeds 0,2 --cand-k 4 --dump-cands`（只做加法，默认不变）：
  - 每块在 A′ 的语速（block pace + 2 + queue 移位）和再快 2 档（即 A′ 口径的 +2 / +4）下各采 4 个，共 8 个候选；
  - 每个候选记录：语音码、收尾 token、是否被守卫强制截断、talker 累积对数概率、CTC 负对数似然（`si_ctc_scorer`，wav2vec2，每字符）、CTC 覆盖率（见 §2）；
  - **续写路径 = A′ 规则**：在 A′ 语速的前 2 个候选里取 CTC NLL 最小者（即 b1 k=2），所以采样沿 A′ 的选择路径前进。
- **数据范围**：dev 只用于预检和评测；训练只用 train。RealSI / ACL 60/60 / AliMeeting 不碰。

## 1. 方案一 PL-EM

### 1.1 格与势函数（落地细节）

- 一条通道 = 一个"窗口 × 说话人"流（si_infer 的一个 item）。块 n 的可放出时刻 a_n 用 `speech_latency_lit.py` 的在线协议：max(write_ms, open_detect_ms)，读到窗口末尾的块再取 ≥ close_detect_ms。
- 状态：积压 b_n ∈ {0, 80 ms, …, 8 s}（101 个状态），b_1 = 0；转移 b_{n+1} = ⌈max(0, a_n + b_n + δ_c − a_{n+1})⌉_80ms（截到 8 s）。
- 边势：ψ_n(b, c) = τ·lp̄_{n,c} + β·f_{n,c} − λ·ℓ_n(b, c)
  - lp̄：talker 每 token 平均对数概率（与 `--best-of` 相同的长度归一化），**τ = 0.5**；
  - f = −min(CTC NLL 每字符, 10)，**β = 5**；
  - ℓ_n(b, c) = W_n·b + δ_c·(W_n + 1)/2（秒·词）：块内 W_n 个词按时长均分，每个词"说完"的时刻相对 a_n 的延迟之和，即语音 LAAL 分子中随选择变化的部分；
  - 说话人相似度项 γ = 0（不算 WavLM）。
- **λ 的对偶二分**：在整个语料上取一个 λ ≥ 0，使后验下按词加权的平均延迟 E_q[Σℓ]/Σ W 等于预算 = A′ 规则路径上的同一量 − 0.3 s；λ = 0 已满足时取 0；λ ∈ [0, 50]，二分 40 次。
- 精确前向-后向（log 空间），Viterbi 取最优路径。

### 1.2 预检（dev，5 个会议：按字母序第 0、3、6、9、12 个 = ES2011a、IB4003、IB4011、IS1008c、TS3004b）

在 A′ 的文本和 talker 输入上采样（一个 talker 种子），对比的参考是**同一次采样里的 A′ 规则路径**（逐块 b1、k=2，与格路径共用同一批候选，严格配对）。实际 A′ 两个种子在这 5 个会议上的数也一并报告。

| 项 | 做法 | GO 条件 |
|---|---|---|
| P1 格选择上界 | Viterbi 路径的语音合成后走现有 ASR 和打分流程（`speechonly_part → asr_transcribe → asr_merge → asr_hyp → score_run`），延迟用 `speech_latency_lit.py --win <c2g06 windows>` | 每条流语音 LAAL（speech_laal_online）比参考降 ≥ 0.25 s 且官方语音 ASR-BLEU ≥ 参考 − 0.2；**或者** LAAL 不变差（差 ≤ +0.05 s）时归一化、按段的语音 ASR-BLEU ≥ 参考 + 0.4。按会议配对 bootstrap 报 P |
| P2 精确性 | 前向-后向的 log Z、边际与暴力枚举一致 | 容差 1e-9（单元测试） |
| P3 可蒸馏性 | 每块后验 q(c\|·) 相对该块候选均匀分布的 KL，按块平均 | ≥ 0.1 nat |

### 1.3 训练（train 子集）

- **语速并入 talker**：新格式（`--backlog-token`）在每块 S2T 头的语速 token（block pace + 2，不含 queue 移位）后面插入一个积压条件 token：积压按 0.5 s 分 8 档（0–0.5, …, ≥ 3.5 s），用 8 个本格式不用的保留 token（180381–180385、180397、180398、180405）；TTS 头的语速 token 由 talker 自己生成。
- 每块的行：历史 = A′ 规则路径（与采样时的条件一致）；该块 8 个候选各一行，**CE 权重 = q(c | b_n = 实际积压)**，即后验在实际历史状态上的条件边际（与 talker 输入里的历史一致；单块流时就是边际 q(n,c)）；权重 < 0.01 的候选不入行。
- 语速损失：TTS 头语速位置，对 35 个语速 token 做 softmax，软标签 q(v | b_n)，每块一次。
- KL 锚：T1c 在**原格式**序列（无积压 token，两个语速 token 都等于候选的语速）上的分布，只在语音码位置，按位置对齐。
- 损失 = Σ_行 w·CE(码 + 收尾×3) / Σ w + 1.0·L_v + 0.5·KL。

### 1.4 推理

- (a) 单采样 k=1：`--backlog-token --self-speed`，talker 先在 35 个语速 token 里取 argmax，再生成语音码；
- (b) A′ 设置：`--backlog-token`（语速仍按 c1+s2 规则给出）+ b1 k=2。

## 2. 方案二 对齐 CRF talker

### 2.1 覆盖率（预检和训练数据）

`si_ctc_scorer.py` 新增可选输出 `cover`（只在请求里带 `"cover": true` 时计算，默认响应不变）：同一段 CTC 输出做贪心解码，与"前缀 + 本块"的词序列求 LCS，本块的词被匹配上的比例。覆盖 = 1 表示本块每个词都在 CTC 贪心转写里按序出现。

### 2.2 预检（同上 5 个会议、同一次采样）

- 选择规则：只在 A′ 语速的 4 个候选里选；先要覆盖 = 1，其中取 CTC NLL 最小者；没有完全覆盖的取覆盖率最高、再按 NLL。
- **GO**：归一化、按段的语音 ASR-BLEU ≥ 参考（A′ 规则路径）+ 0.3，且语音 LAAL 不变差（差 ≤ +0.05 s）。

### 2.3 模型与训练

- 文本单元 = 本块 START_CONTENT…END_CONTENT 之间的文本 token（M 个）；语音帧 = 本块的语音码位置（T 个）。
- 对齐头：两层线性投影 q = W_q h_t、k = W_k h_m（1536 → 256），e(t, m) = q·k/16，h 为 talker 最后一层隐状态。
- 单调半马尔可夫对齐：每个单元占连续 d_m ≥ 1 帧（d ≤ 200），按序覆盖；段势 = Σ_{t∈段} e(t, m) + ω(d)，ω(d) = −(d − μ_m)²/(2σ_m²)，μ_m = T·len_m/Σlen（len = token 的字符数，至少 1），σ_m = max(2, μ_m/2)。
- Z_all：在第 T 帧结束于任意单元 m ≤ M 的对齐之和；Z_cover：结束于第 M 个单元的对齐之和。P_cover = Z_cover / Z_all（可微前向算法）。
- 行：每块取 §2.2 规则选出的候选（原格式，与 A′ 的 talker 输入相同）作为正例：CE（码 + 收尾×3）+ 0.5·KL(T1c) + 1.0·(−log P_cover) + 0.1·时长后验正则（E_q[d_m] 用自动求导求得，惩罚 mean_m((E_q[d_m] − μ_m)/μ_m)²）；同一块覆盖 < 1 的候选作负例，只加 1.0·(−log(1 − P_cover))，不加 CE。

### 2.4 推理

- (a) 默认策略：k=1，c1+s2，加 `--cover-gate`：每轮对所选候选用 talker + 对齐头算 P_cover，< 0.5 的块换种子重采，最多 2 次，取第一个 ≥ 0.5 的，否则取 P_cover 最大的。这是"所有单元被覆盖之前不发收尾 token"的拒绝采样实现（按块整体拒绝，而不是逐 token 屏蔽收尾），报告里如实写明。
- (b) A′ 设置：c1+b1+s2，k=2，只换 talker。

## 3. 训练数据（train 子集，已定）

- 窗口：`build_windows_zh2en --splits train --gap 0.6 --limit-meetings 16` → `eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_train16_v1`（按字母序前 16 个 train 会议，跳过 1 个，15 个会议、5613 个窗口）。
- 用 v2′ step_03000 finalpost 解码（与 A′ 相同的解码配置），再做 talker prep（与 A′ 相同），再按 §0 采候选。
- 行数过多时不做筛选，按 1200 步 × GBS 4 能用到多少用多少（RowGroups 每遍重新打乱）。

## 4. 评测（dev 全集，两个 TTS 种子，与 A′ 配对）

- 每个训练好的 talker 导出为 HF 格式（同 T1c），在 A′ 的文本上跑 `crf_pair_dev_orch.speech` 的链路，只换 `--model` 和 §1.4 / §2.4 的推理标志；种子与 A′ 相同（默认 / 20261109）。
- 报告：门槛 `gate_2seed.py`（文本按构造相同）；语音 LAAL 为主延迟指标，EndOffset 补充（`speech_latency_lit.py --win`）；文本到语音差距在官方口径和归一化按段口径下都报（`speech_gap_decomp.py --win`）；talker 词错率和强制截断块数；UTMOS / WavLM 说话人相似度若仓库有现成打分器则报。
- 门槛：点估计 ≥ A′ − 0.2，且按会议配对 bootstrap P(差 ≤ 0) < 0.95；文本 all、文本 ov、语音（两种子合并）。
