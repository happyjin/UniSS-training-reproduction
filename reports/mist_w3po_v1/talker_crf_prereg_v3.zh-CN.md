# talker 实验 E2′ 的预登记：单通道 PL-EM，修正预算（2026-11-11）

**登记时间**：2026-11-11 约 07:45 UTC。在 E2′ 的任何数字（训练行的预算和 λ、预检、训练、评测）出来之前提交并推送（`git push private HEAD:main`）。
**用户批准**：约 07:30 UTC，作为推荐的修正方案批准。下面每个数值都是**唯一设置**，不扫超参。

前置文件：`talker_crf_prereg_v2.zh-CN.md`（下称 v2；实验二 = E2）、`talker_crf_results_v2.zh-CN.md` §1–§2、`talker_crf_results_v1.zh-CN.md`（PL-EM）。

## 0. 这是什么，不是什么

- **这是同一方法（单通道 PL-EM，会议格）的第二个登记设置**，在看到 E2 的预检结果（NOT GO）之后做出。E2 的结果照原样保留、照原样报告，不被这份登记替代。
- **目的是改正一个设计错误**：E2 的预算（A′ 规则路径的单通道代理延迟 − 1.0 s）与 E2 自己登记的 GO 线不匹配。GO 线要求会议格路径的单通道 LAAL 比**每条流 PL-EM 格的路径**低 ≥ 0.5 s，但预算是相对 A′ 定的。在 dev5 的会议格代理延迟上，预算 3.82 s 比每条流格路径已经达到的 2.61 s 还宽松 1.2 s。也就是说，E2 的格被要求比 GO 线的比较对象更保守，λ 只有 0.004（dev5）/ 0.041（train）。
- **这不是朝结果调参**：
  - 只换预算的参照路径，从 A′ 规则路径换成 GO 线里的比较对象（每条流 PL-EM 格路径）；
  - 降幅取 GO 线本身的 0.5 s；
  - 不试第二个值；其余设置全部与 E2 相同。
- **登记时已知的数据**：E2 的预检数字和训练行统计（results_v2 §1–§2）。E1 / E2 的 dev 评测在登记时还没有结束，执行者没有看到它们的任何 dev 数字。

## 1. 设定（与 v2 §2 的唯一差别是预算）

**预算**：L_ref − 0.5 s。

- **L_ref**：每条流 PL-EM 格的 Viterbi 路径，在**会议格的单通道代理延迟**上的值，即 Σ_n ℓ_n(b_n, c_n) / Σ_n W_n。
  - 链与 v2 §2.1 相同：每个会议一条 FIFO 链，共享积压，Δ = 80 ms，上限 32 s，最近点取整。
  - ℓ_n(b, c) = W_n·b + δ_c·(W_n + 1)/2。
  - b_n 由这条路径自己的选择沿链递推。
- **每条流 PL-EM 格的路径**：在同一批数据上建每条流格（v1 方案一，`talker_rows.lattice`），预算 = A′ 规则路径的每条流代理延迟 − 0.3 s，λ 在这批数据上二分求得；每条流取 Viterbi 路径。
  - **训练行**（train 15 个会议，与 PL-EM、E1、E2 相同的 19077 块）：每条流格在 train 上的 λ 就是 PL-EM 训练时的 λ（0.497）。
  - **dev5 预检**：v1 预检的 `precheck_dev5/sel_lattice.json`（λ 0.80）。它在会议格代理延迟上的值已在 E2 预检里报告：2.6064 s，所以 dev5 预检的预算 = 2.1064 s。
- **λ**：与 v2 相同的对偶二分（λ ∈ [0, 50]，40 次），使后验下按词平均的代理延迟等于预算。
  - 如果预算在 λ = 50 时仍达不到，按 `solve_lambda` 的既有规则取 λ = 50，如实报告，不改预算。
- **其余与 E2 完全相同**：
  - 格：链、状态、转移、ℓ、ψ = τ·lp̄ + β·f − λ·ℓ（τ 0.5，β 5，f = −min(CTC NLL 每字符, 10)）；
  - 条件后验 q(c | b_n = A′ 路径精确共享积压的最近格点)；
  - 训练行：历史 = A′ 规则路径；积压 token = A′ 路径的共享积压分档；Q_MIN 0.01；语速软标签；KL 锚原格式；
  - 训练：`run_talker_crf.sh MODE=plem`，从 T1c 起训，talker 全参数，LR 3e-5（warmup 50，cosine 到 3e-6），1200 步，GBS 4，每组 ≤ 24000 token，KL 0.5，收尾 token CE 权重 3，语速损失 1.0，种子 20261110；
  - 推理：A′ 的文本，`--speed-policy queue --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 --speed-shift 2 --backlog-token --select-k 2 --backlog-scope meeting --backlog-win eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl`，CTC 打分服务；c1 语速规则保持每条流。

**实现**（只做加法，默认值逐位不变，单元测试含改动前取的哈希）：
- `talker_rows.py --mode plem --scope meeting --budget-ref stream_viterbi --drop-1ch 0.5`：先在同一批候选上算每条流格的 Viterbi 路径（`stream_viterbi_paths`），再把它作为 `meeting_lattice(ref_paths=)` 的参照。行文件 `rows/plemm2.pt`，variant `meeting_v3`。
- `precheck_plcrf.py select-meeting --budget-ref stream_viterbi --drop-1ch 0.5 --sel-stream precheck_dev5/sel_lattice.json`，输出目录 `talker_crf/precheck3_dev5`。
- 默认 `--budget-ref aprule --drop-1ch 1.0` 逐位不变：单元测试 `test_meeting_default_bit_identical`（改动前的哈希）、`test_meeting_budget_ref`、`test_select_meeting_budget_ref`。

## 2. 预检（dev5，v1 的预检候选，一个 talker 种子；CPU 选择 + 渲染 + 语音链）

- **三条路径**，候选相同，严格配对：
  - A′ 规则路径；
  - 每条流 PL-EM 格 Viterbi 路径（v1 的渲染）；
  - **E2′ 会议格 Viterbi 路径**（新渲染，`decode/plcrf3_dev5_meeting`）。
- 三条路径的延迟都用当前的 `speech_latency_lit --win <dev5 窗口>` 重算（FIFO 单通道）。另把 E2 的会议格路径（precheck2）作为第 4 行列出，只作描述。
- **GO 线与 E2 相同**，同时满足：
  - 单通道语音 LAAL（E2′）≤ 单通道语音 LAAL（每条流格）− 0.5 s；
  - 官方语音 ASR-BLEU（E2′）≥ A′ 路径 − 0.2。
- **无论 GO 与否都训练**，如实报告。另报每条流 LAAL、EndOffset（每条流 / 单通道）、归一化按段语音分、WER，以及对 A′ 路径的单种子门槛。

## 3. 训练与 dev 评测

- **训练**：0–3 号卡（A 槽），E2 训练结束、导出已存在之后开始。
  - 07:29 UTC 已建 `logs/mist_w3po_v1/slots/slotA.owner`。
  - 训练进程退出、四张卡上它的显存消失后，立即把该文件改名为 `slotA.released_<UTC>`（`command mv -f`）。
  - 启动先拿 `gpu_launch.lock`，再按 PID 停 0–3 号卡的占卡任务。
- **评测**：只用 4–7 号卡（B 槽评测池）。
  - 编排器 `scripts/talker_crf_v3_orch.py`，CTC 打分端口 18801–18804；
  - 每卡 ≤ 135 GiB，vLLM 每对卡每 60 s 最多启动一个；
  - 允许和 Thinker 执行者的标注 / 打分任务叠放，启动前看 nvidia-smi。
- **评测协议与 v2 §4 相同**：
  - 数据：全 dev，A′ 的文本，两个 TTS 种子（20261021 / 20261109），共享积压推理（`--backlog-scope meeting`）；
  - 链路：`speechonly_part → asr_transcribe → asr_merge → asr_hyp → score_run --asr`；
  - 延迟：`speech_latency_lit.py --win <c2g06 dev 窗口>`；
  - 差距：`speech_gap_decomp.py --win`；
  - 质量：`talker_quality.py`；
  - 门槛：`gate_2seed.py` 对 A′；
  - 配对：`gate_2seed.py` 对 PL-EM c1+b1+s2；对 E2 的配对只作描述。
- **主延迟指标**：单通道语音 LAAL（FIFO）。另报每条流语音 LAAL、EndOffset（每条流和单通道）。

## 4. 解读规则（看结果之前定下，与 v2 §2.5 相同）

- **"单通道建模（修正预算）有效"**：以下两条同时满足：
  - 对 A′ 过门槛（语音项：点估计 ≥ A′ − 0.2，且按会议配对 bootstrap P(差 ≤ 0) < 0.95）；
  - 单通道语音 LAAL（两种子均值）比 PL-EM c1+b1+s2 低 ≥ 0.5 s，且两个种子同号。
- **质量代价**：若对 PL-EM 的配对语音分下降超过 0.2，或配对 P(差 ≤ 0) ≥ 0.95，如实写明延迟是用质量换来的。
- **用户的硬约束**：ASR-BLEU 不得低于 A′。门槛不过，就不能作为主系统候选，无论延迟如何。
- **补充（不判定）**：
  - E2′ 对 E2：修正预算是否带来更低的单通道 LAAL，代价多少；
  - 每条流 LAAL、EndOffset、归一化按段语音分与差距、WER、UTMOS / SIM、强制截断块。

## 5. 报告

- 结果写 `talker_crf_results_v3.zh-CN.md`；
- 同时更新 `HANDOFF_20261111.zh-CN.md` §0 和 `LOG.zh-CN.md`（只追加）。
