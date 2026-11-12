# PMC：VE1 + 早一读的写出窗 + 策略边缘化的内容学习（预登记，2026-11-12 布拉格约 08:20；在任何 PMC 数据、预检、训练和结果之前提交）

- **提出者**：协调者（代用户批准"最后一个新方法"），方法设计执行者 v3 细化。
- **照实说明**：这个方法是在**看过 T1–T5、VE1（速查与探索性 dev 全集）、RC5 的结果之后**设计的。所以它不是盲设计，结论只对这一次登记的设定成立。
  - 已知：VE1 在 dev 全集上文本与 A′ 分不开（28.07 / 23.70），VE1rl + PL-EM 语音 23.64、每条流语音 LAAL 3.756 s；VE1 的 dev5 速查是 NOT GO（ov 23.28 < 23.92）。
  - 已知：写得更早的 T4 / T5 / RC5 文本掉约 1 分；RC5 说明碎片不是主因。现有的解释是"证据还没听够就提交了内容"。
- **要回答的问题**：在 VE1 的配方上，让每个 merge3 块可以早一读（1.0 s）写出，用"证据标签路径"的延迟作为后验的期望延迟预算，并且在策略实际会写出的状态上训练内容，能否在文本和语音 BLEU 不降的前提下，把延迟降到 VE1 以下。
- **范围**：只有一种设置，不扫参，不重试。只用 AMI train 和 dev。RealSI、ACL 60/60、AliMeeting、AMI test 不碰；不用 COMET / MetricX。
- **不使用的东西**：不用 QABC（没有 ABC 稳定前缀后验、没有 SDR、没有随队列变化的阈值或等待标记偏置）。证据时刻只作为标签和窗口的边界使用。不引用保密的 ICASSP 审稿论文。

## 0. 新意评估（文献核查，2026-11-12 布拉格 08:00–08:10，约 40 次检索，读了 ar5iv / arXiv 原文；不保证穷尽）

**不新的部分。** (1) 在每个候选写出状态上都训内容，是 HMT（Zhang & Feng, ICLR 2023, arXiv 2303.00257）的 state loss：K 个状态均匀加权；WMT15 De→En 上去掉它 BLEU 从 29.29 降到 27.74（−1.55）。(2) 用后验 p(z | x, y) 加权每个状态的内容交叉熵，就是边缘似然对发射项的梯度（EM 恒等式）：HMT 的 L_hmm、基于 CTC / transducer 的同传（NAST，CAAT，MonoAttn-Transducer）都隐式做了；我们自己的 v2′ / latent-write 的 M 步也早已是这样（`serial_v2p_train.set_coefs`）。所以**"后验加权的内容学习"本身对我们和对文献都不新**。(3) 不看内容的策略分布已经有人用，但只用在期望延迟（HMT 的 L_latency，MILk / MMA，NAST）和期望上下文（MILk / MMA）上。(4) "在策略会去的地方训内容"已有点质量或采样的版本：StreamSpeech 用模型自己的 CTC 对齐，Arthur et al. (EACL 2021) 对读写动作做 scheduled sampling，BS-SiMT 交替训练，Seed LiveInterpret 2.0 用 RL。
**据我们检索、没有找到先例的部分。** 用**策略自身、不看内容的写出时刻分布**去加权每个候选状态的内容交叉熵，并且这个期望在格上用前向-后向**精确**求出（不是采样、不是点质量），分布带一个**由对偶求解的延迟乘子**（posterior regularization 式的期望延迟预算；SiMT 里没有检索到用对偶乘子的 PR 延迟约束，ASR 里的 delay-penalized transducer 和 minimum-latency training 用的是固定 λ）。再加上应用场景：多说话人集合状态 CRF、用强制对齐得到的证据时刻窗口、DART 逾期状态、基于 LLM 的 S2ST Thinker。
**怎么表述。** 把它说成 "HMT state loss 的 on-policy（纠正时机上的 exposure bias）版本"：HMT 在所有状态上均匀训内容；EM 只在参考译文已经好预测的状态上训；PMC 在延迟受预算约束的策略**实际会写出**的状态上训。**局限要写明**：本登记只有一种设置，不跑"均匀权重（HMT 式）"和"只有 EM"两个消融，所以即使有效，也只能说明"这一整套"有效，不能把收益归给权重的选择。最接近的参照是 VE1（只有 EM、窗口 {e, e+1}、没有预算），但它在窗口和预算上也不同。

## 1. 设定（唯一）

**名称**：PMC-1。`train_crf_sft --objective latent_write --lw-mode pmc`（新选择，只做加法）。

### 1.1 数据与标签（与 VE1 完全相同）
- `eval_outputs/mist_w3po_v1/crf/sched_t5_merge3_w10_v1`：T5 的证据时刻规则（OmniAlign 对齐，support + 200 ms，NIR > 0.3 的窗口去掉），merge3 粒度（每块约 6 词），每 1.0 s 一读再加窗口终点。标签读记为 e_i。
- 不新建标签。

### 1.2 写出窗：比 VE1 早一读
- 每块的写出读 z_i ∈ [l_i, h_i]，**l_i = max(0, e_i − 1)**，h_i = min(R − 1, e_i + 1)。VE1（v2p）是 [e_i, e_i + 1]。
- 格的其余部分与 v2′ / latent-write 相同：状态是每个说话人的前缀计数，c_k ∈ [#{h_i ≤ r}, #{l_i ≤ r}]；规范顺序仍按标签 e（`V.kap(rec, e, ·)`，不随 l 改变）；读内顺序隐变量（canon / newfirst）的定义不变（"逾期" = e_i ≤ r − 1，"新到" = e_i = r）；历史是已写集合按 kap 顺序。
- **为什么选这个窗口**（数据依据）：
  1. VE1 训练后 500 步（2501–3000，按词加权）后验的期望延迟是 0.838 s·词 / 词，而 e 标签路径只有 0.602（`ve1_20261111T214454Z/metrics.jsonl`；前 500 步是 0.855 / 0.600）；dev5 上 VE1 的文本 LAAL 3.26 s，而 merge3 证据标签路径是 2.33 s。也就是说，VE1 实际写得比它的标签晚约 0.9 s，延迟的余量在"往标签靠"，不在更细的块。
  2. 更细的块（T4 / T5 的 ≤ 4 词块）把延迟降到 2.0–2.7 s，但文本掉约 1 分；RC5 说明只合块不救。所以保留 merge3 粒度，只放宽时机。
  3. "T5 标签用 merge3 粒度"就是 VE1 的数据本身；T5 的窗口 [e, max(u, e)] 是往**晚**放宽，不是往早。所以往早一读是唯一能让写出早于 VE1 标签、又不换粒度的选择。只放一读（1.0 s），不放两读：放得更早就接近 T5 的提前量，质量风险已知。

### 1.3 后验 q_λ（含内容）与延迟预算
- 边势与 latent-write 相同：ψ_{r,o}(A′ → A) = log p_θ(W 按顺序 o ⊕ e_r | x ≤ t_r, κ(A′))；强制读不打分。
- q_λ(z) ∝ exp(Σ_r [τ ψ_{r,o_r} − λ L_r])，τ = 0.5（v2′ 登记的退火），L_r = Σ_{i∈W} n_words_i (t_r − src_end_i) / 1000。
- **预算**：每个优化步，在本步的全部窗口上求 λ* ≥ 0，使 Σ_w E_{q_λ,w}[Lat] = Σ_w Lat_w(e 路径)；不绑定时 λ* = 0。e 路径就是 VE1 的证据标签路径（每块都在 e_i 写出）。求解器与 T2 相同（从 1 起加倍定界，上限 1e3 并标记，50 次二分）。
- 含义：块可以在 [e − 1, e + 1] 里随内容难易前后移动，但**期望上不晚于证据标签**。VE1 没有这个预算（λ ≡ 0），这是它比标签晚 0.34 s·词 / 词的原因之一。

### 1.4 策略分布 q^pol_λ（不看内容）与 PMC 内容项
- q^pol_λ：同一张格、同一个 τ、同一个 λ*、同样的读内顺序隐变量，**只把边势里的内容 token 对数似然去掉**，只留写 / 等的决定：
  - 一条边在读 r 依次写出 k 个块（块边界位置 b_0 = 0 < b_1 < … < b_k），φ = Σ_{j<k} log(1 − p_end(b_j)) + log p_end(b_k)，p_end 是该位置的结束 token（非最终读是 `<wait>`，最终读是 `<eos>`）的概率，都取自打分前向里已经算出的 `ec`，**不增加前向**；
  - 强制读 φ := 0（路径常数，与 ψ 相同处理）。
  - q^pol_λ(z) ∝ exp(Σ_r [τ φ_{r,o_r} − λ* L_r])，前向-后向精确求边与顺序的边缘；不回传梯度。
- **M 步** = VE1 的 M 步原样（全部 token 按 q_λ 加权，`set_coefs`）**加上** PMC 内容项：每个内容 token 再加上 q^pol_λ 下"经过该 token 的边"的总质量，权重 1（与 HMT 的 λ_state = 1 相同）：
  - c_tok[t] = h_t (Σ_{b>t} cend[b] + Σ_{b>t} cend_pol[b]) / D；c_end 不变（只按 q_λ）。
  - 第 2 遍的行就是格里全部请求的行（与 VE1 相同），PMC 只改系数，不加行。
- 直观：q_λ 决定**何时写**（策略的监督信号）；内容既在参考好预测的状态上训（EM），也在**策略会写的状态上**训（PMC）。训练初期策略还平，q^pol 接近窗口内均匀，相当于 HMT 的 state loss；策略变尖之后，q^pol 集中到策略实际写的地方。

### 1.5 其余与 VE1 完全相同
- `run_crf_v2p.sh` 的全部参数：`--tau 0.5 --final-read-full-post 1`、3000 步、每步约 64 个读行、学习率 1e-4（warm-up 50，余弦到 1e-5）、AdamW (0.9, 0.95)、梯度裁剪 1.0、扰动 p 0.2、v8 merged 初始化、新 LoRA r32 / α64 + 投影器 + 259 个新 token 行、`--feat-per-row 1 --fast-moe-bwd 1 --prefetch 1 --empty-cache 0 --tok-budget 32000 --mem-cap-gib 120`。
- 训练种子：默认 20261109（与 VE1 相同的计划）。
- **DART**：参数与 VE1 相同（扰动窗口比例 0.5、p_item 0.3、推迟 k ∈ {1, 2, 3} 读、DART 抽样种子 20261111）。清单绑定 lattice 模式，所以用同一命令只把 `--lw-mode` 换成 `pmc`，重建到新目录 `crf/pmc_dart_merge3_w10_v1`（模式只影响清单里"增广状态是否在格内"的统计；训练器加载时逐条核对抽样）。
- 1.0 s 读，4 卡（4–7 号卡，slotB），只评 step_03000。

## 2. CPU 预检（训练之前；任何一项不过就不训练，报告协调者）

- **P1 精确性**（单元测试，CPU）：
  - 在合成多说话人记录（2–3 个说话人，R ≤ 6）和真实的 VE1 训练窗口上，pmc 格的 log Z、每条序列的权重（cend）、E_q[Lat]，与**不经过格**的穷举（每块 z_i ∈ [l_i, h_i]、沿每个说话人单调 × 每读的顺序）一致，误差 ≤ 1e-9；λ ∈ {0, 0.3, 2}，τ ∈ {0.5, 1}。
  - q^pol 的每条序列权重（cend_pol）同样与穷举一致（穷举里 φ 直接从 dummy 打分的 `ec` 独立计算），误差 ≤ 1e-9。
  - PMC 系数：c_tok = h (suf(cend) + suf(cend_pol)) / D，c_end = cend / D，与按定义逐 token 计算的一致。
- **P2 默认路径逐位不变**：
  - 模式 t2 / t5 / v2p / label 下，新代码的计划、后验（cend、log Z、q_label、E_q[Lat]）和系数，与改动前的代码（git 里的上一版，作为对照模块加载）在 VE1 数据的 200 个真实窗口上**逐位相同**（`np.array_equal` / `==`）；
  - 已有测试 `test_latent_write_crf`、`test_train_crf_sft`、`test_dart_aug` 全过。
- **P3 标签路径的延迟低于 VE1**：
  - (a) VE1 训练后 500 步（2501–3000）的 metrics：e 路径延迟 `lat_label`（= PMC 的预算）的均值，比 VE1 后验期望延迟 `Eq_lat_T2` 的均值低 ≥ 0.1 s·词 / 词（预算确实比 VE1 实际训练出来的时机早，λ 会绑定）；
    照实说明：这一项用的是 VE1 已有的日志，登记时已经算过（0.602 vs 0.838，见 §1.2），它只是合理性条件，不检验 PMC 本身；
  - (b) 在 VE1 训练计划的前 200 步窗口上，pmc 格最早路径（全部在 l 写出）的每词延迟 < e 路径的每词延迟（窗口确实往早放宽了）；并报告两者以及 v2p 格最晚路径的数值。
- **P4 规模**：在 VE1 的训练计划（3000 步）上用 `--sizes` 统计 pmc 与 v2p 的每步第 1 / 2 遍行数与 token 数、每窗口最大状态数。
  - GO：每步第 2 遍 token 的 p95 ≤ v2p 的 2.5 倍，且 dummy 打分下每步计划 + 后验（含两次前向-后向和 λ 求解）的 CPU 时间 p95 ≤ 2 s。
  - 用实测的倍数估计训练时长（VE1 是 3.8 s/步）。
- **P5 训练器**：`--lw-mode pmc` 参数解析正确；DART 清单（模式 pmc）与计划一致；单卡 CPU 小模型跑一步 `latent_write_train.run_step` → `grad_pass`：损失有限，λ 已定界，Σ 系数 × D = Σ h + R + PMC 项的总质量（逐行核对）。

## 3. 速查与门槛

### 3.1 训后评测链（与 VE1-s2 相同的形状）
1. dev 全集解码：c2g06 窗口，1.0 s 读，8 分片，`--cap135`，只用 4–7 号卡（`--exclude 0 1 2 3`）。
2. dev5 截取（`t_dev5_cut.sh`，只用 CPU）→ 速查 GO 线（3.2）。
3. 读级块合并（`merge_read_segments.py`，不改）。
4. PL-EM talker，两个 TTS 种子，相隔 60 s 启动；门槛参考：A′、A′rl + PL-EM、VE1rl + PL-EM（最后一个只作描述）。
5. 听感指标：UTMOS / SIM（`talker_quality`，一张非训练卡）、talker WER（`speech_gap_decomp`，CPU）、句中 ≥ 0.5 s 停顿每分钟次数与单词块占比（`talker_block_pauses.py`，CPU）。
- **速查 NOT GO 时也跑完 3–5**：理由是这是用户选系统前最后一个方法，卡在这段时间里本来只给占卡程序，完整的数字对用户的选择有用。但**正式判定**仍按 3.4：速查不 GO 就不能说"PMC 有效"。这一条是现在登记的，不是事后偏离。

### 3.2 dev5 速查 GO 线（全部满足才 GO）
1. 文本 all ≥ 28.16 且 ov ≥ 23.92（A′ − 0.2）；
2. 文本 LAAL < 3.26 s（VE1 的 dev5）；
3. 最终读写出的词占比 ≤ 49.7%。
- 照实说明：VE1 自己在 dev5 上 ov 没过第 1 条（23.28）。所以这条线对 PMC 的要求比"追平 VE1"更高。
- 附带报告：dev5 语音代理（`joint_speech_proxy`，读级块）、停顿分类、送到 talker 的块大小、重复率。

### 3.3 dev 全集门槛（`gate_2seed`，两种子合并，按会议配对 bootstrap）
1. **语音**（PMCrl + PL-EM）对 **A′rl + PL-EM**（`dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem` 与 `_tts2`）：差 ≥ −0.2 且 P(差 ≤ 0) < 0.95；
2. **文本 all 和 ov** 对 **A′**：两项都差 ≥ −0.2 且 P(差 ≤ 0) < 0.95；
3. **每条流语音 LAAL**（`speech_laal_online` all，两种子平均）< **3.756 s**（VE1rl + PL-EM）。

### 3.4 判读
- **"PMC 有效"** = 3.2 全部 GO 且 3.3 三条全部满足。
- 只满足 3.3、不满足 3.2：写"全集达标，但登记的速查没过"，不说有效。
- 3.3 的第 3 条不满足：写"没有降低延迟"。第 1 或第 2 条不满足：写"质量没有守住"，并写明是哪一项。
- 不调任何参数，不换种子重试，不扫 λ 或窗口。

### 3.5 只作描述（不是门槛）
- 对 VE1rl + PL-EM 的配对差（语音、文本 all / ov）与 P；EndOffset、单通道 LAAL、排队、最后一块；文本 LAAL；≥ 10 s 拆分（`early_commit_diag --light`、`laal_breakdown`）。
- 听感：UTMOS（对 VE1rl + PL-EM 和 A′rl + PL-EM 的同流配对差，两种子合并，按会议 bootstrap）、SIM、talker WER、句中停顿、单词块占比。
- 训练动态：λ 绑定的步数比例、E_q[Lat] / 词、预算 / 词、q_λ 与 q^pol 在 e − 1 上的期望质量（"提前写出"的块数）、PMC 项占内容权重的比例。

## 4. 算力
- 训练：4 卡（4–7 号卡），时长按 P4 的实测倍数估计（VE1 3.8 s/步 × 倍数 × 3000 步），记在结果节。
- 评测：dev 全集解码约 30 分钟（4 卡），PL-EM 语音链约 50–60 分钟，听感作业约 10 分钟（一张卡）+ CPU。

## 5. 代码与启动（登记之后做，只做加法；脚本都只用 `bash -n` 和 DRY_RUN 检查，不试跑）
- `crf/latent_write_crf.py`：模式 `pmc`（下界 l、预算 = e 路径、策略边势、q^pol 的序列权重）；其他模式逐位不变。
- `crf/latent_write_train.py`：pmc 时算 q^pol，并调用新的系数函数；其他模式不变。
- `crf/train_crf_sft.py`、`crf/dart_aug.py`：`--lw-mode` 加 `pmc` 选择。
- 测试：新增 `crf/test_pmc.py`（P1、P2、P5）；预检脚本 `crf/pmc_precheck.py`（P2–P4，写 `crf/pmc_precheck_v1.json`）。
- 启动：`scripts/start_pmc.sh`（等 slotB 上只剩占卡进程，noclobber 占槽，按 `takeover_train.sh` 按 PID 停占卡程序并立即启动，核对 4 个 rank）；训后：`scripts/pmc_after_train.sh`（照 `ve1s2_after_train.sh` / `ve1x_dev_eval.sh` 的形状）。

## 结果

（登记之后补。）

### 代码与 CPU 预检：全部通过（2026-11-12 布拉格 08:27）

**代码**（只做加法；被改的模块都是写新文件后 `command mv -f` 替换）：
- `crf/latent_write_crf.py`：模式 `pmc`（`lower()`、`e_path_lat()`、预算、`assemble_pol` / `pol_parts`（策略边势 φ）、`exp_early`）；
- `crf/latent_write_train.py`：`set_coefs_pmc`；pmc 时算 q^pol，并记录 `Eq_lat_pol`、`lat_e`、`exp_early_q`、`exp_early_pol`、`n_chunks`；
- `crf/train_crf_sft.py`、`crf/dart_aug.py`：`--lw-mode` 加 `pmc`；
- 新增：`crf/test_pmc.py`、`crf/pmc_precheck.py`、`scripts/start_pmc.sh`、`scripts/pmc_after_train.sh`、`scripts/pmc_dev5_go.py`、`scripts/pmc_listen_compare.py`。
- **注意**：pmc 模式下 metrics 里的 `q_label` / `lat_label` 指格的 0 号状态路径，也就是**最早路径**（每块都在 l = e − 1 写出），不是证据标签路径；证据标签路径的延迟是 `lat_e`（= 预算 = `Eq_lat_T1_target`）。

**数据**：DART 清单 `crf/pmc_dart_merge3_w10_v1`（`logs/mist_w3po_v1/pmc/build_pmc_manifest.sh`，CPU 约 5 s）：15,585 条，抽样和行与 VE1 的清单逐条相同，模式 pmc。

**预检** `eval_outputs/mist_w3po_v1/crf/pmc_precheck_v1.json`（CPU 151 s）：

| 项 | 结果 | GO |
|---|---|---|
| P1 精确性 | 60 个合成记录（最多 4362 条路径）+ 30 个真实 VE1 窗口：q_λ 的 log Z、cend、E[Lat]、E[提前块数]，q^pol 的 cend、E[Lat]、E[提前块数]，与穷举的最大差 2.1e-13；预算 = e 路径的 Lat | 过 |
| P2 默认路径 | t2 / t5 / v2p / label 各 200 个真实 VE1 窗口：计划、格、target / lat_label、后验（log Z、q_label、E[Lat]、exp_nf、cend；λ 0 / 0.37）与改动前的模块（46cc50e4）逐位相同；回归 `test_latent_write_crf` 5/5、`test_train_crf_sft` 8/8、`test_dart_aug` 6/6 | 过 |
| P3 (a) | VE1 后 500 步：后验 0.8375 vs e 路径预算 0.6019 s·词 / 词，差 0.236 ≥ 0.1（登记时已知） | 过 |
| P3 (b) | VE1 计划前 200 步的 2052 个窗口（每词）：pmc 最早路径 −0.284 < e 路径 0.603（= v2p 最早路径）；v2p 最晚路径 1.194 | 过 |
| P4 规模 | 每步第 2 遍 token（hist + 写出部分）均值 2.02 倍、p95 2.34 倍（≤ 2.5）；第 1 / 2 遍行数 62.3 → 131.9、96.6 → 149.6；每窗口最大状态数 p95 8、最大 54；dummy 打分下每步 CPU p95 0.145 s（≤ 2 s） | 过 |
| P5 训练器 | `--lw-mode pmc` 解析为 latent_write / pmc / seed 20261109 / 3000 步 / 64 行；清单通过 `load_manifest` 核对；小模型一步：损失有限，prefetch 1 = 0 逐位相同，Σ c_tok × D = 2 Σ h，Σ c_end × D = Σ R，系数 = set_coefs + h·suf(cend_pol)/D（差 1.7e-18） | 过 |

- **P3 (b) 的一个提示**：最早路径的每词延迟是负的（−0.28 s），说明 e − 1 常常在块的源端说完之前，也就是"提前预测"的区域。预算把期望拉回 e 路径，但这正是质量风险所在，速查的文本线会检验它。
- **训练时长估计**：每行的音频和提示 token 远多于 hist + 写出部分，所以按行数比例估计（第 1 遍 ×2.12、第 2 遍 ×1.55，VE1 后 500 步 0.713 / 3.038 s）约 6.2 s/步，3000 步约 **5.2 小时**；只按 token 比例估计是 7.8 s/步、6.5 小时。登记时写的"4–6 小时"按实测改为 **5–6.5 小时**（4 卡，20–26 卡·时）。
- **两次未出结果的预检尝试**（预检脚本自身的 bug，不是数据问题；两次都在写出 json 之前就退出了，日志保留）：
  1. 用 runpy 跑时多进程无法 pickle `_sizes`（`logs/mist_w3po_v1/pmc/precheck_v1_attempt1_pickle_error.log`）；改为直接 `python -m`，并在模块里打 CPU 补丁；
  2. `T.step_totals` 需要 `set_reach`，这里只用 D，改成直接算（`precheck_v1_attempt2_step_totals_error.log`）。
  - 两次中 P1 / P2 / P5 的测试都已通过，判据没有改。
