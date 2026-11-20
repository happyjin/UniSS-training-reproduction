# 鲁棒第十一波：干净锚定的 Laplace 后验合并（Fisher 加权 WiSE）预登记（2026-11-20 11:50 CET，写在任何 Fisher 计算、合并和评测之前）

## 起因
- 第九波锚定 Thinker 两条主判定都不通过，在第一秒干净的子集上也没有显著增益，所以这个思路关闭，不做 u 的改版。
- 用户 10:50 说："如果没效果就换思路，不要调参数"。
- 现有主系统 ERM-WiSE 的 α = 0.5 对所有参数一视同仁。ERM 的剩余问题是干净中→英 CoVoST 掉分（α = 1 时 −3.63），WiSE 用一个标量把它压住，同时也把带噪增益减半。

## 假设（数学模型）
- 底座 W0 是"干净语音"后验的众数，ERM 的 W_e 是"带噪训练混合"后验的众数。两者各用对角 Laplace 近似：
  - p_c(W) ∝ exp(−½ (W − W0)ᵀ F_c (W − W0))，F_c 为干净数据在 W0 处的经验 Fisher；
  - p_n(W) ∝ exp(−½ (W − W_e)ᵀ F_n (W − W_e))，F_n 为 ERM 训练混合在 W_e 处的经验 Fisher。
- 两个后验之积的众数有闭式解：W* = W0 + Λ ⊙ Δ，其中 Δ = W_e − W0，Λ = F_n / (F_c + F_n)，逐元素计算。
- WiSE α = 0.5 正是 F_c = F_n 时的特例。假设：干净数据敏感的元素（F_c 大）留在 W0，只对带噪敏感的元素采用 ERM 的更新。这样应能在保住干净的同时，保留比 0.5 更多的带噪增益。
- 参考：Fisher merging（Matena & Raffel, NeurIPS 2022）；Laplace 近似 / EWC（Kirkpatrick et al., 2017）。本波把它用作"干净锚定"的鲁棒合并，没有新训练。

## 方法（固定，不扫）
- **参数集合：** 与 ERM LoRA 相同的被适配矩阵（q/k/v/o，相同层）。Δ 用 ERM 的 (alpha/rank)·BA 逐矩阵计算，fp32。
- **Fisher 样本：** 都来自第五波的训练数据（开放数据），按 sha1("20261120|w11|fisher|c|i") 与 sha1("20261120|w11|fisher|n|i") 选，两方向各半。
  - F_c：N = 512 条干净样本（clean 环境，两方向各 256），在 W0 处计算。
  - F_n：N = 512 条 ERM 训练混合样本（8 个环境各 64），在 W_e 处计算（W_e = 底座 + ERM LoRA，α = 1）。
- **经验 Fisher：** 逐样本计算参考译文的 CE（与训练相同的 prompt 和标签掩码，按 token 平均）的梯度平方，对 N 求平均。逐样本反向，batch 1，eval 模式（无 dropout），fp32 累加。
- **合并：** Λ = F_n / (F_c + F_n)。F_c + F_n = 0 的元素取 0.5，即退回 WiSE。W* = W0 + Λ ⊙ Δ，在 fp32 中计算后转回 bf16，与 `wave5_m_merge.py` 相同；没有被适配的权重文件硬链接，拒绝覆盖。
- 合并后仍是一个普通模型：推理零开销，一个模型同时做两个方向。

## 评测与判定（全部写定）
- 链与打分同第五波 / 第十波：si_thinker_stream la2 w1 → T1c（4 个分片，种子 20261021）→ 只对语音做 ASR；用 3 次底座运行、噪声底、bootstrap，单元口径相同。
- **主判定：**
  1. laplace 相对底座的统一规则：所有 (a)/(b) × 干净/带噪单元，text BLEU 和 ASR-BLEU 的点估计 ≥ −0.3，且 CI 上界 ≥ 0；至少一个带噪单元 text 增益的 CI 下界 > 0，且超过噪声底。
  2. 相对 erm_wise05 的增量：至少一个带噪单元的 text BLEU 差 CI 下界 > 0，且点估计超过噪声底；并且没有任何 (a)/(b) 单元的点估计 < −0.3 且 CI 上界 < 0（text BLEU 和 ASR-BLEU 都查）。
  - 第 1 条过、第 2 条不过：记为"Laplace 合并对 WiSE 无增量"，只作分析。
- **辅助（不作判定）：**
  - LAAL；
  - Λ 的分布（逐层均值、分位数）；
  - 有效步长 ‖Λ⊙Δ‖ / ‖Δ‖；
  - Λ 与 |Δ| 的相关性。
- 若两条都过：跑与第五波相同的完整系统组合（+ V1 音色 + T1c），与 ERM-WiSE 的完整系统比较。

## 卡与时间
- 卡 0–1（第九波结束后空出）。Fisher 约 2 × 512 次前向加反向，约 30–40 分钟；合并在 CPU 上约 10 分钟；评测约 1.5 小时。

## 实现说明与实现修正（2026-11-20 11:47 CET 起，写在任何冒烟、Fisher、合并和评测数字之前；只追加）
- **Fisher 脚本：** `voicecond/wave11_fisher.py`（新文件，复用 `wave5_m_train.py` 的 `Sampler`、prompt 和标签掩码，不改原文件）。
  - **样本的具体取法（预登记只规定了 sha1 键）：** 第 i 条（0..511）是第五波 ERM 训练流中的一个样本。k = sha1("20261120|w11|fisher|c|i")（F_n 用 `|n|`），步 s = k mod 2300，环境内 slot m = (k // 2300) mod 8，slot j = 8e + m，然后取 `Sampler.sample(s, j)`。所以行号、噪声文件、BVR 和混合方式都与 ERM 当时看到的样本逐个相同。
    - F_c：i < 256 取 e = 0（zh→en clean），i ≥ 256 取 e = 4（en→zh clean）。
    - F_n：e = i // 64，按 `wave5_m_train.ENVS` 的顺序，8 个环境各 64 条，其中包括两个 clean 环境（它们本来就属于训练混合）。
    - 这个映射会产生少量重复的 (s, j)：F_c 有 5 对，F_n 有 3 对。按预登记 N = 512 不去重，照实记录。
  - **对整个权重矩阵求梯度：** 每个被适配的 Linear 旁边挂一个全零的 fp32 矩阵 D，前向为 out = base(x) [+ (alpha/r)·b a x] + (D x).to(bf16)。D = 0 时前向完全不变，而 ∂L/∂D 就是该点处的 ∂L/∂W。q/k/v/o × 48 层，共 9.06 亿个元素。
  - **W0：** LoRA 关掉（`enabled = False`），即严格的底座。
  - **W_e：** 底座 + ERM LoRA 不合并（a、b 为 fp32，α = 1），和训练时的前向完全相同，不用 bf16 合并后的副本。
  - 逐样本做前向和反向：batch 1，`thinker.eval()`，`use_cache=False`，不开梯度检查点（eval 模式下 HF 本来也不用），SDPA 和专家实现都用默认值（只做一阶反向，不需要第五波 Fishr 的 math SDPA 和 eager 专家）。
  - CE 按目标 token 取平均，与训练相同。梯度平方用 fp32 累加，最后除以 N。出现非有限梯度直接报错退出，不跳过样本。
  - 输出 `eval_outputs/mist_w3po_v1/wave11/fisher/F_{c,n}.pt`、`.json` 和 `.samples.jsonl`（每条样本记 i、s、j、id、CE），拒绝覆盖。
  - 显存分配器加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，只减少碎片，不改数值。
- **合并脚本：** `voicecond/wave11_laplace_merge.py`。
  - Δ = (alpha/rank)·B A 用 fp32 计算；Λ = F_n / (F_c + F_n)，F_c + F_n = 0 的元素取 0.5。
  - W* = W0 + Λ⊙Δ 在 fp32 中计算，再转回 bf16，与 `wave5_m_merge.py` 相同。没有被适配权重的文件硬链接自原版，拒绝覆盖。
  - 辅助统计写在 `merged_laplace.laplace_merge.json`（放在模型目录旁边，不放进模型目录）：
    - Λ 的全局均值，以及逐层、逐投影类型的均值；
    - 全局分位数（1/5/10/25/50/75/90/95/99%），用 [0,1] 上 10⁴ 个 bin 的直方图求，分辨率 10⁻⁴；
    - 有效步长 ‖Λ⊙Δ‖/‖Δ‖，分全局、逐层和逐投影；
    - Λ 与 |Δ| 的 Pearson 相关，在全部元素上用流式和精确计算；
    - Spearman 相关，在固定子样本上算（每个矩阵 2×10⁴ 个元素，seed 0）；
    - 另报 F_c + F_n = 0 的比例、Λ = 0 / 1 的比例，以及 F_c、F_n 的均值。
  - **自检（玩具张量，已过）：** F_c = F_n 时 Λ 全为 0.5；F_c = F_n = 0 时取 0.5；手算的四个值一致；F_c = F_n 时合并后的 bf16 权重与 WiSE 0.5 逐位相同。
- **评测链：** `wave11_eval.sh` / `wave11_eval_orch.sh`，复制自第十波，只改了：
  - 输出路径 `eval_outputs/mist_w3po_v1/wave11/` 和日志路径 `logs/mist_w3po_v1/wave11/`；
  - 卡号：卡 0 跑 F_c、en2zh 和 4 个 en2zh Talker 分片，卡 1 跑 F_n、zh2en 和 4 个 zh2en Talker 分片；
  - 新增 `fisher_c` 和 `fisher_n` 两个阶段，编排器按 Fisher（两卡并行）→ 合并（CPU）→ Thinker → Talker → ASR 的顺序跑。
  - Thinker、Talker、ASR 的命令与第十波逐字相同，包括 vllm_omni_030/bin 在 PATH 上和 SI_KV_CACHE_GIB=7.73。
  - 占位守护用自己的副本 `wave11_noplaceholder.sh`，有自己的 PIDS 和 STOP 文件（`logs/mist_w3po_v1/wave11/w11_workers.pids`、`w11_noplaceholder.stop`），只认 wave11_* / si_thinker_stream / si_infer / asr_transcribe 进程。
- **打分：** `voicecond/wave11_score.py`，复制自 `wave10_score.py`，系统为 base、erm_wise05、laplace，第 1、2 条规则与第十波（即本预登记）逐字相同：
  - 第 1 条：text BLEU 相对 3 次底座均值，ASR-BLEU 相对 S1；
  - 第 2 条：增量只看 text BLEU；"超过噪声底"指该单元 laplace − erm_wise05 的点估计大于 3 次底座两两之差的最大值；"没有退化"同时查 text BLEU 和 ASR-BLEU。
  - **自检（已过）：** 在现有输出上复算 erm_wise05 相对底座，与 `summary_erm_wise05.json` 对比 180 个数（点估计、CI、底座均值、噪声底），全部逐位一致，门判定也一致。
- **冒烟（11:47–11:49 CET，各 12 条，`wave11/fisher_smoke/`，不写 Fisher，不计入结果）：**
  - F_c 在卡 0、F_n 在卡 1，都跑通。峰值显存 70.1 GiB（模型加 D、D 的梯度和 Fisher 累加器，3 × 3.6 GB），预热后约 1.5 s/条，512 条约 13 分钟。
  - 逐样本 CE 合理：F_c 在 W0 的 zh→en clean 上为 0.28–2.43；F_n 在 W_e 的同一环境上为 0.009–0.72。
  - 事实记录：F_n 是在 W_e 处、在 ERM 自己的训练样本上算的。ERM 对这些样本已经拟合得较好（zh→en 只有 7004 行，2300 步里每行平均出现约 2.6 次），所以 F_n 的量级可能整体小于 F_c，Λ 整体偏向 0。这是预登记方法本身的性质，不做任何修正或缩放。
- **正式运行：** 编排器 `wave11_eval_orch.sh`（ARM=laplace CA=0 CB=1），用 setsid nohup 脱离会话，日志 `logs/mist_w3po_v1/wave11/w11_orch.log`。
