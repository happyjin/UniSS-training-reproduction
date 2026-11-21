# 鲁棒 K1：任务向量差不合并部署（τ_noisy − τ_clean 作为一个精确 LoRA）预登记

2026-11-21 07:25 CET，写于任何 K1 数字之前。方案出处：`robust_review_v5_20261121.zh-CN.md` 第 4 节 K1（用户 07:20 决定开跑）。前置：`robust_r1_taskdiff_prereg.zh-CN.md`（R1）。约束同 memory：不蒸馏、一个设置不扫参、只看点估计（不做 bootstrap）、只用开放数据训练。

## 1. 动机
R1 把 θ_base + (ΔW_noisy − ΔW_clean) 合并进 bf16 权重。风格去除成立（不同域干净不掉、长度比回到底座），但不变性只保住一部分；差向量很小（‖Δ‖/‖W‖ ≈ 1.2%），bf16 合并后差向量平均相对误差 0.17（最大 0.49）。K1 不合并：差在激活上逐层计算（y = W x + B A x），W 保持底座原值，误差只剩 LoRA 因子的 bf16 舍入。要回答的问题：收回合并误差后，R1 丢掉的不变性能恢复多少。

## 2. 精确的秩 32 LoRA
- τ_noisy = `wave14/run_unist/adapters_final.pt`，τ_clean = `r1/run_clean/adapters_final.pt`，两者都是 48 个文本层的 q/k/v/o，rank 16，alpha 32，s = alpha / rank = 2。
- 拼接：A = [a_n ; a_c]（32 × in），B = [s_n b_n , −s_c b_c]（out × 32），则 B A = s_n b_n a_n − s_c b_c a_c = ΔW_n − ΔW_c，在数学上精确。
- 存成 PEFT 适配器（布局同 `wave16_make_lora.py`：键 `base_model.model.thinker.model.<stem>.lora_A/B.weight`，fp32），r = lora_alpha = 32，vLLM 的缩放 lora_alpha / r = 1（两边的缩放都已折进 B）。
- 代码 `voicecond/k1_make_lora.py`。**校验**（从存盘文件和配置重读，全部 192 个模块）：fp32 下 max |scale · B A − (ΔW_n − ΔW_c)| 和相对 Frobenius 误差；任一模块相对误差 > 1e-5 就退出，不部署。另报告：因子按 bf16 舍入后（vLLM 默认 lora_dtype = 模型 dtype）的相对误差，与 R1 合并的 0.17 对照。

## 3. vLLM 服务方式
- 第十六波已经发现：vLLM 按 config 的 `Qwen3OmniMoeForConditionalGeneration` 加载 vllm-omni 的外壳类，报 "does not support LoRA yet"；只有其中的 Thinker 类 `Qwen3OmniMoeThinkerForConditionalGeneration` 声明 SupportsLoRA。该 Thinker 类自带权重映射（`thinker.model.` → `language_model.model.`，丢弃 `talker.` / `code2wav.`），可以直接读完整检查点。
- 所以 K1 用包装脚本 `voicecond/k1_si_thinker.py`（不复制、不修改 si_thinker_stream）：在 si_thinker_stream.main() 之前给 `vllm.LLM.__init__` 加上
  `hf_overrides={"architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]}, enable_lora=True, max_loras=1, max_lora_rank=32, lora_target_modules=["qkv_proj", "o_proj"]`，
  并让 `vllm.LLM.generate` 对每个请求都带同一个 LoRARequest。模块名：PEFT 里是 q_proj/k_proj/v_proj/o_proj，vLLM 按该类的 `packed_modules_mapping` 把 q/k/v 打包到 qkv_proj，与第十六波相同。MoE 专家不挂 LoRA。
- **通路自检（同一时段、同卡，先于判定）：** 用同一个包装脚本，把 τ_noisy 单独作为 LoRA（`wave16_make_lora.py --alphas 1`，rank 16，缩放 2）服务，与合并副本 `r1/eval_unist` 比。条件：8 个单元（方向 × {u, b} × {干净, 带噪}）的 text BLEU 差都在 ±0.5 以内。另报告最终译文逐条完全相同的比例，以及 K1 与 base 逐条相同的比例（防止 LoRA 实际没挂上、输出等于底座）。
  - 自检不过：K1 不做判定，只报告"LoRA 通路不可信"，第 2 步不训练。

## 4. 评测（只跑 Thinker，点估计）
- **数据：** 与 R1 完全相同，`eval_outputs/mist_w3po_v1/r1/data` = UniST 同域 dev（devU，set u，每方向干净 300 + 带噪 300）+ S1 bench_dev (b)（英→中 337 + 337，中→英 706 + 706）。
- **Thinker：** si_thinker_stream `--policy la2 --windows 1 --gpu-mem 0.85`，贪心；`--thinker` = 原始 Qwen3-Omni + K1 LoRA。
- **对照（复用已有输出，不重跑）：** base = `r1/eval_base`，unist_wise05 = `r1/eval_unist_wise05`，R1 合并 = `r1/eval_r1`（另列 unist = `r1/eval_unist`）。
- **指标**（`voicecond/k1_score.py`，直接调用 `rx_score.py` 的 `cell()`，口径与 R1 表相同）：每方向 × {u, b} × {干净, 带噪}：text BLEU（论文规范化）、长度比、重复率（规范化后 > 3 倍参考）、text D1 LAAL；带噪 − 干净差距。
- **判定规则（K1 对 R1 合并，4 个单元 = 方向 × {u, b}）：**
  1. 带噪更好：4 个单元带噪 BLEU（K1 − R1）的平均 > 0，且至少 3 个单元 K1 > R1；
  2. 干净不掉：每个单元干净 BLEU(K1) ≥ R1 − 0.3；
  3. 长度比不更偏：每个单元 mean(|长度比 − 1|，干净与带噪) 的 K1 ≤ R1 + 0.005（0.005 只吸收四舍五入）。
  - 三条都满足 = K1 通过。另外报告 K1 对 base、对 unist_wise05（含 `rx_score.py` 的夜间规则，只作信息）。
- **卡：** 只用卡 4、5，每卡一个 vLLM Thinker。K1 英→中在卡 4、中→英在卡 5，然后通路自检同样分配。启动前按 PID 停掉该卡的 gpu_load 占位，并等到卡上没有 gpu_load 进程（≤ 120 s，同 combo_orch.sh launch()）。自带 watcher `voicecond/k1_noplaceholder.sh`（rx_noplaceholder.sh 的副本，POLL 1 s，自己的 PIDS 和 STOP）。
- 正式运行前先在卡 4 上用 `--limit 3` 冒烟，只确认 LoRA 引擎能起、输出非空（冒烟输出放 `k1/smoke`，不进结果）。

## 5. 第 2 步（有条件，预登记于此）：只用带噪环境的 τ_noisy′
- **触发：** 第 3 节通路自检通过，且 K1 按第 4 节规则通过。否则不训练，只报告。
- **设计：** `voicecond/k1_train.py` = `wave14_train.py` 的副本，只多一个开关 `--noisy-only`（与 R1 的 `--clean-only` 对称）：
  - 全局批量 64、步/位置 → 行的映射（sha1 "20261119|w5m|row|s|j"）、参考译文、LoRA（rank 16 / alpha 32）、优化器、余弦调度、种子 20261119、2300 步、world 4（每个 rank 16 条，微批量 16）全部不变；
  - 原来属于 2 个干净环境的 16 个位置（每方向 8 个）不再是干净音频，改为混入背景：位置 j 的环境 = ("music", "crowd", "sfx")[(j mod 8) mod 3]（每方向 music 3 个、crowd 3 个、sfx 2 个），混合随机数仍用同一键 "20261119|w5m|mix|s|j"，混合方式与带噪环境相同；其余 48 个带噪位置与 τ_noisy 完全相同。
  - 于是只剩 6 个带噪环境，64 个位置全部带噪，且每个位置与 τ_clean 同一行、同一参考，τ_noisy′ − τ_clean 在全部 64 个位置上都只差声学条件（τ_noisy 只有 48 个）。
  - 说明：6 个环境无法整除 64，所以不是"每环境 8 条 × 6"，而是保持 64 个位置的配对结构、把干净位置分给三种噪声。这是在"同数据 / 同步数 / 同全局批量 / 同种子"下唯一能保持与 τ_clean 逐位置配对的写法。
- **卡：** 1、2、3、7（world 4）。启动前若这些卡上有非占位的作业（例如 K2 的 Talker 评测在卡 2、3），等它们结束；占位按 PID 停。端口 29693。约 2–2.5 h（R1 τ_clean 同配置 1 h 58 min）。
- **部署和评测：** τ_noisy′ − τ_clean 用同样的 `k1_make_lora.py` 拼成秩 32 LoRA（同样校验），同样的包装脚本在卡 4、5 上评测，同样的指标。
- **第 2 步的判定：** 把第 4 节的三条规则里的 R1 换成 K1（K1′ 对 K1）；另报告对 R1、base、unist_wise05。
- 链条 `voicecond/k1_step2_chain.sh`（脱离会话）：等卡 → 训练 → 拼 LoRA → Thinker → 打分 → `k1/step2/DONE`，任何失败写 `k1/step2/FAILED`。

## 6. 运行和输出
- 第 1 步链条 `voicecond/k1_chain.sh`（setsid nohup，fd 8/9 关闭）：watcher → K1 Thinker（卡 4、5）→ 通路自检 Thinker（卡 4、5）→ STOP watcher → `k1_score.py` → `k1/DONE`；失败写 `k1/FAILED`。
- 不碰 guard 2726951、gpu_watch 52045、sync 1222453，也不碰其他卡上的作业。
- 输出 `eval_outputs/mist_w3po_v1/k1/`，日志 `logs/mist_w3po_v1/k1/`。

## 结果（2026-11-21 07:45 CET，只追加；点估计，不做 bootstrap）
来源：`eval_outputs/mist_w3po_v1/k1/score.json`（`k1_score.py`），`k1/score_rx.json`（`rx_score.py`，仅作信息）。运行：链条 07:31–07:41 CET，卡 4、5；冒烟（卡 4，`--limit 3`）07:27 通过，LoRA 引擎经 Thinker 架构覆盖正常启动。

**LoRA 构造校验（第 2 节）：** 192 个模块，fp32 max |B A − (ΔW_n − ΔW_c)| = 1.9e-9，最大相对误差 6.0e-7，通过。因子 bf16 舍入后的相对误差平均 0.0053（最大 0.018），R1 合并为 0.17（最大 0.49），约小 30 倍。‖ΔW_n − ΔW_c‖ / ‖ΔW_n‖ 平均 0.66（与 R1 相同）。

每格为 text BLEU / 长度比（重复率都 ≤ 0.01；text D1 LAAL 与 R1 相差 ≤ 0.05 s）：

| 单元 | base | unist_wise05 | R1 合并 | **K1 不合并** | 通路自检：unist 合并 / unist LoRA |
|---|---|---|---|---|---|
| 英→中 u 干净 | 52.26 / 0.971 | 56.96 / 0.983 | 52.04 / 0.961 | **52.82 / 0.970** | 58.14 / 58.34 |
| 英→中 u 带噪 | 52.78 / 0.969 | 56.92 / 0.992 | 52.00 / 0.969 | **53.06 / 0.973** | 56.67 / 56.78 |
| 英→中 b 干净 | 37.99 / 0.881 | 37.21 / 0.920 | 37.89 / 0.882 | **38.27 / 0.886** | 36.35 / 36.51 |
| 英→中 b 带噪 | 35.18 / 0.885 | 35.39 / 0.913 | 35.74 / 0.894 | **35.83 / 0.897** | 34.78 / 34.61 |
| 中→英 u 干净 | 35.97 / 1.010 | 48.03 / 1.009 | 38.81 / 0.989 | **39.18 / 0.986** | 46.91 / 46.97 |
| 中→英 u 带噪 | 34.20 / 1.022 | 47.21 / 1.007 | 37.25 / 1.019 | **38.35 / 1.000** | 45.91 / 46.52 |
| 中→英 b 干净 | 27.66 / 0.981 | 25.29 / 1.050 | 27.49 / 0.988 | **27.22 / 0.990** | 24.53 / 24.82 |
| 中→英 b 带噪 | 25.16 / 1.041 | 25.27 / 1.051 | 26.73 / 0.991 | **27.00 / 0.990** | 25.03 / 24.75 |

带噪 − 干净的差距：

| 单元 | base | unist_wise05 | R1 合并 | K1 |
|---|---|---|---|---|
| 英→中 u | +0.52 | −0.04 | −0.04 | +0.24 |
| 英→中 b | −2.81 | −1.82 | −2.15 | −2.44 |
| 中→英 u | −1.77 | −0.82 | −1.56 | −0.83 |
| 中→英 b | −2.50 | −0.02 | −0.76 | −0.22 |

**1. 通路自检（第 3 节）：按预登记不通过。** unist LoRA − unist 合并的 BLEU 差：+0.20、+0.11、+0.16、−0.17、+0.06、**+0.61**、+0.29、−0.28，8 个单元里 7 个在 ±0.5 以内，中→英 u 带噪超出 0.11。逐条完全相同的比例 0.73–0.82（K1 对 base 只有 0.29–0.50，说明 LoRA 确实挂上了；unist LoRA 的输出整体与合并副本一致、与底座明显不同）。超出的方向是 LoRA 高于合并，与"合并副本自身也带 bf16 舍入"一致，但规则写的是硬门槛，不在事后放宽。

**2. K1 对 R1 合并的规则（第 4 节）：三条点估计都满足，但按第 3 节，自检不过时 K1 不做判定。**
- 带噪：4 个单元 K1 − R1 = +1.06、+0.09、+1.10、+0.27，平均 +0.63，4 个单元都更好；
- 干净：+0.78、+0.38、+0.37、−0.27，都在 −0.3 以内；
- 长度比偏离：0.029 / 0.108 / 0.007 / 0.010，均 ≤ R1（0.035 / 0.112 / 0.015 / 0.011）。
- 信息（`rx_score.py` 夜间规则，相对底座）：不通过，中→英 b 干净 −0.44（> 0.3），英→中 b 差距不如 unist_wise05 −0.62。

**3. 第 2 步：按预登记不训练。** τ_noisy′ 的脚本（`k1_train.py`，CPU 上核对过：16 个原干净位置换成 music/crowd/sfx、行号与 τ_clean 相同，其余 48 个位置与第十四波逐样本相同）、`k1_run.sh`、`k1_step2_chain.sh` 都已就绪（bash -n / DRY_RUN 检查），没有启动。需要用户决定：是否接受"7/8 单元在 ±0.5 以内、超出的一格是 LoRA 更好"作为通路有效，从而启动第 2 步（卡 1、2、3、7，约 2–2.5 h，加评测约 10 min）。

**解读（不作判定，只作分析）：** 收回 bf16 合并误差后，同一个差向量在带噪单元上比 R1 合并平均高 0.63，同域（u）的差距明显缩小（中→英 u −1.56 → −0.83，与 unist_wise05 持平），而不同域干净基本不掉、长度比仍在底座水平。这支持 R1 原因分析中的"合并误差吃掉了部分不变性"。但 (b) 上的差距仍不如 unist_wise05（英→中 −2.44 对 −1.82），所以剩余的不变性损失主要不是来自合并误差，而是来自差向量本身（τ_noisy 只有 75% 的位置带噪、两条轨迹不线性对齐），这正是第 2 步要检验的。
