# 鲁棒 R1：抵消风格的任务向量差（τ_noisy − τ_clean）预登记

2026-11-20 21:50 CET，写于任何 R1 训练或评测数字之前。方案出处：`robust_analysis_and_ideas_v3_20261120.zh-CN.md` 第 3 节 R1（用户同意开跑）。约束同 memory：
- 训练只用同一批 UniST 数据；
- 不做蒸馏：目标是参考译文，没有教师模型；
- 一个设置，不扫参；
- 先看点估计。

## 1. 动机
第十四波的 UniST LoRA（τ_noisy）同时学到了两样东西：一是对声学干扰的不变性，二是输出分布的偏移，包括 UniST 合成译文的冗长风格（长度比 1.07–1.12）和解码退化（UE 在 RealSI health 上出现重复循环）。

如果用同一批数据、同样的参考译文、同样的随机数再训一个只用干净音频的 LoRA τ_clean，两者学到的风格和知识偏移应当相同，差别只在声学条件。所以 τ_noisy − τ_clean 近似只剩"对干扰的适应"。

## 2. τ_noisy 是什么（读第十四波代码核实）
- `wave14_train.py`：Qwen3-Omni Thinker，bf16，音频编码器冻结。LoRA 加在全部 48 个文本层的 q/k/v/o 上，rank 16，alpha 32，b 初始化为 0。
- **8 个环境：** {zh2en, en2zh} × {clean, music, crowd, sfx}。每步全局批量 64，每个环境 8 条。
- **采样：** 第 s 步第 j 个位置，行号 = sha1("20261119|w5m|row|s|j") mod n_rows；噪声由 sha1("20261119|w5m|mix|s|j") 决定。
- **优化：** AdamW lr 1e-4，余弦调度到 0，2300 步，种子 20261119。world 4（卡 0–3），每个 rank 16 条。
- **注意：τ_noisy 的 8 个环境里有 2 个是干净环境。**
  - 它的 64 个位置中，有 16 个本来就是干净样本，与 τ_clean 在同一步、同一位置的样本完全相同（同一行、不加噪）。
  - 所以 τ_noisy − τ_clean 中，噪声的贡献只来自其余 48 个位置（75%）。
  - 影响：差分的幅度会比"全带噪对全干净"略小，方向不变。
  - 不另训一个"全带噪"的对称版本，理由：
    1. τ_noisy 是第十四波和 UE / UW 已经评测过的同一个适配器，换掉它，R1 就不再是"在当前方法上去掉风格分量"；
    2. 一个设置不扫参；
    3. 两边在这 16 个位置上完全相同，梯度只在优化轨迹上有间接差别，不会引入新的偏移。

## 3. τ_clean
- 代码：`voicecond/r1_train.py`，它是 `wave14_train.py` 的副本，只多一个开关 `--clean-only`：每个位置照常取同一行、同一参考译文，但不混入背景。其余完全相同：同数据 `eval_outputs/mist_w3po_v1/wave14/data`、同 LoRA、同优化器和调度、同种子、同步数 2300、同全局批量 64。
- **卡：** world 4，卡 1、2、3、7。每个 rank 16 条，微批量 16，也就是每步一个微批（与第十四波每个 rank 16 条相同）。
- **冒烟（卡 2、3，world 2，只确认能跑和显存，不是结果）：**
  - 微批量 16：每个 rank 显存 69.0 GiB，约 6.3 s/步；
  - 微批量 32：78.4 GiB，约 4.4 s/步；
  - 第 0 步的 ce 为 0.91，与第十四波第 0 步同量级。
- **启动时机：** 卡 1、7 正在跑 test_robust 的 UE / UW 测试。等 `test_robust/UE/DONE` 和 `UW/DONE` 都出现、卡 1、7 上没有它们的进程之后，由脱离会话的等待脚本自动启动。

## 4. 部署
- θ = θ_base + c · (ΔW_noisy − ΔW_clean)，c = 1，不扫。
- ΔW_x = (alpha / rank) · b_x a_x，在 fp32 中分别算好再相减，不直接相减 LoRA 因子。
- 合并成完整权重 `eval_outputs/mist_w3po_v1/r1/merged_r1`（`voicecond/r1_merge.py`）。
- 同时报告：
  - bf16 舍入检查：‖(W′_bf16 − W) − cΔ‖ / ‖cΔ‖ 的均值和最大值；
  - ‖cΔ‖ / ‖W‖；
  - cos(ΔW_noisy, ΔW_clean) 的均值。

## 5. 评测（只跑 Thinker，点估计，不做 bootstrap）
- **数据：** `eval_outputs/mist_w3po_v1/r1/data` = UniST 同域 dev（devU，set u，每方向干净 300 + 带噪 300），加上 S1 bench_dev (b)（英→中 337 + 337，中→英 706 + 706）。
- **Thinker：** si_thinker_stream `--policy la2 --windows 1`，贪心，同 wave2 S1。
- **系统：**
  - base（原始 Qwen3-Omni）；
  - unist（第十四波 α = 1，`wave14/merged_unist`）；
  - unist_wise05（`wave14/merged_unist_wise05`）；
  - r1（`r1/merged_r1`）。
- **指标**（每方向 × {u, b, b 无串音} × {干净, 带噪}）：
  - text BLEU（论文口径 `evaluation.text_metrics.corpus_bleu`）；
  - 长度比 sys_len / ref_len；
  - 重复率：规范化后译文单位数 > 3 倍参考的句子比例；
  - text LAAL（si_text_eval.laal）；
  - 噪声差距 = 带噪 − 干净。
- **判断规则**（点估计，r1 对 unist_wise05，每方向 × {u, b}）：
  1. 差距不变差（≥）；
  2. 干净 BLEU 不降（≥）；
  3. |长度比 − 1| 更接近 0（干净、带噪平均）；
  4. 重复率不升（干净 + 带噪）。
  - 四条都满足记为"有效"。另外报告 r1 对 base、r1 对 unist。
- **打分：** `voicecond/r1_score.py`，输出 `r1/score.json`。

## 6. 运行
- 链条 `voicecond/r1_chain.sh`（脱离会话）：等卡 → 训练 → 合并 → 4 个系统的 Thinker → 打分 → DONE。任何失败写 FAILED。
- 自带 watcher `r1_noplaceholder.sh`（POLL 1 s），有自己的 PIDS 和 STOP。启动前 SIGTERM 占位，并等到卡上没有 gpu_load 进程（最多 120 s）。
- 不碰 guard 2726951、gpu_watch 52045、sync 1222453，也不碰其他卡上的作业。
- 输出 `eval_outputs/mist_w3po_v1/r1/`，日志 `logs/mist_w3po_v1/r1/`。

## 变更 1（21:50 CET，写在任何训练数字之前）
- 卡从"卡 2、3，world 2"改为"卡 1、2、3、7，world 4"，原因是用户要求用 4 卡。这样与 τ_noisy 的 world 4、每个 rank 16 条完全一致。
- world 2 的训练从未启动，只在卡 2、3 上跑过冒烟，所以没有需要改名为 superseded 的训练输出。冒烟目录 `r1/smoke16`、`r1/smoke32` 保留。
