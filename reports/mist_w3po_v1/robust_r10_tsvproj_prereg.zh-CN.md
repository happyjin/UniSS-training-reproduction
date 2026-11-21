# 鲁棒 R10：从 τ_noisy 中整体移除 τ_clean 的奇异子空间（Clean-Subspace Removal of the Task Vector）预登记

2026-11-21 05:35 CET（布拉格）写，早于本方法的任何合并或评测数字。夜间执行者按用户授权设计。约束同前：只用同一批 UniST 数据（这里不再训练，只用已有的两个适配器）；不做蒸馏；不用 QABC 组件；一个模型两个方向；一个设置不扫参；点估计；dev 只用 D-U + bench_dev (b)。

## 1. 动机（来自 R1、R5、R6、R7 的结果）
- 夜里只有 R1（事后减法 τ_noisy − τ_clean）同时避开了两种偏移：(b) 干净不降、长度比回到底座、D-U 上没有"带噪比干净好"的条件式偏移；但不变性只保住了一部分。
- R1 的减法按**幅度**扣除：τ_clean 在某方向上的分量多大就扣多大。两条优化轨迹不对齐时，这会在共享方向上扣多或扣少（R1 分析第 2 点）。
- R10 换成按**子空间**移除：τ_clean 能表达的方向（它的 16 维左、右奇异子空间）就是"不用噪声也会学的东西"（风格、知识），把 τ_noisy 在这些方向上的分量整个投影掉，其余全部保留，不管幅度。

## 2. 文献
- **Task Singular Vectors: Reducing Task Interference in Model Merging（TSV-Merge，Gargiulo et al., CVPR 2025）：** 在任务向量的奇异子空间里分析和消除任务间干扰。
- Editing Models with Task Arithmetic（Ilharco et al., ICLR 2023）：任务向量的加减（R1 的来源）。
- **新意：** TSV 用奇异子空间减少不同任务之间的干扰；这里是同一任务、只差声学条件的两个任务向量，用"干净适配器的奇异子空间"定义并整体移除"与噪声无关的偏移"，只保留噪声特有的更新方向。没有看到先例。

## 3. 方法（一个设置）
- **输入：** τ_noisy = `wave14/run_unist/adapters_final.pt`（ERM，与 R1 相同），τ_clean = `r1/run_clean/adapters_final.pt`（R1 训练的干净版）。不训练新东西。
- **对每个被适配的权重**（48 个文本层的 q/k/v/o）：ΔW_c = U Σ Vᵀ（秩 ≤ 16，保留奇异值 > 1e−6·最大值的方向）；ΔW_rob = (I − U Uᵀ) ΔW_n (I − V Vᵀ)（左右两侧都投影）。
- **部署：** W' = W + 1 · ΔW_rob（系数 1，与 R1 相同，不扫）。
- **代码：** `voicecond/r10_merge.py`（`r1_merge.py` 的副本，只改差向量的构造；报告保留范数比例、bf16 舍入误差）。

## 4. 评测与判断规则
- **数据：** `r1/data`（D-U set u + bench_dev set b），si_thinker_stream la2 w1，--gpu-mem 0.85，只新跑 r10（英→中卡 4，中→英卡 5）。
- **对照：** base、unist_wise05（ERM）、r1（`r1/eval_<sys>/`）。
- **判断规则：** 与 R5 预登记第 5 节逐字相同（`rx_score.py`，method = r10，ERM = unist_wise05）。另报 r10 对 r1。

## 5. 运行
- `voicecond/r10_chain.sh`（脱离会话）：合并（CPU）→ 自己的 watcher（`rx_noplaceholder.sh`，POLL 1 s）→ 两个 Thinker（卡 4、5）→ STOP → 打分 → `r10/DONE`（失败写 `r10/FAILED`）。占位只按 PID 停，启动前等卡上没有 gpu_load 进程（最多 120 s）。不碰 guard 2726951、gpu_watch 52045、sync 1222453。
- 输出 `eval_outputs/mist_w3po_v1/r10/`，日志 `logs/mist_w3po_v1/r10/`。

## 结果
（待追加）
