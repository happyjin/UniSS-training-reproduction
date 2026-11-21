# 鲁棒 R13：按 Fisher 信息把"对干净输入重要的参数"退回底座（Clean-vs-Noisy Fisher-Weighted Reversion）预登记

2026-11-21 07:10 CET（布拉格）写，早于本方法的任何估计、合并或评测数字。夜间执行者按用户授权设计。约束同前：只用同一批 UniST 训练数据（这里只用来估计 Fisher，不训练新适配器）；不做蒸馏；不用 QABC 组件；一个模型两个方向；一个设置不扫参；点估计；dev 只用 D-U + bench_dev (b)。

## 1. 动机（来自 R9 与全夜的结果）
- R9（对抗抹去噪声信息）去掉了条件式偏移（D-U 差距不再为正），但无条件的风格偏移照旧（D-U 干净 +2.6 / +7.8，(b) 中→英干净 −1.0），因为翻译交叉熵本身在教风格。
- 全夜只有参数空间的事后处理（R1 / R10）稳定避开了风格偏移。它们按"另一个适配器"去减，受轨迹不对齐和 bf16 精度限制。
- R13 换一个依据：**逐参数地看这个参数对干净输入的预测有多重要**。对干净输入重要（干净 Fisher 大）的参数，ERM 的更新会改变干净输出 → 退回底座；对带噪输入重要而对干净输入不重要的参数，保留更新。系数 α_j = F_noisy,j / (F_noisy,j + F_clean,j)，没有超参。

## 2. 文献
- **Merging Models with Fisher-Weighted Averaging（Matena & Raffel, NeurIPS 2022）：** 用对角 Fisher 作逐参数的合并权重。
- **Overcoming Catastrophic Forgetting in Neural Networks（EWC，Kirkpatrick et al., PNAS 2017）：** 对旧任务 Fisher 大的参数施加保持。
- **新意：** 这里的两个"模型 / 任务"是同一个底座在**配对的干净 / 带噪输入**上的 Fisher，用它们的比值决定 ERM 鲁棒更新在每个参数上保留多少，目的是只保留"只对带噪输入重要"的更新。没有看到先例。

## 3. 方法（一个设置）
- **Fisher 估计（`voicecond/r13_fisher.py`，torchrun world 4，卡 1、2、3、7）：** 底座 Thinker（不加任何适配器），对 48 个文本层 q/k/v/o 的权重求梯度：
  - 512 条 UniST 训练行（每方向 256 条，sha1("20261121|r13|row|<dir>|<k>")），每条一份干净、一份带噪（混噪规则同 R3 / R6 的 `r3_subspace.mix`，键 sha1("20261120|r3sub|mix|<dir>|<k>")）；
  - 每条单独前向-反向（批量 1），损失 = 对参考译文的平均交叉熵（与训练目标相同），累加梯度平方：F_clean、F_noisy（经验对角 Fisher）；跨 rank 求和。
- **合并（`voicecond/r13_merge.py`）：** ΔW = unist_wise05 的稠密增量（`wave14/run_unist_wise05`，alpha 16，即 ERM-WiSE 0.5）；ΔW' = ΔW ⊙ F_noisy / (F_noisy + F_clean + 1e−30)；W' = W + ΔW'。报告 α 的均值、保留范数比例、bf16 舍入误差。
- 也就是说 r13 = "按 Fisher 退回的 unist_wise05"，与 ERM 对照逐参数可比。

## 4. 评测与判断规则
- `r1/data`（D-U set u + bench_dev set b），si_thinker_stream la2 w1，--gpu-mem 0.85，只新跑 r13（英→中卡 1，中→英卡 2）。对照：base、unist_wise05（ERM）、r1、r10。
- 判断规则与 R5 预登记第 5 节逐字相同（`rx_score.py`，method = r13，ERM = unist_wise05）。

## 5. 运行
- `voicecond/r13_chain.sh`（脱离会话，卡 1、2、3、7）：Fisher → 合并 → 两个 Thinker → STOP → 打分 → `r13/DONE`（失败写 `r13/FAILED`）。watcher `rx_noplaceholder.sh`（POLL 1 s），占位只按 PID 停，不碰 guard 2726951、gpu_watch 52045、sync 1222453。预计 08:30 前出分。
- 输出 `eval_outputs/mist_w3po_v1/r13/`，日志 `logs/mist_w3po_v1/r13/`。

## 结果
（待追加）
