# 鲁棒 R11：在冻结的"干净适配器"之上只拟合残差的鲁棒 LoRA（Fitting the Residual over a Frozen Clean Adapter）预登记

2026-11-21 05:45 CET（布拉格）写，早于本方法的任何训练或评测数字。夜间执行者按用户授权设计。约束同前：只用同一批 UniST 训练数据；不做蒸馏（目标是参考译文；冻结的干净适配器只是模型里的一个固定加项，不产生软标签或伪标签）；不用 QABC 组件；一个模型两个方向；一个设置不扫参；点估计；dev 只用 D-U + bench_dev (b)。

## 1. 动机（来自 R1、R10 的结果）
- R1 / R10 在参数空间事后从 τ_noisy 里去掉 τ_clean 的分量，都避开了两种偏移（(b) 干净基本不降、D-U 没有条件式偏移），但不变性只保住一部分：两个适配器是分开训练的，轨迹不对齐，"去掉"总是去多或去少；R10 的差向量只剩 19% 的范数，bf16 合并后相对误差 48%。
- R11 改成**训练时就把"与噪声无关的部分"交给一个已经学好的模型**：把 R1 训练的干净适配器 τ_clean（只在干净 UniST 音频上学到的风格、长度、知识）冻结在模型里，再在带噪样本上训练一个新的 LoRA R。因为风格已经由 τ_clean 解释了，R 只需要拟合"有噪声时还缺的那部分"，即残差。部署时去掉 τ_clean，只用 θ + R。

## 2. 文献
- **Unlearn Dataset Bias in Natural Language Inference by Fitting the Residual（He, Zha, Wang, DeepLo@EMNLP 2019）**；**Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases（Clark et al., EMNLP 2019）**；**End-to-End Bias Mitigation by Modelling Biases in Corpora（Karimi Mahabadi et al., ACL 2020）：** 先训练只会利用偏差的模型，冻结后与主模型组合训练，主模型只学残差，测试时去掉偏差模型。
- **新意：** 这些工作里"偏差模型"是人为设计的只看部分输入的模型；这里的"偏差"是鲁棒适配带来的输出分布偏移，偏差模型就是**同数据、同目标、只用干净音频训练的适配器**。把"fitting the residual"用于把声学不变性从目标风格中分离，没有看到先例。

## 3. 方法（一个设置）
- **代码：** `voicecond/r11_train.py`，`r1_train.py`（即 `wave14_train.py`）的副本，改三处：
  1. 每个文本层 q/k/v/o 上，除了新的可训练 LoRA R（rank 16，alpha 32，b 初始化为 0，fp32），再加一个**冻结**的加项 s·B_c A_c，B_c、A_c 取自 `r1/run_clean/adapters_final.pt`（fp32，不合并进 bf16 权重，避免舍入）；
  2. 每步只用 8 个环境中的 6 个带噪环境（48 / 64 个位置，同一 sha1 行键和混噪键，等于第十四波批次去掉 2 个干净环境）；损失是这 48 句逐句平均交叉熵的平均；
  3. 只训练 R。
- 其余与第十四波逐项相同：AdamW lr 1e-4，betas (0.9, 0.999)，wd 0，余弦到 0，2300 步，种子 20261119，不裁剪。
- **卡：** 4、5（R10 交卡后），world 2，每 rank 24 条，微批量 12（失败则 8、6）。
- **最终系统：** r11 = θ + 1 · R（去掉 τ_clean；系数 1，与 R1 / R10 同口径，不扫），`wave5_m_merge.py`（不改）合并。

## 4. 评测与判断规则
- **数据：** `r1/data`（D-U set u + bench_dev set b），si_thinker_stream la2 w1，--gpu-mem 0.85，只新跑 r11（英→中卡 4，中→英卡 5）。
- **对照：** base、unist_wise05（ERM）、r1、r10。
- **判断规则：** 与 R5 预登记第 5 节逐字相同（`rx_score.py`，method = r11，ERM = unist_wise05）。

## 5. 运行
- `voicecond/r11_chain.sh`（脱离会话）：等 R10 链结束 → 自己的 watcher（`rx_noplaceholder.sh`，POLL 1 s）→ 冒烟 → 训练 → 合并 → 两个 Thinker → STOP → 打分 → `r11/DONE`（失败写 `r11/FAILED`）。占位只按 PID 停，启动前等卡上没有 gpu_load 进程（最多 120 s）。不碰 guard 2726951、gpu_watch 52045、sync 1222453，只用卡 4、5。
- 输出 `eval_outputs/mist_w3po_v1/r11/`，日志 `logs/mist_w3po_v1/r11/`。预计训练约 3 h，09:30 前后出分。

## 结果
（待追加）
