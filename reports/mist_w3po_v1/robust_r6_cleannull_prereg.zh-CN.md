# 鲁棒 R6：限定在干净语音零空间里的编码器 LoRA（Clean-Null-Space Encoder LoRA）预登记

2026-11-20 23:30 CET（布拉格）写，早于本方法的任何估计、训练或评测数字。夜间执行者按用户授权设计并启动。约束同 R5：只用同一批 UniST 训练数据；不做蒸馏（目标是参考译文，无教师、无伪标签）；不用 QABC 组件；一个模型两个方向；一个设置不扫参；点估计；dev 只用 D-U + bench_dev (b)。

## 1. 动机（v3 主线）
编码器 LoRA（第十七波 enc_wise05）用翻译交叉熵训练，干净输入也会被改动，于是学到了 UniST 的风格（D-U 干净中→英 +8.1，训练集里没抽到的句子同样提升）。R3 把 LoRA 的**输出**限定在噪声子空间里，但干净输入经过 LoRA 仍会产生增量（只要它在那些方向上有分量）。

R6 换一个约束位置：把 LoRA 的**输入**限定在"干净语音激活几乎不占据的方向"（干净激活协方差的零空间）里。于是对干净输入，增量 ΔW·x_clean ≈ 0，**从结构上保证干净输入的表示不变**（评价第 2 轴），LoRA 只能对"干净语音里不出现、而噪声带进来"的激活分量起作用，也就是只能学对干扰的响应（第 1 轴），没有通道去改变干净语音的翻译风格。

## 2. 文献
- **AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models（Fang et al., ICLR 2025）：** 把参数扰动投影到"需保持知识的键向量"的零空间上，使保持集的输出不变。
- **Adam-NSCL（Wang et al., CVPR 2021，Training Networks in Null Space of Feature Covariance for Continual Learning）：** 在旧任务特征协方差的零空间里更新。
- 对照 R3：SEEN（ACL 2026）/ O-LoRA（EMNLP 2023 Findings）是在输出侧用噪声子空间约束。

**新意：** AlphaEdit / Adam-NSCL 保护的是旧知识或旧任务；这里保护的是"干净输入的表示"，零空间由**同一编码器在干净语音上的激活**定义，用来把鲁棒适配限定为"只对干扰起作用"。没有看到把零空间约束用于抗噪适配、以从结构上排除干净输入上的输出偏移的先例。

## 3. 零空间估计（一个规则，写定）
- **代码：** `voicecond/r6_nullspace.py`（`r3_subspace.py` 的副本，钩子从模块输出改为模块输入）。
- **数据：** 与 R3 相同的 2000 条 UniST 训练行（每方向 1000 条，sha1("20261120|r3sub|row|<dir>|<k>")），每条一份干净、一份带噪（混噪规则同 R3）。只用训练数据。
- **统计：** 对每层的 q/k/v 共同输入（3 个模块输入相同）和 out_proj 输入，各算干净帧的非中心化协方差 K_c = E[x xᵀ]（fp64）；另算噪声残差 D = x_noisy − x_clean 的协方差 K_d（只用于诊断）。
- **规则：** K_c 特征分解，特征值从小到大累加，取累计和 ≤ 1% · tr(K_c) 的最多个方向，记为 U0（[1280, r0]），投影 P = U0 U0ᵀ。即干净激活在 P 中的能量 ≤ 1%。1% 是事先写定的唯一取值，不扫。
- **诊断（只记录，不据此改规则）：** 每个模块的 r0；噪声残差能量落在零空间里的比例 tr(U0ᵀ K_d U0) / tr(K_d)。
- **说明：** 如果噪声残差几乎不在零空间里，R6 的适配能力会很弱、结果接近底座，这也是如实的结果，不改阈值。

## 4. 训练（一个设置）
- **代码：** `voicecond/r6_enc_train.py`，`r3_enc_train.py` 的副本，只改投影位置：Δh = s · B A (U0 U0ᵀ x)（R3 是 Δh = Q Qᵀ (s · B A x)）。
- 其余与第十七波 / R3 逐项相同：LoRA 在音频编码器 32 层自注意力的 q/k/v/out_proj，rank 16，alpha 32，b 初始化为 0，fp32；8 个环境、同一 sha1 行键和混噪键；AdamW lr 1e-4，余弦到 0，2300 步，全局批量 64，种子 20261119，翻译交叉熵（参考译文）。
- **卡：** 0、6（R3 链结束交卡后），world 2，微批量按 R3 的做法取冒烟能跑的最大值（32、16、8）。
- **最终系统：** r6_wise05，WiSE α = 0.5（alpha 减半为 16），与 enc_wise05 / r3_wise05 同口径；`r6_merge.py`（`r3_merge.py` 的副本）合并：W' = W + s · (B A) U0 U0ᵀ，带前向等价自检。

## 5. 评测与判断规则
- **数据与系统：** `r1/data`（D-U set u + bench_dev set b），si_thinker_stream la2 w1，--gpu-mem 0.85，只新跑 r6_wise05（英→中卡 0，中→英卡 6）。对照：base 用 `r1/eval_base`；ERM 对照 enc_wise05 用 R3 链的 `r3/eval_enc_wise05/devU` 加 `wave17/eval_enc_wise05`（(b)）；另报 r3_wise05（`r3/eval_r3_wise05/{devU,s1}`）。
- **指标与判断规则：** 与 R5 预登记第 5 节逐字相同（`rx_score.py`，method = r6_wise05，ERM = enc_wise05）：以底座为参照的不变性、干净不退化、无输出偏移、延迟四条，加 (b) 上不差于 ERM。五条都满足记为"有效"；有效也先不跑测试。

## 6. 运行
- **链条：** `voicecond/r6_chain.sh`（脱离会话）：等 R3 链结束（`r3/CHAIN_DONE` 或 `CHAIN_FAILED`，没有存活的 r3_* 进程，两卡只剩占位）→ 自己的 watcher（`r3_noplaceholder.sh`，自己的 PIDS / STOP，POLL 1 s）→ 零空间估计（卡 0）→ 冒烟 → 训练 → WiSE 副本 → 合并 → 两个 Thinker → STOP → 打分 → `r6/DONE`（任何失败写 `r6/FAILED`）。
- 占位只按 PID 停，启动前等卡上没有 gpu_load 进程（最多 120 s）。不碰 guard 2726951、gpu_watch 52045、sync 1222453，只用卡 0、6。
- 输出 `eval_outputs/mist_w3po_v1/r6/`，日志 `logs/mist_w3po_v1/r6/`。

## 结果
（待追加）

## 运行记录（2026-11-21 00:55 CET，只追加）
- 零空间估计（00:46–00:49 CET，2000 对，155690 帧）：r0 平均 qkv 303 / out 504（范围 102–1263）。**诊断：噪声残差能量落在零空间里的比例只有约 4%（qkv 0.042、out 0.043）。** 即噪声在编码器内部的扰动绝大部分落在干净语音本身占据的方向上；按预登记不改阈值，如实运行，预计 R6 的适配能力很弱、结果可能接近底座。
- 冒烟 mb 32 通过，00:50 CET 开始训练（卡 0、6）。
