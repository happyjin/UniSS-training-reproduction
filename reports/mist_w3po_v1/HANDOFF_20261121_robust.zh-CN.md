# 交接：鲁棒线（2026-11-21 13:15 CET）

新会话从这里接着做。回复用中文，时间写 CET。**token 不足：** 不做多种子，不自主探索，不开新的子代理或实验，除非用户明确同意（memory no-autonomous-exploration-token-budget）。

## 主线（v6 瓶颈分解：robust_bottleneck_v6_20261121.zh-CN.md）
- **Thinker：** 抗噪空间小（英→中 −0.3，中→英感知 −1.2），而且和参考译文的风格纠缠。R1–R13、K1、K1′ 全部不通过，已停止，只写进分析。
- **Talker 身份：** 是最大且能救的缺口。用完美 token（oracle）收回约 100%；token 翻转本身无害（D1），身份跟着 d-vector 走；BiCodec 的 ECAPA 在噪声下丢信息，TitaNet 不丢（D0）。
- **身份方法的结果**（S1 dev，4 格平均的身份损失收回比例，ECAPA2 / TitaNet）：

| 方法 | 收回比例 | 结论 |
|---|---|---|
| V1 | 47% / 26% | 基线 |
| K2、K3、K4 | — | 无效 |
| K5（软条件） | 51% / 38% | 无效 |
| A（TitaNet 映射） | — | 预检失败：只有 651 个说话人，映射过拟合 |
| **AbS** | 82% / 78% | 两种子都稳定；带噪两格 ASR −0.4 到 −0.5 |
| **AbS-C**（加 STOI 约束） | 76% / 60% | 带噪的 ASR 修好了；英→中干净两格 −0.46 / −0.67，主要来自种子 2；延迟 0.42 s |

- AbS 和 AbS-C 按逐格 ASR 规则都"不成立"，但逐格差异与 TTS 种子带来的波动同量级（V1 自己在两种子间相差 0.8）。
- 预登记：`robust_b_abs_prereg`、`robust_abs_c_prereg`、`robust_oracle_voice_prereg`、`robust_d0_token_stability`、`robust_d1_talker_sensitivity_prereg`。
- **其他：** CV-LA 的作用是"按噪声自适应的保守度"（只在英→中证实），代价是延迟增加。AoV / EITHER 只通过了弱检验。

## 待用户决定
1. 是否以 AbS-C（或 AbS）作为主方法，用非劣性 + 置信区间的规则，在测试集（RealSI、OBSI）上单种子跑一次；需要补写测试预登记变更。
2. 论文形态：诊断（oracle、D0、D1，Thinker 的纠缠）加 AbS-C。

## 卡
- 卡 0–3 归另一个 session，不要碰；本线只用卡 4–7。
- guard 2726951、gpu_watch 52045、sync 1222453 不能碰；只按 PID 停进程；不用 rm。
