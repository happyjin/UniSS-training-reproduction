# 鲁棒第六波：音色条件的身份 / 无关属性解耦 Talker（DisenBooth 式）预登记

2026-11-19 晚（布拉格），写于任何第六波数字之前。用户："开。不算。用另外四卡开始"。用卡 4–7；主线 Fishr 在卡 0–3。约束见 memory `robust-paper-constraints`：
- 一个模型服务两个方向；
- 不用蒸馏：目标是真实语音的语义码，不用任何模型输出作目标；
- 不碰 QABC；
- 训练只用开源非测试数据；
- 一个设置，不扫参，一个 TTS 种子 20261021。

## 动机（已测）
- 第一波：音色 token 改变 Talker 时长 ±5%（Wilcoxon p ≤ 1e-7）。线性语速探针 R² 0.89，但沿它钳制只让时长变异降了 5.7%。
- 第三波 V 组：只换音色 token，D2 LAAL 就稳定变化 −0.04 到 −0.07 s（CI 不含 0）。
- 两者说明音色条件里泄漏了与身份无关的属性（语速、噪声），而且是非线性的。
- 来源：DisenBooth: Identity-Preserving Disentangled Tuning for Subject-Driven Text-to-Image Generation（ICLR 2024，arXiv 2305.03374）。

## 方法（一个设置）
- **身份分支：** c_id = σ(m) ⊙ E(g)。
  - g 是 32 个音色 token，用 Idea 1 C 编码器作用于提示得到（训练时提示随机混入非留出背景）；
  - E 是 T1c 自己的 global token 嵌入；m 是可学习的逐维 mask，初值使 σ(m) ≈ 1。
- **无关分支：** c_irr = h_ψ(提示的 log-mel)，即 4 个软前缀向量。h_ψ 是一个小型卷积 + 注意力池化编码器，只在训练时接入。
- **损失：**
  L = NLL(x | c_id, c_irr) + λ_w·NLL(x | c_id) + λ_c·InfoNCE(c̄_id) + λ_o·cos²(c̄_id, c̄_irr)
  - x 是真实目标语音的语义码；c̄ 表示在位置上取平均；
  - InfoNCE 的正例：同一说话人的另一条提示（另一句话、另一种背景）；负例：batch 内其他说话人；τ = 0.07；
  - λ_w = 1，λ_c = 0.1，λ_o = 0.1。这是 DisenBooth 的量级，固定，不扫参。
- **推理：** 只用 c_id，不接 c_irr。语速完全交给已有的块语速 token（`--speed-policy block`）。
- **可训练部分：** Talker 注意力上的 LoRA（rank 16，α 32）、m、h_ψ。BiCodec 解码器与 C 编码器冻结。
- **训练长度：** 3000 步，lr 1e-4，余弦，batch 32；只存适配器和小模块。

## 对照（同数据、同步数、同随机数）
- **T-ft：** 同一个 LoRA 只用 NLL(x | E(g)) 训练（没有 m、c_irr、λ 项）。用于分清"微调本身"和"解耦"各自的贡献。
- 另外两个参照，不训练：V1（C + T1c 原版，第三波已有）、B（原始 token + T1c，S1 已有）。

## 数据
- 由执行者在看任何数字之前确定：沿用 T1c 训练所用的序列格式和数据管线，只取开源部分（CoVoST 2 train、LibriSpeech、AISHELL-3、Common Voice），排除留出说话人和测试集。
- 实现细节追加在本文件「实现细节」一节，必须写在任何评测数字之前。

## 评测
- 同 S1 / 第三波 V 组：S1 的 LA2 文本，si_infer `--speed-policy block`，4 个 shard，种子 20261021。
- 两个方向，数据 (a) 与 (b) 的无串音子集，干净和带噪都跑。
- 指标：PESQ、ASR-BLEU、说话人相似度、D2 LAAL（主）、D1。
- 另做时长变异检验：同一条目、同一文本、同一种子，在四种音色条件下（原始 token、C、V2 的 token、干净提示的 C token）生成，计算时长变异系数。

## 门槛（解耦臂 vs T-ft，同时对 V1 报告）
1. 四种音色条件之间的时长变异系数下降 ≥ 50%（配对中位，CI 上界 < 0）。
2. 带噪 D2 LAAL：均值差 ≤ 0，且不劣于 V1 超过 0.02 s。
3. 带噪 PESQ 中位差 ≥ −0.05（对 T-ft 和 V1）。
4. ASR-BLEU 差的点估计 ≥ −0.5，且 CI 上界 ≥ 0（对 T-ft 和 V1）。
5. 相似度中位差 ≥ −0.02（对 V1）。

## 实现细节（执行者在任何评测数字之前追加）
