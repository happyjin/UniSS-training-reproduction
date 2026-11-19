# Idea 2 端到端版本（方案 §8.3，按收缩模型改写）：预登记草案 v0 与实现计划

2026-11-19 03:30 布拉格时间。**草案**：在任何端到端训练之前提交；训练开始前会以 v1 定稿（只允许补充数据统计和实现细节，不改损失、权重规则和评测）。

- 依据：方案 §8.0、§8.3；Idea 2 预登记 v1–v3（`voicecond_idea2_prereg.zh-CN.md`）。
- 线性加性模型在 G2.0 未通过；收缩模型在先导和完整数据的 G2.2 都通过，并且在电竞 demo 上声音质量大幅提升。所以端到端版本的结构层用**收缩后验层**，不用原来的线性加性卡尔曼层。
- 本文件同时是实现计划（§5–§7）。

---

## 1. 定义

"端到端"：生成损失的梯度从 Talker 的输出，经过 Talker（LoRA）、收缩后验层、窗口编码器和 λ 头，一直传到窗口音频；这些部分联合训练。Thinker、分离器（Nemotron）、HDemucs、BiCodec 解码器冻结。论文表述为 "end-to-end trainable speaker conditioning and Talker"。

## 2. 前向

对说话人槽位 $j$ 的每个窗口 $k$（L = 2 s，跳步 0.5 s，与推理相同的因果规则）：

1. 窗口编码 $z_k=\phi_\theta(\text{窗口})$：BiCodec 说话人编码器（ECAPA + Perceiver + project_in）全参数微调（BN 和 dropout 用 eval 模式），从冻结权重初始化；
2. 收缩权重 $\lambda_k=\sigma(h_\omega(\tilde v_k))$，$h_\omega$ 为 5→32→1 的 MLP，初始化为当前的 $\alpha+\beta^\top\tilde v$（第一层取恒等式的线性部分，其余接近 0）；另一个输出头给每个窗口的对数方差尺度 $\log r_k$（初值 0），即 $R_k=e^{r_k}R$；
3. 收缩后验层（闭式、可微）：
$$\hat s_j=\Big(\Sigma_0^{-1}+\sum_k\pi_k\lambda_k^2e^{-r_k}R_f^{-1}\Big)^{-1}\Big(\Sigma_0^{-1}\mu_0+R_f^{-1}\sum_k\pi_k\lambda_ke^{-r_k}(z_k-(1-\lambda_k)c_{\rm lang})\Big),\quad \hat u_j=\hat s_j+b_M$$
   $c_{\rm lang}$ 可训练，初值为完整拟合（sha1 25aaf6cd）；$R,\Sigma_0,\mu_0,b_M$ 固定。
4. 量化与桥接：$g=\mathcal Q(\hat u)$；Talker 的 32 个 global token 嵌入用直通估计
$$e=E[g]+\big(q_c-\mathrm{sg}(q_c)\big)W^\top,\qquad q_c=\mathcal B(\hat u)/2$$
   $W$（1536×6）是 global token 嵌入行对其 FSQ 码的最小二乘线性映射（可行性检查：$R^2=0.67$）。
5. Talker：T1c（Qwen2 1.5B），LoRA r = 16、α = 32，作用于 q_proj、v_proj（218 万参数）；推理接口与现在完全相同（只换 32 个 token）。

## 3. 损失（每个设置只跑一次，不扫参）

$$\mathcal L=\underbrace{\mathrm{KL}\big(p_{0}(\cdot\mid g^\star)\,\|\,p_{\psi}(\cdot\mid g(\hat u))\big)}_{\text{特权信息生成监督}}
+\underbrace{\mathrm{KL}\big(p_{0}(\cdot\mid g^\star)\,\|\,p_{\psi}(\cdot\mid g^\star)\big)}_{\text{干净条件无害}}
+w_{\rm code}\|\hat u-u^\star\|^2+w_{\rm anc}\,\mathbb E_{\rm clean}\|\phi_\theta(s)-\phi_0(s)\|^2$$

- 两个 KL 都在语义码位置上逐位计算（教师强制，目标序列是冻结 T1c 在 $g^\star$ 下生成并缓存的语义码），只在语义码子词表（8192）上归一。
- $p_0$ 是冻结 T1c，$g^\star$ 是该说话人干净 6 s 提示中心的 token（特权信息，只在训练时有）。
- **权重规则（事先固定）：** $w_{\rm code}$、$w_{\rm anc}$ 取各自在前 50 个 batch 上均值的倒数乘以第一项 KL 的前 50 个 batch 均值（使三项初始量级相同），之后固定。
- 不用似然项（收缩模型没有隐状态，原 §3.4 的卡尔曼边缘似然不适用）；CN-Celeb 只用于锚点项的真实带噪干净段之外的无害检查（记录，不参与损失）。

## 4. 消融（与主设置同数据、同步数，各跑一次）

1. 黑盒注意力池化代替收缩后验层（窗口编码经 2 层注意力池化直接回归 $\hat u$）；
2. 收缩层但不用 $\tilde v$（$\lambda$ 为可训练常数）；
3. 只训练条件模块、Talker 冻结（没有 LoRA）；
4. 完整版。

预期：4 优于 1（结构化层的价值）、4 优于 3（Talker 联合训练的价值）；2 与 4 的差反映 $\tilde v$ 的作用（目前的线性版里几乎为 0）。

## 5. 数据（全部开源；测试集一律不用）

- **条件模块与编码误差项：** Idea 2 现有 100 h 合成流的 train 划分（窗口、$\tilde v$、分离器槽位都已算好），$u^\star$ 已有。
- **Talker 的内容上下文（源 GLM 与目标文本）：** 合成流里的语料（LibriSpeech、AISHELL-3、Common Voice）没有译文。用 **CoVoST 2 的 train 划分**（zh-CN→en 约 7 千条，en→zh-CN 取约 2 万条子集；Common Voice 说话人，与 test 划分不重叠；再排除 `HELDOUT_SPEECH.txt` 的全部说话人）按同一配方（§1 of v1，背景用 train 部分，排除留出背景）合成约 50 h 的新多人流 `cvt_*`，自带原文、译文和真实说话人。
- **教师目标：** 冻结 T1c 在 $g^\star$ 下对每个训练段生成语义码（vLLM，种子固定），缓存。
- **dev：** `cvt` 的 dev 划分（说话人不重叠）与 Idea 2 合成 dev，只用于记录曲线，不做选择。

## 6. 训练设置（固定）

- 4 卡 DDP（卡 4–7），每步 64 个序列（每卡 16），bf16，梯度检查点；AdamW，LoRA 和 $h_\omega$、$c$ 学习率 1e-4，编码器 5e-5，warm-up 200 步，余弦到 0，共 3000 步；梯度裁剪 1.0；种子 20261119。
- 只保存最终检查点；每 500 步记录 dev 的两个 KL、编码误差和 token 恢复率（不选择）。

## 7. 可行性检查（已完成，`eval_outputs/mist_w3po_v1/voicecond_idea2/e2e/feasibility_v1.json`）

只检查形状、梯度通路和成本；用 4 个合成 train 说话人，GLM、文本和语义码是随机 id。

| 项目 | 结果 |
|---|---|
| global token 嵌入对 FSQ 码的线性替代 $R^2$ | 0.67 |
| 初始 KL（$g^\star$ 对 $g(\hat u)$，随机上下文） | 0.29–0.32 nats / 语义位置 |
| 梯度 | 到达 $\alpha$、$\beta$、$c$ 和 LoRA（均非零） |
| 单卡 16 × 578 token 一步（前向教师 + 前向反向学生） | 0.88 s，峰值 38.5 GiB |

**成本估计：** 训练 3000 步 × 4 卡约 1 h；教师目标生成约 2–3 h（4 卡 vLLM）；`cvt` 合成与预处理约 1.5 h。

## 8. 风险

- 直通桥的线性替代只有 $R^2=0.67$：梯度方向是近似的。若 dev 的第一项 KL 在 1000 步内下降不到 10%，按事先规则改用 4 档 softmax 软嵌入（每个 FSQ 维度在 4 个档位上的 softmax 混合，嵌入为 4096 行的精确混合）作为主设置重跑一次，并在报告中注明。
- 编码器崩塌迎合 KL：锚点项和无害 KL 项限制；监控干净输入的编码偏移。
- Talker LoRA 改变干净条件下的输出：无害 KL 项；测试时 AMI / AliMeeting 无害检验。

## 9. 评测（训练完成后另写测试预登记，与 v3 同一套测试集和门槛）

- ID-1：合成 test 上的编码误差和 token 恢复率；
- 生成语音：电竞 demo（只评测）、OpenBroadcast-SI 的 ID-2 / OOD-1 / clean、OOD-3 子集；对照 baseline B 和冻结的 shrink+Vfb。
