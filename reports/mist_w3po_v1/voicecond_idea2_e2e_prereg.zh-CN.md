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

---

## v1 定稿（2026-11-19 03:44 布拉格；任何训练之前；只补充数据统计和实现细节，损失、权重规则、设置和评测不变）

**数据（已建成，`data/open_voicecond_v1/idea2/cvt/`，`idea2_build_cvt.py`）：**
- 来源：CoVoST 2 的**非测试**划分（zh-CN→en 的 train + validation；en→zh-CN 的 train 与 validation），从未读取 test 划分；`HELDOUT_SPEECH.txt` 的说话人全部排除。
  - zh 源：按 `int(sha1(client_id),16) % 10` 划分，0–7 train、8–9 dev：84 / 20 个说话人，14.2 / 4.1 h 语音；
  - en 源：train 划分中 ≥ 8 条的说话人按 sha1 取前 700 个（每人 ≤ 20 条，实得 697 人、16.2 h）；validation 中 ≥ 5 条的说话人取 80 个（每人 ≤ 5 条，0.7 h）作 dev。
- 多人流：与 v1 合成配方相同（背景用 train / dev 部分，留出背景已排除），train 320 条（zh 120、en 200，39.2 h），dev 24 条（3.1 h），共 42.3 h（计划约 50 h；zh 源只有 104 个非测试说话人，不再加大）。
- 教师条目：未截断、≥ 1.5 s 的段，按 sha1 排序，每种语言 train 4000、dev 400；条目音频是带背景的混合段（与部署时 Talker 看到的源一致），文本是 CoVoST 人工译文，单块（final）。

**教师目标（`scripts/idea2_teacher_chain.sh`，后台等待 M2 测试运行全部结束才用卡 4–7）：**
1. cvt 流的预处理（分离器、HDemucs、2 s 窗口、$v$、$\tilde v$），与主数据相同；
2. cvt 说话人的 $u^\star$ 和 $g^\star$（干净 6 s 提示的中心）；
3. 冻结 T1c（与 T1c talk 作业相同的参数：`--speed-policy fixed --history-ban 8 --max-model-len 32768`，默认种子 20261021），音色 token 用 `idea1_si_infer_vo` 覆盖为 $g^\star$，`--dump-seq --dump-codes` 保存完整输入序列和语义码，作为两个 KL 项的教师强制序列。

**训练不启动，等协调者在测试表出来之后确认。**

---

## 测试补充（2026-11-19 08:35 布拉格；main_v2 训练结束之前写定，只追加）

训练与修复的记录见 `LOG.zh-CN.md`（main_v1 的 NaN 来自 cuDNN SDPA 反向，main_v2 关闭了这个后端并加了非有限步保护；损失、权重规则和设置都不变）。第 1000 步的 §8 检查：dev KL1 相对第 0 步下降 44%（zh2en）和 49%（en2zh），远高于 10%，主设置不改用软嵌入。

**被测对象：** 只测 main_v2 第 3000 步的最终检查点（`ckpt_step03000.pt`），不看中间检查点，也不在测试集上做任何选择或调参。dev 曲线只作记录。

**推理接口（与 M2 的 shrink+Vfb 完全相同，只换两处）：**
1. 音色 token：用训练后的编码器、λ / log r 头和 c_lang 跑闭式收缩后验（R、Σ0、μ0、b_M 固定为完整拟合 sha1 25aaf6cd 的值），窗口、v~、槽位、π 和 t_commit 与 M2 相同（直接读 M2 的特征缓存，只用训练后的编码器重算窗口编码 z）；û = ŝ + b_M 经 FSQ 得到 32 个 token。没有可用窗口的条目按 M2 的规则走 V 回退，直接复用 M2 token 表里这些条目的 ids。
2. Talker：把训练得到的 LoRA 以 fp32 合并进 T1c 的一份副本，再存成 bf16（`idea2_e2e_main_v2_export/talker_s03000/uniss`，原 T1c 不动）；只把 talk 作业的 `--model` 换成这个副本。
3. 其余全部不变：B 的文本（`--speech-from`）、`--talker t1c`、`--speed-policy fixed --history-ban 8`、两个 TTS 种子、`SI_KV_CACHE_GIB=7.73`、`ORCH_TALK_PAIR_GAP_S=0`、`ORCH_LEDGER_VLLM_GAP_S=10`、`--cap135`、`--exclude 0 1 2 3`（卡 0–3 在跑消融）。

**测试集与指标：** 与 M2 的 shrink+Vfb 表相同的 9 组。
- OpenBroadcast-SI 的 ID-2、OOD-1、clean，两个方向，经 `voicecond_bench_orch` 跑、`bench_eval` 评；
- OOD-3 的 ami、en、ali 三个子集，经 `idea2_ood3_orch` 跑，指标为块协议语音 LAAL、`idea1_speech_metrics`、TitaNet 相似度。
- 运行名：`obsi_e2e_<cond>_<dir>__t1c{,_tts2}`、`idea2_ood3_e2e_<ss>__t1c{,_tts2}`。

**比较：** e2e 对 B，以及 e2e 对冻结的 shrink+Vfb（M2 的 `obsi_i2_*` / `idea2_ood3_sv_*`），都用两个种子。统计方法与 `idea2_test_summary` 相同：两种子均值；按流（OOD-3 按条目）先做种子平均再配对，报告中位差、bootstrap 95% CI 和 Wilcoxon 检验；PESQ 族做 Holm 校正。

**门槛（与 M2 相同）：**
- tcpASR-BLEU 不低于对照（两种子均值）；
- 相似度中位差不低于 −0.02；
- 语音 LAAL 按用户认可的规则 "BLEU 优先于 LAAL"：LAAL 不大于对照算通过；小幅超出不否决 BLEU 门槛，但必须单独报告超出量，不得隐去。

**冒烟测试只用 dev：** token 路径用 cvt dev 条目检查（初始化时与冻结方法一致；用检查点时与训练日志的 dev 编码误差和 token 恢复率一致）；合并后的 Talker 在 cvt dev 条目上检查能否加载并生成，并与 PEFT 前向的 logits 对比。不在任何测试条目上做冒烟。
