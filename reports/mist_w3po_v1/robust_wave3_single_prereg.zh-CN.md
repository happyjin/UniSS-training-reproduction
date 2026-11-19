# 鲁棒第三波：全部组合在单说话人底座上并行预验证 预登记

2026-11-19 20:55 布拉格，写于任何第三波数字之前，也写于任何 S1 结果之前。
用户："可以不等 s1，直接把目前所有组合先并行一起预先验证一遍吗？"

## 底座与共同规则
- 底座 = S1 的单说话人系统：QABC 论文里不含 QABC 方法的 LA2 基线（Δ = 1 s）。Thinker 是原始 Qwen3-Omni-30B-A3B-Instruct，Talker 是 T1c export_1500，`si_infer --text-policy file --speed-policy block`，其余参数用默认值。见 `robust_wave2_diag_prereg.zh-CN.md` 与 `voicecond/wave2_s1_run.sh`。
- 文本复用 S1 的 dev Thinker 输出：`eval_outputs/mist_w3po_v1/wave2_S1/thinker_<dir>/blocks_<dir>_w1.jsonl`。
- B 臂直接用 S1 的 `talker_<dir>_s<seed>`。B 与本波各臂用同一份文本、同一顺序、同样两个 TTS 种子（20261021、1），因此可以配对比较。
- 数据（全是开源 dev，不读任何测试集）：
  - (a) CoVoST 2 validation 干净 / 带噪对，每方向 300 句；
  - (b) OpenBroadcast-SI dev 切出的单说话人流。主结果只用**无串音的句子**（meta 中 `ovl = false`），全量作补充。
  - 数据都由 `wave2_s1_prep.py` 生成，在 `wave2_S1/data/` 下。
- 指标只用真实输出：
  - 语音的 SQUIM PESQ；
  - ASR-BLEU（论文的纯语音 ASR 协议，两个种子）；
  - 说话人相似度：输出语音与该说话人干净源语音的相似度，用 idea1_spksim 同一工具；
  - LAAL：D2 语音为主，D1 为补充。
  - 打分代码复用 `wave2_s1_score.py`。
  - 不用码误差。
- 统计：每个条目先对两个种子取平均，再配对。
  - 配对中位差（或语料级差）的 95% CI：2000 次 bootstrap，(a) 按句子重抽，(b) 按流重抽。
  - 按 方向 × 数据集 × 条件（带噪 / 干净孪生）分别报告。
- 每个方法只跑**一个**设置，不扫参。不通过就停止，不改设置重试。
- 不用蒸馏，不用 SEEN。GPU 只用卡 3–7（卡 0–2 归 S1）。
- 输出放在 `eval_outputs/mist_w3po_v1/wave3_combos/`，日志放在 `logs/mist_w3po_v1/wave3/`。

## V 组：音色条件（只换 si_infer 的 32 个 BiCodec 音色 token，其余与 B 完全相同）
si_infer 在每句第一次写出时，把"截至此刻听到的源音频"送进 `BiCodecTokenizer.encode_wav_to_tokens`。本组在同一位置替换这一步的结果（hook 方式同 `idea1_si_infer_vo.py` / `bench_talker_override.py`），每个替换都以该次调用的提示文件为准。

- **V1 Idea 1 C：** 去噪音色编码器 `eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt`（`idea1_tokens.load_encoder`），作用于同一个提示。
- **V2 Idea 2 冻结后验（shrink+Vfb）：**
  - 参数用 `voicecond_idea2/shrink_fit_all/full.npz`（sha1 25aaf6cd…）。
  - 单说话人只有一个槽位，π ≡ 1，不用说话人分离器。
  - 窗口规格与 Idea 2 相同（L = 2 s，hop 0.5 s，冻结编码器 + λ 头）。
  - 观测范围是因果的，只用本句第一次写出之前听到的音频：
    - (a) 只有本句；
    - (b) 同一条流（meta 的 `stream`）中更早的句子，加上本句已听到的部分，按时间顺序。
  - 首次写出时一个可用窗口都没有的条目，按原方法走 V 回退。
- **V3 Idea 6 C_post：**
  - 每个窗口用 C 编码（平铺到 6 s），做高斯共轭后验。
  - 先验 = 上面完整拟合的 μ0 + b_M、Σ0；观测噪声 R_C 用 `idea6_precheckA/R_C.npz`，窗口重叠的缩放与 Idea 6 预检 A 相同。
  - 观测范围同 V2。
- **门槛（每个臂对 B；条件为带噪，两个方向、两个数据集都要报告）：**
  - PESQ 配对中位差 > 0，且 CI 下界 > 0；
  - ASR-BLEU 差的点估计 ≥ −0.5，且 CI 上界 ≥ 0（不能显著变差）；
  - 相似度中位差 ≥ −0.02；
  - D2 LAAL 均值差 ≤ +0.10 s；
  - 干净孪生上：PESQ 中位差 ≥ −0.05，ASR-BLEU 不能显著变差。
- **V2、V3 的主判据**放在 (b) 的无串音子集，中→英为主（每条流中位 56 s）。
  - 另报告 V2、V3 对 V1 的 PESQ / 相似度配对差，说明累积是否比单个提示更好。
  - (a) 上每句只有约 5 s 语音，只用来看短语音时会不会变差。

## G 组：负条件引导 Talker（以 LAAL 为目标）
- 每一步的 logit 改为 l(g⁺) + w·[l(g⁺) − l(g⁻)]，w = 1，固定不调。
  - g⁺ = V1 的 C token；
  - g⁻ = 本句 3 s 间隙的纯背景（wave1 manifest 的 `gap`，即说话前的停顿），同样用 C 编码。
  - 只在数据集 (a) 上跑。
- 在 si_infer 的块协议内实现，每一步对 g⁺ 和 g⁻ 各做一次前向，共享文本和已说出的码。
  - 对照臂 = **同一实现、w = 0**（与 V1 在数学上相同），这样两者用的后端和随机数路径一致。
  - 如果用的后端与 si_infer 的 vLLM 不同，要先在 20 个 dev 条目上核对：w = 0 时该后端与 si_infer 的码长中位比在 [0.98, 1.02] 内。
- 门槛（对 w = 0 臂，带噪；也要报告干净）：
  - D2 LAAL 均值差 < 0，且 CI 上界 < 0；
  - ASR-BLEU 差的点估计 ≥ −0.5，且 CI 上界 ≥ 0；
  - PESQ 中位差 ≥ −0.05；
  - 相似度中位差 ≥ −0.02。

## T 组：Thinker 侧（换成底座的原始 Qwen3-Omni，句级预检，不在 LA2 流式中实现）
- 每句当作一个整段，按 si_thinker_stream 的提示和贪心解码一次读完。数据为 (a)。
- 每臂的设置与第一波 `robust_wave1_prereg.zh-CN.md` 完全相同，只把模型换成底座 Thinker：
  - **P2a** 噪声损失（干净 − 带噪的文本 BLEU）；
  - **K2** P2b 背景对比解码（w = (1 − λ)/λ，截断到 3，自适应可信截断 0.1）；
  - **K3** 间隙偏移校正；
  - **K7** 在线软子空间 Wiener（音频编码器第 16–31 层，ν − d − 1 = 50，δ = 0.1）；
  - **K8** λ 自适应观测叠加，以及纯 HDemucs、α = 0.5 两个参照臂。
- **全部跑**，不以 P2a 作为停止条件。门槛：
  - 英→中带噪的文本 BLEU 差：CI 下界 > 0；
  - 若 P2a 损失 ≥ 0.5，另要求收回 ≥ 40%；
  - 中→英带噪、以及两个方向的干净输入，都不降超过 0.2 BLEU。
- 先核对底座 Thinker 的音频编码器与第一波 B Thinker 的是否逐张量相同，并写明结果。若相同，K3、K7 的特征层统计可以复用第一波的，但仍然要在底座上重新解码。

## 不跑（写在任何数字之前）
- K1 / K4 / K7v：只作用于 Talker 侧，而底座 Talker 就是第一波用过的同一个 T1c，结论直接沿用：
  - K1：logit 不随噪声线性变化（R² 0.50）；
  - K4：语速轴能读出，但钳不住（时长变异只降 5.7%）；
  - K7v：投影为空，token 与 C 逐位相同。
- Idea 5、说话人分离抗噪：单说话人系统里不存在可利用的对象。

## 判定
各项独立判定。通过的组合，再在 (b) 全量（包括串音句）与另一种子组合上确认，然后决定是否进入正式测试（RealSI 等只作测试，不在这里使用）。
