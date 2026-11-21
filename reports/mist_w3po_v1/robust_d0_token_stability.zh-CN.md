# Robust D0：φ0 全局 token 稳定性诊断（只用 CPU）

2026-11-21 09:48 CET（布拉格时间）· 脚本 `experiments/mist_w3po_v1/voicecond/d0_token_stability.py` · 输出 `eval_outputs/mist_w3po_v1/d0/`
（`phi_<dir>.npz`、`sv_<dir>.npz`、`d0_summary.json`；`d0_summary.buggy_spk_by_meeting.json` 是第一版，误把 meeting 当成说话人，已作废，只保留不删）。

## 设置

- 数据：S1 dev（`eval_outputs/mist_w3po_v1/wave2_S1/data`），set a = CoVoST2 clean/noisy 配对，set b = OpenBroadcast-SI DEV 参考轮次（clean 与 id2 noisy 配对）。
- prompt 区间：与 Talker 一致（`oracle_voice_tokens.py`）：`[0, max(first write, 1 s))`，first write 取自该条目自己的 S1 Thinker blocks；每个视图都和 si_infer 的 prompt 文件一样经过一次 PCM_16 wav 读写。
- φ0：冻结的 BiCodec speaker encoder（`bicodec_phi.py` 那条路径：音量归一化 → 6 s 参考段 → mel → ECAPA → Perceiver → project_in → FSQ [4]*6）。
  记录的量：token id、bounded 值、d-vector（`detokenize`）、soft d-vector（不取整的 bounded/2 走同一个 affine 解码；用 round 后再算，与 hard d-vector 的误差为 0）、ECAPA x-vector、连续码 u。
- 一致性检查：CPU 上的 base token 与 GPU 上算的 oracle clean token 比较，token 匹配 99.89% / 99.88%，但 32 个 token 全同的 prompt 只有 96.6% / 96.1%：光是设备数值误差，就已经让约 4% 的 prompt 至少翻一个 token。
- 扰动（每个方向从 clean 条目中按种子抽 300 条）：(a) 起点后移 10 ms，长度不变；(b) 区间加长 0.2 s（条目要够长：235 / 252 条）；(c) 加白噪声，功率比活动语音低 45 dB（活动语音 = 能量在最响 20 ms 帧 30 dB 以内的帧）；(d) 同一说话人的另一条 clean 话语：set b，说话人 = meta `stream`（meeting × speaker），去掉 crosstalk（`ovl`）条目，zh2en 298 对，en2zh 119 对。异说话人的随机配对作为参照。
- 指标：token 位置匹配率（32 个位置）、逐维 level 匹配率（192 维，按每维 level 边际分布算的随机水平约为 0.34）、d-vector 余弦、参考侧落在 level 边界 ±0.1 以内的 FSQ 维比例（边界为 −1.5 / −0.5 / 0.5）。

## Q1 φ0 自身对几乎听不出的扰动是否稳定

| 方向 | 扰动 | n | token 匹配 | 32 个全同 | 维 level 匹配 | d-vec cos | 中位 \|Δbounded\| | 翻转维中原本靠近边界的比例 |
|---|---|---|---|---|---|---|---|---|
| zh2en | 起点 +10 ms | 300 | **0.576** | 0.000 | 0.908 | 0.981 | 0.044 | 0.61 |
| zh2en | 加长 0.2 s | 235 | **0.645** | 0.013 | 0.923 | 0.984 | 0.033 | 0.65 |
| zh2en | 白噪声 −45 dB | 300 | **0.288** | 0.000 | 0.773 | 0.928 | 0.099 | 0.32 |
| zh2en | 同说话人另一句 | 298 | **0.030** | 0.000 | 0.518 | 0.801 | 0.356 | – |
| zh2en | 异说话人（参照） | 298 | 0.004 | 0.000 | 0.341 | 0.499 | 0.762 | – |
| en2zh | 起点 +10 ms | 300 | **0.559** | 0.000 | 0.905 | 0.981 | 0.047 | 0.60 |
| en2zh | 加长 0.2 s | 252 | **0.619** | 0.008 | 0.919 | 0.984 | 0.036 | 0.63 |
| en2zh | 白噪声 −45 dB | 300 | **0.272** | 0.000 | 0.759 | 0.921 | 0.108 | 0.31 |
| en2zh | 同说话人另一句 | 119 | **0.043** | 0.000 | 0.553 | 0.832 | 0.307 | – |
| en2zh | 异说话人（参照） | 164 | 0.002 | 0.000 | 0.339 | 0.502 | 0.777 | – |

- 靠近 level 边界（±0.1）的 FSQ 维占 16.9% / 16.7%。一个 token 由 6 维组成，所以约 1 − 0.83⁶ ≈ 67% 的 token 至少含一个脆弱维。
- 同样的扰动下，连续表示几乎不动：起点 +10 ms 时 soft d-vector cos 0.995，ECAPA x 0.991，TitaNet 0.999，WavLM-sv 0.995；白噪声 −45 dB 时分别为 0.948 / 0.951 / 0.986 / 0.995（zh2en）。

## Q2 噪声不变性与说话人区分度（短 prompt）

twin cos 是 noisy prompt 与其 clean 孪生（同一区间）的余弦，取全部配对：zh2en 1006 对，en2zh 637 对。同 / 异说话人统计与 EER 只用 set b 的 clean、无 crosstalk prompt：zh2en 298 条 / 58 个说话人，en2zh 164 条 / 98 个说话人。EER(n→c) 用 noisy prompt 去比 clean 注册集，排除自己的孪生。NI = (twin cos − 异说话人 cos 均值) / (1 − 异说话人 cos 均值)；twin z = (twin cos − 异说话人均值) / 异说话人标准差。

| 方向 | 编码器 | twin cos | 同说话人 cos | 异说话人 cos | d′ | EER clean | EER n→c | NI | twin z |
|---|---|---|---|---|---|---|---|---|---|
| zh2en | φ0 d-vector | 0.837 | 0.794 | 0.499 | 2.88 | 7.5% | 12.5% | 0.68 | 2.95 |
| zh2en | φ0 soft d-vector | 0.863 | 0.822 | 0.532 | 2.89 | 7.3% | 12.0% | 0.71 | 2.94 |
| zh2en | φ0 ECAPA x | 0.820 | 0.727 | 0.375 | 2.84 | 7.9% | 13.9% | 0.71 | 3.20 |
| zh2en | **TitaNet-large** | 0.918 | 0.598 | 0.191 | **3.02** | **6.5%** | **7.9%** | **0.90** | **4.99** |
| zh2en | WavLM-base-plus-sv | 0.963 | 0.888 | 0.684 | 1.41 | 21.7% | 22.3% | 0.88 | 1.47 |
| en2zh | φ0 d-vector | 0.833 | 0.832 | 0.492 | 4.04 | 3.8% | 10.8% | 0.67 | 3.89 |
| en2zh | φ0 soft d-vector | 0.859 | 0.860 | 0.526 | 4.07 | 3.8% | 10.1% | 0.70 | 3.88 |
| en2zh | φ0 ECAPA x | 0.809 | 0.744 | 0.344 | 3.57 | 6.3% | 10.8% | 0.71 | 4.30 |
| en2zh | **TitaNet-large** | 0.899 | 0.577 | 0.064 | 3.87 | **3.8%** | **3.3%** | **0.89** | **8.60** |
| en2zh | WavLM-base-plus-sv | 0.972 | 0.883 | 0.614 | 1.66 | 12.5% | 16.5% | 0.93 | 1.90 |

- φ0 token 在 noisy 与 clean 孪生之间只匹配 10.6% / 11.7%，维 level 匹配 0.61。
- 有一个关键比较：φ0 的 noisy↔clean twin cos（0.84 / 0.83）**已经与同说话人换一句话（0.79 / 0.83）一样低**。也就是说，在 φ0 眼里，噪声把同一句话的位置移开的距离，相当于换了一句话。TitaNet 的 twin cos 则远高于它自己的同说话人水平（0.92 vs 0.60）。
- 离线没有 speechbrain ECAPA（两个 venv 都没装 speechbrain，HF 缓存里也没有这个模型）。`microsoft/wavlm-large` 不是 SV 模型，没有纳入。
- 注意：同说话人对来自同一场 meeting（信道相同），所以 EER 偏乐观；但各编码器之间的相对次序可以比较。

## Q3 prompt 长度：first-write 区间 vs 整句（φ0 只用前 6 s）

| 方向 | 区间长度 | n | token 匹配 | 32 个全同 | 维 level 匹配 | d-vec cos | TitaNet cos |
|---|---|---|---|---|---|---|---|
| zh2en | 全部（区间中位 3.0 s，整句中位 5.0 s） | 300 | 0.448 | 0.203 | 0.837 | 0.962 | 0.921 |
| zh2en | 2–3 s | 89 | 0.386 | 0.225 | 0.803 | 0.953 | 0.891 |
| zh2en | 3–4 s | 138 | 0.394 | 0.123 | 0.824 | 0.959 | 0.915 |
| zh2en | ≥4 s | 62 | 0.557 | 0.210 | 0.883 | 0.975 | 0.961 |
| en2zh | 全部（区间中位 2.67 s，整句中位 4.64 s） | 300 | 0.389 | 0.147 | 0.811 | 0.953 | 0.892 |
| en2zh | 2–3 s | 150 | 0.341 | 0.133 | 0.781 | 0.941 | 0.857 |
| en2zh | 3–4 s | 106 | 0.347 | 0.057 | 0.811 | 0.958 | 0.909 |
| en2zh | ≥4 s | 32 | 0.526 | 0.188 | 0.876 | 0.975 | 0.988 |

另有 11 / 12 条 1–2 s 的区间就等于整句（短句），结果完全相同，也计入了“全部”那一行。“32 个全同”一列比 Q1 高，原因是有些条目的区间已经接近整句。

## 解读

1. **主要瓶颈是 (i)：φ0 的 FSQ / token 化本身很脆弱。** 起点挪 10 ms，token 只剩 56–58% 匹配，300 条里没有一条 32 个 token 全同；加 −45 dB 白噪声后只剩 27–29%（d-vector cos 0.92–0.93）。可对照的是，连续的 soft d-vector / ECAPA / TitaNet 在同样扰动下都保持在 ≥0.95–0.999。同说话人换一句话只有 3–4% 的 token 相同（维匹配 0.52–0.55，随机水平 0.34），这印证了 CoBit 的报告：BiCodec global token 编码的是“这段录音”，不是“这个说话人”。
2. **(iii) 去噪器的“~20% token 匹配”不能说明它们失败了。** 去噪器的状态（d-vector cos ≈0.93、token ≈20%）与“给 clean prompt 加 −45 dB 白噪声”几乎完全重合（cos 0.92–0.93、token 27–29%）。V1 在 clean 输入上的 45–55%，也落在“挪 10 ms”（56–58%）与“改变区间长度”（39–45%）的区间里。凡是不能逐比特复现 clean 波形的处理，token 匹配率天然就在 20–60%。oracle 能拿回约 100%，原因是它喂进去的是同一区间、逐比特相同的 clean 波形，所以 token 匹配率不适合用来评价去噪器。
3. **(ii) 短 noisy prompt 里的说话人信息基本还在，但 φ0 的编码器丢掉了一部分。** 在 2.7–3 s 的 noisy prompt 上，TitaNet 的 EER(n→c) 只有 7.9% / 3.3%，与它的 clean EER（6.5% / 3.8%）接近；φ0 则从 7.5% / 3.8% 恶化到 12.5% / 10.8%，而且在 ECAPA x 这一层就已经恶化（NI 0.71 vs TitaNet 0.90）。可见信息损失主要发生在 φ0 的编码器上（ECAPA 段对噪声敏感，FSQ 段把小位移放大成 token 翻转），不在 prompt 本身。区间与整句之间 token 匹配只有 39–45%，长度效应也是同一种脆弱性（d-vector cos 仍有 0.95–0.96）。
4. **在这个 prompt 长度下，噪声不变性最好的强 SV 编码器是 TitaNet-large。** 它的 twin z 为 5.0 / 8.6（φ0 只有 2.9–3.9），EER(n→c) 最低，NI 0.89–0.90。WavLM-base-plus-sv 的原始 twin cos 和 NI 最高，但区分度差（EER 13–22%，d′ 1.4–1.7），余弦空间被压缩，不适合作锚点。
5. **建议。** (a) 不再用 token 位置匹配率评价去噪器或 voice 预测器，改看 soft d-vector / TitaNet 空间，或直接看 Talker 输出的 SIM。(b) 下一步做一个 GPU 小检查 D1：把 clean prompt 挪 10 ms 或加 −45 dB 白噪声后得到的 token 喂给 Talker，测输出的 TitaNet SIM。如果 SIM 掉得和去噪器一样多，说明问题出在 Talker 按类别消费 token 这一环，修法应是连续 / 软条件（K5 soft voice）或扰动集成（对多种平移求平均 bounded 再取整），而不是更强的去噪器。如果 SIM 基本不掉，说明 Talker 本身对 token 翻转鲁棒，去噪器的差距来自 0.93 这一档 d-vector 余弦所对应的真实身份偏移。(c) 噪声不变的身份锚点优先用 TitaNet-large。
