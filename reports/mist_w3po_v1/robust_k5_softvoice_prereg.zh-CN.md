# 鲁棒 K5：跳过硬量化的软音色条件（免训练，V1 连续码）预登记

2026-11-21 09:15 CET（布拉格）写，早于 K5 的任何 token、合理性检查或 Talker 数字。

## 1. 动机
- Oracle 诊断（`robust_oracle_voice_prereg.zh-CN.md`）：把带噪条目的 32 个 BiCodec 全局 token 换成干净孪生的 token，S1 dev 带噪相似度收回 B 噪声损失的 92–119%；V1 只收回 −12–63%（K3 14–52%）。
- K4 预检（`robust_k4_tokencls_prereg.zh-CN.md`）：V1 / K3 / K4 的连续码离干净码很近（解码器 d-vector 余弦约 0.93），但 FSQ 硬量化后 32 个位置只有约 20% 与干净 token 相同（V1 在**干净**输入上也只有 45–55%）。很多维度靠近 FSQ 量化边界，身份在硬量化这一步丢掉。
- K5 假设（免训练）：不做硬量化，直接用 V1 的连续码作说话人条件，能保住 V1 连续码里已有的身份信息，从而收回更多 B 的噪声损失。
- 与 voicecond 线（Idea 1 / K2 / K3 / K4）重叠：同一 V1 编码器、同一 S1 评测；用户已决定把 Talker 身份这条线并入鲁棒论文。不是蒸馏：没有训练，没有教师。

## 2. 全局 token 进入系统的两处（读 `si_infer.py` 与 BiCodec 代码）
1. **Talker LM 输入**：`seq = [TASK_S2S, BALANCE, lang, START_GLOBAL] + [GLOBAL_0 + g for g in voice] + [END_GLOBAL] + …`（GLOBAL_0 = 151665，START_GLOBAL = 180387），即位置 4–35 是 32 个普通词表 token，经 Qwen2 Talker 的 `model.embed_tokens`（180407 × 1536，bf16）查表；vLLM 以 `prompt_token_ids` 输入。
2. **BiCodec 解码器**：`front.decode([*voice, *codes])` → `BiCodecTokenizer.decode_tokens_to_audio` → `BiCodec.detokenize`：`d = speaker_encoder.detokenize(ids)` = `project(flatten(ResidualFSQ.project_out(q)))`，q = FSQ 归一化码 = level / 2 ∈ {−1, −0.5, 0, 0.5}^{32×6}（6 维 × 4 级，单层 FSQ，scale 1）；然后 `x = prenet(z_q, d) + d`，`wav = decoder(x)`。d 对 q 是仿射的。
- FSQ：u = project_in(Perceiver 输出) ∈ R^{32×6}，b = bound(u) ∈ (−2.0015, 1.0015)，token 的第 k 维等级 ℓ_k = round(b_k) + 2，token id = Σ_k ℓ_k · 4^k。

## 3. 方法（一个设置；两条通道用同一个软分配）
- V1 连续码：对 V1 表（`wave3_combos/V/tokens/tokens_v1_<d>.jsonl`）每一行同一提示文件（sha1 校验），b = bound(φ_V1(y))，与 V1 token 完全同一计算，只是不做 round。
- **软分配（唯一选择，写在数字之前，不扫）：逐维精确重心插值（多线性插值）。** 把 b 截到 [−2, 1]，每维在两个最近等级之间按线性重心权重分配：lo = min(floor(b_k), 0)，w(lo) = 1 − (b_k − lo)，w(lo + 1) = b_k − lo；32 个位置各自对 4^6 网格取 6 维权重的乘积分布 p(t)（最多 64 个非零角点）。理由：无温度超参；b 正好在等级中心时 p 是硬 token 的 one-hot（精确复现 V1）；最大权重角点就是 FSQ round 的 token；并且对解码器这条仿射通道，E_p[d(t)] 恰好等于 d 在连续码上的值，两条通道是同一个分布下的期望。
- **(1) 解码器（K5-dec 通道）**：d = project(flatten(project_out(b / 2)))，即 `speaker_encoder.detokenize` 去掉 round（= 硬 token d-vector 在 p 下的期望）；替换 `decode_tokens_to_audio` 里的 d，语义码部分不变。
- **(2) Talker LM（K5-lm 通道）**：位置 4–35 的输入嵌入换成 e_k = Σ_t p_k(t) · E[GLOBAL_0 + t]（fp32 求和后转 bf16），其余位置的嵌入 = 原 token 查表；vLLM 以 `prompt_embeds`（`enable_prompt_embeds=True`）输入。条目对应：请求的采样种子 = seed + 1000·n_it + r，按 n_it 对应到 front-end 里第 n_it 次 `encode_wav_to_tokens` 的条目，并校验请求里的 32 个 id 等于该条目的 V1 id（不符即报错）。
- 钩子：`experiments/mist_w3po_v1/voicecond/k5_si_infer_soft.py`（沿用 `idea1_si_infer_vo.py` 的 override 模式：替换 `BiCodecTokenizer.encode_wav_to_tokens` 为 V1 表查表 + sha1 校验；按开关再替换 vLLM `LLM` 与 `decode_tokens_to_audio`），si_infer 本身不改。
- 三个子臂（一个方法的消融，不是扫参）：**K5-dec**（只有解码器用连续码，Talker 用 V1 硬 token）、**K5-lm**（只有 Talker 用软嵌入，解码器用 V1 硬 token）、**K5-full**（两处都用）。

## 4. 合理性检查（Talker 主评测之前，硬门槛）
- token 步骤：用 V1 硬 token 的等级代入连续通道，d-vector 必须与 `speaker_encoder.detokenize(ids)` 一致（最大绝对差 ≤ 1e−5），否则停止。另记录（只记录）：Q(b) 与 V1 表 id 不一致的行数；带噪条目上 d(b) 与 d(oracle token) 的余弦对比 d(V1 token) 与 d(oracle token) 的余弦。
- 端到端：两方向各取 `--limit 48`（均匀抽样，单分片）条目，跑 (i) **K5-full 合理性模式**（b 换成 V1 硬 token 的等级中心，软嵌入与 d-vector 都退化为 one-hot），(ii) 同一子集、同一分片的普通 token 通道对照（`idea1_si_infer_vo.py` + V1 表）。
- 门槛：(i) − (ii) 的 TitaNet 相似度按条配对中位数差在 ±0.003 以内（两方向都要）；另报语义码完全相同的条目比例，以及 (i) 对原 V1 整跑同一条目的配对中位数差（只记录，原跑分片 / 批次不同，vLLM 非确定）。不过门槛就停止，写 `k5/FAILED_SANITY`。

## 5. 评测与判断规则（S1 dev，与 oracle_voice 完全相同）
- 底座 S1 单说话人 LA2（`wave2_S1`），同一 Thinker 文本块，T1c Talker（`si_t1c_20261021T181440Z/export_1500`），种子 20261021，4 分片，提示文件 / sha1 同 V1 表。
- 复用 B、V1、oracle 的已有输出与指标；新跑 K5-dec、K5-lm、K5-full。
- 指标：ASR-BLEU（speech-only，`asr_speech.jsonl`）、SQUIM PESQ 中位数、TitaNet 相似度中位数（对干净源）；点估计。
- 收回比例 = (臂带噪相似度中位数 − B 带噪中位数) / (B 干净中位数 − B 带噪中位数)，按格。
- **判定格：en2zh / zh2en × {a, b}（4 格）**；b_noxt 只报告。
- **判断规则（K5-full，四条都满足才算成立）：**
  1. 带噪相似度收回比例 4 格平均 ≥ V1 平均 + 20 个百分点；
  2. 干净相似度不低于 V1：4 格干净相似度中位数的平均，K5-full ≥ V1；
  3. ASR-BLEU：4 格 × {带噪, 干净} 每一格，K5-full ≥ V1 − 0.3；
  4. PESQ：4 格 × {带噪, 干净} 每一格，K5-full ≥ V1 − 0.1。
- 另报：PESQ 对 oracle / B 干净的差（V1 的 PESQ 是过平滑的"比干净更干净"）；K5-dec / K5-lm 同样数字，用来判断收益来自哪条通道（只解读，不入规则）。

## 6. 运行
- `experiments/mist_w3po_v1/voicecond/k5_orch.sh`（setsid nohup 脱离，fds 8/9 关闭；先 `bash -n` 与 `DRY_RUN=1`）：token（卡 0，`k5_tokens.py`）→ 合理性（卡 0，两方向 × 2 跑，各 1 分片）→ 门槛 → 主评测：卡 0 K5-full en2zh → K5-dec en2zh；卡 6 K5-full zh2en → K5-dec zh2en；卡 4 K5-lm en2zh；卡 5 K5-lm zh2en（每跑 4 分片，间隔 45 s）→ ASR → SQUIM / TitaNet → `k5_score.py` → `k5/DONE`（失败写 `k5/FAILED`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `k5_noplaceholder.sh`（POLL 1 s）。不碰 2726951 / 52045 / 1222453，不碰 K1 第 2 步（卡 1、2、3、7，torchrun 1602215）。
- 输出 `eval_outputs/mist_w3po_v1/k5/`，日志 `logs/mist_w3po_v1/k5/`。
