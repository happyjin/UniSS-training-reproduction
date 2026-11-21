# 鲁棒诊断：全局音色码的"干净孪生"oracle 预登记

2026-11-21 08:05 CET（布拉格）写，早于任何 oracle token 或 Talker 数字。

## 1. 问题
- V1（去噪音色编码器，替换喂给 T1c Talker 的 32 个 BiCodec 全局音色 token）在带噪输入上只收回一部分说话人相似度损失；K3 把码散布从 0.72 恢复到 0.90，Talker 相似度却只多 +0.006–0.011（`robust_k3_discrim_prereg.zh-CN.md`）。
- 诊断问题：剩下的身份损失到底在不在全局音色 token 里？

## 2. Talker 用到的、由源音频导出的输入（读 `experiments/uniss_qwen3_omni_v1/evaluation/si_infer.py`，本次默认路径：无 --voice-ema / --voice-bayes / --voice-context-ms / voice_audio）
1. **32 个 BiCodec 全局 token**：`it["voice"] = encode_wav_to_tokens(tmp/<id>_voice.wav)[:32]`，提示 = 该条源音频的前 max(首次写出, 1 s)。它进两处：Talker 序列开头 `[TASK_S2S, BALANCE, lang, START_GLOBAL, GLOBAL_0+g×32, END_GLOBAL]`，以及最后的 BiCodec 解码 `front.decode([*voice, *codes])`。override 钩子替换 `encode_wav_to_tokens`，所以两处都被替换。
2. **GLM-4 语音 token（另一条通道）**：每个块写出时，把**源音频截至该写出时刻的全部前缀**（不只是提示段）重新用 GLM-4 语音分词器分词（`front.glm4.tokenize(heard)`），按 holdback 截断后以 `GLM_0 + t` 交错放在每个块的文本前面。带噪条目里这一串来自带噪源音频，噪声可以由此进入 Talker。
3. 其余输入是文本与控制 token：Thinker 文本块（带噪条目用的是带噪条件下的 Thinker 输出）、语速 token、STREAMING_MODE / START_SEMANTIC 等，以及前面块已生成的语义码。
4. **没有**提示段的语义 / 声学 token 续写，也没有把提示音频本身作为 Talker 的前缀；BiCodec 解码器只收 32 个全局 token 和生成的语义码。

## 3. Oracle 设计（一个设置）
- 只改带噪条目的 32 个全局 token：换成**干净孪生条目**（同一 set、同一 `pair`、cond = clean）的源音频，在**带噪条目自己的提示段**（前 max(带噪条目首次写出, 1 s)，与带噪条目同一读取 / 截取 / `sf.write` 方式）上，用 B 同一个 `BiCodecTokenizer.encode_wav_to_tokens` 算出的 token，即"全局码的完美去噪器"。
- 干净条目：token = B 自己的计算（同一提示文件），用来检查 override 路径能复现 B 干净结果。
- override 表沿用 wave3 V 表的 key 与 sha1（带噪条目自己的提示文件，钩子运行时校验 sha1），只换 ids。
- 记录：带噪提示与干净孪生提示的波形相关（检查对齐）、oracle 与 B 带噪 token 的平均 Hamming 距离、提示段长度与干净孪生自身提示段不同的条目数。
- 其余与 wave3 V 组 / K3 完全相同：S1 dev（`wave2_S1/data`），同一 Thinker 文本块，T1c Talker（`si_t1c_20261021T181440Z/export_1500`），种子 20261021，4 分片，`idea1_si_infer_vo.py --voice-override`；ASR-BLEU（speech-only，`asr_speech.jsonl`）、SQUIM PESQ 中位数、TitaNet 相似度中位数（对干净源，`wave3_v_spksim.py`）；点估计。

## 4. 判断规则（固定）
- 每格（方向 × 集合 {a, b}）的 **B 噪声损失** L = B 干净相似度中位数 − B 带噪相似度中位数；
  某臂的**收回比例** = (该臂带噪相似度中位数 − B 带噪相似度中位数) / L。对 V1、K3、oracle 都报。
- **若 oracle 在 4 个判定格的收回比例都 ≥ 70%**（带噪相似度接近干净输入相似度）：剩余损失位于全局码，码级方法（K2 / K3）还有空间。
- **若 oracle 在多数（≥ 3/4）判定格收回 < 70%**：剩余损失经另一条通道（第 2 节的 GLM-4 源 token 或带噪 Thinker 文本）进入，码级方法应停止。
- 介于两者之间（2/4）：报告为"混合"，按格解读。b_noxt 只报告。
- 另报 PESQ、ASR-BLEU，只作记录（oracle 不是候选方法）。

## 5. 运行
- `experiments/mist_w3po_v1/voicecond/oracle_voice_orch.sh`（setsid nohup 脱离，fds 8/9 关闭）：token（卡 0，`oracle_voice_tokens.py`）→ Talker（en2zh 4 分片卡 0、zh2en 4 分片卡 6，间隔 45 s）→ ASR → SQUIM / TitaNet → `oracle_voice_score.py` → `oracle_voice/DONE`（失败写 `FAILED`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `oracle_voice_noplaceholder.sh`（POLL 1 s）。不碰 2726951 / 52045 / 1222453、K1 训练（卡 1、2、3、7）和 K2 评测（卡 4、5）。
- 复用：B = `wave2_S1/talker_<d>_s20261021`，V1 = `wave3_combos/V/v1/talker_<d>`，K3 = `k3/talker_<d>` 及各自指标。只新跑 oracle。
- 输出 `eval_outputs/mist_w3po_v1/oracle_voice/`，日志 `logs/mist_w3po_v1/oracle_voice/`。

## 结果
（待运行后追加）
