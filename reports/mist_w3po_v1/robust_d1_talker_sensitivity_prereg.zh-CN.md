# Robust D1：Talker 对"感知等价、但不逐比特相同"的全局 token 是否敏感（预登记）

2026-11-21 10:05 CET（布拉格）写，早于任何 D1 token 或 Talker 数字。前置：`robust_d0_token_stability.zh-CN.md`（D0）、`robust_oracle_voice_prereg.zh-CN.md`（oracle）。

## 1. 问题
- D0：冻结的 BiCodec 全局 tokenizer φ0 很脆弱。clean prompt 起点后移 10 ms，32 个 token 只剩约 57% 匹配（soft d-vector cos 0.995）；加 −45 dB 白噪声只剩约 28%（d-vector cos ≈0.93），与去噪器 V1 / K3 / K4 的 token 匹配、d-vector 余弦落在同一区间。oracle（同区间、逐比特相同的 clean 波形算出的 token）能收回约 100% 的噪声身份损失。
- D1 问题：把 clean prompt 做几乎听不出的扰动后算出的 token 喂给 Talker，输出身份会不会掉？也就是说，去噪器与 oracle 之间的差距，是来自 Talker 按类别消费 token（token 一翻转身份就变），还是来自 ≈0.93 这一档 d-vector 余弦所代表的真实身份偏移？

## 2. 设计（每臂一个设置，不扫参）
- 数据：S1 dev 的 **clean** 条目（`eval_outputs/mist_w3po_v1/wave2_S1/data`），zh2en 1006 条（a 300 / b 706），en2zh 637 条（a 300 / b 337）。noisy 条目不跑。
- prompt 区间与 B 相同：条目源音频前 L = max(首次写出, 1 s)，首次写出取自该条目自己的 S1 Thinker blocks。
- 三臂，只换 32 个全局 token（override 钩子 `idea1_si_infer_vo.py --voice-override`，与 oracle 完全相同：Talker LM 序列开头的全局 token 与最后 BiCodec 解码都用 override；key = `<id>_voice`，sha1 = B 自己的 prompt 文件，运行时校验）：
  - **C0**：φ0(该条目 prompt 文件本身)，即 B 自己的计算（GPU）。用作复现检查：与 B 的相似度按条配对中位数差应在 ±0.003 以内。
  - **C10**：φ0(源音频 [10 ms, 10 ms + L))，长度不变（条目不够长时截到条目末尾，记录条数）。
  - **C45**：φ0(源音频 [0, L) + 白噪声)，噪声功率比活动语音低 45 dB（活动语音 = 能量在最响 20 ms 帧 30 dB 以内的帧），种子 crc32("wn45|<id>")，与 D0 的 wn45 视图完全一致。
  - 每个扰动视图都和 si_infer 的 prompt 一样经过一次 PCM_16 wav 写出，再用 B 同一个 `BiCodecTokenizer.encode_wav_to_tokens`（GPU）编码。
- 记录：C10 / C45 与 C0 的 token 位置匹配率、32 个全同比例、d-vector 余弦（`detokenize`）；C0 与 oracle 表中 clean 行（卡 0 上算的 B token）的一致率；C10 被截断的条数。
- 其余与 oracle / wave3 V 组完全相同：同一 Thinker 文本块（`wave2_S1/thinker_<d>/blocks_<d>_w1.jsonl`），T1c Talker（`si_t1c_20261021T181440Z/autoext/export_1500`），种子 20261021，4 分片，`SI_KV_CACHE_GIB=7.73`，`--text-policy file --speed-policy block --window 1`。
- 指标：TitaNet 相似度中位数（对 clean 源，`wave3_v_spksim.py`，与 oracle 同一打分器）、SQUIM PESQ 中位数、ASR-BLEU（speech-only，`asr_speech.jsonl`）；点估计，每格 = 方向 × 集合 {a, b}，b_noxt 只报告。

## 3. 判断规则（固定）
- 每格**降幅** Δ_X = C0 相似度中位数 − X 相似度中位数（X ∈ {C10, C45}；另报按条配对中位数差）。
- 每格**参照差距** G = oracle 带噪相似度中位数 − V1 带噪相似度中位数（取自 `oracle_voice/summary.json`）：en2zh a 0.046、en2zh b 0.038、zh2en a 0.072、zh2en b 0.055；半数阈值 G/2 = 0.023 / 0.019 / 0.036 / 0.028。另报 oracle − K3 差距作对照。
- 每臂：Δ_X ≥ G/2 的格数 ≥ 3/4 → 该臂"大"；Δ_X < 0.005 的格数 ≥ 3/4 → 该臂"小"；其余为"中间"。
- **结论**：
  - C10 或 C45 为"大" → **Talker 按类别消费 token 是瓶颈**（感知等价的 token 就能造成与去噪器相当的身份损失）；修法方向：连续 / 软条件（如 K5 soft voice）或扰动集成（多种平移下平均 bounded 再取整）。若只有 C45 为"大"、C10 不大，则另注明"损失随 d-vector 偏移量增长，10 ms 级翻转本身无害"。
  - C10 与 C45 都"小" → Talker 对 token 翻转鲁棒，**去噪器的差距来自 ≈0.93 这一档 d-vector 偏移本身**；修法方向：更好的身份信息，如噪声鲁棒的 SV 锚点（D0 推荐 TitaNet-large）。
  - 其余 → "中间"，按格解读。
- C0 复现检查失败（任一判定格配对中位数差超出 ±0.003）时，降幅改用"相对 C0"并在结果中注明。
- PESQ、ASR-BLEU 只作记录。

## 4. 运行
- `experiments/mist_w3po_v1/voicecond/d1ts_orch.sh`（setsid nohup 脱离，fds 8/9 关闭）：先等待 K1 step 2 结束（`eval_outputs/mist_w3po_v1/k1/step2/DONE` 或 `FAILED` 出现，且没有 K1 step 2 进程仍在运行），只用卡 4、5；token（卡 4，`d1ts_tokens.py`）→ 每方向（en2zh 卡 4、zh2en 卡 5）依次跑 C0 / C10 / C45 的 Talker（4 分片），ASR / SQUIM / TitaNet 与下一臂的 Talker 重叠 → `d1ts_score.py` → `d1/DONE`（失败写 `FAILED`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `d1ts_noplaceholder.sh`（POLL 1 s）。不碰 2726951 / 52045 / 1222453、K5（卡 0、6）、K1（卡 1、2、3、7）及以后的 AoV / EITHER。
- 输出 `eval_outputs/mist_w3po_v1/d1/`，日志 `logs/mist_w3po_v1/d1/talker_sens/`（`logs/mist_w3po_v1/d1/` 里已有 11-11 的 dagger D1 日志，故放子目录）。

## 结果
（待运行后追加）
