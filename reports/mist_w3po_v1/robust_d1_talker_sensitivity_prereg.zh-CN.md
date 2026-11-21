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
运行 09:56–10:28 CET（卡 4、5；K1 step 2 已在 09:51–09:55 之间写 DONE，等待器立即放行；占位按 PID 停，自带 watcher 另停 1 次，结束后卡还给 guard）。点估计，种子 20261021；`eval_outputs/mist_w3po_v1/d1/summary.json`，token 统计 `d1/tokens/tokens_stats.json`。无偏差、无扫参、无重跑。

### Token（GPU，与 D0 一致）
| 方向 | n | C0 与 oracle 表 clean 行一致 | C10 token 匹配 / d-vec cos | C45 token 匹配 / d-vec cos | C10 截断条数 |
|---|---|---|---|---|---|
| zh2en | 1006 | 100%（32 个全同 100%） | 0.586 / 0.982 | 0.291 / 0.928 | 206 |
| en2zh | 637 | 100% | 0.562 / 0.981 | 0.266 / 0.920 | 83 |

C10 截断 = prompt 区间已到条目末尾（短句），后移 10 ms 后少 10 ms，按预登记处理。

### S1 dev clean 条目（相似度 = TitaNet 对 clean 源中位数；PESQ = SQUIM 中位数）
| 格子 | n | 相似度 B / C0 / C10 / C45 | 降幅 C10（配对） | 降幅 C45（配对） | 参照差距 G（oracle−V1，带噪）/ G/2 | C45 降幅 / G | PESQ B / C0 / C10 / C45 | ASR-BLEU B / C0 / C10 / C45 |
|---|---|---|---|---|---|---|---|---|
| en2zh a | 300 | 0.472 / 0.468 / 0.485 / 0.439 | −0.017（−0.006） | **0.029**（0.023） | 0.046 / 0.023 | 62% | 3.41 / 3.45 / 3.51 / 3.20 | 31.73 / 32.04 / 31.60 / 31.29 |
| en2zh b | 337 | 0.429 / 0.429 / 0.424 / 0.414 | 0.005（−0.007） | 0.015（0.013） | 0.037 / 0.019 | 41% | 3.27 / 3.29 / 3.32 / 3.08 | 34.03 / 34.13 / 33.95 / 33.95 |
| zh2en a | 300 | 0.537 / 0.534 / 0.528 / 0.508 | 0.006（0.003） | 0.026（0.022） | 0.073 / 0.036 | 36% | 3.35 / 3.37 / 3.30 / 3.16 | 30.50 / 30.16 / 30.41 / 30.37 |
| zh2en b | 706 | 0.487 / 0.494 / 0.488 / 0.488 | 0.006（−0.001） | 0.006（0.006） | 0.055 / 0.028 | 11% | 3.32 / 3.32 / 3.28 / 3.14 | 25.27 / 25.34 / 25.49 / 25.59 |
| en2zh b_noxt（只报告） | 164 | 0.451 / 0.444 / 0.434 / 0.428 | 0.010（0.000） | 0.016（0.018） | 0.041 | 39% | 3.33 / 3.36 / 3.49 / 3.11 | 35.39 / 35.80 / 34.60 / 35.34 |
| zh2en b_noxt（只报告） | 298 | 0.514 / 0.519 / 0.530 / 0.524 | −0.011（−0.005） | −0.005（0.003） | 0.042 | – | 3.36 / 3.37 / 3.38 / 3.25 | 28.02 / 27.89 / 28.31 / 28.32 |

- 复现检查通过：C0 − B 按条配对中位数差 −0.0024 / +0.0005 / +0.0015 / −0.0025（判定格，均在 ±0.003 内）。但非配对的格中位数 C0 与 B 相差可到 0.0065（zh2en b），说明格中位数本身有约 ±0.006 的 vLLM 运行噪声，小于约 0.006 的降幅不可分辨。
- 规则：C10 为"中间"（≥ G/2 的 0/4 格；< 0.005 的 2/4 格），C45 为"中间"（≥ G/2 的 1/4 格；< 0.005 的 0/4 格）→ **verdict = intermediate**。

### 解读（按格）
1. **Talker 不怕 token 翻转本身。** C10 翻掉约 42% 的 token（d-vector cos 0.98），相似度降幅 −0.017 到 +0.006，配对降幅 −0.007 到 +0.003，都在运行噪声内；PESQ、ASR-BLEU 也不变。所以"Talker 按类别消费 token"不是瓶颈，**对多种平移做扰动集成（只为消除翻转）不会有收益**。
2. **身份损失随 d-vector 偏移增长。** C45（−45 dB 白噪声，听不出，d-vector cos 0.92–0.93，与去噪器 V1 / K3 / K4 同档）在 4 个判定格降 0.006–0.029（配对 0.006–0.023），PESQ 也降 0.17–0.25；相当于 V1 与 oracle 差距的 11–62%（平均约 37%），只有 en2zh a 达到半数。
3. **结论：去噪器剩余差距的约 1/3 可以用"φ0 对极小噪声敏感，把 d-vector 推到 0.93 档"解释，其余约 2/3 是去噪器码离 clean 孪生码更远的真实身份偏移。** 两部分都发生在 d-vector 的位置上，不在类别消费上。修法方向是"更好的身份信息"：让码在 d-vector / 解码器说话人嵌入空间里落到 clean 孪生附近（以 Talker 身份或解码器端嵌入为目标），并用噪声不变的 SV 锚点（TitaNet-large）约束；软条件（K5）只能去掉量化，去不掉 0.93 档偏移，这与 K5 只收回 37–38% 一致。
