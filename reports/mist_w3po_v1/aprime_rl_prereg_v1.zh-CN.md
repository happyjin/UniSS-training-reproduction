# A′rl：A′ 的读级块 talker 输入（预登记，2026-11-12 布拉格约 01:45；在任何 A′rl 数字之前提交）

- **提出者**：协调者（代用户），布拉格约 01:35。
- **依据**：t2rl 的结果（`thinker_crf_t_results_v1` 的"T2 读级块"一节）。T2 的 dev 解码只把同一次读、同一说话人的段合成一块，不重训，语音就比逐段高 1.05（P = 0.000），语音 LAAL 低 0.36 s。
- **要回答的问题**：同样的改动能不能直接改进主系统 A′，以及 A′ + PL-EM talker 的主系统候选。

## 设定（唯一）

- **文本**：A′ 的 dev 解码 `decode/dev_crf_v2p_s3000_finalpost_s8_c2g06`（c2g06 窗口，1.0 s 读）。用 `crf/merge_read_segments.py`（539ed495，未改）合成读级块，得到 `decode/dev_crf_v2p_s3000_finalpost_s8_c2g06_rl`。文本、写出时刻（write_ms）和文本分数都不变。
- **两个 talker**，各跑一遍，每个两个 TTS 种子，用 `thinker_t_eval_orch --speech-from`，与 t2rl 完全相同的语音链（prep → 8 个 talker 分片 → ASR → 打分 → `speech_latency_lit`）：
  1. **A′ 原来的 talker c1+b1+s2**（`--talker c1b1s2`）。输出 `dev_crf_v2p_s3000_finalpost_s8_c2g06__rl`（+ `_tts2`），编排器 `t_eval/aprl`。
  2. **PL-EM talker**，与 t2plem 相同的设置（`--talker plem`：PL-EM 导出 + `EVAL_COMMON` + `--backlog-token --select-k 2`，即 `talker_crf_results_v1` 的 A′ + PL-EM c1b1s2）。输出 `dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem`（+ `_tts2`），编排器 `t_eval/aprlplem`。
- **门槛**（`gate_2seed`，两种子合并的语音 tcpBLEU，配对 bootstrap；每个 talker 各算两份）：
  - gate_0 对 A′（c1+b1+s2）：`dev_crf_v2p_s3000_finalpost_s8_c2g06` 与 `_tts2`；
  - gate_1 对 A′ + PL-EM：`dev_crf_v2p_s3000_finalpost_s8_c2g06__plem_c1b1s2` 与 `_tts2`；
  - 通过 = 语音 ≥ 参考 − 0.2，且 P(差 ≤ 0) < 0.95。文本与参考相同（差 0），只看语音这一项。
- **报告**（每个 talker）：语音 all / ov / 1spk；每条流的语音 LAAL（all / ov / 1spk，主指标）、EndOffset、单通道语音 LAAL；单元终点分解里的排队和最后一块时长；语音-文本差距。
- **解读规则**：
  - 只有在语音过门槛、并且语音 LAAL 不高于参考时，才说读级块"改进"了该参考系统；
  - 语音过门槛但 LAAL 更高，或者 LAAL 更低但语音没过，就只在质量-延迟平面上并列报告。
- **不做**：不调任何参数（不扫合并规则、不扫 talker 标志），只用 dev，不碰测试集。

## 卡

- T4 训完后，与 T4 dev5 速查的解码一起叠在 4–7 号卡上（`--cap135`，每卡 ≤ 135 GiB，共享账本）：速查的 4 个解码都开始后再启动，排在 T1 存档评测的续跑之前（T1 等 A′rl 的 talker 全部开始后再续跑）。D1 roll-in 照常叠着跑。
- D1 训练准备好时（run_D1、R4 通过、GO 候选的全集评测都已完成），如果 A′rl 还没跑完：
  - 剩下不到约 40 分钟，D1 等它跑完。判定方法：两个编排器的 talker 分片都已开始，且最后一个开始已超过 10 分钟；或者 talker 都已完成。
  - 否则，A′rl 剩下的作业挪到 VE1 训完后的 0–3 号卡上：A′rl 暂时排除全部卡，按 PID 停掉它在 4–7 号卡上正在跑的作业（之后按重试重跑），等 VE1 的速查解码都开始后再放开 0–3 号卡。D1 照常开训。
