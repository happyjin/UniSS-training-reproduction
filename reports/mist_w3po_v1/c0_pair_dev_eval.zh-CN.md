# C0 / C0-fp dev 评测（step_03000，最终读因果 P 协议）

2026-11-10 05:41–07:44 UTC。只做推理和打分，没有训练。数据只用 dev 在线窗口：3,050 个窗口，15 个会议。协议与 `finalpost_dev_test.zh-CN.md` §1–§3 完全相同。

## 结论

主对照是 v8 + 最终读因果 P（`dev_v8_ctx0_guard_finalpost_s8`）：

- **C0-fp：正式闸门通过，三项点估计都为正。**
  - 文本 all +0.25，ov +0.05；
  - 语音 +0.74（两种子合并），不显著，但没有下降。
- **C0：按闸门规则三项都通过，但语音点估计为 −0.20。**
  - 正好落在 −0.2 容差边上，P(diff ≤ 0) = 0.62，不显著。
  - 若按"语音 ASR-BLEU 不得下降"的硬要求从严判定，C0 不满足。
  - 文本 all +0.33，P = 0.037。
- **对照官方 v8 参考**（次要）：
  - 两者文本 all 都显著提升，约 +1.0；
  - 语音 C0 −0.10，C0-fp +0.85（P = 0.048）。
- **C0-fp vs C0**（次要）：
  - 文本持平：all −0.09，ov −0.17，都不显著；
  - 语音 +0.94，P(diff ≤ 0) = 0.032。

## 1. 运行

- **解码**用新包装 `crf/decode_crf_ckpt_finalpost.py`（提交 24f5e14f）。
  - 先装 `online_decode_finalpost` 的最终读 P 补丁，再由 `decode_crf_ckpt.main` 在内存中折叠 LoRA、说话人投影和 259 个新 token 行，最后调用 `online_decode.main`。两个原包装都没改。
  - CPU 单测 22 项全部通过（`test_decode_crf_ckpt_finalpost.py`）。
- **解码配置**与参考 `queue_v2/jobs/032_dev_v8_ctx0_guard_s8_00.json` 逐项相同，只多 `--crf-ckpt` 和 `--final-read-post close_detect`：
  - v8 merged Thinker，`windows_dev_online.jsonl`；
  - `--policy wait --lag-ms 2000 --context 0 --batch 16 --max-gib 120`，8 分片按会议分；
  - `--loop-guard-k 6 --loop-guard-maxp 12 --budget-tps 25 --budget-slack 50`；
  - 单一解码种子。
- **冒烟。** 05:41 在空卡 0 和 1 上各跑 `--limit 2`：
  - 日志显示 `adapters_folded 192`、`new_rows 259`，FINALPOST_STATS final = extended = 2；
  - 输出与 v8 finalpost 相近，合理。
- **语音**与 finalpost 测试同一流水线：prep、8 个 talker 分片、每分片 2 个 ASR 子分片、asr_merge、asr_hyp、score_run。种子 2 为 talker `--seed 20261109`（同 `mk_tts_seed2_jobs.py`）。
- **编排** `scripts/crf_pair_dev_orch.py`：
  - 导入 `finalpost_dev_orch` 的 Job/Orch，原文件没改。
  - 占用估计不变：ASR 17.5、talker 24、规划余量 12 GiB；每卡 ≤ 135 GiB。
  - 每次 GPU 启动持有 `gpu_launch.lock`；talker 每 GPU 对每 60 s 最多启动一次。
  - 两个候选的解码和语音在 8 卡上交叠运行。
- **输出目录**（`eval_outputs/mist_w3po_v1/decode/`）：
  - `dev_crf_c0_s3000_finalpost_s8{,_tts2}`；
  - `dev_crf_c0fp_s3000_finalpost_s8{,_tts2}`。

## 2. 正式闸门

规则：点估计 ≥ 参考 − 0.2，且按会议配对 bootstrap（10k）P(diff ≤ 0) < 0.95。三项都过才算通过。语音为两个 TTS 种子合并（`gate_2seed.py`）。

| 对比 | 文本 all | 文本 ov | 语音 all（2 种子） | 闸门 |
|---|---|---|---|---|
| **C0 vs v8+finalpost**（主） | 27.75 → 28.08，+0.33 [−0.03, 0.74]，P = 0.037 | 22.82 → 23.04，+0.22 [−0.55, 1.46]，P = 0.31 | 19.17 → 18.97，**−0.20** [−1.45, 1.08]，P = 0.62 | 规则通过；语音点估计下降 |
| **C0-fp vs v8+finalpost**（主） | 27.75 → 28.00，+0.25 [−0.09, 0.61]，P = 0.078 | 22.82 → 22.87，+0.05 [−0.69, 1.11]，P = 0.43 | 19.17 → 19.91，**+0.74** [−0.35, 1.85]，P = 0.094 | **通过** |
| C0 vs 官方 v8 参考 | 27.05 → 28.08，+1.03 [0.62, 1.46]，P = 0.000 | 21.94 → 23.04，+1.10 [0.31, 2.18]，P = 0.003 | 19.07 → 18.97，−0.10 [−1.19, 0.93]，P = 0.57 | 规则通过 |
| C0-fp vs 官方 v8 参考 | 27.05 → 28.00，+0.94 [0.62, 1.28]，P = 0.000 | 21.94 → 22.87，+0.93 [0.13, 1.89]，P = 0.011 | 19.07 → 19.91，+0.85 [−0.16, 1.75]，P = 0.048 | 通过 |
| C0-fp vs C0 | 28.08 → 28.00，−0.09 [−0.36, 0.20]，P = 0.73 | 23.04 → 22.87，−0.17 [−0.65, 0.47]，P = 0.73 | 18.97 → 19.91，+0.94 [−0.06, 1.98]，P = 0.032 | 通过 |

P 为 P(diff ≤ 0)。

- **语音分种子**（tcpBLEU all）：
  - C0：17.93 / 20.13；
  - C0-fp：20.04 / 19.78；
  - v8+finalpost：19.19 / 19.15；
  - 官方参考：18.63 / 19.52。
- **语音噪声。** C0 的两个种子相差 2.2，大于以往约 ±0.8 的单次噪声。所以 C0 的语音 −0.20 和 C0-fp 的 +0.74 都在噪声范围内。

## 3. 次要指标（`finalpost_compare.py`）

| 指标 | 官方参考 | v8+finalpost | C0 | C0-fp |
|---|---|---|---|---|
| tcpBLEU 1spk | 29.31 | 29.93 | 30.31 | 30.26 |
| 短单元召回 ≤0.5 s 1spk / ov | 0.579 / 0.712 | 0.766 / 0.741 | 0.775 / 0.732 | 0.766 / 0.741 |
| 短单元召回 ≤1 s 1spk / ov | 0.670 / 0.783 | 0.829 / 0.812 | 0.850 / 0.811 | 0.851 / 0.812 |
| ≤1 s 孤立：错标率 / 漏译率 | 18.0% / 32.3% | 6.7% / 17.4% | 4.7% / 15.2% | 4.7% / 15.1% |
| ≤1 s 重叠（ov50）：错标率 / 漏译率 | 7.4% / 21.0% | 5.9% / 18.8% | 6.5% / 19.2% | 5.6% / 19.1% |
| ≤0.5 s 孤立：错标率 / 漏译率 | 26.1% / 41.7% | 7.9% / 23.6% | 7.1% / 23.0% | 7.5% / 23.4% |
| collapse | 0.164 | 0.146 | 0.157 | 0.148 |
| CR | 0.830 | 0.849 | 0.838 | 0.845 |
| duplicate rate | 0.066 | 0.063 | 0.052 | 0.053 |
| 失控窗口 | 1 | 1 | 0 | 0 |
| 延迟：ov p50 / 平均滞后（s） | 1.690 / 1.779 | 1.674 / 1.769 | 1.745 / 1.802 | 1.740 / 1.800 |

- **短单元。** 相对 v8+finalpost，两个候选的 ≤1 s 孤立错标率都从 6.7% 降到 4.7%，漏译率 17.4% → 15.1–15.2%，≤1 s 1spk 召回 0.829 → 0.85。
  - 在重叠单元上，C0 的错标率略升（5.9% → 6.5%），C0-fp 略降（5.6%）。
- **重复和失控。** 重复率降低约 1 个百分点，失控窗口为 0。collapse 介于两个参考之间。
- **延迟略增。** ov p50 增加约 +0.07 s，平均滞后增加约 +0.03 s。
- **输出变化。** 对 v8+finalpost，C0 改变了 2,251 / 3,050 个窗口的输出，C0-fp 改变了 2,389 个。C0-fp 对 C0 改变了 2,082 个。

## 4. 运行时间（墙钟）

| 时间（UTC） | 事件 |
|---|---|
| 05:41 | 两个训练都写完 step_03000 并退出（卡 0–7 空） |
| 05:41–05:43 | 冒烟 2 窗口 × 2 候选，约 1 min |
| 05:43 | 编排启动，共 164 个作业 |
| 05:51 | C0-fp 解码开始（与 C0 解码和 C0 语音交叠） |
| 06:08 | C0 文本分数出来 |
| 06:37 | C0-fp 文本分数出来 |
| 07:19 / 07:40 | C0 两个种子的语音分数出来 |
| 07:44 | C0-fp 两个种子的语音分数出来；DONE（0 个失败 / 跳过） |

从训练结束算起共 2 h 03 min，其中正式编排 2 h 01 min。

## 5. 事件与处理

- **监督进程。** 05:38，训练到 step 2950 时，按 PID 停掉 8 个占卡监督进程（`workload_at_20261109T210622Z/card<k>.pid`）。它们没有子进程。nvidia-smi 确认它们没有占用显存。
- **重试一次。** 06:15，`c0_tts2__talk2` 第一次 rc 1，自动重试成功。全程没有显存余量告警。
- **卡 7 空闲（07:04 UTC，协调者发现）。**
  - 原因：`finalpost_dev_orch.plan()` 对非解码作业取 best-fit（`min(free − need)`）。解码结束后，语音作业都堆到忙的卡上，空卡最后才被选。
  - 处理：给 `crf_pair_dev_orch.py` 加了 `--spread`（默认关，行为与原来逐位相同）。
    - 打开后，在没有解码运行、等待或就绪时，改为 worst-fit（`max(free)`）。
    - 锁、≤ 135 GiB、12 GiB 余量和 talker 对间隔都不变。
    - CPU 单测 11 项全部通过（`test_crf_pair_dev_orch.py`），提交 87b8d116。
  - 07:09 按 PID 停掉旧编排（3867716），没有停任何作业。用 `--resume --spread` 重启（477687），接管了 37 个运行中的作业，0 个失败。
    - 重启时其余 38 个等待作业都还在等 talker，暂无可放的作业。
    - 07:18 起，ASR 作业被放到空卡 7、3、6 上。
  - 新编排在结束时照常写出 `events/DONE`。v2′ 启动器（PID 3880632）已在 07:44:47 检测到。
- **占卡程序未由我重启。** 评测结束时，`train_mist_workload.py` 已在全部 8 卡上运行（每卡约 80 GiB），是 07:18 / 07:26 由他人启动的（`workload_at_20261110T07*`）。
  - 再跑一次 `start_workload_at.sh` 会重复占卡，所以没有执行。
  - v2′ 启动器会按自己的流程停掉并在训练后重启它们。

## 6. 注意事项

- **语音不确定性大。** C0 的两个种子相差 2.2 BLEU，两个候选的语音差异都不显著。C0-fp 对 C0 的 +0.94（P = 0.032）是本轮最强的语音信号，但只有 2 次抽取。
- **数据范围。** 只用了 dev。没有读取或评测 RealSI、ACL 60/60、AliMeeting。没有 COMET 或 MetricX。

## 文件

- **代码**：
  - `experiments/mist_w3po_v1/crf/decode_crf_ckpt_finalpost.py`、`test_decode_crf_ckpt_finalpost.py`；
  - `experiments/mist_w3po_v1/scripts/crf_pair_dev_orch.py`、`test_crf_pair_dev_orch.py`。
- **闸门**：`decode/dev_crf_{c0,c0fp}_s3000_finalpost_s8/gate_2seed_vs_{finalpost,ref}.json`，以及 `decode/dev_crf_c0fp_s3000_finalpost_s8/gate_2seed_vs_c0.json`。
- **次要指标**：
  - `eval_outputs/mist_w3po_v1/crf/c0pair_dev_compare.json`；
  - 检查 A：`eval_outputs/mist_w3po_v1/crf/c0pair_dev_diagA/{c0,c0fp}`。参考侧沿用 `finalpost_dev_diagA/{ref,cand}`。
- **编排状态**：
  - 正式运行：`logs/mist_w3po_v1/c0pair_dev/`（orch.log、status.json、events/）；
  - 冒烟：`logs/mist_w3po_v1/c0pair_smoke/` 和 `eval_outputs/mist_w3po_v1/crf/c0pair_smoke_c0pair_smoke/`。
