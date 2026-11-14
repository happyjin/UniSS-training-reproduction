# 联合双向系统：严格基线 vs 方法（dev + test，2026-11-14 08:50 布拉格）

- **严格基线**：joint v8 SFT step 6000（joint_v8h6k_20261113T073658Z）+ 冻结的原始 T1c@1500 talker（训练前的 talker，无 rl / PL-EM / c1b1s2）。
- **方法**：joint SerialCRF v2′ stage 2 step 3500（joint_v2p_20261113T120415Z；规则选 4000，用户改选 3500）+ read-level merge + PL-EM talker（zh→en：talker_plem_20261111T001759Z；En→Zh：talker_plem_en2zh_20261114T045019Z i0400）。
- 指标：文本 tcpBLEU（all / ov / 1spk）；ASR-BLEU = 两个 TTS 种子的平均；LAAL 单位为秒，主延迟 = chunk 协议下的 speech LAAL（括号内为旧协议）。没有 COMET / MetricX。
- dev 数据来自 `eval_outputs/mist_w3po_v1/joint/dev_summary_v1.json`（joint_dev.sh 正式输出）。test 数据用 joint_test.sh 的同一段 summary 代码和 gate_2seed 在 CPU 上算出（暂存在 scratchpad）；joint_test.sh 要等 (a) 的 EN2ZH_TEST_DONE 才会写正式的 test_summary_v1.json，那里只多出“方法 vs 单方向系统”的对比。

## dev

| 方向 | 系统 | text all | ov | 1spk | text LAAL | ASR-BLEU | ASR ov | speech LAAL |
|---|---|---|---|---|---|---|---|---|
| zh→en | 基线 | 27.93 | 23.13 | 30.06 | 2.946 | 19.24 | 14.00 | 3.973 (4.137) |
| zh→en | 方法 | 27.75 | 22.96 | 29.87 | 3.370 | **23.33** | 17.58 | **3.957** (4.158) |
| En→Zh | 基线 | 41.07 | 37.61 | 43.31 | 2.318 | 31.07 | 26.06 | 3.121 (3.257) |
| En→Zh | 方法 | **41.90** | 37.93 | **44.58** | 2.698 | **31.72** | 26.78 | 3.255 (3.427) |

Gate（gate_2seed，按会议做配对 bootstrap，规则：≥ 参照 − 0.5）：
- 方法 vs 基线，zh→en：text −0.18 [−0.60, 0.22] 通过；ASR **+4.09 [2.87, 5.32]** 通过。
- 方法 vs 基线，En→Zh：text **+0.83 [0.15, 1.51]** 通过；ASR **+0.65 [0.30, 1.01]** 通过。
- 方法 vs 单方向 A′rl + PL-EM（参考对比）：zh→en text −0.42、ASR −0.37（按 CI 判为未通过）；En→Zh text −0.32（未通过）、ASR −0.17（通过）。

## test（只评一次，未在 test 上做任何选择）

| 测试集 | 系统 | text all | ov | 1spk | text LAAL | ASR-BLEU | ASR ov | speech LAAL |
|---|---|---|---|---|---|---|---|---|
| AMI zh→en | 基线 | 25.02 | 19.97 | 27.10 | 2.986 | 18.07 | 12.48 | 4.014 (4.198) |
| AMI zh→en | 方法 | 25.09 | 19.96 | 27.18 | 3.364 | **20.48** | 14.75 | **3.986** (4.203) |
| AliMeeting zh→en | 基线 | **21.43** | **16.91** | **23.05** | 2.823 | 16.73 | 11.69 | **4.277** (4.371) |
| AliMeeting zh→en | 方法 | 18.99 | 15.63 | 20.45 | 3.866 | **17.37** | 12.63 | 4.848 (5.002) |
| AMI En→Zh | 基线 | 41.15 | 36.16 | 42.51 | 2.368 | 31.65 | 24.69 | 3.211 (3.374) |
| AMI En→Zh | 方法 | **41.85** | 36.18 | **44.33** | 2.734 | **32.41** | 25.40 | 3.336 (3.544) |

Gate，方法 vs 基线：
- AMI zh→en：text +0.07 [−0.40, 0.56]；ASR **+2.41 [1.56, 3.30]**。
- AliMeeting：text **−2.44 [−2.88, −1.65] 未通过**；ov −1.28 未通过；ASR +0.64 [0.24, 1.65]。
- AMI En→Zh：text **+0.69 [0.11, 1.62]**；ASR **+0.76 [0.42, 1.04]**。

## 要点
1. 方法在全部 5 个评测点（dev 两个方向、test 三个集合）上的 ASR-BLEU 都高于严格基线，且置信区间都在 0 以上：zh→en +2.4 到 +4.1，AliMeeting +0.6，En→Zh +0.7 到 +0.8。
2. 文本 BLEU：En→Zh 两处显著提升；AMI zh→en 持平；**AliMeeting 下降 2.44**。用同一个 v2′ Thinker 的单方向 A′ 在 Ali test 上也只有 18.14（低于 21.43），说明这是 v2′ SerialCRF 写计划在 AliMeeting 上的共同倾向（单方向系统也一样），不是流水线故障（merge 4065/4065，丢失 2 对）。
3. 延迟：zh→en 的 speech LAAL 持平或略低（dev −0.02，AMI −0.03）；En→Zh +0.13 / +0.13；**AliMeeting +0.57**。文本 LAAL 普遍 +0.37 到 +0.42（Ali +1.04），因为 CRF 写计划等待更多上下文，PL-EM talker 在语音端把大部分延迟吃回来了。
4. 联合系统仍略低于单方向 A′rl + PL-EM（dev −0.2 到 −0.4）。

## step 3500 vs 4000（stage-2 dev5，text）
| step | zh→en Q / L | En→Zh Q / L | J | 规则可行 |
|---|---|---|---|---|
| 3000 | 27.81 / 3.664 | 42.48 / 2.879 | 35.14 | 是 |
| 3500（用户选定） | 27.99 / 3.747 | 43.19 / 2.978 | **35.59** | 否（En→Zh L 超出 0.137 > 0.1） |
| 4000（规则选定） | 27.72 / 3.725 | 42.89 / 2.939 | 35.30 | 是 |
