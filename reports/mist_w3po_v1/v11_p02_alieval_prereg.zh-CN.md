# v11 P0-2 登记：AliMeeting Eval 干净确认集（解码之前写定）

写于 2026-11-15（布拉格时间，任何解码之前）。依据：docs/mist_v11_plan.zh-CN.md §3 P0-2；用户 2026-11-15 授权（方案 A、
投主会、需要拍板的事按最佳选择决定）。只解码一次，不做任何选择。

## 1. 数据

- **会议**：AliMeeting Eval 集 8 场（R8001_M8004 R8003_M8001 R8007_M8010 R8007_M8011 R8008_M8013 R8009_M8018
  R8009_M8019 R8009_M8020），OpenSLR 119 `Eval_Ali.tar.gz`，近场 headset 求和成单通道（与测试集同一 `build_alimeeting_test.py prep`），
  共 6,457 句（含 1,084 句语气词），约 4.2 h。从未用于训练、开发、测试或任何决定；与 AliMeeting 测试集 20 场无重叠。只做测试。
- **参考**：DiariST 发布的 dev 人工英文翻译（`data/diarist_alimeeting_refs_v1/dev.txt`，6,454 行；DiariST §4.1：dev / test 由人工翻译）。
  按会议、说话人、起止时间（10 ms）和中文全文匹配到我们的句子：6,451 / 6,457 匹配（6 句无对应行，记为不计分）。
- **与测试集构建的偏差（只有两处，都因为条件不同，不是选择）**：
  1. 测试集的 `refs.jsonl` 是 Hy-MT2 机器译文，本服务器上没有该模型；这里由人工译文代替（参考窗口的 "en"/"ref" 字段）。
  2. 测试集的审核规则用来剔除坏的机器译文（14,148 句中 4 句）；人工译文不做剔除（若套用，长度比规则会误删 222 句所在的窗口）。
     因此参考窗口不依据参考译文做任何筛选。
- **分离器后验**：Streaming Sortformer，offline / lowlat 两个预设（`dump_nemotron_posteriors.py`，与测试集相同）。
- **窗口**：与测试集相同的 C2 在线窗口（gap 0.6 s，`build_test_windows_c2`）和 chunk 协议窗口（`chunk_windows_set`），
  构建代码 `experiments/mist_w3po_v1/data/alieval_build.py`（只在本进程内替换数据路径）。评分的 universe 由
  `reports/gstar_si_v6/meeting_score.py` 在测试集目录里找不到该会议时读 `data/alimeeting_eval_si_v1`（测试集结果不变）。

## 2. 系统（全部与 v10 测试集完全相同的配置，只换数据集）

| 行 | Thinker | 写偏置 | 语音层 | 名字 |
|---|---|---|---|---|
| B | stage 1 s6000（joint_v8h6k merged + identity） | 无 | 冻结 T1c talker，无合并（`--talker t1c`） | `jaev_b_<RT1>__t1c` |
| B+PA | 同 B 的文本 | 无 | 读级合并 + PA talker（`--talker plem`，c1b1s2 = 2 候选 CTC 选择 + backlog token） | `jaev_b_<RT1>__rlplem` |
| M | stage 2 s3500（joint_v2p） | δ_w = −1 | 同 B+PA | `jaev_m_<RT>__rlplem` |
| M4000 | stage 2 s4000（选择规则选的点） | δ_w = −1 | 同 B+PA | `jaev_m4_<RT4>__rlplem` |

- 解码：finalpost close_detect，policy wait，1.0 s 读，greedy，8 个分片（按会议），`--cap135`，与 v10 测试相同的编排器
  （`alieval_orch.py` 包装 `joint_orch.py` / `joint_orch_wb.py`，只替换 AliMeeting 的集合定义）。
- 语音：两个 TTS 种子；ASR Whisper-large-v3；chunk 协议延迟（文本、逐词语音 LAAL_spread、speech start、EndOffset）。

## 3. 假设与检验

**确认性（一个家族，Holm 校正，3 项）**，比较 B+PA − B（同一文本，只有语音层不同）：

- H1a：ASR-BLEU（tcpBLEU of ASR，两个 TTS 种子合并）> 0。
- H1b：常规 block-start 语音 LAAL（LAAL_block，整块的词都记在该块开始播放的时刻）< 0。
- H2：EndOffset < 0。

判定：双侧精确符号翻转检验（按会议，8 场共 2^8 = 256 种翻转；**最小可能的双侧 p = 2/256 ≈ 0.0078**，Holm 下第一项门槛 0.05/3 ≈ 0.0167，
所以只有 8 场中几乎全部同向才可能确认）**且** 10,000 次会议 bootstrap 的 95% 区间不含 0、方向与假设一致，才算"确认"。
统计量和 v10 相同：BLEU 类为合并统计量的语料级 BLEU；延迟类为各会议单元值之和 / 单元数（两个种子平均）。

**探索性（不校正，只报区间和 p）**：
- H3：M − (B+PA) 的文本 tcpBLEU（WA 的文本效应；v10 在 AliMeeting 测试集上为 −0.36）。
- M − (B+PA)、M4000 − (B+PA)、M − B 的 ASR-BLEU、文本 LAAL、LAAL_block、LAAL_spread、speech start、EndOffset。
- B+PA − B 的 speech start、LAAL_spread、语速（词/秒）、≥0.5 s 停顿/分钟、句中停顿、UTMOS、SIM。
- 文本 tcpBLEU ov / 1spk、CR、collapse；去重 tcpBLEU（若 P1-4 的工具届时可用）。

## 4. 不做的事

- 不调任何参数，不重跑，不在这 8 场会议上选 checkpoint、偏置或语音设置；结果无论正负都写进论文。
- 不用于训练、不改写、不调参（记忆 realsi-acl-test-only）。
- 如果某个作业失败，只做技术性重跑（同一命令 --resume），并在 LOG 中记录。
