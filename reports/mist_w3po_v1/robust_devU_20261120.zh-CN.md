# UniST 同域 dev（D-U）

2026-11-20 晚，布拉格。数据位置：`eval_outputs/mist_w3po_v1/devU/data/`。脚本：`experiments/mist_w3po_v1/voicecond/devU_build.py`。只用 CPU，固定种子，只生成一次。

## 为什么要建
- UniST 试点没有官方的 dev 划分（`stage_a_source` 下只有 `parts/train-*`）。
- 第十四波已经把 crc32(id) % 50 == 0 的条目留出、不进训练，D-U 就从这个留出子集里取。
- 原来的 dev (a) 是 CoVoST validation，对 UniST 训练的模型是跨域的；D-U 与训练同域。

## 规则
1. **候选：** UniST 留出子集中，原文和译文都非空，且原文（按 wave14_data.norm 规范化后）既不在任何测试或评测文本集里（与训练排除用同一套来源），也不等于任何第十四波训练行的原文。按 sha1("20261120|devU|<dir>|<id>") 排序。
2. **trim_norm：** 对前 3000 个候选做 trim_norm（去首尾静音，−26 dBFS，长度 1.5–12 s），与训练相同。
3. **时长匹配 S1 dev (a)：** 对该方向 (a) 的 300 句干净句，按时长从短到长，每句取剩下的候选里时长最接近的一条（时长相同时按排序先后）。
4. **带噪版：** 与 S1 dev (a) 带噪版的做法相同（`wave1_data.mix_one`）：
   - 背景：bench_build 的音乐、人群、音效三层时间线；
   - 背景池：Idea 2 背景索引的 **dev 划分**。训练用的是 train 划分；测试留出的 HELDOUT_BACKGROUNDS* 由 bg_pools 断言排除；
   - BVR：OpenBroadcast-SI esports 提示 BVR 分布在 q ~ U(0.05, 0.95) 处的分位数；
   - 逐句缩放，不做 ducking，限幅；不生成纯背景段和间隙段；种子 sha1("20261120|devU|mix|<id>")。
5. **格式：** 与 `wave2_S1/data` 相同：
   - `items_<dir>.jsonl`：{id, audio, dur_ms, translation}；
   - `meta_<dir>.jsonl`：{id, set = "u", cond, pair, stream, ovl = false, dur_ms}；
   - `manifest_<dir>.jsonl`：来源行、BVR、alpha、限幅、背景事件。

## 条目数与时长
| 方向 | 句数（每种条件） | 总时长 | 时长 p10 / p50 / p90（s） | S1 (a) 的 p10 / p50 / p90 | 语料构成 |
|---|---|---|---|---|---|
| 中→英 | 300 × 干净 / 带噪 | 26.5 min | 3.01 / 5.05 / 8.19 | 2.55 / 5.05 / 8.19 | emilia_zh 287，HQ-Conversations 9，NCSSD_R_ZH 4 |
| 英→中 | 300 × 干净 / 带噪 | 25.9 min | 2.72 / 5.17 / 7.48 | 2.72 / 5.17 / 7.48 | CommonVoice_EN 99，emilia_zh 前缀 89，Emilia EN（EN_B*）78，DailyTalk 13，VCTK 12，LibriSpeech 9 |

- 英→中里有 emilia_zh 前缀的条目，原文确实是英语。训练数据里也有同样的命名，是 UniST 的 id 命名问题，不是错配。

## 实际 BVR（dB）
| 方向 | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| 中→英 | −34.4 | −21.0 | −14.0 | −10.9 | −6.6 |
| 英→中 | −31.9 | −21.6 | −14.6 | −11.0 | −6.4 |

- 两个方向都没有触发限幅；带噪句峰值的最大值为 0.92 / 0.70；用到的背景文件为 139 / 136 个，都不在测试留出背景里。

## 重叠检查（verify.json）
- 选中的条目全部属于 crc32 % 50 == 0 的留出子集；id 不在训练行里。
- 原文、译文与训练行的原文、译文重合的条目数都是 0。
- 原文、译文与各测试或评测来源的重合都是 0。这些来源包括 CoVoST dev/test、S1 dev、OBSI bench、Idea 2 cvt、OOD-3、RealSI、ACL 60/60、AliMeeting、AMI、esports。
- 在候选阶段排除的条目：
  - 中→英：166 条原文与训练原文相同；
  - 英→中：2078 条与测试文本重合（主要是 CoVoST dev/test 1918 条），966 条与训练原文相同。
- 干净版与带噪版逐条等长，并与 dur_ms 一致：300/300。

## 偏差与注意
- **中→英的短句偏少：** 候选里短句不多，时长匹配后 p10 为 3.01 s，而 S1 (a) 是 2.55 s，最短 1.68 s 对 1.52 s。中位数和 p90 相同。
- **候选池：** 只对前 3000 个候选做了 trim_norm（成功 2650 / 2555 个），时长匹配在这个池子里进行。
- **语料构成：** 中→英几乎全是 emilia_zh；英→中约三分之一是 CommonVoice_EN（UniST 的 CommonVoice 子集，与 CoVoST dev/test 文本重合的已排除）。
