# v11 P0-1 登记：播放层（PA）逐因子消融，基于基线 B 的文本

2026-11-15 布拉格时间约 09:40 写，**在任何测试集解码之前提交**。依据：docs/mist_v11_plan.zh-CN.md §3 P0-1；
审稿 M3（"PA 整体有效，但学习部分的独立增益不清"）。用户外出，决定按最优选择做（LOG 有记录）。

## 1. 要回答的问题

在完全相同的文本上（B = 联合 deadline Thinker stage 1 第 6000 步，`joint_v8h6k_20261113T073658Z_s06000`），
B → B+PA 的语音收益分别来自哪一部分：读级合并、按积压的语速规则、2 选 1 CTC 选择、学到的 PA Talker。
重点：学到的 PA Talker 在**语速匹配**的规则面前是否还有收益。

## 2. 行（全部两个 TTS 种子：默认种子 / 20261109）

| 行 | 名称后缀 | 合并 | 语速 | 候选 / 选择 | Talker |
|---|---|---|---|---|---|
| R0 = B | `__t1c`（已有） | 否 | 固定 | 1 | T1c |
| R1 | `__rlt1c` | 读级 | 固定 | 1 | T1c |
| R2 | `__rlc1s2` | 读级 | 队列规则 + shift 2 | 1 | T1c |
| R3 | `__rlc1b1s2` | 读级 | 队列规则 + shift 2 | 2 / CTC | T1c |
| R3′ | `__rlc1b1s<N>` | 读级 | 队列规则 + shift N（语速匹配） | 2 / CTC | T1c |
| R4 | `__rlplemk1` | 读级 | 队列规则 + shift 2 | 1 | PA（backlog token） |
| R5 = B+PA | `__rlplem`（已有，E2） | 读级 | 队列规则 + shift 2 | 2 / CTC | PA（backlog token） |

- 命令：`v11_factorial_orch.py` 对 joint_orch 生成的 si_infer 命令做字符串改写；其余（分片、种子、ASR、打分、放置）与 E2 完全相同。
- Talker：Zh→En PA = `talker_plem_20261111T001759Z/export`；En→Zh PA = `talker_plem_en2zh_20261114T045019Z/export_i0400`（E2 用的同一个）。T1c = `si_t1c_20261021T181440Z/autoext/export_1500`。
- **R3′ 的 N**：dev 上跑 N ∈ {2, 5, 8}（N=2 即 R3）。每个方向取 dev 平均语速（Zh→En 英文词/秒，En→Zh 汉字/秒，两个种子平均）最接近 R5 dev 语速的 N。若 N=8 仍比 R5 慢 0.15 以上，加跑一个 N=11 再选（只此一次）。选定后写进 LOG，再跑测试；测试集上不选。

## 3. 集合

- dev：AMI dev Zh→En（devz）、En→Zh（deve），整场在线窗口。
- test：AMI Zh→En（ami）、AliMeeting（ali）、AMI En→Zh（en），每行各跑一次。

## 4. 指标

- ASR-BLEU（tcpBLEU，all / ov / 1spk；Whisper-large-v3 / Paraformer；两个种子合并）。
- 延迟（chunk 协议，`lat_units_dump`）：逐词 LAAL_spread、块起点 LAAL_block、speech start、EndOffset。
- 语速（单位 / 合成语音秒）、合成语音总时长、≥0.5 s 停顿/分钟、句中停顿/分钟（`talker_block_pauses`；只对英文输出）。
- UTMOS、SIM（`talker_quality.py`，种子 1）。

## 5. 对比和检验

- 主对比（回答 M3）：**R5 − R3′**（学到的 PA 层 vs 语速匹配的规则层）。
- 单因子对比：R1 − R0（合并）、R2 − R1（语速规则）、R3 − R2（CTC 选择）、R4 − R2（PA Talker，无选择）、R5 − R4（PA 上的 CTC 选择）、R5 − R3（v10 表 6 式对比）。
- 检验：与 v10 E3 / post_review 相同——按会议配对；BLEU 用两种子合并的统计量；延迟和语速用会议内求和后的比值；95% 区间 10,000 次会议 bootstrap（rng 20261114）；精确双侧符号翻转 p。
- 多重比较：主对比 R5 − R3′ 的 ASR-BLEU 与 LAAL_block 两项 × 三个测试集 = 6 项，Holm 校正；其余单因子对比为描述性（不校正，标"探索性"）。
- 判定（写进论文的措辞）：
  - R5 − R3′ 的 ASR-BLEU 在 ≥2 个测试集 Holm 显著为正 → "学习部分在语速匹配后仍有收益"。
  - 都不显著或为负 → 按 v11 方案 §2.3 的保底写法：播放层收益主要来自规则语速与候选选择，CRF 学习部分不带来额外显著收益。
  - LAAL_block / EndOffset 同时报告，不作为单独判定。

## 6. 卫生

- 文本不变，不解码 Thinker；测试集上不做任何选择（R3′ 的 N 在 dev 上定）。
- 不重跑已有的 R0、R5；它们与新行使用同一文本、同一种子。
- 结果文件：`reports/mist_w3po_v1/v11_p01_factorial_results.zh-CN.md`。
