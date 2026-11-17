# v17 准备：CRF 的有利证据、对方法不利的结果清单、计算时间与停顿的出处（2026-11-17，布拉格时间）

全部来自已有结果，没有新跑实验。

- test 数字的来源是 `eval_outputs/mist_w3po_v1/joint/v11f_analysis/factorial_test_v1.json`：v11 预登记的析因实验，用 B 的文本、两个 TTS 种子，按会议做配对 sign-flip 检验，95% bootstrap。
- 文中所说的表号都是 v16 的编号。

---

## 1. 哪些已有结果最能证明 CRF 有效

### 1.1 四组对比（\*\*\* p<0.001，\*\* p<0.01，n.s. 不显著）

| 对比（test，同一文本） | 控制住的因素 | tcpASR-BLEU（Z→E / E→Z / Ali） | EndOffset（s） | LAAL（s） | 是否“有效” |
|---|---|---|---|---|---|
| **① PA Talker − base Talker，同一积压语速规则，各采样一个**（= v16 Table 5 的 R3 − R2） | 语速规则、采样数 | **+0.29\*\*** / +0.02 n.s. / **+0.33\*\*** | **−0.18\*\*\* / −0.15\*\*\* / −0.22\*\*\*** | −0.01\*\*\* / −0.02\*\*\* / −0.04\*\*\* | **是**：三个集合的时延都显著下降，质量没有一处显著变差 |
| **② PA Talker − base Talker，同一语速规则，都做二选一**（R5 − R3，v16 未展示） | 语速规则、二选一 | +0.07 n.s. / −0.03 n.s. / **+0.27\*\*\*** | **−0.18\*\*\* / −0.14\*\*\* / −0.22\*\*\*** | −0.01\*\*\* / −0.02\*\*\* / −0.04\*\*\* | **是**：同上；这正是 reviewer 要求的“相同语速策略下”的对照 |
| ③ MIST − 规则（语速在 dev 上调到与 MIST 相同，同样二选一）（R5 − R3′） | 语速、二选一 | +0.11 n.s. / +0.05 n.s. / **+0.23\*\*** | −0.06\*\*\* / −0.03\*\*\* / −0.00 n.s. | 0.00 n.s. | 部分有效：质量只在 AliMeeting 显著，时延收益很小 |
| ④ PA − λ=0（BoN，只按质量加权），只有 dev，Zh→En，旧文本 | 一切相同，只差 CRF 的延迟项和 β | +0.14（n.s.） | −0.10（两个种子同号，未做检验） | 每条流持平 | 方向有利，但只有 dev，没有显著性检验 |

### 1.2 建议放进正文的内容：①和②（都满足“有效”）

- 做法：在 v16 Table 5（ablation）中加一行 “R2 + best of two（base Talker）”，表格就成为语速规则固定下的 2×2：{base, PA} Talker × {一个样本, 二选一}。
- 改后的表（数字都取自同一个 JSON；四舍五入，31.645 写作 31.65）：

| Speech layer | tcpASR Z→E | E→Z | Ali | EndOffset Z→E | E→Z | Ali |
|---|---|---|---|---|---|---|
| R0 B（base Talker） | 18.07 | 31.65 | 16.73 | 4.08 | 3.48 | 4.57 |
| R1 + read merging | 18.32 | 31.67 | 16.62 | 4.00 | 3.45 | 4.41 |
| R2 + backlog-aware rate | 19.88 | 31.91 | 17.49 | 3.31 | 3.05 | 3.47 |
| **R3a R2 + best of two（base Talker）**（新增） | 20.28 | **32.26** | 17.81 | 3.30 | 3.04 | 3.44 |
| R3b R2 + PA Talker（一个样本） | 20.17 | 31.93 | 17.82 | 3.13 | **2.90** | 3.24 |
| R4 PA Talker + best of two = MIST | **20.35** | 32.23 | **18.08** | **3.13** | 2.91 | **3.23** |

- 正文可以这样写：在语速规则和选择方式都固定时，把 base Talker 换成 CRF 训练的 PA Talker，三个集合上语音都提前 0.14–0.22 s 结束（p<0.001）；AMI 上质量不变，AliMeeting 上提高 0.27（p<0.001）。也就是说，二选一带来的是质量，CRF 带来的是时延，两者互补。
- **要注意的一点**：加了 R3a 之后，En→Zh 一列的最佳 tcpASR-BLEU 是 R3a 的 32.26，比 MIST 的 32.23 高 0.03（不显著），按“每列最好值加粗”的规则，这一格不再是 MIST。这是如实报告的结果。
- **仍然不能证明的**：①和②证明的是“CRF 训练的 Talker 优于不训练的 base Talker”，**并不能**证明“优于其他方式微调的 Talker”。能回答后一个问题的只有④（BoN，只有 dev）。若只放①②，正文措辞要写成 “the CRF-trained Talker”，不要写成 “the CRF is necessary”。
- ③（R3′）是否放进来由你决定。它最直接回答 reviewer 的问题，但只部分有效。

---

## 2. 计算时间（1.20 s / 52.8 s）出自哪里，能不能删

- **来源**：v11 的 wall-clock 回放实验，预登记文件是 `reports/mist_w3po_v1/v11_p11_wallclock_prereg.zh-CN.md`，结果记在 LOG 里。论文中的文字来自宏 `\num{v11:wc:verdict}`（`generated/macros_v11.tex` 第 30 行），内容是：
  - 每秒音频的 Thinker 计算时间：AMI Zh→En / En→Zh / AliMeeting 分别为 0.65 / 0.80 / 1.20 s；PA Talker 为 0.18 / 0.26 / 0.31 s。
  - 计算使块结束的中位时间再推迟 3.4 / 2.9 / 52.8 s。
- **v16 中出现的位置**：
  - 附录 C 的 *Computation* 段（`sections/appendix_v16.tex`）；
  - Limitations 的 *System* 段最后一句：“as implemented, decoding does not keep up in real time (Appendix C)”；
  - 正文 4.1 已写明 “delay … without computation”。
- **能否删**：可以删掉附录 C 里的具体数字（只回放了少量会议、一种硬件，且实现未优化：batch-1，每次读都重新编码前缀），并在 4.1 加上 “computation-excluded simultaneous evaluation” 一句。
  - **建议保留一句如实的说明**：Limitations 中写 “Latencies exclude computation; our implementation, which re-encodes the prefix at every read, does not run in real time.”。
  - 理由：数字删掉没有问题，但如果连“目前跑不了实时”这个已经测到的事实也去掉，又在标题和正文写 simultaneous，审稿人一旦追问就很被动。
- **建议的 4.1 措辞**：*We evaluate computation-excluded simultaneous translation: LAAL and EndOffset count audio and playback time but not computation, as SimulEval's non-computation-aware latency.*

---

## 3. 停顿 4.4 → 11.8 次/分钟出自哪里，能不能删

- **来源**：v10 的语音质量分析（每分钟内 ≥0.5 s 的块间静音）。数值在 factorial JSON 的 `pauses_per_min` 中。
- **v16 中出现的位置**：
  - 正文 **Table 2**（时延分解表）的 pauses 列：3.3→8.3 / 4.6→9.4 / 4.4→11.8；
  - 4.3 末句 “the cost is more pauses between blocks …”；
  - Limitations 的 *Method* 段 “Faster speech brings more pauses …”；
  - 附录 **Table 12** 的 ≥0.5 s 停顿和句中停顿列（句中停顿 0.8→2.8，0.9→3.9），以及 G.2 的 “pauses become two to three times as frequent”。
- **成因**（已有分析，记忆 *TS-PA closed*）：停顿出现在说话人的语音流**已经追上源语音、正在等下一块文本**的时候，是队列被消除后的另一面。实验也证实，在 Talker 一侧修不掉它：用 oracle 能减少 24–40%，因果的 Talker 只能减少 1–2%。
- **能否删**：
  - **可以**把 pauses 列从正文 Table 2 移出，只留在附录 Table 12，正文和 Limitations 各保留一句。
  - **不建议完全删除**：这是方法真实的代价，模拟审稿已经点名了这个数字，听过 demo 的审稿人也会注意到。完全删除会被视为选择性报告。
  - **更好的写法**是同时给出成因：*Pauses appear when a stream has caught up with its source and waits for the next block — the counterpart of the removed queue (Appendix G).* 这样把代价解释成机制的副作用，读者更容易接受。

---

## 4. 对方法不利的结果清单

| # | 结果 | 位置（v16） | 不利之处 | 建议 |
|---|---|---|---|---|
| 1 | 停顿增加（AliMeeting 4.4→11.8 次/分钟，句中停顿也增加） | 正文 Table 2、4.3、Limitations；附录 Table 12 | 语音更碎 | 正文可移出数字，但保留一句并说明成因（§3） |
| 2 | 计算不实时（AliMeeting 每秒音频 1.20 s，块结束中位推迟 52.8 s） | 附录 C、Limitations | 不能宣称可以实时部署 | 数字可删，Limitations 保留一句（§2） |
| 3 | 外部系统的 LAAL 更低（X-Translator：AMI Zh→En 2.69 vs 3.03、Ali 2.77 vs 2.95；SeamlessStreaming：AMI Zh→En 2.73） | 正文 Table 1（加粗）；附录 Table 10 | 时延不是全面最好 | 保留。正文已解释：它们的质量只有 MIST 的 1/3 到 3/4 |
| 4 | 同数据级联在 AMI Zh→En 上略好于 B（cp 24.88 vs 24.72，tcp 18.37 vs 18.07） | 正文 Table 1 | “说话人优先”只靠 B 体现不出来 | 保留。正文写的是“级联与 B 持平，MIST 两者都优于” |
| 5 | AliMeeting 上说话人 token：全部单元的 tcpBLEU 更低（11.47 vs 12.67），重叠单元持平（预登记检验未通过） | 正文 Table 4、4.5；附录 Tables 18、19 | OOD 上翻译质量没有提高 | 保留。已如实写成“部分迁移”，只有召回提高 |
| 6 | PA Talker 在 En→Zh 上的质量增益只有 +0.02（不显著） | 正文 Table 5 | CRF 的质量收益不是处处都有 | 保留。若采纳 §1.2，可写成“CRF 带来时延，二选一带来质量” |
| 7 | 语音开始时间几乎不变（−0.03 / −0.13 / −0.04 s） | 附录 Table 13 | 播放层不能让第一个词更早被听到 | 保留在附录，不碍事 |
| 8 | DiariST 官方协议：MIST 文本在 **dev** 上略低于 DiariST（SAg 15.26 vs 16.05，SAt 13.74 vs 13.91） | 附录 Table 17 | 不是全面领先 | 保留（只删 dev 等于挑结果），test 上 MIST 领先 |
| 9 | 离线行：在 AliMeeting 上，speaker-blind + 重标的全部单元得分高于离线 MIST（16.27 vs 15.19） | 附录 Table 18 的 offline 两行 | 离线时说话人 token 在 Ali 上没有帮助 | **可以删 offline 两行**：离线模型不是论文要证明的系统，也不属于 12 项预登记检验 |
| 10 | 第二识别器在 En→Zh 上的增益被 Whisper 在 B 上更多的失败放大 | 附录 G.4 | 这一项增益不可信 | 保留（这是如实的说明），正文不引用 +3.74 这个数 |
| 11 | 锁定测试的绝对分数低（10.66→12.89） | 附录 Table 15、Limitations | — | 保留 |
| 12 | X-Translator 不加增益时的原始结果 | 附录 F.2 | 对外部系统做了处理 | 保留，这是增益校准的依据 |

**不在 v16 中、但在项目记录里的不利结果**（论文里已经没有，无需删除，审稿时心里要有数）：
- R3′（语速匹配规则）：MIST 只在 AliMeeting 显著更好；
- BoN-SFT（dev）：质量持平；
- PA-F 流畅度 Talker：降低了 ASR-BLEU，已弃用；
- TS-PA：停顿在 Talker 一侧修不掉；
- 写对齐 WA：时延漂移，弃用；
- 单通道会议格：没有超过每条流的格。

---

## 5. 说话人相似度代理指标：在什么数据上做、怎么做、用什么指标

- **数据**：
  - 已有的 test 合成语音：B 和 MIST，两个 TTS 种子，三个集合；可选再加 X-Translator（它也做声音克隆）。
  - 每个说话人的“原声”参考：
    - AMI：用标注中该说话人的**单人说话片段**，从原始英文 IHM-mix 录音（`data/ami_multispk_si_v1/session_audio/ihm_mix`）中切出，取 ≥1 s 的片段，拼接到约 30 s。中文 AMI 的源语音也是用这些真人的声音克隆的，所以 AMI 两个方向都对比真人原声。
    - AliMeeting：从测试集的混合耳麦录音中，按参考标注切出单人片段。
- **方法**：
  - 用本地已有的 `wavlm-base-plus-sv` 模型（v16 的 SIM 用的也是它），对每条输出流的合成语音和对应说话人的原声各提取一个说话人向量。
  - 输出流与说话人的对应关系使用 cpASR-BLEU 的匹配 σ*。
  - **不生成新语音，不训练**，只提取向量，GPU 或 CPU 几小时。
- **指标**：
  1. **SIM-own**：输出流与**说话人本人原声**的余弦相似度，在所有流上取平均，替代现在的“与 voice prompt 的相似度”。
  2. **说话人辨认准确率**：每条流与同场会议中哪位说话人的原声最近；最近的是本人即算正确。同时报告随机水平（AMI 4 人即 25%，AliMeeting 2–4 人）。这个指标直接回答“听众能不能认出是谁在说”。
- **注意**：输出语言与参考语言不同（跨语言），绝对值会偏低，但 B 与 MIST 的对比是公平的。
- **结果放在哪里**：Table 2 的 SIM 列换成 SIM-own 和辨认准确率，附录补上细节。Limitations 中“SIM 只对比了 voice prompt”这一句随之删掉。
