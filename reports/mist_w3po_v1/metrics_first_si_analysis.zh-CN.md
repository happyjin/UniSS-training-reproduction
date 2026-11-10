# 指标优先：多说话人同传（MIST）该优化什么、问题在哪、CRF 能动多少

> 日期：2026-11-10。只用 CPU，没有训练，也没有启动 GPU 作业。正在跑的 v2′ 训练没有动。
> 数据只用 AMI dev：15 个验证会议，3,050 个在线窗口，源音频 5.69 h。没有读取或评测 RealSI、ACL 60/60、AliMeeting，也没有计算 COMET / MetricX。
> 新代码（都是新文件）：
> - `experiments/mist_w3po_v1/evaluation/metric_headroom.py`
> - `metric_headroom_rids.py`
> - `metric_omission_types.py`
> - `metric_speech_tail.py`
>
> 新输出在 `eval_outputs/mist_w3po_v1/metrics_headroom_v1/`。所有脚本都拒绝覆盖已有文件。
> **复现核对。** 重新实现的打分与 `overlap_score.py` 的结果逐位相同：
> - 4 个系统的文本 tcpBLEU all / ov / 1spk；
> - 8 次语音抽取的 ASR tcpBLEU。
>
> 文献依据见 §7。每条都给出实际读过的 URL 或 arXiv 号，无法核实的标注"未核实"。

---

## 0. 结论

**一、目标指标集**（§2）。按 SimulS2ST 文献的惯例，再加上"说话人"维度：

| 维度 | 指标 |
|---|---|
| 质量（主指标） | 语音 ASR-BLEU（tcp 版：说话人置换、时间约束，下同） |
| 质量（辅助） | 文本 tcpBLEU。两者都分 all、ov、1spk 报告 |
| 延迟 | 文本：StartOffset、EndOffset、按单元的 LAAL<br>语音：StartOffset、EndOffset（SimulEval 定义）<br>计算感知延迟（CA / RTF）：**目前没测** |
| 说话人 | 归属差 Δspk（说话人 oracle 减实际）、CR / collapse、输出重叠率、积压（backlog）<br>每个说话人的音色相似度 SIM-O：**目前没测** |
| 只作诊断，不作目标 | ≤1 s 召回、错标率、CHAIN 数、推迟质量、重复率 |

**二、最近 CRF 预检关注的指标，离同传指标很远。** 它们是"机制量"，和主指标之间隔着两层：机制 → 单元召回 → tcpBLEU，中间还要乘上很小的词数权重。

- **CHAIN。** 68 个单元只含 113 个参考词，占 dev 参考词的 0.18%。即使全部完美补上，文本 all 也只有 **+0.15**、ov **+0.45**。
- **≤1 s 短单元。** 占参考词的 7.4%，当前漏掉的只占 1.1%。全部完美补上：C0-fp 上 all **+0.88**、ov +1.67。
- **噪声地板。** 正式闸门可分辨的最小差：文本 all 约 ±0.35，ov 约 ±0.9，语音约 ±1.1。
- **结论：用户的怀疑成立。** 短单元这条线能拿到的现实收益，大部分已经被"最终读因果 P"（干净效应 +0.39）和 C0 拿走了。

**三、按实测数值排序的三大指标缺口**（C0-fp，dev）：

| # | 缺口 | 数值 | 可分解为 |
|---|---|---|---|
| 1 | **语音通道** | 文本 28.00 → 语音 19.91（−8.1）；ov 22.87 → 12.97（**−9.9**） | ① 打分粒度伪差 −1.9 / ov −3.3：语音按"窗口×说话人"整条打分，见 §4.3<br>② ASR 幻听长尾 −1.2 至 −1.4 / ov −1.9 至 −2.8：3.6% 的流贡献了 92% 的多余词<br>③ 其余约 −4.9 / ov −4.3 是 TTS→ASR 的保真度损失<br>此外，文本上的进步传不到语音 ov：从 v8 到 C0-fp，文本 ov +0.93，语音 ov −0.28 |
| 2 | **内容覆盖** | 长单元（>1 s）漏译的完美补全上界 all **+3.89** / ov +4.65 | 欠译（译文长度不到参考一半）上界 +1.87 / ov +4.26<br>BP = 0.942，相当于少了 1.7 BLEU<br>对照：短单元（≤1 s）的上界只有 +0.88 |
| 3 | **重叠区内容质量** | ov 22.87 对 1spk 30.26（−7.4）；unigram 精确率 55.7 对 64.9 | 补全所有漏译后仍差 5.2（29.17 对 34.38）<br>说话人 oracle 对 ov 几乎为 0（−0.07），**不是标签问题** |

**延迟的现状。**
- 文本 EndOffset p50 1.94 s，按单元 LAAL p50 3.0 s；语音 EndOffset p50 2.2 s、p90 6.2 s。
- 这和 Seamless、Hibiki 处在同一量级。主要由窗口检测器决定：打开 1.24 s、关闭 +1.70 s，41% 的词在最终读才提交。CRF 调度改不了这部分。
- 真正的缺口有两个：
  - **只有单一工作点**，没有质量-延迟曲线（文献都报曲线）；
  - **没有计算感知延迟**。

**四、各条 CRF 线能动多少**（§5）。

| 线路 | 状态 | 可动指标的上界 | 现实预期 |
|---|---|---|---|
| 1′、SerialCRF v1、v2、v2′、TurnCRF、全局归一化头 | 已停止或预期为零 | — | 都在噪声地板以下 |
| **v2′**（正在训练） | 训练中 | CHAIN 上界 all +0.15 / ov +0.45 | 约 +0.0–0.1 / +0.0–0.3，**测不出来** |
| CRF 动态 oracle | 预检 STOP | 改成"覆盖所有漏译与欠译"后，可触及缺口 2 的 +3.9 / +1.9 | — |

整个 CRF 家族（短单元 + 归属 + 调度）在文本 all 上的上界合计约 1.3，现实 ≤ 0.5；**对语音缺口 1 完全没有作用**。

**五、新方向排序**（§6）。按"预期影响 × 新颖度 × 在本代码库的可行性"排序：

1. **MS-SimulEval**：多说话人同传评测协议与指标。修掉约 3 BLEU 的语音测量伪差，是所有工作的前提。
2. **"说你写的"**：基于单调对齐边际的流式 talker 一致性解码与训练，针对缺口 1。
3. **覆盖格动态 oracle**：CRF 主线改换目标，从短单元转到所有漏译和欠译，针对缺口 2。
4. **输出时间线调度 CRF**：针对语音 EndOffset、积压和输出重叠。
5. **说话人路由的结构化注意力**：针对缺口 3。
6. **单模型多延迟**：给出质量-延迟曲线，这是可比性的必需品。

没有一条能保证发顶会。前三条与指标缺口对得最准。

---

## 1. 文献里的同传指标（核实过的）

### 1.1 单说话人同传 / SimulS2ST

| 工作 | 质量指标 | 延迟指标 | 语音与说话人 | 代表数值（质量 / 延迟） |
|---|---|---|---|---|
| **HMT**（Zhang & Feng, ICLR 2023，arXiv 2303.00257） | BLEU | AL（正文）；附录有 CW、AP、DAL。文本同传，token 单位 | — | WMT15 De→En：L3K6 29.29 / AL 3.46；L5K6 30.29 / 4.74 |
| **SimulS2ST-Omni**（He, Liang, Chen, Li, Chen, Wu；arXiv 2607.19810，v2 2026-08-30，无会议信息） | Text-BLEU、ASR-BLEU（sacreBLEU，中文分词 zh、英文 13a；ASR 英文用 Whisper-Large-V3，中文用 Paraformer；另用 Qwen3-ASR 做敏感性检查） | 句级 LAAL 与 LAAL_CA；长音频 StreamLAAL 与 StreamLAAL_CA；Compute RTF。**不报** StartOffset、EndOffset、ATD、AL | SIM-O（WavLM-Large）、AutoPCP、人工 MOS、LLM 评判的 VIP 通过率。**没有多说话人评测** | CVSS-T 离线 ASR-BLEU 31.12 / 25.18（En→Zh / Zh→En）；RealSI 句级 En→Zh 随延迟档 m2→m6 从 24.21 升到 26.32 |
| SimulEval（Ma et al., EMNLP 2020 Demo） | BLEU、ASR_BLEU | AL、AP、DAL、LAAL、ATD、StartOffset、EndOffset、RTF、NumChunks、Discontinuity；`--computation-aware` 给出 _CA 版本 | 语音输出按 300 ms 切 token 计算 ATD；有 SpeechAlign（MFA）变体 | — |
| LAAL（Papi et al., AutoSimTrans@NAACL 2022，arXiv 2206.05807） | — | AL 的分母改为 max(\|Y\|, \|Y*\|)，防止过度生成被奖励 | — | — |
| **StreamSpeech**（Zhang et al., ACL 2024，arXiv 2406.03049） | ASR-BLEU、BLASER 2.0 | AL、AP、DAL、StartOffset、EndOffset、LAAL、ATD，都有 _CA 版本（ms） | Discontinuity、RTF | CVSS-C Fr→En：chunk 320 ms 时 22.89，AL 1270，EndOffset 699；1280 ms 时 25.20，AL 3146 |
| Seamless（arXiv 2312.05187，技术报告） | BLEU、chrF、ASR-BLEU、BLASER 2.0 | 文本 AL、LAAL（s）；语音 **Ending Offset** | VSim、AutoPCP、语速、停顿 | SeamlessStreaming，Fleurs，X→eng / eng→X：S2ST ASR-BLEU 22.1 / 21.4，Ending Offset 2.79 / 4.64 s |
| **Hibiki**（Labiausse et al., arXiv 2502.03382；ICML 2025，只在检索结果里看到） | ASR-BLEU、BLEU | End Offset（s）、LAAL（s） | 说话人相似度、MOS | CVSS-C Fr→En：39.2，End Offset 2.9 s，LAAL 3.4 s；长音频 Audio-NTREX：27.5 / 2.7 / 5.0 |
| Hibiki-Zero（arXiv 2602.11072；会议未核实） | ASR-BLEU、BLEU、XCOMET-XL | End Offset、LAAL | 说话人相似度、MOS | 用只奖励 BLEU 的 GRPO，延迟是隐式优化的 |
| NAST-S2X（Ma et al., arXiv 2406.06937；ACL 2024 依据代码库的 BibTeX） | ASR-BLEU、BLASER 2.0 | AL、AL_EOW / BOW、StartOffset、EndOffset、ACT，分 CA 与非 CA | — | CVSS-C Fr→En：AL 3330 ms 时 20.20 |
| Seed LiveInterpret 2.0（arXiv 2507.17527，技术报告） | 人工 VIP / SVIP；BLEURT、COMET 作补充；**不报 BLEU** | FLAL、AL、LAAL | 声称能区分多说话人，**但没有量化评测** | RealSI 长音频 S2S zh-en：SVIP 67.8，AL 5.18 s |

### 1.2 多说话人：转写与翻译

| 工作 | 指标 | 说明 |
|---|---|---|
| SOT（Kanda et al., Interspeech 2020，arXiv 2003.12687）；t-SOT（Kanda et al., Interspeech 2022，arXiv 2202.00842） | WER；说话人归属时用 cpWER / SA-WER | t-SOT 按时间顺序输出 token，用换道 token 区分虚拟通道 |
| **MeetEval**（von Neumann et al., CHiME-7 2023，arXiv 2307.11394） | cpWER、ORC-WER、MIMO-WER、**tcpWER**、tcORC-WER | cpWER：每个说话人的文本拼接起来，取 WER 最小的置换。tc：时间约束的 Levenshtein，推荐 collar 5 s |
| cpWER 出处：CHiME-6（Watanabe et al., arXiv 2004.09249 §4.6） | cpWER | — |
| **DiariST**（Yang, Kanda et al., ICASSP 2024，arXiv 2309.08007） | **SAgBLEU**：不分说话人，整段拼接<br>**SAtBLEU**：按说话人拼接、取 BLEU 最大的置换<br>另有 DER | **没有延迟，也没有语音输出**。DiariST-AliMeeting Zh→En 测试集 SAtBLEU：IHM-CAT 17.07，SDM 11.67。注意论文里没有 "SAcpBLEU" 这个名字 |
| X-Translator（arXiv 2607.17544，级联系统） | Spk-SIM、Target Purity、Source Purity；短音频另报 Start、ATD、cATD、COMET | 多说话人只评音色和纯度，不评归属翻译质量 |
| STAC-ST（Zuluaga-Gomez et al., EMNLP 2023，arXiv 2311.00697） | （指标未核实） | 单通道、感知说话人轮次的对话翻译 |

### 1.3 对本项目的含义

1. **同传的通用骨架**是"质量 + 延迟（含计算感知）+ 语音自然度 / 说话人相似度"，而且都报**质量-延迟曲线**，不报单点。
   - SimulS2ST-Omni 只用一个模型，靠 m 档给出曲线；StreamSpeech 用 chunk 大小；HMT 用 (L, K)。
2. **多说话人一侧的标准**是"置换后计分 + 时间约束"，即 cpWER / tcpWER / SAtBLEU。我们的 tcpBLEU 正好是 DiariST 的 SAtBLEU 加上 MeetEval 的时间约束（collar 2 s），口径是对的。
3. **检索到的工作里，没有一篇在延迟约束下评测"说话人归属的语音翻译质量"。**
   - Hibiki、SimulS2ST-Omni、Seed LiveInterpret 都是单说话人，或者没有量化多说话人；
   - DiariST 没有延迟和语音；
   - X-Translator 只评音色和纯度。
   - 所以"多说话人 SimulS2ST 的指标体系"本身就是空白（§6 方向 1）。
4. **SimulS2ST-Omni 的测试集正好是 RealSI 和 ACL 60/60。** 按本项目规则，这两个只能作最终测试，所以本文只在 dev 上分析。与它对比要等用户决定跑测试集。

---

## 2. 本项目的目标指标集

### 2.1 主指标（闸门与论文表头）

| 指标 | 定义（本仓库实现） | 与文献的对应 | 现状 |
|---|---|---|---|
| **Q-speech：tcpASR-BLEU** all / ov / 1spk | 语音按"窗口×说话人"合成，Whisper-large-v3 转写后用 tcpBLEU 打分；两个 TTS 种子合并 | SimulS2ST 的主指标 ASR-BLEU，加上说话人置换和时间约束。ASR 模型与 SimulS2ST-Omni 的英文侧相同 | 已有；**粒度有伪差**（§4.3） |
| **Q-text：tcpBLEU** all / ov / 1spk | `overlap_score.py`：按会议做说话人 chrF 匈牙利匹配；单元为同一说话人相隔 <1 s 的话语合并；collar 2 s；13a 分词；corpus BLEU；插入作为 (hyp, "") 对 | DiariST 的 SAtBLEU 加 MeetEval 的 tc | 已有 |
| **L-text**：StartOffset、EndOffset、tcLAAL | 每个单元：首个 / 末个提交时刻减去单元起点 / 终点；LAAL 以单元为源，d_i = 提交时刻 − 单元起点，\|X\| = 单元时长，\|Y\| = max(hyp 词数, ref 词数) | SimulEval 的 StartOffset / EndOffset 与 LAAL，按说话人单元计算 | EndOffset 就是现有的 lag；StartOffset 和 LAAL 是本次新算的 |
| **L-speech**：StartOffset、EndOffset | talker 输出的 `start_offset_ms` / `end_offset_ms`（每条窗口×说话人流） | SimulEval / Seamless / Hibiki 的 (End) Offset | 本次首次汇总 |
| **L-CA**：计算感知延迟或 RTF | — | StreamSpeech、NAST-S2X、SimulS2ST-Omni 都报 | **缺**。离线批量解码（batch 16）给不出这个数，需要单流计时 |

### 2.2 说话人维度（多说话人特有）

| 指标 | 定义 | 说明 |
|---|---|---|
| **Δspk 归属差** | 说话人 oracle 的 tcpBLEU 减去实际 tcpBLEU。oracle 把每段的说话人改成 ±2 s 内参考文本 chrF 最高的那句的说话人，映射取恒等 | 相当于 DiariST 的 SAgBLEU 与 SAtBLEU 之差。它直接回答"说话人错误值多少 BLEU" |
| CR / collapse | 现有定义 | 并发召回；一方被吞进另一方 |
| **输出重叠率** | 同一窗口内，两条以上输出流同时发声的时间占输出语音总时长的比例 | 听众侧的可懂度；单说话人文献没有这一项 |
| **积压（backlog）** | 一个窗口的语音在下一个窗口的语音开始时还在播放的秒数 | 长会议里延迟的累积 |
| SIM-O（每个说话人） | WavLM-Large 余弦 | Hibiki、SimulS2ST-Omni 都报；**我们还没测** |

### 2.3 只作诊断的指标

≤0.5 s / ≤1 s 召回、错标率、CHAIN 数、推迟后验质量、S2′a / S2′b、重复率，以及本次新加的 BP、n-gram 精确率、漏译词占比。

这些指标只用来解释主指标为什么变，**不进闸门，也不作为方法"有效"的证据**。

### 2.4 不计算的

COMET、MetricX（项目规则），BLASER，人工 MOS。论文里可以注明文献报了这些。

---

## 3. 预检关注的指标与目标指标差多远

| 预检指标 | 出现在 | 和主指标的联系 | 能换算成的 tcpBLEU 上界 |
|---|---|---|---|
| ≤1 s 召回（孤立 / 重叠） | finalpost、C0、TurnCRF | 一个单元平均约 1.6 个参考词（≤0.5 s 约 1.06 个） | 全部 ≤1 s 漏译完美补上：C0-fp all +0.88 / ov +1.67 |
| ≤1 s 错标率 | diag_order_attr 检查 A | 错标的短单元会变成插入加漏译 | S1′ 中的 MISLABEL 83 个：+0.10 / +0.29。全局说话人 oracle：+0.45 / −0.07 |
| CHAIN 数（68） | v2′ S1′、动态 oracle | 113 个参考词，占 0.18% | **+0.15 / +0.45** |
| DEFERRAL（121） | v2 S1 | 186 个词 | +0.25 / +0.74 |
| 推迟后验质量 S2 / S2′a | v2、v2′ 预检 | 与任何主指标都没有直接映射 | — |
| C1 / C2 / C3（1′） | 1′ 预检 | C2 是提前多少秒，与 EndOffset 有关 | 延迟 ±0.01 s 量级 |

S1′ 那几行：用 v2′ 预检在 finalpost 解码上的逐单元分类（`serial_v2p_precheck/s1/s1_rows_primary_finalpost.jsonl`），对相应漏译做完美补全，见 `rids_finalpost_s1p.json`。在官方参考解码上（295 个单元），CHAIN 为 +0.17 / +0.49，全部为 +0.53 / +1.42。

**判断。**
- 这些预检指标是对的**机制**指标，但它们瞄准的误差在参考词里只占 0.2–1.1%。即使机制完全修好，对主指标的上界也低于或接近噪声地板：all 约 ±0.35，ov 约 ±0.9。
- 它们与同传主指标的差别不在于"定义不同"，而在于**权重**：它们针对的是 BLEU 几乎不计分的那部分内容。
- 它们对语音主指标没有任何杠杆。语音缺口在 TTS→ASR 和打分粒度，见 §4.3。

---

## 4. 从当前实验结果看问题

### 4.1 四个系统的主指标（dev，同一协议）

| 系统 | 文本 all / ov / 1spk | 语音 all / ov / 1spk（两种子平均） | BP / p1 | 文本 StartOffset p50 | 文本 EndOffset p50 | tcLAAL p50 / 均值 | 语音 EndOffset p50 / p90 | ≤1 s 召回 iso / ov |
|---|---|---|---|---|---|---|---|---|
| v8 参考 | 27.05 / 21.94 / 29.31 | 19.07 / 13.25 / 21.62 | 0.967 / 59.1 | 2.74 | 1.93 | 2.88 / 3.24 | 2.19 / 5.78 | 0.670 / 0.783 |
| v8 + finalpost | 27.75 / 22.82 / 29.93 | 19.17 / 13.96 / 21.48 | 0.974 / 59.9 | 2.65 | 1.94 | 2.83 / 3.20 | 2.07 / 5.75 | 0.829 / 0.812 |
| C0 | 28.08 / 23.04 / 30.31 | 19.03 / 13.09 / 21.95 | 0.944 / 61.9 | 2.84 | 1.94 | 3.01 / 3.44 | 2.19 / 6.13 | 0.850 / 0.811 |
| **C0-fp** | 28.00 / 22.87 / 30.26 | **19.91** / 12.97 / **23.58** | 0.942 / 61.9 | 2.83 | 1.94 | 2.99 / 3.43 | 2.21 / 6.18 | 0.851 / 0.812 |

- 单位：延迟为秒。StartOffset 和 EndOffset 都按单元计算，提交时刻已按在线协议夹到窗口检测时刻。
- 语音 StartOffset 中位数：v8 和 finalpost 3.0 s，C0 和 C0-fp 3.54 s。
- 输出重叠率 17.2–17.7%。积压：33–34% 的窗口有积压，平均 0.81–0.84 s，p90 2.9–3.1 s。四个系统几乎相同。

**观察**

1. **文本进步来自精确率，不来自召回或长度。**
   - 从 v8 到 C0-fp，p1 从 59.1 升到 61.9，插入词数从 2,436 降到 1,769。
   - 但 BP 从 0.967 降到 0.942，译文反而更短。BLEU / BP 从 27.97 升到 29.73。
2. **文本进步传不到语音 ov。** 文本 ov +0.93，语音 ov 为 −0.28。语音 all 的 +0.84 全部来自 1spk（+1.96）。
3. **延迟几乎没动。** EndOffset 由窗口检测器决定：每个窗口打开固定在 1.24 s，关闭在窗口结束后 +1.70 s（p50）。41% 的词在最终读提交，提交时刻被夹到关闭时刻。四个系统的 EndOffset 差别 ≤ 0.02 s。
4. **与文献同量级。**
   - 文本 tcLAAL 约 3 s：Hibiki 短音频 LAAL 3.4 s，长音频 5.0 s；Seamless S2TT 的 LAAL 为 2.1–2.2 s。
   - 语音 EndOffset p50 2.2 s：Seamless 2.79–4.64 s，Hibiki 2.7–2.9 s。
   - 数据和语言对都不同，只能粗比。
   - 延迟不是最大的缺口，但**我们只有一个工作点，也没有计算感知延迟**。

### 4.2 文本 tcpBLEU 的 oracle 上界（C0-fp；v8 参考列在括号里）

每一行单独施加在打分器自己的分配上。"补全"指把参考原文放进去，所以是**上界**；实际能达到的约为一半或更少。

| oracle | Δ all | Δ ov | Δ 1spk | 对应的误差类 |
|---|---|---|---|---|
| ≤1 s 孤立短单元漏译补全 | +0.32（+0.71） | +0.04 | +0.47 | 孤立插话漏译 |
| ≤1 s 重叠短单元漏译补全 | +0.55（+0.63） | +1.63 | +0.01 | 重叠插话漏译 |
| **≤1 s 全部** | **+0.88**（+1.34） | +1.67 | +0.49 | 短单元线的全部空间 |
| **>1 s 话语漏译补全** | **+3.89**（+4.26） | **+4.65** | +3.59 | 长话语漏译 |
| 全部漏译补全 | +4.86（+4.92） | +6.30 | +4.12 | 召回上限 |
| **欠译单元改成参考**（hyp 词数 < 0.5 × ref） | **+1.87**（+2.54） | **+4.26** | +0.84 | 部分翻译 |
| 说话人 oracle（内容感知，所有段） | +0.45（+0.95） | −0.07（+0.40） | +0.71 | 错标、归属 |
| 说话人 oracle（只改 ≤1 s 话语的段） | −0.28（−0.02） | −0.61 | −0.13 | 短单元错标（oracle 在 "Yeah." 这类句上会误判，视为 ≈ 0） |
| 删除全部插入 | −0.24（−0.39） | −0.22 | −0.23 | 插入（删掉后 BP 下降，反而更差） |
| 删除重复段 | −0.44（−0.46） | −0.60 | −0.36 | 重复（同上，BP 下降） |
| 所有有译文的单元改成参考 | +67.8 | +70.9 | +66.4 | 内容质量的天花板（只作参照） |

**参考词的分布**（C0-fp；8,340 句可计分话语，61,108 个参考词）

| 时长 | 句数 | 参考词占比 | 漏译率 | 漏掉的词占全部参考词 |
|---|---|---|---|---|
| ≤0.5 s | 1,103 | 1.9% | 24.8% | 0.48% |
| 0.5–1 s | 1,648 | 5.5% | 12.0% | 0.62% |
| 1–2 s | 2,107 | 16.8% | 5.3% | 0.88% |
| 2–5 s | 2,562 | 45.8% | 3.4% | 1.53% |
| >5 s | 920 | 30.1% | 3.5% | 1.08% |

**解读**

- **短单元的局限在于权重，而不在于漏译率。** ≤1 s 的话语占三分之一的句数，但只占 7.4% 的参考词。它们的漏译率很高（12–25%），漏掉的词却只占 1.1%。
- **长话语漏得少，但量大。** >1 s 的漏译率只有 3–5%，漏掉的词却是 3.5%，**是短单元的 3 倍以上**。欠译还有 +1.9 / ov +4.3 的上界。
- **用户的怀疑可以定量确认。** 从 v8 到 C0-fp，≤1 s 的 oracle 空间从 1.34 缩到 0.88，被拿走了 0.46，这是 +0.95 总提升中短单元份额的上界。finalpost 的干净效应 +0.39 与之一致。
- **漏译的类型**（`omission_types_v2_c0fp.json`；判定宽松，只作参考）。C0-fp 共有 702 句、2,804 个词没被召回：
  - **75% 的词，其内容在 ±2 s 内其他段的译文里能找到**：标到了别人名下、被并进别人的段，或超出了 1 s 召回窗；
  - 23% 真的没译；
  - 2% 根本没进任何窗口。
  - 但内容感知的说话人 oracle 只能救回 +0.45。说明大部分"在别处"的内容是**被合并进了别人的段**，或时间戳偏了，单靠改标签救不回来。这是 collapse 和分段问题，要靠**段级的覆盖与切分**来解决，不靠调度。

### 4.3 语音缺口的分解（C0-fp，两个种子）

| 阶段 | all | ov | 1spk | 说明 |
|---|---|---|---|---|
| 文本 tcpBLEU | 28.00 | 22.87 | 30.26 | — |
| 文本按"窗口×说话人"合并后打分（没有 TTS，也没有 ASR） | 26.12 | 19.59 | 29.13 | **打分粒度伪差**：−1.88 / −3.28 / −1.13 |
| 语音，修掉幻听长尾（oracle） | 21.22 / 21.18 | 15.22 / 15.39 | 24.00 / 23.86 | 种子 1 / 种子 2 |
| 语音 tcpASR-BLEU | 20.04 / 19.78 | 13.32 / 12.62 | 23.51 / 23.64 | 种子 1 / 种子 2 |

**粒度伪差。**
- `asr_hyp.py` 把一个窗口里同一说话人的所有段并成一段，时间跨度取 min 到 max，再交给按时间切分的分配规则。
- 重叠区里一个说话人的段常被别人的段隔开。合并后，按时间比例切分就会把词分错单元。
- 这一步完全不涉及 TTS 和 ASR，只是协议本身造成的。

**幻听长尾**（`speech_tail_*.json`）。
- ASR 词数是送入 TTS 的文本词数的 1.15–1.17 倍。
- 3.6% 的流（164–194 条）的 ASR / 文本词数比 > 1.5，贡献了多余词的 **92–93%**。短流（≤4 个 token）的比值是 1.4–2.1。
- 把这些流的转写换回原文（oracle），语音 all +1.2 到 +1.4，ov +1.9 到 +2.8。v8 参考上是 +1.95 / +2.55。
- 只看 talker 时长的检测器（每 token 语音秒数 > p97）几乎抓不到它们：只占多余词的 0.3–5%，与上面的 190 条只重合 24–32 条。所以这些流的**语音时长正常**，幻听更可能出在 Whisper 一侧，或者 TTS 在正常时长里说了错的内容。这一点需要抽样试听确认（未做）。
- 与 `diag_speech_drop.json` 一致：v8 保真度 WER 17.4%，其中 4% 的流 WER > 0.8，占错误量的 46–56%。

**剩余部分。** 合并文本 26.12，减去修掉长尾后的约 21.2，剩约 4.9 / ov 约 4.3。这才是 TTS→ASR 的常规保真度损失。

**说话人 oracle 在语音上**：all +0.71，ov +0.98（两种子平均）。

**含义**
1. 语音 all 的 8.1 个点里，至少约 3.1 个（粒度 1.9 + 长尾 1.2–1.4）是**测量协议的问题**，不是翻译或同传策略的问题。
2. 语音 ov 是所有指标里最差的（13.0）。它对文本的改进不敏感：四个系统的语音 ov 都在 12.97–13.96 之间，而文本 ov 跨了 0.9 个点。**任何只在文本上起作用的方法（包括所有 CRF 线），都很难让语音 ov 动起来。**
3. 闸门的语音噪声（±1.1，种子间最多相差 2.2）很大一部分就来自这个长尾。修好协议也会缩小噪声地板。

### 4.4 过去各次尝试动了哪些指标（`LOG.zh-CN.md` 汇总）

| 改动 | 文本 all | 语音 | 动的是哪个分量 |
|---|---|---|---|
| 解码端循环保护 | +0.92 | 不降 | 失控窗口从 4–5 个降到 1 个，重复减少 |
| GRPO 系列（W³PO、dia、w3dia、EA、ECF、HiCA） | 最好 +0.25 到 +0.33（dia 50，保护版） | 多次因语音不过闸门被判无效 | 精确率小幅上升；延迟通道曾引起漂移 |
| finalpost（最终读因果 P） | +0.69（干净效应 +0.39） | +0.10 | 短单元召回 |
| C0 / C0-fp（续训 SFT 对照） | +1.03 / +0.94 | −0.10 / +0.85 | 精确率（插入减少）；BP 下降 |
| 1′、SerialCRF v1、v2、TurnCRF | 预检停止或证伪 | — | — |

**共同点。** 所有有效改动都落在"精确率 / 插入 / 失控"这个分量上，单步 ≤ 1 BLEU。召回（尤其是长话语）、语音通道和延迟工作点都没有真正动过。

---

## 5. 每条 CRF 线能解决什么、能动多少

可检测阈值（MDE）：文本 all ≈ 0.35，ov ≈ 0.9，语音 ≈ 1.1。

| 线路 | 针对的问题 | 应该动的主指标 | 上界（本次实测） | 证据 | 现实预期 | 能否检测到 |
|---|---|---|---|---|---|---|
| **1′ 隐调度边际似然**（§10） | 提交时机、碎片 | EndOffset、tcLAAL；对 tcpBLEU 几乎没有杠杆 | EndOffset 被窗口检测器钉住（41% 的词在最终读）；tcpBLEU 对延迟不敏感 | 预检 C1 = 4.1%，后验钉在标签上 → 停止 | 文本 0，延迟 ≤ 0.01 s | 否 |
| SerialCRF v1（读内顺序） | 先写对方导致插话丢失 | ov tcpBLEU | ORDER_OMISSION 行 +0.30 / +0.90 | 事后检查推翻机制 H：顺序不影响能否写完 | ≈ 0 | 否 |
| SerialCRF v2（推迟 1 读的集合格） | 推迟后不再补写 | ov tcpBLEU | DEFERRAL +0.25 / +0.74 | 格里被推迟的条目总排第一，v8 本来就会补写 | ≈ 0 | 否 |
| **SerialCRF v2′**（训练中） | CHAIN：推迟 → 先写新条目 → `<wait>` | ov tcpBLEU、≤1 s ov 召回 | **CHAIN +0.15 / +0.45** | S2′a = 0.066，勉强过线；τ = 1 时 0.044 | all +0.0–0.1，ov +0.0–0.3；语音 0 | **否**：低于 MDE。建议照常评测，但事先承诺"未超过 MDE 就按零效应读" |
| CRF 动态 oracle（§14） | 自己前缀上的暴露偏差（CHAIN、推迟） | ov tcpBLEU | 按原设计（短 ov）≈ DEFERRAL 的 +0.25 / +0.74 | 预检 O1 人工一致率 0.865 < 0.95 → STOP | 按原设计 ≤ 0.2 | 否 |
| 动态 oracle **改换目标后**（§6 方向 3） | 所有未覆盖条目，包括长话语漏译和欠译 | all、ov、语音 | >1 s 漏译 +3.89 / +4.65；欠译 +1.87 / +4.26 | 75% 的漏译内容"在别处"，需要段级覆盖 | +0.3–1.0 / +0.5–1.5（估计，未验证） | 可能 |
| TurnCRF（§3、§12） | 错标、collapse、短插话检测 | Δspk、CR | 说话人 oracle：文本 +0.45 / −0.07，语音 +0.71 / +0.98 | CPU 阶段失败；oracle diarizer 也没有增益（P10） | ≤ +0.1 | 否 |
| 全局归一化策略头（§4） | `<wait>` 的标签偏置 | 延迟稳定性 | tcpBLEU ≈ 0 | 没做 | 固定延迟下为 0；可能有助于做延迟控制 | — |
| CRF 引导 GRPO（§5） | 整组漏掉、优势为零 | all | 参照 GRPO 历史最好 +0.33 | 没做 | ≤ +0.3 | 勉强 |
| semicrf 方案 B（HSMM 说话人轨迹平滑） | 同 TurnCRF | Δspk | ≤ +0.45 | — | ≤ +0.1 | 否 |
| semicrf 方案 C（事件触发强制输出） | 短单元 | ≤1 s 召回 | ≤ +0.88，finalpost 已拿走大部分 | — | ≤ +0.2 | 否 |
| semicrf 方案 E（拼接增强） | 短单元、重叠 | ov | ≤ +1.67（ov） | — | ≤ +0.3 | 勉强 |

**总评**
- 现有 CRF 线全部瞄准"短单元 / 顺序 / 推迟 / 标签"这一块。在文本 all 上，这一块合计的上界约 1.3（≤1 s 的 0.88 加归属的 0.45），**现实 ≤ 0.5**，在语音上为 0。
- 它们的数学是严格的，预检也做得干净，但**瞄准的误差太轻**。这是"预检指标"与"同传主指标"错位的直接后果。
- 若要保留 CRF 作为论文主线，必须把格的对象从"短单元的调度"换成**"覆盖"**（方向 3），或者换到**语音时间线**上（方向 4），或者**语音与文本的对齐**上（方向 2）。

---

## 6. 新方向（按预期影响 × 新颖度 × 可行性排序）

每条写明它针对哪个指标缺口、预期改善哪些指标，以及为什么可能新。**没有一条能保证发表。**排序依据是 §4 的实测缺口。

### 方向 1：MS-SimulEval——多说话人同传的指标体系与评测协议（必做）

- **针对：** 缺口 1 中的测量部分（粒度 −1.9 / ov −3.3；幻听长尾 −1.2 至 −1.4 / ov −1.9 至 −2.8），以及"没有曲线、没有 CA 延迟、没有 SIM-O"。
- **做法：**
  1. **段级语音打分。** 按块（每个块都有 `start_ms` 和 `dur_ms`）分别做 ASR，或者对整条流做强制对齐后把词分回各段，让语音的分配与文本相同。
  2. **对幻听稳健的 ASR 协议。** 例如对 ASR / 文本长度比异常的流用第二个 ASR 复核，或者报告时剔除这些流。参考方与候选方要用同一规则。
  3. **补齐延迟和说话人指标。** 同时报告 tcLAAL、StartOffset、EndOffset（文本与语音）、积压、输出重叠率、Δspk、SIM-O，以及单流计时的 CA 延迟。
- **预期：** 不改模型，语音分数中约 3 BLEU 的测量伪差被修掉；种子噪声变小，MDE 下降；结果能与 SimulS2ST-Omni、Hibiki、StreamSpeech 的指标对齐。
- **新颖性：** §1.3 的检索里没有任何工作在延迟约束下评测说话人归属的语音翻译。DiariST 有 SAtBLEU，但没有延迟和语音。可以作为论文的评测贡献，也可以单独投 Interspeech / ICASSP。
- **可行性：** 高。CPU 部分本次已完成原型；段级 ASR 需要 GPU 重跑 ASR，用户批准后再做。
- **注意：** 协议改变后，闸门的"语音不降"要在新协议下对参考重新测量。这需要用户决定。

### 方向 2："说你写的"：用单调对齐边际约束流式 talker

- **针对：** 缺口 1 的真实部分。去掉伪差后，语音仍比文本低约 4.9 / ov 4.3；还有幻听长尾。
- **做法：** 对每条流，talker 的输出（语音码或 ASR 的 CTC 后验）与已提交文本之间是一条单调对齐。用 HMM / CTC 前向算精确边际 log p(text | speech)，用在两个地方：
  - **(a) 推理期：** 抽 k 个样本，按对齐边际选最好的一个。talker 已经有 `best_of` / `select_k` 字段，见 `uniss_qwen3_omni_v1/evaluation/si_infer_cont.py`。选择器**必须**用与评测 ASR 不同的模型（例如 CTC 声学模型），避免对评测器过拟合。
  - **(b) 训练期：** 把对齐边际作为 talker 的一致性损失。
- **预期：**
  - (a) 只改推理。长尾修复的 oracle 上界是 +1.2–2.0 all / +1.9–2.8 ov；选择器若能抓到一半，约 +0.6–1.0 / +1–1.4。
  - (b) 若能把常规保真度 WER 降低三分之一，估计语音再 +1–2。
  - 这是唯一可能在语音 ov 上超过 MDE 的方向。
- **新颖性：** 中等。码本 TTS 里已有单调对齐的做法，例如 Glow-TTS 的 MAS，以及 VALL-E R、ELLA-V 一类（**本文未核实，写作前要查**）。新的地方在于：流式、分块提交、多说话人分流，而且优化目标是 ASR-BLEU。
- **可行性：** (a) 中高；(b) 中等，需要 talker 训练代码。

### 方向 3：覆盖格动态 oracle（CRF 主线改换目标）

- **针对：** 缺口 2。>1 s 漏译上界 +3.89 / +4.65；欠译 +1.87 / +4.26；BP 0.942。还有 §4.2 的观察：75% 的漏译内容"在别处"，是合并或分段错误。
- **做法：** 把 §14 的残差格从"短条目的何时写"推广为**"覆盖格"**。
  - **状态：** 每个 diarized 源话语的覆盖程度，可以是未覆盖、部分覆盖或已覆盖，并允许"拆分已写的合并段"。
  - **前向：** 在模型自己的前缀上，对所有补全方式精确求和。
  - **训练：** 用后验边际作为 DAgger 的软目标。
  - 覆盖判定必须改成**语义 / 对齐式**，不能用 O1 失败的词面规则。例如用离线翻译模型对齐，或者 chrF 加时间。O1 的人工一致率 0.865 是这条线首先要过的关。
- **预期：** 上界很大，现实不确定。估计文本 all +0.3–1.0，ov +0.5–1.5。语音 1spk 会随之上升；语音 ov 要等方向 1、2 修好才看得出来。
- **新颖性：** 高。在 on-policy 状态上，用 CRF / HMM 残差格的精确后验作为动态 oracle；最接近的是 OCD（ICLR 2019）和 Cross & Huang（EMNLP 2016），见 §14.7。把覆盖（Tu et al. 2016 的 NMT coverage 的结构化版本）加进多说话人集合状态，也没检到先例。
- **可行性：** 中等。残差格的代码已经通过 O3 的精确性检验，要重写的是覆盖函数。先做 CPU 人工一致率预检，**重新预登记**。

### 方向 4：输出时间线调度 CRF（听众侧同传）

- **针对：**
  - 语音 EndOffset p90 6.2 s；
  - 33% 的窗口有积压，p90 约 3 s；
  - 17.5% 的输出语音时间里有两条流同时发声。
  - 这些都是听众真正感受到的延迟和可懂度问题，现有协议看不到。
- **做法：** 给定 Thinker 提交的段（说话人、文本、提交时刻），在离散的播放起点和语速档（talker 有速度参数）上建 semi-Markov CRF / 动态规划：
  - **目标：** 最小化 E[EndOffset] + λ·输出重叠 + μ·积压；
  - **约束：** 每个说话人的段保序；语速不超过可懂度上限；
  - **推断：** 精确 Viterbi 或边际。可以先只在推理期使用，再把 Thinker 的分块偏好作为势函数接入。
- **预期：**
  - 语音 EndOffset p90 和积压明显下降（预计 30–50%，没验证）；
  - 输出重叠接近 0；
  - 在现行"只转写语音段"的 ASR-BLEU 下约为 0，所以必须加上**时间线混音后的可懂度**（例如在混音音轨上做说话人分离后 ASR，或者人工听测）；
  - 语速加快可能损害保真度，必须过 ASR-BLEU 硬闸门。
- **新颖性：** 高。同传里的等时（isochrony）工作只针对单说话人，例如 Tam et al., Interspeech 2022（只在检索结果中看到）。多说话人同传的输出调度没有检到先例。是严格的 CRF。
- **可行性：** 高。纯 CPU 的 DP 加 talker 的速度参数。

### 方向 5：说话人路由的结构化注意力（TurnCRF 的新用法）

- **针对：** 缺口 3。ov 内容质量比 1spk 低 7.4，补全所有漏译后仍低 5.2；说话人 oracle 对 ov 约为 0，瓶颈在**重叠源上的内容识别**。
- **做法：** 不再用 CRF 改标签。把轮次 CRF 的边际作为**音频帧上的结构化注意力掩码 / 偏置**（Kim et al., ICLR 2017），让说话人 k 的段主要关注 k 的语音帧，相当于在 LLM 内部做目标说话人条件的"软分离"。
- **预期：** 如果成功，文本 ov +0.5–2。这是唯一直接针对缺口 3 的方向。
- **新颖性：** 中高。目标说话人 ASR 已有很多工作（SpeakerBeam 一系；基于 LLM 的 TS-ASR 需要检索，**未核实**）。"CRF 边际作为 LLM 内部的说话人路由注意力"在同传里没有检到。
- **可行性：** 中低。要改 Qwen3-Omni 的注意力，重训 LoRA。§12 的教训是：按帧的最大似然会抹掉短轮次，要按轮次计分。

### 方向 6：单模型多延迟，给出质量-延迟曲线

- **针对：** 只有单点，无法与 SimulS2ST-Omni（m1–m12）、StreamSpeech（chunk 大小）、HMT（L、K）比较。
- **做法：**
  - 训练时加延迟档 token，例如"每 m 次读才允许写"，或把延迟目标作为条件；
  - 也可以把 1′ 的隐调度格加上延迟约束的对偶（§10.3）来生成各档的软目标；
  - 窗口检测的关闭延迟（1.7 s）要作为一个可调档位一起扫。
- **预期：** 固定延迟下文本 BLEU 不变，得到曲线，延迟轴可以下探到 EndOffset 约 1 s。
- **新颖性：** 低，SimulS2ST-Omni 已经这样做。但这是投稿的**必需品**。
- **可行性：** 高。

### 推荐的论文主线

"多说话人同传：该测什么、损失在哪里"，用方向 1 加 §4 的分解。方法部分用方向 3（结构化覆盖的动态 oracle）或方向 4（输出时间线 CRF），再加方向 2 作为语音侧的工程贡献。方向 6 用来给出曲线。现有的 v2′ 和 SerialCRF 可以作为"短单元调度不是瓶颈"的消融证据。

---

## 7. 参考文献核对

由一个子代理在 2026-11-10 联网核对；arXiv 元数据用 export.arxiv.org API 查过。标"未核实"的是只看到检索摘要，或者没能打开原文。

### 7.1 对 `crf_novel_ideas.zh-CN.md` §8 与正文的更正

| 原文 | 更正 / 状态 |
|---|---|
| "Xiang & Ou. CTC-CRF ICASSP 2019; CAT toolkit arXiv 1911.08747" | CTC-CRF 原文："CRF-based Single-Stage Acoustic Modeling with CTC Topology"，ICASSP 2019, pp. 5676–5680。**CAT 的作者是 An, Xiang, Ou**（Keyu An 为第一作者），只在 arXiv 上 |
| "Arivazhagan et al. MILk (DAL)" | 正确：DAL 在 MILk §4.2 中提出（arXiv 1906.05218，ACL 2019）。Cherry & Foster, "Thinking Slow about Latency Evaluation for Simultaneous Machine Translation"（arXiv 1906.00048，只在 arXiv 上）给出了 DAL 的分析，建议一起引用 |
| "Moriya et al. 2024, arXiv 2409.20301" | 标题正确。arXiv 只写 "Submitted to ICASSP 2025"，**录用未核实**，按 arXiv 2024 引用 |
| "Johnson & Willsky. HDP-HSMM. UAI 2010. arXiv 1203.3485" | 正确，UAI 2010。JMLR 版是另一篇："Bayesian Nonparametric Hidden Semi-Markov Models"，JMLR 14:673–701, 2013，arXiv 1203.1365 |
| "LSG（2025）" | Guo, Zhang, Ma, Feng, "Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation"，**AAAI 2025**，arXiv 2501.00868 |
| "EMNLP 2021 用完整 NMT 构造读写 oracle" | Alinejad, Shavarani, Sarkar, "Translation-based Supervision for Policy Generation in Simultaneous Neural Machine Translation"，EMNLP 2021, pp. 1734–1744（aclanthology.org/2021.emnlp-main.130）。前身是 Zheng et al., "Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation"，EMNLP-IJCNLP 2019，arXiv 1909.01559 |
| Goyal et al. NAACL 2019 | 正确。全名："An Empirical Investigation of Global and Local Normalization for Recurrent Neural Sequence Models Using a Continuous Relaxation to Beam Search" |
| Sortformer ICML 2025 | 正确。全名："Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems" |
| Fujita EEND | 是两篇："End-to-End Neural Speaker Diarization with Permutation-Free Objectives"（Interspeech 2019，arXiv 1909.05952），"...with Self-Attention"（ASRU 2019，arXiv 1909.06247） |
| DiariST 的指标 | 论文用的是 **SAgBLEU / SAtBLEU**，不是 cpBLEU 或 SAcpBLEU |
| HMT 的数字 "29.29（AL 3.46）" | 与论文 De→En L3K6 一致。"只取最大状态 24.02（AL 1.77）"**本次未核实** |

### 7.2 核对无误的

- Variani et al. GNAT（NeurIPS 2022，2205.13674）
- Kim et al. Structured Attention（ICLR 2017，1702.00887）
- Rush, Torch-Struct（ACL 2020 Demo，2020.acl-demos.38）
- Zheng et al. ACL 2019（1906.01135；作者 Zheng, Zheng, Ma, Huang）
- DiariST（ICASSP 2024，2309.08007）
- Kanda et al. Interspeech 2022（2203.16685）
- LUFFY（NeurIPS 2025，2504.14945）
- Levine（1805.00909，只在 arXiv 上）
- Ganchev et al.（JMLR 11, 2010）
- Andor et al.（ACL 2016，P16-1231）
- Sarawagi & Cohen（NIPS 2004）
- Lafferty et al.（ICML 2001）
- StreamSpeech（ACL 2024）

§14.7 的 Goldberg & Nivre（COLING 2012 / TACL 2013）、Cross & Huang（EMNLP 2016）、DAgger（AISTATS 2011）、SEARN（MLJ 2009）、LOLS（ICML 2015）、OCD（ICLR 2019）、Scheduled Sampling（NeurIPS 2015）：本次**没有重新打开原文**，按记忆与 §14.7 一致，写论文前要逐条核对。

### 7.3 本文新引用的（核实情况见 §1）

| 文献 | 出处 | 核实情况 |
|---|---|---|
| SimulS2ST-Omni | arXiv 2607.19810 | 已读 abs 与 html |
| SimulEval | ACL Anthology 2020.emnlp-demos.19；文档与代码 `latency_scorer.py` | 已核实 |
| LAAL | arXiv 2206.05807 | 已核实 |
| Seamless | arXiv 2312.05187 | 已核实 |
| Hibiki | arXiv 2502.03382 | 已核实；ICML 2025 只在检索结果中看到 |
| Hibiki-Zero | arXiv 2602.11072 | 会议未核实 |
| NAST-S2X | arXiv 2406.06937 | 已核实 |
| Seed LiveInterpret 2.0 | arXiv 2507.17527 | 已核实 |
| StreamUni | arXiv 2507.07803 | 已核实 |
| MeetEval | arXiv 2307.11394 | 已核实 |
| CHiME-6 | arXiv 2004.09249 | 已核实 |
| SOT | arXiv 2003.12687 | 已核实 |
| t-SOT | arXiv 2202.00842 | 已核实 |
| ATD（Kano et al., Interspeech 2023） | arXiv 2211.13173 | 只在检索结果中看到 |
| StreamAtt / StreamLAAL | arXiv 2406.06097 | 只在检索结果中看到 |
| X-Translator | arXiv 2607.17544 | 已核实 |
| Spatial Speech Translation（CHI 2025） | arXiv 2504.18715 | 已核实 |
| Moshi | arXiv 2410.00037 | 已核实 |
| SimulPL | arXiv 2502.00634 | 只在检索结果中看到 |
| SimulS2S-LLM | arXiv 2504.15509 | 只在检索结果中看到 |
| Tu et al. 2016 NMT coverage | — | 按记忆，未核实 |
| VALL-E R、ELLA-V、Glow-TTS MAS | — | 未核实 |

---

## 8. 复现与文件

```
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
D=eval_outputs/mist_w3po_v1/decode; S=eval_outputs/mist_w3po_v1/speech; O=eval_outputs/mist_w3po_v1/metrics_headroom_v1
# 主分析：oracle 上界、参考词分布、文本与语音延迟、语音粒度、语音说话人 oracle
$PY -m experiments.mist_w3po_v1.evaluation.metric_headroom --name c0fp --run $D/dev_crf_c0fp_s3000_finalpost_s8 \
    --speech-dirs $S/dev_crf_c0fp_s3000_finalpost_s8/talker $S/dev_crf_c0fp_s3000_finalpost_s8_tts2/talker \
    --asr-hyps $D/dev_crf_c0fp_s3000_finalpost_s8/hyp_asr.jsonl $D/dev_crf_c0fp_s3000_finalpost_s8_tts2/hyp_asr.jsonl \
    --out $O/c0fp.json          # 另有 v8ref（种子 2 用 _tts2b）、finalpost、c0
# S1′ 各类的精确上界
$PY -m experiments.mist_w3po_v1.evaluation.metric_headroom_rids --run $D/dev_v8_ctx0_guard_finalpost_s8 \
    --rows eval_outputs/mist_w3po_v1/crf/serial_v2p_precheck/s1/s1_rows_primary_finalpost.jsonl --out $O/rids_finalpost_s1p.json
# 漏译类型（宽松判定）
$PY -m experiments.mist_w3po_v1.evaluation.metric_omission_types --run $D/dev_crf_c0fp_s3000_finalpost_s8 --out $O/omission_types_v2_c0fp.json
# 语音幻听长尾
$PY -m experiments.mist_w3po_v1.evaluation.metric_speech_tail --run $D/dev_crf_c0fp_s3000_finalpost_s8 \
    --speech $S/dev_crf_c0fp_s3000_finalpost_s8 --out $O/speech_tail_c0fp_s1.json
```

- 每个主分析约 1–3 分钟，单进程 CPU。
- `omission_types_{c0fp,v8ref}.json` 是第一版：近邻判定太宽，几乎全部归为 OTHER_SPK，已被 v2 取代，保留不删。

**局限**
- 所有"补全"都是放入参考原文的上界，现实收益更小。
- 说话人 oracle 用 chrF 选说话人，在 "Yeah."、"Okay." 这类短回应上会误判，所以"只改 ≤1 s 段"那一行为负值，不可信。
- 漏译类型的判定较宽，"在别处"的 75% 是上界。
- 幻听来源（ASR 还是 TTS）没有试听确认。
- tcLAAL 是本文定义的单元级 LAAL，和文献的句级或流级 LAAL 只能粗比。
