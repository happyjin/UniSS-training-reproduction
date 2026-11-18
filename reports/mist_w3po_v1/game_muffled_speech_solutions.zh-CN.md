# 电竞同传译音"闷"：解决方案、是否需要训练、TSE 的作用与可发表的创新点

2026-11-18，布拉格时间约 23:30。

原因分析见 `reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md`。本文回答四个问题：
- 有哪些解决办法？
- 需不需要训练？
- TSE（目标说话人提取）能不能解决？
- 有没有人研究过？有哪些可以发顶会的新想法？

---

## 0. 简短回答

| 问题 | 回答 |
|---|---|
| 有哪些办法 | 分三档。**免训练**：给提示去背景、挑最干净的片段并锁定音色。**轻训练**：只训一个小的"音色 token 去噪器"，不动 Talker。**重训练**：用加噪提示训练 Talker，或用质量条件加 CFG 引导、强化学习 |
| 需要训练吗 | **不一定。** 免训练的"提示去背景"已经实测有效：PESQ 中→英 +0.94，英→中 +0.66，86% 的语音流变好。但它离会议数据的水平还有差距（英→中 2.46，会议 3.17），而且依赖较重的音源分离。要更好、更省，建议训练一个**轻量的音色 token 去噪器**，几个 GPU 小时即可，**Talker 不用重训** |
| TSE 能解决吗 | **只能解决一部分。** 主要干扰是 BGM、观众和音效这类**非语音**声音，这属于语音增强和音源分离的问题。TSE 擅长的是**其他解说的串音**和游戏内语音这类"语音型"干扰。常规 TSE 还需要目标说话人的注册语音，有"先有鸡还是先有蛋"的问题，输出的波形也带伪影。更好的做法是**用分离器后验当线索，直接提取"目标说话人的音色 token"，不重建波形** |
| 有人研究过吗 | **噪声鲁棒的零样本 TTS 有不少工作**：DINO-VITS，微软的 flow-matching 噪声鲁棒性研究，NR-LauraTTS 的离散 token 去噪，IDEA-TTS 的环境解耦。**S2ST 领域**，Meta 的 DINO-PRETSSEL（ACL Findings 2024）明确指出嘈杂输入会把背景噪声泄漏进译音；Hibiki 用 CFG 提升说话人相似度。**但"多说话人 + 同传 + 广播背景 + 分离器条件 + 定长音色 token 瓶颈"这个组合，我们检索到的文献里没有**。**第二轮查新更正（§7）**：NTT 的 Guided Speaker Embedding（ICASSP 2025）已经能"用活动标签从重叠语音里直接提取目标说话人嵌入"，所以 §3.2 F 和 §4.1 不能把"提取器本身"当作创新。ICLR 级的新 idea 以 §7 为准 |

---

## 1. 动机：为什么值得做，而且现在能做

### 1.1 问题真实存在，而且有代表性

**场景很大：**
- 电竞、体育直播、发布会、带 BGM 的视频，原声几乎都是"人声 + 持续背景"的广播混音。
- 会议数据（AMI、AliMeeting）的背景很安静，所以现有多说话人 S2ST 的评测掩盖了这个问题。

**我们的实测：**
- 电竞提示的背景声比会议高 16–20 dB，译音估计 PESQ 低约 1.5（1.6–1.9 对 3.2–3.3）。
- 背景越响的段越"闷"。最差的是 KPL 总决赛现场那段：背景只比人声低 6.8 dB，PESQ 1.32。

### 1.2 关键发现：问题被压缩进 32 个 token

这是方案设计的出发点。**Talker 只通过一个地方"看到"音色提示**：BiCodec 说话人编码器输出的 32 个 global token（`si_infer.py`：`it["voice"] = encode_wav_to_tokens(prompt)[:32]`）。

- 之前的换提示实验，只改了提示，其他输入完全相同：源语音的 GLM token、译文、随机种子都一样。所以两组之间**所有差异都来自这 32 个 token**。
- 2×2 拆分显示，这 32 个 token 同时污染了两条路径：
  - BiCodec 解码器：PESQ 效应 +0.43 / +0.30；
  - Talker LM 生成的语义码：PESQ 效应 +0.49 / +0.39。
- 推论：**只要把这 32 个 token 修干净，两条路径同时修好。** 这不再是"波形去噪"，而是"32 个离散 token（或其量化前的连续向量）的去噪"。这个问题小得多，便宜得多，而且天然适合流式：
  - 每位说话人只需估计一次，之后可以缓存；
  - 不在译音生成的关键路径上，不增加延迟；
  - 不重建波形，也就没有分离伪影。

### 1.3 现在能做

- **数据现成。** kenxsong 的电竞数据有约 1000 小时真实解说原声。用 HDemucs 分离出的非人声部分，就是大量真实的"电竞背景"，可以直接拿来造训练对。
- **工具链现成。** SQUIM 和 HDemucs 已下载；配对换提示和 2×2 拆分的脚本已写好；Talker、分离器和评测编排器都在。
- **钩子现成。** `si_infer.py` 已有 `BayesVoiceMemory`，它在**量化前的说话人向量**上做精度加权的后验。可以直接把质量加权或去噪器接进去。

---

## 2. 相关工作：有人研究过吗

| 方向 | 代表工作 | 做了什么 | 与我们的差别 |
|---|---|---|---|
| 噪声鲁棒零样本 TTS：训练鲁棒的说话人编码器 | DINO-VITS（arXiv 2311.09770） | 用 DINO 自蒸馏让说话人编码器抗噪 | 单说话人 TTS，提示里是普通噪声；没有多说话人和同传 |
| 噪声鲁棒零样本 TTS：数据和训练策略 | 微软，flow-matching 零样本 TTS 的噪声鲁棒性（arXiv 2406.05699） | 掩码去噪预训练、多说话人检测、DNSMOS 过滤、随机混噪微调。报告比"先对提示做 SE"更好 | 同上 |
| **token 层面去噪** | **NR-LauraTTS**（arXiv 2505.13830，Interspeech 2025） | 在离散声学 token 上去噪，只恢复携带说话人信息的前两组 token。说话人相似度优于 SE 前端 | **思路最接近我们的 1.2 节。** 差别：它是逐帧的声学 token，我们是**定长 32 个的全局说话人 token**；它是单人干净目标，我们是**多说话人、需要分离器线索、流式** |
| 环境与说话人解耦 | Environment-Aware TTS（arXiv 2110.03887）、IDEA-TTS（arXiv 2412.16977） | 分离出说话人嵌入和环境嵌入，可以把环境换掉或保留 | 单说话人 TTS；可借鉴来做"保留广播感"的变体（4.3 节） |
| **S2ST 的噪声泄漏** | **DINO-PRETSSEL**（Meta，arXiv 2406.02733，ACL Findings 2024） | 指出嘈杂录音会让表现力 S2ST 的背景噪声泄漏进译音；用 DINO 自蒸馏和加噪训练表现力编码器 | **与我们的现象最接近。** 差别：单说话人、非同传；需要重训生成器 |
| S2ST 的音色控制 | Hibiki（Kyutai，arXiv 2502.03382） | 按说话人相似度分档打标签训练，推理时用 CFG 提高相似度 | 可借鉴来做"干净度 CFG"（3.3 节 H） |
| 多编码任务的语言模型 | SpeechX（arXiv 2308.06873） | 一个 codec LM 同时做降噪、TSE、零样本 TTS | 不是同传，也不针对提示的音色 |
| 多说话人同传 / 翻译耳机 | Spatial Speech Translation（arXiv 2504.18715）、X-Translator（arXiv 2607.17544） | 先分离说话人再翻译，保留方位和音色。前者发现分离残留会进入译音，于是在分离器输出上微调 | 关注分离和方位，没有分析"背景经音色瓶颈泄漏"这一机制 |
| 用分离器信息做目标说话人处理 | TS-SEP（arXiv 2303.03849）、**DiCoW**（BUT，arXiv 2501.00114） | 用分离器的活动信息代替注册语音，做目标说话人提取或识别 | **线索的形式可以直接借来**：DiCoW 的四类掩码（静音 / 目标 / 非目标 / 重叠）。我们是把它用在音色 token 上，而不是 ASR |
| 配音中的背景保留 | 先分离、配音、再混回（工业通行做法）；JUST-DUB-IT（arXiv 2601.22143） | 后者批评"分离后混回"在时长变化时和环境声不同步 | 同传的背景是实时的，没有对齐问题，可以直接混回（4.3 节） |
| 嵌入层面的说话人增强 | Extended U-Net for SV（arXiv 2206.13044）等 | 在说话人确认任务里增强嵌入 | 目标是识别，不是生成 |

**结论：**
- "噪声提示会损害零样本 TTS"已经有较充分的研究。"在 token 上去噪比在波形上去噪好"也有 NR-LauraTTS 的先例。
- 但以下四点，我们检索到的文献里没有看到：
  1. **多说话人同传场景下的机制分析**：背景经 32 个音色 token 同时污染 LM 和解码器，两条路径大约各占一半；
  2. **以分离器后验为线索、不需要注册语音、在音色 token 空间里做目标说话人提取**；
  3. **广播类真实数据（电竞）上的评测**；
  4. **和同传系统的延迟、ASR-BLEU 一起报告。**
- 检索覆盖不全，投稿前需要再做一次系统的文献检索。

---

## 3. 解决方案（按投入从小到大）

### 3.1 第 0 档：免训练（1–2 天，可以马上用到试听页）

**A. 提示去背景（已验证）**
- 每位说话人的提示先过一遍音源分离，只留人声，再编码成 32 个 token。
- 提示取自已经听过的过去音频，所以**不增加前瞻延迟**；每个说话人处理一次后缓存。
- 实测：PESQ +0.94 / +0.66，SI-SDR +7.7 / +5.8 dB，频谱对比度 +2.2 / +1.7 dB。
- **局限：**
  - 音乐分离模型会把**观众的喊声和解说声一起归入"人声"**，所以去不干净。这是我们的推测，英→中只恢复到 2.46 与它相符。
  - HDemucs 偏重，流式部署需要换成更轻的模型。
  - 分离本身会带来轻微失真。

**B. 挑最干净的片段，锁定音色**
- 现在的做法是取首次输出之前**最近的** 8 s 单人片段。改为：在所有已经听到的单人片段里，按估计质量（SQUIM / BVR / DNSMOS）排序，取最干净的 6–8 s。之后如果出现更干净的片段再更新，否则保持不变。
- 同一个分离器槽位跨窗口复用同一组 token，避免同一个人在不同窗口里音色漂移。
- 一个小创新：`BayesVoiceMemory` 现在用的是精度加权后验，可以**把估计 SNR 作为观测精度**，即"质量加权的音色后验"。

**C. A 和 B 一起用**
- 先选最干净的片段，再去背景。预计效果介于 A 和"干净提示"之间，需要实测。

**D. 播放端（不是"闷"本身的问题，但影响整体听感）**
- 多路译音大量重叠，可以用 MIST 的播放层（PA）处理。
- 可选：把分离出的背景声（BGM 和观众）按较低电平垫在译音下面，让译音听起来"在同一个现场"，减少"干、闷"的对比感（4.3 节）。

### 3.2 第 1 档：轻训练，不动 Talker（推荐主线，约 1 周）

**E. 音色 token 去噪器（Voice-Token Denoiser，VTD）**
- **输入**：带噪提示的 mel 频谱，或 BiCodec 说话人编码器的中间特征。
- **输出**：干净的量化前说话人向量（再经 FSQ 得到 32 个 token），或者直接预测 32 个 token 的类别。
- **训练数据**（可以无限合成）：
  - 干净语音：AISHELL-3、LibriTTS-R、VCTK 等干净多说话人语料；
  - 电竞背景：**用 HDemucs 从电竞原声中分离出的非人声部分**，加上观众声和其他解说的串音；
  - 混合 SNR 在 −5 到 20 dB 之间，按真实电竞提示的 BVR 分布采样，中位数约 −15 dB。
- **教师**：BiCodec 说话人编码器在**干净语音**上的输出。**学生**：用同一个编码器初始化，输入**带噪语音**。
- **损失：**
  - 量化前向量的 L2 和余弦；
  - token 的交叉熵；
  - **解码一致性**：固定语义码，分别用预测的 token 和干净的 token 解码，比较 mel 距离。这样优化的是"听起来一样"，而不只是向量接近；
  - 可选：用 WavLM 说话人相似度防止音色漂移。
- **规模**：一个小 Conformer 或在原编码器上微调，几个 GPU 小时。
- **为什么值得：**
  - 一次修好两条路径（1.2 节）；
  - 不重建波形，所以没有分离伪影；
  - 比 HDemucs 轻得多，每个说话人只算一次；
  - **Talker 不用重训。**

**F. 以分离器为线索的目标说话人音色提取（DiCo-Voice）：E 的多说话人版本，也是 TSE 的推荐用法**
- 输入是**未经挑选的混合音频**（包括重叠段和别人说话的段），加上目标槽位的分离器后验，按 DiCoW 的做法编码成四类掩码：静音 / 目标 / 非目标 / 重叠。
- 输出目标说话人干净的 32 个 token。
- **解决的问题：**
  - 5–6% 的语音流因单人片段不足 1 s 而回退到"整窗音频含别人"，F 可以直接从混合音频里提取；
  - 分离器把同一个人拆成两个槽位、或把别人混进来时，F 可以利用更多音频；
  - 首次输出前能用的单人片段很少时，F 能更早得到可靠的音色，对同传的首句尤其重要。
- 训练数据：在 E 的数据基础上，再按分离器的输出格式混入其他说话人（2–4 人，按真实重叠率）。

### 3.3 第 2 档：训练 Talker（需要重训 T1c，数天到数周）

**G. 加噪提示训练**
- 训练时用"同一说话人的另一句话加电竞背景"作提示，目标是干净语音。这是 DINO-PRETSSEL 和微软 flow-matching 工作的通用做法。
- 优点：LM 自己学会忽略环境。缺点：要重训，而且只解决 LM 那一半；解码器那一半仍取决于 token 是否干净，还需要配合 E。

**H. 干净度条件加 CFG（借鉴 Hibiki）**
- 训练数据按"提示干净度"和"目标干净度"打档（用 SQUIM 或 DNSMOS），训练时随机丢弃这个条件。
- 推理时用 CFG 把生成推向"干净"，同时保留说话人条件。
- 可以做成**双重引导**：说话人相似度一路，干净度一路，分别调节，在"像不像"和"干不干净"之间取舍。文献中反复出现的规律是：去噪越狠，说话人相似度越容易掉。

**I. 强化学习（项目里已有 GRPO）**
- 奖励 = 质量（SQUIM / UTMOS）+ 说话人相似度 + ASR-BLEU。
- 风险：奖励被钻空子，例如合成出"很干净但不像本人"的嗓音。说话人相似度和 ASR-BLEU 必须是硬约束。

---

## 4. 可以发顶会的创新点（建议主打 4.1，4.2 和 4.3 二选一作为加分项）

### 4.1 主打：Voice-Bottleneck Denoising for Multi-Speaker Simultaneous S2ST

> **查新更正（2026-11-18，第二轮检索）：** 本节的"方法"一条与以下工作高度重叠，不能再声称新：
> - Guided Speaker Embedding（Horiguchi et al.，ICASSP 2025）：用目标说话人的活动标签，从重叠语音直接提取目标说话人嵌入，不重建波形；
> - 师生式混合语音嵌入（arXiv 2306.00634）；
> - Unmixing the Crowd（arXiv 2604.03219）；
> - NeurIPS 2025 的"带噪正负注册对比 TSE"（arXiv 2502.16611）。
>
> 仍然成立的部分：机制分析（2×2）、广播场景的评测、面向**生成**而不是识别的目标函数。改进后的版本见 §7 的 idea 3（面向生成器的条件去噪）。§7 的 idea 1 和 idea 2 才是建议主打的 ICLR 方向。

**动机**：广播类场景的同传，音色提示必然含背景。通过分析发现：背景**只经过一个 32 token 的音色瓶颈**，却同时污染 LM 和声码器。

**贡献：**
1. **机制分析（新）**：用配对换提示和 2×2 因子拆分，量化背景经"音色 token 进入 LM 生成"和"音色 token 进入解码器"两条路径的贡献，各约一半，还有正的交互项。这一类因果分析在 TTS 和 S2ST 文献中少见。
2. **方法（新）**：以分离器后验为线索、不需要注册、流式的**目标说话人音色 token 提取**（3.2 节的 F）。它把 TSE 从"波形"搬到"音色 token"上。
   - 与 NR-LauraTTS 的区别：全局音色 token、多说话人、有分离器线索、流式。
   - 与 DiCoW 的区别：用于生成，而不是识别。
3. **评测（新）**：真实电竞广播（中英双向），加上会议数据（AMI / AliMeeting），再加上"干净会议 + 电竞背景"的受控合成集（有干净参考，可以算说话人相似度）。指标包括：
   - 无参考质量：SQUIM、DNSMOS、UTMOS；
   - 频谱纹理：对比度、平坦度；
   - 说话人相似度；
   - **ASR-BLEU 和 LAAL 不得变差**（同传系统的硬约束）。

**预期**：第 0 档的上限已经实测到 +0.66 到 +0.94 PESQ。去噪器的目标是不用重型分离器就达到或超过这个水平，并在多说话人重叠时更稳。

**投稿**：Interspeech / ICASSP 最匹配。如果和 MIST 的同传系统合成一个完整故事（广播场景的多说话人同传），可以考虑 ACL / EMNLP 的语音翻译方向，或 TASLP。

### 4.2 加分项 A：身份与干净度的双重 CFG（Identity–Cleanliness Dual Guidance）

- Talker 同时以说话人 token 和干净度标签为条件，推理时用两路 CFG 分别控制。
- 回答一个实际问题：去噪是否会"洗掉"说话人特征？给出可调的取舍曲线。
- 需要重训 Talker（3.3 节的 H），工作量较大。适合在 4.1 之后做成扩展版（期刊或第二篇）。

### 4.3 加分项 B：保持现场感的同传（Environment-Consistent Simultaneous Dubbing）

- 把源音频分成"各说话人人声"和"背景层"两部分：人声去噪后生成译音，再把**原背景层**按响度规则（类似广播里的自动压低背景）混回去。
- 进一步可以借鉴 IDEA-TTS 的环境嵌入，给干净的译音加上与原广播一致的空间和 EQ 特征。
- 动机：用户听感上的"闷"，有一部分来自"原声是现场广播，译音是干巴巴的另一种质感"的反差。保持现场感对电竞和体育直播的产品体验很关键。
- 同传天然没有"译音变长导致与环境声不同步"的问题（JUST-DUB-IT 对配音的批评不适用于实时背景）。
- 评测需要主观听测（MUSHRA / ABX）。

---

## 5. TSE 能不能解决？详细分析

**提示里的干扰分五类，各自适合的工具不同：**

| 干扰 | 例子 | 音乐分离（HDemucs 等） | 常规 TSE（SpeakerBeam、WeSep 等） | 分离器条件的 token 提取（F） |
|---|---|---|---|---|
| BGM / 音乐 | 赛事配乐 | 好 | 一般（常规 TSE 训练集里很少有音乐） | 好（训练数据里加入） |
| 观众 / 欢呼 | 现场人声起伏 | **差**（会被归入人声） | 一般（像多人说话的嘈杂声） | 好（训练数据里加入） |
| 游戏音效和游戏内语音 | 技能音效、击杀播报 | 音效好，**语音差** | 播报语音：好 | 好 |
| 其他解说串音 | 重叠、分离器错误、回退提示 | 差 | **好**（这正是 TSE 的本职） | **好**（线索来自分离器） |
| 广播处理 | 压缩、EQ | 不能 | 不能 | 部分可以（解码一致性损失学的是"干净音色"） |

**结论：**
1. **单靠 TSE 不够。** 主要干扰是非语音（BGM、观众、音效），常规 TSE 不是为这个设计的。
2. **TSE 加上音乐分离可以解决大部分。** 但是：
   - 常规 TSE 需要注册语音，而我们恰恰缺一段干净的注册语音。用分离器线索的 TS-SEP 或 DiCoW 式做法可以绕开这一点。
   - 两级波形处理会叠加伪影和计算量。
   - 最终只需要 32 个 token，重建整段波形是多余的。
3. **推荐**：把 TSE 的思想（目标说话人、分离器线索）用在**音色 token 空间**（3.2 节的 F）。实验中保留"音乐分离"和"音乐分离 + 波形 TSE"作为基线，以说明 token 空间方法的优势。
4. **可以马上做的 TSE 基线**：
   - (i) HDemucs 人声（已有）；
   - (ii) WeSep 或 TD-SpeakerBeam，用"最干净的单人片段"做注册；
   - (iii) TS-SEP 用分离器活动做线索。
   - 三者都只处理提示，不影响主链路。

---

## 6. 实验计划与门槛（沿用项目惯例：先登记、dev / test 分开、不在测试集上做选择）

### 6.1 数据划分
- 电竞视频按视频切分为训练、dev、test。**已发布的 10 段试听视频只能放在 test。**
- 训练去噪器用的"背景"只取自训练集视频。
- AMI / AliMeeting 按原有的划分使用。

### 6.2 指标

| 类型 | 指标 |
|---|---|
| 主指标 | 译音的 SQUIM-PESQ 和 DNSMOS OVRL；频谱对比度和平坦度 |
| 说话人相似度 | 合成集与干净参考比；真实集与人声分离结果比，仅作参考 |
| **硬门槛** | ASR-BLEU 不得下降（项目规定：流畅度和音质改动不得降低 ASR-BLEU）；LAAL 不得上升 |
| 计算 | 每位说话人提取音色 token 的耗时，并确认它不在关键路径上 |

### 6.3 对照
- B（原样）；
- A（HDemucs）；
- B + 最干净片段；
- A + B；
- 波形 TSE；
- E（VTD）；
- F（DiCo-Voice）；
- 上限：干净提示，只在合成集上有。

### 6.4 步骤
1. **第 0 档（1 天）**：A + B 用到 10 段试听，加 A/B 切换，同时测 ASR-BLEU。
2. **第 1 档的 E（约 1 周）**：造数据（BGM / 观众 / 串音 × 干净语料）、训练、在 dev 上和 A 比。
3. **F 加论文实验（2–3 周）**：多说话人混合、分离器线索、全套基线、受控合成集、真实电竞 test。
4. **可选**：Talker 反正要重训时，再做 H（4.2 节）或 4.3 节。

### 6.5 风险
- **说话人相似度和干净度的取舍**：去噪可能连带去掉嗓音特征。需要用合成集的干净参考量化这一点。
- **无参考质量指标的偏差**：SQUIM 和 DNSMOS 在 BiCodec 合成语音上可能有偏差，主张需要至少两种指标一致；投稿时可能需要小规模主观听测（ABX）。
- **观众喊声这类"语音型"背景的上限**：如果最终仍是瓶颈，F 的多说话人训练数据要特别加入观众声。

---

## 7. ICLR 级新 idea（第二轮：6 路并行文献检索）

### 7.0 检索范围与总判断

**检索时间：** 2026-11-18 晚，布拉格时间。6 个代理并行检索：

| 代理 | 方向 |
|---|---|
| 1 | ICLR / NeurIPS / ICML 上的语音和音频生成 |
| 2 | CVPR / ICCV / ECCV / SIGGRAPH 上的视觉类比：从带背景或降质的参考图做保身份生成 |
| 3 | 推理时免训练的方法：CFG 和负向引导、对比解码、概念擦除、steering、后验采样 |
| 4 | 表示学习：不变性、解耦、因果、多次观测聚合、身份先验 |
| 5 | 分离和 TSE 在顶会上的进展，以及对我们组合的查新 |
| 6 | 其他领域的类比：LLM 的上下文噪声与 RAG 干扰、视觉语言模型幻觉、RL 中的外生干扰、单细胞批次效应、多视图因果表示、仿真推断的分布偏移 |

**抽象出来的问题：** 生成模型的条件信号（参考、提示）被干扰因素污染，生成器忠实地复现了干扰。我们要保留身份、去掉干扰，同时满足：
- 拿不到干净参考；
- 流式；
- 对同一身份有多次带噪观测；
- **干扰本身可以单独观测到**：分离出的背景 b̂、说话人不在说话的帧。

**三个结论：**
1. **最强的信号：** 有 4 个代理各自独立提出了同一个核心思路，即"**把干扰再加一份，然后外推到零干扰**"（idea 1）。思想来源分别是：量子计算的 zero-noise extrapolation、Noisier2Noise / Ambient Diffusion（数据层面）、Autoguidance（NeurIPS 2024，用"变差的自己"引导），以及 VCD / Instructive Decoding（用破坏过的输入做对比）。在"生成模型的条件空间加上自回归 logit"里统一使用、并把它和 CFG 联系起来的工作，没有检索到。
2. **多说话人场景独有的结构**（idea 2）：同一时刻所有解说**共享同一个广播背景**，而每个人的音色在时间上**不变**；再加上背景可以单独观测，这是单说话人 TTS 文献没有利用过的。
3. **查新更正：** "按活动标签从混合语音提取目标说话人嵌入"已有 Guided Speaker Embedding（ICASSP 2025）。所以 idea 3 的卖点改成"**对识别最优的说话人编码，对合成不是最优的**"，训练目标要穿过生成器本身。

**符号：**
- y：8 s 提示，也就是人声加背景；
- b̂：分离出的纯背景；v̂：分离出的人声；
- E：BiCodec 全局编码器；z = E(y) 是量化前的连续向量，g = FSQ(z) 是 32 个 token；
- ℓ_t(g)：以 g 为条件时 Talker 第 t 步的 logits；
- D：BiCodec 解码器。

**排序：** 按"新颖性 × 可行性 × 能否发 ICLR"排。idea 1 和 idea 2 建议作为两篇独立论文的核心；idea 3 和 idea 4 可以作为备选，或并入 idea 2；idea 5 和 idea 6 作为强基线或组件。

---

### 7.1 Idea 1（主推，免训练）：零干扰外推（Zero-Nuisance Extrapolation，ZNE）

**一句话：** 干扰去不干净，但可以精确地"多加一份"。在不同的干扰放大倍数下各算一次条件和 logits，再用 Richardson 外推到零干扰。同一套公式同时作用在音色编码（解码器那一半）和 Talker logits（LM 那一半）上。

**Motivation：**
- 用分离器去背景会引入分离伪影（"人声"本身被处理过），所以去不干净；
- 但我们手里有 b̂，可以**不失真地加**背景：y + β·b̂ 对编码器来说是完全"在分布内"的输入；
- 2×2 实验表明，背景同时经过 32 个 token 污染两条路径，所以修 z 或修 logits 都要做；
- 普通 CFG 把条件整个丢掉，会连说话人身份一起丢。Autoguidance 的消融指出，"变差的分支"只有在**犯同一种错**时才有用。我们的放大分支恰好只在背景这一个轴上更差。

**做法：**
1. **构造放大序列：** y_λ = y + (λ−1)·b̂，λ ∈ {1, 1.5, 2, (3)}，编码得到 z(λ) = E(y_λ)。
2. **隐空间外推（解码器那一半）：**
   - 把 z(λ) 看成 λ 的平滑函数，用低阶多项式拟合，再用 Richardson 权重 γ（满足 Σγᵢ = 1、Σγᵢλᵢᵏ = 0）得到 ẑ₀ = Σγᵢ·z(λᵢ)；
   - 一阶时就是 ẑ₀ = 2z(1) − z(2)；
   - 然后 g̃ = FSQ(ẑ₀)，同时送给解码器和 Talker。
3. **logit 外推（LM 那一半）：**
   - ℓ̂_t = Σγᵢ·ℓ_t(g(λᵢ))；
   - 一阶即 ℓ̂_t = ℓ_t(g₁) + α·[ℓ_t(g₁) − ℓ_t(g₂)]，其中 α = λ₁/(λ₂ − λ₁)；
   - 两个分支打成一个 batch 跑，用法和 CFG 一样。
4. **分离器增益未知时：** 若 b̂ = c·b + e（增益 c 未知），外推目标应是 λ* = 1 − 1/c。c 用一个在合成对上训练的小 SNR 探针 q(z) 来估：解 q(z(λ*)) = 干净值。
5. **工程保护：**
   - 只在 Contrastive Decoding 式的"可信 token 集合"上施加外推，避免外推推高离谱的 token；
   - 对 K 段不同的 b̂ 取平均，降低外推方差（这是量子 ZNE 的已知弱点）；
   - 按 Dynamic Negative Guidance 的思路，根据两个分支的分歧逐 token 调权重；
   - 用 CCA（ICLR 2025 oral）把 logit 外推蒸馏进 LoRA，这样流式推理的代价回到 1 倍，不影响 LAAL。

**理论贡献（让它成为 ICLR 方法论文，而不只是一个技巧）：**
- **命题 1：** 以"干扰放大的条件"作负分支的负向提示 CFG / 对比解码，等价于一阶 Richardson 外推到零干扰。由此 guidance 权重 α 有了**原理性取值**（不再靠扫参），也可以推广到高阶，并且解释了 Autoguidance 的"同类错误"要求。
- **命题 2：** 外推的偏差和方差分析，说明何时失效：编码器非线性饱和、FSQ 量化、分离器把人声漏进 b̂（同一说话人的人声漏进来，影响是良性的）。
- **通用性：** 这个原理不限于语音。投稿时最好再做一个非语音的验证，例如 RAG 里"再加一份干扰文档后外推 logits"，或视觉语言模型里"再加一份背景后外推"。这样它就是一篇方法论文，而不是一篇领域应用。

**借鉴来源（跨领域）：**
- 量子 zero-noise extrapolation（Temme et al.，arXiv 1612.02058；PRL 2017 待核对）；
- Noisier2Noise（CVPR 2020）；
- Ambient Diffusion（NeurIPS 2023）；
- Consistent Diffusion Meets Tweedie（ICML 2024）；
- Autoguidance（NeurIPS 2024）；
- Stay on Topic with CFG（ICML 2024，自回归 LM 上的 CFG）；
- Contrastive Decoding（ACL 2023）；
- Instructive Decoding（ICLR 2024，用故意破坏的指令作负分支）；
- VCD（CVPR 2024）；
- SID（ICLR 2025）；
- ICG（ICLR 2025，用非空条件做"无条件"分支，不需要 dropout 训练）；
- Dynamic Negative Guidance（ICLR 2025）；
- APG（ICLR 2025，防止 guidance 过强导致过饱和）；
- CCA（ICLR 2025 oral）。

**最接近的工作与差异：**

| 工作 | 它做的 | 我们不同在哪 |
|---|---|---|
| Autoguidance | 破坏**模型** | 我们破坏**条件**，而且只沿一个可测量的物理干扰轴 |
| VCD / SID / Instructive Decoding | 启发式地构造负分支，权重靠扫参 | 外推阶数和权重有原理，并统一了隐空间和 logits |
| Noisier2Noise / Ambient Diffusion | 在数据或像素层面，用于训练 | 我们作用在生成器的**条件**上，推理时用，不用训练 |
| MM-Sonate（arXiv 2601.01568） | 用通用噪声作负条件 | 我们用**同一路流、同一说话人**的实测背景 |
| DAIEN-TTS（ICASSP 2026） | 显式的环境提示 + 双路 CFG，需要重训 F5-TTS | 不训练生成器，用在冻结的 codec LM 上 |

**最小验证（几小时内，go/no-go）：**
1. 用"干净 AMI 提示 + 电竞背景"的合成对（z(clean) 已知）画 z(λ) 随 λ 的曲线。低阶拟合 R² > 0.9 就继续。
2. 比较 ẑ₀、z(v̂)（HDemucs 人声）、z(y) 三者到 z(clean) 的距离。
3. 只用解码器重合成（语义码固定），比较 SQUIM、DNSMOS 和说话人相似度；再跑全链路。
4. 在真实电竞的 80 条配对流上和 HDemucs 方案比较。
5. **硬门槛：** ASR-BLEU 不降，LAAL 不升。

**风险：**
- 编码器非线性和 FSQ 量化可能导致外推过冲。对策：二阶外推、多段 b̂ 平均、小 α。
- 文献提示 guidance 在语音里比在图像里更容易失效（Selective CFG for TTS，arXiv 2509.19668），需要逐 token 的自适应权重。

---

### 7.2 Idea 2（主推，可学习）：同场共享背景的流式身份后验（CA-Kalman / Ambient Conditioning）

**一句话：** 把每位说话人的音色当成静态隐变量，把同一时刻所有人共享的广播背景当成随时间变化的共享因子。分离出的背景作为第二路"只含干扰"的观测流，用卡尔曼滤波（或它的学习版本）在线估计每个人的**干净音色后验**，送给生成器。整个过程只用真实带噪数据就能训练。

**Motivation：**
- 现在每个人只用一段 8 s 的带噪提示，相当于只看了一个样本。实际上解说会连续说好几分钟，而且背景不断变化（音乐切换、观众起伏），音色却不变。
- 多视图因果表示学习（ICLR 2024 spotlight；NeurIPS 2021）证明：跨视图共享的信息可以被识别出来。多次观测正好提供了这样的"视图"。
- **多说话人特有的结构：** 同一时刻所有解说的背景相同。所以跨说话人的共同变化，加上"没人说话的帧"，就能直接估计背景因子。这相当于单细胞批次校正里的"阴性对照"，以及对比分析（SepCLR，ICLR 2024；CFS，NeurIPS 2023）里的"背景数据集"。
- 分离器后验天然给出每次观测的置信度。

**做法（先做线性高斯版）：**
1. **状态：**
   - 说话人 j 的静态音色 s_j，先验 N(μ₀, Σ₀) 由干净语料拟合；
   - 共享背景因子 n_k，服从 AR(1)：n_k = ρ·n_{k−1} + η。
2. **观测 1（短窗、单人帧）：** z_k = s_{j(k)} + A·n_k + ε_k。噪声方差按 1/p_k 放大，p_k = 分离器后验 × 语音概率。这就是"把分离器后验当作似然的温度"。
3. **观测 2（只含干扰）：** v_k = E(b̂_k) = B·n_k + ζ_k。b̂_k 来自分离出的背景或没人说话的帧。
4. **滤波与更新：**
   - 联合卡尔曼滤波，同时利用所有说话人和两路观测，从音色里减去背景；
   - 在背景主导的方向上精度低，估计会自然收缩到干净先验；
   - 后验均值经 FSQ 得到 g；
   - 只有当 tr(Σ_post) 下降超过阈值 τ 时才更新 g，并带滞回，避免音色跳变；
   - 只在句子边界切换。
5. **参数估计：** A、B、ρ 和协方差，用 EM 在**无标注的真实电竞流**上估计。静态和漂移两种动态，加上第二路观测，使模型可识别。
6. **学习版（ICLR 的主体）：**
   - **观测编码器：** 输出 (μ_t, σ_t)，即带不确定度的概率嵌入（MCInfoNCE，ICML 2023），再做贝叶斯聚合（Bayesian Context Aggregation，ICLR 2021）。
   - **只用真实带噪数据训练：**
     - 多视图 InfoNCE：正样本是同一说话人、相隔超过 T 秒的片段；困难负样本是同一时刻的 E(b̂) 和"同背景、不同人"的合成混合；
     - Noisier2Noise 式训练：在真实提示上再叠加同域背景，预测加噪前的编码。
   - **合成对只用于校准：** 用高斯 NLL 校准 σ。
   - **先验：** 可以加一个小的流或扩散先验，用一步 Tweedie 或 DPS 得到后验均值。

**理论与可检验主张：**
- **可识别性猜想：** 在"身份静态、背景随时间变化，且有一路只含背景的观测"的条件下，身份块可以识别。这是对多视图因果表示学习的推广，加上了一个背景集。持续不变的背景（例如一直垫着的 BGM）也是跨视图共享的，所以必须用 b̂ 作负样本，这一点要在理论里说明。
- **可检验：** 生成质量随观测秒数大约按 1/√n 改善，直到说话人自身漂移为止。画"观测秒数 vs 质量"曲线，这对同传尤其关键：首句质量和后续质量的取舍。

**借鉴来源：**
- 多视图因果表示学习（ICLR 2024 spotlight）；
- von Kügelgen et al.（NeurIPS 2021，自监督学习可证明地分离内容与风格）；
- Bayesian Context Aggregation（ICLR 2021）；
- MCInfoNCE（ICML 2023）；
- Ambient Diffusion（NeurIPS 2023）；
- SepCLR（ICLR 2024）、CFS（NeurIPS 2023）；
- IFactor（NeurIPS 2023，可识别的分解，把噪声留在模型里但不给策略用）；
- 单细胞 Cell Painting 的 sphering（用阴性对照白化）；
- 说话人识别中的经典 JFA / NAP。

**最接近的工作与差异：**
- **JFA / PLDA / NAP（说话人确认）：** 目标是判别，离线，单人，没有一路只含背景的观测。
- **Mega-TTS 2（ICLR 2024）、PhotoMaker（CVPR 2024）：** 多参考聚合，但没有干扰模型。
- **SEED（Interspeech 2025）：** 单条嵌入的扩散去噪。
- **我们：** 估计的是**生成器的条件**；在线；利用跨说话人共享的干扰；有一路只含干扰的观测；用分离器后验作温度；只用真实带噪数据训练。

**最小验证：**
- 合成多人流：干净说话人加上随时间变化的电竞背景，有干净参考。画误差、生成 SQUIM 和说话人相似度随观测秒数（1–60 s）的曲线。
- 对照组：
  1. 现在的"首 8 s"；
  2. 滑动平均；
  3. 只做先验收缩；
  4. 卡尔曼，不用 b̂ 那路观测；
  5. 完整卡尔曼；
  6. 学习版。
- 然后在真实电竞上做同样的曲线（无参考指标）。

**风险：**
- 短窗编码很嘈杂；线性模型抓不住非线性纠缠；
- 外场采访这类场景，不同说话人的背景并不相同。这正好可以作为模型选择的消融。

---

### 7.3 Idea 3：面向生成器的条件去噪（"对识别最优 ≠ 对合成最优"）

**一句话：** 现有的从混合语音提取说话人嵌入的方法（Guided Speaker Embedding、SEED、师生 d-vector）优化的都是**识别**损失。我们训练一个小的条件适配器，**穿过冻结的生成器本身**来优化：让带噪条件下 Talker 的输出分布，逼近"干净反事实"条件下的输出分布。

**Motivation：**
- 查新显示，"按活动标签从混合语音提取目标嵌入"已有人做，所以卖点要换。
- 关键观察是：生成器读取 32 个 token 的方式，和识别模型不同。识别最优的嵌入（例如映射到 BiCodec token 的 ECAPA）**仍然可能带着背景影响生成**。论文需要先用实验证明这一点。
- "干净的 token 是什么"本身是病态问题：很多组 token 听起来是同一个声音。真正需要的，只是去掉"生成器会读成噪声"的那部分。
- 我们能合成**真正的反事实对**：同一说话人、同一文本、同一对齐，只差背景。反事实不变性的文献（NeurIPS 2021；NeurIPS 2023）通常拿不到这种数据。

**做法：**
1. **适配器 A：** 小 Transformer。输入是带噪的量化前向量，可选再加上若干分离器窗口及其后验，按 DiCoW 的四类掩码（静音 / 目标 / 非目标 / 重叠）编码。输出连续条件向量，用软 FSQ 或直通估计送入冻结的 Talker 和解码器。可以落在 FSQ 格点之间。
2. **损失：**
   - **生成器输出蒸馏：** KL( p_Talker(·|text, src, g_clean) ‖ p_Talker(·|text, src, A(g_noisy)) )，在 teacher forcing 的短片段上计算，配合梯度检查点；
   - **解码器损失：** mel 和多分辨率谱损失，外加频谱平坦度、频谱对比度两项惩罚（对应我们实测到的"闷"的特征）；
   - **不变性正则：** 同一句话配两种不同背景 b 和 b′，要求 A 的输出或隐状态一致（MMD 或方差惩罚）；
   - **说话人相似度保持。**
3. **流式：** 分离器后验是软的、逐块到达的，A 随时间更新。按 token 后验熵决定是否沿用上一次的编码。

**借鉴来源：**
- DA-CLIP（ICLR 2024，降质感知的控制器）；
- SeeSR（CVPR 2024，降质鲁棒的提示提取器）；
- PuLID（NeurIPS 2024，ID 损失加对齐损失）；
- BLIP-Diffusion（NeurIPS 2023，把主体贴到随机背景上）；
- DEADiff（CVPR 2024，非重建式训练）；
- 反事实不变性（Veitch et al. NeurIPS 2021；Feder et al. NeurIPS 2023）；
- DFR（ICLR 2023，只重训最后一层）；
- GenSE（ICLR 2025，token 空间增强）；
- TDSM（ICLR 2024，带噪条件下的扩散模型）。

**最接近的工作与差异：**
- **Guided Speaker Embedding（ICASSP 2025）及其 Interspeech 2025 后续：** 判别式嵌入，用分类损失，活动标签是二值的、离线的，不评测生成质量。
- **师生混合嵌入（arXiv 2306.00634）、Unmixing the Crowd（arXiv 2604.03219）、SEED、NR-LauraTTS、SATTS（Interspeech 2022）：** 都是在嵌入或 token 空间里做重建或对比，目标不穿过生成器。
- **我们：** 目标穿过冻结生成器的输出分布；同时修两条路径；分离器后验是软的四类、流式的。

**必须做的基线：**
- HDemucs / BS-RoFormer 分离后再编码；
- 波形 TSE 后再编码（arXiv 2502.16611、LauraTSE、SoloSpeech）；
- Guided Speaker Embedding 改为输出 token；
- SEED；
- NR-LauraTTS 式的 token 去噪。

**通用性：** 至少两种说话人编码接口（BiCodec 全局 token，加一个连续嵌入的 TTS，例如 CosyVoice）；至少一个公开设定（AMI，或 Libri2Mix 加 MUSAN / 音乐）。

---

### 7.4 Idea 4：干扰寄存器——"改道，而不是删除"（Nuisance Registers）

**一句话：** BiCodec 训练时要重建**整段**提示，所以背景信息除了塞进 32 个音色 token，别无去处。我们给它加几个"干扰寄存器"token 来专门接住背景。推理时把寄存器置为"干净值"，冻结的生成器不需要重训。

**Motivation：**
- ViT Registers（ICLR 2024）发现 Transformer 会把全局或背景信息倾倒进低信息 token，加几个寄存器 token 就能吸收掉。
- Denoised MDP（ICML 2022）和 IFactor（NeurIPS 2023）的做法是：噪声保留在模型里，但不给策略看。
- 实用上的额外好处：寄存器可以**可控地保留现场感**。把原广播的寄存器值喂回去，就是"保持现场氛围"的同传（§4.3），一套模型同时支持去背景和保背景两种模式。

**做法：**
1. **结构：** g_id = 原编码器加 LoRA；r = E_reg(y)，即 R 个寄存器；解码器加一个小 FiLM / LoRA，读取 [g_id; r]。
2. **损失：**
   - **重建：** 用 [g_id, r] 重建**带噪**的 y，所以信息是被改道，而不是丢掉；
   - **寄存器监督：** r 要能预测 φ(b̂)。这个监督来自分离，在无标注的电竞数据上免费可得；
   - **防泄漏：** 每个 batch 用 LEACE 或对抗器，让 g_id 不含 φ(b̂)；
   - **交换一致性：** 对 (s+b₁, s+b₂)，要求 g_id 相同，且 D([g_id(s+b₁), r(s+b₂)]) ≈ s+b₂；
   - **干净锚点：** 在干净语音上，g_id(s) = 原 g(s)，r = r_∅。这样冻结的 Talker 和解码器始终在分布内。
3. **推理：** 默认送 [g_id, r_∅]，即去背景模式；需要现场感时送 [g_id, r(y)]。

**借鉴来源：**
- ViT Registers（ICLR 2024）；
- Denoised MDP（ICML 2022）、IFactor（NeurIPS 2023）；
- InfoGating（NeurIPS 2023）；
- NaturalSpeech 3 的 FACodec（ICML 2024，因子化编码器加梯度反转）；
- Batch Calibration（ICLR 2024，"无内容输入"的思想）；
- Break-A-Scene（SIGGRAPH Asia 2023，用掩码损失把概念分开）。

**最接近的工作与差异：**
- **DAIEN-TTS：** 显式环境提示，需要重训 F5-TTS；
- **Zhang et al.（ICASSP 2025）：** 双说话人编码器加控制信号，需要重训；
- **NaturalSpeech 3：** 从头训练一个因子化编码器；
- **我们：** 给**冻结的** codec LM 的定长身份编码做改装；用干净锚点保证在分布内；用分离结果作为领域内的弱监督，不需要干净参考。

**风险：** 改装 BiCodec 编码器需要训练（约 1 GPU 天）。关键消融是"无寄存器、只做对抗删除"，用来检验"改道"是否优于"删除"。

---

### 7.5 Idea 5（强基线 / 组件）：保护说话人子空间的干扰擦除（闭式，零推理开销）

**做法：**
1. 用干净语料混入电竞背景，得到 d = z(s+b) − z(s)，取前 k 个主成分作为干扰子空间 U_b（NAP / R-LACE 视角）。
2. 用干净向量的说话人间协方差得到说话人子空间 U_s。
3. **斜投影：** z′ = z − U_b(U_bᵀ P U_b)⁻¹U_bᵀ P (z − μ_clean)，其中 P = I − U_sU_sᵀ。它在去掉干扰分量的同时不动说话人分量。之后再做 FSQ。
4. **可替换方案：**
   - LEACE（NeurIPS 2023，以背景强度为标签）；
   - Representation Surgery / Linear-AcT 的均值加协方差最优传输（ICML 2024 / ICLR 2025）；
   - Spectral Editing of Activations（NeurIPS 2024，用 25 个正负示例做 SVD）。
5. **LM 那一半：** 在 Talker 残差流上按 Arditi et al.（NeurIPS 2024）的方向消融，并**并入权重**（权重正交化），推理零开销。也可以只干预能线性预测"有背景"的注意力头（ITI，NeurIPS 2023）。

**定位：** 单独发表时新颖性一般，是已知擦除方法在新场景的应用。但它便宜、闭式，适合作为 idea 1 和 idea 2 的强基线或组件。另外，语音里用 steering 改情感（EmoSteer-TTS）和口音（arXiv 2603.05977）已有先例，但没有检索到用来改"录音环境"的。

### 7.6 Idea 6（组件）：带显式背景项的分析式合成反演

- **做法：** 只优化全局向量 z，穿过冻结的解码器，FSQ 用直通估计：

  z* = argmin_z L_mel(D(FSQ(z), s) + a·b̂, y) + λ₁‖z − z(y)‖² + λ₂[1 − cos(E_spk(D(FSQ(z), s)), E_spk(v̂))]

  损失只算在目标说话人主导的时频点上（Break-A-Scene 式掩码）。因为 b̂ 已经解释了背景，z 没有动力去吸收背景。
- **成本：** 20–50 步 Adam，在后台线程里跑，算好后热替换 g，不阻塞流式。
- **来源：** CSGM（ICML 2017）、Textual Inversion（ICLR 2023）、Arik et al. 只调说话人嵌入（NeurIPS 2018）、DPS（ICLR 2023）、GibbsDDRM（ICML 2023）、ReNO（NeurIPS 2024）。
- **定位：** 适合作为 idea 2 的"精修"步骤或 idea 1 的对照。

### 7.7 跨领域的便宜基线（不新，但应先跑，作为论文的对照组）

| 基线 | 来源 | 做法 |
|---|---|---|
| 无内容偏置减除 | Batch Calibration（ICLR 2024） | 用 b̂ 作"无内容输入"，减去 ℓ(g(b̂)) 引起的偏置 |
| 窗口挑选 | LPR（NeurIPS 2024） | 用 Talker 或解码器的困惑度给同一说话人的候选提示窗排序，取最干净的 |
| 隔离再聚合 | RobustRAG（ICML 2024 workshop） | 各段分别编码，取中位数或截尾均值 |
| 预计算的平均方向 | VTI（ICLR 2025） | 预先算好"多视图平均"带来的位移，推理时直接加 |
| 分路条件 | CosyVoice 2 / Vevo（ICLR 2025）的教训 | Talker 用中性或去噪 token，解码器用本人 token，以此测量 LM 那一半能去掉多少 |
| 限定注意力范围 | Bounded Attention（ECCV 2024） | Talker 对源音频的注意力只看当前说话人的高语音概率帧 |
| 加噪提示 LoRA | OZSpeech（ACL 2025）/ DINO 配方 | 训练时以 0.8 的概率用带噪提示，目标是干净语义码 |
| 泄漏探针 | Self-Supervised Visual Acoustic Matching（NeurIPS 2023）的"声学残留"指标 | 从全局向量线性预测背景类别或 SNR，作为每种干预的快速泄漏度量 |

### 7.8 建议的投稿组合

**论文 1（ICLR，方法论文）：** *Extrapolating Away the Nuisance: Zero-Nuisance Extrapolation for Conditional Generation*，即 idea 1。
- 贡献：原理与命题 1（负向引导等价于一阶外推）、偏差方差分析、隐空间与 logit 统一、CCA 蒸馏。
- 实验：主实验是多说话人同传（电竞两个方向，加 AMI 受控合成）；再加一个非语音领域（RAG 干扰文档或视觉语言模型的背景）证明通用性。
- 免训练，最快能出结果。

**论文 2（ICLR / NeurIPS）：** *Ambient Conditioning: Learning Clean Identity Conditions from Repeated Noisy Observations in Streaming Multi-Speaker Generation*，即 idea 2，并入 idea 3 的生成器目标和 idea 4 的寄存器作为消融或变体。
- 贡献：共享背景加上只含干扰的观测流、可识别性、只用真实带噪数据训练、"观测秒数 vs 质量"的流式曲线。

**Interspeech / ICASSP：** idea 5、idea 6 和 §7.7 的基线组合，加上已有的 2×2 机制分析，可以作为短文。

### 7.9 查新的局限

- 6 个代理共用的检索额度在后期用尽，部分会议信息没有核对，已在文中标出"待核对"。
- idea 1 和 idea 2 的"未检索到先例"建立在有限次数的检索上。投稿前必须针对以下关键词做一次专门查新：
  - "zero-noise / Richardson extrapolation + guidance / conditional generation"；
  - "factor analysis / Kalman + TTS conditioning"；
  - "register tokens + codec"。
- 少数信息来自代理的记忆，没有经过检索核对（Temme et al. 的 PRL、contrastiveVI 的 Nature Methods、Miipher 的会议），已标出。

## 8. 参考（第一轮）

- DINO-VITS：https://arxiv.org/abs/2311.09770
- An Investigation of Noise Robustness for Flow-Matching-Based Zero-Shot TTS：https://arxiv.org/abs/2406.05699
- NR-LauraTTS（Interspeech 2025）：https://arxiv.org/abs/2505.13830
- 连续自回归 TTS 的噪声提示评测：https://arxiv.org/abs/2502.01084
- DINO-PRETSSEL（ACL Findings 2024）：https://arxiv.org/abs/2406.02733
- Hibiki：https://arxiv.org/abs/2502.03382
- Spark-TTS / BiCodec：https://arxiv.org/abs/2503.01710
- SpeechX：https://arxiv.org/abs/2308.06873
- TS-SEP：https://arxiv.org/abs/2303.03849
- DiCoW：https://arxiv.org/abs/2501.00114
- Environment-Aware TTS：https://arxiv.org/abs/2110.03887
- IDEA-TTS：https://arxiv.org/abs/2412.16977
- DenoiSpeech：https://arxiv.org/abs/2012.09547
- Extended U-Net for SV：https://arxiv.org/abs/2206.13044
- Spatial Speech Translation：https://arxiv.org/abs/2504.18715
- X-Translator：https://arxiv.org/abs/2607.17544
- JUST-DUB-IT：https://arxiv.org/abs/2601.22143
- TransVIP：https://arxiv.org/abs/2405.17809
- WeSep：https://arxiv.org/abs/2607.27436
- Translatotron 2：https://arxiv.org/abs/2107.08661

说明：以上文献信息来自检索摘要，部分数字没有逐篇核对原文。投稿前需要重新核对，并做一次系统检索。

## 9. 参考（第二轮，按领域分组）

**会议核对状态：** 未标注的条目，会议已由检索代理在会议论文集、官方虚拟会场或 mlanthology 上核对。标"待核对"的只有 arXiv，或者会议信息来自 arXiv 备注或代理记忆。

### 9.1 语音生成（ICLR / NeurIPS / ICML）
- NaturalSpeech 3（FACodec），ICML 2024：https://arxiv.org/abs/2403.03100
- Voicebox，NeurIPS 2023：https://arxiv.org/abs/2306.15687
- SpeechFlow，ICLR 2024：https://arxiv.org/abs/2310.16338
- Metis，NeurIPS 2025：https://arxiv.org/abs/2502.03128
- GenSE，ICLR 2025：https://arxiv.org/abs/2502.02942
- Vevo，ICLR 2025：https://arxiv.org/abs/2502.07243
- MaskGCT，ICLR 2025：https://arxiv.org/abs/2409.00750
- Mega-TTS 2，ICLR 2024：https://arxiv.org/abs/2307.07218
- UniAudio，ICML 2024：https://arxiv.org/abs/2310.00704
- Self-Supervised Visual Acoustic Matching，NeurIPS 2023：https://arxiv.org/abs/2307.15064
- RecXi（语音与内容解耦，用于说话人识别），NeurIPS 2023：https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract.html
- StyleTTS 2，NeurIPS 2023：https://neurips.cc/virtual/2023/poster/70566
- Hibiki，ICML 2025：https://arxiv.org/abs/2502.03382
- Hibiki-Zero，ICML 2026（待核对）：https://arxiv.org/abs/2602.11072

### 9.2 语音领域的近邻工作（非 ML 顶会，查新必引）
- Zhang et al.，零样本 TTS 的背景去除与保留，ICASSP 2025：https://arxiv.org/abs/2502.07345
- DAIEN-TTS，ICASSP 2026（待核对）：https://arxiv.org/abs/2509.14684；后续工作：https://arxiv.org/abs/2608.03011
- OZSpeech，ACL 2025：https://arxiv.org/abs/2505.12800
- Koel-TTS，EMNLP 2025：https://aclanthology.org/2025.emnlp-main.1076
- MM-Sonate，以噪声作负条件（待核对）：https://arxiv.org/abs/2601.01568
- Selective CFG for zero-shot TTS（待核对）：https://arxiv.org/abs/2509.19668
- ReStyle-TTS（待核对）：https://arxiv.org/abs/2601.03632
- Fujita et al.，SSL 适配器做噪声鲁棒零样本 TTS，ICASSP 2024：https://arxiv.org/abs/2401.05111
- Hsu et al.，说话人与噪声相关时的解耦（数据增强加对抗因子化），ICASSP 2019：https://research.google/pubs/pub47834/
- SEED，说话人嵌入的扩散增强，Interspeech 2025：https://arxiv.org/abs/2505.16798
- Noro，噪声鲁棒的单样本语音转换，APSIPA 2025（待核对）：https://arxiv.org/abs/2411.19770
- SD-Codec，在 codec 中分离语音、音乐和音效，ICASSP 2025：https://arxiv.org/abs/2409.11228
- CosyVoice 2：https://arxiv.org/abs/2412.10117
- EmoSteer-TTS（待核对）：https://arxiv.org/abs/2508.03543
- 口音 steering（基于 Qwen3-TTS，待核对）：https://arxiv.org/abs/2603.05977
- 流式生成式语音增强：Stream.FM https://arxiv.org/abs/2512.19442；Hsieh & Braun，ICASSP 2026（待核对）https://arxiv.org/abs/2510.16997
- Seed LiveInterpret 2.0：https://arxiv.org/abs/2507.17527

### 9.3 分离和 TSE；查新最接近的工作
- CLIPSep，ICLR 2023：https://arxiv.org/abs/2212.07065
- Separate and Diffuse，ICLR 2024：https://arxiv.org/abs/2301.10752
- TIGER，ICLR 2025：https://arxiv.org/abs/2410.01469
- 带噪正负注册对比 TSE，NeurIPS 2025（据 arXiv 备注）：https://arxiv.org/abs/2502.16611
- LauraTSE（据称 ASRU 2025，待核对）：https://arxiv.org/abs/2504.07402；StarTSE：https://arxiv.org/abs/2604.19635
- TSELM：https://arxiv.org/abs/2409.07841
- SoloSpeech（待核对）：https://arxiv.org/abs/2505.19314
- ADEnet，用说话人活动驱动的提取，ICASSP 2021：https://arxiv.org/abs/2101.05516
- SE-DiCoW：https://arxiv.org/abs/2601.19194
- **Guided Speaker Embedding，ICASSP 2025（与本文最接近）**：https://arxiv.org/abs/2410.12182；后续，Interspeech 2025：https://arxiv.org/abs/2506.12500
- 混合语音的师生嵌入（待核对）：https://arxiv.org/abs/2306.00634
- Unmixing the Crowd（投 SLT 2026）：https://arxiv.org/abs/2604.03219
- SATTS，Interspeech 2022：https://arxiv.org/abs/2207.06011
- EEND-VC：https://arxiv.org/abs/2105.09040
- REAL-TSE 挑战数据：https://arxiv.org/abs/2607.15198

### 9.4 视觉类比（CVPR / ICCV / ECCV / SIGGRAPH，以及 ML 顶会上的视觉论文）
- DisenBooth，ICLR 2024：https://iclr.cc/virtual/2024/poster/19050
- MaskST（Less is More，用元素掩码防止内容泄漏），ICLR 2025：https://proceedings.iclr.cc/paper_files/paper/2025/hash/a547d86953a4e36aa8a1390e6f4708e2-Abstract-Conference.html
- PuLID，NeurIPS 2024：https://proceedings.neurips.cc/paper_files/paper/2024/hash/409fcc9d24b549969b8b9be68b56a7be-Abstract.html
- BLIP-Diffusion，NeurIPS 2023：https://papers.nips.cc/paper_files/paper/2023/hash/602e1a5de9c47df34cae39353a7f5bb1-Abstract-Conference.html
- DA-CLIP，ICLR 2024：https://arxiv.org/abs/2310.01018
- Break-A-Scene，SIGGRAPH Asia 2023：https://arxiv.org/abs/2305.16311
- DEADiff，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/html/Qi_DEADiff_An_Efficient_Stylization_Diffusion_Model_with_Disentangled_Representations_CVPR_2024_paper.html
- Arc2Face，ECCV 2024 Oral：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5442_ECCV_2024_paper.php
- B-LoRA，ECCV 2024：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01549.pdf
- SeeSR，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/html/Wu_SeeSR_Towards_Semantics-Aware_Real-World_Image_Super-Resolution_CVPR_2024_paper.html
- SUPIR，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Scaling_Up_to_Excellence_Practicing_Model_Scaling_for_Photo-Realistic_Image_CVPR_2024_paper.pdf
- PFStorer，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/papers/Varanka_PFStorer_Personalized_Face_Restoration_and_Super-Resolution_CVPR_2024_paper.pdf
- WildGaussians，NeurIPS 2024：https://arxiv.org/abs/2407.08447；NeRF On-the-go，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/html/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.html
- Bounded Attention，ECCV 2024：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02245.pdf
- PhotoMaker，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.pdf
- AnyDoor，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_AnyDoor_Zero-shot_Object-level_Image_Customization_CVPR_2024_paper.pdf
- SSR-Encoder，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_SSR-Encoder_Encoding_Selective_Subject_Representation_for_Subject-Driven_Generation_CVPR_2024_paper.html
- RB-Modulation，ICLR 2025 Oral：https://proceedings.iclr.cc/paper_files/paper/2025/hash/8f3705d6354fcecf48515cc43aa16023-Abstract-Conference.html
- Concept Sliders，ECCV 2024：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5660_ECCV_2024_paper.php
- Paint by Example，CVPR 2023：https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Paint_by_Example_Exemplar-Based_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf
- InstantStyle（只有 arXiv）：https://arxiv.org/abs/2404.02733
- Debiasing VLMs via Biased Prompts（只有 arXiv）：https://arxiv.org/abs/2302.00070
- VCD，CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html

### 9.5 推理时引导、表示编辑与后验推断
- Stay on Topic with CFG，ICML 2024：https://proceedings.mlr.press/v235/sanchez24a.html
- Contrastive Decoding，ACL 2023：https://aclanthology.org/2023.acl-long.687
- Context-aware Decoding，NAACL 2024：https://aclanthology.org/2024.naacl-short.69
- Instructive Decoding，ICLR 2024：https://proceedings.iclr.cc/paper_files/paper/2024/hash/1f5c76187c28e8d9d3b938d0c504436c-Abstract-Conference.html
- ICG（无需训练的 CFG），ICLR 2025：https://proceedings.iclr.cc/paper_files/paper/2025/hash/bf331c87e29f473b610336f00fe1cb51-Abstract-Conference.html
- Dynamic Negative Guidance，ICLR 2025：https://iclr.cc/virtual/2025/poster/30862
- Autoguidance，NeurIPS 2024：https://arxiv.org/abs/2406.02507
- APG，ICLR 2025：https://proceedings.iclr.cc/paper_files/paper/2025/file/d1450d6c10c6b6cf1b80964357f5fa08-Paper-Conference.pdf
- SEGA，NeurIPS 2023：https://proceedings.neurips.cc/paper/2023/hash/4ff83037e8d97b2171b2d3e96cb8e677-Abstract-Conference.html
- CADS，ICLR 2024：https://iclr.cc/virtual/2024/poster/17385
- CCA（Condition Contrastive Alignment），ICLR 2025 Oral：https://arxiv.org/abs/2410.09347
- SID（Self-Introspective Decoding），ICLR 2025：https://arxiv.org/abs/2408.02032
- HALC，ICML 2024：https://proceedings.mlr.press/v235/chen24bi.html
- VTI（潜空间 steering），ICLR 2025：https://arxiv.org/abs/2410.15778
- LEACE，NeurIPS 2023：https://arxiv.org/abs/2306.03819
- R-LACE，ICML 2022：mlanthology ravfogel2022icml-linear
- Representation Surgery，ICML 2024：https://proceedings.mlr.press/v235/singh24d.html
- Linear-AcT，ICLR 2025：https://proceedings.iclr.cc/paper_files/paper/2025/hash/df4f6e43446b1ee29c5a33d32c279f83-Abstract-Conference.html
- Spectral Editing of Activations，NeurIPS 2024：https://arxiv.org/abs/2405.09719
- Arditi et al.（单方向消融），NeurIPS 2024：https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract.html
- ITI，NeurIPS 2023：https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract.html；CAA，ACL 2024：https://aclanthology.org/2024.acl-long.828
- DPS，ICLR 2023：https://iclr.cc/virtual/2023/poster/11877
- GibbsDDRM，ICML 2023：https://proceedings.mlr.press/v202/murata23a.html
- SGDD（离散扩散的后验采样），NeurIPS 2025：https://proceedings.neurips.cc/paper_files/paper/2025/hash/d5bda93cd36f470fc9b175bbcd9cddab-Abstract-Conference.html
- Textual Inversion，ICLR 2023：https://iclr.cc/virtual/2023/poster/12191；CSGM，ICML 2017：https://proceedings.mlr.press/v70/bora17a.html；Arik et al.，NeurIPS 2018：https://proceedings.neurips.cc/paper/2018/hash/4559912e7a94a9c32b09d894f2bc3c82-Abstract.html
- ReNO，NeurIPS 2024：https://neurips.cc/virtual/2024/poster/95513
- 量子 zero-noise extrapolation：Temme, Bravyi, Gambetta，https://arxiv.org/abs/1612.02058（PRL 2017 待核对）

### 9.6 不变性、解耦、因果、聚合与干净先验
- Multi-View Causal Representation Learning with Partial Observability，ICLR 2024 spotlight：https://arxiv.org/abs/2311.04056
- von Kügelgen et al.（自监督学习可证明地分离内容与风格），NeurIPS 2021：https://arxiv.org/abs/2106.04619
- Identifiability Results for Multimodal Contrastive Learning，ICLR 2023：https://iclr.cc/virtual/2023/poster/11159
- Disentanglement via Latent Quantization，NeurIPS 2023：https://neurips.cc/virtual/2023/poster/71965；FSQ，ICLR 2024：https://iclr.cc/virtual/2024/poster/19317
- Label-Noise Robust Diffusion Models（TDSM），ICLR 2024：https://iclr.cc/virtual/2024/poster/18991
- Counterfactual Invariance to Spurious Correlations，NeurIPS 2021：https://neurips.cc/virtual/2021/poster/28538；Feder et al.，NeurIPS 2023：https://papers.neurips.cc/paper_files/paper/2023/hash/df88b275bef31ac96c85f0c4013734fc-Abstract-Conference.html
- DFR（只重训最后一层），ICLR 2023：https://mlanthology.org/iclr/2023/kirichenko2023iclr-last
- Bayesian Context Aggregation，ICLR 2021：https://iclr.cc/virtual/2021/poster/3148；MCInfoNCE，ICML 2023：https://proceedings.mlr.press/v202/kirchhof23a.html；UEC，ICLR 2026（仅据 mlanthology）：https://mlanthology.org/iclr/2026/lim2026iclr-uncertaintydriven/
- Ambient Diffusion，NeurIPS 2023：https://arxiv.org/abs/2305.19256；Consistent Diffusion Meets Tweedie，ICML 2024：https://arxiv.org/abs/2404.10177；Ambient Diffusion Omni，NeurIPS 2025：https://neurips.cc/virtual/2025/poster/118464
- Noisier2Noise，CVPR 2020：https://arxiv.org/abs/1910.11908
- Diffusion Forcing，NeurIPS 2024：https://neurips.cc/virtual/2024/poster/93029
- Cascaded Diffusion Models（条件增强），JMLR 2022：https://jmlr.org/papers/v23/21-0635.html

### 9.7 其他领域的类比
- Batch Calibration，ICLR 2024：https://arxiv.org/abs/2309.17249
- LPR（上下文学习在文本生成上的噪声鲁棒性），NeurIPS 2024：https://arxiv.org/abs/2405.17264
- Making RAG Robust to Irrelevant Context，ICLR 2024：https://arxiv.org/abs/2310.01558；LLMs Can Be Easily Distracted，ICML 2023：https://arxiv.org/abs/2302.00093；RobustRAG（workshop）：https://arxiv.org/abs/2405.15556
- Denoised MDPs，ICML 2022：https://arxiv.org/abs/2206.15477；IFactor，NeurIPS 2023：https://arxiv.org/abs/2306.06561
- InfoGating，NeurIPS 2023：https://arxiv.org/abs/2303.06121；ACRO，ICML 2023：https://proceedings.mlr.press/v202/islam23a.html；HRSSM，ICML 2024：https://arxiv.org/abs/2405.06263
- Vision Transformers Need Registers，ICLR 2024：https://arxiv.org/abs/2309.16588
- SepCLR，ICLR 2024：https://arxiv.org/abs/2402.11928；CFS，NeurIPS 2023：https://arxiv.org/abs/2310.18531
- RoPE（仿真推断的错配校准），ICML 2025 Oral：https://proceedings.mlr.press/v267/wehenkel25a.html；Huang et al.，NeurIPS 2023：https://arxiv.org/abs/2305.15871
