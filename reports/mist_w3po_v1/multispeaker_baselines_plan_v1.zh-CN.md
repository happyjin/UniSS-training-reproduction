# 多说话人同传（MIST）外部基线对比方案 v1

整理时间：2026-11-12（布拉格）。只做文献调研和方案，没有运行任何系统。

标注说明：
- 【原文】：数字逐字取自论文。
- 【需核对】：来自网页摘要或二手来源，写进论文前要对照原文核对。

---

## 0. 结论速览

| 问题 | 建议 |
|---|---|
| 有没有直接可比的已发表数字？ | **只有 DiariST-AliMeeting（中→英，说话人归属的语音翻译）这一份。** 我们的 AliMeeting 测试集已经在用 DiariST 公开的人工英文参考。只要用它的评测脚本给我们的输出算 **SAgBLEU / SAtBLEU**，就能直接和它的已发表数字比，不用 GPU，也不用运行 DiariST（符合 11-03 "不做 DiariST 实验"的决定）。限制：它只出文本，没有延迟；模型是 2023 年 Whisper-small 级别，偏弱。 |
| 开源基线要不要自己跑？ | **要。** 2024–2026 年没有发表过"多说话人同传语音到语音"的系统，可比的外部数字几乎为零。审稿人最认的是：(1) **现成开源模块拼的级联**（分离 + ASR + LLM 翻译 + 声音克隆 TTS），离线和流式各一个；(2) **单说话人同传系统直接喂混合音频**（SeamlessStreaming、Hy-SimulMT），用来说明"不分说话人"会出什么问题。 |
| 要不要用 GPT 这类 API？ | **建议用，作为工业参照点，不作为主基线。** GPT-Realtime-Translate、Gemini Live Translate、Qwen-LiveTranslate、Seed LiveInterpret 2.0 都是单路输出、不分说话人，只能按"说话人无关"的指标比。价格很低：GPT-Realtime-Translate 约 $0.034/分钟【需核对】，所有测试集约 $50。离线方面，可以用 GPT-4o 或 Gemini 的音频理解，按说话人逐段翻译，作为"大模型离线上限"的参照。 |
| 优先顺序 | **P0**：零成本的 DiariST 指标对比。→ **P1**：开源离线级联、开源流式级联、Qwen3-Omni 原生、SeamlessStreaming。→ **P2**：GPT 和 Gemini 的 API 参照。→ **P3**：Hy-SimulMT + TTS、X-Translator（之前决定不实测，只在能力对比表里列）。 |

---

## 1. 现状：论文（gstar_si_v9）和 v8/v9 方案里已有的对比

- **内部基线，同一份数据**（`docs/mist_v8_plan.zh-CN.md`，`docs/mist_prereg_v9.zh-CN.md`）：
  - 级联：同一份数据，按轮次重训；
  - 后置重标（Rel，speaker-blind + 因果重标）；
  - D1 / D2 拆段；
  - 同一个模型下的不同策略：LA2、固定滞后 Lag1/2/3、Blk2；
  - 离线 MIST。
- **外部系统**：只有引言里的能力对比表（Seamless、Hibiki、SimulS2S-LLM、X-Translator、DiariST 等），没有实测。
- **已有决定**：
  - 11-03 修订 1：**不做任何 DiariST 实验**，只引用；
  - X-Translator 只放能力对比表，可选做 2–3 场会议的延迟实测；
  - 不做人工评价和 COMET；
  - DiariST 公开的 AliMeeting 人工英文参考**只用来给我们自己的系统打分**。
- **测试集**：
  - AMI 16 场（英→中）；
  - AMI 16 场（中→英，重配音）；
  - AliMeeting 20 场（中→英，DiariST 人工参考）。
  - 测试集只在最后跑一次；dev 用于选配置。
- **问题**：外部可比的数字几乎为零，审稿人很可能会问"和现成的开源级联或商用系统比怎样"。

---

## 2. 文献调研：有没有可比的多说话人语音翻译工作

| 工作 | 任务 | 数据 / 方向 | 输出 | 开源情况 | 能怎么用 |
|---|---|---|---|---|---|
| **DiariST**（Yang et al., ICASSP 2024, arXiv 2309.08007） | 流式语音翻译 + 说话人分离（t-SOT + t-vector） | **DiariST-AliMeeting**，中→英：dev 8 场 4 h，test 20 场 10 h，切成 3–6 分钟的小段（test 195 段）；有 SDM 远场、IHM-MIX、IHM-CAT 三种条件；dev/test 是人工翻译 | 文本 | 评测数据、离线基线、评测代码都开源（MIT；数据 CC BY-SA 4.0）。DiariST 自身的权重是否开源，没能确认 | **直接比文本**，见第 3 节 |
| X-Translator（arXiv 2607.17544，上海交大 X-LANCE） | 流式级联 S2ST：Qwen3-ASR + LMT-60-8B + X-Voice | 中英，长篇对话 | 语音 | 代码 MIT；评测代码"later" | 有逐说话人的音色提示（CAM++），但**不处理重叠**（它自己写在局限里）。之前决定只放能力对比表 |
| Spatial Speech Translation（CHI 2025, arXiv 2504.18715） | 可穿戴设备上分方向翻译多个说话人 | 双耳录音 | 语音 | — | 只在相关工作里引用 |
| Seamless / Hibiki / SimulS2ST-Omni / LiveInterpret 2.0 | 单说话人同传 S2ST | — | 语音 | Seamless 开源；Hibiki 没有中英方向【需核对】 | 作为"不分说话人"的基线直接喂混合音频（第 4.2 节） |
| 多说话人识别方面（SOT / t-SOT、SURT、M2MeT、Raj 的博士论文 2402.08932、SLT 2026 SmartGlasses） | 说话人归属的 ASR | AMI、AliMeeting 等 | 文本（转写） | 部分开源 | 只能提供识别侧的模块或指标思路，不是翻译基线 |

**结论**：多说话人同传 S2ST 目前没有外部系统。唯一有发表数字、能直接对上的是 DiariST-AliMeeting 的文本翻译结果。

---

## 3. P0：零成本对比，DiariST-AliMeeting 的 SAgBLEU / SAtBLEU

**指标定义**【原文】：
- **SAgBLEU**：每个小段内，参考和输出都按开始时间拼接，不看说话人标签，算 corpus BLEU（SacreBLEU 2.3.1）。
- **SAtBLEU**：每个小段内，按说话人拼接，选使 BLEU 最大的说话人排列，说话人数不同时用空串补齐，再算 corpus BLEU（思路来自 cpWER）。

**已发表的 test 数字**【原文】：

| 系统 | SAgBLEU test CAT / MIX / SDM | SAtBLEU test CAT / MIX / SDM |
|---|---|---|
| 离线 SD→ST（ECAPA + NME-SC 聚类，然后 Whisper-small 翻译） | 14.47 / 10.40 / 7.88 | 12.94 / 9.31 / 6.80 |
| 离线 ST→SD（Whisper-small 长音频翻译，然后聚类） | 18.45 / 13.68 / 10.51 | 16.81 / 12.10 / 9.02 |
| 流式 1spk ST（Conformer transducer，1 s 块） | 20.63 / 16.15 / 13.46 | 17.38 / 13.39 / 10.77（+ d-vec） |
| **流式 DiariST**（t-SOT ST + t-vector） | 19.80 / 16.62 / 13.52 | **17.07 / 14.76 / 11.67** |

**做法**（只用 CPU）：
1. 确认我们 AliMeeting 测试窗口用的是哪个通道。原始会议有远场阵列和近讲麦克风两种录音，看对应 SDM、IHM-MIX 还是 IHM-CAT，再把我们的整场输出切成 DiariST 的 195 个小段。
2. 用 DiariST 的评测代码，算我们**文本输出**（以及语音经 ASR 后的文本）的 SAgBLEU / SAtBLEU。
3. 放进论文：一张"与 DiariST 已发表数字对比"的小表，同时脚注写明：
   - 我们是同传加语音输出，它只出文本；
   - 模型规模不同，它是 Whisper-small 和 Conformer；
   - 延迟没有可比数字，它的设置只是 1 s 块。

**注意**：
- 测试集要等最终系统定下来后才跑。这一步只是**给最终系统多算两个指标**，不另外动测试集。
- 不运行 DiariST，符合 11-03 的决定。

---

## 4. P1：开源基线，自己跑（dev 上调配置，test 只跑一次）

### 4.1 离线多说话人

| 基线 | 组成（都是开源模型） | 作用 | 估计 GPU |
|---|---|---|---|
| **B-off-1：现成级联，带说话人** | 说话人分离：pyannote 3.x / community-1 或 DiariZen；中文可以用 FunASR 的 CAM++ 加 Paraformer-zh-spk 一体化方案。<br>ASR：Whisper-large-v3（英）、Paraformer（中），或 Qwen3-ASR。<br>翻译：Qwen3 文本 LLM，或同一个 Qwen3-Omni Thinker 的文本模式（控制住"翻译模型强弱"这个变量）。<br>TTS：CosyVoice2，每个说话人用自己的音色提示克隆。 | 回答"现成开源模块拼起来怎么样"，审稿人最常要的就是这个。与内部"同数据重训的级联"互补 | dev + test 约 20–30 卡·时 |
| **B-off-2：Qwen3-Omni 原生** | 整窗音频输入，用提示要求按说话人逐段翻译并标注说话人；语音用它自带的 Talker | 端到端全能模型的直接对照；和我们是同一个 Thinker，能看出我们的方法带来多少增量 | 约 10 卡·时 |
| B-off-3：WhisperX 风格（Whisper-large-v3 长音频翻译 + pyannote 对齐） | 等于 DiariST 的 ST→SD 基线换成大模型 | 和 DiariST 的离线基线同一路线，规模升级 | 约 10 卡·时 |

### 4.2 同传（流式）多说话人

| 基线 | 组成 | 作用 | 估计 GPU |
|---|---|---|---|
| **B-sim-1：流式级联，带说话人** | 流式分离：NeMo Streaming Sortformer，或 diart，或我们自己的检测器。<br>流式 ASR：Paraformer-online 或 Qwen3-ASR 流式。<br>翻译：LLM 加 LocalAgreement-2 或 wait-k。<br>TTS：CosyVoice2 流式，每个说话人用自己的音色。 | 最强、最标准的模块化同传基线；重叠区、漏说话人、重复翻译这些对比，可以沿用 v8 计划里的指标 | 30–40 卡·时，含调试 |
| **B-sim-2：SeamlessStreaming，直接喂混合音频** | 开源，不分说话人，单路输出 | 说明单说话人同传系统在多人重叠场景里会怎样（重叠区掉分、说话人塌缩） | 约 10 卡·时 |
| B-sim-3：Hy-SimulMT（Apache-2.0，30B-A3B）+ TTS | 开源权重，只出文本；TTS 接 CosyVoice2 或我们的 talker | 2026 年开源同传的代表；和我们同为 30B-A3B | 约 15 卡·时 |
| （之前决定不实测）X-Translator | 开源代码 | 只放能力对比表；可选做 2–3 场会议的延迟实测 | – |

**怎么让这些基线和我们可比**：
- 输出不带说话人的系统（B-sim-2、B-sim-3、各种 API）：
  - 主要报"说话人无关"的指标：SAgBLEU、把所有输出当一路的 BLEU 和 ASR-BLEU、单通道 LAAL；
  - 同时给出 tcpBLEU，把输出全部算作一个说话人，作为下界；
  - **必须分 ov / 1spk 报告**：单人区它们可能不差，重叠区才是我们的卖点。
- 延迟：统一用检测器真实输出时刻的计时口径（11-12 起），报 LAAL、StartOffset、EndOffset。
- 流式基线如果自带计算时间（API 或实时运行），就另报计入计算时间的延迟。我们自己的系统还**没有**这项数字，要么补测，要么只在脚注里说明。

---

## 5. P2：GPT 类 API（工业参照）

| API | 能做什么 | 用法 | 成本（【需核对】） | 注意 |
|---|---|---|---|---|
| **OpenAI GPT-Realtime-Translate**（2026-05） | 流式语音到语音翻译，约 70 种输入语言、13 种输出语言 | 实时推流测试集音频，记下输出音频和时间戳 | 约 $0.034/分钟，所有测试集约 $50 | 不分说话人；延迟包含网络和服务排队，要说明 |
| **Google Gemini Live Translate**（2026-06） | 流式语音到语音翻译 | 同上 | 按 token 计 | 发布状态不确定（可能是预览版），要先确认中英方向是否可用 |
| **Qwen-LiveTranslate**（阿里云 API） | 流式语音到语音翻译 | 同上 | 按 token 计 | 中英强项 |
| **Seed LiveInterpret 2.0**（火山引擎 API） | 流式语音到语音翻译 | 同上 | – | SimulS2ST-Omni 就是用它的 API 测的 |
| **GPT-4o 或 Gemini Pro（离线音频理解）** | 整段音频，提示"按说话人逐段翻译"，输出文本 | 作为"大模型离线上限"的文本参照，语音用 CosyVoice2 合成 | 低 | 输出格式要严格解析；同一个提示，不做提示调优（只在 dev 上定一次） |

**建议**：至少选 **GPT-Realtime-Translate** 和**一家中文厂商**（Qwen-LiveTranslate 或 Seed LiveInterpret）做流式参照，再加 **GPT-4o 或 Gemini** 做离线参照。

**数据合规**：AMI（CC BY 4.0）和 AliMeeting 都是公开数据，可以发给 API，但要确认各家服务条款，以及中文重配音音频的授权。

---

## 6. 建议的论文结构

1. **主表（同传）**：我们的系统；内部策略基线（LA2、Lag、Blk2、Rel）；B-sim-1 流式级联；B-sim-2 SeamlessStreaming；API 参照。指标分 all / ov / 1spk，报 tcpBLEU、ASR-BLEU、SAgBLEU，延迟用统一口径。
2. **离线表**：离线 MIST；内部重训的级联；B-off-1 现成级联；B-off-2 Qwen3-Omni 原生；GPT-4o 或 Gemini 离线参照。
3. **DiariST 小表**（AliMeeting，只比文本）：SAgBLEU / SAtBLEU 对 DiariST 的已发表数字。
4. **能力对比表**（已有）：保留，并加上"论文里实际测过"一栏。

---

## 7. 工作量与排期（建议）

| 优先级 | 内容 | 资源 | 时长 |
|---|---|---|---|
| P0 | 我们 AliMeeting 输出的 SAgBLEU / SAtBLEU（只给最终系统算；开发期可以先在 dev 上验证脚本） | CPU | 半天 |
| P1 | B-off-1、B-off-2、B-sim-2（dev 上定配置，test 跑一次） | 评测类 GPU 作业，可以叠在非训练卡上 | 约 40–50 卡·时，2–3 天 |
| P1 | B-sim-1 流式级联（搭建工作量最大） | GPU + 工程 | 3–5 天 |
| P2 | API 参照（GPT-Realtime-Translate，加一家中文厂商；GPT-4o 或 Gemini 离线） | API 费用约 $100 以内 | 1–2 天 |
| P3 | Hy-SimulMT + TTS | GPU | 1 天 |

**需要你决定**：
1. 要不要上 API 参照：涉及费用，以及数据外发。
2. B-sim-1 流式级联用哪种分离器：Sortformer 流式、diart，还是我们自己的检测器（最公平）。
3. 是否维持"不运行 DiariST、不实测 X-Translator"的原决定。本方案只用 DiariST 的评测脚本和已发表数字。

---

## 来源
- DiariST：https://arxiv.org/abs/2309.08007 ，HTML 版 https://arxiv.org/html/2309.08007v2 ，代码和数据 https://github.com/Mu-Y/DiariST
- X-Translator：https://arxiv.org/abs/2607.17544 ，https://github.com/zhaoyx239/X-Translator
- Raj 博士论文（多说话人识别）：https://arxiv.org/abs/2402.08932
- Seed LiveInterpret 2.0：https://arxiv.org/abs/2507.17527
- SimulS2ST-Omni：https://arxiv.org/abs/2607.19810
- Hy-SimulMT：https://github.com/Tencent-Hunyuan/Hy-SimulMT
- 商用 API 动态（二手来源，需核对）：https://docs.litellm.ai/docs/realtime_translation.md ，https://www.eweek.com/news/google-gemini-live-translate-api-neuron/ ，https://www.orcarouter.ai/blog/qwen-3-8-live-translate-vs-gemini-3-5-live-translate
- 项目内已有的决定：`docs/mist_v8_plan.zh-CN.md`（修订 1）、`docs/mist_prereg_v9.zh-CN.md`
