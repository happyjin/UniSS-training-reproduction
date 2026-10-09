# Offline 翻译质量提升:换模型方案与完整训练计划

2026-09-21。依据两篇论文的实测证据,不是推测:

- **SimulS2ST-Omni**,arXiv [2607.19810](https://arxiv.org/pdf/2607.19810),全文在 `data/external/simuls2st_omni_demo/paper.txt`
- **CUHKSZ IWSLT 2026**,[aclanthology 2026.iwslt-1.13](https://aclanthology.org/2026.iwslt-1.13/)

---

## 一、结论先行

**换大模型有用,但光换大模型不够。** 两件事必须同时做:

1. **换基座**:从"文本 LLM + 追加语音 token"换成**原生音频-文本对齐的 LLM**;
2. **换架构**:从 Dec-only(单解码器同时吐文本和语义码)换成**双流 Thinker–Talker**。

论文给了一个**完全匹配的对照**证明第 2 条不可省略:同样 3B 基座、同样数据、同样 tokenizer,
**Thinker–Talker 比 Dec-only 高 +4.00 ASR-BLEU(en→zh)**,而且 **Dec-only 还多烧 2 倍 GPU 小时**。

另外一个关键的成本事实:**不需要重做我们现在这套 43k 步的全量预训练**。
SimulS2ST-Omni 全程是 LoRA + 三阶段轻量适配,用 **~2k 小时**配对 S2ST(砍 90% 仍鲁棒);
IWSLT 那篇更极端,只在 Thinker 的 q/k/v/o 上插 r=16 的 LoRA、3 个 epoch、
220K 块级样本就拿到了 SOTA 级结果。

**但 IWSLT 那篇不能当作换基座的模板** —— 它禁用了 Talker,是语音→文本系统,
不产生语音。它可借鉴的是**数据构造方法**与 **`<wait>` 策略内化**,见 §2.3 与 §4.1。

---

## 二、两篇论文的核心证据

### 2.1 CVSS-T 同表对照(SimulS2ST-Omni Table 1)

| 系统 | 参数 | 架构 | ASR-BLEU en→zh / zh→en | Text-BLEU en→zh / zh→en |
|---|---|---|---:|---:|
| **我们 Phase3 (Q)** | **0.5B** | **Dec-only** | **23.58 / 12.05** | **24.15 / 15.33** |
| Ours (Dec-only) | 3B | Dec-only | 27.12 / 23.41 | 31.59 / 24.90 |
| **Ours (Thinker–Talker)** | 3B | **双流** | **31.12 / 25.18** | **33.73 / 26.41** |
| UniSS (Q) | 1.5B | Dec-only | 32.04 / 24.72 | 32.95 / 25.51 |
| Seamless-m4t-v2-large | 2.3B | — | 20.86 / 22.25 | 20.75 / 22.60 |
| Qwen-Omni2.5 | 7B | — | 8.04 / 22.66 | 34.85 / 24.39 |

两条独立结论:

* **规模有用**:同为 Dec-only,0.5B → 3B 让 zh→en ASR-BLEU 从 12.05 升到 **23.41(近一倍)**;
* **规模不够**:匹配 3B 下双流仍 **+4.00**。论文原话:
  *"forcing a unified decoder to predict dense semantic codes introduces modality
  interference that harms intermediate text planning"*。

还有一个反向结论值得记住:**Dec-only 在声学质量上反而更好**(A.PCP、SIM-O 更高,甚至超过 UniSS),
论文称之为 *"intelligence-versus-quality trade-off"* —— Dec-only 用整个 3B 去生成码,
声学细节更丰富但语言智能受损;双流把码生成限制在 0.4B 的 Talker,牺牲声学换翻译保真度。
**我们现在的短板是翻译而不是音质,所以应该选双流。**

### 2.2 数据效率:不需要海量配对数据

SimulS2ST-Omni 只用 **~2,000 小时**中英配对 S2ST,由公开 ASR/S2TT 语料经跨语言对齐 +
单调性过滤构造,叠在大规模辅助 ASR/S2TT/MT/TTS 监督之上,
且 *"even when the paired-S2ST budget itself is reduced by 90%"* 仍鲁棒。

CUHKSZ 更进一步:用 **Qwen3-30B-Instruct 合成翻译目标**,从现成 ASR 语料构造
syntax-aware、chunk-aligned 的监督,**完全不需要配对 S2ST 语料**。

**这直接否定了"我们差是因为没有 WMT17 2.3B token"这个解释。**

### 2.3 CUHKSZ 的路线(IWSLT 2026)—— 注意:这是语音→**文本**

**先说清边界,否则会误用。** 这套系统**不产生语音**:论文明确禁用了
Qwen3-Omni 的 Talker(语音合成)与 vision 模块,
原话 *"our task produces text from audio and never routes information through
these modalities"*。因此:

* 它的 **40.5 BLEU 是 MCIF 上的 S2TT BLEU**,与我们 CVSS-T 的 ASR-BLEU **不可比**;
* **它不能作为我们 S2ST 的换基座模板** —— 我们需要语音输出,而它把语音输出模块关掉了。

```
基座    Qwen3-Omni-30B-A3B(原生音频-文本对齐,MoE 30B 总 / 3B 激活)
        音频编码器冻结;Talker 与 vision 训练时禁用
适配    仅在 Thinker 的 q_proj/k_proj/v_proj/o_proj 上插 LoRA
        r=16, α=32, dropout 0.05, peak LR 1e-4(En→Zh), batch 128, 3 epoch, 8×A100
数据    LibriSpeech 960h + CommonVoice17 1,470h + CoVoST2 425h + VoxPopuli 530h
        = 3,385 小时 → 质量过滤后 **220K 块级样本**
执行    vLLM 上的轻量 agent,单张 A800;唯一的延迟参数是 chunk_sec
结果    En→Zh 低延迟档 40.46 BLEU / 73.54 XCOMET @ 1954 ms(CA LongYAAL)
        高延迟档 42.14 / 75.74 @ 2164 ms
```

**但它有两项对我们真正可迁移的贡献:**

**(a) 用一次 LLM 调用取代三个串联模型的数据构造。**
传统做法(含我们的 Stage-A)靠 spaCy 分块 + Whisper 词级时间戳 + SimAlign 双语对齐
三者交集来启发式地推出 `<wait>` 决策,论文批评其
*"compounding errors and disagreements among models that were never explicitly
trained to be mutually consistent"*。他们改用**单个文本 LLM(Qwen3-32B)一次调用**
同时产出:句法感知源分块、块级双语对齐、目标侧重排序、`<wait>` 决策,
输出 JSON;**唯一的非 LLM 信号是一次 Whisper-large-v3 强制对齐**拿时间戳。

四条**可后验校验**的硬约束写在 prompt 里:
每块 ≤7 个源词;所有源块拼接 == 原转录(忽略空白);
所有非 `<wait>` 目标块拼接 == 参考译文;最后一块不得为 `<wait>`。

四维质量过滤(权重 **0.3:0.3:0.3:0.1**):时间戳完整性/单调性、文本长度与完整性、
对齐一致性、**决策平衡度**(奖励每句 READ:WRITE 接近语料均值 **约 3:1**),
并刻意保留一定比例的边缘样本(极短/极长/疑问句/异常标点)防止分布变窄。

**(b) 把 read/write 策略内化成 `<wait>` token,且推理时可无重训调档。**
扫 `chunk_sec` 从 0.64 s 到 6.40 s 即可遍历质量/延迟曲线,**不需要重训**;
而且**改变 chunk 大小不改变 `<wait>` 决策的比例**,模型改为调整每步输出的 token 数 ——
论文据此认为句法感知监督泛化到了训练时的静态边界之外。
919 条中空输出 ≤0.4%。

---

## 三、诊断:我们的差距具体在哪

**不是均匀落后。** 和匹配架构的 3B Dec-only 比:

| 方向 | 指标 | 我们 0.5B | 他们 3B Dec-only | 差距 |
|---|---|---:|---:|---:|
| en→zh | ASR-BLEU | 23.58 | 27.12 | **−3.54** |
| **zh→en** | **ASR-BLEU** | **12.05** | **23.41** | **−11.36** |
| en→zh | Text-BLEU | 24.15 | 31.59 | −7.44 |
| **zh→en** | **Text-BLEU** | **15.33** | **24.90** | **−9.57** |

**zh→en 的差距是 en→zh 的 3.2 倍。问题集中在英文侧。**

再看我们自己内部的 text → speech 损失:

| 方向 | Text-BLEU | Speech-BLEU | 损失 |
|---|---:|---:|---:|
| en→zh | 24.15 | 23.58 | −0.57 |
| **zh→en** | **15.33** | **12.05** | **−3.28** |

所以 zh→en 坏在**两处叠加**:

1. **翻译成英文的文本质量本身就低**(15.33 vs 24.90);
2. **英文语音生成又额外丢 3.28 BLEU**(中文方向只丢 0.57)。

第 2 条正是 Dec-only 的模态干扰在起作用 —— 这也是双流架构直接针对的问题。

---

## 四、选项与成本

0.5B 三阶段实测约 **3.2 秒/步**,8×H200,合计:

```
Phase1 18,765 步 ≈ 16.7 h
Phase2 15,381 步 ≈ 13.7 h
Phase3  9,075 步 ≈  8.1 h
合计            ≈ 38.5 h
```

| 方案 | 基座 | 架构 | 预计收益 | 成本 | 风险 |
|---|---|---|---|---|---|
| **A. 只放大** | Qwen2.5-1.5B(**已在本地**) | Dec-only 不变 | 参考 UniSS 1.5B:ASR-BLEU 约 32/24.7 | 约 **5–6 天**(3.6× 参数) | 低。但保留了论文证明较差的架构 |
| **B. 换 Omni + 双流** | Qwen2.5-Omni-3B/7B | **Thinker–Talker** | 参考论文:**31.12 / 25.18** | LoRA 三阶段,**比 A 省一半 GPU 小时** | 中。需实现双流,DualCodec 适配 |
| ~~C. 走 CUHKSZ 路线~~ | Qwen3-Omni-30B-A3B | 单流 + `<wait>` | **不适用** | — | **排除:该系统禁用 Talker,只出文本,不做 S2ST** |

### 为什么不推荐 A(只放大)

它把 38.5 小时变成 5–6 天,换来的是**沿着一条论文已证明次优的架构往上爬**。
按 Table 1,1.5B Dec-only(UniSS)确实能到 32.04/24.72,但那是 UniSS 用
WMT17 2.3B token + 7.71 万小时语音训出来的;我们只有公开 UniST,
**同样的数据放大模型,很可能到不了那个数**。

### 为什么推荐 B

* 论文在**匹配条件**下证明双流 +4.00 ASR-BLEU,且**省一半 GPU 小时**;
* 原生音频对齐的基座跳过了"给文本 LLM 硬塞 28,471 个语音 token"这一步 ——
  我们当前词表 151,936 → 180,407 的扩展正是模态干扰的来源;
* 它直击我们最大的短板(zh→en 英文生成);
* 数据上只要 ~2k 小时配对 S2ST,我们现有的 UniST 198 shard 远超这个量。

---

### 4.1 从 CUHKSZ 借什么(不换基座也能用)

它的基座路线不适用于 S2ST,但两项方法是**独立于基座**的,可以直接搬到我们现有管线:

| 借鉴项 | 替换我们的什么 | 预期好处 |
|---|---|---|
| **单 LLM 调用产出分块+对齐+重排+`<wait>`** | Stage-A 的多模型对齐链路 | 消除串联误差;我们现有 NIR 单调性过滤可保留为后验校验 |
| **四条硬约束 + 四维质量过滤(3:1 决策平衡)** | 我们的配对筛选 | 我们实测空提交步占 54.2%(约 1:1),与其 3:1 的目标分布差距很大,值得对照 |
| **`<wait>` token 内化策略** | 我们固定的读步 + 门槛 | 一个检查点靠 `chunk_sec` 遍历延迟档,**不重训** |

第三项与 REINA 是**同一目标的两条路**:REINA 外挂一个 6M 策略头估信息增益,
CUHKSZ 把决策写进词表让模型自己学。后者实现更简单(只需在训练数据里插 `<wait>`),
但需要重新构造带 `<wait>` 标注的训练集。

---

## 五、推荐方案:分三步走,每步有决策门

### 第 0 步(先做,2–3 小时,纯推理):判别实验

**在投入任何训练之前,先把"规模 / 架构 / 数据"三个因素拆开。**

| 实验 | 做法 | 回答什么 |
|---|---|---|
| **0a 纯文本 MT 对照** | 用**未经 UniSS 训练的** Qwen2.5-0.5B / 1.5B,在 CVSS-T 的 4,897 条参考文本上做纯文本 zh↔en 翻译,比 BLEU | 同家族仅规模差 3.6×,在这个语言对上 MT 能力差多少 |
| **0b UniSS 训练的代价** | 把 0a 的基座 BLEU 与我们 Phase3 的 Text-BLEU(24.15 / 15.33)比 | **扩词表 + 三阶段训练到底损失了多少原生 MT 能力** |
| **0c 英文生成专项** | 只看 zh→en:基座纯文本 BLEU vs 我们的 15.33 | 确认英文短板是基座自带还是训练引入 |

**决策门**:
* 若 0b 显示我们的 Text-BLEU **远低于**基座原生 MT 能力 → 问题在训练配方,**换模型救不了**,应先修配方;
* 若基座原生能力本身就低 → 规模/基座是真瓶颈,进入第 1 步。

### 第 1 步:获取并验证 Omni 基座(半天)

```bash
# 下载 Qwen2.5-Omni-3B(或 7B,视显存与目标而定)
scripts/download_hf_assets.sh   # 需新增 omni 条目
# 验证:音频编码器、Thinker、Talker 三部分完整,且能离线加载
```

**决策门**:Talker 的 codec embedding 与输出头能否按 DualCodec/BiCodec 词表重新初始化。
论文原文:*"the codec embedding and output head re-initialized for the DualCodec vocabulary"*。
我们用的是 BiCodec,需确认码本规模可对齐。

### 第 2 步:双流三阶段训练(按论文配方)

论文的三阶段,两种架构**共享完全相同的数据混合、优化器与 token 预算**:

```
Stage 1 (Warmup)
  Thinker–Talker:只训 Talker,数据为 TTS
  (Dec-only 需要双分支 LoRA 预热防模态欠拟合 —— 又一条双流更简单的证据)

Stage 2 (Joint Pretraining)
  ASR : S2TT : MT : TTS : S2ST = 0.2 : 1 : 0.5 : 1 : 1.5
  两种架构同一混合比

Stage 3 (Streaming Finetuning)
  合并 Stage-2 adapter,挂新的 LoRA,冻结 embedding 与预测头,
  在增广的流式轨迹上微调(ASR/S2TT/MT/S2ST)
```

**数据构造**(我们已有大部分工具):

| 论文做法 | 我们的现状 |
|---|---|
| 从公开 ASR/S2TT 语料经跨语言对齐 + 单调性过滤构造 ~2k 小时配对 S2ST | `data/processed/phase2_unist198_sharded` 已有 5,935 万条;**NIR 单调性过滤已实现**(`experiments/uniss_streaming_p2st_traj_v1/data/nir_score.py` / `nir_stratify.py`) |
| 按 NIR 做难度/长度分层配额 | 已实现并验证过(池内 NIR 均值 11.64%) |
| 转录级清洗:两套 ASR 交叉核验(Qwen3-ASR-1.7B + Whisper-large-v3),按 WER/长度阈值保留 80–95% | **未做**,需新增 |

### 第 3 步:评估与对照

复用现成的 CVSS-T 评测链(**完全不用改**):

```
experiments/evaluation/cvss_t_zh_en_phase3_v1/run_full_evaluation.sh
```

对照必须包含:
* 我们现在的 0.5B Dec-only(23.58 / 12.05);
* 新基座的 Dec-only(若做了 A 作为对照);
* 新基座的 Thinker–Talker;
* 论文 Table 1 的 3B 两行。

**注意评测口径**:报告须沿用 `whisper-large-v3-attention-mask-v2` 协议
(旧版 batched Whisper 未传 attention mask,曾把 zh→en Speech-BLEU 从 6.99 错报为 1.75),
并**同时报告数据泄漏审计**(当前归一化文本命中 1,705 条训练记录)。

---

## 六、成本汇总

| 阶段 | 内容 | 预计 |
|---|---|---|
| 第 0 步 | 判别实验,纯推理 | **2–3 小时** |
| 第 1 步 | 下载 + 验证 Omni 基座 | 半天 |
| 第 2 步 | 双流三阶段 LoRA 训练 | 按论文,**约为 Dec-only 全量的一半 GPU 小时** |
| 第 3 步 | CVSS-T 完整评测 | 约 1 天(现成链路) |

对比:方案 A(只放大到 1.5B 全量重训)单是训练就 **5–6 天**,且保留次优架构。

---

## 七、风险

1. **BiCodec vs DualCodec 不匹配** —— 论文用 DualCodec,我们用 BiCodec。
   Talker 的 codec embedding/输出头需按我们的码本重新初始化,码本规模若差异大需额外验证。
2. **Qwen2.5-Omni 的许可与可获得性** —— 需确认能离线获取;本机当前 `HF_HUB_OFFLINE=1`。
3. **双流实现是新工作** —— 论文无开源代码(SimulS2ST-Omni 的 GitHub 有 agent/训练骨架,
   见 `data/external/simuls2st_omni_demo/repo/`,但双流的完整训练需自行补齐)。
4. **磁盘** —— 当前 `/opt/dlami/nvme` 余 2.4 TB(92% 已用)。新基座 + 新 checkpoint 需
   先清点空间,**不删任何现有数据**。
5. **不能用论文数字当承诺** —— 他们的 3B 用的是 Qwen2.5-Omni + DualCodec + 自建 2k 小时配对数据;
   我们换基座后的实际值必须自己测出来。

---

## 八、不做的事

* **不动现有的 streaming 实验**(`uniss_streaming_p2st_*`)与其检查点;
* **不删除**任何现有数据、checkpoint 或报告;
* **不在第 0 步之前启动任何训练** —— 判别实验的成本是训练的千分之一,
  而它可能直接否掉"换模型"这个方向;
* 不把论文 Table 1 的数值当作我们换模型后的预期值写进汇报。

---

## 九、立即可执行的下一步

```bash
# 第 0 步判别实验(纯推理,不占训练资源)
# 0a: 基座原生 MT 能力
#   输入: CVSS-T 4,897 条参考文本
#   模型: pretrained_models/Qwen2.5-0.5B-Instruct 与 Qwen2.5-1.5B-Instruct
#   输出: 双向 BLEU,与我们 Phase3 的 24.15 / 15.33 对照
```

该实验的三种结果对应三条不同的路:

| 结果 | 含义 | 下一步 |
|---|---|---|
| 基座原生 BLEU ≫ 我们的 Phase3 | 训练配方把 MT 能力训坏了 | **先修配方,不换模型** |
| 基座原生 BLEU ≈ 我们的 Phase3,且 1.5B ≫ 0.5B | 规模是瓶颈 | 进入第 1 步,换 Omni + 双流 |
| 基座原生 BLEU ≈ 我们的 Phase3,且 1.5B ≈ 0.5B | 瓶颈在数据 | 先补 MT 数据,换模型收益有限 |
