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

## 四之二、保住说话人与韵律:这是我们现在最强的地方

### 4.2.1 先确认优势有多实在

CVSS-T Table 1 排名(1 = 最好,共 10 个方法):

| 指标 | 方向 | 我们的值 | 排名 |
|---|---|---:|---|
| **AutoPCP** | zh→en | 2.8837 / 2.9073 | **第 1/10 名** |
| **UTMOS** | en→zh | 3.8839 / 3.8855 | **第 1/10 名** |
| AutoPCP | en→zh | 2.7715 / 2.7889 | 第 4/10 |
| SLC-0.2 / SLC-0.4 | 双向 | — | 第 3/10 |
| *(对照)* Speech-BLEU | — | — | 第 8–10/10 |

**韵律和音质是我们唯二进前列的指标,换架构时必须当成约束,不能当成可牺牲项。**

### 4.2.2 拆开看:哪部分真的有风险

查了管线,说话人信息的来源是明确的 —— `evaluation/cvss_t/tokenize.py` 里:

```python
"direction": "cmn->eng", ... "bicodec_global": zh_global   # 源音频的 global
"direction": "eng->cmn", ... "bicodec_global": en_global   # 源音频的 global
```

**`bicodec_global` 是从源音频提取的 32 个全局 token,直接交给 BiCodec 解码器,完全不经过 LLM。**
(`merge_tokenized.py` 校验长度必须为 32。)

于是三种能力的风险完全不同:

| 能力 | 承载者 | 换 LLM 架构的风险 |
|---|---|---|
| **音色 / 说话人身份(SIM-O)** | 源音频的 32 个 global token,**绕过 LLM** | **零** |
| **韵律(AutoPCP)** | LLM 生成的语义码轨迹 | 有,见下 |
| **时长一致(SLC)** | 生成的码数量 | 有 |
| 音质(UTMOS) | 码质量 + 声码器 | 部分 |

### 4.2.3 韵律的风险:离线会降,流式会反超

论文 Table 1(**离线** CVSS-T)确实显示 Dec-only 声学更好:

| 架构 | A.PCP en→zh / zh→en | SIM-O en→zh / zh→en |
|---|---:|---:|
| Dec-only | **2.96 / 2.75** | **0.45 / 0.59** |
| Thinker–Talker | 2.70 / 2.64 | 0.35 / 0.45 |

**但那句论断的结尾限定词是 *"prior to trajectory finetuning"*。** 论文 §4(第 545 行)给出之后的情况:

> *"Unlike in the offline setting, **the streaming Talker consistently outperforms Dec-only on A.PCP**.
> More importantly, as the latency multiplier m increases, **the Talker's A.PCP scores actually exceed
> our offline S2ST baseline**. This proves that chunked text-code generation via trajectory supervision
> **actively improves local rhythm**, rather than merely preserving it."*

并且 Dec-only 在流式下还有**额外的延迟惩罚**(第 2193 行):
用单个 3B 主干预测稠密声学码是计算瓶颈,*"severely inflating its delay compared with the
lightweight 0.4B Talker"* —— 我们 AL 已经 4.6 s,这一条对我们是加分项。

**结论:轨迹监督不是可选项,它正是让 Talker 的韵律反超的机制。缺了它,换架构就是净损失。**

### 4.2.4 保住优势的四条硬约束

写进方案,任何一条不满足就不采纳新架构:

1. **保留 BiCodec,不换 DualCodec。** 论文换了码本,我们不换 —— 32 个 global token 的说话人通路、
   现有声码器、`StreamingBiCodecDecoder` 的 50 token 左上下文全部原样保留。
   Talker 的 codec embedding 与输出头按 **BiCodec 语义码本**重新初始化。
2. **轨迹监督必做**(Stage 3),否则只会拿到离线那版更差的 A.PCP。
3. **跨块音色延续要保留。** 论文保留最近 **4 s 源音频 + 2 s 已生成音频**作为 flow-matching 的
   prompt 上下文,*"improves timbre consistency and smooths transitions across chunks"*;
   我们现有的 50 token 左上下文是同一思路,迁移时不能丢。
4. **A.PCP 作为验收门,不是观察项。** 每个阶段都测,**低于当前的 2.7715 / 2.8837 即判定失败**,
   回退而不是"用翻译收益解释掉"。

---

## 五、实施计划:目标函数、训练过程、分阶段验收

### 5.0 先量到的机制:文本规划只拿到 4.8% 的梯度

当前是单一交叉熵、单一 180,480 路 softmax,文本与语义码在同一条序列里。
实测 RealSI 777 条:

| 方向 | 文本 token | 语义码 token | 码:文本 |
|---|---:|---:|---:|
| en→zh | 17.4 | 385.4 | **22.2×** |
| zh→en | 18.1 | 332.0 | **18.3×** |

**全量:文本 13,818 / 语义码 276,457 —— 码预测占梯度的 95.2%,文本规划只占 4.8%。**

这把论文那句 *"modality interference harms intermediate text planning"* 变成了一个具体数字,
也指出了两种强度不同的干预:**重新配平损失** 与 **拆成双流**。前者一行代码,后者要改架构。

---

### 5.1 阶段 A:损失配平 —— 完整的离线实施设计

**先做这一步,因为它不改架构、不改数据、不改推理,而且已核实可以零磁盘成本实现。**

#### A.1 已核实的实现路径

离线训练用的是 **Megatron 原版 `forward_step`,没有任何自定义 loss**
(`training/pretrain_uniss_megatron.py:301`)。Megatron 的 GPT 损失是:

```
loss = Σ(losses × loss_mask) / Σ(loss_mask)
```

而 `loss_mask` 是**逐 token 的 float32 权重**,由数据集直接读入
(`training/megatron_uniss_dataset.py:122`)。**所以配平是给这个数组赋非 1 的值,
不需要动任何训练代码。**

实测一条 18,000 token 的打包记录:

| 被监督位置 | 数量 | 占比 |
|---|---:|---:|
| 文本 | 1,028 | **8.1%** |
| BiCodec 语义码 | 11,352 | **90.0%** |
| 特殊 token | 239 | 1.9% |

**要让文本占 50% 的梯度,`w_text / w_code = 11.0`。**

⚠ **必须按 `labels` 索引,不是 `tokens`。** 实测 `labels[i] == tokens[i+1]` 占 99.761%
(差额来自打包样本边界),两者在模态边界处的计数不同(文本 985 vs 1028)。
按 `tokens` 加权会在每个模态切换点错配一个位置。

#### A.2 代码改动(约 5 行,默认恒等)

改在 `training/megatron_uniss_dataset.py` 的 `packed_json_to_megatron_item`:

```python
loss_mask = _tensor_from_int_list(item, "loss_mask", seq_length, torch.float32)
# 默认 1.0/1.0 即恒等,现有实验完全不受影响
if TEXT_LOSS_WEIGHT != 1.0 or CODE_LOSS_WEIGHT != 1.0:
    is_text = labels <= c.BICODEC_GLOBAL_OFFSET - 1          # 0..151,664
    loss_mask = loss_mask * torch.where(
        is_text, TEXT_LOSS_WEIGHT, CODE_LOSS_WEIGHT)
```

权重由环境变量注入,默认 1.0 —— **不新建数据、不占磁盘、不改既有行为**。
(若改为重写打包文件,每个变体要 385 GB,而当前余量只有 2.4 TB。)

**必须配的测试**:
1. 权重为 1.0 时产出与现在逐位相同(恒等);
2. 权重生效时,文本位置的 mask 恰为 `w_text`、码位置恰为 `w_code`;
3. 按 `labels` 而非 `tokens` 判断 —— 构造一个模态边界样本断言边界那一位的归属。

#### A.3 训练过程

**起点**:现有 Phase2 终检查点,**不重跑 Phase1/Phase2**。

```
LOAD   checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/iter_0015381
语义   FINETUNE=1  LOAD_OPTIM=0  LOAD_RNG=0        # 新阶段,不带上阶段 optimizer
数据   data/megatron/phase3_unist198/packed_train.jsonl   (1,161,587 条,不变)
几何   8×H200,TP1/PP1,mbs 2,gbs 128,seq 18000,BF16
日程   9,075 步,LR 1e-5 → 1e-6 cosine,warmup 200,clip 0.5
           cyclic shuffle,--no-data-sharding,full validation,eval 每 100 步
隔离   新的 SAVE_DIR / RUN_DIR / LOG_PATH,绝不覆盖现有 Phase3
```

**两个臂**(顺序跑,各约 8.1 小时):

| 臂 | w_text | w_code | 文本梯度占比 | 用意 |
|---|---:|---:|---:|---|
| `phase3_wt4` | 4 | 1 | **26.5%** | 温和,先看方向 |
| `phase3_wt11` | 11 | 1 | **49.8%** | 文本与码等权 |

先跑 `wt4`。若 Text-BLEU 有提升且 A.PCP 未跌,再跑 `wt11` 看是否单调。

#### A.4 评估

复用现成链路,**一行不改**:

```bash
CUDA_VISIBLE_DEVICES=0 experiments/evaluation/uniss_full198_phase2_phase3/export_exact.sh phase3
# 注意:HF_OUTPUT 必须是新路径,脚本拒绝覆盖已有导出

experiments/evaluation/cvss_t_zh_en_phase3_v1/run_full_evaluation.sh
```

必须沿用 `whisper-large-v3-attention-mask-v2` 协议,并同时报告泄漏审计。

#### A.5 验收门

对照当前 Phase3(Quality 模式):

| 指标 | 当前 | 判定 |
|---|---:|---|
| **Text-BLEU** en→zh / zh→en | 24.15 / 15.33 | **提升 ≥ +1.5 → 机制证实** |
| Speech-BLEU | 23.58 / 12.05 | 同向变化 |
| **A.PCP** en→zh / zh→en | 2.7889 / **2.9073** | **跌超 0.05 → 判失败回退** |
| **UTMOS** en→zh | **3.8855** | 跌超 0.05 → 判失败回退 |
| SLC-0.2 | 0.7022 / 0.6343 | 不显著退化 |

**A.PCP 与 UTMOS 是否决项,不是观察项。** 削弱码的梯度本来就可能伤韵律与音质,
这正是这一轮最需要盯的风险。

#### A.6 成本与时间线

| 步骤 | 内容 | 时间 |
|---|---|---|
| 1 | 实现 + 单测 | 半天 |
| 2 | `phase3_wt4` 训练 | **8.1 h** |
| 3 | 导出 + CVSS-T 全量评测 | 约 1 天 |
| 4 | (视结果)`phase3_wt11` | 8.1 h + 1 天 |

**约两天拿到第一个答案**,而且不触碰 Phase1/Phase2、不触碰流式线、不新增数据。

#### A.7 这一步也为阶段 B 提供定量输入

阶段 B 的双流损失 `L = L_text + λ·L_code` 里的 `λ`,论文正文没有给。
阶段 A 扫出的最优文本梯度占比,**就是 λ 的直接标定** ——
所以即使阶段 A 的收益有限,它也不是白跑。

### 5.2 阶段 B:拆双流(方案 0 —— 不换基座、不换编码器)

#### B.1 替换清单

| 组件 | 现在 | 改为 | 依据 |
|---|---|---|---|
| 码预测路径 | 与文本共用 AR 头 | **独立 Talker** | 论文 +4.00 ASR-BLEU 的来源 |
| Thinker 输出头 | 180,480 路 | 切到 `0..151,664`(文本) | — |
| Talker 输出头 | — | `155,761..163,952`(**8,192** 码) | 我们的 BiCodec 码本 |
| Talker 初始化 | — | **现有 decoder 的副本** | 它已训 43k 步生成这套码;论文说 Omni 初始化只是 *"a modest optimization aid"* |
| 交叉条件通路 | 无 | **H_θ → Talker**(新增) | 式 (4) |
| 嵌入表 | `tie_word_embeddings=True`,文本与码共享 | **拆开**(新增 8,192×896 ≈ 7.3M 参数) | 共享嵌入是模态耦合的一部分 |
| 音频编码器 | WhisperVQ + GLM bridge | **不动** | 动它会废掉流式管线与本轮 DPO 成果 |
| BiCodec / global / 声码器 / 流式解码器 | — | **不动** | 说话人通路与跨块音色 |
| 评测链 | — | **不动** | 保证与历史可比 |

#### B.2 目标函数

**不变的是轨迹定义与块分解(论文式 1、式 2,两种架构共用):**

```
C_c = ( X_{1:g_c},  Y_{<c}^text,  Y_{<c}^code )                      (1)

log p(Y^text, Y^code | X, τ) = Σ_{c=1..C} log p(Y_c^text, Y_c^code | C_c)   (2)

离线是 C=1、g_1=|X| 的特例;流式是 C>1 —— 同一目标覆盖两者。
```

**变的是 p(· | C_c) 的参数化:**

```
现在(Dec-only,式 5)
    p_θ(Y_c^text, Y_c^code | C_c),   V = V_text ∪ V_code
    L = −Σ_c Σ_{v∈chunk} log p_θ(v | ·)                    # 一个 CE,180,480 路

阶段B(Thinker–Talker,式 4)
    p(Y_c^text, Y_c^code | C_c)
        = p_θ(Y_c^text | C_c) · p_φ(Y_c^code | C_c, Y_c^text, H_θ)

    L = Σ_c [ L_text(c) + λ · L_code(c) ]

    L_text(c) = −Σ_{i∈Y_c^text} log p_θ(y_i | C_c, y_<i)             # 151,665 路
    L_code(c) = −Σ_{j∈Y_c^code} log p_φ(z_j | C_c, Y_c^text, H_θ, z_<j)  # 8,192 路
```

其中 `H_θ` 是 Thinker 生成本块文本时的中间隐状态序列。
**关键:Talker 不预测文本 —— 它在文本已定之后才生成码**,这就是保护文本规划的机制。

**λ 的设定**:论文正文未给。鉴于 5.0 测到的 18–22× token 比,
建议 `λ` 使两项损失**按各自 token 数归一后等权**,并在阶段 A 的结果上校准。

#### B.3 训练过程:混合配方,不照抄任何一方

我们的起点与两篇论文都不同 —— **主干已过完整 UniSS Phase1–3(43k 步)**,
语音-文本对齐早已建立,所以照抄 SimulS2ST-Omni 的三阶段会有冗余。

```
Stage 0  手术与空操作验证                                         【新增,论文没有】
    从现有检查点复制出 Thinker 与 Talker,切头、建交叉通路。
    验收:用 gold 文本喂 Talker、两者同权重初始化时,
          码预测应复现当前模型的输出。不复现 = 手术错了。
    成本:零训练,纯前向比对。

Stage 1  Talker warmup                                    【抄 SimulS2ST-Omni】
    冻结 Thinker,只训 Talker 与新的交叉通路,仅用 TTS 数据。
    论文为 2 epoch;我们的 Talker 已会生成这套码,预计更短。
    验收:TTS 重建的 A.PCP 不低于当前管线。

Stage 2  轻量联合适配                                  【缩水版,非论文完整 Stage 2】
    Thinker 与 Talker 各挂 LoRA。
    目的仅为让 Thinker 适应"不再预测码"的输出分布,
    **不是**重新学对齐 —— 那在 43k 步里已经做过。
    ⚠ 任务定义沿用 UniSS 的 Quality / Performance 双模式,
      不照抄论文 0.2:1:0.5:1:1.5 的混合比 —— 那里面没有 Q/P 概念,
      照抄会丢掉我们 Q 比 P 高 3.57–5.06 BLEU 的能力。

Stage 3  流式轨迹微调                                     【抄 SimulS2ST-Omni】
    合并 Stage-2 adapter,挂新 LoRA,冻结 embedding 与预测头,
    在块轨迹上微调。
    **基建已有**:`uniform_chunk_tasks.py` 的固定块重分箱、
    NIR 单调性过滤、轨迹池都已实现并验证。
    这一步是 A.PCP 反超的来源,不可省。
```

#### B.4 每阶段的验收门

| 阶段 | 必须满足 | 失败处理 |
|---|---|---|
| Stage 0 | 码预测复现当前模型 | 手术有误,修正后重来 |
| Stage 1 | TTS 的 A.PCP ≥ 当前 | Talker 容量或交叉通路有问题 |
| Stage 2 | CVSS-T Text-BLEU > 24.15 / 15.33 | 回退到阶段 A 的结果 |
| **Stage 3** | **A.PCP ≥ 2.7715(en→zh)/ 2.8837(zh→en)** | **回退,不得用翻译收益解释掉** |
| Stage 3 | RealSI 流式指标不低于 pf_guard | 回退 |

---

### 5.3 阶段 C:换基座(只在 A、B 都做完后再评估)

若阶段 B 证实双流有效,再考虑放大主干或换 Omni。
**此时才有依据判断**:双流的收益是否依赖大主干。

**注意成本**:换 Omni 原生编码器会使 p2st 级联、Stage A/B、本轮 DPO 全部失效
(级联的 ASR prompt 建在 `bridge_projection` + `GLM_SEMANTIC_OFFSET` 上,
WhisperVQ 被六个以上实验脚本依赖)。放大主干(Qwen2.5-1.5B,已在本地,
tokenizer 相同)不触发这个问题,是更稳的放大路径。

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
