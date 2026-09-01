# 换思路:把"何时开口"从模型里拿出来 —— 文献扫描 + 本项目实测 + 方案 v4

> 本文取代 plan v3。v3 的核心处方(为 WAIT/WRITE 决策写 margin loss)已被本项目
> **三次独立实验证伪**,而且失败模式在文献里有名字、有诊断、有解法。

---

## 0. 一句话结论

**UniSS 原始配方是纯 next-token CE + CoT prompt(其论文已确认无任何辅助 loss)。
本项目为了做 streaming,在序列里加了 WAIT/WRITE 动作 token,然后把它们的梯度
权重压低了 7 倍,再用 14 个加权目标去补 —— 方向反了。文献里几乎没人用动作
token 的 CE 学时机:主流做法是离线纯 CE 训练 + 推理侧策略,或者把时机烘进
prefix-to-prefix 数据里。而本项目**恰好已经有一条纯 prefix-to-prefix 通路,
并且它是全项目质量最高的通路**。**建议把主线切换到那条通路上。**

---

## 1. 本项目失败的完整账目(实测)

### 1.1 三次尝试用 loss 学"开口决策",三次都反向

推理侧决策 logit 差(`logit[WRITE_GENERATE] − logit[WAIT_READ]`,事件内第 2 次决策,
同一协议 `family_logit_probe` 测得):

| checkpoint | 训练侧 gold 行位移 | **推理侧 gap** |
|---|---:|---:|
| `iter_0002264` 基线 | — | **−2.88** |
| `speak_decision`(margin 1.0,权重 0.5) | +0.105 | **−3.75** |
| `continue_end`(margin **3.0**,权重 0.5) | +0.40 | **−4.97** |

**单调反向,margin 越大偏得越远。** gold 行方向每次都对,推理侧每次都反。

`WRITE_MT`/事件:0.168 → 0.147 → 0.147。`natural_eos`:0.50 连续五个 epoch 不动。

### 1.2 梯度分配:决策被刻意压低 7 倍

交织 gold 序列(50 条 packed record,668,953 个监督 token):

| kind | token 数 | 占比 | 当前权重 | **梯度占比** | 均匀 CE 下 |
|---|---:|---:|---:|---:|---:|
| semantic | 402,953 | 60.2% | 1.00 | **85.8%** | 60.2% |
| **boundary**(含 WAIT/WRITE、family、END_CONTENT) | **219,735** | **32.8%** | **0.10** | **4.7%** | **32.8%** |
| asr | 25,352 | 3.8% | 1.00 | 5.4% | 3.8% |
| mt | 19,377 | 2.9% | 1.00 | 4.1% | 2.9% |

`boundary_eos = 0.10` 在**全部实验脚本里从未被改过**。目标函数 85.8% 被语音
token 预测占据。

### 1.3 零训练的推理侧偏置一次就成功

给续写决策加 δ 个 logit(`biased_family_probe`,不改任何权重):

| δ_cont | `WRITE_MT` | natural_eos | 语义覆盖 | 文本长度比 |
|---:|---:|---:|---:|---:|
| 0 | 0.168 | 0.50 | 0.666 | 1.03 |
| 2 | 0.242 | 0.62 | 0.708 | 1.42 |
| **3** | **0.863** | **1.00** | **0.997** | 2.25 |
| 4 | 0.947 | 1.00 | 0.997 | 2.88 |

**用 loss 一年没推动的东西,推理侧 3 个 logit 一次到位。**

### 1.4 纯 prefix-to-prefix 通路才是本项目最强的通路

`incremental_mt_rollout` 只是遍历源前缀调 `generate_mt_prefix`,**内部没有
`_choice`,不经过任何开口决策**。同一门禁里:

| | `iter_2264` | `continue_end` 新 ckpt | 离线 phase3(增量协议) |
|---|---:|---:|---:|
| gold-src BLEU cmn→eng | 18.98 | **27.67** | 9.97 |
| gold-src BLEU eng→cmn | 27.66 | **35.86** | 3.92 |
| free-src BLEU eng→cmn | 4.99 | **17.51** | 3.92 |
| 英文流式 ASR 错误率 | 0.303 | **0.194** | — |

而交织会话(带学出来的决策)的会话文本覆盖只有 **0.30**。

**能力在 P2P 通路里,坏掉的是动作 token 那条通路。**

### 1.5 关键新证据:模型没有内部时钟

`constants_uniss.py` 里 token 家族全查过 —— **没有任何时间/时长/chunk/step 类
token**。决策位置唯一的隐含时间信号是已追加的 GLM 块数量,埋在 18000 token
的异质长序列里。

实测(按事件的 `source_end_ms` 分桶):

| 已听时长 | 事件数 | `iter_2264` 开口率 | 新 ckpt 开口率 |
|---|---:|---:|---:|
| 0–1s | 19 | 0.158 | 0.158 |
| 1–2s | 25 | 0.200 | 0.200 |
| 2–3s | 23 | 0.087 | 0.043 |
| 3–4s | 14 | 0.143 | 0.143 |
| 4–5s | 10 | 0.300 | 0.300 |
| 5–6s | 3 | 0.000 | 0.000 |
| **开口率对时长的斜率** | | **−0.0124 /秒** | **−0.0111 /秒** |

**斜率约为零且略负 —— 听了 6 秒和听了 1 秒一样不肯开口。**

---

## 2. 文献扫描

### 2.1 UniSS 自己的论文([arXiv:2509.21144](https://arxiv.org/abs/2509.21144),ICLR 2026)

* **loss:标准自回归 CE,`ℒ_AR = −∑ log P_θ(τ_out,t | P, τ_out,<t)`,论文明确
  说是纯 next-token CE,无任何辅助 loss。**
* CoT prompt(quality 模式):
  `[c_task, c_lang_tgt, c_speed, S_src_spk, S_src_ling] → T_src → T_tgt → S_tgt_sem`
  即 listen → translate → speak,一条序列一次前向。
* 三个 tokenizer:GLM-4 linguistic **12.5 tok/s**、BiCodec speaker **固定 32 token**、
  BiCodec semantic **50 tok/s**。
* 三阶段:phase1 文本-语音对齐(ASR/TTS/S2TT/MT,77.1k 小时)→ phase2 引入 CoT
  (UniST General,55B token)→ phase3 高质量精炼(10B token)。
* **论文完全没有讨论 streaming / simultaneous 解码。** 本项目的流式扩展在论文
  之外,没有原作者的配方可循。
* 结果:CVSS-T Speech-BLEU 32.20(EN→ZH)/ 24.28(ZH→EN)。

**这条最重要:本项目引以为据的"最好模型"用的是最简单的 loss。复杂化是本项目
自己加的。**

### 2.2 时机是怎么被处理的 —— 六个系统,只有一个用动作 token 的 loss 学

| 系统 | 时机怎么来的 | 训练 loss |
|---|---|---|
| [SimulS2S-LLM](https://arxiv.org/abs/2504.15509) | **离线训练 + 推理侧 wait-k**,训练中不学任何策略 | CE(文本)+ CTC(语音 token),文本 LLM 冻结 |
| [SpeakStream](https://arxiv.org/html/2505.19206) | **外部规则**:"流式文本够长了就追加一个 BOS,模型开始生成语音" | 标准 LM 训练,**loss 只算在语音 token 上** |
| [AlignAtt4LLM](https://arxiv.org/pdf/2606.03967) (IWSLT 2026) | 注意力对齐策略,**训练无关** | — |
| [DOA](https://arxiv.org/html/2605.31432) | decoder-only 注意力策略,**完全 training-free** | — |
| [CSSEL-P2P](https://arxiv.org/abs/2607.13158) | **烘进数据**:teacher 标注的 prefix-to-prefix 目标 + bounded waiting;推理用固定 chunk + rewind 已提交前缀。**明确宣称"无需架构改动"** | 纯 SFT |
| [EASiST](https://arxiv.org/pdf/2504.11809) | **独立的轻量 policy head** | 主 loss + policy head |
| [REINA / REINA-TAN](https://arxiv.org/html/2604.09916v1) | 专门的 policy loss(全/截断音频的 CE 之差) | 见下 |

**六个里五个不用 loss 学时机。** 唯一用专门 loss 学的 REINA,踩了本项目同一个坑。

### 2.3 REINA:本项目失败模式的文献命名与解法

REINA 报告的退化行为:**"policy 反复预测 Read,尽管持续吃进音频,把所有输出
推迟到句尾。"** 论文的诊断是 **temporal drift**:

> "the policy, lacking an internal clock, fails to increase its emission
> probability as audio duration grows."

**这与本项目 §1.5 实测的 −0.012/秒 斜率是同一件事。**

两个解法:

1. **REINA-TAN(时钟)**:把音频时长的正弦位置编码(base-100)注入 policy 输入,
   `e_time^(2i) = sin(t_audio / 100^(2i/d))`。**不增加任何 loss 项。**
   Pareto 前沿最好(NoSE 0.991 de→en,超过 SeamlessM4T 的 0.925),
   read-loop 发生率 0.063% → **0.024%**。
2. **REINA-SAN(单调对齐弱监督)**:用 LLM 生成的单调对齐给每个目标 token 一个
   理想发射时刻 `t_n*`,soft label `σ((t_audio − t_n*)/τ)`,加一项 BCE(λ=1)。
   NoSE 0.987。
3. **两者合用反而更差** —— 策略冲突。

**结论:加时钟是"零新 loss"的解法,而且是他们最好的解法。**

### 2.4 开源参照

* [StreamSpeech](https://github.com/ictnlp/StreamSpeech)(ictnlp)—— "All in One" 流式
  ASR/ST/TTS,是被 SimulS2S-LLM 超过的基线(26.3 vs 22.94 ASR-BLEU)。
* [Simulstream](https://arxiv.org/html/2512.17648) —— IWSLT 2026 同传赛道官方评测工具,
  WebSocket 服务 + 流式实验工具。**若要对外可比,应该按它的接口出评测。**

---

## 3. 三个方案(全部纯 CE,按推荐度排序)

### 方案 A:给模型一个时钟(最小改动,最高性价比)

**motivation**:REINA 把本项目的失败模式归因于"没有内部时钟",而本项目实测
斜率 −0.012/秒 独立证实了这一点;REINA-TAN 用一个零新 loss 的改动把 read-loop
压掉 2.6 倍并拿到最好 Pareto 前沿。而本项目 token 表里**一个时间 token 都没有**。

**做法**:新增一个 `TOKEN_ELAPSED_<bucket>` 家族(0.5 秒一桶,0–20 秒共 40 个
token),在每个 `START_GLM` 块之前插入当前 `source_end_ms` 对应的桶 token。
`source_end_ms` 已经在 trajectory 的 event 里,**不需要新对齐、不需要新数据源**。

**loss**:完全不变,均匀 CE。新 token 落在 `LOSS_BOUNDARY` 桶里。

**成本**:constants 加一个家族(词表扩容 40)+ builder 插一行 + task pool 重打包
(约 2 小时)+ 一个 epoch(约 7 小时)。

**falsification**:重测 §1.5 的斜率。若斜率显著转正(≥ +0.05/秒)则时钟起效;
若仍 ≈ 0,则"没有时钟"不是本项目的主因,方案 A 关闭。

### 方案 B:回到离线 phase3 的均匀 CE(零代码改动)

**motivation**:UniSS 论文确认最好的模型用纯 CE 无加权;本项目把决策 token 的
梯度压到 4.7%(应为 32.8%),而 `boundary_eos=0.10` **从未被测过**;三次 margin
尝试全败,而最朴素的选项一次没试。

**做法**:`boundary_eos` 0.10 → **1.0**;删除 `continue_after_fragment`、
`content_end_margin`、全部 `semantic_end_*` 与 `rollin_*`(归零);保留
`asr/mt/semantic_ce` 1.0、两个蒸馏 KL、`replay_ce`、`commit_consistency`
(蒸馏项实测一直在下降,是有效的防遗忘;离线 phase3 不需要它们只因为它没有
前序能力要保护)。目标数从 14 降到 **7**。

均匀 CE 顺带把 `END_CONTENT` 的权重给到 1.0 —— 是我那个"唯一成功的项"
(`content_end_margin` 0.25)的 **4 倍**,并且免费。

**成本**:改一个环境变量 + 一个 epoch。**无新代码。**

**falsification**:同 §1.1 的探针。gap 从 −2.88 往 0 移动则方向对;仍不动或反向,
则 teacher-forced 无论怎么加权都修不了这个决策,**margin/CE 这条路彻底关闭**。

### 方案 C:把时机从模型里拿走(与文献主流一致,改动最大但证据最强)

**motivation**:六个文献系统里五个不用 loss 学时机(SimulS2S-LLM 离线 CE +
wait-k;SpeakStream 外部规则;AlignAtt4LLM/DOA training-free;CSSEL-P2P 烘进
数据)。而本项目 §1.4 实测:**纯 P2P 通路的 BLEU 是 27.67/35.86,动作 token
通路的会话覆盖只有 0.30。能力在 P2P 里。**

**做法** —— 序列里**彻底不放 WAIT/WRITE**,三个 prompt 三个纯 CE 任务
(三个 task token 在 `constants_uniss.py` 里**已经存在**):

```
① TOKEN_TASK_STREAMING_ASR
   prompt : c_task ⊕ c_lang_src ⊕ [START_GLM 源块前缀 END_GLM]
   target : ASR 文本增量 ⊕ END_CONTENT

② TOKEN_TASK_STREAMING_TEXT_TRANSLATION
   prompt : c_task ⊕ c_lang_tgt ⊕ 已提交 ASR 文本前缀 ⊕ START_CONTENT
   target : **安全译文前缀** ⊕ END_CONTENT

③ TOKEN_TASK_STREAMING_TTS
   prompt : c_task ⊕ c_lang_tgt ⊕ c_speed ⊕ 说话人 global ⊕ 已提交译文增量
            ⊕ START_SEMANTIC
   target : 该增量对应的 semantic token ⊕ END_SEMANTIC
```

**"安全译文前缀"是全部关键**,也是 CSSEL-P2P 的核心:只包含那些**源端支撑
已经到达**的目标词(bounded waiting)。时机不在模型里,在数据里 —— 给定源前缀,
数据说了该吐多少。构造方式:用离线 phase3 teacher 对完整句产生译文,再用单调
对齐(REINA-SAN 的 `t_n*` 思路)确定每个译文词的最早安全发射时刻,截到当前源
前缀为止。

**loss**:三个任务一个 CE,均匀权重。**没有 boundary loss、没有 margin、
没有动作 token。**

**推理**:固定 320ms chunk 调度 → ASR → local agreement 提交 → 在已提交 ASR
前缀上 MT → local agreement 提交 → 在已提交译文增量上 TTS。
**`StablePrefixCommitter` 与配速器本项目已经写好并测过。**

**成本**:task pool 重建(需要单调对齐,约 1 天)+ 一个 epoch。

**falsification**:会话文本覆盖 ≥ 0.50(现 0.30)且可听起始 ≤ 1500ms 且长度比
∈[0.9,1.2]。

---

## 4. 推荐执行顺序

**B → A → C**,理由是成本单调递增而信息量互补:

1. **先跑 B(零代码,7 小时)。** 它是"离线 phase3 配方"的直接检验,也是唯一
   从未被测过的最朴素选项。无论结果如何都关闭一个大方向。
2. **B 若无效则跑 A(2+7 小时)。** 它直接打 REINA 命名的根因,而本项目已实测
   斜率 −0.012/秒 支持这个根因。
3. **A 若无效则转 C(1 天 + 7 小时)。** 此时"用序列内动作 token 学时机"已被
   三种方式(margin、均匀 CE、时钟)否证,与文献主流一致地把时机移出模型是
   有充分依据的重构,而不是又一次试错。

**并且在跑 B 之前先做一件零成本的事**:在 `iter_0002264` 上重扫 δ 偏置确认
兜底仍在 4/6(约 30 分钟)。这样任何时刻都有可交付的 demo。

---

## 5. 明确不做的事

* **不再为 WAIT/WRITE 决策写 margin loss** —— 三次单调反向(−2.88 → −3.75 → −4.97)。
* **不单独训练 `content_end_margin`** —— 它成功了,而单独使用把长度比压到 0.324、
  覆盖压到 0.448。它只在与可用开口机制联合时正确。
* **不同时使用时钟(A)与对齐弱监督** —— REINA 实测两者合用反而更差。
* **不回离线 phase3 重训** —— 离线模型在增量协议下 BLEU 只有 3.92–9.97,
  它不能做增量翻译。
* **不换 Stage-A v9** —— 内容差 1.65 倍(设计 §27.2)。
* **不在 8 条 train-seen 样本上宣布任何成功** —— 全绿也必须先扩到 64 条复核。

---

## 参考文献

* [UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice](https://arxiv.org/abs/2509.21144) — 本项目基座,纯 next-token CE + CoT,未讨论流式
* [SimulS2S-LLM](https://arxiv.org/abs/2504.15509) — 离线 CE 训练 + 推理侧 wait-k,不学策略
* [SpeakStream: Streaming TTS with Interleaved Data](https://arxiv.org/html/2505.19206) — 标准 LM loss,切换由外部规则
* [CSSEL-P2P: Do LLMs Need Architectural Changes for Simultaneous Speech Translation?](https://arxiv.org/abs/2607.13158) — 时机烘进 prefix-to-prefix 数据
* [REINA-TAN / REINA-SAN](https://arxiv.org/html/2604.09916v1) — read-loop 退化的诊断(temporal drift)与两个解法
* [REINA](https://arxiv.org/html/2508.04946) — 用全/截断音频 CE 之差学 READ/WRITE
* [EASiST](https://arxiv.org/pdf/2504.11809) — 交织 read/write token + 独立 policy head
* [AlignAtt4LLM (IWSLT 2026)](https://arxiv.org/pdf/2606.03967) — 训练无关的注意力策略
* [DOA](https://arxiv.org/html/2605.31432) — training-free decoder-only 注意力策略
* [StreamSpeech](https://github.com/ictnlp/StreamSpeech) — 开源流式 ASR/ST/TTS 基线
* [Simulstream](https://arxiv.org/html/2512.17648) — IWSLT 2026 同传赛道官方评测工具
