# 精读 arXiv 2607.19810:我的训练问题在哪,能借鉴什么

来源:arXiv HTML 全文(含全部附录,已存 `data/external/simuls2st_omni_demo/paper.txt`)
与已 clone 的官方仓库。

## 〇、训练开源了吗?没有

README 原文:*"This repository contains the official **inference code** and interactive demos"*。
`src/train/` 里只有模型定义(`modeling_dual_head.py`、`modeling_omni_talker.py`)和
`prompt_formats.py`,**没有训练循环、优化器、数据管线、轨迹构造代码**。
全仓库不含 external 只有 35 个 py 文件。

**但两样东西等价于训练规格泄漏,可以直接用:**
* `prompt_formats.py` —— 完整的 task taxonomy 与 prompt 模板,含
  `With Latency: {m}` 条件后缀;
* 论文 §3.2 + 附录 A/B —— 轨迹构造与三阶段训练的完整描述。

---

## 一、必须先纠正一个我之前的错误结论(好消息)

我之前反复说"我们落后论文 8–11 BLEU"。**那是拿我们和 Thinker–Talker 比,
而论文自己有一个和我们同架构的基线,叫 Dec-only:**

> *"The unified-decoder baseline unifies this process by **appending 16,384 code
> tokens to the Thinker's vocabulary, modeling with a single autoregressive
> head**... within each chunk the model autoregressively generates the target
> text tokens **followed immediately by** the target code tokens."*

**这就是 C 的架构。** 对上他们表 5 的 Dec-only(RealSI 句级 ASR-BLEU,贪心解码):

| RealSI ASR-BLEU | En→Zh | Zh→En |
|---|---:|---:|
| 他们 Dec-only(**3B**)m1(1000 ms) | 12.70 | 10.88 |
| **我们 C(0.5B)k1(160 ms)** | **14.66** | **10.37** |
| 他们 Dec-only(3B)m4(4000 ms) | 17.44 | 15.45 |
| **我们 C(0.5B)k25(4000 ms)** | **17.33** | **14.41** |
| 他们 Thinker–Talker m4 | **25.54** | **22.47** |

**我们 0.5B 的模型,在同架构类下与他们 3B 的 Dec-only 基本持平,
低延迟档 En→Zh 还略好(14.66 对 12.70),而且读步细 6 倍。**

而那个 8–11 BLEU 的差距,论文自己说是**架构性的**:

> *"trajectory finetuning alone **cannot resolve a unified decoder's internal
> text/code conflict**, cementing the **structural necessity of two-stream
> factorization** for continuous streaming."*

**所以我们的天花板不是训练不好,是单解码器的 text/code 冲突。**
(一处口径提醒:表 5 是贪心解码,主表 Table 3 的 25.54 用了 beam search,
表 5 的 Talker m4 只有 24.04。)

---

## 二、论文照出来的、我目前训练的四个问题

### 问题 1(首要):没有单调性过滤 —— 论文消融说这是低延迟鲁棒性的首要驱动

> *"**Without monotonicity filtering, low-latency (m1) performance collapses
> entirely, plummeting to 4.59 and 3.56 BLEU** in the two directions...
> Introducing NIR-based filtering immediately recovers this gap, proving that
> high-quality, difficulty-controlled trajectory supervision is **the primary
> driver of low-latency robustness**. Furthermore, varying the multiplier
> sampling schedules yields **negligible** differences... **curating a stable,
> filtered data pool is far more critical than the specific multiplier sampling
> schedule.**"*

**NIR(normalized inversion rate)**:设目标词对齐到的源位置序列为 Â,
其逆序数为 I,则

```
NIR = 2I / (|Â|(|Â|−1)) × 100%
```

越低说明重排越弱、读写监督越稳。**他们不是"只留单调的",而是按 NIR 和句长分桶
控制难度分布**:难度权重 `{high:0.1, mid_high:0.3, mid_low:0.4, low:0.2}`,
长度权重 `{short:0.1, medium:0.5, long:0.4}`,**刻意保留 10% 的高 NIR 桶**。
(实测他们分层后的 NIR 均值 13.79/15.86,反而**高于**随机选的 9.66/9.92。)

**我们的训练池完全没有这一步。** 这是本篇给出的、证据最强的一条可借鉴项。

### 问题 2:训练读步不是固定 chunk

论文 §3.2 Step 2:*"group adjacent target words and their codes whose boundaries
fall within the same pre-defined source chunk intervals of **1 second**...
**Chunks without newly committed target content act as read/wait steps**"*。

我们的 `task_samples_p2st.py:410` 是 `for event in trajectory.events`,
切在 gold 词/事件边界。**已实测可重新分箱**(IDLE 率:160 ms 84.1% / 640 ms 50.7% /
960 ms 40.4%),构造所需字段数据里全有。

### 问题 3:语义码没有按词边界切分并继承单调化边界

论文 Step 1+2:先做**强制对齐 + SimAlign 跨语言词对齐**,得到每个目标词对应的源端
结束帧,**单调化** `t̃ᵢ = max(t̃ᵢ₋₁, t_{a(i)})`;然后**把目标音频的语义码按目标词
边界切段,每个词和它的码段继承同一个 t̃ᵢ**。

**这一条直接关系到合成割裂:他们的切口落在词边界上,我实测我的 69–79% 落在
满音量的词中间。**

### 问题 4:没有 latency 条件化

他们在 system prompt 里放 `With Latency: {m}`,训练时采样 m∈{1..12},
**一个 checkpoint 服务所有读步**。
**但论文消融说采样 schedule 影响"negligible",uniform 就够** —— 所以这条优先级低。

---

## 三、合成割裂:论文自己也有,而且承认

附录 J.1 标题就叫 **Boundary Concatenation Artifacts**:

> *"The artifact surfaces as a short transient click or spectral discontinuity at
> chunk boundaries... **it is primarily perceptual and only marginally raises ASR
> WER**. It arises because voiced segments are **hard-cut at systematically tight
> forced-alignment word boundaries and spliced against exact numerical silence**,
> so the resulting join is an **abrupt energy step rather than a natural onset**."*

**逐字命中我实测的现象**(切口前后 0.90–1.17 倍 RMS,对着 0.000000 的数字静音)。
而且 Limitations 里承认声码器也不是流式原生的:

> *"The current chunk-wise Flow Matching decoder is **not yet fully
> streaming-native**, which may limit **inter-chunk coherence**."*

**所以这不是我们独有的缺陷,论文也没解决。** 但有一处真实差距:
**他们切在词边界上,我们切在词中间。** 这是可改的(见问题 3)。

---

## 四、可改进点,按论文的证据强度排序

| # | 改进 | 论文证据 | 成本 | 预期 |
|---|---|---|---|---|
| **1** | **NIR 单调性过滤 + 难度/长度分层采样** | 消融:不做则 m1 塌到 4.59/3.56;做了恢复。明说是"primary driver" | 数据侧,需要跨语言词对齐 | **低延迟档最大收益** |
| **2** | **固定 chunk 网格 + 显式 wait 步** | §3.2 Step 2 | 数据重分箱(已验证可行)+ 一次训练 | 修卡顿 + 细读步鲁棒 |
| **3** | **语义码按词边界切分、继承单调化源边界** | §3.2 Step 1+2;J.1 归因 | 数据侧 | **直接减少词中间切断** |
| 4 | latency 条件化 | §B "negligible differences" | 低 | 便利性为主 |
| 5 | 双流分解(Thinker–Talker) | §4.3 "structural necessity" | **重写架构** | 是天花板,不是移植 |

**一个必须记下的张力:** 论文说 Dec-only *"cannot effectively leverage auxiliary
multitask supervision"*(去掉辅助数据对 Dec-only 影响可忽略)。但我们的 C 恰恰是
多 family + replay 的统一解码器,而且它**确实**赢了 B′ 和 m3。
**所以这条结论不能照单全收 —— 我们的实测与之相左,值得单独验证而不是直接采信。**

## 五、一句话总结

**训练没开源,但方法写得足够清楚可以复现。我们真正缺的是数据侧的两件事:
按 NIR 做单调性过滤与难度分层(论文标为低延迟鲁棒性的首要驱动),
以及按固定 chunk 分组 + 词边界切码 + 显式 wait 步。
合成的割裂论文自己也有并承认;而我们与论文头条数字的差距,
论文自己归因为单解码器的架构上限 —— 在同架构的 Dec-only 基线上,
我们 0.5B 的模型已经和他们 3B 的持平甚至略好。**
