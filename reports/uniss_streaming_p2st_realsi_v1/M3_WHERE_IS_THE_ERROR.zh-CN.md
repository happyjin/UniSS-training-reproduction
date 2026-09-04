# m3 δ=5+rp1.1 声音不卡但翻译错:错在哪一段

**答案:错在 ASR 和 TTS 两段,incremental MT 是清白的。而交织架构本身是第四处
独立损失 —— 在 ASR 不坏的时候,它才是杀死翻译的那一处。**

## ① ASR —— 主要入口,而且不是音频难

这条 demo 样本上:

| | 源 ASR 输出 | 对第三方 Paraformer 的 CER |
|---|---|---:|
| **m3 δ=5+rp1.1** | 对对呃呃他就我这个人叫就就有就险下了于是你像住一个到这个个的这几的这个些的新货荒啊什么新调啊就是这个 | **0.707** |
| C s1t0(同一条音频) | 呃他就这个就就就显现了医生已经注意到这个呃自己的这个心心慌啊什么心跳啊之类的 | **0.171** |
| Paraformer(当作正确源) | 对呃他就这个就就就显现了了是是你才要注意到这个呃自己的这个心心慌啊什么心跳啊之类的 | — |

**C 在同一条音频上只有 0.171,所以这段音频是可识别的,是 m3 没识别对。**
RealSI 全量上同样:m3 WER 1.442 / CER 0.842(WER > 1 说明以插入为主,它在往外吐词),
C 是 0.421 / 0.804。

## ② incremental MT —— 清白,两条独立证据

**(a) 它的英文是它自己那段乱码中文的忠实直译。** 可以逐段对上:

| m3 的 ASR(错) | m3 的 MT 输出 |
|---|---|
| 他就我这个人叫 | he just said I was a person called |
| 就就有就险下了 | just just having a danger down |
| 于是你像住一个到 | so you looked like a body into this place |
| 新货荒 | new stock wasteland |
| 什么新调 | or something new adjustment |

**MT 在正确地翻译错误的输入。** 把它单独拎出来指责是错的。

**(b) 喂它正确的源文本,它的 MT 是好的。** m3 自己门禁里有这个对照
(`e_mt_gold` = 读 gold 源文本翻译):

| m3 门禁 16 条 | cmn→eng | eng→cmn |
|---|---:|---:|
| ① 它自己的 ASR 错误率 | **0.037**(CER) | 0.227(WER) |
| ② 离线 phase3 教师 · gold 源 | 9.97 | 3.92 |
| ③ **m3 · gold 源 MT** | **27.67**(chrF 43.66) | **35.86**(chrF 33.68) |
| ④ m3 · 自己的 ASR MT | 14.93 | 17.51 |
| ⑤ **交织 S2S 会话里的 MT** | **7.45** | **0.10** |

**③ 的 27.67 / 35.86 说明 MT 能力在**(甚至高于离线教师在同口径下的 9.97 / 3.92)。

## ③ TTS —— 第二处独立失效

把 m3 的**输出音频**过 Whisper,再和它**自己写的那段文本**比(词级),
接近 1 说明"听到的"就是"它写的":

| | 音频过 Whisper | TTS 保真度 |
|---|---|---:|
| **m3 δ=5+rp1.1** | Yeah, Ralf just said with first quote, you just have to think that so you look like body into this place that it wasn't full of things any star was like something you could just live with is this | **0.304** |
| C s1t0 | where they just shows that the doctor has already noticed own anxiety other patients and such | **0.706** |
| m3 δ=5+rp1.3 | (空 —— 被判 `implausible_after_single_item_retry`) | **0.000** |

**连它自己产出的那段(已经错了的)文本,音频都没忠实渲染出来:**
`a person called` → 听成 `Ralf just said with first quote`;
`new stock wasteland` → `any star was like`。C 的保真度是它的 2.3 倍。

顺带一个结论:**rp=1.3 对 m3 更糟** —— 虽然它同样把谱冻结压到 0.000,但音频被
Whisper 判为完全读不出。**所以 m3 的重复惩罚最优值是它自己选的 1.1,不是 C 的 1.3。**
(注:0.706 不是上限,Whisper 自身也有误差;有意义的是 0.304 与 0.706 的相对差。)

## ④ 交织架构 —— 在 ASR 不坏的时候,这才是杀手

看上表的 ④ → ⑤ 这一跳:**同样的权重、同样的 ASR 输入,`e_mt_free` 是 17.51,
放进交织会话后 `e_s2s_free` 只有 0.10。** cmn→eng 也从 14.93 掉到 7.45。

差别只有两点:
1. 交织会话里 MT 被**决策 token 门控** —— δ=0 时 `WRITE_MT/事件` 只有 0.147,
   覆盖 0.295;
2. 语义(TTS)token 被**塞进同一个 KV cache**,污染了后续 MT 续写读到的上下文。

**而这条 16 条门禁上 ASR 只有 0.037 —— 也就是说在识别几乎没错的情况下,
翻译仍然从 17.51 塌到 0.10。那一定不是 ASR 的锅,是交织架构的锅。**

我们这次给 rp 的连带观察也印证了这条通路:rp 只改语义解码,但因为语义码和 MT
共用 KV cache,**加了 rp 之后 MT 文本也变了**。这正是 ④ 的机制。

## 归因总结

| 数据 | 主因 | 次因 | 无辜 |
|---|---|---|---|
| **本条 RealSI demo** | **ASR(CER 0.707)** | **TTS(保真度 0.304)** | MT |
| **域内 16 条(ASR 好到 0.037)** | **交织架构(17.51 → 0.10)** | ASR 前缀误差传播(③→④ 掉 12.7 / 18.4) | MT 能力(gold 源 27.67 / 35.86) |

**所以"m3 声音不卡但翻译错"是三件事叠出来的:识别错了(RealSI 上)、
交织会话把 MT 的输出打碎(域内也如此)、TTS 又没把已有文本忠实渲染出来。
唯一没问题的是 MT 本身。**

## 对 C 的两点启示

1. **C 结构上避开了 ④。** 它是三段隔离级联,MT 有自己的任务前缀和干净上下文,
   不和 TTS 共用一次续写。这解释了为什么 C 的 RealSI ASR-BLEU 是 m3 的 10–20 倍,
   而两者 MT 能力(gold 源)本来同源。
2. **C 的 TTS 保真度 0.706 是 m3 的 2.3 倍**,而且它的谱冻结是 0.000、
   >4kHz 只有 0.004。**C 剩下的唯一大问题仍然是 ①(源 ASR)** ——
   这是第四条独立指向同一处的证据。
