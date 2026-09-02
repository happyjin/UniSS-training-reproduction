# 修完之后 —— 四处格式错位 + 一处我自己的 bug

**三条里两条修好了,第三条露出了真正的训练目标。**

方法一直是同一条:**每个任务的 prompt 必须逐 token 复现"训练过它的那个格式"**。
偏离一处就有一个可观测的后果,而且每次都是可定位的。

## 修了什么,后果是什么

| # | 错位 | 观测到的后果 | 修后 |
|---|---|---|---|
| ① | ASR prompt 少了 `*wrap_global_tokens(speaker)`,`lang` 也只在头部、隔着约 90 个 GLM token | **英文音频被转写成中文**(`石雕了现在`) | 四条全部正确英文 |
| ② | MT 用 `TASK_S2T_TRANSLATION`,但 C 的 MT 读的是**已提交的源文本**,不是音频 | — | 改成 `TASK_T2T_TRANSLATION` |
| ③ | MT prompt 少了 `WRITE_GENERATE, lang` 分隔,第二个 `START_CONTENT` 读起来像同一个 content 块的延续 | **把源文抄下来而不翻译**(`I think a → I think a`) | 四条全部正确中文 |
| ④ | TTS prompt 少了 `WRITE_GENERATE, lang, speed`,且 `speed` 位置不对 | — | 对齐 phase3 `build_tts_sample` |
| ⑤ | **我的 runtime 把 `END_SEMANTIC` mask 掉了** —— allowed 集只有 8192 个 BiCodec 码,而终止符不在其中 | **终止在结构上不可能**:每个 TTS 阶段都跑满 384 | 终止率 0.93 → **1.00**(3/4),wall 8–9 s → **2.5–3.0 s** |

参照格式:ASR 对 Stage-A 的 `build_streaming_asr_sample`(它是唯一训练过
`TASK_STREAMING_ASR` 的地方);MT 对离线 phase3 的 `build_mt_sample`(整句 BLEU 33–52);
TTS 对离线 phase3 的 `build_tts_sample`(整句时长比 1.039)。

## 修完的实际输出(20 块,零训练的 checkpoint)

```
NCSSD_R_EN_0000000083  src: I can't think what takes          tgt: 我
NCSSD_R_EN_0000000261  src: Still late now Let's gather       tgt: 现在还晚
NCSSD_R_EN_0000000402  src: What's the buzzing that is in fact tgt: 实际上
NCSSD_R_EN_0000000463  src: Well I'm glad I can inspire you   tgt: 好吧我很高兴能激励你
```

**一个从未在这个格式上训练过的 checkpoint,只靠对齐 prompt 布局就产出了正确的流式中英转写与中文翻译。**
三项机械判据仍然 4/4(规则复现、无决策 token、不早于源音频)。

## 还剩什么问题

**① 长样本的 TTS 仍然不终止(真正的训练目标)。**
`NCSSD_R_EN_0000000463` 有 2 个片段,两个都撞 384 上限,`term = 0.93`。
短片段能终止、多片段/长文本不能 —— `END_SEMANTIC` 在隔离形式下确实需要训练。
**这现在是一个干净的单任务问题,而不是和 WAIT/WRITE 挤在一个桶里的问题。**

**② 译文相对源文太短。** `"I can't think what takes"` 只提交出 `"我"`。
提交器是 local agreement + holdback 2,而 MT 每次输出很短,所以放出的很少。
这是 holdback 的已知取舍,但**它直接决定首次发音延迟**。

**③ 首次发音基本等于源结束。** 4 条里 3 条 `first_audible = 3200 ms`,
而 20 块正好是 3200 ms —— **也就是说现在还不是同传,是"说完才开口"**。
根因是 ②:MT 迟迟提交不出内容。**这是 ⑥ 要改善的核心指标。**

**④ `text_scope` 对比无结论。** delta 与 prefix 两次运行**逐位相同**,
因为 4 条里 3 条只有一个片段(前缀==增量),第 4 条两个片段都撞了上限。
参数保留,默认 `delta` 是按 SpeakStream Scheme 1 的设计判断,**不是测出来的**。

**⑤ 还不是实时。** 约 3 秒音频用了 2.5–17 秒墙钟(RTF ≈ 1–5),
其中 20 次 ASR 调用是主要开销 —— 每块都重跑完整 prompt,没有 KV cache 复用。
这是已知的 S4 工程项,不在 C 的范围内。

## 一个副作用值得记录

ASR 每个样本要付 32 个 speaker token,而 Stage-A 每**整句**付一次、C 每**事件**付一次,
所以 ASR 的 packed 行从 925 涨到 1378(+49%),`used_tokens` 16.1M → 24.0M。
这是隔离序列设计的真实开销,不阻塞,但全量训练时要算进去。
