# SimulS2ST-Omni demo 音频(zh2en/414)上的三方试听

## 输入

从论文的 demo 页下载:
`https://hasaki321.github.io/SimulS2ST-Omni.demo/media/audio/human_eval/zh2en/source/414.wav`

* 原始下载文件:`data/external/simuls2st_omni_demo/zh2en_414.wav`
  (16 kHz,**双声道 float**,7.41 s,sha256 前缀 `46fea516cbb5421e`)
* 两个声道**完全相同**(相关 1.0000,`rms(L−R)/rms(L)` = 0.0000),是双单声道,
  所以均值下混就是原信号,没有内容损失。
* 送进模型的:`data/external/simuls2st_omni_demo/zh2en_414_16k_mono.wav`(16 kHz 单声道)
* `speaker_global` 现算自这条音频的前 32 个 BiCodec global code,写进
  `data/external/simuls2st_omni_demo/SELECTION.json`。

**源音频内容(第三方 Paraformer-zh 转写,作为参照)**:

> 对呃他就这个就就就显现了了是是你才要注意到这个呃自己的这个心心慌啊什么心跳啊之类的

是带大量口头语和重复的自然口语,而且**没有参考译文** —— 论文 demo 页只给源音频,
所以下面只能定性听,不能算 BLEU。

## 三个配置

| | 读步 | 关键设置 |
|---|---|---|
| **C 最佳流式** | 160 ms | `source_holdback=1, target_holdback=0`(s1t0,刚在 RealSI 777 条上验证的新默认) |
| **C 最佳质量** | 4000 ms | k25。它在 RealSI 上 BLEU 最高(17.33/14.41),但块数≈1,**延迟意义上不是流式** |
| **m3 最佳** | 160 ms | `CONTINUE_WRITE_BIAS=5`(δ=5),m3 自己文档称为"全项目最好的流式配置" |

## 结果

| | C 最佳流式 s1t0 | C 最佳质量 k25 | **m3 δ=5** |
|---|---:|---:|---:|
| **首次发音** | 1760 ms | 7410 ms | **320 ms** |
| **真实可听起始** | 2000 ms | 7620 ms | **520 ms** |
| 语音块数 | 5 | 1 | **38** |
| 块间隔 | 1920/2080/960/690 ms | — | **几乎每步 160 ms** |
| 语义 token | 300 | 243 | 424 |
| 译音占用 / 源 | 11.04 / 7.41 s | 12.27 / 7.41 s | 9.60 / 7.41 s |
| 内部静音 | **40.7%** | 12.5% | 10.8% |
| 峰值 | 0.553 | 0.563 | 0.773 |
| 终止率 / 撞上限 | 1.00 / 0 | 1.00 / 0 | 0.95 / 2 |
| RTF | 1.76 | 1.04 | 3.53 |

**译出的英文(把生成音频过 Whisper-large-v3 转写,即"实际听到的")**:

* **C 最佳流式 s1t0** ——
  > But it just shows that the doctor has already noticed his owing anxiety, heart palpitations, and such.
* **C 最佳质量 k25** ——
  > Yes, well, this has manifested, so you need to notice your own heart and heartbeat.
* **m3 δ=5** ——
  > yeah wrong just said it was personal just having danger that's what you look like playing
  > the new play that we all sort of trade out say initial or something like that

## 这条样本正好演示了聚合数字说的事

**m3 δ=5 在 320 ms 就开口,之后几乎每 160 ms 说一句,内部静音只有 10.8% —— 听起来
最"流畅"、最像同传。但内容是词语沙拉。** 它的源 ASR 假设本身就已经崩了:
`对呃呃他就我这个人叫就就有就险下了于是你像住一个到这个个呃自自己的的这个一些的戏或方啊什么新调啊之类的`。
这就是 RealSI 上 ASR-BLEU 1.36/0.50 的样子。

**C 等到 1760 ms 才开口,只发 5 块、内部静音 40.7%(听起来会卡),但输出是通顺可用的
译文。**

**还有一处细节正好对上之前的归因**:C 的 k25 译文 "so you need to notice your own
heart and heartbeat" **比 s1t0 的更忠实** —— s1t0 把源里的"于是你才要注意到"听成了
"医生已经注意到",于是译文里出现了原文没有的 "the doctor"。这正是
"**C 的中文源 ASR 在细读步下退化**"(CER 0.264@k25 → 0.804@k1)在单条上的表现,
也再次说明下一步该修的是 ASR 而不是 TTS 或提交策略。

## 文件路径

每个臂三个音频:

* `<arm>/stereo/omni_demo_zh2en_414.wav` —— **双声道试听:左=源,右=译音**,
  严格按真实发出时间线放置(块在它被发出的时刻才出现)
* `<arm>/translation_placed/omni_demo_zh2en_414.wav` —— 单声道译音,含时间线静音
* `<arm>/translation_concat/omni_demo_zh2en_414.wav` —— 单声道译音,片段背靠背拼接

`arm` ∈ `c_best_streaming_k1_s1t0` / `c_best_quality_k25` / `m3_best_d5_k1`。
每个臂的 `MANIFEST_g0.json` 里有逐块发出时刻、文本假设和全部指标;
ASR 转写在 `asr/{cmn,eng}.jsonl`。
