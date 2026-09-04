# 为什么 SimulS2ST-Omni 的 demo 不割裂而我的割裂

分析对象:demo 站 `hasaki321.github.io/SimulS2ST-Omni.demo`(实际下载了 10 个音频)、
开源仓库 `github.com/hasaki321/SimulS2ST-Omni`(已 clone,读了 s2st agent 全文)。

**结论有四条,其中第三条是我移植他们的机制**失败**才发现的。**

## ① 他们的输出是等时的:一个 chunk 进,必定一个 chunk 出

`src/agents/simuleval_omni_talker_s2st_agent.py` 的 `policy()`:

```python
chunk_samples = max(1, int(self.chunk_duration * sample_rate))   # chunk_duration = m 秒
...
if not has_full_chunk and not has_final_chunk:
    return ReadAction()
...
output_audio = self._synthesize_codes(voicebox_codes, states, current_audio_16k)
```

**每消费 `m` 秒源音频,必定产生一次 `WriteAction`。** 而且不够长时用同分布静音码补齐:

```python
# Pad `generated_codes` with silence codes (in-distribution, audio_tokenizer of
# zero-padded waveform) so we keep a stable margin.
silence_pool = self._get_silence_codes(1.0)...
generated_codes = torch.cat([generated_codes, pad_codes], dim=1)
```

**我的级联是反过来的:提交器放出文本才发一个变长片段,没文本就什么都不发,
`place_on_timeline` 用数字零填空。** 结构上他们不可能有洞,我必然有。

## ② 他们有一个专门为此加的开关,默认关闭

```python
parser.add_argument(
    "--enable-wait-silence-decode", action="store_true", default=False,
    help="On wait/idle chunks, synthesize cached silence codes through VoiceBox "
         "instead of emitting no audio.")
...
# otherwise wait chunks emit no audio unless the talker produced real codes
if is_wait and self.enable_wait_silence_decode:
    voicebox_codes = self._get_silence_codes(self.chunk_duration)
```

注释里 "otherwise wait chunks emit no audio" 说的就是我现在的行为。
**他们自己意识到了这个问题并加了开关。**

## ③ 我移植了这个机制,它没解决问题 —— 因为声码器不同

给 `realsi_rollout.py` 加了 `--silence-fill`:用同一个 BiCodec 对 1 秒数字静音取码
(得到 49 个码),按时间线插进码序列的空隙里,**整条一次性解码**(这样解码器的
80 ms 交叉淡化也跨越静音↔语音边界)。结果:

| | 精确为 0 占比 | **静音帧 RMS 中位** | 内部静音 | 最长空档 |
|---|---:|---:|---:|---:|
| 我的 placed(数字零) | 28.2% | **0.000000** | 0.360 | 3140 ms |
| **我的 isochronous(合成静音)** | 27.5% | **0.000009** | 0.368 | 3580 ms |
| Omni 414 `ours_L5` | 1.0% | **0.000698** | 0.110 | 340 ms |
| Omni 长篇 `m3` | 11.4% | **0.000518** | 0.308 | 5280 ms |

**BiCodec 把静音码解码成近乎真静音(0.000009),他们的 VoiceBox 解出 0.0005–0.0007
的本底噪声 —— 差 60 倍。** 所以同一个机制在我们的声码器上填出来的仍然是"死寂"。

**这一条是我原本会猜错的地方:我以为是策略差异,实测证明还有声码器差异。**
(顺带:他们的 `zh2en_stereo` 右声道本底是 0.000000,长篇和 `ours_L5` 才有噪声底 ——
说明这个开关默认关闭、只在部分渲染里开了。)

## ④ 最重要的一条:粒度和等待

| | SimulS2ST-Omni | 我的 C |
|---|---|---|
| 读一个 chunk | **m 秒**(1000–5000 ms) | **160 ms** |
| 最小起说时间 | `--min-start-sec` 默认 **1.0 s**,且首个 chunk 必须完整 | 提交器一放文本就说 |
| 每步输出 | **恒定一个 chunk 的音频** | 变长,可能为 0 |

实测他们的立体声 demo 前导静音是 **1000 / 2000 / 3000 / 4000 ms**,
精确等于 L1/L2/L3/L4 —— 印证了"首个 chunk 完整才写"。

**用 1 秒的 chunk,模型每次有足够内容可说,写出来的每段都实、且节奏规整;
用 160 ms 的 chunk,大部分步没有新文本,于是变成"说一小口、停很久"。**
我自己的读步扫描也印证了这个方向:C 从 160 ms 到 1920 ms,空档数从 5.2 降到 2.4。

## 一处必须澄清的比较偏差

**他们长篇 demo 的内部静音是 0.308,我的 C 是 0.360 —— 只差 0.05。**
而他们短片段(`ours_L5`)是 0.110、最长空档 340 ms,我的是 0.360、3140 ms。

**所以差距主要在短片段上,长篇上两边其实接近。** 你听到的强烈反差,
很可能来自拿他们的**短 human-eval 片段**对我的**长音频时间线版**。
建议用同长度对比:他们的 `lf_zh2en_m3_Technology.wav`(325 s)对我的长音频臂。

## 能做和不能做的

**能做(已验证有效方向):**
1. **把读步放大到 ~1 秒**。我的扫描显示 C 在 960 ms 上 MT BLEU 最好(37.9/33.2)、
   空档数降到 5.0、内部静音 0.37;代价是首次发音 1120 → 2880 ms。
   **这正是他们的 m=1~2 工作点。**
2. **等时输出**:每个读步都发满一个 chunk(不够用静音码补)。机制已实现
   (`--silence-fill`),但在 BiCodec 上效果有限,见 ③。

**不能做:**
* 指望换个填充就解决 —— 已实测,BiCodec 的静音太"干净";
* 指望淡入淡出 —— 之前测过,切口阶跃只有 p99 的 0.23–0.44,没有咔哒可消;
* 指望语速 —— 之前测过,0.7/0.8/0.9 上最长空档纹丝不动。

**根本差别仍然是:他们每 1 秒必产出一个 chunk 的内容,我每 160 ms 大多数时候
没有新文本可说。** 这回到同一个结论 —— 要么放大读步(立刻可做,牺牲开口延迟),
要么让 ASR→MT 更早更密地放出文本(需要训练)。

## 文件

* 下载的官方 demo 音频:`data/external/simuls2st_omni_demo/site/a/`(10 个)
* clone 的仓库:`data/external/simuls2st_omni_demo/repo/SimulS2ST-Omni/`
* 我的等时渲染 A/B:`reports/uniss_streaming_p2st_realsi_v1/silence_fill/c_k1_silencefill/`
  下的 `translation_placed/`(数字零)、`translation_isochronous/`(合成静音)、
  `translation_concat/`(纯连续)三个目录,同名同内容,可直接对比。
