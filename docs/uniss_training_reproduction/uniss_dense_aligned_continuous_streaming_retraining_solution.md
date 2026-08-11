# UniSS Phase3 Dense-Aligned Continuous Streaming 重构与重新训练完整方案

> 文档日期：2026-08-11  
> 文档状态：实施前完整设计  
> 基础模型：UniSS Phase3 v4 最佳 validation checkpoint  
> 最终目标：在不更换 Phase3 主架构、不额外引入独立 talker 的前提下，训练得到可以真实增量推理、连续生成目标语音、音色稳定，并具备低于 1 秒首次有效发音能力的 simultaneous speech-to-speech 模型。

---

## 1. 最终结论

当前问题不能通过以下方法解决：

- 继续训练当前 pilot15 checkpoint；
- 把当前有缺陷的 trajectory 从 15 shard 直接扩大到 full198；
- 单纯增加 iteration；
- 仅修改 shuffle；
- 只在推理时降低阈值或强制 WRITE；
- 只在推理时固定 speaker token；
- 重新开放没有语音监督的 forced audio。

必须执行的核心修复是：

1. 保留现有 UniST 原始 parquet、source/target 音频、文本和 BiCodec token，不重新下载原始数据。
2. 废弃当前稀疏、短片段、未真实对齐的 streaming trajectory 派生缓存。
3. 从原始 UniST 重新生成一套 dense-aligned continuous streaming trajectory。
4. 每 160ms 建立一个 observation，一直持续到句尾。
5. 每次 WRITE 的新增目标文字必须和目标音频中的真实时间区间、连续 BiCodec semantic span 对齐。
6. 一条样本所有 WRITE 的 semantic span 合并后必须覆盖完整目标语音。
7. 从 Phase3 v4 最佳 checkpoint 重新训练 streaming adapter、policy heads 和 streaming LoRA，不能从当前 pilot15 iter350 续训。
8. 训练与推理使用完全一致的增量前端、历史格式、speaker 条件和 codec state。
9. 固定系统音色或预注册音色，取消 3.2 秒在线 speaker warm-up。
10. 通过真实 free-running PCM rollout，而不是仅依赖 teacher-forced loss 选择 checkpoint。

推荐的总体路线是：

```text
Phase3 v4 最佳 checkpoint
  + Dense bilingual aligned trajectory
  + 完整连续 semantic/audio 覆盖
  + Phase3 replay
  + Balanced READ/WRITE + safe commit
  + Stateful BiCodec streaming runtime
  + 固定或预注册 speaker
```

---

## 2. 当前失败的根本原因

### 2.1 当前 streaming 监督极度稀疏

对当前 15 shard、约 150 万条 session 的数据审计结果：

| 每句自然 WRITE 数 | session 比例 |
|---:|---:|
| 0 | 60.842% |
| 1 | 28.175% |
| 大于等于 2 | 10.983% |

总 natural WRITE 数：

```text
786,294
```

总 streaming semantic 监督：

```text
9,432,700 tokens
```

BiCodec semantic 约为 20ms/token，因此当前每句平均仅有：

```text
约 125.77ms 目标语音被 streaming semantic loss 监督
```

这远不足以让模型学习一整句连续同传音频。

### 2.2 大量样本没有任何 streaming 语音目标

抽查样本：

| 样本 | 完整目标 semantic | 被 streaming 监督的 semantic | 覆盖率 |
|---|---:|---:|---:|
| train_en_zh_01 | 178 | 28 | 15.7% |
| train_en_zh_02 | 243 | 0 | 0% |
| train_zh_en_01 | 206 | 0 | 0% |
| train_zh_en_02 | 314 | 16 | 5.1% |

这意味着模型即使把当前 loss 优化得很好，也只学习了极少量目标语音片段，不能学会：

- 第一次应该说什么；
- 本次新增文字对应哪段声音；
- 下一次如何延续上一段声音；
- 如何避免重复已经说过的内容；
- 如何覆盖整句目标语音；
- 如何在多个 WRITE 之间保持音色和韵律连续。

### 2.3 当前 semantic span 不是真实文字—音频对齐

当前数据中的部分 WRITE 只选择固定的 8/12/16 个 semantic token。这些 token 并不是通过目标文本和目标语音 forced alignment 得到的真实区间。

因此存在如下错误监督：

```text
本次 text_delta = “你好”
本次 semantic_target = 目标音频中任意选择的一小段 token
```

即使文本预测正确，语音 target 也可能对应其他词、停顿、半个音素或不同短语。

### 2.4 forced WRITE 没有对应语音监督

在当前数据中，达到 deadline 后的 forced WRITE 通常只有软文本 teacher target，没有对应 semantic/audio target。

因此模型没有学过：

```text
forced anticipation text
    -> 对应的目标 semantic
    -> 连续可播放 PCM
```

推理时强行让这种事件生成语音，会产生乱码、重复、极短音频或不可懂声音。安全 runtime 将这类输出静音是正确行为，但这也说明训练数据必须补充真实监督，不能只修改推理阈值。

### 2.5 当前 schedule 不连续

当前每句通常只有：

```text
320 / 480 / 640 / 800ms
+ 一个 middle 或 late tick
```

没有从开头每 160ms 持续观察到句尾，也通常没有 final WRITE 提交剩余完整目标内容。

### 2.6 runtime 与训练条件不一致

当前 runtime 还存在会阻止低延迟输出的结构问题：

- speaker warm-up 为 3200ms，天然不可能做到首次发音低于 1 秒；
- UI 使用 640ms chunk 时，800ms deadline 只能在 1280ms 被检查；
- 训练使用完整 `bicodec_global`，runtime 可能从极短前缀重新提取 speaker；
- 当前增量 WhisperVQ token 存在 committed token revision；
- 决策、文字和 semantic 分支必须看到与训练完全相同的 committed history；
- 每个 WRITE 如果重新初始化 codec，会造成音色变化、爆音和不连续；
- scheduler 产生 WRITE 但安全门拒绝音频时，不能把该时刻算作 first useful audio。

当前回归结果可参考：

[true_subsecond_pilot15 regression report](/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/true_subsecond_pilot15_dataset_regression_v1/run_20260811_fixed8_chunk640/REPORT.md)

---

## 3. 哪些数据需要重做，哪些不需要

| 数据/产物 | 是否重做 | 说明 |
|---|---|---|
| UniST 原始 parquet | 否 | 保持原样，不重新下载 |
| source transcription | 否 | 作为 source forced alignment 文本 |
| target translation | 否 | 作为 target forced alignment 文本 |
| 完整 source/target BiCodec token | 否 | 继续作为原始声学数据 |
| source/target 重建 WAV | 按需生成 | forced alignment 和试听审计需要 |
| 当前 sparse trajectory cache | 是 | 不能继续用于正式训练 |
| source/target 单语时间戳 | 是 | 需要重新正式生成或复用已通过质量门的产物 |
| source-target 双语词对齐 | 是 | 用于计算目标词最早安全提交时间 |
| dense READ/WRITE events | 是 | 每 160ms 生成到句尾 |
| text delta—semantic span 映射 | 是 | 必须真实时间对齐 |
| fixed-speaker streaming branch | 建议生成 | 保证低延迟和跨 chunk 音色稳定 |
| Phase3 replay 数据 | 否 | 原样保留，防止基础质量退化 |

重新制作的是“派生 streaming 监督数据”，不是重新收集或下载语料。

---

## 4. 新数据的总体设计

### 4.1 一条样本保存一次完整序列

full198 如果每 160ms 复制一次完整 source prefix、target prefix 和 semantic history，会造成数据量爆炸。

推荐紧凑结构：

```json
{
  "sample_id": "zh_en_xxx",
  "direction": "zh-en",
  "source_text": "欢迎大家参加今天的会议",
  "target_text": "Welcome everyone to today's meeting.",
  "source_codes": ["完整source token，只保存一次"],
  "target_semantic": ["完整target semantic，只保存一次"],
  "speaker": {
    "mode": "fixed_system",
    "global_tokens": ["固定32个speaker token"]
  },
  "source_words": [
    {"word": "欢迎", "start_ms": 0, "end_ms": 200},
    {"word": "大家", "start_ms": 210, "end_ms": 390}
  ],
  "target_words": [
    {
      "word": "Welcome",
      "start_ms": 0,
      "end_ms": 260,
      "semantic_span": [0, 13]
    }
  ],
  "alignment_links": [
    {"source_word_ids": [0], "target_word_ids": [0]}
  ],
  "events": [
    {
      "source_end_ms": 160,
      "action": "READ",
      "committed_text_end": 0,
      "semantic_end": 0
    },
    {
      "source_end_ms": 320,
      "action": "WRITE",
      "text_delta": "Welcome",
      "text_span": [0, 1],
      "target_audio_span_ms": [0, 260],
      "semantic_span": [0, 13],
      "support_bucket": 1,
      "safe": true
    }
  ]
}
```

训练 dataset 在读取时根据 event offset 动态构造 prefix 和 history，避免在磁盘上重复保存。

### 4.2 必须满足的连续覆盖约束

设完整目标 semantic 为：

```text
z_target = [z0, z1, ..., zN-1]
```

所有 WRITE 的 semantic span 必须满足：

```text
WRITE 1: z[0:q1]
WRITE 2: z[q1:q2]
WRITE 3: z[q2:q3]
...
FINAL:   z[q_last:N]
```

严格约束：

1. 第一个 `semantic_start = 0`。
2. 当前 `semantic_end = 下一次 semantic_start`。
3. gap 数为 0。
4. overlap 数为 0。
5. final `semantic_end = len(target_semantic)`。
6. 所有 `text_delta` 拼接后等于规范化后的完整 target text。
7. 所有 semantic span 拼接后等于完整 target semantic。

即：

```text
[0:q1] ∪ [q1:q2] ∪ ... ∪ [q_last:N] = [0:N]
```

正式数据目标：

- semantic coverage 至少 95%；
- 最理想为接近 100%；
- 不允许继续出现平均每句仅约 125ms 的监督。

---

## 5. 文字与目标语音的严格对齐算法

真正的 streaming trajectory 需要同时解决两个问题：

1. 目标文字什么时候已经可以根据 source prefix 安全提交；
2. 这段目标文字对应目标音频中的哪一段声音。

这是两个不同的对齐步骤。

### 5.1 Source transcription 与 source audio 对齐

对 source transcription 做词级或字级 forced alignment：

```text
欢迎      0–200ms
大家      210–390ms
参加      400–560ms
今天      570–750ms
会议      760–1000ms
```

输出：

```text
source_word_i = {
    text,
    start_ms,
    end_ms,
    confidence
}
```

### 5.2 Target translation 与 target audio 对齐

对 target translation 和完整 target WAV 做 forced alignment：

```text
Welcome       0–260ms
everyone      260–540ms
to            540–620ms
today's       620–900ms
meeting       900–1300ms
```

输出：

```text
target_word_j = {
    text,
    start_ms,
    end_ms,
    confidence
}
```

### 5.3 Source word 与 target word 双语对齐

建立 source-target multilingual alignment：

```text
欢迎   <-> Welcome
大家   <-> everyone
参加   <-> to / attend
今天   <-> today's
会议   <-> meeting
```

允许：

- 一对一；
- 一对多；
- 多对一；
- 非单调语言对齐。

但最终 target 输出顺序必须保持单调，不能跳过尚不安全的目标前缀。

### 5.4 计算目标词最早安全提交时间

设目标词 `y_j` 对齐到若干 source 词，目标词最早安全提交时间：

\[
\tau(y_j)=\max_{x_i\in Align(y_j)}EndTime(x_i)+SafetyMargin
\]

建议初始 safety margin：

```text
80–160ms
```

对于目标前缀 `y_1,...,y_k`：

\[
\tau(1:k)=\max_{1\leq j\leq k}\tau(y_j)
\]

在 observation 时间 `t`，安全目标前缀长度：

\[
K(t)=\max\{k:\tau(1:k)\leq t\}
\]

如果：

```text
K(t) > 已提交target长度
```

则存在 natural WRITE；否则为 READ。

### 5.5 语言重排序示例

Source：

```text
I will call you tomorrow.
```

Target：

```text
我明天给你打电话。
```

虽然 source 在听到 `call you` 后已经支持“给你打电话”，但目标第二个词“明天”需要等到 source 的 `tomorrow`。

目标输出必须保持顺序，因此在听到 `tomorrow` 前不能跳过“明天”直接说“给你打电话”。

这类等待是语言重排序的真实下界。若强制低于 1 秒输出，只能使用经过验证的预测式翻译监督，不能简单降低 WRITE 阈值。

### 5.6 Target audio 时间转成 BiCodec semantic span

BiCodec semantic 约 20ms/token，因此：

\[
semantic\_start=\lfloor target\_start\_ms/20 \rfloor
\]

\[
semantic\_end=\lceil target\_end\_ms/20 \rceil
\]

例如：

```text
Welcome: 0–260ms
semantic span = [0:13]
```

### 5.7 不应机械逐词切音频

音频存在共发音、连读和边界模糊。建议先得到词级时间戳，再将相邻词合并为 200–400ms 短语，并把切分点调整到：

- 静音或低能量区；
- 音素边界；
- 标点；
- 自然停顿；
- 不破坏辅音或元音的位置。

标点没有独立音频，应附着到相邻 phrase。词间静音也必须分配给前后 phrase 之一，保证最终 semantic coverage 连续。

### 5.8 低质量样本处理

建议：

- source forced alignment coverage 小于 0.85：过滤或降权；
- target forced alignment coverage 小于 0.85：过滤或降权；
- 双语 alignment coverage 太低：过滤或只用于 Phase3 replay；
- target ASR 与 target translation 严重不一致：过滤；
- 时间戳不单调：过滤；
- semantic gap/overlap 无法修复：过滤；
- 极长静音、损坏 WAV、clipping：过滤。

项目中已有可复用基础组件：

- [prepare_a45.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/simul_uniss/subsecond_v2/prepare_a45.py)
- [prepare_a68.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/simul_uniss/subsecond_v2/prepare_a68.py)
- [formal_supervision.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/simul_uniss/subsecond_v2/formal_supervision.py)
- [neural_word_aligner.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/simul_uniss/subsecond_v2/neural_word_aligner.py)
- [reconstruct_unist_audio.py](/opt/dlami/nvme/jasonleeeli/projects/UniSS/training/simul_uniss/reconstruct_unist_audio.py)

forced aligner 内部即使使用 CTC/Viterbi，也只是离线数据制作工具，不代表重新把 CTC 作为主模型训练 objective。

---

## 6. Dense READ/WRITE trajectory 生成

### 6.1 Observation 粒度

推荐内部固定：

```text
160ms/tick
```

从 160ms 开始，一直生成到：

- source 音频结束；
- target text 全部提交；
- target semantic 全部提交；
- final flush 完成。

Gradio 可以使用 320/480/640ms 作为网络或 UI chunk，但 policy 决策必须仍然按内部 160ms tick 执行。

### 6.2 Support bucket

在 observation `t` 时：

```text
support_bucket =
    当前已安全、但尚未提交的target phrase数量
```

例如：

| support_bucket | 含义 |
|---:|---|
| 0 | 没有安全内容，只能 READ |
| 1 | 有一个 phrase 可提交 |
| 2 | 有两个 phrase 可提交 |
| 3+ | 输出明显落后，应优先 WRITE |

support loss 必须采用类别平衡，不能继续让 0 类占绝对多数后使用普通 CE。

### 6.3 Playback buffer 模拟

仅知道“内容可以翻译”仍不足以学会连续说话。训练数据还应模拟目标音频播放队列：

```text
每过160ms：
    playback_buffer_ms -= 160

若：
    support_bucket > 0
    且 playback_buffer_ms < low_watermark
则：
    WRITE一个200–400ms目标semantic span
否则：
    READ
```

初始建议：

```text
low_watermark  = 240ms
target_buffer  = 320–480ms
max_buffer     = 800ms
```

这样模型学习的是：

- 不要在有安全内容时一直等待；
- 不要一次生成过多内容造成延迟堆积；
- 在缓冲即将耗尽时及时补充语音；
- 以短而连续的音频块维持播放。

### 6.4 Final WRITE

每条样本必须有 final WRITE：

```text
提交剩余全部target text
提交剩余全部target semantic
```

没有 final full-target WRITE 的样本不能进入正式 streaming 训练。

---

## 7. 如何获得完整连续 streaming 声音监督

### 7.1 保留完整 canonical target

不要把 target audio 预先切成许多互不相关的小 WAV 作为独立样本。

应保存：

```text
一个完整target semantic sequence
+ 多个指向该序列的连续span
```

训练第 `i` 次 WRITE 时：

输入：

```text
当前source prefix
已提交target text
已提交semantic history
当前speaker global
当前playback buffer状态
```

目标：

```text
下一个target text delta
下一个连续semantic span
```

### 7.2 训练和推理都必须保留 history

下一次 WRITE 必须基于：

```text
committed_text_ids
semantic_history
source incremental state
```

不能每次只看到 source prefix，然后从目标句第一个词重新生成。

### 7.3 Codec 必须连续解码

同一个 session：

- 只初始化一次 speaker；
- 不重置 target semantic history；
- 保持 decoder cache/state；
- 对新 semantic token 做增量解码；
- 使用小的 holdback 和 overlap；
- final 时只 flush 尚未播放的尾部。

如果每个 WRITE 独立调用完整 BiCodec decode，即使 semantic token 正确，也会产生：

- 音色变化；
- 边界爆音；
- 每段重新起音；
- 重复或吞字；
- chunk 中间空白。

---

## 8. 音色固定：推理和训练都会影响

### 8.1 音色不是纯推理参数

需要区分：

| 问题 | 主要决定因素 |
|---|---|
| 使用哪个说话人 | speaker global token |
| 是否跨 chunk 保持同一音色 | 训练条件、semantic speaker leakage、codec state |
| chunk 连接是否自然 | semantic history、边界监督、连续解码 |
| 韵律是否稳定 | streaming semantic训练和完整句子 replay |

只在推理时固定 speaker global token，并不能保证：

- 每个 chunk 的 semantic 不携带不同speaker残留；
- 每个 chunk 的基频和韵律一致；
- codec 重启后不发生音色变化；
- 首段和后续段落听起来是同一个人。

### 8.2 三种 speaker 模式

#### 模式一：fixed_system

整个系统使用一个预先确定的目标音色：

- 推荐用于低于 1 秒首次发音；
- 不需要等待用户提供3.2秒音色；
- 整个 session 固定一组 speaker token；
- 最容易实现稳定的跨 chunk 音色。

#### 模式二：pre_enrolled

用户在会话开始前提供5–10秒 speaker reference：

- 正式同传开始前已经完成speaker提取；
- 推理时没有在线 warm-up；
- 仍能保持较低首次发音延迟。

#### 模式三：live source cloning

同传开始后再从当前用户语音提取speaker：

- 无法严格保证首音频低于1秒；
- 当前3.2秒warm-up就是硬延迟；
- 不推荐作为第一版低延迟系统。

### 8.3 推荐训练策略

推荐：

```text
streaming trajectory:
    使用与正式推理一致的fixed_system speaker

Phase3 replay:
    保留原始多speaker数据
```

这可以同时达到：

- streaming 音色稳定；
- 不破坏 Phase3 多说话人基础能力；
- 不需要额外训练独立 talker。

可以使用多个预注册固定声音做 augmentation，但同一个 session 内不能切换speaker。

### 8.4 是否需要固定声线重合成

先进行 BiCodec speaker leakage 审计：

1. 固定同一 global speaker token；
2. 使用不同原始 target semantic；
3. 解码并计算 speaker embedding similarity；
4. 检查目标音色是否仍随原始说话人变化。

如果 semantic 中 speaker leakage 较小：

- 可以保留现有 target semantic；
- streaming branch 使用固定 global token；
- 增加 speaker consistency loss/metric。

如果 leakage 明显：

- 必须将 streaming 目标语音做固定声线 voice conversion 或重合成；
- 再用 BiCodec 重新编码；
- 生成与 fixed speaker 完全一致的 semantic/global pairing。

---

## 9. 新训练方法

### 9.1 初始化 checkpoint

必须加载：

```text
Phase3 v4 最佳 validation checkpoint
```

不能加载：

```text
当前有缺陷trajectory训练得到的pilot15 iter350
```

原因是当前 checkpoint 已经适配了错误的稀疏监督分布，继续训练会增加修复难度。

### 9.2 保留的模块

- Phase3 v4 主干；
- 当前 Qwen semantic generation；
- 当前 BiCodec；
- Phase3 原始 text/semantic token objective；
- 原始 Phase3 replay 数据格式；
- 已验证可用的 speaker conditioning。

### 9.3 重新初始化或重新训练的模块

- causal/incremental frontend adapter；
- action head；
- support head；
- safe-commit head；
- streaming Qwen LoRA；
- aligned semantic branch；
- 必要的 playback buffer/state embedding。

第一版不需要替换成 Emformer，也不需要增加额外独立 talker。

### 9.4 推荐数据混合

初始建议：

```text
60%–70% dense aligned streaming trajectory
30%–40% Phase3 replay
```

Phase3 replay 的目的：

- 保持完整句翻译能力；
- 保持语音自然度；
- 保持原有 semantic generation；
- 防止模型只会短 chunk；
- 防止灾难性遗忘。

### 9.5 Shuffle

使用和 Phase3 v4 同等级别的严格全局 shuffle，但 shuffle 单位必须是 session/trajectory。

```text
可以shuffle不同session顺序
不能打乱同一个session内部event顺序
```

如果训练时将事件展开成独立样本，则每条事件必须显式携带完整 committed history；更推荐按 session packing 后动态构造。

---

## 10. Loss 设计

推荐总目标：

\[
\begin{aligned}
L=&1.0L_{phase3\ replay}
+1.0L_{stream\ text}
+1.0L_{aligned\ semantic}\\
&+0.5L_{action}
+0.5L_{support}
+0.5L_{safe}\\
&+0.25L_{prefix\ KD}
+0.1L_{stability}
+0.1L_{boundary}\\
&+0.1L_{speaker}
+\lambda_dL_{deadline}
\end{aligned}
\]

### 10.1 `L_phase3 replay`

保留 Phase3 原始 autoregressive text/semantic CE。

作用：

- 保持 offline 翻译和语音质量；
- 防止 streaming 微调使基础模型退化；
- 保持完整句语义和长程上下文能力。

### 10.2 `L_stream text`

监督每次 WRITE 新增的目标文本：

```text
current source prefix
+ committed target history
-> next text delta
```

不得重复监督已经提交的文本。

### 10.3 `L_aligned semantic`

监督 text delta 对应的真实目标 semantic span：

```text
text_delta
+ previous semantic history
-> target_semantic[q_prev:q_next]
```

这是解决当前声音不正确和不连续的核心 loss。

### 10.4 `L_action`

监督 READ/WRITE。

必须使用：

- class-balanced CE；
- 或正负样本平衡 sampler；
- 报告 WRITE precision、recall、F1；
- 不能只看总 accuracy。

### 10.5 `L_support`

预测当前有多少 target phrase 已经被 source 支持。

建议 inverse-sqrt class weights：

\[
w_c \propto 1/\sqrt{freq_c}
\]

并冻结统计权重，避免 batch 之间波动。

### 10.6 `L_safe`

预测当前 candidate 是否可以安全提交且未来不需要修改。

建议 focal BCE 或高精度优先的加权 BCE。safe precision 应优先于 recall，避免过早输出错误内容。

### 10.7 `L_prefix KD`

使用冻结的 Phase3 v4/offline teacher 指导 streaming prefix：

- 保持翻译含义；
- 降低短前缀生成质量下降；
- 防止 streaming LoRA 远离 Phase3 分布。

### 10.8 `L_stability`

约束已经 committed 的 text/semantic 不回滚、不修改、不重复。

目标：

```text
committed token revision = 0
```

### 10.9 `L_boundary`

约束相邻 semantic/audio chunk 边界连续。

可使用：

- adjacent codec hidden continuity；
- overlap区域的mel/STFT一致性；
- 相邻chunk decoded embedding一致性；
- 边界能量和相位差约束。

如果不能高效反向通过完整 codec，可先作为 validation metric，并用 semantic-history consistency 作为训练代理。

### 10.10 `L_speaker`

约束所有 chunk 与同一个 session speaker embedding 一致。

如果 speaker evaluator 不可微：

- 使用轻量可微 speaker projection；
- 或将 speaker consistency 作为严格 checkpoint gate。

### 10.11 `L_deadline`

只对存在真实、安全 text+semantic target 的事件优化 deadline。

禁止：

```text
没有可发音target
却为了800ms deadline强制模型输出声音
```

初始：

```text
lambda_deadline = 0
```

后期缓慢增加到：

```text
0.1–0.2
```

---

## 11. 单次连续 Curriculum

正式 full198 可以在一个连续 Megatron job 中完成，不需要拆成多个互不相关训练stage。

### 11.1 0%–10%

启用：

- Phase3 replay；
- streaming text；
- aligned semantic；
- prefix KD。

关闭：

- deadline loss。

目标：

```text
先学会“说对”
而不是先强迫“说得早”
```

### 11.2 10%–50%

加入：

- balanced action；
- balanced support；
- safe commit；
- boundary continuity；
- playback buffer supervision。

目标：

- 学会何时 READ；
- 学会何时自然 WRITE；
- 学会生成连续的下一个声音片段。

### 11.3 50%–100%

加入：

- deadline weight 从 0 缓慢增加到 0.1–0.2；
- model-generated history scheduled sampling；
- speaker consistency；
- free-running validation。

scheduled sampling 逐渐让模型看到自己生成的 text/semantic history，而不总是依赖完美 teacher history。

### 11.4 Curriculum 不丢弃前面的 loss

后续 curriculum 不应删除前面已经使用的核心 loss。

正确形式：

```text
前一阶段loss继续保留
+ 新增或提高后续loss权重
```

不是：

```text
进入新阶段后丢弃Phase3 replay或aligned semantic
```

---

## 12. 严格低于 1 秒的预测式监督

对于语言重排序或开头信息不足的样本，reference alignment 可能在 800ms 内没有任何安全目标前缀。

如果要求更高比例样本在 1 秒内发音，需要额外构造安全 anticipation 数据。

### 12.1 生成流程

```text
source prefix
  -> frozen Phase3/offline teacher生成候选target prefix
  -> 与最终reference进行语义一致性检查
  -> 检查多个未来prefix中是否稳定
  -> 只接受高置信、不会回滚的候选
  -> 使用fixed speaker生成候选语音
  -> BiCodec重新编码
  -> 得到text + semantic + audio完整监督
```

### 12.2 示例

Source prefix：

```text
I would like to...
```

安全候选：

```text
我想……
```

如果 teacher 在多个后续 prefix 中都保持“我想”，并与最终句义一致，可以为“我想”生成固定音色目标语音，作为早期 WRITE 监督。

如果 teacher 生成“我希望”，但原 target audio 说的是“我想要”，则不能从原 target semantic 中任意截取片段冒充“我希望”的声音。必须：

- 为“我希望”重新生成语音并编码；
- 或放弃该 anticipation 样本。

### 12.3 不能保证所有句子都低于 1 秒

即使系统架构允许低于 1 秒，部分样本仍受以下因素限制：

- source 开头静音；
- 语言重排序；
- 专有名词尚未出现；
- 否定词出现在后面；
- target 第一个词依赖后续 source；
- 网络、计算和浏览器缓冲。

因此应分别报告：

- eligible 样本 first audio；
- 全量样本 first audio；
- natural WRITE；
- anticipation WRITE；
- deadline miss；
- unsafe rejection。

---

## 13. Runtime 必须同步重构

### 13.1 取消在线 3.2 秒 speaker warm-up

低于 1 秒版本必须采用：

- fixed system speaker；
- 或 pre-enrolled speaker。

不能在同传开始后再等待3.2秒提取speaker。

### 13.2 内部固定 160ms policy tick

UI/network chunk可以是：

```text
320 / 480 / 640ms
```

但内部必须拆分或累计为：

```text
160ms policy tick
```

800ms deadline必须在接近800ms时检查，不能等到1280ms。

### 13.3 增量前端 parity

训练数据生成和runtime必须使用完全相同的：

- 音频归一化；
- sample rate；
- chunk；
- padding；
- WhisperVQ/前端tokenizer版本；
- overlap；
- stable commit规则；
- cache/state。

验收：

```text
token parity >= 99.9%
committed token revision = 0
```

### 13.4 committed history

每次决策、文本生成和semantic生成都必须输入：

```text
committed_text_ids
recent_semantic_history
source_incremental_state
```

避免：

- 每次重新输出首词；
- “各位各位各位”；
- semantic重复；
- chunk之间内容断裂。

### 13.5 Stateful Qwen 和 BiCodec

建议：

- Qwen使用KV cache或有界历史；
- semantic生成仅追加新token；
- BiCodec保存连续decoder state；
- 同一session固定speaker global；
- 不重复解码已经播放的semantic；
- final只flush尾部。

### 13.6 Micro-WRITE 长度

初始建议每次实际生成：

```text
200–400ms有效目标音频
约10–20个50Hz semantic token
```

需要结合 codec holdback/overlap计算真实可立即播放长度。不能生成12个token后扣除100ms holdback和80ms overlap，只剩约60ms可实时播放。

### 13.7 真实 First Audio 指标

必须区分：

1. first scheduler WRITE；
2. first committed text；
3. first semantic token；
4. first emitted PCM；
5. browser first audible sound。

正式低延迟指标应使用：

```text
first emitted PCM
browser first audible sound
```

被safe gate拒绝的forced WRITE不能算首次发音。

---

## 14. Validation 与 checkpoint 选择

仅看训练 loss 或 validation CE 下降不能证明模型会真实同传。

每个候选 checkpoint 必须执行：

### 14.1 数据和teacher-forced指标

- source alignment coverage；
- target alignment coverage；
- bilingual alignment coverage；
- semantic coverage；
- semantic gap/overlap；
- action WRITE precision/recall/F1；
- support macro-F1和confusion matrix；
- safe precision/recall；
- streaming text CE；
- aligned semantic CE；
- Phase3 replay CE；
- teacher agreement；
- committed stability。

### 14.2 Free-running streaming指标

- natural WRITE 数；
- anticipation WRITE 数；
- forced/deadline miss 数；
- first committed text；
- first semantic；
- first emitted PCM；
- first browser sound；
- Average Lagging；
- LAAL/DAL；
- RTF；
- semantic rollback/revision；
- output buffer underflow/overflow；
- silence ratio；
- output duration ratio。

### 14.3 语音和翻译质量

- ASR-BLEU / Speech-BLEU；
- COMET 或文本翻译质量；
- intelligibility；
- AutoPCP 等语音质量指标；
- speaker similarity；
- 跨chunk speaker similarity；
- clipping；
- DC offset；
- RMS；
- 音频重复；
- 丢词和吞字。

### 14.4 Checkpoint 选择

禁止直接使用：

```text
latest checkpoint
```

应生成：

```text
checkpoint_selection.json
```

综合选择：

- 安全性；
- first useful PCM；
- AL/LAAL；
- 翻译质量；
- speaker稳定性；
- semantic coverage；
- Phase3 offline replay退化幅度。

---

## 15. 数据和模型质量门

### 15.1 数据质量门

正式 full198 训练前：

```text
source alignment coverage >= 0.85
target alignment coverage >= 0.85
streaming semantic coverage >= 0.95
semantic gap = 0
semantic overlap = 0
final coverage = full target
```

建议：

- 长于3秒样本多数有至少2次natural WRITE；
- forced/无监督deadline比例低于10%–20%；
- 双向语言分别审计；
- 至少人工试听100条span拼接结果；
- 拼接结果必须与完整目标音频内容一致。

### 15.2 训练小规模放行门

先对32–128条已知正样本overfit：

- 已知320/480/640ms正标签可以恢复；
- WRITE recall明显高于0；
- 生成的text delta正确；
- 生成的semantic span与目标对应；
- 所有span拼接后可懂；
- 没有首词重复；
- 没有semantic gap/overlap。

### 15.3 15-shard放行门

使用固定0–14 shard：

- 完成一个完整trajectory coverage epoch；
- strict session-level global shuffle；
- 真实PCM rollout；
- 双向dev样本；
- natural WRITE和first useful PCM通过；
- Phase3 replay质量没有明显退化。

15-shard不是正式最终训练，只是防止full198运行数天后才发现数据或runtime仍有问题。

### 15.4 Full198目标

建议目标：

```text
first useful PCM p50 < 800ms
first useful PCM p95目标 < 1000ms
committed token rollback = 0
semantic coverage >= 95%
speaker跨chunk稳定
ASR-BLEU/Speech-BLEU相对Phase3下降可控
```

语言重排序导致的不可安全早说样本应单独统计，不能通过不安全强制WRITE掩盖。

---

## 16. 推荐实施顺序

### Step 1：新建独立项目

建议目录：

```text
experiments/uniss_phase3_dense_aligned_streaming_full198_v1/
  data/
  training/
  inference/
  evaluation/
  tests/
  configs/
```

不得覆盖：

- 当前 pilot15；
- 旧 Phase1/2/3；
- 旧 wait-k；
- 旧 StreamSpeech/CTC；
- 已有 Gradio demo；
- 已有报告和checkpoint。

### Step 2：复用正式alignment工具

复用A4–A8中已经存在的：

- source forced alignment；
- target forced alignment；
- bilingual alignment；
- coverage审计；
- 音频重建。

不要继续旧的多stage训练，只复用数据处理思想和可靠代码。

### Step 3：制作固定0–14 shard dense trajectory

并行处理：

- source alignment；
- target alignment；
- bilingual links；
- semantic span；
- 160ms observations；
- playback buffer；
- final full coverage。

### Step 4：数据审计

自动检查：

- coverage；
- monotonicity；
- gap/overlap；
- text拼接；
- semantic拼接；
- event数量；
- natural WRITE分布；
- target audio duration。

人工试听：

- 英中各50条；
- 单独听每个chunk；
- 听拼接后的完整target；
- 对比原target WAV。

### Step 5：小样本overfit

32–128条：

- 覆盖短句、长句；
- 覆盖单调和重排序；
- 覆盖多次WRITE；
- 覆盖双向；
- 验证training/runtime token parity。

### Step 6：15-shard完整验证

- 从Phase3 v4最佳checkpoint加载；
- 跑一个trajectory coverage epoch；
- TensorBoard记录所有新增loss和streaming指标；
- 执行真实PCM rollout；
- 不使用当前iter350续训。

### Step 7：Full198数据制作

只有15-shard通过后，使用完全相同代码生成full198派生数据。

### Step 8：Full198正式单次训练

- 单个连续Megatron curriculum；
- 严格全局session shuffle；
- Phase3 replay；
- 8卡训练；
- 定期free-running streaming validation；
- 根据checkpoint gate自动保留best。

### Step 9：真实在线推理

- fixed/pre-enrolled speaker；
- 160ms内部tick；
- frontend cache；
- Qwen KV cache；
- stateful BiCodec；
- WebRTC/browser实时播放；
- 分别测算法延迟和端到端wall-clock延迟。

---

## 17. 是否只需要正式训练一次

正式 full198 模型可以只进行一次连续训练，但不能省略训练前的正确性验证。

推荐含义：

```text
15-shard：
    数据、loss、runtime正确性门
    不是最终正式模型

full198：
    一次连续Megatron curriculum正式训练
```

如果直接跳过15-shard门，当前已有经验表明很可能在训练完成后才发现：

- WRITE监督仍然稀疏；
- semantic span仍不对应文字；
- runtime历史不一致；
- first WRITE并没有实际PCM；
- 音频仍然重复或静音。

---

## 18. 风险与对应措施

| 风险 | 后果 | 解决方法 |
|---|---|---|
| 双语align错误 | 过早输出错误翻译 | coverage/confidence过滤、safe head |
| target forced align不准 | 文字和声音不匹配 | phrase合并、低能量边界、人工抽检 |
| target semantic带speaker残留 | 固定global后仍变音色 | speaker leakage审计、固定声线重编码 |
| deadline权重过早过大 | 模型胡乱WRITE | 后50%才逐步升权重 |
| action标签仍不平衡 | 模型只会READ | inverse-sqrt权重、balanced sampler |
| teacher-forced正常但free-running失败 | demo重复或静音 | scheduled sampling、每checkpoint rollout |
| UI chunk决定policy | 800ms被拖到1280ms | 内部固定160ms tick |
| codec每chunk重置 | 音色变化和边界爆音 | stateful codec和连续history |
| 只看scheduler WRITE | 延迟指标虚假 | 统计first emitted PCM/browser sound |
| full198派生数据过大 | I/O慢、磁盘爆炸 | 完整序列只存一次，event只存offset |

---

## 19. 最终推荐方案摘要

当前原始 UniST 数据并没有缺少完整目标语音。缺失的是“完整目标语音如何分配到每次流式 WRITE”的派生监督。

正确解决方案不是重新下载数据，而是：

1. 对 source audio/source text 做时间对齐；
2. 对 target audio/target text 做时间对齐；
3. 做 source-target bilingual word alignment；
4. 计算每个目标phrase最早安全提交的source时间；
5. 每160ms建立READ/WRITE observation；
6. 将每次新增文字映射到目标音频中的真实、连续semantic span；
7. final WRITE覆盖剩余完整语音；
8. 模拟播放buffer，学习连续输出节奏；
9. streaming branch使用固定或预注册speaker；
10. Phase3 replay保持原有翻译和语音质量；
11. 从Phase3 v4最佳checkpoint重新训练；
12. runtime使用相同增量前端、committed history和stateful BiCodec；
13. 用真实PCM和browser first sound评估低延迟。

最终模型仍然采用：

```text
Phase3 v4主架构
+ 当前Qwen semantic generation
+ 当前BiCodec
```

无需增加新的独立 talker，也无需切换到 Emformer。核心变化是将当前错误的稀疏监督升级为：

```text
Dense
+ bilingual aligned
+ full-semantic coverage
+ playback-aware
+ speaker-consistent
+ train/runtime parity
```

这套方案针对性解决当前五个核心故障：

1. 模型一直等待、不自然WRITE；
2. 文字与声音不匹配；
3. 每次只输出极短声音、中间长时间空白；
4. 多个chunk之间重复、断裂和音色变化；
5. 报告中的WRITE时刻并不等于用户真正听到声音的时刻。

