# Simul-UniSS 真正亚秒级 Speech-to-Speech 同传研究与实施方案

> 文档日期：2026-07-30
> 目标仓库：`/opt/dlami/nvme/jasonleeeli/projects/UniSS`
> 目标：把当前约3--6秒首包的 pseudo-streaming 系统推进到真实原始音频端到端 p50 < 1秒
> 原则：不覆盖现有 Phase1--3、Stage3/4/6/7A、评估结果、网页服务或 checkpoint；所有新实验使用独立目录和版本号
> 状态：研究与实施计划，不代表亚秒级模型已经训练完成

## 1. 结论先行

当前系统无法通过简单修改 `write_logit_bias` 或把播放器时间轴提前，真正达到1秒以下。
当前主要限制不是 H200 算力，而是系统在收到足够源语音之前没有稳定、因果、可提交的表示：

1. 当前 WhisperVQ/GLM 前端包含约4秒 block 内未来上下文，麦克风累计前缀实验首个稳定 token 约 `4.22 s`。
2. 当前 R2 Stage7A dev First WRITE NCA 约 `4.11 s`，策略本身也明显偏保守。
3. 当前决策间隔是 `640 ms`，即使模型愿意早写，动作机会也不够密。
4. 当前麦克风模式需要约 `3.2 s` 音频提取 source voice speaker token，单这一项就已超过1秒。
5. 当前每次 WRITE 往往生成一个较长短语和大量 semantic token，不适合100--300 ms级持续输出。
6. 当前 Transformers 网页推理重复计算完整 prompt，没有正式的 Qwen KV-cache 会话接口。
7. 当前 BiCodec 依赖 overlap re-decode，已有实现可工作，但不是严格因果 codec decoder。

推荐主线是：

```text
20 ms 麦克风 PCM
  ↓
160 ms chunk + 0--80 ms有限右上下文
  ↓
带显式 cache 的 Causal Audio Streaming Student
  ├─ stable GLM token
  ├─ Source CTC
  ├─ Target CTC capacity
  └─ uncertainty/confidence
  ↓
Bayesian Safe-Commit Gate + Qwen WAIT/WRITE KV cache
  ↓
8--16个 BiCodec semantic token 的 micro-WRITE（160--320 ms语音）
  ↓
低 holdback Streaming BiCodec / 后续因果 codec student
  ↓
80--120 ms浏览器 jitter buffer
```

一个有现实可行性的第一阶段工程目标是：

```text
p50 First Audio CA <= 900 ms
p95 First Audio CA <= 1400 ms
p50 First WRITE NCA <= 640 ms
RTF p95 < 0.6
premature WRITE <= 5%
相对当前R2 Text/Speech质量下降不超过预设gate
```

“所有句子都强制 <1秒”并不科学。中英翻译存在语序重排和否定等歧义，安全系统应允许困难句子
自适应等待。正式目标应是 p50 亚秒、p95 受控，而不是为了数字强制模型猜测。

## 2. 什么叫“真正低于1秒”

必须同时满足：

- 输入是实时到达的原始波形，不允许预先读取完整音频；
- encoder 在时间 `t` 不能访问 `t + future` 的源音频；
- 所有卷积、attention、特征提取和量化都有 streaming cache；
- 第一个目标音频 sample 在源开始后1秒内真实到达声卡；
- 计算、codec、网络和浏览器缓冲全部计入 CA latency；
- 已播放目标语音不可回滚；
- 不允许把完整生成结果简单在时间轴上向前移动；
- 不允许把“第一个无意义音节”当作 First WRITE 改善。

推荐同时冻结以下定义：

```text
First WRITE NCA:
  策略首次提交非空目标内容时已消耗的源音频时长。

First Audio CA:
  从浏览器开始采集源音频，到第一段非静音目标PCM进入播放队列的真实墙钟时间。

Useful First Audio:
  第一段通过prefix correctness和ASR可懂度gate的目标音频到达时间。
```

`Useful First Audio` 可以防止模型提前输出“uh”“the”或噪声来投机延迟指标。

## 3. 当前系统为什么接近整句后才输出

### 3.1 WhisperVQ block 是硬瓶颈

当前 GLM tokenizer 的关键属性是：

```text
encoder_causal_convolution = true
encoder_causal_attention = false
quantize_causal_block_size = 200
quantize_causal_encoder = false
```

Whisper 卷积后约20 ms一帧，block size 200 对应约4秒。块内仍然使用未来上下文，所以当前网页只能：

```text
640 ms  → 重新编码 audio[0:640]
1280 ms → 重新编码 audio[0:1280]
...
约4秒后才观察到足够稳定的GLM前缀
```

这不是缓存不足的小问题，而是模型训练和结构使用了未来上下文。

### 3.2 R2只优化 action head，不能修复源信息到达时间

R2 explicit-latency 相比其他 Reward-v2 action policy 更积极，但其 backbone、source timeline 和
WRITE生成能力仍来自已有 Stage6/7A。action GRPO 可以减少不必要 WAIT，却不能让4秒后才稳定的
source token 在500 ms时提前出现。

### 3.3 640 ms chunk 对亚秒目标过粗

若 chunk 为640 ms：

```text
第1次动作机会 = 640 ms
若再WAIT一次      = 1280 ms
```

即使计算时间为0，`wait-k=2` 已经超过1秒。亚秒模式应主要使用 `80/160/240/320 ms` grid，
640 ms只保留为质量模式。

### 3.4 source voice cloning 与亚秒目标冲突

当前麦克风页面约3.2秒后才冻结 speaker token。若目标是 <1秒，首版必须采用以下之一：

1. 固定目标音色；
2. 用户预先上传/注册一段目标音色参考；
3. 使用历史会话已缓存的 speaker embedding；
4. 首包先用固定音色，2--3秒后切换会造成音色突变，因此不推荐。

“首次进入网页、没有参考音色、同时要求复制当前说话人、还要求500 ms输出”在当前 BiCodec speaker
建模下不可同时满足。

## 4. 端到端亚秒时延预算

建议先以固定目标音色冻结预算：

| 组件 | p50预算 | p95预算 | 说明 |
|---|---:|---:|---|
| 麦克风累积首个chunk | 160 ms | 240 ms | `stream_every=80--160 ms` |
| encoder有限右上下文 | 40--80 ms | 120 ms | 不能使用4秒block |
| causal student计算 | 30--60 ms | 100 ms | H200、batch=1、cache |
| CTC + Safe-Commit policy | 10--30 ms | 50 ms | 小head，不调用长生成 |
| Qwen action + micro-WRITE | 120--220 ms | 350 ms | KV cache，短输出 |
| BiCodec首块 | 60--120 ms | 180 ms | 低holdback或因果student |
| 网络与浏览器buffer | 80--120 ms | 200 ms | 公网可能更高 |
| **总计** | **500--790 ms** | **1000--1240 ms** | 不含语言学必要等待 |

公式为：

```text
L_first_audio_CA =
  L_capture
  + L_right_context
  + L_frontend_compute
  + L_safe_commit
  + L_qwen_micro_write
  + L_codec
  + L_transport_buffer
```

模型因语义不确定额外 WAIT 的时间应单独记为：

```text
L_total = L_system_floor + L_linguistic_wait
```

系统目标是把 `L_system_floor` 压到约500--800 ms；不能消除所有 `L_linguistic_wait`。

## 5. 推荐主方案：Causal Student + Bayesian Safe Commit + Micro-WRITE

## 5.1 Motivation

当前问题可以分成三个不同的不确定性：

1. **声学不确定性**：当前160 ms音频是否已经形成稳定音素/GLM token？
2. **翻译支持度不确定性**：当前源前缀是否足以支持一个不可回滚的目标短语？
3. **语音生成不确定性**：目标 semantic 是否稳定、非塌缩且可以立刻播放？

单个 WAIT/WRITE logit 无法区分三者。推荐让 causal frontend 提供显式证据，再由 safe-commit
controller 决定是否提交，Qwen负责真正的翻译和语音 token 生成。

## 5.2 Causal Audio Streaming Student v2

当前已有：

- `training/simul_uniss/audio_streaming_student.py`
- `training/simul_uniss/streaming_student.py`
- `training/simul_uniss/train_audio_student.py`

这些代码说明项目已经具备 causal log-Mel、causal convolution、Transformer mask、teacher GLM CTC、
Source/Target CTC 的骨架。但当前实现仍是 bootstrap：

- `TransformerEncoder` 每次接收完整 prefix，没有在线 KV/cache；
- prefix label 按文本字符比例截断，不是真实时间戳；
- 固定三次 stride=2，约80 ms一个输出步，但未验证真实 token stability；
- 没有 teacher hidden-state distillation；
- 没有 right-context随机化；
- 没有 batch-one online inference API。

建议新增独立版本：

```text
training/simul_uniss/subsecond_v1/
  causal_audio_student_v2.py
  streaming_cache.py
  alignment_dataset.py
  losses.py
  train_frontend.py
  export_frontend.py
```

结构建议：

```text
16 kHz waveform
→ center=False log-Mel, hop=10 ms
→ causal Conv subsampling ×4 或 ×8
→ Chunk-Conformer/Emformer/Transformer
   left cache = 1.5--4.0 s
   right context = {0, 40, 80, 160} ms
→ dense hidden every 40--80 ms
→ teacher GLM CTC
→ Source CTC
→ Target CTC capacity
→ stability/confidence head
```

不要简单把完整 Whisper attention mask 改成 causal 后直接训练。更稳妥的路线是：

1. 用当前 WhisperVQ 完整句表示作为 teacher；
2. student只看过去和有限右上下文；
3. teacher codebook保持兼容，避免重训整个 Qwen 词表；
4. 先蒸馏 hidden/token，再联合 CTC policy heads；
5. 最后接 Qwen 做端到端 refinement。

建议损失：

```text
L_frontend =
    1.0 * L_teacher_glm_ctc
  + 0.5 * L_hidden_distill
  + 0.3 * L_source_ctc
  + 0.4 * L_target_capacity_ctc
  + 0.2 * L_prefix_stability
  + 0.1 * L_chunk_consistency
```

其中：

```text
L_prefix_stability = KL(p_token(x[0:t]), stopgrad(p_token(x[0:t+Δ])))
```

它鼓励早期输出在看到更多音频后仍保持一致。

## 5.3 真正的时间戳与训练数据

现有 pseudo schedule 按 token/字符比例对齐，只适合打通流程。亚秒训练必须知道“当前源前缀真正支持
哪个目标内容”。推荐对15 shard先构造精细对齐，再扩展 full198：

### Source alignment

- 中文：FunASR/Paraformer timestamp 或 WhisperX/MFA；
- 英文：WhisperX/MFA；
- 保存 word/character start-end time、置信度和静音区间；
- source GLM token 使用 teacher encoder receptive field 映射到音频时间。

### Target support alignment

目标文本不能简单按字符比例切。推荐组合：

1. 双语 word alignment（awesome-align/SimAlign或冻结MT attention）；
2. 依存/短语边界；
3. Hibiki 风格 contextual alignment；
4. 对每个 target prefix 计算“所需source最晚结束时间”；
5. 加语言方向特定重排安全边界。

每个目标 token `y_j` 保存：

```text
support_end_ms(y_j) =
  生成y_j所必需的最后一个source证据到达时间
```

WRITE 只允许提交满足：

```text
support_end_ms(y_j) <= current_source_ms
```

的数据前缀。

## 5.4 Bayesian Safe-Commit Gate

为了在低延迟和不可回滚之间取得平衡，可以显式估计“现在写是否安全”的后验概率。

定义：

```text
z_t = 1：当前时刻提交下一个目标micro-phrase是安全的
z_t = 0：应该继续等待
```

严格 Bayes 形式：

```text
p(z_t=1 | e_t, b)
  ∝ p(e_t | z_t=1) · p(z_t=1 | b)
```

其中：

- `b`：用户选择的 latency budget，例如 fast/balanced/quality；
- `p(z_t=1 | b)`：prior，fast模式更偏向WRITE，quality模式更偏向WAIT；
- `e_t`：likelihood evidence；
- evidence 包括 source CTC增长、target capacity增长、prefix entropy、teacher/student agreement、
  Qwen action margin、静音/标点边界和历史revision；
- posterior 是当前不可回滚提交安全概率。

一个可实现的 log-odds 形式：

```text
logit p_safe =
    logit prior_budget
  + w1 * source_ctc_growth
  + w2 * target_capacity_margin
  - w3 * frontend_entropy
  + w4 * qwen_write_margin
  + w5 * phrase_boundary_score
  - w6 * recent_revision_rate
```

初始阈值：

```text
fast:     WRITE if posterior >= 0.75
balanced: WRITE if posterior >= 0.88
quality:  WRITE if posterior >= 0.95
```

posterior应使用 held-out dev 做温度校准或 isotonic calibration，不能把未校准 softmax 当概率。

## 5.5 Qwen KV-cache 与 micro-WRITE

当前网页 `QwenLiveAdapter` 每次动作重新前向完整 prompt。亚秒系统应提供真正会话状态：

```python
state = adapter.start_session(header)
state.append_source(new_glm_tokens)       # 只prefill新增token
action = state.next_action()              # 单token decode
delta = state.generate_micro_write(8, 16) # 短文本/semantic
```

建议把一个 WRITE 从“完整短语 + 100--700 semantic token”改成：

```text
目标文本：1--4个词或一个中文短语
semantic：8--16 token
目标音频：160--320 ms
```

这样生成和 codec 可以并行：

```text
Qwen生成第2个semantic micro-chunk
同时BiCodec解码并播放第1个micro-chunk
```

训练时必须加入 micro-WRITE 数据，否则只在推理阶段硬截断会破坏模型格式。

建议新 protocol 保持旧词表兼容：

```text
START_GLM x1 x2 END_GLM
WRITE_GENERATE
START_CONTENT "Tomorrow morning" END_CONTENT
START_SEMANTIC s1 ... s12 END_SEMANTIC

START_GLM x3 x4 END_GLM
WRITE_GENERATE
START_CONTENT "at nine" END_CONTENT
START_SEMANTIC s13 ... s24 END_SEMANTIC
```

## 5.6 Streaming BiCodec v2

短期可复用现有 `StreamingBiCodecDecoder`，但参数需要低延迟专用版本：

```text
left_context_tokens = 25--50
holdback_tokens = 2--3        # 40--60 ms
overlap_ms = 30--50 ms
micro semantic chunk = 8--16  # 160--320 ms
```

中期训练 causal codec student：

- teacher 是完整 BiCodec decoder；
- student 使用 causal convolution/cache；
- distill waveform、mel、multi-resolution STFT、speaker embedding；
- 输入 speaker token 在会话开始前固定；
- 每次只解码新增 semantic token；
- 无 overlap re-decode 时目标首包可再减少50--150 ms。

codec损失：

```text
L_codec =
    L_waveform
  + 1.0 * L_multi_resolution_STFT
  + 0.5 * L_mel
  + 0.2 * L_speaker
  + 0.2 * L_boundary_continuity
```

## 5.7 推荐架构的完整数据流与模块边界

推荐方案不是把当前 Phase3 模型整体替换掉，而是在它前后增加真正的在线接口，并针对短提交方式继续训练。
完整架构如下：

```text
浏览器20 ms PCM包
  │
  ├─ Audio Ring Buffer：累计到160 ms，并额外等待0--80 ms右上下文
  │
  ▼
Causal Audio Student v2
  ├─ 在线log-Mel和causal convolution cache
  ├─ Chunk-Conformer/Emformer attention KV cache
  ├─ GLM token CTC
  ├─ source-text CTC
  ├─ target-capacity head
  └─ token stability/entropy head
  │
  ├─ 稳定GLM token ───────────────┐
  └─ acoustic evidence ────────┐  │
                               ▼  ▼
                         Bayesian Safe-Commit
                         ├─ WAIT
                         ├─ internal draft
                         └─ irreversible WRITE
                               │
                               ▼
                  Phase3 Qwen + session KV cache
                  ├─ 1--4个目标文本词
                  └─ 8--16个目标semantic token
                               │
                               ▼
                  Streaming BiCodec decoder
                  ├─ codec state/cache
                  ├─ 30--50 ms overlap
                  └─ 80--120 ms播放缓冲
                               │
                               ▼
                        浏览器连续播放PCM
```

系统内部有两种不同的“提交”，不能混为一谈：

1. **source-token commit**：Causal Student确认某些源 GLM token 已稳定，可加入 Qwen 输入；
2. **target-audio commit**：Bayesian Gate确认某个目标 micro-phrase 已足够安全，可以生成并播放，播放后不可撤销。

前者解决声学表示抖动，后者解决翻译重排和语义不确定性。只有一个 WAIT/WRITE head 时，这两类错误会
混在一起，难以知道系统究竟应该改前端还是改策略。

## 5.8 Whisper到底如何处理

### 5.8.1 项目里存在两种不同角色的 Whisper

本方案中必须区分：

| 名称 | 当前用途 | 是否进入在线主链路 | 推荐处理 |
|---|---|---:|---|
| UniSS WhisperVQ/GLM speech tokenizer | 把源音频编码为 Qwen 能接收的 GLM speech token | 当前进入，但约4秒 block 导致 pseudo-streaming | 保留为冻结 teacher，不再作为亚秒在线前端 |
| Whisper ASR/WhisperX | 转写、时间戳、评价生成语音 | 否 | 只作为离线数据处理和评价工具，不参与在线同传决策 |

因此，“训练 streaming Whisper”在推荐方案中的准确含义是：

> 不覆盖、也不直接破坏原 WhisperVQ，而是额外训练一个只看历史和极少右上下文的 Causal Audio
> Student，使它输出与原 WhisperVQ 相同语义空间的 GLM token。

### 5.8.2 为什么不直接把原 WhisperVQ attention mask 改成 causal

直接翻转 attention mask 风险很高：

1. 原模型在训练时依赖整段或 block 内未来帧，权重已经适应这种信息分布；
2. 同一层的 hidden state 在失去未来上下文后会发生系统性偏移；
3. VQ codebook边界可能改变，导致输出 token 虽然编号合法，但对 Phase3 Qwen 的含义已经漂移；
4. 原 checkpoint 的 offline 复现能力会被破坏；
5. 只改 mask 仍没有 convolution/attention cache，计算上可能继续重复编码整个前缀。

推荐保留原模型作为 teacher，所有 teacher 参数 `requires_grad=False`，teacher只在离线数据准备或训练
时运行。在线部署时移除 teacher，只加载 student。

### 5.8.3 Causal Student是否需要额外训练

需要。当前 `training/simul_uniss/audio_streaming_student.py` 已经是可用的 bootstrap，但还不能直接称为
正式亚秒前端。正式版本至少需要额外完成：

- 从完整 WhisperVQ 蒸馏 GLM token和中间表示；
- 用160/240/320 ms随机 chunk训练，而不是固定完整 prefix；
- 右上下文在 `{0, 40, 80, 160}` ms之间随机化；
- 使用真实 source timestamp和 target support alignment，而不是字符比例截断；
- 实现 convolution state和 attention KV cache；
- 训练 token stability head并在 dev 上校准；
- 用 batch-one 流式回放确认“分块输入”和“一次输入”输出一致。

当前 bootstrap 的三层 stride=2 大约每80 ms产生一个 hidden step。正式版本可以先比较：

```text
Student-S：hidden 512，12层，8 heads，约40--80 ms输出步
Student-B：hidden 768，12--16层，12 heads，约40 ms输出步
```

最终不按参数规模选模型，而按 `first stable GLM`、teacher agreement、RTF和下游翻译质量共同选。

### 5.8.4 可选的 Streaming WhisperVQ 分支

还可以建立独立实验，把原 WhisperVQ 权重复制到新目录后进行 chunk-causal fine-tuning：

```text
原WhisperVQ checkpoint（只读）
  → 复制初始化新模型
  → block mask改为160--320 ms chunk + 0--80 ms lookahead
  → 增加每层KV cache
  → 用原WhisperVQ做self-distillation
  → 输出沿用原VQ codebook
```

这个分支参数兼容性可能更好，但模型较大、缓存工程复杂，并不一定比专门设计的 student 更快。因此它
应该作为 E2b 对照，而不是第一实现。无论走哪条路线，现有 WhisperVQ 文件和 checkpoint 都不修改。

## 5.9 Causal Audio Student的设计、训练与推理

### 5.9.1 原理

令输入波形为 `a[1:T]`。offline teacher在训练时可以看到完整 `a[1:T]`，student在时刻 `t` 只能看到：

```text
a[1 : t + r]
```

其中 `r` 是最多80 ms的有限右上下文。student需要逼近 teacher对“已经有足够证据的部分”的判断，而
不是复现 teacher对整句未来信息的利用。

每个40--80 ms输出步产生 hidden state `h_t` 和四类预测：

```text
q_glm(t)       ：原WhisperVQ codebook token分布
q_source(t)    ：源文字/音素CTC分布
q_capacity(t)  ：当前源前缀最多支持多少目标内容
q_stable(t)    ：当前token未来是否仍会保持不变
```

左侧历史通过 cache保留，右侧只允许固定小窗口。因此算法延迟由 `chunk + right context` 显式控制，
不会随着句子变长而增长。

### 5.9.2 一条训练样本如何构造

原始训练记录需要包含：

```json
{
  "source_audio": "...wav",
  "target_audio": "...wav",
  "source_text": "我明天上午九点去北京开会。",
  "target_text": "I'm going to Beijing for a meeting at nine tomorrow morning.",
  "teacher_glm": [123, 456, 789],
  "teacher_hidden": "可选的离线缓存",
  "source_alignment": [{"word": "明天", "end_ms": 430}],
  "target_support": [{"token": "tomorrow", "support_end_ms": 430}],
  "target_semantic": [31, 92, 18]
}
```

同一句话不会只生成一个训练样本，而是在多个时间点切成前缀。例如：

```text
t = 160, 320, 480, 640, 800, ... ms
right_context = random choice(0, 40, 80, 160) ms
```

每轮 epoch再随机改变切点、chunk size和右上下文。这样模型不能记住固定分块位置，也不会只在静音边界
工作。

### 5.9.3 蒸馏和监督目标

teacher序列与student时间步长度不同，先用 CTC path、单调 DTW或时间戳完成对齐，再计算：

```text
L_glm_ctc  ：student恢复teacher GLM token序列
L_hidden   ：投影后的student hidden逼近已对齐teacher hidden
L_source   ：识别当前已经出现的source prefix
L_capacity ：预测当前最多支持的target prefix
L_stable   ：判断token在未来Δ时间后是否仍一致
L_cons     ：同一音频用不同chunk划分时输出一致
```

推荐第一版总损失：

```text
L_student =
    1.0 L_glm_ctc
  + 0.5 L_hidden
  + 0.3 L_source
  + 0.4 L_capacity
  + 0.2 L_stable
  + 0.1 L_cons
```

`L_stable` 的标签不能按“当前 argmax置信度高”生成，而要真正向未来查看：若 token在 `t`、`t+160 ms`
和 `t+320 ms` 三个前缀中保持相同，且最终teacher序列也包含它，才标记为稳定。未来信息只用于产生训练
标签，不进入student推理输入。

### 5.9.4 在线推理

student推理不是每160 ms重新编码从0到当前时刻的全部音频，而是：

```python
state = student.start_session()
for pcm_20ms in microphone:
    state.append_pcm(pcm_20ms)
    if state.ready_for_tick(chunk_ms=160, right_context_ms=80):
        output = state.forward_new_chunk()
        stable_glm = state.commit_stable_glm(
            output,
            min_posterior=0.90,
            persistence_ticks=2,
        )
```

`forward_new_chunk()`只读取新的 Mel帧，并复用：

- causal convolution左状态；
- 每层 attention key/value；
- CTC prefix beam状态；
- 已提交 GLM token；
- 最近未稳定 token的小缓冲。

这样单次计算量近似与新 chunk长度相关，而不是与会话累计长度相关。

## 5.10 Bayesian Safe-Commit的设计、训练与推理

### 5.10.1 它判断的不是“翻译对不对”，而是“现在播放是否安全”

定义候选目标 micro-phrase为 `m_j`，安全变量为：

```text
z_t,j = 1：在时刻t播放m_j，未来无需因为新增源信息而修改
z_t,j = 0：当前播放有较高误译、漏译或语序回滚风险
```

训练标签由双语 support alignment和未来前缀一致性共同产生：

```text
z_t,j = 1 iff
  support_end_ms(m_j) <= t - safety_margin
  and candidate(t) == candidate(t + Δ)
  and no unresolved negation/entity/reordering risk
```

例如目标短语依赖的最后一个中文词在430 ms结束，安全余量为80 ms，那么最早正标签时间约为510 ms。

### 5.10.2 严格Bayesian实现

第一版建议使用可审计的显式 likelihood，而不是把一个神经网络 softmax直接叫作 Bayes：

```text
posterior odds
  = prior odds
  × p(e_acoustic | safe) / p(e_acoustic | unsafe)
  × p(e_translation | safe) / p(e_translation | unsafe)
  × p(e_boundary | safe) / p(e_boundary | unsafe)
  × p(e_history | safe) / p(e_history | unsafe)
```

对应 log 空间：

```text
log O_post = log O_prior + Σ_k log LR_k
P_safe = O_post / (1 + O_post)
```

各部分含义：

| Bayesian部分 | 实际含义 | 例子 |
|---|---|---|
| prior `p(z=1 | b,h)` | 在看到本tick证据前，根据模式、语言方向、等待时长判断“通常该不该写” | fast模式、已等待800 ms，prior较高 |
| likelihood `p(e | z)` | 如果现在确实安全/不安全，观察到当前声学与翻译证据的可能性 | GLM连续两tick不变在safe类中更常见 |
| posterior `p(z=1 | e,b,h)` | 综合先验和新证据后，现在不可回滚播放的安全概率 | posterior=0.92，balanced阈值0.88，可WRITE |

初始可对 entropy、margin、增长率使用 class-conditional Gaussian/Beta分布；离散边界特征使用 categorical
分布。若证据相关性太强，再升级为 Gaussian mixture、normalizing flow或条件 density-ratio estimator，
但最后仍输出显式 likelihood ratio并做校准。

### 5.10.3 Gate的输入证据

每个tick记录：

```text
acoustic:
  GLM token persistence、CTC blank probability、student entropy、teacher-agreement proxy

translation:
  target-capacity margin、Qwen WAIT/WRITE logit margin、draft在连续tick的一致率

boundary:
  静音、音节/词边界、标点概率、micro-chunk是否能自然结束

history:
  距离上次WRITE时间、最近错误/修订率、播放buffer、累计等待
```

### 5.10.4 Gate如何训练和校准

训练分三步：

1. 用 training split的 `safe/unsafe` 标签拟合 prior和各类 likelihood；
2. 在 validation split选择 posterior threshold，并做 temperature/isotonic calibration；
3. 在独立 dev上画 reliability diagram，验证 `P_safe=0.9` 的样本确实约90%安全。

决策还可以写成风险最小化：

```text
Cost(WRITE) = (1 - P_safe) × C_irreversible_error
Cost(WAIT)  = C_latency_per_tick
```

当 `Cost(WRITE) <= Cost(WAIT)` 时提交。fast/balanced/quality不是随意改 logit，而是使用不同的
`C_latency/C_error` 比率和先验。

推理时使用迟滞避免 WAIT/WRITE来回抖动：

```text
posterior >= τ_write 连续2个tick → WRITE
posterior <  τ_draft             → 不生成draft
τ_draft <= posterior < τ_write   → 只在内部生成draft，不播放
```

## 5.11 Micro-WRITE的设计、训练与推理

### 5.11.1 原理

当前一次 WRITE生成较长文本和大量 semantic token，只有全部或很大一部分生成后 codec才能工作。
Micro-WRITE把一次大提交拆成可流水的短事务：

```text
传统WRITE： [完整短语文本 + 100--700 semantic] → decode → play
Micro-WRITE：[1--4词 + 8--16 semantic] → decode/play
             同时生成下一块semantic
```

若 BiCodec semantic rate约50 Hz，8--16 token对应约160--320 ms语音。这不是简单在推理阶段硬截断，
而是让模型在训练中学习短块的边界、连续性以及什么时候结束一个 micro-WRITE。

### 5.11.2 训练数据如何制作

对目标音频执行 BiCodec，得到 semantic序列，再利用目标词时间戳把它切为小块：

```text
目标文本：Tomorrow morning | at nine | I'm going | to Beijing | for a meeting
semantic：s001...s012     | s013...s024 | s025...s036 | ...
```

为了保持 Phase3词表和 checkpoint兼容，第一版不必增加大量新 token，可以重复使用现有协议：

```text
START_GLM <新增稳定源token> END_GLM
WRITE_GENERATE
START_CONTENT Tomorrow morning END_CONTENT
START_SEMANTIC s001 ... s012 END_SEMANTIC

START_GLM <下一批源token> END_GLM
WRITE_GENERATE
START_CONTENT at nine END_CONTENT
START_SEMANTIC s013 ... s024 END_SEMANTIC
```

每个 target micro-chunk还保存：

- 对应目标词范围；
- semantic start/end index；
- `support_end_ms`；
- 是否自然词/短语边界；
- 与前一块的 codec overlap状态；
- final flush标志。

### 5.11.3 Qwen训练方式

初始化使用当前效果最好的 Phase3 checkpoint，不从随机权重开始。推荐分两步：

1. 冻结 Causal Student和 codec，只训练 Qwen的 action、目标文本和 semantic生成；
2. 加入20%--40%原 Phase3 replay，防止短块训练造成翻译质量和语音能力遗忘。

损失可以写为：

```text
L_micro =
    1.0 L_action
  + 1.0 L_target_text
  + 1.0 L_semantic
  + 0.2 L_chunk_boundary
  + 0.1 L_duration
  + λ_replay L_phase3_replay
```

其中 `L_duration` 约束文本长度和 semantic时长合理，`L_chunk_boundary` 惩罚把词切在不自然的位置。
训练初期使用 teacher forcing；稳定后加入 scheduled sampling，让模型也看到自己上一块生成的 semantic
历史。最后可以用 latency-constrained GRPO优化不可微的延迟、质量和播放连续性指标。

### 5.11.4 在线推理

Qwen会话只建立一次：

```python
qwen_state = qwen.start_session(system_prompt, speaker_token)
qwen_state.append_source(stable_glm_delta)

if safe_commit.allow_write(candidate):
    stream = qwen_state.generate_micro_write(
        max_text_tokens=8,
        min_semantic_tokens=8,
        max_semantic_tokens=16,
    )
    for semantic_delta in stream:
        codec.push(semantic_delta)
```

Qwen必须复用 `past_key_values`。已经提交的 source、target text和 semantic历史留在会话 cache中；每个tick
只 prefill新增 GLM token。codec收到最少一小块 semantic后立即解码，不等待下一个 micro-WRITE。

为避免短块之间出现“咔哒声”或音色跳变，需要：

- 固定整个会话的 speaker token；
- 保留 codec hidden/cache；
- 相邻块使用30--50 ms overlap-add；
- 对边界做 waveform/STFT continuity训练；
- 浏览器使用小而稳定的 jitter buffer，而不是每块创建新播放器。

## 5.12 哪些模块训练、冻结以及训练顺序

| 模块 | 初始化 | 第一阶段 | 后续阶段 | 是否进入在线推理 |
|---|---|---:|---:|---:|
| 原 WhisperVQ | 当前checkpoint | 冻结，生成teacher target | 始终只读 | 否，student通过后移除 |
| Whisper ASR/WhisperX | 公开checkpoint | 冻结，做时间戳/评价 | 冻结 | 否 |
| Causal Audio Student v2 | bootstrap或新建 | 训练 | 可与policy小学习率联合微调 | 是 |
| Bayesian likelihood/prior | 统计初始化 | 训练并校准 | 可随新数据重新校准 | 是 |
| Phase3 Qwen | 最佳Phase3 checkpoint | 先冻结或仅训action head | micro-WRITE SFT，再GRPO | 是 |
| 原 BiCodec | 当前checkpoint | 冻结 | 作为codec teacher | 短期是 |
| Causal BiCodec Student | 可选新模型 | 暂不训练 | E5以后蒸馏 | 中期替换原decoder |

推荐严格按以下顺序，避免所有模块同时变化导致无法定位问题：

```text
Step A：冻结WhisperVQ，离线缓存teacher GLM/hidden和精细时间戳
Step B：只训练Causal Student，Qwen和BiCodec不动
Step C：冻结Student，训练并校准Bayesian Safe-Commit
Step D：从最佳Phase3初始化，做micro-WRITE SFT并加入Phase3 replay
Step E：串联Student + Gate + Qwen + 原BiCodec做端到端流式评估
Step F：只在E通过后训练causal codec student
Step G：小学习率联合refinement或latency-constrained GRPO
```

### 5.12.1 阶段A的性质：它是数据构建，不是模型训练

阶段 A 不更新 Causal Student、Qwen或 BiCodec权重。它负责把当前 UniST token数据转换成后续 B--G
阶段可以使用的、带真实时间关系的 streaming supervision：

```text
UniST token记录
  → 重建source/target waveform
  → 冻结WhisperVQ teacher标签
  → source词/字时间戳
  → target词与semantic时间戳
  → bilingual support alignment
  → safe/unsafe标签
  → micro-WRITE训练事件
  → 独立subsecond manifest
```

因此阶段 A 的“checkpoint”不是模型参数，而是版本化 manifest、索引、统计、哈希和
`STAGE_A_COMPLETE.json`。阶段 A 一旦正确完成，后续不同 Student/Gate/Qwen实验可以重复使用它，
无需每次重新跑 Whisper、BiCodec和对齐模型。

### 5.12.2 当前可用的阶段A输入数据

当前项目实际数据如下：

| 用途 | 当前路径 | 已核对规模 | 阶段A用途 |
|---|---|---:|---|
| 15-shard pilot train | `data/raw/UniST/train-00000.parquet` 至 `train-00014.parquet` | 1,500,000条记录 | 先跑通并比较160/240/320 ms前端 |
| full198 train | `data/raw/UniST/train-00000.parquet` 至 `train-00197.parquet` | 19,286,004条记录 | pilot通过后的正式训练 |
| validation | `data/raw/UniST/dev-00000.parquet` | 当前已处理7,965条 schedule | 模型选择、Bayesian校准和端到端dev |
| final test | `data/raw/UniST/test-00000.parquet` | 独立保留 | 所有阈值冻结后才运行 |
| 当前full198 pseudo数据 | `data/processed/simul_uniss_v3_full198/` | 206,100,650个640 ms pseudo事件 | 只复用ID/token，不把比例边界当真值 |
| 当前音频重建bootstrap | `data/processed/simul_uniss_v2_15shard/stage00_audio_reconstruction/` | 1,000条 | 仅代码smoke，不等于15-shard正式阶段A |

当前 full198 已处理数据中的每条 schedule包含：

- `id`；
- `src_lang`、`tgt_lang`；
- `transcription`、`translation`；
- `source_glm`或其长度；
- `source_bicodec`、`target_bicodec`；
- `speaker_tokens/bicodec_global`；
- 当前640 ms比例式 WAIT/WRITE事件。

其中可以直接复用的是真实样本 ID、文本、token和语言方向。不能直接复用为正式监督的是：

- 按字符或token比例截断得到的 `source_ctc_count_proxy`；
- `target_ctc_count_proxy`；
- `pseudo_proportional_token_alignment`；
- 由固定640 ms和 wait-k构造的 safe/unsafe结论。

阶段 A 必须从音频时间戳和双语依赖重新生成这些标签。

### 5.12.3 UniST只有token时，音频从哪里来

当前 UniST记录主要提供 BiCodec/GLM token，不应假设每条记录都带原始 WAV。阶段 A 首先使用：

```text
source_bicodec → BiCodec decoder → source FLAC
target_bicodec → BiCodec decoder → target FLAC
bicodec_global → 会话speaker/global条件
```

输出统一为16 kHz、mono、无 VAD trim的 FLAC。不能删除句首/句尾静音，因为这会改变真实 streaming
时间戳和延迟指标。

需要明确一个域差异风险：Student如果只在 BiCodec重建音频上训练，真实麦克风没有 codec重建伪影，
两者分布并不完全相同。推荐分两步处理：

1. pilot先用重建 UniST音频验证架构和低延迟能力；
2. 正式模型增加真实波形的 source-only蒸馏数据或麦克风增强数据，只训练
   `L_glm_ctc/L_hidden/L_source/L_stable`，不伪造目标翻译。

阶段 A 应保存 `audio_origin={reconstructed,original,augmented}`，后续评估必须分桶报告，避免重建音频
上的好结果掩盖真实麦克风域退化。

### 5.12.4 训练、校准和测试如何切分

为了防止泄漏，建议固定：

```text
train shards:
  Student训练
  Bayesian likelihood/prior拟合
  Micro-WRITE SFT

dev_calibration:
  posterior temperature/isotonic calibration
  fast/balanced/quality阈值选择

dev_selection:
  checkpoint选择和消融

test:
  冻结模型、阈值和配置后只运行一次正式结果
```

`dev_calibration`和`dev_selection`从 `dev-00000.parquet`按样本 ID哈希确定性划分，不能按文件前后顺序
切分。所有方向都要按 `src_lang→tgt_lang`分层，防止某一方向集中在单一 split。test不能用于选择
Bayesian阈值、chunk size或 micro-WRITE长度。

### 5.12.5 阶段A的完整处理步骤

#### A0：冻结输入和模型身份

记录并校验：

- 15个或198个 Parquet的路径、大小和 SHA256；
- WhisperVQ teacher checkpoint和配置哈希；
- BiCodec checkpoint和配置哈希；
- Phase3 checkpoint iteration；
- 对齐模型、tokenizer和代码 commit；
- 数据 split seed和语言方向。

若任一输入变化，必须产生新的 Stage A版本号，不能覆盖旧目录。

#### A1：抽取原始字段并做基础过滤

从 Parquet抽取：

```text
id
src_lang / tgt_lang
transcription / translation
source_glm
source_bicodec
target_bicodec
bicodec_global / speaker token
source shard和row index
```

拒绝空文本、空token、非法语言方向、异常token范围和重复 ID。此时只做确定性检查，不根据模型质量
过滤“困难样本”。

#### A2：重建source和target音频

使用冻结 BiCodec decoder分布式重建：

```text
GPU rank r处理 record_index % world_size == r
```

每条记录保存：

```json
{
  "source_audio": ".../source/<id>.flac",
  "target_audio": ".../target/<id>.flac",
  "sample_rate": 16000,
  "source_duration_ms": 2140,
  "target_duration_ms": 1980,
  "audio_origin": "reconstructed",
  "decode_status": "ok"
}
```

目标 semantic已经来自 `target_bicodec`，原则上不必重新编码；但需随机抽样执行
`target token → waveform → token`一致性审计。

#### A3：建立冻结WhisperVQ teacher目标

优先复用 UniST已经提供的 `source_glm`作为硬 CTC标签。为了训练 hidden distillation和置信度，还可以
把重建 source音频送入冻结 WhisperVQ，缓存：

```text
teacher_glm_logits（可选top-k）
selected_teacher_hidden（建议只存1--3个选定层）
teacher_token_time_proxy
teacher/released_source_glm agreement
```

重建音频经过有损 codec，重新编码得到的 GLM序列不保证与数据内 `source_glm`完全一致。只有 agreement
达到预设阈值的样本才使用 hidden蒸馏；否则仍保留原 `source_glm`做 CTC，不能强行把两个冲突目标混合。

teacher在此阶段始终 `eval()`、`torch.no_grad()`且参数冻结。它不会被阶段 A训练。

#### A4：得到source真实时间戳

因为已有 transcription，优先使用 forced alignment，而不是完全自由 ASR：

- 中文：FunASR/Paraformer timestamp，再按字/词校正；
- 英文：WhisperX或 MFA word alignment；
- 低置信度样本用第二 aligner复核；
- 保存 word/character start、end、confidence和静音区间。

输出示例：

```json
{
  "source_words": [
    {"text": "我", "start_ms": 40, "end_ms": 130, "confidence": 0.98},
    {"text": "明天", "start_ms": 180, "end_ms": 390, "confidence": 0.96},
    {"text": "上午", "start_ms": 410, "end_ms": 520, "confidence": 0.94}
  ]
}
```

#### A5：得到target文本与semantic时间戳

对 target FLAC做目标词 forced alignment，然后把约50 Hz的 BiCodec semantic index映射到时间：

```text
semantic_start_ms(k) ≈ 20 × k
semantic_end_ms(k)   ≈ 20 × (k + 1)
```

最终得到每个目标词覆盖的 semantic范围。若词边界落在 semantic token中间，边界向外扩展，不切断词的
声学主体。

#### A6：计算bilingual target-support alignment

对 source text和 target text执行双语 word/phrase alignment，并结合：

- source真实词结束时间；
- 目标依存/短语边界；
- 翻译模型 attention或 contextual alignment；
- 否定词、数字、实体和语言方向重排规则。

对于目标 token或短语 `m_j`：

```text
support_end_ms(m_j)
  = max(end_ms(source_word_i))
    for all source_word_i required by m_j
```

如果一个目标短语没有可靠 source支持、不同 aligner严重冲突，标记为 uncertain；不能为了让延迟更好看
而强行给它较早 support时间。

#### A7：生成safe/unsafe和future-revision标签

在160 ms tick上构造候选状态：

```text
t = 160, 320, 480, 640, ...
right_context ∈ {0, 40, 80, 160} ms
```

safe标签至少同时满足：

```text
support_end_ms(m_j) <= t - safety_margin
candidate(t) == candidate(t + 160/320 ms)
没有未解决的否定、实体或长距离重排
```

未来音频只用于离线生成标签，绝不能作为 Student或 Gate在推理时的输入。

full198有约1,928万条记录，如果把每个tick全部展开成 JSON，会产生极大的重复数据。正式实现应保存
utterance-level compact alignment，并在 Dataset中按 seed动态采样 prefix；只把固定 dev/test事件
完全展开，保证评估可重复。

#### A8：生成Micro-WRITE监督

将目标文本、target semantic和 support time联合切分：

```text
micro text：通常1--4词
semantic：目标8--16 token
例外上限：24 token，避免切断一个长词或不自然音节
```

每个事件保存：

```json
{
  "text": "Tomorrow morning",
  "text_token_start": 0,
  "text_token_end": 2,
  "semantic_start": 0,
  "semantic_end": 12,
  "support_end_ms": 520,
  "earliest_safe_ms": 600,
  "natural_boundary": true,
  "final_flush": false
}
```

#### A9：质量过滤、索引和完成标记

检查：

- source/target音频能否完整解码；
- token rate和音频时长是否异常；
- source forced-alignment覆盖率；
- bilingual alignment覆盖率；
- `support_end_ms`是否单调；
- micro-chunk是否覆盖全部目标 semantic且无重叠丢失；
- semantic unique ratio和最大重复 run；
- train/dev/test ID是否有交集；
- 每个语言方向的样本数和时长。

通过后生成 binary offset index、统计报告和 `STAGE_A_COMPLETE.json`。若未生成完成标记，B阶段自动
启动器必须拒绝运行。

### 5.12.6 阶段A建议的独立目录

不能写回当前 `simul_uniss_v1/v2/v3`或 Phase1--3目录。建议：

```text
data/processed/simul_uniss_subsecond_v1/
  pilot_15shard/
    stage_a/
      manifests/
      source_audio/
      target_audio/
      teacher/
      source_alignment/
      target_alignment/
      support_alignment/
      micro_write/
      indexes/
      rejects/
      statistics.json
      STAGE_A_COMPLETE.json
  full198/
    stage_a/
      ...
```

建议的主 manifest：

```json
{
  "id": "sample-id",
  "split": "train",
  "src_lang": "cmn",
  "tgt_lang": "eng",
  "source_audio": "...flac",
  "target_audio": "...flac",
  "audio_origin": "reconstructed",
  "transcription": "...",
  "translation": "...",
  "source_glm": [1, 2, 3],
  "teacher_hidden_ref": "...safetensors#row",
  "source_alignment_ref": "...jsonl#offset",
  "target_alignment_ref": "...jsonl#offset",
  "support_alignment_ref": "...jsonl#offset",
  "target_semantic": [10, 20, 30],
  "speaker_tokens": [100, 101],
  "micro_write_ref": "...jsonl#offset",
  "quality_flags": []
}
```

大数组应存二进制 shard或 safetensors并在 JSONL中保存 offset，不能为1,928万条样本各自创建大量小
文件，否则 inode和随机读取会成为瓶颈。

### 5.12.7 阶段A详细例子

假设原记录为：

```text
源：我明天上午九点去北京开会。
目标：I'm going to Beijing for a meeting at nine tomorrow morning.
```

source alignment得到：

| source词 | 结束时间 |
|---|---:|
| 我 | 130 ms |
| 明天 | 390 ms |
| 上午 | 520 ms |
| 九点 | 830 ms |
| 北京 | 1320 ms |
| 开会 | 1810 ms |

双语 alignment发现目标语序发生了重排：

| target micro-phrase | 依赖source | support end | 加80 ms安全余量后的最早提交 |
|---|---|---:|---:|
| Tomorrow morning | 明天、上午 | 520 ms | 600 ms |
| at nine | 九点 | 830 ms | 910 ms |
| I'm going to Beijing | 我、去、北京 | 1320 ms | 1400 ms |
| for a meeting | 开会 | 1810 ms | 1890 ms |

因此在 Stage A生成的训练状态中：

```text
t=320 ms：Tomorrow morning → unsafe/WAIT
t=480 ms：可生成internal draft，但证据未完整 → unsafe
t=640 ms：support和安全余量均满足 → safe/WRITE
t=800 ms：at nine仍缺少完整“九点”证据 → unsafe
t=960 ms：at nine → safe/WRITE
```

目标音频对应的 semantic再被切为：

```text
Tomorrow morning → s001...s012
at nine          → s013...s022
I'm going        → s023...s034
to Beijing       → s035...s047
for a meeting    → s048...s063
```

B阶段使用 source audio和 teacher GLM训练 Student；C阶段使用上述 safe/unsafe状态训练 Bayes；D阶段
使用文本和 semantic小块训练 Qwen。三种监督都来自同一条 Stage A记录，但各自优化不同模块。

### 5.12.8 阶段A验收条件和预计时间

建议的 pilot gate：

```text
Parquet/record hash完整：100%
音频decode成功率：>= 99%
source alignment覆盖率：>= 95%
高置信bilingual support覆盖率：>= 90%
train/dev/test ID交集：0
micro-write semantic覆盖：100%，无丢失/重复
每个语言方向人工抽查：>= 100条
```

时间取决于是否缓存 teacher hidden：

| Stage A模式 | 15-shard | full198 |
|---|---:|---:|
| smoke 1,000条 | 0.5--2小时 | 不适用 |
| fast：复用source GLM，不保存大hidden | 8--24小时 | 2--5天 |
| formal：完整音频、forced alignment、support alignment和teacher hidden | 1--3天 | 5--12天 |

这是8张H200独占且音频/对齐并行后的工程估算。正式运行前应先用1,000条记录分别测 BiCodec decode、
WhisperVQ、source align和 bilingual align的 RTF，再按实际音频总时长更新 ETA。

### 5.12.9 阶段A如何衔接完整训练和推理

完整训练的数据依赖为：

```text
Stage A/source_audio + teacher_glm/hidden
  → Stage B训练Causal Student

Stage A/prefix + safe标签
  → Stage C拟合prior/likelihood并校准posterior

Stage A/micro_write + target_semantic
  → Stage D从最佳Phase3做Micro-WRITE SFT

Stage A/fixed dev/test streaming events
  → Stage E端到端评估

Stage A/target waveform + semantic boundary
  → Stage F可选Causal BiCodec蒸馏

Stage E结果和在线rollout
  → Stage G可选联合训练/GRPO
```

阶段 A 不进入部署推理。真实在线推理只加载：

```text
Causal Student checkpoint
Bayesian Gate参数和校准表
Micro-WRITE Qwen checkpoint
Streaming BiCodec/原BiCodec checkpoint
固定speaker token
```

不会加载：

- full WhisperVQ teacher；
- WhisperX/FunASR/MFA；
- bilingual aligner；
- target reference audio；
- safe/unsafe训练标签；
- 未来音频。

换句话说，Stage A利用完整训练样本和未来信息制作“老师答案”；部署时模型只依靠当前时刻已经收到的
PCM、cache和历史已提交内容执行因果推理。

### 5.12.10 阶段B：Causal Audio Student正式训练与推理规范

本节是在5.9节模型原理之上的可执行 Stage B规范。阶段 B **必须训练**，目标是得到一个可以替换约4秒
block WhisperVQ在线位置、但仍输出原 GLM token空间的真正因果音频前端。

#### B0：阶段目标、输入和输出

输入来自通过验收的阶段 A：

```text
source_audio
source_audio_origin
source_glm
teacher_glm_logits/top-k（可选）
teacher_hidden_ref（可选）
source word/character alignment
target support alignment
language direction
quality flags
```

阶段 B 不使用 target waveform生成语音，也不更新 Qwen和 BiCodec。主要输出为：

```text
causal_student checkpoint
online cache配置
CTC prefix decoder配置
stability calibration参数
first-stable-GLM评估
teacher agreement评估
Phase3兼容性评估
STAGE_B_COMPLETE.json
```

训练数据选择：

```text
pilot：
  Stage A pilot_15shard train
  1,500,000条记录池
  dev_calibration/dev_selection只用于评估

formal：
  Stage A full198 train
  19,286,004条记录池
  按语言方向、时长和audio_origin全局shuffle

test：
  阶段B完全不读取
```

15-shard用于确认架构、损失和cache正确，不能作为最终泛化结论。pilot通过后，正式 checkpoint应使用
full198数据池训练。

#### B1：哪些参数训练，哪些参数冻结

| 组件 | Stage B状态 | 原因 |
|---|---:|---|
| 原 WhisperVQ teacher | 冻结 | 保持历史Phase1--3可复现，提供完整上下文teacher |
| Whisper ASR/forced aligner | 冻结且离线 | 只提供Stage A时间戳 |
| Causal convolution/subsampling | 训练 | 学习无未来帧的局部声学表示 |
| Chunk-Conformer/Emformer/Transformer | 训练 | 学习有限右上下文和历史cache |
| GLM CTC head | 训练 | 对齐原WhisperVQ GLM codebook |
| Source CTC head | 训练 | 提供可解释的源内容证据 |
| Target-capacity head | 训练 | 估计当前源前缀最多支持多少目标内容 |
| Stability/confidence head | 训练 | 判断候选GLM token是否可提交 |
| Phase3 Qwen | 冻结 | Stage B只测兼容性，不在此阶段改变翻译模型 |
| BiCodec | 冻结 | Stage B不负责目标音频生成 |

原 WhisperVQ checkpoint和代码保持只读；Stage B所有 checkpoint写入新的
`checkpoints/simul_uniss_subsecond_v1/stage_b_*`目录。

#### B2：正式网络结构

输入特征建议与当前 bootstrap保持兼容：

```text
sample rate = 16 kHz
window = 25 ms / 400 samples
hop = 10 ms / 160 samples
n_mels = 128
center = False
```

推荐正式结构：

```text
waveform
  → online log-Mel
  → left-only causal conv
  → causal subsampling ×4
  → 每40 ms一个hidden step
  → 12--16层Chunk-Conformer/Emformer
     left memory = 2--4 s
     current chunk = 160 ms
     right context = 0--80 ms（训练时可包含160 ms curriculum）
     per-layer KV/cache
  → LayerNorm
  ├─ GLM CTC head
  ├─ Source CTC head
  ├─ Target-capacity CTC/count head
  └─ Stability/entropy head
```

当前 bootstrap使用三次 stride=2，约80 ms一步。正式亚秒版本优先比较：

| 版本 | hidden/layers | 输出时间分辨率 | 用途 |
|---|---|---:|---|
| Student-S | 512 / 12 | 40--80 ms | 快速pilot和低计算量基线 |
| Student-B | 768 / 12--16 | 40 ms | 正式质量候选 |
| Streaming-WhisperVQ clone | 原teacher尺寸 | 由chunk设置决定 | E2b兼容性对照，不作为首选 |

不应为了GPU占用直接沿用 Qwen的 `sequence_length=18000`。Stage B处理的是声学帧，应使用
`total_audio_frames/total_audio_seconds per GPU`控制动态 batch，避免长音频padding浪费。

#### B3：因果attention和cache的严格定义

在输出时刻 `t`，每层只允许访问：

```text
历史cache：[max(0, t-left_memory), t]
有限右上下文：(t, t+right_context]
```

不能访问 `t+right_context`之后的任何帧。训练 attention mask、离线回放 mask和在线 cache必须由同一
工具函数生成，避免训练和部署的感受野不同。

每层在线状态至少包含：

```text
conv_left_state
attention_key_cache
attention_value_cache
position_offset
valid_cache_length
```

cache长度必须有上限。超过2--4秒左记忆后删除最旧 K/V，使计算量和显存不会随着通话时长无限增长。

#### B4：训练损失和标签作用域

正式损失为：

```text
L_B =
    1.0 L_glm_ctc
  + 0.5 L_hidden_distill
  + 0.3 L_source_ctc
  + 0.4 L_target_capacity
  + 0.2 L_stability
  + 0.1 L_chunk_consistency
```

各损失的训练范围必须严格定义：

| 损失 | 目标 | 是否允许使用未来信息制作标签 | 推理时是否需要teacher |
|---|---|---:|---:|
| `L_glm_ctc` | Stage A提供的完整 `source_glm`或当前前缀可对齐部分 | 标签可离线生成 | 否 |
| `L_hidden_distill` | 对齐后的冻结teacher hidden | 是，只用于teacher target | 否 |
| `L_source_ctc` | 当前时刻已经结束的源词/字 | 是，用时间戳裁标签 | 否 |
| `L_target_capacity` | 当前源证据支持的目标prefix | 是，用support alignment | 否 |
| `L_stability` | token未来160/320 ms是否保持 | 是，用未来判断标签 | 否 |
| `L_chunk_consistency` | 同一音频不同chunk划分输出应一致 | 不需要未来内容，只改变分块 | 否 |

teacher hidden只能监督当前因果位置可以合理对应的部分。不能要求 Student在320 ms时复现 teacher依赖
整句未来才能形成的 hidden；这种位置应降低 hidden权重或只使用 CTC/稳定性监督。

#### B5：训练课程

不建议从第一步就随机混合所有任务。推荐：

```text
B0 overfit smoke：
  32--128条样本
  验证loss下降、CTC blank和cache实现

B1 GLM bootstrap：
  L_glm_ctc + L_source_ctc
  right context从320/160 ms开始

B2 causal distillation：
  加入L_hidden_distill
  right context逐渐降到160/80 ms

B3 prefix stability：
  加入随机前缀、L_stability和L_chunk_consistency
  chunk ∈ {160, 240, 320} ms
  right context ∈ {0, 40, 80, 160} ms

B4 target capacity：
  加入Stage A support alignment和L_target_capacity

B5 formal refinement：
  以80 ms右上下文为主
  保留少量0/40/160 ms增强
  加入真实麦克风噪声、混响、速度和重采样增强
```

课程从较大右上下文开始是为了先保持 teacher兼容，再逐步压缩延迟。最终 checkpoint必须在固定
`160 ms chunk + 80 ms right context`条件下重新评估，不能用训练时更大的160/320 ms lookahead报告
亚秒结果。

#### B6：一个Stage B训练iteration

假设抽到样本：

```text
我明天上午九点去北京开会。
```

本次随机条件为：

```text
prefix_end = 640 ms
right_context = 80 ms
visible_audio = 0--720 ms
chunk_size = 160 ms
audio_origin = reconstructed
```

iteration执行：

1. 从 Stage A索引读取 source FLAC、`source_glm`、时间戳和 support alignment；
2. 只解码到720 ms，720 ms之后波形不送入 Student；
3. online log-Mel以 `center=False`产生特征；
4. 按160 ms chunk模拟cache前向，而不是把720 ms当普通双向序列；
5. 根据 Stage A时间戳截取当前可监督 source GLM/source text/target capacity；
6. 从 teacher缓存取对应 token、top-k logits和可用 hidden，并对 teacher执行 `stop_gradient`；
7. Student输出四个 head和 hidden；
8. 计算 `L_B`，无有效标签的 head使用 mask，不用伪标签填充；
9. 8个 rank做 DDP gradient all-reduce；
10. gradient clip后 optimizer更新 Student参数；
11. TensorBoard记录总损失、各head损失、blank rate、agreement、cache长度和audio seconds/s；
12. 下一次看到同一句时可能抽到320 ms或960 ms前缀，监督标签会随证据到达而变化。

伪代码：

```python
record = dataset.sample()
t = sample_prefix_end(record)
r = sample_right_context()
wave = load_prefix(record.source_audio, end_ms=t + r)

student_out = student.forward_chunked(
    wave,
    chunk_ms=sample_chunk_size(),
    right_context_ms=r,
    use_cache_simulation=True,
)
targets = build_stage_b_targets(record, prefix_end_ms=t)
losses = stage_b_losses(student_out, targets)

optimizer.zero_grad(set_to_none=True)
losses["total"].backward()
clip_grad_norm_(student.parameters(), 1.0)
optimizer.step()
```

#### B7：8卡训练和初始超参数

建议用8张H200做单个正式 Stage B，而不是4个两卡小模型长期并行。初始配置：

```text
precision = bf16
optimizer = AdamW
Student-S learning rate = 2e-4
Student-B learning rate = 1e-4
weight decay = 0.01
warmup = 5%
schedule = cosine
gradient clip = 1.0
dropout = 0.1
DDP shuffle = true
duration/language bucket = true
```

batch按每卡累计音频秒数控制。先跑100 step吞吐扫描，在不 OOM且 padding效率足够高的情况下逐步增加
`max_audio_seconds_per_gpu`。DataLoader必须：

- 全局shuffle并设置固定 seed；
- 每个 epoch调用 `DistributedSampler.set_epoch()`；
- 按音频时长bucket，但不能把语言方向永久分开；
- 避免同一utterance的多个prefix同时集中在一个batch；
- 对重建音频和真实/增强音频设置可追踪采样比例；
- 使用足够 worker预取，但以 NVMe和CPU实测为准。

15-shard pilot建议训练到20k--50k optimizer steps或dev指标停止改善；full198的正式 step数必须在 Stage A
统计音频总时长和100-step吞吐后计算，不直接照搬 Qwen的4,753 iterations。

预计时间：

```text
15-shard formal pilot：约8--20小时
full198正式训练：约1--3天
```

该时间不包含 Stage A数据处理。

#### B8：验证、checkpoint选择和性能保护

checkpoint不能只按 `valid_total_loss`选择。每次 validation至少记录：

```text
GLM CTC loss
source CER/WER
teacher token agreement
first stable GLM p50/p90/p95
uncommitted revision rate
committed rollback rate
cache/full-causal parity
active RTF p50/p95
audio seconds/s
Phase3 Text-BLEU/COMET兼容性
```

建议的阶段 B pilot gate：

```text
first stable GLM p50 <= 400 ms
first stable GLM p95 <= 720 ms
teacher token agreement >= 90%
committed rollback = 0
active RTF p95 < 0.25
cache/full-causal token parity >= 99.9%
插入冻结Phase3后Text-BLEU下降 <= 2绝对点
COMET下降 <= 0.03
NaN/OOM/decode failure = 0
```

这些是建议门槛，最终应在固定 dev和置信区间上冻结。若 Student延迟达标但 Phase3质量下降过大，应：

1. 优先从 Student-S切换到 Student-B；
2. 把主 right context从40 ms提高到80 ms；
3. 增加 hidden/GLM蒸馏权重；
4. 检查重建音频与真实麦克风域差异；
5. 检查 Student token是否发生 codebook语义漂移；
6. 不要通过降低稳定阈值掩盖表示错误。

历史 offline Phase3不受影响。Stage B只增加新路径，比较方式必须是：

```text
Frozen original WhisperVQ → Frozen Phase3
vs.
Stage B Causal Student    → Frozen Phase3
```

除前端外其他推理配置完全相同。

#### B9：必须执行的真正因果性测试

仅有 causal mask不等于实现正确。需要四类测试：

**Future perturbation test**

给两个音频相同前640 ms、不同640 ms以后内容。对允许的80 ms lookahead之后，640 ms之前已输出 logits/
token必须相同。若改变未来音频会改变更早输出，说明存在未来泄漏。

**Cache parity test**

同一音频分别执行：

```text
一次性full causal forward
160 ms chunk incremental forward with cache
```

有效时间步 logits应在数值容差内一致，最终 CTC token序列应一致。

**Chunk-boundary invariance test**

同一音频用160、240和320 ms分块，在共同可见音频范围内，稳定 token应高度一致，不能只在训练分块
边界正确。

**Long-session bounded-memory test**

连续输入30--60分钟音频，确认：

- cache显存保持有界；
- 每个tick计算时间不随通话长度增长；
- position offset不溢出；
- CTC history不会无限复制；
- 没有越来越大的播放前端延迟。

#### B10：Stage B在线推理状态

Stage B部署时只负责“PCM → 稳定GLM delta + evidence”，状态如下：

```text
pcm_ring_buffer
mel_overlap_state
conv_left_state
per_layer_attention_kv
position_offset
ctc_prefix_beam
uncommitted_glm_tail
committed_glm_history
source_ctc_prefix
target_capacity_prefix
stability_history
```

会话初始化：

```python
frontend = CausalStudentRuntime.load(checkpoint, calibration)
state = frontend.start_session(
    chunk_ms=160,
    right_context_ms=80,
    left_memory_ms=3000,
)
```

每20 ms接收一次PCM：

```python
for pcm_20ms in microphone_stream:
    state.append_pcm(pcm_20ms)

    if not state.has_ready_chunk():
        continue

    output = state.forward_incremental()
    state.update_ctc_prefix(output)

    stable_delta = state.commit_source_tokens(
        min_stability_probability=0.90,
        min_persistence_ticks=2,
        min_ctc_margin=configured_margin,
    )

    emit({
        "stable_glm_delta": stable_delta,
        "source_ctc_prefix": state.source_prefix,
        "target_capacity": output.target_capacity,
        "frontend_entropy": output.entropy,
        "token_persistence": output.persistence,
        "audio_time_ms": state.received_audio_ms,
    })
```

这里的 `commit_source_tokens`不是阶段 C的目标语音 Safe-Commit。它只保证源 GLM token足够稳定，可以
追加到 Qwen上下文；是否播放目标语音仍由阶段 C决定。

#### B11：一个在线推理例子

源音频：

```text
我明天上午九点……
```

假设160 ms chunk和80 ms右上下文：

| 墙钟 | 当前收到语音 | Student候选 | stability | source commit |
|---:|---|---|---:|---|
| 240 ms | “我…” | `g1` | 0.62 | 无 |
| 400 ms | “我明…” | `g1 g2` | 0.84 | 无，未连续达阈值 |
| 560 ms | “我明天…” | `g1 g2 g3` | 0.93 | 提交 `g1 g2` |
| 720 ms | “我明天上午…” | `g1 g2 g3 g4` | 0.96 | 追加 `g3 g4` |
| 880 ms | “九点…” | `g5 g6` | 0.81 | 保留为未提交tail |
| 1040 ms | “九点去…” | `g5 g6 g7` | 0.94 | 追加 `g5 g6` |

560 ms时 Phase3已经可以收到第一批真实因果 GLM token，但此时 Stage B本身不会直接生成目标音频。
阶段 C读取 Student的 entropy、persistence、capacity和 Qwen证据，判断什么时候允许 Micro-WRITE。

#### B12：Stage B完成后可以运行的推理

阶段 B完成后可以立即做一个诊断链路：

```text
Causal Student
  → fixed wait-k或简单稳定阈值
  → 冻结Phase3 Qwen
  → 当前Streaming BiCodec
```

它可以回答：

- 前端是否已经从约4.2秒降到亚秒级稳定token；
- Student token是否与 Phase3兼容；
- cache和真实麦克风输入是否正确；
- 前端替换后翻译质量下降多少。

但这还不是最终低于1秒的完整同传系统，因为：

- 没有阶段 C时无法可靠判断目标语音提交风险；
- 没有阶段 D时 Qwen仍可能一次生成很长的文本/semantic；
- 当前 codec和播放器可能继续增加数百毫秒；
- 固定 wait-k只能作为基线，不适应否定、实体和重排难度。

最终在线链路是：

```text
Stage B stable GLM/evidence
  → Stage C Bayesian Safe-Commit
  → Stage D Qwen KV cache + Micro-WRITE
  → Streaming BiCodec
  → browser playback
```

#### B13：Stage B建议目录、自动恢复和完成标记

建议新增而不覆盖现有 bootstrap：

```text
training/simul_uniss/subsecond_v1/
  causal_audio_student_v2.py
  chunk_attention.py
  streaming_cache.py
  stage_b_dataset.py
  stage_b_losses.py
  train_stage_b.py
  infer_stage_b.py
  test_causality.py

configs/experiments/simul_uniss_subsecond_v1/
  stage_b_pilot_15shard.env
  stage_b_full198.env

checkpoints/simul_uniss_subsecond_v1/
  stage_b_pilot_15shard/
  stage_b_full198/

runs/simul_uniss_subsecond_v1/
  stage_b_pilot_15shard/
  stage_b_full198/
```

checkpoint应包含：

```text
model
optimizer
lr_scheduler
grad_scaler（如使用）
global_step
consumed_audio_seconds
sampler_epoch/state
RNG states
architecture config
Stage A manifest hash
teacher checkpoint hash
best dev metrics
```

自动启动 Stage C前必须同时存在：

```text
best.pt或Megatron等价checkpoint
stage_b_metrics.json
causality_test_passed.json
cache_parity_passed.json
STAGE_B_COMPLETE.json
```

若训练进程中断，从最后 checkpoint恢复 sampler和 RNG，不能从头重复相同前缀顺序，也不能仅加载模型
权重而丢失 optimizer/scheduler状态。

这里的“联合”不等于一开始就让梯度穿过所有离散 token。先用各模块可监督损失稳定训练；最终再通过
scheduled sampling、policy gradient或可微 surrogate处理 WAIT/WRITE和离散 semantic决策。

## 5.13 一个训练iteration如何实际执行

仍以中文到英文为例：

```text
源：我明天上午九点去北京开会。
目标：I'm going to Beijing for a meeting at nine tomorrow morning.
```

### 离线预处理一次

1. 完整 WhisperVQ读取整句，缓存 teacher GLM token和选定层 hidden；
2. source aligner得到“我/明天/上午/九点/……”的真实结束时间；
3. bilingual aligner和目标结构模型得到每个英文词依赖的最晚源证据时间；
4. BiCodec编码目标音频，得到50 Hz左右的 target semantic token；
5. 将目标文本和 semantic切成多个8--16 token micro-chunk；
6. 写入独立 subsecond manifest，不修改原 UniST manifest。

### 某一次训练iteration

假设本轮随机抽到 `t=640 ms`、右上下文 `80 ms`：

1. student只能读取 `0--720 ms`源音频，不能读取后面的“九点去北京开会”；
2. student输出 GLM CTC、source CTC、target capacity和 stability；
3. 从缓存teacher中取出时间上已经可以对齐的 token/hidden，计算 `L_student`；
4. 数据标签显示“Tomorrow morning”的支持证据已经到达，`z=1`；
5. Gate若预测 `P_safe=0.72`，而标签为安全，则 Bayesian NLL/校准损失推动它提高后验；
6. Qwen输入当前稳定 GLM prefix，监督输出一个 micro-WRITE：

```text
CONTENT: Tomorrow morning
SEMANTIC: s001 ... s012
```

7. 计算 action/text/semantic/boundary损失；
8. 当前阶段若 codec冻结，则不反传 codec；若处于 codec蒸馏阶段，再对目标 PCM计算 STFT和边界损失；
9. optimizer更新当前阶段允许训练的参数；
10. 下一 iteration可能仍是同一句，但切点变成320 ms，此时正确 action应为 WAIT或 internal draft。

因此，一条3秒音频可以贡献十几个不同“看到了多少源音频”的训练状态。模型真正学到的是随时间增长的
证据变化，而不是把整句音频映射到整句翻译。

### 一个建议的优化器更新边界

```text
Frontend阶段：
  backward(L_student) → update(student only)

Gate阶段：
  fit likelihood/prior + calibration → update(gate only)

Micro-WRITE阶段：
  backward(L_micro + replay) → update(Qwen/action only)

Joint阶段：
  backward(L_student + L_micro + surrogate_latency)
  → student使用0.1×基础学习率
  → Qwen使用0.05×基础学习率
  → gate保持校准参数或极小学习率
```

## 5.14 一次真实在线推理如何执行

假设用户已经在会话开始前选择固定目标音色。固定音色是亚秒目标的重要条件：如果系统必须先从当前
讲话人的前3秒提取音色，就不可能在1秒内用该音色播放。需要源音色克隆时，应使用会话前 enrollment，
或先使用固定音色，不能把3秒音色收集时间隐藏在指标之外。

一次在线会话的状态变化如下：

| 墙钟 | 系统操作 | cache/后验变化 | 用户听到什么 |
|---:|---|---|---|
| 0--160 ms | 收集首个chunk | Mel/conv cache建立 | 暂无目标音频 |
| 160--240 ms | 等80 ms右上下文并运行student | GLM仍不稳定 | 暂无 |
| 240--400 ms | 第二个chunk增量前向 | 提交少量source GLM，Gate仍WAIT | 暂无 |
| 400--560 ms | 第三个chunk增量前向 | Qwen生成内部draft，不播放 | 暂无 |
| 560--640 ms | 新证据到达 | `P_safe`从0.74升到0.91，触发WRITE | 暂无 |
| 640--740 ms | Qwen用KV cache生成12个semantic | codec边收边解码 | 暂无 |
| 740--840 ms | PCM进入网络和jitter buffer | 播放队列达到安全水位 | 开始听到目标语音 |
| 840 ms以后 | student/Qwen/codec三段并行 | 一边听上一块，一边生成下一块 | 连续同传 |

关键约束是：已经发送到浏览器的 PCM永远不回滚。内部 draft可以修改，未播放的 jitter buffer在严格
规则下可以丢弃，但已播放语音不允许“事后纠正”。评价 First Audio时必须以用户设备实际播放的第一个
有用目标音频为准，不能用模型产生第一个 token的时间代替。

### 推理状态机伪代码

```python
session = start_session(fixed_target_voice)

while session.is_open:
    pcm = receive_pcm_20ms()
    frontend_out = session.frontend.push(pcm)

    if frontend_out.has_stable_glm:
        session.qwen.append_source(frontend_out.stable_glm_delta)

    evidence = collect_evidence(
        frontend=frontend_out,
        qwen=session.qwen,
        playback=session.playback,
    )
    p_safe = session.gate.posterior(evidence, latency_mode="balanced")

    if p_safe >= session.write_threshold_for_two_ticks:
        for semantic_delta in session.qwen.generate_micro_write():
            pcm_delta = session.codec.push(semantic_delta)
            if pcm_delta.is_stable:
                session.playback.enqueue(pcm_delta)
    else:
        session.qwen.maybe_update_internal_draft()
```

三个模块不是串行等待整段完成：student处理第 `n+1` 个音频块时，Qwen可以生成第 `n` 个目标块，codec
同时解码第 `n-1` 个 semantic块。只有形成这种流水，低于1秒首包和长时间不累积延迟才能同时成立。

## 5.15 方法创新性与论文价值判断

需要客观地区分“已知组件”和“组合后的研究贡献”：

| 模块 | 单独看创新性 | 可能形成的当前项目贡献 |
|---|---|---|
| Causal Student蒸馏 | 中等；chunk encoder、蒸馏和CTC已有先例 | 在不改变UniSS Phase3 GLM词表的前提下，把4秒WhisperVQ teacher压成亚秒在线tokenizer |
| Bayesian Safe-Commit | 简单阈值本身较低 | 显式prior/likelihood/posterior、概率校准、不可回滚语音风险和语言重排证据联合建模可达到中高创新性 |
| Micro-WRITE | 短块生成思想已有相关工作 | 同时联合目标文本、BiCodec semantic、codec连续性和真实播放时延，具有中等到中高系统创新性 |
| 三者端到端结合 | 单个组件不是全新 | 如果证明质量接近offline且真实设备首音频低于1秒，整体贡献较强 |

更有研究价值的创新点可以进一步设计为：

1. **Future-Revision Bayesian Likelihood**：用“再听160/320/640 ms后草稿是否改变”直接定义 unsafe
   likelihood，把未来修订风险转成可校准概率；
2. **Uncertainty Propagation**：不只把离散 GLM token交给 Qwen，还把student token posterior/entropy
   压缩后传给 Gate，使声学不确定性真正影响翻译提交；
3. **Risk-budgeted Micro-WRITE**：posterior越高可以写更长块，posterior刚过阈值只写8 token，从而
   动态控制一次不可回滚暴露的风险；
4. **Counterfactual WAIT/WRITE Training**：对同一前缀同时模拟“现在写”和“再等一块”，计算质量增益
   与真实延迟代价，训练决策器选择期望效用更高的动作；
5. **Language-direction Bayesian Prior**：根据语言对的典型重排距离学习 prior，例如中英时间/地点短语
   重排比同语序片段更保守，但 prior只能影响等待倾向，不能绕过实际 likelihood证据。

若只实现“因果encoder + 固定wait-k + 截断semantic”，更像可靠工程实现；若实现严格校准的未来修订
posterior、动态 micro-WRITE长度，并在 computation-aware latency和offline质量上做完整消融，则更可能
形成清晰的研究创新。最终创新性仍需通过文献检索、消融实验和同行评审确认，不能仅凭模块命名判断。

## 6. 详细例子：如何在1秒内开始翻译

源句：

```text
我明天上午九点去北京开会。
```

目标：

```text
I'm going to Beijing for a meeting at nine tomorrow morning.
```

假设160 ms chunk、80 ms右上下文、固定目标音色：

| 墙钟 | 新源信息 | student/CTC状态 | controller | 目标输出 |
|---:|---|---|---|---|
| 0--160 ms | “我…”声学片段 | token不稳定 | WAIT | 无 |
| 160--320 ms | “我明…” | source增长，target capacity低 | WAIT | 无 |
| 320--480 ms | “我明天…” | 支持“tomorrow”概念，但主语结构未稳 | 草稿、不播放 | 内部draft |
| 480--640 ms | “我明天上午…” | posterior safe=0.91 | WRITE | 文本“Tomorrow morning” |
| 640--760 ms | Qwen micro-WRITE | 12 semantic token | codec解码 | 约240 ms目标语音 |
| 760--840 ms | 浏览器buffer | PCM到达 | PLAY | 右声道开始说“Tomorrow morning…” |
| 800--1120 ms | “九点…” | capacity继续增长 | WRITE | “at nine” |
| 1120--1600 ms | “去北京…” | 支持目的地 | WRITE | “I'm going to Beijing” |
| 后续 | “开会” | final flush | WRITE | “for a meeting” |

这时：

```text
First WRITE NCA ≈ 640 ms
First Audio CA  ≈ 840 ms
```

用户在源句尚未说到“九点”时已经听到目标句开头。

### 为什么不能所有句子都一样早

源前缀：

```text
他没有……
```

如果只听到“他”，模型可能草稿为“He…”。听到“没有”后才知道是否要说：

```text
He didn't...
There isn't...
He has not...
```

系统可以在安全 buffer 中生成 draft，但在否定证据到达前不应播放不可回滚语音。Bayesian gate 的
作用不是让所有句子强制早写，而是简单前缀早写、歧义前缀合理等待。

## 7. 备选方案

## 7.1 方案A：Streaming ASR + Incremental MT + Streaming TTS

流程：

```text
causal ASR partial text
→ incremental MT
→ stable target prefix
→ streaming TTS/BiCodec
```

Motivation：文本中间层易于做稳定性、词对齐和回滚 buffer，最快得到 <1秒工程 baseline。

优点：

- 调试最清楚；
- 可以直接计算 partial WER、prefix BLEU；
- 可使用成熟 streaming ASR/RNNT/Emformer；
- 低延迟成功概率最高。

缺点：

- 不再是纯 textless/unit-to-unit；
- ASR错误会传递；
- 韵律和speaker preservation较弱；
- 与当前 UniSS unified token generation 的研究故事不同。

推荐把它作为亚秒可行性下界 baseline，而不是最终主方案。

## 7.2 方案B：Direct Monotonic Speech-to-Unit Transducer

使用 RNN-T、Monotonic Chunkwise Attention、CIF 或 EMMA，直接从 causal audio hidden state 产生目标
text/unit。它天然支持80--160 ms step，理论上可达到300--700 ms。

优点：真正单遍、延迟低、cache简单。

缺点：需要重训较大模型；长距离翻译和开放域质量可能低于Qwen；难以复用现有Phase3能力。

适合作为独立论文分支，与主方案做 latency lower bound。

## 7.3 方案C：Speculative Speech Draft + Verification

创新性较强的方案：

1. 小 causal draft model 每160 ms预测下一个目标 text/unit；
2. 大 Qwen 使用新到 source prefix 验证 draft；
3. 浏览器保留160--240 ms未播放 safety buffer；
4. buffer内允许替换，真正播放后不可回滚；
5. verification通过的 draft 直接播放，不等大模型完整生成。

Motivation：将大模型的计算时间和语言学等待隐藏在很小的安全 buffer 内。

预期：在简单句上可把 Qwen生成部分从200 ms降到50--100 ms，但训练和一致性控制更复杂。

## 7.4 方案D：NAR semantic micro-chunk

借鉴 NAST-S2x：Qwen只生成目标文本/hidden plan，NAR head并行生成下一个8--16个 semantic token。

Motivation：当前 AR semantic 是 WRITE计算时间的主要可扩展部分；micro-chunk NAR 可显著减少每次
目标音频生成时间。

建议只在 profiling 证明 Qwen AR 部分阻塞亚秒目标时启用。当前 H200 计算快，源前端和策略更优先。

## 8. 推荐实验顺序

所有实验新建：

```text
experiments/simul_uniss_subsecond_v1/
checkpoints/simul_uniss_subsecond_v1/
runs/simul_uniss_subsecond_v1/
logs/simul_uniss_subsecond_v1/
eval_outputs/simul_uniss_subsecond_v1/
```

### E0：冻结当前基线

```text
模型：当前R2
前端：WhisperVQ prefix re-encode
chunk：640 ms
音色：3.2 s source voice
```

保存 p50/p95 First Audio CA、First WRITE、RTF、Speech-BLEU和双声道试听。

### E1：固定音色 + 现有前端

只移除3.2秒音色等待，测量 speaker extraction 对当前延迟的独立贡献。它不能突破4秒 frontend，
但能验证音色策略。

### E2：Causal Audio Student v2，固定 wait-k

```text
chunk = 160/320 ms
right context = 0/80 ms
policy = fixed wait-k 2/3
voice = fixed
```

先不接 learned action，确定前端能否在320--640 ms产生稳定 token。

### E3：Source/Target CTC Safe-Commit

比较：

```text
fixed wait-k
hard CTC gate
calibrated Bayesian gate
```

主问题：是否在不增加 premature WRITE 的情况下把p50 First WRITE降到640 ms以内。

### E4：Qwen KV-cache + micro-WRITE SFT

把输出粒度从长短语改为1--4词、8--16 semantic token，并测 action、text、semantic、codec的逐项墙钟。

### E5：Streaming BiCodec低holdback扫描

扫描：

```text
holdback = 2/3/5 token
overlap = 30/40/80 ms
left context = 25/50 token
```

选择 boundary click通过且首包最低的点。

### E6：Latency-Constrained GRPO / Stage7B

SFT已经探索早期WRITE后再做RL。奖励建议：

```text
R =
    1.0 * R_final_quality
  + 0.7 * R_prefix_quality
  + 0.5 * R_coverage
  - 0.8 * max(0, FirstAudioCA - 1.0s)
  - 0.4 * ATD
  - 1.2 * premature_WRITE
  - 1.0 * under_translation
  - 2.0 * semantic_collapse
  - 0.5 * codec_underrun
```

不能只奖励 First WRITE，否则模型会提前说一个词后长期WAIT。

### E7：Speculative draft（可选创新实验）

只在 E4/E6 已接近1秒但仍受 Qwen墙钟影响时进行。

## 9. 训练数据与算力建议

### 9.1 15-shard快速闭环

用当前相同15 shard：

- 精细对齐小集；
- 训练 frontend student；
- 跑通 E2--E5；
- 每个重要 checkpoint 用固定200--1000条 dev sample；
- 目标是证明架构能达到亚秒，不作为最终质量结论。

### 9.2 full198正式训练

15 shard通过后再处理 full198：

- 精细 source timestamps；
- bilingual support alignment；
- 多 chunk/right-context 训练；
- full dev/test评估；
- 与当前R2、R3、offline Phase3做paired comparison。

### 9.3 8 GPU分配

建议单个正式 frontend 训练使用8卡，而不是每个2卡并行4个不充分实验。因果 student需要充分 batch
覆盖不同chunk/right-context。探索阶段可按以下分配：

```text
GPU0--3: 160 ms + 80 ms right-context
GPU4--7: 320 ms + 80 ms right-context
```

完成首轮后选优，再8卡正式训练。Qwen micro-WRITE/GRPO阶段沿用8卡 Megatron。

## 10. 推理实现

服务端必须是异步流水线：

```text
Thread/Stream A: Audio capture + resample
Thread/Stream B: causal frontend cache update
Thread/Stream C: policy + Qwen KV decode
Thread/Stream D: BiCodec decode
Thread/Stream E: browser network/playback queue
```

关键状态：

```text
frontend_conv_cache
frontend_attention_kv
committed_glm_tokens
source_ctc_prefix
target_capacity_prefix
qwen_past_key_values
speaker_tokens
semantic_history
codec_cache
playback_buffer_ms
```

每个160 ms tick：

1. 只处理新PCM；
2. 更新 frontend cache；
3. 产生0个或多个稳定 GLM token；
4. 更新 safe posterior；
5. WAIT 或 micro-WRITE；
6. semantic达到8个即可推给 codec，不等WRITE完整结束；
7. codec输出首个稳定PCM立即发送浏览器；
8. 浏览器保持80--120 ms buffer，防止公网抖动。

## 11. 评价指标与通过条件

### 11.1 延迟

- Useful First Audio CA p50/p90/p95；
- First WRITE NCA；
- StartOffset CA/NCA；
- ATD、LAAL、DAL；
- per-chunk ACT；
- frontend、Qwen、codec、network分项；
- RTF p50/p95；
- browser underrun和buffer growth。

### 11.2 质量

- Text-BLEU/chrF/COMET；
- Speech-BLEU/ASR-COMET；
- prefix BLEU/COMET；
- premature WRITE；
- under-translation、hallucination、revision；
- semantic unique ratio、maximum run、collapse rate；
- UTMOS、AutoPCP、SLC；
- speaker cosine；
- boundary click/spectral distance。

### 11.3 第一阶段 gate

```text
Frontend:
  p50 first stable GLM <= 400 ms
  p95 first stable GLM <= 720 ms
  committed rollback = 0
  token agreement vs teacher >= 90%（先作为研究目标）

Policy:
  p50 First WRITE NCA <= 640 ms
  premature WRITE <= 5%
  final flush = 100%

End-to-end:
  p50 Useful First Audio CA <= 900 ms
  p95 <= 1400 ms
  RTF p95 < 0.6
  semantic collapse = 0
  decode failure = 0
```

质量 gate 应以当前 R2 和 offline Phase3 为基线，根据 dev 置信区间冻结，不能训练结束后临时改变。

## 12. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| student token与teacher不兼容 | Qwen质量大幅下降 | 保持codebook、hidden distill、offline/online混训 |
| 过早翻译 | 否定/语序错误 | target support alignment、Bayesian posterior、premature reward |
| 频繁小WRITE | 语音断裂 | micro-phrase boundary head、codec continuity loss |
| semantic collapse | 滋啦声/超长输出 | 保留现有anti-collapse和Phase3安全回退 |
| 固定音色降低speaker preservation | 不像源说话人 | 明确low-latency模式；另设pre-enrolled voice模式 |
| 公网抖动 | underrun | 80--120 ms jitter buffer、自适应buffer |
| 160 ms计算不过实时 | backlog增长 | cache、CUDA Graph、KV cache、NAR micro-chunk |
| 15 shard过拟合 | dev看似亚秒但full退化 | full198正式训练和跨域CVSS-T测试 |

## 13. 推荐落地优先级

### P0：先证明系统下限

1. 固定目标音色；
2. 冻结真实墙钟 profiler；
3. 为 `AudioStreamingStudent` 增加 cache API；
4. 在15 shard训练160/320 ms因果 frontend；
5. fixed wait-k跑真实麦克风；
6. 若 p50 first stable GLM仍>700 ms，先不要训练GRPO。

### P1：真正亚秒主线

1. 精细 source/target support alignment；
2. Source/Target CTC + calibrated safe posterior；
3. Qwen KV-cache；
4. micro-WRITE SFT；
5. low-holdback codec；
6. dev latency/quality Pareto选择。

### P2：正式质量恢复

1. full198；
2. Stage7B latency-constrained GRPO；
3. bilingual quality constraints；
4. offline replay regularization；
5. full dev/test/CVSS-T和主观试听。

### P3：创新增强

1. Bayesian uncertainty-aware latency budget；
2. speculative speech draft/verification；
3. causal BiCodec student；
4. NAR semantic micro-chunk；
5. source voice预注册和跨会话缓存。

## 14. 最终推荐

如果目标是尽快证明“真正 <1秒可行”，最推荐的实验不是继续微调当前 R2 action bias，而是：

```text
15-shard
+ 固定目标音色
+ 160 ms causal Audio Student v2
+ 80 ms right context
+ fixed wait-k=2 baseline
+ Qwen KV-cache
+ 8--16 semantic micro-WRITE
+ holdback=2 / overlap=40 ms
```

它能快速回答最关键问题：当前 UniSS/Qwen 在不使用4秒未来上下文时，是否仍能基于320--640 ms源
信息生成可懂目标语音。如果这一点成立，再加入 Target CTC、Bayesian safe commit 和 GRPO恢复质量。

如果该 baseline 都无法在320--640 ms前缀上保持翻译质量，那么继续优化 WAIT/WRITE reward 不会解决
问题，应转向更强的 causal frontend、text-pivot baseline 或 direct monotonic transducer。

## 15. 参考工作

- SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation
- StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning
- Textless Streaming Speech-to-Speech Translation using Semantic Speech Tokens
- High-Fidelity Simultaneous Speech-to-Speech Translation / Hibiki
- Simultaneous Speech-to-Speech Translation Without Aligned Data / Hibiki-Zero
- A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Speech Translation / NAST-S2x
- Seamless/SeamlessStreaming 中的 monotonic streaming translation 思路

这些工作共同说明：亚秒级不是单纯“更早按WRITE”，而是源表示必须因果可用、目标提交必须可校准、
语音生成必须足够细粒度，并且所有模块都要以真实墙钟延迟联合评估。
