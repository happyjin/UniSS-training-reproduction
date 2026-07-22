# Simul-UniSS：Simultaneous Speech-to-Speech Translation 详细实施方案

> 文档状态：设计与实施规划，不代表相关模块已经实现或训练完成
> 生成日期：2026-07-22
> 适用仓库：`/opt/dlami/nvme/jasonleeeli/projects/UniSS`
> 设计依据：`Simul_UniSS_方案分析与实施建议.docx`、当前 UniSS 代码与数据，以及 SimulS2S-LLM、Hibiki、Hibiki-Zero、StreamSpeech、NAST-S2x 论文
> 核心约束：所有 Simul-UniSS 代码、数据、配置、checkpoint 和日志均使用独立命名空间，不覆盖或改变现有 Phase1–3 的默认行为与复现路径

## 1. 结论先行

推荐主方案不是照搬某一篇论文，而是保留 UniSS 的 Qwen + GLM linguistic tokens + BiCodec semantic/global tokens 主体，同时组合以下能力：

1. 使用当前离线 GLM/Whisper-VQ tokenizer 作为 teacher，蒸馏一个输出相同 16,384 维离散 token 空间的 multi-chunk、prefix-stable streaming student。
2. 在 streaming student hidden states 上增加 Source CTC 和 Target CTC policy heads，判断当前是否听到新内容，以及当前源前缀最多支持多少目标文本。
3. 使用已有 `<|wait_read|>` / `<|write_generate|>` 训练 Qwen 的 WAIT/WRITE 动作，并将完整句级生成改成“源 chunk → 动作 → 目标文本短语 → 对应 semantic chunk”的时间顺序交错生成。
4. 第一版保留 UniSS 的 AR BiCodec semantic-token generation；BiCodec 采用带左上下文的重叠解码、稳定中心提交与 cross-fade，之后再训练边界连续性。
5. 监督训练稳定后，借鉴 Hibiki-Zero 只对 policy、目标文本前缀和语音提交边界做 GRPO，降低延迟并抑制过早猜测。
6. 只有在真实 profiling 证明 AR semantic generation 无法达到 RTF < 1 时，才增加 NAST-S2x 风格的 NAR semantic generator。

推荐系统可概括为：

```text
流式源音频
   ↓
multi-chunk Streaming GLM Student
   ├─ stable GLM linguistic-token commit
   ├─ Source CTC：是否出现新的源内容
   └─ Target CTC：当前源前缀支持多少目标内容
   ↓
外部流式 Controller + UniSS/Qwen KV cache
   ├─ WAIT
   └─ WRITE → 目标文本短语 → BiCodec semantic-token chunk
   ↓
Streaming BiCodec：重叠解码、稳定区域提交、cross-fade
   ↓
不可回滚的目标语音播放
```

正式研究主线对应原 DOCX 的“方案 B”。SimulS2S-LLM 的 CIF + wait-k 作为低成本、必须复现的 baseline；NAST-S2x 是吞吐优化分支，不应在第一阶段替换 UniSS 的单模型生成方式。

## 2. 目标、非目标与验收原则

### 2.1 目标

- 在源说话尚未结束时开始生成并播放目标语音。
- 尽可能复用现有 UniSS 已学到的 ASR、S2TT、MT、TTS、S2ST、音色与表达能力。
- 支持不同 latency budget，而不是只训练一个固定 chunk size。
- 已播放目标语音不可回滚；所有提交动作都必须有明确稳定性约束。
- 同时报告翻译质量、语音质量、speaker preservation、系统延迟、计算延迟和边界连续性。
- 保持现有离线 Phase1–3 checkpoint、数据、脚本和配置可复现。

### 2.2 第一版非目标

- 不整体复刻 Hibiki 的 temporal/depth 多流生成架构。
- 不直接把当前离线 tokenizer 在推理时机械加 causal mask 后称为正式流式模型。
- 不在没有稳定 SFT policy 的前提下直接做 GRPO。
- 不在第一版扩展全部 CVSS 21 种源语言。当前仓库训练常量只完整支持 `eng` 与 `cmn`；首版应以 English↔Mandarin 为主，CVSS 先使用 `zh→en`。
- 不立即修改 UniSS 的 180,407 大词表或旧 checkpoint embedding。

### 2.3 验收原则

每一阶段只有在以下条件同时满足后才能进入下一阶段：

- 离线能力没有超出预设阈值的退化；
- streaming 指标相对上一阶段有可重复提升；
- 所有数据、配置、代码、checkpoint、日志均可追溯；
- 现有 Phase1–3 单元测试、dry-run 和至少一个历史 checkpoint 推理 smoke test 继续通过。

## 3. 当前仓库能力与真实缺口

### 3.1 已经具备的能力

当前词表已经预留流式任务与动作 token，不需要为第一版改词表：

| ID | Token | 推荐用途 |
|---:|---|---|
| 180382 | `<|task_streaming_tts|>` | 流式 TTS 辅助任务 |
| 180383 | `<|task_streaming_asr|>` | 流式 ASR 辅助任务 |
| 180384 | `<|task_streaming_s2st|>` | Simul-S2ST 主任务 |
| 180385 | `<|task_streaming_s2tt|>` | Simul-S2TT 辅助任务 |
| 180386/180390 | `<|start_content|>` / `<|end_content|>` | 目标文本短语边界 |
| 180387/180391 | `<|start_global_token|>` / `<|end_global_token|>` | 32 个 speaker/global tokens 边界 |
| 180388/180392 | `<|start_semantic_token|>` / `<|end_semantic_token|>` | BiCodec semantic chunk 边界 |
| 180389/180393 | `<|start_glm_token|>` / `<|end_glm_token|>` | 每个源 GLM chunk 边界 |
| 180395 | `<|wait_read|>` | WAIT/READ 动作 |
| 180396 | `<|write_generate|>` | WRITE 动作 |
| 180399–180402 | slow/balance/fast/dynamic mode | latency/quality policy 条件 |
| 180406 | `<|streaming_mode|>` | 流式会话标识 |

现有 token 空间还包括：

- BiCodec global：4,096 个，offset 151,665；每条样本当前使用 32 个 global tokens。
- BiCodec semantic：8,192 个，offset 155,761；约 50 token/s。
- GLM semantic/linguistic：16,384 个，offset 163,953；约 12.5 token/s。
- 现有 Qwen/Megatron 训练路径支持 18,000-token packed sequence、sample boundary 与 FlashAttention packed metadata。

### 3.2 当前 GLM 前端不是严格 streaming

`pretrained_models/UniSS/glm4_tokenizer/config.json` 的关键配置为：

```json
{
  "encoder_causal_convolution": true,
  "encoder_causal_attention": false,
  "quantize_causal_block_size": 200,
  "quantize_causal_encoder": false,
  "pooling_kernel_size": 4
}
```

代码确实在 `quantize_causal_block_size` 非空时构造 block-causal mask。按 Whisper 卷积后的约 20 ms 帧率计算，block size 200 约相当于 4 秒；pooling 后 block size 同比例缩小，因此块内仍有约 4 秒双向上下文。这说明当前前端有因果卷积和 block-causal 基础，但仍存在以下问题：

- `encoder_causal_attention=false`，不是逐帧严格因果 attention；
- 4 秒块内可以看未来，320–1280 ms latency 下会发生明显分布错位；
- `Glm4Tokenizer.tokenize()` 每次重新编码完整输入，没有可复用的流式 state/cache；
- 没有 stable-prefix commit、revision tracking 或不可回滚承诺；
- 当前特征提取路径仍按完整片段工作。

因此它适合初始化 streaming student 和构造 prefix re-encoding baseline，但不能直接作为正式 simultaneous 前端。

### 3.3 当前 BiCodec 不是严格 streaming

当前接口是：

```python
detokenize(semantic_tokens, global_tokens)
```

`WaveGenerator` 使用带对称 padding 的一维卷积、转置卷积和多个残差单元，没有显式流式 cache。直接逐 chunk 独立解码会产生：

- click 或波形相位跳变；
- F0、能量、语速和音色边界不连续；
- 每个短语像独立 TTS 片段；
- 前后 chunk 缺少足够声学上下文。

所以必须先实现 overlap re-decode，再决定是否训练因果/流式 decoder。

### 3.4 当前 Megatron 数据路径需要独立扩展

现有 `training/pack_sequences.py` 只产生 0/1 `loss_mask`，而 Simul-UniSS 需要让动作、文本和 semantic token 使用不同浮点权重。例如：

```text
WAIT/WRITE action : target text : semantic token = 4 : 2 : 1
```

现有 dataset 虽把 loss mask tensor 转成 `float32`，但输入校验要求 JSON 中元素为整数。因此不要直接修改旧 packer/dataset；应新增 Simul 专用 builder、packer、dataset 和 training entrypoint。

## 4. 五篇论文与本方案的对应关系

| 论文 | 采用的核心思想 | 在 Simul-UniSS 中的位置 | 不直接照搬的部分 |
|---|---|---|---|
| SimulS2S-LLM | streaming encoder、CIF boundary prompts、测试时 wait-k、ASR-BLEU/ATD | CIF + wait-k baseline；验证离线 UniSS 能否被“解锁”为流式推理 | 连续 speech prompt 和独立 CTC speech generator 与当前离散 UniSS 不完全兼容 |
| StreamSpeech | chunk-based encoder、multi-chunk training、Source/Target CTC、AR text + NAR unit | 正式 streaming front end 与 READ/WRITE eligibility 的主要依据 | 不整体替换 UniSS 为 Conformer + 两遍 AR/NAR 模型 |
| Hibiki | contextual alignment、alignment-aware TTS、自然停顿、voice transfer | 生成较安全的 phrase schedule、目标语音时间对齐和 speaker-preserving 数据 | 不复刻 temporal/depth 多流架构 |
| Hibiki-Zero | 无精细对齐起步、过程/最终翻译 reward、GRPO 降低延迟 | SFT 后的 policy 优化 | 不在基础 streaming model 尚未稳定时直接 RL |
| NAST-S2x | chunk NAR、CTC blank/repeat、two-step glancing、速度与 discontinuity 分析 | AR semantic generation 不能实时后的可选加速器 | 不在第一版移除目标文本 CoT 或完全替换 UniSS |

## 5. 推荐系统架构

### 5.1 Streaming front end

正式 student 从当前 `WhisperVQEncoder` 初始化，保持输出 token 空间兼容：

- 冻结或低学习率维护 16,384-entry GLM codebook；
- 使用 chunk-causal attention：chunk 内双向、只访问历史 chunk、禁止未来 chunk；
- 支持有限右上下文配置，例如 0/80/160 ms，作为 latency–quality 控制项；
- 训练时随机采样 320/640/960/1280 ms 和 offline chunk；
- 为 convolution、attention 和 feature extraction 增加显式 streaming state；
- 输出 dense hidden states、CTC logits、量化 token 以及 stable-prefix metadata。

建议的 chunk 与 GLM token 对应关系为：

| 原始音频 chunk | encoder post-conv 帧数（约） | pooling 后 GLM tokens（约） |
|---:|---:|---:|
| 320 ms | 16 | 4 |
| 640 ms | 32 | 8 |
| 960 ms | 48 | 12 |
| 1280 ms | 64 | 16 |
| offline | 全长 | 全长 |

这些数值用于 batch 构造与延迟预算，实际输出长度应以 tokenizer 的有效 attention mask 为准。

### 5.2 Policy heads

在 streaming hidden states 上增加三个 head：

1. Teacher-token CTC：对齐离线 teacher 的 GLM token 序列。
2. Source CTC：预测源文本 prefix，用于确认新源内容。
3. Target CTC：预测当前源 prefix 支持的目标文本容量，不直接替代 Qwen 翻译。

不建议 Source/Target CTC 直接使用 Qwen 的约 151k 文本词表，因为会产生两个很大的 CTC projection。建议训练独立的 8k 中英 SentencePiece policy tokenizer：

- 中英文共享 8k vocabulary；
- 保留 blank、unk、bos、eos；
- 仅供 CTC policy 和指标对齐使用；
- Qwen 的目标文本仍使用原 UniSS/Qwen tokenizer；
- policy tokenizer 及其 hash 写入每个 manifest 与 checkpoint metadata。

### 5.3 Qwen interleaved generation

所有新源信息都按时间顺序追加到 decoder-only context 尾部，不把新 source token 插回历史 prompt，从而保留标准 KV cache。

推荐首版序列协议如下：

```text
[TASK_STREAMING_S2ST, STREAMING_MODE, DYNAMIC_MODE,
 TGT_LANG, SPEED,
 START_GLOBAL, g1 ... g32, END_GLOBAL]

START_GLM, x1 ... x8, END_GLM
WAIT_READ

START_GLM, x9 ... x16, END_GLM
WRITE_GENERATE, TGT_LANG,
START_CONTENT, target_phrase_1, END_CONTENT,
START_SEMANTIC, s1 ... s80, END_SEMANTIC

START_GLM, x17 ... x24, END_GLM
WAIT_READ
...
EOS
```

DOCX 中提议的 `<SRC>`/`</SRC>`、`<SPEAK>`/`</SPEAK>` 可直接分别映射到现有 `START_GLM`/`END_GLM` 和 `START_SEMANTIC`/`END_SEMANTIC`，第一版不扩词表。

当前词表没有独立 `SOURCE_EOS`。首版由外部 controller 维护 `is_final`：

- 非 final chunk：允许 WAIT 或 WRITE；
- final chunk：将 WAIT logit mask 为负无穷，强制 flush；
- flush 完成后模型输出已有 `TOKEN_EOS`；
- 如果消融证明显式 SOURCE_EOS 对质量很重要，再创建独立 vocabulary-version 分支，不能静默改变旧词表。

### 5.4 三层 WRITE policy

每个源 chunk 到达后依次执行：

1. **Source CTC gate**：Source CTC stable prefix 没增加时默认 WAIT，避免静音或重复声学帧触发输出。
2. **Target CTC capacity gate**：若当前可支持目标 policy tokens 数 `N_tgt` 不大于已提交数 `M_committed`，则 WAIT。
3. **Qwen action decision**：满足前两层后，Qwen 在 WAIT/WRITE 中选择；即使容量增加，也可等待到自然、可发音的短语边界。

推荐初始 eligibility 条件：

```text
source_prefix_growth >= 1
AND target_supported_tokens - target_committed_tokens >= min_write_tokens
AND target_ctc_confidence >= threshold
```

`min_write_tokens` 和置信度阈值按语言分别在 dev set 标定，不在代码中写死。

## 6. 数据现状与数据源选择

### 6.1 当前 UniST Parquet 的真实字段

当前 `data/raw/UniST` 有 208 个 Parquet 文件，约 29 GB。公开行结构包含：

```text
id
transcription / translation
source_glm / target_glm
source_bicodec / target_bicodec
bicodec_global
dataset_name / src_lang / tgt_lang / split
source_glm_len / target_glm_len
source_bicodec_len / target_bicodec_len
duration_ratio
```

但不包含：

- 原始源/目标音频路径；
- 词级或音素级时间戳；
- source-target alignment；
- chunk 边界；
- WAIT/WRITE 标签。

因此当前数据可直接训练离线 Qwen，却不能不经处理就训练严格 simultaneous policy。

### 6.2 可从现有 token 恢复的内容

- `source_bicodec + bicodec_global` 可重建近似源音频；
- `target_bicodec + bicodec_global` 可重建目标音频；
- `source_glm` 约 12.5 token/s，可把 token index 暂时映射为约 80 ms；
- BiCodec semantic 约 50 token/s，可把 token index 暂时映射为约 20 ms。

这种索引时间只适合 bootstrap 和 baseline，不能当作真实词对齐。正式 schedule 应在原始音频或重建音频上运行时间戳 ASR/forced alignment，并记录对齐置信度。

### 6.3 CVSS 的使用边界

本机 `/opt/dlami/nvme/jasonleeeli/CVSS` 保存 CVSS-C/CVSS-T 官方 target-speech archives。官方数据特性为：

- 21 个 source languages → English；
- CVSS-C 与 CVSS-T 各约 1,900 小时；
- 两类 target speech archives 合计约 198.40 GiB；
- CVSS-T 适合 voice transfer，CVSS-C 适合高质量单 speaker target speech；
- 官方 archive 只含翻译语音和 normalized translation text；源语音必须使用 Common Voice 4，并按文件名匹配，原始翻译文本来自 CoVoST 2。

首版只建议引入 `zh→en`：

- 与当前 `cmn/eng` language tokens 兼容；
- CVSS-T 可增强跨语言 speaker preservation；
- CVSS-C 可用于语音自然度和稳定 target alignment；
- 在 Common Voice 4 source audio 未完成配对、checksum/manifest 未验证前，不得把 target-only archive 当成 S2ST pair。

## 7. 数据处理完整流程

### 7.1 数据目录隔离

建议只新增以下命名空间：

```text
data/processed/simul_uniss_v1/
  manifests/
  audio_reconstructed/
  source_alignments/
  target_alignments/
  contextual_alignments/
  schedules/
  samples/
  packed/
  stats/

data/megatron/simul_uniss_v1/
configs/experiments/simul_uniss_v1/
checkpoints/simul_uniss_v1/
runs/simul_uniss_v1/
logs/simul_uniss_v1/
```

不修改 `data/processed/phase*`、`data/megatron/phase*`、现有 checkpoint 或历史 run。

### 7.2 Step D0：数据审计和不可变 manifest

为每个样本生成一行基础 manifest：

```json
{
  "schema_version": "simul_uniss_manifest_v1",
  "id": "...",
  "dataset_name": "...",
  "split": "train",
  "src_lang": "cmn",
  "tgt_lang": "eng",
  "transcription": "...",
  "translation": "...",
  "source_glm_len": 69,
  "source_bicodec_len": 273,
  "target_bicodec_len": 314,
  "duration_ratio": 0.8734,
  "source_kind": "unist_tokens",
  "source_parquet": "...",
  "source_row_group": 0,
  "source_row_index": 0
}
```

同时写入：

- 输入文件 SHA256、大小、row count；
- tokenizer/config/checkpoint hash；
- 数据处理 git commit；
- 处理命令、随机种子和失败原因；
- split 不可跨版本漂移。

### 7.3 Step D1：音频准备

按优先级选择音频：

1. 原始数据音频，若能够从数据源恢复；
2. CVSS/CoVoST/Common Voice 官方 source-target pair；
3. 由 UniST BiCodec tokens 重建的 bootstrap audio。

重建时必须：

- 统一为 mono 16 kHz；
- 不覆盖 token 原始数据；
- 输出 FLAC/WAV 与 checksum；
- 记录 reconstruction model hash；
- 抽样计算重建音频 ASR WER/CER，质量差的样本只用于 token-index baseline，不用于正式前端训练。

### 7.4 Step D2：源文本时间对齐

对源音频和 `transcription` 生成 word/character timestamps：

- 英文：WhisperX、MFA 或 CTC segmentation；
- 中文：字/词级 CTC alignment，保留分词版本；
- 对每个 token 保存 `start_ms`、`end_ms`、confidence；
- 与 VAD 段落、静音、重叠语音和非语音事件一起保存。

低置信度处理：

- alignment coverage < 95%；
- WER > 20% 或中文 CER > 15%；
- 大段未对齐区间 > 1 秒；
- 音频/文本时长比例异常。

这些阈值是首轮建议，需要根据 dev 分布调整。低置信度样本可保留给离线 replay，但不用于高精度 WAIT/WRITE 监督。

### 7.5 Step D3：目标文本与目标语音对齐

对 target audio 与 `translation` 做 forced alignment，得到目标词/字到 BiCodec semantic index 的映射：

```text
target word/char time [start_ms, end_ms]
     ↓ 50 Hz
semantic token span [floor(start_ms/20), ceil(end_ms/20)]
```

短语边界应优先落在：

- 标点；
- 句法短语边界；
- 目标语音自然静音；
- 0.6–1.8 秒的可播放语音长度；
- 至少 2 个目标 policy tokens，除非 source EOS flush。

### 7.6 Step D4：contextual alignment

借鉴 Hibiki，为每个目标 token/短语估计最早安全源前缀。对于目标 token `y_j` 和源文本前缀 `x_≤i`：

```text
score(i,j) = log P(y_j | x_≤i, y_<j)
gain(i,j)  = score(i,j) - score(i-1,j)
```

候选最早位置可以由最大信息增益、达到完整上下文概率的某个比例，或 prefix translation 的一致性共同决定。为避免过早输出，最终 safe boundary 取保守组合：

```text
safe_source_time(phrase_j) = max(
  contextual_alignment_time,
  source_ctc_stable_time,
  phrase_dependency_time,
  previous_phrase_safe_time
)
```

对英中语序差异较大的句子，contextual alignment 比按线性长度比例切分更重要。

### 7.7 Step D5：构造 multi-chunk schedule

每条音频随机生成多个 chunk 视图：

```text
chunk_ms ∈ {320, 640, 960, 1280, offline}
right_context_ms ∈ {0, 80, 160}
chunk_jitter_ms ∈ {-80, 0, +80}
```

对每个 chunk 生成：

- 音频起止时间；
- teacher GLM 完整序列和当前 prefix；
- student 可见帧范围；
- Source CTC prefix；
- Target CTC supported length；
- safe target phrase 列表；
- action label：WAIT/WRITE；
- 对应 target text span；
- 对应 target semantic span；
- source 是否 final；
- 所有 alignment confidence。

### 7.8 Step D6：动作与交错样本格式

建议的中间 JSONL：

```json
{
  "schema_version": "simul_uniss_schedule_v1",
  "id": "magicdata_...",
  "src_lang": "cmn",
  "tgt_lang": "eng",
  "chunk_ms": 640,
  "speaker_tokens": [2747, 3196],
  "events": [
    {
      "source_start_ms": 0,
      "source_end_ms": 640,
      "source_glm": [10815, 4116],
      "source_ctc_count": 3,
      "target_ctc_count": 0,
      "action": "wait",
      "source_is_final": false
    },
    {
      "source_start_ms": 640,
      "source_end_ms": 1280,
      "source_glm": [7661, 13319],
      "source_ctc_count": 7,
      "target_ctc_count": 5,
      "action": "write",
      "target_text": "I want to use Baidu",
      "target_semantic": [1283, 1717],
      "source_is_final": false
    }
  ]
}
```

真实文件必须保存完整 32 个 speaker tokens 和完整 token spans；上述数组仅为示意。

### 7.9 Step D7：样本过滤与去泄漏

推荐过滤：

- 源或目标空文本、空 token；
- 过短 < 0.8 秒或过长 > 30 秒的首版训练样本；
- ASR/forced-alignment 置信度过低；
- source/target duration ratio 极端；
- semantic rate 显著偏离 50 token/s；
- GLM rate 显著偏离 12.5 token/s；
- target phrase 无法映射到连续 semantic span；
- CVSS 源/目标文件名不匹配或 Common Voice 版本不正确。

split 必须按原始 utterance、speaker 和数据源分组，禁止同一语音的不同 chunk view 跨 train/dev/test。

### 7.10 Step D8：packing 与 loss weights

新增 Simul 专用 packer，允许每个 label 的浮点权重：

| Segment | 初始权重 |
|---|---:|
| 源 chunk、header、外部追加 token | 0.0 |
| WAIT/WRITE action | 4.0 |
| 目标文本短语 | 2.0 |
| 文本/semantic 边界与 EOS | 2.0 |
| BiCodec semantic tokens | 1.0 |

权重只是初始值，应通过 action F1、translation quality 和 semantic perplexity 联合标定。必须保留 sample boundaries，禁止 packed 样本间 attention 泄漏。

## 8. Streaming student 的结构与训练

### 8.1 初始化与冻结策略

推荐配置：

- encoder：从当前 GLM tokenizer 初始化；
- codebook：第一阶段完全冻结；第二阶段只在离线性能退化很小的前提下以极低 LR 解冻；
- Qwen：Stage 1 冻结；
- CTC heads：随机初始化；
- offline teacher：全程冻结，生成 teacher tokens/logits；
- feature extractor：首轮冻结，避免重建音频域引起漂移。

### 8.2 前端损失

```text
L_front = λteach L_teacher_ctc
        + λsrc   L_source_ctc
        + λtgt   L_target_ctc
        + λstab  L_prefix_stability
        + λhid   L_hidden_distill
        + λdown  L_downstream
```

建议初始权重：

```text
λteach=1.0, λsrc=0.3, λtgt=0.3,
λstab=0.2, λhid=0.2, λdown=0.2
```

这些值不是论文固定值，需在小规模 dev experiment 上调参。

各损失定义：

- `L_teacher_ctc`：student dense frames 到离线 GLM token sequence 的 CTC。
- `L_source_ctc`：student hidden states 到源 policy-token sequence 的 CTC。
- `L_target_ctc`：student hidden states 到目标 policy-token sequence 的 CTC。
- `L_prefix_stability`：同一语音相邻 prefix 的 committed logits/token 一致性。
- `L_hidden_distill`：student 与 teacher 对齐时间位置的 hidden-state cosine/MSE。
- `L_downstream`：把 student token/embedding 送入冻结 UniSS，计算 streaming ASR/S2TT loss。

### 8.3 Stable-prefix commit

baseline 和正式 student 都应记录 longest common prefix revision。推荐首版 commit rule：

- 一个 token 在连续 2 个 chunk 的解码 prefix 中保持不变；
- token posterior margin 超过 dev 标定阈值；
- 保留最后 1–2 个不稳定 token 不提交；
- 已提交 token 永不回滚；
- source EOS 时提交剩余 stable prefix，并把不确定尾部交由 final flush 处理。

正式 student 的目标是逐步减少“完整 prefix 重编码”，最终由 cache-aware streaming encoder 原生提供 committed tokens。

### 8.4 Multi-chunk sampling

训练 batch 中建议：

```text
320 ms: 20%
640 ms: 30%
960 ms: 20%
1280 ms: 20%
offline: 10%
```

offline 样本用于维持上界；若小 chunk 性能不足，可提高 320/640 ms 比例，而不是完全移除 offline replay。

## 9. 完整分阶段训练过程

### Stage 0A：现有前端 prefix re-encoding baseline

**目的**：不训练新模型，先测当前 GLM token 的 prefix stability 和计算成本。

流程：

1. 每 320/640/960/1280 ms 把累计源音频重新送入现有 tokenizer。
2. 比较相邻 prefix 的 longest common prefix。
3. 只提交连续两次稳定的 GLM tokens。
4. 使用固定 source-time wait-k 或 GLM-token wait-k。
5. Qwen 暂时只做 Simul-S2TT，确认文本前缀可用后再输出语音。

产物：

- prefix revision curves；
- tokenizer computation-aware RTF；
- wait-k quality–latency Pareto；
- 失败样例集合。

### Stage 0B：CIF + wait-k baseline

**目的**：复现 SimulS2S-LLM 风格、接近文本粒度的固定策略 baseline。

- 在 streaming/student hidden states 上训练 CIF boundary units；
- 使用 `wait-k`，至少评估 `k ∈ {1, 3, 5, 7, 9}`；
- 保持同一 decoder/checkpoint，避免 baseline 因模型容量不同而不可比；
- 报告 ASR-BLEU、ATD、LAAL 和 RTF。

### Stage 1：Streaming GLM student distillation

**训练模块**：streaming encoder、teacher-token CTC、hidden distillation projection。
**冻结模块**：offline teacher、Qwen、BiCodec、首轮 codebook。
**数据**：原始/重建 source audio + offline teacher tokens，混合全部 chunk sizes。
**目标**：输出与现有 UniSS speech embedding 兼容的稳定 token。

建议优化：

- AdamW，encoder LR 从 `1e-5` 量级起做 sweep；
- 新增 head 可用 `1e-4` 量级；
- 5% warmup，cosine decay；
- bf16、gradient clipping 1.0；
- 保存 best offline token agreement 与 best streaming stability 两类 checkpoint。

进入下一阶段前必须满足：

- offline teacher-token agreement 达到可接受水平；
- 640/960 ms 下 committed prefix revision 明显低于 Stage 0A；
- streaming ASR/S2TT 相对当前离线前端的退化在预设范围内；
- cache 与 full-prefix forward 的 committed output 一致。

### Stage 2：Source/Target CTC policy heads

**训练模块**：Source CTC、Target CTC，必要时低 LR 联调 student 顶层。
**数据**：带 source/target text 和 chunk prefix 的数据。
**目标**：估计新源内容与可安全输出目标容量。

训练时同时计算：

- Source CTC loss 与 prefix WER/CER；
- Target CTC loss 与 supported-token-count MAE；
- gate precision/recall；
- premature eligibility rate。

Source/Target CTC 的作用是 eligibility，不要求它们生成最终翻译文本。

### Stage 3：WAIT/WRITE action SFT

**初始化**：使用当前最佳 Phase3/Performance checkpoint 的兼容分支。
**训练模块**：Qwen；streaming front end 可先冻结。
**数据**：只包含 header、source chunks 与 WAIT/WRITE supervision，可暂不生成长 semantic chunks。
**损失**：action CE 为主，少量 target-text phrase CE。

建议先训练：

```text
source chunk → WAIT
source chunk → WRITE → target phrase
```

不要一开始就让大量 semantic tokens 淹没稀疏 action 标签。

### Stage 4：Phrase-level interleaved S2ST SFT

完整序列：

```text
source chunk
→ WAIT/WRITE
→ target text phrase
→ corresponding BiCodec semantic chunk
→ next source chunk
```

训练数据混合建议：

| 数据类型 | 初始比例 | 目的 |
|---|---:|---|
| Simul-S2ST interleaved | 50% | 主任务 |
| Simul-S2TT/action-only | 15% | 强化 policy 与翻译 |
| 原 Phase3 Quality/Performance replay | 20% | 保持离线 S2ST |
| ASR/S2TT/MT replay | 10% | 保持语言和识别能力 |
| TTS/voice replay | 5% | 保持声学生成与 speaker 条件 |

先冻结 streaming front end 训练 Qwen，再以较低 LR joint tune，便于定位退化来源。

### Stage 5：Streaming BiCodec

分两步进行。

#### Stage 5A：无需训练的 overlap decode baseline

- 会话内冻结 32 个 speaker tokens；
- 每次保留 25–50 个 semantic tokens（0.5–1.0 秒）左上下文；
- 加入新 semantic chunk 后重解码上下文窗口；
- 丢弃不稳定左边缘，只提交稳定中心；
- 相邻输出保留 40–120 ms overlap，并做 equal-power cross-fade；
- source final 时完整 flush 尾部。

#### Stage 5B：chunk-aware BiCodec refinement

使用完整解码波形作为 teacher，对随机 semantic chunk 训练：

```text
L_codec = L_wave
        + L_multi_scale_stft
        + L_feature_matching
        + L_boundary
        + L_f0
        + L_energy
        + L_speaker
        + L_full_chunk_consistency
```

重点优化 chunk 边界前后 100–200 ms。若改造 WaveGenerator 为严格 causal decoder，应保留原 decoder checkpoint 和独立配置，不能替换旧 BiCodec 文件。

### Stage 6：Joint low-LR refinement

解冻 student 顶层、policy heads 和 Qwen，以较低 LR 联调：

- Qwen LR：Stage 4 的 0.1–0.3 倍；
- encoder LR：Qwen LR 的 0.2–0.5 倍；
- codebook 默认仍冻结；
- offline replay 不低于 30%；
- 每个 eval interval 同时评估 offline 与 streaming dev。

若 streaming 提升伴随 offline 明显回退，优先增加 replay 或冻结底层，而不是覆盖历史 checkpoint。

### Stage 7：Hibiki-Zero 风格 GRPO

只在 Stage 4–6 已能稳定输出 WAIT/WRITE 与最终正确翻译后开始。

对同一输入采样 4–8 个 action/phrase rollout。建议 reward：

```text
R = w_process * R_prefix_translation
  + w_final   * R_final_translation
  - w_latency * R_latency
  - w_early   * R_premature_write
  - w_repeat  * R_repetition
  + w_voice   * R_voice_continuity
  - w_kl      * KL(policy || SFT_reference)
```

推荐边界：

- RL 主要更新 WAIT/WRITE、目标文本短语和 semantic chunk 结束位置；
- 不把每个 BiCodec semantic token 当作大动作空间直接做 RL；
- final translation reward 权重始终足够高，防止模型通过永远 WAIT 或输出极短内容“优化延迟”；
- source final 必须强制 flush；
- 保存 reward 各分量，不能只看总 reward。

### Stage 8：可选 NAST-S2x NAR semantic generator

只有满足以下任一条件才进入：

- AR semantic generation 在目标硬件上 p95 RTF ≥ 1；
- Qwen 生成一个语音 chunk 的时间长期大于 source chunk interval；
- 外部 source buffer 持续增长；
- 降低 semantic chunk 长度后质量或连续性不可接受。

NAR generator 输入：

- Qwen target phrase hidden states；
- speaker/global tokens；
- streaming source hidden states；
- duration/length condition。

输出使用 CTC blank/repeat 去重和可选 two-step glancing。必须与 AR 系统在同一 quality–latency 图上比较，尤其报告 NAST-S2x 指出的 discontinuity 问题。

## 10. Speaker tokens 与 voice preservation

### 10.1 推荐默认策略

默认等待源语音前 1–2 秒完成 speaker enrollment，然后冻结 32 个 global tokens，整个会话不更新。这会增加首音频延迟，但最稳妥。

支持三种运行模式：

| 模式 | 做法 | 适用场景 |
|---|---|---|
| pre-enrolled | 用户预先提供参考语音 | 产品低延迟、固定用户 |
| initial enrollment | 首 1–2 秒提取后冻结 | 默认研究设置 |
| dynamic update | 随源音频更新 | 只做消融，不作为首版默认 |

动态更新容易让目标音色随时间漂移，且已播放音频不能回滚。

### 10.2 CVSS-C 与 CVSS-T 的不同用途

- CVSS-C：单一 canonical target voice，适合研究 streaming synthesis quality 和边界连续性。
- CVSS-T：target voice 从 source 转移，适合 speaker similarity 与 voice-preserving S2ST。
- 两者不能混为同一种 speaker 监督；manifest 必须保存 `cvss_variant`。

## 11. 在线推理完整过程

### 11.1 Session state

每个会话维护：

```text
audio_ring_buffer
frontend_cache
committed_source_glm
source_ctc_stable_prefix
target_ctc_supported_count
qwen_kv_cache
committed_target_text
generated_semantic_history
bicodec_decode_context
speaker_tokens
source_is_final
input_queue_latency_ms
waveform_commit_time
```

### 11.2 并发原则

源音频采集不能因为 Qwen 正在生成 semantic tokens 而停止。输入线程持续写入 ring buffer；Qwen 完成当前 phrase 后，controller 合并期间到达的 source chunks，再追加到 KV context。

AR 版本必须限制单次语音生成：

- phrase target audio 建议 0.6–1.8 秒；
- 设置 semantic-token chunk 上限；
- 达到自然短语边界即输出 `END_SEMANTIC`；
- 若 input queue 增长，切换更短 phrase 或 fast policy；
- 不允许无限生成整句 semantic tokens 后才回到 READ。

### 11.3 推理状态机伪代码

```python
state = init_session(target_language, latency_mode)

while True:
    pcm, is_final = audio_source.read_chunk()
    state.audio_ring_buffer.append(pcm)

    hidden, glm_candidates, state.frontend_cache = frontend.stream_step(
        pcm,
        cache=state.frontend_cache,
    )

    new_glm = stable_prefix_commit(
        glm_candidates,
        already_committed=state.committed_source_glm,
    )
    src_ctc = source_ctc.prefix(hidden)
    tgt_capacity = target_ctc.supported_length(hidden)

    qwen.append_external_tokens(
        [START_GLM, *encode_glm(new_glm), END_GLM],
        kv_cache=state.qwen_kv_cache,
    )

    eligible = (
        source_prefix_grew(src_ctc, state.source_ctc_stable_prefix)
        and tgt_capacity > committed_policy_tokens(state.committed_target_text)
    )

    if is_final:
        action = WRITE  # WAIT is logit-masked
    elif not eligible:
        action = WAIT
    else:
        action = qwen.generate_action(allowed=[WAIT, WRITE])

    if action == WAIT:
        qwen.commit(WAIT_READ)
    else:
        phrase, semantic = qwen.generate_phrase_and_semantic_chunk(
            max_phrase_tokens=policy.max_phrase_tokens,
            max_semantic_tokens=policy.max_semantic_tokens,
        )
        state.committed_target_text += phrase
        state.generated_semantic_history += semantic
        wav = bicodec_streamer.decode_and_commit(
            semantic,
            speaker_tokens=state.speaker_tokens,
        )
        audio_sink.play(wav)

    if is_final:
        flush_remaining_target_and_codec(state)
        break
```

### 11.4 Source EOS 与异常处理

- source EOS：禁止 WAIT，允许多轮 WRITE 直到 EOS。
- 长静音：不反复追加空 GLM chunk；保留心跳与超时。
- 网络抖动：区分真实静音与缺包，不让缺包触发错误 WRITE。
- source buffer overflow：降级到较大 chunk、短 semantic chunk 或 NAR branch，并记录事件。
- 生成失败：不回滚已播放波形；只允许从最近未提交 semantic 边界重试。

## 12. 评价数据集与实验设置

### 12.1 评价集合

至少包括：

1. UniST 原始 dev/test，保持原 split。
2. 专门的 high-confidence aligned streaming dev/test。
3. CVSS `zh→en` test：CVSS-C 与 CVSS-T 分开报告。
4. 长语音集合：把同 speaker 相邻句拼接成 30–120 秒会话，测试 cache、queue 和跨句 speaker stability。
5. 噪声/停顿/自修正集合：测试 premature WRITE。
6. 人工困难集：英中长距离重排序、否定词、数字、命名实体、句尾消歧。

### 12.2 必须比较的系统

- 当前离线 Phase3/Performance upper bound；
- Stage 0A prefix re-encoding + wait-k；
- CIF + wait-k；
- streaming student + wait-k；
- streaming student + Source/Target CTC；
- + Qwen WAIT/WRITE；
- + streaming BiCodec；
- + GRPO；
- 可选 + NAR semantic generator。

所有系统使用相同 ASR、相同文本 normalization、相同 test split 和相同硬件计时协议。

## 13. 评价指标

### 13.1 翻译与内容质量

| 指标 | 对象 | 说明 |
|---|---|---|
| SacreBLEU | committed target text | 可复现的文本翻译基线 |
| COMET | committed target text | 语义质量，对词面变化更稳健 |
| chrF | target text/prefix | prefix 和中文字符级补充 |
| ASR-BLEU | generated target speech → ASR text | S2ST 论文常用 |
| ASR-COMET | generated speech ASR text | 语义质量补充 |
| BLASER 2.0 | source speech/target speech | 无需只依赖 ASR 文本 |
| WER/CER | ASR/TTS intelligibility | 英文 WER、中文 CER |
| under-translation | 最终输出 | 漏译比例 |
| over-generation | 最终输出 | 幻觉/多译比例 |
| repetition/EOS error | semantic/text | 重复、未停止、异常短输出 |

### 13.2 语音质量与 speaker preservation

- WavLM/ECAPA speaker cosine similarity；
- DNSMOS 或同类自动自然度指标，作为筛查而非最终结论；
- F0 correlation、F0 range、energy correlation；
- source/target duration ratio；
- 人工 MOS：quality、naturalness、intelligibility、speaker similarity；
- preference test：offline vs streaming、AR vs NAR；
- CVSS-C 与 CVSS-T 分开报告 speaker 指标。

### 13.3 算法延迟

- StartOffset / first-audio latency：源音频开始到第一段目标波形可播放。
- EndOffset：源 EOS 到目标语音结束。
- AL：Average Lagging。
- LAAL：Length-Adaptive Average Lagging。
- DAL：Differentiable Average Lagging。
- AP：Average Proportion。
- ATD：Average Token Delay。
- target word/character emission timestamp 和 target audio playback timestamp 两套口径。

必须同时报告 computation-unaware 与 computation-aware 版本。只报告 chunk policy 的理论延迟会掩盖 tokenizer、Qwen 和 BiCodec 实际耗时。

### 13.4 系统性能

- end-to-end RTF；
- frontend/Qwen/BiCodec 分模块 RTF；
- 每 chunk latency 的 p50/p90/p95/p99；
- first-token、first-semantic-token、first-waveform latency；
- GPU memory 峰值；
- source ring-buffer backlog 和最大 queue time；
- 每秒可服务 session 数；
- semantic tokens/s 和 waveform seconds/s。

### 13.5 边界连续性

- discontinuity count / mean / sum；
- chunk boundary waveform derivative jump；
- click detector rate；
- boundary 前后 F0 jump；
- boundary 前后 RMS energy jump；
- speaker embedding drift over time；
- 相邻短语 pause duration 分布；
- 人工 continuity MOS。

### 13.6 稳定性与 policy

- source token prefix revision rate；
- committed source rollback rate，目标必须为 0；
- target text revision rate；
- committed target rollback rate，目标必须为 0；
- WAIT/WRITE accuracy、macro F1；
- premature WRITE rate；
- unnecessary WAIT rate；
- source EOS flush completion rate；
- action entropy 与平均连续 WAIT 次数。

## 14. Quality–latency Pareto 与建议通过门槛

不要只选择单个 BLEU 最高 checkpoint。对每个 chunk size、policy threshold 和 latency mode 绘制：

```text
x 轴：LAAL / ATD / first-audio latency / computation-aware latency
y 轴：ASR-BLEU / COMET / BLASER / MOS
颜色：RTF 或 chunk size
```

第一轮工程 gate 可设为：

- 目标硬件 end-to-end p95 RTF < 0.9；
- initial-enrollment 模式 first-audio median ≤ 2.0 秒；pre-enrolled ≤ 1.2 秒；
- ASR-BLEU 相对离线 Phase3 下降不超过 2.0；
- text COMET 相对离线上界下降不超过 0.03；
- committed rollback = 0；
- committed source-token revision ≤ 0.1%；
- speaker cosine 相对离线输出下降不超过 0.03；
- click/discontinuity rate < 1 次/分钟；
- source EOS flush completion ≥ 99.9%。

这些是建议的首轮目标，不是论文既定阈值；应在获得离线 upper bound 和硬件 profiling 后冻结正式门槛。

## 15. 消融实验矩阵

### 15.1 前端

- prefix re-encoding vs cache-aware student；
- 320/640/960/1280/offline；
- right context 0/80/160 ms；
- fixed chunk vs multi-chunk training；
- teacher CTC only vs + hidden distillation vs + downstream loss；
- codebook frozen vs low-LR tune；
- stable 1/2/3 chunks 后提交。

### 15.2 Policy

- wait-k；
- Source CTC only；
- Target CTC only；
- Source + Target CTC；
- CTC eligibility + Qwen action；
- contextual alignment vs 粗线性 schedule；
- action/text/semantic loss weight 1:1:1、4:2:1、8:2:1。

### 15.3 生成

- direct semantic vs phrase-level text CoT；
- AR semantic vs NAR semantic；
- 0.6/1.0/1.4/1.8 秒 semantic chunk；
- speaker pre-enrolled vs initial enrollment vs dynamic；
- no overlap vs overlap-add vs chunk-aware refinement。

### 15.4 后训练

- SFT only；
- final reward only；
- process + final reward；
- + latency penalty；
- + premature-write penalty；
- 不同 KL anchor 和 rollout group size。

## 16. 推荐新增代码与配置布局

以下是实施时建议新增的文件，不在本方案阶段创建：

```text
training/simul_uniss/
  schema.py
  policy_tokenizer.py
  reconstruct_unist_audio.py
  align_source_text.py
  align_target_speech.py
  build_contextual_alignment.py
  build_schedules.py
  sample_builders.py
  pack_sequences.py
  dataset.py
  streaming_frontend.py
  train_streaming_frontend.py
  pretrain_simul_uniss_megatron.py
  train_streaming_bicodec.py
  train_grpo.py
  eval_simul_s2st.py

uniss/streaming/
  state.py
  frontend.py
  policy.py
  controller.py
  bicodec_streamer.py
  inference.py

configs/experiments/simul_uniss_v1/
  stage0_prefix_baseline.env
  stage1_frontend.env
  stage2_ctc_policy.env
  stage3_action_sft.env
  stage4_interleaved_s2st.env
  stage5_bicodec.env
  stage6_joint.env
  stage7_grpo.env

scripts/simul_uniss/
  prepare_data.sh
  train_stage1_frontend.sh
  train_stage2_policy.sh
  train_stage3_action.sh
  train_stage4_s2st.sh
  train_stage5_bicodec.sh
  train_stage6_joint.sh
  train_stage7_grpo.sh
  run_streaming_eval.sh
  start_tensorboard.sh
```

### 16.1 与旧代码隔离的强制规则

- 不改变 `training/sample_builders.py` 的默认序列。
- 不改变 `training/pack_sequences.py` 的旧 0/1 loss-mask 语义。
- 不改变 `training/megatron_uniss_dataset.py` 的输入校验。
- 不改变 Phase1–3 shell script 默认参数。
- 不覆盖 `pretrained_models/UniSS` 内 tokenizer/BiCodec 文件。
- 不在旧 checkpoint 目录继续保存 Simul checkpoint。
- 新 entrypoint 显式要求 `--simul-schema-version`，防止误读旧 JSONL。

## 17. 测试、回归与可复现要求

### 17.1 单元测试

- token protocol 编解码 round-trip；
- source chunk 只追加、不插入历史位置；
- final chunk 禁止 WAIT；
- weighted loss mask 对齐 next-token labels；
- packed samples attention 不跨界；
- stable-prefix commit 永不回滚；
- cached streaming forward 与 full-prefix committed region 一致；
- BiCodec overlap commit 无重复/丢样；
- schedule 中 target text 与 semantic spans 完整覆盖且不重叠。

### 17.2 回归测试

每次合并 Simul 代码前执行：

1. 当前 `training/tests` 全部通过。
2. 现有 Phase1、Phase2、Phase3 脚本 dry-run 输出不变。
3. 使用历史 Phase2/Phase3 checkpoint 做一次现有音频评测 smoke test。
4. 比较旧 sample builder 生成 JSON hash，确认无意外变化。
5. `git diff` 不包含历史配置、checkpoint、数据 manifest 的覆盖。

### 17.3 训练可复现

每个 run 保存：

- git commit 和 dirty diff；
- 完整 env/config；
- data manifest hash；
- tokenizer/codebook/checkpoint hash；
- 随机种子；
- GPU 型号、CUDA、PyTorch、Megatron commit；
- TensorBoard 和结构化 JSON metrics；
- best-quality、best-latency、best-Pareto 三类 checkpoint 链接。

## 18. 资源与存储规划

### 18.1 GPU

- Stage 0 数据与 baseline profiling 可分 GPU 批量执行，但不要与现有 8-GPU Phase1 训练争抢同一设备。
- Stage 1 前端蒸馏优先使用 8 GPU DDP/FSDP；先用小 shard 测显存，再确定 batch。
- Stage 3–6 可延续现有 Megatron 8-GPU 路径，但使用独立 master port、save dir、TensorBoard dir。
- Stage 7 rollout 生成与 reward 计算可拆成 inference workers + learner，避免 learner 等待音频评测。

### 18.2 存储

- 当前 UniST token 数据可直接复用，不复制原始 Parquet。
- 重建 16 kHz mono PCM 的理论体积约 115 MB/小时；优先 FLAC 并按 shard 保存。
- alignment、schedule 和 JSONL 体积远小于音频，但 multi-chunk views 不应复制音频，只保存时间索引。
- teacher logits 非常大；优先缓存 teacher token、边界和低维 hidden projection，不缓存完整 16k logits。
- CVSS target archives 当前约 198.4 GiB；配套 Common Voice 4 source audio 需要额外空间。

## 19. 主要风险与缓解方案

| 风险 | 表现 | 缓解 |
|---|---|---|
| UniST 缺少原始音频/时间戳 | schedule 不可靠 | bootstrap 与正式数据分轨；优先恢复原始音频；保存 alignment confidence |
| student token 漂移 | Qwen 无法理解新 GLM tokens | teacher CTC、冻结 codebook、hidden/downstream distillation、offline replay |
| 小 chunk 质量差 | 翻译错误、前缀频繁修改 | multi-chunk training、有限右上下文、保守 commit、CTC gate |
| semantic tokens 淹没 policy loss | WAIT/WRITE 学不好 | weighted float loss mask、action-only curriculum |
| Qwen 生成语音时无法 READ | source queue 增长 | 限制 phrase/semantic chunk、异步缓存、必要时 NAR |
| BiCodec 边界不连续 | click、F0/音色跳变 | 左上下文重解码、stable-center commit、cross-fade、边界专项训练 |
| GRPO reward hacking | 永远 WAIT、短输出、幻觉 | final reward、flush 约束、KL anchor、分项监控 |
| streaming 训练破坏离线能力 | 历史实验不可复现 | 独立文件/checkpoint、offline replay、回归测试、禁止覆盖旧资源 |
| CVSS 只下载 target speech | 错配成 S2ST pair | 必须配 Common Voice 4 + CoVoST 2 manifest，文件名和 checksum 双重验证 |

## 20. 推荐实施里程碑

### M0：两周内的可行性基线

- 完成 100–1000 条高质量样本的 audio reconstruction/alignment；
- 跑完 prefix re-encoding stability；
- 建立 wait-k Simul-S2TT；
- 输出 quality–latency–RTF 报告；
- 决定 student 的最小可用 chunk/right-context。

### M1：Streaming front end

- 完成 multi-chunk student、cache、teacher CTC；
- 640/960 ms stable-prefix 指标通过；
- streaming ASR/S2TT 可用；
- Source/Target CTC heads 达到 gate 精度要求。

### M2：可工作的 Simul-S2ST SFT

- Qwen 能输出 WAIT/WRITE；
- phrase-level target text/semantic interleaving 正常；
- source final 能 100% flush；
- overlap BiCodec 无明显重复/漏播；
- 与离线 Phase3 形成首个 Pareto 曲线。

### M3：高保真 streaming audio

- chunk-aware BiCodec refinement；
- speaker similarity、F0、energy、click 指标达到 gate；
- 完成长语音和 CVSS-T voice-preserving evaluation。

### M4：低延迟 policy 优化

- GRPO 相对 SFT 在相同最终质量下降低 LAAL/ATD；
- premature WRITE 不增加；
- reward 各分量无异常；
- 形成论文主系统与完整消融。

### M5：可选实时性能分支

- 仅当 AR profiling 不通过时实现 NAR semantic generator；
- 与 AR 主系统比较质量、RTF 和 discontinuity；
- 保留 AR 主线作为高保真版本。

## 21. 最终推荐执行顺序

严格按照以下顺序推进：

1. 冻结当前 Phase1–3 复现基线、checkpoint hash 和离线指标。
2. 从小规模 UniST 重建数据建立 alignment 与 schedule 工具链。
3. 完成 Stage 0A prefix re-encoding 与 Stage 0B CIF/wait-k baseline。
4. 训练 Stage 1 streaming student，不同时改 Qwen/BiCodec。
5. 训练 Stage 2 Source/Target CTC policy heads。
6. 训练 Stage 3 action-only/Simul-S2TT，确认 WAIT/WRITE 正确。
7. 训练 Stage 4 phrase-level interleaved Simul-S2ST。
8. 先用 Stage 5A overlap decode，再做 Stage 5B BiCodec refinement。
9. 以 Stage 6 低学习率联合优化并持续混合离线 replay。
10. 在 SFT 稳定后执行 Stage 7 GRPO。
11. profiling 失败时才进入 Stage 8 NAR semantic generation。

这一路线把风险按模块拆开：先验证数据与前端，再训练 policy，再接语音，最后做 RL 和吞吐优化。任何阶段失败都可回退到上一阶段，不需要覆盖或破坏现有 UniSS 实验。

## 22. 参考论文与链接

- UniSS: Unified Speech and Text Foundation Model，arXiv:2509.21144：<https://arxiv.org/abs/2509.21144>
- SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation，arXiv:2504.15509：<https://arxiv.org/abs/2504.15509>
- High-Fidelity Simultaneous Speech-to-Speech Translation（Hibiki），arXiv:2502.03382：<https://arxiv.org/abs/2502.03382>
- Simultaneous Speech-to-Speech Translation Without Aligned Data（Hibiki-Zero），arXiv:2602.11072：<https://arxiv.org/abs/2602.11072>
- StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning，arXiv:2406.03049：<https://arxiv.org/abs/2406.03049>
- A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Speech Translation（NAST-S2x），arXiv:2406.06937：<https://arxiv.org/abs/2406.06937>
- CVSS corpus：<https://arxiv.org/abs/2201.03713>

## 23. 当前应立即做与暂时不应做的事项

### 应立即做

- 固化离线 Phase3 upper bound；
- 写独立数据 schema 和小规模 alignment prototype；
- 测当前 GLM prefix revision 与重编码 RTF；
- 建 wait-k Simul-S2TT baseline；
- 验证 CVSS `zh→en` 是否已具备 Common Voice 4 source pair，而不只检查 target archives。

### 暂时不应做

- 直接启动全量 8-GPU Simul-S2ST 训练；
- 修改现有词表或复用旧 checkpoint 目录；
- 用 token index 伪时间直接训练最终 policy；
- 在没有 streaming BiCodec 指标时只看 ASR-BLEU；
- 在 SFT 尚不稳定时做 GRPO；
- 在没有 profiling 证据时提前实现 NAR generator。
