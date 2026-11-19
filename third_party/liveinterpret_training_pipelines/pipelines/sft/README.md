# SFT 训练任务总览

本目录包含 Qwen-OmniSpeak 的 SFT 训练入口。当前 SFT 分为两个阶段：

- **Stage-1：从非实时翻译到实时翻译的过渡阶段**
  - 包含 `normal_translation`、`bilingual_translation`、`multi_speaker_translation` 三个任务。
  - 三个任务可以独立训练，也可以通过 `joint-stage1` 联合训练。
  - 数据中通过样本级 `task` 字段区分任务类型。
- **Stage-2：纯实时同传翻译阶段**
  - 只包含 `simultaneous_policy_learning`。
  - 该阶段使用更精细的实时同传数据标注，直接学习完备语义单元驱动的读写策略。

所有 SFT 任务均为 **codec-only speech-to-speech** 训练：模型直接从源语音学习生成目标语音 codec，不预测目标文本；没有 Thinker 文本 CE，也没有 `token_text_acc`。读写策略由两部分隐式监督：

1. **WRITE**：目标语音按真实时间轴铺设，未到翻译时机的片段用静默 codec 帧填充。
2. **READ**：`codec_to_sem_visible` 流式注意力掩码限制每一帧目标 codec 只能看到当前时刻已到达的源语音前缀。

---

## 1. 任务划分

| 阶段 | 任务 | 数据集注册名 | 模板 | 说明 |
| --- | --- | --- | --- | --- |
| Stage-1 | `normal_translation` | `normal_translation` | `qwen_omnispeak_normal_translation` | 普通语音到语音翻译，使用随机开口时延构造从离线到实时的过渡数据。 |
| Stage-1 | `bilingual_translation` | `bilingual_translation` | `qwen_omnispeak_bilingual_translation` | 双语互译任务，支持 `bilingual_translation` / `bilingual_mutual_translation`。 |
| Stage-1 | `multi_speaker_translation` | `multi_speaker_translation` | `qwen_omnispeak_multi_speaker_translation` | 多说话人、多源语言场景下的流式语音翻译。 |
| Stage-1 联合训练 | `joint-stage1` | `sft_joint_stage1` | `qwen_omnispeak_joint_stage1` | 混合前三类 Stage-1 数据，通过样本 `task` 字段分发。 |
| Stage-2 | `simultaneous_policy_learning` | `simultaneous_policy_learning` | `qwen_omnispeak_simultaneous_policy` | 纯实时同传翻译，使用完备语义单元规则做更细粒度的翻译时机标注。 |

`joint-stage1` 支持的 `task` 取值：

- `normal_translation`
- `speech2speech_translation`
- `bilingual_translation`
- `bilingual_mutual_translation`
- `multi_speaker_translation`

---

## 2. 数据格式

每条样本是一行 JSON，核心字段如下：

```jsonc
{
  "schema_version": "ms-swift-speech-simul-v1",
  "task": "normal_translation",
  "messages": [
    {"role": "user", "content": "<audio>"}
  ],
  "audios": ["/abs/path/to/source.wav"],
  "target_audios": [
    "/abs/path/to/chunk_0000.wav",
    "/abs/path/to/chunk_0001.wav"
  ],
  "alignment": [
    {
      "chunk_id": 0,
      "source_text": "...",
      "target_text": "...",
      "source_start_ms": 0,
      "source_end_ms": 2300,
      "target_start_ms": 3600,
      "target_end_ms": 5900,
      "target_audio": "/abs/path/to/chunk_0000.wav"
    }
  ],
  "chat_template_kwargs": {"sampling_rate": 24000}
}
```

关键要求：

- `messages` 中只需要 user 轮 `<audio>`；不需要 assistant 文本轮。
- `alignment` 是训练读写策略的核心监督来源。
- `target_audios` 可以是顶层数组，也可以放在每个 `alignment[i].target_audio` 中。
- 训练侧不重新采样翻译时延，只信任数据中已经写好的 `target_start_ms` / `target_end_ms`。

共享校验规则：

1. 源语音 chunk 不重叠：`source_end_ms[i-1] < source_start_ms[i]`。
2. 目标语音 chunk 不重叠：`target_end_ms[i-1] < target_start_ms[i]`。
3. 目标总时长不能过长：`max(target_end_ms) < source_duration_ms + 0.5 * last_chunk_duration_ms + 3000ms`。

默认遇到坏样本会跳过；设置对应任务的 `*_STRICT=true` 可改为直接报错。

---

## 3. 四个任务的数据准备与翻译时延策略

### 3.1 `normal_translation`

`normal_translation` 面向普通语音到语音翻译（单个音频句子），是 Stage-1 的基础任务。数据准备时，翻译开始时延采用随机策略：

- 以源语言音频开始时间为基准。
- 从 `3000~5000ms` 中随机采样一个数字作为翻译开始时间。
- 翻译开始前的目标时间轴使用静默音频块填充。

训练时不会再次采样随机时延，只按照 `alignment` 中的时间戳构造静默 codec 与流式注意力掩码。

### 3.2 `bilingual_translation`

`bilingual_translation` 面向双语互译，是 Stage-1 中更接近真实交替双语场景的任务。数据准备从未对齐的语音翻译对 \((X, Y)\) 开始，并使用人工静默构造目标侧时延：

1. 验证句子映射约束：
   - \(X\) 和 \(Y\) 包含相同数量的句子；
   - \(Y\) 中第 \(i\) 个句子是 \(X\) 中第 \(i\) 个句子的翻译。
2. 依靠在 \(Y\) 中插入人工静默，延迟目标内容相对于源语音 \(X\) 的出现时间。
3. 对每个索引为 \(i\) 的句子，在 \(X\) 中第 \(i\) 个句子的开始后引入静默，将 \(Y\) 的第 \(i\) 个句子移动 \(\delta_i\)（即 δ_i）个时间单位：

   \[
   \delta_i \sim U(0, \delta \times d_i)
   \]

   其中 \(d_i\) 是 \(X\) 中第 \(i\) 个句子的持续时间，\(\delta \in [0, 1]\)（即 δ ∈ [0, 1]）是超参数，当前取 \(0.5\)。
4. 如果插入静默后，\(Y\) 的第 \(i\) 个句子的结束时间超过 \(X\) 的第 \(i\) 个句子的结束时间过多，即超过下一句可承受的 \(\delta_{i+1}\)（即 δ_i+1）时延预算，则对 \(Y\) 的第 \(i\) 个句子做音频加速处理，防止时延逐句累积。

最终将处理后的目标语音切成 chunk，并写入 `alignment` 的 `target_start_ms` / `target_end_ms`。

### 3.3 `multi_speaker_translation`

`multi_speaker_translation` 面向多说话人、多源语言场景。它与 `bilingual_translation` 使用相同的时延构造思想：

1. 从未对齐的语音翻译对 \((X, Y)\) 开始。
2. 验证句子映射约束：\(X\) 与 \(Y\) 句子数量相同，且 \(Y_i\) 是 \(X_i\) 的翻译。
3. 在目标语音 \(Y\) 中插入人工静默，使每个目标句子相对对应源句子延迟：

   \[
   \delta_i \sim U(0, 0.5 \times d_i)
   \]

4. 当某个目标句子因静默插入导致结束时间过度落后，并可能影响后续句子的时延预算时，对该句目标音频进行加速，避免延迟持续累积。
5. 额外保留多说话人相关元数据，例如 `speaker_id`、`source_lang`、`source_langs` 等。

### 3.4 `simultaneous_policy_learning`

`simultaneous_policy_learning` 是 Stage-2 的纯实时同传翻译任务。它不再使用 Stage-1 的粗粒度随机或人工静默迁移策略，而是：

- 严格按照完备语义单元规则标注翻译开始时机；
- 对源语音进行更细粒度的实时语义边界判断；
- 将每个可翻译语义单元对应到更准确的目标语音时间轴；
- 用更精细的 `alignment` 监督模型在真实同传场景下学习何时 READ、何时 WRITE。

因此，Stage-2 更适合作为 Stage-1 过渡训练后的实时同传能力强化阶段。

---

## 4. 训练入口

### 4.1 Stage-1 单任务训练

```bash
# normal_translation
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export NORMAL_TRANSLATION_DATA=/abs/path/to/normal_translation.jsonl
bash pipelines/sft/train_normal_translation.sh

# bilingual_translation
export BILINGUAL_TRANSLATION_DATA=/abs/path/to/bilingual_translation.jsonl
bash pipelines/sft/train_bilingual_translation.sh

# multi_speaker_translation
export MULTI_SPEAKER_TRANSLATION_DATA=/abs/path/to/multi_speaker_translation.jsonl
bash pipelines/sft/train_multi_speaker_translation.sh
```

Ray 入口不再使用独立 `_ray.sh` 包装脚本；同一个普通入口通过 `USE_RAY=true` 启用 Ray：

```bash
USE_RAY=true bash pipelines/sft/train_normal_translation.sh
USE_RAY=true bash pipelines/sft/train_bilingual_translation.sh
USE_RAY=true bash pipelines/sft/train_multi_speaker_translation.sh
```

`RAY_DEVICE_GROUPS` 需要是 ms-swift 支持的 JSON 字符串；Anyscale 训练任务会由 `anyscale/train/dispatch_train.py` 根据 compute config 自动派生。

Megatron 入口：

```bash
export TP=2 PP=1 CP=1 EP=1
export SEQUENCE_PARALLEL=false
bash pipelines/sft/train_normal_translation_megatron.sh
bash pipelines/sft/train_bilingual_translation_megatron.sh
bash pipelines/sft/train_multi_speaker_translation_megatron.sh
```

### 4.2 Stage-1 联合训练：`joint-stage1`

`joint-stage1` 使用一个混合 JSONL 文件训练 Stage-1 三个任务。每条样本通过 `task` 字段区分任务类型。

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export SFT_JOINT_STAGE1_DATA=/abs/path/to/sft_joint_stage1.jsonl
bash pipelines/sft/train_joint_stage1.sh
```

Ray / Megatron：

```bash
USE_RAY=true bash pipelines/sft/train_joint_stage1.sh

export TP=2 PP=1 CP=1 EP=1
export SEQUENCE_PARALLEL=false
bash pipelines/sft/train_joint_stage1_megatron.sh
```

`joint-stage1` 的数据 fallback 顺序：

1. `dataset_maker/sft_joint_stage1.jsonl`
2. `dataset_maker/sft_joint_stage1.jsonl`
3. `pipelines/sft/joint_stage1_data_format.jsonl`

### 4.3 Stage-2 训练：`simultaneous_policy_learning`

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export SIMULTANEOUS_POLICY_DATA=/abs/path/to/simultaneous_policy_learning.jsonl
bash pipelines/sft/train_simultaneous_policy_learning.sh
```

Ray / Megatron：

```bash
USE_RAY=true bash pipelines/sft/train_simultaneous_policy_learning.sh

export TP=2 PP=1 CP=1 EP=1
export SEQUENCE_PARALLEL=false
bash pipelines/sft/train_simultaneous_policy_learning_megatron.sh
```

---

## 5. 关键环境变量

| 任务 | 数据路径变量 | Streaming 开关 | 额外时延 | Strict 开关 |
| --- | --- | --- | --- | --- |
| `normal_translation` | `NORMAL_TRANSLATION_DATA` | `NORMAL_TRANSLATION_STREAMING` | `NORMAL_TRANSLATION_EXTRA_LAG_MS` | `NORMAL_TRANSLATION_STRICT` |
| `bilingual_translation` | `BILINGUAL_TRANSLATION_DATA` | `BILINGUAL_TRANSLATION_STREAMING` | `BILINGUAL_TRANSLATION_EXTRA_LAG_MS` | `BILINGUAL_TRANSLATION_STRICT` |
| `multi_speaker_translation` | `MULTI_SPEAKER_TRANSLATION_DATA` | `MULTI_SPEAKER_TRANSLATION_STREAMING` | `MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS` | `MULTI_SPEAKER_TRANSLATION_STRICT` |
| `joint-stage1` | `SFT_JOINT_STAGE1_DATA` | `JOINT_STAGE1_STREAMING` | `JOINT_STAGE1_EXTRA_LAG_MS` | `JOINT_STAGE1_STRICT` |
| `simultaneous_policy_learning` | `SIMULTANEOUS_POLICY_DATA` | `SIMULTANEOUS_POLICY_STREAMING` | `SIMULTANEOUS_POLICY_EXTRA_LAG_MS` | `SIMULTANEOUS_POLICY_STRICT` |

默认配置：

- `*_STREAMING=true`：启用流式注意力掩码。
- `*_EXTRA_LAG_MS=0`：不额外增加 wait-k 风格全局时延。
- `*_SOURCE_DURATION_FROM_FILE=true`：优先从源 wav 文件读取真实源语音时长。
- `*_STRICT=false`：坏样本默认跳过，不中断训练。

---

## 6. 文件说明

| 文件 | 说明 |
| --- | --- |
| `common_args.sh` | DeepSpeed / Ray 共享训练参数与各任务 runner。 |
| `common_args_megatron.sh` | Megatron 共享训练参数与各任务 runner。 |
| `train_normal_translation*.sh` | `normal_translation` 训练入口。 |
| `train_bilingual_translation*.sh` | `bilingual_translation` 训练入口。 |
| `train_multi_speaker_translation*.sh` | `multi_speaker_translation` 训练入口。 |
| `train_joint_stage1*.sh` | Stage-1 联合训练入口。 |
| `train_simultaneous_policy_learning*.sh` | Stage-2 同传训练入口。 |
| `*_sample.json` | 单条样本结构参考。 |
| `*_data_format.jsonl` / `data_format.jsonl` | JSONL 数据格式占位文件，真实训练前需要替换路径。 |
| `Normal-Translation.md` | `normal_translation` 详细设计。 |
| `Bilingual-Translation.md` | `bilingual_translation` 详细设计。 |
| `Multi-Speaker-Translation.md` | `multi_speaker_translation` 详细设计。 |
| `Joint-Stage1.md` | `joint-stage1` 联合训练说明。 |
| `Simultaneous-(Read-Write)-Policy-Learning.md` | `simultaneous_policy_learning` 详细设计。 |

---

## 7. Megatron 约束

Megatron 训练路径与 DeepSpeed 共享同一套 codec-only 监督和 `codec_to_sem_visible` 流式掩码逻辑。当前 codec / streaming 路径要求：

- `PP=1`
- `CP=1`
- `SEQUENCE_PARALLEL=false`
- 不使用 packing / padding-free

Thinker 仍可使用 TP / EP / DP / ZeRO-1；上述约束主要来自 talker codec loss 与 streaming mask 路径。

---

## 8. 编辑模型代码注意事项

`trust_remote_code` 加载的是模型导出目录中的 `modeling_qwen_omnispeak.py`，不是 `custom/qwen_omnispeak/` 下的源文件。如果修改了源模型文件，需要同步到导出模型目录并清理 HF module cache：

```bash
SRC=custom/qwen_omnispeak/modeling_qwen_omnispeak.py
for d in /path/to/project/models/*/; do
  [ -f "$d/modeling_qwen_omnispeak.py" ] && cp -f "$SRC" "$d/modeling_qwen_omnispeak.py"
done
rm -rf "${HF_MODULES_CACHE:-.cache/hf_modules_qwen_omnispeak_simul}" /tmp/hf_modules
```
