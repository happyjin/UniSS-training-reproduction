# 多说话人流式语音翻译（multi_speaker_translation）训练方案说明

> Stage-3 SFT · `multi_speaker_translation`
> 模型：Qwen-OmniSpeak（Thinker + Dual-Route Bridge + Qwen3-TTS Talker + Code2Wav）
> 模板：`qwen_omnispeak_multi_speaker_translation`
> 入口：`pipelines/sft/train_multi_speaker_translation.sh`（DeepSpeed）/ `train_multi_speaker_translation_megatron.sh`（Megatron）

本文档描述 **多说话人、多源语言、语音流同时输入输出翻译** 的训练方案，覆盖任务定义、数据构建、输入输出构造、读写策略监督、损失函数、DeepSpeed/Ray/Megatron 训练流程、关键开关与实现约束。

本任务与 `normal_translation` 在训练机制上完全等价：**codec-only 监督**、真实时间轴目标 codec 铺设、间隙静默填充、单调 `codec_to_sem_visible` 读写掩码，以及任务无关的 main/residual codec CE。差异仅是独立注册名、独立 `MULTI_SPEAKER_TRANSLATION_*` 环境变量前缀，以及数据中携带 `speaker_id` / `source_langs` / 逐 chunk `source_lang` 等多说话人与多源语言元数据。

## 1. 任务说明

目标是训练一个流式语音到语音翻译模型：源音频从 0 时刻开始持续到达，模型在每一刻隐式决定继续 **READ（听输入）** 还是 **WRITE（输出翻译语音）**。源端可以包含多个说话人、多个源语言；目标端固定由样本顶层 `target_lang` 指定（示例为 `zh`）。

本任务不预测目标文本。`target_text` 与 alignment 里的文本字段只作为数据参考与调试信息，不进入 thinker 文本 CE。训练监督完全来自目标语音 codec：

- **WRITE**：目标 codec 被铺到真实时间轴上，开头等待与 chunk 间隔由静默 codec 帧填充；静默帧也参与 CE，模型学习什么时候不说、什么时候开口。
- **READ**：`codec_to_sem_visible` 限制每个目标 codec 帧只能看见当时已经到达的源语音前缀，模型不能依赖未来源音频。

多说话人与多源语言不引入新模型头或新 loss。逐 chunk 的 `speaker_id`、`source_lang` 和顶层 `source_langs` 作为元数据随样本保留，翻译行为由 teacher-forced 的目标 codec 序列和流式掩码隐式学习。

## 2. 数据 Schema

训练数据为 JSONL，每行一条样本。参考 `multi_speaker_translation_sample.json` 与 `multi_speaker_translation_data_format.jsonl`：

```jsonc
{
  "schema_version": "ms-swift-speech-simul-v1",
  "task": "multi_speaker_translation",
  "task_prompt": "Task: Multi speaker translation, translate the following audio to Chinese.",
  "messages": [{"role": "user", "content": "<audio>"}],
  "audios": ["<source.wav>"],
  "target_text": "<全文译文，仅作元数据>",
  "target_audios": ["chunk_0000.wav", "chunk_0001.wav"],
  "source_langs": ["en", "ar", "tr"],
  "target_lang": "zh",
  "alignment": [
    {
      "chunk_id": 0,
      "speaker_id": "SPEAKER_A",
      "source_lang": "en",
      "source_start_ms": 0,
      "source_end_ms": 2260,
      "target_start_ms": 3000,
      "target_end_ms": 5238,
      "target_audio": "chunk_0000.wav"
    }
  ],
  "chat_template_kwargs": {"sampling_rate": 24000}
}
```

训练侧严格信任 `alignment[i].target_start_ms` / `target_end_ms`。源音频起点固定为 0；目标音频的开头等待区和不同 chunk 间间隔用静默音频对应的 codec 帧填充。

## 3. 数据构建与校验

实现位置：`custom/dataset.py::MultiSpeakerTranslationPreprocessor`。预处理器继承 `NormalTranslationPreprocessor`，仅替换任务集合与环境变量前缀，因此复用四个 SFT 任务共享的三条样本舍弃规则。校验在按 `chunk_id` 排序后进行，任一不满足则默认丢弃样本；`MULTI_SPEAKER_TRANSLATION_STRICT=true` 时抛错。

1. **源 chunk 不重叠**：对每个 `i >= 1`，要求：
   ```text
   source_end_ms[i-1] < source_start_ms[i]
   ```

2. **目标 chunk 不重叠**：对每个 `i >= 1`，要求：
   ```text
   target_end_ms[i-1] < target_start_ms[i]
   ```
   这里是严格小于；相等也会被舍弃。

3. **目标总时长上界**：要求：
   ```text
   max(target_end_ms) < source_duration_ms
                        + 0.5 * (last.source_end_ms - last.source_start_ms)
                        + 3000ms
   ```
   `source_duration_ms` 默认优先用 `soundfile` 读取 `audios[0]` 的真实 wav 时长；读取失败或 `MULTI_SPEAKER_TRANSLATION_SOURCE_DURATION_FROM_FILE=false` 时回退到 `max(source_end_ms)`。

## 4. 输入输出构造

实现位置：`custom/template.py::QwenOmniSpeakMultiSpeakerTranslationTemplate`，继承 `QwenOmniSpeakSimultaneousPolicyTemplate`，只设置：

```python
env_prefix = 'MULTI_SPEAKER_TRANSLATION'
```

### 4.1 Thinker 侧

模板保留 user `<audio>` 输入和 `task_prompt`。没有 assistant 文本轮，Thinker 只作为编码器产出 hidden states，经 dual-route bridge 输入 Talker；文本 logits 不参与 loss，也不统计 `token_text_acc`。

### 4.2 目标 codec 真实时间轴

`encode_target_audios_concat(...)` 将每个 `target_audio` 编码为 codec，并按 `target_start_ms` 放到真实时间轴上：

```text
0 ........ target_start[0] .... target_end[0] ... target_start[1] ...
[ 静默等待 ][ chunk_0 codec ][ 静默间隔 ][ chunk_1 codec ] ...
```

帧率由 codec tokenizer 自校准（通常 80ms/frame）；静默帧来自真实静默波形编码后的稳态 codec 帧。顶层 `target_lang="zh"` 会进入 control_prefix，因此本任务可以显式承载目标语言信息。

### 4.3 读写掩码

模板为每个目标 codec 帧构造单调非减的 `codec_to_sem_visible`：

```text
t_ms(j) = j * frame_ms + MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS
arrived(j) = round(M * clamp(t_ms(j) / source_duration_ms, 0, 1))
visible(j) = a0 + arrived(j)
```

其中 `a0` 是源音频 token 在 thinker input 中的起点，`M` 是源音频 token 数。模型侧 `build_block_causal_mask(...)` 让 codec 帧只能注意到 `visible(j)` 之前的 semantic/source 前缀，同时保持 codec 自回归因果约束。

`MULTI_SPEAKER_TRANSLATION_STREAMING=false` 可关闭流式掩码，退化为离线全上下文注意力，用作消融。

## 5. 损失函数

本任务为纯 codec 监督，总损失为 main codec CE 与 residual codec CE 的加权和：

```text
L = CE(main_codec_logits, codec_labels)
    + residual_loss_weight * CE(residual_codec_logits, codec_residual_labels)
```

`codec_labels` 覆盖整条真实时间轴，包括静默帧和语音帧。静默帧监督 WRITE 决策，流式注意力掩码约束 READ 可见性。没有 thinker 文本 CE，没有额外策略头，也没有新的二元分类 loss。

监控指标以 main codec / residual codec token 准确率为主。

## 6. 训练流程

### 6.1 DeepSpeed / Ray

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export MULTI_SPEAKER_TRANSLATION_DATA=/abs/path/to/your.jsonl
export QWEN3_TTS_TOKENIZER=/path/to/project/models/Qwen3-TTS-Tokenizer-12Hz
bash pipelines/sft/train_multi_speaker_translation.sh

# 多机 Ray（先启动 Ray 集群）
USE_RAY=true bash pipelines/sft/train_multi_speaker_translation.sh
```

默认训练 thinker + bridge + talker backbone，冻结 Code2Wav/vocoder。常用超参通过环境变量覆盖：`TUNER_TYPE`、`DEEPSPEED`、`LEARNING_RATE`、`NUM_TRAIN_EPOCHS`、`MAX_LENGTH`、`GRADIENT_ACCUMULATION_STEPS`、`PER_DEVICE_TRAIN_BATCH_SIZE`。

### 6.2 Megatron

Megatron 路径复用任务无关的 mcore talker codec loss。模板产出的 codec batch 与 `codec_to_sem_visible` 和同传、normal、bilingual 管线一致，因此不需要 Megatron 任务特定代码。

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export MULTI_SPEAKER_TRANSLATION_DATA=/abs/path/to/your.jsonl
export QWEN3_TTS_TOKENIZER=/path/to/project/models/Qwen3-TTS-Tokenizer-12Hz
export TP=2 PP=1 CP=1 EP=1
export SEQUENCE_PARALLEL=false
bash pipelines/sft/train_multi_speaker_translation_megatron.sh
```

约束：codec / streaming 路径要求 `PP=1`、`CP=1`、`sequence_parallel=false`、不开 packing/padding-free；thinker 仍可使用 TP/EP/DP/ZeRO-1。

## 7. 关键开关

| Env | Default | 说明 |
| --- | --- | --- |
| `MULTI_SPEAKER_TRANSLATION_DATA` | `sft/multi_speaker_translation_data_format.jsonl` | 训练 JSONL 路径。 |
| `MULTI_SPEAKER_TRANSLATION_STREAMING` | `true` | `false` 时关闭流式掩码，使用离线全上下文。 |
| `MULTI_SPEAKER_TRANSLATION_EXTRA_LAG_MS` | `0` | 在数据时间轴上叠加全局 wait-k 式延迟。 |
| `MULTI_SPEAKER_TRANSLATION_SOURCE_DURATION_FROM_FILE` | `true` | `false` 时用 `max(source_end_ms)` 代替读 wav 时长。 |
| `MULTI_SPEAKER_TRANSLATION_STRICT` | `false` | 坏样本默认跳过；设为 true 时抛错。 |
| `MULTI_SPEAKER_TRANSLATION_CODEC_CACHE` | `.cache/multi_speaker_translation_codec` | codec 编码缓存目录。 |

## 8. 实现文件索引

| 文件 | 作用 |
| --- | --- |
| `custom/dataset.py` | `MultiSpeakerTranslationPreprocessor` + 数据集注册，复用四个 SFT 任务共享的三条时间轴校验规则。 |
| `custom/template.py` | `QwenOmniSpeakMultiSpeakerTranslationTemplate`，继承同传模板并设置独立 env 前缀。 |
| `custom/plugin.py` | 注册数据集与模板。 |
| `pipelines/sft/common_args.sh` | `run_multi_speaker_translation()` DeepSpeed/Ray 参数。 |
| `pipelines/sft/common_args_megatron.sh` | `run_multi_speaker_translation_megatron()` Megatron 参数。 |
| `pipelines/sft/train_multi_speaker_translation.sh` | DeepSpeed 一键入口；设置 `USE_RAY=true` 时作为 Ray 多机入口。 |
| `pipelines/sft/train_multi_speaker_translation_megatron.sh` | Megatron 一键入口。 |
| `pipelines/sft/multi_speaker_translation_sample.json` | Schema 示例。 |
| `pipelines/sft/multi_speaker_translation_data_format.jsonl` | 占位 JSONL，正式训练前替换路径。 |

## 9. 注意事项

1. `multi_speaker_translation_data_format.jsonl` 中的路径是占位绝对路径，正式训练前需替换为可访问的本地/S3 镜像路径。
2. 示例中 chunk2 和 chunk3 的 `target_end_ms == target_start_ms`，会被严格目标轴不重叠规则舍弃；该文件主要作为 schema 参考。
3. 本管线不修改 `modeling_qwen_omnispeak.py`、`codec_tokenizer.py`、`mcore_talker_loss.py` 或 `mcore_adaptation.py`；Megatron 适配复用已有任务无关 codec-loss 链路。
4. 若未来修改模型源码，注意 `trust_remote_code` 从导出目录 `models/custom_models/<variant>/` 加载，需要同步导出副本并清理 HF module cache。
