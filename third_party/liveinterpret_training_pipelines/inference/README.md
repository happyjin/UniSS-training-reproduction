# qwen_omnispeak 推理测试脚本

Omni 官方复现及 S4/S5 的四条跑测链路位于 `custom/streaming_si/`，见 [推理入口导航](../docs/INFERENCE_ENTRYPOINTS.md)。本目录用于 `qwen_omnispeak` 模型。

本目录提供三个公开 CLI：

- `offline_infer.py`：离线推理，支持文本/音频输入，输出文本或音频。
- `realtime_infer.py`：实时测试，`--source file` 模拟文件流式输入，`--source mic` 从麦克风采集整句后推理播放。
- `s2s_infer.py`：真 S2S 专用入口；显式指定训练时的 `model_type` 和 checkpoint，先生成译文再生成语音。

> 说明：`s2s_infer.py` 已提供 Talker、Thinker 和 residual code predictor 的独立 KV-cache
> 路径，并支持固定窗口流式解码。`realtime_infer.py --source file` 仍是 prefix-probe
> 模拟，用于评估“源音频到达多少后模型开始输出非静默音频”；`--source mic` 当前以
> VAD 切句后整句推理为主。加速开关含义与推荐配置见下文「S2S 流式推理加速开关」一节。

## 依赖

先安装根目录依赖：

```bash
pip install -r requirements.txt
```

如需麦克风或实时播放：

```bash
pip install -r inference/requirements.txt
```

## 离线推理

真 S2S checkpoint（推荐使用专用入口）：

```bash
python inference/s2s_infer.py \
  --model-type qwen2_5_3b_omnispeak_a \
  --checkpoint output/qwen_omnispeak_output/bidirectionalS2S/upstream_reply_add-thinker-talker-3b-lr1e-5-bs64x8-s2s/v1-20260726-021439/checkpoint-13500 \
  --val-jsonl output/qwen_omnispeak_output/bidirectionalS2S/upstream_reply_add-thinker-talker-3b-lr1e-5-bs64x8-s2s/v1-20260726-021439/val_dataset.jsonl \
  --index 0 --device cuda:0
```

也可以用 `--audio /path/source.wav --target-lang zh` 传单条源语音。默认启用与训练一致的 speaker x-vector；可用 `--no-speaker-prompt` 关闭。

### S2S 流式推理加速开关

所有加速开关默认关闭并可独立组合。H200 实测最快的安全组合为：

```bash
python inference/s2s_infer.py \
  --model-type qwen2_5_3b_omnispeak_a \
  --checkpoint /path/to/checkpoint \
  --val-jsonl /path/to/val_dataset.jsonl \
  --index 0 --device cuda:0 \
  --streaming --kv-cache \
  --code-predictor-kv-cache \
  --compile-code-predictor \
  --code-predictor-compile-mode default \
  --thinker-kv-cache \
  --gpu-codec-buffer \
  --compile-streaming-decoder
```

生产运行使用 `--code-predictor-compile-mode default`。完整开关表、冷启动时间和
`reduce-overhead` 风险见 上文「S2S 流式推理加速开关」一节。

文本到文本：

```bash
python inference/offline_infer.py \
  --model /path/to/qwen3_omnispeak_d \
  --prompt "Translate to Chinese: hello world" \
  --output text \
  --stream
```

语音到文本：

```bash
python inference/offline_infer.py \
  --model /path/to/qwen3_omnispeak_d \
  --prompt "请识别这段音频" \
  --audio /path/to/source.wav \
  --task s2t \
  --output text \
  --stream
```

文本到语音 / 语音到语音：

```bash
python inference/offline_infer.py \
  --model /path/to/qwen3_omnispeak_d \
  --prompt "Translate this audio to English speech" \
  --audio /path/to/source.wav \
  --task s2s \
  --output audio \
  --target-lang en \
  --stream \
  --output-dir inference_outputs/offline_s2s
```

JSONL 批量输入：

```jsonl
{"id":"utt_001","prompt":"Translate to English speech","audio":"wavs/a.wav","task":"s2s","output":"audio","target_lang":"en"}
{"id":"utt_002","prompt":"请转写","audio":"wavs/b.wav","task":"s2t","output":"text"}
```

运行：

```bash
python inference/offline_infer.py \
  --model /path/to/qwen3_omnispeak_d \
  --samples samples.jsonl \
  --output-dir inference_outputs/offline_batch
```

相对音频路径会按 JSONL 文件所在目录解析。

## 文件流式实时测试

```bash
python inference/realtime_infer.py \
  --model /path/to/qwen3_omnispeak_d \
  --source file \
  --prompt "Translate to English speech" \
  --audio /path/to/source.wav \
  --target-lang en \
  --chunk-ms 320 \
  --infer-every-chunks 4 \
  --play \
  --output-dir inference_outputs/realtime_file
```

默认不启用 `codec_to_sem_visible` mask，因为不同 bridge 的 `semantic_block` 长度和原始 `input_ids` 长度可能不一致。若确认当前模型变体的映射正确，可显式启用：

```bash
python inference/realtime_infer.py ... --use-streaming-mask
```

## 麦克风实时测试

```bash
python inference/realtime_infer.py \
  --model /path/to/qwen3_omnispeak_d \
  --source mic \
  --prompt "Translate my speech to English speech" \
  --target-lang en \
  --play \
  --max-utterances 5
```

常用参数：

- `--mic-sample-rate`：麦克风采样率，默认 `16000`。
- `--mic-device`：指定输入设备。
- `--vad-threshold`：RMS 起音阈值。
- `--utterance-silence-ms`：静音多久后判定一句结束。

## 输出

默认输出目录包含：

```text
inference_outputs/
├── texts/*.txt
├── audios/*.wav
└── metrics.jsonl
```

指标字段包括：

- `total_latency_ms`：整次推理耗时。
- `first_token_latency_ms`：文本流式首 token 时延。
- `first_audio_latency_ms`：音频流式首个非静默 chunk 时延。
- `input_audio_duration_ms` / `output_audio_duration_ms`：输入/输出音频时长。
- `rtf`：`total_latency_ms / output_audio_duration_ms`。
- `generated_text_tokens` / `generated_codec_frames`：生成 token 或 codec 帧数。
- `extra.thinker_generate_ms` / `thinker_finalize_ms`：Thinker 生成与末尾 hidden 整理。
- `extra.talker_prefill_ms` / `talker_step_ms`：Talker cache prefill 与增量 step。
- `extra.code_predictor_ms` / `codec_transfer_ms` / `stream_decoder_ms`：残差预测、codec
  搬运与流式 callback 解码。
- `extra.compile_warmup_ms`：首次编译和 specialization；稳态统计会排除该值。

## 注意事项

- 传入 `--audio` 时，脚本会要求成功构造真正的多模态音频输入；若环境缺少 `qwen_omni_utils` 或 processor 不支持音频，会直接报错，避免静默退化为纯文本推理。
- 音频解码依赖 `qwen-tts` 的 tokenizer decode 接口。不同版本接口名称可能不同，`common.py` 会尝试常见方法并在不兼容时给出错误。
- 这些脚本用于推理测试，不修改训练管线和模型文件。
