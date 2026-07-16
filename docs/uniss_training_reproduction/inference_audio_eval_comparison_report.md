# UniSS 官方推理与当前验证音频生成流程对比分析

生成时间：2026-07-16

## 结论

当前刚生成的 3 个音频目录不是 `cmots/UniSS` 官方 GitHub 中面向用户的完整 speech-to-speech translation 推理结果。它们来自本仓库新增的验证试听脚本，用 Phase 1 的 Qwen2.5-0.5B reproduction checkpoint，在 `tts` mode 下从 UniST 预处理 token row 生成音频。

所以现象是合理的：

- `source_wav/` 是输入源语音 token 的重建音频。
- `reference_wav/` 是参考答案音频；但这次因为 `mode=tts`，参考答案也取 `source_bicodec`，所以它和 `source_wav/` 完全相同。
- `wav/` 是模型生成音频；它不是翻译后的目标语言语音，而是 TTS 任务下按源语言转写生成的语音 token。

这次没有得到翻译后的语音，核心原因有两个：

1. 运行脚本默认 `MODES=tts`，没有运行 `quality`、`performance` 或 `direct_s2st`。
2. 使用的是 Phase 1 checkpoint，Phase 1 主要是 ASR/S2TT/TTS 对齐训练，还不是论文最终用于 S2ST 的 Phase 2/Phase 3 完整模型。

## 官方 GitHub 推理流程

官方仓库在 [README.md](../../README.md) 中说明自己是推理代码，Quick Start 指向 `infer.py`，vLLM 示例指向 `vllm_example.py`。

官方单文件推理入口 [infer.py](../../infer.py) 的流程是：

1. 读取真实 wav 文件，必要时重采样到 16 kHz。
2. 用 VAD 将长音频切成 chunk。
3. 对每个 chunk 调用 `UniSSTokenizer.tokenize()`，得到两类语音 token：
   - `glm4_tokens`：输入语音的 linguistic / semantic token。
   - `bicodec_tokens`：包含 speaker/global token 和 BiCodec semantic token。
4. 调用 [uniss/cli/prompt.py](../../uniss/cli/prompt.py) 的 `process_input()` 拼接任务 prompt。
5. 用 `AutoModelForCausalLM.generate()` 或 vLLM 生成文本形式的特殊 token 序列。
6. 调用 `process_output()` / `process_output_vllm()` 从生成文本中抽取 `<|bicodec_semantic_x|>`，再用 `UniSSTokenizer.decode()` 合成音频。
7. 将各 chunk 的生成音频拼回完整 wav，并输出 transcription / translation 文本。

官方 vLLM 推理入口 [vllm_example.py](../../vllm_example.py) 做的是同一件事，只是用 vLLM 批量生成。README 中的示例命令是 `--task Quality --target_language zh`，也就是真正的 S2ST Quality 模式。

官方 `process_input()` 的关键任务定义如下：

| 官方 task | prompt 目标 | 预期输出 |
| --- | --- | --- |
| `Quality` | `<|task_s2s_translation|><|slow_mode|>` | 先 ASR，再 S2TT，再目标语音 semantic token |
| `Performance` | `<|task_s2s_translation|><|balance_mode|>` | 先目标文本翻译，再目标语音 semantic token |
| `S2ST` | `<|task_s2s_translation|><|fast_mode|>` | 直接目标语音 semantic token |
| `ASR` | `<|task_asr|>` | 源语言转写文本 |
| `S2TT` | `<|task_s2t_translation|>` | 目标语言翻译文本 |

也就是说，官方 speech-to-speech translation 的输出音频只会在 `Quality`、`Performance` 或 `S2ST` 任务下出现。`TTS` 不是翻译任务。

## 当前刚跑的验证音频流程

刚跑的输出目录是：

```text
/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase1_gpu3_validation_audio
```

这个目录由 [scripts/export_and_generate_qwen0p5b_phase1_audio_eval.sh](../../scripts/export_and_generate_qwen0p5b_phase1_audio_eval.sh) 生成。脚本做了两步：

1. 将 Megatron-LM Phase 1 checkpoint 导出为 Hugging Face 格式：

```text
checkpoints/uniss_qwen0p5b_phase1_unist13_full
  -> checkpoints/exported_hf/qwen0p5b_phase1_unist13_full_hf
```

2. 调用 [scripts/generate_unist_audio_eval.sh](../../scripts/generate_unist_audio_eval.sh)，再进入 [training/generate_unist_eval_audio.py](../../training/generate_unist_eval_audio.py) 生成验证音频。

这个评测入口和官方推理入口不同：

| 项目 | 官方 `infer.py` / `vllm_example.py` | 当前 `generate_unist_eval_audio.py` |
| --- | --- | --- |
| 输入 | 真实 wav / wav 文件夹 / jsonl 中的 wav 路径 | UniST parquet 中已经预处理好的 token row |
| 语音 token 来源 | 运行时调用 `UniSSTokenizer.tokenize(wav)` | 直接读取 `source_glm`、`source_bicodec`、`target_bicodec`、`bicodec_global` |
| 模型 | 官方 `pretrained_models/UniSS` 完整权重 | 本地 `qwen0p5b_phase1_unist13_full_hf` Phase 1 权重 |
| prompt 生成 | `uniss/cli/prompt.py::process_input()` 文本特殊 token | `training/sample_builders.py` 直接构造 token id |
| 解码 | 从模型输出文本提取 BiCodec semantic token | 从模型输出 token id 提取 BiCodec semantic token |
| 目的 | 用户侧 S2ST 推理 | 训练复现过程中的固定样本试听/验证 |

当前脚本保留了和官方 prompt 兼容的 token 布局，但它不是官方端到端 wav 推理脚本。

## 为什么生成了三个音频文件夹

[training/generate_unist_eval_audio.py](../../training/generate_unist_eval_audio.py) 中固定创建了三个可能的输出目录：

```text
wav/
source_wav/
reference_wav/
```

含义如下：

| 目录 | 内容 | 来源 token | 用途 |
| --- | --- | --- | --- |
| `wav/` | 模型生成音频 | 模型 generate 后抽出的 semantic token | 听当前 checkpoint 的生成结果 |
| `source_wav/` | 源语音重建 | `record["source_bicodec"]` + `record["bicodec_global"]` | 检查输入源语音/说话人 token 是否正常 |
| `reference_wav/` | 参考答案重建 | `reference_bicodec_values(record, mode)` | 和模型生成音频对照 |

这次脚本传了 `SAVE_SOURCE_AUDIO=1`，而 `generate_unist_audio_eval.sh` 默认总是加 `--save-reference-audio`，所以三个目录都会生成。

本次 `summary.json` 显示：

```json
{
  "failed": 0,
  "generated_audio": 3,
  "reference_audio": 3,
  "source_audio": 3,
  "total": 3
}
```

说明 3 条 validation 样本都成功生成了模型音频、源语音重建和参考语音重建。

## 为什么两个文件夹的音频一样

完全相同的是：

```text
source_wav/
reference_wav/
```

我用 `md5sum` 对比过，三条样本一一相同：

```text
bed39e294e9ab54c156187e401a5b078  source_wav/00000_tts_magicdata_0000000001.wav
bed39e294e9ab54c156187e401a5b078  reference_wav/00000_tts_magicdata_0000000001.wav

6f9346f26f6e993b86bc20bf20ee27fd  source_wav/00001_tts_magicdata_0000000002.wav
6f9346f26f6e993b86bc20bf20ee27fd  reference_wav/00001_tts_magicdata_0000000002.wav

42baa6917356f36bbe26a4799fec5c69  source_wav/00002_tts_magicdata_0000000005.wav
42baa6917356f36bbe26a4799fec5c69  reference_wav/00002_tts_magicdata_0000000005.wav
```

原因在 `reference_bicodec_values()`：

```python
def reference_bicodec_values(record, mode):
    if mode == "tts":
        return record["source_bicodec"]
    return record["target_bicodec"]
```

也就是说：

- `mode=tts` 时，reference 也是源语音 semantic token。
- `mode=quality/performance/direct_s2st` 时，reference 才会切到目标语音 semantic token。

所以这不是保存 bug，而是这次评测模式导致的预期结果。

## 为什么没有得到翻译后的语音

### 1. 这次运行的是 `tts` mode

本次实际运行脚本 [scripts/export_and_generate_qwen0p5b_phase1_audio_eval.sh](../../scripts/export_and_generate_qwen0p5b_phase1_audio_eval.sh) 中默认：

```bash
MODES="${MODES:-tts}"
```

`results.jsonl` 中三条记录也都显示：

```json
{"mode":"tts","src_lang":"cmn","tgt_lang":"eng","transcription_ref":"我想用百度搜索短信","translation_ref":"I want to use Baidu to search for SMS."}
{"mode":"tts","src_lang":"cmn","tgt_lang":"eng","transcription_ref":"水费还可以用多长时间","translation_ref":"How much longer can the water fee be used?"}
{"mode":"tts","src_lang":"cmn","tgt_lang":"eng","transcription_ref":"我现在哪","translation_ref":"Where am I now?"}
```

注意：虽然 row 里有英文 `translation_ref`，但 `tts` prompt 并不会使用它。

### 2. `tts` prompt 明确是源语言 TTS，不是翻译

[training/sample_builders.py](../../training/sample_builders.py) 的 `build_tts_sample()` 逻辑是：

1. 读取 `transcription`，也就是源语言转写文本。
2. 使用 `src_lang`，不是 `tgt_lang`。
3. 目标 semantic token 是 `source_bicodec`，不是 `target_bicodec`。

对应关系：

```text
prompt:
  <|task_tts|> <src_lang> global_tokens
  <|start_content|> source transcription <|end_content|>
  <|write_generate|> <src_lang> <speed> <|start_semantic_token|>

target:
  source_bicodec semantic tokens
```

因此，TTS 模式的正确目标是“按源语言文本和源说话人生成源语言语音”，不是“把源语音翻译成目标语音”。

### 3. Phase 1 checkpoint 还不是最终 S2ST checkpoint

当前 checkpoint 是：

```text
checkpoints/uniss_qwen0p5b_phase1_unist13_full
```

导出后用于推理的是：

```text
checkpoints/exported_hf/qwen0p5b_phase1_unist13_full_hf
```

Phase 1 在本复现计划中主要训练 alignment 任务：

- ASR：源语音到源文本。
- S2TT：源语音到目标文本。
- TTS：源文本到源语音。
- 如启用 MT proxy，则是文本到文本翻译。

真正的 S2ST 语音翻译模式，即 `quality`、`performance`、`direct_s2st`，是在 Phase 2 / Phase 3 中训练或强化的。只拿 Phase 1 checkpoint 做 `tts` 试听，不能期待它输出英文翻译语音。

### 4. 当前 `wav/` 还出现了 max token 打满现象

`results.jsonl` 中三条 `semantic_token_count` 都是 `1500`，正好等于脚本默认的 `MAX_NEW_TOKENS=1500`。这说明模型大概率没有正常生成 EOS，而是一路生成到最大长度。

这和官方 `vllm_example.py` 里 `max_tokens=1500` 的注释一致：1500 token 约对应小于 30 秒音频。当前 Phase 1 小模型输出打满上限，说明它还不是稳定可用的 S2ST/TTS 推理模型，只能证明导出、generate、BiCodec decode 链路跑通。

## 官方推理和当前评测是否“一样”

答案：不是完全一样，但有兼容关系。

相同点：

- 都使用 UniSS 的特殊 token 体系。
- 都使用 `UniSSTokenizer` 的 BiCodec decode 把 semantic token 转为 wav。
- `quality`、`performance`、`direct_s2st` 的 prompt/target 结构在本地 `sample_builders.py` 中刻意对齐了官方 `uniss/cli/prompt.py`。

不同点：

- 官方推理从真实 wav 开始，运行时 tokenize；当前评测从 UniST parquet 的预处理 token 开始。
- 官方推理默认使用完整 `cmots/UniSS` 权重；当前评测使用本地 Phase 1 reproduction checkpoint。
- 官方 S2ST 示例跑 `Quality` / `Performance`；当前这次跑的是 `tts`。
- 官方推理只输出最终翻译 wav 和结果 json；当前评测额外保存 source/reference reconstruction，方便训练调试。

因此，这次生成目录更像“训练中间 checkpoint 的音频 sanity check”，不是“官方 S2ST demo 的等价复现”。

## 正确验证翻译语音的方式

如果目标是验证 speech-to-speech translation，需要换成以下条件：

1. 使用 Phase 2 或 Phase 3 checkpoint，或者官方完整 `cmots/UniSS` 权重。
2. 运行 `quality`、`performance` 或 `direct_s2st`，不要用 `tts`。
3. 保留 `source_wav/` 和 `reference_wav/`，用于对照：
   - `source_wav/`：源语言输入语音。
   - `reference_wav/`：目标语言参考语音，应该来自 `target_bicodec`。
   - `wav/`：模型生成的目标语言翻译语音。

示例命令形态：

```bash
EVAL_CUDA_VISIBLE_DEVICES=3 \
HF_CHECKPOINT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase2_or_phase3_hf \
SPLIT=dev \
LIMIT_RECORDS=3 \
MODES="quality performance direct_s2st" \
SAVE_SOURCE_AUDIO=1 \
OUTPUT_DIR=/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase2_s2st_validation_audio \
scripts/generate_unist_audio_eval.sh
```

预期结果：

- `source_wav/` 和 `reference_wav/` 不应再相同。
- `reference_wav/` 应该是目标语言参考语音。
- `wav/` 应该是模型生成的目标语言翻译语音。
- `results.jsonl` 中 `mode` 应为 `quality`、`performance` 或 `direct_s2st`，不应全是 `tts`。

## 建议修正

为了避免后续混淆，建议做两个小改动：

1. 将 Phase 1 试听输出目录命名得更明确，例如：

```text
eval_outputs/qwen0p5b_phase1_tts_reconstruction_validation_audio
```

2. 给 `export_and_generate_qwen0p5b_phase1_audio_eval.sh` 增加注释或显式变量说明：

```bash
# Phase 1 audio eval defaults to TTS reconstruction sanity check.
# It is not an S2ST translation evaluation.
MODES="${MODES:-tts}"
```

如果要做 S2ST 评测，应另建脚本，例如：

```text
scripts/export_and_generate_qwen0p5b_phase2_s2st_audio_eval.sh
```

并默认：

```bash
MODES="${MODES:-quality performance}"
```

## 最终判断

刚刚生成的音频没有翻译成英文，不是因为 UniSS 官方推理本身不支持 S2ST，也不是因为 `source_wav/` / `reference_wav/` 保存错了。

真正原因是：我们刚跑的是 Phase 1 checkpoint 的 `tts` reconstruction sanity check。它只验证了“checkpoint 导出 -> HF generate -> 抽取 BiCodec semantic token -> UniSSTokenizer decode -> 保存 wav”这条链路能跑通；它没有要求模型执行 S2ST 翻译，也没有使用 target speech token 作为 reference。

要得到翻译后的语音，下一次应使用 Phase 2/Phase 3 S2ST checkpoint，并将评测 mode 改成 `quality`、`performance` 或 `direct_s2st`。
