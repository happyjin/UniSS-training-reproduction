# UniSS Training Reproduction Plan

本文档整理 UniSS 论文、附录、公开 GitHub/Hugging Face 信息和当前仓库推理代码，目标是给出一个可以落地到 Megatron-LM 的训练脚本设计计划。由于官方训练代码尚未开放，本文把严格论文复现和公开数据可复现版本分开说明。

最后更新：2026-07-15 12:15:31 UTC

## 1. 复现边界与依据

### 1.1 已公开信息

- 论文：UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice, arXiv 2509.21144.
- 代码仓库：`https://github.com/cmots/UniSS`。当前仓库主要提供推理和 speech tokenizer 封装，没有训练代码。
- GitHub issue #2 作者回复：UniSS 训练流程与普通 text LLM 一致；完成数据预处理后可用 `transformers`、`LLaMA-Factory` 或 `Megatron`；官方具体训练脚本未来可能开放，当前优先开放数据。
- 公开模型：`https://huggingface.co/cmots/UniSS`。
- 公开数据：`https://huggingface.co/datasets/cmots/UniST`。

### 1.2 训练复现的两个层级

1. 论文严格复现：
   - 使用论文列出的 77.1k 小时 speech alignment 数据、WMT17 的 2.3B MT tokens、UniST General 44.8k 小时、UniST High-quality 19.8k 小时。
   - 按论文数据合成流程重新执行 ASR 清洗、Qwen2.5-72B 翻译、SparkTTS 合成、BiCodec/GLM4 tokenization。
   - 训练环境为 16 张 NVIDIA H800 80G，Megatron-LM，三阶段总计约 6 天。

2. 公开数据可复现版本：
   - 直接使用 HF `cmots/UniST` 的 codec-token Parquet。
   - 公开 UniST 数据不是原始音频，而是 tokenized training data，字段包括 `transcription`、`translation`、`source_glm`、`target_glm`、`source_bicodec`、`target_bicodec`、`bicodec_global` 等。
   - Phase 2/3 可以直接构造 S2ST 样本；Phase 1 可从公开 UniST 派生 ASR/TTS/S2TT 样本，再混入 WMT17 MT 数据。
   - 这个版本不能完全等价于论文 Phase 1 的 77.1k 小时 speech alignment 数据，但工程上最可执行。

### 1.3 当前实现进度

截至 2026-07-15，当前 repo 已经落地并验证了公开 UniST -> Megatron packed JSONL 的核心链路：

- `training/constants_uniss.py`：UniSS vocab/token ID 常量与合法性校验。
- `training/sample_builders.py`：ASR、S2TT、TTS、MT、Quality、Performance、Direct S2ST prompt/target 构造。
- `training/prepare_phase1_alignment.py`：读取公开 UniST parquet，派生 Phase 1 ASR/S2TT/TTS speech alignment JSONL 样本。
- `training/build_mt_wmt17.py`：读取已下载的 WMT17-style 平行文本或 JSONL，生成 Phase 1 MT JSONL 样本。
- `training/prepare_unist_s2st.py`：读取 HF `cmots/UniST` parquet，生成 Phase 2/3 S2ST JSONL 样本。
- `training/mix_sample_jsonl.py`：按整数比例确定性混合样本 JSONL，例如 Phase 2 的 `UniST S2ST : Phase1 replay = 2 : 1`。
- `training/pack_sequences.py`：next-token shift、loss mask 对齐、position reset、packed sample boundary。
- `training/megatron_uniss_dataset.py`：把 packed JSONL 转成 Megatron `pretrain_gpt.py` 可消费的 tensors。
- `training/pretrain_uniss_megatron.py`：复用 Megatron-LM `pretrain_gpt.py` 的 model/forward/loss/training loop，只替换 UniSS dataset provider。
- `scripts/download_hf_assets.sh`：使用正确 conda env 的 `huggingface-cli` 下载 UniSS、Qwen2.5、UniST，显式设置 HF cache/tmp 到 `/opt/dlami/nvme/jasonleeeli`。
- `scripts/train_phase1.sh`、`scripts/train_phase2.sh`、`scripts/train_phase3.sh`：按论文 Implementation Details 设置 Megatron 启动参数；支持 `--dry-run` 打印命令。

已执行验证：

```bash
python training/prepare_unist_s2st.py --help
python training/prepare_phase1_alignment.py --help
python training/build_mt_wmt17.py --help
python training/mix_sample_jsonl.py --help
python training/pack_sequences.py --help
python training/pretrain_uniss_megatron.py --help
python -m unittest discover training/tests -v
python -m py_compile training/*.py training/tests/*.py
bash -n scripts/train_phase1.sh
bash -n scripts/train_phase2.sh
bash -n scripts/train_phase3.sh
scripts/download_hf_assets.sh --dry-run uniss
scripts/train_phase1.sh --dry-run --exit-duration-in-mins 1
scripts/train_phase2.sh --dry-run --exit-duration-in-mins 1
scripts/train_phase3.sh --dry-run --exit-duration-in-mins 1
```

当前验证结果：42 个单元测试通过，所有训练工具脚本通过语法编译；三个 Megatron phase 启动脚本通过 shell 语法检查和 dry-run 命令构造检查。`data/raw/UniST` 下载已在 `tmux` session `unist_download` 中后台执行，数据目录已被 `.gitignore` 排除。

## 2. 模型与 tokenizer 设计

### 2.1 Backbone

UniSS 基于 Qwen2.5-1.5B-Instruct，不修改 transformer 架构，只扩展词表并将 speech token 当作普通离散 token 建模。

公开 UniSS `config.json` 的关键参数：

| 参数 | 值 |
| --- | --- |
| `model_type` | `qwen2` |
| `architectures` | `Qwen2ForCausalLM` |
| `hidden_size` | 1536 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 12 |
| `num_key_value_heads` | 2 |
| `intermediate_size` | 8960 |
| `max_position_embeddings` | 32768 |
| `rope_theta` | 1000000.0 |
| `rms_norm_eps` | 1e-6 |
| `torch_dtype` | `bfloat16` |
| `tie_word_embeddings` | true |
| `vocab_size` | 180407 |

训练初始化策略：

- 从 `Qwen/Qwen2.5-1.5B-Instruct` 初始化 base weights。
- 将 tokenizer 扩展到 `vocab_size=180407`。
- 原 Qwen token embedding / LM head 保留预训练权重。
- 新增 speech/control token embedding 使用正态初始化，建议 `std=0.02`，与 Qwen initializer 一致。
- 因 `tie_word_embeddings=true`，如果 Megatron 实现中 embedding 与 output layer 绑定，需要确保 resize 后继续共享。

### 2.2 Speech tokenizer

UniSS 使用三类 speech token：

- Speaker tokens `S_spk`：来自 BiCodec global encoder，固定 32 个 token，表示说话人音色、情绪、韵律等全局属性。
- Linguistic tokens `S_ling`：来自 GLM-4 speech tokenizer，约 12.5 tokens/s，用于源语音内容理解。
- Semantic tokens `S_sem`：来自 BiCodec semantic tokenizer，约 50 tokens/s，用于目标语音生成，可与 speaker tokens 拼接后用 BiCodec decoder 还原 waveform。

当前仓库中推理 tokenization 入口：

- `uniss/tokenizer.py` 的 `UniSSTokenizer.from_pretrained` 加载 `glm4_tokenizer/` 与 `bicodec/`。
- `uniss/cli/extract_speech_token.py` 的 `tokenize_speech` 返回 `(glm4_tokens, bicodec_tokens)`。
- `bicodec_tokens` 的前 32 个是 global tokens，后续是 semantic tokens。

### 2.3 Token ID 映射

建议训练预处理直接用 token ID 构造样本，避免为每个 speech token 拼字符串再 tokenize。

| token 类型 | ID 范围 |
| --- | --- |
| Qwen 原词表和原特殊 token | `0..151664` |
| `<|bicodec_global_i|>` | `151665 + i`, `i=0..4095` |
| `<|bicodec_semantic_i|>` | `155761 + i`, `i=0..8191` |
| `<|glm_semantic_i|>` | `163953 + i`, `i=0..16383` |
| `<|speed_i|>` | `180337 + i`, `i=0..34` |

关键 control token：

| token | ID |
| --- | --- |
| `<|endoftext|>` / pad | 151643 |
| `<|im_start|>` | 151644 |
| `<|im_end|>` / eos | 151645 |
| `<|cmn|>` | 180372 |
| `<|eng|>` | 180373 |
| `<|task_tts|>` | 180375 |
| `<|task_asr|>` | 180376 |
| `<|task_s2s_translation|>` | 180379 |
| `<|task_s2t_translation|>` | 180380 |
| `<|task_t2t_translation|>` | 180381 |
| `<|start_content|>` | 180386 |
| `<|start_global_token|>` | 180387 |
| `<|start_semantic_token|>` | 180388 |
| `<|start_glm_token|>` | 180389 |
| `<|end_content|>` | 180390 |
| `<|end_global_token|>` | 180391 |
| `<|end_semantic_token|>` | 180392 |
| `<|end_glm_token|>` | 180393 |
| `<|write_generate|>` | 180396 |
| `<|slow_mode|>` | 180399 |
| `<|balance_mode|>` | 180400 |
| `<|fast_mode|>` | 180401 |

速度 token 计算沿用当前推理代码：

```python
speed_x = int((speed - 0.1) / 0.1)
```

论文中一般使用 1.0 维持源/目标时长一致，因此常用 `<|speed_9|>`。

## 3. 数据准备计划

### 3.1 论文数据构建流程

论文中的 UniST 生成流程：

1. 从公开中文/英文 TTS/ASR/S2ST 语音数据集中收集 `(X_src, T_src)`。
2. 所有音频重采样到 16 kHz、mono。
3. 使用 Paraformer 对源语音重新识别，计算 re-ASR text 与原 transcription 的 WER。
4. 丢弃源端 WER `> 0.05` 的样本。
5. 使用 Qwen2.5-72B-Instruct 将 `T_src` 翻译到目标语言，得到 `T_tgt`。
6. 清理翻译模型常见前缀、解释性 note、多余换行和空白。
7. 过滤中英混杂或目标语种不纯的翻译。
8. 使用 SparkTTS，以 `X_src` 为 speaker/style prompt，合成目标语音 `Y_tgt`。
9. 对 `Y_tgt` 执行 ASR，与 `T_tgt` 比较，丢弃目标端 WER `> 0.01` 的样本。
10. 用 duration ratio filter：
    - UniST General：保留目标语音时长在源语音 `[0.5, 2.0]` 倍内。
    - UniST High-quality：额外 VAD 去除首尾静音，使用更严格 `[0.7, 1.5]`。
11. 对源/目标音频抽取 GLM4 和 BiCodec tokens。
12. 存储完整样本：
    - `source_path`, `target_path`
    - `transcription`, `translation`
    - `source_glm`, `target_glm`
    - `source_bicodec`, `target_bicodec`
    - `bicodec_global`
    - `src_lang`, `tgt_lang`, `duration_ratio`, `split`, `dataset_name`

### 3.2 论文列出的源语音数据

附录列出的 source corpora：

- AISHELL-3
- CoVoST2
- Common Voice EN/ZH, paper 使用 Common Voice 4.0
- CVSS-T
- Dailytalk
- Emilia
- FLEURS
- Gigaspeech
- Hi-Fi TTS
- HQ-Conversations
- LibriSpeech
- LibriTTS-R
- MAGICDATA
- NCSSD-C / NCSSD-R
- VCTK
- WenetSpeech4TTS

严格复现需要为这些数据写 source adapter，把不同格式统一成：

```json
{
  "id": "dataset_name/split/sample_id",
  "dataset_name": "librispeech",
  "split": "train",
  "wav_path": "...",
  "speaker_id": "...",
  "src_lang": "eng",
  "tgt_lang": "cmn",
  "transcription": "...",
  "duration": 3.42
}
```

### 3.3 翻译提示词

中文到英文：

```text
Please translate the following text into English (Note that, aside from the translation, no other responses or explanations should be provided.):
```

英文到中文使用对应中文提示词，要求只输出译文，不输出解释。

翻译清洗规则：

- 删除 `Sure, here is the translation:`、`Here is the translation:` 等英文前缀。
- 删除中文对应前缀，如“当然，以下是翻译：”。
- 删除 `Note:`、`注：` 及后续解释性内容。
- 去除换行、多余空白、首尾引号。
- 过滤目标语言中明显混入另一语言的大段内容。
- 过滤空译文、重复译文、异常短/异常长译文。

### 3.4 公开 UniST 数据

HF `cmots/UniST` README schema：

- `id`: sample identifier
- `transcription`: source transcription
- `translation`: Qwen translation, fallback to `trans_text`
- `source_glm`, `target_glm`: GLM token lists
- `source_bicodec`, `target_bicodec`: BiCodec semantic token lists
- `bicodec_global`: source BiCodec global token list
- `dataset_name`, `src_lang`, `tgt_lang`, `split`
- `*_len`, `duration_ratio`: audit fields

公开 README 过滤条件：

- 所有四类 token files 存在。
- source/target BiCodec global tokens 匹配。
- metadata 完整。
- WER 低于阈值。
- `0.7 <= len(source_glm) / len(target_glm) <= 1.3`。

公开 `merge_summary.json` 显示：

- `train`: 198 个 parquet，约 19.8M rows。
- 还有 `dev/test/clean_dev/clean_test/dev_clean/dev_other/test_clean/test_other` 等 split。

注意：`export_summary.json` 与 `merge_summary.json` row count 有差异，原因可能是后续 incremental merge。实际脚本应读取当前 HF dataset files，并以 parquet row count 重新统计为准。

### 3.5 UniSS/UniST 下载计划

训练前需要下载三类公开资源：

1. UniSS 推理模型与 tokenizer side assets：
   - 主模型 tokenizer：`tokenizer.json`、`vocab.json`、`merges.txt`、`tokenizer_config.json`。
   - GLM4 tokenizer：`glm4_tokenizer/`。
   - BiCodec tokenizer/decoder：`bicodec/`。
   - 当前仓库已有 `download_weight.py`，会下载 `cmots/UniSS` 到 `pretrained_models/UniSS`。

2. UniST tokenized training data：
   - HF dataset `cmots/UniST`。
   - parquet 文件包含 Phase 2/3 需要的 codec-token 字段。
   - 数据大小约 70GB+，应下载到数据盘，不建议放进 git repo。

3. Qwen2.5-1.5B-Instruct base weights：
   - 用于从 base LLM 初始化训练。
   - 如果只是继续训练公开 `cmots/UniSS`，可以改为从 UniSS checkpoint load；如果复现论文从 Qwen 开始，应下载 `Qwen/Qwen2.5-1.5B-Instruct`。

推荐目录：

```text
pretrained_models/
  UniSS/
  Qwen2.5-1.5B-Instruct/
data/
  raw/
    UniST/
    WMT17/
  processed/
  megatron/
checkpoints/
```

下载命令计划：

```bash
# 安装依赖
pip install -U "huggingface_hub[cli]" datasets pyarrow

# 下载 UniSS 模型、tokenizer、GLM4 tokenizer、BiCodec
python download_weight.py

# 或者用 huggingface-cli 显式下载
huggingface-cli download cmots/UniSS \
  --local-dir pretrained_models/UniSS \
  --local-dir-use-symlinks False \
  --resume-download

# 下载 UniST 数据集 parquet
huggingface-cli download cmots/UniST \
  --repo-type dataset \
  --local-dir data/raw/UniST \
  --local-dir-use-symlinks False \
  --resume-download

# 下载 Qwen2.5 base checkpoint
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir pretrained_models/Qwen2.5-1.5B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download
```

下载完成后的检查：

```bash
python - <<'PY'
from pathlib import Path
for p in [
    "pretrained_models/UniSS/config.json",
    "pretrained_models/UniSS/tokenizer.json",
    "pretrained_models/UniSS/glm4_tokenizer/config.json",
    "pretrained_models/UniSS/bicodec/config.yaml",
    "pretrained_models/Qwen2.5-1.5B-Instruct/config.json",
]:
    print(p, Path(p).exists())

parquets = sorted(Path("data/raw/UniST").glob("train-*.parquet"))
print("UniST train shards:", len(parquets))
PY
```

注意：

- HF dataset/model 下载可能需要较长时间，建议使用 `tmux` 或 batch job。
- `cmots/UniST` license 是 `cc-by-nc-4.0`，训练用途需要遵守非商业许可。
- 若网络不稳定，优先使用 `huggingface-cli download --resume-download` 分片续传。

### 3.6 数据落盘格式

建议分两层落盘。

第一层：normalized JSONL 或 Arrow/Parquet sample records：

```json
{
  "id": "...",
  "src_lang": "eng",
  "tgt_lang": "cmn",
  "transcription": "...",
  "translation": "...",
  "source_glm": [12, 34, 56],
  "target_glm": [78, 90],
  "source_bicodec": [10, 11, 12],
  "target_bicodec": [20, 21, 22],
  "bicodec_global": [1, 2, "... 32 tokens"],
  "duration_ratio": 1.03,
  "dataset_name": "emilia_en",
  "split": "train"
}
```

第二层：Megatron-ready packed binary：

```json
{
  "input_ids": "uint32/int64 vector length <= 18000",
  "labels": "int64 vector, prompt positions = -100",
  "loss_mask": "float/bool vector",
  "position_ids": "int32 vector reset per sample",
  "sample_boundaries": [[0, 423], [423, 911], "..."],
  "task_ids": ["quality", "performance", "..."],
  "source_ids": ["sample_id_a", "sample_id_b", "..."]
}
```

如果使用 Megatron IndexedDataset，需要额外保存 packed metadata，或者在 dataset class 中并行读取 `.bin/.idx` 与 sidecar `.npy/.jsonl` metadata。

## 4. 训练样本模板

当前推理代码在 `uniss/cli/prompt.py` 中定义了任务 token 映射：

- Quality: `<|task_s2s_translation|><|slow_mode|>`
- Performance: `<|task_s2s_translation|><|balance_mode|>`
- Direct S2ST: `<|task_s2s_translation|><|fast_mode|>`
- ASR: `<|task_asr|>`
- S2TT: `<|task_s2t_translation|>`
- TTS: `<|task_tts|>`

下文用伪 token 表示模板，实际脚本应直接构造 token ID。

### 4.1 通用片段

Global speaker tokens：

```text
<|start_global_token|>
  <|bicodec_global_{g0}|> ... <|bicodec_global_{g31}|>
<|end_global_token|>
```

GLM linguistic tokens：

```text
<|glm_semantic_{x0}|> ... <|glm_semantic_{xN}|>
```

BiCodec semantic tokens：

```text
<|bicodec_semantic_{y0}|> ... <|bicodec_semantic_{yM}|>
```

Language token：

- English: `<|eng|>`
- Mandarin: `<|cmn|>`

### 4.2 Phase 1: Speech-Text Alignment

Phase 1 任务：ASR、TTS、S2TT、MT。

#### ASR

Prompt：

```text
<|task_asr|>{src_lang}
<|start_global_token|>{global_tokens}<|end_global_token|>
{source_glm}
<|write_generate|>{src_lang}<|start_content|>
```

Target：

```text
transcription <|end_content|><|im_end|>
```

Labels：

- prompt 全部 `-100`
- transcription、`<|end_content|>`、`<|im_end|>` 计算 loss

#### S2TT

Prompt：

```text
<|task_s2t_translation|>{tgt_lang}
<|start_global_token|>{global_tokens}<|end_global_token|>
{source_glm}
<|write_generate|>{tgt_lang}<|start_content|>
```

Target：

```text
translation <|end_content|><|im_end|>
```

#### TTS

Prompt：

```text
<|task_tts|>{src_lang}
<|start_global_token|>{global_tokens}<|end_global_token|>
<|start_content|>transcription<|end_content|>
<|write_generate|>{src_lang}<|speed_9|><|start_semantic_token|>
```

Target：

```text
source_bicodec_semantic <|end_semantic_token|><|im_end|>
```

#### MT

Prompt：

```text
<|task_t2t_translation|>{tgt_lang}
<|start_content|>source_text<|end_content|>
<|write_generate|>{tgt_lang}<|start_content|>
```

Target：

```text
target_text <|end_content|><|im_end|>
```

### 4.3 Phase 2: S2ST with CoT

Phase 2 任务：Quality、Performance、Direct S2ST，并混入 Phase 1 数据。论文写法是 new data mixed with Phase 1 data at a 2:1 ratio。计划中按 `UniST S2ST : Phase1 alignment = 2 : 1` 实现。

#### Quality mode

Prompt：

```text
<|task_s2s_translation|><|slow_mode|>{tgt_lang}
<|start_global_token|>{global_tokens}<|end_global_token|>
{source_glm}
<|write_generate|>
<|task_asr|>{src_lang}<|speed_9|><|start_content|>
```

Target：

```text
transcription <|end_content|>
<|task_s2t_translation|>{tgt_lang}<|speed_9|><|start_content|>
translation <|end_content|>
<|start_semantic_token|>
target_bicodec_semantic
<|end_semantic_token|><|im_end|>
```

说明：

- 这是 listen -> translate -> speak 的完整 CoT。
- 当前推理代码 `process_output(..., quality_mode=True)` 通过两个 `<|end_content|>` 分割 transcription 和 translation，因此训练 target 中必须保留两个 content block。

#### Performance mode

Prompt：

```text
<|task_s2s_translation|><|balance_mode|>{tgt_lang}
<|start_global_token|>{global_tokens}<|end_global_token|>
{source_glm}
<|write_generate|>
<|task_s2t_translation|>{tgt_lang}<|speed_9|><|start_content|>
```

Target：

```text
translation <|end_content|>
<|start_semantic_token|>
target_bicodec_semantic
<|end_semantic_token|><|im_end|>
```

#### Direct S2ST / fast mode

Prompt：

```text
<|task_s2s_translation|><|fast_mode|>{tgt_lang}
<|start_global_token|>{global_tokens}<|end_global_token|>
{source_glm}
<|write_generate|><|fast_mode|>{tgt_lang}<|speed_9|><|start_semantic_token|>
```

Target：

```text
target_bicodec_semantic <|end_semantic_token|><|im_end|>
```

### 4.4 Phase 3: Annealing / Refinement

Phase 3 只使用 UniST High-quality，并训练完整 S2ST task 的 CoT prompting modes：

- Quality mode
- Performance mode

不再混 Direct S2ST，也不混 Phase 1 alignment 数据。

## 5. Script 设计

建议新增目录：

```text
training/
  constants_uniss.py
  build_tokenizer_and_init.py
  convert_qwen2_hf_to_megatron.py
  prepare_phase1_alignment.py
  prepare_unist_s2st.py
  build_mt_wmt17.py
  mix_sample_jsonl.py
  pack_sequences.py
  pretrain_uniss_megatron.py
  export_megatron_to_hf.py
scripts/
  train_phase1.sh
  train_phase2.sh
  train_phase3.sh
  eval_cvss_t.sh
  export_megatron_to_hf.sh
```

### 5.1 `constants_uniss.py`

职责：

- 固化 token ID offset。
- 提供 `encode_global_tokens`、`encode_glm_tokens`、`encode_bicodec_semantic_tokens`。
- 提供 control token ID 常量。

接口草案：

```python
QWEN_BASE_VOCAB_END = 151664
BICODEC_GLOBAL_OFFSET = 151665
BICODEC_GLOBAL_SIZE = 4096
BICODEC_SEMANTIC_OFFSET = 155761
BICODEC_SEMANTIC_SIZE = 8192
GLM_SEMANTIC_OFFSET = 163953
GLM_SEMANTIC_SIZE = 16384
SPEED_OFFSET = 180337

TOKEN_CMN = 180372
TOKEN_ENG = 180373
TOKEN_TASK_TTS = 180375
TOKEN_TASK_ASR = 180376
TOKEN_TASK_S2S_TRANSLATION = 180379
TOKEN_TASK_S2T_TRANSLATION = 180380
TOKEN_TASK_T2T_TRANSLATION = 180381
TOKEN_START_CONTENT = 180386
TOKEN_START_GLOBAL = 180387
TOKEN_START_SEMANTIC = 180388
TOKEN_END_CONTENT = 180390
TOKEN_END_GLOBAL = 180391
TOKEN_END_SEMANTIC = 180392
TOKEN_WRITE_GENERATE = 180396
TOKEN_SLOW_MODE = 180399
TOKEN_BALANCE_MODE = 180400
TOKEN_FAST_MODE = 180401
TOKEN_EOS = 151645
TOKEN_PAD = 151643
```

需要校验：

- global token 必须在 `[0, 4095]`。
- bicodec semantic token 必须在 `[0, 8191]`。
- GLM token 必须在 `[0, 16383]`。
- `bicodec_global` 长度必须为 32。

### 5.2 `build_tokenizer_and_init.py`

职责：

- 从 `Qwen/Qwen2.5-1.5B-Instruct` 或公开 `cmots/UniSS` tokenizer 构造扩展 tokenizer。
- 如果从 base Qwen 构造，需要添加所有 speech/control tokens，保证最终 ID 与公开 UniSS 对齐。
- resize model embedding 到 180407。
- 初始化新增 embedding。
- 保存 HF intermediate checkpoint，作为 Megatron 转换输入。

实际更稳的做法：

- 直接下载 `cmots/UniSS` 的 `tokenizer.json/vocab.json/merges.txt/tokenizer_config.json`。
- 使用 base Qwen2.5-1.5B weights 初始化 model，然后套用 UniSS tokenizer。
- 这样能保证 token ID 与推理仓库完全一致。

### 5.3 `prepare_phase1_alignment.py`

职责：

- 读取论文源语音数据或公开 UniST 派生数据。
- 生成 ASR/TTS/S2TT 训练样本。
- 可选执行原始音频 tokenization。

当前已实现输入模式：

1. `--input`：公开 UniST token parquet 或 glob，直接派生 ASR/S2TT/TTS。
2. `--tasks`：可选 `asr s2tt tts` 子集；如果启用 `tts`，输入必须包含 `source_bicodec`。

严格论文复现中的原始音频 manifest tokenization 尚未实现；该部分需要先完成 ASR 清洗、SparkTTS 合成、GLM4/BiCodec tokenization 后，再落成同 schema 的 parquet/JSONL。

输出：

```text
data/processed/phase1/alignment.jsonl
```

每行包含：

```json
{
  "phase": "phase1",
  "task": "asr",
  "id": "...",
  "prompt_ids": [180376, 180373, "..."],
  "target_ids": [1234, 5678, 180390, 151645],
  "prompt_length": 456,
  "target_length": 32,
  "segment_spans": {"transcription_text": [0, 30]}
}
```

当前可执行命令：

```bash
python training/prepare_phase1_alignment.py \
  --input "data/raw/UniST/train-*.parquet" \
  --tokenizer pretrained_models/UniSS \
  --tasks asr s2tt tts \
  --output data/processed/phase1/alignment.jsonl
```

### 5.4 `build_mt_wmt17.py`

职责：

- 读取已下载的 WMT17 中英 MT text pair；不在脚本内联网下载。
- 使用 UniSS/Qwen tokenizer 编码文本。
- 生成 MT prompt/target。
- 控制 MT token 总量接近论文的 2.3B tokens。

过滤：

- 删除空行。
- 删除长度过长 pair。
- 过滤目标/源语言异常混杂。
- 限制 max sample length，避免单条样本过长影响 packing。

当前可执行命令：

```bash
python training/build_mt_wmt17.py \
  --source-text data/raw/WMT17/train.en \
  --target-text data/raw/WMT17/train.zh \
  --src-lang eng \
  --tgt-lang cmn \
  --tokenizer pretrained_models/UniSS \
  --max-sample-tokens 18000 \
  --output data/processed/phase1/mt.jsonl
```

也支持预先整理好的 JSONL：

```bash
python training/build_mt_wmt17.py \
  --input-jsonl data/raw/WMT17/train_pairs.jsonl \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/phase1/mt.jsonl
```

### 5.5 `prepare_unist_s2st.py`

职责：

- 读取 HF `cmots/UniST` parquet。
- 生成 Quality、Performance、Direct S2ST 样本。
- 根据 phase 参数选择 task mix：
  - `phase2`: Quality + Performance + Direct S2ST。
  - `phase3`: Quality + Performance only。
- 可按 `src_lang/tgt_lang` 做双向平衡采样。

建议参数：

```bash
python training/prepare_unist_s2st.py \
  --input "data/raw/UniST/train-*.parquet" \
  --phase phase2 \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/phase2/unist_s2st.jsonl
```

后续需要补的采样参数：

- `--quality-ratio`
- `--performance-ratio`
- `--direct-ratio`
- `--max-unpacked-length`

质量检查：

- `len(bicodec_global) == 32`。
- 所有 token ID 在合法范围。
- `transcription` 和 `translation` 非空。
- `source_glm_len`、`target_bicodec_len` 与实际数组长度一致。
- `src_lang != tgt_lang`。
- 生成后的 target 至少包含一个可学习 token。

### 5.6 `mix_sample_jsonl.py`

职责：

- 在 packing 前按整数比例混合不同来源的 sample JSONL。
- Phase 2 使用 `UniST S2ST : Phase1 replay = 2 : 1`，对应论文的 new data mixed with Phase 1 data at a 2:1 ratio。
- 输出仍然是标准 sample JSONL，只额外添加 `mix_group` 审计字段；`pack_sequences.py` 会忽略该字段。

当前可执行命令：

```bash
python training/mix_sample_jsonl.py \
  --group unist=2:data/processed/phase2/unist_s2st.jsonl \
  --group phase1=1:data/processed/phase1/alignment.jsonl,data/processed/phase1/mt.jsonl \
  --output data/processed/phase2/phase2_mixed.jsonl
```

### 5.7 `pack_sequences.py`

职责：

- 将单条样本 packing 到固定 `seq_length=18000`。
- 避免 padding 浪费。
- 生成 sample boundary，使 packed sequence 内不同样本互不 attention。

核心算法：

1. 对每条 sample 得到：
   - `ids = prompt_ids + target_ids`
   - `labels = ids[1:] + [pad]` 或在 dataset 内 shift。
   - `loss_mask = 0 for prompt positions, 1 for target positions`。
2. 按长度 bucket 排序或 first-fit decreasing。
3. 将多条 sample 拼进一个 18k sequence。
4. 每条 sample 内 position ids 从 0 重新开始。
5. 记录 `sample_boundaries`。
6. 不能让不同 sample 互相 attention。

关键点：

- 如果 Megatron 版本支持 packed sequence / reset attention mask，可用内置参数。
- 如果不支持，需要自定义 dataset/collate，构建 block-diagonal causal mask。
- 绝不能只简单 concatenate 后用普通 causal mask，否则样本之间会互相看到答案，训练污染。

输出建议：

```text
data/megatron/phase1_text_document.bin/.idx
data/megatron/phase1_loss_mask.npy
data/megatron/phase1_position_ids.npy
data/megatron/phase1_boundaries.npy
```

或者直接使用 WebDataset/Arrow + custom Megatron dataset，减少 sidecar 对齐风险。

### 5.8 `pretrain_uniss_megatron.py`

职责：

- 基于 Megatron-LM `pretrain_gpt.py` 改造。
- 加载 packed dataset。
- 支持 `loss_mask`。
- 支持 per-sample boundary attention mask 或 packed sequence metadata。
- 支持 Qwen2 architecture 参数。

损失：

```python
loss = cross_entropy(logits, labels)
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

不要对 prompt、padding 或跨样本 boundary 后的无效位置算 loss。

### 5.9 `export_megatron_to_hf.py`

职责：

- Megatron checkpoint 转 HF `Qwen2ForCausalLM`。
- 保存 `config.json`，`tokenizer.json`，`generation_config.json`。
- 输出目录兼容当前 `infer.py` 和 `vllm_example.py`。
- 同步复制 `glm4_tokenizer/` 与 `bicodec/`，以便 `UniSSTokenizer.from_pretrained` 可直接加载。

验收：

```bash
python infer.py
python vllm_example.py --task Quality --target_language zh --input_path ... --output_path ...
```

### 5.10 Megatron-LM Framework 使用方式

完整实现计划明确使用论文中提到的 Megatron-LM Framework，而不是只用普通 HF Trainer。建议做法是 fork 或 vendor 官方 Megatron-LM：

```text
third_party/
  Megatron-LM/
training/
  pretrain_uniss_megatron.py
  datasets/
    uniss_packed_dataset.py
  checkpointing/
    qwen2_hf_megatron_bridge.py
```

接入点：

1. 以 Megatron-LM `pretrain_gpt.py` 为入口模板。
2. 保留 Megatron 的 distributed 初始化、tensor/pipeline/data parallel、optimizer、lr scheduler、checkpointing。
3. 替换 dataset provider：
   - 原 GPT dataset 只返回连续 text tokens。
   - UniSS dataset 返回 `tokens`、`labels`、`loss_mask`、`position_ids`、`attention_mask` 或 packed metadata。
4. 替换/扩展 model provider：
   - 使用 Megatron 支持的 GPT/Qwen2 decoder-only model。
   - 配置 Qwen2 的 GQA、RMSNorm、RoPE theta、SwiGLU/SiLU。
5. 替换 loss function：
   - 使用 UniSS masked next-token CE。
   - prompt、padding、packed sequence boundary 无 loss。
6. 保留 Megatron checkpoint：
   - Phase 1/2/3 训练中间 checkpoint 都保存为 Megatron format。
   - 最终用转换脚本导出为 HF format，兼容当前 `infer.py`/`vllm_example.py`。

推荐 Megatron 相关脚本：

```bash
# 克隆 Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM third_party/Megatron-LM

# 训练入口使用 Megatron-LM torchrun 方式
PYTHONPATH=third_party/Megatron-LM:$PYTHONPATH \
torchrun --nproc_per_node=16 training/pretrain_uniss_megatron.py ...
```

如果所选 Megatron-LM 版本没有完整 Qwen2 支持：

- 优先尝试 Megatron Bridge / Megatron-Core 的 HF Qwen2 import/export。
- 或使用 Megatron-LM 中已有 LLaMA/Mistral-style decoder 作为基底，补 Qwen2 config 差异。
- 必须验证 HF Qwen2 -> Megatron -> HF round trip 后，同一输入 logits 误差足够小，再启动大规模训练。

### 5.11 Megatron-LM 集成策略

需要 clone Megatron-LM，但不建议把当前 UniSS repo 搬进 Megatron-LM 里重构。当前开源 UniSS 仓库是推理/tokenizer 发布包，不是 Megatron-LM fork；最稳的工程策略是保留当前 repo 作为 UniSS-specific 适配层，把 Megatron-LM 当外部训练引擎。

推荐职责划分：

```text
UniSS 当前 repo
  uniss/                         # 原推理/tokenizer 代码，尽量不动
  training/                      # UniSS -> Megatron 的适配层
    constants_uniss.py           # speech/control token ID
    prepare_unist_s2st.py        # UniST parquet -> task samples
    prepare_phase1_alignment.py  # ASR/TTS/S2TT/MT samples
    pack_sequences.py            # 18k packing + boundary/loss masks
    pretrain_uniss_megatron.py   # Megatron training entry wrapper
    checkpointing/
      convert_hf_qwen2_to_megatron.py
      convert_megatron_to_hf.py
    patches/
      megatron_qwen2.patch
      megatron_packed_uniss.patch

third_party/Megatron-LM
  Megatron 原始框架代码，默认不提交到本 repo
```

训练时通过 `PYTHONPATH` 使用外部 Megatron-LM：

```bash
PYTHONPATH=third_party/Megatron-LM:$PYTHONPATH \
torchrun --nproc_per_node=16 training/pretrain_uniss_megatron.py ...
```

这样做的原因：

1. UniSS-specific 逻辑与 Megatron 训练基础设施解耦：
   - UniSS 负责 prompt/token/data/loss mask/checkpoint conversion。
   - Megatron-LM 负责 distributed training、parallelism、optimizer、scheduler、checkpoint sharding。
2. Megatron-LM 体量很大，直接复制/大改会让 diff 难维护，也不利于跟进上游。
3. 当前 UniSS 推理代码仍保持干净，后续导出 HF checkpoint 后可继续复用 `infer.py` 和 `vllm_example.py`。
4. 如果 Megatron-LM 版本升级，只需要重新验证适配层和少量 patch。

只有下面情况才修改 Megatron-LM 文件：

- 当前 Megatron-LM 没有 Qwen2 architecture/GQA/RoPE theta/RMSNorm 支持。
- dataset/training loop 不支持传入 `loss_mask`、`position_ids`、packed sequence boundary。
- attention mask 无法实现 packed sample 间互相不可见。
- checkpoint converter 不支持 Qwen2 HF <-> Megatron。

如果必须修改 Megatron-LM：

- 优先把改动做成 `training/patches/*.patch`，而不是直接在 `third_party/Megatron-LM` 中留下不可追踪修改。
- 每个 patch 都要配一个验证脚本：
  - Qwen2 forward logits round-trip test。
  - packed attention boundary test。
  - masked CE loss alignment test。
  - small overfit test。
- 在 README 或脚本中记录 Megatron-LM commit hash，确保训练可复现。

推荐记录：

```bash
cd third_party/Megatron-LM
git rev-parse HEAD > ../../training/MEGATRON_COMMIT
git diff > ../../training/patches/local_megatron_changes.patch
```

### 5.12 当前已实现的 Megatron 入口

当前实现采用 `training/pretrain_uniss_megatron.py`，不直接修改 `third_party/Megatron-LM`。入口脚本做法：

- 启动时把当前 repo 和 `third_party/Megatron-LM` 加入 `PYTHONPATH`。
- 导入并复用 Megatron-LM `pretrain_gpt.py` 的 `model_provider`、`forward_step`、`loss_func`、`get_embedding_ranks`。
- 自定义 `train_valid_test_datasets_provider`，返回 `UniSSPackedJsonlDataset`。
- 依赖 Megatron 的 `--sft` packed sequence 路径，让 `cu_seqlens` 进入 `PackedSeqParams`，从而避免 packed samples 互相 attention。
- 保留 Megatron 的 distributed 初始化、optimizer、LR scheduler、checkpointing、tensor/pipeline/data parallel。

当前 dataset item schema：

```python
{
    "tokens": int64[seq_length],
    "labels": int64[seq_length],
    "loss_mask": float32[seq_length],
    "position_ids": int64[seq_length],
    "cu_seqlens": int32[seq_length + 1],
    "max_seqlen": int32[],
}
```

必须传入的 UniSS 参数：

```bash
--sft
--uniss-packed-train data/megatron/phaseX/packed_train.jsonl
--uniss-packed-valid data/megatron/phaseX/packed_valid.jsonl   # 如果 eval_iters > 0
--vocab-size 180407
--seq-length 18000
--global-batch-size 128
```

如果暂时不做验证集评估，需要显式关闭 Megatron 默认 eval：

```bash
--eval-iters 0
```

建议正式复现实验加：

```bash
--uniss-strict-paper-config
```

它会检查 `seq_length=18000` 和 `global_batch_size=128`，对应论文的 18k packing 和 2.304M tokens/global step。

当前限制：

- `context_parallel_size` 暂时限制为 1；如果后续要启用 context parallel，需要补 `cu_seqlens_padded` 与 CP 切分验证。
- dataset 不生成 dense attention mask，因此不能启用 `--create-attention-mask-in-dataloader`。
- 当前环境 Python 是 3.14，入口脚本里包含两个 argparse 兼容 shim，用于绕过当前 Megatron-LM 参数定义中的 `BooleanOptionalAction(type=bool)` 和未转义 `%` help 文案问题。真实训练建议新建独立 conda 环境，使用 Python 3.12、Megatron-LM 要求的 `torch>=2.6`、Transformer Engine/Apex/NCCL 组合。

最小执行链路：

```bash
python training/prepare_unist_s2st.py \
  --input "data/raw/UniST/train-*.parquet" \
  --phase phase2 \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/phase2/unist_s2st.jsonl

python training/pack_sequences.py \
  --input data/processed/phase2/unist_s2st.jsonl \
  --output data/megatron/phase2/packed_train.jsonl \
  --seq-length 18000 \
  --drop-overlong

PYTHONPATH=third_party/Megatron-LM:$PYTHONPATH \
torchrun --nproc_per_node=16 training/pretrain_uniss_megatron.py \
  --sft \
  --uniss-packed-train data/megatron/phase2/packed_train.jsonl \
  --eval-iters 0 \
  --uniss-strict-paper-config \
  ...
```

## 6. Loss 设计

论文训练目标是标准 autoregressive next-token prediction。UniSS 没有额外训练 BiCodec decoder、GLM4 tokenizer 或声学重建 loss；这些 tokenizer/decoder 在 LLM 训练时视为冻结的离散 token 产生器/还原器。因此所有训练阶段的核心 loss 都是 masked token-level cross entropy。

### 6.1 通用 masked CE

对每条样本：

```text
ids = prompt_ids + target_ids
labels[t] = ids[t + 1]
loss_mask[t] = 1 if labels[t] belongs to target_ids else 0
```

训练 loss：

```python
token_loss = cross_entropy(logits[:, :-1], labels[:, :-1], reduction="none")
loss = (token_loss * loss_mask[:, :-1]).sum() / loss_mask[:, :-1].sum()
```

多 GPU 训练时，分子和分母都需要在 data parallel group 上 all-reduce：

```python
global_loss = all_reduce(sum_loss) / all_reduce(num_loss_tokens)
```

不计算 loss 的位置：

- 所有 prompt token。
- padding token。
- packed sequence 中 sample boundary 外的位置。
- 如果使用 block-diagonal attention mask，跨样本不可见；如果用 reset attention，也要保证 boundary 后不 attend boundary 前。

### 6.2 Phase 1 Loss

Phase 1 包含 ASR、S2TT、TTS、MT。每个任务的 loss 都是同一个 CE，只是 target token 类型不同。

#### ASR loss

Prompt 输入 source speaker/global + source GLM linguistic tokens，target 是源语言 transcription text。

```text
L_ASR = - mean_t log P(T_src[t] | prompt_asr, T_src[<t])
```

计算 loss 的 token：

- transcription 的文本 token。
- `<|end_content|>`。
- `<|im_end|>`。

不计算 loss 的 token：

- `<|task_asr|>`、语言 token、global tokens、source GLM tokens、`<|write_generate|>`、`<|start_content|>`。

#### S2TT loss

Prompt 输入 source speaker/global + source GLM linguistic tokens，target 是目标语言 translation text。

```text
L_S2TT = - mean_t log P(T_tgt[t] | prompt_s2tt, T_tgt[<t])
```

计算 loss 的 token：

- translation 的文本 token。
- `<|end_content|>`。
- `<|im_end|>`。

#### TTS loss

Prompt 输入 speaker/global + source text，target 是 source speech 的 BiCodec semantic tokens。

```text
L_TTS = - mean_t log P(S_src_sem[t] | prompt_tts, S_src_sem[<t])
```

计算 loss 的 token：

- `<|bicodec_semantic_i|>` 序列。
- `<|end_semantic_token|>`。
- `<|im_end|>`。

不计算 waveform loss、mel loss、STFT loss 或 speaker embedding loss，因为 LLM 不直接输出 waveform，BiCodec decoder 不参与训练。

#### MT loss

Prompt 输入 source text，target 是 target text。

```text
L_MT = - mean_t log P(T_tgt[t] | prompt_mt, T_tgt[<t])
```

计算 loss 的 token：

- target translation 文本 token。
- `<|end_content|>`。
- `<|im_end|>`。

Phase 1 总 loss 不是把四个 loss 固定相加，而是通过数据采样混合形成：

```text
L_phase1 = E_batch[masked_CE(sample)]
```

如果实现按任务分别采样并手动加权，可写成：

```text
L_phase1 = w_asr L_ASR + w_s2tt L_S2TT + w_tts L_TTS + w_mt L_MT
```

起始权重建议 `0.25/0.25/0.25/0.25`，最终以 token-level sampling 和验证指标调整。

### 6.3 Phase 2 Loss

Phase 2 引入 S2ST CoT，并混入 Phase 1 数据。

#### Quality mode loss

Quality target 包含三段：source transcription、target translation、target semantic tokens。

```text
target_quality = T_src + <|end_content|>
               + task_s2tt/lang/speed/start_content
               + T_tgt + <|end_content|>
               + <|start_semantic_token|>
               + S_tgt_sem + <|end_semantic_token|><|im_end|>
```

Loss：

```text
L_quality = L_quality_asr_text
          + L_quality_translation_text
          + L_quality_semantic
```

实现上不需要拆成三次 backward；同一条序列上用一个 masked CE 即可：

```text
L_quality = - mean_t log P(target_quality[t] | prompt_quality, target_quality[<t])
```

建议日志拆分三项：

- `loss/quality_transcription_ce`
- `loss/quality_translation_ce`
- `loss/quality_semantic_ce`

拆分方法是额外生成 segment mask；训练主 loss 仍用总 mask。

#### Performance mode loss

Performance target 包含 target translation 和 target semantic tokens，不包含 source transcription。

```text
L_performance = - mean_t log P([T_tgt, S_tgt_sem][t] | prompt_performance, previous target)
```

建议日志拆分：

- `loss/performance_translation_ce`
- `loss/performance_semantic_ce`

#### Direct S2ST loss

Direct S2ST target 只有 target semantic tokens。

```text
L_direct = - mean_t log P(S_tgt_sem[t] | prompt_direct, S_tgt_sem[<t])
```

该 loss 只训练语音 token 生成，不训练中间文本 CoT。论文 ablation 显示 Direct S2ST inference 明显较弱，因此 Phase 2 中可低比例采样。

#### Phase 2 mixed loss

论文说明 UniST General 与 Phase 1 data 以 2:1 混合。实现上建议：

```text
L_phase2 = E_batch[
  2/3 * sample_from({Quality, Performance, Direct})
  + 1/3 * sample_from(Phase1 ASR/TTS/S2TT/MT)
]
```

如果用显式权重表达：

```text
L_phase2 = w_q L_quality
         + w_p L_performance
         + w_d L_direct
         + w_align L_phase1_replay
```

推荐起点：

```text
w_q = 0.40 * 2/3
w_p = 0.40 * 2/3
w_d = 0.20 * 2/3
w_align = 1/3
```

### 6.4 Phase 3 Loss

Phase 3 只在 UniST High-quality 上训练完整 S2ST CoT prompting modes：

```text
L_phase3 = w_q L_quality + w_p L_performance
```

默认：

```text
w_q = 0.5
w_p = 0.5
```

如果优先复现 Quality mode 指标，可尝试：

```text
w_q = 0.7
w_p = 0.3
```

但这会牺牲 Performance mode。

### 6.5 Megatron 中应记录的 loss

训练日志不应只记录一个 `lm loss`。建议每步或每 N 步记录：

```text
loss/total
loss/asr_text
loss/s2tt_text
loss/tts_semantic
loss/mt_text
loss/quality_total
loss/quality_transcription_text
loss/quality_translation_text
loss/quality_semantic
loss/performance_total
loss/performance_translation_text
loss/performance_semantic
loss/direct_semantic
tokens/active_loss_tokens
tokens/prompt_tokens
tokens/semantic_tokens
tokens/text_tokens
data/task_mix_actual
data/pack_efficiency
```

这些日志来自同一次 forward 的 segment masks，不需要改变训练目标。它们能帮助判断问题来源：

- text loss 降、semantic loss 不降：语音 token 生成学习不足。
- semantic loss 降、Text-BLEU 差：CoT translation 能力不足或 MT replay 不够。
- Quality transcription loss 高：GLM source tokens 到 text 对齐不足，Phase 1 ASR/S2TT 需要加强。
- pack efficiency 低：packing 策略浪费，global batch 实际 token 数不稳定。

## 7. 三阶段训练配置

论文统一设置：

- Optimizer: AdamW
- `weight_decay=0.1`
- `betas=(0.9, 0.95)`
- global batch size: 2.3M tokens
- sequence packing length: 18k tokens
- global batch size by sequence: 128
- hardware: 16 x NVIDIA H800 80G
- all audio: 16 kHz
- vocab size: 180407
- total training time: approximately 6 days

### 7.1 Batch 与 step 计算

```text
tokens_per_step = 128 sequences * 18000 tokens = 2,304,000 tokens
```

| Phase | 训练 token | 估算 steps |
| --- | ---: | ---: |
| Phase 1 | 32B/epoch * 3 = 96B | 41667 |
| Phase 2 | 55B total | 23872 |
| Phase 3 | 10B total | 4341 |

### 7.2 Phase 1

数据：

- 77.1k hours speech data。
- WMT17 2.3B MT tokens。
- 任务混合：ASR、TTS、S2TT、MT。

训练：

- epochs: 3
- tokens: about 32B per epoch
- LR: `8e-4`
- schedule: constant after warmup
- warmup: 1 epoch, about 13889 steps
- batch: 2.3M tokens

脚本草案：

```bash
torchrun --nproc_per_node=16 training/pretrain_uniss_megatron.py \
  --sft \
  --uniss-packed-train data/megatron/phase1/packed_train.jsonl \
  --uniss-packed-valid data/megatron/phase1/packed_valid.jsonl \
  --uniss-strict-paper-config \
  --vocab-size 180407 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 28 \
  --hidden-size 1536 \
  --ffn-hidden-size 8960 \
  --num-attention-heads 12 \
  --group-query-attention \
  --num-query-groups 2 \
  --seq-length 18000 \
  --max-position-embeddings 32768 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --train-iters 41667 \
  --lr 8e-4 \
  --min-lr 8e-4 \
  --lr-warmup-iters 13889 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --bf16 \
  --use-flash-attn \
  --recompute-activations \
  --save checkpoints/uniss_phase1 \
  --load checkpoints/qwen2_1p5b_uniss_vocab
```

实际 Megatron 参数名需按所用 Megatron-LM 版本微调。

### 7.3 Phase 2

数据：

- UniST General。
- mixed with Phase 1 data at 2:1 ratio。
- S2ST task 包括 Quality、Performance、Direct S2ST。

训练：

- epochs: 1
- total tokens: about 55B
- LR: `2e-4`
- schedule: constant after warmup
- warmup: 5% epoch, about 1194 steps
- batch: 2.3M tokens

脚本草案：

```bash
torchrun --nproc_per_node=16 training/pretrain_uniss_megatron.py \
  --sft \
  --uniss-packed-train data/megatron/phase2_mix/packed_train.jsonl \
  --uniss-packed-valid data/megatron/phase2_mix/packed_valid.jsonl \
  --uniss-strict-paper-config \
  --vocab-size 180407 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --seq-length 18000 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --train-iters 23872 \
  --lr 2e-4 \
  --min-lr 2e-4 \
  --lr-warmup-iters 1194 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --bf16 \
  --use-flash-attn \
  --recompute-activations \
  --load checkpoints/uniss_phase1 \
  --save checkpoints/uniss_phase2
```

### 7.4 Phase 3

数据：

- UniST High-quality only。
- Quality + Performance。

训练：

- epochs: 1
- final checkpoint 选 0.9 epoch 左右。
- total tokens: about 10B
- LR: cosine `5e-5 -> 5e-6`
- warmup: none
- batch: 2.3M tokens

脚本草案：

```bash
torchrun --nproc_per_node=16 training/pretrain_uniss_megatron.py \
  --sft \
  --uniss-packed-train data/megatron/phase3_hq/packed_train.jsonl \
  --uniss-packed-valid data/megatron/phase3_hq/packed_valid.jsonl \
  --uniss-strict-paper-config \
  --vocab-size 180407 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --seq-length 18000 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --train-iters 4341 \
  --lr 5e-5 \
  --min-lr 5e-6 \
  --lr-warmup-iters 0 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --bf16 \
  --use-flash-attn \
  --recompute-activations \
  --load checkpoints/uniss_phase2 \
  --save checkpoints/uniss_phase3
```

保存策略：

- 每 200-500 steps 保存一次。
- 重点保留 step 3900 左右，即约 0.9 epoch。
- 也保留最后 step 4341 方便比较。

## 8. 数据混合与采样

### 8.1 Phase 1 task mixture

论文没有给 ASR/TTS/S2TT/MT 的精确采样比例。建议保守方案：

- 先按 token-level 均衡 ASR、TTS、S2TT、MT。
- MT tokens 不低于总训练 tokens 的 20%-25%，用于减少文本翻译能力遗忘。
- 每个 mini-batch 内允许多任务混合，但每条 packed sample 自带 task token。
- 监控各任务 loss，避免 TTS/S2ST semantic token loss 因序列长而主导。

可选采样比例起点：

```text
ASR 25%
S2TT 25%
TTS 25%
MT 25%
```

若 MT 退化，提升 MT 到 30%-35%。若 speech token 生成质量差，提升 TTS。

### 8.2 Phase 2 task mixture

论文只说明三条 generation paths，没有给精确比例。建议起点：

```text
Quality 40%
Performance 40%
Direct S2ST 20%
```

理由：

- Quality/Performance 是最终推理主要模式。
- Direct S2ST 用于让模型学习直接 speech generation path，但论文 ablation 显示 Direct S2ST inference 明显弱，因此不应过高。

Phase 2 总体混合：

```text
UniST S2ST data : Phase 1 data = 2 : 1
```

### 8.3 Phase 3 task mixture

```text
Quality 50%
Performance 50%
```

如果目标优先 Quality，可调为：

```text
Quality 70%
Performance 30%
```

但应注意 Performance mode 的部署表现可能下降。

## 9. Megatron 关键工程点

### 9.1 Qwen2 支持

需要确认所用 Megatron-LM/Megatron-Core 版本是否原生支持：

- Qwen2 architecture。
- GQA / grouped-query attention。
- Qwen2 RoPE theta。
- RMSNorm。
- tied word embeddings。
- HF Qwen2 checkpoint import/export。

如果没有原生支持，优先选择：

1. Megatron Bridge / Megatron-Core 最新版本转换 HF Qwen2。
2. 或从现有 Qwen2 Megatron conversion 脚本改。
3. 最后再手写 HF <-> Megatron 参数映射。

### 9.2 Packed sequence attention

这是最容易出错的一点。

普通 causal mask 对 packed sequence 是错误的，因为后一个样本会看到前一个样本的 target。必须实现：

- 每条 sample 内 causal attention。
- 不同 sample 之间 attention 全部 mask。
- 每条 sample position ids 重新从 0 开始。
- loss mask 只覆盖 target。

如果 Megatron 支持 reset attention mask：

- 设置 EOD 或 boundary，并启用 reset attention/position。
- 仍要确认 boundary 后 token 不能 attend 到 boundary 前 token。

如果 Megatron 不支持：

- 自定义 attention mask。
- 或暂时关闭 packing，用 padding 先做小规模 sanity check，再实现 packing。

### 9.3 Loss mask 与 label shift

推荐 dataset 直接返回：

- `tokens`: length `seq_length`
- `labels`: next-token labels
- `loss_mask`: prompt/pad 为 0，target 为 1
- `attention_mask`
- `position_ids`

注意 target 的第一个 token 应由 prompt 最后一个 token 预测，因此 loss mask 对应 shift 后位置需要仔细对齐。建议写单元测试：

```text
input:  [P0, P1, T0, T1, EOS]
labels: [P1, T0, T1, EOS, PAD]
mask:   [0,  1,  1,  1,   0]
```

这里 `T0` 的 loss 发生在位置 `P1 -> T0`，所以 mask 应落在 label 为 `T0` 的位置。

### 9.4 长度控制

训练前过滤：

- 单条 unpacked sample 长度不得超过 18000。
- 特别长的 source_glm 或 target_bicodec 样本先丢弃，不建议截断 target semantic tokens，否则会训练出不完整音频。
- 可记录被过滤比例，按 source dataset 和语言方向统计。

推理代码默认 `max_new_tokens=1500`，约小于 30 秒。训练中超长音频会造成性能和稳定性问题，因此应优先保留短中句。

## 10. 验证与评估

### 10.1 预处理 sanity check

每个阶段正式训练前做：

- 随机抽 100 条样本 decode 成可读模板，确认 task/lang/speed/control token 顺序。
- 检查 global token 长度为 32。
- 检查 prompt positions 的 labels 全为 `-100` 或 loss mask 为 0。
- 检查 packed sample boundary attention。
- 检查每个 phase 的 token count 与计划一致。

### 10.2 小规模 overfit

先用 1k-10k rows 训练几百步：

- ASR loss 应快速下降，生成可读 transcription。
- S2TT/MT 应生成目标语言文本。
- TTS/S2ST 输出应包含 `<|bicodec_semantic_i|>` 序列，并能用 BiCodec 解码。
- Quality output 应含两个 `<|end_content|>`，否则当前 `process_output` 会解析失败。

### 10.3 正式评估

论文评估集：

- CVSS-T Chinese/English test sets。
- FLEURS Chinese/English subsets。
- 情感保持主观评估：ESD、CREMA-D 随机样本。

推理配置沿用当前 README/vLLM example：

```text
temperature = 0.7
top_p = 0.8
top_k = -1
repetition_penalty = 1.1
max_new_tokens = 1500
```

指标：

- Speech-BLEU：对生成语音做 ASR 后与参考翻译算 BLEU。
- Text-BLEU：模型中间 translation text 与参考翻译算 BLEU。
- A.PCP：prosody similarity。
- SLC 0.2 / SLC 0.4：duration compliance。
- UTMOS：speech quality。
- Speaker similarity / emotion MOS：主观或辅助模型评估。

### 10.4 导出后推理验收

导出到 HF 后，用当前仓库脚本验证：

```bash
python infer.py
python vllm_example.py --task Quality --target_language zh --input_path /path/to/input --output_path /path/to/output
python vllm_example.py --task Performance --target_language en --input_path /path/to/input --output_path /path/to/output
```

需要确认：

- tokenizer 路径内有 `glm4_tokenizer/` 与 `bicodec/`。
- `AutoModelForCausalLM.from_pretrained` 能加载。
- `AutoTokenizer.from_pretrained` 的 token ID 与训练常量一致。
- `process_output` 可以正确抽取 text 和 bicodec semantic tokens。

## 11. 文件与命令执行顺序

### 11.1 公开数据可复现版本

```bash
# 0. 准备目录与下载工具
mkdir -p pretrained_models data/raw data/processed data/megatron checkpoints third_party
pip install -U "huggingface_hub[cli]" datasets pyarrow

# 1. 下载 UniSS 模型 tokenizer、GLM4 tokenizer、BiCodec assets
python download_weight.py

# 可选：不用 download_weight.py 时显式下载
huggingface-cli download cmots/UniSS \
  --local-dir pretrained_models/UniSS \
  --local-dir-use-symlinks False \
  --resume-download

# 2. 下载 UniST parquet tokenized training data
huggingface-cli download cmots/UniST \
  --repo-type dataset \
  --local-dir data/raw/UniST \
  --local-dir-use-symlinks False \
  --resume-download

# 3. 下载 Qwen2.5-1.5B-Instruct base checkpoint
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir pretrained_models/Qwen2.5-1.5B-Instruct \
  --local-dir-use-symlinks False \
  --resume-download

# 4. 克隆论文中使用的 Megatron-LM Framework
git clone https://github.com/NVIDIA/Megatron-LM third_party/Megatron-LM

# 5. 检查下载结果
python - <<'PY'
from pathlib import Path
required = [
    "pretrained_models/UniSS/config.json",
    "pretrained_models/UniSS/tokenizer.json",
    "pretrained_models/UniSS/glm4_tokenizer/config.json",
    "pretrained_models/UniSS/bicodec/config.yaml",
    "pretrained_models/Qwen2.5-1.5B-Instruct/config.json",
]
for p in required:
    print(p, Path(p).exists())
print("UniST train shards:", len(list(Path("data/raw/UniST").glob("train-*.parquet"))))
print("Megatron-LM:", Path("third_party/Megatron-LM").exists())
PY

# 6. 准备 tokenizer/model init
python training/build_tokenizer_and_init.py \
  --base-model pretrained_models/Qwen2.5-1.5B-Instruct \
  --uniss-tokenizer pretrained_models/UniSS \
  --output checkpoints/qwen2_1p5b_uniss_vocab_hf

# 7. 转 Megatron checkpoint format
python training/convert_qwen2_hf_to_megatron.py \
  --input checkpoints/qwen2_1p5b_uniss_vocab_hf \
  --output checkpoints/qwen2_1p5b_uniss_vocab

# 8. Phase 1: 从 UniST 派生 alignment + WMT17 MT
python training/prepare_phase1_alignment.py \
  --input "data/raw/UniST/train-*.parquet" \
  --tokenizer pretrained_models/UniSS \
  --tasks asr s2tt tts \
  --output data/processed/phase1/alignment.jsonl

python training/build_mt_wmt17.py \
  --source-text data/raw/WMT17/train.en \
  --target-text data/raw/WMT17/train.zh \
  --src-lang eng \
  --tgt-lang cmn \
  --tokenizer pretrained_models/UniSS \
  --max-sample-tokens 18000 \
  --output data/processed/phase1/mt.jsonl

python training/pack_sequences.py \
  --input data/processed/phase1/alignment.jsonl data/processed/phase1/mt.jsonl \
  --seq-length 18000 \
  --output data/megatron/phase1/packed_train.jsonl \
  --drop-overlong

bash scripts/train_phase1.sh

# 9. Phase 2
python training/prepare_unist_s2st.py \
  --input "data/raw/UniST/train-*.parquet" \
  --phase phase2 \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/phase2/unist_s2st.jsonl

python training/mix_sample_jsonl.py \
  --group unist=2:data/processed/phase2/unist_s2st.jsonl \
  --group phase1=1:data/processed/phase1/alignment.jsonl,data/processed/phase1/mt.jsonl \
  --output data/processed/phase2/phase2_mixed.jsonl

python training/pack_sequences.py \
  --input data/processed/phase2/phase2_mixed.jsonl \
  --seq-length 18000 \
  --output data/megatron/phase2_mix/packed_train.jsonl \
  --drop-overlong

bash scripts/train_phase2.sh

# 10. Phase 3
python training/prepare_unist_s2st.py \
  --input "data/raw/UniST/train-*.parquet" \
  --phase phase3 \
  --tokenizer pretrained_models/UniSS \
  --output data/processed/phase3/unist_s2st.jsonl

python training/pack_sequences.py \
  --input data/processed/phase3/unist_s2st.jsonl \
  --seq-length 18000 \
  --output data/megatron/phase3_hq/packed_train.jsonl \
  --drop-overlong

bash scripts/train_phase3.sh

# 11. 导出与评估
bash scripts/export_megatron_to_hf.sh
bash scripts/eval_cvss_t.sh
```

### 11.2 论文严格复现版本

公开数据可复现版本的 Phase 1 预处理之前，需要额外完成：

```bash
# A. 标准化论文列出的所有源语音数据
python training/normalize_source_corpora.py --configs configs/data_sources/*.yaml --output data/manifests/source_speech.jsonl

# B. 源端 ASR 清洗
python training/filter_source_asr_wer.py \
  --manifest data/manifests/source_speech.jsonl \
  --asr paraformer \
  --max-wer 0.05 \
  --output data/manifests/source_speech_clean.jsonl

# C. Qwen2.5-72B 翻译
python training/translate_with_qwen72b.py \
  --input data/manifests/source_speech_clean.jsonl \
  --output data/manifests/source_speech_translated.jsonl

# D. 翻译清洗
python training/clean_translations.py \
  --input data/manifests/source_speech_translated.jsonl \
  --output data/manifests/source_speech_translation_clean.jsonl

# E. SparkTTS 合成目标语音
python training/synthesize_unist_with_sparktts.py \
  --input data/manifests/source_speech_translation_clean.jsonl \
  --output-root data/synth/unist

# F. 目标端 ASR 过滤
python training/filter_target_asr_wer.py \
  --manifest data/synth/unist/manifest.jsonl \
  --max-wer 0.01 \
  --output data/manifests/unist_general_candidates.jsonl

# G. duration + VAD 生成 General/HQ
python training/build_unist_splits.py \
  --input data/manifests/unist_general_candidates.jsonl \
  --general-ratio-min 0.5 \
  --general-ratio-max 2.0 \
  --hq-ratio-min 0.7 \
  --hq-ratio-max 1.5 \
  --output data/manifests/unist

# H. 抽 GLM4/BiCodec token
python training/extract_speech_tokens.py \
  --input data/manifests/unist/general.jsonl \
  --output data/tokens/unist_general
```

然后继续执行 Phase 1/2/3 预处理和训练。

## 12. 风险点与补救

1. 官方训练脚本未开放：
   - 本计划是基于论文和公开推理格式的工程复现，不保证 bitwise 对齐官方训练。

2. 公开 UniST 与论文 UniST General/HQ 命名不完全等价：
   - HF README 显示公开数据已经有较严格过滤。
   - 需要按字段和 row count 重新统计，再决定是否能模拟 General/HQ。

3. Phase 1 数据缺失：
   - 严格复现需要大量原始 speech data 与 WMT17。
   - 公开复现可从 UniST 派生 alignment task，但缺少论文中额外 77.1k 小时 speech data 的多样性。

4. Megatron packed sequence 实现：
   - 如果 boundary mask 错，训练会产生隐蔽泄漏。
   - 必须写单元测试和小批量可视化检查。

5. Speech token ID offset：
   - ID 一旦错位，模型会学到完全错误的语义。
   - 每次构造样本后用 HF tokenizer decode 抽样对比字符串模板。

6. Loss mask shift：
   - prompt loss 如果没有 mask，会浪费能力学习复制 prompt。
   - target 首 token loss 如果 mask 对齐错，会影响所有任务。

7. Long audio：
   - 当前推理建议输入 30 秒以内。
   - 训练也应过滤过长样本，避免 target semantic tokens 过长导致 packing 和生成不稳定。

## 13. 最小可执行里程碑

### Milestone 1: Token constants 与模板验证

- 实现 `constants_uniss.py`。
- 从 10 条 UniST parquet 构造 Quality/Performance/Direct 样本。
- 用 `AutoTokenizer.decode` 验证字符串与 `uniss/cli/prompt.py` 风格一致。

### Milestone 2: Packing 单元测试

- 构造 3 条 toy samples。
- pack 到一个 sequence。
- 检查 attention mask block diagonal。
- 检查 position ids reset。
- 检查 loss mask shift。

### Milestone 3: 单 GPU HF/小 Megatron overfit

- 1k samples，训练 200-500 steps。
- 验证 ASR/S2TT 文本输出。
- 验证 TTS/S2ST bicodec tokens 可解码。

### Milestone 4: 16 GPU Phase 1/2/3

- 完整跑三阶段。
- 保留所有 intermediate checkpoints 和 tensorboard logs。
- Phase 3 比较 0.9 epoch 与 1.0 epoch checkpoint。

### Milestone 5: HF 导出与论文评估

- 导出到 HF。
- 跑 CVSS-T 和 FLEURS。
- 生成 Text-BLEU/Speech-BLEU/SLC/UTMOS 报告。

## 14. 参考链接

- UniSS paper: https://arxiv.org/pdf/2509.21144
- UniSS GitHub: https://github.com/cmots/UniSS
- GitHub issue #2 author reply: https://github.com/cmots/UniSS/issues/2#issuecomment-3663882681
- UniSS model: https://huggingface.co/cmots/UniSS
- UniST dataset: https://huggingface.co/datasets/cmots/UniST
- Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- GLM-4-Voice tokenizer: https://github.com/zai-org/GLM-4-Voice
- Spark-TTS / BiCodec: https://github.com/SparkAudio/Spark-TTS
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Megatron-Core docs: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html
