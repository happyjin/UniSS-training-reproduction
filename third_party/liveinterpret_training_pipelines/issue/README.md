# text_to_speech.jsonl 数据问题记录

> 记录时间: 2026-07-02
> 涉及文件: `datasets/jsonl/text_to_speech.jsonl` (10,230,427 条, 8.4 GB)
> 生成脚本: `dataset_maker/make_text_to_speech.py`

---

## 问题 1: 多段共享音频 — 未切分原音频，丢失 segment 时间戳

### 现象

部分语言的条目共享同一个完整长音频文件（通常 10-35 分钟），但每条数据的 `target_text` 只是该音频中的一句话（1-2 秒）。输出 JSONL 中所有条目都指向同一个音频文件路径，且没有记录 segment 的起止时间。

**受影响语言（条目/音频比极高）:**

| 语言 | 条目数 | 唯一音频数 | 平均条目/音频 |
|------|--------|-----------|-------------|
| pt   | 141,696 | 1,342 | 105.6 |
| it   | 85,462  | 816   | 104.7 |
| es   | 213,730 | 34,789 | 6.1  |
| fr   | 242,229 | 48,171 | 5.0  |
| bn   | (大量)  | (少量) | ~100+ |

**不受影响的语言（1:1 或接近 1:1）:** en, zh, ko, de, ru — 这些数据源的原始音频本身就是按句切分的。

### 示例

音频文件: `TgDwUxXF-QA.flac` (完整 TED 演讲)

Parquet 原始数据:
```
example_id: ...TgDwUxXF-QA_0000  text: "É aqui que é aqui?"     start_ms: 37650  duration_ms: 1310
example_id: ...TgDwUxXF-QA_0001  text: "É agora que é agora?"   start_ms: 39490  duration_ms: 1130
example_id: ...TgDwUxXF-QA_0002  text: "Então é suposto..."     start_ms: 41980  duration_ms: 1690
... (共 294 条 segment 共享同一音频)
```

生成的 JSONL 输出:
```json
{
  "target_audio": "/mnt/fsx/data/raw/.../TgDwUxXF-QA.flac",
  "target_text": "É aqui que é aqui?",
  "speaker_profile": {"speaker_id": "...TgDwUxXF-QA_0000"},
  // ❌ 没有 start_ms / duration_ms 字段
  // ❌ target_audio 指向完整演讲音频而非 segment 级片段
}
```

### 根因分析

在 `dataset_maker/make_text_to_speech.py` 的 `convert_record()` 函数（第 332-423 行）中:

1. **读取了 `audio_uri`** 并通过 `s3_to_fsx()` 映射为本地 FSx 路径，直接写入 `target_audio`
2. **读取了 `example_id`** 作为 `speaker_id`（其中包含 `_0000` 段号后缀）
3. **没有读取 `start_ms` 和 `duration_ms`** — 这两个字段存在于 parquet 原始数据中但被完全忽略
4. **没有执行音频切分** — 没有将长音频按 segment 裁剪为独立的短音频文件

```python
# convert_record() 中相关代码 (第 371-392 行):
audio_uri = clean_str(record.get("audio_uri"))
# ... 映射为 FSx 路径 ...
target_audio_path = local_audio   # ← 直接使用完整音频路径

# ❌ record 中的 start_ms 和 duration_ms 从未被读取
```

### 影响

- **训练时数据错位**: 模型训练时读取的是完整长音频，但 `target_text` 只是其中一句话，导致 text-audio 对齐错误
- **时长筛选失效**: 无法按 "2 分钟以内" 筛选，因为整个音频文件可能 10-35 分钟但实际 segment 只有 1-2 秒
- **pt/it 语言大量误删**: 如果按文件时长过滤，pt 的 141,696 条中绝大多数会被错误丢弃

### 建议修复方案

**方案 A (推荐): 在 convert_record 中保留时间戳字段**
- 在输出 JSONL 中增加 `audio_start_ms` 和 `audio_duration_ms` 字段
- 训练 dataloader 根据这两个字段从长音频中裁剪对应片段

**方案 B: 在数据制作阶段预切分音频**
- 在 `convert_record` 或后处理步骤中，使用 `start_ms` / `duration_ms` + ffmpeg/sox 将长音频切分为独立的 segment 文件
- 输出独立的 `target_audio` 路径指向切分后的短音频

---

## 问题 2: LibriSpeech 音频路径重复 — 原始数据目录结构冗余

### 现象

LibriSpeech 数据集的同一个音频文件 `637-127579-0032.flac` 在 JSONL 中出现了 4 条完全相同的记录，分为两种 `target_audio` 路径:

```
路径 1: .../train-clean-360/LibriSpeech/train-clean-360/637/127579/637-127579-0032.flac  (2 条)
路径 2: .../train-clean-360/train-clean-360/637/127579/637-127579-0032.flac              (2 条)
```

四条数据的 `target_text`、`speaker_id`、`source_lang` 完全一致，是纯粹的重复数据。

### 根因分析

LibriSpeech 原始数据目录结构存在冗余:

```
/mnt/fsx/data/raw/dataset=librispeech/source_version=openslr_slr12/
└── train-clean-360/
    ├── LibriSpeech/               # 原始 LibriSpeech 目录结构
    │   ├── BOOKS.TXT
    │   ├── CHAPTERS.TXT
    │   ├── LICENSE.TXT
    │   ├── README.TXT
    │   ├── SPEAKERS.TXT
    │   └── train-clean-360/       # ← 实际音频文件所在
    │       └── 637/127579/637-127579-0032.flac
    └── train-clean-360/           # ← 重复拷贝（非软链接）
        └── 637/127579/637-127579-0032.flac
```

两个目录下的文件 MD5 完全一致 (`edd15f547bcbb8e0e3c8e04b07602a22`)，`train-clean-360/train-clean-360/` 是 `train-clean-360/LibriSpeech/train-clean-360/` 的完整副本。

数据 curate 阶段扫描了两个目录，生成了重复的 parquet 行，进而导致 JSONL 中出现 4 倍重复（2 个路径 × 每个路径 2 条）。

### 影响范围

- LibriSpeech 在 parquet 中有 1,169,468 行，其中约一半是重复数据
- 对应 JSONL 中约 584,734 个 "2 条共享" 的音频文件，实际都是同一条数据的重复
- 英文总条目 2,386,551 中估计约 50 万条是重复

### 建议修复方案

**数据源层面:**
- 删除 `/mnt/fsx/data/raw/dataset=librispeech/.../train-clean-360/train-clean-360/` 目录（它是 `LibriSpeech/train-clean-360/` 的冗余拷贝）
- 或在 curate 阶段对 `audio_uri` 做去重

**JSONL 层面:**
- 按 `(example_id, target_text)` 去重，保留唯一路径
- 或在 `make_text_to_speech.py` 中增加去重逻辑（对同 `example_id` 的行只保留第一条）

---

## 附: Parquet 原始数据字段说明

```
example_id       — 唯一标识符（含 segment 段号后缀 _0000, _0001 ...）
dataset          — 数据集名称
split            — 数据集 split (train/dev/test)
src_lang         — ISO 639-1 语言代码
audio_uri        — S3 音频文件 URI (s3://bucket/data/raw/...)
duration_ms      — segment 时长（毫秒）  ← 问题1中被忽略
start_ms         — segment 起始时间（毫秒） ← 问题1中被忽略
sample_rate      — 采样率
prompt           — 提示文本
text             — 文本内容（优先使用）
transcription    — 转录文本（text 的 fallback）
source_type      — 来源类型
source_field     — 来源字段
source_file      — 来源文件
source_row_index — 来源行号
```

---

## 筛选后数据排查结果 (jsonl_zeyang/text_to_speech_zh_en.jsonl)

> 筛选条件: 仅保留 `en`/`zh`，去除问题 1（多段共享）和问题 2（路径重复）
> 筛选后: 2,542,975 条 (en: 1,509,450 / zh: 1,033,525)
> 重复来源: librispeech (877,101 条重复已去除), primewords_chinese_corpus_set_1 (1,554 条重复已去除)

### 问题 3: thchs_30 数据集 — 文本字段存储的是文件路径而非实际内容

**严重程度: 高**

13,388 条 `thchs_30` 数据集的 `target_text` 不是实际文本内容，而是文件路径，如 `../data/A22_1.wav.trn`。

```
line 564983: target_text = "../data/A11_101.wav.trn"  (应为实际中文转录文本)
line 565063: target_text = "../data/A22_1.wav.trn"
...
```

**根因**: `make_text_to_speech.py` 中 `convert_record()` 优先读取 `text` 字段，而 thchs_30 parquet 的 `text` 字段存储的是 `.trn` 文件路径而非转录文本。应 fallback 到 `transcription` 字段，或读取 `.trn` 文件内容。

**影响**: 13,388 条数据完全无法用于训练（文本是路径而非内容）

**示例 JSON:**
```json
{
  "target_text": "../data/A11_101.wav.trn",
  "target_audio": "/mnt/fsx/data/raw/dataset=thchs_30/source_version=openslr_slr18/data_thchs30/dev/A11_101.wav",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "thchs_30_data_thchs30_dev_A11_101", "gender": null, "style": null, "pitch": null}
}
```

**处理**: 已在 `text_to_speech_zh_en_clean.jsonl` 中移除

### 问题 4: 重复字符异常文本

**严重程度: 中**

53 条数据的 `target_text` 包含连续重复字符（≥10 次重复），全部来自中文数据集。

**示例 JSON 1:**
```json
{
  "target_text": "快快快快快快快快快快",
  "target_audio": "/mnt/fsx/data/processed/dataset=aishell_3/source_version=external_openslr_slr93/process_version=audio_16k_mono_flac_v1/audio/lang=zh/shard=000000/aishell_3_SSB05340458.flac",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "aishell_3:SSB05340458", "gender": null, "style": null, "pitch": null}
}
```

**示例 JSON 2:**
```json
{
  "target_text": "减减减减减减减减减减减减减减减减减减减减减减减减减减减合格钡",
  "target_audio": "/mnt/fsx/data/raw/dataset=primewords_chinese_corpus_set_1/source_version=openslr_slr47/audio_files/2/25/25dd1a96-af22-45a0-9b72-295b24cbbabe.wav",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "primewords_chinese_corpus_set_1_25dd1a96-af22-45a0-9b72-295b24cbbabe", "gender": null, "style": null, "pitch": null}
}
```

**根因**: 来自 ASR 转录数据（如 aishell_3、primewords），转录了口语中的重复/结巴现象。部分可能是转录标注错误。

**影响**: 少量噪声数据，对训练影响有限但可能影响模型稳定性

**处理**: 已在 `text_to_speech_zh_en_clean.jsonl` 中移除

### 问题 5: 过短文本（< 2 字符）

**严重程度: 低**

280 条数据的 `target_text` 少于 2 个字符，如单字 `好`、`停`、`嗯`、`A`、`C` 等。

**示例 JSON 1:**
```json
{
  "target_text": "光",
  "target_audio": "/mnt/fsx/data/processed/dataset=aishell_3/source_version=external_openslr_slr93/process_version=audio_16k_mono_flac_v1/audio/lang=zh/shard=000000/aishell_3_SSB08090178.flac",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "aishell_3:SSB08090178", "gender": null, "style": null, "pitch": null}
}
```

**示例 JSON 2:**
```json
{
  "target_text": "呢",
  "target_audio": "/mnt/fsx/data/processed/dataset=aishell_3/source_version=external_openslr_slr93/process_version=audio_16k_mono_flac_v1/audio/lang=zh/shard=000000/aishell_3_SSB12740095.flac",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "aishell_3:SSB12740095", "gender": null, "style": null, "pitch": null}
}
```

**影响**: 文本过短可能导致 TTS 模型无法生成有效语音

**处理**: 已在 `text_to_speech_zh_en_clean.jsonl` 中移除

### 数据集分布汇总

> 以下为 `text_to_speech_zh_en.jsonl` 筛选后的分布；`text_to_speech_zh_en_clean.jsonl` 额外移除了问题 3/4/5（共 13,721 条），最终 2,529,254 条。

| 数据集 | 条目数 | 语言 | 采样率 | 备注 |
|--------|--------|------|--------|------|
| magicdata_mandarin_chinese_read_speech_corpus | 609,552 | zh | 16kHz | |
| libritts_corpus | 375,086 | en | **24kHz** | |
| libritts_r | 374,999 | en | **24kHz** | |
| hi_fi_multi_speaker_english_tts_dataset | 323,978 | en | **44.1kHz** | |
| librispeech | 292,367 | en | 16kHz | 去重后保留 |
| aishell_1 | 141,600 | zh | 16kHz | |
| free_st_chinese_mandarin_corpus | 102,600 | zh | 16kHz | |
| aishell_3 | 88,035 | zh | 16kHz | |
| vystadial | 78,029 | en | 16kHz | |
| primewords_chinese_corpus_set_1 | 50,384 | zh | 16kHz | 去重后保留 |
| vctk_092 | 44,283 | en | 16kHz | |
| thchs_30 | 26,776 | zh | 16kHz | ⚠️ 问题3: 13,388 条文本为路径，已移除 |
| shalcas22a | 14,578 | zh | 16kHz | |
| ljspeech | 13,100 | en | 16kHz | |
| speechocean762 | 5,000 | en | 16kHz | |
| mini_librispeech | 2,608 | en | 16kHz | |

---

## 问题 9: 音频采样率不统一 — Qwen3-TTS-Tokenizer 期望 24kHz

**严重程度: 高**

JSONL 中 `chat_template_kwargs.sampling_rate` 全部标注为 16000，但实际音频文件的采样率不一致：

| 采样率 | 数据集 | 条目数 |
|--------|--------|--------|
| 16000 Hz | aishell, librispeech, magicdata, vystadial, vctk 等 | ~1,831,187 |
| 24000 Hz | libritts_corpus, libritts_r | 750,085 |
| 44100 Hz | hi_fi_multi_speaker_english_tts_dataset | 323,978 |

而 Qwen3-TTS-Tokenizer-12Hz 的 `config.json` 中 `input_sample_rate: 24000`。

**根因**: `make_text_to_speech.py` 中 `convert_record()` 将 `chat_template_kwargs.sampling_rate` 硬编码为 16000，未检查实际音频文件采样率。训练管线 `codec_tokenizer.py` 中的 `encode_codes()` 直接把文件路径传给 `tokenizer.encode()`，没有重采样。

**影响**: 不匹配的采样率会导致 tokenizer 生成错误的 codec token，进而导致 codec loss 计算错误。

**处理**: 已在 `custom/codec_tokenizer.py` 的 `encode_codes()` 中添加 `_encode_with_resample()` 方法，使用 `torchaudio.transforms.Resample` 将所有音频重采样到 24000 Hz 后再送入 tokenizer。

---

## 问题 10: 中文文本中混入 ASCII 字符（码切换/OCR 噪声）

**严重程度: 低**

11,027 条中文（`zh`）数据中混入了 ASCII/拉丁字符。

**来源数据集分布:**

| 数据集 | 条目数 | 噪声类型 |
|--------|--------|---------|
| magicdata_mandarin_chinese_read_speech_corpus | 11,012 | 中文中夹杂英文缩写/单词（如 "家n次方"、"qq"） |
| thchs_30 | 12 | OCR 噪声（如 `l =` 代替"六"） |
| free_st_chinese_mandarin_corpus | 3 | 日常用语夹杂英文（如 "qq"） |

**示例 JSON (magicdata):**
```json
{
  "target_text": "请给我播放家n次方第十二集",
  "target_audio": "/mnt/fsx/data/raw/dataset=magicdata_mandarin_chinese_read_speech_corpus/source_version=openslr_slr68/dev_set/dev/38_5730/38_5730_20170915100910.wav",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "magicdata_mandarin_chinese_read_speech_corpus_38_5730_20170915100910", "gender": null, "style": null, "pitch": null}
}
```

**示例 JSON (thchs_30 — OCR 噪声):**
```json
{
  "target_text": "徐 希君 肖 金生 刘 文华 屈 永利 王开 宇 骆 瑛 等 也 被 分别 判处 l = 六年 至 十 五年 有期徒刑",
  "target_audio": "/mnt/fsx/data/raw/dataset=thchs_30/source_version=openslr_slr18/data_thchs30/data/B11_374.wav",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "thchs_30_data_thchs30_data_B11_374", "gender": null, "style": null, "pitch": null}
}
```

**示例 JSON (free_st — 日常用语):**
```json
{
  "target_text": "把安娟和杨洋的qq告诉我嗯嗯",
  "target_audio": "/mnt/fsx/data/raw/dataset=free_st_chinese_mandarin_corpus/source_version=openslr_slr38/20170001P00169A0018.wav",
  "source_lang": "zh", "target_lang": "zh",
  "speaker_profile": {"speaker_id": "free_st_chinese_mandarin_corpus_20170001P00169A0018", "gender": null, "style": null, "pitch": null}
}
```

**影响**: 少量噪声，不影响训练但可能影响文本 tokenization 质量

**建议**: 可选择性清洗，或保留（模型需要学习处理此类噪声）

---

## 问题 11: 文本中存在控制字符

**严重程度: 低**

1 条数据（line 1,978,397）包含控制字符 `\x14` (DC4) 和 `\x17` (ETB)：

```
"大家早上好\x14\x17下雨天积水出行的友友们注意安全哦"
```

来自 magicdata_mandarin_chinese_read_speech_corpus 数据集。

**影响**: 控制字符可能被 tokenizer 当作特殊 token 处理

**建议**: 在数据加载阶段 strip 控制字符

---

## 问题 12: gender/style/pitch 和 reference_audios 字段未使用

**严重程度: 无（设计确认）**

数据中 `speaker_profile.gender`、`style`、`pitch` 全部为 `null`，`reference_audios` 全部为空数组 `[]`。

经代码确认，训练管线**不使用**这些字段：
- `modeling_qwen_omnispeak.py` 第 977-981 行：`speaker_profile` 和 `reference_audios` 从 `thinker_kwargs` 中被 `pop` 移除，不传入模型 forward
- `template.py` 第 107 行注释明确说明：`"no speaker id: the base talker's spk_id map is empty"`
- `_build_control_prefix()` 只使用 `target_lang` 构建 codec 控制前缀，不使用 speaker 信息

**结论**: 这些字段为空不影响训练，无需处理。

---

## 问题 13: 英文文本大小写不一致

**严重程度: 低（设计特征）**

英文数据中存在三种大小写模式：

| 模式 | 条目数 | 主要数据集 |
|------|--------|-----------|
| ALL CAPS | 376,658 | LibriSpeech (292,320), vystadial (70,864) |
| all lower | 329,251 | libritts_corpus, libritts_r |
| Mixed case | 789,325 | hi_fi_tts, vctk, ljspeech 等 |

**影响**: 不同数据源的转录规范不同。需确认 Qwen3-OmniSpeak 模型的 tokenizer 是否大小写敏感。如果 tokenizer 会将所有文本转为小写，则无影响；否则需要统一大小写格式。

---

## 训练前需确认的配置清单

| # | 确认项 | 状态 | 说明 |
|---|--------|------|------|
| 1 | Python 环境 | ❌ 未安装 | torch / ms-swift / deepspeed 等需安装 |
| 2 | 数据路径 | 需设置 | `BRIDGE_ALIGNMENT_DATA` 指向 `datasets/jsonl_zeyang/text_to_speech_zh_en_clean.jsonl` |
| 3 | 音频重采样 | ✅ 已修复 | `codec_tokenizer.py` 已添加 24kHz 重采样 |
| 4 | gender/style/pitch | ✅ 无需处理 | 训练管线不使用这些字段 |
| 5 | WANDB | 需配置 | `WANDB_API_KEY` 或 `REPORT_TO=none` |
| 6 | MAX_LENGTH=8192 | 需确认 | 文本中位数 30 字符，p99=278，codec 按 12.5Hz 算，10 分钟音频 ≈ 7500 帧，接近上限 |
| 7 | 训练参数 | 需确认 | 1 epoch / lr=1e-4 / freeze_ratio=1.0（只训练 bridge + talker 前 2 层） |
