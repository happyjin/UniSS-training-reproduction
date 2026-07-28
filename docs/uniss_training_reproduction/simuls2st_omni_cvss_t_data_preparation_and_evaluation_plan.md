# SimulS2ST-Omni / UniSS 的 CVSS-T 数据准备与评估实施计划

更新日期：2026-07-28

适用仓库：`/opt/dlami/nvme/jasonleeeli/projects/UniSS`

数据根目录：`/opt/dlami/nvme/jasonleeeli/CVSS`

## 1. 结论先行

本机已经下载了 CVSS 官方发布的全部目标语音压缩包，但“已经有 CVSS-T”不等于“已经具备论文级 CVSS-T 评估条件”。当前准确状态如下：

- CVSS-C 21/21、CVSS-T 21/21，共 42/42 个官方 target-speech archives 已下载；archives 合计约 `198.40 GiB`，整个 CVSS 根目录约 `225 GiB`。
- 中文到英文的 CVSS-T 已解压到：

  ```text
  /opt/dlami/nvme/jasonleeeli/CVSS/extracted/cvss_t_zh_en_v1.0
  ```

- `test.tsv` 有 4,897 条，`test/` 下也有 4,897 个英文目标 WAV；总时长约 `6.2917 h`。
- 这 4,897 个 WAV 均为可读的 `24 kHz / mono / PCM16`。
- 对应 archive 为：

  ```text
  /opt/dlami/nvme/jasonleeeli/CVSS/archives/cvss_t_v1.0/cvss_t_zh_en_v1.0.tar.gz
  size   = 3103162925 bytes
  sha256 = 00904f6f952a308024f6d1d8af8127b01bac8d78d46ee69ffc3715e3376c21b2
  ```

- CVSS 官方 archive 只提供目标英文语音和规范化英文翻译文本。对应的真实中文源语音和中文 transcript 必须来自 **Common Voice release v4 的 zh-CN 数据**，并按文件名逐条配对。
- 2026-07-28 已补齐 CVSS-T test 所需的最小 Common Voice v4 / CoVoST 2 中文源数据：
  - 中文源音频：`4,897 / 4,897`
  - 中文源文本：`4,897 / 4,897`
  - 全部源音频可读，为 `48 kHz / mono / MP3`，总时长约 `8.2519 h`
  - 两个独立公开 CoVoST 2 镜像中，这 4,897 条音频的 SHA256 逐条一致
  - 两个镜像共同多出 `common_voice_zh-CN_18747192.mp3` 一条，因不在 CVSS-T test ID 集中而被严格排除
- 当前具备完整 ZH→EN 与论文反向 EN→ZH CVSS-T 数据条件；但保存的是 CVSS-T test 所需的可验证最小子集，不是完整 Common Voice v4 zh-CN archive。

旧的阻塞状态审计文件为：

```text
experiments/evaluation/uniss_full198_phase2_phase3/manifests/cvss_t_manifest_summary.json
```

它记录了补齐源数据之前的历史状态：

```json
{
  "pair_count": 4897,
  "target_wav_count": 4897,
  "missing_source_count": 4897,
  "missing_source_text_count": 4897,
  "target_en_hours": 6.2916875,
  "source_zh_hours": null,
  "ready_for_bidirectional_evaluation": false
}
```

新的可评估状态审计文件为：

```text
/opt/dlami/nvme/jasonleeeli/CVSS/manifests/cvss_t_zh_en_v1/cvss_t_manifest_summary.json
```

关键状态：

```json
{
  "pair_count": 4897,
  "target_wav_count": 4897,
  "missing_source_count": 0,
  "missing_source_text_count": 0,
  "target_en_hours": 6.2916875,
  "source_zh_hours": 8.2518970,
  "ready_for_bidirectional_evaluation": true
}
```

下一步不再是下载 CVSS-T test 数据，而是制作不可变 canonical audio、UniSS tokenized parquet，并分别运行 UniSS 原协议和 SimulS2ST-Omni 统一协议。

## 2. 本计划要回答的问题

本计划围绕四个问题展开：

1. CVSS-T 官方数据究竟包含什么，源语音、目标语音和文本如何配对？
2. SimulS2ST-Omni、UniSS 以及其他使用 CVSS-T 的工作采用了什么评估协议？
3. 当前本地数据、代码和模型还缺什么，如何在不影响历史实验的情况下补齐？
4. 如何用当前最好的 UniSS full198 Phase3 checkpoint 得到可与论文对比的指标、可试听音频和完整报告？

## 3. 文献协议审计

### 3.1 CVSS 原始论文与官方数据定义

CVSS 来源于两部分：

- 源语音与源 transcript：Mozilla Common Voice release v4。
- 翻译文本：CoVoST 2。
- 目标英文语音：CVSS 使用 TTS 合成。

CVSS 有两个版本：

- `CVSS-C`：目标英文全部使用同一 canonical speaker。
- `CVSS-T`：使用对应源语音做跨语言 voice transfer，使源语音和目标英文语音具有相近说话人声音。

每个官方 `tar.gz` 包含：

```text
train/
dev/
test/
train.tsv
dev.tsv
test.tsv
```

其中 TSV 每行是：

```text
<Common Voice filename>\t<normalized English translation>
```

必须使用相同文件名到 Common Voice v4 中查找真实源音频。例如：

```text
common_voice_zh-CN_18885718.mp3
```

不能只使用 CVSS-T 的英文 WAV 做 ZH→EN 评估，也不能用当前最新版 Common Voice 随意替代 v4，因为后续版本可能改变文件覆盖、metadata 和 split。

CVSS 目标英文规范化文本与实际合成发音匹配，主要处理数字、货币、缩写等。它通常为 lowercase、去掉标点但保留 apostrophe；评估时应优先使用 archive 内的 normalized text，而不是用未经规范化的 CoVoST 原始 translation 直接代替。

### 3.2 使用 CVSS-T 的论文协议对比

| 工作 | CVSS-T 用途 | 方向与数据选择 | 解码 | 主要指标 | 对本项目的意义 |
|---|---|---|---|---|---|
| CVSS Corpus and Massively Multilingual S2ST | 建立多语种 X→EN S2ST benchmark | 官方方向 X→EN | 论文自有模型配置 | ASR-BLEU、MOS、speaker similarity 等 | 规定 CVSS 与 Common Voice v4 的配对方式 |
| UniSS: Unified Expressive S2ST with Your Voice | 主要离线 S2ST benchmark | 4,897 对，ZH→EN 与反向 EN→ZH | `temperature=0.7, top_p=0.8, top_k=-1, repetition_penalty=1.1` | Text-BLEU、Speech-BLEU、AutoPCP、SLC 0.2/0.4、UTMOS、主观 MOS | 本项目最直接的论文复现协议 |
| SimulS2ST-Omni | **CVSS-T 只用于 offline S2ST** | 4,897 对，EN→ZH 与 ZH→EN | offline greedy | Text-BLEU、ASR-BLEU、AutoPCP、SIM-O | 提供统一重评分协议；streaming 不在 CVSS-T 上做主结论 |
| PROST-LLM | SFT/偏好优化和评估 | French-English CVSS-T，双向通过反转 pair | 论文配置 | ASR-BLEU、UTMOS 等 | 说明 CVSS-T 可用于多说话人 voice-preserving S2ST，但任务与本项目语言对不同 |
| COMPASS-S2ST framework（论文题名：Benchmarking Speech-to-Speech Translation Models） | 大规模离线 S2ST 评估框架 | CVSS 只做 X→EN 真实源输入；明确排除 EN→X synthetic-source bias；常取每语言前 1,000 条 | greedy、固定 seed | BLEU、ChrF++、COMET、TER、UTMOS/NISQA、speaker/prosody/timing 等 | 提醒反向 CVSS-T 方向存在合成输入偏置，并提供扩展诊断指标 |

与当前 ZH/EN 模型直接相关的两组数值锚点如下。它们使用不同协议，必须分表对比：

| 协议来源 | 模型 | ASR/Speech-BLEU EN→ZH / ZH→EN | Text-BLEU EN→ZH / ZH→EN | AutoPCP EN→ZH / ZH→EN | 其他 |
|---|---|---|---|---|---|
| UniSS 原论文 Table 1 | UniSS (Q), 1.5B | 32.20 / 24.28 | 32.95 / 26.28 | 2.71 / 2.74 | SLC-0.2 0.98/0.87，SLC-0.4 0.99/0.97，UTMOS 3.76/3.86 |
| SimulS2ST-Omni unified re-score Table 1 | UniSS (Q), 1.5B | 32.04 / 24.72 | 32.95 / 25.51 | 2.71 / 2.75 | SIM-O 0.40/0.42 |

上述差异不是可以忽略的抄表误差，而是提醒我们：解码方式、ASR backend、文本规范化和 metric implementation 必须随每个结果一起保存。

### 3.3 SimulS2ST-Omni 的关键边界

SimulS2ST-Omni 的主方法是显式 trajectory supervision：把目标文本和 acoustic semantic codes 放入同一条 commitment path，在源语音 chunk 推进过程中学习 READ / WAIT / WRITE。其 streaming Stage 3 还对 latency multiplier `m` 进行采样，用一个 checkpoint 覆盖多种延迟档位。

但该论文的数据集分工非常明确：

- CVSS-T：offline S2ST，验证基础翻译和语音生成能力。
- RealSI：sentence-level 与 long-form streaming S2TT/S2ST。
- ACL60/60-dev：long-form EN→ZH streaming S2TT。

所以本项目可以把 CVSS-T 音频按时间 chunk 后用于额外的 sentence-level streaming diagnostic，但结果必须标记为：

```text
CVSS-T sentence-level streaming diagnostic
```

不能标记为：

```text
SimulS2ST-Omni streaming benchmark reproduction
```

原因是 CVSS-T 是朗读句子，且其中一侧为合成英文语音；它不等价于 RealSI 的真实同传场景，也不适合替代 long-form streaming 数据。

## 4. CVSS-T zh-en test 的双向构造

原始 pair 定义为：

```text
真实中文 Common Voice 语音
    + 中文 transcript
    + CoVoST/CVSS 英文 translation
    + CVSS-T voice-transferred 英文语音
```

### 4.1 ZH→EN：真实源输入方向

| 字段 | 内容 |
|---|---|
| input speech | Common Voice v4 zh-CN 真实中文 MP3 |
| source text | Common Voice/CoVoST 2 中文 sentence |
| reference translation | CVSS `test.tsv` 中 normalized English text |
| reference target speech | CVSS-T 英文 WAV |
| benchmark 属性 | 真实输入，适合主结果与泛化结论 |

### 4.2 EN→ZH：反向 synthetic-source 方向

| 字段 | 内容 |
|---|---|
| input speech | CVSS-T 合成英文 WAV |
| source text | CVSS normalized English text |
| reference translation | Common Voice/CoVoST 2 中文 sentence |
| reference target speech | Common Voice v4 真实中文音频 |
| benchmark 属性 | reversed synthetic-source benchmark |

UniSS 和 SimulS2ST-Omni 都报告双向结果，因此论文复现 track 应保留两个方向。但报告中必须同时给出：

- 双向平均结果，用于对齐 UniSS/SimulS2ST-Omni 表格。
- ZH→EN 单独结果，作为更可信的真实输入结论。
- EN→ZH 明确带上 `synthetic_source=true`，不得把它解释成真实 English speech 泛化能力。

## 5. 数据目录设计

所有新数据放在现有 CVSS 根目录，不修改已下载的 archive 和已解压文件：

```text
/opt/dlami/nvme/jasonleeeli/CVSS/
├── archives/                         # 已有，只读保存
│   ├── cvss_c_v1.0/
│   └── cvss_t_v1.0/
├── extracted/                        # 已有官方解压内容，只读保存
│   └── cvss_t_zh_en_v1.0/
├── source/
│   └── common_voice_v4_zh-CN/        # 待补齐
│       ├── clips/
│       ├── validated.tsv
│       ├── train.tsv
│       ├── dev.tsv
│       └── test.tsv
├── metadata/
│   └── covost_v2.zh-CN_en/           # 待下载/生成 split metadata
├── canonical_16k/
│   └── cvss_t_zh_en_test/
│       ├── source_zh/
│       └── target_en/
├── manifests/
│   └── cvss_t_zh_en_v1/
├── tokenized/
│   └── cvss_t_zh_en_v1/
│       ├── cvss_t_zh_en_test.parquet
│       └── cvss_t_en_zh_test.parquet
├── audits/
└── logs/
```

约束：

- `archives/` 和 `extracted/` 视为不可变原始数据。
- canonical 音频、manifest、tokenized parquet 均写入新目录。
- 不把大型数据、tokenized parquet、生成音频或 metric cache 提交到 Git。
- Git 只保存代码、轻量 manifest summary、配置、运行脚本和最终 Markdown 报告。

## 6. 数据准备实施步骤

### P0：固定版本与校验原始 CVSS-T

目标：证明后续所有结果都来自同一份官方数据。

执行内容：

1. 记录 archive 路径、size、SHA256。
2. 检查 `test.tsv` 恰好 4,897 行且 ID 唯一。
3. 检查 `test/<filename>.wav` 与 TSV 一一对应，没有缺失或额外文件。
4. 对全部 WAV 记录 sample rate、channels、subtype、frames、duration。
5. 将审计写到：

   ```text
   /opt/dlami/nvme/jasonleeeli/CVSS/audits/cvss_t_zh_en_test_inventory.json
   ```

验收门槛：

- pair count = 4,897
- unique IDs = 4,897
- missing target WAV = 0
- unreadable target WAV = 0
- target duration 约 6.3 h

### P1：补齐 Common Voice v4 zh-CN（test 所需子集已完成）

必须获取 **release v4** 的 zh-CN 包。推荐目录：

```text
/opt/dlami/nvme/jasonleeeli/CVSS/source/common_voice_v4_zh-CN
```

最低需要：

```text
clips/<4,897 个目标 filename 对应的 MP3>
test.tsv 或 validated.tsv
```

完整 v4 zh-CN archive 仍然需要 Common Voice/Hugging Face 授权。当前为优先完成 CVSS-T evaluation，使用公开 CoVoST 2 test parquet 镜像抽取严格匹配的 4,897 条 v4 音频、中文 transcript 和 client ID。来源固定到具体 repository revision，并使用两个独立镜像逐条校验音频 hash。不能用 Common Voice 新版本的数据替代 v4。

当前数据来源与下载产物：

```text
官方 CoVoST 2 metadata:
/opt/dlami/nvme/jasonleeeli/CVSS/metadata/covost_v2.zh-CN_en

主镜像（含 audio / sentence / client_id）:
/opt/dlami/nvme/jasonleeeli/CVSS/source/common_voice_v4_zh-CN_test_fixie_parquet

交叉验证镜像（含 audio / original filename）:
/opt/dlami/nvme/jasonleeeli/CVSS/source/common_voice_v4_zh-CN_test_mirror

严格抽取后的 4,897 条评估源数据:
/opt/dlami/nvme/jasonleeeli/CVSS/source/common_voice_v4_zh-CN
```

下载完成后的验收：

- CVSS `test.tsv` 的 4,897 个 filename 在 `clips/` 中全部存在。
- metadata 中 4,897 条 source sentence 全部可找到。
- 无重复 filename。
- 能解码全部 MP3；源中文总时长应接近论文报告的 `8.2 h`。
- 记录 Common Voice 包的来源、版本、文件 hash 和许可说明。

### P2：下载并交叉验证 CoVoST 2 metadata

CoVoST 2 的 zh-CN→en metadata 用于交叉验证 translation、speaker/client ID 和标准 split。它不是 CVSS target normalized text 的替代品。

建议保存到：

```text
/opt/dlami/nvme/jasonleeeli/CVSS/metadata/covost_v2.zh-CN_en
```

交叉验证规则：

1. CVSS filename 必须出现在 CoVoST 2 的 test split。
2. CoVoST source sentence 与 Common Voice metadata 一致或仅存在可解释的 Unicode/空白差异。
3. CVSS normalized English 作为正式 BLEU reference；CoVoST original translation 作为审计字段保留。
4. 保存 `client_id`，用于 speaker 分组分析，但不把它写入公开试听页面。

### P3：严格 filename join

项目已有：

```text
evaluation/cvss_manifest.py
evaluation/tests/test_cvss_manifest.py
```

在 Common Voice 补齐前，现有命令只能生成 pending manifest：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
  -m evaluation.cvss_manifest \
  --cvss-root /opt/dlami/nvme/jasonleeeli/CVSS/extracted/cvss_t_zh_en_v1.0 \
  --output-dir experiments/evaluation/uniss_full198_phase2_phase3/manifests
```

补齐后使用：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
  -m evaluation.cvss_manifest \
  --cvss-root /opt/dlami/nvme/jasonleeeli/CVSS/extracted/cvss_t_zh_en_v1.0 \
  --common-voice-root /opt/dlami/nvme/jasonleeeli/CVSS/source/common_voice_v4_zh-CN \
  --output-dir /opt/dlami/nvme/jasonleeeli/CVSS/manifests/cvss_t_zh_en_v1
```

准备完成时应生成：

```text
cvss_t_zh_en_test.jsonl
cvss_t_en_zh_test.jsonl
cvss_t_manifest_summary.json
```

正式实现时还要在 manifest 中增加以下审计字段：

| 字段 | 说明 |
|---|---|
| `id` | Common Voice filename |
| `pair_id` | 双向共用的稳定 pair ID |
| `direction` | `cmn->eng` 或 `eng->cmn` |
| `source_audio_path` | 本方向输入音频 |
| `reference_audio_path` | 本方向目标 reference 音频 |
| `source_text` | 本方向输入文本 |
| `translation_ref` | 本方向翻译 reference |
| `cvss_normalized_en` | CVSS normalized English |
| `covost_original_en` | CoVoST original English，用于审计 |
| `common_voice_client_id` | 说话人分组字段 |
| `synthetic_source` | ZH→EN 为 false，EN→ZH 为 true |
| `synthetic_reference` | ZH→EN 为 true，EN→ZH 为 false |
| `source_sha256` | 源音频内容 hash |
| `reference_sha256` | reference 音频内容 hash |
| `source_duration_seconds` | 输入时长 |
| `reference_duration_seconds` | reference 时长 |

### P4：建立 canonical 16 kHz 音频 cache

原始 Common Voice MP3 和 CVSS-T 24 kHz WAV 保持不变。为 tokenizer、ASR 和 speech metrics 生成统一 cache：

```text
16 kHz / mono / PCM16 WAV
```

处理原则：

- 只做可靠解码、单声道合并和高质量重采样。
- 不做 VAD trim，因为删除句首句尾静音会改变 duration、SLC 和 streaming latency。
- 不做 loudness normalization，因为它可能改变 speaker/prosody/naturalness 指标。
- 不做 denoise，因为 speaker/prosody 指标应基于原始 benchmark 条件。
- 保留原始与 canonical 文件之间的 hash、时长和 sample rate 映射。
- 如果输入存在 clipping、NaN、空音频或异常超长，记录到 quarantine，不静默修复。

建议每条 canonical 文件名保持原 ID，两个方向共享同一份音频 cache，不复制两份。

### P5：数据泄漏审计

CVSS-T test 来自 Common Voice/CoVoST 2，而当前 UniSS 使用 UniST 训练。正式声称 benchmark 结果前必须检查是否存在训练泄漏。

至少做四层检查：

1. filename exact match：CVSS test ID 是否出现在所有训练 parquet 的 ID/path 字段。
2. audio exact hash：canonical audio SHA256 是否与训练数据已知原音频 hash 相同。
3. audio near-duplicate：对 source audio embedding 或音频 fingerprint 做近重复搜索。
4. text overlap：规范化 source/translation 后检查 exact match 与高相似句子。

报告必须区分：

- `clean_test_count`
- `exact_id_overlap_count`
- `exact_audio_overlap_count`
- `near_audio_overlap_count`
- `source_text_overlap_count`
- `translation_text_overlap_count`

若发现 exact audio 或 exact ID 泄漏，主表应同时给出全量结果与去泄漏子集结果，并把 clean subset 设为主要可信结论。

### P6：CVSS raw audio 转 UniSS tokenized parquet

当前 vLLM batch evaluator 读取的是 UniST-style tokenized record，不直接读取 raw WAV。正式评估不建议循环调用 Web demo 的单请求 engine；应新增可恢复、可分片、可 8 GPU 数据并行的 CVSS tokenizer adapter。

每个方向的 record 至少包含：

```text
id
dataset_name
src_lang
tgt_lang
transcription
translation
source_glm
source_bicodec
target_bicodec
bicodec_global
source_audio_path
reference_audio_path
synthetic_source
synthetic_reference
```

使用 `UniSSTokenizer.tokenize(audio)` 时：

```text
source_glm       = source linguistic tokens
bicodec_global   = source BiCodec tokens[:32]
source_bicodec   = source BiCodec tokens[32:]
target_bicodec   = reference-target BiCodec tokens[32:]
```

这里必须使用 **source 的 32 个 global tokens** 作为说话人条件；不能误用目标 reference 的 global tokens，否则会改变 UniSS 的 “your voice” 任务定义。

重要实现要求：

- 以 `direction + pair_id` 作为唯一主键。
- 8 卡时按 pair ID 稳定分片，每卡独立写 part parquet，完成后排序合并。
- 每条处理后立即写 progress/summary，支持 `--resume`。
- 若一条音频失败，写 failure JSONL，不允许整个 job 悄悄少样本。
- 保存 tokenizer checkpoint hash/version、代码 commit、GPU、seed 和 canonical audio manifest hash。
- tokenized parquet 之外仍保留真实 `source_audio_path` 与 `reference_audio_path`；正式 AutoPCP/SIM-O 直接使用真实音频，而不是先将 reference token 解码再当 reference。

这一点非常关键：现有 UniST evaluator 的 `--save-reference-audio` 可从 semantic tokens 重建 reference，但 CVSS 论文指标应比较官方 paired waveform。重建音频只能做调试，不能替代正式 CVSS reference。

### P7：建立 deterministic evaluation manifest

建议生成以下 manifest：

```text
cvss_t_zh_en_test_all.jsonl          # 4,897，真实中文输入
cvss_t_en_zh_test_all.jsonl          # 4,897，合成英文输入
cvss_t_bidirectional_test_all.jsonl  # 9,794 direction records
cvss_t_bidirectional_smoke_20.jsonl  # 每方向10条
cvss_t_bidirectional_listen_100.jsonl# 每方向50条
cvss_t_zh_en_compass_1000.jsonl      # 可选 X→EN 扩展评估
```

抽样规则：

- 固定 seed，例如 `20260728`。
- smoke/listen 按方向、时长 bucket、speaker/client ID 做分层抽样。
- 所有 manifest 记录源数据 inventory hash；任何源数据变化都应使 manifest version 变化。

## 7. 评估 Track 设计

### Track A：UniSS 原论文兼容评估

目的：与 UniSS Table 1 和项目已有 baseline JSON 直接对比。

模型：

```text
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
```

建议分别运行 `quality` 和 `performance` 两种 mode，保留当前项目任务定义。生成参数固定为：

```text
temperature         = 0.7
top_p               = 0.8
top_k               = -1
repetition_penalty  = 1.1
seed                = 20260728
```

指标：

- Text-BLEU
- Speech-BLEU / ASR-BLEU
- AutoPCP
- SLC at 0.2 / 0.4
- UTMOS
- 可选主观 speaker / emotion / naturalness MOS

对比数据已保存在：

```text
evaluation/reference_data/uniss_paper_cvss_t_table1.json
```

结果表按方向分别列出，不只报告双向平均。

### Track B：SimulS2ST-Omni 统一重评分协议

目的：按照 SimulS2ST-Omni Table 1 的 unified protocol 重新评估同一个 UniSS checkpoint。

解码：

```text
offline full-utterance
greedy decoding
fixed seed
```

指标：

- Text-BLEU
- ASR-BLEU
- AutoPCP / A.PCP
- SIM-O

ASR 与 text normalization：

- 英文生成语音：Whisper-Large-v3。
- 中文生成语音：Paraformer-zh。
- English：NFKC、lowercase、去标点但保留 apostrophe，SacreBLEU `13a`。
- Chinese：繁体转简体、去标点、字符间插空格，SacreBLEU `zh`。

AutoPCP：

- source waveform 与 generated target waveform 比较。
- 统一转 16 kHz。
- Wav2Vec2-Large-XLSR-53 hidden layer 9。
- released `AutoPCP-multilingual-v2` comparator。
- 使用与现有评估兼容的 symmetrized protocol，并在报告写清实现版本。

SIM-O：

- generated target waveform 与 paired reference-target waveform 比较。
- 使用 WavLM-Large speaker embedding 的 cosine similarity。
- ZH→EN 的 reference target 是 CVSS-T 合成英文；EN→ZH 的 reference target 是真实中文 Common Voice。

注意：SimulS2ST-Omni 对公开 UniSS checkpoint 重评分后的绝对数字与 UniSS 原论文表格略有不同。因此 Track A 与 Track B 必须写入两个独立结果表，不能把不同协议的数值混在同一排名表中。

### Track C：COMPASS 风格的真实输入扩展诊断

目的：避免 EN→ZH synthetic-source bias，并扩大对 translation、naturalness、speaker、timing 的观察。

主集合：

```text
cvss_t_zh_en_compass_1000.jsonl
```

仅使用 ZH→EN 真实中文源输入。优先实现 compact 指标集：

- ASR-BLEU
- ASR-ChrF++
- ASR-TER
- COMET 或 BLASER（在模型权重与许可满足时）
- UTMOS
- NISQA
- WavLM speaker similarity
- AutoPCP
- source/generated duration ratio
- generated/reference duration ratio
- characters-per-second / speech-rate ratio
- failure rate、empty audio rate、truncation rate

Track C 是扩展诊断，不应拿它的 1,000 条结果与 Track A/B 的 4,897 条结果直接比较绝对数值。

### Track D：CVSS-T sentence-level streaming diagnostic

目的：评估当前 streaming Stage4/Stage6 checkpoint 在与 offline CVSS-T 相同内容上的 quality-latency trade-off。

限制：CVSS-T 只包含 sentence-level 朗读语音，因此：

- 可以测试 chunked sentence streaming。
- 不可声称 long-form continuous streaming 能力。
- 不可替代 RealSI/ACL60/60-dev 主 benchmark。
- ZH→EN 作为主要 streaming diagnostic；EN→ZH 保留但标记 synthetic source。

每个 checkpoint 固定跑多个 latency point，例如与模型训练定义一致的 chunk/multiplier 档位。每个 operating point 必须同时保存事件日志和生成音频。

指标分两类：

质量指标：

- ASR-BLEU
- Text-BLEU（若模型显式输出 text）
- AutoPCP
- SIM-O
- UTMOS
- SLC

streaming 指标：

- First WRITE / first audio latency
- LAAL
- computation-aware LAAL (`LAAL_CA`)
- Average Token Delay 或项目现有 ATD 定义
- active compute RTF
- wall-clock RTF
- wait/write event count
- chunk boundary discontinuity / click diagnostics
- truncation、repetition、premature emission、empty chunk rate

展示方式必须是 quality-latency frontier，例如：

```text
ASR-BLEU vs LAAL_CA
AutoPCP vs LAAL_CA
SIM-O vs LAAL_CA
ASR-BLEU vs active RTF
```

不能只给一个延迟数字，也不能只给一个 BLEU 数字。

## 8. 指标定义和对比规则

| 维度 | 主指标 | Reference | 方向 | 说明 |
|---|---|---|---|---|
| 文本翻译 | Text-BLEU ↑ | target text | 双向 | 只评价模型显式生成的翻译文本 |
| 语音内容 | ASR-BLEU ↑ | target text | 双向 | 先 ASR 生成语音再算 BLEU；受 ASR backend 影响 |
| 韵律保持 | AutoPCP ↑ | source audio | 双向 | 比较跨语言 prosody consistency |
| 说话人/音色 | SIM-O ↑ | paired target audio | 双向 | WavLM speaker embedding cosine |
| 时长一致性 | SLC 0.2/0.4 ↑ | source/generated duration | 双向 | 输出时长是否落在源时长的 ±20% / ±40% 内 |
| 自然度 | UTMOS ↑ | generated only | 双向 | 无 reference MOS predictor |
| 延迟 | LAAL / LAAL_CA ↓ | event timeline | streaming | sentence-level latency |
| 计算效率 | active RTF ↓ | source duration | streaming | 只累计非 idle inference time |
| 运行可靠性 | failure rate ↓ | manifest count | 全部 | 任何缺失样本必须显式统计 |

对比规则：

1. 同一张主表只放相同数据、相同方向、相同 normalization、相同 ASR backend、相同 decoding protocol 的数值。
2. UniSS 原论文数值与 SimulS2ST-Omni 重评分数值分表。
3. 4,897 全集和 COMPASS 风格 1,000 子集分表。
4. ZH→EN 与 EN→ZH 分列；双向平均仅作汇总。
5. EN→ZH 表头注明 `reversed synthetic-source`。
6. 对每个系统级指标计算 paired bootstrap 95% confidence interval；模型差异还要报告 paired delta 的 CI。
7. ASR-BLEU 最好抽样使用第二个 ASR backend 做 sensitivity audit，避免结论完全依赖 Whisper/Paraformer。

## 9. 当前代码可复用部分与待实现部分

### 9.1 可直接复用

```text
evaluation/cvss_manifest.py
evaluation/tests/test_cvss_manifest.py
evaluation/text_metrics.py
evaluation/asr_transcribe.py
evaluation/slc_metrics.py
evaluation/utmos_metrics.py
evaluation/autopcp_metrics.py
experiments/evaluation/uniss_full198_phase2_phase3/run_vllm_eval.sh
experiments/evaluation/uniss_full198_phase2_phase3/run_objective_metrics.sh
web_demo/offline_s2st_phase3_v1/inference_engine.py
```

其中 `text_metrics.py` 已实现目标语言相关的 UniSS/SimulS2ST-Omni normalization；`asr_transcribe.py` 已按英文 Whisper-large-v3、中文 Paraformer-zh 分流。

### 9.2 必须新增或扩展

建议将所有 CVSS 专用代码放在独立目录，避免修改历史 UniST 评估入口的默认行为：

```text
evaluation/cvss_t/
├── inventory.py
├── canonicalize_audio.py
├── join_metadata.py
├── leakage_audit.py
├── tokenize_to_parquet.py
├── build_manifests.py
├── sim_o_metrics.py
├── bootstrap_ci.py
├── report.py
└── tests/

experiments/evaluation/cvss_t_zh_en_phase3_v1/
├── README.md
├── configs/
│   ├── uniss_original.yaml
│   ├── simuls2st_omni_greedy.yaml
│   └── compass_zh_en_1000.yaml
├── prepare_data.sh
├── run_smoke.sh
├── run_offline_8gpu.sh
├── run_metrics_8gpu.sh
├── run_streaming_diagnostic.sh
├── verify_outputs.sh
└── write_report.sh
```

实现原则：

- 不改变历史 `uniss_full198_phase2_phase3` 实验目录的配置和结果。
- 若必须扩展通用 evaluator，应新增 opt-in 参数，保持旧参数的默认行为不变，并补 regression tests。
- 所有输出目录如果已存在且没有 `--resume`，必须拒绝覆盖。
- config、manifest hash、checkpoint hash 不一致时禁止 resume。

## 10. 实验与输出目录

建议本次独立实验目录：

```text
experiments/evaluation/cvss_t_zh_en_phase3_v1
```

建议输出目录：

```text
eval_outputs/cvss_t_zh_en_phase3_full198_iter_0009075_v1/
├── protocol_a_uniss_original/
│   ├── zh_en/
│   └── en_zh_synthetic_source/
├── protocol_b_simuls2st_omni_greedy/
│   ├── zh_en/
│   └── en_zh_synthetic_source/
├── protocol_c_compass_zh_en_1000/
├── streaming_diagnostic/
├── listening_samples/
├── audits/
├── plots/
└── report/
```

每个 generation 子目录至少包含：

```text
run_config.json
generation_results.jsonl
results.jsonl
wav/
source_wav/ 或原始路径索引
reference_wav/ 或原始路径索引
failures.jsonl
summary.json
metrics/
logs/
```

## 11. 分阶段执行与验收 Gate

### Gate 1：数据完整性

- 4,897 target WAV 全部存在且可读。
- 4,897 source MP3 全部存在且可读。
- source text、normalized English reference 全部非空。
- source 时长约 8.2 h，target 时长约 6.3 h。
- 双向 manifest 分别恰好 4,897 条。

未通过 Gate 1，不启动模型推理。

### Gate 2：tokenization smoke

对每方向 10 条运行：

- source GLM tokens 非空。
- source/target semantic tokens 非空。
- global tokens 恰好 32 个且范围合法。
- reference semantic tokens 可由 BiCodec 解码。
- tokenized record 能通过 `normalize_unist_record` 和 sample builder。

未通过 Gate 2，不启动全量 8 卡 tokenization。

### Gate 3：generation smoke

每方向 10 条、每种 protocol/mode 运行：

- 生成结果数与预期一致。
- dummy vocabulary token count = 0。
- empty semantic output = 0，或失败被明确记录。
- 音频可读、16 kHz mono、非空、无 NaN/Inf。
- ASR 与 BLEU 至少能完整跑通。

### Gate 4：100-pair pilot

每方向固定 100 条：

- 运行全部主指标。
- 人工听至少每方向 10 条。
- 检查语言方向、source/reference 是否接反。
- 检查 output 时长、重复 token、异常静音和截断。
- 验证 AutoPCP 使用 source audio，SIM-O 使用 paired target audio。

### Gate 5：4,897 × 2 全量

- generation completion = 100%。
- 每个失败有 ID、异常类型、log 和重试次数。
- 初次报告 failure rate；修复后 resume，不覆盖成功样本。
- 所有系统级 metric 同时保存 per-sample score。

### Gate 6：报告与可复现性

- 保存数据 inventory hash、manifest hash、checkpoint/export manifest、Git commit。
- 保存 Python/CUDA/PyTorch/vLLM/transformers/sacrebleu/ASR model 版本。
- 报告两种方向、两种 protocol，不混表。
- 报告 bootstrap CI、失败率和泄漏审计。
- 输出固定的试听样本索引与 HTML/Markdown 表格。

## 12. GPU 并行和性能计划

### 12.1 Tokenization

CVSS 两侧共 9,794 个方向音频记录，但原始音频只有 4,897 对，应缓存每个 waveform 的 token，避免双向重复编码。

建议：

- 8 GPU，按 stable hash 分成 8 份。
- 每 GPU 一个 tokenizer worker。
- part parquet 独立写入，最后按 ID 合并。
- canonical audio 与 token cache 以音频 SHA256 去重。

### 12.2 Offline generation

当前 0.5B Phase3 模型不需要 4/8 卡 tensor parallel；应采用 data parallel：

- 每 GPU 独立加载一份模型。
- manifest 稳定切成 8 份。
- 每 GPU 运行 vLLM worker。
- 合并时检查 ID/mode 唯一性与 expected count。

优先提高 scheduler queue 和 `max_num_seqs`，但不能为了追求 GPU utilization 改变 decoding 语义。greedy Track B 与 sampling Track A 必须保持各自协议。

### 12.3 ASR 和 speech metrics

建议初始设置：

- ASR：8 GPU，每卡 1 个进程；在显存稳定后，可尝试每卡 2 worker。
- UTMOS：按样本 8-way shard。
- AutoPCP：每卡 1 worker，`batch_size=16`，`chunk_size=512` 起步；此前更大 chunk 有 OOM 风险。
- SIM-O：8-way shard，长度排序后 batching。

所有 batch 增大都先跑 100-pair pilot，并验证数值与 batch=1 的 tolerance 内一致。

GPU 满载不是结果正确性的验收条件。首要条件是 protocol 不变、样本无丢失、指标 reference 正确；利用率只在不改变输出和 metric 的前提下优化。

## 13. 推荐的最终报告结构

报告建议保存为：

```text
docs/uniss_training_reproduction/cvss_t_zh_en_phase3_full198_evaluation_report.md
```

章节：

1. 模型、checkpoint 与代码版本。
2. 数据版本、pair 数、时长、hash 和许可。
3. Common Voice/CoVoST/CVSS join 审计。
4. 训练集泄漏审计。
5. Protocol A：UniSS-original 双向结果与 Table 1 对比。
6. Protocol B：SimulS2ST-Omni unified greedy 双向结果与 Table 1 对比。
7. Protocol C：真实 ZH→EN 扩展诊断。
8. 可选 streaming diagnostic 的 quality-latency frontier。
9. 95% bootstrap CI 与 per-sample error analysis。
10. 合成源偏置分析。
11. 失败样本、重复、截断、静音和 ASR backend sensitivity。
12. 固定试听样本及生成音频路径。
13. 与当前 UniST dev/test 离线结果的差异和可能的 domain shift。
14. 结论与下一步实验。

## 14. 试听样本设计

每方向固定选择 50 条，不按模型结果重新挑选，以避免 cherry-picking。分层维度：

- 短/中/长 source duration。
- 不同 Common Voice client ID。
- 数字、专有名词、缩写等 normalized text 难例。
- 高/中/低 ASR-BLEU。
- 高/中/低 AutoPCP 和 SIM-O。
- 生成时长过短/过长样本。

每条试听记录同时展示：

```text
pair ID
direction 与 synthetic-source 标记
source audio
source transcript
reference translation
reference target audio
generated text
ASR transcript
generated audio
per-sample metrics
```

## 15. 风险与处理原则

### 15.1 Common Voice v4 获取困难

旧版 Common Voice 的完整 archive 需要授权，匿名请求不可用。当前已经通过两个公开 CoVoST 2 test 镜像补齐并交叉验证 CVSS-T 所需的 4,897 个 v4 clips；报告必须声明这是可验证的 test 最小子集，不是完整 v4 archive。若未来获得官方完整包，应逐条比较音频 hash，不静默替换当前数据。

### 15.2 EN→ZH 合成输入偏置

反向输入来自 CVSS-T 合成英文，会比真实开放域英文更干净，也可能保留特定 TTS artifact。双向论文对比仍然保留，但真实性结论以 ZH→EN 为主。

### 15.3 ASR-BLEU backend 偏置

不同 ASR 对口音、韵律和合成 artifact 的容忍度不同。主表固定 Whisper-large-v3/Paraformer-zh，另取固定子集用第二 ASR backend 重评，检查模型排名是否稳定。

### 15.4 Speaker metric reference 选择错误

AutoPCP 与 SIM-O 的 reference 不同：前者按 SimulS2ST-Omni 使用 source audio，后者使用 paired target audio。代码和报告必须分别记录路径，不能复用一个模糊的 `reference_wav` 字段。

### 15.5 重建 reference 污染指标

正式 CVSS metric 必须使用官方/真实 paired waveform。不能将 reference semantic tokens 经本项目 BiCodec 重建后作为正式 reference，因为这会把 codec 误差引入 reference，并可能偏向本模型。

### 15.6 Streaming 结论外推

CVSS-T streaming 只能作为 sentence-level 附加诊断。长时连续输入、自然停顿、说话人打断、上下文累积和真正同传延迟仍需 RealSI/ACL60/60-dev。

## 16. 推荐执行优先级

推荐按以下顺序实施：

1. 已完成：获取并验证 CVSS-T 所需的 Common Voice v4 zh-CN test 最小子集。
2. 已完成：严格 filename join 和双向 raw-audio manifest。
3. 下一步：建立 canonical audio，并完成训练数据 leakage audit。
4. 实现可恢复的 8-GPU CVSS tokenization adapter。
5. 先跑 20 条 smoke，再跑每方向 100 条 pilot。
6. 对 full198 Phase3 iter 9075 跑 Track A 全量。
7. 对同一 checkpoint 跑 Track B greedy 全量。
8. 补 SIM-O、bootstrap CI 和最终报告。
9. 再决定是否做 Track C 扩展指标。
10. CVSS-T streaming diagnostic 最后做；正式 streaming 主结果仍以 RealSI/ACL60/60-dev 为准。

当前 CVSS-T test 数据下载不再构成阻塞。剩余工作是数据转换、泄漏审计、推理和指标实现。

## 17. 参考资料

1. Rongshen He et al., **SimulS2ST-Omni: Data-Efficient Streaming Speech-to-Speech Translation via Explicit Trajectory Supervision**, arXiv:2607.19810, 2026.

   <https://arxiv.org/abs/2607.19810>
2. Ye Jia et al., **CVSS Corpus and Massively Multilingual Speech-to-Speech Translation**, LREC 2022, arXiv:2201.03713.

   <https://arxiv.org/abs/2201.03713>
3. **UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice**, arXiv:2509.21144.

   <https://arxiv.org/abs/2509.21144>
4. Changhan Wang et al., **CoVoST 2: A Massively Multilingual Speech-to-Text Translation Corpus**, arXiv:2007.10310.

   <https://arxiv.org/abs/2007.10310>
5. **PROST-LLM: Progressively Enhancing the Speech-to-Speech Translation Capability in LLMs**, arXiv:2601.16618.

   <https://arxiv.org/abs/2601.16618>
6. **Benchmarking Speech-to-Speech Translation Models**（文中提出 COMPASS-S2ST framework）, arXiv:2606.03241.

   <https://arxiv.org/abs/2606.03241>
7. CVSS official repository/data description.

   <https://github.com/google-research-datasets/cvss>
8. CoVoST official repository and split construction instructions.

   <https://github.com/facebookresearch/covost>
