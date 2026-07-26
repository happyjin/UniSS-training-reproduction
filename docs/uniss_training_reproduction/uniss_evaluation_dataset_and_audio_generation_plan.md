# UniSS / UniST 评估数据集、论文复现与可试听音频生成计划

> 审计日期：2026-07-26
> 项目目录：`/opt/dlami/nvme/jasonleeeli/projects/UniSS`
> 参考资料：[UniSS 论文（arXiv:2509.21144）](https://arxiv.org/pdf/2509.21144)、[cmots/UniSS 官方仓库](https://github.com/cmots/UniSS)、[CVSS 论文](https://arxiv.org/pdf/2201.03713)

## 1. 结论摘要

1. **UniST 有独立 evaluation 数据。** 本地 `data/raw/UniST` 除 198 个 train shard 外，还包含独立的 `dev-00000.parquet`、`test-00000.parquet`，以及若干 HiFi-TTS/LibriSpeech 衍生的额外 dev/test split。
2. **当前完整 UniST dev 已经处理为训练期 validation 数据。** Phase1、Phase2、Phase3 的 JSONL 和 Megatron packed validation 文件都存在；Simul-UniSS 的 validation 也来自完整的 7,965 条 UniST dev。
3. **当前 UniST test 尚未处理成 Phase1-3 的 Megatron packed 文件，但不妨碍推理评估。** 现有音频生成程序可以直接读取 tokenized parquet，因此可先对原始 `test-00000.parquet` 做推理；只有在需要用训练程序计算 teacher-forced validation loss 时，才需要额外生成 packed test 数据。
4. **可以采用论文相同的指标定义评估当前模型，但 UniST test 结果不能冒充论文 CVSS-T 结果。** UniST dev/test 适合模型选择、Phase2/Phase3 对比和内部回归；论文主表用的是 CVSS-T test。
5. **论文主评测不是 UniST。** 论文使用 CVSS-T 的 4,897 对中英语音，中文 8.2 小时、英文 6.3 小时，同时报告 EN→ZH 与 ZH→EN；补充评测使用 FLEURS，情感主观评测使用 ESD 和 CREMA-D。
6. **本地已具备生成可试听 WAV 的基础链路。** 已有 13-shard Phase2/Phase3 结果，均包含 `source_wav/`、`reference_wav/`、`wav/`、`results.jsonl` 和 `summary.json`。full198 checkpoint 尚未导出成独立 Hugging Face checkpoint，需要先精确导出再推理。
7. **CVSS archive 已下载完成，但论文级 CVSS-T 中英评测尚缺源语音配对。** `/opt/dlami/nvme/jasonleeeli/CVSS` 约 224 GB，日志显示 42/42 archive 下载完成，`cvss_t_zh_en_v1.0.tar.gz` 也存在；但 CVSS archive 主要提供英文翻译语音及翻译文本，本地尚未发现配套 Common Voice v4 中文 source WAV 和已验证的 4,897 条映射 manifest。因此当前可以马上做 UniST 内部评估，暂时不能声称完整复现论文 CVSS-T 主表。

## 2. 论文中的正式评估口径

### 2.1 数据集

论文的主要 S2ST 评测设置为：

| 用途 | 数据集 | 规模/说明 |
| --- | --- | --- |
| S2ST 主结果 | CVSS-T test | 4,897 对语音；中文 8.2 小时、英文 6.3 小时；EN→ZH 与 ZH→EN |
| 泛化补充结果 | FLEURS test | 中英文方向的补充评测 |
| 情感保持主观评测 | ESD、CREMA-D | 每个数据集随机 300 条；happy、sad、angry、neutral 四类情感 |

因此应严格区分：

- `UniST dev/test`：当前训练数据发布包内的独立验证/测试集，用于内部评估和模型间公平对比。
- `CVSS-T test`：论文主表对应的正式 benchmark。
- `FLEURS/ESD/CREMA-D`：论文的补充泛化或主观评测，不是当前第一优先级。

### 2.2 推理参数

论文使用 vLLM，设置为：

```text
temperature = 0.7
top-k = -1
top-p = 0.8
repetition_penalty = 1.1
```

论文分别报告：

- **UniSS (Q)**：Quality mode。
- **UniSS (P)**：Performance mode。

本地 `direct_s2st` 可以作为额外消融实验，但不能混入论文主结果，也不能替代 Quality/Performance。

### 2.3 指标

#### Text-BLEU

使用 SacreBLEU 的 corpus-level `corpus_score`：

- 英文：转小写，去标点，但保留 apostrophe。
- 中文：转简体，去标点，逐字用空格分隔，使用 `zh` 模式。
- 对整个 split 汇总计算 corpus BLEU，不能先算逐句 BLEU 再简单平均。

注意：现有 `training/generate_unist_eval_audio.py` 中的 `generated_text_clean` 只是删除形如 `<|...|>` 的控制 token，尚不足以保证准确抽取 Quality/Performance 输出里的“翻译文本字段”。正式 Text-BLEU 前必须按 UniSS 控制 token 边界解析 transcription、translation 和 speech token 区段。

#### Speech-BLEU / ASR-BLEU

对生成语音先做 ASR，再与目标翻译文本计算 BLEU：

- 生成英文：Whisper-large-v3。
- 生成中文：Paraformer-zh。
- 文本规范化与 Text-BLEU 相同。

#### AutoPCP / A.PCP

比较 source speech 与 generated target speech 的音高、能量、时间结构等韵律特征，报告 prosody similarity。论文给出指标定义，但官方 UniSS 仓库没有附完整 AutoPCP 评估实现，因此需要单独接入与固定版本。

#### SLC-0.2 / SLC-0.4

设：

```text
duration_ratio = generated_duration / source_duration
```

- `SLC-0.2`：`abs(duration_ratio - 1) <= 0.2` 的样本比例。
- `SLC-0.4`：`abs(duration_ratio - 1) <= 0.4` 的样本比例。

#### UTMOS

对生成 WAV 计算神经网络预测 MOS，并报告 split 均值。必须记录 UTMOS 实现、checkpoint 和 commit/version，防止不同实现造成不可比结果。

#### 主观 MOS（后续可选）

论文由 6 名双语评测者以 5 分制评价：

- emotion similarity；
- speaker similarity；
- naturalness。

内部模型验证应先完成客观指标；只有在选定最终 checkpoint 后再组织盲测，避免过早投入人工成本。

## 3. 官方 GitHub 提供了什么、缺少什么

当前项目的 `origin` 指向官方 `cmots/UniSS`，本地已包含官方核心推理文件：

| 文件 | 能力 |
| --- | --- |
| `infer.py` | 单个真实 WAV 的 Transformers 推理示例 |
| `vllm_example.py` | 文件夹或 JSONL 的批量 vLLM 推理示例 |
| `configs/uniss.yaml` | 模型/语音 tokenizer 配置 |
| `uniss/cli/prompt.py` | Quality、Performance 等 prompt 组织 |
| `uniss/tokenizer.py` | GLM linguistic tokenizer 与 BiCodec 编解码入口 |

标准真实语音推理链路为：

```text
真实 source WAV
  -> UniSSTokenizer
       -> GLM linguistic tokens
       -> BiCodec global/speaker tokens
  -> Quality 或 Performance prompt
  -> UniSS LLM generation
  -> 提取生成的 BiCodec semantic tokens
  -> BiCodec decode
  -> generated WAV
```

官方仓库当前没有提供完整的一键论文 evaluation pipeline，尤其没有完整提供：

- CVSS-T 4,897 条中英配对 manifest 构造；
- Text-BLEU 的精确字段解析与语言规范化脚本；
- Whisper/Paraformer 双 ASR 的 Speech-BLEU 汇总；
- AutoPCP；
- UTMOS；
- 统一的 per-sample 指标、bootstrap 置信区间和最终报告生成。

因此，“能按照论文方式评估”的准确含义是：论文给出了数据、推理参数和指标定义，本地核心推理能力也存在；但还需要补齐 benchmark manifest 和指标实现，不能只运行官方 `infer.py` 就称为论文评估复现。

## 4. 当前本地 UniST 数据审计

### 4.1 总体状态

```text
data/raw/UniST
大小：约 29 GB
train shard：198 个（train-00000.parquet 到 train-00197.parquet）
```

### 4.2 主 dev/test

| 文件 | 条数 | 方向 | 来源数据集 | token 完整性 |
| --- | ---: | --- | --- | --- |
| `dev-00000.parquet` | 7,965 | cmn→eng 6,531；eng→cmn 1,434 | magicdata 6,531；gigaspeech 1,434 | 关键 token 字段无空值 |
| `test-00000.parquet` | 23,369 | cmn→eng 14,257；eng→cmn 9,112 | magicdata 13,206；gigaspeech 6,339；vctk 1,997；commonvoice_cn 1,051；commonvoice_en 776 | 关键 token 字段无空值 |

关键字段包括：

```text
transcription
translation
source_glm
source_bicodec
target_glm
target_bicodec
bicodec_global
src_lang
tgt_lang
duration_ratio
```

`bicodec_global` 的本地记录长度固定为 32；审计未发现 `source_glm`、`source_bicodec`、`target_glm`、`target_bicodec` 或 `bicodec_global` 为空的记录。

### 4.3 额外 dev/test split

| 文件 | 条数 | 方向 | 数据来源 |
| --- | ---: | --- | --- |
| `clean_dev-00000.parquet` | 83 | eng→cmn | hifi_tts |
| `clean_test-00000.parquet` | 185 | eng→cmn | hifi_tts |
| `other_dev-00000.parquet` | 270 | eng→cmn | hifi_tts |
| `other_test-00000.parquet` | 545 | eng→cmn | hifi_tts |
| `dev_clean-00000.parquet` | 1,074 | eng→cmn | librispeech |
| `dev_other-00000.parquet` | 957 | eng→cmn | librispeech |
| `test_clean-00000.parquet` | 2,927 | eng→cmn | librispeech 822；libritts_r 2,105 |
| `test_other-00000.parquet` | 3,140 | eng→cmn | librispeech 976；libritts_r 2,164 |

本次对上述 10 个非 train split 的 `id` 做了两两交集检查，未发现重叠。正式内部主结果建议先固定使用 `dev-00000.parquet` 和 `test-00000.parquet`；额外 split 单独报告，不能与主 test 拼接后只给一个总分，否则数据构成会改变且难以复现。

### 4.4 原始 WAV 与 token 重建的区别

本地 UniST parquet 是 codec/linguistic token 数据，**不包含原始 WAV 文件本身**。现有程序可以通过 BiCodec token 重建：

- `source_wav/`：由 `source_bicodec` 重建的输入语音；
- `reference_wav/`：由 `target_bicodec` 重建的目标参考语音；
- `wav/`：由模型生成 semantic token 后解码的输出语音。

这些文件足够用于试听、内部 SLC/AutoPCP/ASR-BLEU 对比，但必须在报告中注明 source/reference 是 codec reconstruction，而非数据集原始波形。论文 CVSS-T 评估应使用真实配对 WAV，不能用 UniST 重建音频替代。

## 5. 当前已经处理好的 validation 状态

### 5.1 Phase1-3 validation

完整 UniST dev 已处理为：

| 文件 | 行数 |
| --- | ---: |
| `data/processed/validation_unist_dev/phase1_dev.jsonl` | 31,860 |
| `data/processed/validation_unist_dev/phase2_dev.jsonl` | 23,895 |
| `data/processed/validation_unist_dev/phase2_dev_mix_2to1.jsonl` | 35,842 |
| `data/processed/validation_unist_dev/phase3_dev.jsonl` | 15,930 |
| `data/megatron/validation_unist_dev/phase1_valid_packed.jsonl` | 241 |
| `data/megatron/validation_unist_dev/phase2_valid_packed.jsonl` | 571 |
| `data/megatron/validation_unist_dev/phase3_valid_packed.jsonl` | 331 |

因此当前 TensorBoard 中 Phase1-3 的 validation loss 已经有独立 dev 数据来源，并不是直接拿 train shard 做验证。

### 5.2 Simul-UniSS validation

```text
data/processed/simul_uniss_v1/validation_dev/
```

其中包含 `manifest.json`、`samples.jsonl`、`schedules.jsonl`、`action_samples.jsonl` 和 `stats.json`。该 validation 来自完整的 7,965 条 UniST dev，已生成约 62,688 个 streaming events。它用于 simultaneous/streaming 训练验证，与本文档的离线 Phase2/Phase3 S2ST 评估应分开报告。

### 5.3 尚未处理的部分

- `test-00000.parquet` 尚无对应的 Phase1/2/3 Megatron packed test 文件。
- 这不阻塞当前 HF/vLLM 推理，因为推理可以直接读 parquet。
- 若以后需要报告 test teacher-forced loss，应在新目录创建 test JSONL/packed 文件，不修改或覆盖现有 validation 文件。

## 6. 当前已有可试听结果与可复用程序

### 6.1 已有输出

```text
eval_outputs/qwen0p5b_phase2_unist13_s2st_dev_20260716T174233Z/
eval_outputs/qwen0p5b_phase3_unist13_s2st_dev_20260717T114748Z/
```

两个目录均包含：

```text
source_wav/       9 WAV
reference_wav/    9 WAV
wav/              9 WAV
results.jsonl
summary.json
```

两次运行都记录为 9/9 生成成功、0 failure。9 个输出来自 3 条记录 × Quality/Performance/direct_s2st 三种 mode。论文式主对比时只应使用 Quality 与 Performance，因此 3 条记录会对应每个 checkpoint 6 个论文主模式结果。

另有真实测试音频：

```text
prompt_audio.wav
16 kHz，mono，约 4.3 秒
```

它适合做 arbitrary-WAV 单样本 sanity check，但不属于固定 benchmark，不能用于计算正式 BLEU。

### 6.2 已有脚本

```text
scripts/export_and_generate_qwen0p5b_phase2_audio_eval.sh
scripts/export_and_generate_qwen0p5b_phase3_audio_eval.sh
scripts/generate_unist_audio_eval.sh
training/generate_unist_eval_audio.py
scripts/convert_uniss_checkpoint.sh
```

当前链路是：

```text
Megatron checkpoint
  -> convert_uniss_checkpoint.sh export
  -> isolated Hugging Face checkpoint
  -> generate_unist_eval_audio.py 读取 UniST parquet
  -> 构造 Quality/Performance prompt
  -> 模型生成 text/semantic token
  -> BiCodec decode
  -> source/reference/generated WAV + JSONL
```

### 6.3 full198 checkpoint 现状

本次审计时的 tracker 为：

| 阶段 | checkpoint | latest iteration |
| --- | --- | ---: |
| Phase2 full198 | `checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4` | 15,381 |
| Phase3 full198 | `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4` | 9,075 |

注意：在真正执行前必须再次读取 tracker，并将“实际导出的 iteration”写入 manifest。不能仅使用目录名中的 `latest`，否则训练继续后同一个命令可能导出不同权重。

当前尚未发现这两个 full198 checkpoint 对应的固定、完成验证的 HF export。现有 Phase2 wrapper 默认面向 `unist13_full`，且不像 Phase3 wrapper 那样显式拼出 `iter_XXXXXXX` 子目录。因此正式 full198 评估应新建独立 wrapper 或增强参数校验，不要直接修改历史 13-shard 脚本的默认行为。

## 7. 推荐的三层评估方案

### Level A：可试听 sanity check

目标：最快确认 checkpoint 导出、prompt、生成 token、BiCodec 解码和方向都正常。

顺序：

1. 固定 3 条记录，至少包含 1 条 eng→cmn 和 1 条 cmn→eng。
2. Phase2 与 Phase3 使用完全相同的 sample ID。
3. 每条分别运行 Quality、Performance。
4. 检查 source/reference/generated 三组 WAV 是否可播放、采样率是否为 16 kHz、是否出现空音频/极短音频/重复 token。
5. 通过后扩展到固定 10 条，再扩展到分层抽样 50 条供人工试听。

50 条建议按方向与来源分层，不采用 parquet 的“前 50 条”，否则容易只覆盖单一方向或单一来源。

### Level B：UniST dev/test 内部完整评估

目标：公平比较 full198 Phase2 与 Phase3，并建立以后实验的回归基线。

推荐矩阵：

| 维度 | 取值 |
| --- | --- |
| checkpoint | Phase2 exact iteration；Phase3 exact iteration |
| split | UniST dev；UniST test |
| direction | eng→cmn；cmn→eng |
| mode | Quality；Performance |
| decoding | temperature 0.7；top-k -1；top-p 0.8；repetition penalty 1.1 |

执行优先级：

1. 3 条 smoke。
2. 50 条固定试听集。
3. 完整 dev 7,965 条。
4. 完整 test 23,369 条。

完整 test 对单 checkpoint、两种 mode 会产生 46,738 次生成；Phase2 与 Phase3 共 93,476 次生成，因此应先用 dev 验证指标和吞吐，再安排完整 test。可先完成无音频 decode 的 Text-BLEU 推理，再对同一生成结果批量解码，避免模型推理与 codec 解码互相阻塞。

报告时：

- 每个方向单独报告，不能只给混合方向总分。
- Quality 和 Performance 单独报告。
- Phase2、Phase3 使用同一 manifest、同一 seed、相同推理参数和相同程序 commit。
- sampling 是随机的；正式主运行应固定 seed。为了估计随机波动，可在 200～500 条子集上附加 3 个 seed，不必一开始就对全 test 做三次。
- `direct_s2st`、TTS、simultaneous/streaming 实验放到独立消融表。

### Level C：CVSS-T 论文级评估

目标：得到能够与论文主表同一 benchmark 口径比较的结果。

必要条件：

1. 解压并校验 `cvss_t_zh_en_v1.0.tar.gz`。
2. 获取与该版本完全匹配的 Common Voice v4 中文 source speech。
3. 按 CVSS metadata/filename 验证每条 source Chinese WAV、target English CVSS-T WAV 和 translation text 的一一映射。
4. 构造固定的 4,897-pair manifest，记录相对路径、sample ID、时长、文本和校验和。
5. 明确两方向构造：ZH→EN 使用 Common Voice 中文为 source、CVSS-T 英文为 target；EN→ZH 反向使用同一配对时，必须验证与论文做法一致并在报告中写清楚。
6. 分别运行 Quality/Performance 和 Phase2/Phase3。
7. 计算 Text-BLEU、Speech-BLEU、AutoPCP、SLC、UTMOS。

本地当前 CVSS 状态：

```text
/opt/dlami/nvme/jasonleeeli/CVSS             约 224 GB
CVSS archive download                         42/42 complete
CVSS-T zh_en archive                          3,103,162,925 bytes（约 2.89 GiB）
CVSS-C zh_en archive                          2,961,540,319 bytes（约 2.76 GiB）
```

论文主表明确使用 **CVSS-T**，不能用 CVSS-C 替换后仍称为论文主表结果。当前 archive 下载成功不等于 benchmark 已准备好；缺少配套 Common Voice v4 source WAV 和经过验证的 4,897 条 manifest 时，论文级推理仍未具备前置条件。

## 8. Phase2 与 Phase3 的公平对比设计

### 8.1 固定不变的项目

- 同一数据 split 与 sample ID 顺序。
- 同一语言方向划分。
- 同一 Quality/Performance prompt 实现。
- 同一推理参数。
- 同一 `max_new_tokens` 或按同一规则动态设置上限。
- 同一 BiCodec/UniSSTokenizer 权重。
- 同一 ASR、AutoPCP、UTMOS 版本。
- 同一硬件与 vLLM 配置，或至少记录差异。
- 同一代码 commit 和环境快照。

### 8.2 只改变的项目

- checkpoint：Phase2 exact iteration 对 Phase3 exact iteration。

### 8.3 需要报告的结果

每个 checkpoint × mode × direction 至少报告：

```text
sample_count
generation_success_rate
empty/no-semantic-token rate
Text-BLEU
Speech-BLEU
AutoPCP
SLC-0.2
SLC-0.4
UTMOS mean/std
input/output duration statistics
generation throughput / RTF（可选，但建议）
```

Phase3 是 high-quality fine-tuning，预期更应关注音质、韵律和自然度是否改善；不能只看训练/validation loss 判断 Phase3 一定优于 Phase2。

## 9. 建议的输出目录格式

每次运行使用唯一目录，禁止复用历史目录。建议：

```text
eval_outputs/
  qwen0p5b_phase2_unist198_iter15381_unist_dev_quality_<UTC>/
  qwen0p5b_phase2_unist198_iter15381_unist_dev_performance_<UTC>/
  qwen0p5b_phase3_unist198_iter9075_unist_dev_quality_<UTC>/
  qwen0p5b_phase3_unist198_iter9075_unist_dev_performance_<UTC>/
  qwen0p5b_phase2_unist198_iter15381_unist_test_quality_<UTC>/
  qwen0p5b_phase3_unist198_iter9075_unist_test_quality_<UTC>/
  qwen0p5b_phase3_unist198_iter9075_cvss_t_zh_en_quality_<UTC>/
  qwen0p5b_phase3_unist198_iter9075_cvss_t_en_zh_quality_<UTC>/
```

iteration 数字只是本次审计快照；执行时应替换为实际冻结的 checkpoint iteration。

每个目录建议包含：

```text
source_wav/             source 音频或 UniST token 重建音频
reference_wav/          target reference 音频或 token 重建音频
wav/                    模型生成音频
results.jsonl           每条生成文本、路径、错误、token 数、时长
summary.json            生成成功/失败统计
manifest.json           样本 ID、输入文件、方向、顺序、校验和
run_config.yaml         checkpoint、mode、seed、sampling 参数、版本
metrics/
  per_sample.jsonl
  text_bleu.json
  speech_bleu.json
  autopcp.json
  slc.json
  utmos.json
  aggregate.json
logs/
```

目录创建规则应改为：目标目录已存在则报错退出，除非显式传入专门的 resume 参数。现有生成脚本传入了 `--overwrite`，它只会清理当前目录的 metadata，但正式新 wrapper 应通过唯一 `RUN_ID` 和存在性检查降低误覆盖风险。

## 10. 实际实施步骤

### Step 0：冻结评估对象

1. 等待目标 checkpoint 完整写盘，确认不存在临时 shard。
2. 记录 `latest_checkpointed_iteration.txt`。
3. 检查目标 `iter_XXXXXXX` 目录结构与权重文件完整性。
4. 记录 Git commit、dirty status、conda package snapshot、CUDA/PyTorch/Transformers/vLLM 版本。
5. 不修改、移动或覆盖训练 checkpoint。

### Step 1：新增独立 full198 评估目录和 wrapper

建议新增：

```text
experiments/evaluation/uniss_full198_phase2_phase3/
  README.md
  export_phase2_exact.sh
  export_phase3_exact.sh
  run_unist_smoke.sh
  run_unist_dev.sh
  run_unist_test.sh
  configs/
  manifests/
```

新脚本只调用现有通用转换和推理代码，不更改历史 13-shard wrapper 的默认 checkpoint、目录或行为。

### Step 2：精确导出 HF checkpoint

建议输出：

```text
checkpoints/exported_hf/qwen0p5b_phase2_unist198_iterXXXXXXX_hf/
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iterXXXXXXX_hf/
```

导出后验证：

- tokenizer/control token 逻辑词表为 180,407，且 token ID 与训练配置一致；
- Megatron 导出的 embedding/config 词表为补齐后的 180,480（额外 73 个 dummy rows），与已有 13-shard HF export 一致；
- 权重能被 `AutoModelForCausalLM.from_pretrained` 加载；
- 用固定 prompt 做一次 forward/generate；
- 记录源 Megatron checkpoint 路径和导出日志。

按本次审计快照，精确导出的命令形态如下（这是执行示例，本次仅写计划，未实际运行导出）：

```bash
scripts/convert_uniss_checkpoint.sh export \
  --hf-model checkpoints/qwen2_0p5b_uniss_vocab_hf \
  --megatron-path checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/iter_0015381 \
  --hf-output checkpoints/exported_hf/qwen0p5b_phase2_unist198_iter0015381_hf \
  --model-type gpt \
  --no-progress

scripts/convert_uniss_checkpoint.sh export \
  --hf-model checkpoints/qwen2_0p5b_uniss_vocab_hf \
  --megatron-path checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075 \
  --hf-output checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter0009075_hf \
  --model-type gpt \
  --no-progress
```

正式执行前仍应重新读取 tracker，不应无条件复制上述 iteration。

### Step 3：生成固定 manifest

从 parquet 按 `id` 构造，不依赖“前 N 行”：

- `unist_smoke_3.jsonl`；
- `unist_listen_50.jsonl`；
- `unist_dev_all.jsonl`；
- `unist_test_all.jsonl`。

manifest 至少包含：

```text
id
parquet_path
row_index
dataset_name
src_lang
tgt_lang
transcription
translation
source/target token length
duration_ratio
```

### Step 4：3 条 smoke 与 50 条试听

先沿用现有 `training/generate_unist_eval_audio.py` 链路，确保：

- Phase2/Phase3 均可加载；
- Quality/Performance 均产生 semantic token；
- 所有 WAV 可以被 `ffprobe`/`soundfile` 正确读取；
- 采样率 16 kHz；
- 输出长度合理；
- `results.jsonl` 无 error；
- 同一 sample 的 source/reference 在不同 checkpoint 目录中内容一致。

通过后，用户即可直接试听各目录中的：

```text
source_wav/<same_sample>.wav
reference_wav/<same_sample>.wav
wav/<same_sample>.wav
```

在 HF checkpoint 已完成精确导出的前提下，现有通用生成脚本的 smoke 命令形态为：

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
HF_CHECKPOINT=checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter0009075_hf \
SPLIT=dev \
LIMIT_RECORDS=3 \
MODES="quality performance" \
SAVE_SOURCE_AUDIO=1 \
EVAL_CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR="eval_outputs/qwen0p5b_phase3_unist198_iter9075_unist_dev_smoke_${RUN_ID}" \
scripts/generate_unist_audio_eval.sh
```

该现有命令取 parquet 的前 3 行，只适合初步链路检查；公平双方向对比应先实现 manifest 选择，不能把它直接当正式 3 条测试集。

### Step 5：补齐论文参数与批量 vLLM

现有 HF `model.generate` 脚本已经支持 temperature、top-p、repetition penalty，但没有显式 top-k 参数，也不是论文使用的 vLLM 批量路径。正式全量前应：

1. 为 HF smoke 路径加 `top_k=-1` 的明确配置/记录，或说明 Transformers 中等价的禁用 top-k 行为。
2. 使用官方 `vllm_example.py` 的输入/输出处理构建 manifest 驱动的批量 runner。
3. 固定 seed、batching 策略、tensor parallel 和 vLLM 版本。
4. 确保 HF smoke 与 vLLM 对同一输入的 prompt token 完全一致。
5. 将生成 token/文本先落盘，再批量解码 WAV，使失败可续跑。

### Step 6：实现指标流水线

建议新增独立目录：

```text
evaluation/
  text_normalization.py
  parse_uniss_outputs.py
  compute_text_bleu.py
  transcribe_generated_audio.py
  compute_speech_bleu.py
  compute_autopcp.py
  compute_slc.py
  compute_utmos.py
  aggregate_report.py
  tests/
```

每个指标脚本都读取 `results.jsonl/manifest.json`，输出 per-sample 和 aggregate 文件；任何失败样本必须进入 failure table，不能静默排除。

### Step 7：运行 UniST dev，再运行 test

验收门槛：

- generation success rate 接近 100%；
- 无系统性空 semantic token；
- 两方向都有样本与结果；
- Phase2/Phase3 样本 ID 完全一致；
- source/reference 音频校验和跨 checkpoint 一致；
- BLEU 输入条数与成功样本数一致；
- 指标汇总可从 per-sample 文件重复计算。

### Step 8：补齐 CVSS-T source 并构造论文 manifest

1. 只解压到新的只读式数据目录，不修改 archive。
2. 获取与 CVSS v1.0 对应的 Common Voice v4 中文数据。
3. 按官方 metadata 配对并验证恰好 4,897 条。
4. 用 `ffprobe`/`soundfile` 校验所有 WAV 可读、采样率和时长。
5. 复核总时长是否接近论文的中文 8.2 小时、英文 6.3 小时。
6. 生成带 SHA256 的冻结 manifest。
7. 先 3/50 条 smoke，再做完整双向评估。

### Step 9：生成最终报告

最终至少形成：

- UniST dev/test Phase2 vs Phase3 的 Q/P 双方向指标表；
- 50 条可试听对照清单；
- 失败样本与异常类型统计；
- CVSS-T 条件具备后生成论文口径主表；
- 所有命令、commit、checkpoint、manifest 和指标版本记录。

## 11. 当前即可使用的试听方式与执行边界

现有 13-shard 音频可以马上试听，无需重新运行：

```text
eval_outputs/qwen0p5b_phase2_unist13_s2st_dev_20260716T174233Z/
eval_outputs/qwen0p5b_phase3_unist13_s2st_dev_20260717T114748Z/
```

full198 推荐在新增 wrapper 后执行，而不是临时覆盖默认变量后直接长跑。最安全顺序为：

```text
精确导出 Phase2/Phase3 HF checkpoint
  -> 固定 3 条双方向 smoke
  -> 固定 50 条试听
  -> UniST dev 全量
  -> UniST test 全量
  -> 补齐 CVSS-T source
  -> CVSS-T 论文级评估
```

如果当前 8 卡训练仍在运行，建议不同时启动大规模 vLLM/ASR/UTMOS GPU 作业。CPU 侧 manifest 构建和脚本测试可以并行准备；GPU smoke 应指定未被训练占用的卡，或等训练 checkpoint 稳定后再运行，以避免训练吞吐下降和显存争抢。

## 12. 实施时的“不破坏历史实验”要求

- 不修改或删除任何已有 `eval_outputs/` 子目录。
- 不覆盖 13-shard 的 export、日志、checkpoint 和结果。
- full198 HF export 使用带 exact iteration 的新目录。
- 新评估脚本放入独立目录，公共代码修改必须保持原 CLI 默认行为。
- 所有新输出使用 UTC timestamp 和 checkpoint iteration。
- 在大规模运行前执行单元测试、`--help`、dry-run 和 3 条 smoke。
- 每个重要且通过测试的增量修改再独立 commit/push；本计划文档本身不授权立即启动 GPU 全量评估。
- 保留当前未跟踪的 `Simul_UniSS_方案分析与实施建议.docx`，不得纳入或修改本评估工作。

## 13. 最终回答用户问题

### UniST 有单独 evaluation 数据集吗？

有。本地有独立 UniST dev 7,965 条、test 23,369 条，以及多个额外 dev/test split；非 train split 的关键 token 字段完整，本次检查的 split ID 之间无重叠。

### 当前处理好的数据里有吗？

有。完整 dev 已处理为 Phase1/2/3 validation 和 Simul-UniSS validation。test 原始 tokenized parquet 已下载完整，但尚未做成 Megatron packed test；音频推理可以直接读取它，不受影响。

### 可以按照论文方式评估吗？

可以分两层回答：

- **UniST 内部评估：现在即可开始。** 可以使用论文相同的 Q/P 推理参数与 Text-BLEU、Speech-BLEU、AutoPCP、SLC、UTMOS 定义，但结果应标记为 UniST dev/test。
- **论文 CVSS-T 主表复现：当前还差数据配对和指标流水线。** CVSS archive 已全部下载，仍需 Common Voice v4 中文 source WAV、4,897 条严格映射 manifest，以及完整指标实现。

### 如何得到类似 `eval_outputs` 中可试听的音频？

使用现有链路将 exact Megatron checkpoint 导出为独立 HF checkpoint，再让 `training/generate_unist_eval_audio.py` 读取固定 UniST parquet 记录。它会分别保存 source token 重建音频、target reference token 重建音频和模型生成音频。先生成 3 条 smoke，再做同一批 50 条 Phase2/Phase3 对照，最后才扩展到 dev/test 全量。每次运行必须写入新的唯一目录，不能覆盖当前已有结果。
