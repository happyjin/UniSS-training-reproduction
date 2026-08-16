# Stage A 数据准备结果报告：因果 ASR 事件与 OOV-free source CTC

## 1. 结论

Stage A 的**数据准备硬门已经通过**。这只授权继续实现和 smoke test 原生 Megatron Stage A 训练器，不表示 Stage A 模型训练已经完成。

最终权威运行使用：

- source snapshot：`source_snapshot_v5.json`；
- CTC map：`ctc_maps_utf8_byte_v5`；
- 全量数据审计：`audit_20260816T062547Z`；
- 代码截至 commit：`32d1b95`。

最终结果中，train 1,325,243 条、validation 13,469 条全部通过；无缺失音频、无 schema rejection、无重复 ID、train/validation ID 重叠为 0、CTC OOV 为 0、CTC 不可对齐样本为 0。

## 2. 隔离与不可变性

本阶段只在以下实验专属位置新增代码和产物：

- 代码：`experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_whisper_asr`；
- 数据审计：`data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr`；
- 日志：`logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a`；
- 报告：`reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_whisper_asr`。

历史 manifest、CTC map、失败审计、训练脚本、checkpoint 和评估结果均未被覆盖或修改。所有新构建器在输出目录已经存在时直接退出。

## 3. 冻结输入

| 项目 | 数值 |
|---|---|
| train manifest | `formal_train_manifest.jsonl` |
| train records | 1,325,243 |
| train bytes | 40,573,534,124 |
| train SHA256 | `cc2c850e20f5d5b346be5fd44b8dfa98093fd44be1f99d0276e57cc28a3716f3` |
| validation records | 13,469 |
| validation bytes | 411,707,502 |
| validation SHA256 | `2981204ad0db9ca08bbc5ec206203ed1c8f7a410ad7a211cd632661e14d91483` |
| native initialization | `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075` |
| source snapshot SHA256 | `43f4abc9d06b3d7532222db6309dad2e4d43c8b5e8d126055db81ffdc9fefb3c` |

固定 speaker condition 来自 validation 第一条跨会话样本 `NCSSD_R_EN_0000000083`，严格为 32 个 global token。它不是当前 utterance 的完整音频编码，因此不会把当前 utterance 的未来信息泄漏到最早 ASR prompt。

## 4. Append-only ASR 事件

事件构造只使用已经结束的 source word 和 causal source GLM timestamp：

1. word end 对齐到 160 ms tick；
2. 每次只追加新结束的 word，不允许 rollback；
3. 连续 1,280 ms 没有完整 word 时插入 empty-delta 事件；
4. 最后一个事件必须覆盖完整 source text 与全部 source GLM；
5. 相邻事件的 word span 和 GLM span 必须零 gap、零 overlap。

### 4.1 全量统计

| 指标 | Train | Validation |
|---|---:|---:|
| records | 1,325,243 | 13,469 |
| 音频小时 | 2,393.2504 h | 24.3185 h |
| events | 22,791,563 | 230,832 |
| 平均 events/record | 17.1980 | 17.1380 |
| events with text | 21,689,016 | 219,628 |
| empty-text events | 1,102,547 | 11,204 |
| empty-text event 比例 | 4.8375% | 4.8537% |
| pre-final text commit | 1,325,199 | 13,469 |
| final-only text | 44 | 0 |
| maximum events | 89 | 74 |
| source words | 23,163,695 | 234,452 |
| source GLM tokens | 108,489,720 | 1,102,419 |

Train 中 44 条 final-only 样本占 `0.00332%`。它们没有伪造早期文字 commit，仍保留用于最终完整 ASR replay；正式训练/validation 必须单列这个计数，不能把它们误报为 pre-final streaming 成功。

### 4.2 方向与时长分布

| 分布 | Train | Validation |
|---|---:|---:|
| EN→ZH | 759,975 / 57.3461% | 7,761 / 57.6212% |
| ZH→EN | 565,268 / 42.6539% | 5,708 / 42.3788% |
| `<4 s` | 26.7481% | 26.4830% |
| `4–8 s` | 48.1233% | 48.6302% |
| `8–15 s` | 21.4306% | 21.1671% |
| `>=15 s` | 3.6980% | 3.7197% |

Validation 与 train 的方向及时长分布接近，且 ID 集合严格不相交。

## 5. CTC 词表失败、根因与最终修复

### 5.1 失败 1：复用历史 joint-data Qwen map

保留的失败审计：`audit_20260816T041412Z`。

| Split | CTC target token | OOV token | OOV rate |
|---|---:|---:|---:|
| Train | 23,629,870 | 8,954 | 0.03789% |
| Validation | 239,095 | 108 | 0.04517% |

根因不是数据损坏，而是 provenance 不一致：历史 map 来自另一套 joint source/target 原始文本；当前 Stage A canonical transcript 来自强制对齐的 word sequence。两套 normalization 的罕见拼写、大小写、无标点形式会产生旧 map 未包含的 Qwen token。

### 5.2 失败 2：只由当前 train canonical transcript 建 Qwen map

保留的失败构建：`ctc_maps_train_canonical_v2`。

- train OOV 可降为 0；
- validation 仍有 90 个 OOV token：ENG 45、CMN 45；
- 这些 OOV 主要是只在 validation 出现一次的罕见 subword、专名或字符。

不能把这 90 个 validation token 直接加入 map，因为那会让 validation label 决定训练词表，造成评估泄漏。把 CTC head 扩到完整 180,407 logical Qwen vocabulary 又会产生不必要的大分类头。

### 5.3 最终方案：固定 UTF-8 byte CTC

最终 source CTC 使用与标签无关的固定词表：

```text
label 0..255 = UTF-8 byte
blank        = 256
output size  = 257
```

这只改变辅助 source CTC head。主 AR-ASR 仍使用 Phase3 的 Qwen tokenizer、原始 special-token grammar 和原生 Qwen decoder，不会把 Phase3 主生成路径改成 byte language model。

该设计的直接收益：

- train/validation/未来文本天然零 OOV；
- ENG、CMN 和混合 Unicode 使用同一无标签泄漏 inventory；
- CTC head 从数万类缩小到 257 类；
- UTF-8 round-trip 可做逐条无损硬门。

代价是 CTC label 序列比 Qwen subword 更长，因此另外执行了 `target length + adjacent-repeat blanks <= 20 ms encoder frames` 的逐样本可行性审计。

## 6. UTF-8 byte CTC 可行性

| 指标 | Train | Validation |
|---|---:|---:|
| UTF-8 byte labels | 118,758,864 | 1,201,387 |
| 平均 labels/record | 89.6129 | 89.1965 |
| minimum CTC steps | 120,287,693 | 1,216,930 |
| available 20-ms frames | 430,785,076 | 4,377,335 |
| aggregate minimum/frame ratio | 27.9229% | 27.8007% |
| maximum minimum CTC steps/record | 524 | 435 |
| CTC infeasible records | 0 | 0 |
| OOV tokens | 0 | 0 |

`minimum CTC steps` 已包含相邻重复 byte 必须插 blank 的额外步数。0 条 infeasible 表示所有目标都能在严格 20 ms pre-VQ frame 时间轴上完成标准 CTC 对齐。

## 7. 最终硬门

权威审计 `audit_20260816T062547Z` 的所有检查均为 true：

| Gate | 结果 |
|---|---|
| train/validation 全记录通过 | PASS |
| source audio 存在 | PASS |
| train/validation ID 唯一 | PASS |
| train/validation ID overlap = 0 | PASS |
| train/validation CTC OOV = 0 | PASS |
| train/validation CTC infeasible = 0 | PASS |
| train/validation 有 pre-final commit | PASS |
| event text append-only | PASS |
| word/GLM span 零 gap、零 overlap | PASS |

最终审计 SHA256：`fab404612a88154aabc736be5d85008bbe90370314db6cce6b2f01d6f55a074f`。

## 8. 测试与提交

最终 CPU test：`27 passed`，同时通过全部 shell `bash -n`、Python compilation 和 `git diff --check`。

与本阶段数据门相关的提交：

| Commit | 内容 |
|---|---|
| `a0177dd` | append-only Stage A 事件与全量审计 |
| `c22023b` | Stage A 数据脚本格式与启动规范化 |
| `d142646` | train-derived Qwen map 与 validation OOV 审计 |
| `1a2a170` | 固定 UTF-8 byte CTC fallback |
| `69617f8` | byte/Qwen map 通用 gate 修复 |
| `32d1b95` | worker maximum 聚合修复 |

## 9. Stage A 训练入口约束

下一步可以开始实现训练器，但必须继续满足：

1. 正式 Qwen trainer 使用原生 Megatron，不使用 HF Qwen 作为权威训练模型；
2. fresh load native Phase3 `iter_0009075`；
3. causal WhisperVQ、byte CTC head、incremental AR-ASR 和 exact Phase3 replay 同时存在；
4. codebook/EMA/post-VQ 冻结，不能改变 WhisperVQ token geometry；
5. 使用 strict global shuffle；
6. 先 CPU tests，再单卡 real-checkpoint test，再 8 卡 20–50 step smoke；
7. smoke 的 loss、gradient、native checkpoint handoff、GPU utility/power 全通过后，才允许启动 3-epoch formal training；
8. formal training 前停止已知 synthetic GPU load，但不得杀死身份不明的 GPU 进程。

## 10. 原始证据

- 最终 snapshot：`data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json`
- 最终 CTC report：`data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/ctc_maps_utf8_byte_v5/ctc_map_build_report.json`
- 最终 data audit：`data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/audit_20260816T062547Z/stage_a_data_audit.json`
- 旧 map 失败：`data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/audit_20260816T041412Z/stage_a_data_audit.json`
- train-only map 失败：`data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/ctc_maps_train_canonical_v2/ctc_map_build_report.json`
- 最终 CTC report SHA256：`52a4f956981adbb9563cc90675e103bed77bf7743756e72f5635560b2d30e804`
