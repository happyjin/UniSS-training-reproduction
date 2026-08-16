# Stage A 正式 18k Pack 构建报告

日期：2026-08-16（UTC）

## 结论

Stage A pilot15 正式 train/validation 数据已使用既定 Phase3 v4 tokenizer、固定 speaker snapshot 和 seq=18000 完成并行 packing。构建过程未覆盖任何历史 pack，train 使用 26 workers、validation 使用 4 workers。

正式 pack Gate：**PASSED**。

## 输入

| Split | Manifest | 源记录数 |
|---|---|---:|
| train | `data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal/formal_train_manifest.jsonl` | 1,325,243 |
| valid | `data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal/formal_valid_manifest.jsonl` | 13,469 |

共同 provenance：

- tokenizer/model：`checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf`；
- source snapshot：`data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json`；
- sequence length：18,000；
- speaker global tokens：固定 32 tokens；
- packing：按 worker 区间保持源顺序，最终按 part 编号稳定合并并生成二进制 offset index。

## 输出统计

| 指标 | Train | Valid |
|---|---:|---:|
| packs | 16,195 | 167 |
| used tokens | 288,826,134 | 2,936,165 |
| fill ratio | 99.0793% | 97.6768% |
| acoustic annotations | 1,060,452 | 10,809 |
| streaming ASR samples | 794,291 | 8,026 |
| causal full ASR samples | 266,161 | 2,783 |
| offline ASR replay samples | 198,675 | 1,979 |
| Phase3 replay samples | 66,116 | 681 |
| packed JSONL size | 8,060,657,892 bytes | 82,833,002 bytes |

输出路径：

- `data/megatron/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/train_packs_18k_v1.jsonl`
- `data/megatron/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/valid_packs_18k_v1.jsonl`

索引一致性：

- train JSONL 行数、build report pack count、offset index records 均为 16,195；
- valid JSONL 行数、build report pack count、offset index records 均为 167。

Build report SHA-256：

- train：`df1f4e58946c23b4f97f000e1abba9d02c98014b96d0075ae5839038bfd8d159`
- valid：`175f916ce6b99524dd02fd7a880a1daeadda6da7b7a2f9b58bb0dbdc49ec1bcd`

## 正式训练几何

采用 8 GPU、MBS=1、GBS=128、三轮严格 coverage：

```text
train packs              = 16,195
steps per coverage epoch = ceil(16,195 / 128) = 127
epoch samples            = 127 * 128 = 16,256
coverage epochs          = 3
train iterations         = 127 * 3 = 381
train samples            = 381 * 128 = 48,768
validation packs         = 167
eval global batch        = 8
eval iterations          = ceil(167 / 8) = 21
warmup iterations        = min(200, ceil(0.03 * 381)) = 12
```

每个 coverage epoch 仅有 61 个 padding schedule slots；`ThreeEpochStageASchedule` 会对每轮独立做确定性全局 shuffle，并保证 16,195 个真实 packs 每轮全部覆盖一次。padding 只用于对齐 GBS，不改变真实 pack 内容。

## 训练配置

正式入口：原生 Megatron compound Stage A。

| 参数 | 值 |
|---|---:|
| GPUs | 8 |
| TP / PP | 1 / 1 |
| seq | 18,000 |
| MBS / GBS | 1 / 128 |
| gradient accumulation | 16 microbatches/rank/update |
| precision | BF16 |
| train iterations | 381 |
| warmup | 12 |
| save interval | 100 |
| eval interval | 50 |
| eval iterations | 21（完整固定 valid） |
| max acoustics/pack | 2 |

fresh run 从 Phase3 v4 native iteration 9075 做受审计的 non-strict handoff；任何后续恢复必须走独立 strict-resume 脚本，载入 optimizer、RNG 和 sampler state。

