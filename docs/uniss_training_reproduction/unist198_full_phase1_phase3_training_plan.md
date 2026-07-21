# UniST-198 完整数据 Phase 1–3 训练就绪性检查与隔离实施方案

日期：2026-07-21

## 1. 结论

当前本机已经具备完整的 UniST 本地训练原始数据和 Phase 1/2/3 分 shard 中间数据，但**还不能直接开始完整数据训练**，原因是三个最终 Megatron packed 训练文件尚未完成。

当前状态可以概括为：

| 环节 | 状态 | 是否可直接训练 |
| --- | --- | --- |
| UniST train 原始 Parquet | 198/198 shards 完整 | 否，仍是原始格式 |
| Phase 1 JSONL | 198/198 shards 完整 | 否，未完成 18k packing |
| Phase 2 JSONL | 198/198 shards 完整 | 否，未完成 18k packing |
| Phase 3 JSONL | 198/198 shards 完整 | 否，未完成 18k packing |
| Phase 2 2:1 mix | 已完成，89,036,658 samples | 否，未完成 18k packing |
| Phase 1 packed | 中断，仅有不可用临时文件 | 否 |
| Phase 2 packed | 缺失 | 否 |
| Phase 3 packed | 缺失 | 否 |

因此下一步应先完成 packing 和验证，再新增隔离的 UniST-198 runner/config。不得把当前 Phase 1 临时文件改名为正式文件，也不得直接修改已有 UniST-13 训练脚本。

## 2. “完整 UniST”与“论文完整 UniSS”的边界

本方案中的“完整数据”准确含义是：

- 本地 `cmots/UniST` 合并后的全部 198 个 train shards；
- 共 19,785,924 条 train rows；
- 使用当前 UniSS tokenizer、prompt layout、BiCodec/GLM token 和 Megatron-LM 训练适配层。

它不等价于论文严格的完整 UniSS 训练语料：

1. 论文 Phase 1 包含约 77.1k hours speech data 和 WMT17 2.3B MT tokens。
2. 当前 Phase 1 使用 UniST 的 ASR/S2TT/TTS 数据，并把同一条记录的 transcription→translation 作为 MT proxy。
3. 当前 Phase 3 代码只是在所有 UniST rows 上去掉 `direct_s2st`，保留 `quality` 和 `performance`；没有额外实现论文所称的 high-quality-only 样本筛选。
4. 因此本次实验应命名为 **UniST-198 full-data reproduction/validation**，不能标记为论文完整语料严格复现。

如果后续要声称严格 Phase 3 high-quality-only，需要先定义、评审并冻结额外筛选规则，然后生成另一个独立数据版本，不能覆盖本文的 `phase3_unist198`。

## 3. 当前数据盘点

### 3.1 原始数据

目录：

```text
data/raw/UniST/
```

当前信息：

- 总大小约 29 GB；
- 208 个 Parquet 文件；
- train：198 shards，19,785,924 rows；
- dev：7,965 rows；
- test：23,369 rows；
- 还包括 `clean_dev/clean_test/dev_clean/dev_other/other_dev/other_test/test_clean/test_other` 等辅助 split；
- `merge_summary.json` 记录总计 19,826,439 rows。

### 3.2 已完成的分阶段 JSONL

```text
data/processed/phase1_unist198_sharded/
data/processed/phase2_unist198_sharded/
data/processed/phase3_unist198_sharded/
```

检查结果：

| 阶段 | shard 数 | 大小 | samples |
| --- | ---: | ---: | ---: |
| Phase 1 | 198 | 115 GB | 79,143,696 |
| Phase 2 | 198 | 220 GB | 59,357,772 |
| Phase 3 | 198 | 151 GB | 39,571,848 |

任务构成：

```text
Phase 1，每条 raw row 生成：
  asr + s2tt + tts + mt_proxy = 4 samples

Phase 2，每条 raw row 生成：
  quality + performance + direct_s2st = 3 samples

Phase 3，每条 raw row 生成：
  quality + performance = 2 samples
```

已对三个目录中的全部 594 个 shard 做轻量完整性检查：

- 文件非空；
- 首条 JSON 可解析；
- 末条 JSON 可解析；
- 文件以完整换行结束。

594/594 均通过。

### 3.3 Phase 2 mix

正式文件：

```text
data/processed/phase2_unist198_mix/phase2_mix_2to1.jsonl
```

状态：

- 大小：279,434,498,149 bytes；
- 总 samples：89,036,658；
- Phase 2 UniST samples：59,357,772；
- Phase 1 replay samples：29,678,886；
- 比例严格为 2:1；
- 首尾 JSON 检查通过。

### 3.4 packing 中断状态

完整数据目标文件应该是：

```text
data/megatron/phase1_unist198/packed_train.jsonl
data/megatron/phase2_unist198_mix/packed_train.jsonl
data/megatron/phase3_unist198/packed_train.jsonl
```

当前三个正式文件都不存在。

遗留临时文件：

```text
data/megatron/phase1_unist198/packed_train.jsonl.tmp.1176053
```

该文件：

- 大小约 82.5 GB；
- 包含 237,778 行；
- 缺少结尾换行；
- 上次流程在 Phase 1 packing 中被服务器迁移打断；
- 不支持安全续写；
- 不得改名为 `packed_train.jsonl`。

`pack_sequences.py` 当前是顺序、非断点续跑 packer。要保证 sample boundary 和最后一个未满 buffer 正确，Phase 1 packing 必须从头重新执行。

## 4. 历史训练脚本与完成度

### 4.1 已存在的训练脚本

1.5B backbone：

```text
scripts/train_phase1.sh
scripts/train_phase2.sh
scripts/train_phase3.sh
```

0.5B fast-validation backbone：

```text
scripts/train_phase1_qwen0p5b.sh
scripts/train_phase2_qwen0p5b.sh
scripts/train_phase3_qwen0p5b.sh
scripts/run_qwen0p5b_unist13_all_phases.sh
```

这些脚本已经支持通过环境变量覆盖：

```text
TRAIN_DATA
VALID_DATA
LOAD_CHECKPOINT
SAVE_DIR
TRAIN_ITERS
LR_WARMUP_ITERS
SAVE_INTERVAL
EVAL_INTERVAL
EVAL_ITERS
LOG_INTERVAL
MASTER_PORT
CUDA_VISIBLE_DEVICES
NPROC_PER_NODE
TP
PP
MICRO_BATCH_SIZE
```

因此完整数据实验不需要修改这些脚本本体。正确做法是新增一个 UniST-198 wrapper/config，只向原脚本传入新路径和由完整 packed count 推导出的 schedule。

### 4.2 历史实际完成度

| 实验 | 目标 | 实际 | 状态 |
| --- | ---: | ---: | --- |
| Qwen0.5B Phase 1 UniST-13 | 1269 | 1269 | 完成，最终 checkpoint 已保存 |
| Qwen0.5B Phase 2 UniST-13 | 1045 | 1045 | 完成，最终 checkpoint 已保存 |
| Qwen0.5B Phase 3 UniST-13 | 4341 | 最后日志 iteration 4180，checkpoint 4100 | 被外部 signal 中断，未完成 |
| Qwen1.5B Phase 1 UniST-13 | 1269 | checkpoint 1200 | 手动中断，未完成 |

最可靠、可直接作为完整数据基线的配置是 Qwen0.5B Phase 1/2。Qwen0.5B Phase 3 已稳定运行到 4180，但不能称为完成实验。

## 5. 不破坏旧实验的强制隔离规则

以下目录和文件必须保持不变：

```text
data/megatron/phase1_unist13/
data/megatron/phase2_unist13_mix/
data/megatron/validation_unist_dev/phase1_valid_packed.jsonl
data/megatron/validation_unist_dev/phase2_valid_packed.jsonl

checkpoints/uniss_qwen0p5b_phase1_unist13_full/
checkpoints/uniss_qwen0p5b_phase2_unist13_full/
checkpoints/uniss_qwen0p5b_phase3_unist13_full/

logs/uniss_qwen0p5b_phase1_unist13_full.log
logs/uniss_qwen0p5b_phase2_unist13_full.log
logs/uniss_qwen0p5b_phase3_unist13_full.log

scripts/train_phase1_qwen0p5b.sh
scripts/train_phase2_qwen0p5b.sh
scripts/train_phase3_qwen0p5b.sh
scripts/run_qwen0p5b_unist13_all_phases.sh
```

完整数据新实验统一使用新名字：

```text
data/megatron/phase1_unist198/
data/megatron/phase2_unist198_mix/
data/megatron/phase3_unist198/
data/megatron/validation_unist_dev/phase3_valid_packed.jsonl

checkpoints/uniss_qwen0p5b_phase1_unist198_full_v1/
checkpoints/uniss_qwen0p5b_phase2_unist198_full_v1/
checkpoints/uniss_qwen0p5b_phase3_unist198_full_v1/

logs/uniss_qwen0p5b_phase1_unist198_full_v1.log
logs/uniss_qwen0p5b_phase2_unist198_full_v1.log
logs/uniss_qwen0p5b_phase3_unist198_full_v1.log

runs/uniss_qwen0p5b_unist198_full_v1/
```

实施时只允许新增：

```text
scripts/run_qwen0p5b_unist198_all_phases.sh
configs/experiments/uniss_qwen0p5b_unist198_full_v1.env
```

不得修改旧 runner 默认值，以保证旧 UniST-13 命令和测试仍可复现。

## 6. 第一步：完成三个 packed 文件

### 6.1 packing 固定参数

三个阶段都保持：

```text
seq_length = 18000
drop_overlong = true
output = JSONL
```

对应工具：

```text
training/pack_sequences.py
```

### 6.2 中断文件处理原则

在重新 packing 前，把中断文件保留为审计文件，不直接删除：

```text
data/megatron/phase1_unist198/packed_train.interrupted_20260718.jsonl
```

新的 packing 必须写入新的临时文件，例如：

```text
packed_train.jsonl.tmp.<new-pid>
```

只有满足全部检查后才原子改名为：

```text
packed_train.jsonl
```

### 6.3 packing 顺序

顺序执行，避免三个超大 JSON 写任务同时争抢磁盘：

1. Phase 1：198 个 `phase1_unist198_sharded/train-*.jsonl`；
2. Phase 2：单个 `phase2_mix_2to1.jsonl`；
3. Phase 3：198 个 `phase3_unist198_sharded/train-*.jsonl`。

每阶段成功后再开始下一阶段。

### 6.4 每个 packed 文件的完成检查

必须同时满足：

1. packer 正常退出，记录 `packed_sequences`；
2. 正式文件存在且临时文件不再增长；
3. `wc -l` 与 packer 输出一致；
4. 第一行和最后一行 JSON 可解析；
5. 文件以换行结束；
6. 第一条和最后一条记录的以下字段长度均为 18000：
   - `tokens`
   - `labels`
   - `loss_mask`
   - `position_ids`
7. `sample_boundaries` 单调且不越界；
8. 用 `MegatronUniSSDataset` 读取首、中、尾记录成功；
9. 用新数据运行 2-GPU/4-GPU 10-step smoke 成功；
10. 生成该阶段的 count、size、mtime 和任务分布 summary。

packing 全部完成后才创建新的完成标记，例如：

```text
runs/unist198_preprocess/PACKING_COMPLETE_V1
```

原来的 `PIPELINE_COMPLETE` 不应伪造，因为此前流程确实被中断。

### 6.5 空间预算

按 UniST-13 packed 文件线性估算：

- Phase 1：约 0.28–0.32 TB；
- Phase 2：约 0.70–0.80 TB；
- Phase 3：需等待实际任务 token length，建议预留 0.4–0.8 TB；
- 加上临时文件和中断文件，packing 阶段建议至少保留 2 TB 可用空间。

当前磁盘剩余约 24 TB，空间足够。

## 7. Validation 数据

现有 validation：

```text
Phase 1:
data/megatron/validation_unist_dev/phase1_valid_packed.jsonl
packed records = 241

Phase 2:
data/megatron/validation_unist_dev/phase2_valid_packed.jsonl
packed records = 571
```

它们已经被历史 Phase 1/2 成功训练使用，应保持不变。

Phase 3 应新增独立 validation：

```text
data/processed/validation_unist_dev/phase3_dev.jsonl
data/megatron/validation_unist_dev/phase3_valid_packed.jsonl
```

生成原则：

- 输入仍使用同一个 `dev-00000.parquet`；
- `--phase phase3`；
- 只包含 `quality/performance`；
- 不混入 Phase 1 replay；
- 不覆盖 Phase 2 validation；
- 完成后做与 train packed 相同的首尾和字段长度验证。

预期未 packed samples：

```text
7,965 dev rows * 2 tasks = 15,930 samples
```

## 8. 完整数据训练配置

### 8.1 固定不变的模型/优化配置

以历史实际跑通的 Qwen2.5-0.5B 配置为基线，以下参数保持不变：

```text
backbone initialization:
  checkpoints/qwen2_0p5b_uniss_vocab

vocab_size: 180407
num_layers: 24
hidden_size: 896
ffn_hidden_size: 4864
num_attention_heads: 14
num_query_groups: 2
group_query_attention: true
normalization: RMSNorm
swiglu: true
disable_bias_linear: true
add_qkv_bias: true
position_embedding_type: rope
rotary_base: 1000000
seq_length: 18000
max_position_embeddings: 32768
micro_batch_size: 1
global_batch_size: 128
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
bf16: true
flash_attention: true
recompute_activations: true
weight_decay: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
```

阶段切换保持：

```text
FINETUNE=1
LOAD_OPTIM=0
LOAD_RNG=0
```

即每个新 phase 只加载上一阶段模型权重，重置 optimizer、RNG 和 iteration。

同一 phase 的断点续训才允许：

```text
FINETUNE=0
LOAD_OPTIM=1
LOAD_RNG=1
LOAD_CHECKPOINT=<同一 phase 的 SAVE_DIR>
```

### 8.2 Phase 1 schedule

Phase 1 保持历史规则：3 epochs，第一整个 epoch warmup，之后恒定 LR `8e-4`。

packed 完成后计算：

```text
P1 = wc -l(data/megatron/phase1_unist198/packed_train.jsonl)
P1_EPOCH_ITERS = ceil(P1 / 128)
P1_TRAIN_ITERS = 3 * P1_EPOCH_ITERS
P1_WARMUP_ITERS = P1_EPOCH_ITERS
```

按 UniST-13 比例粗略估计：

```text
P1 ≈ 824,069 packed records
P1_EPOCH_ITERS ≈ 6,439
P1_TRAIN_ITERS ≈ 19,317
P1_WARMUP_ITERS ≈ 6,439
```

最终值必须以正式 packed 文件行数为准，不能直接硬编码估算值。

### 8.3 Phase 2 schedule

Phase 2 保持历史规则：1 epoch，5% epoch warmup，恒定 LR `2e-4`。

```text
P2 = wc -l(data/megatron/phase2_unist198_mix/packed_train.jsonl)
P2_EPOCH_ITERS = ceil(P2 / 128)
P2_TRAIN_ITERS = P2_EPOCH_ITERS
P2_WARMUP_ITERS = ceil(0.05 * P2_EPOCH_ITERS)
```

按 UniST-13 比例粗略估计：

```text
P2 ≈ 2,034,617 packed records
P2_TRAIN_ITERS ≈ 15,896
P2_WARMUP_ITERS ≈ 795
```

最终值同样必须按正式 packed 文件计算。

### 8.4 Phase 3 schedule

Phase 3 保持：

```text
lr = 5e-5
min_lr = 5e-6
lr_decay_style = cosine
warmup_iters = 0
```

但训练步数需要明确实验目标。

#### 推荐：UniST-198 全数据 1 epoch

用户目标是使用完整 UniST 数据，因此推荐：

```text
P3 = wc -l(data/megatron/phase3_unist198/packed_train.jsonl)
P3_TRAIN_ITERS = ceil(P3 / 128)
P3_90_PERCENT = floor(0.9 * P3_TRAIN_ITERS)
```

保留：

- 约 0.9 epoch checkpoint；
- 1.0 epoch final checkpoint。

这与论文“Phase 3 high-quality data 1 epoch”的 schedule shape 一致，但数据选择并不严格等价。

#### 可选：论文 10B token budget 对照

论文 Phase 3 约 10B tokens：

```text
10,000,000,000 / (128 * 18,000) ≈ 4,341 steps
```

可额外做一个独立对照实验，固定 `TRAIN_ITERS=4341`，但必须使用不同 checkpoint/log 名称：

```text
checkpoints/uniss_qwen0p5b_phase3_unist198_10b_v1/
logs/uniss_qwen0p5b_phase3_unist198_10b_v1.log
```

不能把“完整数据1 epoch”和“10B token budget”混为同一实验。

## 9. 新 runner/config 设计

实施时新增以下独立 tracked 文件，不修改旧训练脚本：

```text
configs/experiments/uniss_qwen0p5b_unist198_full_v1.env
scripts/pack_unist198_full.sh
scripts/run_qwen0p5b_unist198_all_phases.sh
scripts/run_unist198_full_pipeline.sh
scripts/start_unist198_tensorboard.sh
training/validate_packed_jsonl.py
training/tests/test_unist198_full_scripts.py
```

### 9.1 config 内容

config 只记录实验变量，不复制训练实现：

```text
ENV_ROOT=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train

PHASE1_TRAIN=data/megatron/phase1_unist198/packed_train.jsonl
PHASE1_VALID=data/megatron/validation_unist_dev/phase1_valid_packed.jsonl

PHASE2_TRAIN=data/megatron/phase2_unist198_mix/packed_train.jsonl
PHASE2_VALID=data/megatron/validation_unist_dev/phase2_valid_packed.jsonl

PHASE3_TRAIN=data/megatron/phase3_unist198/packed_train.jsonl
PHASE3_VALID=data/megatron/validation_unist_dev/phase3_valid_packed.jsonl

BASE_CHECKPOINT=checkpoints/qwen2_0p5b_uniss_vocab
PHASE1_SAVE=checkpoints/uniss_qwen0p5b_phase1_unist198_full_v1
PHASE2_SAVE=checkpoints/uniss_qwen0p5b_phase2_unist198_full_v1
PHASE3_SAVE=checkpoints/uniss_qwen0p5b_phase3_unist198_full_v1
```

`TRAIN_ITERS` 和 warmup 不应手写估算值。runner 在启动前读取正式 packed 行数并计算，同时把最终值写入 run manifest。

### 9.2 runner 职责

新 runner 只负责：

1. source 已修复的 Conda activation；
2. 校验 packed completion marker；
3. 校验所有 train/valid 文件；
4. 计算 Phase 1/2/3 iteration；
5. 输出 `--dry-run` 完整命令；
6. 写入 immutable run manifest；
7. 调用现有 `train_phase*_qwen0p5b.sh`；
8. Phase 1 成功后才允许 Phase 2；
9. Phase 2 成功后才允许 Phase 3；
10. 每阶段使用独立 log、port、save dir。

它不应复制或修改 `pretrain_uniss_megatron.py` 的训练逻辑。

### 9.3 run manifest

每次启动前保存：

```text
runs/uniss_qwen0p5b_unist198_full_v1/manifest.txt
```

至少包括：

- 时间；
- 主仓库 commit `0c7b743` 或启动时实际 commit；
- Megatron-LM commit；
- Megatron-Bridge commit；
- Python/Torch/Transformers/FlashAttention 版本；
- GPU/driver/CUDA/NCCL 信息；
- packed 文件绝对路径、字节数、行数；
- Phase 1/2/3 train/warmup/save/eval iterations；
- base/load/save checkpoint；
- `git status --short`；
- 实际 dry-run command。

## 10. GPU 与并行配置

历史成功的 Qwen0.5B 配置是：

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
NPROC_PER_NODE=4
TP=1
PP=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
```

新服务器有 8 张 H200。为了只改变数据，第一版正式基线建议仍使用4卡：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4
TP=1
PP=1
MICRO_BATCH_SIZE=1
```

先保持与历史成功实验相同的 data-parallel world size，减少变量。

8卡优化应作为后续独立实验：

- global batch 仍保持128；
- 先跑50-step loss/throughput parity smoke；
- 确认无NaN、sample consumption和4卡一致后再使用；
- 使用不同 run ID，不覆盖4卡基线。

## 11. 执行顺序

### Gate A：packing

1. 保留并标记中断 Phase 1 临时文件；
2. 从头完成 Phase 1 packing；
3. 完成 Phase 2 mix packing；
4. 完成 Phase 3 packing；
5. 生成 Phase 3 validation；
6. 创建 packing summary 和 completion marker。

### Gate B：数据 smoke

对三个正式 packed 文件分别：

1. dataset 首/中/尾读取；
2. DataLoader batch shape；
3. 2-GPU NCCL 读数据；
4. 各 phase 10-step smoke；
5. checkpoint 保存和 reload；
6. validation loss 为有限值。

smoke 使用独立目录：

```text
checkpoints/smoke/uniss_qwen0p5b_phase1_unist198_v1/
checkpoints/smoke/uniss_qwen0p5b_phase2_unist198_v1/
checkpoints/smoke/uniss_qwen0p5b_phase3_unist198_v1/
```

### Gate C：Phase 1 full

- load：`qwen2_0p5b_uniss_vocab`；
- train：完整 Phase 1 packed；
- 3 epochs；
- 1 epoch warmup；
- final tracker 必须等于计算出的 train iters；
- final validation 成功；
- checkpoint reload 成功。

### Gate D：Phase 2 full

- load：本次 UniST-198 Phase 1 final；
- 禁止加载 UniST-13 Phase 1；
- train：完整 Phase 2 2:1 mix packed；
- 1 epoch；
- 5% warmup；
- final validation/checkpoint reload 成功。

### Gate E：Phase 3 full

- load：本次 UniST-198 Phase 2 final；
- train：独立 Phase 3 `quality/performance` packed；
- 默认1 epoch；
- 记录0.9 epoch和1.0 epoch checkpoint；
- 使用独立 Phase 3 validation；
- 导出并运行固定 S2ST audio evaluation。

## 12. save/eval 策略

为了最大程度复用历史成功配置，第一版保持：

```text
SAVE_INTERVAL=100
EVAL_INTERVAL=100
EVAL_ITERS=10
LOG_INTERVAL=10
```

如果 checkpoint 数量过多，可在不改变优化过程的前提下把完整数据正式实验调整为500，但必须在 manifest 中明确记录，且不能修改旧脚本默认值。

Phase 3 必须额外保留最接近：

```text
floor(0.9 * P3_TRAIN_ITERS)
P3_TRAIN_ITERS
```

的两个 checkpoint。

## 13. 预计耗时

历史4卡 Qwen0.5B 每 iteration 约7.4–7.7秒。

按当前粗略 packed 数估计：

```text
Phase 1: 约19.3k steps，约40小时，加eval/save开销
Phase 2: 约15.9k steps，约33小时，加eval/save开销
Phase 3: 取决于正式P3行数；若4341 steps约9小时
```

4卡基线总计约3–4天。H200实际速度可能更快，应以50-step smoke实测吞吐重新估算。

packing 是大规模单进程 JSON 解析/写入任务，也可能需要数小时到一天以上，必须在 tmux 中运行并持续记录日志。

## 14. 完成判定

### 数据完成

- 三个正式 packed 文件存在；
- 无未解释的活跃 temp 文件；
- 行数、字段长度、首尾 JSON、dataset 读取全部通过；
- Phase 3 valid packed 存在；
- packing summary 和 marker 存在。

### 每个 phase 训练完成

- final iteration 等于 manifest 目标；
- `latest_checkpointed_iteration.txt` 指向 final；
- log 中有 `[after training is done]`；
- final checkpoint successfully saved；
- 无持续NaN、Inf或skipped iterations；
- validation loss为有限值；
- final checkpoint可reload；
- Phase 2/3可导出HF并生成非空semantic tokens/audio。

### 复现保护

- 主仓库旧脚本未修改；
- UniST-13数据、checkpoint、log、eval output未覆盖；
- 新实验所有路径包含`unist198`和版本号；
- 主仓库单元测试仍全部通过；
- `git diff`只包含明确新增的UniST-198 runner/config/文档。

## 15. 执行顺序

正式任务按以下顺序自动执行：

1. 从头完成并验证三个packed文件；
2. 新增Phase 3 validation；
3. 根据正式packed行数计算schedule；
4. 新增隔离runner/config；
5. 使用8卡完成micro batch smoke；
6. smoke通过后依次执行Phase 1→Phase 2→Phase 3 full。

该顺序能最大限度复用历史已经验证的训练实现，同时保证旧实验、旧checkpoint和旧评测结果完全不受影响。

## 16. 2026-07-21实施验证记录

已经完成：

- 新增脚本和配置全部通过`bash -n`；
- 完整`training/tests`共83项通过，包括小型端到端原子packing fixture；
- 旧Qwen0.5B、Qwen1.5B、UniST-13训练与评测脚本测试保持通过；
- 8张H200上完成Phase 1两步真实smoke；
- `MICRO_BATCH_SIZE=2`、`GLOBAL_BATCH_SIZE=128`、DP=8正常；
- 两步均无OOM、NaN、Inf或skipped iteration；
- 单卡实测峰值allocated约83GB，H200显存余量足够；
- iteration 2分布式checkpoint保存成功且tracker为2；
- TensorBoard已记录loss、learning rate、memory、world size和throughput。

恢复环境中Triton随包携带的`ptxas/cuobjdump/nvdisasm`最初缺少执行位，已在以下环境内恢复为可执行，不涉及仓库或旧实验文件：

```text
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/lib/python3.12/site-packages/triton/backends/nvidia/bin/
```

smoke确认后，正式默认micro batch固定为2。由于global batch 128和8卡数据并行要求micro batch整除每卡16个样本，下一档有效值是4；当前不采用4，以保留长序列训练的显存安全余量。
