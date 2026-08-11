# UniSS True-Subsecond：15-Shard 单个完整 Epoch 训练验证计划

> 实现目录：`experiments/uniss_true_subsecond_pilot15_epoch1_v2/`
>
> 2026-08-11实施状态：旧full198任务已安全停在cache `108/198`、pack
> `106/198`并保留断点；固定shard `0..14`的v2修复代码、数据硬门、严格全局
> shuffle epoch构建器、8卡cache/训练自动流水线和TensorBoard 6071脚本已实现。
> 修复数据需要GPU重新生成，因为精确800ms、`+160/+320ms`与完整上下文teacher
> 分布不能从旧sidecar可靠补造。实测batch 128达到约102GiB、100% utility、
> 约584W和21.90 rows/s；batch 160稳定达到22.77 rows/s。因此正式生成从160
> 开始，OOM时只对未完成shard自动回退到144/128/96/64；teacher请求按prefix
> time去重并以512请求子批推理。

> 方案对象：`experiments/uniss_phase3_true_subsecond_deadline_full198_v1/`
>
> 目标：在停止当前 full198 GPU cache worker、释放 8×H200 后，使用与正式训练相同的 Megatron 主训练入口和 `seq-length=18000` 几何，在 15 个有代表性的 UniST shard 上完成一个独立、可恢复、可审计的 packed epoch，验证 READ/WRITE timing、safe commit、deadline 和 Phase3 replay 是否真正能够共同学习。
>
> 本计划是 pilot gate，不替代 full198 正式实验；所有数据、checkpoint、TensorBoard、日志和报告必须写入新目录，禁止覆盖当前 full198、历史 Phase1/2/3 或已有 `pilot15_native_50step_v2`。

---

## 1. 结论与数据选择原则

### 1.1 不应只选择 natural WRITE 比例高的 shard

主 pilot 如果只选择 natural WRITE 比例高的 shard，会产生三种偏差：

1. **容易样本偏差**：teacher 本来就容易在前缀上预测稳定，无法验证低支持率样本会不会造成 always-WAIT；
2. **语言方向偏差**：当前 shard 的 natural WRITE 分布与 `eng->cmn` / `cmn->eng`、语速和领域相关，只挑高值可能几乎变成单一方向；
3. **错误乐观结论**：模型在高-WRITE数据上可以得到好看的 recall，但放到 full198 后仍可能因 READ 占优而塌缩。

高-WRITE shard 只适合作为额外的“能否快速过拟合”诊断，不适合作为主实验。

### 1.2 推荐固定使用 shard 0–14

当前已经完成并打包的前15个shard具有以下真实统计：

| 项目 | 数值 |
|---|---:|
| shard | `0..14` |
| accepted raw rows | 1,500,000 |
| trajectory points | 3,000,000 |
| packed trajectory records | 13,686 |
| `eng->cmn` | 816,659，54.44% |
| `cmn->eng` | 683,341，45.56% |
| aggregate natural WRITE | 21.28% |
| shard natural WRITE范围 | 15.37%–25.58% |
| aggregate deadline-forced | 46.67% |

优点：

- 双向语言接近均衡；
- 同时包含低、中、高 natural WRITE shard；
- 已存在不可变trajectory pack和Phase3 replay pack；
- 与此前50-step smoke使用相同raw范围，方便定位“代码修改”而不是“换数据”带来的差别。

因此主 pilot 固定使用 `0..14`，不要事后根据结果重新挑 shard。

---

## 2. 当前v1数据不能直接放行一个epoch

已有15-shard 50-step运行已经证明Megatron链路、Phase3 handoff、反向传播、checkpoint和严格resume都可以工作，但算法指标没有通过：

```text
iteration 1 : predicted WRITE=48.73%, natural WRITE=13.60%, safe F1=0.2121
iteration 10: predicted WRITE= 6.43%, natural WRITE=12.49%, safe F1=0.00067
iteration 20: predicted WRITE= 0.031%,natural WRITE=16.39%, safe F1=0.1128
iteration 50: predicted WRITE= 0.46%, natural WRITE= 8.35%, safe F1=0.00209
```

这属于工程smoke通过、策略门失败。当前数据审计还发现：

1. 每条语音只有一个early和一个middle/late snapshot，并非稠密多事件trajectory；
2. 部分样本在640ms标签为READ，却没有800ms真实观察点，deadline survival只能在640ms上施加“800ms前必须WRITE”的相反压力；
3. 当前完成shard的deadline-forced比例约47%，自然安全内容不足；
4. 两次自然WRITE都发semantic的样本中，只有约4.6%的semantic区间严格连续，多数存在gap；
5. packed sample是单动作snapshot，尚未真正包含同一session的历史text/semantic事件。

所以一个epoch训练必须以修复后的独立数据版本为输入，例如：

```text
data/processed/uniss_true_subsecond_pilot15_epoch1_v2/
data/megatron/uniss_true_subsecond_pilot15_epoch1_v2/
```

旧目录必须保留用于复现：

```text
data/megatron/uniss_true_subsecond_pilot15_v1/
checkpoints/uniss_true_subsecond_pilot15_native_50step_v2/
runs/uniss_true_subsecond_pilot15_native_50step_v2/
```

---

## 3. 训练前必须通过的数据硬门

以下任一项失败都不允许启动一个epoch训练。

### 3.1 时间轴和deadline门

每个训练session必须满足二选一：

1. 显式包含 `320/480/640/800ms` 中计划要求的tick，且必须有精确800ms observation；或
2. 若没有精确deadline tick，则该session不能计算grouped deadline survival，禁止把640ms READ同时当成800ms WRITE监督。

要求：

```text
deadline label/action CE conflict count = 0
future leakage count                  = 0
chunk_end_ms monotonic violation      = 0
hard deadline accounting coverage     = 100%
```

### 3.2 semantic连续性门

为每个session维护独立的：

```text
previous_semantic_end
```

下一次真实semantic WRITE必须满足：

```text
semantic_target_start == previous_semantic_end
semantic_target_end   > semantic_target_start
```

禁止继续用“目标文本进度比例”重新计算start并跳过尚未播放的semantic token。

要求：

```text
natural WRITE semantic continuity >= 99.9%
semantic gap rate                    <= 0.1%
semantic overlap rate                <= 0.1%
speaker_global length                = 32 for 100% sessions
```

### 3.3 action/safe标签门

建议对0.60/0.65/0.70三个confidence threshold生成审计报告，再冻结正式值。不能为了提高WRITE比例只看覆盖率，还要看未来稳定性和premature commit。

最低要求：

```text
natural WRITE overall       : 15%–35%
natural WRITE per direction : >= 10%
deadline-forced overall     : 建议 <= 35%
safe positive fraction      : 每个方向均非0
forced WRITE hard text CE   : 0
forced WRITE hard semantic CE:0
```

如果0.70导致forced仍接近47%，优先修正teacher前缀质量、时间采样或使用0.65并重新验证premature WRITE，不应直接用action class weight掩盖坏标签。

### 3.4 session序列门

每个session的训练表示至少需要满足：

```text
READ/WRITE事件按时间排序
previous committed text可见
previous emitted semantic可见或由持久state提供
新WRITE只监督delta
已提交内容不会回滚
position/KV语义与在线推理一致
```

如果继续使用snapshot训练，必须明确把它定义成“单步policy监督”，不能声称已经训练了append-only multi-event KV trajectory。

---

## 4. “一个epoch”的精确定义

### 4.1 不采用现有full replay schedule的原因

现有15-shard数据有：

```text
trajectory packed records = 13,686
Phase3 replay records      = 91,852
```

若把全部91,852条replay都消费一次，同时保持curriculum约38% replay / 62% trajectory，当前`JointPackedEpochGeometry`会生成：

```text
train_iters          = 1,873
replay scheduled     = 91,856
trajectory scheduled = 147,888
```

也就是13,686条trajectory平均被重复约10.8次。这不是“每条pilot数据只训练一遍”，会放大15-shard过拟合和标签噪声。

### 4.2 推荐定义：一个trajectory coverage epoch

主pilot将一个epoch定义为：

> 每条修复后的15-shard trajectory packed record恰好被消费一次；按同一curriculum比例，从相同15-shard Phase3 replay pack中确定性、均匀地选取所需replay记录；除最后DP对齐padding外不重复trajectory。

对于当前13,686条trajectory，使用DP microbatch：

```text
8 GPU × micro-batch 2 = 16 packed records/group
global batch 128      = 8 groups/optimizer step
```

当前两trajectory/row几何下，最小完整schedule为：

```text
trajectory scheduled = 13,696  # 13,686 + 10 DP padding
replay subset         = 8,448
total schedule        = 22,144 packed records
TRAIN_ITERS           = 22,144 / 128 = 173
```

如果修复后增加了tick或改变packing，禁止硬编码173。重新读取最终trajectory count `T`，求最小的8倍数group count `G`，使：

```text
curriculum_trajectory_groups(G) × 16 >= T
TRAIN_ITERS = G / 8
REPLAY_SUBSET = curriculum_replay_groups(G) × 16
```

训练manifest必须记录最终：

```text
trajectory source count
trajectory scheduled count
trajectory padding
replay source count
replay selected count
schedule count
train iters
curriculum boundaries
```

---

## 5. 单epoch curriculum

继续使用当前`training/curriculum.py`的四段比例，但按最终epoch进度计算，不按full198固定iteration。

若最终为173 iterations，边界为：

| 区间 | Iteration | Replay/Trajectory | Deadline权重 | Frontend LR multiplier | 目的 |
|---|---:|---:|---:|---:|---|
| C0 | 1–15 | 45/55 | 0→0.10 | 0.25→1.0 | 新head和格式稳定，不立刻强迫WRITE |
| C1 | 16–58 | 40/60 | 0.10→0.30 | 1.0 | 学prefix、support和safe commit |
| C2 | 59–130 | 35/65 | 0.30 | 1.0 | 主deadline/micro-WRITE区间 |
| C3 | 131–173 | 40/60 | 0.30 | 0.5 | 恢复Phase3质量、降低前端漂移 |

所有loss从iteration 0就存在；curriculum只改变数据编排、deadline coefficient和frontend LR，不重置optimizer，不替换loss。

### 5.1 Pilot warmup

当前正式代码默认：

```text
warmup = max(200, ceil(2.5% * TRAIN_ITERS))
```

这对173-step pilot不适用，因为会导致整个epoch都处于warmup。pilot单独覆盖为：

```text
RUN_WARMUP_ITERS=20
```

full198正式实验仍保留原来的最少200步规则。

---

## 6. 固定训练超参数

除pilot专用warmup、保存和验证频率外，训练几何与正式Phase3/当前true-subsecond脚本保持一致：

```text
GPU                 = 8
TP / PP             = 1 / 1
micro batch         = 2
global batch        = 128
sequence length     = 18000
BF16                = true
Flash Attention     = true
activation recompute= true
dataloader          = cyclic
no-data-sharding    = true
global shuffle seed = 20260810

Qwen LoRA LR        = 1.0e-5
frontend LR         = 5.0e-6
new heads LR        = 5.0e-5
minimum LR          = 1.0e-6
weight decay        = 0.1
Adam betas          = 0.9 / 0.95
clip grad           = 0.5
warmup              = 20 pilot steps
decay               = cosine over exactly one pilot epoch
```

当前50-step collapse首先是数据/标签/类别不平衡问题，不建议在修复数据前靠降低学习率解决。

---

## 7. 独立目录

建议使用：

```text
experiment code/config:
experiments/uniss_phase3_true_subsecond_deadline_full198_v1/

pilot repaired data:
data/processed/uniss_true_subsecond_pilot15_epoch1_v2/
data/megatron/uniss_true_subsecond_pilot15_epoch1_v2/

checkpoint:
checkpoints/uniss_true_subsecond_pilot15_epoch1_v2/

TensorBoard:
runs/uniss_true_subsecond_pilot15_epoch1_v2/tensorboard/

log:
logs/uniss_true_subsecond_pilot15_epoch1_v2.log

report:
reports/uniss_true_subsecond_pilot15_epoch1_v2/
```

禁止使用full198正式run名或已有pilot15 v1/v2目录。

---

## 8. GPU释放与自动流水线隔离

用户决定切换到pilot时，先停止自动full198 pipeline，避免数据完成后自动抢占GPU启动正式训练：

```bash
tmux kill-session -t uniss_true_subsecond_full198_pipeline
tmux kill-session -t uniss_true_subsecond_cache_full198
```

CPU packing进程不使用GPU，可选择保留。不要使用无范围的`pkill python`或kill其他用户进程。

确认8卡释放：

```bash
nvidia-smi
ps -eo pid,args | rg 'build_trajectory_cache|pretrain_true_subsecond_megatron'
```

完成marker使用原子写入，已经完成的shard可以保留；停止时正在生成的临时part不能作为正式输入，重新启动cache时应由无marker状态恢复/重建。

---

## 9. 数据构建和offset设计

### 9.1 trajectory

修复后的shard 0–14重新pack到新目录并组装为一个不可变JSONL和uint64 offsets：

```text
packed_trajectory.jsonl
packed_trajectory.jsonl.count
packed_trajectory.offsets.u64
packed_trajectory.offsets.u64.json
ASSEMBLY_COMPLETE.json
```

### 9.2 replay

Phase3 replay本身不依赖新timing标签，可以复用现有15-shard replay packed JSONL：

```text
data/megatron/uniss_true_subsecond_pilot15_v1/packed_replay.jsonl
data/megatron/uniss_true_subsecond_pilot15_v1/packed_replay.offsets.u64
```

但训练必须创建新的uniform replay subset offsets，不复制32GB JSONL。当前13,686 trajectory示例选择8,448条replay：

```bash
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python

"$PYTHON" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_offset_subset \
  --kind replay \
  --packed data/megatron/uniss_true_subsecond_pilot15_v1/packed_replay.jsonl \
  --source-offsets data/megatron/uniss_true_subsecond_pilot15_v1/packed_replay.offsets.u64 \
  --output-offsets data/megatron/uniss_true_subsecond_pilot15_epoch1_v2/replay_subset_8448.offsets.u64 \
  --records 8448
```

若最终trajectory count改变，先计算新的`REPLAY_SUBSET`，再生成offset，不能继续使用8448。

### 9.3 validation

训练期间使用固定、双向均衡且不与训练重叠的dev offset subset：

```text
512 trajectory packed records
512 Phase3 replay packed records
eng->cmn / cmn->eng各50%
固定sample IDs和offset checksum
```

每15或20步运行短validation；epoch结束后再运行完整dev teacher-forced validation和256条真实streaming rollout。

---

## 10. 使用当前Megatron训练入口

继续复用：

```text
experiments/uniss_phase3_true_subsecond_deadline_full198_v1/
scripts/run_megatron_training.sh
```

建议新增一个独立wrapper，例如：

```text
scripts/run_pilot15_epoch1_8gpu.sh
```

wrapper只负责计算几何和传入独立路径，不复制trainer代码。

参考启动环境：

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
source experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env

NAME=uniss_true_subsecond_pilot15_epoch1_v2
ROOT="$REPO_ROOT/data/megatron/uniss_true_subsecond_pilot15_epoch1_v2"
SAVE="$REPO_ROOT/checkpoints/$NAME"
RUN="$REPO_ROOT/runs/$NAME"
LOG="$REPO_ROOT/logs/$NAME.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RUN_NAME="$NAME" \
RUN_SAVE_DIR="$SAVE" \
RUN_TB_DIR="$RUN/tensorboard" \
RUN_LOG="$LOG" \
RUN_TRAJECTORY_PACKED="$ROOT/packed_trajectory.jsonl" \
RUN_TRAJECTORY_OFFSETS="$ROOT/packed_trajectory.offsets.u64" \
RUN_REPLAY_PACKED="$REPO_ROOT/data/megatron/uniss_true_subsecond_pilot15_v1/packed_replay.jsonl" \
RUN_REPLAY_OFFSETS="$ROOT/replay_subset.offsets.u64" \
RUN_VALID_TRAJECTORY_PACKED="$ROOT/valid/packed_trajectory.jsonl" \
RUN_VALID_TRAJECTORY_OFFSETS="$ROOT/valid/trajectory_512.offsets.u64" \
RUN_VALID_REPLAY_PACKED="$REPO_ROOT/data/megatron/validation_unist_dev/phase3_valid_packed.jsonl" \
RUN_VALID_REPLAY_OFFSETS="$ROOT/valid/replay_512.offsets.u64" \
RUN_FULL_VALIDATION=0 \
RUN_TRAIN_ITERS="$TRAIN_ITERS" \
RUN_WARMUP_ITERS=20 \
RUN_NPROC=8 RUN_MBS=2 RUN_GBS=128 RUN_SEQ_LENGTH=18000 \
RUN_SAVE_INTERVAL=15 RUN_EVAL_INTERVAL=15 RUN_LOG_INTERVAL=1 \
RUN_MASTER_PORT=29721 \
RUN_LOAD="$PHASE3_NATIVE_CHECKPOINT" \
RUN_FINETUNE=1 RUN_LOAD_OPTIM=0 RUN_LOAD_RNG=0 \
RUN_STRICTNESS=log_all RUN_SMOKE=0 \
bash experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_training.sh
```

真实wrapper需要先从offset metadata自动计算`TRAIN_ITERS`和replay subset count，并拒绝手工不一致值。

### 10.1 Resume验证

建议在第15步主动退出一次，验证Phase3 handoff后的optimizer、scheduler、RNG和全局shuffle均可恢复：

```text
Phase A: Phase3 checkpoint → iteration 15，保存后退出
Phase B: 从pilot checkpoint严格恢复 → iteration TRAIN_ITERS
```

Phase B必须设置：

```text
RUN_LOAD=$SAVE
RUN_FINETUNE=0
RUN_LOAD_OPTIM=1
RUN_LOAD_RNG=1
RUN_STRICTNESS=raise_all
```

恢复后的第16步必须继续原sample schedule，不能重新从dataset开头开始。

---

## 11. TensorBoard

建议独立端口：

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/tensorboard \
  --logdir /opt/dlami/nvme/jasonleeeli/projects/UniSS/runs/uniss_true_subsecond_pilot15_epoch1_v2/tensorboard \
  --host 0.0.0.0 \
  --port 6071
```

浏览器：

```text
http://<服务器IP>:6071
```

必须重点观察：

```text
phase3_replay
interleaved_trajectory
real_prefix_kd
support_ordinal
token_safe_commit
deadline_survival
prefix_stability
ar_semantic_microblock
boundary_continuity

support_accuracy
support_mae
safe_commit_precision
safe_commit_recall
safe_commit_f1
predicted_write_fraction
natural_write_fraction
deadline_forced_fraction
frontend_residual_rms
grad_norm
learning_rate各参数组
```

训练loss下降本身不是通过标准。尤其不能只看`deadline_survival`下降而忽略模型是否依靠forced WRITE或是否没有安全内容。

---

## 12. 分阶段训练门

### Gate A：iteration 1–15

目标：检查初始化、数据格式和梯度。

通过条件：

```text
NaN/skipped iteration = 0
所有核心loss有限
grad norm有限，无持续尖峰
frontend residual不是NaN
predicted WRITE不是恒0或恒1
两个语言方向均进入batch
```

### Gate B：iteration 16–58

目标：确认action和safe head没有再次快速塌缩。

通过条件：

```text
predicted_write_fraction >= 0.5 × natural_write_fraction
predicted_write_fraction <= 1.5 × natural_write_fraction
safe_commit F1不持续下降到接近0
support MAE总体下降
Phase3 replay validation无突然上升
```

如果到iteration 30已经连续10次log满足：

```text
predicted WRITE < 1%
且 natural WRITE > 10%
```

立即停止pilot，不继续浪费整个epoch。

### Gate C：iteration 59–130

目标：检验deadline压力下是否仍保持自然安全WRITE。

通过条件：

```text
natural WRITE recall继续提高
safe_commit precision不因deadline明显恶化
deadline-forced rollout rate下降
premature WRITE不上升
semantic continuity rollout无gap/重复块
```

### Gate D：iteration 131–173

目标：恢复offline质量并选择checkpoint。

通过条件：

```text
Phase3 replay loss/validation不比初始化恶化超过10%–15%
predicted WRITE仍与natural WRITE同量级
safe F1保持而非再次塌缩
两语言方向均通过
```

---

## 13. Epoch结束后的真实有效性评估

一个epoch完成并不自动表示方案有效。必须在固定dev上执行真实chunk rollout，而不是只做teacher-forced loss。

### 13.1 Streaming指标

```text
First natural WRITE NCA p50/p95
First natural WRITE CA p50/p95
First useful audio CA p50/p95
WRITE-to-PCM p50/p95
LAAL / AL / AP
RTF p50/p95
natural WRITE by 640ms
natural WRITE by 800ms
deadline-forced WRITE rate
premature WRITE rate
rollback rate
empty-after-WRITE rate
semantic gap/overlap rate
```

### 13.2 质量和speaker指标

```text
ASR-BLEU / BLEU / COMET（按现有可用链路）
source/target transcription consistency
AutoPCP
SLC / speaker similarity
音频可懂度和空白比例
```

### 13.3 Pilot放行full198的最低条件

建议至少满足：

```text
predicted WRITE与natural WRITE同量级，不发生always-WAIT
safe_commit F1相对50-step v2显著改善且不在后半程塌缩
deadline-forced rollout rate <= 20%–30%，并明显低于当前标签47%
premature WRITE <= 5%
rollback = 0
semantic gap/overlap <= 0.1%
First natural WRITE CA p95接近或低于800ms目标
Phase3 offline质量下降在预先允许范围内
中英双向都通过，而不是只通过一个方向
```

若延迟达标完全依靠hard scheduler，而natural WRITE仍接近0，则判定失败。

---

## 14. 预计耗时

在8×H200、MBS=2、GBS=128、seq-length=18000条件下：

```text
数据审计和offset构造：约0.5–2小时
dev subset/cache准备：约0.5–2小时
173-step单epoch训练：预计约1–3小时
完整dev rollout和报告：约1–3小时
```

若修复后trajectory count增加，训练步数和时间按最终count线性增加。不能为了维持173步而丢弃新增的deadline tick。

---

## 15. 最终执行顺序

```text
1. 停止full198自动pipeline和8个GPU cache worker
2. 保留所有已完成PART_COMPLETE/PACK_COMPLETE资产
3. 修复deadline tick、semantic cursor和session history
4. 在shard 0–14生成全新v2 trajectory/cache/pack
5. 运行数据硬门审计
6. 计算最终T、replay subset和TRAIN_ITERS
7. 构造固定双向dev subset
8. 1GPU执行1–2 step native smoke
9. 8GPU执行iteration 1–15并验证严格resume
10. 继续至一个完整trajectory coverage epoch
11. 运行完整streaming rollout和Phase3质量评估
12. 写对比报告：Phase3、旧50-step v2、新epoch v2
13. 所有门通过后才恢复full198数据制作/正式训练
```

最终原则：15-shard pilot的目的不是得到最好看的WRITE比例，而是尽早发现full198会遇到的真实失败模式。只挑高-WRITE shard会破坏这个目的。
