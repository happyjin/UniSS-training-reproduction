# Simul-UniSS v1：15-shard 端到端执行报告

## 1. 执行结论

`simul_uniss_v1` 已使用固定的 UniST 前 15 个 train shard 跑通独立的 simultaneous speech-to-speech 原型流程。GPU smoke 与 15-shard short training 均已生成完成标记，且没有修改或覆盖原 UniSS Phase1–3 的默认脚本、配置、checkpoint 与评测输出。

本次执行是工程闭环与可训练性验证，不代表论文规模训练已经收敛，也不代表已经达到最终 BLEU、语音自然度或实时延迟目标。

完成标记：

```text
runs/simul_uniss_v1/bootstrap_15shard/gpu_smoke/GPU_SMOKE_COMPLETE
runs/simul_uniss_v1/bootstrap_15shard/short_training/SHORT_TRAINING_COMPLETE
```

完成时间：

```text
GPU smoke:     2026-07-22T19:20:04Z
short training: 2026-07-22T19:27:21Z
```

## 2. 数据范围与隔离

固定输入：

```text
data/raw/UniST/train-00000.parquet
...
data/raw/UniST/train-00014.parquet
```

处理结果：

```text
records:             1,500,000
events:             16,870,993
WAIT events:        12,566,074
WRITE events:        4,304,919
packed sequences:      228,870
```

主要数据目录：

```text
data/processed/simul_uniss_v1/bootstrap_15shard/
data/megatron/simul_uniss_v1/bootstrap_15shard/
data/megatron/simul_uniss_v1/validation_dev/
```

action packed 数据已验证首尾 JSON、4096 定长字段、行数和末尾换行：

```text
data/megatron/simul_uniss_v1/bootstrap_15shard/packed_action_train.jsonl
data/megatron/simul_uniss_v1/bootstrap_15shard/ACTION_PREPARE_COMPLETE
```

## 3. 蒸馏 teacher 与 checkpoint 使用

Stage 1 的 teacher 是计划中定义的离线 GLM speech tokenizer token sequence。训练使用已缓存的 `source_glm` teacher tokens 做 CTC sequence distillation，因此不需要占用当前正在训练的 Phase2 Qwen checkpoint 运行在线 teacher forward。

本次 Qwen action/interleaved/joint 分支从以下已完成 Phase1 checkpoint 初始化：

```text
checkpoints/uniss_qwen0p5b_phase1_unist198_recovery_b1_v2
iteration: 15465
```

当前原 UniSS Phase2 训练保持独立运行，不被 Simul-UniSS 读取、写入或抢占 GPU。若未来要比较“Phase1 teacher 初始化”和“Phase2 teacher 初始化”，必须使用新的实验名、checkpoint root、TensorBoard root 和 master port，不能覆盖本报告中的结果。

## 4. 各阶段实际产物

### Stage 0：schedule、latency 与 prefix baseline

```text
runs/simul_uniss_v1/bootstrap_15shard/stage0_metrics.json
runs/simul_uniss_v1/bootstrap_15shard/validation_latency_metrics.json
runs/simul_uniss_v1/bootstrap_15shard/stage0_prefix/record_0.json
```

全量 schedule 指标：

```text
first WRITE mean:          3631.97 ms
first WRITE p50:           3200 ms
first WRITE p95:           6400 ms
proxy lag mean:             630.37 ms
final flush failure rate:     0
```

真实 GLM cumulative-prefix baseline：

```text
prefix revision rate: 0.896714
committed rollback events: 0
computation-aware RTF: 0.191–0.206
first stable token: 4220 ms
```

较高的 candidate prefix revision rate 说明离线 tokenizer 的累计重编码不适合作为最终严格 streaming 前端，也验证了训练 causal streaming student 的必要性。

### Stage 1–2：token/audio student 与 CTC heads

```text
checkpoints/simul_uniss_v1/stage1_bootstrap_student_short/best.pt
checkpoints/simul_uniss_v1/stage1_bootstrap_student_short/last.pt
checkpoints/simul_uniss_v1/stage1_audio_student_short/last.pt
```

TensorBoard 摘要：

```text
token student train total: 40.7946 -> 16.3678
token student valid total: 19.0060 -> 14.1796
token student grad norm:   57.2219 -> 13.2907
audio student total:       28.4252 -> 18.8354
audio source CTC:          28.1555 -> 15.6476
audio target CTC:          33.6911 -> 16.2118
```

### Stage 3：WAIT/WRITE action SFT

```text
checkpoints/simul_uniss_v1/stage3_action_qwen_15shard/
latest iteration: 20
```

```text
train lm loss:      16.8721 -> 0.6360
validation lm loss: 13.6559 -> 0.6069
NaN/skipped:        0/0
```

action-only objective 较简单，20-step short run 的快速下降只证明 loss mask、checkpoint load/save、validation 与 optimizer 闭环正常，不能作为最终 policy 泛化结论。

### Stage 4：phrase-level interleaved S2ST SFT

```text
checkpoints/simul_uniss_v1/stage4_interleaved_qwen_15shard/
latest iteration: 50
```

```text
train lm loss:      8.2014 -> 4.2664
validation lm loss: 6.2082 -> 3.9923
grad norm:          133.79 -> 2.57
NaN/skipped:        0/0
```

### Stage 5：BiCodec overlap replay 与 refinement

```text
checkpoints/simul_uniss_v1/stage5_bicodec_refinement_short/bicodec_streaming_refinement.pt
runs/simul_uniss_v1/bootstrap_15shard/short_training/stage5_replay/record_0_bicodec.wav
runs/simul_uniss_v1/bootstrap_15shard/short_training/stage5_replay/record_0_bicodec.json
```

真实 replay 指标：

```text
first audio:             1920 ms
output duration:          3.56 s
boundary jump mean:       0.048286
boundary jump max:        0.089563
source revision events:   0
WAIT/WRITE:               4/3
```

### Stage 6：joint low-LR refinement

```text
checkpoints/simul_uniss_v1/stage6_joint_low_lr_15shard/
latest iteration: 20
```

```text
train lm loss:      4.5916 -> 4.2438
validation lm loss: 4.3359 -> 3.9991
grad norm:          7.28 -> 4.49
NaN/skipped:        0/0
```

### Stage 7：GRPO policy

```text
checkpoints/simul_uniss_v1/stage7_policy_grpo_short/policy_grpo.pt
```

```text
SFT loss:             0.6873 -> 0.5072
reward mean:         -0.4370 -> 0.4922
premature WRITE rate: 0.2075 -> 0.0278
KL:                   0 -> 0.3158
```

### Stage 8：NAR semantic diagnostic branch

```text
checkpoints/simul_uniss_v1/stage8_nar_semantic_short/nar_semantic.pt
```

100-step diagnostic run 已完成，所有 CTC、length、total loss 与 grad norm 均为有限值。该分支曲线仍有明显 batch 波动，只能证明可训练与可保存，尚不能证明 NAR 质量优于 autoregressive 分支。

## 5. TensorBoard 验收

TensorBoard 进程：

```text
tmux session: simul_uniss_v1_tensorboard
port: 6008
URL: http://<server-ip>:6008
logdir: runs/simul_uniss_v1/bootstrap_15shard/tensorboard
```

已读取并验证以下目录中的全部 scalar；所有数值均为 finite：

```text
stage0_prefix/
validation_latency/
gpu_smoke_stage1_audio/
gpu_smoke_stage1_token/
gpu_smoke_stage3_action/
gpu_smoke_stage4_interleaved/
gpu_smoke_stage5_bicodec/
gpu_smoke_stage5_bicodec_refinement/
gpu_smoke_stage6_joint/
short_stage1_token/
short_stage1_audio/
stage3_action/
stage4_interleaved/
stage6_joint/
short_stage5_bicodec/
short_stage5_replay/
short_stage7_grpo/
short_stage8_nar/
```

## 6. 测试与旧实验保护

最后一次代码回归：

```text
python -m unittest discover training/tests -q
Ran 118 tests
OK
```

已覆盖：

- 原 Phase1–3 数据、训练脚本与 dry-run 回归；
- Simul packed float loss weights；
- causal token/audio student 与 CTC；
- action/interleaved Megatron entrypoint；
- latency、controller、stable prefix 与 streaming codec；
- GRPO、NAR 与 BiCodec refinement；
- background launcher、completion marker 与 recovered CUDA library path。

用户提供的原始 DOCX 保持未跟踪，没有加入 commit：

```text
docs/uniss_training_reproduction/Simul_UniSS_方案分析与实施建议.docx
```

## 7. 实际修复记录

真实 GPU 执行发现并修复了两项只在干净 tmux/torchrun 环境出现的问题：

1. 绝对路径 torchrun 未包含 repository root，导致 `ModuleNotFoundError: training`；
2. recovered conda 环境中的 pip NVIDIA library directories 未进入 `LD_LIBRARY_PATH`，导致 Transformer Engine 找不到 `libcudnn_graph.so.9`。

对应提交：

```text
2c74266 Fix Simul-UniSS torchrun import path
f816a99 Restore CUDA libraries for Simul-UniSS Qwen stages
```

修复后 action、interleaved、joint GPU smoke 和完整 short pipeline 均实际跑通。

## 8. 当前限制与下一步

当前已经达到“15-shard 全流程工程闭环”的目标，但没有达到最终论文训练完成标准。下一步应使用新的独立实验目录执行：

1. 扩大 Stage 1 audio student 的真实音频覆盖和 multi-chunk curriculum；
2. 在固定 dev 集计算 BLEU/ASR-BLEU、speaker similarity、UTMOS/DNSMOS；
3. 对 320/640/960/1280 ms chunk size 绘制 quality–latency Pareto；
4. 增加 streaming wall-clock RTF、GPU memory、first audio latency profiling；
5. 对 Stage 3 action collapse 做 held-out language/domain 检查；
6. 将 Phase1 与未来 Phase2 checkpoint initialization 作为独立消融，禁止覆盖本次 Phase1-anchor 结果；
7. 只有当短训指标和音频人工检查通过后，再扩到更多 shard 或长训。

原 UniSS 全量 Phase2/Phase3 训练是另一条独立流水线，当前 Phase2 已开始实际 iteration，不应为 Simul-UniSS 可选蒸馏实验而中断。
