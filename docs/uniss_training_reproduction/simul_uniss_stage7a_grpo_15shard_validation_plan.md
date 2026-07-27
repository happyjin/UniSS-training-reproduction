# Simul-UniSS Stage7A：15-shard Action-only GRPO 快速验证计划

> 文档状态：Stage7A action-head v1、E0–E3 训练和 23,369 条 full test 已完成；第 17 章记录 E3 复盘与 Reward v2 路线；不代表 full-Qwen/semantic-token GRPO
>
> 生成日期：2026-07-27
>
> 适用仓库：`/opt/dlami/nvme/jasonleeeli/projects/UniSS`
>
> 目标：用最小、可归因、可复现的实验判断 GRPO 是否改善 Stage6 的 simultaneous quality-latency Pareto frontier
> 资源：单机 8 × NVIDIA H200，计划并行运行 4 个实验，每个实验固定 2 GPU

## 1. 结论先行

可以使用 8 张 H200 同时运行四个 2-GPU 实验，但需要满足两个前提：

1. 使用从 Stage6 Qwen/Hugging Face checkpoint 的 WAIT/WRITE LM-head 两行精确初始化的 action head；旧 `training/simul_uniss/policy_grpo.py` 的 5 维 MLP bootstrap 不用于本实验。
2. 正式计时前临时停止 GPU0 上的公网 Phase3 demo。demo 只占约 5.8 GiB，但并发训练会争用 GPU0 计算资源，使 baseline 延迟和 RTF 不可比较，也会让网页推理变慢。

四个最重要的首轮实验为：

| 实验 | GPU | 训练/评估 | 核心问题 |
|---|---|---|---|
| E0：Stage6 + learned policy/fixed wait-k baseline | 0,1 | 只评估 | 当前质量–延迟基线和 wait-k Pareto frontier 在哪里？ |
| E1：Stage6 + continued action SFT | 2,3 | 训练后评估 | 改善是否仅来自多训练、LoRA 或额外数据？ |
| E2：Stage6 + GRPO，group size 4 | 4,5 | 训练后评估 | 小 group GRPO 是否已能减少不必要 WAIT？ |
| E3：Stage6 + GRPO，group size 8 | 6,7 | 训练后评估 | 更大的组内探索是否带来更稳定的 Pareto 改善？ |

### 1.1 2026-07-27 实现与压测记录

独立实现路径：

```text
training/simul_uniss/stage7a/
experiments/simul_uniss_stage7a_15shard_v1/
training/tests/test_simul_stage7a.py
```

E0、E1、E2、E3 的 2-GPU smoke 均已完成；checkpoint、TensorBoard、validation、固定 wait-k 和 GPU monitor 均正常，无 OOM/NaN。Stage7A 样本长度中位数约 387、p95 约 602，故不把短样本补齐到 Megatron 的 13000/18000。正式 H200 参数采用每卡动态 `524288` padded tokens、最多 `1024` samples，同时保留 `max_sequence_length=18000` 作为异常长样本安全上限。实测稳态 GPU utility 为 100%，功率约 631–686 W/700 W，resident memory 约 38–40 GiB/卡。

当前 v1 的边界必须如实表述：冻结 Stage6 backbone，只训练由 Stage6 LM head 初始化的二分类 action head；实际 GRPO reward 使用 pseudo-alignment action、很小的 early-safe-WRITE bonus 和 final-WAIT penalty，尚未接入真实翻译质量、显式 First-WRITE/ATD/LAAL、结构或音频 reward。它是 action-policy proof，不是 semantic-token GRPO，也不更新翻译内容或声音生成权重。是否真正改善端到端质量–延迟 Pareto，必须由后续 free-running streaming dev/test 评估决定，不能由训练 reward 单独下结论。

有效性的核心判据不是训练 reward 或 GRPO loss，而是独立 dev/test 上：

```text
翻译质量基本不下降
+ first-WRITE / StartOffset / ATD / LAAL 明显降低
+ premature WRITE、漏译、重复和结构恢复不恶化
+ 优于相同训练量的 continued SFT
+ 至少一个 operating point 位于 fixed wait-k Pareto frontier 左上方
```

## 2. 当前基线与研究动机

### 2.1 固定初始化

四个实验统一使用当前已完成并导出的 Stage6：

```text
Megatron source:
checkpoints/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/
  stage06_joint_refinement/iter_0001189

Hugging Face export:
checkpoints/exported_hf/simul_uniss_stage6_streaming_v1_iter_0001189_hf

model.safetensors SHA256:
d211cf41ece3ed8843ac7d3a1d3027262268bb5f700e27a815de09393fbfee47
```

Stage6 checkpoint 只能以只读方式加载。所有 Stage7A checkpoint、optimizer、日志、TensorBoard 和评估结果必须写入新的独立命名空间。

### 2.2 当前策略问题

已有 Stage6 batch-one streaming 评估表明：

| 指标 | 当前量级 |
|---|---:|
| 中文→英文平均首次 WRITE | 约 4.50 秒 |
| 英文→中文平均首次 WRITE | 约 4.79 秒 |
| 每句平均目标音频 chunk | 约 1.12–1.25 |
| premature WRITE | 约 1%–2.6% |
| unnecessary WAIT | 约 13%–17.8% |
| batch-one RTF | 约 0.16–0.18 |

这说明当前瓶颈主要不是推理速度，而是 learned policy 过于保守：模型常常具备实时计算能力，却继续 WAIT，最后接近句尾时只输出一个较大的语音块。

### 2.3 为什么 SFT 不足

SFT 为每个 source event 提供一个唯一 WAIT/WRITE 标签，但真实 simultaneous translation 往往存在多个合理动作：

- 更早 WRITE：延迟更低，但可能依赖尚未出现的未来信息；
- 晚一点 WRITE：翻译更稳定，但首包延迟和 ATD 更高；
- 短 phrase：响应更快，但语音碎片和 boundary gap 可能增加；
- 长 phrase：自然度更好，但会退化成接近 offline 的一次性输出。

GRPO 可以对同一输入采样多个完整决策轨迹，用最终翻译、prefix 安全性、延迟、结构完整性和语音连续性共同排序，而不要求训练单独的 value critic。

## 3. 本实验能证明什么、不能证明什么

### 3.1 可以证明

- action-only GRPO 能否比原 Stage6 更合理地选择 WAIT/WRITE；
- 改善是否超出 continued SFT 带来的变化；
- group size 4 与 8 的探索收益是否值得额外生成开销；
- learned policy 是否超过固定 wait-k 的 quality-latency frontier；
- 15-shard 条件下是否值得继续做 full198 Stage7。

### 3.2 不能直接证明

- full198 正式 Stage7 已完成；
- GRPO 已优化 BiCodec semantic token；
- GRPO 已解决真实 streaming encoder 的所有问题；
- pseudo proportional alignment 等价于真实 contextual alignment；
- 15-shard 结果可以直接作为论文最终数字。

当前 full198 schedule 使用：

```text
alignment_kind=pseudo_proportional_token_alignment
```

因此 Stage7A 是方法可行性验证。若成功，full198 正式实验前仍应优先补充真实或高置信 contextual alignment、full Stage1/2 和正式 joint Stage6。

## 4. 数据与切分

### 4.1 训练数据

复用已经冻结的 15-shard Simul-UniSS train subset，不重新制作或覆盖历史数据：

```text
data/processed/simul_uniss_v2_15shard/
data/megatron/simul_uniss_v2_15shard/
```

训练前记录：

- 15 个输入 shard 的文件名、大小和 SHA256；
- schedule、packed action 数据和 tokenizer hash；
- 样本数、event 数、WAIT/WRITE 分布；
- source/target language 分布；
- 是否与 dev/test ID 有交集，交集必须为 0；
- alignment kind 和 chunk 配置。

### 4.2 调参与最终评估

```text
train: 固定 15-shard train subset
dev: 当前固定 simultaneous dev，允许选择 reward 权重和 checkpoint
test: 当前固定 simultaneous test，只在配置冻结后运行一次
```

严禁根据 test 结果重新选择：

- reward 权重；
- group size；
- action temperature；
- WAIT/WRITE logit bias；
- chunk size；
- fixed wait-k；
- checkpoint iteration。

## 5. Stage7A 模型边界

### 5.1 推荐结构

Stage7A 只训练 action policy：

```text
Stage6 Qwen hidden state at action position
                  ↓
        action-only LoRA / adapter
                  ↓
             WAIT or WRITE
```

冻结：

- Stage6 基础 Qwen 权重；
- tokenizer 和 embedding；
- Streaming GLM/CTC 模块；
- target phrase 内容生成；
- semantic token generation；
- BiCodec decoder；
- speaker/global tokens。

训练：

- Qwen 顶部少量层的 action-only LoRA，或独立 action head；
- 只在 WAIT/WRITE action position 计算 RL log probability；
- adapter 在生成 target text 和 semantic tokens 时关闭，或保持完全冻结。

该边界保证首轮实验主要回答“什么时候说”，而不是同时改变“说什么”和“声音怎么生成”。

### 5.2 强制安全规则

- source final 时屏蔽 WAIT，强制 WRITE/flush；
- 已提交 target text 和 waveform 永不回滚；
- 未满足 Source/Target CTC eligibility 时默认 WAIT；
- 单个 rollout 设置最大连续 WAIT、最大 WRITE 次数和最大 token budget；
- 重复、空输出、无 EOS、结构恢复都进入 reward penalty；
- 训练崩溃、NaN 或 reward 异常时不覆盖任何历史 checkpoint。

## 6. 四个实验的严格定义

### 6.1 E0：Stage6 learned policy + fixed wait-k baseline

GPU：`0,1`

性质：只评估，不更新模型。

在同一 Stage6 checkpoint、相同 source schedule、相同生成参数下运行：

```text
E0-A: Stage6 learned WAIT/WRITE policy
E0-B: fixed wait-k=1
E0-C: fixed wait-k=2
E0-D: fixed wait-k=3
E0-E: fixed wait-k=5
E0-F: oracle schedule（只作为上界，不可部署）
```

产物：

- dev 逐句结果；
- batch-one latency 结果；
- quality-latency Pareto 图；
- Stage6 当前 unnecessary WAIT 和 premature WRITE 分布；
- test 仅在配置冻结后运行。

### 6.2 E1：continued action SFT

GPU：`2,3`

初始化：Stage6 iteration 1189。

训练参数容量：与 E2/E3 使用相同 action LoRA/adapter。

训练数据：相同 15 shards。

训练 step、optimizer、学习率和 replay 比例：尽可能与 GRPO 实验匹配。

目标：排除以下混淆：

- 只是多训练了一段时间；
- 只是增加了 LoRA；
- 只是重新看了 15-shard 数据；
- 只是使用了新的 optimizer 或 scheduler。

E1 仅使用 action CE/SFT loss，不使用 group rollout reward。

### 6.3 E2：GRPO group size 4

GPU：`4,5`

初始化、adapter、数据和基础超参数与 E1 相同。

每个 source schedule 采样 4 条 action rollout。

目的：验证较低生成开销下，group-relative advantage 是否足以减少 unnecessary WAIT。

### 6.4 E3：GRPO group size 8

GPU：`6,7`

其余设置与 E2 相同，只把 group size 从 4 改为 8。

在 2 GPU 上，group size 8 并不表示 8 个候选同时分别占一张 GPU。建议每卡处理 4 个候选或采用 micro-batched rollout，因此功能可行，但 wall-clock 时间预计高于 group size 4。E2/E3 必须报告真实生成吞吐和每 step 时间。

## 7. GRPO rollout 与 reward

### 7.1 Rollout 单位

一个 rollout 是同一 source utterance 从第一个 source chunk 到 final flush 的完整 action 轨迹：

```text
chunk_0 → WAIT
chunk_1 → WAIT
chunk_2 → WRITE
chunk_3 → WAIT
chunk_4/final → WRITE + flush
```

同一输入的 4 或 8 条 rollout 使用相同 source chunks、reference 和冻结的 Stage6 phrase/semantic generator，仅 action sampling 不同。

### 7.2 首轮 reward

建议首轮使用可解释的分量，而不是只保存总 reward：

```text
R = 2.0 * R_final_translation
  + 0.5 * R_prefix_translation
  - 0.4 * R_latency
  - 1.0 * R_premature_write
  - 0.5 * R_repetition_or_structure
  + 0.2 * R_voice_boundary_proxy
  - 0.02 * KL(policy || Stage6_reference)
```

其中：

- `R_final_translation`：最终 text COMET/chrF/BLEU 或冻结 teacher score；
- `R_prefix_translation`：已提交 prefix 是否被后续完整上下文支持；
- `R_latency`：StartOffset、ATD、LAAL 的归一化组合；
- `R_premature_write`：输出依赖尚未出现的未来信息、与 safe boundary 冲突；
- `R_repetition_or_structure`：重复、空输出、漏译、无 EOS、forced recovery；
- `R_voice_boundary_proxy`：semantic repetition、chunk 时长和 boundary proxy；
- `KL`：限制 action policy 偏离冻结 Stage6 reference。

首轮不建议为每个 rollout 都完整运行昂贵的 ASR、COMET、UTMOS 和 speaker encoder。可以采用两级 reward：

1. 每步训练使用快速 teacher/text/latency/structure proxy；
2. 每个 eval interval 对固定 dev subset 做真实 end-to-end audio decode 和完整指标。

### 7.3 Group-relative advantage

同一输入的组内 reward 标准化：

```text
A_i = (R_i - mean(R_group)) / max(std(R_group), epsilon)

L_GRPO = -mean(A_i * log P(action_trajectory_i))
         + beta * KL(policy || reference)

L_total = L_GRPO + lambda_sft * L_action_replay
```

建议保留 20%–30% action SFT replay，防止策略为了降低延迟而遗忘 Stage3/4/6 行为。

## 8. 详细决策例子

源语音：

```text
The bank by the river was covered in snow.
```

目标：

```text
河岸被雪覆盖了。
```

音频逐块到达：

```text
0.00–0.64s: The bank
0.64–1.28s: by the river
1.28–1.92s: was covered
1.92–2.56s: in snow
```

第一块只有 `The bank`，`bank` 可能是“银行”或“河岸”。组内候选可能为：

| Rollout | 决策 | 翻译质量 | 延迟 | premature 风险 | 示例总 reward |
|---|---|---:|---:|---:|---:|
| A | 0.64 秒 WRITE“银行” | 0.30 | 好 | 严重 | -0.44 |
| B | WAIT 到 1.28 秒后 WRITE“河岸” | 0.95 | 较好 | 无 | 2.39 |
| C | 一直 WAIT 到 2.56 秒再输出整句 | 0.95 | 差 | 无 | 2.17 |
| D | 0.64 秒猜测 WRITE“河岸” | 0.95 | 最好 | 高 | 1.69 |

GRPO 应提高 B 的概率：它不是最早，也不是最晚，而是在歧义解除后立即输出。A 会因最终错误和 premature penalty 被压低；C 虽然正确，但延迟 penalty 更大；D 即使碰巧正确，也会因 prefix 不安全受到惩罚。

## 9. 公平对比约束

E1/E2/E3 必须尽量保持以下完全一致：

- Stage6 初始化权重和 hash；
- 15-shard train IDs；
- dev/test IDs；
- action adapter 结构、rank 和参数量；
- optimizer、学习率和训练 step；
- 最大 source/target 长度；
- source chunk size；
- phrase/semantic frozen generation 参数；
- SFT replay 数据和比例；
- checkpoint/eval interval；
- 三个随机种子；
- TensorBoard 指标命名；
- batch-one latency 测量硬件与脚本。

允许的主要差异：

```text
E1: action CE
E2: GRPO, G=4
E3: GRPO, G=8
```

## 10. 指标与验收阈值

### 10.1 翻译和内容

- Text BLEU；
- chrF；
- COMET；
- speech/ASR-BLEU；
- ASR-COMET；
- under-translation；
- repetition；
- EOS/empty-output/structure failure。

### 10.2 Streaming latency

- first-WRITE time；
- StartOffset CA/NCA；
- EndOffset CA/NCA；
- ATD；
- AL、LAAL、DAL、AP；
- 每句 WRITE 数和目标音频 chunk 数；
- 最大连续 WAIT；
- playback gap。

### 10.3 Action policy

- WAIT/WRITE precision、recall、F1；
- premature WRITE rate；
- unnecessary WAIT rate；
- final forced action rate；
- structural recovery rate；
- action entropy 和 write rate。

### 10.4 计算与音频

- batch-one RTF；
- TTFT；
- rollout samples/s；
- training step time；
- peak GPU memory、utilization、power；
- UTMOS、AutoPCP、speaker similarity；
- boundary click、RMS jump、spectral distance；
- output/source duration ratio。

### 10.5 首轮成功标准

| 指标 | 建议门槛 |
|---|---:|
| first-WRITE | 相比 Stage6 降低 ≥15%，或绝对降低 ≥500 ms |
| StartOffset | 降低 ≥10%–15% |
| ATD/LAAL | 降低 ≥10% |
| unnecessary WAIT | 从约 13%–18% 降至 8%–12% |
| Text BLEU | 相比 Stage6 下降不超过 0.5 |
| COMET | 下降不超过 0.01 |
| premature WRITE | 增加不超过 1 个百分点 |
| under-translation/repetition | 不显著增加 |
| RTF | 保持 `<1` |
| structure recovery | 不得显著增加 |

不能只满足延迟条件。例如 first-WRITE 从 4.5 秒降到 2.5 秒，但 BLEU 下降 5 分、premature WRITE 上升到 9%，只能说明策略变得激进，不能证明 GRPO 有效。

## 11. Pareto 与统计检验

### 11.1 Pareto 曲线

横轴至少使用 StartOffset、ATD 或 LAAL，纵轴使用 Text BLEU、COMET 或 ASR-BLEU。

E0 提供：

```text
fixed wait-k = 1, 2, 3, 5
Stage6 learned policy
oracle schedule
```

GRPO 提供：

```text
G4 balanced
G8 balanced
可选：通过 action logit bias 得到 conservative / fast operating points
```

GRPO 有效的严格定义：

- 相同质量下延迟更低；或
- 相同延迟下质量更高；并且
- 至少一个 GRPO 点位于 fixed wait-k frontier 的左上方。

### 11.2 统计检验

保存每条样本的 paired 结果，对相同 dev/test utterance 计算：

```text
delta Text BLEU / COMET
delta first-WRITE / StartOffset / ATD / LAAL
delta premature WRITE / unnecessary WAIT
```

使用：

- 10,000 次 paired bootstrap；
- 95% confidence interval；
- 至少 3 个训练随机种子；
- 报告 mean ± std。

示例可接受结果：

```text
first-WRITE delta = -0.82 s
95% CI = [-0.91, -0.73]

Text BLEU delta = -0.18
95% CI = [-0.36, +0.02]
```

这表示延迟改善稳定，而 BLEU 没有显著下降。

## 12. 8-GPU 并行资源设计

### 12.1 推荐固定分配

| GPU | 实验 | 建议 tmux | 建议端口 | 独立输出根目录 |
|---|---|---|---:|---|
| 0,1 | E0 baseline/wait-k eval | `simul_stage7a_e0_baselines` | 30470 | `eval_outputs/simul_uniss_stage7a_15shard_v1/e0_baselines/` |
| 2,3 | E1 continued SFT | `simul_stage7a_e1_sft` | 30471 | `checkpoints/simul_uniss_stage7a_15shard_v1/e1_continued_sft/` |
| 4,5 | E2 GRPO G4 | `simul_stage7a_e2_grpo_g4` | 30472 | `checkpoints/simul_uniss_stage7a_15shard_v1/e2_grpo_g4/` |
| 6,7 | E3 GRPO G8 | `simul_stage7a_e3_grpo_g8` | 30473 | `checkpoints/simul_uniss_stage7a_15shard_v1/e3_grpo_g8/` |

所有实验额外使用独立：

```text
experiments/simul_uniss_stage7a_15shard_v1/<experiment>/
logs/simul_uniss_stage7a_15shard_v1/<experiment>/
runs/simul_uniss_stage7a_15shard_v1/tensorboard/<experiment>/
eval_outputs/simul_uniss_stage7a_15shard_v1/<experiment>/
```

严禁共享：

- output checkpoint directory；
- TensorBoard event directory；
- master port；
- temporary rollout cache；
- generated audio/result JSONL；
- random seed state。

### 12.2 GPU0 公网 demo 冲突

当前 GPU0 驻留：

```text
tmux: uniss_offline_phase3_demo
GPU memory: 约 5.8 GiB
```

如果四个实验必须各使用 2 GPU并且需要可比较的 latency/RTF，正式运行前应执行：

```bash
web_demo/offline_s2st_phase3_v1/stop.sh
```

实验结束后再重新启动 demo。若 demo 必须持续在线，则建议改成：

```text
GPU0: demo only
GPU1: E0 baseline（单卡）
GPU2,3: E1
GPU4,5: E2
GPU6,7: E3
```

这种分配不再是“四个实验各 2 卡”，但比让 benchmark 与公网请求共享 GPU0 更可信。

### 12.3 并行瓶颈

四路同时运行时必须监控：

- CPU decode/tokenization 是否成为瓶颈；
- 15-shard schedule 是否被四个进程重复顺序扫描；
- NVMe read bandwidth；
- `/tmp` 与 rollout cache 空间；
- vLLM/HF cache 是否互相覆盖；
- H200 power、utilization、memory；
- E0 latency 是否受到 E1–E3 的 CPU/I/O 干扰。

如 E0 是正式 batch-one 延迟测量，最严谨的方法是先完成 E1–E3 训练，再在 GPU 空闲时单独复测 E0/E1/E2/E3 的 batch-one latency。并行 E0 适合先生成质量结果，不适合作为最终 computation-aware latency 数字。

## 13. 实施顺序

### Phase A：实现与单元测试

1. 新增独立 Stage7A action adapter/LoRA 模块；
2. 新增 Stage6 HF action-position rollout wrapper；
3. 新增 group rollout、reference KL 和 reward 分量；
4. 新增 continued SFT control；
5. 新增 fixed wait-k/oracle evaluator；
6. 新增逐句 paired metrics 和 Pareto report；
7. 单元测试 reward、advantage、action mask、final flush 和 checkpoint isolation。

### Phase B：单卡 smoke

- 4–16 条 schedule；
- E1/E2/E3 各 2–5 step；
- 检查显存、NaN、reward 分量、LoRA 更新和 frozen weight hash；
- 真实生成至少一条翻译文本和音频；
- 确认旧 Stage6 checkpoint hash 不变。

### Phase C：每实验 2-GPU smoke

- 每个实验 20–50 step；
- 检查 DDP/rollout 数据无重复或漏样；
- 检查两个 rank 的 reward、KL 和 optimizer 同步；
- 检查独立端口、tmux、日志和 TensorBoard。

### Phase D：四路 15-shard 正式运行

并行启动 E0–E3，训练过程不共享可写目录。每个实验完成后写入不可变 marker，包含：

- git commit；
- config 和输入 hash；
- 训练 step；
- final/best checkpoint；
- dev metric summary；
- NaN/OOM/skip count；
- GPU 监控摘要。

### Phase E：统一评估

1. 固定每个实验的 best dev checkpoint；
2. 在 GPU 空闲状态下使用同一脚本逐个跑 batch-one latency；
3. 完整 dev 指标和三随机种子统计；
4. 冻结配置；
5. test 只运行一次；
6. 输出统一 Markdown 报告、CSV、JSON 和 Pareto 图。

## 14. 决策规则

### 14.1 继续 full198 Stage7

满足以下条件才进入 full198：

- E2 或 E3 明显优于 E1；
- 至少一个 GRPO point 超过 fixed wait-k Pareto frontier；
- 延迟达到预设改善；
- BLEU/COMET/premature/under-translation 通过门槛；
- 三个随机种子趋势一致；
- reward 没有出现 always-WAIT、always-WRITE 或短输出投机。

### 14.2 暂停 GRPO，先修数据或 SFT

以下情况优先返回 alignment/Stage1–6，而不是扩大 GRPO：

- E1 与 E2/E3 同样改善，说明 GRPO 没有独立贡献；
- reward 上升但独立 dev 质量下降；
- premature WRITE 显著增加；
- 中文→英文和英文→中文趋势相反；
- GRPO 只学到 pseudo alignment 或 fixed wait-k 模式；
- 每句仍只有约一个 WRITE，无法形成真正 phrase-level streaming；
- structural recovery 或 final forced action 增加。

### 14.3 Stage8 NAR

Stage7A 不涉及 NAR。只有在经过 GRPO 后，真实 batch-one p95 RTF 仍接近或超过 1、生成时间持续大于 source chunk interval时，才有必要正式训练 Stage8。

## 15. 第一轮报告必须包含

1. 四个实验的唯一配置和 checkpoint hash；
2. 训练曲线：SFT/GRPO loss、reward 各分量、KL、write rate；
3. 质量表：BLEU、chrF、COMET、ASR-BLEU、UTMOS、AutoPCP；
4. 延迟表：first-WRITE、StartOffset、ATD、AL、LAAL、DAL、EndOffset；
5. action 表：premature WRITE、unnecessary WAIT、WRITE F1、forced actions；
6. fixed wait-k 与 GRPO Pareto 图；
7. 三随机种子和 paired bootstrap 置信区间；
8. 至少 20 个逐句案例，包括成功提前输出、歧义等待、失败早写和长句退化；
9. 是否建议进入 full198 Stage7 的明确结论；
10. 所有限制：15 shards、pseudo alignment、frozen phrase/semantic generation。

## 16. 最终判断模板

只有类似以下结果才能支持“Stage7A GRPO 有效”：

```text
相对 Stage6：
- first-WRITE -0.8 s，95% CI 不跨 0；
- ATD/LAAL -10% 以上；
- Text BLEU -0.2，未超过 -0.5 门槛；
- premature WRITE +0.4 个百分点，未超过 +1；
- unnecessary WAIT 从 13% 降至 8%；
- GRPO 明显优于同 step continued SFT；
- 至少一个 GRPO operating point 位于 fixed wait-k frontier 左上方；
- 三个随机种子方向一致。
```

如果只看到 reward 上升、loss 下降或训练集动作准确率提高，不能据此宣称 GRPO 改善 simultaneous speech-to-speech translation。

## 17. 2026-07-27 Full-test 复盘与 Reward v2 优化路线

### 17.1 本章目的、证据范围与追踪入口

本章把 Stage7A v1 的实验结果、失败机制、reward 实现差异、优化 motivation、下一轮消融矩阵和验收规则放在同一个位置，避免后续只看到某个 TensorBoard reward 上升就误判方法有效。

本章结论基于：

- E0 Stage6、E1 continued SFT、E2 GRPO G4、E3 GRPO G8；
- 完全相同的 `23,369` 条 free-running streaming S2ST test schedules；
- 完整的文本生成、BiCodec 音频解码、ASR、Text/Speech BLEU、UTMOS、AutoPCP；
- streaming policy/latency 指标和每组 200 条 batch-one latency audit；
- E1/E2/E3 相同数据、初始化、训练步数范围、action-head 容量和评估协议。

完整结果入口：

```text
eval_outputs/simul_uniss_stage7a_15shard_v1/full_test_e2e_v1/
├── stage7a_four_way_full_test_report.md
├── comparison.json
├── e0_stage6/full_test_v1/
├── e1_continued_sft/full_test_v1/
├── e2_grpo_g4/full_test_v1/
└── e3_grpo_g8/full_test_v1/
```

对应结果 commit：

```text
4055697 Add Stage7A four-way full test results
```

必须注意：本轮已经查看并使用 test aggregate 诊断 Reward v1，因此后续 Reward v2 不能反复用同一个 test 调权重。Reward、checkpoint、WRITE bias 和阈值只能在 dev 上选择；正式确认最好使用新的冻结 holdout。如果没有新 holdout，后续结果必须标注为 iterative test，而不能再称完全独立的 confirmatory test。

### 17.2 四组 full-test 结果与 E3 定位

#### 17.2.1 质量

| Experiment | Text BLEU zh→en | Text BLEU en→zh | Speech BLEU zh→en | Speech BLEU en→zh | UTMOS zh→en | UTMOS en→zh |
|---|---:|---:|---:|---:|---:|---:|
| E0 Stage6 | 26.378 | 40.560 | 1.155 | 37.251 | 3.558 | 3.362 |
| E1 continued SFT | 26.240 | 40.480 | 1.164 | 37.187 | 3.560 | 3.361 |
| E2 GRPO G4 | 27.587 | 40.224 | 1.127 | 36.990 | 3.556 | 3.361 |
| E3 GRPO G8 | **28.109** | 40.355 | **1.215** | 37.010 | **3.561** | **3.362** |

E3 相对 matched E1：

- zh→en Text BLEU `+1.869`；
- en→zh Text BLEU `-0.126`；
- zh→en Speech BLEU `+0.050`；
- en→zh Speech BLEU `-0.176`；
- UTMOS/AutoPCP 基本不变。

因此 E3 不是“完全没有作用”。更大的 group exploration 确实产生了可测量的 policy shift，并在 zh→en 上改善质量；但收益具有明显方向不对称，不能用 zh→en 的提升掩盖 en→zh 的下降。

#### 17.2.2 同传时延和 action 行为

| Experiment | First WRITE ms | ATD ms | LAAL proxy | Premature WRITE | Unnecessary WAIT | Reported final flush | Forced final flush |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 Stage6 | **3986.4** | **1807.3** | 44.58 | 0.031 | 0.158 | 1.000 | 10.065% |
| E1 continued SFT | 3991.0 | 1808.4 | **44.54** | 0.030 | **0.157** | 1.000 | 9.945% |
| E2 GRPO G4 | 4055.4 | 1839.0 | 45.54 | **0.027** | 0.164 | 1.000 | 9.530% |
| E3 GRPO G8 | 4032.8 | 1827.5 | 45.10 | 0.029 | 0.162 | 1.000 | **9.187%** |

E3 相对 E1：

- First WRITE `+41.8 ms`，更慢；
- ATD `+19.1 ms`，更慢；
- LAAL proxy `+0.564`，更差；
- premature WRITE `-0.002`，略好；
- unnecessary WAIT `+0.005`，更差；
- forced final flush 从 `9.945%` 降到 `9.187%`，说明 E3 对最终 WRITE 有小幅正作用，但仍有约 `9%` 的样本依赖 runtime 强制恢复；
- batch-one source RTF `0.167`，计算速度不是主要瓶颈。

结论不是“GRPO 没学习”，而是“GRPO 学到了错误的 trade-off”：它更保守，减少了少量 premature WRITE，却通过更多 WAIT 换取安全性和部分 zh→en 质量，因而没有实现降低 simultaneous latency 的主要目标。

#### 17.2.3 Action validation 暴露的隐藏问题

best checkpoint 的 512 条 action validation：

| Metric | E1 | E2 G4 | E3 G8 |
|---|---:|---:|---:|
| First-write MAE ms | **498.89** | 526.73 | 526.98 |
| Premature given WAIT | 0.01288 | **0.01090** | 0.01255 |
| Unnecessary WAIT given WRITE | **0.32050** | 0.32600 | 0.32187 |
| Predicted writes/sample | **1.041** | 1.021 | 1.037 |
| Reference writes/sample | 1.420 | 1.420 | 1.420 |
| Predicted final-flush success | 0.867 | 0.875 | **0.877** |

这里有两个不能被 full-test aggregate 掩盖的问题：

1. E3 平均只预测约 `1.037` 次 WRITE，而 reference 为 `1.420`，说明策略存在 under-WRITE/过度等待倾向。
2. action head 在 512 条 validation 上的 final-flush success 只有约 `87.7%`；full test 中 E3 仍有 `2,147/23,369=9.187%` 的样本由 runtime 把 final WAIT 强制改为 WRITE。full-test 报告的 `1.000` 是强制恢复后的结构成功率，不是 policy 原始 final-WRITE 成功率。下一轮必须单独报告 `predicted_final_flush` 与 `forced_final_flush`，不能把强制恢复算作 policy 自己成功。

### 17.3 当前代码中的实际 Reward v1

第 7.2 节描述的是目标形态，但 Stage7A v1 代码实际实现的是简化 action-only pseudo-label reward。当前默认权重为：

```text
correct          = +1.0
incorrect        = -1.0
premature_write  = -2.0
unnecessary_wait = -0.5
final_wait       = -5.0
safe_early_write = +0.2 * (1 - event_fraction)
```

由于 `incorrect` 会与 action-specific penalty 相加，实际单事件收益大致为：

| Action case | 实际 reward |
|---|---:|
| 正确 WAIT | `+1.0` |
| 正确 WRITE | `+1.0` 到 `+1.2` |
| premature WRITE | `-1.0 - 2.0 = -3.0` |
| unnecessary WAIT | `-1.0 - 0.5 = -1.5` |
| final WAIT | 约 `-1.0 - 0.5 - 5.0 = -6.5` |

训练还使用：

```text
group size: E2=4, E3=8
KL beta: 0.02
SFT replay weight: 0.2
前 100 steps: SFT warmup
```

Reward v1 没有直接使用：

- free-running First WRITE；
- StartOffset、ATD、LAAL、DAL；
- 最终 Text/Speech BLEU、COMET 或 teacher translation score；
- prefix consistency/歧义解除时间；
- under-translation、重复和真实 forced recovery；
- 音频 continuity/UTMOS/AutoPCP。

因此不能期待只靠当前 reward 自动优化完整 quality-latency Pareto。它本质上是在模仿 pseudo WAIT/WRITE label，并对两类错误施加不对称 cost。

### 17.4 为什么 Reward v1 会得到当前 E3 结果

#### 17.4.1 Premature 与 unnecessary 的惩罚不对称

premature WRITE 的有效 cost 是约 `-3.0`，unnecessary WAIT 只有约 `-1.5`。在存在不确定性时，WAIT 的最坏风险明显更低，最容易学到的策略就是“宁可晚一点，不要早写”。

这与 full test 完全一致：

```text
premature WRITE: 0.030 → 0.029
unnecessary WAIT: 0.157 → 0.162
First WRITE:     3991 → 4033 ms
ATD:             1808 → 1828 ms
```

#### 17.4.2 没有真正的 trajectory latency reward

`safe_early_write` 只在 pseudo label 已经是 WRITE 时给最多 `0.2` 的小 bonus。它不能奖励“比 pseudo schedule 更早但依然语义安全”的 WRITE，也没有直接比较 rollout 与 E1/Stage6 的 First WRITE、ATD 或 LAAL。

换句话说，Reward v1 能学“不要违反 pseudo label”，却缺少“在安全边界出现后立即写”的强梯度。

#### 17.4.3 事件平均稀释关键决策

当前把一条 trajectory 的 event reward 按事件数取平均，再把整条 trajectory 的 log probability 也按事件平均。长句中一个关键 first-WRITE 或 final-flush 决策会被大量普通 WAIT 事件稀释；同一个 trajectory advantage 又被平均分配给所有 action，credit assignment 不够精确。

#### 17.4.4 Pseudo label 不是最终目标

Pseudo action label 只能表示“当前处理流程认为何时可以写”，不是端到端翻译质量和 latency 的真实 Pareto optimum。模型把 pseudo accuracy 做高，不等于 streaming BLEU、First WRITE 或 ATD 会更好。

#### 17.4.5 Checkpoint selection 没有选择端到端 latency

当前 best score 是：

```text
write_f1
- 0.25 * premature_write_given_wait
- 0.10 * unnecessary_wait_given_write
```

虽然 validation 已经计算 `first_write_mae_ms`，selection score 并未使用它，也没有使用 predicted/forced final flush、writes/sample、Text BLEU 或 free-running ATD。E3 best step 700 因而是 action proxy 最优，不保证是端到端 quality-latency 最优。

#### 17.4.6 Group size 8 只改善探索，不修正 reward 方向

G8 比 G4 有更多组内候选，E3 的质量和 latency 都比 E2 更好，说明增加 exploration 有价值。但 group size 只能更充分地优化当前 reward；如果 reward 偏向保守，G8 不会自动把目标改成低延迟。

### 17.5 Reward v2 优化原则

Reward v2 的目标不应是“最小化 latency，不管质量”，也不应继续用一个固定大 premature penalty 把模型锁成 always-WAIT。推荐把问题定义为：

```text
在 translation quality、premature、coverage 和 final flush 约束通过的前提下，
最大化相对 matched E1 的 First-WRITE / ATD / LAAL 改善。
```

也就是 constrained quality-latency optimization，而不是把所有指标随意相加成一个无法解释的总分。

### 17.6 优化方向一：重构 mutually-exclusive action cost

不要再让 `incorrect=-1` 与 premature/unnecessary penalty 隐式重复叠加。改成互斥 case，使每个数字的意义清楚：

```text
correct WAIT       → +r_wait_correct
correct WRITE      → +r_write_correct
premature WRITE    → -lambda_premature
unnecessary WAIT   → -lambda_unnecessary
final WAIT         → -lambda_final
```

首轮不应直接固定一个“最优”数字，而应在 dev 做小网格：

```text
lambda_premature  ∈ {1.0, 1.5, 2.0}
lambda_unnecessary∈ {1.0, 1.5, 2.0, 3.0}
lambda_final      ∈ {3.0, 5.0}
```

核心是把 `lambda_unnecessary / lambda_premature` 从当前约 `0.5` 提高到 `0.75–1.5` 区间，同时用独立 premature gate 防止策略变成 always-WRITE。

Motivation：当前失败不是 premature 太高，而是 unnecessary WAIT 和 first-WRITE 没有改善；继续增大 premature penalty 只会加重保守行为。

### 17.7 优化方向二：加入显式 trajectory-level latency delta

每条 rollout 应与同一 utterance 的 matched reference/E1 schedule 比较，而不是使用跨句绝对毫秒值：

```text
R_first = clip((FirstWrite_ref - FirstWrite_rollout) / scale_first, -1, 1)
R_atd   = clip((ATD_ref - ATD_rollout) / scale_atd, -1, 1)
R_laal  = clip((LAAL_ref - LAAL_rollout) / scale_laal, -1, 1)

R_latency = 0.4 * R_first + 0.35 * R_atd + 0.25 * R_laal
```

这里的系数只是首轮可解释起点，必须在 dev 冻结。使用 per-utterance delta 有三点好处：

1. 消除长短句绝对时长差异；
2. 直接优化要超过的 matched E1；
3. group 内候选能明确排序“同质量下谁更早写”。

First WRITE 不能单独占全部 reward，否则模型可能尽早写一个 token 后长期 WAIT。ATD/LAAL 和 coverage 必须同时存在。

### 17.8 优化方向三：safe-boundary 后的递增 WAIT penalty

一旦 contextual/pseudo eligibility gate 判断当前可以安全 WRITE，继续 WAIT 的 cost 应随连续等待次数递增：

```text
eligible_wait_count = 从首个 safe boundary 起连续 WAIT 的次数

R_late_wait = -lambda_wait * min(eligible_wait_count, cap)
```

在首个或较早 safe boundary WRITE 时给 potential-based bonus：

```text
R_safe_commit = +lambda_commit / (1 + eligible_wait_count)
```

这比当前 `0.2 * (1-event_fraction)` 更准确，因为它奖励的是“歧义已经解除后的立即提交”，不是简单奖励 utterance 前半段的所有 label-WRITE。

### 17.9 优化方向四：用约束而不是固定大 penalty 保证安全

推荐使用 Lagrangian/dual update：

```text
maximize  E[R_latency + R_quality + R_coverage]

subject to:
  premature_write <= E1 + 0.01
  Text BLEU(direction) >= E1(direction) - 0.5
  predicted_final_flush >= 0.99
  forced_recovery <= 0.01
```

当某个约束超标时自动提高对应 multiplier；约束满足后不再持续用过大的固定 penalty 压制 WRITE。

Motivation：当前固定 premature cost 即使 premature 已经低于 E1，仍持续推动模型更保守。自适应约束只在真正违规时施压，更适合寻找 Pareto frontier。

### 17.10 优化方向五：加入 coverage、under-WRITE 和真实 final-flush reward

E3 的 `predicted_writes/sample=1.037`，明显低于 reference `1.420`。Reward v2 应包含：

```text
R_write_coverage = -abs(num_writes - target_writes) / max(1, target_writes)
R_max_wait       = -normalized_max_consecutive_wait
R_under_trans    = -under_translation_or_missing_prefix_score
R_forced_flush   = -I(runtime had to force final WRITE)
R_structure      = -I(empty/repetition/missing EOS/recovery)
```

`target_writes` 不能机械要求等于 reference 每个点位，可以使用允许区间；但必须防止模型用“一次很晚的 WRITE + runtime forced flush”伪装成结构正确。

下一轮报告必须同时列出：

- predicted final flush；
- forced final flush；
- writes/sample；
- max consecutive WAIT；
- audio chunks/sample；
- under-translation 和 repetition。

### 17.11 优化方向六：加入双语平衡的质量 reward

E3 呈现 zh→en 改善、en→zh 下降，说明单一全局 reward/采样分布可能偏向某个方向。建议：

1. 分方向计算 reward mean/std，再做 group normalization；
2. batch 内强制 zh→en 与 en→zh 平衡；
3. checkpoint selection 使用 worst-direction retention，而不是双向平均掩盖下降；
4. 使用 prefix teacher score、chrF/BLEU 或冻结翻译模型 log-prob 作为快速质量 reward；
5. 每个 eval interval 在固定 dev subset 上运行真实 free-running Text/Speech BLEU。

推荐的质量项：

```text
R_quality = 0.6 * R_final_translation
          + 0.4 * R_prefix_consistency
          - R_under_translation
          - R_repetition
```

对于 action-only GRPO，backbone 虽然冻结，action schedule 仍会影响 prefix commit、截断、重复和最终音频内容，因此质量 reward 仍然必要。

### 17.12 优化方向七：改善 trajectory credit assignment

当前整条 trajectory 共享一个 sample-level advantage。Reward v2 建议增加 event-level return-to-go：

```text
G_t = r_t + gamma * r_{t+1} + ... + gamma^(T-t) * r_T
```

并对关键事件单独加权：

- first safe boundary；
- first WRITE；
- 每个 unnecessary WAIT；
- ambiguity-resolution boundary；
- final predicted flush。

这样模型能知道“哪一个 WAIT 导致了额外 640 ms”，而不是把整个句子的好坏平均归因到所有 WAIT/WRITE。

如果完整 event-level GRPO 改动过大，可以先使用两段式 advantage：

```text
A_trajectory：最终质量、coverage、结构
A_event：当前 action 的 premature/unnecessary/latency shaping

L = L_trajectory + alpha_event * L_event
```

### 17.13 优化方向八：KL、SFT replay 与探索强度

Reward v1 使用 `KL beta=0.02`、`SFT replay=0.2`。E3 与 E1 的 action 指标差异很小，可能同时受到 reward 信号弱和 reference anchor 偏强影响。

在 Reward v2 主体正确后，dev 上做以下小网格：

```text
KL beta          ∈ {0.005, 0.01, 0.02}
SFT replay weight∈ {0.05, 0.10, 0.20}
group size       = 8 优先；G4 只保留成本对照
sampling temp    ∈ {0.8, 1.0, 1.2}
```

必须记录：

- group reward std；
- unique action trajectories/group；
- action entropy；
- KL；
- always-WAIT/always-WRITE 比例。

如果 group 内轨迹几乎相同，GRPO advantage 接近零，单纯继续增加训练步数没有意义。可以使用温度 curriculum：早期略高温探索，后期降温稳定。

KL 最好改为 target-KL adaptive controller，而不是固定 beta：偏离过小时降低 beta，偏离过大或 safety gate 变差时提高 beta。

### 17.14 优化方向九：两级 multi-fidelity reward

每条 rollout 都运行 ASR、COMET、UTMOS 会过慢，因此采用两级 reward：

#### Fast reward：每 step

- pseudo/contextual eligibility；
- explicit latency delta；
- prefix teacher score；
- coverage、repetition、structure；
- predicted/forced final flush；
- KL 和 action constraints。

#### Full reward：每个 eval interval 的固定 dev subset

- free-running Text BLEU/chrF；
- Speech BLEU；
- First WRITE、StartOffset、ATD、LAAL；
- UTMOS/AutoPCP；
- forced recovery、under-translation、repetition；
- batch-one RTF/TTFT。

Fast reward 只负责训练吞吐，Full reward 负责 checkpoint 选择和校准。二者相关性必须记录；如果 fast reward 上升而 full dev Pareto 变差，应停止训练并修改 proxy。

### 17.15 优化方向十：先做 WRITE-logit bias sweep

在重新训练前，先对 E3 G8 checkpoint 在 dev 扫描 WRITE logit bias：

```text
logit(WRITE) = logit(WRITE) + b
b ∈ {0.00, 0.10, 0.20, 0.30, 0.50}
```

这是最低成本诊断，不是最终 reward 方案：

- 如果小正 bias 就能降低 First WRITE/ATD，且 BLEU/premature 仍通过，说明 E3 已学到可用 representation，只是 operating point 过保守；
- 如果 bias 一增加就出现 premature、漏译或结构错误，说明必须修改训练 reward/eligibility，而不能只调阈值；
- bias 只能在 dev 选择，不能根据已查看的 test 反复调整。

### 17.16 推荐 Reward v2 形式

建议把 reward 明确拆成可记录分量：

```text
R_v2 = lambda_latency   * R_latency_delta
     + lambda_commit    * R_safe_commit
     + lambda_quality   * R_quality
     + lambda_coverage  * R_coverage
     + lambda_structure * R_structure
     + lambda_audio     * R_audio_proxy
     - beta_kl          * KL

constraints:
  premature <= threshold
  BLEU retention per direction >= threshold
  predicted final flush >= threshold
  forced recovery <= threshold
```

首轮实现优先级：

1. `R_latency_delta`；
2. safe-boundary 后递增 WAIT penalty；
3. mutually-exclusive premature/unnecessary cost；
4. coverage、predicted/forced final flush；
5. dev free-running checkpoint selection；
6. 双语 quality constraint；
7. event-level credit assignment；
8. 更昂贵的 audio/full-quality reward。

不要第一轮同时加入十几个无法解释的分量。每新增一类 reward，都必须有单独 ablation 和 TensorBoard scalar。

### 17.17 下一轮最小可归因实验矩阵

保留 E1 和 E3 v1 作为冻结对照，不覆盖任何历史目录：

| ID | 训练 | 主要改动 | 要回答的问题 |
|---|---|---|---|
| R0 | 不训练 | E3 v1 WRITE-bias sweep | 当前 checkpoint 是否只是 operating point 过保守？ |
| R1 | G8 | 重平衡 premature/unnecessary + coverage/final-flush | 是否能减少 unnecessary WAIT 而不提高 premature？ |
| R2 | G8 | R1 + explicit First-WRITE/ATD/LAAL delta | 直接 latency reward 是否进入端到端指标？ |
| R3 | G8 | R2 + bilingual quality constraints + adaptive KL | 能否同时保持双向质量并降低延迟？ |

只有 R2 在 dev 明显改善后才运行 R3。G4 不作为首轮主实验，因为 E2 在质量与 latency 上整体弱于 E3；它只在需要验证 group-size 成本时补跑。

每个训练实验至少保存：

```text
reward component curves
group reward std / unique trajectories / entropy
predicted vs forced final flush
writes/sample / max consecutive WAIT
dev Text/Speech BLEU by direction
dev First WRITE / ATD / LAAL
Pareto checkpoints: best-quality, best-latency-under-gate, non-dominated
```

### 17.18 Reward v2 checkpoint 选择

禁止继续只用 action F1 标量选 best。每个 eval interval 在固定 dev subset 产生候选点：

```text
quality = 双向 Text/Speech BLEU retention
latency = First WRITE + ATD + LAAL
safety  = premature + forced recovery + final flush
coverage= writes/sample + under-translation
```

保存三类不可变 checkpoint：

1. `best_quality`；
2. `best_latency_under_quality_gate`；
3. `pareto_non_dominated`。

推荐主 checkpoint 的 gate：

```text
双向 Text BLEU 相对 E1 最差下降 <= 0.5
premature WRITE <= E1 + 0.01
predicted final flush >= 0.99
forced recovery <= 0.01
batch-one RTF < 1
```

在通过 gate 的 checkpoint 中最小化 First WRITE/ATD/LAAL。test 只评估 dev 冻结后的最多三个 Pareto checkpoints。

### 17.19 Reward v2 分阶段验收目标

#### 15-shard pilot gate

| Metric | 相对 E1 的最低目标 |
|---|---:|
| First WRITE | 降低 ≥5% |
| ATD/LAAL | 降低 ≥5% |
| Unnecessary WAIT | 相对降低 ≥20%，或绝对 ≤0.12 |
| Text BLEU | 每个方向下降 ≤0.5 |
| Premature WRITE | 增加 ≤0.01 |
| Predicted final flush | ≥0.99 |
| Forced recovery | ≤0.01 |
| Batch-one RTF | `<1` |

#### 扩大训练前的正式 gate

| Metric | 目标 |
|---|---:|
| First WRITE | 降低 ≥10%–15%，或 ≥500 ms |
| ATD/LAAL | 降低 ≥10% |
| Unnecessary WAIT | 进入 0.08–0.12 |
| 双向质量 | 均通过 retention gate |
| Paired bootstrap | latency 95% CI 不跨 0 |
| 随机种子 | 至少 3 个方向一致 |
| Fixed wait-k | 至少一个点位于 frontier 上或之外 |

### 17.20 当前明确结论

1. E3 G8 是 Reward v1 中最值得保留的起点，因为它优于 E2，且证明 group size 8 的探索能改善部分质量。
2. E3 不能称为 latency improvement：First WRITE、ATD、LAAL 和 unnecessary WAIT 均未超过 E1。
3. 当前失败的首要原因不是 GPU、RTF 或 group size，而是 reward 与 checkpoint selection 没有直接优化端到端 latency，且 premature/unnecessary cost 把策略推向保守。
4. 下一轮先做 E3 WRITE-bias dev sweep，再做 R1/R2；不要立即扩大 full198，也不要覆盖 E0–E3 v1。
5. 只有在 matched E1 上形成质量保持且 latency 更低的 Pareto 点，才能声称 GRPO 对 simultaneous S2ST 有独立贡献。
