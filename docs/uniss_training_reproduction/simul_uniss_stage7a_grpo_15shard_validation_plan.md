# Simul-UniSS Stage7A：15-shard Action-only GRPO 快速验证计划

> 文档状态：实验设计，不代表 full-Qwen GRPO 已实现或已经开始训练
>
> 生成日期：2026-07-27
>
> 适用仓库：`/opt/dlami/nvme/jasonleeeli/projects/UniSS`
>
> 目标：用最小、可归因、可复现的实验判断 GRPO 是否改善 Stage6 的 simultaneous quality-latency Pareto frontier
> 资源：单机 8 × NVIDIA H200，计划并行运行 4 个实验，每个实验固定 2 GPU

## 1. 结论先行

可以使用 8 张 H200 同时运行四个 2-GPU 实验，但需要满足两个前提：

1. 先实现真正从 Stage6 Qwen/Hugging Face checkpoint 初始化的 action-only LoRA/adapter GRPO；当前 `training/simul_uniss/policy_grpo.py` 只是 5 维输入的小型 MLP bootstrap，不能作为正式 Stage7A。
2. 正式计时前临时停止 GPU0 上的公网 Phase3 demo。demo 只占约 5.8 GiB，但并发训练会争用 GPU0 计算资源，使 baseline 延迟和 RTF 不可比较，也会让网页推理变慢。

四个最重要的首轮实验为：

| 实验 | GPU | 训练/评估 | 核心问题 |
|---|---|---|---|
| E0：Stage6 + learned policy/fixed wait-k baseline | 0,1 | 只评估 | 当前质量–延迟基线和 wait-k Pareto frontier 在哪里？ |
| E1：Stage6 + continued action SFT | 2,3 | 训练后评估 | 改善是否仅来自多训练、LoRA 或额外数据？ |
| E2：Stage6 + GRPO，group size 4 | 4,5 | 训练后评估 | 小 group GRPO 是否已能减少不必要 WAIT？ |
| E3：Stage6 + GRPO，group size 8 | 6,7 | 训练后评估 | 更大的组内探索是否带来更稳定的 Pareto 改善？ |

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
