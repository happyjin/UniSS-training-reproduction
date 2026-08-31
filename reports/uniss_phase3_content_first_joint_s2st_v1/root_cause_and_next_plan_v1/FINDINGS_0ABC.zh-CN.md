# 0 阶段证伪实验结论(0-A / 0-C / 0-B)

- 执行时间:2026-08-31
- 目标任务:**streaming / simultaneous speech-to-speech translation**(中↔英双向,边听边说)
- 实验代码:[experiments/uniss_phase3_content_first_diagnostics_v1](../../../experiments/uniss_phase3_content_first_diagnostics_v1)
- 原始数据:同目录 `BRIDGE_PARITY.json`、`TEACHER_FORCED_CEILING.json`、`prior_lineage_reeval/METRICS.json`
- 声明范围:全部 **train-seen**,不含泛化结论
- 隔离性:未修改任何既有实验、脚本或 checkpoint。0-C 通过在本实验内临时重绑定 `_nearest_codes` 做代码源消融,session 结束即还原;0-B 直接调用旧血脉自己的 rollout 脚本

---

## 0. 三句话结论

1. **0-A:推理时喂给模型的离散语音码,和它训练时用的码只有 14.5% 相同。** 这不是桥的算术问题(桥与前端 token_ids 一致率 1.000),而是前端本身:content-first 运行时加载的是一个**未经训练的**块因果 WhisperVQ,而所有旧血脉加载的是**训练过的 Stage-A 因果前端**。
2. **0-C:把训练时的码原位喂回去,ASR 从 0.183 涨到 0.294、MT 从 0.101 涨到 0.127、译文长度比从 0.304 涨到 0.400 —— 但 coverage 仍然只有 7.9%。** 所以输入不匹配是真实且昂贵的,但**完整性是能力缺口**,靠修推理链路修不好。两个原因独立并存。
3. **0-B:两条血脉本来就是同一套评测器、同一批 64 条 episode、同一几何、同一采样温度。** 差距不是评测口径造成的:旧血脉 ASR 0.261 / MT 0.178 / coverage 17.1% / 首发声 4.45 s,content-first Round 2 是 0.048 / 0.030 / 3.2% / 19.63 s。

**另外一个此前完全没被发现的问题(0-C 副产物):同一个 checkpoint 在单条 3–17 秒音频上表现远好于 60–70 秒长 episode。** 见 §3。

---

## 1. 0-A:推理码一致性

对 64 条 episode 里前 8 个互不相同的 component(3.1–17.0 s 真实音频),同时计算三路离散码:

| 路径 | 含义 |
|---|---|
| `offline` | 非因果 GLM4 tokenizer(`extract_speech_token`),**即训练数据里 `source_glm` 的生成器** |
| `content_first_bridge` | 未训练的预训练 WhisperVQ 按 160 ms 块因果推进 + float32 L2 argmin,**即 `load_content_first_models` 实际构造的东西** |
| `stage_a` | 训练过的 Stage-A 因果前端 `stage_a_formal8_20260816T224100Z/iter_0000381`,**即所有旧血脉 `strict_cascade._load_models` 加载的东西** |

### 结果

| 指标 | 值 |
|---|---:|
| `offline` vs `content_first_bridge` 逐帧一致率(均值) | **0.1446** |
| 同上(最小) | **0.0649** |
| 前端自身 `token_ids` vs 桥的 argmin 一致率 | **1.0000** |
| 序列长度完全相等的比例 | **1.0000** |
| 选中码向量的余弦相似度(均值) | 0.9632 |
| `offline` vs `stage_a` 一致率 | 0.0016 |
| `offline` vs `stage_a` 余弦相似度 | 0.7789 |
| `content_first_bridge` vs `stage_a` 一致率 | 0.0038 |
| 编码器位置重置次数 | 全部 0 |

### 读法

- **长度 100% 对齐、余弦 0.963,但逐帧只有 14.5% 相同**:隐状态方向接近,但 argmin 在密集码本里绝大多数时候落到了**不同的邻居**。模型看到的是一串陌生的离散 token。
- **桥的算术没有问题**(一致率 1.000),`_nearest_codes` 精确复现了前端自己的量化,平方 L2 argmin 与参考实现 `uniss/speech_tokenizer/glm4/modeling_whisper.py:68-88` 度量一致。
- **旧血脉的 Stage-A 码与 offline 只有 0.16% 一致,却能work** —— 因为 Stage-A 的 `bridge_norm` / `bridge_projection` 是**和这个因果前端一起、在真实波形上训练出来的**(`StageAObjective.prepare` 直接调 `self.frontend(waveform)`)。残差投影的职责正是把因果码校正回 Phase-3 能用的表示。
- **content-first 的 `frontend_adapter` / `frontend_projection` 从头到尾只见过 offline 码的嵌入**(训练消费预计算的 `frontend_ids`,前端根本不参与训练),推理时却被喂进因果码。它没有任何机会学会这个校正。

> 结论:content-first 的推理链路把一个从未在训练中出现过的输入分布喂给了一个专门为另一个分布训练的残差适配器。

---

## 2. 0-C:能力上限与代码源消融

用**未修改的** `evaluate_event_policy_session` 跑真实级联,只消融一件事:离散码从哪来。为保证替换严格逐位对齐(0-A 已验证长度 8/8 完全相等),session 用单个 component 而非长 episode,因此无需处理拼接静音。

替换忠实性(每条都核对):`consumed == gold_codes`、每块 `memoized_repeats == 1`(嵌入与残差两个消费者拿到同一串码、游标只前进一次)、`exhausted == 0`。

### 汇总(8 条 component 均值)

| 指标 | `causal`(现网推理) | `gold_offline`(训练时输入) | 变化 |
|---|---:|---:|---|
| ASR teacher similarity | 0.1828 | **0.2940** | **+60.8%** |
| MT teacher similarity | 0.1013 | **0.1268** | **+25.2%** |
| translation_length_ratio | 0.3044 | **0.3996** | **+31.3%** |
| target_coverage | 0.0923 | 0.0787 | −14.7% |
| spoken_target_coverage | 0.0923 | 0.0787 | −14.7% |
| first_write | 5478 ms | 5760 ms | +5.2% |
| max internal silence | 950 ms | 738 ms | −22.3% |
| audio_writes | 1.375 | 1.500 | +9.1% |
| premature_end | 0.0 | 0.0 | — |

### 逐条

| sample | 时长 | causal ASR | gold ASR | causal MT | gold MT | causal cov | gold cov | causal len | gold len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emilia_zh_0006544609 | 5.74 s | 0.000 | **0.452** | 0.026 | 0.024 | 0.000 | 0.000 | 0.062 | 0.062 |
| emilia_zh_0006616205 | 16.96 s | 0.064 | 0.064 | 0.045 | 0.027 | 0.064 | 0.021 | 0.064 | 0.128 |
| emilia_zh_0006974291 | 9.32 s | 0.170 | **0.213** | 0.056 | **0.131** | 0.080 | **0.120** | 0.200 | **0.480** |
| emilia_zh_0006787038 | 3.12 s | 0.048 | **0.190** | 0.017 | **0.122** | 0.000 | **0.077** | 0.154 | **0.538** |
| emilia_zh_0004032736 | 3.98 s | 0.000 | 0.059 | 0.017 | 0.004 | 0.000 | 0.000 | 0.077 | 0.077 |
| emilia_zh_0004958827 | 4.28 s | **0.680** | 0.640 | **0.473** | 0.391 | **0.417** | 0.167 | 0.833 | **1.000** |
| emilia_zh_0005086952 | 6.10 s | 0.310 | **0.448** | 0.147 | **0.261** | 0.067 | **0.133** | **0.933** | 0.467 |
| emilia_zh_0006068322 | 3.30 s | 0.190 | **0.286** | 0.028 | **0.054** | 0.111 | 0.111 | 0.111 | **0.444** |

### 读法

- **ASR 侧,输入不匹配的代价被明确量化**:6/8 条在换成训练输入后 ASR 上升,其中一条从 0.000 直接到 0.452。这与 0-A 的 14.5% 一致率完全吻合。
- **但 coverage 在两个 arm 下都只有 8–9%**。给它完全正确的输入,它依然译不完整。判定阈值(gold coverage < 0.10)因此落在 `capability_gap_dominates_data_and_budget_must_change`。
- **coverage 在 gold arm 反而略降**是采样噪声:8 条样本、单次采样、温度 0.70/0.90/0.80,并且 `emilia_zh_0004958827` 一条(causal 0.417 → gold 0.167)就足以翻转均值。译文长度比同时从 0.304 涨到 0.400,方向与 ASR/MT 一致。这一项需要更多样本才能定量,不作为结论。
- **两个原因是独立并存的**,不是二选一:修推理链路能把 ASR 拉回来一大截,但完整性必须靠重做数据构造和训练预算。

---

## 3. 新发现:长 session 崩塌

0-C 的副产物。**同一个 checkpoint、同一套推理代码、同一批音频素材**,只是 session 长度不同:

| 指标 | 单 component(3.1–17.0 s) | 10-component episode(60–70 s) | 恶化 |
|---|---:|---:|---:|
| ASR teacher similarity | 0.1828 | 0.0479 | **3.8×** |
| MT teacher similarity | 0.1013 | 0.0300 | **3.4×** |
| translation_length_ratio | 0.3044 | 0.0740 | **4.1×** |
| target_coverage | 0.0923 | 0.0324 | **2.8×** |
| first_write | 5,478 ms | 19,630 ms | **3.6×** |
| **max internal silence** | **950 ms** | **27,415 ms** | **28.9×** |
| audio_writes / 秒源音频 | 0.243 | 0.068 | 3.6× |
| premature_end | 0.0 | 5.28 | — |

`max internal silence` 从不到 1 秒变成 27 秒,`premature_end` 从 0 变成 5.28,这不是"能力差一点",而是**长 session 上的状态崩塌**。可能的机制(需要进一步定位):

1. **acoustic rollover**:`--acoustic-rollover-ms 24000`,60–70 s 的 episode 会触发 2–3 次声学提示重建。级联源码里明确写着重建后"因果前端与已提交文本保留,但 LLM 声学提示被重置"。
2. **encoder position reset**:0-A 在单 component 上测得重置 0 次;长 episode 一定会超过 WhisperVQ 位置表并触发 `encoder_resets`。
3. **误差累积 + EOS 过预测**:训练日志的 `runtime_predicted_eos_fraction` 0.0168 vs target 0.0078(2.15 倍过预测)在长序列上会被反复触发,对应 `premature_end` 5.28。
4. **component 拼接边界**:episode 由 10 段独立音频 + 160 ms 间隔拼成,`boundary_masked` 事件占相当比例,模型可能在边界处误判 END。

**这条必须单独定位。** 在此之前,任何在 64 条长 episode 上做的训练/RL 都会同时受"单句能力不足"和"长 session 崩塌"两个因素污染,无法归因。

---

## 4. 0-B:两条血脉的同评测器对照

旧血脉与 content-first 本来就共享:

| 共享项 | 值 |
|---|---|
| episode 协议 | `data/processed/uniss_phasea_event_constrained_grpo_long_v2/protocol64_v1/episodes.jsonl` |
| 几何 | 64 episodes × 4 candidates = 256 |
| decision chunk | 320 ms |
| acoustic rollover | 24,000 ms |
| 采样温度 | policy 0.70 / top-p 0.90 / action 0.80 |
| merged rollout schema | `uniss_event_constrained_rollout_merged_v2` |
| reward / coverage / 评测代码 | `uniss_phasea_coverage_constrained_grpo_v3`(同一份) |

唯一的差异只有两处:**base HF**(旧:`uniss_stage_a_formal8_iter_0000381_hf`;新:`qwen0p5b_phase3_unist198_iter_0009075_hf`)和**前端/objective**(旧:训练过的 Stage-A;新:未训练因果前端 + content-first export)。

因此差距不可能是评测口径造成的。用**同一次** `write_report` 调用把六个 arm 一起打分:

| arm | reward | ASR | MT | 长度比 | coverage | spoken cov | 首发声 | 最大空白 | audio writes | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prior_stateful_archived | 2.505 | 0.300 | 0.190 | 0.363 | n/a | n/a | 21,945 | 31,075 | 9.85 | 4.63 |
| **prior_post_round3_archived** | −5.554 | **0.261** | **0.178** | **0.339** | **0.171** | **0.170** | **4,445** | 18,895 | **18.14** | 11.95 |
| prior_final_round3_archived | −5.851 | 0.261 | 0.178 | 0.340 | 0.171 | 0.169 | 4,500 | 18,453 | 18.57 | 11.84 |
| content_first_pre_grpo | −10.482 | 0.048 | 0.026 | 0.068 | 0.031 | 0.030 | 21,194 | 28,465 | 4.14 | 7.80 |
| content_first_round1 | −10.376 | 0.048 | 0.029 | 0.072 | 0.031 | 0.031 | 19,156 | 28,388 | 4.40 | 7.80 |
| content_first_round2 | −10.564 | 0.048 | 0.030 | 0.074 | 0.032 | 0.032 | 19,630 | 27,415 | 4.61 | 7.85 |

产物:[prior_lineage_reeval/METRICS.json](prior_lineage_reeval/METRICS.json)、[prior_lineage_reeval/REPORT.zh-CN.md](prior_lineage_reeval/REPORT.zh-CN.md)

第二阶段(`REEVAL=1`)用今天的代码在 8 卡上重跑旧血脉最优 adapter
(`uniss_phasea_event_constrained_grpo_long_v2/event_grpo_round3_g4_w64_formal_v1/iter_0000142`),
在"可比"之外再加一层"可复现"。结果见 `prior_lineage_reeval_stage2/`。

---

## 5. 对下一版方案的修正

原 [ANALYSIS.zh-CN.md](ANALYSIS.zh-CN.md) 的诊断成立,但 0 阶段实验改变了**优先级**,并新增一条必修项。

### 5.1 起点结论(可以直接执行)

**下一版必须从旧血脉出发,而不是从 offline Phase-3 v4 出发。** 具体是:

```
base HF     : checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf
前端/objective: checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/
                stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381
adapter     : checkpoints/uniss_phasea_event_constrained_grpo_long_v2/
                event_grpo_round3_g4_w64_formal_v1/iter_0000142
```

理由不只是"它指标更好",而是 0-A 给出的机制:**只有这一套里的残差桥是和因果前端一起训练出来的。**

### 5.2 新增必修项:训练必须让因果前端进环

这是 0-A 直接推出的结论,原方案没有覆盖到这个层面。

| 方案 | 做法 | 代价 | 说明 |
|---|---|---|---|
| **A(推荐)** | 训练时不再消费预计算的 offline `frontend_ids`,而是**在线跑 Stage-A 因果前端**产出码 | 前端前向开销;需要在 dataloader 里带上波形 | 一次性消除训练/推理分布差,和 Stage-A 的做法一致 |
| B | 保留离线预计算,但**用因果前端重新生成一份 `causal_frontend_ids`** 替换 `source_glm` | 一次性重跑 15-shard 的码提取 | 便宜得多,不改训练图;代价是前端从此冻结 |
| C | 训练 offline 码,推理时也用 offline 码 | 不可行 | offline 码需要整句非因果注意力,与 streaming 目标矛盾 |

**建议先做 B**(几小时的数据重生成),因为它能在不动训练图的前提下把 0-A 这个变量彻底消掉;若 B 之后 ASR 仍不达标,再上 A。

**无论 A 还是 B,都必须加一个断言:训练消费的码流与推理产生的码流,在同一条音频上一致率 ≥ 0.99。** 0-A 的脚本可以直接当这个门禁用。

### 5.3 优先级重排

| 优先级 | 项目 | 依据 | 原方案位置 |
|---|---|---|---|
| **P0** | 换回旧血脉起点 | 0-B + 0-A | §4.3(A),现已从"若 0-B 证实"变为确定 |
| **P0** | 消除训练/推理码流不匹配(方案 B) | 0-A(14.5%)+ 0-C(ASR +61%) | **新增,原方案未覆盖** |
| **P0** | 数据构造:目标文本 token 占比 1% → ≥25% | 训练日志 435,315 / 40,057,425 | §4.2 |
| **P0** | 语音 token 头(准确率 1.5%)重做监督 | 训练日志 | §4.2 |
| **P1** | 定位长 session 崩塌(空白 0.95 s → 27.4 s) | **0-C 新发现** | **新增** |
| **P1** | 补上三项恒零 loss + 零 loss 断言 | 训练日志 | §4.2 |
| **P1** | EOS 过预测 2.15× | 训练日志 | §4.2 |
| **P1** | 训练量 717 → ≥4000 步 + 自由运行在环 | §2.1(E) | §4.3(B)(C) |
| **P2** | reward 五处死梯度修复 | §2.2 | §4.4 |
| **P2** | 阶段门禁 | §2.3(B) | §4.5 |

reward 修复被降到 P2 **不是因为它不重要**,而是因为在 coverage 3% 的工作点上它无论怎么改都拿不到有效梯度;它必须在内容达标之后才有意义。

### 5.4 立刻可加的两个门禁

1. **码流一致性门禁**:`run_bridge_parity.sh` 的 `offline_vs_bridge_agreement_mean ≥ 0.99`,不达标不允许开训。
2. **长度尺度门禁**:任何 checkpoint 的自由运行评估必须**同时**报单 component 和长 episode 两组数字。本轮如果一开始就这样报,长 session 崩塌会在 8 月 30 日就暴露,而不是拖到今天。

---

## 6. 复现命令

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS

# 单元测试(CPU)
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python -m pytest -q \
  experiments/uniss_phase3_content_first_diagnostics_v1/tests

# 0-A:约 3 分钟,单卡
COMPONENTS=8 GPU=0 bash experiments/uniss_phase3_content_first_diagnostics_v1/scripts/run_bridge_parity.sh

# 0-C:约 18 分钟,单卡
COMPONENTS=8 GPU=0 bash experiments/uniss_phase3_content_first_diagnostics_v1/scripts/run_teacher_forced_ceiling.sh

# 0-B 第一阶段:秒级,无 GPU
bash experiments/uniss_phase3_content_first_diagnostics_v1/scripts/run_prior_lineage_reeval_8gpu.sh

# 0-B 第二阶段:约 1.2 小时,8 卡
REEVAL=1 OUTPUT_DIR=<新目录> \
  bash experiments/uniss_phase3_content_first_diagnostics_v1/scripts/run_prior_lineage_reeval_8gpu.sh
```

GPU 占用程序:`tmux kill-session -t uniss_gpu_load_60` 停止,
`bash experiments/uniss_phase3_content_first_joint_s2st_v1/scripts/start_gpu_holder.sh` 恢复。
