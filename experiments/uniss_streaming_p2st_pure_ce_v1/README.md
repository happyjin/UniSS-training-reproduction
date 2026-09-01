# 纯 CE 流式 S2ST:从离线 phase3 出发,把时机移出模型

本实验**不修改** `uniss_phase3_v4_e2e_simuls2st_pilot15_v1` 及任何既有实验的
文件。它是一条新主干,与之前两周的思路在三个层面上不同。

---

## 1. 与之前两周的三处根本不同

| | 之前 26+ 次实验 | 本实验 |
|---|---|---|
| **起点权重** | Stage-A V1 `iter_0000381` —— 设计文档 §27.2 定位为"仅用于生成 Stage B noisy source prefix"的流式 ASR 组件,中文 CER 21.01% / 英文 WER 35.34%,**权重里没有翻译、没有 TTS** | 离线 phase3 `iter_0009075` —— **翻译与 TTS 已在权重里**,CVSS-T Speech-BLEU 32.20/24.28 |
| **模型要预测什么** | 序列里含 `WAIT_READ` / `WRITE_GENERATE` / `TASK_*` 动作 token,**模型必须决定何时开口** | **序列里没有任何动作 token。** 模型只回答"给定这个前缀,该输出什么" |
| **目标函数** | 14 个加权目标(3 个 CE + 2 个 KL + replay + commit + 7 个 margin/roll-in) | **1 个均匀 next-token CE** |
| 时机在哪 | 模型权重里(三次尝试全部反向) | **数据里(安全前缀)+ 推理调度器里(固定 chunk + 已提交前缀)** |

**为什么之前必须用 14 个目标**:student 从一个 ASR 组件出发,要同时学翻译、学
TTS、学流式三件事,所以需要 `phase3_kl`(0.25)和 `replay_ce`(0.50)去"进口"
本该作为初始权重存在的能力。从 phase3 出发后,**只剩流式一件事要学**。

**为什么动作 token 必须移除** —— 三次独立实验的推理侧决策 logit 差:

| checkpoint | 训练侧 gold 行位移 | 推理侧 gap |
|---|---:|---:|
| `iter_0002264` 基线 | — | −2.88 |
| `speak_decision`(margin 1.0) | +0.105 | −3.75 |
| `continue_end`(margin 3.0) | +0.40 | **−4.97** |

单调反向。而 [REINA](https://arxiv.org/html/2604.09916v1) 把这个失败模式命名为
**read loop**,归因于 **temporal drift**(*"the policy, lacking an internal
clock, fails to increase its emission probability as audio duration grows"*)。
本项目实测开口率对已听时长的斜率 **−0.012/秒** —— 独立证实。

---

## 2. 文献依据:三篇论文各管一块

| 论文 | 本实验采用的部分 |
|---|---|
| [SimulS2S-LLM](https://arxiv.org/abs/2504.15509) | **战略**:从离线训练好的 speech LLM 出发,流式性由推理侧策略提供。论文把 boundary-aware prompt 称为 *"the key to unlocking simultaneous inference for offline-trained speech LLMs"* |
| [CSSEL-P2P](https://arxiv.org/abs/2607.13158) | **增量 MT 的数据配方**:teacher 标注的 prefix-to-prefix 目标 + **bounded waiting**;推理用固定 chunk + rewind 已提交前缀。明确宣称无需架构改动,纯 SFT |
| [SpeakStream](https://arxiv.org/html/2505.19206) | **流式 TTS 的配方**:交织数据上标准 LM 训练,**loss 只算在语音 token 上**;何时切换由外部规则决定 |
| [UniSS](https://arxiv.org/abs/2509.21144) | **loss 与 prompt 形态**:`ℒ_AR = −∑ log P_θ(τ_out,t | P, τ_out,<t)`,纯 next-token CE,无辅助 loss;CoT prompt `listen → translate → speak` |

**不移植它们的代码。** StreamSpeech 是 fairseq 非 LLM 架构;另三篇的框架与本项目
(Megatron + 三 tokenizer 栈)不同,移植代码是净负收益。**移植的是配方。**
唯一值得直接用的外部代码是 [Simulstream](https://arxiv.org/html/2512.17648)
(IWSLT 2026 同传赛道官方评测工具),用于将来对外可比的评测接口。

---

## 3. Loss:一个均匀 next-token CE

```
ℒ = −∑_t  loss_mask_t · log P_θ(y_t | prompt, y_<t)
```

* `loss_mask = 1` 在三个任务的**全部 target token** 上(含 `END_CONTENT` /
  `END_SEMANTIC` / `EOS`),`0` 在 prompt token 上。
* **没有** boundary 权重、没有 margin、没有 roll-in、没有 KL、没有 policy head。
* 防遗忘沿用 phase3 自己的方式:**靠数据混合**,把两个 `phase3_*_replay` 家族
  按权重 1.0 混进来,不加任何 loss 项。

这与 phase3 训练时的 `megatron_uniss_dataset` 接口完全一致(只输出
`tokens / labels / loss_mask / position_ids / cu_seqlens`,`segment_weight`
出现 0 次)。

---

## 4. 三个任务(prompt 形态)

三个 task token **已在 `constants_uniss.py` 中定义**,且 phase3 词表
(180,480)**已包含全部**(最大 180,406),因此可直接 `--load` phase3,
**无需词表嫁接**。

```
① 流式 ASR   TOKEN_TASK_STREAMING_ASR
   prompt : c_task ⊕ c_lang_src ⊕ [START_GLM  源 GLM 前缀  END_GLM] ⊕ START_CONTENT
   target : ASR 文本增量 ⊕ END_CONTENT

② 增量 MT    TOKEN_TASK_STREAMING_TEXT_TRANSLATION
   prompt : c_task ⊕ c_lang_tgt ⊕ 已提交源文本前缀 ⊕ START_CONTENT
   target : **安全译文增量** ⊕ END_CONTENT

③ 流式 TTS   TOKEN_TASK_STREAMING_TTS
   prompt : c_task ⊕ c_lang_tgt ⊕ c_speed ⊕ 说话人 global ⊕ 已提交译文增量
            ⊕ START_SEMANTIC
   target : 该增量对应的 semantic token ⊕ END_SEMANTIC
```

**"安全译文增量"是全部关键。** 它只包含源端支撑已经到达的目标词 ——
即 CSSEL-P2P 的 bounded waiting。**这个信号已经在数据里**:
`TrajectoryEvent.target_support_end_ms`。实测(40 条轨迹 674 个事件):

* `target_support_end_ms` 有值:252/674(37%)
* `target_text_delta` 非空:252/674 —— **与上者完全重合**
* `target_semantic_delta` 非空:252/674 —— **同样完全重合**
* support 相对 `source_end_ms` 的滞后:**中位 −160 ms**(p10 −160,p90 −80)

**滞后为负意味着:这些目标增量的源端支撑在事件结束之前就已到达 ——
它们天生 anticipation-free,不需要任何重新对齐。**

---

## 5. 数据:零新生成

| 任务需要的 | 已有字段 | 状态 |
|---|---|---|
| ① 源 GLM 前缀 / ASR 增量 | `source_glm_delta`, `gold_source_delta`, `gold_source_prefix` | 已有 |
| ② 已提交源前缀 | `gold_source_prefix`(teacher-forced)/ `v1_source_prefix`(模型自己的 ASR) | 已有,**两种都有** |
| ② 安全译文增量 | `target_text_delta` + `target_support_end_ms` | 已有 |
| ③ 译文增量↔语音 token 对齐 | `target_text_delta` + `target_semantic_delta` | **逐 event 已对齐** |
| 说话人音色 | `E2ETrajectory.speaker_global` | 已有 |
| 语速 | phase3 的 `speed_token_id` | 已有 |
| 对齐质量过滤 | `alignment_confidence`, `noise_severity` | 已有 |

**结论:不需要新的对齐、不需要新的 teacher 推理、不需要新的音频处理。**
只需要新写 builder,从现有 `train_gold_trajectories.jsonl`(23.3 GB)
重新打包成三个家族。

已存在的两个 builder(`build_streaming_asr_task` /
`build_incremental_mt_tasks`)形态上已经是"不同 prompt + 纯 CE",但:

* 它们产出的 `loss_kinds` 里 boundary 占比极高(`streaming_asr_event` **74.7%**、
  `incremental_mt_event` **34.3%**),在旧目标函数里被权重 0.10 饿着;
  本实验均匀 CE 后它们自动被解放。
* **`build_streaming_tts_task` 不存在** —— `TOKEN_TASK_STREAMING_TTS` 在
  `task_samples.py` 里出现 **0 次**。**这是唯一真正缺失的一块。**

---

## 6. 推理:时机由调度器提供

```
每 320 ms 音频块到达:
  1. WhisperVQ 因果前端 → 新的源 GLM token(12.5 Hz)
  2. 任务 ① :在累积源前缀上生成 ASR 增量
  3. StablePrefixCommitter(local agreement, holdback 2)提交稳定的 ASR 前缀
  4. 任务 ② :在**已提交**的 ASR 前缀上生成译文增量
  5. StablePrefixCommitter 提交稳定的译文前缀
  6. 若有新提交的译文增量 → 任务 ③ :生成该增量的 semantic token
  7. 配速器(50 tok/s,margin 1200 ms)按源时间轴放行语音
```

**模型在任何一步都不决定"要不要输出"。** 它只回答"给定这个前缀,输出什么"。
何时输出由 chunk 时钟决定;输出多少由 local agreement 的稳定性决定;
说多快由配速器决定。

`StablePrefixCommitter`、`PacedInterleavedSession`、`allowed_event_tokens`
**本项目已经写好并测试过**,直接复用。

---

## 7. 门禁(与既有测量同协议,可直接对照)

frozen fixed-16 selection,与 `stage2_paced_m1200_iter0002264` 逐字节同一份
`SELECTION.json`。

| 判据 | `iter_2264` 基线 | 门线 |
|---|---:|---:|
| 会话文本覆盖 | 0.30 | ≥ 0.50 |
| 语义覆盖 | 0.666 | ≥ 0.666 |
| 文本长度比 中位 | 1.033 | ∈ [0.9, 1.2] |
| 可听起始 | 1214 ms | ≤ 1500 ms |
| 英文流式 ASR 错误率 | 0.303 | ≤ 0.20 |
| free-src BLEU eng→cmn | 4.99 | ≥ 17.51(本血脉当前最好) |
| free-src BLEU cmn→eng | 16.73 | ≥ 16.73 |

**注意没有 `WRITE_MT/event` 与 `natural_eos` 这两项** —— 本方案里它们不存在:
没有动作 token,也没有会话级 EOS 决策(每个任务各自以 `END_CONTENT` /
`END_SEMANTIC` 结束)。这是设计变化,不是回避门禁。

---

## 8. 文件清单

| 文件 | 状态 |
|---|---|
| `data/builders.py` — 三个任务的 builder(① ② 改写自既有,③ 新写) | 待写 |
| `data/build_pool.py` — 从 trajectories 打包成三个家族 | 待写 |
| `training/pretrain_pure_ce_megatron.py` — 均匀 CE 训练入口 | 待写 |
| `evaluation/run_cascade.py` — 三任务级联 + committer + 配速器 | 待写 |
| `scripts/run_8gpu.sh` — 从 phase3 `iter_0009075` 起训 | 待写 |
| `tests/` — builder 对齐不变量、loss_mask 覆盖、级联语法 | 待写 |

## 9. 不做的事

* 不修改任何既有实验目录的文件。
* 不为 WAIT/WRITE 写任何 loss(三次单调反向)。
* 不移植三篇论文的代码(框架不同,净负收益);只移植配方。
* 不在 8 条 train-seen 样本上宣布成功;全绿必须先扩到 64 条复核。
