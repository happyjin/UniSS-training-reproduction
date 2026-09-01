# 下一步训练计划 v5:先修目标函数,暂不扩全量

> 本文回答四个具体问题:下一步训什么、要不要扩全量 shard、载入哪个 checkpoint、
> 之前是不是从 phase1 蒸馏的。全部结论有实测支撑。

---

## 0. 一处必须更正

我此前说过"离线 phase3 的权重从未进入 student,它只是冻结 KL teacher"。**这是错的。**

`run_stage_a_formal_8gpu.sh:39`:

```bash
export RUN_LOAD="${PHASE3_NATIVE_ROOT}"   # = checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4
```

**真实血统链:**

```
离线 phase3  iter_0009075      (CVSS-T Speech-BLEU 32.20/24.28)
  └─ Stage-A 因果 ASR 训练 381 步 → V1 iter_0000381  (CER 21.01% / WER 35.34%)
       └─ e2e 交织训练 26+ 次 → iter_0002264 → 今天的 iter_0001132
```

**所以"从 phase3 重启"不是一条新路,而是"跳过 Stage-A 那一段"。**

### 之前是从 phase1 蒸馏的吗 —— 不是

蒸馏 teacher 从来不是 phase1。两处蒸馏:

| 阶段 | 蒸馏项 | 权重 | teacher |
|---|---|---:|---|
| Stage-A | `offline_teacher_kl` | 0.20 | 离线 phase3 |
| Stage-A | `phase3_replay` | **1.00** | 离线 phase3 的任务重放 |
| e2e | `v1_asr_kl` | 0.30 | 冻结的 Stage-A V1 |
| e2e | `phase3_kl` | 0.25 | 冻结的离线 phase3(top-32,温度 1.5) |
| e2e | `replay_ce` | 0.50 | 离线 phase3 任务重放 |

**phase1 只在 UniSS 论文的三阶段课程里出现(文本-语音对齐),本项目的流式血脉
从 phase3 起步,没有回到 phase1。**

---

## 1. 数据现状:15 shard 用了多少、全量有多少

| | 记录数 | 磁盘 | 说明 |
|---|---:|---:|---|
| **当前 e2e 源**(`simul_uniss_subsecond_v2/formal_15shard_v1`) | **1,325,243** | **411 GB** | 15 个 shard,含 `source_glm` / `target_bicodec` / `bicodec_global` / 对齐 |
| 当前 e2e 派生数据(轨迹/rollout/teacher cache/task pool) | — | **371 GB** | |
| **原始语料** `data/raw/UniST` | — | 29 GB | **208 个 parquet shard**(上限) |
| `simul_uniss_v3_full198` | 19,286,004 | 425 GB | **198 shard 已 prepare,但 schema 不同** |
| `phase3_unist198_sharded` | — | 151 GB | 198 shard,离线 phase3 的训练数据 |

### `simul_uniss_v3_full198` 能直接用吗 —— 不能

它的 `samples.jsonl` 字段是 `input_ids / token_weights / length`(**已 tokenize 成
另一种任务格式**),而 e2e 流水线需要的是
`source_glm / target_bicodec / bicodec_global / source_glm_end_ms / source_audio`。
而且它 manifest 里列的 `packed_interleaved.jsonl` **198 个 shard 全部没有落盘**
(实测 0/198)。

### 扩到全量的真实成本 —— 磁盘挡住了

| | 15 shard 实测 | 198 shard 外推 |
|---|---:|---:|
| Stage-A 源快照 | 411 GB | **≈ 5.4 TB** |
| e2e 派生(轨迹+rollout+teacher cache+task pool) | 371 GB | **≈ 4.9 TB** |
| **合计** | **782 GB** | **≈ 10.3 TB** |

**当前磁盘:28 TB 总量,已用 23 TB,可用 4.1 TB。装不下。**

要扩全量必须先做其中之一:清理旧实验(19 个 rollout 目录、4240 个候选)、
分批处理+即用即删、或加盘。

---

## 2. 该不该现在扩全量 —— **不该**

**因为瓶颈不是数据量,是目标函数。** 实测证据:

| 问题 | 数字 | 更多同类数据能修吗 |
|---|---|---|
| **eng→cmn 逐词直译不重排** | 译文 chrF **15.0** vs cmn→eng 53.3,而两方向 ASR chrF 几乎相同(78.6 vs 77.2) | **不能** —— 是监督形态问题,不是数据量问题 |
| **开口决策学不会** | 三次训练把推理 gap 从 −2.88 推到 −4.97,单调反向 | **不能** —— teacher-forced 结构性反效果 |
| 决策 token 被饿着 | boundary 占监督 token **32.8%**,只拿 **4.7%** 梯度 | 不能 |
| 英文流式 ASR | 0.280(已从 0.396 改善) | **能**,这一项确实吃数据 |

**四项里只有一项吃数据。** 在目标函数修好之前扩 13.9 倍数据,等于把同一个错误
放大 13.9 倍,而且每次迭代从 7 小时变成 4 天。

---

## 3. 下一步训练:B′ → C,都在 15 shard 上做

### Step 0(修正:先做单变量版本)

初稿直接跳到 B′(同时关掉两个 KL、commit、以及全部 margin),那是**同时改动
10 个权重**,失败了无法归因。**更好的科学是先做单变量:**

| | Step 0 = B(单变量) | Step 1 = B′(全 CE) |
|---|---|---|
| `boundary_eos` | **0.10 → 1.0** | 0.10 → 1.0 |
| `asr_ce`/`mt_ce`/`semantic_ce` | 1.0(不变) | 1.0(不变) |
| `replay_ce` | 0.50(**不变**) | → 1.0 |
| `v1_asr_kl` / `phase3_kl` | 0.30 / 0.25(**不变**) | **→ 0** |
| `commit_consistency` | 0.20(**不变**) | → 0 |
| 全部 margin / roll-in | **→ 0** | → 0 |
| 改动的权重数 | **8**(7 个 margin 归零 + boundary 提升) | 10 |
| 回答的问题 | "决策 token 被饿死是不是主因" | "phase3 原配方能不能行" |

**两个 KL 实测一直在下降(`phase3_kl` −0.181、`v1_asr_kl` −0.013),是有效的防遗忘。
先留着。** 若 Step 0 有效就不必再做 Step 1;若 Step 0 无效,再用 Step 1 检验
"是不是 KL 和 commit 在拖后腿"。

### Step 1(若 Step 0 不足,零代码,7 小时):B′ 均匀 CE

**动机:** UniSS 论文确认最好的模型用的是纯 next-token CE 无加权
(`ℒ_AR = −∑ log P_θ`,无辅助 loss)。而本项目把决策 token 的梯度压到 4.7%
(应为 32.8%),`boundary_eos = 0.10` **在全部实验脚本里从未被改过**。

**配置**(只改环境变量):

| 权重 | 值 |
|---|---|
| `asr_ce` / `mt_ce` / `semantic_ce` / **`boundary_eos`** / `replay_ce` | **全部 1.0** |
| `v1_asr_kl` / `phase3_kl` / `commit_consistency` | **0** |
| 全部 margin / roll-in / `content_end_*` / `semantic_end_*` | **0** |

结果就是**一个均匀 CE 覆盖全部监督 token** —— 字面意义的 phase3 配方。
防遗忘沿用 phase3 自己的方式:两个 `phase3_*_replay` 家族按权重 1.0 混在数据里,
不加 loss 项。

**父 checkpoint:`iter_0002264`**(不是今天的 iter_0001132)——
因为决策 gap 正是被测对象,−2.88 是全部历史测量的参照点,而 iter_0001132 已被
推到 −4.97,会污染读数。

**falsification:** `family_logit_probe` 测推理侧决策 gap。
* 从 −2.88 往 0 移动 → 方向对,继续
* 仍不动或反向 → **teacher-forced 无论怎么加权都修不了这个决策,margin/CE 这条路
  彻底关闭**,直接转 Step 2

顺带检验:eng→cmn 的 chrF 是否从 15.0 上升 —— 因为 `incremental_mt_event` 家族有
**34.3% 的监督 token 是 boundary**,均匀 CE 会把它解放出来。

### Step 2(1 天数据 + 7 小时训练):C 前缀到前缀重构

**动机:** eng→cmn 的 chrF 15.0 是最大阻塞,而它是**语序重排**问题 ——
CSSEL-P2P 论文点名的 *"limited context and cross-lingual reordering"*。
它的解法是 teacher 标注的 prefix-to-prefix + bounded waiting,
**而本项目的 `incremental_mt_event` 家族已经是这个形态**
(`_mt_request` 的 target 就是 `event.target_text_delta`,
`target_support_end_ms` 相对 `source_end_ms` 滞后中位 **−160 ms**,天生
anticipation-free)。

**三个 prompt,一个 CE,序列里不放任何 WAIT/WRITE:**

```
① TOKEN_TASK_STREAMING_ASR                  源 GLM 前缀 → ASR 增量 ⊕ END_CONTENT
② TOKEN_TASK_STREAMING_TEXT_TRANSLATION     已提交源前缀 → 安全译文增量 ⊕ END_CONTENT
③ TOKEN_TASK_STREAMING_TTS                  已提交译文增量 ⊕ lookahead L 个后续增量
                                            → 仅当前增量的 semantic ⊕ END_SEMANTIC
```

三个 task token **已在 `constants_uniss.py` 中定义**,phase3 词表 180,480
**已包含全部**(最大 180,406),可直接 `--load`。
①② 的 builder **已存在**,**只缺 ③ 流式 TTS**
(`TOKEN_TASK_STREAMING_TTS` 在 `task_samples.py` 里出现 **0 次**)。

lookahead L 来自 SpeakStream 的消融(最优 m=5, n=1,即 4 词 lookahead),
初值 L=2 个 event 增量,进门禁扫描。

**数据零新生成** —— `target_text_delta` / `target_semantic_delta` /
`target_support_end_ms` / `speaker_global` 全部已在轨迹里。

### Step 3(条件性):扩全量

**只有在 Step 1 或 Step 2 的门禁通过后才做。** 且必须先解决磁盘:

1. 清理:19 个历史 rollout 目录 + 4240 个候选 + 已废弃的 26 个 checkpoint 血脉
2. 分批:一次处理 32 个 shard,build → train → 删中间产物 → 下一批
3. 或加盘(需 ≥ 10.3 TB)

**优先扩的是英文侧** —— 英文流式 ASR 0.280 是唯一确实吃数据的一项,
而 S3 的门线是 ≤ 0.20。

---

## 4. 载入哪个 checkpoint

| 用途 | checkpoint | 理由 |
|---|---|---|
| **Step 1(B′)** | **`iter_0002264`** | 决策 gap −2.88 是全部历史测量的参照点 |
| Step 2(C) | 视 Step 1 结果:通过则用 Step 1 产出;否则 `iter_0002264` | |
| **不要用** | 今天的 `iter_0001132` | 决策 gap 已被推到 −4.97,污染读数 |
| **不要用** | 离线 phase3 `iter_0009075` 直接起 | 它在增量协议上 BLEU 只有 3.92–9.97,不能做增量翻译;而且现有血脉本来就是它的后代 |

**保留今天的 `iter_0001132` 作为部署候选** —— 它在 δ=5 + rp=1.1 下拿到 6/6,
是目前唯一可交付的配置。

---

## 5. 排除的做法

* **不扩全量再训一遍**(磁盘不够,且瓶颈不是数据量)
* **不用 `simul_uniss_v3_full198`**(schema 不同,packed 数据 0/198 未落盘)
* **不再为 WAIT/WRITE 写 margin loss**(三次单调反向 −2.88 → −3.75 → −4.97)
* **不单独训练 `content_end_margin`**(单用会把长度比压到 0.324)
* **不回 phase1**(蒸馏链从来不经过 phase1)
* **不在 8 条样本上宣布成功**(全绿也要先扩到 64 条复核)

---

## 6. 时间与成本

| 步骤 | 人力 | GPU | 磁盘 |
|---|---|---|---|
| Step 1 B′ | **0**(改环境变量) | 7 h | 0 |
| Step 2 C 的 ③ builder + 打包 | 半天 | 2 h | +40 GB |
| Step 2 C 训练 | 0 | 7 h | 0 |
| Step 3 扩全量(条件性) | 2 天(含清理) | 3–5 天 | +10.3 TB |

**建议今晚就跑 Step 1** —— 零代码、零磁盘,而且是"能不能照离线最好的方式训流式"
这个问题的直接答案。
