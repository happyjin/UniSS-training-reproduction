# Streaming / Simultaneous S2ST 执行计划 v2 —— 自底向上、每层设硬门禁

- 时间:2026-08-31
- 取代:`/root/.claude/plans/streaming-simultaneous-speech-to-speech-synthetic-mccarthy.md`(v1,其中两条建议已被实测推翻)
- 证据来源:本目录同级的 `streaming_s2st_metrics_v1/`、`streaming_s2st_demos_v1/`、`reports/uniss_phase3_content_first_joint_s2st_v1/root_cause_and_next_plan_v1/`
- 声明范围:全部数字为 train-seen 的 fixed-16 selection

---

> **2026-08-31 更正 —— 本文的 Phase 1 已作废。**
> 我当时把 Stage-A v9 的 `FORMAL_GATE.json`(`passed: true`)与 v1 的 `GATE_FAILED.json` 直接对比,
> 得出「有一个通过门禁的 v9 没人用」。**这个前提是错的**:两个门禁量的不是同一件事 —— v9 通过的是
> **训练健康度**门禁(`ctc_blank_ratio` 0.188、无 NaN、curriculum 饱和),v1 失败的是**更严格的内容**
> 门禁(streaming error 不得超 offline+15%)。计划 §27.2 在同一 free-running 协议上实测过两者:
> v1 中文 CER 21.01% / 英文 WER 35.34%,v9 是 34.79% / 55.94% —— **v9 几何更健康但内容差 1.65 倍**,
> 已被明确降级为 anti-collapse 研究证据。**换 v9 会让结果更差。**
>
> 完整复盘与修正后的计划见 [../streaming_s2st_plan_v3/PROJECT_REVIEW.zh-CN.md](../streaming_s2st_plan_v3/PROJECT_REVIEW.zh-CN.md)。
> 本文其余部分(Phase 0 的内容门控开口策略、去污染 canary、指标入门禁、Phase 2 的开口决策新 loss)仍然成立。

---

## 0. 为什么两周的 loss 调参不管用

不是权重没调对。是**每一次实验都在修屋顶**。

能力链自底向上,以及每一层的真实状态:

| # | 层 | 状态 | 证据 |
|---|---|---|---|
| 1 | 因果前端 Stage-A | **地基是自己判定失败的版本,且逐位冻结** | `GATE_FAILED.json`,`blocked_next_stage: stage_b_incremental_mt`,`ctc_blank_ratio` 0.80(eng)/0.97(cmn);254 张量 `exact_bitwise_match: true`,自 08-16 未训练 |
| 2 | 流式 ASR | 中文可用,英文是瓶颈 | cmn CER **0.089**(离线 0.0547),eng CER **0.303–0.396** |
| 3 | 增量 MT | 能力在,**但被提交层丢弃** | 单条最好 0.900;换提交策略 goldcov 0.211 → **0.495**,零训练 |
| 4 | **开口决策 WRITE vs WAIT** | **0.168 次/事件,且从未被实验过** | 95 个事件:`WRITE_ASR` 0.821、`WRITE_MT` **0.168**、`WRITE_SEMANTIC` 0.158;只被无差别的 `boundary_ce` 覆盖,无 margin/无类别平衡/无 roll-in |
| 5 | 语音 token 生成 | 过度生成、长片段退化成近静音、重复循环 | 语义/参考 1.8–2.4×;可听起始 **7703 ms**(源 5862 ms 就结束);`of of`/`new new`/`waiting for the right time to wait for the right time` |
| 6 | 长度与收尾 | `natural_eos` 三个 epoch 死在 0.50 | iter 1132 / 1207 / 2264 全是 0.50 |

**26 次实验全部训练第 5、6 层,而第 1 层是坏的、第 3 层的测量是坏的、第 4 层从没被碰过。** 上层的 loss 权重在数学上无法修复下层缺陷,而唯一会暴露这件事的指标(gold 覆盖率)自己就被提交层封顶了。

收益对照,同一口径:

| 干预 | 规模 | goldcov 增量 |
|---|---|---:|
| 11 个 loss 形状变体 | 各 100 步 | ≤ +0.014 |
| epoch 2 → 3 | 1057 步 | +0.006 |
| **修提交策略** | **0 步** | **+0.284** |

**结构性缺陷的收益是 loss 扫参的 20–47 倍。** 而提交策略、码流不匹配(一致率 14.5%)、可听起始 7.7 秒这三个结构性缺陷,都是最近几小时才发现的 —— 它们被 26 次训练实验完整地掩盖了两周。

---

## 1. 关于"一定能解决"

研究问题我不能承诺结果。但计划的**结构**可以承诺一件事:

> **每一层在被训练之前先被独立验证,失败永远定位在一层,而不是被上一层吸收。**

这就是"一定"的含义:执行完 Phase 0–2 之后,要么 demo 成立,要么我们**知道墙在第几层、数值是多少**。过去两周之所以走不动,正是因为失败可以被任意一层吸收,26 次实验因此无法归因。

每个 Phase 的门禁都是**可证伪的数字**,不达标即停、即定位,不允许"再调一版看看"。

---

## 2. Phase 0 —— 把已知的结构缺陷吃干(零训练,约 1 天)

顺序即优先级。全部零训练,单条门禁 8 分钟。

### 0.1 内容门控的开口策略(直接打第 4 层)

Oracle 已经量出两个极端:

| 策略 | 开口率 | `natural_eos` | 首次发音 | cmn 会话覆盖 | 症状 |
|---|---:|---:|---:|---:|---|
| 保守(现状) | 0.168 | 0.50 | 680 ms | **0.514** | 覆盖被憋住 |
| 强制(oracle) | 1.000 | **1.00** | **280 ms** | 0.478 | 重复退化,malformed 2.0→7.75 |

**中间地带从未被试过。** 交织语法里 ASR 在同一事件先于 MT 执行,所以 `asr_deltas` 在做 MT 决策时已知 —— 用"**本事件 ASR 产出新增内容才继续到 MT/TTS,否则 WAIT**"门控。实现与已写好的 `EagerSpeakSession` 同构,只换判据。

**门禁**:cmn→eng 会话自身文本覆盖率 **≥ 0.60**(现状 0.514)、`natural_eos` **≥ 0.875**、`malformed` **≤ 2.0**、可听起始 **≤ 1500 ms**。四项同时达标才算通过。

### 0.2 去污染那 11 个历史 canary

用 local agreement(holdback 2)重测 `decisionrow` / `sembinary` / `fragend` 三个最好的 checkpoint。

- **若分化**:知道该放大哪个 loss,Phase 2 有了先验。
- **若仍不分化**:得到"loss 形状确实无效"的**干净**证据,Phase 2 直接跳过权重扫参,只做新写的 loss。

这一步的价值不是提升指标,是**让 Phase 2 不再瞎猜**。

### 0.3 指标进正式门禁

- **可听起始**(右声道首个 |x|>0.01 的时刻)—— 现有 `first_emission_ms` 会给出 320 ms 而实际听到是 7703 ms,必须替换为可听口径。
- **会话自身文本覆盖率** —— `mt_gold`/`mt_free` 走独立路径,对动作策略完全不敏感,不能作为唯一判据。
- **未截断的语义长度比** —— gate 的 `min(1, 生成/参考)` 单边饱和,看不到过度生成。

### 0.4 修紧音频判据

`rms > 1e-5` → **`rms > 0.01`**,并加 `peak <= 1.0`。当前判据同时放过了近静音(rms 0.0005–0.0016)和削波(peak 1.223)两类坏样本。

---

## 3. Phase 1 —— 换掉失败的地基(约 2 天)

**只有在 Phase 0 之后做**,因为 Phase 0 会告诉我们第 4 层能在推理侧走多远,从而决定这一步值不值得付重建代价。

- 改动点:`experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/experiment.env:14` 的 `V1_CHECKPOINT` → `.../pilot15_v9/stage_a_formal/stage_a_v9_bridgefreeze_formal8_20260817T130814Z/iter_0000381`(`FORMAL_GATE.json` = `passed: true`,`stage_b_authorized: true`,`ctc_blank_ratio` 0.188)。HF 导出已存在。
- **代价必须说清**:e2e 把 `stage_a_objective.*` 逐位冻结并审计,而 task pool 与 v1 ASR teacher cache 是**用 v1 的因果码和教师分布建的**。换 Stage-A 必须重建:`run_v1_teacher_cache_8gpu.sh` → `run_phase3_teacher_cache_8gpu.sh` → `run_formal_task_pools.sh`,全部用新的不可变 RUN_ID,旧 pool 保留。原始 pool 是 90.9 GB / 132 万条 / 64 worker。
- **开训门禁**:`run_bridge_parity.sh` 的 `offline_vs_bridge_agreement_mean ≥ 0.99`。当前 content-first 那条线是 **0.1446** —— 这个门禁本身就是为防止再犯那个错而建的。
- **收益门禁**:eng CER 必须从 0.303 **降到 ≤ 0.20**(v9 的 blank ratio 0.188 对 v1 的 0.80/0.97,主要救英文)。**不降就立即停止这条支线**,并记录"Stage-A 不是英文的瓶颈"这个结论。

---

## 4. Phase 2 —— 唯一真正缺失的训练项(约 2 天)

Phase 0.1 只能在推理侧门控开口。要让**模型自己**学会何时开口,必须写代码里不存在的 loss。

把 `semantic_boundary_binary` 对 END 决策做的三件事,搬到 `WRITE_GENERATE` vs `WAIT`:

| 要素 | 为什么 |
|---|---|
| softplus 校准的二元分数 | **无死区**,margin 满足后仍有梯度;现有 `boundary_ce` 是无差别 CE,`WAIT` 占 0.863 会直接压死正类 |
| 两类各占一半权重 | `WRITE_MT` 0.168 对 `WAIT` 0.863,是 5:1 的类别不平衡 |
| roll-in 版本 | 在模型自己的历史下监督;推理时的条件只有这一种 |
| **重复惩罚** | oracle 证明开口变多就出现 `of of`/`new new`;**现有 loss 一项都不惩罚重复** |

**门禁**:`WRITE_MT`/事件 从 0.168 升到 **≥ 0.50**,**且**会话自身文本覆盖率不低于 Phase 0 的水平,**且**语义长度比 ∈ [0.9, 1.2]。三项同时成立 —— 只升开口率而覆盖率下降,就是 oracle 已经证明过的失败模式,不算通过。

---

## 5. Phase 3 —— 长度与实时(约 1 天)

- 训练侧:把长度比直接写进 loss(当前没有任何一项约束输出/输入时长比,而离线基线本身就是 3.06×)。
- 推理侧:LLM KV cache + 异步 TTS,打 RTF 7.85。
- **这一步不早于 Phase 2 门禁通过**。首次发音 400 ms、AP 0.404 已经足够好,瓶颈全在内容。

---

## 6. 当前在跑的那一版怎么处理

`rollin_continue_epoch4_20260831T183621Z`,1132 步,预计 5.9 小时。

**不停。** 但要如实降低预期:

- 它开的 `semantic_rollin_end_ce` / `semantic_rollin_continue_decision_margin` 的掩码选在 `labels == TOKEN_END_SEMANTIC` 和 gold 为语音 token 的行上 —— **只管语音片段何时结束,完全不影响开口频率**。这是我的靶子选错了,已纠正。
- 它的价值是**第一次让 roll-in 监督跑在 epoch 尺度**(历史全是 100 步),而且是在提交策略修好之后测量。停掉会永久留下"到底是 loss 无效、还是没跑够、还是测量坏了"这个悬案。
- 预期:`natural_eos` 与语义长度比改善,覆盖率小幅动。**风险是重复退化加重**,它没有任何一项惩罚重复。

跑完后用 Phase 0.3 的指标(可听起始 + 会话自身覆盖率 + 未截断长度比)对比 iter_2264,并生成双声道 demo。这个结果直接决定 Phase 2 里 roll-in 相关权重要不要保留。

---

## 7. 时间线

| Phase | 内容 | 8 卡耗时 | 门禁 |
|---|---|---:|---|
| — | 当前 roll-in 续训跑完 + 评测 | 5.9 h + 0.3 h | 记录,不设门禁 |
| 0.1 | 内容门控开口策略 | 0.3 h | 覆盖 ≥0.60、natEOS ≥0.875、malformed ≤2、可听起始 ≤1500 ms |
| 0.2 | 去污染 3 个历史 canary | 0.5 h | 分化 / 不分化(二者都是有效结论) |
| 0.3–0.4 | 指标与判据入门禁 | 0(CPU) | 单测 |
| 1 | Stage-A v9 重基 + 数据重建 | 1–2 天 | 码流一致率 ≥0.99;eng CER ≤0.20 |
| 2 | 开口决策的新 loss + 重复惩罚 | 1–2 天 | WRITE_MT ≥0.50 且覆盖不降 且长度比 ∈[0.9,1.2] |
| 3 | 长度 loss + KV cache/异步 TTS | 1 天 | 覆盖 ≥0.80、可听起始 ≤1500 ms、RTF <1 |

**Phase 0 全部零训练,合计不到 1 天,却直接打在两周来从未被碰过的第 4 层上。这是整份计划里信噪比最高的部分。**

---

## 8. 不做的事

- 不再做 loss 权重扫参。历史上限 +0.014,而结构性修复是 +0.284。
- 不在 Phase 0 之前动 Stage-A 重建 —— 代价太大,且 Phase 0 会告诉我们值不值得。
- 不关闭任何审计。`GATE_FAILED`、`audit_frozen_stage_a`、handoff 审计都正确地报告过真实问题,是它们被忽略而不是它们错了。
- 不用 gate 的 `first_emission_ms` 单独判断延迟(它给 320 ms 而实际听到是 7703 ms)。
- 不用 `mt_gold`/`mt_free` 判断动作策略(路径独立,对策略完全不敏感)。
- 内容未达标前不谈墙钟实时。
