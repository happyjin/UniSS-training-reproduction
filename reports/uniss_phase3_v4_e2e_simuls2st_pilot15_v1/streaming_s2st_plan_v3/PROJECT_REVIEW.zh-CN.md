# UniSS 全项目复盘:从 offline phase1-3 到 streaming,问题出在哪里

- 时间:2026-08-31
- 依据:Codex 会话 `01a04876-dd46-75f1-9c57-a250cd095b39`(922 条用户消息)、`docs/uniss_training_reproduction/` 的计划与评估文档、全部 gate/report artifact、以及本轮实测
- 取代:`streaming_s2st_plan_v2/PLAN.zh-CN.md` 的 Phase 1(其中"换 Stage-A v9"一条基于错误前提,已作废)

---

## 0. 一句话结论

**思路不需要换。** `uniss_phase3_v4_quality_first_true_streaming_asr_mt_tts_training_plan.md` 第 1–26 节的设计是正确的,而且**逐条预判了后来真实发生的每一种失败**。走不动的原因是三处**实现背离了设计**,其中一处是决定性的。

---

## 1. 完整弧线

### 1.1 Offline 阶段(phase1 → phase2 → phase3)

| 阶段 | 结果 |
|---|---|
| phase1 | ASR / S2TT 能力奠基 |
| phase2 | `quality` 模式只能产出中文转写/改写;`performance` 出现严重重复坍缩(25.54 s、semantic 1277、最长连续重复 **343**) |
| **phase3** | `quality` 模式**稳定产出"源语言转写 + 目标语言翻译"两段文本**;重复坍缩被修复(5.42 s、271、最长重复 **8**) |

phase3 正式离线基线(`stage00_phase3_offline_20260816T031129Z`):

| 指标 | 值 |
|---|---:|
| BLEU(quality,中→英 / 英→中,文本) | **33.23 / 52.42** |
| BLEU(quality,音频子集) | 36.78 / 51.43 |
| ASR CER / WER | **0.0547 / 0.0561** |
| `semantic_tokens` 均值 | 785.8 |
| **`direct_s2st` 输出/参考时长比** | **3.06(中→英)、3.27(英→中)** |
| **`slc_0_2`(时长落在 ±20%)** | **0.129 / 0.031** |
| **`missing_eos`** | **92 / 256** |

**offline 阶段就带着两个未修复的缺陷进入 streaming:输出音频比参考长 3 倍、36% 的样本不产生 EOS。** phase3 报告自己也标注了边界:"当前 Phase3 训练脚本复用了 Phase2 mixed packed 数据,不等价于论文严格的 high-quality-only Phase3 配方。"

### 1.2 转向 streaming 的设计(两版)

**第一版:DC-CMW(Deadline-Constrained Causal Micro-WRITE)。** 核心原则写得非常清楚:

```
延迟由确定性调度器保证
内容由模型预测
安全性由 target-support/anticipation 监督保证
```

并且明确写了 **"不再让 WAIT/WRITE 模型无限 WAIT"**。它还给了历史实验的 NCA/CA 对照表:

| 实验 | First WRITE | First Audio NCA | **First Audio CA** |
|---|---:|---:|---:|
| Stage10 EN→ZH | 560 ms | 880 ms | **5.16 s** |
| wait-k=0 | 605 ms | — | 受 RTF 2.83 限制 |
| Prefix V3 短样本 | 3.68 s | 4.16 s | pseudo-streaming |
| 5 分钟模式 | 25.72 s | 25.72 s | 约 306 s |

**这张表说明:项目在 8 月就已经知道"源时间轴延迟低"不等于"用户听得早",并且知道差距可以是 6 倍。**

**第二版:quality-first 计划(§1–26)。** 结构是 Stage A(流式 ASR)→ Stage B(增量 MT)→ Stage C(分段 TTS)→ Stage D(冻结串联),原则是"**只有上游通过,才允许下游开始;下游不能用自己的能力掩盖上游错误**"。

§2.1 的"已排除的误区"里逐条列着:

- 单样本会 WRITE 不等于多样本可用;
- 训练用 oracle 历史、推理用自己的历史 → 严重 exposure mismatch;
- **"仅增加 epoch、shuffle、WAIT/WRITE 权重或 GPU 吞吐,不能修复监督对象和部署状态机不一致。"**

**最后这一条,就是过去两周 26 次实验做的事。计划两周前就写明它无效。**

### 1.3 Stage A 的九个版本与 V1/V9 的选择

Stage A 跑了 v1–v9,**Stage B / C / D 从未被建立**(`experiments/` 里只有 `stage_a_causal_whisper_asr`)。

§27.2 在同一 free-running 协议上实测了两个候选:

| checkpoint | causal-full error | event-streaming error | 中文 CER | 英文 WER |
|---|---:|---:|---:|---:|
| **V1** `iter_0000381` | 14.45% | 27.14% | **21.01%** | **35.34%** |
| V9 `iter_0000381` | 24.86% | 43.84% | 34.79% | 55.94% |

**必须纠正一个容易犯的误判**(我自己在 plan v2 里犯了):V9 的 `FORMAL_GATE.json` 是 `passed: true`,V1 的是 `GATE_FAILED.json` —— 但这两个门禁量的不是同一件事。V9 通过的是**训练健康度**门禁(`ctc_blank_ratio` 0.188、无 NaN、curriculum 饱和、codebook 几何);V1 失败的是**更严格的内容**门禁(streaming error 不得超 offline+15%)。**V9 几何更健康但内容差 1.65 倍**,因此被明确降级为 anti-collapse 研究证据。选 V1 是对的。

### 1.4 §27:把 A→B→C→D 塌缩成一个 student

§27 标题是"最终权威方案:冻结 V1 Causal Whisper 的单模型 E2E Simultaneous S2ST",它**替代**了分阶段方案:Stage A 只作为初始化与 ASR teacher,Stage B 只提供标签构造方法,Stage C 只提供对齐方法,Stage D 只作为 oracle 上限。正式训练只跑一个 E2E student run。

这就是 `uniss_phase3_v4_e2e_simuls2st_pilot15_v1`,以及它之后的全部血脉。

### 1.5 E2E student 的真实成绩(本轮实测)

| 指标 | Stage-A V1(初始化) | **E2E student iter_2264** | offline phase3 |
|---|---:|---:|---:|
| 中文 ASR | 21.01% | **8.9%** | 5.47% |
| 英文 ASR | 35.34% | **30.3%** | 5.61% |
| 源结束前发出目标语音 | — | **6/8** | 不适用 |
| gold MT 覆盖(修提交策略后) | — | **0.495** | — |
| `natural_eos` | — | 0.50 | 0.64(= 1 − 92/256) |
| 输出/源时长比 | — | 2.04 | 3.06 |

**student 的 ASR 明显好于它的初始化(中文 21.0 → 8.9),所以"单模型 student"这条路本身没错。** 中文流式 ASR 距离离线只差 3.4 个点,英文差 24.7 个点。

---

## 2. 三处实现背离设计

### 2.1 决定性的一处:WAIT/WRITE 从"确定性分隔符"变成了"未被监督的采样策略"

计划 §24 写得非常明确:

> `WAIT_READ`/`WRITE_GENERATE` 在首版主要是**确定性的 event delimiter,不是单独策略分类目标**。

而实现里(`evaluation/runtime.py::PersistentInterleavedSession.run_event`)它是 `self._choice(continuation_candidates)` **采样出来的模型决策**。后果链条完整闭合:

| 环节 | 事实 |
|---|---|
| 设计说它不是策略目标 | → **没有人为它写 loss**,只有把 `WAIT`/`WRITE_GENERATE`/`START_GLM`/`EOS`/`END_CONTENT`/`END_SEMANTIC` 全塞在一起的无差别 `boundary_ce`,无 margin、无类别平衡、无 roll-in |
| 实现让它变成采样决策 | → 95 个事件里 `WAIT` 0.863、`WRITE_ASR` 0.821、**`WRITE_MT` 0.168、`WRITE_SEMANTIC` 0.158** |
| 说得少 | → 每次说就得说很多 → **过度生成 1.8–2.4 倍** |
| 中间大段 WAIT | → **最大空白 1240–1640 ms** |
| 从没走到该收尾的地方 | → **`natural_eos` 三个 epoch 恒为 0.50** |

**四个看起来独立的症状是同一个根因。** 而且第一版 DC-CMW 设计早就写过 "不再让 WAIT/WRITE 模型无限 WAIT" —— 这个保护在第二版被 §4.2 第 6 条("禁止 forced WRITE")取消了,取消的理由是评测纯度,但代价是**唯一能结构性保证延迟的机制被移除,同时没有任何 loss 接手**。

### 2.2 A→B→C→D 塌缩,失败无法归因

§2.1 的核心原则是"下游不能用自己的能力掩盖上游错误",§27 却把四层合成一层。结果:26 次实验的失败可以被任意一层吸收,没有一次能定位。

最直接的证据是本轮几小时内找到的三个结构性缺陷,它们被 26 次训练实验完整掩盖了两周:

| 缺陷 | 数值 | 掩盖方式 |
|---|---|---|
| 提交层丢弃译文 | 82.3% 事件冲突,`That's` 覆盖整句 | 唯一的覆盖率指标自己被封顶 |
| 训练/推理码流不匹配(content-first 支线) | 一致率 **14.5%** | 无人测量 |
| 可听起始 vs 上报延迟 | **7703 ms vs 320 ms** | 门禁只报 NCA |

### 2.3 NCA/CA 区分被丢掉

DC-CMW 设计里有完整的 NCA/CA 对照表,而当前 gate 只有 `first_emission_ms`(NCA)。本轮双声道 demo 实测:`emilia_zh_0006199435` 上报 320 ms、**实际可听 7703 ms**,而源音频 5862 ms 就结束了 —— **那根本不是同传**。项目自己 8 月就知道这个陷阱。

---

## 3. offline 阶段的遗留债务

streaming 一直在被要求修一个它看不见的 offline 问题:

| 缺陷 | offline phase3 | streaming e2e |
|---|---:|---:|
| 输出/参考时长比 | **3.06** | 2.04 |
| 时长落在 ±20% 的比例 | **0.129** | — |
| 不产生 EOS | **92/256 = 0.36** | `natural_eos` 0.50 |

**streaming 的时长比(2.04)其实比 offline(3.06)更好,收尾(0.50)比 offline(0.64 产生 EOS)更差但同量级。** 也就是说:**长度失控和不收尾是从 offline 继承的,不是 streaming 引入的。** 而在 streaming 条件下修它,每单位信号的代价是 offline 的 20 倍以上(1132 步 ≈ 6 小时,vs offline 数据侧修正)。

---

## 4. 要不要换思路

**不换。** 但要做三件"回到设计"的事,和一件设计里没有的事。

### 4.1 回到设计(三件)

**A. 把 WAIT/WRITE 恢复成设计所说的样子。** 两条路,不互斥:

- **推理侧(零训练,当天可得结论)**:内容门控 —— 本事件 ASR 有新增稳定内容就继续到 MT/TTS,否则 WAIT。这就是"确定性 event delimiter"的原意。Oracle 已经量出上下界:保守 0.168 开口 / 会话覆盖 0.514;强制 1.000 开口 / `natural_eos` 1.00 / 首次发音 280 ms 但覆盖 0.478 且重复退化。**中间地带从未被试。**
- **训练侧**:若要模型自己学,必须**新写** loss —— 把 `semantic_boundary_binary` 对 END 做的三件事(softplus 无死区、两类各半权重、roll-in)搬到 `WRITE_GENERATE` vs `WAIT`,并加**重复惩罚**(现有 loss 一项都没有)。代码里不存在这一项。

**B. 恢复按能力的门禁,而不是恢复三个 Qwen。** 单 student 是对的(它的 ASR 比 V1 好一倍),要恢复的是**归因能力**:每次评测必须分别报 ASR / 增量 MT / 会话自身文本覆盖 / 语音,任一项退化即定位,不允许被总分吸收。本轮已经把这套指标写好了。

**C. 延迟指标改成 CA/可听口径。** `first_emission_ms` 保留为诊断,门禁改用右声道首个 |x|>0.01 的时刻。同时修紧音频判据(`rms > 0.01`、`peak <= 1.0`),当前判据同时放过近静音(rms 0.0005)和削波(peak 1.223)。

### 4.2 设计里没有的一件:回到 offline 修长度与收尾

这是**唯一一个真正的"换思路"建议**。

时长比 3.06、`slc_0_2` 0.129、`missing_eos` 92/256 都是 offline 缺陷,streaming 继承了它们。在 offline 修:

- 条件更好(无因果约束、无 exposure mismatch、无提交层);
- 信号更便宜(不需要 event rollout、不需要 teacher cache);
- 而且 phase3 自己承认"复用了 Phase2 mixed packed 数据,不等价于论文严格的 high-quality-only 配方" —— **配方本身就没按论文做完**。

具体:用严格 high-quality-only 数据重做一轮 phase3(或续训),把"目标音频时长 ≈ 参考时长"和"必须产生 EOS"直接写进 loss 与数据筛选。**offline 的时长比不降到 1.2 以内,streaming 永远在修一个上游问题。**

---

## 5. 计划

| Phase | 内容 | 代价 | 门禁 |
|---|---|---:|---|
| **0** | 内容门控开口策略(零训练) | 0.3 h | 会话覆盖 ≥0.60、`natural_eos` ≥0.875、malformed ≤2、**可听起始** ≤1500 ms,四项同时 |
| **0b** | 去污染 3 个历史 canary(local agreement 重测) | 0.5 h | 分化 / 不分化(二者都是有效结论) |
| **0c** | 可听起始 + 会话自身覆盖 + 未截断长度比 入门禁;`rms>0.01`、`peak≤1.0` | 0(CPU) | 单测 |
| **1** | **回到 offline 修长度与收尾**(严格 high-quality-only 配方) | 1–2 天 | 时长比 ≤1.2、`slc_0_2` ≥0.6、`missing_eos` ≤0.05,且 BLEU 不低于 33.2/52.4 |
| **2** | 开口决策的新 loss + 重复惩罚(streaming) | 1–2 天 | `WRITE_MT`/事件 ≥0.50 **且**覆盖不降 **且**长度比 ∈[0.9,1.2] |
| **3** | 英文流式 ASR(30.3% → ≤20%) | 1–2 天 | 英文 WER ≤0.20;不达标即记录"Stage-A 不是英文瓶颈"并停止该支线 |
| **4** | CA 延迟工程:KV cache + 异步 TTS | 1 天 | 可听起始 ≤1500 ms、RTF <1 |

**不再做的事**:loss 权重扫参(历史上限 +0.014,而结构性修复是 +0.284);换 Stage-A v9(内容差 1.65 倍,已被 §27.2 实测排除);在 Phase 0 之前重建 task pool。

---

## 6. 当前在跑的那一版

`rollin_continue_epoch4_20260831T183621Z`,1132 步,约 5.9 小时。

它开的 `semantic_rollin_end_ce` / `semantic_rollin_continue_decision_margin` 掩码选在 `labels == TOKEN_END_SEMANTIC` 和 gold 为语音 token 的行上 —— **只管语音片段何时结束,不影响开口频率**。靶子选错了。

不停的理由:这是**第一次让 roll-in 监督跑在 epoch 尺度**(历史 11 个 canary 全是 100 步),而且是在提交策略修好、指标去污染之后测量。停掉会永久留下"loss 无效 / 没跑够 / 测量坏了"的三义悬案。跑完用本轮指标对比 iter_2264,结果直接决定 Phase 2 里 roll-in 权重要不要保留。
