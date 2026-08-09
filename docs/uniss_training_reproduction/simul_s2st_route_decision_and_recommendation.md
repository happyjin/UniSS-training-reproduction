# UniSS Simultaneous S2ST 路线选型与建议

> 日期：2026-08-09
> 综合材料：Student v2 完整报告、Stage-B latent 15-shard 报告、Phase3-joint V6 失败分析、StreamSpeech CTC v1 Stage00–13、codex session `019f6fc5`（08-06 至 08-08）、以及 FAST / SimulS2S-LLM / SimulS2ST-Omni / InfiniSST / Hibiki 等外部方案
> 前序：[`phase3_joint_v6_root_cause_reframe_and_streaming_interface_redesign.md`](./phase3_joint_v6_root_cause_reframe_and_streaming_interface_redesign.md)、[`streamspeech_failure_diagnosis_and_alternative_simul_s2st_ideas.md`](./streamspeech_failure_diagnosis_and_alternative_simul_s2st_ideas.md)

---

## 0. 结论先行

**第一，Student v2 的延迟不高，反而是整个项目里唯一延迟达标的部分。** 它的实测是 active RTF `0.0985`、first stable p50/p95 `320 / 480 ms`、future perturbation `0.0`、cache parity `1.0`。结构门全部通过。它没过的是**质量门**（teacher agreement 29.29%），不是延迟门。你记忆里的"延迟高"来自端到端链路（Stage4 StartOffset CA `7.37 s`、Stage12 CA `5.16 s` / RTF `2.23`），不是来自 Student v2。

**第二，三次大规模尝试全部投在源侧，而实测瓶颈在目标侧。** Student v2/v3、Phase3-joint V4/V6、StreamSpeech CTC Stage00–13，主要训练量都花在"怎么把源语音变成流式 token"。但源前端独立测量是 RTF `0.098`，Stage12 端到端却是 RTF `2.23`——**约 95% 的计算感知延迟在源 tokenizer 下游**（Qwen 自回归 + BiCodec + 策略）。这部分从来没有被专门优化过。

**第三，亚秒目标本身校准错了。** 你的天花板审计说全上下文正确 token 中位数 2.9 s 才稳定；SimulS2ST-Omni 与 LiveInterpret 2.0 在中英上的竞争性工作点是 **LAAL 3–6 秒**；HPO 报告的工作点是 1.5 秒。**中英同传的学术竞争区间是 2–4 秒，不是亚秒。** 把目标定在亚秒，等于在一个本来就最难的语言对上再加一个业界没人达到的约束。

**建议的路线：保留 Student v2 前端和冻结 Phase3，把下一轮全部投入目标侧生成，延迟目标改为 LAAL 2–3 秒。** 这是唯一一条能复用你全部已通过资产、且直击实测瓶颈的路。

---

## 1. Student v2 到底怎么样：逐项复盘

### 1.1 通过的部分（这些是真资产，不要丢）

| 指标 | 结果 | 判断 |
| --- | ---: | --- |
| active RTF | `0.0985` | 约实时预算的 1/10 |
| long-session RTF | `0.0971` | 长会话无退化 |
| first self-stable p50 / p95 | `320 / 480 ms` | 前端出 token 很快 |
| correct-stable p50 / p95 | `320 / 480 ms` | 正确的那些也一样快 |
| future perturbation max | `0.0` | 真因果，不偷看未来 |
| cache / full committed parity | `1.0` | 增量推理与一次性 forward 完全一致 |
| cache max abs | `2.86e-6` | 数值误差可忽略 |
| structural gate | pass | — |

121.1M 参数、16 层 Emformer、160 ms chunk + 80 ms lookahead、2 s 显式左上下文、每 80 ms 出一个 1280 维 latent。**"把 WhisperVQ 从反复 prefix 重编码改成有状态因果编码器"这件事，Student v2 已经做完了。**

### 1.2 没通过的部分

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| target position agreement | `0.2929` | `0.90` |
| target edit agreement | `0.2924` | `0.70` |
| correct-stable coverage | `31.25%` | — |
| full-context teacher agreement | `0.1075` | — |
| quality gate | **fail** | — |

下游冻结 Phase3 的敏感性探针：

| 源 GLM 流 | EN→ZH | ZH→EN |
| --- | ---: | ---: |
| released（离线全上下文） | 33.45 | 26.61 |
| **exact prefix-causal 80 ms** | **31.22** | **25.21** |
| streaming clone 160×80 ms | 22.95 | 22.46 |
| Student v2 prefix-80 | 21.13 | 15.32 |
| latent Student v1 | 18.69 | 12.86 |

### 1.3 这张表里最重要的一行

**`exact prefix-causal 80 ms` 拿到 31.22 / 25.21，只比离线低 2.2 / 1.4 点。**

这意味着：**80 ms 前瞻下因果可得的源信息，对冻结 Phase3 来说已经基本够用了。** 源侧的"信息量"问题不存在，剩下的全是"怎么便宜地算出这个流"的工程问题。

再看后面两行的落差在哪：

| 从 → 到 | EN→ZH 损失 | 含义 |
| --- | ---: | --- |
| exact prefix-causal 80 ms → streaming clone 160×80 | **−8.27** | prefix 重算 → 单遍 chunk-causal 的代价 |
| streaming clone → Student v2 | −1.82 | 蒸馏成小模型的代价（EN→ZH） |
| streaming clone → Student v2 | −7.14 | 同上（ZH→EN，明显更差） |

**最大的单项损失不是蒸馏，是"从 prefix 重算改成单遍 chunk-causal"这一步（−8.27 BLEU）。** 这一点此前没有被当作主要矛盾。

### 1.4 v2 质量没过的四个原因（报告自述，我认为都成立）

1. prefix80 阶段**没有 pre-VQ hidden target**，日志里的 `hidden_l1 = 0` 不是完美对齐而是根本没有这项监督；
2. 100k prefix 数据从 manifest 前部连续选取，**方向不平衡**；
3. 第二阶段**替换**而不是混合 clone 监督，可能发生表示遗忘；
4. 16,384 类离散 exact match 过难，落在声学近邻 cell 也记为错。

v3 针对这四条做了修复（双向平衡 50k+50k、真 pre-VQ hidden、1:1 混训、双向 Phase3 BLEU 选模），结果是 agreement 17.70%、Stage C 三个工作点 recall 全为 0。**修复方向对，但没救回来**——因为天花板在那里（320 ms 上限约 0.5465，640 ms 约 0.6814）。

---

## 2. 三次尝试的共同盲点

| 尝试 | 时间 | 投入 | 目标 | 结果 |
| --- | --- | --- | --- | --- |
| Student v2 / v3 | 07 月末–08-03 | 15-shard，8×H200 | 因果流式源 tokenizer | 结构过、质量不过 |
| Phase3-joint V4 → V6 | 08-06 – 08-08 | full198，8×H200，9075 iter | 源前端 + Phase3 接口联合 | V4 发散、V6 安全门终止 |
| StreamSpeech CTC v1 | 07 月–08-05 | Stage00–13 | 源侧 CTC 多任务 + 计数策略 | 门未过，Stage12 BLEU 2.26 |

**三次都在回答同一个问题：源语音怎么变成流式 token。没有一次在回答：目标语音怎么增量生成。**

而实测告诉你瓶颈在哪：

| 环节 | 实测 |
| --- | ---: |
| 源前端（Student v2 独立测量） | RTF **0.098** |
| Stage12 端到端 | RTF **2.23**（EN→ZH）/ **7.49**（ZH→EN） |
| Stage12 首音频 NCA → CA | 880 ms → **5164 ms** |

NCA 到 CA 的 **4.3 秒差额全是纯计算**，而源前端只占 0.098。**剩下约 2.1 的 RTF 在 Qwen 自回归 + BiCodec 解码 + 策略上。**

具体机制：Phase3 用单个 LM head 在扩展词表上自回归生成 **50 Hz** 的 BiCodec semantic token。一秒目标语音要 50 步 AR。这在离线是可以接受的，在流式下就是灾难——而且早期结构不完整导致 Stage11 大量 reject（2/8、0/3 valid）。

---

## 3. 延迟目标应该定在哪

| 来源 | 中英/类似难度的工作点 |
| --- | --- |
| 你的天花板审计 | 全上下文正确 token 中位稳定 **2.9 s** |
| SimulS2ST-Omni 竞争性区间 | latency multiplier **m3–m6**（1 s 源块，即 3–6 s） |
| LiveInterpret 2.0（闭源 SOTA） | 同区间对比 |
| HPO (ACL 2026) | **1.5 s** 下 +7 COMET |
| 人类同传 ear-voice span（中英） | 通常 3–6 s |

**把目标从"亚秒"改成"LAAL 2–3 秒、CA 可控、RTF < 0.5"，会让整个问题从不可能变成可做。** 这不是降低标准——这是业界在中英方向上的实际竞争标准。你现在 Stage12 的 CA 5.16 s 配 BLEU 2.26，问题从来不是延迟不够低。

---

## 4. 推荐路线

### 4.1 总体：保源侧、修目标侧

```text
保留（已验证通过，不动）
  Student v2/v3 causal Emformer   RTF 0.098, 320/480ms, 因果+cache 全过
  冻结 Phase3 Qwen                 CVSS-T 32.04/24.72，外部论文引为 SOTA
  BiCodec global speaker token     音色保持，UniSS 差异化优势

替换（实测瓶颈所在）
  Qwen AR 逐 token 出 50Hz semantic  →  NAR CTC 语音生成头
  每 chunk 重算历史                  →  Λ 形 KV cache
  CTC 计数 / WAIT-WRITE 动作 token   →  wait-k on stability/CIF 计数

改判据（此前用错了）
  teacher agreement  →  冻结 Phase3 双向 Text-BLEU + LAAL/CA/RTF
```

### 4.2 逐组件建议与依据

| 组件 | 现状 | 建议 | 学术依据 |
| --- | --- | --- | --- |
| 源前端 | Student v2/v3，结构全过、agreement 29% | **原样保留**，停止用 agreement 判它 | 你自己的天花板审计 + agreement 与 BLEU 反相关证据 |
| 源→Qwen 接口 | 离散 GLM 最近邻 | **保留**（Phase3 已证明能读），但若要再训 Student，target 换成 FAST 的**同模型 oracle-future hidden** | FAST (EMNLP 2023) |
| 何时 WRITE | CTC 双计数 + 二次确认 | **wait-k**，由 Student v2 **已训好的 stability head** 或 CIF 发放计数驱动 | SimulS2S-LLM、FAST |
| Qwen 推理 | 每次重算 | **Λ 形 KV cache**：只留 system prompt KV + 最近 w 个 token，存前去 RoPE、拼后重施 | InfiniSST (ACL 2025) |
| 目标 semantic 生成 | AR 逐 token，50 Hz | **NAR CTC 头**：LLM hidden → 上采样 U → causal Transformer → CTC over BiCodec 词表；配 incremental beam search + 语音 token n-gram LM shallow fusion | SimulS2S-LLM |
| 音色 | BiCodec global 32 token | 保留，作为 NAR 头的 speaker condition | — |
| 评测 | UniST dev（重建音频） | 换 **CVSS-T**（已下载对齐，ZH→EN test 4,897 条齐备）；报 LAAL / StreamLAAL、NCA/CA、RTF、ASR-BLEU、SIM-O、AutoPCP | SimulS2ST-Omni 协议 |

### 4.3 为什么 NAR CTC 头是第一优先级

它同时解决三件事：

1. **RTF**：一秒目标语音从 50 步 AR 变成一次 NAR CTC 前向；
2. **早期结构不完整 / 重复**：NAR 不会因为前缀短就退化成重复串，Stage11 的大量 reject 应显著减少；
3. **CA 延迟**：NCA→CA 的 4.3 秒差额主要来自这里。

而且它**不动 Phase3 主干**——NAR 头挂在 Qwen hidden 上，Qwen 本身可以先完全冻结。这是所有候选改动里"收益/风险"最高的一个。

---

## 5. 三条路线的取舍

| | 路线 A（推荐） | 路线 B | 路线 C |
| --- | --- | --- | --- |
| 名称 | 保 Phase3 + 修目标侧 | A + FAST 重训源前端 | SimulS2ST-Omni 轨迹 + Thinker–Talker |
| 改动 | NAR CTC 头 + KV cache + wait-k | 再加 Student 用 FAD 重训 | 重建目标侧为双流 + 轨迹监督数据 |
| Phase3 是否保留 | **完全保留（冻结）** | 保留 | Thinker 保留，Talker 新建 |
| 需要词对齐 | 否 | 否 | **是**（SimAlign + NIR 过滤） |
| 预期 RTF | 2.23 → **< 0.5** | 同 A | < 0.5 |
| 预期质量 | 接近 `exact prefix-causal` 上限（约 31 / 25 的 70–85%） | 更接近上限 | 最高 |
| 工期 | **2–3 周** | +2–3 周 | +6–8 周 |
| 风险 | **低** | 中 | 中高 |

**建议顺序：A → 评估 → B 或 C。** A 的价值不只在于它便宜，还在于它是第一次真正测量目标侧——跑完之后你会第一次知道"在源 token 足够好的前提下，目标侧能做到多快多好"，这个数字决定 B 和 C 哪个值得做。

同时，**路线 C 的数据工作（SimAlign 跨语言对齐 + NIR 单调性过滤）可以并行开始**，因为它对 A/B/C 都有用，而且 SimulS2ST-Omni 的消融显示这是低延迟档收益最大的单项（不做过滤时 m1 从 21.14 崩到 4.59）。你已经有 `source_words` / `target_words` 时间戳。

---

## 6. 执行顺序

**第 0 步：RTF 分解（1–2 天，只跑推理，不训练）**

在 Stage12 同一批样本上，把端到端 wall-clock 拆成：源前端 / 策略判定 / Qwen prefill / Qwen AR decode / BiCodec decode / 音频拼接。

这一步决定后面所有优先级。**如果 Qwen AR decode 占 60% 以上，NAR 头就是对的；如果 BiCodec decode 占大头，优先做增量 codec；如果 Qwen prefill 占大头，优先做 KV cache。** 现在没有这个分解，任何目标侧改造都是猜。

**第 1 步：D1 checkpoint 复检（1 天，与第 0 步并行）**

对 V6 Stage A 和 iter 750/1500/3000/5000 跑冻结 Phase3 的 32 条双向 BLEU 探针。确认 agreement 掉的同时 BLEU 是否也掉。这决定 V6 那批 checkpoint 要不要留、agreement 要不要彻底降级为诊断项。

**第 2 步：NAR CTC 语音生成头（1–2 周）**

- Qwen 完全冻结，只训 NAR 头；
- 上采样率 `U` 由 `target_bicodec_length / target_text_token_length` 的 p50/p95 定，**不要照搬 SimulS2S-LLM 的 25**；
- 先做 CTC path feasibility 检查（你在 `nar_semantic.py` 和 `streaming_student.py` 里已有 CTC 基础设施）；
- 门：离线条件下 ASR-BLEU 相对 Phase3 AR 基线掉 ≤ 2 点，RTF 降到 < 0.3。

**第 3 步：Λ 形 KV cache + wait-k 串联（3–5 天）**

用 Student v2 的 stability head 驱动 wait-k，接上第 2 步的 NAR 头，在 CVSS-T 上扫 k，画 LAAL–ASR-BLEU 帕累托曲线。

**第 4 步：决策**

看帕累托曲线离 SimulS2ST-Omni / LiveInterpret 的工作点差多少，再决定投 B（修源前端）还是 C（换双流架构）。

---

## 7. 明确不建议做的事

| 不建议 | 原因 |
| --- | --- |
| 继续提高 teacher agreement | 天花板审计已证明 320 ms 上限约 0.5465、640 ms 约 0.6814；且 agreement 与下游 BLEU 在你的数据里反相关 |
| 提高 `MAX_COMMITMENT` 后从 V6 iter 5000 续训 | 只是允许模型沿已错方向继续走，不修 teacher 闭环、bridge 容量或 loss 冲突 |
| 先花几天全量重建 `source_glm_reencoded` | 它能把 offline agreement 修到 ~99%，但修不了 320 ms 的上限。先用 2000 条小样本判定值不值 |
| 继续缩短 Stage05 的 WRITE 阈值 | 策略触发已经够早（first WRITE 560 ms），问题是内容质量和 CA |
| 现在就换 Mimi 重训 Phase3 | 代价极大，且换 tokenizer 不会自动带来 simultaneous 能力（需要双流架构 + 延迟对齐数据） |
| 坚持亚秒目标 | 中英学术竞争区间是 LAAL 2–4 秒 |

---

## 8. 一句话总结

> Student v2 已经把"流式源前端"这件事做完了（RTF 0.098、320/480 ms、真因果），它被一个不可达且与下游质量反相关的指标判了不及格。真正没做的是目标侧：约 95% 的计算感知延迟在 Qwen 自回归 + BiCodec 上。下一轮应该冻结 Phase3、保留 Student v2、加一个 NAR CTC 语音生成头和 Λ 形 KV cache，用 wait-k 串起来，把目标定在 LAAL 2–3 秒，在 CVSS-T 上画帕累托曲线。

---

## 9. 参考

**本地证据**
- [`student_v2_complete_process_and_colleague_briefing.md`](./student_v2_complete_process_and_colleague_briefing.md)
- [`stage_b_latent_15shard_h200_execution_report.md`](./stage_b_latent_15shard_h200_execution_report.md) §10 天花板、§11 下游敏感性、§13 v2 正式结果
- [`uniss_phase3_whisper_streamspeech_joint_v6_full198_stage_b_failure_analysis.md`](./uniss_phase3_whisper_streamspeech_joint_v6_full198_stage_b_failure_analysis.md)
- [`uniss_emformer_stages_vs_streamspeech_original_training_audit.md`](./uniss_emformer_stages_vs_streamspeech_original_training_audit.md)
- [`simuls2st_omni_cvss_t_data_preparation_and_evaluation_plan.md`](./simuls2st_omni_cvss_t_data_preparation_and_evaluation_plan.md)
- `reports/uniss_streamspeech_ctc_v1/stage12_stage09_11_bilingual_eval_v1.md`
- `reports/simul_uniss_subsecond_v2/stage_b_v2_prefix80_validation.json`

**外部方案**
- Deng et al. *SimulS2S-LLM*. ACL 2025. [arXiv:2504.15509](https://arxiv.org/abs/2504.15509) — NAR CTC 语音生成头、CIF、incremental beam search
- Ouyang et al. *InfiniSST*. ACL 2025 Findings. [ACL](https://aclanthology.org/2025.findings-acl.157/) — Λ 形 KV cache
- Fu et al. *FAST*. EMNLP 2023. [PDF](https://aclanthology.org/2023.emnlp-main.1033.pdf) — 离线模型改流式、FAI/FAD
- He et al. *SimulS2ST-Omni*. 2026. [arXiv:2607.19810](https://arxiv.org/html/2607.19810) — 轨迹监督、NIR 过滤、Thinker–Talker、CVSS-T/RealSI 协议
- Labiausse et al. *Hibiki*. ICML 2025. [项目页](https://hibiki-s2st.github.io/) — 多流恒定帧率、延迟编进数据
- Ouyang et al. *HPO*. ACL 2026. [ACL](https://aclanthology.org/2026.acl-long.80/) — 延迟-质量分层奖励后训练
