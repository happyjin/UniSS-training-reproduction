# Streaming / Simultaneous S2ST 指标分析报告

- 时间:2026-08-31
- 被测 checkpoint:`uniss_phase3_v4_e2e_simuls2st_pilot15_v1` / `extended_canaries/endmargin_epoch23_15shard_20260824T190227Z/iter_0002264`
- 评测:该实验自己的 free-running gate,冻结的 fixed-16 selection(8 条跑完整 S2S),`MAX_S2S_SEMANTIC_TOKENS=384`
- 离线对照:`stage00_phase3_offline_20260816T031129Z/baseline_summary.json`
- 原始数据:同目录 [METRICS.json](METRICS.json)
- 指标代码:[experiments/uniss_phase3_e2e_commit_policy_v1/evaluation/streaming_metrics.py](../../../experiments/uniss_phase3_e2e_commit_policy_v1/evaluation/streaming_metrics.py)
- 声明范围:**train-seen**,16 条样本(S2S 仅 8 条),不构成泛化结论

---

## 0. 结论摘要

1. **同传这件事本身已经成立。** 中→英 8/8 样本在源音频结束**前**就已发出目标语音,首次发音中位数 **400 ms**、均值 680 ms(源时间轴),零回滚。
2. **流式 ASR 已基本追平离线。** 中→英 CER **0.069**(p50 0.059),离线 Phase-3 是 0.0547 —— 因果流式只付了约 1.4 个点的代价。
3. **真正的阻塞是语音时长,不是延迟也不是识别。** 中→英输出音频是源音频的 **2.04 倍**,语义 token 是参考的 **1.80 倍**。一个同传系统的输出时长必须≈输入时长,否则物理上无法跟上。这个缺陷是从离线 Phase-3(时长比 **3.06**、`slc_0_2` 仅 0.129)继承的,流式只是把它改善了一部分。
4. **译文完整性的一大半是提交层 bug,已修复。** 把 `append_only_commit` 换成复用现有的 `StablePrefixCommitter`(local agreement,holdback=2),gold-source 覆盖率从 **0.211 → 0.495**、英→中 chrF 从 **4.58 → 30.48**,零训练、零参数改动。
5. **有一条样本全部指标同时成立**:`emilia_zh_0004122419`,CER 0.000、译文覆盖 0.800、首次发音 480 ms、音频/源时长 1.102、自然结束。见 §6。

---

## 1. 指标口径与方法论

### 1.1 延迟定义

源轴以**输入音频毫秒**度量(而不是源 token),沿用 SimulEval 的语音形式:

$$d_i = \text{已消耗的源时间(ms),当第 } i \text{ 个目标单位被发出}$$

$$\mathrm{AL} = \frac{1}{\tau}\sum_{i=1}^{\tau}\Big(d_i - (i-1)\frac{T_{\text{src}}}{|Y^*|}\Big),\quad \tau=\min\{i: d_i \ge T_{\text{src}}\}$$

$$\mathrm{LAAL} = \text{同上,但 } |Y^*| \to \max(|Y|, |Y^*|)$$

$$\mathrm{AP} = \frac{\sum_i d_i}{T_{\text{src}}\cdot|Y|},\qquad
\mathrm{DAL} = \frac{1}{|Y|}\sum_i\Big(d'_i-(i-1)\frac{T_{\text{src}}}{|Y^*|}\Big),\ d'_i=\max\big(d_i,\ d'_{i-1}+\tfrac{T_{\text{src}}}{|Y^*|}\big)$$

**LAAL 不是 AL 的别名。** 项目里 [training/simul_uniss/latency_metrics.py:48](../../../training/simul_uniss/latency_metrics.py#L48) 写的是 `"laal_glm_tokens": al`,只在假设长度不超过参考长度时才成立(此时 `max` 取到参考,两者恒等)。**本模型语音过度生成 1.8–2.4 倍,这个别名会给出误导结论**,因此本报告按定义重新实现,并用单测钉住两种情形(等长时相等、过度生成时 LAAL 更严格)。

### 1.2 过度生成会让 AL 失去意义

实测中→英语音 **AL = −1990 ms**。负值不代表"提前于源音频说出",而是过度生成的算术后果:理想时刻 $(i-1)T/|Y^*|$ 用的是**参考**长度,当实际发出的单位数远多于参考时,后面的单位被减去一个越来越大的量。

| 指标 | 中→英语音 | 是否可解释 |
|---|---:|---|
| AL | −1990 ms | ❌ 被长度失配污染 |
| LAAL | −375 ms | ⚠️ 仍受影响但已大幅缓解 |
| **DAL** | **988 ms** | ✅ 单调延迟下界,稳健 |
| **首次发音** | **680 ms** | ✅ 定义无关,最可信 |
| AP | 0.404 | ✅ 有界 [0,1] |

**结论:在长度比修正到 ≈1.0 之前,应以 `first_emission_ms` 与 `DAL` 为主指标,AL 仅作记录。** 这是本报告的方法论要点。

### 1.3 非计算感知

以上全部是**源时间轴**指标:它回答"消耗了多少输入之后这个单位可以被发出",不回答"听众什么时候听到"。墙钟成本单列为 RTF,当前该血脉未逐事件记录时间戳,故本报告不给计算感知延迟(computation-aware latency)。

---

## 2. 与离线 Phase-3 基线的对照

| 指标 | 离线 Phase-3 | 流式 e2e(中→英) | 差异 |
|---|---:|---:|---|
| ASR 错误率 | 0.0547(CER) | **0.069** | +1.4 点 |
| ASR 错误率(英→中) | 0.0561(WER) | 0.396 | **+34 点** |
| BLEU(quality,中→英,文本) | 33.23 | 18.98(gold-source 增量) | 口径不同,见下 |
| BLEU(quality,英→中,文本) | 52.42 | 27.66(gold-source 增量) | 同上 |
| 输出音频/参考时长比 | 3.061 | **2.041** | 改善 33% |
| `slc_0_2`(时长落在 ±20%) | 0.129 | — | — |
| `semantic_tokens` 均值 | 785.8 | — | — |
| `missing_eos` | 92 / 256 | `natural_eos` 仅 0.50 | 同类缺陷 |

三点说明:

1. **中文流式 ASR 代价很小(+1.4 点)**,英文代价巨大(+34 点)。英文因果 ASR 是当前最大的单点缺口,并且它直接拖累 free-source 的英→中翻译(chrF 仅 5.88,而 gold-source 是 30.48)。
2. **BLEU 不可直接比较**:离线是整句一次性翻译,流式是增量提交后的最终假设,且流式受覆盖率(0.495)限制。可比的是同一评测器下的策略消融(§5)。
3. **`missing_eos` 92/256 与流式 `natural_eos` 0.50 是同一个缺陷**:模型不会自然收尾。这不是流式引入的。

---

## 3. 分能力评估(local agreement hb=2)

### 3.1 ASR

| 指标 | 全部(16) | 中→英(8) | 英→中(8) |
|---|---:|---:|---:|
| 错误率 均值 / p50 / p90 | 0.233 / 0.145 / 0.533 | **0.069 / 0.059 / 0.154** | 0.396 / 0.354 / 0.717 |
| 错误率 max | 0.833 | 0.171 | 0.833 |
| `empty_events` 均值 | 3.375 | 3.250 | 3.500 |
| `early_eos_events` 均值 | 0.062 | **0.000** | 0.125 |
| `source_rollbacks` | 0 | 0 | 0 |

中→英识别质量已达可用水平(最差样本 CER 0.171)。**`empty_events` 均值 3.4 是结构缺陷**:平均每条有 3.4 个事件产出空 ASR 输出,这是 gate 里 `e_asr_structure_valid` 失败的原因,虽然不直接损害最终转写。

### 3.2 增量 MT

| 指标 | 全部 | 中→英 | 英→中 |
|---|---:|---:|---:|
| gold-source 覆盖率 均值 / p50 | 0.495 / 0.464 | 0.422 / 0.304 | 0.568 / 0.500 |
| gold-source 长度比 | 0.678 | 0.620 | 0.737 |
| gold-source 提交冲突 均值 | 7.25 | 6.50 | 8.00 |
| free-source 覆盖率 | 0.292 | 0.387 | **0.197** |
| free-source 长度比 | 0.538 | 0.676 | 0.400 |
| 回滚事件 | **0** | 0 | 0 |

**gold→free 的落差就是 ASR 误差的传播代价**:中→英 0.422→0.387(小),英→中 0.568→0.197(**−65%**)。这再次指向英文因果 ASR。

### 3.3 语音输出(TTS / 语义 token)

| 指标 | 全部 | 中→英 | 英→中 |
|---|---:|---:|---:|
| 非静音样本比例 | 0.875 | **1.000** | 0.750 |
| `natural_eos` 比例 | 0.500 | 0.500 | 0.500 |
| 语义 token / 参考(未截断) | 1.299 | **1.800** | 0.797 |
| **输出音频 / 源音频时长比** | 1.444 | **2.041** | 0.847 |
| `malformed_segments` 均值 | 1.125 | 1.500 | 0.750 |
| `invalid_semantic_tokens` | 0 | 0 | 0 |

注意 gate 报告的 `semantic_coverage` 是 `min(1.0, 生成/参考)`,**单边饱和,完全看不到过度生成**。本报告的 `semantic_length_ratio` 未截断,才暴露了 1.80 倍这个事实。

两个音频层面的具体缺陷:
- `emilia_zh_0006795452` 的 **peak = 1.223 > 1.0**,存在削波。
- `emilia_zh_0005215832` 的 **rms = 0.0016 / peak = 0.099**,虽通过了 `rms>1e-5` 的非静音判据,但实际上几乎听不见。**非静音阈值过松**,建议改为 `rms > 0.01`。

### 3.4 延迟

| 指标 | 中→英 | 英→中 |
|---|---:|---:|
| 首次发音 均值 / p50 / min | **680 / 400 / 320 ms** | 1747 / 1920 / 1280 ms |
| 语音 DAL | 988 ms | 1747 ms |
| 语音 LAAL | −375 ms | 804 ms |
| 语音 AP | 0.404 | 0.616 |
| 文本 LAAL 均值 / p50 | 493 / 230 ms | 1238 / 1252 ms |
| 语音最大空白 均值 | 1640 ms | 533 ms |
| 源结束前发声比例 | **1.000** | 0.500 |

中→英以 320–480 ms 开口、AP 0.404(即平均只等了 40% 的输入就发出),**这是真正的低延迟同传行为**。最大空白 1640 ms 偏大但远好于 phasea 血脉的 27 秒。

---

## 4. 动作分布

事件级 `chosen_continuations` 统计(全部样本)显示交织语法被正确执行:每个源事件最多各出一个 ASR / MT / TTS 片段,`WAIT` 用于让出到下一个源块。详细计数见 `METRICS.json` 中每条样本的 `s2s.actions`。

---

## 5. 提交策略消融(唯一变量:commit policy)

同一 checkpoint、同一 selection、同一评测器,只替换增量 MT 的提交策略。**零训练。**

| 策略 | gold 覆盖 | 覆盖 min | 冲突 | 中→英 chrF | 中→英 BLEU | 英→中 chrF | 英→中 BLEU | free 覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `append_only`(现网) | 0.211 | 0.000 | 260 | 31.56 | 10.49 | 4.58 | 0.04 | 0.161 |
| local agreement hb=1 | 0.377 | 0.030 | 159 | 33.34 | 12.61 | 18.86 | 13.29 | 0.186 |
| **local agreement hb=2** | 0.495 | **0.092** | 116 | 38.64 | **18.98** | **30.48** | **27.66** | 0.292 |
| local agreement hb=3 | **0.507** | 0.077 | **106** | **39.24** | 18.68 | 30.18 | 26.64 | **0.299** |

**hb=2 是拐点**:hb=3 只多 0.011 覆盖,而覆盖最小值、中→英 BLEU、英→中 chrF 都回退,延迟还更高。

两个必须记录的事实:

1. **英→中方向从未坏掉。** 在 `append_only` 下它的 chrF 是 4.58、BLEU 0.04,看起来像一个死掉的方向,并且据此做过范围决策;换成 local agreement 后同一份权重是 chrF 30.48、BLEU 27.66。**那是提交层的假象。**
2. **该消融不影响 demo 音频。** 交织 S2S 会话不走 `incremental_mt_rollout`,所以四个 run 的语音延迟与长度指标逐位相同(LAAL 130.7 ms、首次发音 1137 ms、长度比 1.299)。**要改善 demo 音频,必须把同类策略也用到交织会话上。**

---

## 6. Demo 样本

全部来自 `stage1_la_hb2_iter0002264_20260831T170422Z/audio/`。

| 样本 | 源时长 | CER | 译文覆盖 | 首次发音 | 语音 LAAL | 最大空白 | 语义长度比 | 音频/源 | 自然结束 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **emilia_zh_0004122419** | 4.12 s | **0.000** | **0.800** | **480 ms** | 1091 ms | 1760 ms | **1.201** | **1.102** | **True** |
| emilia_zh_0006795452 | 3.26 s | 0.000 | 0.778 | **320 ms** | −1254 ms | 1600 ms | 2.436 | 2.436 | False |
| emilia_zh_0006199435 | 6.06 s | 0.087 | 0.625 | 320 ms | −1303 ms | 2880 ms | 1.913 | 2.261 | True |
| emilia_zh_0005215832 | 3.28 s | 0.000 | 0.231 | 1600 ms | −32 ms | 320 ms | 1.651 | 2.366 | False |

### 6.1 首选试听:`emilia_zh_0004122419`

**这是当前唯一一条所有指标同时成立的样本。**

```
源音频      4.12 s
ASR 参考    他们普遍感觉到某种新的因素正在诞生
ASR 假设    他们普遍感觉到某种新的因素正在诞生        (逐字一致,CER 0.000)
译文参考    They generally feel that some new factor is being born
模型译文    They generally feel that some new factor is emerging   (语义正确)
输出音频    4.54 s  (源的 1.102 倍)
首次发音    480 ms
自然结束    是
```

[试听 WAV](../formal_gold_20260818T090515Z/free_running_gates/stage1_la_hb2_iter0002264_20260831T170422Z/audio/worker_07/emilia_zh_0004122419.wav)

### 6.2 其余三条

- [emilia_zh_0006795452](../formal_gold_20260818T090515Z/free_running_gates/stage1_la_hb2_iter0002264_20260831T170422Z/audio/worker_06/emilia_zh_0006795452.wav) —— 320 ms 开口、ASR 完美、译文 `Or do you have anything special you want to eat` 正确,但音频 2.44 倍过长且 **peak 1.223 削波**。
- [emilia_zh_0006199435](../formal_gold_20260818T090515Z/free_running_gates/stage1_la_hb2_iter0002264_20260831T170422Z/audio/worker_01/emilia_zh_0006199435.wav) —— 最长的一条(6.06 s),译文 `He is no longer waiting to catch mosquitoes but is like a ...` 方向正确但未译完,最大空白 2.88 s。
- [emilia_zh_0005215832](../formal_gold_20260818T090515Z/free_running_gates/stage1_la_hb2_iter0002264_20260831T170422Z/audio/worker_05/emilia_zh_0005215832.wav) —— ASR 完美但覆盖仅 0.231,且 **rms 0.0016 几乎听不见**,属于非静音判据过松的漏网样本。

---

## 7. 已达成 / 未达成

| 同传 S2ST 的必要条件 | 状态 | 证据 |
|---|---|---|
| 因果流式识别可用 | ✅ 中→英 | CER 0.069,离线 0.0547 |
| 源结束前开始发目标语音 | ✅ 中→英 8/8 | `pre_eos_speech` 1.000 |
| 首次发音亚秒级(源轴) | ✅ | p50 400 ms,min 320 ms |
| 已提交内容不回滚 | ✅ | 所有 rollback 计数为 0 |
| 语义 token 合法 | ✅ | `invalid_semantic_tokens` 0 |
| 产出非静音音频 | ⚠️ 中→英 8/8,英→中 6/8 | 且阈值过松 |
| **译文完整** | ❌ 覆盖 0.495 | 修复提交层后从 0.211 提升 |
| **输出时长≈输入时长** | ❌ 2.04 倍 | 离线继承(3.06 倍) |
| **自然收尾** | ❌ 0.50 | 离线 `missing_eos` 92/256 |
| 英文方向可用 | ❌ CER 0.396 | 拖累英→中 free-source |
| 墙钟实时 | ❌ | 未做 KV cache / 异步 TTS |

**一句话:同传的时序行为已经做到了,内容完整性和语音长度控制没有做到。**

---

## 8. 下一步(按信噪比排序)

1. **把 local agreement 用到交织 S2S 会话上。** 当前 demo 音频路径无条件追加每个 delta。`AppendOnlyDeltaCommitter`(同一份 `commit.py` 里已有,专为"模型被喂入已提交历史、只产出新 delta"设计)正是对应工具。纯推理侧,零训练。
2. **语音长度控制 —— 这是最大阻塞。** 输出 2.04 倍不是靠调采样能解决的。需要:(a) 每次提交的语义 token 数与该短语的文本长度挂钩,而不是固定 384 上限;(b) 训练侧把 `semantic_end_ce` / `semantic_end_margin` 与**长度比**直接关联;(c) 注意这是离线继承缺陷,单纯在流式阶段修可能治不了根。
3. **修紧非静音判据**:`rms > 1e-5` → `rms > 0.01`,并加削波检查 `peak <= 1.0`。当前判据同时放过了近静音和削波两类坏样本。
4. **英文因果 ASR(CER 0.396)。** 这是英→中方向的唯一阻塞。已通过质量门的 `stage_a_v9_bridgefreeze_formal8_20260817T130814Z`(`ctc_blank_ratio` 0.188 对现用 v1 的 0.80/0.97)是明确候选,代价是要重建 task pool 与 teacher cache。
5. **`empty_events` 均值 3.4**,`e_asr_structure_valid` 失败项,需要单独定位。
6. **扩大评测样本**:S2S 只有 8 条,`min` 类门禁检查在这个规模上不可信。

延迟仍不作为优化目标 —— 首次发音 400 ms、AP 0.404 已经足够好,当前瓶颈全在内容与长度。

---

## 9. 复现

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python

# 单测(39 项,CPU)
$PY -m pytest -q experiments/uniss_phase3_e2e_commit_policy_v1/tests

# 带 local agreement 的门禁(RUN_ROOT 内需先放好 SELECTION.json 与 CANDIDATE_HF_FINGERPRINT.json)
RUN_ID=<fresh> CANDIDATE_HF=<hf> CANDIDATE_CHECKPOINT=<ckpt> \
  MAX_S2S_SEMANTIC_TOKENS=384 UNISS_E2E_MT_HOLDBACK=2 \
  bash experiments/uniss_phase3_e2e_commit_policy_v1/scripts/run_gate_local_agreement_8gpu.sh

# 指标与本报告的数据源
$PY -m experiments.uniss_phase3_e2e_commit_policy_v1.evaluation.streaming_metrics \
  --run "label=<gate run root>" ... \
  --offline-baseline eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_phase3_offline_20260816T031129Z/baseline_summary.json \
  --output reports/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/streaming_s2st_metrics_v1/METRICS.json
```

单条门禁约 8 分钟(8 卡、16 样本);GPU 占用程序用
`tmux kill-session -t uniss_gpu_load_60` 停止、
`bash experiments/uniss_phase3_content_first_joint_s2st_v1/scripts/start_gpu_holder.sh` 恢复。
