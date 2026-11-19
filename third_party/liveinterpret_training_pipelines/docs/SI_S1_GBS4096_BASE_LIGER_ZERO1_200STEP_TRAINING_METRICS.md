# streaming_si S1 · GBS=4096 对照（A: zero2 无liger · B: zero1+liger，200 step）训练指标记录

> 记录日期：2026-10-14（创建）/ 2026-10-15（B 列完成，回填）
> 本文档结构/表格格式与 `GBS512_GBS4096_BASELINE_ABCD_200STEP_TRAINING_METRICS.md` 保持一致，
> 便于与既有 omnispeak 基线文档并排查阅。
>
> **A（GBS4096 基线，torchrun + ZeRO-2，无 liger）**：任务名 `si-s1-b4k-pgao`，本次实测。
> **B（GBS4096，torchrun + ZeRO-1 + liger）**：任务名 `si-s1-l4k-sori`，本次实测。
>
> ## 核心结论（详见第 2 节）
>
> **B 相对 A：稳态 s/it 8.80 → 7.10（↓19.3%），吞吐 465 → 577 样本/s（↑24.1%），
> 全程预估 1d 21h → 1d 12h17m（省约 8h43m）；稳态显存基本不变（50.96 → 50.54 GiB）；
> loss 曲线一致（step≥100 的 |Δloss| 均值 0.00704，双向波动，判定数值行为未变）。**
>
> 两个必须注意的限制：
> 1. **A→B 是三变量同时变**（ZeRO 2→1、liger 关→开、镜像底座），↓19.3% **不可单独归因**；
>    要拆分须补跑 C 列（zero1 + 无liger），见 2.4。
> 2. **liger 实际只融合了 talker**（RMSNorm+SwiGLU）。镜像内 liger-kernel 0.5.10 没有
>    `apply_liger_kernel_to_qwen3_next`，thinker 未融合 —— 而 S1 的 GPU 时间大头恰是冻结
>    thinker 4B 的前向，所以这里还有未取的收益，见 1.5.3。
>
> ## ⚠️ 与 GBS512_GBS4096_BASELINE_ABCD 文档的口径差异（不可跨文档比绝对值）
>
> 那份文档的 A-F 列是 **`qwen2_5_3b_omnispeak_a` 的 adapter+talker SFT**；本文档是
> **`streaming_si_4b`（Thinker Qwen3.5-4B 冻结 / Talker 0.6B）的 S1 阶段（Talker 发声预训练）**。
> 模型、训练目标、数据集、loss 口径全部不同，**仅 GBS=4096 这一个量相同**，
> 任何 s/it / 显存 / loss 的绝对值都不可跨文档比较，只能各自组内对照。
>
> 具体差异：
> 1. **可训参数不同**：那份是 `thinker_adapter + talker`（≈1.93B）；本文档是 `--trainable_parameters talker.`
>    （talker 0.6B，28L/1024d），thinker / audio_encoder 全冻结。
> 2. **loss 字段不同**：那份输出 `loss/token_acc/token_text_acc/token_acoustic_token/eos_boundary_top1_acc/
>    premature_eos_top1_rate`；本文档 S1 的 `text_w=0 speech_w=1`，只有
>    `loss`（= speech_loss）/ `speech_loss` / `speech_acc` / `speech_acc10`，**`text_loss` 恒为 0**
>    （`skip_text_logits=True`，thinker 的 lm_head 压根不算），无 eos/acoustic 系列指标。
> 3. **per_device 拆分不同**：那份 C/D 列是 `64×GA8` / `32×GA8`；本文档是 **`256×GA1`**。
>    不照搬 `32×GA8` 的原因：streaming_si 在每卡 64 时显存仅 21.6/141 GiB（见下方 1.4 节的
>    GBS1024 参照任务），`32×GA8` 会掉到约 11 GiB、更浪费，且 GA=8 把单 optimizer step 拉长 8 倍。
> 4. **总步数不同**：那份 C/D/E/F = 9364 步；本文档 = **18608 步**（2 epoch × 38,105,174 样本 / 4096）。
>
> ## ⚠️ MAX_STEPS 口径（本文档的关键前提）
>
> 照参考文档 C 列"`--force`（无 `TRAIN_STEP_LIMIT`）"、"跑满 200 步 `ktrain job stop` 人工停止"
> （实际运行 203/210/220 步）的做法，**A/B 两列均不设 `MAX_STEPS`**，按真实总步数 18608 走
> `cosine_with_min_lr` schedule，跑到 200 步后人工停。
>
> 本次曾先用 `MAX_STEPS=200` 跑过一轮（任务 `si-s1-g4k-t5od`，已废弃），该口径下 cosine 会把
> **整个衰减压缩进 200 步**：step200 时 lr 已从 1.4e-4 掉到 1.4e-5，loss 为 5.623；
> 而本文档 A 列同 step 为 **5.284**（lr 全程恒 1.400e-04）。**差 0.34 完全来自 schedule 而非配置**，
> 故 `MAX_STEPS=200` 的 loss 曲线不等于正式训练的前 200 步，已弃用、不录入本文档。
> （该轮的 s/it 与显存不受 schedule 影响，作为旁证记于 1.4 节。）

---

## 1.0 通用环境（各任务一致，差异已标注）

| 项 | A（GBS4096 基线，zero2 无liger） | B（GBS4096，zero1+liger） |
|---|---|---|
| 集群 | speech-h200-hyperpod（ns `speech-h200`） | 同 |
| 节点 | `hyperpod-i-08e3257c716579618`(master) + `hyperpod-i-03e7d1f6d95864e1a`(worker)（08e3+03e7） | 同 A（08e3 master + 03e7 worker） |
| 机型 / GPU | ml.p5e.48xlarge，8 × H200（141 GiB/卡），2 节点 | 同 |
| 镜像 | `…:p5-cuda12.8-torch2.8-te-v9-streaming-si-v1` | `…:p5-cuda12.8-torch2.8-te-v9-ray2-nsys-liger-streaming-si-v1` |
| 模型 | `streaming_si_4b`（Thinker Qwen3.5-4B / Talker 0.6B，`talker_preset=0.6b`，28L/1024d） | 同 |
| 基座 | `base_streaming_si_4b`（未训练基座，6.32B 参数含 Voxtral 音频塔） | 同 |
| 训练目标 | S1 Talker 发声预训练（文本 -> FSQ speech token） | 同 |
| 可训参数 | `--trainable_parameters talker.`（scope=talker，thinker / audio_encoder / talker_text_embed 冻结） | 同 |
| 数据集 | `streaming_si_s1`（Emilia-FSQ v3 parquet，2060 文件，20.5 GiB，38,105,174 样本） | 同 |
| 阶段开关 | `text_w=0 speech_w=1 cond_mode=text_only hidden_noise=0 talker_bridge_layer=-1` | 同 |
| `max_length` | 512 | 同 |
| `num_train_epochs` | 2 | 同 |
| LR / scheduler / warmup | 1.4e-4 / `cosine_with_min_lr`(`min_lr_rate` 0.1) / `warmup_ratio=0` | 同 |
| `split_dataset_ratio` | 0 | 同 |
| `lazy_tokenize` | true | 同 |
| `group_by_length` | false | 同 |
| `weight_decay` / `max_grad_norm` | 0.01 / 1.0 | 同 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 同 |
| 加速核 | fla 0.4.2 / causal_conv1d 1.5.2（4B 线性注意力走加速核，`SI_CODE_CHECK` 两节点均已验证） | 同 |
| 停止方式 | 跑满 200 步（实际 210）后 `ktrain job stop` 人工停止 | 同 A（实际 210 步） |
| 代码 commit/来源 | `b6638d2` + 本地未提交的 ray/liger 改动（同步进训练） | 同 A |

## 1.1 任务级配置对比表

| 参数 / 指标 | A（GBS4096 基线，zero2 无liger） | B（GBS4096，zero1+liger） |
|---|---|---|
| 任务 | `si-s1-b4k-pgao` | `si-s1-l4k-sori`（前一次 `si-s1-l4k-40fc` 因 liger 断言失败废弃，见 1.5.4） |
| 任务 yaml | `ktrainjob_streaming_si_s1_gbs4096_base.yaml` | `ktrainjob_streaming_si_s1_gbs4096_liger_z1.yaml` |
| framework / 后端 | pytorch(torchrun) / deepspeed | 同 |
| DeepSpeed 引擎 | **zero2**（`common_args.sh` 默认） | **zero1** |
| 节点数 / GPU | 2×8 | 2×8 |
| 并行策略 | DP=16 | DP=16 |
| 微批 / 累积 | `PER_DEVICE=256`×`GA=1` | 同 A |
| **有效batch(GBS)** | **4096** | **4096**（同 A） |
| liger 开关 | **关**（镜像内无 liger_kernel 包） | **`STREAMING_SI_LIGER_OPS=1`** |
| liger 版本 | 不适用 | **0.5.10** |
| liger 实际融合范围 | 不适用 | **仅 talker**（RMSNorm+SwiGLU）；thinker 因 0.5.10 无 `qwen3_next` patch 未融合 |
| liger 挂载方式 | 不适用 | `custom/streaming_si/integration/liger_ops.py`（见 1.5 节） |
| `MAX_STEPS` | 不设（`--max_steps -1`） | 同 A |
| 总步数 | 18608 | 18608 |
| `logging_steps` | 10 | 10 |
| `save_steps` / `save_total_limit` | 2000 / 5 | 同 |
| `eval_strategy` | no | 同 |
| `DATASET_NUM_PROC` / `DATALOADER_NUM_WORKERS` | 32 / 4 | 同 |
| `SI_CODE_CHECK` | 1（两节点通过） | 1（额外断言 liger 真的挂上） |
| W&B / 上报 | `--report_to none` | 同 |
| 输出目录 | `/nvme/output/junyaoyang/streaming_si_gbs4096_base/s1_tts` | `/nvme/output/junyaoyang/streaming_si_gbs4096_liger_z1/s1_tts` |
| MFU 日志 | 未采集 | 未采集 |
| Map 预处理耗时 | 约 4m49s（38,105,174 样本） | 未单独记录（同数据/同 `DATASET_NUM_PROC=32`，预期与 A 相当） |
| 实际运行step数 | 210（人工停止） | 210（人工停止） |

> **口径说明**：A→B 的变量有 **两个**（ZeRO stage 2→1、liger 关→开），外加镜像底座变化
> （liger 包只在 ray2-nsys-liger 底座里，torch / TE / EFA 与基线镜像相同）。
> 因此 A↔B **不是单变量对照**，若需拆分二者贡献须另跑 "zero1 无liger" 或 "zero2+liger" 列。
> 本组按用户要求的"4096 baseline vs 4096+liger+zero1"口径组织，与参考文档 C/D 列
> （也是同时含 zero1+liger）的组织方式一致。

## 1.2 性能数据对比表（稳态）

> ### ⚠️ 本文档出现两种 s/it 口径，引用时务必说明是哪一种
>
> * **分段 s/it**：相邻两条 logging 的 `elapsed_time` 差 ÷ 步数 → **真实稳态，结论用这个**
> * **累计 s/it**：HF Trainer 日志里的 `train_speed(s/it)` 原值 → 从第 1 步起的累计均值，
>   被首步编译/预热（各列 18-60s）长期摊薄，比稳态大 1-1.3 秒
>
> 逐 step 两种口径分别见 1.3.7（累计）与 1.3.8（分段）。
>
> ### ⚠️ C 列是单机，s/it 不可与其他列并排比较（每卡样本量是 2 倍），见文首说明

| 指标 | A | B | **C**(单机) | D | **E**(全优化) |
|---|---|---|---|---|---|
| 拓扑 | 双机 DP16 | 双机 DP16 | **单机 DP8** | 双机 DP16 | 双机 DP16 |
| ZeRO / liger / SKIP | 2 / — / — | **1 / ✓ / —** | 2 / — / **✓** | 2 / — / — | **1 / ✓ / ✓** |
| 微批 × 累积 | 256×1 | 256×1 | **256×2** | 256×1 | 256×1 |
| 每optimizer step样本数 | 4096 | 4096 | 4096 | 4096 | 4096 |
| **稳态s/it（190→200 分段）** | **8.80** | **7.10** | 12.60 | **8.70** | **4.70** |
| 稳态s/it（150→200 均值） | 8.92 | 7.30 | 12.62 | 8.92 | **4.80** |
| 累计s/it@200（含冷启动） | 10.003 | 8.395 | 12.637 | 10.001 | **4.882** |
| **每秒样本吞吐（稳态）** | **465** | **577** | 325 | **471** | **871** |
| 稳态MFU | 未采集（见注） | 未采集 | 未采集 | 未采集 | 未采集 |
| **显存@200(GiB)** | **50.96** | **50.54** | 47.75 | **50.96** | **40.60** |
| 显存@step1(GiB) | 46.41 | 41.17 | 46.34 | 46.41 | 40.56 |
| **200step耗时** | **33m21s** | **27m59s** | 42m7s | **33m20s** | **16m16s** |
| 首步耗时（含编译/预热） | 60s | 57s | 18s | 57s | 20s |
| **全程预估**（HF `remaining`@200） | 2d 3h 8m | 1d 18h 56m | 2d 16h 37m | 2d 3h 8m | **1d 0h 58m** |
| **全程预估**（按稳态重算） | **44.5 h** | 36.3 h | — | **44.5 h** | **24.0 h** |
| 实际运行step数 | 210 | 210 | 200 | 220 | **300** |
| 总步数（2 epoch） | 18608 | 18608 | 18608 | 18608 | 18608 |

### 1.2.1 关键对照

| 对照 | 变量 | 稳态 s/it | 吞吐 | 显存 | 性质 |
|---|---|---|---|---|---|
| **E vs D** | zero1 + liger + SKIP（三项叠加） | 8.70 → **4.70**  **↓46.0%** | 471 → **871**  **↑85.1%** | −10.36 GiB | **优化总收益** |
| **E vs B** | **仅 SKIP_THINKER**（两列均 zero1+liger） | 7.10 → **4.70**  **↓33.8%** | 577 → **871**  **↑51.1%** | −9.94 GiB | **严格单变量，本组最大单项** |
| B vs A | zero1 + liger（双变量） | 8.80 → 7.10  ↓19.3% | 465 → 577  ↑24.1% | −0.42 GiB | 双变量，不可单独归因 |
| D vs A | 仅 profiler 开关 | 8.80 → 8.70  −1.1% | — | ±0.00 | **验证 profiler 零开销 + 可复现** |

> **MFU 为何未采集**：C/D/E 三列已启用 DeepSpeed `flops_profiler`，但其 `fwd flops` 是按
> module 公式估算，对本链路大量 Triton / 自定义核（`fla` 的 `chunk_gated_delta_rule`、
> `causal_conv1d`、liger 的 RMSNorm/SwiGLU、flash-attn）统计不全；且 SKIP_THINKER 列中
> thinker 的 latency 已为 0 而 FLOPs 公式仍计入，导致数值虚高。
> 故本文档**只采用 profiler 的 latency 分解，不用其 FLOPs 推算 MFU**（详见 3.5）。

## 1.3 训练指标对比（step 1~200，每10步一条logging，按指标分表）

> 字段口径：A/B 均为 deepspeed 链路 + `swift sft`，HF Trainer logging 原始日志。
> S1 阶段 `text_w=0`，故 **`text_loss` 恒为 0.0、无 `text_acc`**（`res.logits is None`，
> 见 `custom/streaming_si/model/modeling_streaming_si.py::compute_si_losses`）；
> `loss` 与 `speech_loss` 数值上相等（`loss = 0×text_loss + 1×speech_loss + aux_loss`，aux=0）。
> 参考文档中的 `token_acc` / `token_text_acc` / `token_acoustic_token` / `eos_boundary_top1_acc` /
> `premature_eos_top1_rate` 系列为 omnispeak 链路专属，**本模型不产出这些字段**，故本文档不设对应小节。

### 1.3.1 loss

<!-- METRIC_loss_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 10.79004 | 10.79105 | 10.79042 | 10.79004 | 10.79105 |
| 10 | 8.39849 | 8.42686 | 10.30129 | 8.39708 | 8.42029 |
| 20 | 7.55676 | 7.56520 | 8.12540 | 7.54483 | 7.55761 |
| 30 | 7.30798 | 7.32147 | 7.55058 | 7.29443 | 7.32626 |
| 40 | 7.13203 | 7.15080 | 7.35707 | 7.12367 | 7.16016 |
| 50 | 6.95257 | 6.97669 | 7.24452 | 6.95755 | 6.98703 |
| 60 | 6.75921 | 6.77451 | 7.16760 | 6.77124 | 6.81462 |
| 70 | 6.54584 | 6.52339 | 7.10174 | 6.55262 | 6.57910 |
| 80 | 6.28564 | 6.26702 | 7.02834 | 6.31076 | 6.31908 |
| 90 | 6.05520 | 6.03686 | 6.94065 | 6.10096 | 6.07633 |
| 100 | 5.85870 | 5.86485 | 6.82354 | 5.89795 | 5.88064 |
| 110 | 5.72981 | 5.73871 | 6.67721 | 5.75660 | 5.73936 |
| 120 | 5.61982 | 5.62169 | 6.51123 | 5.63118 | 5.62511 |
| 130 | 5.54452 | 5.56217 | 6.37357 | 5.55682 | 5.55791 |
| 140 | 5.48255 | 5.49249 | 6.24482 | 5.48945 | 5.50036 |
| 150 | 5.43842 | 5.45039 | 6.14221 | 5.43385 | 5.44686 |
| 160 | 5.40580 | 5.40405 | 6.04915 | 5.39756 | 5.41180 |
| 170 | 5.36347 | 5.36843 | 5.96512 | 5.36354 | 5.37747 |
| 180 | 5.32377 | 5.33009 | 5.88760 | 5.32928 | 5.33653 |
| 190 | 5.31267 | 5.31082 | 5.82483 | 5.30127 | 5.31047 |
| 200 | 5.28384 | 5.28765 | 5.76980 | 5.28302 | 5.28611 |
<!-- METRIC_loss_END -->

### 1.3.2 speech_loss（S1 专属；= loss，text_w=0）

<!-- METRIC_speech_loss_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 10.79017 | 10.79114 | 10.79007 | 10.79017 | 10.79114 |
| 10 | 8.39862 | 8.42700 | 10.30138 | 8.39722 | 8.42043 |
| 20 | 7.55711 | 7.56555 | 8.12533 | 7.54521 | 7.55798 |
| 30 | 7.30798 | 7.32148 | 7.55062 | 7.29444 | 7.32627 |
| 40 | 7.13200 | 7.15078 | 7.35720 | 7.12363 | 7.16014 |
| 50 | 6.95267 | 6.97680 | 7.24447 | 6.95766 | 6.98715 |
| 60 | 6.75920 | 6.77451 | 7.16772 | 6.77124 | 6.81462 |
| 70 | 6.54584 | 6.52337 | 7.10171 | 6.55262 | 6.57910 |
| 80 | 6.28566 | 6.26705 | 7.02842 | 6.31078 | 6.31911 |
| 90 | 6.05525 | 6.03690 | 6.94072 | 6.10102 | 6.07637 |
| 100 | 5.85847 | 5.86463 | 6.82349 | 5.89773 | 5.88041 |
| 110 | 5.72983 | 5.73872 | 6.67718 | 5.75662 | 5.73937 |
| 120 | 5.61970 | 5.62157 | 6.51124 | 5.63106 | 5.62499 |
| 130 | 5.54434 | 5.56198 | 6.37337 | 5.55663 | 5.55773 |
| 140 | 5.48250 | 5.49245 | 6.24488 | 5.48942 | 5.50032 |
| 150 | 5.43822 | 5.45020 | 6.14220 | 5.43366 | 5.44665 |
| 160 | 5.40590 | 5.40415 | 6.04906 | 5.39767 | 5.41191 |
| 170 | 5.36322 | 5.36820 | 5.96487 | 5.36330 | 5.37722 |
| 180 | 5.32371 | 5.33002 | 5.88755 | 5.32921 | 5.33647 |
| 190 | 5.31260 | 5.31074 | 5.82444 | 5.30119 | 5.31039 |
| 200 | 5.28384 | 5.28764 | 5.76973 | 5.28302 | 5.28610 |
<!-- METRIC_speech_loss_END -->

### 1.3.3 speech_acc（top-1，FSQ speech token，词表 6565）

<!-- METRIC_speech_acc_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 0.012% | 0.012% | 0.012% | 0.012% | 0.012% |
| 10 | 2.084% | 1.840% | 1.089% | 2.048% | 1.871% |
| 20 | 4.249% | 4.142% | 3.727% | 4.347% | 4.537% |
| 30 | 6.044% | 6.001% | 4.772% | 6.115% | 5.898% |
| 40 | 6.711% | 6.557% | 5.371% | 6.742% | 6.574% |
| 50 | 7.351% | 7.247% | 5.932% | 7.296% | 7.197% |
| 60 | 7.989% | 7.913% | 6.331% | 7.912% | 7.752% |
| 70 | 8.505% | 8.579% | 6.573% | 8.537% | 8.446% |
| 80 | 9.097% | 9.115% | 6.767% | 9.046% | 9.035% |
| 90 | 9.606% | 9.617% | 7.012% | 9.383% | 9.506% |
| 100 | 10.144% | 10.062% | 7.366% | 9.934% | 10.007% |
| 110 | 10.356% | 10.303% | 7.693% | 10.278% | 10.332% |
| 120 | 10.704% | 10.702% | 8.217% | 10.706% | 10.708% |
| 130 | 10.918% | 10.863% | 8.555% | 10.919% | 10.840% |
| 140 | 11.090% | 11.105% | 8.870% | 11.125% | 11.058% |
| 150 | 11.268% | 11.216% | 9.085% | 11.274% | 11.251% |
| 160 | 11.354% | 11.376% | 9.337% | 11.394% | 11.277% |
| 170 | 11.487% | 11.481% | 9.549% | 11.503% | 11.367% |
| 180 | 11.715% | 11.677% | 9.866% | 11.693% | 11.630% |
| 190 | 11.645% | 11.692% | 10.032% | 11.785% | 11.703% |
| 200 | 11.721% | 11.715% | 10.142% | 11.760% | 11.734% |
<!-- METRIC_speech_acc_END -->

### 1.3.4 speech_acc10（top-10）

<!-- METRIC_speech_acc10_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 0.133% | 0.131% | 0.133% | 0.133% | 0.131% |
| 10 | 6.023% | 6.179% | 1.866% | 6.031% | 6.111% |
| 20 | 10.730% | 10.657% | 8.306% | 11.034% | 10.631% |
| 30 | 13.479% | 13.404% | 10.827% | 13.667% | 13.329% |
| 40 | 15.605% | 15.401% | 12.108% | 15.696% | 15.193% |
| 50 | 17.614% | 17.285% | 13.543% | 17.486% | 17.128% |
| 60 | 19.972% | 19.810% | 14.353% | 19.732% | 19.332% |
| 70 | 22.561% | 22.765% | 15.138% | 22.469% | 22.259% |
| 80 | 25.094% | 25.253% | 16.015% | 24.825% | 24.806% |
| 90 | 27.231% | 27.361% | 17.190% | 26.697% | 27.005% |
| 100 | 29.126% | 28.995% | 18.633% | 28.702% | 28.883% |
| 110 | 30.390% | 30.259% | 20.223% | 30.103% | 30.275% |
| 120 | 31.588% | 31.552% | 21.954% | 31.474% | 31.521% |
| 130 | 32.377% | 32.174% | 23.334% | 32.246% | 32.201% |
| 140 | 33.026% | 32.900% | 24.615% | 32.941% | 32.794% |
| 150 | 33.552% | 33.403% | 25.628% | 33.616% | 33.447% |
| 160 | 33.897% | 33.936% | 26.553% | 34.027% | 33.799% |
| 170 | 34.385% | 34.347% | 27.487% | 34.391% | 34.221% |
| 180 | 34.884% | 34.823% | 28.350% | 34.825% | 34.735% |
| 190 | 34.964% | 35.027% | 28.999% | 35.137% | 35.014% |
| 200 | 35.311% | 35.316% | 29.581% | 35.343% | 35.295% |
<!-- METRIC_speech_acc10_END -->

### 1.3.5 grad_norm

<!-- METRIC_grad_norm_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 105.3740 | 105.3501 | 105.4091 | 105.3737 | 105.3507 |
| 10 | 1.7134 | 1.9040 | 4.3240 | 1.6034 | 4.7035 |
| 20 | 0.9162 | 1.1155 | 6.6261 | 1.0457 | 1.4412 |
| 30 | 0.6743 | 0.4762 | 1.0307 | 0.6691 | 0.5934 |
| 40 | 0.8194 | 0.8852 | 1.0179 | 0.5681 | 0.9124 |
| 50 | 0.7118 | 0.9698 | 0.7955 | 0.9487 | 0.9513 |
| 60 | 0.9185 | 0.7889 | 0.9131 | 1.2578 | 1.0081 |
| 70 | 0.8910 | 1.8701 | 1.1187 | 0.7774 | 1.2465 |
| 80 | 1.1267 | 1.3023 | 1.1587 | 2.6584 | 1.3657 |
| 90 | 1.1669 | 1.6254 | 1.0900 | 1.2483 | 0.9208 |
| 100 | 1.7906 | 1.9287 | 1.2677 | 1.3158 | 1.1632 |
| 110 | 1.1552 | 1.2603 | 1.1500 | 1.0285 | 1.4974 |
| 120 | 1.2193 | 1.1333 | 1.1795 | 1.3867 | 1.8939 |
| 130 | 2.0107 | 1.1204 | 1.1761 | 1.5306 | 1.7413 |
| 140 | 2.0240 | 1.4579 | 1.4399 | 1.6088 | 1.5129 |
| 150 | 1.7329 | 1.3287 | 1.9466 | 1.5188 | 1.9662 |
| 160 | 1.1969 | 1.5397 | 1.6556 | 1.5919 | 1.9925 |
| 170 | 1.1914 | 1.5753 | 1.9297 | 1.6752 | 1.4673 |
| 180 | 1.5496 | 1.1250 | 1.7402 | 1.2752 | 1.4996 |
| 190 | 1.4378 | 1.1765 | 1.4332 | 1.5352 | 1.1449 |
| 200 | 1.6796 | 1.1754 | 1.1927 | 1.9241 | 1.5304 |
<!-- METRIC_grad_norm_END -->

### 1.3.6 显存(GiB)

<!-- METRIC_mem_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 46.41 | 41.17 | 46.34 | 46.41 | 39.04 |
| 10 | 46.42 | 41.18 | 46.35 | 46.42 | 39.05 |
| 20 | 46.42 | 41.79 | 46.35 | 46.42 | 39.12 |
| 30 | 46.42 | 41.79 | 46.35 | 46.42 | 39.12 |
| 40 | 46.42 | 41.79 | 46.35 | 46.42 | 39.12 |
| 50 | 47.57 | 44.43 | 47.75 | 47.57 | 40.60 |
| 60 | 47.57 | 44.43 | 47.75 | 47.57 | 40.60 |
| 70 | 50.46 | 50.04 | 47.75 | 50.46 | 40.60 |
| 80 | 50.46 | 50.04 | 47.75 | 50.46 | 40.60 |
| 90 | 50.48 | 50.05 | 47.75 | 50.48 | 40.60 |
| 100 | 50.48 | 50.05 | 47.75 | 50.48 | 40.60 |
| 110 | 50.48 | 50.05 | 47.75 | 50.48 | 40.60 |
| 120 | 50.48 | 50.05 | 47.75 | 50.48 | 40.60 |
| 130 | 50.48 | 50.05 | 47.75 | 50.48 | 40.60 |
| 140 | 50.48 | 50.05 | 47.75 | 50.48 | 40.60 |
| 150 | 50.95 | 50.54 | 47.75 | 50.95 | 40.60 |
| 160 | 50.95 | 50.54 | 47.75 | 50.95 | 40.60 |
| 170 | 50.95 | 50.54 | 47.75 | 50.95 | 40.60 |
| 180 | 50.96 | 50.54 | 47.75 | 50.96 | 40.60 |
| 190 | 50.96 | 50.54 | 47.75 | 50.96 | 40.60 |
| 200 | 50.96 | 50.54 | 47.75 | 50.96 | 40.60 |
<!-- METRIC_mem_END -->

### 1.3.7 s/it（train_speed，累计均值口径）

<!-- METRIC_sit_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 60.028 | 57.254 | 17.544 | 56.931 | 11.852 |
| 10 | 17.953 | 16.408 | 12.978 | 17.654 | 5.590 |
| 20 | 13.989 | 12.478 | 12.745 | 13.862 | 5.227 |
| 30 | 13.295 | 11.746 | 12.712 | 13.218 | 5.129 |
| 40 | 12.741 | 11.205 | 12.680 | 12.697 | 5.083 |
| 50 | 12.665 | 11.109 | 12.722 | 12.626 | 5.101 |
| 60 | 11.979 | 10.419 | 12.709 | 11.953 | 5.064 |
| 70 | 11.761 | 10.188 | 12.684 | 11.741 | 5.020 |
| 80 | 11.391 | 9.818 | 12.672 | 11.375 | 5.000 |
| 90 | 11.300 | 9.714 | 12.664 | 11.282 | 4.976 |
| 100 | 11.155 | 9.562 | 12.646 | 11.141 | 4.950 |
| 110 | 10.937 | 9.342 | 12.637 | 10.924 | 4.934 |
| 120 | 10.772 | 9.173 | 12.640 | 10.759 | 4.928 |
| 130 | 10.595 | 8.993 | 12.637 | 10.584 | 4.918 |
| 140 | 10.460 | 8.855 | 12.628 | 10.451 | 4.905 |
| 150 | 10.366 | 8.760 | 12.638 | 10.361 | 4.906 |
| 160 | 10.209 | 8.601 | 12.634 | 10.206 | 4.897 |
| 170 | 10.183 | 8.574 | 12.637 | 10.182 | 4.895 |
| 180 | 10.194 | 8.587 | 12.639 | 10.194 | 4.890 |
| 190 | 10.068 | 8.465 | 12.638 | 10.067 | 4.887 |
| 200 | 10.003 | 8.395 | 12.637 | 10.001 | 4.882 |
<!-- METRIC_sit_END -->

### 1.3.8 分段 s/it（相邻两条 logging 的真实步耗时，去掉冷启动摊薄）

> 累计均值口径（1.3.7）会被首步的 60s 长期摊薄，看稳态须用本表。

<!-- METRIC_sit_seg_START -->
| step 区间 | A | B | C | D | E |
|---|---|---|---|---|---|
| 1→10 | 13.33 | 11.89 | 12.44 | 13.33 | 4.89 |
| 10→20 | 10.00 | 8.60 | 12.50 | 10.00 | 4.90 |
| 20→30 | 11.90 | 10.20 | 12.60 | 12.00 | 4.90 |
| 30→40 | 11.10 | 9.60 | 12.60 | 11.10 | 4.90 |
| 40→50 | 12.30 | 10.70 | 12.90 | 12.30 | 5.20 |
| 50→60 | 8.60 | 7.00 | 12.70 | 8.60 | 4.90 |
| 60→70 | 10.40 | 8.80 | 12.50 | 10.50 | 4.70 |
| 70→80 | 8.80 | 7.20 | 12.60 | 8.80 | 4.90 |
| 80→90 | 10.60 | 8.90 | 12.60 | 10.50 | 4.80 |
| 90→100 | 9.80 | 8.20 | 12.50 | 9.90 | 4.70 |
| 100→110 | 8.80 | 7.20 | 12.50 | 8.80 | 4.80 |
| 110→120 | 9.00 | 7.30 | 12.70 | 8.90 | 4.80 |
| 120→130 | 8.40 | 6.80 | 12.60 | 8.50 | 4.80 |
| 130→140 | 8.70 | 7.10 | 12.50 | 8.70 | 4.80 |
| 140→150 | 9.10 | 7.40 | 12.80 | 9.10 | 4.90 |
| 150→160 | 7.80 | 6.20 | 12.50 | 7.90 | 4.80 |
| 160→170 | 9.80 | 8.20 | 12.70 | 9.80 | 4.80 |
| 170→180 | 10.40 | 8.80 | 12.70 | 10.40 | 4.80 |
| 180→190 | 7.80 | 6.20 | 12.60 | 7.80 | 4.90 |
| 190→200 | 8.80 | 7.10 | 12.60 | 8.70 | 4.70 |
<!-- METRIC_sit_seg_END -->

### 1.3.9 elapsed（自训练循环起点）

<!-- METRIC_elapsed_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 1m 0s | 57s | 18s | 57s | 12s |
| 10 | 2m 60s | 2m 44s | 2m 10s | 2m 57s | 56s |
| 20 | 4m 40s | 4m 10s | 4m 15s | 4m 37s | 1m 45s |
| 30 | 6m 39s | 5m 52s | 6m 21s | 6m 37s | 2m 34s |
| 40 | 8m 30s | 7m 28s | 8m 27s | 8m 28s | 3m 23s |
| 50 | 10m 33s | 9m 15s | 10m 36s | 10m 31s | 4m 15s |
| 60 | 11m 59s | 10m 25s | 12m 43s | 11m 57s | 5m 4s |
| 70 | 13m 43s | 11m 53s | 14m 48s | 13m 42s | 5m 51s |
| 80 | 15m 11s | 13m 5s | 16m 54s | 15m 10s | 6m 40s |
| 90 | 16m 57s | 14m 34s | 18m 60s | 16m 55s | 7m 28s |
| 100 | 18m 35s | 15m 56s | 21m 5s | 18m 34s | 8m 15s |
| 110 | 20m 3s | 17m 8s | 23m 10s | 20m 2s | 9m 3s |
| 120 | 21m 33s | 18m 21s | 25m 17s | 21m 31s | 9m 51s |
| 130 | 22m 57s | 19m 29s | 27m 23s | 22m 56s | 10m 39s |
| 140 | 24m 24s | 20m 40s | 29m 28s | 24m 23s | 11m 27s |
| 150 | 25m 55s | 21m 54s | 31m 36s | 25m 54s | 12m 16s |
| 160 | 27m 13s | 22m 56s | 33m 41s | 27m 13s | 13m 4s |
| 170 | 28m 51s | 24m 18s | 35m 48s | 28m 51s | 13m 52s |
| 180 | 30m 35s | 25m 46s | 37m 55s | 30m 35s | 14m 40s |
| 190 | 31m 53s | 26m 48s | 40m 1s | 31m 53s | 15m 29s |
| 200 | 33m 21s | 27m 59s | 42m 7s | 33m 20s | 16m 16s |
<!-- METRIC_elapsed_END -->

### 1.3.10 learning_rate

<!-- METRIC_learning_rate_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 10 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 20 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 30 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 40 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 50 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 60 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 70 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 80 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 90 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 100 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 110 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 120 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 130 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 140 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 150 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 160 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 170 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 180 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 190 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
| 200 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 | 1.400e-04 |
<!-- METRIC_learning_rate_END -->

> `warmup_ratio=0` + 总步数 18608 的 cosine，前 210 步只衰减了不到 0.01%，
> HF Trainer 四舍五入后全程显示 1.400e-04。**这正是本文档废弃 `MAX_STEPS=200`
> 口径的原因**：那种口径下同样的 210 步内 lr 会从 1.4e-4 掉到 1.4e-5。

### 1.3.11 remaining_time（HF Trainer 按累计均值速率外推全程 18608 步）

> **本组最重要的工程指标**：直接回答"这个配置跑完 2 epoch 要多久"。
> 口径 = `(18608 - step) × 累计s/it`（已逐条校验，与日志完全吻合），
> 因此它继承了累计均值的冷启动摊薄 —— 早期偏大，随 step 增加单调收敛到真值。
> 参考文档 A/B 外推 75070/74160 步、C/D/E/F 外推 9364 步，**本文档外推 18608 步**，
> 跨文档不可比。

<!-- METRIC_remaining_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 12d 22h 15m 41s | 12d 7h 55m 29s | 3d 18h 40m 47s | 12d 6h 15m 13s | 2d 13h 15m 31s |
| 10 | 3d 20h 44m 41s | 3d 12h 45m 56s | 2d 19h 2m 43s | 3d 19h 12m 11s | 1d 4h 52m 47s |
| 20 | 3d 0h 13m 55s | 2d 16h 25m 39s | 2d 17h 48m 19s | 2d 23h 34m 34s | 1d 2h 59m 28s |
| 30 | 2d 20h 36m 29s | 2d 12h 36m 54s | 2d 17h 35m 55s | 2d 20h 12m 37s | 1d 2h 28m 8s |
| 40 | 2d 17h 42m 57s | 2d 9h 47m 29s | 2d 17h 24m 6s | 2d 17h 29m 10s | 1d 2h 13m 3s |
| 50 | 2d 17h 17m 21s | 2d 9h 16m 7s | 2d 17h 34m 59s | 2d 17h 5m 15s | 1d 2h 17m 37s |
| 60 | 2d 13h 43m 8s | 2d 5h 40m 52s | 2d 17h 28m 50s | 2d 13h 35m 11s | 1d 2h 5m 24s |
| 70 | 2d 12h 33m 50s | 2d 4h 27m 43s | 2d 17h 18m 59s | 2d 12h 27m 42s | 1d 1h 50m 58s |
| 80 | 2d 10h 37m 26s | 2d 2h 31m 42s | 2d 17h 13m 10s | 2d 10h 32m 39s | 1d 1h 43m 58s |
| 90 | 2d 10h 7m 30s | 2d 1h 57m 58s | 2d 17h 8m 23s | 2d 10h 2m 3s | 1d 1h 35m 46s |
| 100 | 2d 9h 20m 49s | 2d 1h 9m 25s | 2d 17h 0m 58s | 2d 9h 16m 31s | 1d 1h 26m 54s |
| 110 | 2d 8h 11m 46s | 2d 0h 0m 1s | 2d 16h 55m 55s | 2d 8h 7m 49s | 1d 1h 21m 10s |
| 120 | 2d 7h 19m 11s | 1d 23h 6m 29s | 2d 16h 54m 45s | 2d 7h 15m 13s | 1d 1h 18m 26s |
| 130 | 2d 6h 22m 60s | 1d 22h 9m 32s | 2d 16h 51m 41s | 2d 6h 19m 40s | 1d 1h 14m 27s |
| 140 | 2d 5h 39m 43s | 1d 21h 25m 29s | 2d 16h 46m 53s | 2d 5h 36m 56s | 1d 1h 9m 40s |
| 150 | 2d 5h 9m 4s | 1d 20h 54m 51s | 2d 16h 47m 49s | 2d 5h 7m 18s | 1d 1h 9m 23s |
| 160 | 2d 4h 18m 49s | 1d 20h 4m 37s | 2d 16h 44m 29s | 2d 4h 17m 59s | 1d 1h 5m 38s |
| 170 | 2d 4h 9m 19s | 1d 19h 54m 53s | 2d 16h 43m 22s | 2d 4h 8m 47s | 1d 1h 4m 12s |
| 180 | 2d 4h 10m 60s | 1d 19h 57m 19s | 2d 16h 41m 55s | 2d 4h 10m 46s | 1d 1h 1m 56s |
| 190 | 2d 3h 30m 40s | 1d 19h 18m 37s | 2d 16h 39m 30s | 2d 3h 30m 19s | 1d 1h 0m 11s |
| 200 | 2d 3h 8m 51s | 1d 18h 55m 42s | 2d 16h 37m 10s | 2d 3h 8m 23s | 1d 0h 57m 51s |
<!-- METRIC_remaining_END -->

> **A 列全程预估：约 2 天 3 小时**（step200 口径）。
> 注意该值仍含冷启动摊薄（累计 10.003 s/it vs 稳态 8.80 s/it）；
> **按稳态 8.80 s/it 重算，真实全程约 `18408 × 8.80 ≈ 1d 21h`**，
> 比 HF 显示的 2d 3h 少约 6 小时。两个口径都列出，取用时注意区分。

### 1.3.12 epoch

<!-- METRIC_epoch_START -->
| step | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 30 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 40 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 50 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 60 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 70 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 80 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 90 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 110 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 120 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 130 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 140 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 150 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 160 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 170 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 180 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 190 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 200 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
<!-- METRIC_epoch_END -->

> 200 步仅为 2 epoch 的 **1.07%**（200/18608），故 epoch 停在 0.02。
> 本组对照只看启动段的效率与收敛趋势，不代表最终收敛质量。

---

## 1.4 旁证：GBS1024 参照任务与废弃的 MAX_STEPS=200 轮次

本组对照之外另有两次运行，**不属于 A/B 对照，仅作参照**：

| 项 | GBS1024 参照 | MAX_STEPS=200 废弃轮 |
|---|---|---|
| 任务 | `si-s1-tr-5jc4` | `si-s1-g4k-t5od` |
| 配置 | `PER_DEVICE=64`×GA1，GBS=1024，zero2，无liger | `PER_DEVICE=256`×GA1，GBS=4096，zero2，无liger |
| `MAX_STEPS` | 200（同样被压缩，loss 不可用） | 200（同上） |
| `logging_steps` | 50 | 10 |
| 稳态 s/it | 2.28（step150→200） | 10.00（step10→20） |
| 累计 s/it@200 | 2.619 | — |
| 显存@200 | 21.61 / 141（15%） | 46.42（@40步） |
| 吞吐 | 约 449 样本/s | 约 410 样本/s |
| 实际步数 | 200（跑满 `MAX_STEPS` 自然结束） | 40（人工停止） |

> 两轮的 **loss 曲线均因 `MAX_STEPS=200` 压缩 schedule 而不可用**；
> s/it 与显存不受 schedule 影响，可作性能参照。
> GBS1024（449 样本/s，21.6 GiB）→ GBS4096（465 样本/s，50.96 GiB）：
> **batch 放大 4 倍，吞吐只 +3.6%，显存涨 2.4 倍**。

---

## 1.5 liger 在 streaming_si 上的挂载方式（B 列专属说明）

### 1.5.1 为什么不能用 `--use_liger_kernel`

swift 的 SFT 路径**完全不处理** `use_liger_kernel`（只有 `rlhf_trainers/grpo_trainer.py`、
`gkd_trainer.py` 等 RLHF 路径用它）；真正应用它的是 transformers：

```python
# transformers/trainer.py L516-525
if self.args.use_liger_kernel:
    if is_liger_kernel_available():
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model, **kernel_config)
```

而 `_apply_liger_kernel_to_instance` 是拿 **`model.config.model_type`** 去
`MODEL_TYPE_TO_APPLY_LIGER_FN` 查表的。streaming_si 导出目录的 config 是：

```
model_type    = streaming_si
architectures = ['StreamingSIForConditionalGeneration']
thinker_arch  = qwen3_5
```

查表必然 miss —— liger **只打一条 warning 就返回，不报错、不生效**（静默空转）。
A 列日志里的 `use_liger_kernel=False` 也印证了该参数在本链路的实际状态。

### 1.5.2 为什么 streaming_si 挂得上（与 qwen_omnispeak 不同）

`custom/qwen_omnispeak/modeling_qwen_omnispeak.py` L99-111 的注释说明：omnispeak 的
talker/adapter 是 project-local 的 `_RMSNorm` / `_MLPProj` 子类，**liger 对官方 HF 类的
monkey-patch 永远碰不到**，所以那边只能用 `QWEN_OMNISPEAK_LIGER_OPS=1` 直接调底层
autograd Function（`LigerRMSNormFunction` / `LigerSiLUMulFunction` / `liger_cross_entropy`）。
参考文档 C/D 列的 liger 收益实际来自这个旁路，**不是** `--use_liger_kernel`。

streaming_si 相反，两塔的 norm / MLP / attention **全是官方 HF 类**：

| 塔 | 实现 | 出处 |
|---|---|---|
| talker（S1 训的就是它） | `transformers.models.qwen3.Qwen3Model` | `model/model_si.py` L273 `self.decoder = Qwen3Model(...)` |
| thinker（冻结，但前向照跑） | `Qwen3NextRMSNorm` / `Qwen3NextRMSNormGated` / `Qwen3NextMLP` / `Qwen3NextAttention` | `model/qwen3_5_compat.py` L177-221，全部取自 `transformers.models.qwen3_next` |

所以只需按官方 patch 函数把这两个 HF 模块 patch 掉，**不必手写 autograd Function 调用**。

### 1.5.3 实现与时机

`custom/streaming_si/integration/liger_ops.py`，在 `StreamingSILoader.get_model()` 里
**`from_pretrained` 之前**调用：

```python
# integration/register_streaming_si.py::get_model
from .liger_ops import apply_liger_to_streaming_si
apply_liger_to_streaming_si()
model = StreamingSIForConditionalGeneration.from_pretrained(model_dir, config=config, **model_kwargs)
```

必须在建模型之前的原因：liger 对 RMSNorm 这类是**替换模块里的类对象**
（`modeling_xxx.XxxRMSNorm = LigerRMSNorm`），只影响此后新建的实例；对 MLP/attention 则多是
替换 `forward` 方法（对已有实例也生效）。两者混用意味着"建完模型再 patch"只有一半生效，
且失效的那半同样是静默的。

挂载目标与开关：

| 目标 | patch 函数 | liger 0.5.10 是否有 | 必需性 | 融合算子 |
|---|---|---|---|---|
| talker(Qwen3) — **S1 训的就是它** | `apply_liger_kernel_to_qwen3` | ✅ 有 | **必需** | `rms_norm=True, swiglu=True` |
| thinker(Qwen3Next) — 冻结，仅前向 | `apply_liger_kernel_to_qwen3_next` | ❌ **没有** | 可选 | `rms_norm=True, swiglu=True` |

> **2026-10-15 实测（镜像 `…-ray2-nsys-liger-streaming-si-v1`）**：
> liger-kernel **0.5.10**（该版本包内**没有 `__version__` 属性**，版本号须走
> `importlib.metadata.version('liger-kernel')` 取），共 19 个 `apply_liger_kernel_to_*`：
> `gemma/gemma2/gemma3/gemma3_text/glm4/granite/llama/llava/mistral/mixtral/mllama/olmo2/
> paligemma/phi3/qwen2/qwen2_5_vl/qwen2_vl/qwen3/qwen3_moe`
> —— **有 `qwen3`，没有 `qwen3_next`**。
>
> 所以 B 列的实际融合范围是 **只有 talker**。这对 S1 恰好是可接受的：`scope=talker`，
> 只有 talker 在训练（前向+反向都吃融合核），thinker 全程冻结、只有前向，
> 少一块前向收益不影响实验成立。**要让 thinker 也融合，需升级 liger 并重建镜像。**

**`cross_entropy` / `fused_linear_cross_entropy` 一律关闭**，原因：
1. talker 不走 `Qwen3ForCausalLM` 的 loss —— 它用自己的 `llm_decoder`（`nn.Linear`）出 logits，
   loss 在 `model/modeling_streaming_si.py::compute_si_losses` L289-295 用
   `F.cross_entropy(sl[valid].float(), st[valid])` 算，patch HF 的 CE 路径对本模型无效。
2. speech token 词表只有 **6565**（FSQ `3^8=6561` + 4 special，见 `model/base_model.py` L302-307），
   融合 CE 的收益主要来自超大词表时省掉 logits 物化，这里意义不大。
3. `compute_si_losses` 需要**已物化的 logits** 算 `speech_acc` / `speech_acc10`（`topk(10)`），
   与 fused-linear-CE 的"不物化 logits"设计直接冲突。

### 1.5.4 防静默空转的断言（已实际拦下一次故障）

`SI_CODE_CHECK=1` 时（`pipelines/streaming_si/ktrain_entry.sh`），训练前断言 liger 真的挂上：

```python
if os.environ.get("STREAMING_SI_LIGER_OPS") == "1":
    from streaming_si.integration.liger_ops import (apply_liger_to_streaming_si,
                                                    required_patch_fns, _liger_version)
    from liger_kernel.transformers import monkey_patch as _mp
    # 只断言「必需」目标（talker=Qwen3）。thinker(Qwen3Next) 可选：全程冻结、只有前向，
    # 且 liger 0.5.10 还没有 apply_liger_kernel_to_qwen3_next。
    _miss = [f for f in required_patch_fns() if getattr(_mp, f, None) is None]
    assert not _miss, f"... liger-kernel {_liger_version()} 缺必需的 {_miss}"
    _done = apply_liger_to_streaming_si()
    assert _done, "STREAMING_SI_LIGER_OPS=1 但没有任何目标被融合"
    print(f"[code-check] liger {_liger_version()} -> {_done}")
```

**这个断言在首次提交 B 列时就拦下了一次真实故障**（任务 `si-s1-l4k-40fc`，已废弃）：

```
[code-check] fla 0.4.2 / causal_conv1d 1.5.2
AssertionError: STREAMING_SI_LIGER_OPS=1 但 liger-kernel ? 缺 ['apply_liger_kernel_to_qwen3_next']
[streaming_si/ktrain][FATAL] SI_CODE_CHECK 失败 @ si-s1-l4k-40fc-master-0
```

当时断言把 thinker 也当作必需，于是任务在**排队 21 小时后一启动就失败**。经探测镜像确认
liger 0.5.10 确实没有该函数后，已把 talker 改为必需、thinker 改为可选（见 1.5.3 表），
`si-s1-l4k-sori` 为修正后的重跑。

价值在于对比：若没有这个断言，这一轮会**正常跑完 200 步**，但 liger 实际一个算子都没融合
（`_apply_liger_kernel_to_instance` 查表 miss 只打 warning），B 列就会退化成"zero1 单变量"
而我们浑然不知 —— 正是 1.5.1 节描述的坑。
`common_args.sh` 的日志行也会打印 `ds=<stage> liger=<0|1>` 便于事后核对
（A 列实测输出：`... ga=1 ray=false ds=zero2 liger=0`）。

---

## 2. 结论

### 2.1 A 列（GBS4096 基线）

待 B 列完成后补充 A↔B 对照结论。当前已确立的 A 列基线数值：

- 稳态 **8.80 s/it**（step190→200）/ **8.92 s/it**（step150→200 均值），累计@200 **10.003**
- 显存 **50.96 / 141 GiB（36%）**，阶梯式从 46.41 增长而来
- 吞吐 **约 465 样本/s**
- 200 step 耗时 **33m21s**
- **全程（2 epoch / 18608 步）预估：HF 口径 2d 3h 9m，按稳态重算约 1d 21h**
- step200 loss **5.28384**，`speech_acc` **11.721%**，`speech_acc10` **35.311%**
- 收敛正常：loss 10.790→5.284，`speech_acc` 0.012%→11.72%，`grad_norm` 首步 105 后迅速收敛到 1-2 区间

### 2.2 B 列（GBS4096 + zero1 + liger）

- 稳态 **7.10 s/it**（step190→200）/ **7.30 s/it**（step150→200 均值），累计@200 **8.395**
- 显存 @step1 **41.17**、@200 **50.54 / 141 GiB（36%）**
- 吞吐 **约 577 样本/s**
- 200 step 耗时 **27m59s**
- **全程（2 epoch / 18608 步）预估：HF 口径 1d 18h 56m，按稳态重算约 1d 12h 17m**
- step200 loss **5.28765**，`speech_acc` **11.715%**，`speech_acc10` **35.316%**
- liger 实际融合范围：**仅 talker 的 RMSNorm + SwiGLU**（thinker 因 liger 0.5.10 无
  `apply_liger_kernel_to_qwen3_next` 未融合，两节点日志均已确认，见 1.5.3 / 1.5.4）

### 2.3 A↔B 对照

**吞吐提升显著，稳态显存基本不变。**

| 维度 | A | B | 变化 |
|---|---|---|---|
| 稳态 s/it | 8.80 | **7.10** | **↓19.3%** |
| 吞吐（样本/s） | 465 | **577** | **↑24.1%** |
| 200 step 耗时 | 33m21s | **27m59s** | **↓5m22s** |
| 全程预估（稳态口径） | 约 1d 21h | **约 1d 12h 17m** | **↓约 8h 43m** |
| 显存 @200 | 50.96 | 50.54 | ↓0.42（-0.8%） |
| 显存 @step1 | 46.41 | **41.17** | ↓5.24（-11.3%） |

**数值一致性（关键）**：两列 loss 曲线高度吻合，`step≥100` 的 `|Δloss|` 均值仅 **0.00704**，
范围 `-0.00185 ~ +0.01765`，且**双向波动而非单向偏移**；step200 的
`speech_acc` 差 -0.006pp、`speech_acc10` 差 +0.005pp。这个量级与 bf16 下
归约顺序差异（zero1 vs zero2 的梯度规约路径不同）+ Triton 融合核的浮点结合律差异相符，
**可判定 zero1 + liger(talker) 未改变训练数值行为**，吞吐收益是"免费"的。

**⚠️ 归因限制**：A→B 同时改了 **ZeRO stage（2→1）**、**liger（关→开，仅 talker）**、
**镜像底座**三项，因此 **↓19.3% 不可单独归因于 liger 或 zero1 任何一方**。
可做的定性推断：
- zero1 相对 zero2 省掉了梯度分片的 reduce-scatter/all-gather 往返（本任务可训参数只有
  talker 0.6B，优化器状态分片收益小、通信省下的更可能是主要项）；
- liger 只融合了 talker 的 RMSNorm/SwiGLU，而 talker 是唯一在训练的塔（前向+反向都受益），
  但 S1 的 GPU 时间大头是**冻结 thinker 4B 的前向**，那部分完全没被融合。

要拆分二者贡献，需补跑第三列（见 2.4）。

### 2.4 待办

- [x] A 列（GBS4096 / zero2 / 无liger）200 步
- [x] B 列（GBS4096 / zero1 / liger）200 步
- [ ] **C 列（GBS4096 / zero2 / 无liger / SKIP_THINKER）** —— 代码与单测已完成，等节点（见第 3 节）
- [ ] **拆分 zero1 与 liger 的各自贡献**：补跑 D 列 = GBS4096 / **zero1** / **无liger**
      （与 B 列唯一差异 = liger，可直接做单变量对照；与 A 列差异 = zero stage）
- [ ] 若要让 thinker 也吃到融合核：升级 liger-kernel（需 ≥ 支持 `qwen3_next` 的版本）并重建镜像。
      **但优先级应低于 C 列** —— 见 3.1：S1 的 thinker 前向本身就是无用计算，
      与其让一段无用计算跑得更快，不如直接不跑它
- [ ] nsys 采样定位"GBS 放大 4 倍吞吐仅 +3.6%"的真实瓶颈（见 1.2 节注），
      以及实测 thinker 前向占比，用于校验 3.1 的 FLOPs 估算
- [ ] 显存仍只用 36%（50.5/141 GiB），C 列跑完后显存应进一步下降，
      那时再评估放大 per_device（此时放大才是喂给真正在训练的 talker）

---

# 3. S1 计算结构分析与 SKIP_THINKER 优化（C 列）

## 3.1 S1 的可训参数只有 467M，但每步白跑 4B thinker 前向

### 3.1.1 阶段定义（源码权威表）

```
pipelines/streaming_si/common_args.sh L5-11
# 阶段 -> 可训参数（括号内为源 streaming-si-0.6b 的阶段名与 trainer）：
#   talker        S1 / S3      只训 talker.*
#   talker_bridge S3-w / S4-w  只训 talker.thinker_proj.* + talker.fusion.*
#   thinker_audio S2           audio_proj.* + thinker_model.*
#   joint         S4           audio_proj.* + thinker_model.* + talker.*
#   stream_lora   S5           LoRA(thinker/talker) + 4 行流式 token + thinker_proj/fusion
```

**S1 训 talker，S2 才训 thinker**，两者是可并行的独立支线。A/B 两列实测日志印证：
`scope=talker` + `--trainable_parameters talker.` + `text_w=0 speech_w=1 cond_mode=text_only`。

### 3.1.2 可训参数精算（`--trainable_parameters talker.`）

`talker_preset=0.6b` 只是**命名**（取自 Qwen3-0.6B 的结构：28L/1024d/16-8头/head_dim128/ffn3072），
实际参数远小于 0.6B —— 因为 Qwen3-0.6B 那个"0.6B"里很大一块是 15 万词表的 embedding，
而 talker 把词表换成了 **6565** 的 speech token 词表（FSQ `3^8=6561` + 4 special）。

| 模块 | 参数量 | 说明 |
|---|---|---|
| `talker.decoder` | **447.19M** | `Qwen3Model` 28L/1024d |
| └ 28 × layer | 440.47M | 每层 15.73M（attn 6.29M + mlp 9.44M） |
| └ `embed_tokens` | 6.72M | `6565×1024`，**它本身就是 speech_embedding**（零冗余） |
| └ final norm | 1K | |
| `talker.llm_decoder` | 6.72M | speech-token head `1024×6565`，不与 embedding tie |
| `talker.thinker_proj` | 6.30M | GatedBridge 2560→1024 |
| `talker.text_proj` | 2.62M | Linear 2560→1024 |
| `talker.fusion` | 4.20M | TalkerFusion |
| **合计可训** | **467.02M = 0.467B** | |

对照全局：基座总参数 **6.32B**（含 Voxtral 音频塔）→ **可训占比仅 7.4%**，
冻结 **5.85B**（thinker 4B + Voxtral + `talker_text_embed`）。

### 3.1.3 thinker 的前向结果在 S1 被整段丢弃

```custom/streaming_si/model/model_si.py L862-867
    def _fuse_talker_condition(self, bridge_hidden, text_hidden):
        cond = self.talker.thinker_proj(bridge_hidden)
        if getattr(self.config, 'cond_mode', 'full') == 'text_only':
            cond = torch.zeros_like(cond)          # <<< thinker hidden 整段置零
        text = self.talker.text_proj(text_hidden.to(cond.dtype))
        return self.talker.fusion(cond, text)
```

而 talker 真正用的 text 路与 thinker **无关**：

```custom/streaming_si/model/model_si.py L902-904
        # text 路用 talker 自己的 embedding 副本（见 __init__ 的说明）：
        # 这里只是查表、不过 thinker 的层，故与 thinker 解耦后 cond 路仍完整保留语义。
        text_embeds, _ = self.gather_answer_hidden(self.talker_text_embed(input_ids), answer_mask)
```

源码注释自己也写明了设计意图（L883-886）：
> `cond_mode='text_only'`（**S1-0**）：把 `cond` 路整段置零，只留 text 路。
> 用途：S1-0 让 talker 先学「怎么说」（FSQ token 的语言模型），此时**不依赖 thinker hidden**。

**既然"不依赖 thinker hidden"，那 `SI.forward` L914 那次无条件的 `forward_thinker(...)`
在 S1 就是纯浪费**：4B 模型跑完整前向，结果经 `thinker_proj` 后被 `zeros_like` 覆盖。

注意作者已为 `text_loss_weight=0` 做过一次同类优化（`skip_text_logits` 省掉 `lm_head` + 文本 CE，
理由写的就是"它们的结果会被乘 0，纯属浪费"），但**同样的逻辑没有应用到 thinker 主干本身**。

### 3.1.4 收益估算

单步计算构成：

| 部分 | 参数量 | 梯度 | 结果是否被用到 |
|---|---|---|---|
| Voxtral 音频塔 | 冻结 | 否 | S1 无音频输入（`encode_s1` 不产出 `audio_inputs`），根本不调用 |
| **thinker 主干** | **4B，冻结** | **否** | **❌ 被 `zeros_like` 丢弃** |
| thinker `lm_head` | 冻结 | 否 | ✅ 已被 `skip_text_logits` 跳过 |
| talker | 0.467B，训练 | 是 | ✅ |

按 FLOPs 粗估（前向 `2NT`、反向 `4NT`）：
`thinker 空转 2×4B : talker 前向+反向 6×0.467B ≈ 8 : 2.8`，
**thinker 空转可能占单步计算的约 74%**。真实比例取决于序列长度分布与 kernel 效率，需实测。

这也解释了 1.2 节的疑问：**GBS 放大 4 倍吞吐只涨 3.6%**，因为放大的 batch 主要在喂那个空转的 4B 前向，
而真正做反向传播的只有 467M。

## 3.2 SKIP_THINKER 实现

开关 `SI_SKIP_THINKER=1`（env → `STAGE_ENV` → `config.skip_thinker_when_text_only`，
沿用 `SI_COND_MODE` 的注入模式），默认关 = 行为零回归。

### 3.2.1 为什么数学上无损（不是近似）

`GatedBridge`（`thinker_proj`）全部由 **`bias=False` 的 Linear + RMSNorm** 组成：

```custom/streaming_si/model/model_si.py L228-238
        self.norm_in = RMSNorm(in_dim, eps=eps)
        self.up = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Linear(in_dim, out_dim, bias=False)
        self.out = nn.Linear(out_dim, out_dim, bias=False)
        self.norm_out = RMSNorm(out_dim, eps=eps)
    def forward(self, x):
        x = self.norm_in(x)
        return self.norm_out(self.out(self.up(x) * torch.sigmoid(self.gate(x))))
```

因此 `thinker_proj(0) == 0`，与被 `zeros_like` 覆盖的结果**逐位相同**。
把 thinker 的真实输出换成同形状零张量后，**talker 看到的 condition 完全不变**。
（`hidden_noise` 也不受影响：`rms(zeros)=0` ⇒ 噪声为 0；且 S1 已设 `SI_HIDDEN_NOISE=0.0`。）

### 3.2.2 双层守卫

**入口层**（`pipelines/streaming_si/ktrain_entry.sh`）按 `SI_STAGE` 提前拦误用：
非 `s1` 直接 `die`。不检查 `SI_COND_MODE` 的原因：它由 `train_tts.sh` 导出，
在 `ktrain_entry.sh` exec 之前还没设。

**模型层**（`SI.can_skip_thinker()`）校验 7 个前置条件，任一不满足即照常跑 thinker：

| 条件 | 理由 |
|---|---|
| `config.skip_thinker_when_text_only` | 显式打开，默认关 |
| `cond_mode == 'text_only'` | 只有此时 cond 路才被置零；S3/S3-w/S4 的 `full` 真的要用 |
| `skip_text_logits`（即 `text_w==0`） | 否则 `lm_head` 要用 `last_hidden_state` 算文本 CE |
| `not use_cache` | 推理路径要 `thinker_out.past_key_values` |
| `talker_bridge_layer < 0` | ≥0 时要取 `hidden_states[idx]` 中间层，零张量替代未实现 |
| `audio_inputs is None` | 有音频说明不是纯 TTS 的 S1，音频特征必须经 thinker 注入 |

### 3.2.3 验证结果（2026-10-15，容器内 CPU 对拍）

`custom/streaming_si/tests/test_skip_thinker.py` —— **11 passed**：

| 测试 | 断言 |
|---|---|
| `test_skip_thinker_bitwise_equal_under_text_only` | **`torch.equal(speech_logits)` 逐位相等**（不是 `allclose`） |
| `test_gated_bridge_maps_zero_to_zero` | `GatedBridge(0) == 0`，等价性的数学依据 |
| `test_full_mode_unaffected_by_flag` | **`full` 模式开启开关后结果不变** → S3/S3-w/S4 回归保护 |
| `test_can_skip_thinker_guards`（6 组） | 每个前置条件的守卫行为 |
| `test_mid_layer_bridge_blocks_skip` | `talker_bridge_layer>=0` 时不跳 |
| `test_zero_thinker_out_matches_real_shape` | 零张量的 shape / dtype / device 与真实输出一致 |

**全量回归**：`custom/streaming_si` **44 passed / 28 skipped / 0 failed**（115s）。

## 3.3 C 列状态：等节点

任务 `si-s1-skt-snwd` 已提交，`Pending` 排队中。配置与 A 列**只差 `SI_SKIP_THINKER=1`**
一个变量（同为 zero2 / 无 liger / 同镜像 / `256×GA1` / lr 1.4e-4），故 C↔A 是干净的单变量对照。
yaml：`ktrain/ktrainjob_streaming_si_s1_gbs4096_skipthinker.yaml`，preflight PASS。

**集群资源阻塞**（2026-10-15 05:52）：kenxsong 的 7 个 `kx-next-w02~w08` 小任务
（每个 2 卡，21:02 同一秒批量提交）把 `08e3` 铺满 8/8、`03e7` 占 6/8；
全集群仅余 3 张散卡，凑不出我们需要的 2×8。

> 这种"2 卡小任务铺满整机"的占用模式对 8 卡整机需求特别不利 —— 对方只占 25% 容量，
> 我们就完全无法调度。本次实验过程中已因此被阻塞多次（详见 4. 过程记录）。

**跑起来后的正确性自检**：C 列 loss 曲线应与 A 列**逐步几乎一致**（数学等价，仅剩 bf16
归约顺序差异）。若出现系统性偏移，说明跳过逻辑有误，须排查而非采信其加速数字。

## 3.4 ⚠️ 首次提交失败：rendezvous 超时死锁（提交策略教训）

C 列首个任务 `si-s1-skt-snwd`（与 S2 baseline `si-s2-b4k-c6kr` 同时）在排队
**7h08m / 6h49m 后双双 Failed**，根因**不是代码**，是提交策略：

**master 侧**
```
torch.distributed.DistStoreError: Timed out after 901 seconds waiting for clients.
1/2 clients joined.
```

**worker 侧（init 容器 `init-pytorch`）**
```
nslookup: can't resolve 'si-s1-skt-snwd-master-0': Name does not resolve
waiting for master
→ Exit Code: 1
```

**完整死锁链条**：

1. master 抢到 `08e3` 的 8 卡，启动 torchrun、开 TCPStore 等 worker
2. worker 调度不上（`03e7` 被一个 4 卡 workspace 占着，只剩 4 卡 < 8）
3. master 等满 **901 秒**（torch elastic 静态 rendezvous 默认超时）后抛错退出
4. master Pod 退出 ⇒ 其 headless service 的 DNS 记录消失
5. worker 的 init 容器永远 `nslookup` 不到 master ⇒ Exit Code 1
6. 两个 Pod 双双 Error ⇒ KtrainJob Failed

**旁证**：失败后 `kubectl get pods -o wide` 显示 master 与 worker **都在 `08e3`**。
两个 Pod 各请求 8 卡、节点仅 8 卡，正常不可能同时落上 —— 这是 master 超时退出释放
8 卡后调度器才放 worker 进来，属失败的**结果**而非原因。

### 教训

**`framework: pytorch`（torchrun）的多节点任务是「全有或全无」**：master 与 worker 必须
在 901 秒内同时到位。因此：

* **资源不足 `numNodes` 台整机时，不要提交多节点 torchrun 任务**。否则 master 会先抢到
  一台、空占 8 卡干等 15 分钟，最后双输（白占卡 + 任务失败 + 重新排队）。
* `submit.sh` 的 step 3 预检只校验白名单节点 **Ready 且未 cordon**，
  **不检查 GPU 是否空闲**（preflight_streaming_si.py 会 WARN 但不阻止），
  所以"提交后会排队"这个 WARN 要当真 —— 排队到的是 Pod，不是原子的 gang。
* 本次代价：`08e3` 8 张卡被空占约 15 分钟，两个任务合计 14 小时排队作废。
* 可选缓解：若集群长期凑不出多台整机，改用单机 8 卡配置
  （`numNodes: 1` + `per_device×GA×8 = 4096`），单机无 worker、不存在该死锁；
  代价是与现有 A/B 列（双机 DP16）不能直接比绝对值，需另开一组单机列。
