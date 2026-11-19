# 训练性能优化方向与问题记录

> 记录日期：2026-08-03（修订于 2026-08-04）
> 数据来源：`hist-s2s-1node-9hgb`（speech-h200 集群，Succeeded）全量日志本地分析
> 日志落盘：`/tmp/pytorchjob-collect/hist-s2s-1node-9hgb/`（含 `master-ts.log` 带时间戳版本）
> 本文档按训练环节记录：现状 → 已落地优化（小节标题标 ✅）→ 待讨论项。
> 章节：第 0 章基准快照 → 第 1 章训练前数据 → 第 2 章训练中速度 → 第 3 章 eval/ckpt 开销。
> 注意力后端与通信-计算重叠等参数取值见 `pipelines/bridge_alignment/PERF_TUNING.md`；原第 3 章关于"FA2 无法释放"的 4D mask / FlexAttention 分析因前提不成立已删除。

---

## 0. 基准快照（本次分析的事实底座）

| 项 | 值 |
|---|---|
| 任务 | `hist-s2s-1node-9hgb`，1 节点 8×H200（ml.p5e.48xlarge），单 master |
| 模型 | `qwen2_5_3b_omnispeak_a`（总 6.73B 参数，可训练 2.07B / 30.7%；冻结 thinker 36 层 6.79GB + audio tower，训 talker 28 层 3.83GB + bridge 0.23GB） |
| 并行 / batch | TP=PP=CP=1，DP=8；MBS=8，GBS=1024（grad_accum=16）；max_length=4096（当前环境已下调至 2048，见 2.2）；packing=false，padding_free=false |
| attention | `--attention_backend unfused`（当前环境已改 `flash` 并验证可用，见 2.2） |
| 数据 | 252.7 万行 manifest；`load_from_cache_file=true`；`lazy_tokenize=True`；dataset_num_proc=150；dataloader workers=16 / prefetch=8 / persistent |
| 训练规模 | 2468 步，总时长 **8h16m**（17:12:43 → 次日 01:29:35） |
| 稳态 step | **12.08 s/step**（min 11.55 / max 13.66，波动极小） |
| 显存 | 稳态 **95.5 / 141 GiB ≈ 68%**（余量 ~45GB） |
| 吞吐 | 84.8 samples/s |
| 周期开销 | eval 每 200 步一次 ≈ 158s（24 iter，首个 eval iter 冷启动 48s），占 **~6.4%**；ckpt 每 500 步 ~45–60s，占 <1% |

**核心结论**：训练全程 step 速度波动 ±0.5s 且显存 68%，说明是**计算受限、而非喂数受限**；dataloader 不是瓶颈。已落地的两大计算侧改进：① `attention_backend` 由 unfused 改 flash（消除 unfused 最慢路径）；② 显存与重计算用 `recompute_granularity=none` + `max_length=2048` 优雅解决 OOM（见 2.2）。batch 内 padding 浪费率未经实测，待讨论（见 2.3）。

---

## 1. 训练前的数据 map / flattening indices

### 1.1 现状时间线（均取自 `master-ts.log`）

| 时刻 | 事件 |
|---|---|
| 17:09:40 | launch，容器启动 |
| 17:10:02 | 装入两个数据集加速补丁（见下） |
| 17:10:12 | main 启动 |
| 17:11:28 | routing-tag 注入 **2,527,660 行**（`without flatten_indices`，日志在 add_column 结束时打印） |
| 17:11:29 | routing-tag 注入 25,531 行 |
| 17:11:51 | train/val 切分（split_ratio=0.01）完成 |
| 17:12:37 | VRAM profile → **17:12:43 训练开始（step 0）** |

数据准备全程约 **3 分钟**，其中约 99s 为读取/解析 2.5M 行 jsonl manifest（arrow 缓存命中后的 preprocess 重放）。

### 1.2 已解决的问题（不要重复投入）

历史上该环节是最大黑洞：**`add_column` 打 routing tag 时触发整表 `flatten_indices()` 全量重写**（datasets 3.6.0 在 `_indices` 非空时强制 flatten，且未传 num_proc，单进程 ~2600 examples/s；unist 1899 万行实测单 rank 约 2 小时，且 8 个 rank 各跑一遍、缓存膨胀到 784G）。已通过 `custom/dataset_routing_tag_patch.py` 两个补丁解决：

1. **routing-tag fast patch**：常量列与行序无关，直接对底层 Arrow 表横向拼接、保留 indices，跳过 flatten（O(全表字节数) → O(行数)）。
2. **parallel flatten_indices patch（兜底）**：若别处仍触发 flatten，补默认 `num_proc`（默认 = `CONTAINER_CPUS/8`，可用 `SWIFT_FLATTEN_NUM_PROC` 覆盖），避免退化成单进程。

本次日志中未出现 `[flatten_indices] no num_proc given` 触发记录，确认 flatten 已被彻底消除。

### 1.3 (✅已完成) 问题 1（保护失效，需修）：WDS 数据上超长音频过滤"假工作"

**背景**：`drop_row_if_invalid_audio`（`custom/dataset.py:652`）是训练前（数据切分前）唯一真删除样本的闸门，负责丢弃缺失/超长音频，防止训练期显存 OOM。**该环节必须保留，优化目标是提效，不是删除。**

**原问题（保护失效）**：对 `wds://` URI，`_resolve_audio_path`（`custom/codec_tokenizer.py:92`）解析出的是 **tar shard 路径**而非成员音频；`_read_audio_duration_ms` 用 `soundfile.info()` 嗅探 tar 文件**必然抛异常 → 返回 None → 按"读不出就放行"策略保留样本**。即：

- **保护失效**：对当前主力 WDS manifest，超长音频过滤实际不生效，靠下游 `MAX_LENGTH` 兜底；
- **性能空耗**：冷启动 map 时，2.5M 行 × 每行 2 个音频 = 数百万次 `urlparse + stat + sf.info` 打开 tar 嗅探失败，全是无用 IO，是该阶段最大的单点浪费。

**解决方案（已落地 ✅）**：manifest 每条记录已**预置音频时长字段**（`source_duration_ms` / `target_duration_ms`），无需解析音频。将消费侧（`drop_row_if_invalid_audio` / `_read_audio_duration_ms`）从"嗅探 tar 取时长"改为"直接读 `source_duration_ms` / `target_duration_ms`（必要时配 `codec_len` 估目标帧数）做阈值比较"——零 IO、成员级精确，保护真正生效。

**manifest 字段实证（2026-08-03 抽样，两个 yaml 指向的 manifest 协议一致）**：

| 字段 | 含义 | 示例（unist / common_voice） |
|---|---|---|
| `source_duration_ms` | 源（输入）语音时长(ms) | 1120 / 5501 |
| `target_duration_ms` | 目标（输出）语音时长(ms) | 1080 / 4032 |
| `target_codec_codes` / `codec_len` | 目标 codec 码 & 帧数 | `codec_len=14` |
| `audios` / `target_audio` | `wds://...tar?offset=&length=&format=` | 含成员 offset/length，可定位成员 |
| `source_audio_encoded_bytes` | 源音频编码字节数 | 19079 |
| `provenance.source_bicodec_len` / `target_bicodec_len` | 双码本长度 | 56 / 54 |

- 抽样来源：
  - unist（FSX 直读）：`fsx://zeyangsong/unist/converted/unist_qwen3_s2s_v2/manifests/manifest-train-*.jsonl`（198 个 train manifest，每文件 10 万行；`pytorch_job_unist_1node.yaml:160` 的 `BRIDGE_ALIGNMENT_DATA`）
  - common_voice（S3）：`s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/audio/manifests/manifest_version=common_voice_bidirectional_ms_swift_codec_20260727_v1/task=speech_to_speech_translation/split=train/manifest-*.jsonl`（`pytorch_job.yaml:96` 的 `BRIDGE_ALIGNMENT_DATA`）

**变更风险**：过滤从"对 WDS 不生效"变为真过滤后，**会真实丢弃一部分超长样本**，训练集规模与长度分布发生变化（首次启用应打印丢弃计数确认比例）；legacy 独立文件路径行为不变。

### 1.4 问题 2（低优先级）：每次启动仍花 ~99s 解析 jsonl

根因：2.5M 行 manifest 文本解析；arrow 缓存命不中时要全量重解析。方向：① 保持 `load_from_cache_file=true` 复用同节点 arrow 缓存；② **严禁在 map 阶段中途 kill 任务**——NVMe 上 datasets arrow 缓存会被截断成半截文件，下次复用报 `unexpected EOF`（处置：先 `df -h /nvme` 排除 ENOSPC，再删损坏缓存目录重提）；③ 需要时把 jsonl 预烘焙成 arrow 一次。

---

## 2. 训练中速度与 GPU 负载

### 2.1 现状

- 稳态 **12.08 s/step**，84.8 samples/s；step 1 = 54s（编译/预热）→ step 20 = 13.7s → 稳态 12s。
- 显存 68%，GPU 未被喂数拖慢（否则 step 速度会大幅抖动）。
- **有效序列长度远低于 `max_length`**：按 6N 公式（满 4096 token）反推单卡 TFLOP/s 达 H200 峰值 175%，物理不可能，说明有效平均长度约 2000–2500；这并不直接等于 padding 浪费率（见 2.3）。

### 2.2 (✅已完成) 显存与重计算：recompute_granularity=none 优雅解决 OOM

**现象（原问题）**：原始配置 `recompute=none` + `attention_backend=unfused` + 长序列 S×S 注意力矩阵物化集中，单卡显存 80G→140G→OOM（chm6 崩溃于 139.78GiB）。talker 28 层此前靠 HF gradient-checkpoint（`mcore_talker_loss.py:441-443` 的 `_grad_checkpoint`）包成一段压制激活，未重算时峰值约 89GB。

**解决（已落地，经 loep / qtyx 任务验证）**：H200 显存充裕（141GB），不靠复杂的重算/K 调参或冻结段 `no_grad` 改造，而是用更直接的方式消除 OOM 根因：

- `attention_backend` unfused→flash：thinker 实为纯 causal+padding，flash 可用，消除 unfused 最慢路径（含其 S×S 物化）；
- `max_length` 4096→2048：序列算力与长序列注意力集中度双降；
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（写入任务 yaml 运行环境层，符合训练参数/运行环境分层规则）：降碎片；
- `lengths` 列修复：修复 dataset `lengths` 字段缺失/错误，配合 `lazy_tokenize` 采样正常；
- 保持 `--recompute_granularity none`：显存充裕下不重算即最快，无需 talker K 调参或 MBS 重算等改造。

**验证**：loep 任务（flash + max_length=2048 + recompute=none + expandable_segments）跑到 step30 显存 ~97.46GiB（最差 rank；原 139.78GiB 崩溃），loss 20.83→13.12，grad_norm 762→36，**无 OOM，训练健康**。

> 注：`overlap_param_gather=true` 已开但重叠退化、`overlap_grad_reduce=false` 暂锁（限制见 `pipelines/bridge_alignment/PERF_TUNING.md` 二章，收口见该文档行动项）。

### 2.3 padding 浪费与 group_by_length（待验证，不做 packing 改造）

**事实**：`packing=false` + dynamic padding 下，序列只 pad 到 **batch 内最长样本**（≤max_length）。padding 浪费率 = `1 − 平均长度/batch 内最大长度`，取决于 batch 内长度方差，**当前未经实测**。

**多模态训练普遍不支持 packing**：拼接破坏逐样本 codec↔semantic 1:1 对齐（codec loss 对 packing/padding_free 有硬 guard），且 talker 的 block-causal 4D mask 难以用拼接流表达。因此**不做 `packing=true` 改造**——原文档中"packing 带来性能提升"的论断缺乏数据与理论支撑，已删除。

**可选小风险项：`--group_by_length` 按长度分桶采样**：它不改样本、不拼接，只把相近长度的样本聚到同一 batch，把 pad 目标从"桶间最长"压成"桶内典型长度"，属开关级改动、零语义风险，但**须先测再开**：

- 先在 collator 加统计日志跑几十步，量化真实 padding 浪费率；若 <15% 则收益小、优先级后移，若音频长度方差大（>30%）则值得做；
- 确认 Megatron 路径真正消费 `--group_by_length`（否则开关形同虚设），且 `lazy_tokenize=True` 下 sampler 能拿到长度（`lengths` 列已修复 ✅，可得）；
- shuffle 折衷：swift 实现为"按长度排序 → 分桶 → 桶内随机"，跨桶顺序仍按长度单调，首次启用建议对比 loss 曲线。

### 2.4 其他

- **GBS 上调**：更少 step、更高 DP 效率，但影响收敛，需配 LR/warmup 验证，非纯工程改动，排最后。
- **dataloader 加深（workers/prefetch）**：当前不是喂数瓶颈，**勿改**。
- **训练参数归属**：所有训练超参只能改 `pipelines/bridge_alignment/common_args_megatron.sh` / 各 `train_*_megatron.sh`（必要时加 SCENE 预设或按 WORLD_SIZE/NNODES 派生），**禁止写进任务 yaml**。

---

## 3. eval / checkpoint 周期性开销（低优先级）

- `eval_steps=200`，全程 12+ 次 eval；单次 ≈158s（24 iter，其中首个 eval iter 恒 ~48s——val loader 每轮冷启动，persistent workers 未覆盖 val，可查），占比 **~6.4%**。
- checkpoint 保存（save_steps=500，~45–60s/次）与 eval 后 `Loading checkpoint shards`（~2s）开销均可忽略。
- 处置：`eval_steps` 调大到 500，零风险、立即可做，省 ~3–4% 总时长。

---

## 4. 性能基准记录

> 说明：本表采用**横向任务、纵向参数/指标**的布局，每新增一个任务就在右侧追加一列。
> 标记为「—」的单元格表示该后端没有此参数/未记录该值（该任务未采集或该后端 CLI 不存在）。
> 「通用环境」表记录所有任务一致的环境信息，不重复放进任务列。

### 4.0 通用环境（所有任务一致）

| 项 | 值 |
|---|---|
| 任务 yaml | `ktrain/ktrainjob_sts_zeyang_custom_data_train.yaml` |
| 集群 | speech-h200-eks |
| 节点 | `hyperpod-i-03e7d1f6d95864e1a`（ml.p5e.48xlarge） |
| 节点数 / GPU | 1 节点 × 8 × H200（141 GiB/卡） |
| 镜像 | `124355679795.dkr.ecr.ap-southeast-3.amazonaws.com/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9` |
| CPU / Memory（cgroup） | 135 核 / 1350 GiB |
| 模型 | `qwen2_5_3b_omnispeak_a` |
| 模型路径 | `s3://iegg-speech-model-models/origin_models/custom_models/qwen2_5_3b_omnispeak_a` |
| 模型总参数 | 6.73B |
| 可训练参数 | ~2.07B / 30.7% |
| 冻结部分 | thinker 36 层 + audio tower |
| 训练部分 | talker 28 层 + thinker_adapter |
| 训练目标 | adapter（reply_add） |
| 数据集 | unist（FSX 直读） |
| Manifest 路径 | `fsx://zeyangsong/unist/converted/unist_qwen3_s2s_v2/manifests/manifest-train-*.jsonl` |
| 音频 shards | `s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup/opt/dlami/nvme/zeyangsong/data/unist/`（跨区域 S3，同步到节点 NVMe） |
| 训练集行数 | 18,994,932 |
| 验证集行数 | 191,867（split_ratio=0.01） |
| QWEN3_TTS_TOKENIZER | `s3://iegg-speech-model-models/origin_models/Qwen3-TTS-Tokenizer-12Hz` |
| QWEN3_TTS_SPEAKER_ENCODER | `s3://iegg-speech-model-models/origin_models/Qwen3-TTS-12Hz-1.7B-Base` |
| 训练脚本 | `pipelines/bridge_alignment/train_speech_to_speech_translation.sh` |
| 参数脚本 | `pipelines/bridge_alignment/common_args.sh` |
| `max_length` | 2048 |
| `num_train_epochs` | 2 |
| `learning_rate` / `lr` | 1e-4 |
| `eval_steps` | 200 |
| `save_steps` | 500 |
| `save_total_limit` | 3 |
| `logging_steps` | 10 |
| `split_dataset_ratio` | 0.01 |
| `group_by_length` | false |
| `lazy_tokenize` | true |
| `dataloader_num_workers` | 8 |
| `dataloader_prefetch_factor` | 8 |
| `dataloader_persistent_workers` | true |
| `dataset_num_proc` | 135 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |

### 4.1 任务级配置对比表

> 各任务差异化的配置（后端 / 并行 / batch / 优化开关等）。

| 参数 / 指标 | otty（megatron） | bxa0（deepspeed） |
|---|---|---|
| 记录日期 | 2026-08-07 | 2026-08-07 |
| 数据来源日志 | `logs ...-otty-master-0` | `logs ...-bxa0-master-0` |
| 数据采集进度 | step 240 / 9,274 | step 100 / 9,276 |
| **训练后端** | megatron (mcore) | **deepspeed（swift sft）** |
| SCENE | dense | — |
| DEEPSPEED | — | zero2 |
| ATTN_IMPL | — | flash_attention_2 |
| `attention_backend` | flash | — |
| **微批 batch** | `micro_batch_size`=8 | `per_device_train_batch_size`=64 |
| `gradient_accumulation_steps` | —（=num_microbatches=64） | 8 |
| `per_device_eval_batch_size` | — | 4 |
| **有效 batch size** | GBS=4096 | per_device(64) × GPU(8) × grad_accum(8) = **4096** |
| `tensor_model_parallel_size` (TP) | 1 | — |
| `pipeline_model_parallel_size` (PP) | 1 | — |
| `context_parallel_size` (CP) | 1 | — |
| `sequence_parallel` | false | — |
| `recompute_granularity` | none | — |
| `recompute_method` | uniform | — |
| `recompute_num_layers` | 1 | — |
| `gradient_checkpointing` | — | true |
| `use_liger_kernel` | — | false |
| `packing` | false | — |
| `padding_free` | false | — |
| `cross_entropy_loss_fusion` | true | — |
| `overlap_grad_reduce` | false | — |
| `overlap_param_gather` | true | — |
| `align_param_gather` | true | — |
| `use_distributed_optimizer` | true（ZeRO-1） | — |
| `freeze_llm` | true | — |
| `freeze_vit` | true | — |
| `freeze_aligner` | false | — |
| `tuner_type` | — | full |
| `trainable_parameters` | — | model.thinker_adapter |
| `min_lr` | 1e-5 | — |
| `lr_warmup_fraction` | 0.03 | — |
| `warmup_ratio` | — | 0.03 |
| `save_safetensors` | true | — |
| `no_save_optim` | true | — |
| `no_save_rng` | true | — |
| `torch_dtype` | bfloat16 | — |
| `finetune` | true | — |
| **总步数** | 9,274 | 9,276 |
| **覆盖总样本** | 4096 × 9,274 = 37,986,304 | 4096 × 9,276 = 37,994,496 |
| **真实数据量** | 18,994,932 × 2 epochs = 37,989,864 | 37,989,864 |

### 4.2 性能数据对比表

> **公平对比口径**：两个任务的有效 batch 都是 **4096**（otty GBS=4096；bxa0 64×8×8=4096），总步数也几乎一致（9,274 vs 9,276），**可直接公平对比 s/it**。
> 总步数均覆盖完整数据量：9,274 × 4096 = 37,986,304 ≈ 1900W×2；9,276 × 4096 = 37,994,496 ≈ 1900W×2。

| 指标 | otty（megatron） | bxa0（deepspeed） |
|---|---|---|
| **每步样本数（有效 batch）** | 4096 | 4096 |
| **每秒样本吞吐**（有效 batch / s/it） | 4096 / 69.5 = **58.9 样本/s** | 4096 / 32.0 = **128.0 样本/s** |
| **稳态训练速度** | **69.5 s/it**（step 20-200，±0.2） | **32.0 s/it**（step 70-100，收敛至 31.99） |
| **显存（稳态）** | **103.42 GiB / 139.8 GiB（74%）** | **105.8 GiB / 139.8 GiB（76%）** |
| step 1 冷启动 | 89.2 s/it | 79.76 s/it |
| step 10 | 70.0 s/it | 35.35 s/it |
| step 20 | 69.5 s/it | 34.05 s/it |
| step 30 | 69.6 s/it | 35.30 s/it |
| step 40 | 69.6 s/it | 34.38 s/it |
| step 50 | 69.7 s/it | 33.74 s/it |
| step 60 | 69.7 s/it | 32.96 s/it |
| step 70 | 69.6 s/it | 32.69 s/it |
| step 80 | 69.6 s/it | 32.43 s/it |
| step 90 | 69.6 s/it | 32.20 s/it |
| step 100 | 69.6 s/it | 31.99 s/it |
| step 120 | 69.6 s/it | — |
| step 200 | 69.5 s/it（eval 前） | — |
| step 210（eval 后） | 76.4 s/it | — |
| step 240 | 75.5 s/it | — |
| 累计 elapsed | 5h 2m 1s（step 240） | 53m 19s（step 100） |
| 预估剩余时间 | 7d 21h 28m 33s（step 240） | 3d 9h 31m 58s（step 100） |
| 显存爬升 | 89.86 → 95.57 → 96.7 → 103.42 GiB（step 1→10→20→60 稳定） | 71.96 → 104.21 → 105.8 GiB（step 1→10→90 缓慢爬升） |
| OOM | 无 | 无 |
| 崩溃 | 无 | 无 |

> **备注**：otty 在 step 200 触发首次 eval 后，训练速度从 69.5 s/it 明显变慢到 75-76 s/it（约慢 9%）。
> 疑似 eval 后 dataloader 状态或显存缓存变化导致，需进一步排查是否为周期性退化。

### 4.3 训练健康度对比表

> 两个后端的 loss 指标口径不同（megatron 用 codec_loss 等；deepspeed 用 loss / token_text_acc 等），按各自记录。
> 两个任务每步有效 batch 都是 4096，采样 step 对齐为 step 1 / 60 / 100，方便同 step 比对。

| 指标 | otty（megatron） | | | bxa0（deepspeed） | | |
|---|---|---|---|---|---|---|
| step | step 1 | step 60 | step 100 | step 1 | step 60 | step 100 |
| loss | 20.98（codec_loss） | 9.34 | 8.95 | 22.96 | 23.61 | 23.52 |
| codec_token_acc / token_acc | 0.054% | 23.3% | 25.9% | 0.23% | 0.22% | 0.21% |
| codec_residual_loss | 8.13 | 5.73 | 5.57 | — | — | — |
| token_text_acc | — | — | — | 72.6% | 72.8% | 72.7% |
| token_acoustic_token | — | — | — | 0.44% | 0.40% | 0.40% |
| eos_boundary_top1_acc | 0.15% | 45.6% | 59.8% | 1.29% | 7.78% | 7.56% |
| grad_norm | 508.23 | 4.75 | 6.44 | 49.05 | 3.00 | 4.34 |
| learning_rate | 3.6e-07 | 2.157e-05 | 3.594e-05 | 3.6e-07 | 2.151e-05 | 3.584e-05 |

> **收敛对比说明**：同 step（1/60/100）下 megatron（otty）的 loss 从 20.98 降至 8.95，而 deepspeed（bxa0）始终在 ~23.5 附近，
> codec_token_acc 也显著更高（otty 25.9% vs bxa0 0.21%）。两者**收敛轨迹差异极大**，
> 但这与速度对比无关，需结合 batch/LR 语义确认是否为预期（可能是 bxa0 用的 deepspeed 路径 loss 口径或初始化差异）。
