# 三个 3B 级模型训练优化方案对比

> 2026-10-10 · 范围：只讲训练工程（并行、显存、算子、数据通路、调度），不涉及模型结构与训练目标。
> 1 号模型的数字取自已落地的实测（`docs/TRAINING_METRICS_MERGED.md`、`PERFORMANCE_OPTIMIZATION_PLAN.md`）；
> 2、3 号模型尚未在 GPU 上跑过基线，表中是**方案**，预期收益按 1 号模型的实测外推，需逐项验证。

| 编号 | 模型 | 代码 | 规模 | 状态 |
|---|---|---|---|---|
| ① | **qwen_omnispeak**（Qwen2.5-Omni-3B Thinker + Qwen3-TTS Talker） | `custom/qwen_omnispeak/` `pipelines/bridge_alignment/` | Thinker 3B + Talker 1.7B | **已优化、生产在用** |
| ② | **streaming_si**（Voxtral + Qwen3 Thinker + CosyVoice Talker） | `custom/streaming_si/` `pipelines/streaming_si/` | 0.6B/0.1B；4B/0.6B 两种变体 | 迁移完成，只有 CPU 冒烟 + 一次 S1 双节点 |
| ③ | **simuls2st_omni**（Qwen2.5-Omni-3B + DualCodec Talker） | `custom/simuls2st_omni/` `pipelines/simuls2st_omni/` | Thinker 3B + Talker 约 0.4B | 迁移完成，只有 CPU 对拍；GPU 未跑 |

---

## 1. 总览对比

| 优化项 | ① omnispeak（已落地） | ② streaming_si（方案） | ③ simuls2st_omni（方案） |
|---|---|---|---|
| **并行 / ZeRO** | DeepSpeed **zero1**（单机）/ **zero0**（双节点更快 5.3%） | 现为 **zero2** → 改 zero1，双节点测 zero0 | 现为 **DDP** → 改 DeepSpeed **zero1** 与 ① 统一（Stage 2/3 需补丁，见 §4.1） |
| **多节点** | ray / torchrun 双、三节点，SE≈86% | torchrun 双节点已有；ray 只支持 S1 | torchrun 双节点 yaml 已有 |
| **注意力后端** | flash_attention_2 | 未显式指定（HF 默认 sdpa）→ 改 FA2 | 写死 `sdpa` → 改 FA2 |
| **融合算子（liger）** | RMSNorm / SwiGLU / RoPE / 融合 CE，提速 9–10% | Qwen3 Thinker/Talker 可直接套 liger 的 Qwen3 补丁；4B 的 Qwen3.5 兼容层需手动替换 | Qwen2.5 Thinker 可复用 ① 的补丁；loss 自算，**不用融合 CE**，只上 RMSNorm/SwiGLU/RoPE |
| **显存 / 重计算** | flash + max_length 2048 + expandable_segments；峰值 139.78→97.46 GiB | 各阶段 batch 沿用 0.6B 默认值，4B **需先探测显存** | Stage 1 关、Stage 2 开梯度检查点；先测 tb24k 是否能关 |
| **batch 策略** | 固定 MBS × GA，GBS 4096 | 固定 per-device batch（S1 32 / S2 8 / S3 16 / S4 48） | **token-budget 打包**（自带长度分桶），已有 `stage1_tb24k.yaml` 放大 1.5× |
| **数据通路** | WDS → **Lance**，提速 1.15–1.51× | parquet + tar（S4 每节点约 11 TiB） | npy codes + jsonl manifest |
| **数据预处理** | flatten_indices 补丁、预置时长过滤 | S4 指针行 + 节点上一次性准备（`SI_PREP_ONLY`） | 指针行（一行 = 一个 batch），加载时一次物化 |
| **DataLoader** | 8 worker × prefetch 2，persistent；已证明不是瓶颈 | 默认 4 worker | 源配置 8（Stage 1/3）/ 4（Stage 2）；OMP 限 4 线程 |
| **通信重叠** | Megatron 下 overlap_grad_reduce 已修好待验证 | 跟随 DeepSpeed 默认 | DDP 下可调 `ddp_bucket_cap_mb`；切 DeepSpeed 后跟随默认 |
| **profiling** | nsys 单步抓取（torchrun / ray 均有补丁） | 可直接复用 ① 的 `nsys_step_profile.py` | 同左 |

---

## 2. ① qwen_omnispeak：已落地方案（基准）

生产配置（AO 任务，2026-09-04）：Lance + DeepSpeed zero0 + liger + ray 双节点，GBS 4096，**吞吐 216 样本/s，显存 100.86 GiB**。

| 优化项 | 做法 | 实测收益 |
|---|---|---|
| 数据通路 | WDS → Lance，本地 NVMe 全表直读 | 墙钟 1.15–1.51× |
| 预处理 | routing-tag 补丁跳过 flatten_indices；WDS 超长过滤改读 manifest 时长 | 单 rank 预处理从约 2 小时降到秒级 |
| ZeRO 档位 | zero2 → zero1 → zero0（双节点） | −12.33% step，再 −5.3% |
| liger | RMSNorm / SwiGLU / RoPE / 融合 CE（`QWEN_OMNISPEAK_LIGER_OPS=1`） | 稳态 9–10% |
| 显存 | flash attention + max_length 2048 + expandable_segments | 峰值 139.78→97.46 GiB，无 OOM |
| 多节点 | deepspeed 三节点 / megatron 双节点 | SE≈86% / 80% |
| DataLoader | 8 worker × prefetch 2，persistent | 队列 Q=1 已喂满，不再加深 |
| 不采纳 | group_by_length（loss 换族、显存翻倍）、DataLoader 加深 | — |

剩余待办：Lance 进程隔离死锁、overlap_grad_reduce 实测、align_param_gather 生效。

---

## 3. ② streaming_si：优化方案

**现状**：DeepSpeed zero2、HF 默认注意力、固定 per-device batch；只有 CPU 冒烟和一次 S1 双节点运行，没有吞吐 / 显存基线。

| 优先级 | 优化项 | 做法 | 依据 / 预期 | 风险 |
|---|---|---|---|---|
| P0 | **建立 GPU 基线** | 真实权重单机多卡跑 50–100 步，记录 s/it、显存、loss 起点 | README 明确要求上集群前先做 | — |
| P0 | **4B 变体显存探测** | 248k 词表下各阶段 batch 先减半，按显存余量回调 | README：0.6B 默认值大概率不适用 | — |
| P1 | ZeRO 档位 | zero2 → zero1；双节点再测 zero0 | ① 实测 zero1 比 zero2 快 12.33% | 可训参数小，显存影响不大 |
| P1 | 注意力 | Thinker / Talker 显式 `flash_attention_2` | ① 已用 FA2；Qwen3.5 兼容层声明支持 flash | Voxtral 编码器冻结，可不动 |
| P1 | liger | 0.6B：Qwen3 Thinker 与 Talker（Qwen3 结构）直接套 liger Qwen3 补丁；4B：兼容层 RMSNorm `(1+w)` 形式需单独确认能否替换 | ① 实测 9–10% | 4B 线性注意力层不在 liger 覆盖范围 |
| P1 | 4B 线性注意力加速核 | 镜像必须带 flash-linear-attention / causal-conv1d | 无此核时走纯 torch，很慢 | 已有 `streaming-si-v1` 镜像 |
| P2 | 梯度检查点 | 只在 4B 显存不够时开 | 用算力换显存 | 速度下降约 20–30% |
| P2 | 多节点 ray | S2–S4 迁移到 ray（多机数据合并、S4 准备、全局 batch 计算） | 与 ① 调度方式统一 | 工作量中等 |
| P2 | 数据 | S4 音频 5.4 TiB 每节点一份，长期考虑 Lance 化或 FSX 直读 | 节点 NVMe 已因数据被 cordon（磁盘压力） | 改数据格式需重新对拍 |

---

## 4. ③ simuls2st_omni：优化方案

**现状**：DDP、`sdpa`、token-budget 打包；Stage 1 GPU 试跑一次（32 分钟后手动停，无日志），没有基线。

| 优先级 | 优化项 | 做法 | 依据 / 预期 | 风险 |
|---|---|---|---|---|
| P0 | **建立 GPU 基线** | 现配置（DDP + sdpa）Stage 1 跑 50 步：首步 loss ≈ ln 16388 ≈ 9.7，记录 s/it、显存 | 后续每项优化的对照 | — |
| P0 | **改 DeepSpeed zero1**（与 ① 统一） | Stage 1 直接加 `--deepspeed zero1`；Stage 2/3 加参数组快照补丁（§4.1） | 配置统一、每卡省约 4–5 GB；速度预计与 DDP 持平或略慢 | Stage 2/3 不加补丁会改变训练语义 |
| P1 | 注意力 | `--attn_impl sdpa` → `flash_attention_2` | ① 已验证 | 不再与源项目逐位一致，只能比 loss 曲线 |
| P1 | liger | 复用 ① 的 Qwen2.5 Thinker 补丁：RMSNorm / SwiGLU / RoPE；**不开融合 CE**（loss 在 template 里自算） | ① 实测 9–10%（其中一部分来自 CE，这里会少一些） | 同上 |
| P1 | 调大 token_budget | 已有 `stage1_tb24k.yaml`（16384→24576，max_batch 64→96，阈值同步翻倍）；FA2 + liger 后显存再降，可继续上调 | Stage 1 注释：16k 时峰值约 100 GB，仍有余量 | 全局 batch 变大，lr 不变时收敛会变，需对比 loss |
| P2 | 梯度检查点 | Stage 2 现在开着；FA2 后测能否关 | 关掉通常快 20–30% | 显存可能不够 |
| P2 | DataLoader | collator 在 CPU 上逐条 tokenize，batch 变大后观察 GPU 是否饿 | 用 `fill_logging` 测 FillRate | — |
| P2 | 多节点步数对齐 | 16 卡时 Stage 2 设 `MAX_STEPS=11719` | README 3.3 | 不设则总 token 翻倍 |

### 4.1 Stage 2/3 切 DeepSpeed 的补丁

问题：Stage 2 每步只抽一个任务。文本任务（MT / S2TT / ASR，约 40% 的步数）时 talker 全参没有梯度；无音频的 batch 时音频塔上的 thinker LoRA 没有梯度。DDP 下这些参数梯度为 None，优化器直接跳过；DeepSpeed 会把它们当作 0 梯度照样做一步 Adam，动量和 weight decay 继续生效，训练语义改变。

做法：
1. 可训参数拆成三个 param group：talker / thinker LLM LoRA / 音频塔 LoRA。
2. 用梯度 hook 记录本步哪些 group 收到了梯度。
3. 本步没收到梯度的 group，在 `optimizer.step` 前快照其 fp32 分片和 `exp_avg` / `exp_avg_sq`，step 后恢复（zero1 下每卡几百 MB，开销小）。
4. 用 `tests/parity/run_parity.sh` 的 2 进程对拍，确认与 DDP 版本 loss 一致。

另需验证：DeepSpeed 下续训是否绕过作者的 `_load_from_checkpoint` 补丁；`save_training_checkpoint` 在 zero1 下可直接取 `state_dict`，**不要上 zero3**（参数分片后需要先 gather）。

---

## 5. 统一建议

1. **先拿基线再优化**：② 和 ③ 都没有 GPU 吞吐 / 显存数据。每个模型先用现配置跑 50–100 步，后面每项优化单独开、和基线比 s/it 与 loss。
2. **统一到 ① 的技术栈**：DeepSpeed zero1（双节点测 zero0）+ FA2 + liger + expandable_segments。三套模型的启动方式、排查经验、nsys 工具可以共用。
3. **对拍和提速分两步走**：② 和 ③ 迁移时都做了与源项目的逐位对拍。FA2、liger、调 batch 都会打破逐位一致，所以在对拍通过、拿到基线之后再开，之后只看 loss 曲线是否同族。
4. **节点资源**：目前两台空节点因 NVMe 磁盘压力被 cordon；② 的 S4 音频（每节点约 11 TiB）和 ③ 的 Stage 2 语料（WenetSpeech、GigaSpeech 各约 1.1 TB）都很占盘。上 ②、③ 之前要先规划节点磁盘，否则会重复出现磁盘压力导致的 cordon。
