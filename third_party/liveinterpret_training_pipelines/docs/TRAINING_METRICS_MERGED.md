# 训练实验参数与结果总表

> 整理日期：2026-09-04（基于远端 `master` @ `3e88d11`）。
> 本文档把分散在多份实验记录里的全部训练任务**统一编号为 A/B/C/…**，
> 并给出每个任务的**完整参数设置**与**最终结果数据**，用作唯一的对照基线。
> 所有数值均原样摘录实测日志，未重新计算。
>
> 阅读方式：
> - **第 1 章** 通用环境与各任务参数设置（按实验族分表）。
> - **第 2 章** 数学结果比对（七方主对照 A–G）。
> - **第 3 章** 训练优化比对（各参数配置下的 loss / s/it / remaining_time）。
> - **第 4 章** 指标口径与可比性判定。

---

## 1. 参数设置

### 1.0 通用环境（除注明外全部任务一致）

| 项 | 值 |
|---|---|
| 集群 / 命名空间 | speech-h200-eks / `speech-h200` |
| 机型 / GPU | ml.p5e.48xlarge，8 × H200（141 GiB/卡）/ 节点 |
| 模型 | `qwen2_5_3b_omnispeak_a`（zeyangsong_share；config 自带 `use_upstream_talker_input: true`，adapter 前向原生生效） |
| 训练目标 | adapter 式 upstream text-hidden 叠加 codec，`reply_add` 布局；speaker prompt on；ICL off |
| 可训参数 | `model.thinker_adapter + model.talker`（≈1.93B），thinker 冻结（`freeze_parameters_ratio=1.0`） |
| loss 权重 | residual 0.3 / codec_eos 1.0 / text 1.0 |
| 精度 / 注意力 | bf16；flash（thinker 侧），talker 4D block-causal mask 走 SDPA |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |
| 归约环境 | `NCCL_NVLS_ENABLE=0`（走 ring/tree 软件归约）；`NCCL_IB_DISABLE=0`、`FI_PROVIDER=efa` |
| 容器资源 | cpu 135、memory 1350Gi、gpu 8、`shm_size` 64Gi |

### 1.0.1 编号速览

| 编号 | 一句话定位 | 编号 | 一句话定位 |
|---|---|---|---|
| A | 本地参考基线（deepspeed 单机，宽 glob） | P | GBS4096 单机 deepspeed zero2 无 liger |
| B | WDS 单机 deepspeed | Q | GBS4096 单机 deepspeed zero2 + liger |
| C | WDS 单机 megatron | R | GBS4096 单机 deepspeed zero1 + liger |
| D | WDS 双节点 ray + deepspeed | S | GBS4096 双节点 ray + zero1 + liger |
| E | WDS 双节点 megatron | T | GBS4096 单机 megatron + liger |
| F | WDS 双节点 deepspeed 直连（torchrun） | U | GBS4096 单机 megatron 无 liger（T 的反向对照） |
| G | WDS 三节点 deepspeed 直连 | V | GBS4096 双节点 deepspeed zero0 + liger |
| H | Lance 单机 deepspeed（B 的 Lance 版） | W | DataLoader 16×8，Q=128（基线队列） |
| I | Lance 单机 megatron（C 的 Lance 版） | X | DataLoader 8×16，Q=128（等深换绑） |
| J | Lance 双节点 ray + deepspeed（D 的 Lance 版） | Y | DataLoader 16×16，Q=256（加深） |
| K | Lance 双节点 megatron（E 的 Lance 版） | Z | DataLoader 8×24，Q=192（加深 + 更好绑核） |
| L | Lance 双节点 deepspeed 直连（F 的 Lance 版） | AA | DataLoader 0 worker（关掉 pipeline） |
| M | Lance 单机 + `group_by_length=true` | AB | DataLoader 1×1，Q=1（最小队列） |
| AN | 汇总优化单机：Lance + zero0 + GBS=2048 | AC | DataLoader 2×2，Q=4（浅队列） |
| AO | 汇总优化双节点：Lance + zero0 + GBS=4096 + liger | AD | DataLoader 4×8，Q=32（中等队列） |
| AE | 200K 子集，`group_by_length=false` | AH | 1/10 数据 megatron |
| AF | 200K 子集，`group_by_length=true` | AI | 1/10 数据 deepspeed |
| AG | 200K 子集，1 worker（FillRate 首测） | | |

### 1.1 七方主对照（A–G）：变量 = 后端 × 拓扑

| 参数 | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| 任务 | `jowz50ie`（zeyang 本地 run） | `sts-cmp200-deepspeed-1node-m3gc` | `sts-cmp200-megatron-1node-7m7o` | `sts-cmp200-ray-zero2-2node-hrbt` | `sts-cmp200-megatron-2node-jkwl` | `junyaoyang-mfu-ds-cmp-2node-v3` | `junyaoyang-mfu-ds-cmp-3node` |
| 任务 yaml | —（本地 run 脚本） | `ktrainjob_sts_cmp200_deepspeed_1node.yaml` | `ktrainjob_sts_cmp200_megatron_1node.yaml` | `ktrainjob_sts_cmp200_ray_zero2_2node.yaml` | `ktrainjob_sts_cmp200_megatron_2node.yaml` | `ktrainjob_junyaoyang_mfu_ds_cmp_2node.yaml` | `ktrainjob_junyaoyang_mfu_ds_cmp_3node.yaml` |
| 框架 / 后端 | deepspeed（本地 8 卡） | pytorch / deepspeed | pytorch / megatron（mcore） | ray（KubeRay）/ deepspeed | pytorch / megatron | pytorch / deepspeed（直连 torchrun） | pytorch / deepspeed（直连 torchrun） |
| DeepSpeed 引擎 | zero2 | zero2 | —（megatron 分布式优化器） | zero2 | — | zero2 | zero2 |
| 节点 | 本地 | `hyperpod-i-02848679822afbe3b` | `hyperpod-i-08e3257c716579618` | 双节点（028 + 08e3） | 双节点 | 双节点（08e3 + 028） | 三节点（07d3 + 028 + 08e3） |
| 节点数 / GPU | 1 × 8 | 1 × 8 | 1 × 8 | 2 × 8（+0GPU head） | 2 × 8 | 2 × 8 | 3 × 8 |
| 并行策略 | DP=8 | DP=8 | TP=PP=CP=EP=1，DP=8 | DP=16 | TP=PP=CP=EP=1，DP=16 | DP=16 | DP=24 |
| 微批 / 累积 | 64 × ga1 | 64 × ga1 | MBS=16 × 4 microbatch | 32 × ga1（512/DP 派生） | MBS=16 × 2 microbatch | 32 × ga1 | 21 × ga1 |
| **有效 batch（GBS）** | **512** | **512** | **512** | **512** | **512** | **512** | **504**（21×24，512 不整除 24） |
| 总步数（2 epochs 派生） | 75070 | 74912 | 74910 | 74912 | ≈74157 | 74912 | 76100 |
| warmup 步数（0.03 派生） | ≈2252 | ≈2247 | ≈2247 | ≈2247 | ≈2247 | ≈2247 | ≈2247 |
| LR / 调度 / warmup | 1e-5 / constant_with_warmup / 0.03 | 同 | 同 | 同 | 同 | 同 | 同 |
| `max_length` | 576 | 576 | 576 | 576 | 576 | 576 | 576 |
| `num_train_epochs` | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| `eval_steps` / `save_steps` | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| `logging_steps` | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| `split_dataset_ratio` | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| `lazy_tokenize` | true | true | true | true | true | true | true |
| 双层 shuffle | 开 | 开 | 开 | 开 | 开 | 开 | 开 |
| dataloader workers / prefetch | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 |
| 数据集 | unist 全量，宽 glob `manifest-*.jsonl`（19.23M 行，含 dev/test） | 窄 glob `manifest-train-*.jsonl`（18,994,930 行） | 同 B | 同 B | 同 B | 同 B | 同 B |
| 超长音频处理 | 截断（`MAX_CODEC_FRAMES=384`，cap 391） | >120s 整行丢弃 | 同 B | 同 B | 同 B | 同 B | 同 B |
| 数据通路 | FSX manifest + S3 WDS shards 同步 NVMe | 同 A | 同 A | 同 A | 同 A | 同 A | 同 A |
| 镜像 | —（本地 conda，swift 4.3.2） | `p5-cuda12.8-torch2.8-te-v9` | 同 B | 同 tag `-ray2`（+ray 2.55.1） | 同 B | 同 B | 同 B |
| 输出目录 | 本地 | `/nvme/output/sts_cmp200_deepspeed_1node` | `/nvme/output/sts_cmp200_megatron_1node` | `/nvme/output/sts_cmp200_ray_zero2_2node` | `/nvme/output/sts_cmp200_megatron_2node` | `/mnt/fsx/junyaoyang/output/mfu_ds_cmp_2node_v3` | `/mnt/fsx/junyaoyang/output/mfu_ds_cmp_3node` |
| 上报 | W&B 在线 | W&B 在线 | W&B 在线 | W&B 在线 | W&B 在线 | disabled | disabled |

> 口径：B/C/D/E/F/G 训练参数同源（同一 `SCENE=batch4_8_adapter_cmp` 预设），**唯一变量 = 后端与拓扑**；
> 有效 batch 与 eval/save 节奏全面对齐 A（GBS=512、eval/save=1000）。与 A 的其余差异：数据 glob、
> 超长音频处理、镜像 swift 版本（4.4.0 vs 4.3.2）。

### 1.2 Lance 数据通路（H–M）：变量 = 数据通路 / DataLoader / 分桶

| 参数 | H | I | J | K | L | M |
|---|---|---|---|---|---|---|
| 任务 | `sts-cmp200lance-ds-1n-v9pm` | `sts-cmp200lance-mg-1n-eoft` | `sts-cmp200lance-ray-2n-d7is` | `sts-cmp200lance-mg-2n-l8ut` | `sts-cmp200lance-ds-2n-xzvy` | `sts-cmp200lance-ds-gbl-ya22` |
| 后端 / 拓扑 | deepspeed / 单机 | megatron / 单机 | ray+deepspeed / 双节点 | megatron / 双节点 | deepspeed 直连 / 双节点 | deepspeed / 单机 |
| 节点 | 028 | 028 | 028 + 07d3 | 028 + 07d3 | 028 + 07d3 | 028 |
| GPU / DP | 8 / DP=8 | 8 / DP=8 | 16 / DP=16 | 16 / DP=16 | 16 / DP=16 | 8 / DP=8 |
| 微批 / 累积 | 64 × ga1 | MBS=16 × 4 | 32 × ga1 | MBS=16 × 2 | 32 × ga1 | 64 × ga1 |
| **GBS** | **512** | **512** | **512** | **512** | **512** | **512** |
| 总步数 | 74912 | 74910 | 74912 | 74910 | 74912 | 74912 |
| LR / 调度 / warmup | 1e-5 / constant_with_warmup / 0.03 | 同 | 同 | 同 | 同 | 同 |
| `max_length` | 576 | 576 | 576 | 576 | 576 | 576 |
| `eval_steps` / `save_steps` | 1000 / 1000 | 同 | 同 | 同 | 同 | 同 |
| `split_dataset_ratio` | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| `lazy_tokenize` | true | true | true | true | true | **true（GBL 硬性要求）** |
| **`group_by_length`** | false | false | false | false | false | **true** |
| 双层 shuffle | 开 | 开 | 开 | 开 | 开 | 开 |
| dataloader workers × prefetch | 16 × 8 | 16 × 8 | 16 × 8 | 16 × 8 | 16 × 8 | 16 × 8 |
| `persistent_workers` / 绑核 | true / 1 worker=1 核 | 同 | 同 | 同 | 同 | 同 |
| 数据 | Lance v506 `split='train'` 19,177,206 行（val 9,593） | 同 | 同 | 同 | 同 | 同 |
| Lance 读取 | 本地 NVMe 全表，离线 mel + speaker，批量 take，主进程 lance 隔离 | 同 | 同 | 同 | 同 | 同 |
| 镜像 | `p5-cuda12.8-torch2.8-te-v9` | 同 | 同 | 同 | 同 | 同 |

### 1.3 GBS=4096 引擎 / 融合算子对照（P–V）

| 参数 | P | Q | R | S | T | U | V |
|---|---|---|---|---|---|---|---|
| 任务 | `junyaoyang-ds-gbs4096-noliger-1node` | `junyaoyang-ds-gbs4096-liger-1node` | `sts-cmp200-zero1-liger-gbs4096-1node` | `junyaoyang-ray2node-gbs4096-liger-z1-07d3-028` | `junyaoyang-mg-gbs4096-liger-1node` | `junyaoyang-mg-gbs4096-noliger-1node` | `junyaoyang-ds-gbs4096-liger-zero0-2node-07d3-028` |
| 后端 / 拓扑 | deepspeed / 单机 | deepspeed / 单机 | deepspeed / 单机 | ray+deepspeed / 双节点 | megatron / 单机 | megatron / 单机 | deepspeed 直连 / 双节点 |
| **DeepSpeed 引擎** | **zero2** | **zero2** | **zero1** | **zero1** | —（分布式优化器） | — | **zero0（纯 DDP）** |
| **liger-kernel** | **关** | **开**（`USE_LIGER_KERNEL=true` + `QWEN_OMNISPEAK_LIGER_OPS=1`） | 开（同 Q） | 开（同 Q） | **开**（仅 `QWEN_OMNISPEAK_LIGER_OPS=1`，megatron 不读 `USE_LIGER_KERNEL`） | **关**（不设 `QWEN_OMNISPEAK_LIGER_OPS`） | 开（同 Q） |
| 节点 | 028 | 028 | 028 | 07d3 + 028 | 028 | 07d3 | 07d3 + 028 |
| GPU / DP | 8 / DP=8 | 8 / DP=8 | 8 / DP=8 | 16 / DP=16 | 8 / DP=8 | 8 / DP=8 | 16 / DP=16 |
| 微批 / 累积 | 64 × ga8 | 64 × ga8 | 64 × ga8 | **32** × ga8 | MBS=16 → 32 microbatch | MBS=16 → 32 microbatch | **32** × ga8 |
| **GBS** | **4096** | **4096** | **4096** | **4096** | **4096** | **4096** | **4096** |
| 总步数 | 9364 | 9364 | 9364 | 9364 | 9363 | 9363 | 9364 |
| LR / 调度 / warmup | 1e-5 / constant_with_warmup / 0.03 | 同 | 同 | 同 | 同 | 同 | 同 |
| `max_length` | 576 | 576 | 576 | 576 | 576 | 576 | 576 |
| `eval_steps` / `save_steps` | 1000 / 1000 | 同 | 同 | 同 | 同 | 同 | 同 |
| `split_dataset_ratio` | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| 双层 shuffle | 开 | 开 | 开 | 开 | 开 | 开 | 开 |
| dataloader workers / prefetch | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 | 16 / 8 |
| 数据集 | unist 全量窄 glob（18,994,930 行） | 同 | 同 | 同 | 同 | 同 | 同 |
| 镜像 | `p5-cuda12.8-torch2.8-te-v9-nsys-liger`（liger-kernel 0.5.10，由开关控制） | 同 | 同 | 同 | 同 | 同 | 同 |
| 上报 | W&B disabled | 同 | 同 | 同 | 同 | 同 | 同 |

### 1.4 DataLoader 队列与分桶（W–AG，200K 热缓存子集）

公共配置：节点 08e3（W–Z、AG）或 07d3（AA–AF）；deepspeed zero2 / 单节点 8×H200；
`SCENE=batch4_8_adapter_cmp`，GBS=512（64×8）；LR 1e-5 / constant_with_warmup / 0.03；
`max_length` 576；`LAZY_TOKENIZE=true`；`DATALOADER_PERSISTENT_WORKERS=true`；
`DATALOADER_CPU_BIND=1`（1 worker = 1 核）；`SPLIT_DATASET_RATIO=0`（无 eval，train/eval worker 不共存）；
批量 take 与主进程隔离开；`shm_size` 除注明外 64Gi；统计窗口 **step 50–199**（150 步）。

| 参数 | W | X | Y | Z | AA | AB | AC | AD | AE | AF | AG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 别名 | P0 | P1 | P2 | P3 | W0 | W1 | W2 | W4 | GBL-G0 | GBL-G1 | 200K w1 |
| **workers** | 16 | **8** | 16 | **8** | **0** | **1** | **2** | **4** | 16 | 16 | **1** |
| **prefetch_factor** | 8 | **16** | **16** | **24** | 8（忽略） | **1** | **2** | **8** | 8 | 8 | 8 |
| **Q / rank** | 128 | 128 | **256** | 192 | **0** | **1** | 4 | 32 | 128 | 128 | 8 |
| `shm_size` | 64Gi | 64Gi | **128Gi** | **128Gi** | 64Gi | 64Gi | 64Gi | 64Gi | 64Gi | 64Gi | 64Gi |
| **`group_by_length`** | false | false | false | false | false | false | false | false | **false** | **true** | false |
| 步数 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 |
| 数据 | Lance v506 子集 200,000 行（48 GiB，进页缓存） | 同 | 同 | 同 | 同 | 同 | 同 | 同 | 同 | 同 | 同 |
| 提交时间 | 2026-09-02 | 同 | 同 | 同 | 2026-09-03 | 同 | 同 | 同 | 同 | 同 | 2026-09-02 |

### 1.5 1/10 数据后端 parity（AH / AI）

| 参数 | AH | AI |
|---|---|---|
| 任务 | `ktrainjob-sts-zeyang-megatron-1t10-lium` | `ktrainjob-sts-zeyang-deepspeed-1t10-jc85` |
| 任务 yaml | `ktrainjob_sts_zeyang_megatron_data1t10.yaml` | `ktrainjob_sts_zeyang_custom_data_train.yaml` |
| 后端 | megatron（mcore） | deepspeed（swift sft） |
| 节点 | `hyperpod-i-02848679822afbe3b` | `hyperpod-i-08e3257c716579618` |
| 可训参数 | **仅 `thinker_adapter`** | 同 |
| 数据量 | 全量 18,994,930 行的 **1/10**：顺序采样注册数据集名 `#1899493` | 同 |
| 采样 / shuffle | `dataset_shuffle=false` + `train_dataloader_shuffle=false`（双层禁用）+ `seed=42` + `data_seed=42` | 同 |
| 微批 / 累积 | `micro_batch_size=16`，num_microbatches=32 | `per_device_train_batch_size=64` × ga8 |
| **GBS** | **4096** | **4096** |
| 总步数 | 918 | 920 |
| LR / warmup | **1e-4** / 0.03（`min_lr` 1e-5） | 同 |
| `max_length` | **2048** | 2048 |
| `eval_steps` / `save_steps` | 200 / 500 | 200 / 500 |
| `split_dataset_ratio` | 0.01 | 0.01 |
| dataloader workers / prefetch | 8 / 8 / persistent=true | 同 |
| `attention_backend` | flash（recompute_granularity=none） | flash_attention_2（gradient_checkpointing=true） |
| 分布式优化器 / overlap | `use_distributed_optimizer=true`（ZeRO-1）、`overlap_param_gather=true`、`align_param_gather=true`、`cross_entropy_loss_fusion=true` | zero2 |
| ICL / speaker prompt | off / off | off / off |
| 终态 | step 620/918 被外部 SIGTERM（checkpoint-500 已存） | Succeeded 920/920，总耗时 8h40m26s |

### 1.6 汇总优化任务（AN 单节点 / AO 双节点）

把实测最优的项组合到一起的生产配置（2026-09-04 落地）。

| 参数 | AN | AO |
|---|---|---|
| 任务 | `ktrainjob-sts-zeyang-custom-data-lance-train-ifcv` | `sts-zy-lance-2n-p88m` |
| 任务 yaml | `ktrainjob_sts_zeyang_custom_data_lance_train.yaml` | 同 |
| 框架 / 后端 | pytorch / deepspeed **zero0** | **ray（KubeRay）/ deepspeed zero0** |
| 拓扑 / 节点 | 单机 `hyperpod-i-02848679822afbe3b`，8×H200，DP=8 | **双节点 0284 + 08e3**，numNodes=3（head+2 workers），16×H200，DP=16 |
| 入口 | `/opt/launcher/launch_train_lance.sh`（Lance 通路） | `ray_entrypoint.py` → `LAUNCHER_PATH=/opt/launcher/launch_train_lance.sh` |
| SCENE | `batch4_8_adapter_cmp` | 同 |
| 微批 / 累积 | 32 × ga8 | 32 × ga8 |
| **GBS** | **2048**（32 × 8 × 8） | **4096**（32 × 16 × 8） |
| 总步数（2 epochs 实测） | **18728** | **9364** |
| LR / 调度 / warmup | 1e-5 / constant_with_warmup / 0.03（warmup ≈562） | 同（warmup ≈281） |
| `max_length` / epochs | 576 / 2 | 576 / 2 |
| `eval_steps` / `save_steps` / `logging_steps` | 1000 / 1000 / 10 | 同 |
| `split_dataset_ratio` | 0.0005 | 同 |
| 双层 shuffle | 开 | 开 |
| `group_by_length` / `lazy_tokenize` | false / true | 同 |
| DataLoader | 8 workers × prefetch 2（Q/rank=16），persistent=true，1 worker=1 核 | 同 |
| 注意力 / 重计算 | flash_attention_2 / `gradient_checkpointing=true` | 同 |
| liger | 关（`USE_LIGER_KERNEL=false`） | **开**（`USE_LIGER_KERNEL=true` + `QWEN_OMNISPEAK_LIGER_OPS=1` + 镜像 `-ray2-nsys-liger`） |
| 兼容开关 | — | `SWIFT_SINGLE_DEVICE_MODE=1`（绕过 swift `is_mp()` 对 ray driver 的误判，让 liger 校验通过） |
| 数据 | Lance v506 `split='train'` 19,177,206 行，本地 NVMe 全表直读 | 同（两节点各一份全表） |
| 归约环境 | `NCCL_NVLS_ENABLE=0`、`NCCL_IB_DISABLE=0`、`FI_PROVIDER=efa` | 同 |
| 镜像 | `p5-cuda12.8-torch2.8-te-v9` | `p5-cuda12.8-torch2.8-te-v9-ray2-nsys-liger` |
| 输出目录 | `/nvme/output/sts_zeyang_lance_adapter_cmp` | `/nvme/output/sts_zeyang_lance_ray2n_gbs4096_liger` |

> AN 与 AO 的 GBS 不同（2048 vs 4096），**不与对方或 GBS=512/4096 族做 loss 横比**；
> 只有 AO 属 GBS=4096 族，可与 P–V 直接横比（见 3.2 的对照表）。

## 2. 数学结果比对（七方主对照 A–G）

### 2.1 step 200 终值

| 指标 | A | B | C | D | E | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss | 6.3941 | 6.3625 | 6.3719 | 6.3605 | 6.3765 | 6.3605 | 6.3782 |
| token_acc | 16.89% | 17.09% | 16.92% | 17.13% | 16.89% | 17.13% | 16.95% |
| eos_boundary_top1_acc | 36.99% | 36.13% | 33.98% | 36.31% | 34.45% | 36.31% | 35.69% |
| premature_eos_top1_rate | 1.55% | 1.46% | 1.37% | 1.43% | 1.40% | 1.43% | 1.40% |
| grad_norm | 9.16 | 6.27 | 8.95 | 6.51 | 7.39 | 6.51 | 10.96 |
| learning_rate @200 | 8.900e-07 | 8.900e-07 | 8.900e-07 | 8.900e-07 | 8.900e-07 | 8.900e-07 | 8.800e-07 |

### 2.2 loss 逐 10 步

| step | A | B | C | D | E | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.7711 | 16.9301 | 16.6700 | 16.9308 | 16.6630 | 16.9308 | 16.9126 |
| 10 | 16.8859 | 17.0034 | 16.6273 | 17.0021 | 16.6337 | 17.0021 | 17.0228 |
| 20 | 16.8164 | 17.0098 | 16.6189 | 17.0248 | 16.6144 | 17.0248 | 17.0237 |
| 30 | 16.6570 | 16.8115 | 16.4360 | 16.8048 | 16.4316 | 16.8048 | 16.8221 |
| 40 | 16.1107 | 16.2513 | 15.8992 | 16.2445 | 15.8957 | 16.2445 | 16.2661 |
| 50 | 14.9654 | 15.0257 | 14.7175 | 15.0348 | 14.7148 | 15.0348 | 15.0905 |
| 60 | 13.9372 | 13.9580 | 13.7180 | 13.9665 | 13.7156 | 13.9665 | 14.0447 |
| 70 | 11.9275 | 11.8405 | 11.7181 | 11.8416 | 11.7141 | 11.8416 | 11.9077 |
| 80 | 11.2074 | 11.0488 | 10.9460 | 11.0529 | 10.9453 | 11.0529 | 11.0929 |
| 90 | 9.9928 | 9.9282 | 9.8515 | 9.9223 | 9.8497 | 9.9223 | 10.0259 |
| 100 | 9.2921 | 9.2522 | 9.2203 | 9.2530 | 9.2191 | 9.2530 | 9.3027 |
| 110 | 8.9054 | 8.8862 | 8.8511 | 8.8833 | 8.8507 | 8.8833 | 8.9193 |
| 120 | 8.4904 | 8.4513 | 8.4329 | 8.4577 | 8.4345 | 8.4577 | 8.5127 |
| 130 | 7.9651 | 7.9267 | 7.8890 | 7.9337 | 7.8857 | 7.9337 | 7.9861 |
| 140 | 7.4702 | 7.4423 | 7.4311 | 7.4404 | 7.4348 | 7.4404 | 7.5050 |
| 150 | 7.1266 | 7.0984 | 7.0953 | 7.1075 | 7.0977 | 7.1075 | 7.1340 |
| 160 | 6.8995 | 6.8584 | 6.8616 | 6.8573 | 6.8541 | 6.8573 | 6.8988 |
| 170 | 6.7146 | 6.6770 | 6.6989 | 6.6720 | 6.6950 | 6.6720 | 6.6902 |
| 180 | 6.5790 | 6.5506 | 6.5572 | 6.5525 | 6.5500 | 6.5525 | 6.5732 |
| 190 | 6.4761 | 6.4365 | 6.4513 | 6.4358 | 6.4527 | 6.4358 | 6.4616 |
| 200 | 6.3941 | 6.3625 | 6.3719 | 6.3605 | 6.3765 | 6.3605 | 6.3782 |

### 2.3 token_acc 逐 10 步（deepspeed `token_acc` ↔ megatron `codec_token_acc`）

| step | A | B | C | D | E | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.19% | 0.26% | 0.30% | 0.26% | 0.30% | 0.26% | 0.26% |
| 10 | 0.16% | 0.20% | 0.24% | 0.20% | 0.24% | 0.20% | 0.20% |
| 20 | 0.18% | 0.18% | 0.25% | 0.18% | 0.26% | 0.18% | 0.18% |
| 30 | 0.19% | 0.21% | 0.25% | 0.22% | 0.26% | 0.22% | 0.21% |
| 40 | 0.18% | 0.18% | 0.23% | 0.18% | 0.23% | 0.18% | 0.17% |
| 50 | 0.21% | 0.18% | 0.23% | 0.18% | 0.24% | 0.18% | 0.18% |
| 60 | 0.20% | 0.17% | 0.20% | 0.17% | 0.20% | 0.17% | 0.17% |
| 70 | 0.30% | 0.31% | 0.32% | 0.32% | 0.32% | 0.32% | 0.30% |
| 80 | 0.47% | 0.56% | 0.54% | 0.55% | 0.53% | 0.55% | 0.54% |
| 90 | 0.92% | 0.93% | 1.03% | 0.93% | 1.03% | 0.93% | 0.88% |
| 100 | 1.40% | 1.63% | 1.63% | 1.60% | 1.62% | 1.60% | 1.48% |
| 110 | 2.42% | 2.64% | 2.56% | 2.62% | 2.53% | 2.62% | 2.51% |
| 120 | 4.12% | 4.38% | 4.35% | 4.32% | 4.33% | 4.32% | 4.00% |
| 130 | 6.72% | 6.76% | 7.06% | 6.69% | 7.04% | 6.69% | 6.51% |
| 140 | 9.54% | 9.70% | 9.62% | 9.67% | 9.57% | 9.67% | 9.23% |
| 150 | 11.96% | 11.95% | 11.78% | 11.84% | 11.71% | 11.84% | 11.72% |
| 160 | 13.27% | 13.66% | 13.42% | 13.64% | 13.50% | 13.64% | 13.20% |
| 170 | 14.64% | 15.02% | 14.77% | 14.99% | 14.77% | 14.99% | 15.01% |
| 180 | 15.45% | 15.90% | 15.53% | 15.88% | 15.60% | 15.88% | 15.59% |
| 190 | 16.28% | 16.60% | 16.23% | 16.66% | 16.22% | 16.66% | 16.52% |
| 200 | 16.89% | 17.09% | 16.92% | 17.13% | 16.89% | 17.13% | 16.95% |

### 2.4 token_text_acc（deepspeed 链路专属）

| step | A | B | D | F | G |
|---|---:|---:|---:|---:|---:|
| 1 | 72.57% | 72.91% | 72.90% | 72.90% | 72.90% |
| 50 | 72.89% | 72.86% | 72.86% | 72.86% | 72.92% |
| 100 | 73.17% | 72.75% | 72.71% | 72.71% | 72.79% |
| 150 | 73.01% | 72.84% | 72.87% | 72.87% | 73.09% |
| 200 | 72.54% | 72.67% | 72.68% | 72.68% | 72.96% |

### 2.5 codec_loss / codec_residual_loss（megatron 链路专属）

| step | C codec_loss | E codec_loss | C residual | E residual |
|---|---:|---:|---:|---:|
| 1 | 16.6413 | 16.6352 | 8.4929 | 8.5001 |
| 50 | 14.7053 | 14.7031 | 8.2136 | 8.2127 |
| 100 | 9.2198 | 9.2191 | 7.0214 | 7.0220 |
| 150 | 7.0969 | 7.0997 | 6.4219 | 6.4225 |
| 200 | 6.3667 | 6.3733 | 6.0739 | 6.0762 |

### 2.6 结论

1. **七列在同一收敛轨迹上**：loss@200 六列（B–G）两两差 ≤0.018，含 A 后两两差 ≤0.032；
   token_acc 差 ≤0.24pt；`premature_eos_top1_rate` 全部落在 1.37–1.55% 带内。
   **后端（deepspeed / megatron）与拓扑（单 / 双 / 三节点、torchrun / ray）不引入可见数学偏差。**
2. **D 与 F 逐点完全一致**（同 GBS=512、同数据顺序）：ray 与 torchrun 数学等价。
3. **G（GBS=504）与 512 家族同族**：loss@200 与 F 差 0.018，token_acc 逐 step 差 ≤0.32pt。
4. `token_text_acc` 全程钉死在 72.5–73.2%（thinker 冻结的直接证据），是判断 thinker 是否被训练的可靠探针。

---

## 3. 训练优化比对（统一维度：所有任务直接对比）

> 不按可比族分章。所有全量 200-step 任务用同一组维度直接对比：
> **batch（GBS）、稳态 step、样本吞吐（样本/s）、显存 @200、remaining @200、loss（0–200 每 10 步）**。
> 稳态口径：A–M 为 step 190→200 瞬时；P–V 为 step 191–200 均值；AN/AO 为 step 191–200 中位。

### 3.1 性能对比总表（全量 200 step）

| 编号 | 数据通路 | 后端 / 拓扑 | GBS | 稳态 step（s） | 吞吐（样本/s） | 显存 @200 (GiB) | remaining @200 | 200 步耗时 |
|---|---|---|---|---:|---:|---:|---|---|
| A | WDS | deepspeed / 单机 | 512 | 4.09 | ~125 | 82.82 | 3d 13h 2m 26s | 13m38s |
| B | WDS | deepspeed / 单机 | 512 | 4.79 | ~138 | 77.19 | 4d 3h 20m 32s | 15m57s |
| C | WDS | megatron / 单机 | 512 | 5.43 | ~98 | 96.34 | 4d 16h 38m 47s | 18m6s |
| D | WDS | ray+deepspeed / 双节点 | 512 | 2.76 | ~283 | 42.59 | 2d 9h 12m 32s | 9m11s |
| E | WDS | megatron / 双节点 | 512 | 3.40 | ~170 | 94.97 | 2d 22h 37m 1s | 11m21s |
| F | WDS | deepspeed 直连 / 双节点 | 512 | 2.69 | ~190 | 42.59 | 2d 7h 53m 17s | 8m59s |
| G | WDS | deepspeed 直连 / 三节点 | 504 | 2.08 | ~242 | 34.45 | 1d 19h 48m 57s | 6m56s |
| H | Lance | deepspeed / 单机 | 512 | 3.23 | ~177 | — | 2d 19h 4m 29s | 10m46s |
| I | Lance | megatron / 单机 | 512 | 4.63 | ~98 | 96.18 | 4d 0h 7m 7s | 15m26s |
| J | Lance | ray+deepspeed / 双节点 | 512 | 1.82 | ~283 | 72.09 | 1d 13h 43m 30s | 6m4s |
| K | Lance | megatron / 双节点 | 512 | 2.71 | ~170 | — | 2d 8h 14m 21s | 9m2s |
| L | Lance | deepspeed 直连 / 双节点 | 512 | 2.35 | ~190 | 72.09 | 2d 0h 40m 28s | 7m49s |
| M | Lance + GBL | deepspeed / 单机 | 512 | 2.23 | — | 132.68 | 1d 22h 18m 16s | 7m26s |
| P | WDS | deepspeed zero2 无 liger / 单机 | 4096 | 31.59 | ~143 | 133.89 | 3d 8h 24m 51s | 1h 45m 18s |
| Q | WDS | deepspeed zero2 + liger / 单机 | 4096 | 28.74 | ~155 | 134.06 | 3d 1h 9m 18s | 1h 35m 48s |
| R | WDS | deepspeed zero1 + liger / 单机 | 4096 | 24.99 | ~177 | 136.88 | 2d 15h 36m 46s | 1h 23m 18s |
| S | WDS | ray+deepspeed zero1 + liger / 双节点 | 4096 | 13.59 | ~335 | 82.77 | 1d 10h 35m 37s | 45m 18s |
| T | WDS | megatron + liger / 单机 | 4096 | 36.59 | — | 110.15 | 3d 21h 8m 30s | 2h 1m 59s |
| U | WDS | megatron 无 liger / 单机 | 4096 | 40.63 | — | 132.91 | 4d 7h 25m 34s | 2h 15m 27s |
| V | WDS | deepspeed 直连 zero0 + liger / 双节点 | 4096 | 12.85 | ~354 | 100.04 | 1d 8h 42m 26s | 42m 50s |
| **AN** | **Lance** | **deepspeed zero0 / 单机** | **2048** | **10.610（中位）/10.769（均值）** | **193.0** | **101.11** | **2d 7h 47m 46s** | **36m 8s** |
| **AO** | **Lance** | **ray+deepspeed zero0 + liger / 双节点** | **4096** | **18.983（中位）/19.410（均值）** | **215.8** | **100.86** | **2d 4h 26m 27s** | **1h 8m 40s** |

### 3.2 loss 逐 10 步（0–200，全量任务）

同一 GBS 的任务 loss 可直接横比；跨 GBS（512/2048/4096）loss 量级与步语义不同，仅看趋势形态。

**GBS≈512 族（A–M）**

| step | A | B | C | D | E | F | G | H | I | J | K | L | M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.7711 | 16.9301 | 16.6700 | 16.9308 | 16.6630 | 16.9308 | 16.9126 | 17.0005 | 16.5168 | 17.0315 | — | 17.0315 | 16.0321 |
| 10 | 16.8859 | 17.0034 | 16.6273 | 17.0021 | 16.6337 | 17.0021 | 17.0228 | 16.9965 | 16.6273 | 17.0031 | — | 17.0035 | 16.8724 |
| 20 | 16.8164 | 17.0098 | 16.6189 | 17.0248 | 16.6144 | 17.0248 | 17.0237 | 16.9984 | 16.6020 | 17.0135 | — | 17.0062 | 17.1419 |
| 50 | 14.9654 | 15.0257 | 14.7175 | 15.0348 | 14.7148 | 15.0348 | 15.0905 | 15.0212 | 14.7187 | 15.0201 | — | 15.0196 | 14.8959 |
| 100 | 9.2921 | 9.2522 | 9.2203 | 9.2530 | 9.2191 | 9.2530 | 9.3027 | 9.2422 | 9.2151 | 9.2501 | — | 9.2542 | 9.1210 |
| 150 | 7.1266 | 7.0984 | 7.0953 | 7.1075 | 7.0977 | 7.1075 | 7.1340 | 7.1098 | 7.1248 | 7.1140 | — | 7.1215 | 7.2296 |
| 200 | 6.3941 | 6.3625 | 6.3719 | 6.3605 | 6.3765 | 6.3605 | 6.3782 | 6.3696 | 6.3742 | 6.3619 | 6.3704 | 6.3645 | 6.3530 |

**GBS=4096 / 2048 族（P–V、AN、AO）**

| step | P | Q | R | S | T | U | V | AN（GBS=2048） | **AO（GBS=4096）** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.00736 | 17.00934 | 17.00934 | 17.01063 | 16.64840 | 16.63654 | 17.01063 | 16.9974 | **17.0007** |
| 10 | 16.95137 | 16.95098 | 16.95006 | 16.94766 | 16.58880 | 16.58691 | 16.96716 | 17.0044 | **16.9641** |
| 20 | 15.47657 | 15.48022 | 15.48258 | 15.47841 | 15.16160 | 15.16254 | 15.53949 | 16.5343 | **15.5216** |
| 50 | 8.39469 | 8.39426 | 8.39267 | 8.42175 | 8.41107 | 8.41596 | 8.42119 | 9.8140 | **8.4174** |
| 100 | 6.06572 | 6.06791 | 6.06713 | 6.03539 | 6.07335 | 6.06909 | 6.02928 | 6.4781 | **6.0685** |
| 150 | 5.73252 | 5.73236 | 5.72729 | 5.72092 | 5.73519 | 5.73250 | 5.71611 | 5.9534 | **5.7254** |
| 200 | 5.59450 | 5.59395 | 5.58948 | 5.59102 | 5.58777 | 5.59254 | 5.59615 | 5.7571 | **5.5979** |

### 3.3 DataLoader 队列专项（200K 热缓存子集，step 50–199）

以下任务在 200K 子集（48 GiB，进节点页缓存）上测，**不能与全量任务横比**；只比吞吐 / FillRate / MFU。

| 编号 | workers × prefetch | Q/rank | shm | FillRate | Wait p95 (ms) | step 中位（s） | median tokens/s | median MFU |
|---|---|---:|---|---:|---:|---:|---:|---:|
| W | 16 × 8 | 128 | 64Gi | 0.99776 | 4.2 | 1.512 | 38,907 | 7.6% |
| X | 8 × 16 | 128 | 64Gi | 0.99776 | 3.8 | 1.515 | 38,993 | 7.6% |
| Y | 16 × 16 | 256 | 128Gi | 0.99779 | 3.9 | 1.505 | 39,186 | 7.7% |
| Z | 8 × 24 | 192 | 128Gi | 0.99730 | 3.6 | 1.510 | 39,174 | 7.7% |
| AA | 0 × 8 | 0 | 64Gi | **0.88904** | **213** | **1.700** | **34,709** | 6.8% |
| AB | 1 × 1 | 1 | 64Gi | 0.99807 | 3.2 | 1.500 | 39,399 | 7.7% |
| AC | 2 × 2 | 4 | 64Gi | 0.99780 | 3.9 | 1.509 | 39,212 | 7.7% |
| AD | 4 × 8 | 32 | 64Gi | 0.99802 | 3.3 | 1.501 | 39,314 | 7.7% |
| AG | 1 × 8 | 8 | 64Gi | 0.99799 | 3.39 | 1.510 | 39,103 | 7.64% |

结论：队列膝点在 **Q=0 与 Q=1 之间**（AA 掉到 FillRate 0.889、Wait p95 213 ms；AB 已喂满）。
tokens/s 差 ≤0.7%，**保持 8 × prefetch 2（Q=16）即可**，加深无收益、过深会爆 shm（16×32 → ~82 GiB 必爆）。

### 3.4 优化项结论

| 优化项 | 对照 | 结论 |
|---|---|---|
| 数据通路 WDS → Lance | B→H、C→I、D→J、E→K、F→L | **采纳**：同族，200 步墙钟提速 1.15–1.51× |
| 拓扑扩展（deepspeed 直连） | B→F→G | **采纳**：SE≈86%；三节点（G）全场最快、显存最省 |
| 拓扑扩展（megatron） | C→E | **采纳**：SE≈80% |
| 后端 deepspeed vs megatron | B vs C、F vs E | deepspeed 更快（单机 1.4×） |
| liger-kernel 融合算子 | P→Q、U→T | **采纳**：稳态提速 ≈9–10%，跨后端一致 |
| ZeRO 引擎 zero2→zero1 | Q→R | **采纳**：−12.33% step，代价 +2.8 GiB |
| ZeRO-1→zero0（双节点） | S→V | 有条件采纳：−5.3% step，显存 +20.9%（单机 zero0 OOM，双节点才可行） |
| GBS 512→4096 | B→P | 吞吐 138→143 样本/s（微增），loss 换族 |
| DataLoader 队列深度 | W–Z、AA–AD | **不加深**：膝点在 Q=0→1，Q=16 已喂满 |
| group_by_length | AE→AF、H→M | 有条件：QPS +43.7%，但 loss 变族 + 显存翻倍 |
| GBS→2048（单机 per_device 32 × ga8 + zero0） | AN | 吞吐 193 样本/s，显存 101.11 GiB 稳定 |
| GBS→4096 + liger（双节点 ray + zero0） | AO | 吞吐 216 样本/s，显存 100.86 GiB，loss@200 与 P–V 一致 |

---

## 4. 指标口径与可比性

### 4.1 指标定义

| 指标 | 定义 | 读法 |
|---|---|---|
| s/it（稳态瞬时） | step 190→200 的瞬时步时 | 排除冷启动的纯吞吐口径 |
| s/it（累计均值） | HF Trainer `train_speed @200 = elapsed / step` | 含冷启动（map / 首步编译 / warmup） |
| remaining_time | `(max_steps − step) × train_speed`，trainer 自报 | 外推**全程**；总步数因 GBS 而异 |
| FillRate | `1 − mean(t_wait) / mean(t_step)`，`t_wait` = 阻塞在 `next(train_dataloader)` 的墙钟 | ≥0.95 即 DataLoader 不是瓶颈；稳态窗口 step 50–199 |
| MFU | `FLOP_FACTOR × N × tokens_per_step / s_per_step / DP / 1e12 ÷ 989 TFLOPS`（H200 bf16 dense 峰值） | 6N 上界对多模态偏乐观，**只做同配置前后对比** |
| PadWaste | `1 − mean(batch 内真长) / max(batch 内真长)` | 高说明 GPU 时间耗在 padding |
| SE（Scaling Efficiency） | 加速比 ÷ 节点数比 | 含冷启动，故低于稳态瞬时口径 |

### 4.2 可比族

| 族 | 成员 | 前提 |
|---|---|---|
| **GBS≈512 全量族** | A、B、C、D、E、F、G、H、I、J、K、L | 同 LR 日程（warmup ≈2247）、同 max_length、同 eval/save=1000；loss/acc 可逐 step 比 |
| **GBS=4096 全量族** | P、Q、R、S、T、U、V | 同 ga=8、同 9364 步；loss 可逐 step 比（T/U 的 `codec_token_acc` 仅看趋势） |
| **200K 热缓存族** | W、X、Y、Z、AA、AB、AC、AD、AE、AF、AG | 同子集同窗口（step 50–199）；只比吞吐 / FillRate / MFU，**不外推全量 NVMe 随机读** |

### 4.3 明确不可横比

| 组合 | 原因 |
|---|---|
| A–G / H–L ↔ P–V | GBS 512 vs 4096：optimizer step 语义、warmup 步数、loss 量级都不同 |
| AH ↔ AI 的 loss 数值 | megatron 输出 codec 侧、deepspeed 输出总体，口径不同 |
| 200K 子集族 ↔ 全量族 | 48 GiB 子集进节点页缓存，测不出 6.2 TiB 全表的随机读压力 |
| 任何列 ↔ AJ / AK / AL | 长期训练（多天、多阶段、不同 LR），非 200 step 对照 |
