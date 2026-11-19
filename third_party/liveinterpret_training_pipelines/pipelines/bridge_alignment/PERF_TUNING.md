# 训练性能参数参考（ms-swift megatron + PyTorch/CUDA/NCCL/TE/transformers）

本文覆盖**训练速度、GPU 负载、数据预处理效率**相关的参数，范围不限于当前
`pipelines/bridge_alignment` 脚本已用的参数。

数据来源（均实测/源码核实，非臆造）：
- ms-swift：`swift/megatron/arguments/megatron_args.py`、`swift/megatron/trainers/`（容器内 `/opt/venv/lib/python3.12/site-packages`）
- 运行时环境变量：训练进程 `/proc/<pid>/environ`
- 本环境版本：CUDA 12.8 / NCCL 2.25.1 / PyTorch 2.8.0+cu128 / transformers 4.57.6 /
  TransformerEngine；硬件 H200（`TORCH_CUDA_ARCH_LIST=8.9;9.0`），单节点 8 GPU，
  当前拓扑 TP=8 / PP=1 / CP=1 / DP=2。

> 标注约定：**默认值**取自源码 dataclass 默认；**本环境**给出结合 H200/当前拓扑的建议或约束。

---

## 一、并行策略（决定 GPU 负载分布与通信开销）

ms-swift megatron 参数（CLI 前缀 `--`）：

| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `tensor_model_parallel_size` (TP) | 1 | 每层权重矩阵切到多卡，一层前向需多卡拼算；每层 2 次 activation all-reduce（高频通信） | 当前 8。3B 模型 TP=8 会把 GEMM 切得过小、kernel 填不满 SM，是利用率波动的结构性来源；模型能单卡放下时可降 TP、增 DP |
| `pipeline_model_parallel_size` (PP) | 1 | 按层切分成流水段 | codec loss 要求 **=1**（约束，勿改） |
| `context_parallel_size` (CP) | 1 | 序列维切分（长序列） | codec loss 要求 **=1** |
| `expert_model_parallel_size` (EP) | 1 | MoE 专家切分 | thinker 冻结，当前 1 |
| `sequence_parallel` | false | TP 基础上再切 LayerNorm/Dropout 的序列维，省激活显存 | codec loss 要求 false；**注意：`tp_comm_overlap` 依赖它为 true** |
| `virtual_pipeline_model_parallel_size` | None | 交错式流水（减少 PP 气泡） | PP=1 时无意义 |
| `use_distributed_optimizer` | true | ZeRO-1：优化器状态分片到 DP 组，省显存 | 保持 true |

DP（数据并行度）不直接设，由 `world_size / (TP×PP×CP)` 推导。

---

## 二、通信 / 计算重叠（直接影响 GPU 负载平滑度）

| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `tp_comm_overlap` | false | 把 TP 的 all-reduce 与 GEMM 计算重叠，削掉卡间等待谷 | **需 `sequence_parallel=true`**（源码强制：否则 `raise ValueError`）。当前 codec loss 要求 sequence_parallel=false，故**当前无法开启** |
| `overlap_grad_reduce` | false | 反向时梯度 all-reduce 与反向计算重叠 | **本环境暂锁 false（不可试开）**：0.17.0 引入「首 batch 每个 param 必须登记 grad-ready」不变量（`_ParamAndGradBucketGroup.reset()` 的 `assert len(per_param_grad_ready_counts)==len(params)`）；根因在 talker 损失 `mcore_talker_loss.py:515` 的 `codec_loss` 仅在 `num_tokens>0` 时计算，`num_tokens==0` 的 microbatch 的 talker 参数首步拿不到梯度 → assert 失败。须先修复「未用/条件 param」（结构性未用→`requires_grad=False`；数据相关→保证每 microbatch ≥1 有效 codec 标签或加梯度锚点 loss）后方可开启 |
| `overlap_param_gather` | false | DistributedOptimizer 的参数 all-gather 与前向重叠 | **当前本环境设 true**（依赖 `use_distributed_optimizer=true`，已满足，安全可开）；但本 swift 版本**不接收 `--align_param_gather` CLI**，导致重叠退化为「未对齐 gather」（`finish_param_sync` 告警），实测无提速。须配 plugin hook 强制 `ddp_config.align_param_gather=True`（见下 `align_param_gather`）才是真正生效 |
| `overlap_param_gather_with_optimizer_step` | false | 参数 gather 与 optimizer step 重叠 | 进阶，可试（同受 `align_param_gather` CLI 不生效约束） |
| `overlap_p2p_comm` | true | PP 的 P2P 通信重叠 | PP=1 时无效 |
| `align_param_gather` | true | 对齐参数 gather 时机，使 `overlap_param_gather` 真正生效（官方文档 0.17.0 配对项） | **CLI 不生效**：此 swift 版本 `MegatronSftArguments` 未暴露 `--align_param_gather` 可覆盖字段，传参被丢弃（args dump 仍 `align_param_gather=False`，而同组 `overlap_param_gather` 正常透传）。dataclass 默认虽为 true，但 swift 构造 `ddp_config` 时未透传，实际以 False 运行 → `overlap_param_gather` 重叠退化。修复：在 `custom/plugin.py` 增加 hook 强制 `ddp_config.align_param_gather=True` |

---

## 三、显存 / 重计算（用算力换显存；显存富裕时应关）

| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `recompute_granularity` | selective | `full`=反向重算整层前向（省显存最多、算力开销 +30~40%）；`selective`=只重算 attention 等热点；`none`=不重算 | H200 143GB 显存富裕，当前设 `none` 最快 |
| `recompute_method` | None | `uniform` / `block` | **仅 granularity=full 可设**；selective 下设置会 `raise ValueError` |
| `recompute_num_layers` | None | 重算的层数 | 仅 full 生效 |
| `recompute_modules` | ['core_attn'] | selective 模式重算哪些模块 | selective 时可调 |
| `moe_layer_recompute` | false | MoE 层单独重算 | thinker 冻结，无关 |

---

## 四、Batch / 吞吐

| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `micro_batch_size` (MBS) | 1 | 单次 forward 的样本数；越大 GEMM 越大、SM 越满 | 当前 8。`num_microbatches = GBS/(MBS×DP)` |
| `global_batch_size` (GBS) | 16 | 每次 optimizer.step 的有效 batch；**改它会改变收敛，需同步调 LR** | 当前 16。想减 step 数才动它 |
| `num_microbatches` | 派生 | = GBS/(MBS×DP)；>1 时步内梯度累积可流水（平滑），=1 时单步单次 fwd/bwd（尖锐） | 当前 16/(8×2)=1 |

关联：**平滑负载**可用 MBS 从 8→4 使 num_microbatches=2（流水削峰，GBS 不变、收敛不变）。

---

## 五、算子融合 / 精度

| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `attention_backend` | flash | `flash`/`fused`/`unfused`/`local`/`auto`。flash 最快但只接受 causal/varlen mask | 本环境已改 **flash 并验证可用**：thinker 实际构造纯 causal+padding mask，flash 接受（运行时打印 `(flash_attention): FlashAttention()`）。原「多模态非因果 mask 只能 unfused」断言已证伪。talker（HF 侧）的 4D block-causal mask 仍走 eager/SDPA，与 mcore 后端开关无关 |
| `cross_entropy_loss_fusion` | true | 融合 CE，省显存提速 | 保持 true |
| `cross_entropy_fusion_impl` | native | `native`/`te` | 默认 |
| `gradient_accumulation_fusion` | true | 梯度累积融合进 GEMM 反向 | 保持 true（某些配置会自动关） |
| `bf16` | None | bf16 混合精度 | 本环境 bf16 |
| `main_grads_dtype` | fp32 | 主梯度精度 | 保持 fp32（稳定） |
| `exp_avg_dtype` / `exp_avg_sq_dtype` | fp32 | Adam 一/二阶矩精度，可设 bf16/fp8 省显存 | 显存富裕，无需降 |
| `use_precision_aware_optimizer` | false | 低精度优化器状态 | 显存富裕，不需要 |

### FP8（H200 支持，可显著提速，但影响数值）
| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `fp8_format` | None | `e4m3`/`hybrid`，开启 FP8 训练 | H200 支持 FP8；开启需验证 codec loss 数值稳定性，谨慎 |
| `fp8_recipe` | delayed | `tensorwise`/`delayed`/`mxfp8`/`blockwise` | 配合 fp8_format |
| `fp8_param_gather` | false | FP8 下参数 gather 用 FP8 | 进阶 |
| `fp8_amax_history_len` | 1024 | amax 历史长度 | 默认 |

---

## 六、优化器 / GC / 显存管理

| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `optimizer_cpu_offload` | false | 优化器状态卸载到 CPU，省显存但慢 | 显存富裕，**不要开**（会拖慢） |
| `calculate_per_token_loss` | 派生 | 按 token 归一 loss | causal_lm 默认 true |
| `manual_gc` | false | 手动控制 Python GC，减少训练中 GC 抖动 | 长训练可试开，配 `manual_gc_steps` |
| `manual_gc_steps` | 0 | 每 N step 手动 GC 一次 | 配合 manual_gc |
| `manual_gc_eval` | true | eval 时手动 GC | 默认 |

---

## 七、MoE（thinker 冻结，当前基本无关，列作参考）

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `moe_grouped_gemm` | true | 分组 GEMM，MoE 提速 |
| `moe_permute_fusion` | false | 融合 permute |
| `moe_token_dispatcher_type` | alltoall | `allgather`/`alltoall`/`flex` |
| `moe_shared_expert_overlap` | false | 共享专家与路由重叠 |
| `moe_expert_capacity_factor` | None | 专家容量因子 |

---

## 八、数据加载 / 预处理（数据供给效率）

### DataLoader（steady-state 供给）
| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `dataloader_num_workers` | 4 | 每个 DataLoader fork 的子进程数。**实际常驻 worker = num_workers × 2(train+val loader) × rank 数**（源码 `_create_dataloader` 被 train/val 各调一次） | 当前 8 → 8×2×8=128 worker/节点。cgroup 限 150 核，勿超订 |
| `dataloader_prefetch_factor` | 2 | 每 worker 预取的 batch 数（深度）。加深比加 worker 更能覆盖 codec 编码长尾 | 当前 4~6 |
| `dataloader_persistent_workers` | false | worker 跨 epoch 保活，避免每 epoch 重建（重载 codec tokenizer）的停顿 | 当前 true |
| `dataloader_pin_memory` | true | 锁页内存加速 H2D 拷贝 | 保持 true |
| `train_dataloader_shuffle` | true | 训练集 shuffle | 保持 true |

### 数据构建 / 编码
| 参数 | 默认 | 作用 | 本环境 |
| --- | --- | --- | --- |
| `dataset_num_proc` | — | `load_dataset .map/filter` 的构建期并发进程数（**仅构建期，不影响 steady-state**） | 当前 32，构建 arrow cache 时吃满 CPU |
| `packing` | — | 多样本拼成定长序列提吞吐 | codec loss 要求 **false**（拼接破坏 codec↔semantic 对齐） |
| `padding_free` | true | 去 padding 拼接 | codec loss 要求 **false**（guard 硬失败） |
| `max_length` | — | 序列截断长度；越大单样本算力越大 | 当前 **2048**（由 4096 下调，已验证：省算力 + 降 OOM） |
| `streaming` | — | 流式数据集（不全量 load） | 用时需显式 `train_iters`；当前非流式 |

---

## 九、环境变量（PyTorch / CUDA / NCCL / TE / OMP / transformers）

以下为**当前进程 environ 实测存在**的值。

### CUDA 显存分配器
| 变量 | 当前值 | 作用 |
| --- | --- | --- |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 可扩展显存段，减少碎片与 OOM。长序列变长场景强烈建议开 |

### NCCL（集合通信 / 跨节点）
| 变量 | 当前值 | 作用 |
| --- | --- | --- |
| `NCCL_TIMEOUT` / `TORCH_NCCL_TIMEOUT` | 1800 | 集合通信超时（秒）。看门狗：错位/死锁时硬失败而非静默挂起 |
| `NCCL_ASYNC_ERROR_HANDLING` / `TORCH_NCCL_ASYNC_ERROR_HANDLING` | 1 | 异步错误处理，通信出错快速抛出 |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 300 | 心跳超时 |
| `NCCL_IB_DISABLE` | 0 | 启用 InfiniBand/EFA RDMA（跨节点带宽关键，勿设 1） |
| `NCCL_SOCKET_IFNAME` | eth0 | NCCL 走的网卡 |
| `NCCL_DEBUG` | INFO | 日志级别。`INFO` 刷屏严重，稳定后可设 `WARN` 减少 stdout 淹没 |
| `NCCL_DEBUG_SUBSYS` | INIT,NET,ENV | 调试子系统过滤 |
| `TORCH_DISTRIBUTED_DEBUG` | OFF | `DETAIL` 可做集合通信 shape 校验（定位错位），有性能开销，平时 OFF |

### CPU 线程 / GEMM
| 变量 | 当前值 | 作用 |
| --- | --- | --- |
| `OMP_NUM_THREADS` | 1 | 每进程 OpenMP 线程数。多进程（如 128 dataloader worker）场景设 1 防线程爆炸；纯 CPU 计算密集可适当增 |
| `CUDA_DEVICE_MAX_CONNECTING` (可选) | 未设 | 增大可提升 TP overlap 的并发流（配合 tp_comm_overlap 才有意义） |

### TransformerEngine / torch.compile
| 变量 | 当前值 | 作用 |
| --- | --- | --- |
| `NVTE_TORCH_COMPILE` | 0 | 关闭 TE 的 torch.compile |
| `TORCH_COMPILE_DISABLE` | 1 | 全局关闭 torch.compile。本项目因 mcore MoE grouped-GEMM 的 Dynamo 假张量 tracing 失败而必须关（约束，勿开） |
| `TORCH_CUDA_ARCH_LIST` | 8.9;9.0 | 编译目标架构（H200=9.0） |

### transformers / tokenizers
| 变量 | 建议 | 作用 |
| --- | --- | --- |
| `TOKENIZERS_PARALLELISM` | false（多 worker 时） | HF fast tokenizer 的 Rust 并行；在 dataloader 多进程 fork 下设 false 避免死锁与 CPU 争抢 |

---

## 十、关键关联关系（改一个必须考虑另一个）

1. **`QWEN3_TTS_TOKENIZER_DEVICE=cpu` → dataloader 并发才安全生效**
   codec tokenizer 在 CPU（见 `custom/codec_tokenizer.py::_default_device_map`）→ 无 fork/CUDA
   重初始化冲突 → `dataloader_num_workers>0` 的多进程编码才可用。若设为 GPU，多 worker fork
   会触发 CUDA re-init 错误，此时 dataloader 并发相关参数（workers/prefetch/persistent）失去意义。

2. **`tp_comm_overlap=true` 依赖 `sequence_parallel=true`**（源码硬约束）
   而 codec loss 要求 `sequence_parallel=false` → 当前**无法用 tp_comm_overlap** 削 TP 通信谷。

3. **`recompute_method`/`recompute_num_layers` 仅当 `recompute_granularity=full`**
   selective 下设 method 会 `raise ValueError`；none 下全不传。

4. **`num_microbatches = GBS/(MBS×DP)`**
   改 MBS 或 DP 会改 num_microbatches → 影响步内流水粒度（负载平滑度），但只要 GBS 不变，
   收敛不变。改 GBS 才动收敛（需调 LR）。

5. **`packing`/`padding_free` 必须为 false**（codec loss 约束）
   否则多样本拼接破坏 codec↔semantic 对齐 + guard 硬失败。

6. **`overlap_param_gather` 依赖 `use_distributed_optimizer=true`**。

7. **dataloader 常驻 worker 数 = `dataloader_num_workers × 2 × rank数`**，且受 pod cgroup
   CPU 限额约束；配 `persistent_workers=true` 时全程常驻，调大 workers 前先算总数勿超核数。

---

## 十一、针对当前环境的可动/不可动清单

**可正向调（不改收敛）**：`recompute_granularity`(none)、`micro_batch_size`、
`dataloader_num_workers/prefetch_factor/persistent_workers`、
`manual_gc`、`NCCL_DEBUG`(→WARN)、`OMP_NUM_THREADS`、`TOKENIZERS_PARALLELISM`。

**已启用但受约束**：`overlap_param_gather=true`（安全可开，但须 plugin hook 强制 `align_param_gather=True` 才真正生效，否则重叠退化、无提速）；`attention_backend=flash`（已验证 thinker 可用，原 unfused 约束已证伪）。

**暂不可动（待修复）**：`overlap_grad_reduce`——0.17.0 首 batch grad-ready 不变量，因 talker 损失 `num_tokens>0` 门控导致部分 param 首步无梯度而 assert 失败，须先修「未用/条件 param」方可开。

**需谨慎/验证**：`fp8_format`（数值）、`global_batch_size`（需调 LR）、`exp_avg_dtype`(fp8)。

**不可动（硬约束）**：`pipeline_model_parallel_size=1`、`context_parallel_size=1`、
`sequence_parallel=false`、`packing=false`、`padding_free=false`、
`TORCH_COMPILE_DISABLE=1`、`NVTE_TORCH_COMPILE=0`（均为 codec loss / 多模态 mask / mcore
compile 的既有约束，改动会导致 guard 失败或报错）。

---

## 附录：训练优化系统方法论（通用参考）

> 本附录与正文解耦，从**训练完整生命周期**角度梳理通用优化地图。正文各环节（数据准备 / 训练中速度 / eval-ckpt）都能在下方找到其所属阶段与方法类别。附录内容不针对本 repo，可迁移到任意大模型训练任务。

### A.1 预处理阶段（离线 / 训练启动前）

在训练启动前把能离线完成的工作一次性做完，避免每次启动重算：

- **数据清洗与过滤**：去重、超长/超短样本剔除、缺失样本丢弃、异常/低质样本过滤；既防训练期 OOM，也提升数据质量（对应 `PERFORMANCE_OPTIMIZATION_PLAN.md` 1.3 的时长阈值过滤）。
- **特征 / 中间产物预烘焙**：tokenize、音频时长、特征抽取、统计字段等可离线计算的量一次性落盘，后续直接读（对应 1.4 的 jsonl→arrow 预烘焙）。
- **数据格式与 IO**：文本 jsonl → 预转 arrow/parquet；webdataset（tar shard）按成员 offset/length 精确定位，避免整文件读取；对象存储（S3/FSX）与本地 NVMe 缓存的取舍。
- **数据集索引操作**：向 Arrow 表追加常量列时避免整表 flatten 重写（保留 indices 横向拼接）、flatten_indices 并行化（对应 1.2）。
- **缓存管理**：`load_from_cache_file` 复用同节点 arrow 缓存；严禁中途 kill 导致缓存截断（对应 1.4、纪律 4）。

**效果归类**：提高启动速度、降低 IO、缩减不必要的重复计算。

### A.2 DataLoader 阶段（在线喂数）

决定 GPU 是否"饿着"，目标是让喂数永不成为瓶颈：

- **并行与预取调优**：`num_workers` / `prefetch_factor` / `persistent_workers` 配合，瓶颈诊断看 GPU 利用率是否随 step 抖动。
- **group_by_length / bucket batching**：按长度分桶采样，把 batch 内 pad 目标从"全局最长"压到"桶内典型长度"（对应 `PERFORMANCE_OPTIMIZATION_PLAN.md` 2.3）。
- **packing / sample concatenation**：多样本拼成定长流，提升 token 密度；需 loss 与 mask 适配（多模态语音训练普遍不支持，见 2.3 说明）。
- **排序去 pad / padding_free / THD varlen**：去除 pad token，走变长注意力。
- **解码后端切换**：音频 librosa→soundfile、图像 decode 提前到 CPU 预取流水线，降低训练期主线程开销。
- **数据增强前置**：把 augment 放到预处理，避免训练期 CPU 抖动。

**效果归类**：提高 GPU 喂数利用率、缩减无效 padding 计算、降低 CPU↔GPU 等待。

### A.3 前向计算

决定单 step 的计算量与算力利用率：

- **融合注意力 kernel**：FlashAttention（causal / varlen / sliding-window）避免物化 S×S；FlexAttention 用 `mask_mod` 闭式表达不规则 mask（需 `torch.compile`）（注意力后端选型见本文第二/五章）。
- **注意力后端选型**：flash / fused / unfused / eager 按 mask 形态选择，融合后端需 mask 可闭式表达。
- **冻结模块 `torch.no_grad()` 包裹**：省去冻结段的激活留存与反向（仅当显存/计算确有收益时考虑，本任务已由 `recompute_granularity=none` + `max_length=2048` 解决，见 `PERFORMANCE_OPTIMIZATION_PLAN.md` 2.2）。
- **混合精度 / 低精度**：bf16 / fp16 / fp8（TransformerEngine），直接抬高算力上限。
- **CUDA Graph**：定长输入下消除 kernel launch 开销（packing 后重新评估）。
- **算子融合与量化**：layernorm+geglu 融合、weight-only / activation 量化、MoE 路由优化。
- **KV cache / 前缀共享**：生成与推理阶段的显存与计算优化。

**效果归类**：提高速度、提高算力利用率、缩减不必要计算与显存占用。

### A.4 反向计算

决定显存峰值与"草稿纸"成本：

- **梯度检查点 / recompute**：用重算换激活显存；粒度可选（整段 vs 每层 vs 前 K 层）。本任务 H200 显存充裕，保持 `recompute_granularity=none` 即最快（对应 `PERFORMANCE_OPTIMIZATION_PLAN.md` 2.2）。
- **冻结参数不进 grad buffer**：DDP/ZeRO 下冻结参数不入梯度缓冲，省通信与显存。
- **混合精度 backward**：bf16 主权重、fp16 loss scaling。
- **选择性反向**：只对可训分支建图（no_grad 包裹冻结塔）。
- **offload / CPU 卸载激活**：ZeRO-offload 类手段缓解显存。

**效果归类**：节省显存以换更大 batch/MBS、缩减不必要激活留存与重算量。

### A.5 梯度同步（通信）

多卡训练下通信往往与计算争抢时间：

- **并行策略**：DDP vs ZeRO-1/2/3，按参数/梯度/优化器状态切分；TP/PP/CP/EP 的层内与层间切分。
- **通信-计算重叠**：梯度通信与反向传播 overlap（backward 过程中逐层 all-reduce）；本环境 `overlap_param_gather` 已开但退化、`overlap_grad_reduce` 暂锁（见本文二章）。
- **通信后端与拓扑**：NCCL + NVLink / IB / RDMA，NCCL 线程与网络参数调优。
- **通信压缩**：fp16 梯度、1-bit Adam、top-k 稀疏化。
- **梯度累积**：多 micro-batch 只在最后一步 all-reduce，摊薄通信频率。
- **异步 / 延迟通信**：降低关键路径上的同步阻塞。

**效果归类**：提高硬件利用率（隐藏通信）、缩减通信时间、改善多卡扩展比。

### A.6 检查点保存（ckpt）

周期性开销，目标是"保存不卡训练"：

- **异步保存**：保存与训练重叠（background / async ckpt），缩短卡训练时长（对应 `PERFORMANCE_OPTIMIZATION_PLAN.md` 第 3 章）。
- **分片保存**：sharded / distributed checkpoint 按 rank 写各自分片，避免单点 I/O 瓶颈。
- **频率与保留数**：`save_steps` / `save_total_limit` 平衡可恢复性与开销（对应第 3 章）。
- **增量保存 / 只存可训参数**：减少写入量。
- **格式**：safetensors / mcore distributed checkpoint 等快速读写格式。

**效果归类**：降低周期性开销、提高训练连续性、缩短故障恢复时间。

### A.7 优化效果分类总览

| 阶段 | 典型手段（部分） | 主要效果 |
|---|---|---|
| 预处理 | 数据过滤、特征预烘焙、jsonl→arrow、缓存复用 | 提高启动速度、降低 IO、缩减重复计算 |
| DataLoader | group_by_length、packing、padding_free、解码后端切换 | 提高喂数利用率、缩减 padding 计算 |
| 前向 | FlashAttention/FlexAttention、no_grad、fp8、CUDA Graph、算子融合 | 提高速度、提高算力利用率、缩减计算 |
| 反向 | recompute K、冻结不建图、混合精度 backward、offload | 节省显存换大 batch、缩减激活重算 |
| 梯度同步 | DDP/ZeRO、通信-计算重叠、通信压缩、梯度累积 | 提高利用率、缩减通信时间 |
| ckpt 保存 | 异步保存、分片保存、频率调优 | 降低周期开销、缩短恢复时间 |

> 优化效果可归纳为三大类：**① 提高速度**（缩短 step 时间）、**② 提高硬件使用率**（消除 GPU 饿喂 / 通信空闲 / 显存碎片）、**③ 缩减不必要的计算**（冻结段免反向、padding 免算、缓存免重算、坏样本早滤）。正文各环节方案均可对号入座到上述三类之一。
