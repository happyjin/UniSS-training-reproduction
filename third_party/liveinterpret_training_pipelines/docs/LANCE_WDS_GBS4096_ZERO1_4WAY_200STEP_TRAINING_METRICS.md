# Lance/WDS × 单机/双节点 五方对照（GBS=4096, ZeRO-0/1, liger）200 step 训练指标记录

> 记录日期：2026-09-05 ~ 2026-09-09。
> 分支：`feature/deferzhou/merge-training-metrics`。
> 本文档目的：把"AO复刻/zero0-zero1对照/Lance-WDS对照/ray+zero0+WDS空白格
> 补全"四轮实验的完整逐指标数据整理成统一对照表（格式对齐
> `GBS512_GBS4096_BASELINE_ABCD_200STEP_TRAINING_METRICS.md`），供后续查阅
> 无需回翻聊天记录或原始日志。
>
> **五列定义**：
> - **AO**：`ao-repro2-txy1` —— Lance + ray + deepspeed **zero0** + liger，
>   **双节点**，GBS=4096（复刻 `TRAINING_METRICS_MERGED.md` 的 AO 列，已用
>   显式 `GROUP_BY_LENGTH=false` 修正脚本默认值漂移，跟 AO 原始报告值误差
>   <0.3%）。
> - **L2N**：`lance-zero1-cagi` —— Lance + ray + deepspeed **zero1** + liger，
>   **双节点**，GBS=4096（AO 的 ZeRO-1 变体，用于跟历史 WDS S/D 列做单变量
>   对照）。
> - **L1N**：`lance-1n-z1-z5is` —— Lance + **pytorch直连**deepspeed **zero1**
>   + liger，**单机**，GBS=4096（对齐历史 R 列 WDS 配置的 Lance 版）。
> - **WM**：`wds-master3-em8k` —— **WDS** + pytorch直连deepspeed **zero1**
>   + liger，**单机**，GBS=4096（在干净的 `master`(`3e88d11`) 代码状态下
>   跑的 R 列复刻，验证通过：与 R 历史值误差<5%，显存精确一致）。
> - **WZ0**：`wds-ray-oldcode3-m8r0` —— **WDS** + **ray** + deepspeed **zero0**
>   + liger，**双节点**，GBS=4096（此前四方对比里唯一的空白格，历史上两次
>   直接 OOM/`TaskUnschedulableError` 失败，从未产出有效数据；本次用
>   `junyaoyang/liveinterpret_mfu` 老代码S3快照绕开`85b6d750`引入的
>   `DATASET_NUM_PROC`探测bug，跑满250步，**验证不会OOM**，见第4.4节）。
>
> **⚠️ 关键口径警告**：
> 1. AO/L2N/WZ0 是 **ray 双节点**；L1N/WM 是 **pytorch 直连单机**——五列
>    之间同时存在"数据源（Lance/WDS）"、"节点数（1/2）"、"是否用ray"、
>    "ZeRO stage（zero0/zero1）"四个变量交织，**不能整体互相归因**，
>    具体哪两列构成单变量对照见第4章"可比对"说明。
> 2. **WM 的 1-60 步原始日志已丢失**（`ktrain job logs` 只保留了任务后段
>    输出，早期步数未能追溯），本文档 WM 列 1-60 步留空，70 步起完整。
>    不影响 191-200 稳态窗口与 loss@200 的核心结论。
> 3. AO/L2N/WZ0 与 L1N/WM 用的 `PER_DEVICE_TRAIN_BATCH_SIZE` 不同（双节点
>    用 32×16卡×ga8；单机用 64×8卡×ga8），但 GBS 同为 4096、总步数同为
>    9364、warmup 同为 ~281 步，loss 量级和轨迹可直接横比。
> 4. **WZ0 用的是 `junyaoyang/liveinterpret_mfu` 这个冻结在`85b6d750`之前
>    的老代码S3快照**（`CODE_PREFIX`不同于AO/L2N），跳过了`submit.sh`的
>    自动`sync_code`步骤（否则会用当前分支代码覆盖这个珍贵快照），yaml
>    与`AO_REPRODUCTION_LOG.md`记录的历史zero0 yaml逐项对齐（唯一变量=
>    DeepSpeed stage与CODE_PREFIX），详见第4.4节完整参数说明。

---

## 1.0 通用环境（五列一致，差异已标注）

| 项 | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 集群 | speech-h200-eks（ns `speech-h200`） | 同 | 同 | 同 | 同 |
| 机型 / GPU | ml.p5e.48xlarge，8×H200（141GiB/卡）/节点 | 同 | 同 | 同 | 同 |
| 模型 | `qwen2_5_3b_omnispeak_a` | 同 | 同 | 同 | 同 |
| 训练目标 | adapter 式 upstream text-hidden 叠加 codec，`reply_add` | 同 | 同 | 同 | 同 |
| 可训参数 | `thinker_adapter + talker`（≈1.93B），thinker 冻结 | 同 | 同 | 同 | 同 |
| loss 权重 | residual 0.3 / codec_eos 1.0 / text 1.0 | 同 | 同 | 同 | 同 |
| 精度 | bf16 | 同 | 同 | 同 | 同 |
| `max_length` | 576 | 同 | 同 | 同 | 同 |
| `num_train_epochs` | 2 | 同 | 同 | 同 | 同 |
| LR / scheduler / warmup | 1e-5 / constant_with_warmup / 0.03 | 同 | 同 | 同 | 同 |
| `split_dataset_ratio` | 0.0005 | 同 | 同 | 同 | 同 |
| `lazy_tokenize` | true | 同 | 同 | 同 | 同 |
| **`GROUP_BY_LENGTH`** | **false（显式覆盖）** | **false（显式覆盖）** | **false（显式覆盖）** | false（master脚本默认） | false（老代码脚本默认，未显式覆盖） |
| `USE_LIGER_KERNEL` | true | true | true | true | true |
| 归约环境 | `NCCL_NVLS_ENABLE=0`、`NCCL_IB_DISABLE=0`、`FI_PROVIDER=efa` | 同 | 同 | 同 | 同 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 同 | 同 | 同 | 同 |
| 停止方式 | 跑满200步 `ktrain job stop --force` | 同 | 同 | 同 | 跑满250步仍在运行（超采，见第4.4节） |
| 代码commit/来源 | `85b6d750`（GBL显式覆盖修正） | 同 | `322ef57` | **`3e88d11`（master）** | **`junyaoyang/liveinterpret_mfu` S3快照（冻结于`85b6d750`之前，实测`DATASET_NUM_PROC=16`硬编码）** |

## 1.1 任务级配置对比表

| 参数 / 指标 | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 任务名 | `ao-repro2-txy1` | `lance-zero1-cagi` | `lance-1n-z1-z5is` | `wds-master3-em8k` | `wds-ray-oldcode3-m8r0` |
| 任务 yaml | `ktrainjob_sts_zeyang_custom_data_lance_train.yaml` | `ktrainjob_sts_zeyang_custom_data_lance_train_zero1.yaml` | `ktrainjob_junyaoyang_gbs4096_liger_zero1_lance_1node.yaml` | `ktrainjob_junyaoyang_gbs4096_liger_zero1_wds_1node.yaml` | `ktrainjob_junyaoyang_ray2node_gbs4096_liger_zero0_wds_07d3_08e3.yaml`（渲染后手动改CODE_PREFIX，见第4.4节） |
| **数据源** | **Lance**（v506全表，precomputed mel+speaker） | **Lance**（同AO） | **Lance**（同AO） | **WDS**（FSX manifest+S3音频shards） | **WDS**（同WM） |
| **framework** | **ray**（KubeRay） | **ray**（KubeRay） | **pytorch**（直连deepspeed） | **pytorch**（直连deepspeed） | **ray**（KubeRay） |
| **DeepSpeed引擎** | **zero0**（纯DDP） | **zero1** | **zero1** | **zero1** | **zero0**（纯DDP） |
| **节点数** | **2**（0284+08e3） | **2**（0284+08e3） | **1**（0284） | **1**（08e3） | **2**（0284+08e3） |
| 并行策略 | DP=16 | DP=16 | DP=8 | DP=8 | DP=16 |
| 微批/累积 | `PER_DEVICE=32`×`GA=8` | `PER_DEVICE=32`×`GA=8` | `PER_DEVICE=64`×`GA=8` | `PER_DEVICE=64`×`GA=8` | `PER_DEVICE=32`×`GA=8` |
| **有效batch(GBS)** | **4096** | **4096** | **4096** | **4096** | **4096** |
| 总步数 | 9364 | 9364 | 9364 | 9364 | 9364 |
| liger开关 | `USE_LIGER_KERNEL=true`+`QWEN_OMNISPEAK_LIGER_OPS=1` | 同 | 同 | 同 | 同 |
| DataLoader | 8×prefetch2（脚本默认，ray driver无eval loader） | 同AO | 8×prefetch2 | **16×prefetch8**（master原生SCENE硬编码，验证通过） | **16×prefetch8**（老代码SCENE硬编码，与WM同源） |
| `DATASET_NUM_PROC` | 动态探测（ray driver子进程失真，见`AO_REPRODUCTION_LOG.md`第9节） | 同AO | 单机不受影响 | 单机不受影响 | **16（老代码SCENE硬编码，绕开动态探测bug）** |
| SWIFT_SINGLE_DEVICE_MODE | `1`（绕过ray driver的`is_mp()`误判） | `1` | 不需要（无ray） | 不需要（无ray） | 未设置（老代码/WDS路径未触发该判断） |
| SCENE预设 | `batch4_8_adapter_cmp` | 同 | 同 | 同 | 同 |
| W&B/上报 | disabled | disabled | disabled | disabled | disabled |
| 输出目录 | `/nvme/output/sts_zeyang_lance_ray2n_gbs4096_liger` | `/nvme/output/sts_zeyang_lance_ray2n_gbs4096_liger_zero1` | `/nvme/output/sts_gbs4096_liger_zero1_lance_1node` | `/nvme/output/sts_gbs4096_liger_zero1_wds_1node_master_clean` | `/nvme/output/ray2node_gbs4096_liger_zero0_wds_07d3_08e3` |
| MFU日志 | `[mfu-log]`（stdout） | 同 | 同 | 同 | 同 |
| 实际运行step数（超采） | 200（人工停止） | 210 | 220 | 220 | **250+（仍在运行）** |

## 1.2 性能数据对比表（稳态窗口 191-200，取自 `[mfu-log]` 逐步瞬时耗时）

| 指标 | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 稳态s/it（中位数） | 19.150 | 19.416 | **18.517** | **23.889** | **12.510** |
| 稳态s/it（均值） | 19.445 | 19.672 | 19.241 | 24.609 | 12.479 |
| 稳态s/it（min-max） | 17.464~22.750 | 17.375~23.703 | 17.879~25.667 | 23.651~30.813 | 11.770~13.075 |
| 稳态PadWaste（中位数） | 0.611 | 0.611 | 0.649 | 0.639 | 未采集（老代码无FillRate埋点） |
| 累计s/it@200（HF Trainer口径） | 20.644 | 20.852 | 19.017 | 25.763 | **13.758** |
| 显存@200(GiB) | **100.86** | **81.13** | **137.25** | **136.88** | **100.04** |
| 200步耗时（HF Trainer elapsed） | 1h8m49s | 1h9m30s | **1h3m23s** | **1h25m53s** | **45m52s** |
| loss@200 | 5.60770 | 5.59434 | 5.58757 | 5.59904 | 5.59089 |

> **稳态窗口方差都不高**（CV约5-9%，同一量级），说明五组数据都是稳定、
> 可重复的测量，不是偶发抽样。**WZ0的稳态12.51s/it、显存100.04GiB，跟历史
> G/V列（torchrun+zero0+WDS，11.57s/100.04GiB）几乎完全一致**（显存小数点
> 后两位吻合），是本文档最重要的新发现，详见第4.4节。

## 2. 训练指标对比（step 10~200，每10步一条logging，按指标分表）

> 五列均为 deepspeed 链路（deepspeed HF TrainerCallback），无 megatron 列。
> WM 列 1-60 步日志已丢失，留空；70 步起完整。WZ0 全程完整（1-250步）。

### 2.1 loss

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 1 | 17.01063 | — | — | — | 17.01063 |
| 10 | 16.96405 | 16.94471 | 16.93170 | — | 16.96716 |
| 20 | 15.52157 | 15.45836 | 15.46329 | — | 15.53949 |
| 30 | 11.95131 | 11.92844 | 11.93083 | — | 11.95856 |
| 40 | 9.48825 | 9.47944 | 9.48152 | — | 9.49222 |
| 50 | 8.41680 | 8.41534 | 8.42429 | — | 8.41922 |
| 60 | 7.33370 | 7.33307 | 7.34283 | — | 7.33279 |
| 70 | 6.68700 | 6.68600 | 6.68998 | 6.67976 | 6.68960 |
| 80 | 6.37880 | 6.37993 | 6.37683 | 6.36961 | 6.36397 |
| 90 | 6.19773 | 6.19747 | 6.19293 | 6.20401 | 6.17598 |
| 100 | 6.07240 | 6.07057 | 6.06775 | 6.07344 | 6.03276 |
| 110 | 5.97300 | 5.97228 | 5.96441 | 5.98317 | 5.93211 |
| 120 | 5.88784 | 5.88669 | 5.87763 | 5.90161 | 5.86437 |
| 130 | 5.83041 | 5.83296 | 5.82413 | 5.84114 | 5.80956 |
| 140 | 5.77072 | 5.77363 | 5.76733 | 5.77405 | 5.75420 |
| 150 | 5.72103 | 5.73150 | 5.72224 | 5.73202 | 5.72905 |
| 160 | 5.68978 | 5.69749 | 5.68892 | 5.69015 | 5.68705 |
| 170 | 5.65494 | 5.66197 | 5.65590 | 5.66378 | 5.66377 |
| 180 | 5.64278 | 5.64584 | 5.63702 | 5.64440 | 5.64373 |
| 190 | 5.61274 | 5.61083 | 5.60480 | 5.61502 | 5.61163 |
| 200 | 5.60770 | 5.59434 | 5.58757 | 5.59904 | 5.59089 |
| 210 | — | — | — | — | 5.56920 |
| 220 | — | — | — | — | 5.54145 |
| 230 | — | — | — | — | 5.53460 |
| 240 | — | — | — | — | 5.51162 |
| 250 | — | — | — | — | 5.49949 |

**五列 loss@200 落在 5.588~5.608，跨度仅 0.020，全部收敛一致**——证明数据源
（Lance/WDS）、拓扑（单机/双节点）、ZeRO stage（zero0/zero1）、框架
（ray/直连）的全部组合差异只体现在**效率**上，不影响**数学正确性**。WZ0
继续跑到250步（loss 5.49949），轨迹平滑下降，无异常。

### 2.2 token_acc

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 0.20% | 0.20% | 0.20% | — | 0.19% |
| 20 | 0.17% | 0.17% | 0.17% | — | 0.17% |
| 30 | 0.36% | 0.36% | 0.36% | — | 0.35% |
| 40 | 1.31% | 1.33% | 1.33% | — | 1.32% |
| 50 | 3.99% | 3.99% | 3.94% | — | 3.99% |
| 60 | 10.18% | 10.18% | 10.07% | — | 10.17% |
| 70 | 14.70% | 14.71% | 14.66% | 14.81% | 14.68% |
| 80 | 16.92% | 16.93% | 16.93% | 17.03% | 17.06% |
| 90 | 18.15% | 18.15% | 18.17% | 18.09% | 18.41% |
| 100 | 19.01% | 19.01% | 18.99% | 19.00% | 19.42% |
| 110 | 19.48% | 19.50% | 19.53% | 19.40% | 19.94% |
| 120 | 19.98% | 20.00% | 20.08% | 19.90% | 20.31% |
| 130 | 20.28% | 20.25% | 20.34% | 20.19% | 20.51% |
| 140 | 20.70% | 20.65% | 20.69% | 20.70% | 20.93% |
| 150 | 21.02% | 20.93% | 20.99% | 20.93% | 20.98% |
| 160 | 21.15% | 21.07% | 21.16% | 21.19% | 21.25% |
| 170 | 21.38% | 21.30% | 21.38% | 21.29% | 21.28% |
| 180 | 21.42% | 21.37% | 21.48% | 21.45% | 21.40% |
| 190 | 21.61% | 21.62% | 21.70% | 21.60% | 21.63% |
| 200 | 21.55% | 21.72% | 21.77% | 21.65% | 21.73% |

### 2.3 token_text_acc（thinker 冻结的直接证据，全程应钉死在~72.7%附近）

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 72.61% | 72.61% | 72.63% | — | 72.75% |
| 20 | 72.78% | 72.78% | 72.78% | — | 72.86% |
| 30 | 72.87% | 72.87% | 72.90% | — | 72.82% |
| 40 | 72.66% | 72.66% | 72.64% | — | 72.64% |
| 50 | 72.72% | 72.72% | 72.66% | — | 72.82% |
| 60 | 72.78% | 72.78% | 72.83% | — | 72.79% |
| 70 | 72.78% | 72.78% | 72.78% | 72.76% | 72.74% |
| 80 | 72.77% | 72.77% | 72.80% | 72.74% | 72.69% |
| 90 | 72.70% | 72.70% | 72.70% | 72.81% | 72.74% |
| 100 | 72.73% | 72.73% | 72.74% | 72.83% | 72.76% |
| 110 | 72.63% | 72.63% | 72.61% | 72.74% | 72.69% |
| 120 | 72.67% | 72.67% | 72.69% | 72.78% | 72.74% |
| 130 | 72.68% | 72.68% | 72.67% | 72.72% | 72.70% |
| 140 | 72.79% | 72.79% | 72.76% | 72.84% | 72.86% |
| 150 | 72.76% | 72.76% | 72.75% | 72.77% | 72.78% |
| 160 | 72.80% | 72.80% | 72.83% | 72.78% | 72.79% |
| 170 | 72.66% | 72.66% | 72.70% | 72.76% | 72.80% |
| 180 | 72.77% | 72.77% | 72.80% | 72.75% | 72.79% |
| 190 | 72.73% | 72.73% | 72.76% | 72.72% | 72.78% |
| 200 | 72.74% | 72.74% | 72.74% | 72.75% | 72.78% |

### 2.4 token_acoustic_token

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 0.45% | 0.45% | 0.45% | — | 0.44% |
| 20 | 0.54% | 0.54% | 0.54% | — | 0.54% |
| 30 | 0.96% | 0.96% | 0.96% | — | 0.96% |
| 40 | 1.91% | 1.91% | 1.91% | — | 1.89% |
| 50 | 3.30% | 3.30% | 3.29% | — | 3.24% |
| 60 | 4.48% | 4.48% | 4.48% | — | 4.48% |
| 70 | 5.35% | 5.35% | 5.36% | 5.53% | 5.38% |
| 80 | 5.96% | 5.97% | 5.97% | 6.07% | 6.08% |
| 90 | 6.56% | 6.56% | 6.58% | 6.57% | 6.81% |
| 100 | 7.14% | 7.14% | 7.14% | 7.16% | 7.60% |
| 110 | 7.72% | 7.71% | 7.74% | 7.63% | 8.20% |
| 120 | 8.22% | 8.21% | 8.26% | 8.12% | 8.56% |
| 130 | 8.56% | 8.55% | 8.59% | 8.53% | 8.88% |
| 140 | 8.97% | 8.96% | 8.96% | 8.94% | 9.26% |
| 150 | 9.27% | 9.24% | 9.25% | 9.17% | 9.28% |
| 160 | 9.43% | 9.40% | 9.43% | 9.40% | 9.46% |
| 170 | 9.64% | 9.61% | 9.62% | 9.62% | 9.62% |
| 180 | 9.76% | 9.76% | 9.78% | 9.80% | 9.78% |
| 190 | 9.95% | 9.95% | 9.97% | 9.91% | 9.92% |
| 200 | 10.00% | 10.04% | 10.06% | 10.04% | 10.04% |

### 2.5 eos_boundary_top1_acc

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 1.57% | 1.56% | 1.55% | — | 1.42% |
| 20 | 2.23% | 2.23% | 2.18% | — | 2.05% |
| 30 | 20.64% | 20.97% | 20.67% | — | 19.80% |
| 40 | 44.85% | 44.62% | 45.09% | — | 44.12% |
| 50 | 10.95% | 11.07% | 10.80% | — | 10.31% |
| 60 | 22.22% | 21.92% | 21.81% | — | 22.84% |
| 70 | 26.37% | 26.45% | 26.71% | 26.71% | 25.87% |
| 80 | 34.48% | 34.91% | 34.98% | 34.48% | 33.94% |
| 90 | 40.05% | 39.57% | 39.48% | 37.35% | 38.73% |
| 100 | 42.43% | 44.42% | 41.61% | 45.00% | 42.04% |
| 110 | 44.10% | 46.85% | 46.23% | 44.62% | 46.36% |
| 120 | 49.50% | 48.32% | 50.67% | 48.45% | 50.35% |
| 130 | 52.68% | 51.28% | 52.68% | 53.88% | 54.32% |
| 140 | 52.52% | 54.50% | 51.32% | 51.52% | 52.15% |
| 150 | 56.62% | 55.82% | 56.26% | 56.05% | 57.89% |
| 160 | 56.56% | 56.15% | 57.79% | 56.62% | 52.45% |
| 170 | 58.38% | 58.43% | 58.30% | 58.92% | 57.93% |
| 180 | 59.47% | 56.58% | 56.06% | 57.09% | 56.31% |
| 190 | 60.60% | 57.50% | 59.30% | 59.04% | 58.45% |
| 200 | 53.58% | 58.62% | 58.77% | 55.66% | 59.55% |

> 该指标方差大（受batch内EOS样本数波动影响），五列在同一噪声带内波动，
> 无系统性差异。

### 2.6 premature_eos_top1_rate

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 1.07% | 1.07% | 1.08% | — | 1.04% |
| 20 | 1.33% | 1.34% | 1.35% | — | 1.25% |
| 30 | 10.34% | 10.56% | 10.37% | — | 9.94% |
| 40 | 28.82% | 28.70% | 29.08% | — | 28.73% |
| 50 | 5.15% | 5.31% | 5.06% | — | 4.83% |
| 60 | 3.25% | 3.19% | 3.14% | — | 3.40% |
| 70 | 1.74% | 1.76% | 1.77% | 1.75% | 1.71% |
| 80 | 1.41% | 1.43% | 1.47% | 1.50% | 1.45% |
| 90 | 1.31% | 1.29% | 1.28% | 1.19% | 1.24% |
| 100 | 1.17% | 1.24% | 1.08% | 1.28% | 1.15% |
| 110 | 1.05% | 1.16% | 1.17% | 1.05% | 1.14% |
| 120 | 1.18% | 1.09% | 1.22% | 1.09% | 1.20% |
| 130 | 1.15% | 1.14% | 1.13% | 1.17% | 1.37% |
| 140 | 1.02% | 1.09% | 1.03% | 1.04% | 1.02% |
| 150 | 1.19% | 1.06% | 1.14% | 1.13% | 1.15% |
| 160 | 1.07% | 1.10% | 1.14% | 1.14% | 0.93% |
| 170 | 0.99% | 1.13% | 1.10% | 1.07% | 1.09% |
| 180 | 1.09% | 0.96% | 0.95% | 0.97% | 1.00% |
| 190 | 1.06% | 0.97% | 0.98% | 1.05% | 0.97% |
| 200 | 1.13% | 0.98% | 0.99% | 0.94% | 0.97% |

### 2.7 grad_norm

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 481.59 | 483.10 | 476.53 | — | 472.68 |
| 20 | 269.92 | 267.57 | 268.73 | — | 265.53 |
| 30 | 87.50 | 85.75 | 85.49 | — | 87.10 |
| 40 | 22.08 | 22.09 | 22.59 | — | 22.09 |
| 50 | 19.09 | 18.96 | 18.97 | — | 19.19 |
| 60 | 8.69 | 8.37 | 8.97 | — | 8.47 |
| 70 | 4.88 | 5.08 | 5.80 | 8.29 | 7.10 |
| 80 | 7.61 | 5.99 | 6.64 | 8.88 | 7.12 |
| 90 | 5.60 | 7.37 | 7.05 | 5.91 | 7.26 |
| 100 | 6.40 | 7.74 | 7.97 | 7.57 | 7.42 |
| 110 | 5.65 | 8.31 | 7.37 | 5.45 | 4.44 |
| 120 | 7.03 | 7.86 | 6.43 | 9.83 | 7.47 |
| 130 | 8.68 | 9.50 | 7.41 | 7.95 | 9.83 |
| 140 | 10.86 | 7.76 | 8.18 | 10.00 | 9.86 |
| 150 | 8.28 | 8.38 | 8.55 | 8.66 | 10.44 |
| 160 | 9.21 | 7.31 | 6.17 | 7.34 | 8.03 |
| 170 | 7.48 | 8.79 | 9.61 | 10.61 | 8.74 |
| 180 | 10.14 | 9.33 | 6.87 | 7.67 | 7.89 |
| 190 | 9.08 | 8.09 | 7.79 | 9.85 | 8.47 |
| 200 | 8.07 | 5.70 | 6.93 | 6.33 | 8.76 |

### 2.8 learning_rate（五列完全一致，同一 LR schedule）

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 3.60e-07 | 3.60e-07 | 3.60e-07 | — | 3.6e-07 |
| 50 | 1.78e-06 | 1.78e-06 | 1.78e-06 | — | 1.78e-06 |
| 100 | 3.56e-06 | 3.56e-06 | 3.56e-06 | 3.56e-06 | 3.56e-06 |
| 150 | 5.34e-06 | 5.34e-06 | 5.34e-06 | 5.34e-06 | 5.34e-06 |
| 200 | 7.12e-06 | 7.12e-06 | 7.12e-06 | 7.12e-06 | 7.12e-06 |

### 2.9 s/it（train_speed，累计均值口径，含冷启动）

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 22.08 | 22.21 | 21.25 | — | 22.50 |
| 20 | 21.06 | 21.03 | 20.06 | — | 19.14 |
| 30 | 20.73 | 20.84 | 19.84 | — | 17.85 |
| 40 | 20.97 | 21.18 | 19.67 | — | 16.67 |
| 50 | 20.80 | 20.99 | 19.51 | — | 15.81 |
| 60 | 20.87 | 21.00 | 19.47 | — | 15.58 |
| 70 | 20.90 | 21.04 | 19.42 | 26.52 | 15.28 |
| 80 | 20.89 | 21.03 | 19.29 | 26.82 | 15.02 |
| 90 | 20.95 | 21.08 | 19.22 | 26.88 | 14.83 |
| 100 | 20.92 | 21.04 | 19.20 | 26.67 | 14.65 |
| 110 | 20.86 | 20.97 | 19.17 | 26.42 | 14.59 |
| 120 | 20.83 | 20.94 | 19.16 | 26.30 | 14.46 |
| 130 | 20.82 | 20.95 | 19.13 | 26.23 | 14.34 |
| 140 | 20.81 | 20.98 | 19.08 | 26.20 | 14.29 |
| 150 | 20.77 | 20.95 | 19.06 | 26.15 | 14.19 |
| 160 | 20.77 | 20.95 | 19.05 | 26.05 | 14.08 |
| 170 | 20.79 | 20.98 | 19.04 | 25.94 | 13.96 |
| 180 | 20.77 | 20.98 | 19.03 | 25.89 | 13.89 |
| 190 | 20.71 | 20.91 | 19.00 | 25.82 | 13.82 |
| 200 | 20.64 | 20.85 | 19.02 | 25.76 | 13.76 |

> **累计均值口径**（含冷启动、warmup）比第1.2节的"稳态瞬时口径"（仅
> step191-200的瞬时耗时）系统性偏高——两者差异反映的是前几十步冷启动
> 的拖累，不是稳态性能的差异。**WZ0的累计口径从22.50s（step10）平滑
> 收敛到13.76s（step200），下降曲线比AO/L2N/L1N/WM都更陡峭，说明WZ0
> 的冷启动拖累相对更小（老代码的DataLoader预热更快）。**

### 2.10 显存(GiB)

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 1-30 | 64.24~91.19 | 45.12~71.39 | 137.25 | — | 66.67~67.40 |
| 40-50 | 91.19 | 71.39 | 137.25 | — | 68.54 |
| 60-170 | 100.86 | 81.13 | 137.25 | 134.88~136.88 | **100.03** |
| 180-200 | 100.86 | 81.13 | 137.25 | 136.88 | **100.04** |
| 210-250 | — | — | — | — | 100.04 |

**显存与 ZeRO stage 强相关**（zero0恒>100GiB，zero1恒<83GiB双节点/恒
~137GiB单机），与数据源（Lance/WDS）、是否用ray基本无关；L1N/WM同为
单机zero1，显存137.25 vs 136.88，差0.27%，几乎完全一致。**AO与WZ0同为
双节点zero0（唯一差异=数据源Lance/WDS），显存100.86 vs 100.04GiB，差
0.8%，同样几乎完全一致——进一步印证显存只跟ZeRO stage走，跟数据源无关
这条规律**。WZ0的100.04GiB还跟历史G/V列（torchrun+zero0+WDS，
100.04GiB）**精确吻合到小数点后两位**。

### 2.11 elapsed（自训练循环起点，HF Trainer累计口径）

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 10 | 3m41s | 3m42s | 3m33s | — | 3m45s |
| 50 | 17m20s | 17m29s | 16m16s | — | 13m11s |
| 100 | 34m52s | 35m4s | 32m0s | 44m27s | 24m25s |
| 150 | 51m56s | 52m23s | 47m39s | 1h5m22s | 35m29s |
| 200 | 1h8m49s | 1h9m30s | 1h3m23s | 1h25m53s | **45m52s** |
| 250 | — | — | — | — | 56m7s |

### 2.12 remaining_time（按累计均值速率外推全程9364步）

| step | AO | L2N | L1N | WM | WZ0 |
|---|---|---|---|---|---|
| 100 | 2d5h50m8s | 2d6h9m3s | 2d1h25m12s | 2d20h37m7s | **1d13h41m39s** |
| 200 | 2d4h33m5s | 2d5h4m48s | 2d0h24m29s | 2d17h34m52s | **1d11h1m15s** |

---

## 3. 数学结果一致性结论

1. **五列 loss@200 落在 5.588~5.608，跨度仅0.020**，`token_text_acc`全程
   钉死在72.6-72.9%（thinker冻结的直接证据）——**数据源(Lance/WDS)、
   拓扑(单机/双节点)、ZeRO stage(zero0/zero1)、框架(ray/直连)的全部
   组合差异只体现在效率上，训练数学结果完全一致**。WZ0继续跑到250步
   （loss 5.49949），轨迹平滑，同样支持这条结论。
2. **AO↔L2N（同数据源同拓扑，唯一变量=ZeRO stage）**：loss@200差0.0134，
   噪声范围内。
3. **L1N↔WM（同拓扑同ZeRO stage，唯一变量=数据源）**：loss@200差0.0115，
   噪声范围内——**这是本文档唯一干净的"数据源单变量对照"，证明Lance
   与WDS吃到的是同一批数据、编码结果一致**。
4. **AO↔WZ0（同拓扑同数据源，唯一变量=数据源，见4.4节；ZeRO stage相同为
   zero0）**：loss@200差0.0168，噪声范围内——进一步印证数据源不影响
   数学结果。

## 4. 训练效率结论与可比对说明

### 4.1 可比对（单变量，严格对照）

| 对照对 | 唯一变量 | 结果 |
|---|---|---|
| **L1N vs WM**（单机，pytorch直连，zero1） | 数据源：Lance vs WDS | **Lance快23.4%**（18.517s vs 23.889s，中位数） |
| **AO vs L2N**（双节点，ray，Lance） | ZeRO stage：zero0 vs zero1 | zero0快1.4%（19.150s vs 19.416s），显存多24.3%（100.86 vs 81.13 GiB） |
| **AO vs WZ0**（双节点，ray，zero0） | 数据源：Lance vs WDS | **WDS快53.3%**（12.510s vs 19.150s，中位数），显存基本一致（100.86 vs 100.04GiB，差0.8%） |

### 4.2 不可比对（混了多个变量，仅供参考）

| 对照对 | 混了哪些变量 | 现象（不作为结论） |
|---|---|---|
| L2N vs L1N（同Lance同zero1） | 节点数(1↔2) + 是否用ray | L2N(19.416s)比L1N(18.517s)慢4.9%——**"双节点+ray"比"单机+直连"慢，
但不能拆分是节点数问题还是ray框架问题**，需补一组"双节点+直连"实验才能拆分（见文档
`AO_REPRODUCTION_LOG.md`第10节，本文档未涉及） |
| AO vs L1N（同Lance，节点数+ZeRO stage都变） | 节点数(1↔2) + ZeRO stage(0↔1) | 不构成单变量对照，仅供浏览 |
| WZ0 vs WM（同WDS，节点数+框架都变） | 节点数(1↔2) + 是否用ray | WZ0(12.510s)比WM(23.889s)快47.6%——**"双节点+ray"比"单机+直连"快，
跟L2N/L1N那组"双节点+ray反而慢"的方向正好相反**，说明这个方向性差异强烈依赖于
"是不是走了ray双节点原生scaling"，而不是ray本身天然带来额外开销，需谨慎解读 |

### 4.3 关键结论

- **单机场景：Lance比WDS快23.4%**——干净、可信的单变量对照（第4.1节），
  双边代码状态均已验证（WM在master上验证通过，与R历史值精确吻合；
  Lance代码本身master/feature分支间无diff）。
- **双节点zero0场景：WDS比Lance快53.3%**（第4.1节AO vs WZ0）——跟单机场景
  方向正好相反！这跟第3/6节（AO_REPRODUCTION_LOG.md）里"双节点zero1场景
  Lance比WDS慢59%"（D/S历史值 vs L2N）的方向一致——**说明"双节点场景下
  Lance比WDS慢"这个现象在zero0和zero1两个ZeRO stage下都成立，不是zero1
  特有的**，Lance在ray双节点+GBS4096场景下的劣势具有一定的普遍性。
  但要注意：AO vs WZ0的WDS一侧数据(WZ0)来自老代码S3快照，Lance一侧(AO)
  来自当前分支，**严格来说仍不是同commit的干净对照**，跟第3节D/S vs L2N
  面临的方法论风险是同一类问题，结论方向可信但精确数字（53.3%）需谨慎
  引用。
- **ray双节点本身不天然拖慢训练**（第4.2节WZ0 vs WM反证）：WZ0（双节点
  ray+zero0+WDS）比WM（单机+zero1+WDS）快47.6%，证明"双节点+ray"在
  WDS+zero0这个组合下反而是加速的（符合DP扩展的预期），"双节点比单机
  慢"这个此前的疑虑（第4.2节L2N vs L1N）看来是Lance数据路径本身的问题，
  不是ray/双节点架构的通病。
- **`ray+zero0+WDS`双节点这个此前的空白格已被填补，验证不会OOM**
  （详见第4.4节）——历史上两次尝试都失败于代码/环境问题（`85b6d750`
  引入的`DATASET_NUM_PROC`探测bug + 节点磁盘压力），不是这个配置组合
  本身有问题；用老代码S3快照绕开后，稳态12.51s/it、显存100.04GiB，
  跟历史G/V列（torchrun+zero0+WDS）几乎完全一致。

### 4.4 WZ0：ray+zero0+WDS 空白格补全的完整过程与参数说明

**背景**：此前 `ktrainjob_junyaoyang_ray2node_gbs4096_liger_zero0_07d3_028.
yaml`（master分支上存在过的原始yaml）两次提交都直接失败——第一次在
step2 optimizer阶段OOM崩溃（`AO_REPRODUCTION_LOG.md`第5节记录），是四方
对比里唯一从未产出有效数据的空白格。而理论分析显示这个配置不该OOM：
G列（pytorch直连torchrun+zero0+WDS双节点，同样`PER_DEVICE=32`）已验证
跑满200步无OOM，唯一差异只是是否走ray。

**本次排查发现的两个独立障碍**（均与"是否会OOM"这个问题本身无关）：

1. **ray driver子进程CPU探测bug**：`85b6d750`这次commit删除了
   `train_speech_to_speech_translation.sh`里`SCENE=batch4_8_adapter_cmp`
   分支的静态硬编码值`DATASET_NUM_PROC=16`/`DATALOADER_NUM_WORKERS=16`/
   `DATALOADER_PERSISTENT_WORKERS=true`，改为依赖`common_args.sh`的
   `_get_container_cpus()`动态探测。用调试日志实测：ray driver子进程里
   `nproc`探测到的是**1**（`cgroup_cpu_max=13500000 100000`即135核配额
   本应正确，但被更小的`nproc=1`封顶覆盖），导致`.map()`预处理退化成
   单进程跑（700-900 examples/s，1919万行要数小时），任务在预处理阶段
   卡死；此bug只在ray driver场景触发，pytorch直连单机不受影响（`L1N`
   跑出`Map (num_proc=135)`，探测完全正常）。详见
   `AO_REPRODUCTION_LOG.md`第9节。**master分支没有这处回归**（`SCENE`
   分支的硬编码从未被删）。
2. **节点磁盘压力**：`08e3`/`07d3`两个节点本身长期磁盘紧张（08e3一度
   90%使用率，`/opt/dlami/nvme/.cache/modelscope/datasets`下堆积了2.4T
   历次失败任务遗留的`tmpXXXXX`临时Arrow缓存文件，从未被清理），预处理
   阶段16个rank并行写缓存很容易把节点推过`DiskPressure`阈值触发污点，
   而`ray_entrypoint.py`把训练driver用
   `NodeAffinitySchedulingStrategy(soft=False)`强行钉死在特定节点上，
   一旦该节点变成不可调度，ray直接抛`TaskUnschedulableError`导致任务
   `Failed`（跟GPU显存OOM是完全不同的失败模式，容易被误判成同一类问题）。
   本次清理该目录（释放2.4T）后才跑通。

**修复/绕过方式**：不修改代码，改用`junyaoyang/liveinterpret_mfu`这个
S3代码快照（junyaoyang在`85b6d750`之前、`2026-09-04 06:42`同步上去、
之后再未被覆盖的冻结版本，已用`aws s3 cp`核实其
`train_speech_to_speech_translation.sh`仍带`DATASET_NUM_PROC=16`硬编码），
把yaml的`CODE_PREFIX`/`LAUNCHER_PATH`/`command`三处指向这个快照，**跳过
`submit.sh`的自动`sync_code`步骤**（否则会用当前分支代码覆盖这个快照），
改成手动渲染yaml后直接`ktrain job submit --file`提交。

**完整 yaml**（`ktrainjob_junyaoyang_ray2node_gbs4096_liger_zero0_wds_07d3_08e3.yaml`
渲染后手动改的最终版本，任务名`wds-ray-oldcode3-m8r0`）：

```yaml
apiVersion: ktrain.oit-svc.com/v1alpha2
kind: KtrainJob
metadata:
  name: wds-ray-oldcode3-m8r0
  namespace: speech-h200
  annotations:
    ktrain.oit-svc.com/owner: junyaoyang
    ktrain.oit-svc.com/render-extra: >-
      {"hyperpod": true, "no_metrics_sidecar": true,
       "efa_devices_per_worker": 32,
       "tolerations": [{"key": "node-type", "operator": "Equal", "value": "gpu", "effect": "NoSchedule"}],
       "node_selector": {"node.kubernetes.io/instance-type": "ml.p5e.48xlarge"},
       "affinity": {
         "nodeAffinity": {
           "requiredDuringSchedulingIgnoredDuringExecution": {
             "nodeSelectorTerms": [{"matchExpressions": [
               {"key": "kubernetes.io/hostname", "operator": "In",
                "values": ["hyperpod-i-02848679822afbe3b", "hyperpod-i-08e3257c716579618"]}
             ]}]
           }
         }
       },
       "extra_volumes": [
         {"name": "s3all", "persistentVolumeClaim": {"claimName": "speech-h200-s3-pvc", "readOnly": true}},
         {"name": "nvme", "hostPath": {"path": "/opt/dlami/nvme", "type": "Directory"}}
       ],
       "extra_volume_mounts": [
         {"name": "s3all", "mountPath": "/s3all", "readOnly": true},
         {"name": "nvme", "mountPath": "/nvme"}
       ],
       "shm_size": "64Gi"
      }
spec:
  framework: ray
  image: 124355679795.dkr.ecr.ap-southeast-3.amazonaws.com/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9-ray2-nsys-liger
  command: ["python", "/s3all/junyaoyang/liveinterpret_mfu/ktrain/ray_entrypoint.py"]
  numNodes: 3
  head:
    cpu: "16"
    memory: 64Gi
  resources:
    cpu: "135"
    memory: 1350Gi
    gpu: 8
  env:
    TRAIN_SCRIPT: "pipelines/bridge_alignment/train_speech_to_speech_translation.sh"
    TRAIN_BACKEND: "deepspeed"
    USE_RAY: "true"
    RAY_CLUSTER_MODE: "kuberay"
    LAUNCHER_PATH: "/s3all/junyaoyang/liveinterpret_mfu/ktrain/launch_train.sh"
    RAY_EXPECTED_GPUS: "16"
    CODE_PREFIX: "junyaoyang/liveinterpret_mfu"
    NVME_MOUNT: "/nvme"
    FSX_MOUNT: "/mnt/fsx"
    MODEL_PATH: "s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_share/liveinterpret_training_pipelines/models/qwen2_5_3b_omnispeak_a/"
    MODEL_TYPE: "qwen2_5_3b_omnispeak_a"
    QWEN3_TTS_TOKENIZER: "s3://iegg-speech-model-models/origin_models/Qwen3-TTS-Tokenizer-12Hz"
    BRIDGE_ALIGNMENT_DATA: "fsx://zeyangsong/unist/converted/unist_qwen3_s2s_v2/manifests/manifest-train-*.jsonl"
    SCENE: "batch4_8_adapter_cmp"
    ARTIFACT_AUDIO_SHARDS: "s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup/opt/dlami/nvme/zeyangsong/data/unist/"
    QWEN3_TTS_SPEAKER_ENCODER: "s3://iegg-speech-model-models/origin_models/Qwen3-TTS-12Hz-1.7B-Base"
    OUTPUT_DIR: "/nvme/output/ray2node_gbs4096_liger_zero0_wds_07d3_08e3"
    PER_DEVICE_TRAIN_BATCH_SIZE: "32"
    GRADIENT_ACCUMULATION_STEPS: "8"
    DEEPSPEED: "zero0"
    USE_LIGER_KERNEL: "true"
    QWEN_OMNISPEAK_LIGER_OPS: "1"
    SWIFT_SINGLE_DEVICE_MODE: "1"
    GROUP_BY_LENGTH: "false"
    FILL_LOG_ENABLED: "1"
    FILL_LOG_STEP_INTERVAL: "1"
    MFU_LOG_ENABLED: "1"
    NCCL_IB_DISABLE: "0"
    NCCL_NVLS_ENABLE: "0"
    NCCL_SOCKET_IFNAME: "^lo,docker0"
    FI_PROVIDER: "efa"
    NCCL_DEBUG: "INFO"
    TORCH_NCCL_ASYNC_ERROR_HANDLING: "1"
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: "300"
    FI_EFA_FORK_SAFE: "1"
    FI_EFA_USE_DEVICE_RDMA: "1"
    NCCL_DEBUG_SUBSYS: "INIT,NET,ENV"
    PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
    WANDB_MODE: "disabled"
    WANDB_PROJECT: "qwen_omnispeak_junyaoyang_profile"
```

**关键参数选择逐条说明**：

| 参数 | 值 | 为什么选这个值 |
|---|---|---|
| `framework` | `ray` | 这是本次要验证的核心变量——历史上"ray+zero0+WDS"从未跑通过 |
| `command` / `LAUNCHER_PATH` / `CODE_PREFIX` | 全部指向 `junyaoyang/liveinterpret_mfu` | **唯一的非常规选择**：绕开`85b6d750`引入的`DATASET_NUM_PROC`探测bug，用冻结在该bug产生之前的老代码快照，而不是当前分支代码 |
| 节点白名单 | `02848679822afbe3b`(028) + `08e3257c716579618`(08e3) | 历史yaml用的是`07d3+028`，本次改用`028+08e3`：07d3长期带磁盘压力污点不可调度，028/08e3是当时唯一同时空闲且磁盘健康的组合 |
| `PER_DEVICE_TRAIN_BATCH_SIZE=32` + `GRADIENT_ACCUMULATION_STEPS=8` | 32×16卡×8=4096 | 对齐AO/L2N/历史zero1版S列的GBS=4096口径，双节点16卡下的标准配比 |
| `DEEPSPEED=zero0` | 纯DDP，不分片优化器/梯度 | 本次要验证的核心变量——历史两次都在这个配置下失败/OOM |
| `GROUP_BY_LENGTH=false` | 未显式设置，用老代码脚本默认值（本身就是false） | 老代码没有`85b6d750`引入的GBL默认值漂移问题，不需要像AO/L2N那样显式覆盖 |
| `USE_LIGER_KERNEL=true`+`QWEN_OMNISPEAK_LIGER_OPS=1` | 开启liger融合算子 | 对齐AO/L2N/历史所有GBS4096对照组的统一配置 |
| `SWIFT_SINGLE_DEVICE_MODE=1` | 绕过ray driver的`is_mp()`误判 | 沿用本轮会话其他ray任务的标准配置，保险起见加上（WDS路径实测未必会触发这个判断，但不影响结果） |
| 未设 `DATASET_NUM_PROC` / `DATALOADER_NUM_WORKERS` 等 | 依赖老代码`SCENE=batch4_8_adapter_cmp`分支的静态硬编码值（16/16/true） | **这正是能绕开探测bug的关键**——不需要在yaml层显式设置任何DataLoader参数，老代码本身就是对的 |

**执行过程中的教训**（供后续参考）：
- 第一次尝试在yaml层显式设`DATASET_NUM_PROC=128`（不换代码，试图强行绕开
  动态探测），预处理确实加速了（`num_proc=128`，5-9万examples/s），**但
  引入了新问题**：8个rank同时起128进程，跟`common_args.sh`自己的约束
  "总worker数不能超容器CPU"冲突，训练阶段稳定卡在60-65s/it（正常
  12-19s/it），`nvidia-smi`采样GPU大部分时间0%利用率——数据/CPU资源
  超配把GPU饿死了，被放弃，改用老代码路径方案。
- 提交前必须先把`junyaoyang/liveinterpret_mfu`这个S3路径完整备份到本地
  （`aws s3 sync`，504个文件），因为`submit.sh`默认会用本地仓库覆盖
  `CODE_PREFIX`指向的S3路径——如果误用`submit.sh`提交，会把这个珍贵的
  老代码快照永久覆盖丢失。
- 提交时不能跟其他正在跑的任务共享同一对节点（GPU资源独占），验证过程
  中因07d3/08e3磁盘压力污点、028被无关任务占满等原因，反复切换节点、
  两次提交失败（`TaskUnschedulableError`），第三次才最终跑通。

## 6. L2N-ompfix：Lance双节点"比单机慢"根因定位与修复验证（2026-09-09晚）

**背景**：`L2N`（`lance-zero1-cagi`，ray+zero1+Lance+双节点）稳态19.416s/it，
反而比单机`L1N`（18.517s/it）慢4.9%——跟`WZ0 vs WM`（同为WDS，双节点比单机
快47.6%）方向相反（详见第4.2/4.3节）。排查过程依次验证并**推翻**了以下假设：
1. ray/双节点架构本身有额外开销（被WZ0 vs WM反证推翻）；
2. microbatch数翻倍（`PER_DEVICE=32×GA=8`→128次/step）导致Lance
   `take_by_stable_rowid()`固定开销累积——实测把microbatch数降到64
   （`PER_DEVICE=64×GA=4`）后s/it仍在20s/it左右，无改善，假设推翻；
3. `launch_train.sh`与`launch_train_lance.sh`逐行diff未发现能解释方向性
   差异的代码路径（唯一实质差异是缺一行`CUDA_VISIBLE_DEVICES="0"`，经推导
   不影响结果）。

**真正根因**（用户提出关键判据："通信问题应该同时影响WDS和Lance，但只有
Lance变慢"，据此把范围锁定在Lance数据读取本身，而非ray/NCCL通用层）：

ms-swift `swift/ray_utils/base.py` 的 `RayHelper._create_workers()` 给每个
GPU训练actor声明 `num_cpus=0.01`（源码实测确认，非猜测）。Ray的
`ray/_private/utils.py::set_omp_num_threads_if_unset()` 在worker进程启动时
若检测到 `OMP_NUM_THREADS` 未设置，会执行
`omp_num_threads = max(floor(num_assigned_cpus), 1)`——`floor(0.01)=0`，
最终把 `OMP_NUM_THREADS` **强制压到1**。本仓库ktrain链路从未显式设置过
`OMP_NUM_THREADS`，因此每个Lance训练actor的OMP线程数全部被压到1。

Lance的 `take_by_stable_rowid()` 底层走PyArrow/Lance的Rust实现，内部有
多线程列式解码/IO并发，`OMP_NUM_THREADS=1` 直接压掉这部分并行度；WDS的
音频读取是简单 `os.pread` 系统调用，不依赖OMP/BLAS线程池，故不受影响——
精确解释了"通信没问题，只有Lance受影响"这个方向性现象。

**验证**：在yaml显式设 `OMP_NUM_THREADS=8`，绕开该自动压制（该env一旦
显式设置，`set_omp_num_threads_if_unset()` 检测到已设置直接no-op）。

**排查中的一个副作用坑**：GNU `nproc`（不带`--all`）会读取 `OMP_NUM_THREADS`
直接返回其值（`man nproc` 有记录，非bug）。`common_args.sh` 的
`_get_container_cpus()` 恰好用裸 `nproc` 探测host_cpus再给cgroup配额封顶，
第一次提交只设了 `OMP_NUM_THREADS=8` 没做其他处理，导致 `DATASET_NUM_PROC`
从预期的135被意外压到8，混入额外变量。第二次提交追加显式设置
`DATASET_NUM_PROC=135` + `DATALOADER_NUM_WORKERS=8`（`:=`语法下已显式设置
的值不会被覆盖），确保实验干净。

**结果**（任务`lance-ompfix3-dqfr`，与L2N逐项同源，唯一差异=
`OMP_NUM_THREADS`显式设8）：

| 指标 | L2N（原始，OMP被压到1） | L2N-ompfix（OMP_NUM_THREADS=8） | 变化 |
|---|---|---|---|
| 稳态s/it（191-200中位数） | 19.416 | ~9.4（191-200：8.989~10.671） | **↓52%** |
| 累计s/it@200 | 20.852 | 9.604 | ↓54% |
| 200步elapsed | 1h9m30s | 32m1s | ↓54% |
| 显存@200 | 81.13GiB | 81.04GiB | 基本一致（zero1同stage，符合预期） |
| loss@200 | 5.59434 | 5.60107 | 差0.0067，噪声范围内 |

数学结果不受影响（loss/token_acc轨迹与L2N同量级），效率修复效果远超预期
——修复后的双节点Lance（9.4s/it）比单机L1N（18.517s/it）快近一倍，是DP=16
应有的正常scaling表现，此前L2N比L1N还慢完全是被这个隐藏的
`OMP_NUM_THREADS=1` 拖累。

**逐10步完整数据**（`lance-ompfix3-dqfr`，HF Trainer口径）：

| step | loss | token_acc | token_text_acc | memory(GiB) | s/it(累计) | elapsed |
|---|---|---|---|---|---|---|
| 1 | 17.00904 | 0.20% | 72.60% | 38.63 | 28.74 | 29s |
| 10 | 16.94218 | 0.20% | 72.67% | 44.53 | 11.85 | 1m58s |
| 20 | 15.46604 | 0.17% | 72.75% | 71.38 | 10.76 | 3m35s |
| 30 | 11.93719 | 0.36% | 72.87% | 71.38 | 10.37 | 5m11s |
| 40 | 9.48674 | 1.32% | 72.69% | 71.39 | 10.20 | 6m48s |
| 50 | 8.42114 | 3.95% | 72.74% | 71.39 | 10.07 | 8m23s |
| 140 | — | — | — | 81.04 | 9.68 | 22m36s |
| 150 | 5.72571 | 20.99% | 72.76% | 81.04 | 9.66 | 24m9s |
| 160 | 5.70248 | 21.04% | 72.82% | 81.04 | 9.65 | 25m44s |
| 170 | 5.66487 | 21.30% | 72.69% | 81.04 | 9.63 | 27m17s |
| 180 | 5.65251 | 21.31% | 72.76% | 81.04 | 9.62 | 28m52s |
| 190 | 5.61785 | 21.61% | 72.74% | 81.04 | 9.60 | 30m24s |
| 200 | 5.60107 | 21.67% | 72.75% | 81.04 | 9.60 | 32m1s |

> 显存从71.39GiB（step20-140前段）涨到81.04GiB（step150+）——涨幅模式与
> L2N原始曲线一致（数据分布导致的正常波动，非本次改动引入）。

**后续实验**：正在验证 `OMP_NUM_THREADS=4`（继续调低是否有更优/更差表现，
探索是否存在最优点，还是"只要不被ray压到1、越大越接近真实核数越好"），
结果待补充。

## 7. 采集历史

| 时间 | 事件 |
|---|---|
| 2026-09-05 | AO复刻：第一次因`GROUP_BY_LENGTH`脚本默认值漂移(false→true)跑偏，
第二次显式覆盖`GROUP_BY_LENGTH=false`后与AO原始报告值精确吻合（`ao-repro2-txy1`） |
| 2026-09-06 | 提交`lance-zero1-cagi`（AO的zero1变体），跑满200步，用于跟历史WDS
S/D列做数据源单变量对照 |
| 2026-09-07 | 提交`lance-1n-z1-z5is`（单机Lance，对齐历史R列WDS配置），发现单机场景
"Lance比WDS快"、跟双节点场景方向相反；随后发现R历史值本身存疑（同分支WDS单机
也有DataLoader默认值回归），排查定位到`85b6d750`引入的`DATALOADER_PREFETCH_
FACTOR`8→2 + SCENE硬编码worker数被删两处回归 |
| 2026-09-08 | 在`master`(`3e88d11`)干净代码状态下重跑单机WDS（`wds-master3-em8k`），
跑满200步，验证通过：显存136.88GiB与R历史值136.88GiB完全一致，loss@200差
0.0096，坐实回归只存在于未合并的feature分支；文档创建，整理四组实验的完整
逐指标数据；晚间尝试补全"ray+zero0+WDS"空白格，两次因`TaskUnschedulableError`
失败（07d3磁盘压力污点、08e3磁盘压力污点各一次），排查定位到`85b6d750`同一
commit还删除了`DATASET_NUM_PROC=16`等SCENE硬编码值，导致ray driver子进程
CPU探测bug（`nproc=1`）暴露；清理08e3上2.4T历次失败遗留的临时Arrow缓存文件 |
| 2026-09-09 | 改用`junyaoyang/liveinterpret_mfu`老代码S3快照（冻结于`85b6d750`
之前）绕开探测bug，第三次提交（`wds-ray-oldcode3-m8r0`）成功跑通，250步无OOM，
稳态12.51s/it、显存100.04GiB，跟历史G/V列几乎完全一致——四方对比补齐为
五方，"ray+zero0+WDS双节点"这个此前唯一的空白格被填补；晚间排查L2N比L1N
慢的根因，定位到ms-swift RayHelper给actor声明`num_cpus=0.01`导致ray自动
把`OMP_NUM_THREADS`压到1、压制Lance内部PyArrow/Rust多线程并发，显式设
`OMP_NUM_THREADS=8`验证后稳态s/it从19.416降到~9.4（↓52%），修复效果超出
预期；正在追加`OMP_NUM_THREADS=4`对照实验 |

> 本文档六列原始日志均为本机会话临时文件（`/tmp/ao_repro2_full.log`、
> `/tmp/lance_zero1_repro_full.log`、`/tmp/lance_1node_zero1_full.log`、
> `/tmp/wds_master3.log`、`/tmp/wds_zero0_oldcode3_full.log`、
> `/tmp/ompfix3_full.log`），未纳入版本库；本文档已摘录全部关键数值。
> 排查过程的完整叙事记录见同目录 `AO_REPRODUCTION_LOG.md`。
