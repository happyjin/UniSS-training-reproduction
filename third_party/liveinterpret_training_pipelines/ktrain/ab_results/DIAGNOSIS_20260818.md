# AB 对照（thinker-only, 4× g6e）结论：**本次测量无效，不能作为数据格式的速度结论**

日期：2026-08-18　集群：speech-model-data-eks (HyperPod)　namespace：curion

## 1. 实测数字（口径已修正）

| 臂 | 数据格式 | steady 单步 | steady samples/s (eff_bs=4) | 步数 | 显存 |
|---|---|---|---|---|---|
| A | 生产 manifest(parquet 派生) + 原始 wav + 在线 Whisper | **20.0 s/it** | 0.200 | 43 | 35.54 GiB |
| B | Lance 表 + 离线 mel + 离线 speaker | **68.5 s/it** | 0.058 | 16 | 35.38 GiB |

> 口径修正：swift 打印的 `train_speed(s/it)` 是**累计均值**，不是单步耗时。B 的累计均值
> 一路从 20.6 爬到 56.8（仍在上升），而按 `elapsed_time` 差分算的真实单步稳定在 68–69s。
> `compare_ab_throughput.py` 已改为以 `elapsed_time`/`global_step` 差分为准。

## 2. 为什么这个数字不能用

单步 20s / 68.5s 对「3B 模型、bs=1、序列几百 token」来说都**高出应有量级 1~2 个数量级**。
排查结论：两臂的单步耗时几乎全部是 **ZeRO-2 跨节点梯度同步**，数据格式的差异被彻底淹没。

证据链：

1. **数据侧不是瓶颈**。在 B 的 pod 内实测 Lance 点查延迟（中位）：
   mel 56ms、speaker 27ms、codec codes 24ms —— 单样本合计约 **0.1s**，
   而两臂单步差 48s，相差约 500 倍。
2. **4 个 rank 全部卡在计算/通信，没有一次落在数据加载**。py-spy 对 master + 3 worker
   共采样 17 次，栈全部落在
   `backward` / `deepspeed.runtime.zero.stage_1_and_2.get_grad_norm_direct` /
   `cuda.synchronize`。也**不存在单节点 straggler**（4 个 rank 表现一致）。
3. **GPU 100% util 但功耗仅 122W / 350W**。这是 NCCL kernel 在 GPU 上自旋等网络的典型
   特征，不是算力吃满。
4. **网络就是天花板**：节点 `ml.g6e.xlarge`，**单 GPU / 4 vCPU / 单张 10 Gbit/s ENI**，
   无 EFA（容器内无 `/dev/infiniband`），且 `NCCL_IB_DISABLE=1` + `FI_PROVIDER=tcp`
   → NCCL 全程走 TCP socket。
   可训练 4.72B × bf16 = **9.44 GB 梯度**；ZeRO-2 reduce-scatter 每 rank 出向
   (3/4)×9.44 = **7.08 GB**，10Gbit/s 理论线速下**下限约 5.7s**，叠加 step 后
   参数 all-gather 与 TCP 实际效率（常只有线速 40~70%），**每步 12~25s 纯通信**
   —— 正好吃掉 A 的 20s。
5. **B 自己的历史记录反证「Lance 慢」**：2026-08-12 的 `lance-full-mel` 同样是
   Lance + 离线 mel，可训练参数**更多**（adapter+talker+thinker，显存 43.69 GiB），
   单步只有 **28.3 s/it**。同一条数据路径在不同批次波动 28→68.5s，说明主导因素是
   网络/节点条件的run-to-run 方差，不是数据格式。

## 3. 结论

- **不要**把 A=20s / B=68.5s 报给 mentor 当作「parquet+wav 比 Lance+mel 快 3.4 倍」。
  这个差值是 10GbE 上的梯度同步方差，与数据格式无因果关系。
- 在**每节点单卡 + 10GbE 无 EFA**的 g6e 上，只要解冻整颗 thinker（4.72B 可训练），
  跨节点通信必然主导单步耗时，**该拓扑下无法测出数据管线差异**。

## 4. 有效的替代测法（任选）

1. **只测数据管线**（最干净、最省卡、完全不受网络影响）：不训练，纯迭代 dataset/dataloader
   N 个样本，比两种格式的 samples/s。这正好对应要证明的命题「离线 mel 喂数更快」。
2. **单卡 + 小可训练集**：把可训练参数收到 `model.thinker_adapter`（够小，单卡 44GB 放得下，
   也不再 OOM），无跨节点通信，单步耗时由「数据 + 前反向」主导，格式差异可见。
3. **保持 4 卡 thinker-only，但把 `GRADIENT_ACCUMULATION_STEPS` 提到 16~32**：
   一次通信摊到 16~32 个样本的数据处理上，per-sample 吞吐里数据差异才显现，
   且更接近真实生产配置。

## 5. 附：公平性已核对通过的项

两臂样本分布确认对齐（所以上面的问题**不是**样本长度不公平）：

| 指标 | A 臂 (4096 行) | B 臂 (3149 行) |
|---|---|---|
| source_duration_ms 均值 / 中位 | 4735 / 4515 | 4682 / 4363 |
| source_duration_ms p10 / p90 | 2167 / 7603 | 2095 / 7747 |
| target codec 帧数 均值 / 中位 / p90 | 66 / 63 / 99 | 73 / 67 / 105 |
| codec codebooks/frame | 16 | 16 |
| trainable_params | `model.thinker_adapter model.thinker` | 同 |

其他一致项：同集群、4× g6e、ZeRO-2、同 model/tokenizer、max_length=2048、bs=1、
grad_accum=1、dataloader_num_workers=0、方向 zh→en。

## 6. 数据说明

- `ab_mel.4gpu.log`：B 臂 master 完整日志（16 步）。
- `ab_raw.steady.txt`：A 臂稳态汇总（原始日志被残留的 `kubectl logs -f | tee`
  在 pod 删除后覆盖成一行报错，仅存汇总）。
- `ab_mel.steady.txt`：B 臂稳态汇总，注意其中 `steady_median_s_per_it=57.6`
  是旧的累计均值口径，**正确值为 68.5**。
