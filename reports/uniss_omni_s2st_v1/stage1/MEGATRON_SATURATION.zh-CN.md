# Megatron 版 Stage 1:批量与显存占用实测

**日期** 2026-10-09 · 8×H200(单卡 139.8 GiB 可用)

## 为什么换到 Megatron

训练栈要与项目其余部分一致。原先那轮用的是 torch DDP + HuggingFace,
Megatron 带来的是项目既有的分布式优化器、梯度累积、`torch_dist` 检查点与日志。

实测也更快:同样 GBS=128,DDP 是 **397 样本/秒**,Megatron **601 样本/秒**(+51%)。

## 移植要点

Megatron 自带 HF 桥接层,但 `QwenHuggingFaceModel` **硬编码为 `Qwen2ForCausalLM`**
(纯 decoder-only),装不下 Omni 的"音频编码器 + 36 层 Thinker + 24 层 Talker"。
它的基类 `HuggingFaceModule(MegatronModule)` 是通用的,于是派生自基类。

Megatron 的 DDP 随后正确接管了 **384,604,928** 个可训练 Talker 参数(单个 bucket),
Thinker 的 4,703.5M 保持冻结。

三个集成点,每个都沿用仓库里已有的做法:

| 问题 | 做法 |
|---|---|
| Megatron 的 loader 用 `default_collate`,不会 padding | patch `build_pretraining_data_loader` 认 `dataset.collate_fn`(本地副本,不 import 已完成实验的那份) |
| `initialize_megatron` 要现场编译 `helpers.cpp`,缺 pybind11 头文件 | 将 `compile_helpers` 置为 no-op;该扩展只服务 Megatron 自己的索引数据集 |
| `full_config.model = None` 时不传 config 给 `model_provider` | provider 自己用 `gpt_config_from_args` 建一个 |

## 批量扫描(无干扰,60 步/档)

| MBS | GBS | ms/iter | 样本/秒 | 峰值显存 | 占可用显存 |
|---|---|---|---|---|---|
| 16 | 128 | 213.1 | 601 | 41.7 GB | 30% |
| 32 | 256 | 351.1 | 729 | 69.6 GB | 50% |
| **48** | **384** | **502.0** | **765** | **93.5 GB** | **67%** |
| 64 | 512 | 716.1 | 715 | 121.6 GB | 87% |

**取 MBS=48。** 它同时是吞吐最优点;MBS=64 显存占用更高但吞吐反降 6.5%,
且 121.6 GB 是 60 步采样下的峰值 —— 分桶之后最长的那批(600 个码)未必被采到,
用几小时跑到一半 OOM 的风险去换那点占用不划算。46 GB 的余量正是留给它们的。

### 第一次扫描的数字作废

第一轮扫描全程与 `run_embed_scale_probe.sh` 抢卡(每卡被占 31 GB),
MBS=48 测出 1277.6 ms/iter、MBS=64 直接 OOM,两个都是干扰造成的,不是真实上限。
上表是 GPU 空闲后重测的。

## 正式配置

```
MBS 48   GBS 384   TRAIN_ITERS 5000
LR 3.5e-4   warmup 200   cosine → 2e-6
embed_scale 8.0   head_scale 0.1
```

5000 步 × 384 = **1.92M 样本**,与 DDP 基线(GBS 128 × 15000)**等数据量**,
两条曲线因此可在相同数据量上对比。学习率按批量放大 3 倍取 √3 倍。

预计 5000 × 0.502 s ≈ **42 分钟**。

## 未解决:embed_scale 的对照被 LR 调度混淆

`embed_scale=8` 的那次对照用的是 `--steps 2500`,cosine 在 2500 步衰减到底;
基线是 `--steps 15000`,同一步数上 LR 还在高位。所以:

| step | scale=1 | scale=8 | 差值 | LR 比值 |
|---|---|---|---|---|
| 250 | 8.9420 | 8.8956 | −0.046 | 0.97 |
| 750 | 8.8630 | 8.6996 | −0.163 | 0.89 |
| 1250 | 8.7292 | 8.4401 | **−0.289** | 0.74 |
| 2500 | 8.3236 | 8.2598 | −0.064 | ~0 |

早期(LR 比值 0.97–0.89)那段的优势可归因于尺度,后面分不开 ——
**这个实验证明不了 −0.289**。正式训练取 scale=8 的依据是实测的 36 倍失衡
(见 `CONDITIONING_SCALE.zh-CN.md`)加上早期这一小段,不是那张表的最大值。
干净的对照需要两臂用**同一条 LR 曲线**,后续补。
