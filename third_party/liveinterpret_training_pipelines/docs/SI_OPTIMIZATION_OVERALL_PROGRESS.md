# streaming_si 训练性能优化 · 总进展（S1-S4）

> 更新：2026-10-20
> 逐阶段详细数据见：
> * `SI_S1_GBS4096_BASE_LIGER_ZERO1_200STEP_TRAINING_METRICS.md`（A-E 五列）
> * `SI_S2_GBS4096_BASE_VS_FUSEDCE_200STEP_TRAINING_METRICS.md`（A/B 两列）
> * `SI_S3_GBS4096_BASE_VS_LIGER_TRAINING_METRICS.md`（A/B 两列）
> * S4：**尚无结果**（见第 4 节）
>
> 所有实验口径：GBS=4096、2 节点 × 8 × H200(141GiB)、torchrun + DeepSpeed、
> 不设 `MAX_STEPS`（避免 cosine schedule 被压缩）、`logging_steps=10`、
> DeepSpeed `wall_clock_breakdown` + `flops_profiler@step20`。

---

## 1. 各阶段最优配置与收益（当前结论）

| 阶段 | 最优配置 | 稳态 s/it | 吞吐(样本/s) | 显存(GiB) | 全程 | 相对 baseline |
|---|---|---|---|---|---|---|
| **S1** | zero1 + liger + **SKIP_THINKER** | **4.70** | 871 | 40.60 | 24.0h / 18608步 | **↓46.0%** |
| **S2** | **baseline**（所有候选优化均无效或负收益） | 6.10 | 671 | 66.4 | 7h58m / 4697步 | — |
| **S3** | **liger** | **3.709** | **1104** | 77.41 | 9h35m / 9304步 | **↓12.5%** |
| **S4** | ⏳ 未知 | — | — | — | — | — |

> ⚠️ **跨阶段只能比量级、不能比优劣**：四个阶段的可训参数、loss 口径、数据集、
> `max_length`、micro-batch 拆分全都不同，仅 GBS=4096 相同。

## 2. 优化手段 × 阶段 适用矩阵

| 优化 | S1 | S2 | S3 | S4（预判） |
|---|---|---|---|---|
| **SKIP_THINKER** | ✅ **↓33.8%** | ❌ thinker 是训练对象 | ❌ `cond_mode=full` 需要 hidden | ❌ 同 S3，且 thinker 也是训练对象 |
| **liger ops** | ✅ 有效（含在三变量里） | ❌ 无可挂架构 | ✅ **↓12.5%**（纯单变量） | 🔄 **待测**（覆盖面比 S3 小） |
| **zero1** | ✅ 有效 | ❌ 通信仅 1.5% | ❌ 通信 3.6% 且增显存 | ❌ joint 全参，增显存风险最大 |
| **fused text CE** | ❌ `text_w=0` | ❌ **实测慢 13.8%** | ❌ `text_w=0` | ⚠️ 技术可用，但按 S2 教训不优先 |
| 大 micro-batch | 已用 256 | 候选（32→128） | 候选（256→384） | 候选 |
| 音频塔优化 | — | 候选（50.1% MACs） | ❌ 无音频输入 | 候选 |
| thinker hidden 缓存 | — | — | 候选（66.3% MACs） | ❌ thinker 在训练，hidden 每步都变 |

## 3. 三条可复用的经验（按收益排序）

1. **先找"能完全不算的计算"** —— S1 的 `SKIP_THINKER` 是唯一的数量级收益（↓33.8%），
   因为它消除的是**真正的无用功**：`cond_mode='text_only'` 时 thinker 输出被
   `torch.zeros_like` 丢弃，且 thinker 冻结无反向。已用严格单测证明**逐位等价**。
2. **再找"算力利用率"** —— S3 的 `per_device=256`（S2 的 8 倍）让
   `fwd+bwd FLOPS` 从 29.94 → 172.31 TFLOPS（**5.8 倍**），几乎免费。
   这是"显存换算力"的正确姿势。
3. **然后才是"换更省的算子"** —— liger 融合带宽型算子（RMSNorm/RoPE/SwiGLU），
   在 S3 稳定拿到 ↓12.5% 且数值无损。
4. **警惕"拆分型省显存"** —— FLCE 在 S2 省了 42.28 GiB 却**慢 13.8%**。

### 3.1 核心判据：看它是"减少工作量"还是"重排工作量"

| | **S3 liger** ✅ | S2 FLCE ❌ |
|---|---|---|
| 显存 | 省 9.82 GiB | 省 42.28 GiB（多 4.3 倍） |
| 速度 | **↓12.5%（变快）** | **+13.8%（变慢）** |
| 机理 | **融合**多算子 → kernel 数与中间张量读写双减 | **拆分**大 GEMM 成 chunk → 省显存但增 launch 开销 |
| 省显存的身份 | **副产品** | **主效果** |

> **在 compute-bound 的训练里，"省显存是副产品"的优化通常赢，"省显存是主效果"的通常输。**
> **"省显存"本身不是目标** —— 必须能换成更大 micro-batch，或作为融合的副产品出现。

### 3.2 口径陷阱（S3 踩到的，后续必须沿用）

`GROUP_BY_LENGTH=true` 时分段 s/it 呈 **50 步周期锯齿**（= swift `LengthGroupedSampler`
的 mega-batch 跨度），S3 实测峰谷比 **7.2 倍**（11.50 ↔ 1.60）。

* 取单个 10 步区间可以让同一次运行"看起来快 3 倍或慢 2 倍"
* HF 日志的累计 `train_speed(s/it)` 被首步（S3 为 152s）长期摊薄，200 步时高估 **66%**
* **唯一可信口径：≥400 步的长窗口均值**（S3 用 200→880）

### 3.3 资源抢占的两个坑（2026-10-17 实战）

1. **查空闲必须同时看 cpu/memory/gpu/efa 四项**。torchrun worker 要整机
   `cpu 135核 / memory 1350Gi / gpu 8 / efa 32`。曾出现 **GPU 全空(0/8) 却调度失败**
   —— 因为别人投了只吃 CPU 不要 GPU 的任务占掉 74 核。
2. **K8s 调度器失败后进入指数 backoff，资源后来够了也不回来重试**。
   worker Pending 7 分钟、最后一条 Event 却是 7 分钟前 ⇒
   **Pending 超 2 分钟且无新 Event 就果断 `kubectl delete pod` 触发重建**。
3. 多节点 torchrun 是全有或全无，资源不足时 master 会等 **901s** 后
   `DistStoreError` 失败。正确做法：**先提交下一列排队、再停当前占用方**，实现无缝接管。

---

## 4. ⏳ S4 现状：尚无结果

**2026-10-17 曾提交 `si-s4-b4k-gs8f`**（03e7+08e3 双机），经历：
* 停 5 个 xiaoxing 任务 + 删 1 个空占 7 卡的 `si-eval-5g`（`sleep infinity`、14.5h 零使用）才凑出两台整机
* worker 两次 Pending（CPU 不足 + 调度器 backoff），删 pod 重建后成功
* master/worker 均 Running、rendezvous 握手成功、阶段开关验证正确：
  `scope=joint text_w=2.5 speech_w=1 cond_mode=full bs=32x16gpu ga=8 liger=0`
* **之后任务消失，未留下任何指标**（`.codebuddy/si_metrics/` 无 `S4_*`，ktrain 列表已查不到）

**当时停在 1928 万行数据预处理阶段（`DATASET_NUM_PROC=12`），尚未出首批指标。**

### 4.1 S4 的阶段特征（与前三阶段的差异）

| | S1 | S2 | S3 | **S4** |
|---|---|---|---|---|
| `scope` | `talker` | `thinker_audio` | `talker` | **`joint`** |
| 可训参数 | 467M | 4.21B+ | 467M | **≈4.7B（thinker+talker+audio_proj，最多）** |
| loss | `speech` only | `text` only | `speech` only | **`text`(w=2.5) + `speech`(w=1) 双 loss** |
| `cond_mode` | `text_only` | `full` | `full` | `full` |
| 音频输入 | 无 | 有 | 无 | **有（P2 data + wav）** |

### 4.2 已备好、等机器即可跑

```
ktrain/ktrainjob_streaming_si_s4_gbs4096_base.yaml    # A 列基线
ktrain/ktrainjob_streaming_si_s4_gbs4096_liger.yaml   # B 列，唯一变量 STREAMING_SI_LIGER_OPS=1
```

两列均为 32×GA8（GBS4096，与 S2 同拆分）、`zero2_prof.json`、不设 `MAX_STEPS`。

**S4 的 liger 预期收益小于 S3 的 12.5%**：S4 的 thinker（4.21B，`qwen3_next`）是训练对象
且有完整反向，而 liger 0.5.10 无 `apply_liger_kernel_to_qwen3_next` ⇒ 只能融合
talker（467M）那一小部分，相对占比比 S3 低。本列就是要测出这个数。

### 4.3 重跑 S4 的注意事项

1. **先查 P2 wav 数据缓存**：S4 需同步 `SI_P2_DATA_DIRS` + `SI_P2_WAV_DIRS`（各 3 个目录）。
   03e7+08e3 上已有 S2 时期的缓存（10-17 实测 3 个同步任务很快完成）；换节点可能很慢。
2. **显存风险**：S4 是三阶段里可训参数最多的（≈4.7B，优化器状态最大）。
   stevehguo 原版每卡 16 时峰值 39.4 GiB；现在每卡 32，预估 90-110 GiB。
   **若 OOM 降为 `16×GA16`**（GBS 仍 4096）。
3. **`GROUP_BY_LENGTH=true`** ⇒ 速度口径必须用长窗口（见 3.2）。
4. xiaoxing 是**活跃投递的任务流**，抢资源前建议先与对方同步。

---

## 5. 代码改动清单（本轮优化相关）

### 5.1 新增

| 文件 | 内容 |
|---|---|
| `custom/streaming_si/integration/liger_ops.py` | 绕过 swift 的 `model_type` 查表直接 patch（`streaming_si` 必然 miss 且**静默空转**）。talker(`qwen3`) 必需、thinker(`qwen3_next`) 可选 |
| `custom/streaming_si/tests/test_skip_thinker.py` | 11 passed，含 `torch.equal` **逐位相等**断言 + `full` 模式回归保护 |
| `custom/streaming_si/tests/test_fused_text_ce.py` | 6 passed / 1 skipped，loss rtol 1e-4、`lm_head` 梯度 rtol 1e-3 |
| `ktrain/deepspeed/{zero1,zero2}_prof.json` | = swift 预设 + `wall_clock_breakdown` + `flops_profiler@step20`。**保留的 yaml 都引用它** |

### 5.2 修改

| 文件 | 内容 |
|---|---|
| `model/model_si.py` | `can_skip_thinker()`（7 个前置条件）、`zero_thinker_out()`、`use_fused_text_ce()`、`_liger_flce()`；`SI.forward` 三分支 |
| `model/base_model.py` | config 新增 `skip_thinker_when_text_only`、`fused_text_ce`，**均默认 `False`（零回归）** |
| `model/modeling_streaming_si.py` | `compute_si_losses` 加 fused 分支（该分支**不产出 `text_acc`**） |
| `integration/register_streaming_si.py` | `from_pretrained` 前挂 liger；`STAGE_ENV` 加两个开关 |
| `pipelines/streaming_si/ktrain_entry.sh` | liger 生效断言（只断言必需目标）；`SI_SKIP_THINKER` 的 `SI_STAGE` 守卫 |
| `pipelines/streaming_si/common_args.sh` | 日志行加 `ds=` / `liger=` 标记 |

### 5.3 本次清理（2026-10-20）

**删除 7 个实验中间产物 yaml**：S1 的 base / liger_z1 / skipthinker / 1n_skipthinker、
S2 的 opt(FLCE 负收益)、S3 的 base、以及 `s1_ray.yaml` + `preflight_streaming_si.py`。

**保留各阶段最优 + S4 两列**：
```
S1  ktrainjob_streaming_si_s1_gbs4096_2n_allopt_prof.yaml   liger+SKIP+zero1
S2  ktrainjob_streaming_si_s2_gbs4096_base.yaml             baseline
S3  ktrainjob_streaming_si_s3_gbs4096_liger.yaml            liger
S4  ktrainjob_streaming_si_s4_gbs4096_base.yaml             基线（待跑）
S4  ktrainjob_streaming_si_s4_gbs4096_liger.yaml            liger（待跑）
```

**已还原**（按要求回到仓库原样）：`ktrain/image/Dockerfile.streaming-si`、
`ktrain/image/build_image_streaming_si.sh`、`ktrain/launch_train.sh`。

> ⚠️ 还原后 Dockerfile 不再有 `REQUIRE_EXTRAS` 构建期校验。
> **构建 liger 镜像时需人工确认底座选对**（S1 曾因底座缺 liger 而废掉一个排队 21h 的任务）。
> 当前 ECR 两个镜像均在：
> `…-v9-streaming-si-v1`（基线）、`…-v9-ray2-nsys-liger-streaming-si-v1`（含 liger）。

---

## 6. 待办

- [ ] **S4 两列**（yaml 已备好，缺机器）
- [ ] S3 C 列：`per_device 256→384`（liger 后显存降到 77.41，**余量 63.6 GiB**）。
      3.1 已证明 S3 对 micro-batch 极敏感，可能比 liger 收益更大
- [ ] S2 的 42 GiB 兑现：FLCE + `per_device 32→128`（**唯一**能让 FLCE 变正收益的路径）
- [ ] S2 音频塔优化：Voxtral 吃 50.1% MACs，可离线特征缓存 / flash-attn / 部分层冻结
- [ ] S3 thinker hidden 离线缓存：66.3% MACs 在冻结 thinker 的重复前向上，
      参数冻结+无 dropout ⇒ 同样本每 epoch hidden 完全相同
- [ ] S5（`stream_lora`）尚未做对照
