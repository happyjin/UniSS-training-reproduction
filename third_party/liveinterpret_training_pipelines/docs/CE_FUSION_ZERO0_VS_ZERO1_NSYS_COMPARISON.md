# CrossEntropy 融合 + ZeRO0/ZeRO1 对照实验记录（2026-09-14~15）

> 记录日期：2026-09-15。本文档记录一次完整的"nsys 诊断 → 代码改动 → 正确性/
> 性能验证"闭环：从单 step nsys 报告发现 kernel-launch bound → 排查出
> CrossEntropy 从未接入 liger 融合 → 实现并验证该融合 → 顺带做了 ZeRO0 vs
> ZeRO1 的对照。**核心结论：两项改动都验证通过（正确、不倒退），但都不是
> 能解决 kernel-launch bound 瓶颈的关键杠杆**，收益微小或为零。

---

## 0. 背景：起点诊断

对双节点 ray + deepspeed zero1 + Lance + GBS4096 任务在 step 30 抓的单 step
nsys 报告（`lance_ray2node_step30_worker_1532.nsys-rep`）做官方 5 视图分析，
结论：

- 单 step 91,623 个 GPU kernel，平均单 kernel 仅 83us
- GPU idle 占 16.7%（1.53s / 9.16s wall），其中 **52.5% 是 10-100us 的碎片
  小 gap**（26,330 次），NCCL 通讯全程仅占 1.3%（<1.5ms 都不到）
- 诊断为 **kernel-launch bound**：CPU 发指令跟不上，不是数据/通信/显存瓶颈

排查代码发现：`QWEN_OMNISPEAK_LIGER_OPS=1` 只融合了 RMSNorm/SwiGLU/RoPE，
liger 集成文档写着应覆盖 CrossEntropy，但 codec/text loss 实际全程走原生
`F.cross_entropy`，是**唯一一处"说了要做但没做"的融合空间**。

## 1. 代码改动

### 1.1 CrossEntropy 融合接入

`custom/qwen_omnispeak/modeling_qwen_omnispeak.py` 新增 `_cross_entropy()`
封装（复用 `QWEN_OMNISPEAK_LIGER_OPS` 开关，lazy import + 异常兜底 + 对
`reduction='none'` 做 shape 校验防御历史版本 bug），替换 3 处调用：
codec loss（EOS 加权）、text loss、residual codec loss。Megatron 路径
（`mcore_talker_loss.py`）同步接入，保持两条后端覆盖率一致。

### 1.2 过程中顺带修复的两个独立 bug

1. **`custom/qwen_omnispeak/speaker_encoder.py` 死代码触发
   `trust_remote_code` 加载崩溃**：文件顶部有一段从未被用到的
   `from .codec_tokenizer import _resolve_audio_path`（该名字后面被同名
   真实实现覆盖），但 `transformers.dynamic_module_utils.
   get_relative_imports` 是纯正则扫描（不看 try/except），扫到这行就无条件
   去 `open()` 一个从不存在的同级文件，代码 hash 变化触发重新校验时直接
   `FileNotFoundError` 崩溃。已删除死代码。
2. **nsys Diagnostics 面板报错**：`nsys profile` 直接 launch 目标程序时
   `--sample` 默认值是 `process-tree`（不是 `none`），容器无 perf_event
   权限导致采集必然失败。已在 `custom/ray_nsight_patch.py` 的
   `_nsight_cfg` 里显式加 `'s': 'none'`，与本次分析用到的 CUPTI 数据无关，
   纯粹清掉一条注定失败、从未用到的诊断噪音。

## 2. 200 步稳态性能对照

基线来自 `GBS512_GBS4096_BASELINE_ABCD_200STEP_TRAINING_METRICS.md` 的
**F 列**（双节点 ray+deepspeed zero1+liger+Lance+GBS4096+OMP修复，同拓扑
同数据源同并行度，唯一差异是本次的两项代码改动）。

| 指标 | F（历史基线，无 CE 融合，zero1） | 本次 zero1+CE融合 | 本次 zero0+CE融合 |
|---|---|---|---|
| 稳态 s/it（190→200 窗口） | 9.44s（min8.99/max10.67） | 未跑满200步（见2.1节说明） | **9.4s**（几乎一致） |
| 累计 s/it@200（含冷启动） | 9.604 | — | 9.678 |
| 显存@200（GiB） | 81.04 | — | **100.86**（+24%） |
| loss 轨迹 | — | 与本次两个变体在 step30 高度一致 | 单调下降，无异常 |

### 2.1 zero1+CE融合任务未跑满 200 步的原因

`lance-ce-liger-mg1j` 任务只跑到 step 60+ 就手动停止了——因为它的唯一目的
是拿 step30 的 nsys 报告做 kernel 层面对比（见第 3 节），跑更多步对"验证
CE 融合是否正确"这个目标没有增量价值，遂提前停止腾出资源给 zero0 对照。
step30 处的 loss（11.93220367）与 zero0 版本（11.95398865）、原始基线
（11.93719406）三者高度一致，已经足以确认融合的数学正确性。

### 2.2 结论

**ZeRO stage（0 vs 1）在这个模型规模/GPU 数下对速度几乎没有影响
（9.4s vs 9.44s，差异在噪声范围内），但 zero0 显存开销明显更高（+24%）**
——zero0 只有在显存本身是瓶颈、且愿意牺牲显存换取（这里没有）速度收益时
才有意义；本场景下没有理由从 zero1 切到 zero0。

## 3. nsys 报告三方对照（同 step30，同拓扑同数据源）

| | zero1 baseline（无融合） | zero1 + CE融合 | zero0 + CE融合 |
|---|---|---|---|
| 报告文件 | `lance_ray2node_step30_worker_1532.nsys-rep` | `lance_ce_liger_step30_worker_1566.nsys-rep` | `lance_zero0_liger_step30_worker_1531.nsys-rep` |
| wall / GPU busy | 9.16s / 7.63s | 9.07s / 7.59s | 9.16s / 7.59s |
| idle 占比 | 16.7% | 16.3% | 17.2% |
| kernel 总数 | 91,623 | 91,823 | 95,461（+4%左右，量级不大） |
| NCCL 总耗时 / 次数 | ~115ms（14 AllReduce + 3 AllGather） | ~116ms（同量级） | **~76ms**（仅 4 AllReduce + 1 AllGather） |
| `liger_cross_entropy_kernel` | 无（原生 F.cross_entropy） | 24 次 / 17.3ms | 24 次 / 17.3ms |

**发现**：
1. **CE 融合确认真实生效**（`liger_cross_entropy_kernel` 实测出现），但
   总耗时仅 17.3ms，占 GPU 总耗时 **不到 0.3%**——CrossEntropy 从来不是
   kernel-launch bound 的主要贡献者，融合它对整体速度几乎没有可观测影响
   （wall/busy/idle/s_it 三份报告全部在噪声范围内持平）。
2. **zero0 的 NCCL 通信次数明显更少**（4+1 vs 14+3），符合原理（zero0
   不用像 zero1 那样为优化器状态分片做额外 gather），但因为 NCCL 本来
   占比就 <1.5%，这点差异同样不影响整体速度。
3. **kernel-launch bound 的真正大头**（91k+ 个 attention/RoPE/SiLU/
   elementwise 小 kernel）在这两项改动前后**完全没有变化**——这才是后续
   如果要真正提速应该攻的方向。

## 4. 总体结论 & 后续方向

- ✅ CrossEntropy liger 融合：**正确、安全，可以保留**，但不解决速度瓶颈
- ✅ ZeRO0 vs ZeRO1：**验证过没有速度差异**，zero0 显存开销更高，本场景
  没有切换必要
- ❌ 两项改动均**不是**能解决 kernel-launch bound 的关键杠杆
- 真正有想象空间但当前均被挡住的方向（详见此前代码库排查结论）：
  - `torch.compile`：被 MoE grouped-GEMM 的 Dynamo tracing 失败**主动锁死**
  - CUDA Graph：训练侧**从未实现**，只有推理代码有雏形
  - 需要进一步排查 attention/RoPE/SiLU 相关的 elementwise kernel 是否有
    更多融合空间（liger 目前只覆盖 RMSNorm/SwiGLU/RoPE/CE，SiLU 在
    SwiGLU 融合里已含，但 talker 主干之外是否还有零散调用点未排查）

## 5. 涉及文件

- 代码改动：`custom/qwen_omnispeak/modeling_qwen_omnispeak.py`、
  `custom/qwen_omnispeak/mcore_talker_loss.py`、
  `custom/qwen_omnispeak/speaker_encoder.py`（死代码修复）、
  `custom/ray_nsight_patch.py`（`-s none` 修复）、
  `pipelines/bridge_alignment/common_args.sh`（新增
  `CROSS_ENTROPY_FUSION_IMPL` 环境变量，Megatron 后端专用，本次未实测）
- 任务 yaml：`ktrain/ktrainjob_lance_ray2node_nsys180_tmp.yaml`（zero1+CE
  融合验证用）、`ktrain/ktrainjob_lance_ray2node_nsys180_zero0_tmp.yaml`
  （zero0 对照，唯一差异 `DEEPSPEED: zero0`）
- nsys 报告（本地 `/data/workspace/nsys/`）：
  `lance_ray2node_step30_worker_1532.nsys-rep`（baseline）、
  `lance_ce_liger_step30_worker_1566.nsys-rep`（zero1+CE融合）、
  `lance_zero0_liger_step30_worker_1531.nsys-rep`（zero0+CE融合）
