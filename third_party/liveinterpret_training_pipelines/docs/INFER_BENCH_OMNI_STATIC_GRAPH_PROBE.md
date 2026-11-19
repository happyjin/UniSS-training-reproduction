# 官方 Omni talker StaticCache / CUDA Graph 实验

最新状态：优化开关默认关闭；已完成 96 步静态 KV 控制实验、static eager / Graph 的 40 句 codes 对齐及 L4 全量质量跑测（见文末）。L2/L3 最新全量指标见 [系统跑测报告](INFER_BENCH_REPRO_OMNI_STREAMING.md)。以下失败/诊断段落按实验发生顺序保留，不代表当前仍停留在首轮失败阶段。线上长会话、并发与回退行为仍需另行验证。

实现文件：

- `pipelines/streaming_si/omni_talker_accel.py`：固定地址 KV、显式有效长度、1024/2048/4096 容量分档；首个 prefill 保留官方 DynamicCache 路径，后续单 token 可进入静态 decoder / Graph 路径。
- `patch_omni_repo.py`：将 helper 复制到官方仓库的本地副本，只为后续 prefill / step 添加可选入口，不修改原资产。
- `custom/streaming_si/eval/omni_graph_probe.py`：同一模型依次跑 eager / static / graph，记录 semantic codes 和文本，有差异即停止。
- `custom/streaming_si/tests/test_omni_talker_accel.py`：缓存扩容、超长回退、跨句缓冲区清理与存储复用。

环境变量：`SIMULS2ST_TALKER_ACCEL=static|graph`，不设则原路径；`SIMULS2ST_TALKER_CACHE_BUCKETS=1024,2048,4096`。
超过最大容量时将有效 KV 复制到 DynamicCache，在当前序列内保持 eager；新序列可重新使用静态缓存。
helper 属于单会话推理对象，不能被多个并发会话共享。Graph 捕获 incompatibility 只对识别到的 capture 错误尝试 static eager，其他错误传播。

探针复用 L40S 原镜像/PVC、L4、beam 4、8 步 BF16、FAST_TALKER、共享模型，设置：

```bash
SIMULS2ST_SHARED_MODEL=1 SIMULS2ST_TALKER_ACCEL=static \
SIMULS2ST_TRACE_CODES=1 SIMULS2ST_PROBE_MODES=eager,static,graph \
SHARED_LIMIT_PER_DIRECTION=20 STOP_AFTER=infer \
SYSTEM=rep_L4 bash pipelines/streaming_si/bench_entry.sh
```

每方向 20 句，另外各加一条预热；严格比较输出，不以小样本总墙钟报告全量 RTF。
eager 首轮包含冷启动，其 elapsed_sec 不能直接与随后模式的 elapsed_sec 比较；应看每方向 evaluate 耗时，Graph 首次捕获成本也在其中。

## 首轮结果

`bench-omni-graph-probe-20261029-181437-r2`：eager 与 static 均跑完 40 句及两条预热，各 74 个 semantic code 输出块。
文本全部一致，但 semantic codes 不完全一致；探针按预设报错停止，未执行 Graph，未启动全量。
检查中的首个不同输出块为第 2 块，前 39 个 codes 相同，随后出现分叉；不能仅凭这一点认定是数值误差，仍需诊断。

本轮还发现第三方 logger 引用已关闭的模式日志，导致 static 阶段出现 logging error；探针已修复为保留日志句柄。
因此本轮不提供有效的速度提升结论。失败产物发布功能也已扩展为保留 probe 的 infer 子目录，便于追踪每块 code。

缓存单元测试已在实际 Omni transformers 4.52 环境运行：2 项通过，覆盖 KV 扩容、超限转 DynamicCache、回退后数据不被缓冲区清零影响、新句尾部清理。
这些测试不等同于实际长语音、并发、重连和 Graph 正确性验证。

## 进一步诊断

Pod `bench-omni-static-shadow-20261029-182753`：每方向 2 句，加预热，eager / static 对照。
设置 `SIMULS2ST_TALKER_VALIDATE_STEPS=96`，对同一份输入/KV，额外运行 DynamicCache eager，记录 logits 最大/平均绝对误差与 top1 是否相同。
该模式有额外复制和双前向，完全不用于性能计时。

诊断已完成：eager/static 各 9 个 semantic 输出块，5 块 codes 不同，文本全部一致。
96 次相同输入/KV 的单 token 对照中，最大 logits 绝对误差 0.265625；逐步平均绝对误差的均值约 0.014425；
第 13、71、88 步的原始 logits argmax 不同。这里的 argmax 尚未施加解码约束，不能直接解释为 3 个最终 token 错误。
静态缓存与固定 attention mask 改变 attention 运算形状/内核是一种可能解释，但尚未通过控制实验排除位置/mask 适配问题。
下一项诊断应比较相同静态 KV 下的官方 outer forward 与直接 decoder forward，并确认二者 logits 对齐，再单独研究 Dynamic/Static 的内核差异。

该诊断阶段结论：当时仅完成实验实现与静态路径探针，Graph 尚未实机捕获验证；由于静态路径的严格 codes 一致性检查失败，按预设停止，未跑 Graph 或全量。后续控制实验和全量验收见下文。
不得用这些模式的 elapsed_sec 推导 L2/L3/L4 的加速倍数，也不得替换此前验证过的共享模型生产候选路径。
诊断产物在 PVC `junyaoyang/infer_bench/bench-omni-static-shadow-20261029-182753/_partial/infer/`，包括模式日志与 probe_results.json。

## 静态 KV 控制实验（已完成）

Pod `bench-omni-static-control-20261029-185718`，每方向 2 句加预热。新增 `SIMULS2ST_TALKER_VALIDATE_STATIC=1`：
在相同输入及相同静态 KV 下运行官方 outer forward，与新 direct decoder 路径对照，同时保留 DynamicCache 对照。
前 96 步输出 `[talker-static-control]` 的 exact、max_abs、mean_abs、top1_equal。
这样可把“静态路径适配是否正确”与“Dynamic/Static 的算术差异”分开；后者即使存在，也应经语音质量评测决定可否接受，不能仅凭 codes 不同认定质量下降或仅凭文本一致认定质量不变。

控制实验已完成：96/96 步静态 KV 下官方 outer forward 与新路径的 logits 逐位一致，最大绝对误差为 0，top1 全部相同。
同轮 DynamicCache 对照仍有最大 0.265625 的误差与 3 步 top1 差异。该结果支持已测单步的适配正确，不能推广为所有长度/分支均已验证。
下一阶段以 static eager 为对照验证 CUDA Graph；这不等同于要求 Graph 与原 DynamicCache 逐 code 相同，原动态路径仍是后续质量评估的基线。

## 全量验收（L4，1193 句，2026-10-30）

配置：VoiceBox 8 步 bf16 + FAST_TALKER + 共享模型单进程 + `SIMULS2ST_TALKER_ACCEL=graph`，
其余（L4 分块、thinker beam 4、1193 句串行）与 eager 基线完全一致。
Pod `curion/bench-graph-l4-l40s`；产物 `junyaoyang/infer_bench/20261029-194426-l40s-graph-l4/`。

| 指标 | eager（共享模型，2.10） | **CUDA Graph** | 变化 |
|---|---|---|---|
| 模型内 RTF（Σwall/Σ源） | 0.524 | **0.339** | −35.3% |
| 含 I/O RTF（推理墙钟/源时长） | 0.592 | **0.413** | −30.2% |
| 推理墙钟 | 5044s | **3521s** | −1523s |
| 单句均值 / 中位 / p90 | 3.74s / 3.49s / 5.77s | 2.42s / 2.25s / 3.61s | −35% |
| 显存峰值 | 14.4GB | 14.2GB | — |
| 失败 / 空译文 / 空语音 | 0 / 0 / 0 | 0 / 0 / 0 | — |

质量对照（eager → graph）：

| 指标 | RealSI en→zh | RealSI zh→en | ACL en→zh |
|---|---|---|---|
| Text-BLEU | 28.04 → 28.04 | 20.06 → 20.06 | 45.20 → 45.20 |
| ASR-BLEU | 26.50 → 26.04 | 19.70 → 19.75 | 42.62 → 42.55 |
| UTMOS | 2.925 → 2.923 | 3.430 → 3.440 | 2.848 → 2.847 |
| SIM-o | 0.808 → 0.813 | 0.795 → 0.792 | 0.946 → 0.947 |
| LAAL | 4.496 → 4.500 | 4.440 → 4.440 | 4.499 → 4.500 |

结论：

1. **质量门通过**：译文（Text-BLEU）三个数据集与 eager 完全相同（thinker 不走静态 cache，符合预期）；
   语音侧 ASR-BLEU 最大差 0.46（en→zh），UTMOS ≤0.01、SIM-o ≤0.003、LAAL ≤0.004，
   都在此前 bf16/采样带来的波动带内，不是退化。
2. **RTF 进到 0.339（模型内）/ 0.413（含 I/O）**，L4 从「刚过 0.6」变成有 30% 余量。
3. graph 与 static eager 的输出在 40 句探针里逐位一致（74/74 块），本轮全量也确认无失败、无回退。
4. L2 / L3 的全量已完成，最终结果见 [系统跑测报告](INFER_BENCH_REPRO_OMNI_STREAMING.md)；更长会话、不同长度分档和回退行为仍需补充验证。
