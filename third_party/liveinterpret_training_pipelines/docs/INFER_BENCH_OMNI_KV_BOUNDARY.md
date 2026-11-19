# Omni streaming：音频时长 / KV 回退 / 显存边界探针

状态：隔离 L40S 探针 `curion/bench-omni-kv-boundary-20261030-190550` 已完成（`Succeeded`），推理及最终压力试验结果已发布。
代码：[omni_kv_boundary_probe.py](../custom/streaming_si/eval/omni_kv_boundary_probe.py)。

## 要回答的问题

1. 默认历史窗口下，连续长语音的 KV 是否会达到 4096？
2. 关闭历史窗口后，首次超限回退对应源音频的哪个时段？
3. 模型实际加载、推理完成后的空闲显存有多少，额外 KV 等效分配在何处触发 CUDA OOM？

## 已确认的计算与限制

当前资产 offline/config.json 的 talker：24 层，2 个 KV heads，head_dim=64，BF16 每元素 2 字节。
纯 KV 每 token 字节数为 `2(K/V) × 24 × 2 × 64 × 2 = 12288`，4096 个 token 为 **48 MiB**。
这不包括 thinker、模型权重、attention 临时张量、Graph 内存池或其他模块，也不能据此计算服务并发容量。

目标语音 codec 的每个 code 对应约 80ms；只包含 4096 个输出 codes 的理想化时长为 327.68s。
真实 talker KV 还含条件输入、文本和边界 token，因此 **4096 KV token 不等于 327.68 秒源语音**。

官方 agent 默认按历史窗口裁剪上下文并清空 thinker/talker KV。当前 L4 参数 28/16 经 L=4 换算成 7/4 个 turns，
若每 turn 对应完整 4 秒块，约 28 秒历史后开始裁剪，并保留重叠历史；有 WAIT、尾块等情况时具体事件以日志为准。
长会话不一定达到静态缓存上限，不能预先承诺“超过固定秒数必定回退”。

## 实验设计

- 原优化组合：L4、beam 4、8 步 BF16、FAST_TALKER、共享模型、CUDA Graph、1024/2048/4096 档位。
- 从真实测试音频片段拼接/重复生成 en、zh 各 192 秒输入，同时保留 3 秒短样本及原预热。
- 同一加载模型依次跑 `windowed`（默认窗口）与 `unwindowed`（关闭窗口）两种模式。
- 每次决策记录源音频已消费秒数、输出时长、KV 长度/类型、历史 turns、回退次数、Graph 重放次数及显存使用。
- 回退发生于一个决策内部，源时间只能先定位在相邻决策边界之间，不能将事件结束时间当作毫秒级首次超限时间。
- 这批拼接音频用于容量压力探针，不代表自然长会话分布，不做翻译质量结论。

推理完成后，在独占 GPU 上保留加载模型，用 256 MiB 块逐步增加设备分配，捕获 `torch.cuda.OutOfMemoryError` 并释放压力缓冲区。
该步骤由 `SIMULS2ST_BOUNDARY_STRESS_OOM=1` 单独打开，默认关闭。
记录的是 **KV 等效分配的显存边界**，不是多个真实请求同时生成的吞吐/并发 OOM 阈值。
真实服务需要独立会话状态与并发调度实现；当前 inference 对象有共享 RoPE 状态，不支持直接被多个并发会话复用。

## 启动与产物

复用原 GPU 跑测镜像与资产，设置：

```bash
SYSTEM=rep_L4 STOP_AFTER=infer SIMULS2ST_SHARED_MODEL=1 \
SIMULS2ST_FM_STEPS=8 SIMULS2ST_VOICEBOX_DTYPE=bf16 SIMULS2ST_FAST_TALKER=1 \
SIMULS2ST_TALKER_ACCEL=graph SIMULS2ST_BOUNDARY_PROBE=1 \
SIMULS2ST_BOUNDARY_SECONDS=192 SIMULS2ST_BOUNDARY_STRESS_OOM=1 \
bash pipelines/streaming_si/bench_entry.sh
```

路径与当前环境变量由 Pod 配置传入；原始资产不修改。
产物前缀为 PVC `junyaoyang/infer_bench/bench-omni-kv-boundary-20261030-190550/`：

- `infer/boundary_manifest.jsonl`：生成的压力输入清单；
- `infer/{windowed,unwindowed}/run_{en2zh,zh2en}/kv_boundary.jsonl`：逐决策数据；
- `infer/boundary_results.json`：各模式是否完成；
- `infer/memory_capacity.json`：KV 字节计算、分配前空闲显存、最后成功额外分配量和 OOM 信息；
- `logs/infer.1.shared.log`：初始化、分档、回退和探针日志。

若长语音推理先触发 OOM，模式结果会先落盘后抛出，跳过最终额外分配试验；失败发布逻辑会保留探针 infer 目录。

## 实测结果（2026-10-31 核对）

Pod 正常退出，shell 探针总墙钟 1036s；默认窗口与关闭窗口两种模式均完成到 192 秒源音频。
总墙钟包含首次初始化和额外显存试验，不能作为标准语料 RTF 成绩；各模式 Python 计时也不宜直接比较，首轮包含冷启动。

| 模式 | 方向 | 源音频长度 | 最大 talker KV | 首次 DynamicCache 回退 | 自然推理 OOM |
|---|---|---|---|---|---|
| 默认历史窗口 | en→zh | 192s | 1343 | 无 | 无 |
| 默认历史窗口 | zh→en | 192s | 1221 | 无 | 无 |
| 关闭历史窗口 | en→zh | 192s | 7309 | 消费源音频 96～100s 区间 | 无 |
| 关闭历史窗口 | zh→en | 192s | 7282 | 消费源音频 96～100s 区间 | 无 |

英文实际超限请求 `required=4162`，中文 `required=4097`；第一次回退后决策结束的 KV 分别为 4181 / 4144。
关闭窗口后能继续处理到 192 秒，说明本轮确实经过了“4096 静态上限 → DynamicCache 继续推理”的路径。
这里约 100 秒是这批拼接输入和 L4 配置的观测值，不是所有语音、L 档位或语速通用的阈值。

默认窗口阶段 PyTorch 最大已分配显存约 **12.73 GiB**；关闭窗口长上下文阶段累计峰值约 **30.75 GiB**。
这是整个推理过程的分配峰值，包含 thinker 和临时张量，不能将增加量全部归因于 talker KV 或 DynamicCache。
未测到单会话自然 OOM 的音频长度，只能说本轮 192 秒内没有 OOM。

### 额外显存分配压力试验

- GPU 总可见显存 **44.39 GiB**（该 L40S 显示 46068 MiB）。
- 保留模型和已有缓存后，成功额外分配 **30.5 GiB** 的原始缓冲区，再申请 256 MiB 时触发 CUDA OOM。
- 报错时设备空闲约 233 MiB；报错文本显示本进程总占用约 44.16 GiB。
- 捕获后记录的 PyTorch allocated / reserved 为约 **42.51 / 43.59 GiB**；Graph 私有池等统计口径另有差别。
- 分配前设备空闲约 7.42 GiB，但 PyTorch 还保留可复用的 reserved 内存，因此成功额外分配量可以大于设备当时显示的 free。

该结果证明隔离压力测试捕获了显存分配 OOM，并不提供真实并发请求数，也不能换算成“某秒语音必定 OOM”。
真实并发服务还需考虑独立会话缓存、临时计算峰值、调度与请求取消，另行设计并发试验。

### 数据标签限制

首轮使用官方 agent 的 `_source_entry_for_state` 标记源路径；上游 SimulEval 将 instance index 留在 0，导致长样本事件也被标成 `*_short.wav`。
本轮按模式、方向、`source_consumed_sec`（达到 192s）及 manifest 顺序辨认长样本，不能按原 `source` 字段区分三个输入记录。
后续探针代码已改用顺序 sidecar record_id 标记路径；这是日志标签修正，未改变本轮音频、KV 或显存数值。

## 默认窗口长输入复测（已完成）

Pod `curion/bench-omni-kv-normal-20261030-201644`，L4 与原默认历史窗口 28/16 不变；
英文、中文各构建一条 600 秒连续输入，从更多不同测试片段拼接，材料不足才重复，并在 `input_composition.json` 记录是否重复。
仅执行 `windowed`，关闭额外显存分配压力试验。保留短样本与预热，额外记录 `history_resets.jsonl` 和逐决策 monotonic 时间。
目的为观察默认窗口下持续输入是否回退，而不是保证某个总时长必定达到 4096。若 600 秒仍未回退，只能报告该观测下限。

两轮测试均为文件模拟流式：同一长请求按 4 秒块推进，没有人为等待或“100 秒请求 → 休息 → 另一个 100 秒请求”。
96～100 秒回退区间指**源音频已消费的位置**，不是计算用墙钟。吞吐受计算速度影响，测试未按真实采集/网络速度 pacing。
本轮产物前缀 `junyaoyang/infer_bench/bench-omni-kv-normal-20261030-201644/`，Pod 正常退出并完成发布。

### 600 秒结果

英文使用 84 个不同片段（原材料 606.961s），中文使用 99 个不同片段（原材料 601.707s），均截取为 600s，没有循环重复材料。
它们仍是片段拼接的长输入，不是真实单场自然对话，也没有按麦克风/网络真实时间 pacing。

| 方向 | 连续源音频 | 长样本决策次数 | 最大 talker KV | DynamicCache 回退 | 历史裁剪次数 |
|---|---|---|---|---|---|
| en→zh | 600s | 150 | 1364 | 0 | 48 |
| zh→en | 600s | 150 | 1352 | 0 | 48 |

两方向的裁剪事件首次记录于处理完源音频 32s 的决策，最后记录于 596s；裁剪前 7 个历史 turns，
保留重叠 4 个再加当前 turn，因此事件记录的裁剪后历史为 5。后续约每 12 秒源音频发生一次裁剪。
累计峰值 PyTorch allocated 约 12.73 GiB，无自然推理 OOM，额外分配压力试验关闭。
shell 总墙钟 800s，包含初始化、输入构建、短样本、预热与两方向，不能解释为每条 600 秒输入的独立 RTF。

结论：默认历史窗口的这两条 10 分钟输入均未达到 4096，不能给出一个必定回退的源时长。
测试只支持“这批输入默认窗口下到 600 秒未回退”，不证明所有内容/档位都不会回退，也不证明长期上下文记忆不丢失。

## 同一会话停发后恢复（已完成，真实输入节奏）

Pod `curion/bench-omni-kv-pause-20261030-205536`，英文、中文分别一条 1200 秒输入，默认 L4 历史窗口不变。
本轮设置 `SIMULS2ST_BOUNDARY_REALTIME=1`，在输入分块的发送边界按每秒 1 秒音频的节奏推进；
处理过 600 秒源音频后，在发送下一块之前实际停发 30 秒，再继续同一输入到 1200 秒。
期间不发送静音、不调用 reset、不重建 agent/evaluator，不做显存分配压力试验。
若计算未跟上输入节奏，后续块会按已逾期的截止时间尽快发送；本轮不是 HTTP/WebSocket 网络服务压测。

暂停前后同步 GPU，记录 talker KV 对象 ID、类型、有效长度和有效 K/V 内容 SHA256，以及已处理样本数、历史 turns、Graph 重放与回退计数。
若这些状态在无输入期间发生变化，则探针报错；恢复后的首个决策另记耗时与 KV。
正常恢复后仍可能按原历史窗口裁剪，不能将恢复后的正常窗口裁剪误认为暂停自动丢缓存。

新增参数：`SIMULS2ST_BOUNDARY_PAUSE_AFTER=600`、`SIMULS2ST_BOUNDARY_PAUSE_SECONDS=30`。
参数不设置时保持之前的快速文件模拟；暂停默认不启用。为避免误配置，单次暂停限制为 0～60 秒。

产物前缀 `junyaoyang/infer_bench/bench-omni-kv-pause-20261030-205536/`：
`infer/input_schedule.json` 记录节奏配置，`infer/windowed/run_*/pause_events.jsonl` 记录 pause_begin、pause_end 和 first_decision_after_resume。
每方向约 20.5 分钟输入/停发时间，两个方向串行。Pod 已正常退出并完成产物发布，shell 总墙钟 2906s，
Python 模式计时 2868.421s；这些时间包含实时 pacing、停发和初始化，不能用作模型纯推理 RTF。

### 暂停恢复结果

| 方向 | 暂停前 KV | 停发 30 秒后 KV | 恢复后首个决策 KV | 全程最大 KV | 回退 / 自然 OOM |
|---|---|---|---|---|---|
| en→zh | 957 | 957 | 1092 | 1380 | 0 / 0 |
| zh→en | 1062 | 1062 | 1231 | 1360 | 0 / 0 |

暂停实际 sleep 为 30.000075 / 30.000092s。两方向暂停前后 talker KV 对象 ID、类型、长度、
有效 K/V 的 SHA256、已处理样本数、历史 turns、Graph replay 和 fallback 计数全部一致，changed_fields 均为空。
PyTorch allocated / reserved 字节数在各方向暂停前后也相同。

恢复后两个方向均继续使用 PositionedCache 和已有 Graph，首个新决策位于源音频 604s，
距恢复约 3.908 / 3.837s；该耗时包含等待后续实时输入，不是纯计算延迟，也不能解释为 Graph 额外卡顿。
两方向各完成长样本 300 次决策至源音频 1200s，均未回退 DynamicCache，未发生自然推理 OOM；
累计 PyTorch allocated 峰值约 12.74 GiB。英文 158 个片段、中文 181 个片段，原始材料均足够 1200s，没有循环重复。

本轮验证：推理层同一会话单纯停发 30 秒不会自动增减/清空已有 KV，恢复后能继续。
暂停前保留的是默认窗口内的近期上下文，不是前 10 分钟的全部历史；窗口裁剪仍按原规则发生。
未覆盖 HTTP/WebSocket 超时、断线重连、VAD 切会话或服务端主动销毁状态等连接/调度层行为。
