# 方案二 端到端 TurnCRF：可训练代码、测试与 GPU 冒烟（crf_novel_ideas.zh-CN.md §3、§9、§11）

**结论**：代码、CPU 测试和 GPU 冒烟都已完成，没有启动任何正式运行。

- g = 0 时，解码与 v8 逐 token 一致；训练第 1 步的 LM 损失与 C0 逐位相等。
- 3 步训练的两项损失都有限，g 和层参数都收到梯度，checkpoint 能加载并完成端到端解码。

**不建议在当前设计下做方案二的正式运行**，是否运行由用户决定。原因是 CPU 阶段没有通过预登记的停止规则（`turn_crf_cpu_stage.zh-CN.md`，commit bcd57411）：
- 流式 semi-CRF 的轮次 F1 提高 0.06；
- 但 ≤1 s 轮次召回比原始 P 低 0.21，置信区间 [−0.233, −0.188]；
- 最大似然训练学成了强平滑：内容权重约 0.26，所有转移分都是负的。

按协调方的指示，层默认使用**中性初始化**（独立 P），CPU 阶段训练出的参数只作为可选项（`--turn-init cpu_stage`）。

本文所有数据：训练只用 AMI train（sched_v1 窗口），冒烟解码只用 dev 的 2 个窗口。没有读取 RealSI、ACL 60/60、AliMeeting。

---

## 1. 文件

全部是新文件，没有改动已有文件。

| 文件 | 作用 |
|---|---|
| `experiments/mist_w3po_v1/crf/turn_layer.py` | `TurnLayer`：可微的右截尾 semi-CRF 前向-后向（torch 对数空间）、门控残差、辅助损失、checkpoint 读写 |
| `experiments/mist_w3po_v1/crf/turn_refs_e2e.py` | 参考帧标签：解析行格式目标，做窗口级重映射，取前 2 个槽；附带槽位约定统计的命令行 |
| `experiments/mist_w3po_v1/crf/train_turncrf_sft.py` | 薄封装：共用训练器 `train_crf_sft.py --objective heuristic`，加上 `post_transform` / `extra_loss` / `extra_params` |
| `experiments/mist_w3po_v1/crf/turn_decode.py` | 推理：`TurnEngine(Engine)`，以及 online_decode 的包装入口 `--turn-crf` / `--turn-ckpt`，默认关闭 |
| `experiments/mist_w3po_v1/crf/test_turn_layer.py` | CPU 测试，11 项 |
| `experiments/mist_w3po_v1/scripts/run_turncrf_sft.sh` | 正式运行启动脚本：4 卡，setsid nohup，拒绝覆盖。只做过 `bash -n` 检查 |

Commits：`6b173ddb`（层、标签、解码包装、9 项测试）和 `1d03e734`（训练封装、`--turn-ckpt`、启动脚本、11 项测试）。两者都已推送到 `private/main`。

## 2. 槽位约定（核查结论）

**v8 的行和投影器用的是会议级（session-level）槽。**
- v8 的训练行（`e2e_gstar_stream_zh2en_v1`，由 `build_ami_gstar_stream_rows_zh2en.py` 生成）写 `<|spk k|>`，其中 k = `slot_map.json["lowlat"][会议][说话人]`。
- 这个映射在整场会议上做匈牙利匹配（`build_ami_gstar_si_rows.slot_map`），从 1 开始编号，k 对应 npy 的第 k−1 列。
- 投影器直接读取窗口后验的各列，不做任何重映射。
- sched_v1 记录里的 `slot` / `S` 沿用同一约定。

**CPU 阶段（`turn_data.py`）用的是窗口级映射（S0-5）**：在窗口内按 P > 0.5 的帧重叠做匈牙利匹配；没有重叠的说话人保留会议级槽，被占用时改用最好的空闲槽。

**两种约定差别不小。** 在 v8 的 27,953 个 train 窗口上统计（`python -m experiments.mist_w3po_v1.crf.turn_refs_e2e`）：

| 统计项 | 窗口级槽 ≠ 会议级槽的比例 |
|---|---|
| 说话人-窗口 | 9.1 %（3,893 / 42,908） |
| 窗口 | 11.8 % |
| 语音帧的联合标签 | 7.3 % |

**辅助损失采用窗口级映射（一个预登记设置）。**
- 说话人取自窗口的**完整** v8 目标，即 `meta["rec"]["S"]` 中的 `<|spk k|><|t a|>…<|t b|>`，每段覆盖帧 [a, b)。
- 这些说话人在整窗后验（未扰动）上重映射到 diarizer 的列。
- 不用读次行自己的前缀加目标，因为其中缺少读次之后才提交的块，会把正在说话的帧标成"空"。
- 理由：CRF 的势函数在槽之间共享，只能清理 P 在 diarizer 自己那一列上的值，无法在列之间搬移概率质量。用会议级标签，等于在 9 % 的说话人-窗口上惩罚它跟随 P。
- 同时有超过 2 个槽活跃时，保留窗口内覆盖最长的 2 个（同长取槽号小的），与 `turn_data.joint_labels` 相同。犹豫语不在 v8 的目标里，所以标为"空"，与 CPU 阶段一致。

## 3. 层的设计

**输入与输出。**
- 输入：读次可见的前缀 P[:T]，T = max(0, t − 1040 ms) // 80，正是 v8 未置零的帧。还有音频编码状态 e，按训练器的 `enc_to_frames` 线性插值到 80 ms 帧上。
- 输出：P' = P + g·(π − P)，T 之后的帧保持 0。
  - π 是右截尾前缀的槽边际（最后一段用生存势，没有结束分）。
  - g 是标量，初始为 0。

**前向-后向。**
- 用列表和 stack 实现，没有原地写入，因此 autograd 经边际回传到全部势函数，即 Kim et al. 2017 的结构化注意力。
- 计算全部在 float64 中进行，参数以 float32 存储，因为训练器的梯度 all-reduce 只用一种 dtype。

**势函数。**
- 沿用 CPU 模块按类型共享的参数化：内容 a·log P(c) + b，时长表加尾部，生存势，关系转移 A，dP 边界权重 w，开始分。
- 已与 `turn_crf.SemiTurnCRF.potentials` 逐元素对拍。
- 再加一个与槽无关的 MLP(e)，结构 2048→256→9，最后一层零初始化：
  - 每帧对 u 加类型偏置（空 / 单人 / 两人），3 个数；
  - 进入第 t 帧的转移加 left[新标签类型] + right[旧标签类型]，共 6 个数。

**辅助损失**（权重 1）。
- 每行为 (log Z̃_T − log Z̃_{T,y}) / T。
- 在 micro-batch 内求和后除以本步全局行数，所以训练器对 micro-batch 和 rank 求和后，得到的是每步按行的平均。T = 0 的行贡献 0。
- 直接复用前向时缓存的 log Z 和势函数，不再算一遍。

**g = 0 处的一个性质。**
- 此时 ∂P'/∂φ = g·∂π/∂φ = 0，所以第一步 CRF 参数只从辅助损失得到梯度。
- g 一旦不为 0（一步之后），翻译损失的梯度也会到达 CRF 参数。

## 4. 测试

`PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_turn_layer` 只用 CPU，**11 项全部通过**（约 40 s）。

| 测试 | 内容 |
|---|---|
| (a) | g = 0 时输出与 P 逐位相同（float32 / float64，CRF 参数和 e-MLP 都是随机非中性的）；g 收到非零梯度；g ≠ 0 时所有 CRF / e-MLP 参数都有梯度；g = 0 时辅助损失能把梯度传到 CRF 参数 |
| (b) | 小实例穷举对拍：14 组，闭合与截尾、有无 e 项都覆盖，log Z、帧边际、槽边际都与穷举一致；K = 8、T = 60 时等于 `turn_crf.semi_forward_backward`，也等于 ∂logZ/∂u；对 u、tr、d、ds 的 gradcheck 通过；势函数与 CPU 模块一致 |
| (c) | 辅助损失与穷举对拍，16 条标签序列，缓存路径与重算路径都覆盖；`path_score` 等于 `turn_crf.semi_path_score` |
| (d) | 用 stub Thinker、真实特征提取器和真实 SDProjector：开关关闭时，以及 g = 0 层开启时，说话人流嵌入、模型输入（7 次前向）、提示长度和贪心 token 全都与原 `Engine` 相同；g ≠ 0 时嵌入会变。`turn_decode.main` 不带开关时就是 `online_decode.main`，参数与 Engine 都原样；`apply_ckpt` 正确折叠 LoRA，并载入投影器、新行和层 |
| (e) | 标签：段解析、窗口级重映射、前 2 个槽的规则 |
| (f) | 训练钩子：g = 0 时输出等于输入；辅助损失的归一化正确；梯度经训练器的 `post_to_tokens_torch` 到达 g；e 对齐与训练器的 `enc_to_frames` 逐位一致；`extra.pt` 能往返读写 |

另外核对过：训练器的 `post_to_tokens_torch` 与 v8 的 numpy 版 `post_to_tokens` 在 300 个随机实例上逐位相等。

## 5. GPU 冒烟

**环境**：卡 4–7，同事每卡占约 39–44 GB。每个作业都配了显存守护：余量低于 8 GiB 就按 PID 结束，实际没有触发。输出目录在 `/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/crf_smoke_turn/` 下。

**(i) g = 0 与 v8 一致**，目录 `i_g0_identity_20261109T162822Z`。
- 用 dev 的两个窗口 `ES2011a_on_0072`（15.7 s）和 `_0073`（20.5 s）。
- 解码设置：context 2，wait 策略，贪心。
- 对比原 online_decode 与 `turn_decode --turn-crf`。后者的 g = 0 层故意设成非中性：CPU 参数加随机 e-MLP，这样 π ≠ P。
- **两个窗口的 raw 和 read_log 完全相同**（16 / 21 次读；3 / 22 段）。
- 显存：卡总占用峰值 108.5 / 111.3 GB。

**(ii) 3 步训练**，单卡，16 行/步，`--mem-cap-gib 85`。
- 默认的中性初始化：目录 `ii_train3_neutral_20261109T165213Z`（下表）。
- 改默认之前跑过一次 CPU 参数初始化：目录 `ii_train3_20261109T164609Z`。
- 对照用的 C0 也在后一个目录里，设置相同。

| 第几步 | LM 损失 | C0 的 LM 损失 | 辅助损失 | 全局 gnorm（C0） | g 的局部梯度 | 每步耗时 s（C0） |
|---|---|---|---|---|---|---|
| 1 | 0.8372076154 | 0.8372076154（逐位相等） | 0.289 | 6.62（6.61） | 2.5e-5 | 5.1（7.1） |
| 2 | 0.9670 | 0.9790 | 1.210 | 5.60（5.31） | 5.5e-4 | 6.9（4.1） |
| 3 | 0.9358 | 0.9296 | 0.553 | 5.02（4.91） | 2.3e-4 | 4.0（3.1） |

- 所有损失都是有限值。
- 层的全部参数都收到梯度：a、b、A、w、d_tab、d_tail、start、e-MLP 两层。e-MLP 第一层的梯度从第 2 步才开始出现，因为它前面的最后一层是零初始化的。
- 3 步后 g = 2.2e-6，此时学习率预热中，约 2e-6–6e-6。
- 第 3 步保存了 checkpoint：`ckpt/step_00003/{lora.pt, extra.pt, optim.pt, DONE}`。
- 显存：分配器峰值 63.9 GiB，卡总占用 ≤ 107.3 GB。
- 第 2 步起与 C0 有约 1e-3 的差。原因还没有单独验证，推测是 GPU 反向的非确定性，被 Adam 第一步（≈ lr·sign）放大。
- CPU 参数初始化那次结果类似：辅助损失 0.087 / 0.093 / 0.170，第 1 步 LM 损失同样与 C0 逐位相等。

**(iii) checkpoint 端到端解码**，目录 `iii_decode_20261109T164854Z`。
- 命令：`turn_decode --turn-ckpt <(ii) 的 CPU 参数初始化那次的 step_00003>`。
- 加载情况：192 个 adapter 折叠进权重，投影器和 259 个新行已载入，层 g = 7.4e-6。
- 两个 dev 窗口都解码完成（16 / 21 次读；3 / 24 段）。
- 输出与 v8 不同。这是预期的：3 步之后 LoRA、投影器和新行都已改变。
- 显存：卡总占用峰值 111.6 GB。

## 6. 正式运行命令（未执行，等用户决定；当前不建议运行）

```bash
CARDS="4 5 6 7" OUT=eval_outputs/mist_w3po_v1/crf/turncrf_sft_v1 \
  bash experiments/mist_w3po_v1/scripts/run_turncrf_sft.sh
# 实际执行：CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port 29871 \
#   -m experiments.mist_w3po_v1.crf.train_turncrf_sft --out $OUT --steps 3000 --save-every 500 --mem-cap-gib 0
# 日志在 logs/mist_w3po_v1/turncrf_sft_v1.log(.rc)；层的旁路日志在 $OUT/turn_layer.jsonl；对照是共用的 C0
```

解码某个 checkpoint：`python -m experiments.mist_w3po_v1.crf.turn_decode --thinker <v8 merged> --windows … --turn-ckpt $OUT/ckpt/step_NNNNN`，其余参数与 online_decode 相同。

## 7. 风险与对 HOOK API 的意见

1. **前提不成立。** CPU 阶段已经失败（见开头），而中性初始化就是"独立 P"的起点。端到端训练能否修好短插话召回，没有证据支持。
2. **速度。** 钩子按行调用，CRF 只能一行一行算，主要开销在 kernel 启动上：在繁忙的 H200 上，T = 120 每行约 0.25 s，T = 237 每行 0.7–0.8 s。冒烟中每步比 C0 多 1–3 s，约 +30–70 %。**HOOK API 缺少批量形式的 `post_transform`**，即一次传入整个 micro-batch 的行列表。如果有，可以做左填充的批量前向-后向。推理时已经按 T 分组批量处理，16 行、T = 237 约 0.14 s。
3. **g 的步长。** g 与 LoRA 共用学习率（1e-4，余弦衰减到 1e-5）。Adam 每步最多移动约 lr，所以 3000 步后 |g| 大约不超过 0.2。中性初始化下 π ≈ P，g 的梯度很小。端到端的耦合可能一直很弱，这是设计上的限制，不是 bug。
4. **全局梯度裁剪包含层的梯度。** 冒烟中层只给 gnorm 加了 0.01–0.3，而 gnorm 本身在 5–6，与 C0 一样每步都会被裁剪，影响可以忽略。
5. **正式评测入口。** `scripts/eval_pipeline.sh` 和 `launch_decode.sh` 把解码模块写死为 online_decode，而 `--turn-ckpt` 不能通过 EXTRA 传进去（online_decode 会拒绝未知参数）。正式评测前，需要给这两个脚本加一个默认关闭的 `DECODE_MODULE` 环境变量。我没有改这两个脚本，避免改动正在使用的已有脚本。
