# 方案一′ CPU 阶段：调度格、数据准备与预检脚本（2026-11-09）

> 对应 `reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md` §10.1–§10.5、§11。本阶段只用 CPU，没有用任何 GPU；
> 只读了 AMI 训练窗口与 v8 的训练行，没有读 RealSI、ACL 60/60、AliMeeting。

## 0. 结论

| 项 | 结果 |
|---|---|
| 调度格模块 | `experiments/mist_w3po_v1/crf/schedule_lattice.py`（提交 `9afb128c`） |
| 单元测试 | `experiments/mist_w3po_v1/crf/test_schedule_lattice.py`：9 项全部通过（2.6 s） |
| 数据准备 | `experiments/mist_w3po_v1/crf/build_sched_windows.py`（提交 `94b2a138`），输出 `eval_outputs/mist_w3po_v1/crf/sched_v1/`（56 个分片，82 MB） |
| 预检脚本 | `experiments/mist_w3po_v1/crf/precheck_sched.py`：只做了 `py_compile` 和 `--dry-run`，**没有运行** |
| **精确分解前提（§10.2）** | **成立。** 行构建器与推理引擎两侧都核对过：第 r 次读的已提交历史恰好是 S 的前缀（token 级），不含读标记、分隔符，也没有依赖提交时刻的时间或说话人 token。v8 的 130,560 条读取行与内存重建逐条一致，0 处不符。注意事项见 §4.3 |
| L_ref（λ\* 的目标） | 启发式调度的平均滞后 **0.826 s**（按窗口平均；按 token 加权 1.086 s） |
| β_heur 越界 | **0**（27,953 个窗口、915,457 个 token 中没有一个越出 lo/hi） |
| 格的可行性 | 27,953 个窗口用随机 log 概率全部可行（log Z 有限）；W = 128 时同样全部可行 |
| CPU 耗时 | 单线程，单窗口前向-后向平均 2.4 ms（p99 10.8 ms），Viterbi 1.0 ms |
| GPU 预检的阻塞项 | 无代码层面的阻塞。需要 GPU（约 2 张卡，`--max-gib 40`），目前 8 张卡都被同事占用 |

## 1. 调度格模块

**模型。** 一个窗口有 R 次读，S 有 N 个 token。调度 β_0 = 0 ≤ β_1 ≤ … ≤ β_R = N，允许空块。第 r 次读提交 S[β_{r−1}:β_r)，再接结束符 e_r（`<wait>`，最后一次读为 `<eos>`），块势函数为

ψ_r(β,β′) = Σ_{j=β}^{β′−1} tok_lp[r,j] + end_lp[r,β′] − (λ/N)·Σ_{j=β}^{β′−1} lag[r,j]

所以 Σ_r ψ_r = log p(S,z|x) − λ·Lat(z)。可行性：token j 只能在 lo[j] ≤ r ≤ hi[j] 的读提交（闭区间），块长不超过 W。

**实现要点。**
- 全部用 torch float64，设备无关（CPU 上测试；以后可直接放到 GPU）。
- 支持填充批：token j ≥ n_tok 与读 r ≥ n_read 都不可行；填充读只能提交空块、得分为 0，所以每个窗口的 log Z 在其自身终态 n_tok 处读出。
- 前缀和：每次读先对 tok_lp − (λ/N)·lag 做前缀和 P_r，并算出"以状态 k 结尾、全部可行的最长连续块" m_r[k]（含 W 上限）。于是 α_r(β′) = P_r[β′] + end_lp[r,β′] + LSE_{w ≤ m_r[β′]} (α_{r−1}(β′−w) − P_r[β′−w])，用 `unfold` 一次算完。复杂度 O(R·N·W)。
- 不可行用 NEG = −1e30 表示（不是 −inf），因此对数和与梯度都保持有限；`end_lp = −inf` 会被钳到 NEG，`tok_lp = −inf` 视为该 (r,j) 不可行。

**接口。**

| 函数 | 输出 |
|---|---|
| `pack(windows, W)` / `Lattice.single(...)` | 填充批 / 单窗口 |
| `marginals(lat, λ)` | log Z；w_tok[r,j] = P(β_{r−1} ≤ j < β_r)；w_end[r,b] = P(β_r = b)；E[Lat]；`feasible`；后向算出的 log Z（自检） |
| `forward(lat, λ)` | α 与 log Z，可对 tok_lp、end_lp、lag、λ 求导 |
| `viterbi(lat, λ)` | MAP 调度及其得分 |
| `sample(lat, λ, n, generator)` | 精确 FFBS（前向滤波、后向 Gumbel 采样） |
| `solve_lambda(lats, target, weights)` | 全局 λ\*：先倍增找区间，再二分，使（加权）平均 E[Lat] 等于目标 |
| `blocks_ok`、`schedule_score`、`schedule_lat` | 给定调度的可行性、得分、Lat |

w_tok 用累计分布的差得到：P(β_{r−1} ≤ j < β_r) = P(β_{r−1} ≤ j) − P(β_r ≤ j)（因为 β 单调），所以边际是 O(R·N) 的附加计算。不可行窗口的权重置 0，E[Lat] 置 NaN，`solve_lambda` 会跳过它们。

## 2. 单元测试（全部通过）

运行：`PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.test_schedule_lattice`（本环境没有 pytest，函数名按 pytest 约定命名，也可以用 pytest）。

| 测试 | 内容 | 结果 |
|---|---|---|
| (a) `test_brute_force_single` | 600 个随机小实例（N ≤ 5，R ≤ 4，随机 lo/hi/W/λ，含 −inf 项）：log Z、后向 log Z、w_tok、w_end、E[Lat]、Viterbi 得分与调度（唯一最大时）、每个 token 恰好提交一次、每次读恰好一个终态；穷举无可行调度时格必须报告不可行 | 通过（293 个可行、307 个不可行），误差 ≤ 1e−10 |
| (a) `test_brute_force_padded_batch` | 40 批 × 6 个不同 R、N（含 N = 0）的填充批，逐窗口与穷举比较 | 通过 |
| (a) `test_ffbs_frequencies` | 6 个实例各采 40,000 个样本，每条调度的经验频率与精确概率之差 ≤ 5σ + 1e−3；采到不可行调度即失败 | 通过 |
| (a) `test_blocks_ok_and_schedule_score` | MAP 调度可行，得分与穷举一致 | 通过 |
| (b) `test_autograd_marginals` | 60 个窗口（填充批）：∂logZ/∂tok_lp = w_tok，∂logZ/∂end_lp = w_end，∂logZ/∂λ = −Σ E[Lat] | 通过，误差 ≤ 1e−10 |
| (c) `test_hmt_oracle` | 150 个随机 HMT 实例（N ≤ 6，K ≤ 4，L ≤ 3）：格的 log Z = HMT 前向算法的 log p(y\|x)；格的 w_tok 在 t_{i,k} 处 = HMT 的状态后验 p(z_i = k\|x,y)（对 HMT 前向自动求导得到）；小实例再与穷举对拍 | 通过，误差 ≤ 1e−10 |
| (c) `test_hmt_normalised` | 预言机自检：发射概率全为 1 时 HMT 的转移归一，log p = 0 | 通过 |
| (d) `test_solve_lambda` | 二分命中目标（\|E[Lat] − 目标\| < 1e−6）；E[Lat] 随 λ 单调不增 | 通过 |
| `test_empty_stream` | N = 0 的窗口 | 通过 |

**HMT 预言机。** 按论文公式（不是 fairseq 代码）直接实现：状态 t_{i,k} = L + (i−1) + (k−1)（不截断到 \|x\|），
p(z_i = k \| z_{i−1} = k′) = c_{i,k} ∏_{l: t_{i−1,k′} ≤ t_{i,l} < t_{i,k}} (1 − c_{i,l})（t_{i,k} < t_{i−1,k′} 时为 0），c_{i,K} = 1，第一个 token 从其第一个状态开始。映射到格上：读 r ↔ t = L − 1 + r（多放一次在第一个状态之前的读，用来检验"状态范围外 end_lp = 0"）；tok_lp[t_{i,k}, i] = log c_{i,k} + log p(y_i\|k)；token i 的状态范围内 end_lp[t, i] = log(1 − c_{i,·})，范围外为 0；end_lp[·, N] = 0；lo/hi = 状态范围；W 不设上限。
推导：状态 β_t = i 的读，恰好是"写出 y_i 之后、写出 y_{i+1} 之前"的那些读，所以 HMT 转移里的 (1 − c) 连乘正好落在这些读的结束项上；c_{i,K} = 1 让 log(1 − c) = −inf 只出现在本来就不可行的状态上。
另做了一次变异检验：把状态范围外的 end_lp 从 0 改成 log 0.5，50 个实例全部不再相等，说明这个对拍有区分力。

## 3. v8 配方核对

- **v8 是谁。** `thinker_gstar_s2dl_zh2en_3k_20261030`（LOG 第 232 行），由 `reports/gstar_si_paper_v1/run_v5_train.sh:38` 用 `data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_train.jsonl` 训练（`experiments/uniss_qwen3_omni_v1/training/pretrain_thinker_gstar.py`，从 O2 合并权重出发 3000 步）。这些行由 `experiments/uniss_qwen3_omni_v1/data_clean/build_ami_gstar_stream_rows_zh2en.py`（种子 20261030）从 `e2e_gstar_zh2en_v1/windows_train_ref.jsonl` 构建。**v8 的行文件存在**，所以直接读它做核对。
- **`build_s0_rows_zh2en.py` 不带开关时是否复现 v8：配方相同，但行文件不同。**
  - 相同的部分：分块和串行化（`chunks_of_window` 不带 jitter 时等于 v8 构建器第 57–67 行），读取循环（`read_rows`，第 167–184 行，等于 v8 第 70–86 行），提示格式（ctx 为空时 `Prompter` 断言与 v8 提示相同，`core/engine.py:72`）。所以 **S 和启发式调度与 v8 完全相同**。
  - 不同的部分：
    1. 种子 20261106（第 122 行）而不是 20261030，所以保留哪些空读、抽哪些离线行和单说话人行、打乱顺序都不同；
    2. `--n-read 130560`（第 110 行）是从读取行池里**再抽样**：池的大小随保留空读的随机性在 130,560 上下浮动，池更大时会随机丢掉一些读取行（包括非空行），更小时全取；v8 是全取；
    3. 行 id 写 `|read|`（v8 是 `|lowlat|`），多了 `"ctx": ""` 字段，要用 S0 的训练入口 `pretrain_thinker_gstar_s0.py` 训练（ctx 为空时提示与 v8 相同）；
    4. 不生成验证集行。
  - 结论：分布意义上是 v8 配方，不是逐字节复现。一′ 只用 S 和启发式调度，这两样与种子无关。
- **核对结果。** `build_sched_windows.py` 按 v8 构建器的代码在内存里重建每个窗口，再与 v8 行文件逐条比较 (prefix, target, end)：**130,560 条读取行全部一致，0 处不符**；27,953 个训练窗口都有 v8 行。

## 4. 精确分解前提的核对（§10.2）

**结论：成立。** 第 r 次读的模型输入只依赖 (t_r, c, S^{<β_{r−1}})，历史是 S 的 token 级前缀，不含读标记。

### 4.1 行构建器（训练标签）

- v8 构建器第 67 行把块按 (commit_ms, src_start_ms) 排序；S 就是这个顺序下各块 piece 的拼接。
- 第 74–75 行：读 t 的 prefix = commit_ms ≤ 上一次读时刻的块，target = prev < commit_ms ≤ t 的块（最后一次读取剩余全部）。由于已按 commit_ms 排序，prefix 恰好是排序列表的前缀，prefix(r+1) = prefix(r) + target(r)；`prev` 在每次读之后都会更新（第 86 行），与该读的行是否被保留无关。
- 第 64–65 行：每块的 piece = `<|spk k|><|t a|>文本<|t b|>`。k 来自整场会议的 lowlat 槽位映射，a、b 是**源语**时间（相对窗口起点，80 ms 量化），都与提交时刻、读的网格无关。块之间没有分隔符；`<wait>` / `<eos>` 只作为该行目标的最后一个 token（训练器 `pretrain_thinker_gstar.py:76`），从不进入 prefix。
- token 级：训练器把 `head + prefix` 作为一个字符串分词，target 单独分词（`pretrain_thinker_gstar.py:74–76`）。时间和说话人 token 是加入词表的特殊 token（`processor()` 第 38 行），分词器先按它们切分，而每个 piece 以特殊 token 开头和结尾，所以在块边界切开的分词等于整串分词。`build_sched_windows.py` 对 27,953 个窗口逐个断言"整串 S 的分词 = 各 piece 分词的拼接"，全部通过；也验证了 `…assistant\n` + piece 的联合分词等于分别分词之和。

### 4.2 推理引擎（解码与 RL）

- 提示：`Prompter.ids` = head_a（音频 pad 展开）+ ctx + head_b + 已提交 id（`core/engine.py:85–88`），已提交 id 之间什么都不插。
- 已提交 id = 之前各读生成的 id 去掉 `<wait>` / `<eos>` 后的拼接：`rl/rollout.py:133`（以及 211 行的分支 rollout）、`evaluation/online_decode.py:169–179`。
- 依赖时间的只有：音频截到 t（`engine.py:191`，每次读重新编码，没有跨读缓存），说话人后验从 t − 1.04 s 起置零（`engine.py:205`），二者都只依赖 t_r。训练器同样截断音频、置零后验（`pretrain_thinker_gstar.py:125, 133–135`）。
- ctx：v8 协议为空（RL 用 `--ctx-max 0`）；带 `--context N` 时 ctx 是上一窗口的输出，在本窗口内固定，仍属于 c，不破坏精确性。

### 4.3 注意事项（不是前提失败，但实现时必须照做）

1. **最后一次读屏蔽 `<wait>`。** 解码时最后一次读把 `<wait>` 的 logit 设为 −1e9（`engine.py:246`），所以"v8 解码生成 (S,z) 的概率"在最后一次读上是去掉 `<wait>` 后重新归一的 softmax。`precheck_sched.py` 默认照此计算（`--no-final-bar-wait` 可关）；SFT 的交叉熵本身不做屏蔽。
2. **S 的串行化顺序是固定的。** 排序键是 (min(src_end + 1 s, dur), src_start)，即按块的源语结束时间排序，只有窗口末尾被截到 dur 的块改按开始时间排序。它由源语时间决定，与读的网格无关，所以任何调度都生成同一个 S；但格无法表示别的交错顺序（这正是 §10.3 的一′-b）。统计：在至少有 2 块的 17,879 个窗口中，S 顺序与"按源语结束时间排序"不同的有 1,150 个（全部来自末尾截断：未截断的块 commit = src_end + 1 s，必然排在截断的块之前），与"按开始时间排序"（离线目标的顺序）不同的有 5,874 个；有 3,090 个窗口（11%）在 S 里出现"某单元的两块之间插着别的单元的块"。
3. **v8 从未在块中间见过 `<wait>`。** v8 的标签只在块边界（结束时间 token 之后）或空读处放结束符，所以块中间的 end_lp 会非常低。"任意位置可结束"是精确模型；`precheck_sched.py` 同时分析"只允许在块边界结束"的版本。
4. **解码时的改写。** `max_new = 300`：一块写满 300 个 token 时该读没有结束符（gold S 最长 219 个 token，不受影响）；lag 策略的 `lag_cut` 和评测用的循环守卫/预算（`online_decode.py:173–179`，`loop_guard.py:27–39`）会截断一块。它们改变的是哪些 (S,z) 会出现，历史仍是已提交 id 的前缀、没有标记；但开了守卫的解码不再严格是从 p_θ 采样。
5. **时间 token 截顶。** 窗口最长 21.3 s，超过 20 s 的时间 token 截成 `<|t250|>`；112 个块的结束时间 token 被截顶，可行下界按 token 的取值计算。

## 5. 数据准备：`sched_v1`

**输出。** `eval_outputs/mist_w3po_v1/crf/sched_v1/shard_000..055.jsonl`（每行一个窗口）和 `summary.json`。用 32 个工作进程（每进程 1 个线程），总用时 17 s。每个窗口：
- `S_ids`（训练器同款分词器与特殊 token）、`S`、逐 token 的 `kind`（0 说话人、1 开始时间、2 文本、3 结束时间）、`chunk`、`unit`、`wstart`（开始一个词的文本 token，用于块的词数）；
- `t_ms`（读取时刻：每 1 s 一次，加窗口末尾，与 `engine.read_times` 相同）；
- `lo`、`hi`（闭区间读下标），`req_ms`，`tau_ms`；`lag[r,j] = (t_ms[r] − tau_ms[j]) / 1000` 以分解形式存储（`build_sched_windows.lag_matrix` 还原成 [R,N] 矩阵，结果精确）；
- `beta_heur`（启发式调度：每次读后已提交的 token 数）、`lat_heur`、`seg_bounds`（块边界的 token 偏移）；
- `chunks`（piece、源语起止、commit_ms 等）、`units`（a_ms、b_ms、槽位、与其他说话人话语的重叠秒数 `ov_s`）。

**界的定义**（D_max = 4 s）：
- 说话人 token 和开始时间 token：第一个 t_r ≥ 开始时间 token 取值（q·80 ms）的读；
- 文本 token：第一个 t_r ≥ a_u（单元开始）的读；
- 结束时间 token：第一个 t_r ≥ 其取值的读；
- hi（单元 u 的所有 token）：第一个 t_r ≥ b_u + 4 s 的读，没有则为最后一次读；没有满足下界的读时也取最后一次读。
- τ_j = 所在块的源语结束时间 src_end_ms（没有词对齐，块是最细的对齐单位；单元末块即 b_u）。Lat(z) = (1/N)·Σ_j (t_{r(j)} − τ_j)。

**统计。**

| 量 | 值 |
|---|---|
| 窗口 | 27,953（全部有 v8 行；S 为空的窗口 0 个） |
| 块 / 单元 / 重叠单元 | 95,488 / 69,540 / 33,853（48.7%） |
| N（token） | 平均 32.7，p10 5，p50 23，p90 75.8，p99 131，最大 219 |
| R（读） | 平均 6.76，p10 2，p50 5，p90 16，p99 20，最大 22；共 188,853 次读 |
| 启发式非空块长（token） | 72,427 块，平均 12.6，p10 5，p50 11，p90 23，p99 36，最大 77 |
| 启发式非空块含块数 | 平均 1.32，p50 1，p90 2，最大 10 |
| 空读占比 | 全部读 61.6%（116,426）；非最后一次读 72.4%；最后一次读从不为空 |
| **L_ref = 平均 Lat(heur)** | **0.826 s**（按窗口；p10 0.42，p50 0.82，p90 1.34，最大 1.77）；按 token 加权 1.086 s |
| β_heur 越出 lo/hi | **0 个 token，0 个窗口** |

与 v8 构建器自己的统计一致：非空读 72,427 = v8 读取行 130,560 − 保留的空读 58,133；空读 116,426 中约一半被保留。
L_ref 不到 1 s，是因为窗口末尾的块的提交被截到 dur（滞后 < 1 s），而窗口很短（时长中位数 4.3 s）。λ\* 的二分用的是"按窗口平均"，与这个 L_ref 口径一致；`solve_lambda` 也支持按 token 加权。

## 6. 格在真实形状上的 CPU 耗时（随机 log 概率）

- 在全部 27,953 个窗口上，用真实的 R、N、lo、hi、lag 和随机 log 概率运行：**全部可行**（log Z 有限），启发式调度在格里都是可行路径，`schedule_lat(β_heur)` 与 `lat_heur` 一致。W = 128 时同样全部可行（启发式块最长 77）。
- 单线程、单窗口：前向-后向平均 **2.4 ms**（p50 1.5，p99 10.8，最大 24.6 ms），Viterbi 平均 1.0 ms；W = 128 时前向-后向平均 2.9 ms。
- 批处理（单线程，1,280 个窗口，W = 128）：批大小 1 / 16 / 64 时分别为 3.0 / 4.0 / 4.6 ms/窗口。CPU 上填充反而更慢，批处理是给 GPU 用的。λ\* 二分（1,280 个窗口，批 64，单线程）23 次迭代、约 31 次求值，用时 182 s。
- 估算：全训练集一次前向-后向约 67 s 单线程；λ\* 二分约 30 次求值，单线程约 35 min，32 进程约 1–2 min，在 GPU 上批处理更快。相对于每次读一次 30B-A3B 的前向，格的开销可以忽略。

## 7. 预检脚本 `precheck_sched.py`（已写，未运行）

- **打分（GPU）。** 对 500 个训练窗口（固定种子从 sched_v1 抽样）的每次读，做一次教师强制前向：提示 = 音频到 t_r + ctx ""（v8 协议）+ 完整 S。模型加载、音频编码、提示构建都复用 `Engine.from_merged`、`Engine.audio_embeds`、`Prompter.ids`，批处理方式与 `Engine.read` 相同（左填充 + mask）。在 S^{<b} 之后的位置同时取 log p(S_b) 和 log p(e_r)，得到 tok_lp[r,:] 与 end_lp[r,:]。最后一次读按解码方式屏蔽 `<wait>`。开始前检查 50 个窗口的 S 分词与引擎分词器一致。每个窗口的 log 概率存成 `logprobs/<id>.npz`，可以用 `--analysis-only` 在 CPU 上重做分析。
- **分析（float64）。** λ\* 二分到 L_ref（默认用这 500 个窗口自己的平均 Lat(heur)；`--lref train` 用全训练集的 0.826 s）；λ\* 下的 MAP 和每窗口 16 个 FFBS 样本；同时报告 λ = 0 时的 E[Lat]（模型自己的偏好）。两种边界模式都分析："任意位置"（精确模型）和"只在块边界"。W = 128。
- **§10.5 的三个停止准则：**
  1. 提交读不同的单元（参考话语；块级也报告）占比 < 10% → 停止；
  2. 重叠单元的块的平均提交时间偏移（MAP；E_q 也报告）早于 −0.3 s → 停止（S0 风险）；
  3. ≤ 2 词块的占比（MAP 与 FFBS 样本）比启发式高出 5 个百分点以上 → 标记"加入 g"。5 个百分点是脚本自己定的阈值，计划里只写了"明显上升"。
  另外报告：≤ 1 s 单元的提交读差异、块长、空读占比、块中间结束的次数、MAP 与启发式的平均 Lat。
- **输出。** `report.json`（两种模式）、`per_window_{any,chunk}.jsonl`、`windows.json`、`logprobs/`。输出目录已存在时拒绝运行。
- **检查。** `py_compile` 通过。`--dry-run` 不加载模型、不写文件，打印计划，并用随机 log 概率在 20 个窗口上把分析流程跑通：

  | 计划项 | 值 |
  |---|---|
  | 窗口 / 读（前向） | 500 / 3,093 |
  | 前向批次（每批 4 次读） | 963 |
  | S token / 打分位置 | 14,753 / 155,498 |
  | 最大 N / 最大 R | 153 / 21 |
  | 每批 logits | 约 357 MB（fp32） |

- **资源估计。** 每次读一次约 300–600 token 的前向（无生成），3,093 次；v8 合并权重按 `--max-gib 40` 分到 2 张卡，约 0.5–1 GPU 小时。启动方式（以后，GPU 空出来时）：
  `setsid nohup python -m experiments.mist_w3po_v1.crf.precheck_sched --out eval_outputs/mist_w3po_v1/crf/precheck_v8_500 > logs/mist_w3po_v1/crf_precheck_v8_500.log 2>&1 &`

## 8. 设计说明：逐 token 权重（w_tok、w_end）接入 SFT 训练器（不改代码）

**现状。**
- 入口 `experiments/mist_w3po_v1/training/pretrain_thinker_gstar_s0.py`：`GStarRowsS0`（继承 `pretrain_thinker_gstar.GStarRows`）读行，`build_batch`（第 26–61 行）把每行变成 prompt id + 目标 id，标签只在目标上（第 42–44 行：`r_ids = tok(target) + [<eos> 或 <wait>]`）。
- 模型 `ThinkerGStarModel.forward`（`experiments/uniss_qwen3_omni_v1/training/thinker_gstar_model.py:173–184`）：只在有标签的位置算 logits，交叉熵乘层级权重（时间 token ×1.5，说话人 token ×2），再按 Σw 归一：`loss = Σ ce·w / Σ w`（第 183–184 行）。
- `forward_step` / `loss_func` 在 `pretrain_thinker_gstar.py:179–186`，s0 入口直接复用。

**一′ 的梯度需要什么。** 对窗口的每次读 r，∇J = Σ_j w_tok[r,j]·∇log p(S_j \| x≤t_r, S^{<j}) + Σ_b w_end[r,b]·∇log p(e_r \| x≤t_r, S^{<b})。两项都在同一条序列"提示(t_r) + 完整 S"上：位置 b 同时有两个目标——S_b（权重 w_tok[r,b]）和 e_r（权重 w_end[r,b]）。普通 SFT 每个位置只有一个标签，所以要加第二个"结束符标签"通道。

**加法式改动（新文件 + 新开关，默认行为逐位相同）：**
1. **行格式**（新的 kind，例如 `"kind": "sched"`）：`prefix = ""`，`target = S`（不接结束符），`end` 照旧表示 e_r 的类型，另带 `w_tok`（N 个数）和 `w_end`（N+1 个数）。离线版本（先用 v8 算一次 q，相当于精确 E 步的 EM 第一轮）可以直接由 `precheck_sched.py` 的打分结果扩展到全训练集后写出。
2. **新入口** `experiments/mist_w3po_v1/training/pretrain_thinker_gstar_sched.py`（不改 s0 入口）：
   - `GStarRowsSched(GStarRowsS0)` 把 `w_tok`、`w_end` 带进 item；
   - `build_batch_sched` 先调用原来的 s0 `build_batch` 逻辑；对 sched 行不追加结束符，另外输出 `tok_w`（[B, L]，标签位置上的 w_tok，其他行为 1）、`end_w`（[B, L]，位置 b 上的 w_end[r,b]，其他为 0）、`end_id`（[B]，`<wait>` 或 `<eos>`）。批里没有 sched 行时这三个张量都不生成，batch 与 v8 完全相同。
3. **模型**：在新文件里子类化 `ThinkerGStarModel`，`forward(..., tok_w=None, end_w=None, end_id=None)`；三者都为 None 时直接调用父类 `forward`（逐位相同）。否则：
   - 标签位置的权重改为 层级权重 × tok_w（再加上状态损失项，见下）；
   - 结束符项：在 end_w > 0 的位置取 logits（同一个 `logits_at`），加上 Σ end_w·(−log_softmax[end_id])；
   - 归一化：分母用 Σ(层级权重 × tok_w) + Σ end_w。对一个窗口而言 Σ_{r,j} w_tok = N、Σ_{r,b} w_end = R，与硬标签时的 token 总数相同，所以损失尺度与 v8 的 SFT 一致；把权重换成 β_heur 的 0/1 指示时，退化为 v8 的 SFT（只差 v8 随机丢掉一半空读；可作为单元测试）。
4. **状态损失** L_state（§10.3，权重 1，预登记）：对所有可行的 (r,j)（lo[j] ≤ r ≤ hi[j]）再加均匀权重 1/\|A\|（按窗口），直接加到 tok_w 上，不需要新通道。
5. **开关**：`--gstar-sched-weights {off,offline,online}`，默认 `off`；`model_provider` 只有在开关打开时才构造子类。

**在线精确版（每步用当前 θ 重新算 q）。** 权重是对当前模型 log 概率的无梯度函数，所以：
- 一个样本 = 一个窗口的全部 R 次读（R 最大 22）；`forward_step` 里先对 R 条序列做一次无梯度前向，取出 tok_lp/end_lp，在 GPU 上用 `schedule_lattice.marginals`（float64）得到权重，再对其中若干次读做带梯度的加权前向。
- 也可以对全部 R 次读一次带梯度前向，log 概率 detach 后算权重，再用同一图求加权损失（只需一次前向，但显存按 R 条序列计）。
- λ\* 用训练集上离线二分的值固定（等延迟约束）；如果在线更新导致 E_q[Lat] 漂移，每 N 步在一小批窗口上重新二分。
- 读的子采样：只对一部分读求梯度时，乘以 R/采样数，估计仍无偏；无梯度前向必须覆盖全部读，因为前向-后向需要所有 ψ_r。

## 9. GPU 预检的阻塞项

- 代码层面没有阻塞：数据就绪（`sched_v1`），格经过穷举、自动求导和 HMT 预言机验证，预检脚本 dry-run 通过。
- 打分部分（`score_window`）用到的引擎接口与 `Engine.read` 相同，但因为不能用 GPU，这一段**没有实际执行过**。建议上卡时先用 `--n 5` 跑一次小样（输出到新目录），确认 log 概率合理（例如启发式块的 tok_lp 应明显高于随机）后再跑 500 个窗口。
- 需要约 2 张卡（v8 合并权重，`--max-gib 40`）、约 0.5–1 小时。8 张卡目前都在同事的训练上，需要等卡。
