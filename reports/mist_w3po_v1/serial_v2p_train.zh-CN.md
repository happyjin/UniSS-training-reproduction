# SerialCRF v2′ 正式训练：预登记（2026-11-10）

> 设计：`crf_novel_ideas.zh-CN.md` §13.2–§13.3（模型、端到端学习）、§13.10（v2）、§13.11（v2′）。预检：`serial_crf_v2p_precheck.zh-CN.md`（按预登记规则 GO：S3′ 通过，S1′ 25.9%，S2′a 0.066，S2′b 0.274）。
> 用户决定：`LOG.zh-CN.md` 2026-11-10 04:55 UTC。预检 GO、代码通过测试和 GPU 冒烟、C0 / C0-fp 的 dev 评测完成后，直接正式训练，不再另行确认。
> **本节（§0–§8）在正式训练开始前提交。** §9 的结果在冒烟和启动后补写。

## 0. 一句话

在 v2′ 的格上做在线确定性退火 EM（DAEM）。每一步：
1. 用当前模型对格的全部边打分，包括有真选择的读的两种读内顺序；
2. 精确算出边-顺序后验，联合退火，τ = 0.5；
3. 对所有不同的（上下文，有序写集 ⊕ e_r）序列，按后验边际加权做交叉熵。

其余全部与 C0-fp 相同：数据流、步数、学习率、初始化、可训练参数、检查点格式。单一设置，不扫参。

## 1. 训练目标（`train_crf_sft.py --objective serial_v2p`，代码 `crf/serial_v2p_train.py`）

**格。** 直接调用预检自己的函数，未改动：`precheck_serial_v2.lattice`、`precheck_serial_v2p.plan_window_p`、`assemble_p`、`edge_terms`、`forward_backward`。训练优化的正是预检打过分的那个格：
- 条目是块，可松弛 1 读，最终读的条目不能推迟；
- 状态是集合 D_r，历史是 κ，读内按 kap 序写；
- 顺序隐变量 o_r ∈ {canon, newfirst}。只有待补写集合和 N_r 都非空时才是真选择。

**pass 1（无梯度，`model.eval()`）。**
- 对每个非强制读的每个前向请求（canon 覆盖前向和 newfirst 前向），读出 m 个强制 token 的 log p，以及 m + 1 个位置上的 log p(e_r)。
- 最终读在 softmax 前屏蔽 `<wait>`，与 `--final-bar-wait 1`、`Engine.read`、预检的 `score_reqs_v2` 相同。
- 输入走训练器自己的行：v8 行布局；扰动与 C0 同一抽样，由 (seed, step, window, read) 决定；最终读的 P 规则见 §3 第 1 条。
- 强制读只有一个上下文和一个写集，给所有路径加同一常数，不打分（ψ := 0），后验不变（同预检）。

**后验。**
- 联合退火：q_τ(ρ, o) ∝ exp(τ Σ_r ψ_{r,o_r})，τ = 0.5（预登记，同预检 `TAU`）。
- 在集合状态上做前向-后向，每条边做 logsumexp_o。
- 边-顺序边际：q(r, i→j, o) = exp(α_{r−1}(i) + τ ψ_{r,o}(i,j) + β_r(j) − log Z)，CPU float64。

**pass 2（有梯度，`model.train()`）。** EM / DAEM 的 M 步代理目标，q 不回传梯度：

L = −(1/D) Σ_{r,i,j,o} q(r, i→j, o) · [ Σ_t h_t log p(W_{r,o}(i,j)_t | x, κ(A_{r−1}), W_{<t}) + log p(e_r | …) ]

- 每一读都用普通 softmax 的交叉熵，与 C0 / v8 相同。最终读也一样：屏蔽只用在 pass 1 的格分数上，同 sched 训练器。
- h 是 v8 的分层权重：时间 token ×1.5，说话人 token ×2。它是 token id 的函数。
- D 与 C0 完全相同：本步可行窗口的 Σ(Σ_j h_j + R)。每条路径把每个条目恰好写一次、每读恰好结束一次，所以每个窗口的期望总权重等于 C0 的（测试 5）。
- 共享前缀的序列合并成一行，各位置系数相加：
  - c_end[b] = 在边界 b 结束的序列的 q 之和；
  - c_tok[t] = h_t · Σ_{b>t} c_end[b]。
- 每个计划请求都是一行，q 下溢为 0 的也保留，所以精确，且行结构只取决于计划。
- 标签路径（q = 1）上的行就是 C0 的行（测试 3，逐位相同）。

**规模**（`--check-all`，CPU，3000 步实际用到的全部 27,953 个训练窗口）：
- 标签路径的行与 C0 的行逐 token、逐位置相同：27,953 / 27,953；
- 6,801 个窗口含真顺序选择；
- 每步 pass 1 平均 61.5 行（中位数 57，p95 110，最大 274），pass 2 平均 100.4 行（中位数 95，p95 144，最大 313）。C0 每步 64 行。

**记录**（`metrics.jsonl`，只有 serial_v2p 有）：
- `sft_loss`：同一步、同一批窗口在当前参数下的 C0 损失，取自 pass 2 前向，可与 C0-fp 逐步配对比较（测试 7）；
- `q_label`、`exp_nf`（期望 newfirst 读数）、`exp_defer`、`n_real`；
- 两个 pass 的行数、token 数、micro-batch 数；
- 各段时间。

## 2. 配置（与 C0 / C0-fp 对齐）

| 项 | v2′ | C0-fp（配对对照） | C0 |
|---|---|---|---|
| 训练目标 | **serial_v2p**（τ = 0.5） | heuristic（标签 SFT） | heuristic |
| 数据 | sched_v1（AMI train），`plan_steps(seed 20261109)`，每步窗口相同 | 同 | 同 |
| 步数 / 每步行数 | 3000 / ~64 读行（按窗口计；v2′ 每个窗口展开成格的序列） | 同 | 同 |
| 学习率 | 1e-4，warm-up 50，余弦到 1e-5；AdamW (0.9, 0.95)，eps 1e-8，wd 0，梯度裁剪 1.0 | 同 | 同 |
| 初始化 / 可训练参数 | v8 merged + LoRA r32 / α64（q/k/v/o，48 层，seed 同）+ 投影器 + 259 个新 token 行 | 同 | 同 |
| 扰动 | p 0.2，同一抽样 | 同 | 同 |
| 最终读 P | `--final-read-full-post 1`（§3 第 1 条） | 1 | 0 |
| 检查点 | `ckpt/step_NNNNN`（lora.pt / extra.pt / optim.pt / DONE），每 500 步；`decode_crf_ckpt(_finalpost).py` 不改即可用 | 同 | 同 |
| 卡数 | 8（只影响打包，§4） | 4 | 4 |

## 3. 设计文档没写死的细节：选与预检一致的最小选项

1. **最终读 P 用 `--final-read-full-post 1`。**
   - 预检给训练窗口打分时，最终读的 P 完整可见（`precheck_serial_v2.final_read_post(split="train")`），所以选 1。
   - 训练器已有的规则是 t // 80，即与 C0-fp 相同，也就是 dev 预检所用的 `final_cut_ms`（cut = t）规则。它屏蔽跨过窗口末端、不足 80 ms 的最后一帧。预检的 train 打分把这一帧也给了模型（93% 的训练窗口有这一帧）。
   - 选训练器现有规则的理由：不新增代码路径；标签路径上与 C0-fp 逐位一致，配对比较只差训练目标；与 dev 预检（主要证据）相同。
   - 因此**配对对照是 C0-fp**，C0 也报告。
   - 任务说明里"训练窗口用与 C0 相同的最终读规则"，这里理解为：最终读不可推迟，以 `<eos>` 结束，pass 1 屏蔽 `<wait>`，损失用普通交叉熵，与 C0 相同。P 的可见范围按预检，即 C0-fp 的设置。**若用户要求与 C0 完全相同（fp = 0），需要另起一次运行。**
2. **pass 1 用训练行（含 p 0.2 扰动、右侧填充的 v8 行布局），不用预检的引擎路径（无扰动、左侧填充）。** 这样后验和梯度落在同一组输入上，是严格的 EM。数值上两条路径只差 bf16 误差，预检 §2.4 的对齐是同一量级。
3. **强制读**：不打分，按权重 1 训练。后验不变，是精确的。
4. **q 极小的序列也保留**，不剪枝，所以是精确的 E 步。
5. **τ 只用 0.5**，联合退火（预检的定义）。不跑 τ = 1、η 消融等。
6. **`sft_loss` 只用于记录**，不进入损失。
7. **log-mel 逐行计算（`--feat-per-row 1`）。**
   - C0 每个 micro-batch 调一次特征提取器，所有波形补零到最长，较短行的最后一帧随 batch 组成变化：在微型模型上改变打包会使梯度变化 7×10⁻⁴（测试 4）。
   - 逐行提取后，每行的输入与打包无关，同一读的特征在一步内缓存复用。
   - 与 C0 的差别只在"同一 micro-batch 里较短行的最后一帧"。C0 换成 8 卡打包后，这一帧本来也会变。

## 4. 打包与 GPU 利用（只改速度，不改全局 batch 和数学）

C0 实测（2026-11-10 05:33 UTC，C0 / C0-fp 各 4 卡，60 s 采样）：
- 利用率 44–54%，功率 220–290 W（上限 700 W）；
- 每卡 ~70 GB，3.4–3.9 s/步，每 rank 每步 1 个 micro-batch，约 5k 填充 token。

瓶颈是 transformers 的 MoE：torch 2.6 没有 grouped_mm，走 `grouped_mm_fallback`，每层每个专家一次 `torch.mm`，加上 `offs.tolist()` 同步。这类开销主要受 kernel 启动次数限制，不受显存限制。

| 开关 | 作用 | 等价性 |
|---|---|---|
| 8 卡，行级拆分 | pass-1 请求、pass-2 行各自按估计长度做 LPT 分配到 rank（窗口内的行也拆开），比按窗口分配更均衡；pass-1 结果 all-gather 后，每个 rank 计算全部窗口的后验（输入相同，确定性） | 测试 4：1 / 3 / 8 个模拟 rank，后验相同，梯度相对差 2.9×10⁻⁶（fp32） |
| `--tok-budget 32000`，`--mem-cap-gib 120` | 大多数步每 rank 每个 pass 1 个 micro-batch，大步也只要 2 个；分配器上限 120 GiB，加上 context 约 124 GiB，单卡合计 ≤ 135 GiB，留约 11 GiB 余量 | 梯度累积，同上 |
| `--fast-moe-bwd 1` | fallback 的反向对冻结的专家权重也会 `zeros_like(weight)`（每层 1.2 GB），并且每个专家多做一次 mm，结果随即丢弃。改为只在权重需要梯度时计算，offs 只同步一次 | 测试 6：前向和输入梯度逐位相同；对可训练权重，梯度也相同 |
| `--prefetch 1` | pass 2 的 micro-batch 在 pass 1 跑 GPU 时，在后台线程里整理好（读音频、扰动、特征、token） | 测试 7：与 `--prefetch 0` 逐位相同 |
| `--feat-per-row 1` | 见 §3 第 7 条；同一读的特征在 pass 1 / pass 2 之间缓存 | 测试 3、4 |
| `--empty-cache 0` | 不再每步清空 CUDA 缓存，免得每步重新分配 | 无数值影响 |

冒烟里会对比两种配置（§6）：A 用 C0 的打包参数（12000 / 88 GiB / 无 fast-moe / 无 prefetch / 每步 empty_cache），B 用上表的配置。两者都按 1 s 采样记录 `utilization.gpu`、`power.draw`、步时和 pass-1 / pass-2 拆分。

## 5. 测试（CPU，`__main__` 运行器；pytest 未安装）

`PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_serial_v2p_train`：**ALL 8 PASSED（23 s）**

| # | 内容 | 结果 |
|---|---|---|
| 1a | 哑 LM，500 个真实窗口（115 个含真顺序选择）：训练器后验 = 预检 `forward_p`（log Z、状态 / 推迟 / 顺序边际）；140 个窗口 = (ρ × 顺序) 暴力枚举，含每条序列的权重 | 最大 \|Δ\| 4.7×10⁻¹⁴ |
| 1b | 微型模型，3 个含真选择的窗口：训练器 pass 1 的后验 = 预检 `forward_p`（1e-9），= 暴力枚举（每条序列单独打分） | 2.2×10⁻⁶ |
| 2 | pass-2 梯度 = 在暴力枚举序列上对 EM 代理目标求 autograd（τ 0.5、屏蔽、h、C0 的 D）；τ 1、不屏蔽、h = 1、D = 1 时 = −∇log Z（Fisher 恒等式） | 相对 1.4×10⁻⁶ / 1.8×10⁻⁶ |
| 3 | 格退化为标签路径（把状态打补丁成只有标签，两种特征模式；以及 2 个天然只有一条路径的真实窗口）：行 = C0 的行，损失和梯度与 `--objective heuristic` 逐位相同 | 逐位相同 |
| 4 | 不同 micro-batch 预算 / rank 拆分：后验权重、梯度相同 | 0 / 2.9×10⁻⁶ |
| 5 | 每个窗口所有系数之和 × D = Σh + R（与 C0 相同），标签系数同样；每读边-顺序质量和为 1 | 500 个窗口 |
| 6 | `--fast-moe-bwd` 与 transformers 的 fallback 逐位相同 | 逐位相同 |
| 7 | 步路径（run_step → grad_pass → 日志）：`--prefetch 1` = `--prefetch 0` 逐位相同；记录的 sft_loss = C0 损失 | 通过 |

原有测试仍全部通过：
- `test_train_crf_sft`：8 个；
- `test_final_read_full_post`：4 个；
- `test_precheck_serial_v2p`：10 个；
- `test_decode_crf_ckpt_finalpost`：通过。

默认参数下 heuristic / sched 的行为不变：新字段默认 None，新开关默认关；测试 (c) 的逐位一致检查仍然通过。

## 6. 冒烟与启动（`scripts/start_v2p_after_c0eval.sh`，只用 `bash -n` / `--dry` 检查）

等待以下事件（每 60 s 查一次，单进程）：
1. C0 / C0-fp 评测的编排进程启动并退出。进程是 `crf_pair_dev_orch.py --run <dir>`，从 ps 字段取 PID 和运行目录；
2. `<run>/events/DONE` 出现。若进程退出时没有 DONE，再等 20 分钟看是否重启；
3. 卡上除占卡程序外，没有其他 GPU 进程。

然后：
1. 按 PID 停掉占卡程序：先停 supervisor（它会结束自己的 worker 进程组），再停残留的树内进程和孤儿 worker；
2. 持 `gpu_launch.lock`，确认每卡合计 + 126,976 MiB ≤ 138,240 MiB；
3. 冒烟 A（C0 打包参数，3 步）和冒烟 B（正式参数，5 步），都是同一 3000 步计划的第 1–3 / 1–5 步，各写到新目录；
4. 闸门：两次都正常结束，损失有限，B 的 `mem_gib_max` ≤ 120，采样到的单卡合计 ≤ 138,240 MiB；
5. 正式运行：8 卡，setsid，在锁内启动，看到每卡显存可见后才释放锁；
6. 看门程序：训练进程退出（正常结束或崩溃）后，在没有 supervisor 的卡上重启占卡程序。

闸门不过或启动失败时，也重启占卡程序，卡不会空着。

## 7. 评测计划（预登记；与 C0 / C0-fp 的评测协议相同）

- **数据**：只用 AMI dev，`windows_dev_online`。RealSI、ACL 60/60、AliMeeting 不碰；不用 COMET / MetricX。
- **检查点**：只评 step_03000，与 C0 / C0-fp 相同，不在检查点之间挑选。
- **解码**：`decode_crf_ckpt_finalpost --final-read-post close_detect`，与参考配置完全相同（v8 merged、policy wait、lag 2000、context 0、batch 16、loop guard、8 分片），然后 `score_run`、检查 A（`diag_order_attr --only A`），语音两个 TTS 种子 → ASR-BLEU。机制与 `crf_pair_dev_orch.py` 相同，只换候选。
- **对照**：C0-fp（配对，主要），C0，v8 + 最终读因果 P。
- **指标**：
  - 文本 all / ov tcpBLEU；
  - ASR-BLEU（两个种子）；
  - ≤1 s、≤0.5 s ov50 召回；
  - 用 S1′ 代码路径在新解码上重算 CHAIN 数和占比；
  - 检查 A 的 ORDER_OMISSION；
  - 重复 / collapse 率；
  - ov 延迟。
  - 配对检验与 C0 对评测相同。
- **判定**（v2′ 对 C0-fp）：
  - 机制目标：≤1 s ov50 召回上升，CHAIN 数下降；
  - 硬闸门：ASR-BLEU 不低于 C0-fp（两个种子），文本 all tcpBLEU 不出现显著下降（P < 0.05）；
  - 都满足才算有效。否则如实报告为负结果，不在 dev 上调参重试。

## 8. 训练中的监控（不是停止规则，不改超参）

- 每步看 `sft_loss`（同窗口、同参数的 C0 损失），与 C0-fp 同一步的 loss 配对比较；同时看 `q_label`、`exp_nf`、`gnorm`、`mem_gib_max`。
- 出现以下情况只报告用户，不自行改设置：loss 非有限；最后 500 步 `sft_loss` 均值比 C0-fp 高 0.05 nats 以上；单卡显存超过 135 GiB。

## 9. 结果（冒烟与启动后补写）

### 9.1 早期 2 卡冒烟（05:42–05:44 UTC，卡 6、7；评测编排启动前的空档，持锁，符合房间规则）

- 配置：正式参数（TOK 32000、CAP 120、fast-moe、prefetch、feat-per-row），2 个 rank，计划的第 1–3 步。
- 目录：`eval_outputs/mist_w3po_v1/crf/train/crf_v2p_smoke2c_20261110T054247Z`；日志：`logs/mist_w3po_v1/crf_v2p_smoke2c_20261110T054247Z.log`。
- 结果：正常结束（`CRF_SFT_TRAIN_DONE`，rc = 0），两个 rank 参数校验和一致。

| 步 | v2′ loss（EM 代理） | v2′ sft_loss（同窗口 C0 损失） | C0-fp 同步 loss | D（v2′ / C0-fp） | gnorm（v2′ / C0-fp） | pass 1 / pass 2 行 | q_label | E[newfirst] | 步时（pass 1 + pass 2） |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1041 | 1.0882 | 1.0878 | 479 / 479 | 2.95 / 2.98 | 87 / 137 | 0.84 | 0.11 | 10.4 s（3.3 + 7.9，含预热） |
| 2 | 1.0167 | 1.0169 | 1.0056 | 492 / 492 | 3.83 / 4.10 | 75 / 108 | 0.68 | 0.88 | 6.0 s（1.2 + 4.8） |
| 3 | 0.9343 | 0.9250 | 0.9250 | 399 / 399 | 3.37 / 3.44 | 39 / 79 | 0.78 | 0.16 | 5.6 s（0.9 + 4.7） |

- 第 1 步参数与 C0-fp 相同：v2′ 的 sft_loss 1.0882 对 C0-fp 的 1.0878，只差逐行特征和 bf16 打包噪声。D 逐步相同。
- 显存：`max_memory_allocated` 80.1 GiB，卡合计峰值 85.3 GB，远低于 135 GiB。

8 卡的冒烟 A / B 和正式启动由启动器在评测结束后执行（§6）。

### 9.2 8 卡冒烟 A / B 与正式启动（启动器 `logs/mist_w3po_v1/v2p_at_20261110T054434Z/`）

- 评测编排进程 3867716 在 07:09 退出，没有写 DONE；07:10 重启为 477687。07:44:47 出现 `events/DONE`，卡上没有评测 GPU 进程。
- 07:48 按 PID 停掉占卡程序，8 卡全空。随后持锁跑冒烟 A、B，闸门通过。

| | C0（4 卡，05:33 采样） | 冒烟 A：v2′ + C0 打包参数 | 冒烟 B / 正式：v2′ + 新打包参数 |
|---|---|---|---|
| 步时（第 2 步起均值） | 3.4–3.9 s | 3.34 s（pass 1 0.60 + pass 2 2.73） | 冒烟 2.79 s（0.59 + 2.16）；正式第 6–85 步 2.94 s（0.59 + 2.33） |
| 利用率 / 功率 | 44–54% / 220–290 W | 54% / 259 W | 冒烟 48% / 276 W；正式 90 s 采样 47% / 274 W |
| 显存 | ~70 GB | 65.1 GiB 分配，卡峰值 71.9 GB | 65–66.5 GiB 分配，卡峰值 73.5 GB（≤ 135 GiB） |

- 同样的工作量，v2′ 每步前向约为 C0 的 2.5 倍（pass 1 约 61 行 + pass 2 约 100 行，C0 为 64 行）。8 卡、行级拆分、fast-moe-bwd、prefetch 之后，步时仍低于 C0。prefetch 后 pass 2 的整理时间约为 0。
- **利用率没有明显提高（约 50%）。** 瓶颈是 torch 2.6 下 transformers MoE 的逐专家 `torch.mm` 循环和 `offs.tolist()` 同步，受 kernel 启动次数限制。加大 micro-batch 不起作用：每 rank 每个 pass 本来就只有 1 个 micro-batch。要真正满载，需要分组 GEMM 内核（torch ≥ 2.8 的 `_grouped_mm` 或自写 Triton 内核）。这会改变数值实现，超出"只改打包"的范围，本次没有做。
- 确定性：正式第 1 步 loss 1.1003，与冒烟 B 第 1 步相同。
- 正式运行：`crf_v2p_20261110T075131Z`，07:51:31 UTC 启动，root pid 1013964。日志 `logs/mist_w3po_v1/crf_v2p_20261110T075131Z.log`，输出 `eval_outputs/mist_w3po_v1/crf/train/crf_v2p_20261110T075131Z`。
  - 前 85 步：loss 有限，gnorm 3–5，q_label 约 0.4–0.9，显存稳定。
  - 第 6–85 步 sft_loss 均值 1.010，C0-fp 同步为 0.997。
  - 预计 3000 步约 2.5 h，约 10:20 UTC 结束。
  - 启动器（pid 3880632）在训练进程退出后会重启 8 卡占卡程序。
