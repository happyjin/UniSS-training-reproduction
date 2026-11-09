# 方案一′ 可训练代码与 GPU 冒烟测试（2026-11-09）

> 对应 `reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md` §10.2–§10.5、§11；CPU 阶段见 `sched_crf_cpu_stage.zh-CN.md`。
> 本阶段只在卡 0–3 上做了几分钟的冒烟测试，**没有启动任何正式运行**（500 窗口预检、C0、一′ 都等用户命令）。
> 训练只用 AMI 训练窗口（`sched_v1`），解码冒烟只用 dev 的 2 个窗口；没有碰 RealSI、ACL 60/60、AliMeeting。

## 0. 结论

| 项 | 结果 |
|---|---|
| 训练器 | `experiments/mist_w3po_v1/crf/train_crf_sft.py`：`--objective heuristic`（C0）与 `--objective sched`（一′），HOOK API 已固定 |
| 测试 | `crf/test_train_crf_sft.py`：8 项全部通过（CPU，约 12 s） |
| C0 = v8 SFT | 在同样的读上，C0 的损失与 v8 自己的 `ThinkerGStarModel.forward` 相同（单个 micro-batch：相对误差 0，梯度 5e−7） |
| Fisher 恒等式 | 一′ 的梯度 = 对 −log Z_λ 经动态规划自动求导（相对误差 3.4e−7） |
| 不挂钩子时 | 批张量与 v8 `build_batch` 逐位相同，嵌入与隐状态 `torch.equal` |
| GPU 冒烟 | (i) 预检 `--n 5`、(ii) 两种目标各 3 步、(iii) 用现有解码引擎解码 2 个 dev 窗口、(iv) 2 进程续训 2 步，全部成功；另测了 4 进程计时 |
| C0 第 1 步损失 | **1.101**；v8 训练最后 500 步的损失均值 1.011（0.84–1.16） |
| 一′（4 卡） | λ 每步在 −13.8 到 +13.1 之间，E_q[Lat] 每步都等于 0.8256 s；H(q) 0.57–0.72，q(启发式) 0.57–0.75 |
| 速度与显存（4 卡，每步约 64 行） | C0 每步 4–6 s，一′ 每步 6–9 s；每卡最大分配 66.3 GiB。3000 步约 C0 3.5–5 h、一′ 5–7.5 h |
| 冒烟中发现并已修正 | 最后一次读：引擎在**每个位置**都屏蔽 `<wait>`，起初只屏蔽了结束符；已改为 token 与结束符都屏蔽，与 `precheck_sched.py` 一致 |
| 主要风险 | 见 §7：每批 λ 的噪声；L_state 约为主损失的 2 倍；层级权重使更新成为 EM 变体；v8 的 bf16 MoE 打分随批组成抖动；C0 的数据与 v8 不同 |

## 1. 文件与提交

| 文件 | 内容 |
|---|---|
| `crf/train_crf_sft.py` | 训练器（两种目标、HOOK API、存档/续训） |
| `crf/test_train_crf_sft.py` | 8 项 CPU 测试，`__main__` 直接运行 |
| `crf/decode_crf_ckpt.py` | `evaluation/online_decode.py` 的包装：在内存里把存档应用到 v8 上再解码（不写合并权重） |
| `crf/export_merged.py` | 把存档写成合并 Thinker 目录（v8 `thinker_merged` 的格式），供 `eval_pipeline.sh` 等现有工具直接使用 |
| `scripts/run_crf_c0.sh`、`scripts/run_crf_sched.sh` | 正式运行的启动脚本（4 卡，setsid nohup，拒绝覆盖）；**只做了 `bash -n`，没有运行** |

提交（均已推送到 `private main`）：
- `366e9f0d`：训练器与 HOOK API、CPU 测试（在 GPU 冒烟之前提交，供方案二对接）；
- `8d289426`：最后一次读的屏蔽修正、rank 校验和、`--stop-after`、解码包装、导出、两个启动脚本；
- 本报告单独提交。

没有修改任何已有文件。所有已有代码（引擎、v8 模型与损失、格、预检、`online_decode`）都是 import 复用。

## 2. 训练器

### 2.1 模型（即 v8 的 SFT 模型）

- v8 合并权重（bf16，冻结），加一个新的 LoRA：r 32 / α 64，48 个文本层的 q/k/v/o，共 192 个适配器。b 初始为 0，所以第 0 步就是 v8。
- 说话人投影器 `sd_proj`：fp32，从 v8 的 `sd_proj.pt` 初始化。
- 259 个新 token 行 `new_emb` / `new_head`：fp32，从 v8 合并权重的对应行初始化。
- 可训练参数共 34.14 M，与 `ThinkerGStarModel(init_merged=True)` 相同。
- `ThinkerGStarModel.embed` 和 `.logits_at` 以非绑定方式在本模块上调用，所以前向就是 v8 的 SFT 代码。加载方式同 `rl/train_w3po.py`：每卡一份完整 Thinker，talker 放 CPU 后丢弃。

### 2.2 数据与行

- `sched_v1` 的 27,953 个窗口，共 188,853 次读。每个窗口的 R 次读各成一行。
- 第 r 行 = v8 提示（音频截到 t_r；后验从 max(0, t_r − 1040)//80 起置零；v8 的 `--gstar-perturb 0.2` 扰动，每行独立种子；ctx 为空）+ S 的前缀 + e_r。
- 在 S^{<b} 之后的位置同时读出 tok_lp[r,b] = log p(S_b) 和 end_lp[r,b] = log p(e_r)。
- 行在最后一个需要的位置之后截断，并以 e_r 作为最后一个输入，与 v8 行的布局完全一致。在因果注意力下，这两点都不改变读出的值。
  - C0 的行恰好就是 v8 的行：前缀 S[:β_{r−1}]，目标块，结束符。
  - 一′ 第一遍的行截到 b_max(r)，即第一个 lo > r 的 token；之后的状态不可达。

### 2.3 两个目标

- **C0（`heuristic`）**：β 的 one-hot 权重，即只在已提交块和它的结束符上算损失。
- **一′（`sched`）**：
  1. **第一遍（无梯度）**：对本步全部窗口的全部读算 tok_lp / end_lp。
     - 最后一次读在 softmax 之前屏蔽 `<wait>`，而且**每个位置都屏蔽**（token 和 `<eos>` 都在去掉 `<wait>` 的分布下重新归一），与解码时 `engine.read` 的做法以及 `precheck_sched.py` 的打分相同。所以 q 是 v8 自己解码的精确后验。
     - 起初只屏蔽了结束符。这只会让 log Z 平移一个常数；但 token 也屏蔽之后，q 本身会变（测试中最多变 0.50）。已修正并补了测试。
  2. **λ**：rank 0 收集全局批，用 `schedule_lattice.solve_lambda` 二分，使批内各窗口 E_q[Lat] 的均值等于 L_ref = 0.8256 s。没有括住解时沿用上一步的 λ（冒烟中每步都括住了）。
  3. **权重**：`marginals` 给出 w_tok、w_end，由 rank 0 广播。
  4. **第二遍（有梯度）**：
     loss = −(Σ h_j·w_tok·tok_lp + Σ w_end·end_lp) / D + L_state，
     L_state = 可行 (r,j) 上 −tok_lp 的均值，在全局批内合并计算，权重 1（预登记）。
- **𝒜（L_state 的集合）**：至少有一条可行调度在第 r 次读提交 token j 的 (r,j)。用"所有得分为 0 的格"的边际 > 0 精确算出，包括 W 上限。
  - 在 lo/hi 允许的对中，有 5.9% 不可达：它们的前缀里含有尚未听到的译文。

### 2.4 归一化与层级权重

- D = 全局批内 Σ(Σ_j h_j + R)。h_j 是 v8 的层级权重：时间 token ×1.5，说话人 token ×2，其余 ×1。
- 两个目标的 D 相同，而且不只是期望相同，是逐窗口精确相同：每个 token 在各读上的 w_tok 之和为 1，每次读在各位置上的 w_end 之和为 1。所以 Σ w_tok = N、Σ w_end = R（测试 (d)）。
- 每个 micro-batch 的损失已经除以全局 D，梯度在 micro-batch 和 rank 之间求和。
- **最后一次读的损失**：两个目标都保持 v8 的普通交叉熵（不屏蔽），这样两臂只在权重上不同，C0 也严格等于 v8 SFT。
- **两处与精确 ∇J 的差别**（都在两臂之间共享）：
  - 层级权重：h ≠ 1 时，一′ 的更新是一个 EM 步——E 步用精确的模型后验，M 步用 v8 的加权对数似然。
  - 最后一次读：屏蔽开着时，更新在解码模型的 ∇J 之上，额外按 q 加权压低最后一次读各位置上的 p(`<wait>`)，这正是 v8 SFT 本来就在做的事。
  - 只有 `--hier-weights 0 --final-bar-wait 0` 时，更新恰好等于 ∇J（测试 (b)）。

### 2.5 日志（`metrics.jsonl`，每步一行）

- 损失：`loss`、`loss_main`、`loss_state`、`loss_extra`，以及 `sft_loss`（一′ 模式下用第一遍的 log 概率算出的 C0 损失，可与 C0 直接比较）。
- 后验：`lam`、`lam_bracketed`、`Eq_lat`、`lat_target`、`lat_heur`（本批启发式的平均滞后）、`H_q`（后验熵的窗口均值）、`q_heur`（启发式调度所占后验质量的窗口均值）。
- 其他：`gnorm`、行数与 token 数、`mem_gib_max`，以及 `t_pass1`、`t_lambda`、`t_pass2`、`t_step`。
- 每次存档时，各 rank 对可训练参数求校验和并比较，打印 `RANKS_IN_SYNC` 或 `RANK_DIVERGENCE`。

### 2.6 与 v8 配方的对齐

v8 的配方来自 `thinker_gstar_s2dl_zh2en_3k_20261030`：`run_v5_train.sh` → `run_thinker_gstar.sh`，Megatron。

| 项 | v8 | 本训练器 |
|---|---|---|
| 起点 | O2 合并权重 + 新 LoRA | v8 合并权重 + 新 LoRA（b = 0） |
| 可训练参数 | LoRA r32/α64 q/k/v/o ×48 层、sd_proj、259 个新 token 行 | 相同（34.14 M） |
| 优化器 | Adam β (0.9, 0.95)，eps 1e−8，wd 0 | AdamW，参数相同（wd 0 时与 Adam 相同） |
| 学习率 | 1e−4，50 步线性预热，cosine 降到 1e−5（3000 步） | 相同；测试核对了 v8 日志里第 1、10、20、2980、2990、3000 步的 lr |
| 梯度裁剪 | 全局范数 1.0 | 相同 |
| 精度 | bf16（Float16Module：可训练参数也以 bf16 计算，fp32 主副本） | 底座 bf16；可训练参数以 fp32 存储和计算，同 `train_w3po` 的 LoRA。差别只是舍入 |
| 后验扰动 | 0.2 | 0.2 |
| 损失 | 交叉熵，时间 ×1.5、说话人 ×2，按 Σw 归一 | 相同；C0 与 v8 的 forward 数值一致 |
| **全局批（读取行）** | 每步 64 行（8 卡 × MBS 8），其中约 81% 是读取行，约 52 行 | 每步约 64 个读取行，按整窗口组成：一个窗口在"已有行数 + R/2 ≤ 64"时加入。实测均值 64.2 行（约 9.5 个窗口） |
| 数据 | 160,957 行：130,560 个读取行（空读只保留约一半），26,112 个离线行，4,285 个 sd 行 | 27,953 个窗口的全部 188,853 次读：含全部空读，没有离线行 |
| 遍数 | 192,000 行，约 1.19 遍 | 约 1.02 遍 |
| 批内归一 | 每个 micro-batch 求均值，再在 micro-batch 与 rank 之间平均 | 全局按 token 加权的均值 |

### 2.7 存档与评测

- 存档路径为 `<out>/ckpt/step_NNNNN/`，内含：
  - `lora.pt`：`train_w3po` 格式，键为 `model.layers.N.self_attn.X_proj.{a,b}`；
  - `optim.pt`；
  - `extra.pt`：sd_proj、new_emb、new_head、extra_params、最后的 λ、rank/alpha；
  - `DONE`。
- 默认每 500 步存一次，最后一步也存。`--resume` 从最新的 DONE 存档续跑；数据流只取决于步号，与卡数无关。
- **只有 `lora.pt` 不是完整模型**：投影器和新 token 行也在训练。因此：
  - 正式评测：先用 `crf/export_merged.py` 写合并目录，再作为 `THINKER=` 交给 `scripts/eval_pipeline.sh`（`launch_decode.sh`、`train_w3po --base`、`precheck_sched --thinker` 也都能直接用）。
    - 合并公式同 `merge_thinker_gstar.py`：W' = bf16(fp32(W) + (α/r)·b@a)，新 token 行写回，sd_proj 替换。
    - 每次导出约 66 GB，几乎每个分片都要重写。
  - 快速解码：`crf/decode_crf_ckpt.py`，在内存里按同样的公式应用存档，其余参数原样交给 `online_decode`。
  - 测试核对了两条路径给出逐位相同的权重、行和投影器。

## 3. HOOK API（给方案二；默认全部为 None，与 v8 逐位一致）

```python
from experiments.mist_w3po_v1.crf import train_crf_sft as T
T.main(argv, post_transform=fn, extra_loss=fn2, extra_params=[module, ...])
```

- **P 进入模型的位置**：
  - v8 的数据加载把 diarizer 后验 P [F, 8]（80 ms 帧）线性插值到音频 token 网格（`post_to_tokens`）。
  - `ThinkerGStarModel.embed` 把结果送进 sd_proj，再按 K:1 与音频编码器输出交错。推理时 `Engine.audio_embeds` 做的是同样的事。
  - 钩子就放在 `post_to_tokens` 之前，在 80 ms 网格上。
- **`post_transform(P_prefix, enc, meta) -> [T, 8]`**：
  - `P_prefix`：float32 [T, 8]，扰动之后的后验，T = max(0, t_r − 1040)//80，恰好是 v8 在 t_r 读到的未置零帧；T 可以为 0。
  - `enc`：float32 [T, 2048]，已 detach。这是同一次前向里、紧挨投影器之前算出的冻结音频编码器状态。
    - 编码器本身是每秒 13 帧，中心在 (i+0.5)·dur/n；按 np.interp 语义线性重采样到 80 ms 帧中心。
    - 代价只是一次 [T × n] 插值，**所以传了**。
    - 可调用对象带属性 `wants_enc = False` 时传 None。
  - `meta`：id、widx、read、t_ms、final、dur_ms、T、F、n_audio、frame_s、lookahead_ms、phase（"score" 为一′ 的无梯度第一遍，"train" 为梯度遍）、step、rec（sched_v1 记录，含 units 的 a_ms/b_ms/slot/ov_s）。
  - 返回值替换 P_prefix，T 之后的帧保持 0，然后继续 v8 的路径。插值用 torch 重写，与 numpy 版逐位相同，梯度可以流回钩子。
- **`extra_loss(info) -> 标量`**：
  - 在每个梯度 micro-batch 反向传播之前加到损失上。
  - info 包含 metas、P_in、P_out、enc、rows_global、windows_global、step、device。
  - 梯度在 micro-batch 与 rank 之间求和，所以钩子要用全局计数自行归一。
- **`extra_params`**：nn.Module 列表。
  - 移到训练设备（dtype 不变），参数加入同一个 AdamW 和学习率调度、梯度 all-reduce 与全局裁剪。
  - 存入 `extra.pt["extra_params"]`，续训时恢复。
  - `decode_crf_ckpt` / `export_merged` 不会应用这些模块，只打印警告。推理时的开关由方案二自己实现。

## 4. CPU 测试（`PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_train_crf_sft`）

测试用小型随机因果 LM（2 层 Qwen3，d = 32）代替 v8，但分词器、特征提取、提示都是真实的，窗口是真实的 sched_v1 AMI 训练窗口（带音频与后验）。

| 测试 | 结果 |
|---|---|
| (a) C0 = v8 SFT：同样的读按 v8 行（v8 自己的 `build_batch`）喂给 `ThinkerGStarModel.forward`，比较损失与全部可训练参数的梯度 | 单个 micro-batch：损失相对误差 0，梯度 5.1e−7；拆成 2 个 micro-batch：7.0e−8 / 1.3e−4（float32 累加顺序） |
| (b) Fisher 恒等式（h = 1，D = 1，不含 L_state，λ = 0.7，屏蔽关）：训练器梯度 vs 对 −log Z 经动态规划自动求导 | 3.4e−7 |
| (b′) 屏蔽开：格的最后一次读 = 屏蔽 `<wait>` 后的 log-softmax（token 与结束符），其余读与损失用的值不变，后验确实改变 | 通过，后验最多变 0.50 |
| (c) 不挂钩子时：批张量 = v8 `build_batch`（逐位）；嵌入与隐状态 `torch.equal`；恒等钩子的差为 0；带参数的钩子能拿到梯度；P_prefix / enc 形状和 extra_loss 的 info 与文档一致 | 通过 |
| (d) 300 个真实窗口、随机 log 概率、λ\* 下：Σ w_tok = N、Σ w_end = R（两种目标）；两种目标的 D 相同；H(q) 与 q(启发式) 和穷举一致（60 个小实例） | 最大误差 7.4e−13 |
| torch 版 post_to_tokens vs numpy | 逐位相同 |
| lr 调度 vs v8 日志、步计划（确定性，每遍是一个排列，均值 64.2 行） | 通过 |
| 存档往返：lora.pt 可在新注入的 LoRA 上加载；extra_params、λ | 通过 |
| `export_merged` = 内存中合并（假的两分片底座） | 逐位相同 |

## 5. GPU 冒烟测试（卡 0–3，输出在 `/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/crf_smoke/`）

- **安全措施**：
  - 同事每卡占 39–44 GB。我的训练进程用 `--mem-cap-gib 80` 限制分配器。
  - 另有一个看门狗：任何一张我的卡剩余显存低于 8 GiB，就按 PGID 结束我自己的作业。全程没有触发。
  - 每卡总占用峰值约 43 + 68 ≈ 111 GB。
  - 所有作业都已结束，看门狗已停，卡上没有我的进程。

**(i) 预检 `precheck_sched.py --n 5`**（卡 3，`--max-gib 80`）：
- 真实 v8 打分路径跑通，输出 `report.json`。
- 5 个窗口的结果：L_ref（样本）0.959，λ\* = −7.20，λ = 0 时 E[Lat] = 0.910，MAP 平均 Lat 0.910；C1（提交读不同的单元）11.1%，C2 = 0.0 s，C3 上升 1.7 个百分点，`go = true`。
- 只有 5 个窗口，只说明流程通了，不代表结论。

**(ii) 训练器，两种目标各 3 步**（单卡，用 `--stop-after 3` 跑正式调度的前 3 步，lr 为 2e−6、4e−6、6e−6）：

| 步 | C0 损失 | 一′ loss_main / L_state | 一′ sft_loss | λ | E_q[Lat] | 本批 lat_heur | H(q) | q(启发式) |
|---|---|---|---|---|---|---|---|---|
| 1 | **1.101** | 1.161 / 2.276 | 1.101 | −14.08 | 0.8256 | 0.627 | 0.764 | 0.700 |
| 2 | 1.021 | 1.039 / 2.120 | 1.021 | +12.40 | 0.8256 | 0.901 | 0.753 | 0.554 |
| 3 | 0.868 | 0.866 / 1.900 | 0.869 | +1.88 | 0.8256 | 0.867 | 0.571 | 0.710 |

- 损失都是有限值。
- C0 第 1 步（即 v8 本身）损失 1.101，落在 v8 训练最后 500 步的范围内（均值 1.011，0.84–1.16）。v8 从 O2 出发时第 1 步是 2.81。
- 一′ 的 sft_loss 与 C0 一致，差别不超过 1.3e−3，来自 bf16 与不同的批组成（见 (v)）。
- 192 个 LoRA b 张量全部非零（|b| 最大 1.2e−5，约等于各步 lr 之和）；sd_proj 变化 1.2e−5。
- 存档已写出（lora.pt 107 MB，optim.pt 273 MB，extra.pt 30 MB）。
- 单卡每步：C0 12–15 s；一′ 16–21 s（第一遍 4–5 s，λ 0.4–0.7 s，第二遍 12–15 s）。显存最大分配 66.9 / 67.5 GiB。
- 修正后的一′ 重跑了一次（`sched_fix_3step`），数值与修正前几乎相同。

**(iii) 用现有解码引擎解码 2 个 dev 窗口**（卡 3）：
- `decode_crf_ckpt.py --crf-ckpt sched_3step/ckpt/step_00003`，加上 `online_decode` 原有参数（`--policy wait --context 0 --limit 2`）。
- 应用存档时报告：折叠 192 个适配器，写入 259 行，替换投影器。随后 `W3PO_ONLINE_DONE`。
- 对照组用 v8 原样解码同样的 2 个窗口：
  - 第 2 个窗口的原始输出逐字相同；
  - 第 1 个窗口只差一个结束时间 token（`<|t22|>` → `<|t23|>`），译文相同。
  - 3 步极小的更新，这个结果符合预期。

**(iv) 多卡**：
- 2 进程 torchrun（卡 1、2）：用 `--resume` 从 `sched_3step` 的第 3 步续跑第 4–5 步。λ 为 −4.25 / −2.10，每步 10.4 / 12.9 s，存档时校验和 `RANKS_IN_SYNC`。
- 4 进程计时（卡 0–3，正式启动脚本的配置）：
  - 一′ 第 1–4 步：每步 10.0 / 8.1 / 9.3 / 6.0 s（第一遍 1.4–3.4 s，λ 0.4–0.7 s，第二遍 4.0–6.4 s）；
  - C0：每步 9.7 / 5.1 / 6.2 / 4.0 s；
  - 每卡最大分配 66.3 GiB，C0 的校验和 `RANKS_IN_SYNC`。
  - 一′ 4 进程那次的控制台日志因为我拼命令时的重定向疏漏没有保存，`metrics.jsonl` 和 step_00004 存档完好；一′ 路径的 rank 同步已由 2 进程那次证明。

**(v) 打分交叉核对**：训练器第一遍的打分（v8 SFT 路径：fp32 投影器和新行，右填充）对比预检的打分（引擎路径：bf16，左填充），用同样的 5 个窗口。
- 逐窗口的 E[Lat] 基本一致：λ = 0 时差 0.0002–0.011 s，λ = −7.2 时差 ≤ 0.03 s。
- 逐位置的差别不小：token 均值 0.08，p99 0.59；结束符均值 0.22，p99 0.98，最大 1.9 nats。
- 诊断：**引擎自己**对同样的读，批大小 4 与 1 之间的差别同样大（token 均值 0.082，结束符均值 0.228，最大 2.5）；训练器单行与 micro-batch 之间也一样。训练器与引擎之间的差别没有超出这个范围，把新 token 行和投影器改成 bf16 也不变。
- 结论：这是 v8（bf16 MoE，专家路由会随批组成翻转）本身的数值噪声，不是训练器的错误。

## 6. 正式运行命令（未执行，等用户命令）

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
# C0（共用对照）与一′，各 4 卡；同事仍在卡上时加 EXTRA="--mem-cap-gib 88"
CARDS=0,1,2,3 bash experiments/mist_w3po_v1/scripts/run_crf_c0.sh
CARDS=4,5,6,7 bash experiments/mist_w3po_v1/scripts/run_crf_sched.sh
# 续跑：RESUME=1 RUN_ID=<已有 RUN_ID> CARDS=... bash 同一脚本
```

- 输出在 `eval_outputs/mist_w3po_v1/crf/train/<RUN_ID>`（默认 RUN_ID 为 `crf_c0_<UTC>` / `crf_sched_<UTC>`），日志在 `logs/mist_w3po_v1/<RUN_ID>.log`。目录或日志已存在时拒绝运行。
- 第 1000 步的中期评测（§11）：

```bash
python -m experiments.mist_w3po_v1.crf.export_merged --ckpt eval_outputs/mist_w3po_v1/crf/train/<RUN_ID>/ckpt/step_01000 \
    --out eval_outputs/mist_w3po_v1/crf/models/<RUN_ID>_s1000/thinker_merged
NAME=<RUN_ID>_s1000 THINKER=eval_outputs/mist_w3po_v1/crf/models/<RUN_ID>_s1000/thinker_merged CARDS="..." \
    CONTEXT=<与 v8 参照相同> setsid nohup bash experiments/mist_w3po_v1/scripts/eval_pipeline.sh > logs/mist_w3po_v1/eval_<NAME>.log 2>&1 < /dev/null &
```

## 7. 风险与需要用户决定的事项

1. **每批 λ 的噪声。**
   - 现象：按要求，每步都把**全局** L_ref = 0.826 s 强加给约 10 个窗口。但各批自己的启发式平均滞后在 0.63–0.90 s 之间，所以 λ 每步在约 −14 到 +13 之间摆动：一批里的窗口被整体推晚，下一批被整体推早。整体期望仍满足约束，但给后验加了与内容无关的噪声。
   - 已备选项：`--lat-target-batch`（目标改为本批自己的启发式均值，默认关）。是否启用请用户决定。
2. **L_state 的量级。** L_state（约 1.9–2.3）是主损失（约 0.9–1.2）的 2 倍左右，在梯度里占主导。它在"已可行但音频还没听完"的早期读上训练发射，相当于训练"预判"。这是预登记的权重 1，没有改。正式运行中要盯住重复率和过早提交。
3. **层级权重。** 默认按 v8 配方打开（×1.5 / ×2），一′ 的更新因此是 EM 变体（§2.4），不是严格的 ∇J。严格版本需要 `--hier-weights 0`，那样 C0 也要一起关，而且会偏离 v8 的配方。默认值请用户确认。
4. **v8 打分本身有噪声。** 同一个读在不同批组成下，log 概率平均差 0.08（token）/ 0.23（结束符）nats，最大约 2.5。所以每步的 q 带有这层噪声（权重差最多约 0.2）。500 窗口预检的 C1/C2/C3 判定也会受影响，建议预检时固定批大小。
5. **C0 的数据与 v8 不同。** 两臂用相同数据，但它们都含有全部空读（v8 只保留了约一半），也都没有离线行和 sd 行。C0 本身可能偏离 v8，比如更爱等待、离线能力退化，这会影响"对 v8 的门槛"。C0 与一′ 之间的比较仍然公平。
6. **评测必须带上投影器和新 token 行**：用 `export_merged.py` 或 `decode_crf_ckpt.py`。只给 `--lora lora.pt` 会丢掉这两部分的更新。
7. **资源。**
   - 每卡约 66–68 GiB，加上约 1.5 GiB 的上下文。同事 40–44 GB 时，每卡合计约 111 GB。同事的占用如果涨过约 65 GiB，合计就会超出 135 GiB。
   - 4 卡 3000 步：C0 约 3.5–5 h，一′ 约 5–7.5 h；两个同时跑需要 8 张卡。
8. **最后一次读：格屏蔽，损失不屏蔽（两臂一致）。** 这是为了让 C0 严格等于 v8 SFT。如果希望一′ 训练的就是解码模型的精确似然，需要给损失也加屏蔽，而且两臂都得加。目前没有实现这个开关。
9. 方案二的 `extra_params` 不会被解码包装和导出应用，推理开关由方案二负责。加载时出现的 "Some parameters are on the meta device" 来自 talker 被放到 CPU，Thinker 完整在 GPU 上，不影响结果。
