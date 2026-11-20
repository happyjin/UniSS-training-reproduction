# 鲁棒第十七波：感知层 LoRA（只在音频编码器上加 LoRA）

方法与判定见主线方案 4.3（44823eb32），逐字有效。即 `reports/mist_w3po_v1/robust_mainline_plan_20261120.zh-CN.md` 第 4.3 节；本文件只追加实现说明和结果。

## 实现说明（2026-11-20 15:47 CET，写在任何冒烟、训练和评测数字之前）

### 1. 音频编码器的结构（transformers 5.19.0，`modeling_qwen3_omni_moe.py`）
- 模块：`thinker.audio_tower`，类 `Qwen3OmniMoeAudioEncoder`。32 层（`encoder_layers` = 32），d_model 1280，20 个头，FFN 5120，输出投影到 2048。
- 每层 `thinker.audio_tower.layers.<i>.self_attn` 的四个投影是 `q_proj`、`k_proj`、`v_proj`、`out_proj`（编码器里输出投影叫 `out_proj`，不叫 `o_proj`），都是 1280×1280、带 bias 的 nn.Linear。
- LoRA 加在这 32 × 4 = 128 个投影上，rank 16，alpha 32，b 初始化为 0，参数用 fp32，共 5.24 M 个参数。第十四波在 48 层文本层上的 LoRA 是 13.37 M（384 个张量）。
- 编码器的权重全部在 release 的 `model-00001-of-00015.safetensors` 一个分片里，键名形如 `thinker.audio_tower.layers.0.self_attn.q_proj.weight`。

### 2. 梯度能否到达编码器
- `Qwen3OmniMoeThinkerForConditionalGeneration.forward` 调用 `get_audio_features` → `audio_tower(...)`，路径上没有 `torch.no_grad`，也没有 `detach`。音频特征通过 `inputs_embeds.masked_scatter(audio_mask, audio_features)` 进入文本层，所以损失的梯度会穿过冻结的 48 层文本层，回传到编码器。
- 第十四波对编码器做了两件事：`requires_grad_(False)`（参数冻结）和 `audio_tower.eval()`。冻结只影响参数，不会把编码器从计算图里去掉；注入 LoRA 后，只有 a/b 需要梯度。eval 模式保留：编码器的 dropout、attention_dropout、activation_dropout 都是 0，train 和 eval 的前向完全相同，与推理时一致。
- **梯度检查点：**
  - 文本层：与第十四波相同，第 0–43 层非重入式检查点，最后 4 层不检查点，最后 4 层的专家用 eager 循环，全局 math SDPA。这些都原样保留，好让两波唯一的差别是 LoRA 的位置。
  - 非重入式检查点在层内参数全部冻结时，仍然会把梯度传给输入，所以梯度能穿过文本层。
  - 编码器层（`GradientCheckpointingLayer`）只在 training 模式下做检查点。编码器处于 eval 模式，所以不做检查点。它的激活很小：每条约 10 s 的音频，CNN 之后约 130 帧，32 层 × 1280 维，可以忽略。
- **自检（写进训练脚本）：**
  - 构建时断言：需要梯度的参数恰好是 `audio_tower.*` 下的 256 个 a/b 张量，文本层没有任何参数需要梯度，并且 LoRA 层数是 32 × 4。
  - 第一步反传后断言：编码器 LoRA-b 的梯度范数 > 0，并写进日志。b 初始为 0、a 不为 0，所以只要有梯度回传到编码器，b 的梯度范数就一定大于 0。

### 3. 训练（`experiments/mist_w3po_v1/voicecond/wave17_enc_train.py`）
- 是 `wave14_train.py` 的副本，只做了这些改动：
  - LoRA 位置，见上；
  - 去掉 Fishr 专用的"最后 4 层 LoRA"列表，只跑 ERM（`--arm erm`，脚本断言这一点）；
  - adapters_final.json 记 `module: "thinker.audio_tower"`、`targets` 和 `mb`；
  - 第一步的梯度断言。
- 数据、8 个环境、Sampler 的 sha1 键（"20261119|w5m|row|s|j"、"20261119|w5m|mix|s|j"）、2300 步、全局批量 64、AdamW 1e-4、余弦调度到 0、种子 20261119 都不变。数据用第十四波已有的 `eval_outputs/mist_w3po_v1/wave14/data/`，不重建。
- **world 2（卡 6–7）：** rank r 取 (j % 8) % 2 == r 的 slot，每个 rank 32 条。各 rank 的梯度相加后除以 world，所以损失仍是 64 条的平均 CE，与第十四波同一批样本（论证同第八波实现修正 2、第十四波实现说明 1）。
- **mb：** 先试 `--mb 32`（一个微批），冒烟 6 步（`--max-steps-run 6`，不留 checkpoint）。显存不够就降到 16。mb 只影响梯度累积的切分。
- **数据加载：** 第十四波记录过瓶颈在数据加载，所以每个 rank 用 12 个 DataLoader worker（第十四波是 6 个）。每个 worker 处理整步，DataLoader 按轮转顺序取，所以步序和样本不变。worker 数只影响速度。
- **启动：** `wave17_run.sh`（setsid nohup，关闭 fd 8/9）。launcher 启动后会复查 `$!` 的 argv 必须是 torchrun 本身，并登记到 `logs/mist_w3po_v1/wave17/w17_workers.pids`。
- **watcher：** `wave17_noplaceholder.sh`，PIDS 和 STOP（`logs/mist_w3po_v1/wave17/w17_noplaceholder.stop`）都是这一波自己的，只让卡 6–7 上的占位进程退让。卡 0–5 不碰。

### 4. 合并与 WiSE（`wave17_merge.py`）
- 是 `wave5_m_merge.py` 的副本，改成按 json 里的 `module` 加前缀：W' = W + (alpha / rank) · b @ a，键 `thinker.audio_tower.<层>.self_attn.<proj>.weight`。
- 断言：恰好 128 个被改的键，全部在 `thinker.audio_tower.` 下。没有被改动权重的分片用硬链接，有的分片（只有分片 1）重写。拒绝覆盖。
- **自检（合并后，不通过就不输出成功标记）：**
  - 编码器以外的权重逐位不变：重写分片里所有未改动的张量，与 release 逐字节比较（按 uint8 视图），dtype 和形状相同；其余文件与 release 是同一个 inode 或逐字节相同。
  - 被改动的 128 个张量都等于 (W.float() + Δ).to(bf16)，并且确实与原权重不同。
- **enc_wise05 = alpha 减半：** `run_enc_wise05/adapters_final.pt` 是 `run_enc/adapters_final.pt` 的硬链接，json 里 alpha 为 16，即 W0 + 0.5·Δ，与第十四波相同。enc 是 α = 1。

### 5. 评测（`wave17_eval.sh`、`wave17_eval_orch.sh`，第十四波的副本）
- 链与第十四波相同：si_thinker_stream la2 w1 → T1c export_1500（每个方向 4 个分片，种子 20261021）→ 只对语音做 ASR。条目是 wave2_S1 的 dev 条目。vLLM 进程的 PATH 里有 vllm_omni_030/bin。
- 只用卡 6–7：每个系统 en2zh 的 Thinker 在卡 6，zh2en 在卡 7；Talker 分片 0–3 跟着各自方向的卡；ASR 也一样。先评 enc_wise05，再评 enc，依次进行。
- **机理检验的 Thinker 运行**（在 enc 之后，同样在卡 6–7，`STAGE=mech`）：
  - **600 条训练子集：** 第十三波的 600 条是 CoVoST 中→英训练集（第五波数据），不在第十四波数据里。所以按第十三波的 sha1 规则，从第十四波的 `rows_zh2en.jsonl` 里重新选：行号 i 按 int(sha1("20261120|w13|train|i")) 排序，取前 600（`wave17_items.py` → `eval_outputs/mist_w3po_v1/wave17/data/items_zh2en_train600.jsonl`），用 la2 w1 跑 base 和 enc。
  - 另加一个对照，也是辅助：用第十四波的 unist（文本层 LoRA，同一数据、同样步数，即 `wave14/merged_unist`）在同一批 600 条上跑。这样"编码器不记忆"有同数据的对照，而不只是和第十三波的 +13.31 比。merged_unist 不存在时跳过，并记录。
  - 每条另记录它在 2300 步训练中被 Sampler 抽中的次数（按 sha1 键复算，中→英的 32 个 slot）。主数字用全部 600 条；被抽中至少一次的子集作为辅助一并报告，因为 0.6 个 epoch 时，约一半的行从未被训练过。
  - **oracle 分解：** enc 用 `--policy oracle`，在第十二波的 (a) 条目 `items_<dir>_a.jsonl` 上跑，两个方向。底座的 oracle 用 `wave3_combos/D/oracle_<dir>`，与第十三波相同。
- 每个 mech 作业必须留下 600 行。

### 6. 打分
- **`wave17_score.py`**（`wave14_score.py` 的副本）：
  - 系统：base（S1 + 2 次 Thinker 重跑）、erm_wise05、unist_wise05（第十四波 dev 输出）、enc_wise05、enc。
  - 第 1 条：enc_wise05 相对底座，用统一规则，与第十四波第 1 条的代码相同。
  - 第 2 条：enc_wise05 − unist_wise05。增量看方向 × {a, b, b_no_crosstalk} × {干净, 带噪} 的 text，要求 CI 下界 > 0 且点估计大于该单元的底座噪声底；不伤看方向 × {a, b} × {干净, 带噪} × {text, ASR}，不能有点估计 < −0.3 且 CI 上界 < 0 的单元。单元范围与第十四波第 2 条相同。
  - 判定：两条都过为 PASS；只过第 1 条记为"相对 unist_wise05 没有增量"；第 1 条不过为 FAIL。
  - **自检：** erm_wise05 相对底座的结果须与 `wave5/M/summary_erm_wise05.json` 完全一致；如果有第十四波的 `summary.json`，unist_wise05 的分数和它相对底座的差也须与之完全一致。每个单元的 bootstrap 重采样只取决于单元，与系统集合无关，所以加入系统后数字不会变。
  - 辅助：enc（α = 1）的全部差值。zh2en/a/clean 上 enc − base_mean 单列，对应机理预测 1："不显著掉分"即 CI 上界 ≥ 0。
- **`wave17_mech_score.py`（机理检验，辅助）：**
  - **预测 2：** 600 条上 enc − base 的点估计和 CI（句子为单位，paired bootstrap 2000，种子 0），与第十三波的 +13.31 [11.19, 15.63] 并列。判定沿用第十三波 H4 的记忆门槛：如果 enc − base ≥ 3.0 且 CI 下界 > 0，就算"有记忆"，预测不成立；否则预测成立。unist − base 作对照，同样报告。
  - **预测 3：** 两个方向 × 干净/带噪的 (a) 单元：
    - 离线差 off(enc) − off(base)；
    - LA2 差 la2(enc) − la2(base_mean)；
    - 提交损失 CL = off − la2，dCL = CL(enc) − CL(base)。于是 LA2 差 = 离线差 − dCL。
    - 英→中带噪上，预测成立的条件是：离线差的 CI 下界 > 0，并且离线差 ≥ −dCL，即离线翻译贡献的增益不少于提交贡献的。如果英→中带噪的 LA2 差本身不显著，就照实报告分解，预测记为"不适用（没有可分解的增益）"。

### 7. 自动衔接（`wave17_chain.sh`，setsid nohup）
- 流程：等训练的 torchrun 退出 → 检查 adapters_final → 建 WiSE 副本 → 评测 enc_wise05（卡 6–7）→ 评测 enc（卡 6–7）→ 跑 mech 作业（卡 6–7）→ touch STOP，把卡交还 guard → `wave17_mech_score.py` → 等第十四波的 `eval_unist_wise05/EVAL_DONE` 出现（等待脚本自动衔接；若第十四波写了 CHAIN_FAILED 或 EVAL_FAILED，就停下并记录）→ `wave17_score.py` → CHAIN_DONE。
- 链条本身不停任何进程；占位进程由 launcher 和 watcher 按 PID 停，并在 kill 前复查 argv。

### 实现说明补记（2026-11-20 15:54 CET）
- **冒烟**（`wave17/smoke`，卡 6–7，`--mb 32`，6 步，15:48–15:49 CET，不留 checkpoint）：
  - 构建断言全部通过：256 个 adapter 张量，5.24 M 参数，文本层没有可训练参数；
  - 第一步两个 rank 上编码器 LoRA-b 的梯度范数都是 7.54e-2（> 0），梯度确实到达了编码器；
  - 峰值显存 78.8 GiB/卡，装得下；
  - 所以正式训练取 `--mb 32`，每个 rank 一个微批，没有梯度累积。
- **机理检验的 600 条：** `wave17_items.py` 已建好 `eval_outputs/mist_w3po_v1/wave17/data/items_zh2en_train600.jsonl`。其中 289 条在 2300 步中至少被 Sampler 抽中一次，311 条从未被抽中。
- **其余脚本：** `wave17_merge.py`、`wave17_eval.sh`、`wave17_eval_orch.sh`、`wave17_score.py`、`wave17_mech_score.py`、`wave17_chain.sh` 都只用 bash -n、DRY_RUN 或 py_compile 检查过。
  - `wave17_score.py` 在现有数据上跑过一次自检，输出写到 scratchpad，没有 enc 的数字：erm_wise05 相对底座的 180 个数值和门与第五波完全一致。
  - `wave17_mech_score.py` 的分解代码用第十三波的 erm 当作"enc"试跑过一次，输出写到 scratchpad：复现了第十三波的离线 −2.77、LA2 −3.63，说明代码路径正确。这不是第十七波的结果。

## 训练进度
- 2026-11-20 15:49:53 CET 启动 `run_enc`：卡 6–7，world 2，`--mb 32`，每个 rank 12 个 worker，torchrun 的 pid 是 2212086。watcher 的 pid 是 2191759。链条 `wave17_chain.sh` 的 pid 是 2246267（15:52 CET 起）。
- 第 4 步的 CE 与冒烟第 4 步一致（1.00120），同一批样本得到确认。第 10–30 步平均每步 3.8 s，2300 步约 2.4 h，预计 18:15 CET 前后结束。
- 之后链条自动接续：
  - WiSE 副本，然后合并（CPU，只重写一个分片）；
  - enc_wise05 评测，约 1.5 h；
  - enc 评测，约 1.5 h；
  - 机理检验的 Thinker 作业，约 0.7 h；
  - touch STOP；
  - 打分，需要第十四波的 unist_wise05 dev 输出，按第十四波的进度应在 19:30 CET 前后出来。
- 预计 22:00 CET 前后出结果。
