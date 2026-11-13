# 实验 (b)：联合双向 Simul（一个模型同时做 zh→en 和 en→zh）——计划与预登记 v1

写于 2026-11-12 22:40（布拉格），在 (b) 的任何 GPU 作业或结果之前。用户决定：B1 两段式、卡 4–7 与 (a) 并行、训练长度由 dev 代理决定（无固定步数）、每个检查点保留优化器状态。

## 0. 一句话

先训一个**联合双向基线**（第 1 段：从 O2 出发的联合 v8 Simul SFT），再在它上面做**用户的方法**（第 2 段：联合 SerialCRF v2′ + 读级合并 + 各方向 PL-EM talker）。两段的训练长度都由两个方向 dev5 文本代理的早停规则决定，15000 步只是防失控的上限。

## 1. 为什么是两段：分开的系统没有共用的 base

- zh→en A′（`crf_v2p_20261110T075131Z`，见 args.json）的 base 是 `thinker_gstar_s2dl_zh2en_3k_20261030`；en→zh (a) 的 base 是 `thinker_gstar_s2dl_20261029b`。
- 两者都是从 O2（`thinker_gstar_o2_20261029b`，用 `e2e_gstar_bidir_v2` 训的双向离线模型）出发、在**单方向** stream rows 上做 3000 步 Megatron LoRA（GBS 64，MBS 8，8 卡，种子 20261026，lr 1e-4→1e-5 余弦，warm-up 50）再合并得到的。
- 所以"同一个双向 base + 联合 LoRA"不存在。按用户决定（2026-11-12，"先训练双向的 baseline……再在这个基础上进行我的方法"）走 B1：
  - **第 1 段 = 联合双向基线**，单独作为一行报告；
  - **第 2 段 = 方法**，在第 1 段选中的检查点上训练。

## 2. 第 1 段：联合 v8 Simul SFT（双向基线）

### 2.1 数据（已建好，CPU，提交 a00e3a112）

- `data/ami_multispk_si_v1/e2e_gstar_stream_bidir_v2/rows_train.jsonl`，960,000 行，由 `joint/build_joint_v8_rows.py --per-dir 480000` 生成，有 manifest。
- 构造方式：把两个分开 run 的 `GStarRows` 样本顺序（default_rng(20261026)，每遍一个排列）**逐条复现**，然后交错排列：每个 GBS-64 的 batch 前 32 条是 zh→en、后 32 条是 en→zh。
- 读取时用 `--gstar-sequential`（新选项，默认关）。第 t 步恰好是 zh→en 的第 [32t, 32t+32) 个样本加上 en→zh 的第 [32t, 32t+32) 个样本。
- 结果：到第 6000 步时，每个方向见过的样本**逐条等于**它那个分开 run 的 192,000 个样本，顺序也相同；6000 步以后按同一 rng 的后续排列继续。
- 单元测试 `joint/test_build_joint_v8_rows.py` 覆盖了这一点。v2 的前 100 MB 与 6000 步版本 v1（`e2e_gstar_stream_bidir_v1`，未使用）逐字节相同。
- 每行的提示词用它自己的 `dir`（build_batch 原本就按行取方向）。

### 2.2 设置（唯一设置，不扫参）

`run_thinker_gstar.sh` 的参数如下，其余与两个分开的 v8 逐项相同：
- `BASE=O2 INIT_MERGED=1 GSTAR_SEQUENTIAL=1 ROWS=e2e_gstar_stream_bidir_v2`
- `NPROC=4 MBS=8 GBS=64`：4 卡、2 个 micro-batch，全局 batch 与原来相同
- `TRAIN_ITERS=15000`（上限）`LR_DECAY_ITERS=12000`
- `SAVE_INTERVAL=500 SAVE_OPTIM=" "`（每个检查点带优化器和 RNG 状态）
- 种子 20261026
- LoRA r32/α64 + 投影器 + 新 token 行；lr 1e-4，warm-up 50

### 2.3 检查点代理（第 1 段）

- 每 500 步：`joint/megatron_to_crf_ckpt.py` 把 Megatron 检查点转成 CRF 格式。在 O2 上于内存中折叠，结果与 `merge_thinker_gstar` 的合并模型**逐位相同**（单元测试已覆盖），所以每个候选不需要合并出 66 GB 的模型。
- 然后在**两个方向的 dev5** 上做纯文本解码，流水线与 A′ 的 dev5 代理逐项相同：finalpost close_detect、wait 策略、lag 2000、ctx 0、batch 16、1.0 s 读、4 个窗口分片、`--cap135`、loop guard。
- 延迟参照：分开的 v8（zh2en_3k、20261029b）各配一个 `joint/identity_ckpt.py` 恒等检查点，走同一条 dev5 流水线，各解码一次。
- 选中后再做一次真正的合并（`merge_thinker_gstar`），作为第 2 段的 base。合并模型配恒等检查点做 dev 全集评测，等价性由单元测试保证。

## 3. 第 2 段：联合 SerialCRF v2′（用户的方法）

- `train_crf_sft --objective serial_v2p --tau 0.5 --final-read-full-post 1`
- `--base` = 第 1 段选中检查点的合并模型
- `--sched crf/sched_v1 --dir zh2en --sched2 crf/sched_en2zh_v1 --dir2 en2zh`（新选项）：每步是 `plan_joint`，两个方向各约 32 个读行；每个窗口用自己方向的 v8 提示词（JointPrompter）。
- 训练长度：`--steps 15000`（上限）`--lr-horizon 12000`（新选项，默认 0 表示不变）。
- 其余与 A′ / (a) 相同：lr 1e-4，warm-up 50，余弦退到 1e-5；种子 20261109；LoRA r32/α64 + 投影器 + 新行；TOK 32000，CAP 120，fast-moe，prefetch，逐行特征。
- `--save-every 500`：每个检查点是 lora + extra + optim，约 392 MB。
- `--stop-file <run>/STOP`（新选项）：文件出现后，在下一步存一个检查点并干净退出（`STOP_FILE_SEEN`）。
- 代理：每个检查点在两个方向 dev5 上做文本解码（同 2.3），另加两个方向的 held-out 损失（`crf/heldout_loss.py`，dev5 的 sched 窗口：zh→en 用 `sched_dev_v1`，en→zh 用 `sched_en2zh_dev_v1`）。
- 延迟参照：zh→en 是 A′ step_03000，en→zh 是 (a) 选中的检查点。都走同一条 dev5 流水线，(a) 的数直接取它自己的 dev5 代理输出。

## 4. 为什么可以只看 dev5：排名一致性（CPU，只用已有输出）

zh→en 共 53 个全 dev 解码，按会议切成 dev5（5 场 1092 窗）和其余 10 场：

| 系统集合 | n | ρ / τ（dev5 对全 dev） | ρ / τ（dev5 对其余 10 场，相互独立） | 最优者一致？/ 后悔 |
|---|---|---|---|---|
| s3000 各方法 | 8 | 0.976 / 0.929 | 0.929 / 0.857 | 一致 / 0 |
| RL 轨迹 dia（step 0–150） | 5 | 0.90 / 0.80 | 0.70 / 0.60 | 一致 / 0 |
| RL 轨迹 w3dia | 5 | 0.90 / 0.80 | 0.70 / 0.60 | 一致 / 0 |
| 正常系统（windows_v1） | 22 | 0.848 / 0.726 | 0.762 / 0.654 | 一致 / 0；全 dev 上相差 ≥ 0.5 的 72 对中 0 对排反 |
| 16 个 guard pilot checkpoint | 16 | 0.640 / 0.527 | 0.467 / 0.403 | 不一致，后悔 0.08 BLEU |
| 退化系统（无 guard 等） | 12 | 0.797 / 0.576 | −0.42 | 不一致，后悔 1.60 |

- LAAL 的排名：dev5 与全 dev 的 ρ = 0.96–1.00；dev5 会把系统间的差放大约 1.1 倍。
- 局限：
  - 全 dev 上相差 < 0.3 BLEU 的系统对，dev5 的顺序接近随机；
  - dev5 的绝对值偏高 0–0.9 BLEU，不能和全 dev 的数直接比；
  - 只发生在其余会议上的崩溃，dev5 看不见。所以选中后的 dev 全集评测必须做，第 7 节的门槛也查崩溃。
  - 没有任何 run 有中间步的全 dev 解码，也没有 held-out 损失的历史数据。所以 held-out 损失**只用作平局时的判据**，不作主信号。
- 解码耗时（`t_eval/*/orch.log`）：
  - zh→en dev5 每个检查点 15–20 分钟墙钟，37–60 GPU 分钟（4 卡）；
  - en→zh 的 dev5 约大 1.2 倍；
  - **两个方向合计每个检查点约 80–130 GPU 分钟**，4 张卡并行时约 25–35 分钟。
- 结论：用 dev5，两个方向都用。
- 表格和统计复制到了 `eval_outputs/mist_w3po_v1/joint/devrank_v1/`（`analysis.txt`、`table.json`、`analyze.py`、`cut_score.sh`）。

## 5. 早停与选择规则（两段相同，`scripts/joint_select.py` 实现，写在任何结果之前）

- **评估点**：每 500 步一个检查点，s = 500, 1000, …。代理异步运行，训练从不等评测。
- **每个检查点**，d ∈ {zh2en, en2zh}：Q_d = dev5 文本 tcpBLEU-all，O_d = tcpBLEU-ov，L_d = dev5 文本 LAAL（现行协议，各方向自己的单位）。
- **可行**：两个方向都满足 L_d(s) ≤ L_d^ref + 0.1 s。第 1 段的 ref 是分开的 v8，第 2 段的 ref 是 A′ / (a) 选中的检查点。
- **联合信号**：J(s) = ½ (Q_zh2en(s) + Q_en2zh(s))，只对可行的 s 计算。各方向 BLEU 的尺度不同不影响排序：同一方向内是常数平移。
- **改进**：J(s) > best + ε，ε = 0.2 BLEU。依据：单方向 dev5 上相差 < 0.3 的排序接近随机，J 平均了两个方向，噪声约降为 1/√2。
- **停止**：s ≥ s_min = 3000（每个方向见过分开 run 一半的样本），且最近一次改进之后已经连续 2 个评估点（1000 步）没有改进。此时写 STOP 文件，记 `JOINT_EARLY_STOP <stage> <s>`。
  - 第 1 段：按 PID 停掉 Megatron（已存的检查点是完整的）；
  - 第 2 段：用 `--stop-file`。
- **欠拟合规则**：只要最近一次评估还在改进，就继续训练。上限 15000 只防失控：到了上限会记 `JOINT_CEILING`，并问协调者。
- **选择**：在决定停止时已经评估过的可行检查点里取 round(J, 2) 最高的。打平时依次比较：两个方向 held-out ce_label 的均值更低（仅第 2 段）→ O 的均值更高 → L − L^ref 的均值更低 → 步数更大。
  - STOP 之后才存的检查点不参加选择，只记录。
  - 某个检查点的代理同配置重试后仍失败：该点不参加，按"没有改进"计。
  - 没有任何可行点：不选择，记 `JOINT_STOP`，问协调者。
- **与分开 run 的差别和公平性**：
  - 分开的 run 是 3000 步余弦到底（fixed），这里是 12000 步余弦、早停。到第 3000 联合步时 lr ≈ 9.2e-5，到第 6000 步时约 5.5e-5，退火没有走完。
  - 公平性靠三点：报告选中的步数和每个方向实际见过的样本数；分开系统 (a) 的检查点同样由 dev5 规则选出（zh→en A′ 用的是最后一步 s3000，这一点照实报告）；第 6000 步以内两个方向见过的样本逐条与分开的 run 相同。
  - 不再加 WSD 衰减阶段：只一个设置。

## 6. 检查点大小

- 第 2 段每个检查点约 392 MB（lora 107 + extra 30 + optim 273），**保留 optim.pt**（用户决定）。每 500 步一个，最多 30 个，约 12 GB。
- 第 1 段的 Megatron 检查点原来不存优化器，约 66 MB；加了 `SAVE_OPTIM` 之后估计约 0.3–0.5 GB。
- 不存完整模型。只在选中后合并一次（66 GB）。磁盘还剩 21 TB。

## 7. 评测、门槛与结论写法

**dev 全集**（选中检查点；两个方向；与分开系统用同一条流水线、同样的两个延迟协议；**每条流语音 LAAL 放在最前**）：
- 第 1 段基线行：联合 v8 对分开 v8（各配恒等检查点），在 C2 gap-0.6 dev 全集上做文本解码：tcpBLEU all/ov/1spk，文本 LAAL（现行协议和 chunk 协议）。语音（rl + PL-EM，两个种子）只记录。
- 第 2 段方法行：联合 v2′ → 读级合并（zh→en `--sep ' '`，en→zh `--sep ''`；pair_lost ≤ 2 照用）→ 各方向的 PL-EM talker（zh→en 用现有的；en→zh 用 (a) 选中的，在它就绪之前先跑 zh→en）→ 两个 TTS 种子 → ASR-BLEU，语音 LAAL / EndOffset（两个协议）。参照是分开系统的 A′rl + PL-EM dev 输出。
- 比较：
  - 联合基线 对 分开 v8；
  - 联合方法 对 联合基线；
  - 联合方法 对 分开 A′rl + PL-EM。
- **门槛**（每个方向分开判）：
  - 文本 tcpBLEU-all、语音 ASR-BLEU（两个种子合并）都 ≥ 参照 − 0.5，用 `gate_2seed` 的点估计；
  - 同时报告按会议配对 bootstrap 的 P(差 ≤ 0)；
  - LAAL 只报告。
- **结论写法**（先写定）：
  - **持平**（两个方向都在 ±0.5 以内）：一个双向模型达到分开模型的水平，参数和部署减半；论文里回答审稿人"为什么分方向"。
  - **更好**（某个方向 ≥ +0.5，另一个方向不低于门槛）：报告双向迁移带来的增益。
  - **更差**（某个方向低于门槛）：如实写成局限，给出差值。主表仍用分开的模型，并说明分开模型的额外开销只有每个方向一份 LoRA。
- **test 只跑一次**（AMI zh→en、AMI en→zh、AliMeeting zh→en），在 `JOINT_DEV_DONE` 之后，配置全部在 dev 上定好。不用 COMET / MetricX。

## 8. GPU 计划与墙钟（布拉格时间）

- **启动条件**（写进 launcher）：
  - 终测完成：`chain_v2_ami.log` 和 `chain_v2_ali.log` 里都有 FT_ALL_DONE；
  - 卡 4–7 上只有占卡程序，或单说话人论文的基线评测（只推理，与训练峰值合计 ≤ 135 GiB）；
  - noclobber 占 slotB，用 takeover 接手，按 PID 停掉占卡程序；守护脚本 v3 不碰。
- **(a) 的冲突已改**（提交 2d7b53d58）：(a) 的 dev5 代理改为等 (a) 自己的训练退出（rc 0 时释放 slotA）之后，在 0–3 上跑（`--exclude 4 5 6 7`，`--cap135` 叠放）。选择规则、命令和输出都不变。等待进程按原环境重启：旧 PID 3192300 已按 PID 停掉，新 PID 3951662。
- **代理用的卡**：约 06:00 之前，0–3 被 (a) 训练占着、4–7 被 (b) 训练占着，没有空卡。held-out 损失也要整张卡（约 65 GiB 以上，和训练叠放会超过 135 GiB），所以也只能排队。之后代理在 0–3 上与 (a) 的评测叠放，几个检查点并行追赶。
- **墙钟**（4 卡；第 1 段每步约 5.0–5.5 s：8 卡是 2.5–3.3 s/步，4 卡要两次 micro-batch；第 2 段每步约 4.2–4.6 s：4 卡同类训练是 3.8–4.5 s/步，en→zh 的行更重）：

| 阶段 | 时长 | 说明 |
|---|---|---|
| 第 1 段训练 | 预计停在 6000–9000 步：8.5–13 h；上限 15000 步约 22 h | 每 500 步约 43 min |
| 第 1 段代理追赶，以及停止判定的滞后 | +0.5–1.5 h | 06:00 以后才有卡 |
| 合并 + 第 2 段启动 | 0.5 h | |
| 第 2 段训练 | 预计停在 4000–8000 步：5–10 h | 每 500 步约 37 min |
| dev 全集（两个方向，基线行和方法行） | 3–4 h | |
| test 一次 | 3–4 h | |

- 如果 (b) 在 23:30 左右启动：第 1 段约在 11/13 12:00–15:00 停止；第 2 段约在 11/14 凌晨停止；dev 约在 11/14 上午完成，test 约在 11/14 中午完成。

## 9. 实现状态（截至本文件；都在 CPU 上测过，没有试跑）

已完成并提交：
- `crf/train_crf_sft.py` 新增 `--sched2/--dir2`、`--stop-file`、`--lr-horizon`。默认路径不变：test_train_crf_sft 8/8、test_serial_v2p_train 8/8、test_heldout_loss 通过；新测试 `crf/test_joint_bidir.py` 5/5 通过。
- `pretrain_thinker_gstar --gstar-sequential`；`run_thinker_gstar.sh` 新增 `GSTAR_SEQUENTIAL`、`LR_DECAY_ITERS`、`SAVE_OPTIM`，默认都不变。
- `joint/build_joint_v8_rows.py`、`joint/megatron_to_crf_ckpt.py`、`joint/identity_ckpt.py`，各有测试。
- (a) 的代理改卡（上文）。

**还没写**（交给下一个执行者，规则都在本文件里定死了）：
- `scripts/joint_orch.py`：包装 thinker_t_eval_orch / en2zh_orch / final_test_orch。只替换 `--thinker` 的 base，另外给 zh→en 加 held-out 作业。按 `--dry` 检查命令，并写测试。
- `scripts/joint_select.py` 加测试：实现第 5 节的 stop / select。
- launcher：`joint_stage1.sh`（等启动条件 → 占 slotB → takeover → Megatron；每 500 步转换并排代理；早停后按 PID 停止 → 合并 → `JOINT_S1_SELECTED`）、`joint_stage2.sh`（`JOINT_TRAINING_STARTED` → 代理 → STOP 文件 → `JOINT_SELECTED` / `JOINT_EARLY_STOP`）、`joint_dev.sh`（`JOINT_DEV_DONE`）、`joint_test.sh`（`JOINT_TEST_DONE`）。遇到 `EN2ZH_STOP` / `EN2ZH_PROXY_FAILED` 时记 `JOINT_STOP` 并退出。
- 都写完、提交、只用 `bash -n` 和 DRY_RUN 检查过之后，用 setsid nohup 启动等待进程。

## 修正 1（2026-11-12 23:10 布拉格，执行者；在 (b) 的任何 GPU 作业或结果之前）

第 9 节的清单已实现（`scripts/joint_orch.py`、`joint_select.py`、`joint_lib.sh`、`joint_proxy.sh`、`joint_stage1.sh`、`joint_stage2.sh`、`joint_dev.sh`、`joint_test.sh`），只用单元测试、`bash -n` 和 DRY_RUN 检查。下面是与正文不同或正文没写明的地方。

1. **修了一个会出错的地方**：带优化器的 Megatron 检查点里，优化器张量的键是 `optimizer.state.<exp_avg|exp_avg_sq|fp32_param>.<模型键>`，同样以 `.a/.b`、`sd_proj.*`、`new_*` 结尾。原来的 `merge_thinker_gstar.py` 和 `joint/megatron_to_crf_ckpt.py` 会把它们也读进来，并覆盖真正的权重（不报错）。现在两者都跳过 `optimizer.` 开头的键；不带优化器的检查点结果不变。测试已扩展：带优化器的检查点合并、转换的结果与不带的逐位相同。
2. **第 1 段的优化器状态**（用户要求每个检查点都有）：
   - 代码核对：`--bf16` 时 Float16Module 把所有可训练参数（LoRA、投影器、新行）都转成 bf16，所以都在 `float16_groups` 里；模型的 sharded_state_dict 用 `keep_vars=True`，优化器能按 id 对上每个参数。这个模式（冻结主干、只存可训练部分、存优化器）在 si_model / phase_f 上用过，e2 也从带优化器的检查点续训过。
   - 本 run 不用 `--decoupled-lr`，所以记忆里那个续训时学习率互换的问题不适用；如果以后续训，仍要核对每组学习率。
   - 新工具 `joint/check_megatron_optim.py`（有测试）：第 500 步检查每个可训练张量都有 exp_avg、exp_avg_sq 且形状对，二阶矩不全为 0。不通过：按 PID 停训练，释放 slotB，记 `JOINT_STOP`，不退回到不存优化器（要用户决定）。
   - 第 2 段同样在第 500 步检查 `optim.pt`。
3. **不存完整模型**：`run_thinker_gstar.sh` 新增 `NO_MERGE=1`（默认不变），第 1 段结束时不再自动合并最后一步。只在选中后合并一次（与 2.3 一致）。
4. **规则中正文没写明的细节**（`joint_select.py`，有 11 个测试）：
   - 评估点按步数顺序，只看从 500 开始连续完成的前缀；
   - best = 最近一次改进时的 J（通常的 min-delta 早停：没达到 ε 的升高不移动 best）；
   - 没有任何改进时，最近一次改进按第 0 步计；
   - 选择的候选 = 决定停止时所有代理已完成、可行的检查点，包括前缀之后但在 STOP 之前存的；
   - 平局比较时，O 的均值按 2 位小数，L − L^ref 的均值按 3 位小数；第 2 段缺 held-out 的点按 +∞。
5. **代理重试**：同配置重试是用新的编排器、名字加 `_r2` 重跑。`--resume` 不会重跑已用完重试次数的作业，所以不用它。
6. **到上限**：记 `JOINT_CEILING` 之后也记 `JOINT_STOP` 并退出，不做选择，问协调者。
7. **停止**：
   - 第 1 段在没有存盘进行中时按 PID 停 torchrun（没有比计数器更新的 iter 目录时）；决定停止之后才存的 iteration 不再启动代理。
   - 第 2 段用 STOP 文件；STOP 之后存的那个检查点只记录。
8. **参照的 dev5 解码**（同一条流水线，第 1 段启动时就排上）：
   - 分开的 v8，两个方向，各配恒等检查点（第 1 段的延迟参照）；
   - A′ step_03000（第 2 段的 zh→en 参照）；
   - 第 2 段的 en→zh 参照是 (a) 的 `thinker_selection.json` 里的 `chosen_name`。
9. **卡**（用户 / 协调者 2026-11-12）：
   - (b) 的所有评测作业（参照、代理、dev）在 (a) 的 Thinker 训练退出之前都不上 0–3。
   - 训练期间 4–7 上只允许占卡程序，或单说话人论文基线评测：按进程或祖先进程的命令行认，与训练峰值合计 ≤ 135 GiB。第 1 段峰值按 75 GiB 计（8 卡 MBS 8 实测 63.6 GiB），第 2 段按 120 GiB 计（`--mem-cap-gib`）。
   - 先占 slotB，再等 4–7 清空：全局 exclude 生效后不会再有新作业进来，不会被饿死。第 2 段直接接手第 1 段的 slotB（改名释放后立即重新占）。
10. **读级合并**：pair_lost > 2 时，只在协调者 11-12 22:20 预授权的条件下继续，记 `JOINT_MERGE_DEVIATION`：
    - 丢的窗口全是已知机制（`merge_pairlost_check`）；
    - pair_lost ≤ 1%；
    - 副本完整。
    否则记 `JOINT_STOP`。
11. **dev 和 test 的范围**：
    - dev 上，联合基线对分开 v8 只比文本；联合基线的语音只记录。
    - test 跑两行（基线、方法），三个集合，每行都含语音，只跑一次。
    - 门槛（`gate_2seed`）比较方法对联合基线、方法对分开的 A′rl + PL-EM；文本差在 summary json 里。
12. **zh→en dev5 的 chunk 协议窗口**：从 `detector_v1/windows_chunk_v1` 切出 dev5 的 1092 个窗口，放在 `joint/windows_zh2en_chunk_dev5`，只用于记录。
13. **标记行**：
    - `JOINT_TRAINING_STARTED s1|s2 <run>`
    - `JOINT_EARLY_STOP s1|s2 <决定步>`
    - `JOINT_SELECTED s1|s2 <步>`（同时写标记文件 `logs/mist_w3po_v1/joint/JOINT_S1_SELECTED` / `JOINT_S2_SELECTED`）
    - `JOINT_DEV_DONE`、`JOINT_TEST_DONE`
    - `JOINT_STOP <段> <原因>`、`JOINT_CEILING`、`JOINT_MERGE_DEVIATION`


## 修正 2（2026-11-13 07:20 布拉格，执行者；只改卡的放置和速查顺序，不改选择规则、任何门槛或解码配置）

用户（约 06:50、06:55 布拉格）："都改成叠加跑dev5速查"；"0-3叠加第一个训练的dev5速查，4-7叠加第二个训练的dev5速查"。

1. **放置（已实现，提交 26ac3caf3）**：`joint_proxy.sh` 在 `logs/mist_w3po_v1/joint/<run>/proxy_stack.conf` 存在时，给两个方向的编排器加 `--stack-cards logs/mist_w3po_v1/slots/stack_cards_b47 --stack-rank <步> --stack-group <run>/stack_group`（`stack_cards.py`）。文件存在时，分片解码只放 4–7，每卡最多 1 个；按 nvidia-smi 实际占用（180 s 内最大值）+ 64 GiB ≤ min(135, total − 5.4)，在 gpu_launch.lock 下复核。其他作业照旧。文件不存在时，行为与修正前完全相同。
2. **顺序**：新检查点优先，旧检查点有空位再补。参照解码 v8e（第 1 段 L^ref 要用）排最前，A′ 参照（只供第 2 段）排最后。`<run>/stack_group/order` 写 `oldest` 就能改成旧的优先，不用重启。注意：§9 修正 1 第 4 条的规则只看从 500 开始连续完成的前缀，所以新检查点先出结果并不能让早停提前判定。
3. **held-out**：第 1 段的代理本来就没有 held-out 作业（§2.3）。第 2 段的 held-out 仍排除 4–7（它约 65 GiB，叠上会超过 135 GiB），0–3 有空时才放。它只用于打平，不阻塞早停判断。
4. **看门狗** `stack47_watchdog.py`：
   - 触发条件：任一卡超过 135 GiB；训练步时连续 3 个日志点超过基线 4650 ms 的 1.4 倍；训练日志报错。
   - 触发后：把 stack 文件改名，按 PID 停掉 4–7 上的解码，记 `STACK47_WITHDRAWN`。训练不动。
5. **实测结果（07:10–07:11 布拉格）**：
   - 4 个 z_s03000 分片放上 4–7。启动后每个解码占 66.2–66.5 GiB，训练本身占 68.4–70.3 GiB，卡 6、7 合计 135.3–135.4 GiB，超过 135。看门狗在第一个解码上卡后约 26 s 撤卡（`stack_cards_b47.withdrawn_20261113T061116Z`）。
   - 训练步时没有变化（4770、4803 ms），没有报错。
   - 这 4 个分片各用掉 1 次重试（rc 143），按原规则回到 0–3 排队。
   - 结论：按现有解码配置（batch 16），4–7 每卡只剩 64.7–66.6 GiB 预算，放不下一个解码，严格 135 GiB 线下不能叠。是否放宽上限，或改用更小的解码占用，要用户决定。在此之前 stack 文件保持撤下，(b) 的速查按修正前的规则放在 0–3。
6. **用户批准**（2026-11-13 约 07:35 布拉格）：4–7 的上限放宽到 138 GiB（每卡仍最多 1 个解码），看门狗另加一条：物理剩余少于 1.5 GiB 也撤卡；顺序改为旧检查点优先（`stack_group/order` = `oldest`），v8e 参照仍排最前。看门狗重启为 PID 1279715（`--cap-gib 138 --min-free-gib 1.5`，日志 `stack47_watchdog_v2.log`）。
7. **卡 4 撤出，只叠 5–7**（协调者 2026-11-13 约 07:25 布拉格，在用户批准的 138 GiB 范围内）：07:24 卡 4 上训练瞬时到 72.2 GiB，加一个解码到了 138.0 GiB，第二次撤卡。卡 4 是训练的 rank 0，以后不再叠。看门狗 v3 只管 5、6、7（上限 138 GiB，物理剩余 ≥ 1.5 GiB，每卡 1 个解码），日志 `stack47_watchdog_v3.log`。stack 文件先建成空文件，90 s 后再写入卡号（`--warmup-s 90`），让各编排器先报上自己的待放数，旧检查点优先从第一次放置起就生效。


## 修正 3（2026-11-13 07:47 布拉格（服务器时钟 06:47 UTC），执行者；在任何第 1 段 dev 全集结果之前：第 1 段尚未选中，`JOINT_S1_SELECTED` 不存在，`jdev_*` 一个都没有）

用户（2026-11-13 早晨，协调者转述）："不能用pl-em talker吧，需要用原始的之前单说话人同传的talker才算是基线吧"。

1. **第 1 段基线行的语音**（改 §7 第 112 行、修正 1 第 11 条里"语音（rl + PL-EM，两个种子）只记录"）：
   - talker 改为**原始的单说话人同传 talker T1c@1500**（`eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss`），冻结；
   - 在第 1 段的**原始文本解码**上直接出语音（`jdev_b{z,e}_<rt>` → `<名>__t1c`、`<名>__t1c_tts2`）：**不做读级合块（rl），不用 PL-EM**，也不用任何多说话人线新加的 talker 改进；
   - en→zh 不再等 (a) 的 PL-EM talker。
2. **T1c 的配置 = T1c 的原始默认设置**：`crf_pair_dev_orch.speech` 的命令逐字不变，即 `si_infer --model <T1c> --window 1 --text-policy file --speed-policy fixed --history-ban 8 --max-model-len 32768`，prep 为 `gstar_talker_v6 prep --causal --blocks --preset lowlat`（zh→en `--session zh_mix_v8 --diar nemotron_lowlat_zh8 --sep ' '`；en→zh 由 en2zh_orch 换成 `--session ihm_mix --diar nemotron_lowlat --sep ''`、`--direction en2zh`，T1c 输出中文），`drop_long_streams 400`。不起 CTC 打分服务。
   - **c1 / b1 / s2 不用**。查证来历：三者都是多说话人这条线 2026-11-10 的"语音缺口计划"（`text_to_speech_gap_plan.zh-CN.md` §5，`speech_var_orch.py`，提交 ec8062961 / 730452748 / c41e4ad87）在 C0-fp 文本上新加的 talker 改进：c1 = `--speed-policy queue`（替换 fixed），b1 = `--select-k 2` + 独立 wav2vec2 CTC 打分选候选，s2 = `--speed-shift 2`。之后它们才成为 A′ 的 talker 设置。
   - 以前单向 v8 的语音**都不是**这么配的：dev 上 v8 参考、"v8 + finalpost"（语音 19.17，`metrics_first_si_analysis` §4.1）、`c0_pair_dev_eval`、`serial_v2p_dev_eval` 里的 v8 行都走 `finalpost_dev_orch` / `make_eval_jobs` 的链，即 T1c + `--speed-policy fixed --history-ban 8`；MIST v7（单说话人论文同一 talker）的 zh→en 和 en→zh 语音作业（`reports/gstar_si_v7/jobs/sp_*`）也是同一组参数。所以 fixed + history-ban 8 既是 T1c 的默认，也与以前 v8 的语音一致。
3. **单向 v8 参照的语音行**：分开的 v8（`thinker_gstar_s2dl_zh2en_3k_20261030`、`thinker_gstar_s2dl_20261029b`，各配恒等检查点）在 C2 gap-0.6 dev 全集上本来只做文本（`jdev_v8sep_{z,e}`）。c2g06 dev 全集上没有任何 v8 的现成语音（19.17 那一行是旧窗口 `windows_dev_online`，不能比），所以**补跑**：在它们的文本上用**完全相同**的 T1c 链出语音（`jdev_v8sep_{z,e}__t1c[_tts2]`）。只推理。
4. **完全相同**：联合基线和两个单向 v8 的语音用同一 talker 和参数、同一窗口（zh→en `windows_zh2en_c2g06_v1/windows_dev_online.jsonl`，en→zh `windows_en2zh_c2g06_v1/windows_dev_online.jsonl`）、同一解码窗口（1.0 s 读）、同样两个 TTS 种子（种子 1 = talker 默认；种子 2 = `--seed 20261109`）、同一 ASR 和打分。
5. **打分口径不变**：ASR-BLEU（tcpBLEU all / ov，两个种子），每条流语音 LAAL 现行协议（编排器的 lat 作业）和 chunk 协议（`chunk_speech`），EndOffset 作补充。
6. **语音行仍是描述性对比**：联合 v8 对分开 v8 只比文本，语音只记录。另记一个描述性的 `gate_2seed`（联合 v8 T1c 对分开 v8 T1c，`gate_vs_separate_v8_t1c.json`），不是门槛。**不改任何门槛数值、选择规则或解码配置。**
   - 由此带来的一处变化，照实写明：§7 "联合方法 对 联合基线" 的比较（`gate_2seed`，规则 ≥ 参照 − 0.5 不变）里，联合基线的语音现在就是它的 T1c 语音（基线行的定义变了，比较对象跟着变）。于是这一比较的语音部分衡量的是"整套方法（v2′ + rl + PL-EM）对纯基线"。文本部分不变。
7. **其他地方的检查**：
   - 第 2 段方法行（联合 v2′ → rl → 各方向 PL-EM）和"联合方法 对 分开 A′rl + PL-EM"：PL-EM 和 rl 在那里是方法本身或分开系统的方法，**不改**。
   - 第 1、2 段的 dev5 代理和延迟参照只做文本，不涉及 talker，不改。
   - **test**（修正 1 第 11 条"每行都含语音"）：基线行（`jtest_b{ami,ali,en}_<rt>`）的语音同样改成 T1c（`__t1c[_tts2]`，不做 rl、不用 PL-EM），test 上的"方法 对 联合基线"同第 6 条。方法行不变。test 上不加分开 v8 行（原计划没有）。
8. **实现**：`thinker_t_eval_orch.py --talker t1c`（新选项；其余选项逐位不变）；`joint_dev.sh` / `joint_test.sh` 用新开关 `BASE_TALKER=t1c`（默认 `plem` = 修正前，逐位不变）。两个等待进程按原环境加 `BASE_TALKER=t1c` 重启。只用单元测试、`bash -n` 和 DRY_RUN 检查。
9. **放置**：这些都是推理作业，按现有规则：dev / test 的作业照旧排除 4–7，在 0–3 上按 `--cap135` 叠放（≤ 135 GiB）。分开 v8 的文本解码按原计划在第 1 段选中后才开（现在 0–3、5–7 被两个实验的 dev5 速查占满，提前开会和速查抢卡），文本一完成就接 T1c 语音，不等方法行。
10. **实施记录**（07:51 布拉格）：脚本改动见提交（`thinker_t_eval_orch.py`、`joint_dev.sh`、`joint_test.sh`、`test_joint_orch.py` 9/9）。默认路径逐位不变：编排器 `--dry` 输出（none / c1b1s2 / plem / speech-from、en2zh、final_test）、两个脚本的 DRY_RUN 输出、dev 的 summary 代码与改前逐字相同；test 的 summary 代码只把后缀改成变量，默认取值相同。等待进程：dev 564748 → 1594999，test 564751 → 1595000，按 PID 停，用原 environ 加 `BASE_TALKER=t1c` 重启。

## 修正 4（2026-11-13 08:40 布拉格（服务器时钟 07:40 UTC），执行者；在新第 1 段 run 的任何训练步之前、在第 2 段任何结果之前）

用户决定（布拉格约 08:10–08:45，协调者转达）：第 1 段按 H6000 从头重训、8 卡、不做 dev5；第 2 段严格照单向 A′ 配方，只换双向数据，8 卡，dev5 从第 3000 步起叠在训练卡上，早停照原规则；暂停单向 (a) 和 v8 测试集解码，双向优先。

**第 1 段（严格双向基线）**

1. **重训原因**：旧 run `joint_v8_20261112T220626Z` 的学习率按 H12000 余弦，到第 6000 步时 lr 还是 5.53e-5，没退到 1e-5，和单向 v8（3000 步余弦到底）不对应。用户先选"接着训到 6000"，后确认学习率有问题，改为 H6000 从头重训。旧 run 在约第 4330 步按 PID 停掉，检查点 iter 500–4000 和它的 dev5 速查输出全部保留，**废弃，只作记录**，不参加任何选择或结论。
2. **配方**（与单向 v8 `thinker_gstar_s2dl_zh2en_3k_20261030` / `thinker_gstar_s2dl_20261029b` 逐项对应）：从 O2 起训，`GSTAR_SEQUENTIAL=1`，rows `e2e_gstar_stream_bidir_v2`（每个全局 batch = 32 条 zh→en + 32 条 en→zh，各自按单向 run 的原顺序），TRAIN_ITERS 6000，LR_DECAY_ITERS 6000，cosine，lr 1e-4 → min_lr 1e-5，**warm-up 100**，种子 20261026，LoRA r32 / α64，其余参数同一个启动脚本（`run_thinker_gstar.sh`），每 500 步存检查点（带优化器）。
   - warm-up 100 和 H6000 的理由：联合第 2k 步时每个方向见过的样本 = 单向第 k 步；lr_joint(2k) = lr_v8(k)（warm-up 段 1e-4·2k/100 = 1e-4·k/50；余弦段 (2k−100)/(6000−100) = (k−50)/(3000−50)）。
3. **8 卡的 GBS 与学习率**：NPROC 8 × MBS 8 = GBS 64，不做梯度累积，与单向 v8（DP 8 × MBS 8）完全相同，学习率不改。核对：`--dataloader-type single` 用 `MegatronPretrainingSampler`，样本按 consumed_samples 顺序取，第 t 个全局 batch 恒为样本 [64t, 64t+64)，每个 DP rank 取其中连续的 MBS 条；所以数据顺序只取决于 GBS 和行文件（`--gstar-sequential` 下即文件顺序），与 DP 数无关。旧 run 的日志显示 `use_distributed_optimizer False`；新 run 同一启动脚本，也不用 distributed optimizer。损失是各 micro-batch 均值的平均，DP 8 无累积与 DP 4 × 累积 2 是同一组 8 个 micro-batch，数值语义相同（只差浮点求和顺序）。
4. **选择**：不做 dev5 速查、不早停、不用 J；最终模型就是 iter 6000（与单向 v8 用最后一步一致）。第 500 步照旧做优化器检查（第 6000 步再查一次，只记录）。训练在 6000 前退出 → `JOINT_STOP`。训完合并 iter 6000（`merge_thinker_gstar` 到 O2）+ identity 检查点，写 `JOINT_S1_SELECTED`（格式不变，zname / ename 为 "-"），触发第 2 段开训和基线行 dev 评测（修正 3：冻结 T1c；分开 v8 用同一 T1c 链）。
5. **占卡**：slotA、slotB 都占（noclobber，说明里写"用户 2026-11-13 08:30 决定 8 卡"），takeover 只按 PID 停占卡程序；第 2 段接手两个槽。
6. **实现**：`joint_stage1.sh` 新开关 `S1_FULL=1`（默认空 = 修正前，DRY_RUN 输出逐位相同）。

**第 2 段（方法）**

7. **配方**：serial_v2p、τ 0.5、final-read-full-post 1、扰动 0.2、LoRA r32/α64、lr 1e-4 → 1e-5、AdamW (0.9, 0.95)、种子 20261109、rows_per_step 64、TOK 32000、CAP 120 等与 `crf_v2p_20261110T075131Z` 的 args.json 逐项相同；变化只有：双向数据（`--sched2 sched_en2zh_v1 --dir2 en2zh`），`--lr-horizon 6000`，`--warmup 100`，`--seed2 20261109`（en→zh 数据流与 (a) 同一个 plan 种子；train_crf_sft 新选项，默认 None = seed + 1，计划和 args.json 逐位不变，提交 d8d138506），上限 `--steps 15000`。
   - 理由同第 2 条：每步 ≈ 32 条 zh→en + 32 条 en→zh 读行，联合第 2k 步时每个方向见过的读行与单向第 k 步相同（窗口流同一排列，只是切分不同），lr 曲线逐点对应单向 H3000。
   - **6000 步之后**：`lr_at` 里 ratio = min(1, (k − warm-up)/(H − warm-up))，超过 horizon 后 lr 恒为 min_lr 1e-5（单元测试 `test_lr_horizon` 覆盖）。
8. **卡**：8 卡（0–7）。全局 batch = 每步 64 读行，与卡数无关（`plan_steps` 先定每步的窗口，再按卡均分），与 v2′（8 卡）相同；学习率不改。
9. **早停与选择**（替换原 §5 对第 2 段的评估起点；其余照 §5）：
   - 第 3000 步之前不做速查。评估点 3000, 3500, 4000, …（每 500 步）；
   - 可行：每个方向 dev5 文本 LAAL ≤ L^ref + 0.1 s，参照照原登记：zh→en 为 A′ s3000 的 dev5 解码，en→zh 为 (a) 选出的第 2000 步（`dev5_en2zh_v2p_20261112T204539Z_s02000`）；
   - J = 两个方向 dev5 文本 tcpBLEU-all 的平均；J > best + 0.2 才算改进；从 3000 起的连续前缀里，最近一次改进之后连续 2 个评估点（1000 步）没有改进就停训（STOP 文件，`JOINT_EARLY_STOP s2 <s>`）；还在改进就继续训，可以超过 6000，上限 15000（`JOINT_CEILING`）；
   - 选择：停止时已评的可行点里 round(J, 2) 最高；打平：held-out 均值 → O → L − L^ref → 步数（同 §5）。
   - 这与单向 v2′ 的"跑满 3000 步取最后一步"不同，是用户的选择（"如果underfitting本身训练就不停，直到dev5没效果了就停止训练"）。
   - 实现：`joint_select.py` 新选项 `--first-step 3000`（默认 500 = 修正前）。A′ 参照的 dev5 解码在第 1 段旧驱动停掉时尚未完成，第 2 段驱动在第一次速查时补跑（新名字，不覆盖旧输出）。
10. **dev5 叠放**（用户："8卡训练不改batch 直接尝试dev5速查，如果所有显存放不下再说"）：
    - 速查直接叠在 8 张训练卡上，每卡最多 1 个解码，上限按物理显存：实际占用（180 s 内最大值）+ 64 GiB ≤ 总显存 − 1.5 GiB（不再用 135 / 138 GiB 软上限）；
    - 看门狗：任何卡物理剩余 < 1.5 GiB，或训练日志出现 OOM / Traceback / CUDA / NCCL 错误 → 撤 stack 文件，只按 PID 停解码，绝不碰训练；
    - 错开：一次只在一张卡上启动一个解码，间隔 ≥ 60 s；
    - 先在一张卡上试一个解码，实测训练 + 解码的峰值，可行再铺开到 8 卡；
    - 放不下（没有卡满足上限）或看门狗跳闸 / 训练 OOM：不再试第二次、不改 batch，记 `JOINT_STACK_FAILED`，交用户决定。
11. **talker** 不变：zh→en 用 `talker_plem_20261111T001759Z`，en→zh 用 (a) 选出的 PL-EM export。

**暂停与顺序**

12. 暂停（只按 PID 停，什么都不删）：(a) 的 after_select / en2zh_test 等待进程、dev 解码、train16 解码、采样编排器及其解码；v8 测试集解码（final_test_v8_launch 和 ali / ami 编排器及解码）。每个 PID 的 cmdline、environ、cwd 记在 `logs/mist_w3po_v1/paused_20261113/paused.tsv`，进度在 `progress.txt`，以后用原环境续跑。第 1 段旧 run 的 dev5 速查、A′ 参照解码（未完成）和看门狗一并停掉，stack 文件撤下。
13. 第 2 段训完、dev5 选完后：0–3 恢复 (a) 的 en→zh PL-EM 路径（train16 解码 → 采样 → rows → PL-EM → 选 export），其余卡做方法行的 rl 和 zh→en 语音；en→zh PL-EM 就绪后补方法行 en→zh 语音，再出 dev 全集结论，最后测试集。(a) 的 dev / test 和 v8 测试集解码等双向卡空时再续跑。

**笔误改正**

14. §5"与分开 run 的差别"里"到第 3000 联合步时 lr ≈ 9.2e-5"应为 **8.71e-5**（H12000、warm-up 50 的余弦；第 6000 步 5.53e-5 无误）。该段描述的是已废弃的旧 run。
