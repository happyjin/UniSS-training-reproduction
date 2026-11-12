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
