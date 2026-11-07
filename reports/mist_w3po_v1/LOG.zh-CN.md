# MIST W³PO 执行日志（决策与偏离）

方案：`docs/mist_grpo_plan.zh-CN.md`（修订 1、2 优先）。所有新代码在 `experiments/mist_w3po_v1/`，输出在
`eval_outputs/mist_w3po_v1/`、`logs/mist_w3po_v1/`、`reports/mist_w3po_v1/`；临时文件在
`/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/`（2026-11-06 用户新规：不写 /opt/dlami/nvme/jasonleeeli 以外的任何路径）。

## 2026-11-06

### 环境与资源核查
- GPU：0–3、7 空闲；4–6 有 kenxsong 的作业（约 43–45 GB/卡），不动、不让卡。
- **COMET / MetricX 本地不存在**（`evaluation_models/` 只有 AutoPCP、wav2vec2、wavlm、whisper 等；`comet` 包在
  qwen3omni 与 uniss-eval 环境都没有）。按任务说明不擅自下载大模型，STAMP 质量项 Q 退回 **chrF++ 与句级 BLEU 的平均**
  （修订 2 的 min(COMET, MetricX) 项暂缺），最终报告中说明。
- `data/ami_multispk_si_v1/zh_revoice_v1/talker`（Paraformer 复核 ASR）在新服务器上不存在，S3 备份里也没有。
  v8 构建的 `e2e_gstar_zh2en_v1/summary.json` 只有 `utt_ok / utt_removed` 两类，`zh_mix_v8/audit.json` 的 bad 为空，
  说明 v8 时所有未被剔除的句子都通过了 CER 闸门；因此新窗口构建跳过 CER 闸门（其余闸门不变），结果与 v8 一致。

### 评测协议（dev）
- AMI dev = validation 中有参考的 15 场会议（3 场因 A8 无参考，跳过），整场在线窗口 3050 个、5.69 h，
  规则与 test 完全相同（`reports/gstar_si_v6/build_online_windows.py`，zh_mix_v8 + nemotron_lowlat_zh8）。
  构建：`experiments/mist_w3po_v1/data/build_windows_zh2en.py` → `eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/`。
- 打分：`reports/gstar_si_v8/overlap_score.py`（tcpBLEU all/ov/1spk、CR、重复、短插话召回、延迟），参考为
  `e2e_gstar_zh2en_v1/windows_validation_ref.jsonl`，universe 为 zh_mix_v8 utts。
- 训练会议在线窗口 28106 个（127 场），其中 23742 个满足"重叠句子 ≥0.9 在窗内、无闸门失败"，作为 S0-3 训练窗口与
  RL rollout 窗口（`windows_train_diar.jsonl`，每句带会议级槽位 slot_sess 与窗口级匈牙利槽位 slot_win；7.8% 的句子两者不同）。

### S0-4 跨窗口上下文的实现
- 上下文 = 同一会议上一个窗口最后 N 段已提交译文，格式 `" Already translated just before this audio (do not translate it
  again): <|spk k|>text ..."`，放在用户轮指令之后；空上下文时提示与 v8 完全相同（代码中断言）。
- 偏离：上一个窗口的"最终读取"在真实系统里由关窗检测触发，最坏情况（20 s 截断窗）比下一个窗口第一次读取晚 ≤0.24 s；
  解码器用上一个窗口的完整输出作上下文，即这 ≤0.24 s 的超前。影响极小，如实记录。

### 闸门判定规则（在看到任何 S0 / RL 结果之前写定）
- 对照：同一 dev 协议下的 v8 Simul SFT（`thinker_gstar_s2dl_zh2en_3k_20261030`，带 2 段跨窗口上下文，修订 2）；
  另报不带上下文的 v8（原协议）作参考。
- 指标：文本 tcpBLEU（整体、重叠区）与语音 ASR-tcpBLEU（整体），`overlap_score.py` online 模式。
- "不低于"的判定：点估计 ≥ 对照 − 0.2，且按会议配对 bootstrap（10k 次，`evaluation/compare.py`）
  P(差值 ≤ 0) < 0.95（即不显著变差）。三项都满足才算通过；任何一项不满足，该阶段 / 该项不进入下一步。

### 工具（已提交）
- `core/engine.py`：读取步（与 v6s 逐字节一致：12 个 test 窗口、batch 12，原解码器与新解码器输出 12/12 完全相同）。
- `rl/stamp.py` + `rl/test_stamp.py`：STAMP 奖励与手工单元测试（完美输出、置换不变、分块、重复、幻觉、漏译回溯、
  说话人错标与 RBC、语言门控、零长段、空输出、FLAD 支配判定、延迟），全部通过。

### GPU 实测（独立解码分片）
- 单卡整模型（max-gib 120）约 62–64 GB/进程。空闲卡（7 号）与 kenxsong 占用卡（4 号，util 99%）上的 dev 解码分片
  前 25 分钟进度相近（254 vs 259 个窗口，后者晚启动 60 s），即对这种 kernel-launch 受限的解码，占用卡几乎不减速；
  因此独立分片两类卡都用。空闲卡上可叠两个 62 GB 解码进程（约 128 GB，余量 > 8 GiB）。

### RL 冒烟测试（2026-11-06 19:14–19:25 UTC，7 号卡，单进程，v8 模型作底座，2 窗口/步，G=4）
- `rl/train_w3po.py` 端到端跑通：rollout → 锚点 → STAMP → W³ 优势 → 投影比率 / GSPO / KL → 更新 → 存档 → 续跑
  （从 step_3 续到 step_5，正常）。
- 首版锚点（lag 策略每次读取生成到结尾）2 个窗口要 190 s；加"出现晚于 t−lag 的时间 token 即停止"后降到 7–16 s
  （被丢弃的续写本来就不会被 lag_cut 提交，提交结果不变，除非出现结束早于开始的畸形段）。
- CPU 检查（`rl/test_pipeline_cpu.py`，真实 DAgger 解码构造的组）：去掉一段 / 全部重复 / 说话人整体轮换三种变体，
  STAMP 与所有模式的优势都按预期排序；整体轮换说话人槽位在窗口内置换不变（R 相同），这是窗口级映射的设计后果，
  跨窗口一致性只靠 RBC 的局部性项与 KL 约束，需在 dev 的会议级指标上监控。

### 2026-11-06 20:05 UTC：作业队列与时间表
- 新增显存感知队列 `scripts/gpuq_w3.py`（按 nvidia-smi 实际空闲显存起作业，每卡留 ≥8 GiB，GPU 作业间隔 60 s；
  多卡作业就绪但放不下时，其可用卡对后续作业"排空"）。评测拆成独立作业（`scripts/make_eval_jobs.py`：
  解码分片 66 GB/卡 → 打分 CPU → 语音 prep → 8 个 talker 26 GB → ASR 14 GB → 语音打分）。队列目录
  `logs/mist_w3po_v1/queue/`。
- 不能并行的部分：S0 SFT 每卡约 74–80 GB，空闲卡再叠一个 66 GB 解码会低于 8 GiB 余量，所以 SFT 期间只有
  kenxsong 的卡上还有约 20 GB 余量（只够 talker/ASR）。为让 SFT 尽早开始，停掉了刚启动的 test v8 解码和
  AliMeeting v8 解码（我自己的进程，按 PID），二者在 SFT 之后重新排队。
- 实测速度：dev 整场在线解码（3050 窗口）4 个分片、每片 ≤4 条会议通道约 75 min（无上下文、共享队列 40 min）；
  语音 talker 8 片约 12 min，ASR 约 20–30 min；v8 SFT 2.49 s/步（8 卡，3000 步 ≈ 2.1 h），合并约 10 min；
  RL 冒烟（单卡，2 窗口×G4）rollout 10–15 s、锚点 7–16 s、更新 4–9 s。按此外推：8 卡、64 窗口×G8 的一步约 3–4 min。

| 步骤 | 依赖 | 预计（UTC） |
|---|---|---|
| S0 数据（6 套行，CPU 并行） | DAgger 解码收尾 | 11-06 20:20 |
| SFT 吞吐基准（8 卡混合 vs 4 空闲卡，各 25 步） | 数据、排空 | 20:30–20:50 |
| S0-all SFT（3000 步）+ 合并 | 基准定布局 | 20:50 → 11-07 00:00 |
| S0-all dev 评测（文本 + 语音）与闸门 | 合并 | 11-07 01:30 |
| 5 个留一消融 SFT + dev 评测（可与 RL 小测分卡并行） | 闸门 | 11-07 01:30 → 约 16:00 |
| RL 50 步小测（ACR、方差、吞吐） | S0-all 过闸门 | 11-07 01:30 → 04:30 |
| 小测检查点 dev 评测 = 第一个"W³PO 是否有效"信号 | 小测 | 11-07 约 06:00 |
| W³PO 主训练（约 300 步，单种子） | 消融结论（若某项被剔除需换底座） | 11-07 16:00 → 11-08 08:00 |
| 对照 RL (a)–(d) 与 C1–C4 消融（各约 150 步，4 卡×2 并行） | 主训练之后或分卡并行 | 11-08 → 11-10 |
| AMI test + AliMeeting 评测、bootstrap、H1–H7 | 全部模型 | 11-10 → 11-11 |
| Zh→En 里程碑报告 | | 约 11-11 12:00 |

### 方案修订 3（用户决定，2026-11-06）
- 全程不用 COMET、不用 MetricX：STAMP 质量项 Q = (chrF++ + 句级 BLEU)/2；评测也不报 COMET；不下载这两个模型。
- 若小测显示 chrF++/BLEU 在组内区分不开 rollout，就测量并报告 ACR / 奖励方差，不自行加入神经指标。
- 其它不变：所有 8 张卡（含 kenxsong 占用的 4–6）都按实际空闲显存叠作业，每卡留 ≥8 GiB；持续监控 kenxsong 显存增长，
  不让我的作业把任何卡推到 OOM。Zh→En 证明 W³PO 有效后再按同样流程做 En→Zh；无效则停止并给出诊断。

### S0 数据（2026-11-06 20:08 UTC，`eval_outputs/mist_w3po_v1/data/rows_s0_{all,no1..no5}`）
- 6 套都按 v8 的读取行预算 130,560 行 + 20% 离线 + 15% 单人 = 160,957 行。
- **偏离**：DAgger 自前缀行池只有约 2.19 万行（14 场整会 3273 个在线窗口 + 1500 个参考窗口的 v8 自解码），
  占读取行 16.7%，低于方案的 30%；为不拖慢关键路径，先按 16.7% 训练，留一消融 no2 检验其作用。
  其中 64% 的 DAgger 行前缀与金标前缀不同（14,042 / 21,867），18 个读取因前缀停在未闭合段内被跳过。
- 带上下文的行 11.85 万（no4 为 0）。

### 结束时的 GPU 处理（用户新指令，取代"结束后空卡"）
全部工作（Zh→En 里程碑，及 En→Zh 若进行）完成后才执行：按 PID 停掉自己所有作业；在 kenxsong 未用的卡（届时用
nvidia-smi 复核，今日为 0,1,2,3,7）上从仓库根目录启动
`setsid nohup bash scripts/gpu_load/run_mist_workload.sh --devices 0,1,2,3,7 --wait-idle --max-used-mib 0 > logs/mist_w3po_v1/workload_after.log 2>&1 &`；
约 5 分钟后确认 5 张卡都接近满显存运行，并写入最终报告。绝不提前启动，绝不放到 4–6 号卡。

## 2026-11-06（接手执行者）

### 20:16–20:35 UTC：接手后的队列修正
- dev_v8_ctx2 语音：talk7 失败且无重试、talk5b 三次都因 vLLM 显存 profiling 断言失败（kenxsong 的 vlm_tta 作业在
  所有 8 张卡上频繁起停，邻居释放显存触发 `init_free_memory >= free_gpu_memory`），下游 asr/sscore 被跳过。
  补 talk7c、talk5d（重试 20 次）与新的 asrd/sscored；`make_eval_jobs.py` 的 talker 重试改为 20 次，ASR 作业先确认
  8 个 part 文件都在。
- **SFT 布局实测**（25 步，GBS 64 不变；基准 warmup 改 5 步，否则 Megatron 断言 warmup<decay）：
  - 4 卡（0–3）MBS 8：4.8 s/步，单卡峰值 reserved 64.6 GB（含上下文约 68 GB，队列按 72 GB 计）；
  - 8 卡 MBS 4：7.5 s/步（更慢），且单卡约 66 GB，叠在 kenxsong 卡上把 5 号卡压到只剩 3 GB 空闲 → 立即按 PID 停掉；
  - 8 卡 MBS 8（约 70 GB/卡）在 4–6 号卡上会使 kenxsong 作业有 OOM 风险（其用量在 60–74 GB 间波动），启动后立即
    按 PID 停掉，未测速。
  - **结论：S0 SFT 用 4 卡 MBS 8 布局（只在 0,1,2,3,7 上）**，3000 步约 4 h；kenxsong 卡上不放训练。
- 偏离：原计划的 8 卡布局在 kenxsong 当前负载下不可行（不是速度问题，是显存安全）。

### 20:35 UTC：用户命令——我方 5 张卡（0,1,2,3,7）预留守护
- 背景：kenxsong 的调度器（他机 ssh 进来，`coordinator.py`）在某卡空闲显存连续 3 次（间隔 20 s）≥ 90000 MiB 且
  util ≤ 75% 时放作业（execute_job 还要求 ≥ 88000 MiB）；20:00 左右我方作业交接时 0–3 号卡空了约 1 分钟，被他占上。
- 实现：`experiments/mist_w3po_v1/scripts/gpu_keeper.py`，每卡一个独立进程（setsid nohup），用 2 GiB 的普通 CUDA
  块把该卡真实空闲显存压在 80000 MiB 以下（空闲 ≥ 78000 MiB 时才加块）；空闲 < 12 GiB 时立即释放块，不会挤爆任何人；
  状态写 `logs/mist_w3po_v1/queue/keeper/card<k>.state`（持有量、空闲量），日志 `card<k>.log`。
- `gpuq_w3.py` 联动：keeper 持有的显存算作我方可用；在某卡起作业前写 `card<k>.req` 让 keeper 释放到
  need + 8 GiB（再 +1 GB）的真实空闲，nvidia-smi 确认后才启动（最多等 150 s，否则撤销请求、稍后重试）；启动后
  180 s 内 keeper 给新作业保留"尚未分配的那部分"空间（随作业增长递减），其余照常补满，所以交接和作业启动期间
  该卡也不会出现 ≥ 88 GB 空闲。（第一版启动后改为"180 s 内不补"，实测 SFT 加载期间 0–3 号卡空闲到 ~100 GB，
  立即改为上述"只保留未分配空间"的版本并替换进程：先起新 keeper，再按 PID 停旧 keeper。）
- 不碰 kenxsong 的任何进程；4–6 号卡不放 keeper。他已在 0–3 号卡上的作业自然结束后，那部分显存由 keeper 接管。
- 结束时：按 PID 逐卡停 keeper 并立即起用户的 workload。**注意**：`run_mist_workload.sh --wait-idle` 要求所选卡
  无任何进程且已用 ≤ 1000 MiB 并持续 60 s；若届时 kenxsong 的作业仍在 0–3/7 上，workload 会一直等待（不是死锁，
  但这段时间该卡没有 keeper），需要届时按实际情况处理并报告。
- 当前进程：keeper PID 1035049(卡0) 1035050(卡1) 1035051(卡2) 1035053(卡3) 1035052(卡7)；队列 PID 1038406。

### 20:42–21:00 UTC：叠作业实测与决定（用户要求）
- 用户在 20:42 启动 `scripts/gpu_guard/guard_cards.py`（PID 见 `logs/gpu_guard/guard.pid`），清除 0,1,2,3,7 上所有
  非 root 计算进程；我方作业全部以 root 运行，不受影响；keeper 继续作为第一道防线。收尾时与 keeper 一起按 PID 停。
- `make_eval_jobs.py` 新增 `--dec-cards/--dec-mem/--max-gib/--dec-only`：整模型解码可拆到 2 卡（`--max-gib 34`，
  实测单卡 33.9 + 29.4 GB，队列按 38 GB/卡计）。
- **实测**：S0-all SFT（0–3 号卡）单独跑 3.41 s/步（第 40–150 步均值）；在 0,1 号卡上叠一个 2 卡拆分解码
  （test_v8_ctx2r 第 0 片）后 3.63 s/步（第 160–330 步），**慢 6.5%**；该解码片约 8.2 窗口/分钟（306 窗口约 37 min）。
  若整个 SFT 期间都叠，关键路径约多 25–30 min。
- **决定**：SFT 卡（0–3）上不再叠解码（已在跑的第 0 片跑完为止）；AMI test 与 AliMeeting 的 v8 参考解码
  （新名字 `test_v8_ctx2r`、`ali_v8_ctx2r`，旧的半成品 `test_v8_ctx2`/`ali_v8_ctx2` 不复用不覆盖）改为 2 卡拆分、
  只放 4,5,6,7 号卡（38 GB/卡，kenxsong 卡上留出 ≥ 35 GB 余量，远大于其实测 ~15 GB 的波动）。这样不拖慢 SFT，
  同时用掉闲置显存。talker/ASR 等轻作业仍按显存自由叠放。
- 21:05 UTC 修正：队列曾把两个拆分解码的半片放到同一张 kenxsong 卡（5 号卡只剩 12 GB 空闲），与上面"≥35 GB 余量"
  不符。立即按 PID 停掉刚启动、仍在加载的 test_v8_ctx2r 第 3 片（未产出任何结果），`gpuq_w3.py` 增加按卡余量：
  4,5,6 号卡每卡至少留 25 GB（其余卡仍 8 GiB）；新增 `scripts/requeue.py`，把失败 / 停掉的作业及其下游以新 id
  重新排队（输出目录不变、不覆盖）。重启队列时我的模式匹配误杀了自己的工具 shell 与旧队列进程（未伤及任何作业，
  作业的 rc 由各自的包装进程写），之后一律用 `ps -C python -o pid=,args=` 的字段精确取 PID。队列 PID 1237923。
- 21:20 UTC：旧队列进程在被替换前（21:06:15）又把 test_v8_ctx2r 第 4 片放到 4,6 号卡，4 号卡只剩 16 GB 且 kenxsong
  刚起新进程 → 按 PID 停掉第 4 片（已完成 50 个窗口），以 `--resume-from` 续跑到新文件（`hyp_s4_r3.jsonl`）；第 3 片
  同样续跑（`hyp_s3_r2.jsonl`）。打分/语音链以 `_c` 后缀重建（`make_eval_jobs.py --existing 8 --id-suffix _c`）。
  `make_eval_jobs.py` 生成的解码作业今后每次尝试都写新文件并从之前的尝试续跑，因此可安全重试（retries 2）。
- 21:25 UTC：AliMeeting 参考解码作业改为可续跑版本重建（`_n` 后缀，旧作业未启动，移入 held_superseded）。
  新增 `scripts/yield_guard.py`（后台常驻）：4,5,6 号卡真实空闲连续两次 < 4 GB 时，SIGTERM 我方在该卡上最新启动的
  GPU 进程（只限 root 且为我方解码/talker/ASR/RL），保证 kenxsong 不因我方叠放而 OOM；被停的作业由队列重试并续跑。

### 2026-11-06 22:00 UTC：闸门规则修订（用户硬规则"永不低于现有系统"；此时尚未看到任何 S0 结果——S0-all SFT 仍在第 ~1300/3000 步，无任何 S0 解码）
- 原因：dev 上跨窗口上下文反而伤害未经训练的 v8：v8 ctx0 tcpBLEU all/ov = 26.25/21.88（CR 0.822、塌缩 16.9%），
  ctx2 = 24.96/18.57（CR 0.776、塌缩 21.4%），重复率不变（6.55% vs 6.42%）。以 v8_ctx2 作对照等于和一个被削弱的基线比。
- 修订：每个闸门指标（文本 all、文本 overlap、语音 ASR all）的对照取 **v8_ctx0 与 v8_ctx2 中该指标更好的那个**
  （实际上即 v8 原协议）；判定阈值不变（点估计 ≥ 对照 − 0.2 且配对 bootstrap P(差 ≤ 0) < 0.95）。之后每个阶段的
  闸门、以及 test / AliMeeting 的对比都如此：v8 一律按其最佳协议报告，ctx2 作为次要行保留。
- 实现：`evaluation/gate.py --refs <v8_ctx0> <v8_ctx2>`（逐指标取点估计更高的对照）。
- 补齐：dev v8_ctx0 语音分（`run_speech.sh`，ASR 进行中，完成后写 `decode/dev_v8_ctx0/score_asr.json`）；
  新排队 `test_v8_ctx0`、`ali_v8_ctx0`（--context 0，文本 + 语音，2 卡拆分，只在 4–7 号卡）。

### 2026-11-06 23:25 UTC：空闲回退（用户新规则：我方 5 张卡空闲时跑无存档 workload）
- `scripts/cardctl.py`：每张我方卡（0,1,2,3,7）的交接原语，均在 `keeper/card<k>.lock`（flock）下进行，只按 PID、
  只对 cmdline 匹配的我方进程发信号。workload 改为**每卡一个**监督进程（`--devices k`），以便单卡收回；
  PID 记在 `logs/mist_w3po_v1/queue/workload/card<k>.json`，日志 `logs/mist_w3po_v1/workload_card<k>.log`。
  - keeper → workload：先起 workload 监督进程（`--wait-idle --idle-seconds 0 --poll-seconds 1 --max-used-mib 0`，
    keeper 在卡上时它等待），再 SIGTERM keeper；keeper 退出后约 1 s 内 worker 启动。workload 本身要求卡上无其他进程
    （启动时检查，运行中发现别的进程会自停），因此二者不能共存，**剩余空档 = worker 加载模型到驻留 ~56 GB 之前**
    （估计数十秒）；这段时间由用户的 guard（清除非 root 进程）兜底。原 `--wait-idle` 默认要 60 s 连续空闲，已用
    `--idle-seconds 0` 规避。
  - workload → keeper：先在卡上起 keeper（卡满时它不分配），再 SIGTERM workload 监督进程（它看到别的进程也会自停），
    workload 释放后 keeper 1 s 内补满，空闲始终 < 80000 MiB，无死锁（各步有超时，失败保留 keeper）。
- `scripts/idle_fallback.py`（root、setsid 常驻，独立于执行者）：某卡连续 10 min 没有 W³PO 作业进程（不计 keeper/
  workload）且 gpuq 中没有在该卡上运行或可运行的 GPU 作业 → keeper → workload；卡上既无 keeper 也无 workload →
  立即起 keeper；workload 监督进程 240 s 内没起 worker（例如有外人进程）→ 换回 keeper。
- `gpuq_w3.py`：workload 占用的显存算作我方可用；在跑 workload 的卡上起作业前先 `to_keeper()`，再走原来的
  keeper 释放流程。
- 验证（未运行任何启动脚本）：`scripts/test_idle_fallback.py` 模拟决策时间线、合成队列目录、交接顺序（记录器替代
  进程操作）全部通过；`idle_fallback.py --dry-run` 对当前 5 张卡判定均为 keep（SFT/解码在跑）；workload 参数用其
  自带 `--dry-run` 校验通过。
- 进程：队列 PID 2225627，idle_fallback PID 2225628，yield_guard PID 1356920；guard 不动。
- 23:31 UTC（用户批准方案 A）：`scripts/gpu_load/train_mist_workload.py` 新增**可选**参数 `--tolerate-keeper`：
  启动检查和运行中检查都忽略 root 拥有、cmdline 含 `experiments/mist_w3po_v1/scripts/gpu_keeper.py` 的进程，
  其显存不计入"忙"；其他任何进程照旧阻止启动 / 使负载自停。不带该参数时行为完全不变（`scripts/gpu_load/test_*.py`
  22 项原测试全部通过）；新增 `experiments/mist_w3po_v1/scripts/test_workload_tolerate_keeper.py`（5 项）通过。
  空闲回退改为**无空档**顺序：workload（带 `--tolerate-keeper`）在 keeper 仍在时启动，keeper 在空闲 < 12 GiB 时自动
  让出块，workload 填到只剩 8 GiB；待 worker 显存 ≥ 56000 MiB 且 keeper 已不持有（或卡上空闲 < 30000 MiB）才停
  keeper；600 s 内没接管成功则停 workload、保留 keeper。决策与交接顺序的模拟测试、workload 自带 `--dry-run`、
  watcher `--dry-run` 均通过（未运行任何启动脚本）。队列与 watcher 已重启以载入新代码。

### 2026-11-06 23:38 UTC：4–6 号卡接管准备（用户经 K8s 取得 4–6 号卡）
- kenxsong 的 zh2en TTS 流水线（`ktrain_hok_to1000`，控制进程 1730109，voxcpm 服务 2243463/65/67）仍在 4–6 上，
  其 PROGRESS.json 预计 11-07 06:05 UTC 结束；不碰。
- `scripts/takeover_4_6.py`（root、setsid 常驻，PID 2316702）：4–6 号卡上 cmdline 含 `ktrain_hok_to1000` 的 GPU
  进程全部消失并保持 2 分钟后，写 `logs/mist_w3po_v1/queue/cards.json`（8 卡皆为我方、无额外余量），立即在 4–6
  起 keeper，写 `takeover_4_6.done` 后退出。
- `gpuq_w3.py` 与 `idle_fallback.py` 每轮读 `cards.json`（`cardctl.load_cfg`）：接管后 4–6 与其他卡同等（8 GiB 余量、
  keeper、空闲 10 分钟无作业 → 无空档切换到 workload、起作业前按 PID 收回）。kenxsong 其他仍在 4–6 上的进程不动：
  有非 root 进程的卡不切 workload（workload 也起不来），keeper 在空闲 < 12 GiB 时让出，`yield_guard.py` 照旧保护。
- 验证：`test_takeover_4_6.py`（模拟触发时序、cards.json、keeper 请求、done 标记）、`test_idle_fallback.py` 通过；
  `takeover_4_6.py --dry-run` 看到流水线 GPU 进程 9 个；`idle_fallback.py --dry-run` 正常。
- 进程：队列 2316700、idle_fallback 2316701、takeover 2316702、yield_guard 1356920。

### 2026-11-07 01:05 UTC：S0-all 闸门 —— **未通过**
S0-all SFT（4 卡，3000 步，3.4–3.7 s/步，23:40 完成合并）。AMI dev，对照 = 逐指标取 v8 ctx0 / ctx2 中较好者
（文本 all 26.25、ov 21.88，均来自 ctx0；语音 all 18.31 来自 ctx0，ctx2 为 17.80）：

| 系统（dev） | tcpBLEU all | overlap | 1spk | CR | 重复率 | ≤0.5 s 召回 1spk/ov | ov 延迟 p50 |
|---|---|---|---|---|---|---|---|
| v8 ctx0（对照） | 26.25 | 21.88 | 28.14 | 0.822 | 6.55% | 57.6/70.5% | — |
| v8 ctx2 | 24.96 | 18.57 | 28.21 | 0.776 | 6.42% | 59.0/67.9% | 1.66 s |
| S0-all ctx2 | 24.87 | 17.26 | 28.95 | 0.804 | 9.45% | 52.4/65.6% | 0.96 s |
| S0-all ctx0 | 22.71 | 16.08 | 26.33 | 0.811 | 7.90% | 49.6/62.8% | 0.89 s |

- 配对 bootstrap（vs v8 ctx0）：S0-all ctx2 all −1.37（P≤0 0.79）、**ov −4.62（95% CI [−7.01, −1.44]，P≤0 0.9998）**；
  S0-all ctx0 all −3.53、ov −5.80（P≤0 0.998）。两种协议下文本 all 都低于对照 −0.2 的容差，ov 显著变差 → 闸门不通过
  （语音分不影响结论，仍在跑，留作报告）。
- 观察：S0 让提交大幅提前（ov 延迟 p50 1.66 → 0.96 s，窗口末尾前提交占比 0.36 → 0.42），单人区略升（+0.8），但重叠区
  明显变差、重复率升高（6.4 → 9.5%）。即"更早提交"的行为在 SFT 里学到了，但质量代价落在重叠区。
- **决定**（按规则：未过闸门的阶段不进入下一步）：
  1. S0 的组合改动不作为 RL 底座；5 个留一 SFT（no1..no5，`sft_s0no*`）照常跑，用来找出拖后腿的那一项；若某个留一
     版本在 dev 上通过闸门，再考虑把它作为 RL 底座（只用于之后启动的运行）。
  2. **RL 底座 = v8 SFT 本身**（`thinker_gstar_s2dl_zh2en_3k_20261030`，即闸门对照），提示不带跨窗口上下文
     （新参数 `train_w3po.py --ctx-max 0`，与 v8 训练/最佳评测协议一致）。W³PO 50 步小测
     `rl_pilot_w3po_v8`（4 卡、64 窗口/步、G=8、锚点 Lag1/2/3）已排队，之后 dev 评测 `dev_pilot50_ctx0`（文本+语音）。
  3. 原定的单卡 RL 显存探测作业取消（小测本身给出显存与吞吐）。

### 2026-11-07 01:45 UTC：W³PO 小测启动（v8 底座）与锚点预计算
- `rl_pilot_w3po_v8` 01:22 在 0–3 号卡启动（每卡约 104–107 GB）。前两步：
  step 1：R 0.419、组内 R 标准差 0.105、ACR when/who/what = 0.11/0.33/0.09、丢弃零方差组 6/64、锚点 R 0.299、
  KL 0.0026、用时 655 s（rollout 116、锚点 204、更新 334）；step 2：472 s（rollout 102、锚点 151、更新 219）。
  组内有足够方差（chrF++/BLEU 能区分 rollout，无需另议神经指标）。按 ~470 s/步，50 步约 6.5 h（约 07:50 结束）；
  第 20 步存档后先做一次 dev 文本评测（`dev_pilot20_ctx0`），50 步后文本+语音（`dev_pilot50_ctx0`）。
- 每步都换新窗口，锚点缓存基本不命中，锚点占每步约 1/3。新增 `rl/precompute_anchors.py`：按训练器自己的窗口流
  （同一 Sampler 种子、每步窗口数、步数范围）和 gold_ctx 预先算好锚点，训练器新参数 `--anchor-cache` 每步增量读入。
  核对：前 3 步的窗口流 192 个，小测实际算过的 144 个锚点窗口全部在其中。主训练将使用它。
- 参考解码（test/ali 的 v8 ctx0/ctx2）排到小测评测与留一 SFT 之后（作业文件改名为 07x 前缀，id 不变）。

### 2026-11-07 02:08 UTC：4–6 号卡接管完成
- kenxsong 的 `ktrain_hok_to1000` GPU 进程 02:05:47 全部退出（预计 06:05，提前约 4 h）；takeover_4_6.py 在 02:07:51
  写 cards.json（8 卡皆我方、8 GiB 余量），在 4/5/6 起 keeper（3279279/3279373/3279489）；idle_fallback 覆盖 8 卡。
  4–6 上 kenxsong 的其他进程（vlm_tta）也已自行退出。
- 锚点预计算（`anchors_main_*`，单卡）实测约 15 窗口/分钟/卡（训练器内约 6.4，因其与 rollout/更新同卡串行）。
- 计划：小测第 10 步存档后按 PID 停 4 卡小测，以 8 卡从 step_00010 续跑（`rl_pilot_w3po_v8_b`，同一窗口流）实测
  每步用时，与 4 卡第 2–10 步比较后决定 8 卡还是 4+4 并行。

### 2026-11-07 02:30 UTC：其他容器的 root 作业
- 02:19 起 zezhongjin 的 `streaming_si.inference.infer_s5_streaming`（/opt/venv，root，约 16 GB/个）出现在 0、1、5 号卡；
  root 运行，guard 不处理，也不受 keeper 的 80 GB 门槛影响。按规则不动，已报告。
- 应对：`yield_guard.py` 扩到全部 8 张卡（任何卡真实空闲连续两次 < 4 GB → 停我方最新的解码/talker/ASR/锚点进程，
  RL 训练只在卡上没有别的我方进程时才停），保证不因我方叠放把任何人推到 OOM；`cardctl.is_ours()` 改为"root 且
  自身或父进程命令行在 /opt/dlami/nvme/jasonleeeli/ 下"，避免把别人的 root 作业当成 W³PO 作业（影响空闲判定）。
- 拆分解码实测单卡峰值可达 ~40 GB，待启动的拆分解码作业改按 44 GB/卡计；锚点预计算作业限定在 4–7 号卡，不与小测同卡。
- 小测第 5 步因同卡 ASR/解码争用用了 16 min；第 10 步存档后转 8 卡实测。

### 2026-11-07 02:32 UTC：小测被外部作业挤爆（OOM）
- 02:32 zezhongjin 在全部 8 张卡各起一个 `/opt/launcher/asr_latency.py`（root，约 55 GB/卡，PID 3453970–77）。
  小测 rank0 在卡 0 上 OOM 崩溃（第 6–9 步未存档，丢失；无存档 → 续跑将从第 0 步重来，metrics.jsonl 会有重复步号，
  以最后一次为准）。yield_guard 停掉了我方 3 个锚点预计算和 1 个解码（均可续跑）。已报告协调者，不动对方进程。
- 处理：小测作业改为每 5 步存档（`rl_pilot_w3po_v8_c`，含 `--anchor-cache`），在对方占用期间放不下（每卡约需 100–105 GB），
  暂移入 held；先放行 5 个留一 SFT（每卡约 70 GB）与评测作业利用剩余显存。AliMeeting 第 5 片以 `_r1` 重新排队续跑。

### 2026-11-07 03:00–03:31 UTC：小测提速与一次全机 SIGTERM
- 03:21 在 step_00005 把小测换成 `--tok-budget 24000`（`rl_pilot_w3po_v8_d`，每卡约 92 GB）：第 6 步更新 245 s
  （此前 12000 时 260–480 s），整步 366 s（锚点已缓存）。
- 03:29:50 全部 8 张卡上的 GPU 进程（含 keeper）几乎同时收到 SIGTERM（非我方 yield_guard、非 guard_cards 所为），
  已报告。自动恢复：idle_fallback 1 分钟内重起 8 个 keeper；队列从 step_00005 重启小测（03:30:44）；锚点预计算续跑；
  两个被停的 ASR 以 `_r1` 重新排队。
- 03:34:45 第二次全机 SIGTERM：8 个 keeper、队列（3420256）、idle_fallback（3420255）、重启后的小测、锚点、ASR 全部被停
  （yield_guard 与 guard_cards 未受影响）。03:36 起 zezhongjin 的 K8s 24 卡作业 `s5-audio-lora-smoke-24gpu-1107-r1`
  （swift sft，torchrun 3917184，本机为 node_rank 0）占用全部 8 张卡（初始约 25 GB/卡）。已报告并请求把本节点从该作业排除。
  03:40 重启队列（3955873）与 idle_fallback（3959883），keeper 全部重起；小测在显存允许前排队等待（每卡需约 110 GB）。
- 03:50:55 第三次：keeper、队列（3955873）、idle_fallback（3959883）、重启后的小测（torchrun 4018220）再次同时被 SIGTERM；
  zezhongjin 的作业 03:52 重新起在 8 张卡上（PID 4049858…，各约 0.6 GB 起步）。三次中从未被杀的是 yield_guard 和
  guard_cards ——它们都不是 GPU 进程的父进程；被杀的守护进程都是 GPU 进程的父进程。推断外部清理会连同父进程一起杀。
- 对策（04:00 生效）：队列作业、keeper、workload 一律用 `setsid -f` 启动（立即过继给 init，守护进程不再是任何 GPU
  进程的父进程）；`yield_guard.py` 每 30 s 检查队列与 idle_fallback，缺了就用 `setsid -f` 重启（`<qdir>/NO_RESURRECT`
  可关闭，收尾时用）。实测：重启后的 yield_guard（4107441）30 s 内把队列（4109794）与 idle_fallback（4109796）拉起。
  小测改用 `rl_pilot_w3po_v8_d_e`（重试 10 次，每 5 步存档，从 step_00005 续跑）。ASR 作业改为 `--resume` 续写。
- 04:16:48 与 04:18:01 第四/五次：小测（04:01 起的 torchrun 及 04:16 重启的 38553）与 8 个 keeper 被 SIGTERM；zezhongjin 的
  swift sft（父进程 56491）04:17 又起在 8 张卡上。这次队列、idle_fallback、yield_guard 都活着（`setsid -f` 起效），keeper
  1 分钟内重起。外部作业约每 15–20 min 重启一次并清理 GPU 进程，长于这个间隔的 RL 进度会全部丢失 →
  小测改为**每步存档**（`rl_pilot_w3po_v8_f`，重试 50 次，从最新存档续跑）。

### 2026-11-07 05:45 UTC：小测改 8 卡（实测）
- 外部 swift sft 涨到约 38–45 GB/卡且算力占满后，4 卡小测（tok-budget 24000）每步 10–13 min（step 7–10）。
  每卡约 92 GB 的 4 卡布局已放不下，改为 **8 卡、tok-budget 12000**（每卡约 70 GB），从 step_00010 续跑
  （`rl_pilot_w3po_v8_g`，每步存档，重试 50 次）。锚点预计算暂停（作业移入 held，进度保留，可续跑）。
- 首次 8 卡尝试在第一步因 NCCL 默认 10 min 超时崩溃（共享卡上各 rank 进度差很大）→ `train_w3po.py` 的进程组超时改为 2 h。
- **实测**：8 卡第 11 步 389 s（rollout 105、更新 280，锚点已缓存），同时段 4 卡为 600–780 s/步 → 8 卡约快 1.6–2 倍，
  按指令采用 8 卡。剩余 39 步预计约 4–4.5 h。
- 队列新增：作业包装进程写 pid，若包装进程被外部杀掉、没留下退出码，队列记 137 并按重试处理（此前会被永远当作"运行中"）。

### 2026-11-07 06:08 UTC：旧服务器全部停止；08:30 UTC 新服务器恢复
- 停止原因：zezhongjin 的 K8s 24 卡作业 `s5-audio-lora-smoke-24gpu-1107-r1` 被调度到本机，每 15–20 min 重启并 SIGTERM 全机 GPU 进程。
  停止时小测存档到 step_00013（metrics 1–13），队列放 `NO_RESURRECT`，守护/占位/队列全部按 PID 停止。
- 新服务器（ip-172-16-197-160，8×H200 全空）：`restore_from_zip/apply.py --apply` 恢复 28,492 个文件权限与 6,990 个软链接；
  git 中 1 个仍缺执行位的文件按 index 补齐；S3 恢复把 `idle_fallback.py` 还原成旧版（与 fee74ed2 一致），已 `git checkout`
  回 HEAD。S3 两路同步重启（项目 stamp 设为恢复时刻，full_ok 已置，SKIP_ZIP=1）。GitHub `private` 远端可达，HEAD 与 private/main 一致。
- 小测第 1–13 步：训练奖励 R 0.34–0.44 无趋势；KL 0.0025–0.006（第 6 步 0.047，续跑后）；组内 R 标准差约 0.10；
  ACR who 0.23–0.48。锚点 R 在第 6、12、13 步异常低（0.07/0.17/0.19，其余 0.26–0.42），需在续跑前核对锚点缓存与在线锚点是否一致。

### 2026-11-07 09:05 UTC（布拉格 10:05）：锚点奖励异常核查 —— 不是缓存/对齐错误，是 Lag1 锚点退化；加锚点入组门
- 工具 `rl/check_anchor_rewards.py`（CPU）：按训练器窗口流重建第 1–13 步的窗口，读取全部锚点记录（小测 `anchors/rank*.jsonl`
  与 `anchor_cache/v8_ctx0_s20261107_w64/part_*.jsonl`），重算锚点 R。结果：
  - 13 步重算的 anchor_R 与 metrics.jsonl **完全一致**（到 1e-15），第 6/12/13 步 0.071/0.173/0.193 均复现；
  - 同一窗口多份记录内容不一致：**0**；read 时刻、akey、ctx（ctx-max 0）、before 前缀与窗口不匹配：**0**；
  - 事实更正：第 1–13 步用到的锚点**全部**来自小测自己的 rank 文件（训练器内计算；第 6、11 步的"已缓存"是此前崩溃的
    尝试写下的同一批记录），`anchor_cache` 目录只覆盖第 51–300 步，从未被小测用到。4→8 卡、tok-budget 变化只改变
    窗口在 rank 间的分配（按 id 查缓存），不影响对齐。
- 根因：Lag1（1000 ms）锚点在部分窗口陷入碎片循环（"the, | the, | the, ..."、"I'm. | I'm. | ..."），每个碎片段都算
  spurious（−0.5），单个窗口锚点 R 低至 −27.5（第 6 步 ES2015a_on_0014）、−32.2（第 12 步 ES2003b_on_0008）；Lag2/3 同窗口
  正常（0.3–0.7）。一个这样的窗口就能把该步 64 组平均的 anchor_R 拉低 0.1–0.17。各步锚点平均 R < 0 的窗口 3–13 个。
  即"难窗口 + 短 lag 锚点退化"，不是缓存或对齐 bug。
- 但它确实伤害了更新：锚点在组内（LUFFY），退化锚点把组内各归一化标准差（what 的段级 / 轨迹级、who、when）撑大。
  在 ES2015a_on_0014 上实测（`rl/test_anchor_gate.py`）：策略成员 what 优势的平均绝对值 0.27（含退化锚点）vs 0.91
  （剔除后），即受影响组的策略信号被压缩约 3.4 倍；方向没有变（不是符号错误），退化锚点自身得到很大的负优势，
  LUFFY 形式 π/(π+0.1) 下这些低概率 token 的梯度很小。
- **修复**：`advantage.admit_anchors` + `train_w3po.py --anchor-gate {none,minpol}`（默认 minpol）：锚点只有在 R ≥ 组内
  策略最低 R 时才入组做 LUFFY 引导；所有锚点仍作 FLAD 延迟参照（退化锚点不可能"更好"，在 FLAD 中本就无作用）。
  `--anchor-gate none` 与此前逐位相同。新增指标 `anchor_admit`（入组锚点比例）。另：锚点 credit 此前算两遍，现只算一遍（结果相同）。
  单测 `test_anchor_gate.py` 全部通过（手工用例 + 真实退化锚点）。提交 6cf45eae。
- **决定**：第 1–13 步保留（更新方向未被破坏，只是约 5–20% 的组信号被稀释）；从 step_00013 续跑时启用 minpol
  （小测第 14–50 步 = 主训练将使用的目标；LOG 记录此处分界）。
- 另核对预计算锚点与训练器内锚点是否等价：新预计算 `anchor_cache/v8_ctx0_s20261107_w64_p12`（第 13–50 步，8 卡各一分片，
  与训练器同为每批 8 窗口），其中第 13 步的 64 个窗口与小测 rank 文件里训练器算的记录逐 token 比较（结果见下一条）。

### 2026-11-07 09:00 UTC：新服务器调度
- 旧队列目录 `logs/mist_w3po_v1/queue` 中有大量旧服务器遗留的待运行作业（旧 4 卡小测 `045_rl_pilot`、bench、probe、
  旧 ctx2r 解码等），直接重启会把它们一起放出。改用**新队列目录 `logs/mist_w3po_v1/queue_v2`**（只放本轮作业）；
  旧目录保持 `NO_RESURRECT`，不再运行。
- 用户新令：训练卡上不叠任何其他 GPU 进程（含 keeper/workload）；任何卡没有我方作业 → 立即 workload。本机独占，
  keeper 不再需要：`cards.json` 新增 `"keepers": false, "idle_s": 45`（cardctl/idle_fallback/gpuq_w3 读取；keepers 关闭时
  队列起作业前直接按 PID 停 workload，空闲 45 s 起 workload，残留 keeper 在作业卡上立即停）。测试通过，提交 c3622897。
- 08:57 按 PID 停掉 newserver workload（监督进程 2125558 及 8 个 worker），起 queue_v2 的 gpuq / idle_fallback / yield_guard
  （setsid -f）。GPU 功耗/利用率采样：`work/mist_w3po_v1/power/smi_20261107.csv`（2 s）。

### 2026-11-07 09:10 UTC：预计算锚点 vs 训练器内锚点（第 13 步，逐 token）
- 比较对象：小测 rank 文件中训练器算的第 13 步锚点（8 卡，每 rank 8 窗口一批，音频嵌入取自同步 rollout 的缓存）与
  `precompute_anchors`（`Engine.from_merged`，同样每批 8 窗口）新算的同一批窗口，约 40 个窗口。
- 结果：39 个窗口至少一个 lag 的 token 序列不同；按 lag 统计 R 完全相同的窗口数 Lag1/2/3 = 3/18/21（共 39）。
  均值：Lag2 0.393 vs 0.406，Lag3 0.400 vs 0.403（无系统偏差）；Lag1 −0.26 vs −0.54（中位数 −0.055 vs −0.083），
  差别来自少数窗口的碎片循环（如 ES2007c_on_0014：−2.6 vs −12.8）。
- 解释：贪心解码对 bf16 数值（不同批次形状下的音频编码/注意力 kernel）敏感，长流式解码中一处 argmax 翻转后序列分叉；
  Lag1 读次最多、截断最频繁，最不稳定。没有发现逻辑差异（窗口、ctx、read 时刻、akey 全部一致，Lag2/3 分布一致）。
  所以两条路径是"同分布、非逐位一致"，不是 bug。退化的 Lag1 锚点由新的锚点入组门（minpol）挡在组外，对训练没有影响。
- 小测第 14–50 步与主训练都使用预计算锚点，口径一致。
