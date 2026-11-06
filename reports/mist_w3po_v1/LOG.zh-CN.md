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
