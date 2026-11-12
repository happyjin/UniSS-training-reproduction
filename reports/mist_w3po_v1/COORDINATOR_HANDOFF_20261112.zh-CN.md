# 协调者交接（2026-11-12 约 21:30 布拉格）

新会话的协调者从本文件接手。各执行者的细节见 `HANDOFF_20261111.zh-CN.md` §0h–§0k，以及 `LOG.zh-CN.md` 末尾。

## 1. 规则（用户定的，长期有效）
- 不 `rm`，不覆盖旧文件。改正在运行的脚本：先写新文件，再用 `command mv -f` 替换。启动脚本只做 `bash -n` 和 DRY_RUN 检查。用 setsid nohup 启动，按 PID 停进程。
- GPU 不能空闲。守护脚本 `keep_cards_busy_v3.sh`（PID 2726951）**永远不停**。
  - 训练独占 4 张卡；评测作业可以叠放，每卡上限 135 GiB，按实际占用另留约 5 GiB 余量。
  - 单说话人论文的基线评测（只做推理）可以叠在训练卡上，条件是训练每步不能变慢超过 40%。
- 延迟同时报旧口径和新的 chunk 口径（按检测器真实输出时刻）。
- 不用 COMET 或 MetricX；多说话人工作不用 QABC 方法；测试集只用于最终评测。
- 回复用中文，时间用布拉格时间。
- 每次改动都提交，并 `git push private HEAD:main`。

## 2. 正在跑（截至 21:30）
| 项目 | 执行者 | 产出 | 预计完成 |
|---|---|---|---|
| 最终测试集评测：AMI 中→英 和 AliMeeting × A′（c1b1s2）、A′+PL-EM、A′rl+PL-EM、VE1rl、D1v rl | 执行者 v9 | 预登记 `final_test_prereg_v1.zh-CN.md`；结果写入 `final_test_results_v1.zh-CN.md` | 第一组约 00:30；全部约 01:30–02:00 |
| X-Translator 在 RealSI 上跑（只推理） | R4 | `paper/baselines_results/xtranslator/` | 约 23:45 |
| QABC 提交块 + Qwen3-Omni 原生 talker（RealSI） | R5 | `paper/baselines_results/qabc_omnitalker/` | 约 22:10，可能更晚 |
| 英→中补训的准备（只用 CPU） | 英→中准备代理 | `en2zh_full_plan_prereg_v1.zh-CN.md` 和脚本；已批准下载中文 CTC 模型 | 今晚 |

## 3. 已完成的结论
- **主系统**：A′rl + PL-EM，dev 上语音 23.69，LAAL 4.050（chunk 口径 3.857）。
- **有效方法**：PL-EM talker、读级块（同读同说话人合块）、chunk 计时口径。
- **折中点**：VE1（两个种子平均：语音 23.44，LAAL 3.689）；D1v（语音 23.36，LAAL 3.548，chunk 口径 3.391）。
- **无效**：T1、T2、T4、T5、RC5、PMC、SpecClose；不训练：T3、J1、J1-v2、D1、PMC-noC。
- **单说话人论文 v12**：`paper/bayes_simul_s2st_v12_baselines/main.pdf`。已加入 Qwen3-Omni 原生、SeamlessStreaming、Hy-SimulMT、LiveInterpret 和级联基线，并改正了引用。用户说"先不动 v12"。
- **Hy-SimulMT + Qwen3-TTS**：`paper/baselines_results/hysimulmt_qwen3tts/`（英→中持平，中→英较差）。

## 4. 待办
1. R4 和 R5 跑完后，新建 **v13**（复制 v12），把 X-Translator、Hy-SimulMT + Qwen3-TTS、QABC + Omni 原生 talker 填进对比表和附录 Table 16；可以用 `experiments/s2st_baselines_v1/` 里的 `fill_paper_v12.py` 做参照。v12 不动。
2. 测试集评测结束后，给英→中训练分卡：Thinker 训练 4 卡，检查点 dev5 速查 4 卡；接着 PL-EM、dev 全集、测试集。只训一个种子；按登记规则用 dev5 文本速查选检查点。约 9–9.5 小时。
3. 等用户决定：v12 正文篇幅；是否把 RealSI 对比表移进正文；Qwen3-TTS 长度上限（推荐保留现在的设置）。
4. 用户已决定不做：多说话人外部基线、多种子。

## 5. 节省 token 的做法
- 执行者类的代理（只跑既定流程、盯进度）用 sonnet 模型；只有设计类任务用 opus。
- 代理只在出结果或出问题时回报；上下文到约 250k 就写交接、交回。
- 每小时的 GPU 巡检改为每 2 小时一次；守护脚本 v3 已经负责补卡。
