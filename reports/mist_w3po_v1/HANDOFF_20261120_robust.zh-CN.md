# 交接：鲁棒单说话人同传线（2026-11-20 16:20 CET）

新会话从这里接着做。回复用简体中文，时间写布拉格时间（CET），其余规则见 memory（MEMORY.md）。

## 1. 先读这三个文件
- **总方案和第十五到十七波的预登记：** `reports/mist_w3po_v1/robust_mainline_plan_20261120.zh-CN.md`。包括已确定的事实、主线"边缘化声学干扰变量"、三个方法、测试数据和论文定位。
- **决策日志：** `reports/mist_w3po_v1/robust_overnight_plan_20261119.zh-CN.md`，看末尾几条。
- **各波的预登记和结果：** `reports/mist_w3po_v1/robust_wave{5,8,9,10,11,12,13,14,15,16,17}_*prereg.zh-CN.md`。

## 2. 目前的结论
- **主系统：** 现在是 ERM-WiSE（CoVoST 加噪 LoRA，α = 0.5）+ V1 音色 + T1c，它通过了全部门槛。
- **失败、已关闭、不重跑的方法：**
  - Fishr：惩罚几乎不起作用；
  - 第九波音色锚定：特征被串音污染；
  - 第十一波 Laplace 合并：Λ ≈ 0.5；
  - 第十波 TIES：两个方向的更新正交，没有冲突可消解；
  - 第八波 AUL：模型学会改口，变得更差。
- **诊断：**
  - 第十二波：WiSE 优于训练轨迹上任何检查点。
  - 第十三波：中→英干净掉分来自 LLM 层记住了窄的 CoVoST 中→英数据（7004 条、22 人），整句翻译 −2.77，训练子集 +13.31。这不是提交问题，也不是双向干扰。
  - 第三波：中→英带噪的损失约 71% 来自提交；英→中的损失在感知。
- **用户的决定（2026-11-20）：**
  - 换用 UniST 15 分片 pilot（与基线 Talker 同一数据域）作训练数据，几个新方法同时跑，都用同一份 UniST 数据；
  - 失败的方法不重跑；
  - 测试用基线的 RealSI 干净版加带噪版，再加真实噪声的分布外测试（见方案第 5 节）。

## 3. 正在运行的（都是脱离会话的链条，会话结束后仍会继续跑）

| 波 | 内容 | 卡 | 链条和输出 | 预计 |
|---|---|---|---|---|
| 14 | UniST-ERM-WiSE（新底座） | 0–3 | `voicecond/wave14_chain.sh`，输出 `eval_outputs/mist_w3po_v1/wave14/`，日志 `logs/mist_w3po_v1/wave14/` | 训练约 17:40 结束，结果约 19:30 |
| 15 | CV-LA 跨视角一致提交 | 4–5 | `wave15_*`，`eval_outputs/mist_w3po_v1/wave15/`。先评 cvla_base；第十四波合并完成后再评 cvla_unist | cvla_base 约 16:30；cvla_unist 约 19:30–20:00 |
| 16 | NCWI 噪声条件的权重插值 | 第十四波结束后用 0–3，必要时加 4–5 | `wave16_wait.sh` → `wave16_orch.sh`，`eval_outputs/mist_w3po_v1/wave16/` | 约 20:15–21:00 |
| 17 | 感知层 LoRA（只训练音频编码器） | 6–7 | `wave17_chain.sh`，`eval_outputs/mist_w3po_v1/wave17/` | 训练约 18:15 结束，结果约 21:00–22:00 |

- **数据：** UniST 训练数据在 `eval_outputs/mist_w3po_v1/wave14/data/`，中→英 105253 条（179.6 h），英→中 103236 条（161.8 h）。排除统计在 `stats.json`。
- **每一波都有自己的 noplaceholder watcher**，用独立的 PIDS 和 STOP 文件，跑完后自己 touch STOP，把卡交还给 guard。

## 4. 新会话要做的事
1. **核对各波是否跑完。** 看各 `eval_outputs/.../waveNN/` 下有没有 `CHAIN_DONE`、`EVAL_DONE` 或 `*_FAILED`，以及 `summary.json`。
   - 原会话里的执行者子代理会随原会话一起结束。如果某一波的结果还没写进预登记"结果"一节，新会话要按预登记的判定规则补写，记入决策日志，然后 commit 并 push。
   - 某一波失败或卡住时，看对应的日志，按 PID 处理。
   - watcher 如果没有退出，touch 它的 STOP 文件。
2. **判读。** 每一波都有两条主判定，第一条对底座，第二条对参照，具体见总方案第 4 节。
3. **组合。** 把通过的方法组合成完整系统：UniST-ERM-WiSE（若第十七波通过，就用 enc 版）+ NCWI + CV-LA + V1 音色 + T1c。先在 dev 上跑（参照 `wave5_m_combo_*`），再做逐项去掉一个组件的消融。
4. **测试预登记。** 要写在任何测试数字之前。内容包括：
   - RealSI 干净版：英→中 346 句，中→英 431 句，原始数据在 `data/raw/realsi`；论文的基线数字是英→中 29.75 / 2.51 s，中→英 23.19 / 2.70 s；
   - RealSI 带噪版：用 `data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS*.txt` 里留出的背景，按音乐、人群、音效三类混入，BVR 按 OpenBroadcast-SI 的分布，固定种子，只生成一次；
   - OOD-3，以及 OpenBroadcast-SI test 里的单说话人片段；
   - 测试只跑一次。
5. **论文。** 主线写法、命题和对 ICLR 的评估，见总方案第 6 节。

## 5. 不能碰的
- 进程：guard 2726951、gpu_watch 52045、sync 1222453。只按 PID 停进程。
- 不用 rm，拒绝覆盖，旧文件改名为 superseded。
- 不就地修改正在运行的脚本。启动脚本只用 bash -n 或 DRY_RUN 检查。
- RealSI、ACL、AliMeeting、OOD-3、电竞只用于测试。
- 不做蒸馏，不用 QABC 的组件，一个模型同时做两个方向，每个方法只用一个设置。
- 每次改动都要 commit，并 `git push -q private HEAD:main`。
