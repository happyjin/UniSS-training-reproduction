# 交接：鲁棒性（声音发闷 / 抗噪同传）线 — 2026-11-19 晚（布拉格）

新会话从这里开始。回复用简体中文，时间用布拉格时间。规则见 memory（MEMORY.md）。

## 1. 当前在跑（唯一）

**S1：单说话人底座诊断。**
- 预登记 `reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md`（看"变更"一节）。
- 底座 = QABC 论文里**不含 QABC 方法**的基线：LA2（LocalAgreement-2，Δ = 1 s）。
  - Thinker = 原始 Qwen3-Omni-30B-A3B-Instruct，不加 ABC 适配器、不用 SDR，`si_thinker_stream --policy la2 --windows 1`；
  - Talker = T1c export_1500，`si_infer --text-policy file --speed-policy block`。
  - 论文 RealSI 基线数字：英→中 29.75 / 2.51 s，中→英 23.19 / 2.70 s。执行者已核对存档块 346/346、431/431 一致。
- 数据：
  - (a) CoVoST validation 干净 / 带噪对：`eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl`，每方向 300 句；
  - (b) dev 流切成单说话人流：`data/open_voicecond_v1/bench_dev/`。
- 指标：文本 BLEU、ASR-BLEU（两种子）、LAAL（D1 + D2）、PESQ。
- 输出 `eval_outputs/mist_w3po_v1/wave2_S1/`，日志 `logs/mist_w3po_v1/wave2/`。
- 结果会追加到预登记的"## S1 结果"和 LOG。
- 判读：单说话人带噪损失 < 0.5 BLEU → 翻译模型本身不怕噪声，Thinker 侧组合不跑。
- 由执行者在后台运行。若新会话看不到它的汇报，直接检查上述输出目录和预登记。

## 2. 用户最新决定（必须遵守）

- **不再做多说话人基线验证。** 所有鲁棒组合改为在**单说话人底座（上面的 LA2 基线，不含 QABC）**上验证。S2 和 D0 已停止，数据保留。
- **不要"老师教学生"的蒸馏思路**（候选 E、K5 已取消）。
- **不要用 SEEN**（已发表，不够新）。K6 已取消，K9 只保留 λ 部分。
- 偏好：有机器学习推导、能写公式的方法；推理时 / 贝叶斯 / 闭式优先。
- 所有方案尽量**并行**，GPU 按实际空余显存叠加（每卡 ≤ 约 135 GiB），不让卡空着。
- **门槛只用真实输出指标**（PESQ、ASR-BLEU、相似度、LAAL），**不再用码误差**：它和音质反向。

## 3. S1 出结果后的下一步（单说话人底座上，并行）

| 组合 | 跑不跑 | 说明 |
|---|---|---|
| Idea 1 C（去噪音色编码器，`voicecond_idea1/m2/ctrl/encoder.pt`） | 跑 | 强参照 |
| Idea 2 冻结流式后验（`shrink_fit_all/full.npz`，sha1 25aaf6cd） | 跑 | 单槽位 |
| Idea 6 C_post（C 逐窗口 + 闭式后验，`idea6_precheckA/R_C.npz`） | 跑 | |
| 负条件引导 Talker（w = 1 固定，负条件 = 句间停顿的纯背景，C 编码） | 跑 | 第一波发现语音变短 3–5%，PESQ/WER/相似度不变 → 可能降 LAAL；要以 LAAL 为目标另写预登记 |
| Thinker 侧 K2 / K3 / K7 / K8 | 看 S1 | 损失 < 0.5 BLEU 就不跑 |
| K1 零噪声外推、K4 语速钳制、K7v | 不跑 | 第一波前提不成立 |
| Idea 5 跨说话人共享背景、说话人分离抗噪 | 不跑 | 只在多说话人存在 |

每个方法一个设置、不扫参；先写预登记再跑；只用开源非测试数据（RealSI / ACL / AliMeeting / 电竞只作测试）。

## 4. 今天的结论（详细见 LOG 与各预登记）

**已上完整测试的 6 个版本（OpenBroadcast-SI 9 组，多说话人）** 都显著好过 B 的音质（带噪 PESQ +0.8 到 +1.4）：
- 音质最好：**Idea 1 C**；
- 最稳：**Idea 2 冻结版**，BLEU 8/9、相似度 9/9；
- 汇总：`eval_outputs/mist_w3po_v1/idea6_test/test_summary_v2.json`，以及 Idea 2 的 `test_summary_{bench,ood3}_v2.json`、Idea 1 的 `m3/test_summary.json`。

**关键发现：**
1. **码误差与音质反向**：码误差 E2E 7.36 < C_post 7.95 < C 9.58，PESQ 排序正相反。
2. **"BLEU 下降"多在种子噪声内**：B 自己两种子差可达 2.14；逐窗口 chrF 无显著变差。在声音最闷的地方，音色去噪能把 ASR-BLEU 拉高 +2.45。
3. **单句 Thinker 几乎不受噪声影响**：英→中 −0.20 BLEU。bench 上 1.5–1.9 的损失推测来自多说话人链路（日志器 / 切分），未验证。
4. **音色 token 改变 Talker 语速**（时长 ±5%）→ LAAL 漂移；语速能从隐状态读出（R² 0.89），但钳不住。
5. 背景跨说话人共享（ICC 0.26），但可利用的增益小（−4.9%）。噪声残差与身份纠缠，找不到可单独去除的方向（K7v 投影为空）。

**预检不通过：** Idea 4、5、6 端到端、7、K1–K4、K7、K7v、K8、K9。

**文献调研：** `reports/mist_w3po_v1/robust_novel_ideas_v1.zh-CN.md`（v1.3）。其中的 E 与 SEEN 相关部分按用户要求不再考虑。

## 5. 其他产出

- 试听页：
  - 开源基准 A/B：https://claude.ai/artifact/EWmZbQGyxYUAGVyYQ8c21F
  - E10 电竞（8 段）：https://claude.ai/artifact/NYTNGwETRvgWgHneDCdrta
  - 另一会话做的 E10 电竞长版：https://claude.ai/artifact/Epz2hEbw86rzgG9KCBbng7
- OpenBroadcast-SI dev 版：`data/open_voicecond_v1/bench_dev/`，泄漏检查为 0。其中中→英只有 15 个说话人。

## 6. 省 token 的做法（用户关心）

- 只在完成或失败时汇报；不转达中间状态。
- 少派代理、合并任务；等待用后台脚本 + 完成标记，不让代理挂着等。
- 机械任务（监控、汇总、重跑）用 Sonnet / Haiku；派任务时指向文件，不复述内容。
- 上下文超过约 40 万就写交接、换新会话。
