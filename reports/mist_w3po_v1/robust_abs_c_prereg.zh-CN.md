# 鲁棒 B′：受约束的分析–合成音色 token 搜索（AbS-C，免训练）预登记

2026-11-21 12:20 CET（布拉格）写，早于 AbS-C 的任何搜索、预检或 Talker 数字。前身 `robust_b_abs_prereg.zh-CN.md`（AbS，含"确认运行"节）。一个设置，不扫参，不因结果重跑。

## 0. 已知事实（来自 AbS 两个种子）
- AbS 收回 B 噪声损失（ECAPA2，4 格平均）84% / 81%，V1 46% / 48%；两种子稳定。
- 但两种子方向一致的小幅 ASR-BLEU 下降只在带噪格：en2zh b 带噪 −0.42、zh2en a 带噪 −0.51（两种子均值，AbS − V1；合并 CI 含 0）；PESQ −0.04..−0.09。
- D0：带噪 / 干净孪生的 TitaNet 余弦约 0.92，效用目标（带噪提示的 TitaNet 嵌入）不是完全噪声不变的。

## 1. 假设
AbS 的效用有一部分把音色推向"带噪声色彩 / 更粗糙"的声音，降低可懂度。只接受"重合成提示的无参考可懂度估计不低于 V1 重合成"的候选，可保留大部分身份收益，同时把 ASR-BLEU 拉回 V1 水平。

## 2. 方法（一个设置；与 AbS 完全相同，只多一个可接受性条件）
记号、起点 s0（V1 表的 32 个 token）、FSQ 维度队列 Q（按边界距离 m_i 升序）、翻转方向、效用 U(s) = cos(TitaNet(R(s)), TitaNet(带噪提示))、R(s) = BiCodec 解码 [全局 token s，该提示自己的语义 token]、预算（4 轮 × 16 次候选评估 + 起点 1 次）、补位规则——全部照抄 `b_abs_search.py`。新增：
1. **可懂度估计**：q(s) = torchaudio `SQUIM_OBJECTIVE`（DNS2020 权重，与评测 SQUIM PESQ 同一模型）在 R(s) 上的 STOI 输出（整段提示，提示约 3 s，评测侧 20 s 截断不起作用），fp32，与 U 同一个 R(s) batch。同时记录 SQUIM PESQ（只记录，不入约束）。
2. **门槛**：τ = q(s0)（V1 重合成的 STOI，每行一次）。候选 c **可接受** ⇔ q(c) ≥ τ。V1（s0）本身总是可接受（搜索起点）。没有权重，没有 "≥ V1" 以外的阈值。
3. **每轮**：15 个单翻转候选与 AbS 相同（Q 的下一批维度，基于当前 s）。"提高集" = **可接受**且 U > U* 的单翻转；若提高集 ≥ 2 个，第 16 个候选为同时翻转它们的组合候选，否则为 Q 中下一个维度的单翻转（补位，与 AbS 同）。本轮结束：s ← 本轮**可接受**候选中 U 最大者（只接受严格大于 U*）。不可接受的候选不能被选中，也不进提高集。
4. 输出 4 轮后的 s（65 次评估中可接受且 U 最大者）。全部行（带噪与干净）都搜索。
5. 另外（不计时、不影响搜索）对每行算 AbS token 的重合成 q(s_AbS) 与 SQUIM PESQ，用于"所选候选 STOI 对比 AbS"。
6. 运行环境：nemo_diar + `.tmp/b_abs_pyshim/`（同 AbS，不安装任何包）；SQUIM 权重已在 torch hub 缓存。
7. 延迟：与 AbS 同口径；**额外延迟 = 搜索耗时**（TitaNet 参考嵌入 + 4 轮，含 SQUIM 前向），中位数 / p90。
8. 脚本：`experiments/mist_w3po_v1/voicecond/abs_c_search.py`、`abs_c_precheck.py`、`abs_c_score.py`、`abs_c_orch.sh`、`abs_c_noplaceholder.sh`。

## 3. 预检（Talker 之前，硬门槛）
S1 dev 带噪条目（en2zh 637、zh2en 1006）：
- （判定）ECAPA2 cos(ECAPA2(R_noisy(s)), ECAPA2(干净孪生提示))，s = AbS-C 对 s = V1。**两个方向都满足 mean(AbS-C) − mean(V1) ≥ 0.02 才继续**；否则写 `abs_c/STOPPED_PRECHECK`，不跑 Talker。
- （记录）同样的 ECAPA2 值对 AbS（复用 `b_abs/search/resynth`）；所选候选的 SQUIM STOI / PESQ：AbS-C 对 AbS 对 V1（带噪行与全部行的均值）；被拒候选比例；接受轮数；与 AbS token 的汉明距离；延迟。

## 4. Talker 评测（预检通过才跑）
- 与 AbS 完全相同：S1 LA2 底座、同一 Thinker 文本块、T1c Talker、`idea1_si_infer_vo --voice-override tokens_absc_<d>.jsonl`、`--window 1 --text-policy file --speed-policy block --gpu-mem 0.1`、`SI_KV_CACHE_GIB=7.73`、每方向 4 分片。**两个 TTS 种子 20261021、20261022 都只跑 AbS-C**；B / V1 / AbS 复用已有输出与指标（种子 1：`wave2_S1`、`wave3_combos/V`、`a_sv/metrics`、`b_abs`；种子 2：`b_abs_seed2`）。
- 同一流程：speech-only ASR（`asr_speech.jsonl`）、SQUIM PESQ（`wave2_s1_score.py pesq`）、TitaNet（`wave3_v_spksim`）、ECAPA2（`a_sv_ecapa2_spksim`）。
- 指标：ASR-BLEU、PESQ 中位数、相似度中位数（ECAPA2 主、TitaNet 次）、收回比例（每个种子用**该种子自己的 B**），格 = en2zh / zh2en × {a, b, b_noxt} × {带噪, 干净}（b_noxt 只报告）。

## 5. 判断规则（两种子均值；四条都满足才算成立）
两种子均值 = 两个种子各自格值（语料 BLEU、中位数、收回比例）的算术平均。
1. ECAPA2 带噪收回比例 4 格（en2zh / zh2en × {a, b}）平均 ≥ V1 平均 + 15 个百分点；
2. 干净相似度不低于 V1：4 格干净 ECAPA2 相似度中位数的平均 ≥ V1；
3. ASR-BLEU：8 个主格（4 格 × {带噪, 干净}）每格 ≥ V1 − 0.3；
4. PESQ：8 个主格每格 ≥ V1 − 0.1。

### 另报（不入规则，读法事先写定）
- **AbS-C 对 AbS**：
  - "身份保住" ⇔ AbS-C 的 ECAPA2 收回比例 4 格平均 ≥ AbS − 10 个百分点（两种子均值）；
  - "ASR 收回" ⇔ 两个种子一致下降的格（en2zh b 带噪、zh2en a 带噪）上 AbS-C − V1 两种子均值都 ≥ −0.3。
  - 另给每格 AbS-C − AbS、AbS-C − V1 的两种子均值与合并 bootstrap CI（单位 = stream，两种子充分统计量相加，2000 次，种子 0）。
- 延迟（第 2 节第 7 条）；PESQ / STOI 对 AbS 的差。

## 6. 运行
- `experiments/mist_w3po_v1/voicecond/abs_c_orch.sh`（setsid nohup 脱离，fds 8/9 关闭；先 `bash -n` 与 `DRY_RUN=1`），**只用卡 4–7**：
  1. 搜索：卡 4–7 各一个进程，进程 k = 两个方向的第 k 分片（行号 mod 4）；
  2. 预检（卡 4）；不过就停；
  3. Talker：16 个（2 种子 × 2 方向 × 4 分片），en2zh 分片 0,1 → 卡 4、2,3 → 卡 6；zh2en 分片 0,1 → 卡 5、2,3 → 卡 7，每卡 4 个（固定 KV），同卡相邻启动间隔 ≥ 45 s；
  4. 指标：4 条链（种子 × 方向），然后 `abs_c_score.py` → `abs_c/summary.json`、`abs_c/DONE`（失败写 `abs_c/FAILED`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `abs_c_noplaceholder.sh`（POLL=1）。不碰 2726951 / 52045 / 1222453。拒绝覆盖已有输出。
- 输出 `eval_outputs/mist_w3po_v1/abs_c/`，日志 `logs/mist_w3po_v1/abs_c/`。

## 结果
（待填）
