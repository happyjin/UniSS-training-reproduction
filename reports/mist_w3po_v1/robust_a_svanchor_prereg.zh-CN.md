# 鲁棒 A：用抗噪说话人验证嵌入锚定音色码（SV-anchor）预登记

2026-11-21 10:40 CET（布拉格）写，早于 A 的任何训练、预检或 Talker 数字。唯一已看过的数字是独立评测器的可用性检查（第 1 节，只看评测器本身，不涉及任何方法臂）。

## 0. 已知事实（来自 D0 / D1 / K5 / oracle 的结果节）
- oracle（带噪条目换成干净孪生提示的 32 个全局 token）收回 B 噪声损失约 100%；V1（`voicecond_idea1/m2/ctrl/encoder.pt`）约 26%，K5-full 约 38%。
- token 翻转本身无害（D1：起点 +10 ms 翻掉约 42% token 无损失）；身份跟着解码器 d-vector 的偏移走（d-vector 对 FSQ 码仿射；V1 码对干净孪生码的 d-vector 余弦约 0.89）。
- 信息在约 3 s 的带噪提示里，但 BiCodec 的 ECAPA（φ0）在噪声下丢掉（EER 7.5% → 12.5%），TitaNet-large 几乎不变（6.5% → 7.9%）。

## 1. 评测器独立性（先做）
- TitaNet-large 既是本方法的输入，又是一直以来的相似度指标，存在"自己考自己"的风险，所以加一个独立的强 SV 模型。
- 可用性检查（10:20–10:35 CET）：HF 可直连。**ECAPA2**（`Jenthe/ECAPA2`，TorchScript 单文件 `ecapa2.pt`，VoxCeleb 训练，192 维）只需要 torch + huggingface_hub，`venvs/nemo_diar` 里已有，**不安装任何包、不新建环境**；下载到 `cache/huggingface/hub/models--Jenthe--ECAPA2`。speechbrain / CAM++（modelscope 3D-Speaker）需要往允许的环境里装包，不用。
- 评测器自检（只看评测器）：S1 dev en2zh set a 前 20 对源音频，同说话人"干净 vs 带噪"余弦中位数 0.961，不同说话人干净对中位数 0.069（最大 0.417），区分正常。
- ECAPA2 与 TitaNet（NVIDIA，NeMo）、BiCodec 内部 ECAPA-TDNN 都是不同训练、不同权重；本方法没有用到 ECAPA2。
- 使用方式：`a_sv_ecapa2_spksim.py`，与 `wave3_v_spksim.py` 完全同一流程（生成语音整句 vs 干净孪生源音频，fp32，L2 归一化余弦）。对 B / V1 / K5-full / oracle 的已有输出与 A / A-soft 的新输出都算。

## 2. 假设
用噪声下几乎不变的 SV 嵌入（TitaNet-large）当锚，从同一个带噪提示估计音色码，能让码在解码器 d-vector 空间里更接近干净孪生码，从而收回比 V1 多得多的噪声损失。不是蒸馏：没有教师模型输出当软目标；目标是干净孪生提示的 φ0 码（与 V1 / oracle 同一个监督信号）。

## 3. 方法（一个设置，不扫参）
- **输入**：x = [TitaNet-large(带噪提示段)（192 维，L2 归一化，前向与 `wave3_v_spksim.emb` 相同），flatten(φ_V1(带噪提示段))（192 维，V1 连续 pre-bound 码）]，按训练集均值 / 标准差标准化。
- **模型**：û = φ_V1 + reshape(MLP(x), 32×6)；MLP 384 → 1024 → 1024 → 192，GELU，dropout 0.1，最后一层零初始化（第 0 步 û 恰好等于 V1 的码）。
- **损失**：mean ‖d(û) − d(u*)‖² + 0.1 · mean ‖û − u*‖²（后者在 pre-bound 码空间）。d(u) = `speaker_encoder.project(flatten(project_out(clip(bound(u), −2, 1) / 2 · scale)))`，即去掉 round 的 `detokenize`（与 `k5_soft.d_cont` 同一条仿射通路，可微；两边都用 bound 后未 round 的码）。脚本先检查：等级中心代入时与 `speaker_encoder.detokenize(ids)` 最大绝对差 ≤ 1e−5。
- **训练数据**：voicecond Idea 1 训练对（`data/open_voicecond_v1/idea1/train_en` + `train_zh`，各 20000 对，6 s 片段；与 V1 对照臂同一数据；与 S1 dev 说话人不相交）。u* = φ0（冻结 BiCodec）在**同一段**干净片段上的码。
- **提示段裁剪（唯一选择，写在数字之前）**：部署时提示是源音频前 max(首次写出, 1 s)，S1 dev 提示 10–90% 分位为 2–4 s（中位数 3 s），而训练片段都是 6 s。为让 TitaNet 输入的时长分布与部署一致，每个训练对**一次性**抽一个裁剪（种子 20261119）：长度 L ~ U[1.5, 4.5] s，起点 ~ U[0, 6 s − L]；带噪 y 与干净 s 用同一裁剪；TitaNet、φ_V1、u* 都在这段上算（φ 内部照常音量归一化并平铺到 6 s）。之后不再换裁剪（TitaNet 预先算好）。
- **优化**：AdamW，lr 1e−3，weight decay 0.01，batch 256，5000 步（约 32 个 epoch），200 步线性预热后余弦到 0，梯度裁剪 1.0，fp32，种子 20261119；只存最后的权重。每 500 步在 200 个 ID-1 diag 监控对（V1 训练日志用的同一批，同样裁剪，种子 + 1）上记录损失与 d-vector 余弦，**只记录，不选点**。
- **已知风险（记录）**：V1 本身就是在这批训练对（6 s 片段）上训练的，训练集上 φ_V1 的误差偏小（样本内），MLP 可能低估 V1 在新数据上的误差；训练日志里记 V1 在训练子集与监控对上的 d-vector 余弦，供解读。
- **部署**：
  - **A（主臂）**：硬 token round(û)（FSQ 量化，即 `P.tokens(û)`），走 `idea1_si_infer_vo.py --voice-override`（与 V1 / oracle 同一路径，Talker LM 与 BiCodec 解码器都用这 32 个 token）。
  - **A-soft（唯一消融）**：K5-lm 软嵌入通道（`k5_si_infer_soft.py --k5-lm 1 --k5-dec 0`），b = bound(û) 做重心插值软嵌入进 Talker LM，解码器用硬 token round(û)。
- 脚本：`experiments/mist_w3po_v1/voicecond/a_sv_common.py`、`a_sv_titanet.py`（nemo_diar）、`a_sv_codes.py`、`a_sv_train.py`（vllm_omni_030）、`a_sv_ecapa2_spksim.py`、`a_sv_score.py`、`a_sv_orch.sh`、`a_sv_noplaceholder.sh`。

## 4. 预检（Talker 评测之前，硬门槛）
- S1 dev 带噪条目（wave3 V1 表同一提示文件，sha1 校验），每方向：
  中位数 cos(d(round(û)), d(oracle token)) ≥ 同一批条目上 V1 的中位数 cos(d(V1 token), d(oracle token)) + 0.02（V1 参考值约 0.891 / 0.897）。
- **两个方向都要满足**，否则停止，写 `a_sv/STOPPED_PRECHECK`，不跑 Talker。
- 另记录（只记录）：连续 d(bound(û)) 的余弦、干净条目上对 B token 的余弦、与 oracle token 的平均汉明距离；`a_sv_codes.py` 检查 φ_V1 重算的 token 与 V1 表一致、φ0(孪生提示) 与 oracle 表一致（各 ≥ 95% 行，否则失败）。

## 5. Talker 评测（S1 dev，与 oracle_voice / K5 完全相同）
- 底座 S1 单说话人 LA2（`wave2_S1`），同一 Thinker 文本块，T1c Talker（`si_t1c_20261021T181440Z/export_1500`），种子 20261021，每跑 4 分片，提示文件 / sha1 同 V1 表。
- 复用 B、V1、K5-full、oracle 的已有输出与 TitaNet / SQUIM 指标；新跑 A、A-soft；所有六个臂都补算 ECAPA2 相似度。
- 指标：ASR-BLEU（speech-only，`asr_speech.jsonl`）、SQUIM PESQ 中位数、相似度中位数（对干净源；TitaNet 与 ECAPA2 各一份）；点估计。

## 6. 判断规则（A 主臂，四条都满足才算成立；A-soft 同样报告）
- 收回比例 = (臂带噪相似度中位数 − B 带噪中位数) / (B 干净中位数 − B 带噪中位数)，按格、按评测器。判定格 en2zh / zh2en × {a, b}；b_noxt 只报告。
1. 带噪相似度收回比例 4 格平均 ≥ V1 平均 + 20 个百分点，**TitaNet 与 ECAPA2 都要满足**；
2. 干净相似度不低于 V1：4 格干净相似度中位数的平均 ≥ V1，**两个评测器都要满足**；
3. ASR-BLEU：4 格 × {带噪, 干净} 每一格 ≥ V1 − 0.3；
4. PESQ：4 格 × {带噪, 干净} 每一格 ≥ V1 − 0.1。
- 另报：PESQ 对 oracle 的差；A-soft 相对 A 的差（只解读，判断 K5-lm 软通道能否叠加）。

## 7. 运行
- `experiments/mist_w3po_v1/voicecond/a_sv_orch.sh`（setsid nohup 脱离，fds 8/9 关闭；先 `bash -n` 与 `DRY_RUN=1`），**只用卡 4、5、6、7**（卡 0–3 属于另一会话，不碰）：
  1. 卡 4 TitaNet 特征、卡 5 BiCodec 码（并行）；卡 6、7 先给 B / V1 / K5-full / oracle 已有输出补算 ECAPA2；
  2. 卡 4 训练 + 预检 + token 表；预检不过就停；
  3. 卡 4 A en2zh、卡 5 A zh2en、卡 6 A-soft en2zh、卡 7 A-soft zh2en（各 4 分片，间隔 45 s）→ ASR → SQUIM / TitaNet / ECAPA2 → `a_sv_score.py` → `a_sv/DONE`（失败写 `a_sv/FAILED`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `a_sv_noplaceholder.sh`（POLL 1 s）。不碰 2726951 / 52045 / 1222453。
- 输出 `eval_outputs/mist_w3po_v1/a_sv/`，日志 `logs/mist_w3po_v1/a_sv/`。

## 结果
（待填）
