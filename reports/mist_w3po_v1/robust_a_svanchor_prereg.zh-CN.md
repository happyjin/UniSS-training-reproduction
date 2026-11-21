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
- **损失**：batch 平均的 ‖d(û) − d(u*)‖²（d-vector 上的平方 L2 范数，对维度求和）+ 0.1 · MSE(û, u*)（对 32×6 pre-bound 码逐元素平均，与 V1 的损失同一定义）。（10:55 CET 澄清，早于任何真实数字：CPU 冒烟测试〔300 对，只查代码能跑〕发现初稿代码把 d 项也按逐元素平均，与上面"平方范数"的写法不一致，按本行定义改正。）d(u) = `speaker_encoder.project(flatten(project_out(clip(bound(u), −2, 1) / 2 · scale)))`，即去掉 round 的 `detokenize`（与 `k5_soft.d_cont` 同一条仿射通路，可微；两边都用 bound 后未 round 的码）。脚本先检查：等级中心代入时与 `speaker_encoder.detokenize(ids)` 最大绝对差 ≤ 1e−5。
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
运行 10:48–10:57 CET（只用卡 4、5、6、7；占位只按 PID 停，watcher POLL 1 s；结束后 guard 已把占位放回，卡 4 约 1 分钟内重新占满）。种子 20261119（训练）；`eval_outputs/mist_w3po_v1/a_sv/`，日志 `logs/mist_w3po_v1/a_sv/`。

### 特征与校验（10:48–10:54 CET）
- `a_sv_codes.py`：φ_V1 重算的 token 与 V1 表不一致 4 / 1274（en2zh）、11 / 2012（zh2en）行（跨设备边界翻转，与 K5 相同），φ0(孪生提示) 与 oracle 表 0 不一致；通过（≤ 5%）。TitaNet 训练特征 40000 条 318 s。
- `a_sv_train.py`：d 通路检查（等级中心 vs `detokenize`）最大差 0.0，通过。

### 预检（10:55 CET，`a_sv/train/precheck.json`）：**不通过，按规则停止，未跑 Talker**
| 方向 | 带噪 n | cos(d(round û), d(oracle)) 中位数 A / V1 | 连续 d(bound û) A / V1 | 汉明对 oracle A / V1 | 干净 cos A / V1 | 门槛（V1 + 0.02） |
|---|---|---|---|---|---|---|
| en2zh | 637 | **0.873** / 0.891 | 0.885 / 0.904 | 29.5 / 28.6 | 0.940 / 0.964 | 0.911 ✗ |
| zh2en | 1006 | **0.882** / 0.897 | 0.895 / 0.910 | 29.3 / 28.5 | 0.944 / 0.964 | 0.917 ✗ |

- A 在两个方向上都**比 V1 更差**（−0.018 / −0.015），干净条目也从 0.964 掉到 0.940–0.944。写 `a_sv/STOPPED_PRECHECK`，A / A-soft 的 Talker 评测没有跑，判断规则第 6 节不适用。
- 训练日志（只记录）：训练集 d 损失 2.78（第 100 步）→ 1.08（第 5000 步），而 200 个 diag 监控对（V1 没训练过）的 d 损失从第 0 步（û = V1）的 5.30 **升到** 5.61，d-vector 余弦中位数 0.940 → 0.934。V1 自身在训练子集上的余弦 0.959，在监控对上 0.940（样本内偏差，第 3 节已记为风险）。

### 解读
- MLP 学到的修正只在训练分布上有效，换到训练外说话人就变成噪声：从第 0 步（恰好等于 V1）开始，监控对上就单调变差。
- 主要原因（推断，未单独验证）：训练集只有 651 个说话人（LibriSpeech 251 + AISHELL 400），TitaNet 输入是几乎完美的说话人 ID，MLP 可以直接记住"说话人 → 干净码"，这对 S1 dev 的新说话人没有用；再加上 V1 码在训练集上是样本内（误差偏小），MLP 学到的"V1 误差模式"在新数据上不成立。
- 所以"把 TitaNet 当锚、再学一个映射回 BiCodec 码"这条路在现有开放数据规模下不成立；TitaNet 的抗噪信息没有被这种小 MLP 以可泛化的方式转到 φ0 码空间。要用 SV 锚点，可能需要不依赖说话人数量的形式（例如推理时在 φ0 码上做以 TitaNet 一致性为目标的优化 / 检索，而不是训练映射），或大得多的说话人集合；本次没有试（一个设置，不扫参）。

### 独立评测器检查（ECAPA2，已有四个臂；`a_sv/ecapa2_existing.json`，`a_sv_ecapa2_existing.py`）
相似度中位数在 Talker 已有输出上补算，按格收回 B 噪声损失的比例（TitaNet 一列与 K5 / oracle 报告一致）：

| 格子 | B 损失 TitaNet / ECAPA2 | V1 TitaNet / ECAPA2 | K5-full TitaNet / ECAPA2 | oracle TitaNet / ECAPA2 |
|---|---|---|---|---|
| en2zh a | 0.076 / 0.065 | 38% / 55% | 56% / 57% | 98% / 77% |
| en2zh b | 0.067 / 0.066 | 63% / 68% | 58% / 66% | 119% / 101% |
| zh2en a | 0.069 / 0.088 | −12% / 21% | 10% / 30% | 92% / 100% |
| zh2en b | 0.067 / 0.081 | 15% / 39% | 26% / 52% | 96% / 108% |
| en2zh b_noxt（只报告） | 0.071 / 0.058 | 56% / 67% | 45% / 74% | 113% / 96% |
| zh2en b_noxt（只报告） | 0.057 / 0.079 | 12% / 27% | 30% / 46% | 85% / 94% |
| **4 格平均** | | **26% / 46%** | **38% / 51%** | **101% / 96%** |

- ECAPA2 同样看到 B 的噪声损失（0.065–0.088）和 oracle 约 100% 的收回，"剩余身份损失在全局码"这一结论不依赖 TitaNet。
- 但 ECAPA2 下 V1 的收回比例高很多（46% vs 26%），K5-full 相对 V1 的增益从 +12 点缩到 +5 点：TitaNet 对 V1 偏严，之前用 TitaNet 算出的 V1 → K5 增益有一部分是评测器特有的。之后的身份结论都应同时报两个评测器。
- ECAPA2 干净相似度：V1 与 B 相差 −0.011 到 +0.010，与 TitaNet 一致（V1 不伤干净输入）。

### 偏差
1. 10:55 CET 澄清损失写法（见第 3 节，早于真实数字）：初稿代码 d 项按逐元素平均，改成平方范数（与预登记写法一致）。同一时间修正了 `detokenize` 的输入形状（(B, 1, 32)，与 `d1ts_tokens.py` 相同）。两处都在 CPU 冒烟测试（300 对）里发现，冒烟输出放在 scratch，不作为结果。
2. `a_sv_codes.py` 的 ≥ 95% 一致性检查只在 `--limit` 冒烟模式下放宽为只记录；正式运行照常检查（通过）。
3. 独立评测器检查（ECAPA2 对 B / V1 / K5-full / oracle 的收回比例）是预检停止后加的只记录分析，不入任何规则。
4. 其余与预登记一致：一个设置，没有扫参，没有重跑。
