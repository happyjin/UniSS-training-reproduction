# 鲁棒 K4：直接分类干净孪生的离散全局音色 token（逐维 FSQ 等级交叉熵）预登记

2026-11-21 08:45 CET（布拉格）写，早于 K4 的任何训练、预检或评测数字。

## 1. 动机
- Oracle 诊断（`robust_oracle_voice_prereg.zh-CN.md`）：把带噪条目的 32 个 BiCodec 全局 token 换成干净孪生的 token，S1 dev 带噪相似度收回 B 噪声损失的 92–119%；V1（MSE 回归连续码）只收回 −12–63%，K2（W2 映射）、K3（InfoNCE）也远低于 oracle。
- V1 / K3 的码余弦约 0.95，但 token 与 oracle 在约 28–29 / 32 个位置不同；它们的带噪 PESQ（约 3.7）**高于**干净提示的 PESQ（3.3–3.45），是"比干净还干净"的平均音色。
- 结论：损失在**离散** token 上，连续 MSE 目标不会落到正确的量化格里。BiCodec 全局量化器是 FSQ（6 维 × 4 级，`fsq_levels: [4,4,4,4,4,4]`，码本 4096），token = 6 个维度各自 round(bound(u)) 的组合。
- K4 假设：把目标从"连续码 MSE"换成"干净提示 token 的逐维等级分类（交叉熵）"，训练直接优化落在哪个量化格，能让带噪提示的 token 更接近干净孪生 token，从而收回更多身份损失。
- 相关：编解码 token 去噪做抗噪 TTS（LauraTTS 去噪器，arXiv 2505.13830）；FSQ（Mentzer et al. ICLR 2024）。新意：中低（组件常见），价值在于直接针对 oracle 诊断出的"离散格错位"。

## 2. 方法（一个设置，其余与 V1 对照臂完全相同）
- 训练脚本 `experiments/mist_w3po_v1/voicecond/k4_tokencls_train.py`：逐行复制 `k3_discrim_train.py`（= `idea1_e2e_train.py --mode ctrl` + InfoNCE），去掉 InfoNCE，把两项 MSE 换成两项交叉熵。
- 与 V1 相同：数据（`data/open_voicecond_v1/idea1/train_en` + `train_zh`，各 20000 对）、初始化（BiCodec 预训练编码器，与 V1 对照臂同一起点；"V1 编码器骨干"按此理解，**不**从 V1 检查点继续）、可训练模块（ECAPA-TDNN + Perceiver + project_in，BN / dropout eval 模式）、AdamW lr 5e−5、200 步预热 + 余弦到 0、6000 步、每批 32 对、权重衰减 0、裁剪 1.0、fp32、种子 20261119（含 DataLoader 生成器）、只存最终检查点、ID-1 dev 监控只记录。
- **头 / 目标 / 损失（唯一改动）：**
  - 标签：干净提示的 32 个全局 token 拆成 6 维 FSQ 等级，ℓ*_{k,d} = round(bound(u*_{k,d})) + 2 ∈ {0,1,2,3}，u* = φ₀(s)（冻结编码器，即 B 的 tokenizer 在干净语音上的 token，与 V1 的监督来源相同）。
  - 头：无参数的距离分类头，在 FSQ 的 bound 值 b = bound(φ(x)) 上，对 4 个等级中心 c ∈ {−2,−1,0,1} 给 logit_c = −(b − c)² / τ，τ = 0.25（固定，不扫）。其 argmax 恰好等于 FSQ 的 round，所以部署时 token 计算与 V1 完全相同（`Q(φ(y))`），推理路径不变。
  - 损失：L = CE(φ(y), ℓ*) + 1.0 · CE(φ(s), ℓ*)，CE 在 32 × 6 = 192 个维度上取平均；第二项就是 V1 的干净输入一致项（‖φ(s) − u*‖²）换成干净输入上的 CE。
- **不是蒸馏：** 标签是同一条语句干净提示经冻结 tokenizer 得到的 token，与 V1 的 u* 同源同监督，只是离散化；没有教师软输出、伪标签或学生-教师结构。
- **与 voicecond 线的重叠：** K4 是 voicecond Idea 1 对照臂的修改版，与 voicecond 线（Idea 1 / Idea 7、K2、K3）重叠，归属待用户决定。

## 3. 预检（ID-1 diag dev，Talker 评测之前）
- 数据：`data/open_voicecond_v1/idea1/diag_{en,zh}`，split = dev，各 500 对（40 / 107 人），有干净 s、带噪 y。
- 编码器：φ₀（记录）、V1、K3（`k3/encoder/encoder.pt`）、K4。对每种语言报告：
  1. **token 位置完全匹配率**（主指标）：Q(φ(y)) 与干净 token Q(φ₀(s)) 在 32 个位置上完全相同的比例；
  2. 维度等级匹配率（192 维）；干净输入 Q(φ(s)) 的同两项；
  3. BiCodec 说话人条件（`se.detokenize`，即解码器的 d-vector）的余弦：d(Q(φ(y))) 对 d(Q(φ₀(s)))，中位数。（用 d-vector 代替"解码音频再测说话人嵌入"，因为不需要语义 token、几乎零成本；只作记录。）
- **停止规则（固定）：** K4 带噪位置完全匹配率必须在**两种语言**上都 ≥ V1 + 5 个百分点（绝对值；每语言 16000 个位置，抽样标准误约 0.3 点，5 点远大于噪声，且相对 V1 约 10% 的基线是约 1.5 倍）。否则停止，不做 Talker 评测，报告预检。
- 另在 token 步骤记录（不入规则）：S1 dev 带噪条目上 V1 / K3 / K4 token 与 oracle token 的位置完全匹配率与维度匹配率。

## 4. 评测与判断规则（S1 dev，与 oracle_voice / K3 完全相同）
- 底座 S1 单说话人 LA2（`wave2_S1`），同一 Thinker 文本块，T1c Talker（`si_t1c_20261021T181440Z/export_1500`），种子 20261021，4 分片，`idea1_si_infer_vo.py --voice-override`，提示文件与 sha1 同 V1 表（`wave3_combos/V/tokens/prompts`）。K4 token = Q(φ_K4(y))（与 K3 同一计算，只换编码器）。
- 复用：B、V1、K3、oracle 的已有输出与指标；只新跑 K4。
- 指标：ASR-BLEU（speech-only，`asr_speech.jsonl`）、SQUIM PESQ 中位数、TitaNet 相似度中位数（对干净源）；点估计。
- 收回比例（同 oracle 预登记第 4 节）：(臂带噪相似度中位数 − B 带噪中位数) / (B 干净中位数 − B 带噪中位数)，按格（方向 × 集合）。
- **判定格：en2zh / zh2en × {a, b_noxt}（按任务说明，共 4 格）。** 另报 {a, b} 4 格的同样数字（只报告）。
- **判断规则（三条都满足才算 K4 成立）：**
  1. 带噪相似度收回比例在 4 个判定格上的平均 ≥ V1 的平均 + 20 个百分点；
  2. 干净相似度不低于 V1：4 个判定格干净相似度中位数的平均，K4 ≥ V1；
  3. ASR-BLEU：4 个判定格 × {带噪, 干净} 的每一格，K4 ≥ V1 − 0.3。
- 另报 PESQ（记录），以及 K4 对 B、对 oracle 的差距。

## 5. 运行
- `experiments/mist_w3po_v1/voicecond/k4_orch.sh`（setsid nohup 脱离，fds 8/9 关闭；先 `bash -n` 与 `DRY_RUN=1`）：训练（卡 0）→ 预检（卡 0）→ 停止规则 → token（卡 0）→ Talker（en2zh 4 分片卡 0、zh2en 4 分片卡 6，间隔 45 s）→ ASR → SQUIM / TitaNet → `k4_score.py` → `k4/DONE`（失败写 `k4/FAILED`，停止写 `k4/STOPPED_PRECHECK`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `k4_noplaceholder.sh`（POLL 1 s）。不碰 2726951 / 52045 / 1222453，不碰 K1 第 2 步训练（卡 1、2、3、7，torchrun 1602215）。
- 输出 `eval_outputs/mist_w3po_v1/k4/`，日志 `logs/mist_w3po_v1/k4/`。

## 结果
（待填）
