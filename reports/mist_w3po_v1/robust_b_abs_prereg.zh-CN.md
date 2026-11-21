# 鲁棒 B：分析–合成式音色 token 搜索（AbS-Voice，免训练）预登记

2026-11-21 11:05 CET（布拉格）写，早于 B 的任何搜索、预检或 Talker 数字。方案来源 `robust_idea_abs_voice_20261121.zh-CN.md`；本文按调度给出的设定落实（效用用 TitaNet、评判用 ECAPA2，与方案草稿里"评测用 TitaNet"相反，以本文为准）。

## 0. 已知事实（来自 A / D1 / D0 / oracle 的结果节）
- oracle（带噪条目换成干净孪生提示的 32 个全局 token）收回 B 噪声损失约 100%（TitaNet 101% / ECAPA2 96%）；V1 26% / 46%；K5-full 38% / 51%。
- A（训练 TitaNet → 码的 MLP）在 651 个训练说话人上过拟合，预检比 V1 还差。所以这次**不训练任何东西**：搜索在每个提示上单独进行，没有说话人可记。
- D1：身份跟着解码器 d-vector 的位置走，token 翻转本身无害。D0：约 3 s 带噪提示上 TitaNet-large 几乎噪声不变，BiCodec φ0 不是。

## 1. 假设
在 V1 码附近的离散 token 空间里，以"BiCodec 解码器重合成的提示音 与 带噪提示本身 的 TitaNet 余弦"为效用做有限预算的贪心搜索，选出的 32 个全局 token 在**独立评测器 ECAPA2** 下更接近干净孪生的身份，从而让 Talker 收回比 V1 多的噪声损失。

## 2. 方法（一个设置，不扫参；精确流程只写这一次）
记号：一行 = S1 dev 一个条目（wave3 V1 表的行，提示文件 `wave3_combos/V/tokens/prompts/<d>/<key>.wav`，sha1 校验）。FSQ 每个 token 6 维、每维 4 级，等级 ℓ ∈ {−2, −1, 0, 1}，id = Σ_j (ℓ_j + 2)·4^j。
1. **起点 s0**：V1 表里该行的 32 个 token（即 V1 臂实际用的 token）。另用 V1 编码器在同一提示上重算连续码 u（`load_encoder` + `phi_wave(read_mono64(p))`，与 V1 表同一方式），b = bound(u)（32×6）。
2. **候选维度与翻转方向**（每行一次性定好）：对 192 个维度 i，当前等级 ℓ_i（取自 s0），翻转目标 a_i = ℓ_i + 1（若 b_i > ℓ_i）否则 ℓ_i − 1；越界（< −2 或 > 1）时取另一侧。边界距离 m_i = 0.5 − |b_i − ℓ_i|（越小越靠近取整边界；s0 与 round(b) 不一致的维度 m_i < 0，排最前）。按 m_i 升序排成队列 Q（并列按维度序号）。
3. **效用** U(s) = cos(TitaNet-large(R(s)), TitaNet-large(带噪提示))，R(s) = BiCodec 解码（`model.detokenize`）[全局 token s，**该提示自己的语义 token**]（语义 token 由冻结 BiCodec 对同一提示文件 `tokenize` 得到，与 B 推理时 `encode_wav_to_tokens` 同一计算）。TitaNet 前向与 `wave3_v_spksim.emb` 相同（fp32，L2 归一化）。
4. **贪心坐标搜索，固定预算 64 次候选评估**（另加起点 1 次）：当前最优 s = s0，U* = U(s0)。共 4 轮，每轮 16 次评估，作一个 batch：
   - 前 15 个候选：s 上各翻转 Q 中**下一个未试过**的维度（单维翻转，依次取 Q 的第 1–15、16–30、31–45、46–60 个，若某轮触发下面的"补位"则顺延）；
   - 第 16 个候选：若本轮单翻转中有 ≥ 2 个使 U 超过 U*，则为"s 同时翻转所有这些维度"的组合候选；否则改为再取 Q 中下一个未试维度做单翻转（补位）。
   - 本轮结束：s ← 本轮所有候选与 s 中 U 最大者（只接受严格提高），U* 随之更新。下一轮的单翻转都基于新的 s。
   - 输出：4 轮后的 s（即 65 次评估中 U 最大者）。全部行（带噪与干净）都做搜索：部署时不知道输入是否带噪。
5. **不用 ECAPA2**：搜索只用 TitaNet，ECAPA2 只在评估里出现。
6. 运行环境：一个 nemo_diar 进程里同时放 TitaNet 和 BiCodec。nemo_diar 缺 BiCodec 需要的纯 Python 包 einx / frozendict，用 `.tmp/b_abs_pyshim/` 里指向 vllm_omni_030 已装包的符号链接加到 PYTHONPATH（**不安装任何包、不新建环境**）。fp32，batch = 每轮 16 个候选。
7. **延迟**：每行记录（cuda 同步后的墙钟）语义 token 化、V1 码、搜索（TitaNet 参考嵌入 + 4 轮）各自耗时；**额外延迟 = 搜索耗时**（语义 token 与 V1 码在 B / V1 里本来就要算）。每卡只跑一个搜索进程，报中位数 / p90。
8. 脚本：`experiments/mist_w3po_v1/voicecond/b_abs_search.py`（搜索 + 重合成 + d-vector）、`b_abs_precheck.py`（ECAPA2 重合成评估 + 合并 token 表 + 预检判定）、`b_abs_score.py`（Talker 评分）、`b_abs_orch.sh`、`b_abs_noplaceholder.sh`。

## 3. 预检（Talker 评测之前，硬门槛）
S1 dev 带噪条目（en2zh 637、zh2en 1006）：
1. （记录）中位数 cos(d(s_AbS), d(oracle token)) 与 V1 的中位数 cos(d(V1 token), d(oracle token))（V1 参考约 0.891 / 0.897）；
2. （判定）ECAPA2 相似度 cos(ECAPA2(R_noisy(s)), ECAPA2(干净孪生提示))，R_noisy(s) = 带噪提示的语义 token + 全局 token s 的重合成；干净孪生提示 = `oracle_voice/tokens/twin_prompts/<d>/<key>.wav`。比较 s = AbS 与 s = V1。
- **停止规则（事先写定）**：两个方向都满足 mean(ECAPA2 相似度 AbS) − mean(ECAPA2 相似度 V1) ≥ 0.02 才继续；否则写 `b_abs/STOPPED_PRECHECK`，不跑 Talker。
- 另记录（只记录）：同样的重合成相似度对 B 带噪 token、oracle token 的值；TitaNet 版本的重合成相似度（是搜索效用的近亲，不作判定）；搜索效用 U 的提高量、接受的翻转数、与 oracle 的汉明距离、延迟分布。

## 4. Talker 评测（预检通过才跑；S1 dev，与 oracle_voice / K5 / A 完全相同）
- 底座 S1 单说话人 LA2（`wave2_S1`），同一 Thinker 文本块，T1c Talker（`si_t1c_20261021T181440Z/export_1500`），种子 20261021，每方向 4 分片，`idea1_si_infer_vo.py --voice-override tokens_abs_<d>.jsonl`（与 V1 / oracle 同一路径，Talker LM 与 BiCodec 解码器都用这 32 个 token）。
- 复用 B、V1、K5-full、oracle 的已有输出与 TitaNet / SQUIM / ECAPA2 指标（ECAPA2 来自 `a_sv/metrics/`）；只新跑 AbS。
- 指标：ASR-BLEU（speech-only，`asr_speech.jsonl`）、SQUIM PESQ 中位数、相似度中位数（对干净源；**ECAPA2 为主评判**，TitaNet 次要，因为它是搜索效用）；点估计。

## 5. 判断规则（四条都满足才算成立）
- 收回比例 = (臂带噪相似度中位数 − B 带噪中位数) / (B 干净中位数 − B 带噪中位数)，按格；判定格 en2zh / zh2en × {a, b}；b_noxt 只报告。
1. **ECAPA2** 带噪收回比例 4 格平均 ≥ V1 平均 + 15 个百分点（TitaNet 同样报告，不入规则）；
2. 干净相似度不低于 V1：4 格干净 ECAPA2 相似度中位数的平均 ≥ V1（TitaNet 同样报告）；
3. ASR-BLEU：4 格 × {带噪, 干净} 每一格 ≥ V1 − 0.3；
4. PESQ：4 格 × {带噪, 干净} 每一格 ≥ V1 − 0.1。
- 另报：延迟（第 2 节第 7 条）、PESQ 对 oracle 的差。

## 6. 运行
- `experiments/mist_w3po_v1/voicecond/b_abs_orch.sh`（setsid nohup 脱离，fds 8/9 关闭；先 `bash -n` 与 `DRY_RUN=1`），**只用卡 4、5、6、7**（卡 0–3 属于另一会话，不碰）：
  1. 搜索：4 个进程（卡 4–7 各一个），进程 k 依次处理 en2zh 与 zh2en 的第 k 个分片（行号 mod 4）；
  2. 预检（卡 4）：ECAPA2 + 合并 token 表 + `precheck.json`；不过就停；
  3. Talker：AbS en2zh 分片 0、1 在卡 4，2、3 在卡 6；zh2en 分片 0、1 在卡 5，2、3 在卡 7（间隔 45 s）→ ASR → SQUIM / TitaNet / ECAPA2 → `b_abs_score.py` → `b_abs/DONE`（失败写 `b_abs/FAILED`）。
- 占位只按 PID 停（启动前 SIGTERM，最多等 120 s 直到卡上没有 gpu_load 进程）；自带 watcher `b_abs_noplaceholder.sh`（POLL 1 s）。不碰 2726951 / 52045 / 1222453。
- 输出 `eval_outputs/mist_w3po_v1/b_abs/`，日志 `logs/mist_w3po_v1/b_abs/`。

## 结果
（待填）
