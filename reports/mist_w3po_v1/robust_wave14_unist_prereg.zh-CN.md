# 鲁棒第十四波：ERM-WiSE 换用 UniST 训练数据（2026-11-20 15:30 CET，写在任何数据构建、训练和评测之前）

## 起因
- 第十三波确认，ERM 在干净中→英上掉分的原因是 CoVoST 中→英训练集太窄：只有 7004 条、22 个说话人，训了约 10.5 个 epoch。证据：训练子集上 +13.31，dev 上 −3.63；离线翻译 −2.77。
- 不含 QABC 的 LA2 基线中，Talker（T1c）是在 UniST 上训练的。UniST 是开源数据集（HuggingFace cmots/UniST）。本机有它的 15 分片 pilot：中→英 683341 条，约 1361 h；英→中 816659 条，约 1502 h，音频齐全。
- 用户 15:20 要求不要让卡空着。按"有需要点头的事按最优选择决定"的授权，开始本波。

## 唯一的变化：训练数据来源
- 两个方向都改从 UniST pilot 取：`data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source`；参考译文用 manifest 的 `translation` 字段。
- 每个方向按 sha1("20261120|w14|<dir>|<id>") 排序取前 120000 条。英→中与第五波相同规模；中→英从 7004 条增加到 120000 条。两个方向都约 0.6 个 epoch。
- 预处理同第五波：`bench_build.trim_norm`（切边、−26 dBFS、1.5–12 s），失败的行丢弃。
- **排除（构建时断言并记录数量）：**
  - UniST 留出验证句：`crc32(id) % 50 == 0`；
  - 与任何评测或测试条目重叠的行：S1 dev 的 (a)/(b) 条目、Idea 2 的 cvt 流、RealSI、ACL、AliMeeting、OOD-3、电竞。按源转写文本归一化后完全相同，或音频文件名、Common Voice clip id 相同来判定。重点是 UniST 英→中里的 CommonVoice_EN 与 CoVoST dev/test 的重叠；
  - bench/HELDOUT_SPEECH.txt 中的说话人（能对上的话）。
- 其余全部与第五波 ERM 相同：
  - 8 个环境（两个方向 × 干净/音乐/人群/音效），噪声库、BVR 取值、限幅相同；
  - 2300 步，全局批量 64，LoRA rank 16、alpha 32，q/k/v/o；
  - AdamW 1e-4，余弦调度，种子 20261119；行号和噪声的 sha1 键沿用 "20261119|w5m|..."。
- **并行方式：** world 4（卡 0–3），每个 rank 16 条，一个微批。只影响梯度累积切分，与第八波实现修正 2 的论证相同。
- **最终系统：** unist_wise05 = W0 + 0.5·Δ（alpha 减半），与 ERM-WiSE 的协议完全相同。unist（α = 1）作为辅助一并评测。

## 评测与判定（写定）
- 链与打分同第五、十、十一波：si_thinker_stream la2 w1 → T1c（4 个分片，种子 20261021）→ 只对语音做 ASR；用 3 次底座运行、噪声底、bootstrap。
- **主判定：**
  1. unist_wise05 相对底座，用统一规则：
     - 所有 (a)/(b) × 干净/带噪单元，text BLEU 和 ASR-BLEU 的点估计 ≥ −0.3，且 CI 上界 ≥ 0；
     - 至少一个带噪单元 text 增益的 CI 下界 > 0，且超过噪声底。
  2. unist_wise05 相对 erm_wise05 的增量：
     - 至少一个单元（干净或带噪）text BLEU 差的 CI 下界 > 0，且点估计超过噪声底；
     - 没有任何 (a)/(b) 单元的点估计 < −0.3 且 CI 上界 < 0（text 和 ASR 都查）。
- **预先说明：**
  - UniST 的译文是合成的，风格与 CoVoST 的人工参考不同，(a) 单元可能因此吃亏。所以 unist（α = 1）在中→英干净上的结果另列，作为第十三波诊断的延续：数据变宽后 α = 1 还掉不掉分。
  - 若第 1 条过、第 2 条不过，记为"换数据对 ERM-WiSE 没有增量"。
- **两条都过时：** 跑完整系统组合（+ V1 音色 + T1c，`wave5_m_combo_*` 的同一协议），再与 ERM-WiSE 的完整系统比较。
- **辅助：** LAAL。

## 卡与时间
- 卡 0–3。数据构建（CPU）约 30 分钟，训练预计 2–3 小时，评测约 1.5 小时。
