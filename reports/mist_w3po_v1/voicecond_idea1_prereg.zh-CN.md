# Idea 1（零干扰外推 ZNE）预登记 v1

2026-11-19 00:35 布拉格时间。在**外推族选择**、**E1.0 / E1.1 的任何结果**以及**任何电竞运行**之前提交。

- 方案：`reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md` §1、§2（尤其 §2.10、§2.12）、§7.2c、§7.2e、§7.2f。
- 代码：`experiments/mist_w3po_v1/voicecond/idea1_*.py`；共享模块 `voicecond/bicodec_phi.py`（只读）。
- 输出：`eval_outputs/mist_w3po_v1/voicecond_idea1/`；数据：`data/open_voicecond_v1/idea1/`。
- 本文件提交之后，下面任何一项若要改动，只能以"v2 修订"的形式追加在文末，写明原因和时间，原条目保留。

---

## 1. 固定设置（验证阶段只跑这一个设置，不扫参）

**编码映射 φ：** `BiCodecPhi.phi_wave`，即 BiCodec 全局编码量化前的连续码 $u\in\mathbb R^{32\times6}$（音量归一、取前 6 s、不足平铺，与 `encode_wav_to_tokens` 逐 token 一致）。距离一律在 $u$ 空间（192 维）用欧氏范数，另报 token 汉明距离。

**背景估计 $\hat b$：** torchaudio `hdemucs_high`（MUSDB18-HQ+ 预训练，权重 `cache/torch/hub/checkpoints/hdemucs_high_trained.pt`），调用 `game_muffle_analysis.demucs()`：
- $\hat v$ = vocals 声部（重采样回 16 kHz）；
- $\hat b=y-\hat v$，按时间对齐，长度不足处补零到 $\mathrm{len}(y)$。

**放大路径：** $y_\lambda=y+(\lambda-1)\hat b$，功率比 $t=\lambda^2$，$f(\lambda)=\phi(y_\lambda)$。

**三个候选外推族（§2.12，公式固定）：**

| 族 | 节点 | 估计量 | 编码次数 |
|---|---|---|---|
| R-t | $t\in\{1,2\}$，即 $\lambda\in\{1,\sqrt2\}$ | $\hat u=2f(1)-f(\sqrt2)$ | 2 |
| Q-λ | $\lambda\in\{1,\sqrt2,2\}$ | Lagrange 外推到 $\lambda=0$ | 3 |
| M-anc | $t\in\{1,2\}$ 加锚点 $u_b=\phi(\hat b)$ | $A=f(1)-u_b$，$B=f(\sqrt2)-u_b$，$\hat\rho=\max\!\big(0,\langle A-2B,B-A\rangle/\|A-2B\|^2\big)$，$\hat u=u_b+(1+\hat\rho)A$ | 3 |

输出 $\hat g=\mathcal Q(\hat u)$（BiCodec FSQ），同时送给解码器和 Talker。

**默认族：** R-t（预先指定的主设置）。最终使用的族由 §3 的规则在开源 dev 数据上选定，之后不再更改。

---

## 2. 开源诊断集（ID-1）

**构造脚本：** `experiments/mist_w3po_v1/voicecond/idea1_build_diag.py`，随机种子 20261119。

**每种语言 1000 对 = 4 种背景 × 5 档 BVR × 50；每个格子 25 对 dev、25 对 test。**
- 每对：6 s（96000 采样点）干净语音 $s$、真实背景 $b$、混合 $y=s+b$、HDemucs 的 $\hat v$ 和 $\hat b$。
- 若 $y$ 的峰值超过 0.99，则 $s$、$b$ 同比例缩小（混合关系不变）。

**语音（干净）：**

| 语言 | dev | test |
|---|---|---|
| 英 | LibriSpeech dev-clean（40 人） | LibriSpeech test-clean（40 人） |
| 中 | AISHELL-3 test 说话人中按说话人 ID 的 sha1 排序取前一半 | 另一半 |

- 每对的 $s$：同一说话人的若干条语句，首尾去掉静音（25 dB 以下）后拼接，截取 6 s；
- 说话人在 dev 和 test 之间不重叠。

**背景：** 背景文件在 dev 和 test 之间按文件名 sha1 的奇偶不重叠；并且**排除** `data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS.txt` 中留给 OpenBroadcast-SI 的全部文件（该清单出现之前不抽背景）。

| 类别 | 来源 | 6 s 的构成 |
|---|---|---|
| music | MUSAN music，只用无人声（ANNOTATIONS 中 vocals = N）的曲目 | 随机起点截 6 s |
| crowd | FSD50K eval 中标签含 Crowd / Cheering / Applause、且不含 Speech / 男声 / 女声 / 儿童语音 / Conversation / Singing / Music 的片段 | 同类片段随机拼接到 6 s |
| sfx | FSD50K eval 中标签含 Explosion / Gunshot_and_gunfire / Whoosh_and_swoosh_and_swish / Boom / Fireworks / Crack、且不含上述人声与音乐标签的片段 | 同类片段随机拼接到 6 s |
| mixed | 各取一段 music、crowd、sfx，先各自归一到单位 RMS | $b=m+0.7c+0.7x$ |

**BVR：**
- 定义：$10\log_{10}(\sum b^2/\sum s^2)$，在 6 s 上计算。
- 五档：电竞提示实测 BVR（`game_muffle_v1/game_{zh2en,en2zh}.jsonl` 的 `prompt.bvr_db`，444 条）的五分位档。边界为 −51.37 / −23.97 / −16.31 / −12.97 / −9.09 / −0.75 dB。
- 每对的目标 BVR：从该档内的电竞实测值中随机抽一个。这是一个统计量，只用来设定混合比例。

**"见过"的语料（供 §7.2e 的 OOD 矩阵登记）：**
- 语音：LibriSpeech dev-clean、test-clean；AISHELL-3 test 说话人；
- 背景：MUSAN music（无人声）；FSD50K eval 的上述 9 个类别；
- 冻结模型：HDemucs（MUSDB18-HQ）、BiCodec、Talker T1c；
- ZNE 没有学习到的参数。只有"外推族选择"这一个离散决定，它只用 dev 部分。

**划分用途：**
- dev：E1.0 诊断、外推族选择、G1.0、E1.1；
- test（ID-1）：M1 中不运行，留到之后一次性评测。

---

## 3. 外推族选择规则（§2.12，只执行一次）

- **数据：** 诊断集 dev 部分，中英合并，共 1000 对。若中文数据到 2026-11-19 03:00（布拉格）仍未就绪，就只用英文 dev 500 对，并在 LOG 记录。
- **统计量：** 每个族用自己的节点得到 $\hat u$，计算 $\|\hat u-u^\star\|_2$ 的中位数，其中 $u^\star=\phi(s)$。
- **选择：** 按简单程度 R-t < Q-λ < M-anc（线性两节点 < 线性三节点 < 非线性带锚点），取**最简单的、且中位数不超过最小中位数 1.05 倍的族**（即差距小于 5% 时取更简单的）。
- 选定之后，E1.1、E1.2 及以后的所有运行只用这一族。另外两族只作为机制分析报告。
- **补充（只报告，不影响选择）：** 只在 music 背景的 dev 对上重复选择，看结论是否相同（§7.2c 的留一类别检查）。

---

## 4. E1.0 诊断与门槛 G1.0

**测量（dev 对）：**
- 真实路径 $f_T(\lambda)=\phi(s+\lambda b)$ 和估计路径 $f_E(\lambda)=\phi(y+(\lambda-1)\hat b)$，在 $\lambda\in\{0,0.25,0.5,\sqrt{0.5},1,\sqrt2,1.5,2,3\}$ 上测量，另加锚点 $\phi(b)$、$\phi(\hat b)$；
- $\lambda$ 上的二次拟合 $c_0+c_1\lambda+c_2\lambda^2$，逐分量最小二乘，取 $\lambda\in[0,2]$ 内的节点，用真实路径；报告 $\|c_1\|_2/\|c_2\|_2$（命题 9）；
- 尺度不变性：$\|\phi(\alpha y)-\phi(y)\|_2/\|\phi(y)-u^\star\|_2$，$\alpha\in\{0.5,2\}$；
- 分离器参数：在波形上把 $\hat b$ 对 $[b,s]$ 做最小二乘，$\hat b=cb+\kappa s+e$，报告 $c$、$\kappa$ 和 $\|e\|/\|\hat b\|$ 的分布；
- 曲率与斜率：
  - 在所选族的外推变量上（R-t 和 M-anc 用 $t$，节点 $t\in\{0,0.0625,0.25,0.5,1,2\}$；Q-λ 用 $\lambda$，节点 $\lambda\in\{0,0.25,0.5,\sqrt{0.5},1,\sqrt2,1.5,2\}$），对真实路径逐分量拟合二次函数；
  - $M_2=\|2c_2\|_\infty$，斜率 $=\|c_1\|_\infty$；
  - 另报二范数版本和 $M_2$/斜率之比的分布。

**G1.0（两条同时满足才通过；在中英合并的 dev 对上判定）：**
1. 至少 70% 的对满足 $M_2<\|f'(0)\|$，即 $\|2c_2\|_\infty<\|c_1\|_\infty$；
2. 所选族的 $\|\hat u-u^\star\|_2$ 中位数小于 $\|\phi(\hat v)-u^\star\|_2$ 的中位数（HDemucs 人声提示）。另报按对配对的胜率和配对差的 bootstrap 95% 置信区间。

**不通过：** 停止，汇报（M1 规定）。

---

## 5. E1.1 只用解码器重合成（dev 对）

- 语义码：干净 $s$ 的 BiCodec 语义 token，固定不变。
- 全局 token 四种：
  - raw：$\phi(y)$；
  - V：$\phi(\hat v)$；
  - ZNE：所选族；
  - oracle：$\phi(s)$。
- 指标：
  - 对 $g^\star$ 的汉明距离和完全恢复率；
  - 余量 $\delta_r$ 的分布和命题 5 的检验：$\|\hat u-u^\star\|_\infty<\min_r\delta_r/h$ 时是否完全恢复；逐维 $|\hat u_r-u^\star_r|\ge\delta_r/h$ 的计数与实际翻转数的对照；
  - 解码语音的 SQUIM PESQ / SI-SDR、频谱对比度、平坦度（`game_voice_texture.feats`）；
  - 说话人相似度：独立的说话人确认模型 TitaNet-large（`nvidia/speakerverification_en_titanet_large`，本地缓存，nemo_diar 环境），计算对干净 $s$ 的余弦；
  - 拟合引理 0 的常数：PESQ 对 $\|q-q^\star\|$ 和汉明距离回归。

---

## 6. E1.2 电竞 dev（只做评测）

### 6.1 探索性探针（协调者 2026-11-19 00:30 要求；在外推族选择之前运行）

- 设置：用 R-t 族，在 80 条配对流 × 2 个方向上运行，与 `game_swap_*` 完全相同（同一 items，同一顺序，同一种子），只把 voice token 换成 R-t 的结果。
- $\hat b$：O 提示减去已有的 HDemucs 人声（`game_muffle_v1/vocals_prompt/<方向>/<流 id>.wav`）。
- 标注为**探索性**：结果**不参与外推族选择**（选择只按 §3 在开源数据上做）。
- 若选定的族就是 R-t，这次运行就作为 E1.2a 的正式结果（设置完全相同，token 表确定性生成）；否则用选定的族重跑 E1.2a。

### 6.2 E1.2a（80 条配对流 × 2 个方向）

- 三组：O（原提示）、V（HDemucs 人声）、Z（ZNE）。O 和 V 用已有的运行；
- Z 的 voice token 用 `idea1_si_infer_vo.py` 覆盖：按提示文件 sha1 校验，缺失即报错；
- 指标：生成语音的 SQUIM PESQ（主）、SI-SDR、频谱对比度、平坦度；
- 统计：按流配对；Wilcoxon 符号秩检验；配对均值差和中位数差的 bootstrap 95% 置信区间（10000 次，种子 0）；主要比较（Z 对 O 的 PESQ，两个方向）做 Holm 校正。

### 6.3 E1.2b（demo 的 10 段，两个 TTS 种子）

- 用编排器 `game_orch.py` 的同一条 T1c 语音链；只把 si_infer 换成 `idea1_si_infer_vo`（voice token 覆盖），其余参数、分片、种子不变；
- 新运行名 `game_b_<方向>__t1c_zne`（以及 `_tts2`），保留全部流的完整输出，布局与 `speech/game_b_*__t1c/talker` 相同，供试听网页做 A/B 切换；
- 指标：
  - 不分说话人的 ASR-BLEU（`game_blind_bleu.py`，银标参考），每个种子分别计算；
  - 语音 LAAL（编排器的 `speech_latency_lit`，`unit.speech_laal_online.all`）；
  - 全部流配对的 SQUIM PESQ、频谱对比度、平坦度，对比 baseline B（`game_b_*__t1c`，同种子）。

### 6.4 门槛 G1.2（两条同时满足才通过）

1. **E1.2a：** 两个方向合并（160 对），Z 相对 O 的配对 PESQ 均值提升 ≥ 0.3，且 bootstrap 95% 置信区间下界 > 0；
2. **E1.2b 硬门槛：** 每个方向两个 TTS 种子平均的不分说话人 ASR-BLEU 不低于 baseline B 的两种子平均；另报逐种子的结果。
   - 语音 LAAL 两种子平均不得比 baseline 差 0.05 s 以上（voice token 不在逐 token 路径上，预期不变）。

### 6.5 试听验收线（§7.2f，协调者要求在 G1.2 一并报告；不替代 G1.2）

在 demo 的 10 段、全部流上，相对 baseline B：
- 配对 SQUIM-PESQ 中位数提升 ≥ +0.5；
- 频谱对比度提升 ≥ +1.5 dB，平坦度下降；
- 不分说话人的 ASR-BLEU 不降。

---

## 7. 数据卫生

- 选择和拟合只用开源 dev 数据；
- 电竞数据（80 条配对流、demo 10 段）只做评测；
- M1 中不跑开源诊断集的 test 部分、OpenBroadcast-SI、AMI / AliMeeting、任何电竞测试片段；
- 不在 OpenBroadcast-SI 上做任何选择或调参。
