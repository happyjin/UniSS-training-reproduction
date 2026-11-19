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

---

## v2 修订（2026-11-19 00:52 布拉格；诊断集构建之前、任何选择之前）

原因：协调者转达了基准（OpenBroadcast-SI v1，提交 2563d6b94）的留出规则和 OOD-1 语料。

1. **背景留出：** 排除以下两份清单中的全部文件，并对每个文件再按基准自己的规则复核一遍：
   - 清单：`bench/HELDOUT_BACKGROUNDS.txt`（FSD50K 1560 条）和 `bench/HELDOUT_BACKGROUNDS_MUSAN.txt`；
   - 规则：`int(sha1('OpenBroadcast-SI-v1|' + 路径)) % 1000 < 150`。
   - 实测：已解压的 686 个 FSD50K 片段中有 91 个属于留出，清单与规则逐条一致。
2. **MUSAN music 只用 hd-classical、jamendo、rfm 三个子集**（仍只取无人声曲目），不用 fma 和 fma-western-art。
   - 原因：这两个子集来自 Free Music Archive，而 FMA 是 OOD-1 的背景语料，整体留给基准。
3. **永不使用：** FMA、DEMAND、ESC-50、CoVoST 2，以及 `HELDOUT_SPEECH.txt` 中的说话人。
4. **背景池规模（留出之后）：**

| 划分 | music | crowd | sfx |
|---|---|---|---|
| dev | 137 | 94 | 182 |
| test | 114 | 68 | 176 |

5. §2 的"见过的语料"相应改为：MUSAN music 的 hd-classical / jamendo / rfm（无人声），以及 FSD50K eval 的 9 个类别（去掉留出部分）。

---

## v3：M2 = 端到端 ZNE-in-the-loop（编码器阶段）

2026-11-19 01:20 布拉格。协调者决定（用户授权按最优选项决定）：免训练 ZNE 到此为止，作为负结果和机制分析保留。本节在任何训练之前提交；除本节写明的内容外，§1–§7 不变。

### v3.1 训练什么

- **可训练：** BiCodec 说话人编码器的全部参数，约 990 万：ECAPA-TDNN 887 万、Perceiver 99 万、`project_in` 774。
- **冻结：** mel 前端、FSQ、解码器和 Talker 全部冻结。
- **选全量微调而不是 LoRA，理由：**
  - 模型小，全量微调的显存和存储代价都可以忽略；
  - ECAPA 是 Conv1d 加 BatchNorm 结构，LoRA 需要自写卷积适配器，引入额外的设计自由度；
  - 与冻结的 Talker 和解码器的兼容性由下面的干净锚点项约束，不依赖低秩限制。
- **BN 和 dropout 一律用 eval 模式**：BN 用冻结的运行统计量，dropout 关闭。只更新权重。

### v3.2 两个模型（同数据、同步数、同参数、同优化器）

记 $\phi_0$ 为原编码器（冻结），$\phi_\theta$ 为微调后的编码器，$u_0^\star=\phi_0(s)$，$y_t=y+(\sqrt t-1)\hat b$，$\hat b=y-\hat v$（HDemucs）。

**(Z) ZNE-e2e：**

$$
L=\big\|\,2\phi_\theta(y_1)-\phi_\theta(y_2)-u_0^\star\big\|^2+1.0\,\|\phi_\theta(s)-u_0^\star\|^2+0.1\,\|\phi_\theta[1,2,4]\|^2
$$

- 曲率项是 $t\in\{1,2,4\}$ 上的二阶均差：$\phi_\theta[1,2,4]=\dfrac{\frac{\phi_\theta(y_4)-\phi_\theta(y_2)}{2}-\big(\phi_\theta(y_2)-\phi_\theta(y_1)\big)}{3}$；
- 推理：$\hat u=2\phi_\theta(y)-\phi_\theta(y+(\sqrt2-1)\hat b)$。

**(C) 去噪编码器对照：**

$$
L=\|\phi_\theta(y)-u_0^\star\|^2+1.0\,\|\phi_\theta(s)-u_0^\star\|^2
$$

- 推理：$\hat u=\phi_\theta(y)$。

**两者共用的设置：**
- 范数为 $u$ 空间（32×6）的逐维均方；
- AdamW，学习率 5e−5，前 200 步线性预热，然后余弦衰减到 0；共 6000 步，每批 32 对；
- 权重衰减 0，梯度裁剪 1.0，fp32，随机种子 20261119；
- 只保存最终检查点。不做早停，不按验证结果挑检查点。
- 训练中每 1000 步在 ID-1 dev 的 200 对上记录损失，只作记录。
- 每个方法只跑这一个设置，不扫参。

### v3.3 训练数据（全部开源；新的 train 划分）

- **英文语音：** LibriSpeech train-clean-100，251 个说话人。LibriSpeech 的划分本身按说话人互斥，与 ID-1 的 dev-clean / test-clean 不重叠。
- **中文语音：** AISHELL-1（OpenSLR 33，Apache 2.0），全部 400 个说话人。
  - 原因：AISHELL-3 的 test/wav 目录里有 170 个训练说话人（Idea 2 的发现）。ID-1 中文的 214 个说话人已经覆盖了 AISHELL-3 的 218 人中的 214 人，所以 AISHELL-3 里不存在与 ID-1 不重叠的说话人。
- **对 M1 的影响：** ID-1 中文的 214 个说话人其实大多是 AISHELL-3 的"训练说话人"。M1 没有任何训练，所以不受影响。
  - 但 BiCodec 本身的预训练数据（Spark-TTS 的 VoxBox）可能包含 AISHELL-3 和 LibriSpeech；这一点对所有方法相同，只影响绝对数值，不影响对比。
- **背景：** 只用 dev 奇偶池（即 ID-1 dev 用的同一批文件），背景类别和构成与 §2 相同。
  - 已排除基准的全部留出文件，也不用 MUSAN 的 fma 子集；
  - ID-1 test 的背景文件（test 奇偶池）不进入训练。
- **规模：** 每种语言 20000 对 = 4 类背景 × 5 档 BVR × 1000，每对 6 s；BVR 规则同 §2（电竞五分位档内抽样）。
- **分离：** $\hat v$ 由 HDemucs 在 GPU 上算（`hdemucs_high`，与 `demucs()` 相同的归一化和 16k↔44.1k 重采样）。
- **存储：** $s$、$y$、$\hat v$ 存成 float16 的 npy，放在 `data/open_voicecond_v1/idea1/train_{en,zh}/`。

### v3.4 评测（M2 范围；不跑 OOD、基准和电竞测试片段）

1. **ID-1 test**（中英各 500 对，说话人和背景文件与训练都不重叠）：
   - 对比组：raw、V（$\phi_0(\hat v)$）、R-t（免训练）、C、Z、oracle；
   - 指标：$\|\hat u-u_0^\star\|$、token 对 $g_0^\star$ 的汉明距离和完全恢复率、只用解码器重合成（干净语义码）的 SQUIM PESQ / 对比度 / 平坦度。
2. **E1.2a（80 条配对流）：** Z 和 C 相对 O、V；指标同 §6.2。
3. **E1.2b（demo 10 段，两个种子，全部流）：** Z 和 C 相对 baseline B；指标同 §6.3。
4. **参照组：** HDemucs 人声提示（V）在 demo 全部流、两个种子上跑一遍，作为试听页的参照，并报告其 demo 数值（不设门槛）。

**门槛 G2a（demo 全部流，按方法分别判定；相对 baseline B，同种子配对）：**
- 两个方向合并的配对 SQUIM-PESQ 中位数提升 ≥ +0.5；
- 频谱对比度中位数提升 ≥ +1.5 dB；
- 每个方向两种子平均的不分说话人 ASR-BLEU 不低于 baseline B；
- 另报：语音 LAAL（两种子平均，容差 0.05 s）、平坦度，以及分方向的结果。

### v3.5 可选（只报告，不设门槛，≤ 1 h，只用开源 dev）

- 幂律重参数化：$x=\lambda^p$，$p$ 在 dev 的真实路径上拟合；节点 $\lambda\in\{1,\sqrt2\}$。
- "经过 V 的外推"：节点 $\lambda\in\{0,1\}$，外推到 $\lambda^\star=1-1/\hat c$，$\hat c$ 取 dev 上 $c$ 的中位数。
- 报告这两者在 $u$ 空间的误差中位数，与 V 对比。

---

## v4：M3 = 冻结编码器 Z 和 C 的测试评测

2026-11-19 03:09 布拉格。协调者决定（用户夜间授权）。本节在任何测试集运行之前提交。

### v4.1 冻结对象与推理设置

**编码器（不再改动）：**

| 编码器 | 文件 | sha1 |
|---|---|---|
| Z（ZNE-e2e） | `m2/zne/encoder.pt` | `f81f65f21818bb936242f7e371f7d5d8d768f90d` |
| C（去噪编码器对照） | `m2/ctrl/encoder.pt` | `fce60570b701e5f18bd7d7fb2ffca7793e06c9bc` |

**推理设置（固定）：**
- 提示 $：baseline B 运行中该条目的音色提示文件，即 `talker_in_all/voice_{causal,fallback}/<id>.wav`；
- $\hat b=y-\hat v$，$\hat v$ 由 HDemucs（`game_muffle_analysis.demucs`，CPU）对整段提示分离得到，长度不足处补零；
- Z 的 token：$\mathcal Q\big(2\phi_Z(y)-\phi_Z(y+(\sqrt2-1)\hat b)\big)$；C 的 token：$\mathcal Q(\phi_C(y))$；
- **全部条目都覆盖**，没有回退，也没有阈值。

### v4.2 OpenBroadcast-SI v1（按 `voicecond_bench.zh-CN.md` 的钩子）

- **条件：** ID-2、OOD-1、clean × zh→en、en→zh × 两个 TTS 种子。
- **方式：** 复用 B 的文本解码，只重跑 Talker。
  - sidecar 为 `{条目 id: 32 个 token}`，经 `scripts/voicecond_bench_orch.py --voice-override` 注入；
  - 参数 `--cap135 --exclude 4 5 6 7`。
- **运行名：** Z 为 `obsi_i1z_<条件>_<方向>__t1c`，C 为 `obsi_i1c_<条件>_<方向>__t1c`。
- **评测：** `bench_eval`，指标为文档 §5 的全部指标，两个方向分开报告。

### v4.3 门槛（与 Idea 2 测试预登记 §D 完全相同的严格规则，便于两条线可比）

按方法、按测试集、按方向分别判定。

- **硬门槛：**
  - tcpASR-BLEU 两种子平均 ≥ B 的两种子平均；另报不分说话人的 ASR-BLEU；
  - 语音 LAAL（分离器块协议，两种子平均）≤ B。另报 0.05 s 容差下的结论；
  - 身份：按流配对的说话人相似度中位数差（方法 − B，两种子平均后）≥ −0.02。OpenBroadcast-SI 用 WavLM-SV（对干净 CoVoST 片段）。
- **主质量指标：** 按流配对的 SQUIM-PESQ（两种子平均后）。报告中位数差、bootstrap 95% CI 和 Wilcoxon；每个方法在 6 个"条件 × 方向"上做 Holm 校正。
- **clean 无害：** clean 条件上的硬门槛同上；PESQ 中位数差的 95% CI 上界 ≥ 0，即不得显著变差。

### v4.4 主要假设 H1：Z − C 在 OOD-1 上大于在 ID-2 上

- **主统计量（每个方向分别）：**

2712446
\mathrm{DiD}_m=\big(\mathrm{PESQ}_Z-\mathrm{PESQ}_C\big)_{\text{OOD-1},m}-\big(\mathrm{PESQ}_Z-\mathrm{PESQ}_C\big)_{\text{ID-2},m}
2712446

  - $ 为孪生流对（ID-2 与 OOD-1 的语音流逐条相同）；先取流内条目的均值，再对两个种子取平均。
  - 对流做 bootstrap（10000 次，种子 0），取单侧 95% 下界；两个方向做 Holm 校正。
  - 下界 > 0 → 支持 H1。
- **次要（只报告）：**
  - 同样的 DiD，用在对比度、说话人相似度和编码误差上；
  - 编码误差 = $\|\hat u-\phi_0(y_{\rm clean})\|$，{\rm clean}$ 取 clean 孪生条件中同一条目 id、且时长相差不超过 0.1 s 的提示；报告覆盖率。

### v4.5 OOD-3 无害（AMI / AliMeeting 测试集，只测）

- **子集规则：** 与 Idea 2 预登记 §C 相同，按 `sha1('idea2-ood3-v1|' + 会议ID)` 从小到大取前 2 场。
  - AMI：EN2002d、IS1009c，两个方向都用这两场；
  - AliMeeting：R8009_M8021、R8009_M8024。
- **对照：** B 用同一子集链路。
  - 若 Idea 2 已发布子集链路和 B 子集运行（`idea2_ood3_b_*`），就直接复用，只读；
  - 否则由本线构建，并在 `logs/mist_w3po_v1/voicecond/ood3_subset_coord.txt` 中登记，供 Idea 2 复用。
- **运行名：** `idea1_ood3_{z,c}_{ami,en,ali}__t1c{,_tts2}`。
- **门槛：** 同 v4.3 的硬门槛；身份用 TitaNet，参照为整段 HDemucs 人声中该槽位的全部单人片段（与 Idea 2 v2.1 的定义相同）。

### v4.6 M2 的 demo 身份指标（补充，协调者要求）

- TitaNet 对 Idea 2 定义的参照：整段 HDemucs 人声中，该槽位全部单人片段的拼接（`data/open_voicecond_v1/idea2/game/<方向>/spkref/<会议>_S<k>.wav`，只读）；
- 与 B 按流配对。
