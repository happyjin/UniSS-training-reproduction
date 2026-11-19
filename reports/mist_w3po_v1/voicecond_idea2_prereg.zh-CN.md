# Idea 2（共享背景的流式身份后验 CA-Kalman）预登记 v1

2026-11-19 00:50 布拉格时间。在**合成流抽取背景**、**E2.0 的任何结果**、**任何 EM 拟合**以及**任何电竞或测试集运行**之前提交。

- 方案：`reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md` §1、§3（3.1–3.7）、§7.2–7.3、§7.2d–7.2f。
- 代码：`experiments/mist_w3po_v1/voicecond/idea2_*.py`、`experiments/mist_w3po_v1/scripts/idea2_*.sh`；共享模块 `voicecond/bicodec_phi.py`（只读）。
- 数据：`data/open_voicecond_v1/idea2/`；输出：`eval_outputs/mist_w3po_v1/voicecond_idea2/`。
- 本文件提交之后，任何改动只能以"vN 修订"的形式追加在文末，写明原因和时间，原条目保留。
- 本阶段（M1）只做：数据、预处理、E2.0（G2.0）、E2.2 线性版（G2.2）。不做学习版、端到端、Talker 运行，不碰任何正式测试集（ID-2、OOD-1/2/3）。

---

## 0. 见过的语料（OOD 定义，§7.2e 要求）

M1 及之后的训练、拟合和所有选择**只**用下面这些语料；没有列出的语料对 Idea 2 一律算 OOD。

| 类型 | 语料 | 用到的部分 |
|---|---|---|
| 英文说话人 | LibriSpeech | train-clean-100、train-clean-360（训练）；dev-clean（dev）；test-clean（ID-1 测试） |
| 中文说话人 | AISHELL-3 | train 说话人（训练）；test 说话人按说话人 ID 的 sha1 排序，前一半 dev、后一半 ID-1 测试（与 Idea 1 相同） |
| 中文说话人 | Common Voice zh-CN（团队 S3 本地镜像 `asr.tar.zst`） | 见 §1.1 的筛选；**排除 CoVoST 2 zh-CN→en 的 test / test_dup / test_covost 中出现过的全部说话人（805 个 client_id）和语句** |
| 背景：音乐 | MUSAN music | 只用无人声（vocals = N）的曲目；**不用** `fma`、`fma-western-art` 两个子目录（来源是 FMA，FMA 预定为 OOD-1 背景语料） |
| 背景：噪声 | MUSAN noise | 全部（free-sound、sound-bible） |
| 背景：观众与音效 | FSD50K eval | 只用 §1.2 列出的类别 |
| 轮次统计 | AMI 人工分段标注 | 只用 full-corpus ASR 划分的训练会议的时间统计（不含音频和文本） |
| BVR 分布 | 电竞提示 `prompt.bvr_db` | 只是一个统计量，用来设定混合比例 |
| 真实带噪（无监督项） | CN-Celeb v1 | M1 不用；从 E2.3（学习版）开始用 |

- 不用 MUSAN speech；不用 FMA、DEMAND、WHAM（任何部分）；不用 CoVoST 2 的 test 说话人和语句；AMI / AliMeeting / RealSI / ACL 的音频只做测试；电竞数据只做评测。
- 所有背景抽取排除 `data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS.txt` 中的文件（该清单出现之前不抽背景）。

---

## 1. 合成多人流（E2.1 / D2，ID-1）

构造脚本：`experiments/mist_w3po_v1/voicecond/idea2_build_streams.py`，种子 20261119（每条流的随机数由流 ID 的 sha1 派生）。

### 1.1 说话人与划分（按说话人不重叠）

| 语言 | train | dev | test |
|---|---|---|---|
| 英 | LibriSpeech train-clean-100 + 360 | dev-clean | test-clean |
| 中 | AISHELL-3 train 说话人 + Common Voice（sha1(client_id) 末位十六进制数 mod 10 ∈ 0–7） | AISHELL-3 test 前一半 + CV（mod 10 = 8） | AISHELL-3 test 后一半 + CV（mod 10 = 9） |

- 语句预处理：重采样到 16 kHz；首尾静音（低于语句峰值 −35 dB 的 10 ms 帧）去掉。
- Common Voice 筛选（数据清洗，不是方法超参）：只用 validated 中的语句；单条语句的能量分位 SNR（25 ms 帧能量的 P90 与 P10 之差，dB）≥ 20 dB；去静音后每个说话人累计 ≥ 30 s 才保留；每个说话人最多用 80 条。
- 干净参考 $u^\star_j$：该说话人最多 5 个**干净 6 s 提示**（按固定随机序拼接其语句并切成不重叠的 6 s 段）的 $\phi$ 的中心（量化前 192 维平均）；$g^\star_j=\mathcal Q(u^\star_j)$。

### 1.2 背景素材（按来源文件划分：sha1(相对路径) mod 10 → 0–7 train、8 dev、9 test）

| 层 | 来源 | 用法 |
|---|---|---|
| 音乐床（持续） | MUSAN music（vocals = N，非 fma） | 同一划分里随机选曲首尾相接（0.5 s 交叉淡化）铺满整条流；增益做慢变随机游走（每 5 s 一个节点，标准差 1 dB，限制在 ±4 dB） |
| 环境底噪（50% 的流） | MUSAN noise | 同上方式铺满，电平比音乐床低 6–15 dB（均匀） |
| 观众爆发 | FSD50K eval：标签含 Crowd / Cheering / Applause / Chatter / Clapping，且不含 Speech、Male/Female/Child speech、Conversation、Speech_synthesizer、Singing、Male/Female_singing、Music | 泊松到达，平均每 25 s 一次；每次 2–8 s（截取或拼接），50 ms 淡入淡出；电平比音乐床高 0–8 dB |
| 稀疏音效 | FSD50K eval：标签含 Explosion / Gunshot_and_gunfire / Whoosh_and_swoosh_and_swish / Boom / Fireworks / Crack，且不含上面的人声与音乐标签 | 泊松到达，平均每 15 s 一次；单个片段（最长 3 s）；电平比音乐床高 −3 到 +6 dB |
| 侧链压低 | 有人说话时（任一说话人的干净语音包络 > 峰值 −40 dB）背景整体衰减 $d\sim U(3,8)$ dB；启动 0.15 s、释放 0.6 s 的一阶平滑 | |

- **BVR：** 每条流从电竞提示 `prompt.bvr_db` 中落在 [−25, −3] dB 的经验值里有放回抽一个；定义为（压低之后）真实背景与干净语音在**有人说话的采样点**上的能量比，按此缩放背景轨。
- 混合后峰值超过 0.99 时语音和背景同比例缩小。

### 1.3 轮次结构（AMI 训练会议的统计，`ami_turn_stats.json`）

- 每条流 2–4 个说话人（均匀），语言单一；时长 $U(300,600)$ s。
- 说话人的发言权重 ~ Dirichlet(2)；下一个说话人按权重从"除当前说话人以外"的人中抽。
- 轮次时长从 AMI 经验轮次时长（≥ 0.5 s，≤ 40 s）中有放回抽；轮次间隔从 AMI 经验间隔中抽，负值（重叠）截断到不超过前后两轮中较短者的 80%。
- 每个轮次用该说话人的语句（顺序随机，语句间 0.1–0.4 s 停顿）填满，最后一句在轮次结束处截断（20 ms 淡出）。

### 1.4 规模与划分

- 总计约 100 h：train 640 条、dev 80 条、test 80 条；每个划分中英各半。
- **先导（pilot，约 10 h）：** 每个划分、每种语言的前 10% 条（train 32+32、dev 4+4、test 4+4），配方和划分规则完全相同。先导时训练集英文说话人池只有 train-clean-100（train-clean-360 尚未下载完）；其余 90% 的流使用完整说话人池。所有流的元数据记录所用的说话人池。
- 每条流保存：混合 `mix.wav`、干净语音 `speech.wav`（所有说话人之和，即干净孪生）、真实背景 `bg.wav`（压低之后）、`meta.json`（每段语音：说话人、起止时间、来源语句；背景各层的来源文件与时间；BVR；压低参数）。

---

## 2. 预处理（卡 4–7）

- **分离器：** Nemotron-3-Diarization，`lowlat` 预设（与 `dump_nemotron_posteriors.py` 相同，1.04 s），每条流整条流式处理，保存 80 ms 后验 `post80 [T, 8]`。
- **HDemucs：** `hdemucs_high`，与 `game_muffle_analysis.demucs()` 相同的归一和声部定义（$\hat v$ = vocals，$\hat b$ = drums + bass + other），整条流按 10 s 分段、1 s 重叠的线性交叉淡化处理（分段带来的少量前瞻是本阶段的近似，记录在案）。
- **窗口编码 $z_k$：** 窗长 $L$（2 s 或 4 s，见 §3.1），跳步 0.5 s；`BiCodecPhi.phi_wave`（音量归一、平铺到 6 s，与分词器相同）。对混合、干净孪生、HDemucs 人声三路都编码。
- **干扰特征 $v_k$：** 64 个 mel 带（n_fft 512、窗 400、跳 160，0–8 kHz）。$P_b$ = 窗内 $\hat b$ 每带 mel 功率的时间平均；$\bar P_k$ = 截至该窗结束、分离器判为有人说话（任一槽位后验 > 0.5）的所有帧上 $\hat v$ 的平均 mel 功率（各带平均；还没有说话帧时用本窗的 $\hat v$ 平均功率）；$v_k=(P_b/\bar P_k)^{0.3}$。锚点：$v=0\iff$ 没有背景。
- **槽位与权重：** 窗 $k$ 的帧上，$j(k)=\arg\max_i\sum_t p_i(t)$；$\pi_k=\mathrm{mean}_t\,[p_{j}(t)\prod_{i\ne j}(1-p_i(t))]$（"只有 $j$ 在说"的概率）；$\pi_k<0.3$ 时该窗没有观测 1（$j(k)=\varnothing$），只用观测 2。
- **因果性：** 每个窗口在其结束时刻加上分离器延迟 1.04 s 之后才可用；所有方法（包括首 8 s 提示）都用同一规则。

---

## 3. E2.0 假设检验与 G2.0（先导数据上判定，完整数据上复核）

所有 E2.0 统计只用**oracle 单人窗口**：按 `meta.json`，窗内只有一个说话人在说、且他的语音覆盖窗长 ≥ 90%。拟合用 train，报告用 dev。

### 3.1 窗长规则（先写好，E2.0 中执行一次）

- 对 $L\in\{2,4\}$ s：在干净孪生上，用 oracle 单人窗口的滑动平均（§4 的方法 2，权重 1），经 §3.2 的映射 $M_L$ 到提示空间，计算 dev 说话人在观测到 8 s 语音时的误差 $e_L(8)$（RMS of $\|\hat u-u^\star\|$）。
- **若 $e_2(8)\le1.1\,e_4(8)$ 选 2 s，否则选 4 s。** 之后固定。

### 3.2 窗口空间到提示空间的映射 $M$（规则，不是调参）

窗口编码（短窗平铺）和 6 s 提示编码之间有系统差。三个候选，都只用 train 说话人的**干净**语音拟合（窗口均值 → $u^\star$）：(a) 只加偏置；(b) 逐维仿射；(c) 全矩阵线性 + 偏置，岭系数固定为 $10^{-3}\,\mathrm{tr}(X^\top X)/192$。**在 dev 干净孪生上 60 s 误差最低的那个被选中，之后固定**，对所有窗口类方法统一使用。

### 3.3 四项检验

1. **持续偏差：** 对每个（流，说话人），$b_{js}=\overline{z}^{\rm noisy}_{js}-\overline{z}^{\rm clean}_{js}$（同一批 oracle 单人窗口的平均）。说话人间距 $\Delta$ = 同一划分、同一语言中不同说话人干净窗口均值两两距离的中位数。报告 $\mathrm{median}\|b_{js}\|/\Delta$。
2. **低秩：** train 上窗口级偏移 $\delta_k=z^{\rm noisy}_k-z^{\rm clean}_k$ 的 PCA；**$r$ = 解释 90% 方差所需的最少主成分数**（上限 96，若触顶在日志中记录）。$r$ 在先导数据上定下后固定。
3. **$v$ 的预测力：** train 上做 $\delta_k=Gv_k+c$ 的最小二乘，dev 上算 $R^2=1-\sum\|\delta-\hat\delta\|^2/\sum\|\delta-\bar\delta\|^2$。同时报告不带截距、以及流级平均（持续部分）的 $R^2$。
4. **高斯性：** 残差在前 10 个主成分上的偏度和超额峰度，只报告。

**G2.0（两条都满足才通过）：** $\mathrm{median}\|b_{js}\|/\Delta\ge0.2$；$R^2$（带截距，dev）$\ge0.3$。不通过则停下汇报，不做 E2.2。

---

## 4. E2.2 线性 CA-Kalman 与 G2.2

### 4.1 模型（§3.1）

$z_k=s_{j(k)}+A n_k+\varepsilon_k$，$\varepsilon_k\sim\mathcal N(0,R/\pi_k)$；$v_k=Bn_k+\zeta_k$，$\zeta_k\sim\mathcal N(0,S)$（$S$ 对角，**没有截距**，锚点）；$n_k=\bar n+\tilde n_k$，$\bar n\sim\mathcal N(\mu_{\bar n},\Sigma_{\bar n})$，$\tilde n_k=\rho\tilde n_{k-1}+\eta_k$（$\rho$ 为标量，规范固定为 $\tilde n$ 的平稳方差 $=I$）；$s_j\sim\mathcal N(\mu_0,\Sigma_0)$。

- $\mu_0,\Sigma_0$：train 说话人**干净**窗口均值的样本均值与协方差（加 $10^{-4}\,\mathrm{tr}/192$ 的对角稳定项），不参加 EM。
- 维度：$D=192$，$m=64$，$r$ 由 §3.3 定。

### 4.2 EM（只在开源合成流的 train 部分上拟合；M1 不用 CN-Celeb）

- 观测用**不重叠**的窗口（跳步 = $L$），使 $\varepsilon_k$ 近似独立；每条流的 E 步是全部隐变量 $(s_{1:J},\bar n,\tilde n_{1:K})$ 的精确高斯后验（信息形式，稠密 Cholesky，等价于 RTS 平滑）；M 步闭式更新 $A,B,S,R,\rho,\mu_{\bar n},\Sigma_{\bar n}$（$Q$ 通过规范固定）。
- 初始化：$A,B$ 取窗口级交叉协方差 $\mathrm{Cov}(z,v)$ 的前 $r$ 个奇异向量（各乘奇异值的平方根）；$R$ 取干净窗口的说话人内协方差；$\rho=0.9$；$S$ 取 $v$ 的方差；$\mu_{\bar n}=0$，$\Sigma_{\bar n}=I$。
- 最多 30 次迭代；每次记录边缘对数似然，相对改进 < $10^{-6}$ 时提前停止。唯一设置，不扫参。
- 先导数据上拟合一次（早期信号），完整 train 上再拟合一次（G2.2 用后者）。

### 4.3 在线滤波（评测，跳步 0.5 s，严格因果）

参数换算到 0.5 s 跳步：$\rho_f=\rho^{0.5/L}$，$Q_f=(1-\rho_f^2)I$；重叠窗口的误差相关按等效独立观测数处理：$R_f=(L/0.5)R$，$S_f=(L/0.5)S$。协方差形式卡尔曼滤波（float64），状态 $(s_{\rm 槽位},\bar n,\tilde n)$。

### 4.4 比较的方法（每个（流，真实说话人）对，说话人与分离器槽位按全流重叠最大匹配，仅用于评测）

| # | 方法 | 定义 |
|---|---|---|
| 1 | 首 8 s 提示（现状） | 该槽位的单人片段（$p_j>0.5$ 且其他 $<0.5$，每段 ≥ 0.4 s），按时间取到累计 8 s 为止，拼接后 $\phi$；不足 1 s 时用最近 2 s 原始音频 |
| 2 | 滑动平均 | $\pi_k$ 加权平均 $z_k$（$j(k)$ = 该槽位），→ $M$ |
| 3 | 中位数 | 同一批窗口逐维中位数，→ $M$ |
| 4 | 只做先验收缩 | 卡尔曼，$A=0$、无 $v$（同一 $\mu_0,\Sigma_0,R_f$），→ $M$ |
| 5 | 卡尔曼不用 $v$ | 完整模型同一组参数，去掉观测 2，→ $M$ |
| 6 | 完整线性 CA-Kalman | → $M$ |
| 参照 | 干净孪生滑动平均（oracle 标签）；HDemucs 人声窗口滑动平均 | 不是方法，只作上界和外部参照 |

### 4.5 指标与统计

- 时间轴：该说话人**oracle 累计语音秒数** $t\in\{1,2,4,6,8,10,15,20,30,40,50,60\}$；在流时间上取刚达到 $t$ 的时刻，每个方法用此刻之前可用的全部信息。
- 指标：$e=\|\hat u-u^\star\|_2$（192 维，量化前）；token 恢复率 = 32 个 token 中与 $g^\star$ 相同的比例；汉明距离。
- 统计：以流为簇的 bootstrap（2000 次，种子 20261119），配对差值的 95% CI。曲线拟合只用累计语音 ≥ 60 s 的说话人：每个方法在 $t\in[8,60]$ 上最小二乘拟合 $\overline{e^2}(t)=a+b/t$。

### 4.6 G2.2（完整数据的 ID-1 test 划分；三条都满足才通过）

- **(a)** 60 s 时，完整版的误差低于滑动平均、也低于首 8 s：两个配对差值的 95% CI 上界都 < 0。
- **(b) 不用 $v$ 时出现平台（定理 2.1）：** 方法 5 的 $a_{\rm nov}$ 的 95% CI 下界 > 0，且 $a_{\rm nov}\ge b_{\rm nov}/180$（即拟合曲线在 60 s 处平台至少占误差的一半：$\sqrt{a}\ge0.5\,\hat e(60)$）。
- **(c) 用 $v$ 时没有平台（定理 2.2）：** 以干净孪生的平台 $a_{\rm clean}$ 为基准，完整版的超额平台不超过不用 $v$ 时的一半：$D=(a_{\rm full}-a_{\rm clean})-0.5\,(a_{\rm nov}-a_{\rm clean})$ 的 95% CI 上界 < 0。

另外报告（不是门槛）：首句（$t\le2$ s）结果；token 恢复率；中英分开；§4.7 的切换统计；先导数据上的同一套结果。

### 4.7 输出与切换策略（§3.5，固定）

- 每个说话人在句子边界（M1 中用 oracle 轮次结束）取 $g_j=\mathcal Q(M(m_{j|k}))$；
- 只有同时满足才切换：自上次切换以来 $\mathrm{tr}(P_{j|k})$ 相对下降 ≥ $\tau=20\%$，且 $\mathrm{Ham}(g^{\rm new},g^{\rm old})\ge4$；
- 首句：新说话人只有先验加共享背景估计时，输出 $\mathcal Q(M(m_{j|k}))$（已有观测的后验）。
- 报告每人切换次数、60 s 时输出 token 的恢复率。

### 4.8 之后阶段的固定值（这里先登记，M1 不执行）

- 学习版（E2.3）：$\lambda_{\rm anc}=1$，$\lambda_{\rm syn}=0.1$；
- 流式推理接口：`idea2_cakalman.py` 中的滤波器类按（窗口结束时刻、槽位、$\pi_k$、$z_k$、$v_k$）逐窗更新，可在任意时刻按槽位输出 32 个 token。之后用于电竞 demo 10 段（已有的 Nemotron 后验和 C2 窗口）与 OpenBroadcast-SI 的评测钩子，不在其上做任何选择。

---

## v1.1 修订（2026-11-19 01:00 布拉格，在构建任何合成流之前）

- §1.1 Common Voice 的划分规则原写"sha1(client_id) 末位十六进制数 mod 10"，末位只有 16 个取值，mod 10 会让 0–5 的概率加倍、dev/test 只各占约 6%。改为与背景相同的规则：**int(sha1(client_id) 的十六进制全文, 16) mod 10**，0–7 train、8 dev、9 test。
- 本地镜像中 Common Voice zh-CN 只有 train 划分的音频（29406 条、658 个说话人；去掉 CoVoST 2 test 的 30 个说话人后 628 个）。筛选规则不变。

## v1.2 修订（2026-11-19 01:05 布拉格，在构建任何合成流之前；测试集代理提交 2563d6b94 之后）

- 背景排除：`bench/HELDOUT_BACKGROUNDS.txt`（1560 个 FSD50K eval 片段）和 `HELDOUT_BACKGROUNDS_MUSAN.txt`；MUSAN 清单写出之前，直接套用该文件头登记的同一规则（`int(sha1('OpenBroadcast-SI-v1|' + 路径), 16) % 1000 < 150`，路径为 MUSAN tar 成员名或 `FSD50K.eval_audio/<fname>.wav`），所以结果与清单一致。
- 语音排除：`bench/HELDOUT_SPEECH.txt` 中的 553 个 Common Voice 说话人（我的 CV 说话人池中有 1 个在内，已排除）及其全部语句；CoVoST 2 test 划分（两个方向）整体不用。
- OOD-1 语料 FMA、DEMAND、ESC-50 整体不用（§0 的"不用"清单加上 ESC-50）。

## v1.3 修订（2026-11-19 01:05 布拉格，在任何中文合成流构建之前）

- AISHELL-3 的 `test/wav` 目录里除了 44 个只在测试集出现的说话人，还有 170 个**训练说话人**的少量语句（同一说话人 ID 同时出现在 train 和 test 目录）。§1.1"test 说话人"按目录划分会让说话人跨划分重叠。改为按说话人划分：只要在 train 目录出现过就归 train（其全部语句）；其余 44 个只在 test 目录出现的说话人按 sha1 排序，前 22 个 dev、后 22 个 test。按目录划分的旧索引改名为 `index/speakers_aishell3.jsonl.bad_split_by_folder`（未使用）。

---

# 预登记 v2（M1b：收缩模型）

2026-11-19 01:30 布拉格时间。在**任何收缩模型的拟合**、**任何 E2.2 结果**和**任何电竞运行**之前提交。

- 背景：G2.0 在先导数据上未通过（$R^2(v\to\delta)=0.075$）；诊断显示背景的主要作用是把说话人编码向一个公共点收缩（$z^{\rm noisy}\approx0.63\,z^{\rm clean}+c$），与 Idea 1 的发现一致（编码器饱和，方案命题 10）。
- 协调者决定（用户夜间授权"按最佳选项"）：改用收缩感知模型，先在先导数据上跑 E2.2 拿早期信号，完整 100 h 数据就绪后正式重跑。
- v1 中与本节不冲突的设置（数据、划分、预处理、窗口 L = 2 s、映射 M = 只加偏置、先验 $\mu_0,\Sigma_0$、评测时间点、因果规则、首 8 s 定义、以流为簇的 bootstrap）全部沿用。

## A. 模型

- 观测：$z_k=\lambda_k\,s_j+(1-\lambda_k)\,c+\varepsilon_k$，$\varepsilon_k\sim\mathcal N(0,R/\pi_k)$，$j=j(k)$ 为分离器槽位（$\pi_k\ge0.3$）。
- 收缩权重（每个窗口一个标量）：$\lambda_k=\sigma(\alpha+\beta^\top\tilde v_k)$。
- $s_j$ 在窗口空间；训练目标 $s^\star_j=u^\star_j-b_M$（$b_M$ 为 E2.0 先导选定的偏置映射的偏置，即 $u^\star$ 在窗口空间的像）。输出 $\hat u=\hat s+b_M$。
- **$c$ 的选择规则（先写好）：** 在 train 上同时拟合"全局 $c$"和"按语言 $c_{\rm en},c_{\rm zh}$"两个版本；若 $\|c_{\rm en}-c_{\rm zh}\|\ge0.2\,\bar\Delta$（$\bar\Delta=17.58$，E2.0 先导 train 两种语言说话人间距的平均），用按语言的版本（推理时语言已知），否则用全局 $c$。

## B. 低维干扰摘要 $\tilde v_k$（5 维，固定）

只用该窗口内的 HDemucs 估计（`idea2_feats2.py`）：64 带 mel 功率在窗口内的时间平均 $P_v$（人声）、$P_b$（背景）。
- $\tilde v_{k,1}=10\log_{10}(\sum P_b/\sum P_v)$：窗口级背景对人声比（dB）；
- $\tilde v_{k,1+g}=10\log_{10}(\mathrm{mean}_{G_g}P_b/\mathrm{mean}P_v)$，$g=1..4$，$G_g$ 为 mel 带 0–15、16–31、32–47、48–63；
- 截断到 [−60, 40] dB，按 train 的均值和标准差标准化。

## C. 拟合（只用开源合成流的 train 部分）

- 样本：分离器窗口（$\pi_k\ge0.3$，不重叠，即跳步 = L = 2 s），其槽位按全流重叠匹配到真实说话人（`slot2spk`，匹配不到的窗口不用），目标 $s^\star_j$。
- 目标函数：$\sum_k\pi_k\|z_k-\lambda_k s^\star_j-(1-\lambda_k)c\|^2$，对 $(\alpha,\beta,c)$ 用 L-BFGS（float64，最多 500 步）。初值 $\alpha=\mathrm{logit}(0.63)$、$\beta=0$、$c$ 取对应的闭式最优。
- $R=\frac1N\sum_k\pi_k e_ke_k^\top$（残差加权协方差）。
- **常数 $\lambda$ 版本**（"不用 $v$ 的收缩反解"）：同样拟合，但固定 $\beta=0$。
- 先导：在先导 train 上拟合（早期信号）；正式：在完整 train 上重新拟合（G2.2 用后者）。

## D. 推理（严格因果，跳步 0.5 s）

给定 $\lambda_k$ 时模型是精确的线性高斯模型。每个槽位的后验（信息形式，等价于静态状态的卡尔曼滤波）：
$$\Lambda_j=\Sigma_0^{-1}+\sum_{k}\pi_k\lambda_k^2R_f^{-1},\qquad \eta_j=\Sigma_0^{-1}\mu_0+R_f^{-1}\sum_k\pi_k\lambda_k\big(z_k-(1-\lambda_k)c\big),\qquad \hat s_j=\Lambda_j^{-1}\eta_j$$
$R_f=(L/0.5)R=4R$（重叠窗口按等效独立观测数）。窗口在结束时刻加 1.04 s 后才可用。

## E. 方法（E2.2）

| # | 方法 | 定义 |
|---|---|---|
| 1 | 首 8 s 提示（现状） | 同 v1 §4.4 |
| 2 | 滑动平均 | 同 v1 |
| 3 | 中位数 | 同 v1 |
| 4 | 只做先验收缩 | §D 中令 $\lambda_k\equiv1$、无 $c$，同一 $R_f$（完整模型的） |
| 5 | 常数 $\lambda$ 收缩反解（不用 $v$） | §D，用 §C 常数 $\lambda$ 版本的参数 |
| 6 | 完整收缩模型 | §D，用完整版本参数 |
| 参照 | HDemucs 人声窗口的滑动平均（V 参照）；干净孪生滑动平均（oracle） | 不是方法 |

## F. G2.2（改写）

- 单元：累计 oracle 语音 ≥ 60 s 的（流，说话人）；60 s 时刻的编码误差 $e=\|\hat u-u^\star\|$。
- **通过条件（两条都要）：** $\mathrm{median}(e_{6})-\mathrm{median}(e_{2})$ 和 $\mathrm{median}(e_{6})-\mathrm{median}(e_{1})$ 的 95% 以流为簇的 bootstrap CI（2000 次，种子 20261119）上界都 < 0。
- 同时报告（不是门槛）：相对 V 参照的同一统计量；配对差值的中位数；token 恢复率；曲线 $e(t)$ 和平台拟合 $\overline{e^2}(t)=a+b/t$。
- **理论预测（事先写下）：** 给定 $c$ 全局、$\lambda$ 可观测，$s$ 在每个窗口上可辨识；误差平台等于收缩模型本身的误差，所以完整模型的平台应低于滑动平均的平台。
- 先导 test（8 条流）上的结果是早期信号；正式结论用完整数据的 test 划分。

## G. 先导 G2.2 通过后的电竞 demo（OOD-2，只评测）

- 10 段 demo（`data/game_esports_si_v1`，两个方向）：已有的 Nemotron lowlat 后验和 C2 窗口；HDemucs 按 §2 的方式处理整段原声；窗口编码、$\tilde v$、槽位与 $\pi$ 的计算与合成流完全相同。
- 参数：先导 train 上拟合的完整模型（电竞数据不参与任何拟合或选择）。
- 每个流（item）的 32 个 token = 该槽位在**该流首次输出时刻**（窗口起点 + 首个 block 的 write_ms）的后验均值经 $M$、FSQ 得到；用 `idea1_si_infer_vo.py`（按 baseline 提示文件 sha1 校验）覆盖 Talker 的音色 token，其余与 baseline B 完全相同（两个 TTS 种子、全部流、条目顺序一致），只在卡 4–7 上运行。
- 指标：按流配对的 SQUIM-PESQ、频谱对比度和平坦度（相对 baseline B），盲 ASR-BLEU（`game_blind_bleu.py`），语音 LAAL。
- 验收线（§7.2f）：PESQ 中位数提升 ≥ +0.5，对比度提升 ≥ +1.5 dB，BLEU 不降。
