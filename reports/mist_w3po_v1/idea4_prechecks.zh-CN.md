# Idea 4 预检（干扰寄存器 + GMM 先验 + 单调证据门控）：三项 CPU 预检

## 0. 预登记

2026-11-19 07:45 布拉格时间（CET）写定，**在计算任何一项统计量之前**。本节之后的改动只能以“修订”追加在文末，原条目保留。

- 目的：判断增强版 Idea 4（BiCodec 说话人条件的干扰寄存器 / 子空间、干扰的 GMM 先验、单调证据硬门控）是否值得实现。
- 只做分析：CPU（`CUDA_VISIBLE_DEVICES=`，`OMP_NUM_THREADS=8`，单进程），不训练，不用 GPU，不编码新音频（全部复用已存的预量化编码）。
- 只用开源数据的 dev 部分；不碰任何测试集（ID-1/ID-2/OOD-1/2/3）、电竞、RealSI、ACL、AliMeeting。
- 脚本：`experiments/mist_w3po_v1/voicecond/idea4_prechecks.py`；输出：`eval_outputs/mist_w3po_v1/idea4_prechecks/summary.json`（拒绝覆盖）。
- 每项只有一个设定；不在结果出来后调设定。

### 0.1 数据（全部已存在）

| 记号 | 来源 | 内容 |
|---|---|---|
| D1（检查 1、2 主数据） | Idea 1 诊断集 dev：`data/open_voicecond_v1/idea1/diag_{en,zh}/manifest.jsonl` + 已存编码 `eval_outputs/mist_w3po_v1/voicecond_idea1/e10/{en,zh}_dev_u.npz` | 1000 对（英 500、中 500）同一语句的干净 / 带噪 6 s 提示；$u(s)$ = `ustar`，$u(s+b)$ = `true[:, 4]`（网格 λ = 1）；192 维预量化编码（提示空间）；标签：背景类别 `cat` ∈ {music, crowd, sfx, mixed}、BVR 分箱 `bin`（0–4）、真实 BVR `bvr_true`（SNR = −BVR） |
| D2（检查 1、2 次要数据） | Idea 2 合成流 dev（80 条，英 40、中 40）：`pre/win_L2`、`pre/feats_L2`、`pre/feats2_L2`、`streams/*/{speech,bg}.wav`、`meta.json` | oracle 单人窗口（`single ≥ 0`），不重叠（每 4 个 0.5 s 跳取 1 个，即 2 s）；$d=z_{\rm mix}-z_{\rm speech}$（窗口空间）；标签：窗口内活跃的背景层（只有音乐床 / +观众 / +音效 / +观众+音效，由 `meta.background` 的事件时间确定）；窗口真实 SNR 由 speech.wav 和 bg.wav 算 |
| D3（检查 3） | 同 D2 的 80 条 dev 流 | Idea 2 冻结参数 `eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all/full.npz`（sha1 25aaf6cd…）；干净参考 $u^\star$ 用 `refs/ustar_v3.npz`（`load_refs` 取最新版本） |
| 说话人子空间 | `refs/ustar_v3.npz` 中只在 **train** 流里出现的说话人 | 干净编码 $u^\star$ 的中心化 PCA |

### 0.2 检查 1：干扰的低秩性

- $d=u(s+b)-u(s)$，展平 192 维。
- **主统计量（判定用）：** D1 全部 1000 对，**不中心化** SVD，能量占比 $E(k)=\sum_{i\le k}\sigma_i^2/\sum_i\sigma_i^2$。不中心化是因为公共偏移本身就是寄存器可以吸收的一个方向。
- **通过线：$E(8)\ge 0.80$。**
- 同时报告（只作信息）：
  - $k=1..16$ 的 $E(k)$，达到 80% / 90% 所需的 $k$；中心化 PCA；分语言；
  - 零假设参照：每一维在样本间独立打乱（保留每维方差，破坏维间相关）后的 $E(8)$；
  - 每个 token 6 维汇集（$32\times1000$ 个 6 维样本）的不中心化 $E(k)$，$k=1..6$；
  - 去掉收缩部分后的残差 $r=d-(\lambda-1)u(s)-(1-\lambda)c$：D1 上 λ、c 无冻结版（Idea 2 的 λ 依赖窗口特征 ṽ，D1 没有），所以在 D1 上样本内拟合两个版本：S1 = 常数 λ + 按语言 c（最小二乘）；S2 = 每对一个 λ_i + 按语言共享 c（交替最小二乘，20 轮；对收缩最宽松，是“收缩能解释多少”的上界）。D2 上用冻结参数（λ_k 由 ṽ_k 给出，按语言 c），$r=z_{\rm mix}-\lambda_k z_{\rm speech}-(1-\lambda_k)c$；
  - 主角：干扰子空间 $N_8$（d 的前 8 个右奇异向量；残差 S2 另算一次）与说话人子空间 $S_8$（train 说话人干净 $u^\star$ 的前 8 个主成分）之间的主角余弦；重叠度 $\|P_{S_8}N_8\|_F^2/8$（随机子空间期望 8/192 = 0.042）；d 的能量落在 $S_8$、$S_{16}$ 内的比例。

### 0.3 检查 2：混合结构

- **主设定（判定用）：** D1 的原始 d（未去收缩），中心化 PCA 降到 16 维；`sklearn.mixture.GaussianMixture`，`covariance_type="full"`，K = 1..10，`n_init=3`、`max_iter=500`、`reg_covar=1e-6`、`random_state=0`；按 BIC 选 K；BIC 选中模型的 argmax 簇对 `cat`（4 类）算 ARI、NMI。
- **通过线：BIC 选的 K ≥ 2 且 ARI(簇, cat) ≥ 0.3。**
- 同时报告（只作信息）：diag 协方差族；簇对 BVR 分箱、对语言的 ARI / NMI；SNR 在簇间的 η²（相关比）；‖d‖ 与 SNR 的 Spearman 相关；同样流程用在残差 S2 和 D2 上（D2 标签 = 活跃背景层，SNR = 窗口真实 SNR）。

### 0.4 检查 3：门控的上界

- 数据 D3：dev 80 条流；L = 2 s；与 G2.2 完全相同的因果后验（`idea2_shrink.run_stream` 的 full 模式：槽位 = `slot_for_speaker`，用该槽位 `pi > 0` 的窗口，可用时刻 = 窗口结束 + 1.04 s；后验均值 + $b_M$ 与 $u^\star$ 比较）。
- 单元：（流，说话人）中累计语音 ≥ 60 s 的单元；评估点 = 第 60 s 语音时；误差 $e=\|\hat u-u^\star\|_2$。
- 方法：
  - (a) Idea 2 软收缩后验：全部可用窗口；
  - **(b) oracle 硬门控（主）：** 同一后验，只用可用窗口中**真实窗口 SNR**（窗口 [end−2 s, end] 内 speech.wav 与 bg.wav 的能量比）最高的一半（⌈n/2⌉ 个）；
  - (b′) oracle 上限（信息）：按单窗口码误差 $\|(z_k-(1-\lambda_k)c)/\lambda_k-(u^\star-b_M)\|$ 最小的一半；
  - (b″) 可实现门控（信息，非 oracle）：按拟合的 λ_k（由 ṽ 给出）最大的一半；
  - (c) 均匀平均：`pi` 加权的 $z_{\rm mix}$ 平均 + $b_M$（即 G2.2 的 runmean）。
- 统计量：相对误差下降 $R=1-\overline{e_b}/\overline{e_a}$（单元平均）；95% CI 用按流聚类的配对 bootstrap（2000 次，种子 20261119）；同时报告中位数版本。
- **通过线：主 oracle (b) 的 $R\ge 0.10$（点估计）。**
- 同时报告（信息）：流结束时（全部窗口，全部单元）和第 8 s、20 s 语音时的同样比较。

### 0.5 结果对 Idea 4 组件的含义（预先写定）

| 检查 | 通过 | 不通过 |
|---|---|---|
| 1 低秩 | 少量寄存器 / 低维干扰子空间能吸收大部分干扰，寄存器设计有依据；若与说话人子空间重叠小，可投影去除而不伤身份 | 干扰是高维的，少量寄存器不够，“干扰子空间”的设定不成立；若与说话人子空间重叠大，投影去除会伤身份 |
| 2 混合结构 | 干扰有按背景类型分开的模式，GMM 先验有依据 | 单个高斯（或收缩模型）先验就够，GMM 组件不值得做 |
| 3 门控上界 | 软后验之外，硬门控还有空间，单调证据门控值得做 | 软收缩后验已接近“挑最干净窗口”的上界，门控不值得做 |
