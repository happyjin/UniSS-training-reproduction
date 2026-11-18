# 两个 ICLR 方向的并行实施方案：零干扰外推（ZNE）与共享背景的流式身份后验（CA-Kalman）

2026-11-18 晚，布拉格时间。
- 来源：`reports/mist_w3po_v1/game_muffled_speech_solutions.zh-CN.md` §7.1、§7.2。
- 现象和因果证据：`reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md`。

本文给出两个 idea 的**严格数学模型**、可检验的定理和命题、实验设计、门槛、GPU 分配和时间表。目标是：如果有提升，能从理论上解释为什么提升、什么时候会失效。

---

## 0. 两个 idea 能否同时进行？——能

| | Idea 1：ZNE | Idea 2：CA-Kalman |
|---|---|---|
| 改哪里 | 只改推理时从提示得到 32 个音色 token 的那一步；另有一个可选的 logit 变体 | 新增一个流式的身份估计模块，输出同样是 32 个 token |
| 需要训练吗 | 不需要（logit 变体的 CCA 蒸馏可选） | 线性版用 EM 拟合；学习版要训练小的观测编码器（约 1 千万到 3 千万参数） |
| GPU | 卡 0–3（slot A）。主要是 Talker 推理、BiCodec 编解码、ASR 评测 | 卡 4–7（slot B）。主要是数据处理（分离器、HDemucs、编码）和小模型训练，然后是 Talker 推理评测 |
| 相互依赖 | 无。共享的组件（BiCodec 编码器、Talker 评测链、`game_muffle_analysis.py`）都只读 | 同左 |
| 冲突点 | CPU（SQUIM、HDemucs）。192 核足够，各限 48 个进程 | 同左 |

**并行规范（沿用项目惯例）：**
- 两条线的输出目录、ledger owner、端口段、日志各自独立；
- 每张卡的占位程序由各自的启动器按 PID 让出，guard 不停；
- 每条线有自己的预登记文件和 `LOG.zh-CN.md` 条目；
- 验证阶段**每个方法只跑一个预先固定的设置**，不扫超参（用户规定）。

**最后合并：** 两条线各自通过门槛后，再做一组组合消融（§4.3）。ZNE 可以作为 Kalman 每个观测的前处理，两者在理论上可以叠加。

---

## 1. 共同的形式化框架

### 1.1 BiCodec 全局编码的精确形式

以下按 `uniss/speech_tokenizer/bicodec` 的实现写。

**输入：** 提示波形 $y\in\mathbb{R}^N$（16 kHz），只取前 6 s，不足则平铺：$\mathrm{clip}_6(y)$。然后做音量归一 $\mathcal{V}$：按 90–99 分位的幅度缩放到 0.2。

**编码映射：**
$$
X=\mathrm{Perceiver}\big(\mathrm{ECAPA}(\mathrm{logmel}(\mathcal{V}(\mathrm{clip}_6(y))))\big)\in\mathbb{R}^{32\times128},\qquad
u_i = W_{\rm in}X_i+b_{\rm in}\in\mathbb{R}^{6},\ i=1..32
$$

记 $u=\phi(y)\in\mathbb{R}^{D}$，$D=32\times6=192$。这是**量化前的连续编码**，两个 idea 都在这个空间里操作。

**有界化与量化**（FSQ，每维 $L=4$ 档）：
$$
\tilde u=\mathcal{B}(u)=\tanh(u+\sigma)\,h-o,\quad h=\tfrac{(L-1)(1+\epsilon)}{2},\ o=\tfrac12;\qquad
q=\mathcal{Q}(u)=\mathrm{round}(\tilde u)/2\in\{-1,-\tfrac12,0,\tfrac12\}^{D}
$$

每个 token $g_i\in\{0..4095\}$ 是 $q_i$ 的 4 进制编号。

**解码器条件是 q 的仿射函数：**
$$
d=P\,\mathrm{vec}(W_{\rm out}q+b_{\rm out})+c=:A_d\,\mathrm{vec}(q)+c_d\in\mathbb{R}^{512}
$$
解码器 $\mathrm{wav}=G(z_{\rm sem},d)$ 通过 prenet（FiLM 式）加上加性项 $+d$ 使用 $d$。

**Talker 的条件：** 32 个 token 的嵌入 $e(g_1),\dots,e(g_{32})$，作为前缀进入自回归 LM。第 $t$ 步的 logits 记为 $\ell_t(g)$。

### 1.2 观测模型

- 提示：$y=s+b$。$s$ 是目标说话人的语音，$b$ 是背景（BGM、观众、音效、其他解说）。
- 分离器（HDemucs）给出背景估计 $\hat b$ 和人声估计 $\hat v$。一般地：
$$
\hat b=c\,b+\kappa\,s+e,\qquad \hat v=y-\hat b
$$
  其中 $c$ 是背景增益误差，$\kappa$ 是目标语音漏进背景的比例，$e$ 是其余误差。
- **理想条件：** $u^\star=\phi(s)$，也就是同一说话人干净提示的编码，对应 $q^\star=\mathcal Q(u^\star)$、$g^\star$。

### 1.3 引理 0：条件误差决定生成质量（两条路径的统一界）

**假设：**
- 解码器 $G$ 对 $d$ 局部 $L_G$-Lipschitz；
- Talker 的 logits 对前缀嵌入局部 Lipschitz，且单个 token 的嵌入差有界：$\|\ell_t(g)-\ell_t(g')\|_\infty\le L_E\sum_i\mathbb 1[g_i\ne g'_i]$。

**结论：**
$$
\underbrace{\|G(z,d(q))-G(z,d(q^\star))\|}_{\text{解码器那一半}}\le L_G\|A_d\|\,\|q-q^\star\|_F,
\qquad
\underbrace{\mathrm{KL}\big(p_t(\cdot|g^\star)\,\|\,p_t(\cdot|g)\big)}_{\text{LM 那一半，逐步}}\le 2L_E\,\mathrm{Ham}(g,g^\star)
$$

第二个不等式用到：$|\log\mathrm{softmax}(a)_x-\log\mathrm{softmax}(b)_x|\le2\|a-b\|_\infty$。

**解释：**
- 两条路径都由条件误差控制：解码器那一半看欧氏距离 $\|q-q^\star\|$，LM 那一半看 token 的汉明距离。
- 2×2 拆分实验（global token 效应和语义码效应约各占一半）正是这两项的经验对应。
- 所以**任何能降低 $\|\hat u-u^\star\|$ 和 $\mathrm{Ham}(\hat g,g^\star)$ 的方法，都同时修两条路径**。两个 idea 的理论目标都写成"降低条件误差"，再用引理 0 解释为什么生成质量会提升。
- 常数 $L_G\|A_d\|$ 和 $L_E$ 用已有的配对数据（80 条换提示流、2×2 的四个格子）经验拟合（E0.2），用于事前预测收益。

### 1.4 评价与硬门槛（两条线相同，事先登记）

**主指标：** 生成语音的 SQUIM-PESQ。

**次指标：**
- DNSMOS OVRL；
- 频谱对比度和频谱平坦度（"闷"的声学特征）；
- 条件误差 $\|\hat u-u^\star\|$、token 汉明距离（只在有干净参考的合成集上算）；
- 说话人相似度：合成集上对干净参考；真实集上对 HDemucs 人声，仅作参考。

**硬门槛：**
- **ASR-BLEU（两个 TTS 种子）不得低于基线**（用户规定：流畅度或音质改动不得降低 ASR-BLEU；选择时 BLEU 优先于 LAAL）；
- 语音 LAAL（分离器块协议）不得变差。

**统计：** 按语音流配对；用 Wilcoxon 和 bootstrap 95% 置信区间；主要比较做 Holm 校正。

---

## 2. Idea 1：零干扰外推（Zero-Nuisance Extrapolation，ZNE）

### 2.1 定义

取放大倍数 $1=\lambda_1<\lambda_2<\dots<\lambda_m$，构造放大序列：
$$
y_\lambda=y+(\lambda-1)\hat b
$$

记 $f(\lambda):=\phi(y_\lambda)\in\mathbb R^D$。用 Lagrange 外推到零干扰：
$$
\hat u_{\rm ZNE}=\sum_{i=1}^m\gamma_i f(\lambda_i),\qquad \gamma_i=\prod_{j\ne i}\frac{\lambda_j}{\lambda_j-\lambda_i},\qquad
\sum_i\gamma_i\lambda_i^k=\delta_{k0}\ (k<m)
$$

输出 $\hat g=\mathcal Q(\hat u_{\rm ZNE})$，**同时**送给解码器和 Talker。

**常用节点：**

| 节点 | 权重 γ | ‖γ‖₂² |
|---|---|---|
| $\{1,2\}$ | $(2,-1)$ | 5 |
| $\{1,1.5,2\}$ | $(6,-8,3)$ | 109 |
| $\{1,2,3\}$ | $(3,-3,1)$ | 19 |

**预登记：** 主设置为 $m=2$、$\Lambda=\{1,2\}$。其他节点只在机制诊断（E1.0）中测量曲率，不作为可选设置。

> **重要修正（见 §2.12 命题 9）：** 外推变量应该是**功率比** $t=\lambda^2$，而不是幅度倍数 $\lambda$。也就是 $y_t=y+(\sqrt t-1)\hat b$，节点 $t\in\{1,2\}$（幅度 $\{1,\sqrt2\}$），估计量 $\hat u=2f(t{=}1)-f(t{=}2)$。
>
> 原因：独立相加时，mel 功率统计量对幅度 $\lambda$ 是偶函数，主导项是 $\lambda^2$。在 $\lambda$ 上做一阶外推会把误差放大 2 倍，而不是消去。
>
> 本节的定理 1 到定理 4 在把 $\lambda$ 换成 $t$ 后全部照样成立。外推族（$t$ 上的 Richardson、$\lambda$ 上的二次外推、带锚点的 Möbius 外推）按 §2.12 的预登记规则，**在 E1.0 的合成 dev 数据上选定一个**，之后不再更改。

### 2.2 假设

- **(A1) 光滑性。** 沿路径 $\lambda\mapsto\phi(s+\mu b)$，在 $\mu\in[0,\mu_{\max}]$ 上 $C^m$，且 $\|\partial_\mu^m\phi\|_\infty\le M_m$。这在每个分量上逐点成立。tanh 有界化发生在 $\phi$ **之后**，不影响这里的光滑性。
- **(A2) 加性。** 混合是加性的：$y=s+b$。广播里经过压缩的背景只近似满足，所以加性残差并入 $e$。
- **(A3) 近似尺度不变。** $\phi(\alpha x)\approx\phi(x)$。依据是音量归一 $\mathcal V$ 按幅度分位数缩放，只在 0.01 和 0.1 两个阈值处破坏尺度不变性。E1.0 会实测 $\|\phi(\alpha y)-\phi(y)\|$。

### 2.3 定理 1（偏差界：外推消去一阶干扰）

**条件：** $\hat b=b$，即 $c=1$、$\kappa=0$、$e=0$。此时 $f(\lambda)=\phi(s+\lambda b)$，$f(0)=u^\star$。

**结论：** 由 Lagrange 插值余项，对每个分量 $r$ 存在 $\xi_r\in(0,\lambda_m)$：
$$
u^\star_r-\hat u_{{\rm ZNE},r}=\frac{f_r^{(m)}(\xi_r)}{m!}\prod_{i=1}^m(0-\lambda_i)
\ \Rightarrow\
\|\hat u_{\rm ZNE}-u^\star\|_\infty\le\frac{M_m}{m!}\prod_i\lambda_i
$$

对比原始提示的误差：$f(1)-f(0)=f'(0)+\tfrac12f''(\xi)$，是 $O(\|f'\|)$ 的一阶量。

**推论 1（可预测的收益条件）：** 当 $m=2$、$\Lambda=\{1,2\}$ 时，ZNE 的误差不超过 $M_2$，而原始误差约为 $\|f'(0)\|$。所以
$$
\text{ZNE 优于原始提示}\iff M_2\ \lesssim\ \|f'(0)\|\quad(\text{曲率小于斜率})
$$

这个条件**在生成之前就能检验**：在合成对上测 $f(\lambda)$ 在 $\lambda\in\{0,0.5,1,1.5,2,3\}$ 处的值，拟合二次函数，比较曲率和斜率。E1.0 的 go/no-go 就是这一条。

### 2.4 命题 2（对分离误差的鲁棒性）

**(a) 增益误差。** 设 $\hat b=c\,b$。则 $y_\lambda=s+\mu(\lambda)b$，$\mu(\lambda)=1+(\lambda-1)c$。
- 外推到 $\lambda=0$ 时，剩余的干扰水平是 $\mu(0)=1-c$。所以一阶干扰效应被去掉的比例是 $c$。例如 $c=0.8$ 时去掉 80%。
- 如果要完全去掉，应外推到 $\lambda^\star=1-1/c$。$c$ 可以用合成对上训练的 SNR 探针 $\varsigma(u)$ 估计。这是 E1.4 的变体，不进入主设置。

**(b) 目标语音漏进背景（自泄漏）。** 设 $\hat b=c\,b+\kappa s$。在 (A3) 下：
$$
\phi(y_\lambda)=\phi\big((1+(\lambda-1)\kappa)s+(1+(\lambda-1)c)b\big)=\phi\big(s+\rho(\lambda)b\big),\qquad \rho(\lambda)=\frac{1+(\lambda-1)c}{1+(\lambda-1)\kappa}
$$
零干扰点 $\rho=0\iff\lambda^\star=1-1/c$，**与 $\kappa$ 无关**。$\kappa$ 只让路径参数化变成非线性，从而增加二阶曲率，可以在 E1.0 中测量。

**(c) 其余误差 e。** 一阶近似下 $f(\lambda)\approx\phi(s+\lambda b)+(\lambda-1)J(\lambda)e$。若 $J$ 近似常数，由于 $\sum_i\gamma_i(\lambda_i-1)=-1$，有
$$
\hat u_{\rm ZNE}-u^\star\approx R_m-J e
$$

**含义：** ZNE 的误差只线性依赖于"背景估计"的误差 $e$。**目标语音本身从不经过分离器。** 这正是它相对"HDemucs 人声提示"方案的结构性优势：后者的误差 $\phi(s+e_v)-\phi(s)$ 包含了分离器对人声的失真，也就是闷化。

### 2.5 命题 3（干净输入无害）

若 $b=0$ 且 $\hat b=e$（只有分离噪声），则 $\hat u_{\rm ZNE}=\phi(y)-Je+O(\|e\|^2)$。若分离器在干净输入上输出 $\hat b\approx0$，则 $\hat u_{\rm ZNE}=\phi(y)$，因为 $\sum\gamma_i=1$。

**含义：** 在安静的会议数据（AMI、AliMeeting）上，ZNE 应当**几乎不改变**结果，这就是"无害"检验（E1.5）。

### 2.6 定理 4（偏差–方差与外推阶数）

若每次编码有独立误差 $f(\lambda_i)+\varepsilon_i$，$\mathrm{Cov}(\varepsilon_i)=\sigma^2I$（来源可以是背景片段的选择或分离器的随机性），对 K 段取平均后：
$$
\mathrm{MSE}(\hat u_{\rm ZNE})=\Big\|\sum_i\gamma_iR(\lambda_i)\Big\|^2+\frac{\sigma^2D}{K}\|\gamma\|_2^2
$$

**含义：**
- 高阶外推能降低偏差，但会把噪声放大 $\|\gamma\|_2^2$ 倍（5、19、109 倍）；
- 最优阶数是两者的权衡；
- 主设置用对齐的 $\hat b$（就是该提示自己的背景），此时 $\sigma\approx0$，所以 $m=2$ 是稳妥的选择。

### 2.7 命题 5（量化下的 token 恢复保证）

**格点边界余量：** 记 $\delta_r=\min_k|\mathcal B(u^\star)_r-(k+\tfrac12)|$，即理想编码到最近取整边界的距离。由于 $\mathrm{Lip}(\mathcal B)\le h\approx1.5$：
$$
\|\hat u-u^\star\|_\infty<\min_r\delta_r/h\ \Rightarrow\ \mathcal Q(\hat u)=\mathcal Q(u^\star)\ (\text{完全恢复 }g^\star)
$$

更一般地，期望汉明距离 $\mathbb E\,\mathrm{Ham}(\hat g,g^\star)\le\sum_r\Pr\big(|\hat u_r-u^\star_r|\ge\delta_r/h\big)$。

**含义：** 定理 1 的偏差界可以经由余量分布，换算成 token 恢复率，再由引理 0 换算成两条路径的质量。E1.1 会实测余量分布和 token 恢复率。

### 2.8 logit 侧：对数线性外推等价于负向 CFG

**命题 6（等价性）：** 记 $L_t(x;\lambda)=\log p_t(x\mid g(\lambda))$。若它在 $[0,\lambda_2]$ 上对 $\lambda$ 是仿射的，那么对 $m=2$：
$$
\log\tilde p_t(x)=\sum_i\gamma_iL_t(x;\lambda_i)-\log Z
=(1+w)\log p_t(x|g(\lambda_1))-w\log p_t(x|g(\lambda_2))-\log Z,\qquad w=\frac{\lambda_1}{\lambda_2-\lambda_1}
$$
并且 $\tilde p_t=p_t(\cdot\mid g(0))$ 精确成立。

**含义：**
- 用"干扰放大的条件"作负分支的 CFG / 对比解码，就是一阶 Richardson 外推。guidance 权重有原理性取值：$\Lambda=\{1,2\}$ 时 $w=1$，不需要扫参。
- 若存在余项 $r_t$，则 $\mathrm{KL}(p_t(\cdot|g(0))\|\tilde p_t)\le 2\sup_x|\sum_i\gamma_ir_t(x;\lambda_i)|$。

**命题 7（身份保持：解释 Autoguidance 的"同类错误"要求）：**
- 设在 $u^\star$ 附近存在局部坐标 $u=\Psi(\iota,\nu)$，$\iota$ 是身份，$\nu$ 是干扰，且放大路径保持 $\iota$ 不变。这由构造保证：放大只加 $\hat b$，不改变 $s$。
- 那么外推只沿 $\nu$ 方向进行：$\hat u=\Psi(\iota,\hat\nu(0))+O(R_m)$，**身份在外推的阶数内被精确保持**。
- 普通 CFG 的负分支是空条件 $u_\varnothing$，方向 $u-u_\varnothing$ 在 $\iota$ 上有分量，身份漂移约为 $w\,\partial_\iota\Psi\,(\iota-\iota_\varnothing)$。
- E1.3 用"说话人相似度 vs guidance"对比 ZNE-logit 和 CFG-null，检验这一点。

**命题 8（离散条件的抖动平滑）：**
- 问题：由于取整，$\lambda\mapsto g(\lambda)$ 是分段常数，$L_t(x;\lambda)$ 是阶跃函数，命题 6 的仿射假设不成立。
- 做法：使用抖动条件 $\bar p_t(x\mid\lambda)=\mathbb E_\xi\big[p_t(x\mid\mathcal Q(\mathcal B(f(\lambda))+\xi))\big]$，$\xi\sim U(-\tfrac12,\tfrac12)^D$（以量化步长为单位）。
- 性质：$\Pr[\mathrm{round}(a+\xi)=k]$ 是 $a$ 的分段线性函数，斜率不超过 1。所以 $\bar p_t(x|\lambda)$ 对 $\lambda$ 是 Lipschitz 的，常数不超过 $h\,\|f'\|_1\max_x p$，从而命题 6 可以近似使用。
- 实现：每个分支采样 $K_d$ 个抖动 token 序列，每个都是合法 token，不会出分布。概率在 batch 内取平均。代价是 $K_d$ 倍。主设置用 $K_d=4$。

**可信集合掩码：** 只在 $\mathcal V_t=\{x:p_t(x|g(\lambda_1))\ge\alpha\max p_t\}$（$\alpha=0.1$）上做外推，集合外的 token 概率置 0。这保证外推后分布的支撑集包含在正分支的支撑集内，参考 Contrastive Decoding（ACL 2023）。

### 2.9 算法（每个说话人槽位更新提示时执行一次，不在逐 token 的关键路径上）

```
输入：提示 y（因果截取）、背景估计 b̂（HDemucs 对 y 的非人声部分，与 y 时间对齐）
1. 对 t ∈ {1,2}：u_t = φ(y + (√t − 1)·b̂)           # 功率比参数化（命题 9）；2 次编码器前向，约 10 ms 量级
2. û = 2·u_1 − u_2；ĝ = Q(û)                         # 隐空间 ZNE（主设置；若 E1.0 选中 Möbius 族，则换成命题 10 的公式）
3. 把 ĝ 交给解码器和 Talker                          # 两条路径一起修
可选（logit 变体，E1.3）：Talker 每步并行跑两支（ĝ_1, ĝ_2），每支 K_d 个抖动样本；
   ℓ̃ = ℓ(ĝ_1) + 1·(ℓ(ĝ_1) − ℓ(ĝ_2))，只作用在 V_t 上
```

**流式成本：**
- 隐空间 ZNE 只在提示更新时多做一次编码，HDemucs 处理 6 s 提示可以放在后台线程。它不增加 Talker 的逐 token 计算，**LAAL 不受影响**；
- logit 变体使 Talker 的计算量变为 2×K_d 倍，只用于机制实验，或之后用 CCA（ICLR 2025 oral）蒸馏进 LoRA。

### 2.10 实验（卡 0–3）

| 编号 | 内容 | 数据 | 产出与门槛 |
|---|---|---|---|
| E1.0 机制诊断（约 0.5 天） | 测 $f(\lambda)$，$\lambda\in\{0,0.25,0.5,\sqrt{0.5},1,\sqrt2,1.5,2,3\}$ 以及 $\lambda\to\infty$（只有背景，$\phi(\hat b)$）；拟合 $c_0+c_1\lambda+c_2\lambda^2$，比较 $|c_1|$ 与 $|c_2|$（命题 9）；按 §2.12 的预登记规则，在三个外推族里选一个；测尺度不变性 $\|\phi(\alpha y)-\phi(y)\|$；测 $c$、$\kappa$ 的分布（用合成对的真值） | **开源**合成对（§7）：干净语音（LibriTTS-R 英文、AISHELL-3 中文；AMI train IHM 作为真实会议语音的补充）+ 开源背景（MUSAN 音乐与噪声、FSD50K 中的观众/欢呼/鼓掌/爆炸/枪声等音效类、DnR 的音乐与音效轨），按电竞提示实测的 BVR 分布（−25 到 −3 dB）混合，每种语言 500 对 | **G1.0：** 在 ≥ 70% 的样本上 $M_2<\|f'(0)\|$；ẑ₀ 到 $u^\star$ 的距离中位数小于 $\phi(\hat v)$ 到 $u^\star$ 的距离。不通过：改为只做 logit 变体，或终止 |
| E1.1 只用解码器重合成（约 0.5 天） | 固定语义码（干净语音的编码），只换全局条件：raw / HDemucs 人声 / ZNE / oracle | 同上 | token 恢复率、余量分布（验证命题 5）、PESQ、频谱对比度、说话人相似度；拟合引理 0 的常数 |
| E1.2 全链路 dev（约 1 天） | Talker（T1c）在 80 条配对流 × 两个方向上运行：raw、HDemucs、ZNE；外加 AMI-N 那一组 | 已有的 `game_swap_*` 输入，只换 voice_audio 或 token | **G1.2：** ZNE 相对 raw 的 PESQ 提升 ≥ 0.3，且 bootstrap 置信区间下界 > 0；ASR-BLEU 不降。另报告与 HDemucs 的对比 |
| E1.3 logit 侧（约 1 天） | ZNE-logit（$w=1$，抖动 $K_d=4$，可信集合掩码）对比 CFG-null（$w=1$）；二者再分别叠加隐空间 ZNE | 80 条流的子集。需要 HF 版双分支 Talker 推理（vLLM 的自定义 logits 处理器只在 V1 runner 可用，先做 HF 版） | 验证命题 6、7：说话人相似度在 ZNE-logit 下不降、在 CFG-null 下下降 |
| E1.4 增益校正（可选） | 用 SNR 探针估计 $c$，外推到 $\lambda^\star=1-1/c$ | 合成集 | 验证命题 2(a) |
| E1.5 无害检验（测试） | AMI zh→en、AMI en→zh、AliMeeting 测试集各随机 300 条 | 只做测试 | PESQ 和 ASR-BLEU 不显著变差（命题 3） |
| E1.6 主测试（预登记后只跑一次） | 新选的未见电竞视频，每方向 5 段 × 5 分钟，用同样的选段规则 | 与训练集和 dev 视频不重叠 | 主结论 |

**卡的使用：**
- Talker 每个 si_infer 约 14 GB，每卡可以叠 6–8 个；
- ASR（Whisper / Paraformer）和 BiCodec 编解码叠在同一组卡上；
- 每卡不超过 135 GiB。

### 2.11 风险与诊断

| 风险 | 诊断 | 对策 |
|---|---|---|
| 编码器在强背景下饱和（曲率大），外推过冲 | E1.0 的曲率和斜率比 | 只在 BVR 较低的段使用，或改用更近的节点 {1, 1.5}（会放大方差），或走 logit 变体 |
| 分离器增益 $c$ 偏小，去除不彻底 | E1.0 的 $c$ 分布 | E1.4 的 $\lambda^\star$ 校正 |
| 尺度不变性不成立 | E1.0 的尺度测试 | 在编码前统一做响度归一 |
| logit 外推改变内容 | ASR-BLEU | 可信集合掩码；只用 $w=1$，不加大 |

### 2.12 与量子误差缓解（QEM）的严格对应：novelty 的理论支点

**查新（2026-11-18）：** 没有检索到把 zero-noise extrapolation 用于扩散模型或语言模型的引导、条件生成的工作。
- 最接近的是 RX-DPM（ICLR 2025）。它把 Richardson 外推用于扩散 ODE 采样器的**时间离散误差**，而不是条件里的干扰，问题不同。
- 量子方面：ZNE 的原始文献是 Temme et al.（arXiv 1612.02058）和 Giurgica-Tiron et al. 的数字 ZNE（arXiv 2005.10921）；误差界分析见 arXiv 2502.20673；最新的"经典锚点增强 ZNE"见 arXiv 2607.25746。

**对应关系：**

| 量子误差缓解 | 本文（条件生成中的零干扰外推） |
|---|---|
| 含噪电路，可观测量期望 $E(\varepsilon)$，$\varepsilon$ 为噪声强度 | 含背景的提示，条件编码 $f(t)=\phi(s+\sqrt t\,b)$；logit 版本是 $L_t(x;t)$ |
| 放大噪声：脉冲拉伸（模拟，连续倍数）；酉折叠 $G\to G(G^\dagger G)^k$（数字，$\lambda=1+2k$） | 放大干扰：$y_t=y+(\sqrt t-1)\hat b$，连续倍数。**比量子更强**：量子只能放大噪声，看不到噪声本身；我们能直接观测干扰的一个样本 $\hat b$ |
| 外推族：Richardson / 多项式、指数（去极化信道下期望值随噪声指数衰减）、自适应 | Richardson（$t$ 上）、二次（$\lambda$ 上）、Möbius（命题 10，能量加权池化下精确） |
| 偏差–方差：高阶外推降偏差，采样开销约为 $(\sum_i|\gamma_i|)^2$ | 定理 4：方差放大 $\|\gamma\|_2^2$；logit 抖动版的开销 $(\sum|\gamma_i|)^2K_d$ |
| 噪声参数化必须让期望对它解析、近似线性（门错误率随脉冲拉伸线性变化） | **命题 9：必须按功率比 $t$ 参数化，不能按幅度 $\lambda$** |
| 经典锚点增强 ZNE：用经典模拟得到零方差的锚点节点 | **命题 10：只有背景的帧给出 $t\to\infty$ 的精确锚点 $u_b=\phi(\hat b)$**，2 个节点加锚点即可精确外推 |
| 概率误差消除（PEC）：用准概率（带负权重）组合实现逆噪声信道 | logit 外推 $\tilde p\propto\prod_ip(\cdot|g(t_i))^{\gamma_i}$，含负指数，是条件分布的"准概率专家积"（命题 6）。这一类比也给出了"概率干扰消除"这条后续方向 |

**命题 9（噪声尺度的正确参数化）：**
- 设 $s$ 与 $b$ 独立，帧级特征是 mel 功率的光滑函数（log-mel 加上 ECAPA 卷积）。
- 对时频单元 $(f,\tau)$：
$$
|S+\lambda B|^2=|S|^2+\lambda^2|B|^2+2\lambda\,\mathrm{Re}(S\bar B)
$$
- 交叉项在频带和帧上求和后，按 $1/\sqrt{N_{\rm bins}}$ 的速度不相干地相消。所以编码可以展开为
$$
f(\lambda)=u^\star+a\,\lambda+b_2\lambda^2+O(\lambda^3),\qquad \|a\|=O\big(N_{\rm bins}^{-1/2}\big)\ll\|b_2\|\ (\text{典型情况})
$$
- 三种估计量的误差：
  - 原始提示：$a+b_2$；
  - 在 $\lambda$ 上取 $\{1,2\}$ 外推：$2f(1)-f(2)=u^\star-2b_2$，**$b_2$ 被放大 2 倍**；
  - 在 $t$ 上取 $\{1,2\}$ 外推，即幅度 $\{1,\sqrt2\}$：$2f(1)-f(\sqrt2)=u^\star+(2-\sqrt2)a+O(\lambda^3)$，**完全消去 $b_2$，只剩 $0.59a$**。
- 结论：在典型情况（$\|a\|\ll\|b_2\|$）下，必须在功率比 $t$ 上外推。E1.0 拟合 $c_1$ 和 $c_2$，直接检验这一点。如果 $|c_1|$ 不可忽略，就用 $\lambda$ 上的 3 节点二次外推，同时消去 $a$ 和 $b_2$，代价是方差放大（见下面的预登记规则）。

**命题 10（能量加权池化模型下，Möbius 外推是精确的）：**
- 一阶模型：如果池化后的编码近似是语音和背景编码的能量加权平均（ECAPA 的统计池化在语音和背景帧上的一阶近似），则
$$
f(t)=\frac{u_s+\rho t\,u_b}{1+\rho t},\qquad \rho=\frac{P_b}{P_s}
$$
  这是 $t$ 的 Möbius（分式线性）函数，$f(0)=u_s=u^\star$，$f(\infty)=u_b$。
- **Richardson 的残差：** $2f(1)-f(2)-u^\star=(u_b-u_s)\dfrac{2\rho^2}{(1+\rho)(1+2\rho)}=O(\rho^2)$。原始提示是 $O(\rho)$。
- **带锚点的精确外推：** 用只有背景的帧得到 $u_b=\phi(\hat b)$（对应经典锚点增强 ZNE 的锚点节点）。由
$$
\frac{f(1)-u_b}{f(2)-u_b}=\frac{1+2\rho}{1+\rho}
$$
  按最小二乘解出 $\hat\rho$，然后 $\hat u=u_b+(1+\hat\rho)\big(f(1)-u_b\big)$。在模型成立时**精确等于** $u^\star$，与 $\rho$ 的大小无关，对强背景也成立。
- **代价：** 依赖模型假设。真实编码器的注意力池化与 Perceiver 只是近似满足，所以需要在 E1.0 上检验拟合度。

**预登记的外推族选择规则（模仿 QEM 的做法，只在 E1.0 的合成 dev 数据上执行一次）：**
- 候选族：
  - (R-t) $t$ 上的 Richardson，节点 $t\in\{1,2\}$；
  - (Q-λ) $\lambda$ 上的 3 节点二次外推，节点 $\lambda\in\{1,\sqrt2,2\}$；
  - (M-anc) 带锚点的 Möbius 外推，节点 $t\in\{1,2\}$ 加 $u_b$。
- 准则：在合成对上，各族用自己的节点估计 $\hat u$，计算 $\|\hat u-u^\star\|$ 的中位数，取最小者。若差距小于 5%，取最简单的 R-t。
- 选定之后，E1.1 到 E1.6 只用这一族，不再更改（符合"每个方法只跑一个设置"）。
- 另外两族的结果只作为机制分析写进论文（它们对应 QEM 文献里不同外推族的取舍）。

**这对 novelty 和可解释性的意义：**
1. **方法上：** 把 QEM 的"放大噪声再外推"系统地迁移到生成模型的条件空间。它比量子情形多一个结构：干扰本身可以观测，因此有锚点，可以做精确的 Möbius 外推。
2. **理论上：**
   - 命题 6 和 7 把负向 CFG、对比解码、Autoguidance 统一解释为"沿干扰轴的一阶外推"，并给出 guidance 权重的原理性取值 $w=t_1/(t_2-t_1)$；
   - 命题 9 和 10 给出何时外推有效、按哪个变量外推、何时精确的判据；
   - 定理 1 到 4 给出偏差、方差和量化的误差界。
3. **可解释性：** 如果有提升，可以按"曲率与斜率之比 → 条件误差 → 引理 0 → 生成质量"这条链定量解释。如果没有提升，E1.0 的诊断能指出是哪一条假设不成立（$a$ 项、曲率、尺度不变性、分离增益），这同样是一个可以发表的分析结论。

---

## 3. Idea 2：共享背景的流式身份后验（CA-Kalman → Ambient Conditioning）

### 3.1 状态空间模型（线性高斯版）

**时间与窗口：**
- 时间按窗口 $k=1,2,\dots$ 推进：长 2 s，跳步 0.5 s，严格因果；
- 分离器给出每个窗口的主导说话人 $j(k)$ 和置信度 $\pi_k$（目标槽位后验 × 语音概率 × [非重叠]）；
- 没人说话的窗口记为 $j(k)=\varnothing$。

**状态：**
- 说话人 $j$ 的静态音色：$s_j\in\mathbb R^{D}$，先验 $s_j\sim\mathcal N(\mu_0,\Sigma_0)$，用干净语料的说话人中心估计；
- 共享背景因子：$n_k=\bar n+\tilde n_k\in\mathbb R^{r}$，其中 $\bar n$ 是本场的持续背景（静态、未知，先验 $\mathcal N(0,\Sigma_{\bar n})$），$\tilde n_k=\rho\,\tilde n_{k-1}+\eta_k$，$\eta_k\sim\mathcal N(0,Q)$，表示随时间变化的部分。

**观测 1（有人说话的窗口）：**
$$
z_k=\phi(y_k)=s_{j(k)}+A\,n_k+\varepsilon_k,\qquad\varepsilon_k\sim\mathcal N\big(0,\,R/\pi_k\big)
$$

**观测 2（只含干扰，每个窗口都有）：**
$$
v_k=h(\hat b_k,\hat v_k)=B\,n_k+\zeta_k,\qquad \zeta_k\sim\mathcal N(0,S)
$$
- 这里 $h$ 是**相对背景功率谱**：$h=\big(P_{\rm mel}(\hat b_k)/\bar P(\hat v_k)\big)^{0.3}$，即背景 mel 功率除以人声平均功率，再做 0.3 次幂压缩；
- 选择它的原因：它天然满足**锚点条件** $h=0\iff$ 没有背景，所以 $n=0$ 对应干净；
- 不用 BiCodec 编码器作 $h$ 的原因：音量归一会把近静音放大，破坏锚点。

**联合状态与观测矩阵：** $x_k=(s_1,\dots,s_J,\bar n,\tilde n_k)$，
$$
H_k=\big[e_{j(k)}^\top\!\otimes I_D,\ A,\ A\big],\qquad H^v=[0,\ B,\ B]
$$
动力学是块对角的：$s$ 和 $\bar n$ 恒等，$\tilde n$ 按 $\rho$ 衰减。这是**标准的线性高斯状态空间模型**，卡尔曼滤波给出精确后验 $p(s_j\mid z_{1:k},v_{1:k})=\mathcal N(m_{j|k},P_{j|k})$。

**为什么用 $R/\pi_k$（加权似然）：** 等价于把每个观测的似然取 $\pi_k$ 次幂（generalized / power posterior），也就是对观测做加权最小二乘。分离器越不确定，观测的权重越低。这给"分离器后验应该怎么用"一个原理性的答案。

### 3.2 定理 2.1（持续背景下，只做平均不可辨识）

**条件：** 没有观测 2，只有 $\{z_k\}$。

**结论：** 对任意 $J$，$\{z_k\}$ 的联合分布只通过 $s_j+A\bar n$ 依赖于 $(s_{1:J},\bar n)$。因为 $\tilde n_k$ 零均值，且与 $\bar n$ 独立。所以：
1. **只用 z 的任何估计量，都不能一致地估计 $s_j$**；
2. 样本均值（也就是多参考平均：Mega-TTS 2、PhotoMaker、滑动平均）收敛到 $s_j+A\bar n$，偏差 $A\bar n$ **不随观测增多而消失**；
3. 跨说话人只能辨识差值 $s_j-s_{j'}$；
4. 用先验 $\mu_0$ 做收缩，只是一个贝叶斯折中，后验均值仍然有偏。

**含义：** 这从理论上解释了为什么"多拿几段提示平均"解决不了电竞的问题：BGM 和观众声是**持续存在**的。这一点也作为 E2.0 的经验检验：画"平均误差 vs 观测数"，应该在 $\|A\bar n\|$ 处出现平台。

### 3.3 定理 2.2（有干扰观测流时可辨识，并给出误差率）

**条件：** 加入观测 2，$B$ 列满秩，锚点 $v=0\iff n=0$。

**可辨识性：**
- $\bar n$ 可辨识，至多差一个 $r\times r$ 的可逆变换 $T$：$T\bar n=B_T^{+}\mathbb E[v]$；
- $A$ 由交叉协方差 $\mathrm{Cov}(z_k,v_k)=A\,\mathrm{Cov}(n_k)B^\top$ 辨识，同样至多差 $T$；
- $s_j=\mathbb E[z\mid j]-A\bar n=\mathbb E[z|j]-(AT^{-1})(T\bar n)$ 与 $T$ 无关，**所以 $s_j$ 可以唯一辨识**。

**误差率：** 在平稳的近似下，
$$
\mathbb E\|m_{j|k}-s_j\|^2\ \approx\ \mathrm{tr}\Big[\big(\Sigma_0^{-1}+K_j^{\rm eff}\,\tilde R^{-1}\big)^{-1}\Big]\ +\ \mathrm{tr}\big[A\,\mathrm{Var}(\hat{\bar n}_k)A^\top\big],
\qquad K_j^{\rm eff}=\sum_{k:j(k)=j}\pi_k,\ \ \mathrm{Var}(\hat{\bar n}_k)=O(1/k)
$$
其中 $\tilde R=R+A\,\mathrm{Var}(\tilde n)A^\top$。

**两个可检验的推论：**
1. 误差随说话人自己的有效观测量按约 $1/K_j^{\rm eff}$ 下降，**没有平台**；对比定理 2.1 的平台。
2. **跨说话人共享：** $\hat{\bar n}$ 用所有窗口估计，包括其他解说的窗口和没人说话的窗口，所以每个说话人的偏差项按 $1/k$（总窗口数）下降，比各自独立估计更快。这对首句尤其重要：新说话人一开口，就已经有对背景的估计。

### 3.4 参数学习：只用真实带噪数据（不需要干净参考）

**线性版：** 参数 $\theta=(A,B,\rho,Q,R,S,\mu_0,\Sigma_0,\Sigma_{\bar n})$。
- 在**开源**数据上用 EM（§7）：开源合成多人流，加上 CN-Celeb 这类真实带噪的开源语料。E 步用 RTS 平滑，M 步有闭式解。电竞数据不参与任何参数拟合；
- 维度 $D=192$；$r$ 用 E2.0 中 $\{\phi(s+b)-\phi(s)\}$ 的 PCA 来定，**事先固定**为解释 90% 方差所需的维数；
- 合成对**只用来校验**，不用来拟合。

**学习版（ICLR 的主体）：**
- 观测编码器 $f_\psi$：窗口音频 → $(z_k,R_k)$，带不确定度的观测，从 BiCodec 编码器初始化，只训练一个 LoRA 和一个方差头；
- 干扰编码器 $h_\omega$：$(\hat b_k,\hat v_k)\to(v_k,S_k)$，满足锚点约束 $h_\omega(0,\cdot)=0$；
- 结构仍然是上面的线性高斯状态空间模型，所以**边缘似然可以由卡尔曼新息精确算出**：
$$
\mathcal L_{\rm ML}(\psi,\omega,\theta)=-\sum_k\log\mathcal N\big(\mathrm{innov}_k;\,0,\,S^{\rm inn}_k\big)
$$
- 总目标：
$$
\mathcal L=\mathcal L_{\rm ML}\ +\ \lambda_{\rm anc}\underbrace{\mathbb E_{\rm clean}\|f_\psi(x)-\phi(x)\|^2}_{\text{干净锚点：在干净输入上不偏离原编码空间}}\ +\ \lambda_{\rm syn}\underbrace{\mathbb E_{\rm syn}\big[(m_{j|K}-u_j^\star)^\top P_{j|K}^{-1}(m_{j|K}-u_j^\star)+\log\det P_{j|K}\big]}_{\text{合成对上的高斯 NLL：校准不确定度}}
$$
- 第一项是**无监督**的，在开源的真实带噪数据（CN-Celeb 的直播、娱乐、vlog 等类别）和开源合成多人流上计算，不使用电竞数据；
- 干净锚点防止编码器塌缩去"迎合"似然，并保证输出仍落在冻结生成器的条件空间里；
- 第三项校准后验的不确定度。
- 预登记：$\lambda_{\rm anc}=1$，$\lambda_{\rm syn}=0.1$，不扫参。

**可选的第三阶段（并入 §7.3 的面向生成器的目标）：** 用 KL$\big(p_{\rm Talker}(\cdot|g^\star)\|p_{\rm Talker}(\cdot|\mathcal Q(m))\big)$ 在合成对上微调 $f_\psi$。只有当 E2.4 发现"条件误差下降了，但生成质量没有按引理 0 提升"时才做。

### 3.5 输出与切换策略（流式）

- 每个说话人在句子边界取 $g_j=\mathcal Q(m_{j|k})$；
- 只有同时满足以下两条才切换：
  - (i) 自上次切换以来，$\mathrm{tr}(P_{j|k})$ 相对下降超过 $\tau=20\%$；
  - (ii) $\mathrm{Ham}(g_j^{\rm new},g_j^{\rm old})\ge4$（滞回，避免音色抖动）；
- 首句：新说话人在前 1 s 内，用先验加上共享的背景估计，给出 $m_{j|k}=\mu_0+$（已有观测的后验更新）。

### 3.6 实验（卡 4–7）

| 编号 | 内容 | 数据 | 产出与门槛 |
|---|---|---|---|
| D2 数据（约 1 天，GPU） | **全部用开源数据**（§7）：下载 LibriTTS-R、AISHELL-3、MUSAN、FSD50K（选类）、CN-Celeb v2；按 AMI 标注的轮次和重叠统计，合成约 100 h 的多人流（2–4 人，随时间变化的广播式背景轨）；对 CN-Celeb 的真实带噪片段和合成流跑 Nemotron 低延迟分离器；HDemucs 在 GPU 上分出背景和人声；用 BiCodec 编码器对窗口编码；计算 $h(\hat b,\hat v)$ | 开源语料。电竞数据只用于 dev 和测试评测 | 特征缓存。4 卡并行：分离器约 1.5 h，HDemucs 约 2 h，下载时间取决于带宽 |
| E2.0 假设检验（约 0.5 天） | (i) 持续偏差 $\|A\bar n\|$ 相对说话人间距的大小；(ii) 干扰是否低秩，即 r；(iii) $v$ 对 $z$ 偏移的预测力 $R^2$；(iv) 高斯性 | 合成对和真实流 | **G2.0：** $\|A\bar n\|$ 不小于说话人间距的 20%（否则平均就够了，idea 2 的必要性不成立）；$R^2(v\to z\text{ 偏移})\ge0.3$ |
| E2.1 合成多人流（约 0.5 天） | 2–4 个干净说话人轮流说话，按真实分离器的统计决定轮次与重叠，叠加**随时间变化的真实电竞背景轨道**（来自训练视频的连续 HDemucs 背景），有干净参考 $u_j^\star$ | 开源干净语料（LibriTTS-R 和 AISHELL-3 的 test 说话人，与训练说话人不重叠）加开源背景的留出部分 | 评测集 |
| E2.2 线性 CA-Kalman（约 1 天） | 用 EM 在开源数据（合成流的训练部分和 CN-Celeb）上拟合，在开源合成评测流上评测；画"误差、token 恢复率 vs 观测秒数"（1–60 s） | E2.1 | **G2.2：** 60 s 时误差低于"滑动平均"且低于"首 8 s"；"无 $v$ 流"版本出现平台（验证定理 2.1），完整版没有平台（验证定理 2.2）|
| E2.3 学习版（约 1.5 天，4 卡 DDP） | 按 §3.4 的目标训练 $f_\psi$ 和 $h_\omega$ | 开源的真实带噪数据 CN-Celeb（无监督）加开源合成对（校准） | 同 E2.2 的曲线，以及 NLL 校准度（覆盖率） |
| E2.4 全链路 dev（约 1 天） | Talker 用 $g_j$ 生成：dev 电竞片段（新选的 dev 视频，用 demo 的选段规则），以及 80 条配对流 | dev | **G2.4：** PESQ 相对"首 8 s"提升 ≥ 0.3，置信区间下界 > 0；ASR-BLEU 不降；LAAL 不变差。首句和后续句分开报告 |
| E2.5 无害与测试 | 同 E1.5 和 E1.6 | 测试 | 主结论 |

**基线（都在同一管线里跑）：**
1. 首 8 s（现状）；
2. 滑动平均（Mega-TTS 2 式）；
3. 中位数聚合（RobustRAG 式）；
4. 只做先验收缩；
5. 卡尔曼但不用 $v$ 流；
6. 完整线性版；
7. 学习版。

另外报告"HDemucs 人声加滑动平均"作为外部参照。

### 3.7 风险

| 风险 | 对策 |
|---|---|
| 2 s 窗口的编码噪声大（编码器为 6 s 设计，短窗会被平铺） | E2.0 比较 2 s 和 4 s 窗口的噪声，事先只选一个 |
| 非线性纠缠（强背景） | 学习版用异方差观测；可以和 ZNE 组合（§4.3），先把每个观测的大部分干扰去掉 |
| 外场采访这类场景，不同说话人的背景不同 | 模型选择的消融：允许每个说话人有自己的 $\bar n_j$ |
| 学习版用似然"作弊"（编码器塌缩） | 干净锚点项；监控在干净输入上的编码偏移 |

---

## 4. 统一的理论解释与组合

### 4.1 两个 idea 在同一框架下

两个 idea 都在**同一个条件空间** $u\in\mathbb R^{192}$ 里降低 $\|\hat u-u^\star\|$，然后由引理 0 同时改善两条路径。

| | ZNE | CA-Kalman |
|---|---|---|
| 利用的结构 | 单个观测内，**干扰可以精确放大**（加性、尺度不变） | 多个观测之间，**身份静态、背景共享且可以单独观测** |
| 消掉的误差 | 一阶干扰偏差（定理 1） | 持续偏差 $A\bar n$（定理 2.1、2.2）和随机误差（$1/K$） |
| 剩下的误差 | 曲率余项 $M_2$，加上背景估计误差 $Je$（命题 2） | 模型误差，以及有限观测带来的方差 |

### 4.2 可预测的收益

用 E0.2 拟合的引理 0 常数，加上 E1.1 和 E2.2 测得的条件误差下降，可以**事先预测** PESQ 的提升，再与 E1.2 和 E2.4 的实测对照。如果预测和实测一致，论文就有"理论解释了提升"的闭环。

### 4.3 组合（两条线都过门槛之后）

- 把 ZNE 当作每个观测的前处理：$z_k\leftarrow\hat u_{\rm ZNE}(y_k)$。由命题 2，等效的 $A$ 变成 $(1-c)A$，卡尔曼再处理剩下的部分。
- 消融：{无、ZNE、Kalman、ZNE + Kalman} × 两个方向，在测试集上跑一次。

---

## 5. 通用前置任务 E0（两条线共享，第 0 天，CPU 加少量 GPU）

- **E0.1 数据划分（预登记文件）：**
  - **训练、拟合和所有选择：只用开源数据**（§7）。
  - 电竞数据（S3 上 kenxsong 的内部数据）只用于评测，不参与任何训练、参数拟合或外推族选择：
    - dev 视频：新选，每方向 5 段，只用于 go/no-go 判断；
    - 测试视频：新选，每方向 5 段，选段规则与 demo 相同。demo 的 10 段已经分析过，不再作为测试；
  - AMI、AliMeeting 只做测试。AliMeeting、RealSI、ACL 绝不用于训练或调参。
- **E0.2 拟合引理 0 的常数：** 用已有的 80 条配对流和 2×2 的四个格子，回归 PESQ 对 $\|q-q^\star\|$ 和 Ham 的关系。
- **E0.3 评测脚本统一：** 复用 `game_muffle_analysis.py`、`game_voice_texture.py` 和 ASR-BLEU 链。

---

## 6. 时间表（两条线并行，约 6 个工作日）

| 天 | 卡 0–3（Idea 1：ZNE） | 卡 4–7（Idea 2：CA-Kalman） |
|---|---|---|
| D0 | E0（共享）；E1.0 机制诊断 → **G1.0** | E0；D2 数据下载与预处理（分离器、HDemucs、编码） |
| D1 | E1.1 只用解码器重合成；开始 E1.2 | D2 收尾；E2.0 假设检验 → **G2.0**；E2.1 合成流 |
| D2 | E1.2 全链路 dev → **G1.2** | E2.2 线性 CA-Kalman → **G2.2** |
| D3 | E1.3 logit 侧（HF 双分支）；E1.4（可选） | E2.3 学习版训练（DDP） |
| D4 | E1.5 无害检验；**预登记** E1.6 | E2.3 收尾；E2.4 全链路 dev → **G2.4** |
| D5 | E1.6 测试（只跑一次）；写结果 | E2.5 测试（只跑一次）；写结果 |
| D6 | §4.3 组合消融（8 卡）；汇总报告 | 同左 |

**卡的空闲：** 一条线在等 CPU 阶段时，卡由 guard 的占位程序补上（不存 checkpoint 的训练占位），GPU 阶段开始前按 PID 让出。不跨 slot 借卡，除非另一条线明确空闲，并在 slot 文件里登记。

---

## 7. 数据方案：训练只用开源数据，电竞数据只做评测

### 7.1 每个 idea 需要什么数据

**Idea 1（ZNE）不需要训练。** 推理时只用到三样东西：冻结的 BiCodec 编码器、冻结的 Talker，以及预训练的 HDemucs（在开源的 MUSDB18-HQ 上训练）。只有两个地方需要数据：
1. **机制诊断与外推族选择（E1.0、E1.1）**：需要有干净真值的**开源合成对**，用来测曲率与斜率、测 token 恢复率，并按预登记规则选定一个外推族。这是"测量和选择"，没有梯度训练。
2. **评测**：
   - 开源受控评测：用开源合成对的留出部分，有干净参考，可以算条件误差和说话人相似度；
   - 真实电竞：dev 和 test，内部数据，只做评测；
   - AMI 和 AliMeeting 测试集：无害检验，只做测试。

两个可选的小模块，即 E1.4 的 SNR 探针和 logit 变体的 CCA 蒸馏，如果要做，也只用开源合成数据训练。

所以 **Idea 1 可以先用开源合成数据做半天诊断，然后直接在电竞和会议数据上测试**。

**Idea 2（CA-Kalman）需要训练**，包括线性版的 EM 拟合和学习版的观测编码器。所需数据分三类，全部开源：
1. 干净说话人语料：用于先验 $\mu_0,\Sigma_0$，也用作合成流的说话人；
2. 背景素材：拼成随时间变化的广播式背景轨；
3. 真实带噪语料：用于无监督的似然项。

### 7.2 开源数据清单

服务器已实测可以直接下载 OpenSLR、Zenodo 和 HuggingFace，团队 S3 上也已有 Emilia。

| 用途 | 数据 | 语言 | 规模 | 许可 | 来源 |
|---|---|---|---|---|---|
| 干净说话人（主） | LibriTTS-R（train-clean-360；test-clean 留作评测说话人） | 英 | 约 190 h，900+ 说话人 | CC BY 4.0 | openslr.org/141 |
| 干净说话人（主） | AISHELL-3（train / test 按说话人划分） | 中 | 约 85 h，218 说话人 | Apache 2.0 | openslr.org/93 |
| 真实会议语音（补充） | AMI IHM（本机已有，train 部分） | 英 | — | CC BY 4.0 | 本机 `data/ami_multispk_si_v1` |
| 大规模真实说话人（可选） | Emilia ZH/EN（团队 S3 已有）。已做过人声分离，按 DNSMOS ≥ 3.5 筛选 | 中、英 | 约 10 万 h | CC BY-NC 4.0（待核对） | 团队 S3 `open_data_0803_fortrain/TTS` |
| 背景：音乐、噪声、人声嘈杂 | MUSAN | — | 约 11 GB | CC BY 4.0（待核对） | openslr.org/17 |
| 背景：观众与游戏类音效 | FSD50K（选类：Crowd、Cheering、Applause、Explosion、Gunshot、Whoosh、Beep 等） | — | 按类抽取 | 每条 CC0 / CC BY / CC BY-NC | Zenodo 4060432 |
| 背景：音乐加音效轨，也可作开源基准 | DnR（Divide and Remaster，由 LibriSpeech、FMA、FSD50K 合成，带干净语音真值） | 英 | 按需下载 test 和部分 train | 待核对 | Zenodo |
| 真实带噪、多次观测（Idea 2 的无监督项） | CN-Celeb v2（直播、娱乐、vlog、访谈等 11 个类别；同一人的多个会话） | 中 | 约 1000 说话人 | 研究用途（待核对） | openslr.org/82 |
| 轮次与重叠统计 | AMI 标注（本机已有） | — | — | CC BY 4.0 | 本机 |

**电竞数据（S3，kenxsong 的内部数据）：只用于评测。**
- 包括 dev 和 test 视频、demo 的 10 段、80 条配对流；
- 不参与任何训练、EM 拟合、外推族选择或超参决定；
- 论文里作为"真实广播场景"的评测集。是否公开发布要单独确认来源许可：元数据里 `license` 为空，来源是 YouTube。

### 7.2b 团队 S3 现有数据盘点（2026-11-18 实查）与下载实测

**语音部分，S3 上已有开源数据**（在 `speech_model_backup/zeyangsong/data/*.tar.zst` 打包里，需要流式解包取子集）：

| 数据 | 内容 | 规模 |
|---|---|---|
| LibriSpeech | 英文朗读，较干净，有说话人 ID | 打包里是 parquet，`train.clean.100` 等 |
| Common Voice | 中英文，众包录音，CC0 | — |
| CoVoST 2 | en→zh 429 h，zh→en 10.4 h。**自带翻译参考，可以在开源数据上直接算 ASR-BLEU** | — |
| WenetSpeech | 中文野外语音，约 1 万 h，来源是 YouTube 和播客，带真实背景音乐 | manifest 已确认；**音频是否也在 S3 上待确认**（6.8 GB 的包可能只有 cuts） |
| GigaST | 只有索引，GigaSpeech 音频不在 S3（gigaspeech 包是空的） | — |
| Emilia ZH/EN | 已做过人声分离 | 团队 S3 有 |
| CVSS | — | 团队 S3 有 |
| AMI | — | 本机已有 |

**背景部分，S3 上没有开源数据**：MUSAN、FSD50K、DNS、LibriTTS-R、AISHELL-3、CN-Celeb 都没有，需要下载。kenxsong 的噪声基准里有 MS-SNSD 和 LoL 音效，但 LoL 音效是 Riot 的素材，不是开源的，不用于训练。

**下载实测（2026-11-18）：**
- OpenSLR：单连接 1.4 MB/s，8 路分段并行 10 MB/s；
- Zenodo：单连接 0.45 MB/s，8 路并行 4.6 MB/s。

按并行速度估算：

| 数据 | 大小 | 预计耗时 |
|---|---|---|
| MUSAN | 11.1 GB | 约 20 分钟 |
| FSD50K eval | 3.0 GB | 约 11 分钟 |
| AISHELL-3 | 19.1 GB | 约 32 分钟 |
| CN-Celeb v1 | 22.3 GB | 约 37 分钟（只有当 WenetSpeech 音频不在 S3 时才需要） |
| LibriTTS-R train-clean-360 | 29 GB | 约 48 分钟（可以不用，改用 S3 上的 LibriSpeech） |

### 7.2c 各 idea 的数据量与 OOD 分析

**Idea 1（不训练）：**
- 诊断和选择集：中英各 1000 对，按 4 种背景类型（音乐、观众、音效、混合）× 5 档 BVR × 50 分层，每对 6 s，**合计约 3.3 h 语音素材**。
  - 语音：LibriSpeech test/dev（英，S3），AISHELL-3 test 说话人（中，下载）；
  - 背景：MUSAN、FSD50K eval。
- 开源全链路评测集（可选，但对论文很有价值）：用 CoVoST 2 的源语音（S3，自带翻译参考）拼成 2–4 人、带广播式背景的 5 分钟流，中→英和英→中各约 2 h，可以在开源数据上算 ASR-BLEU。
- **OOD：**
  - ZNE 没有学习到的参数，不存在训练和测试分布不一致的问题。它是否有效，取决于每条样本是否满足 (A1)–(A3)、交叉项 $a$ 是否远小于 $b_2$、分离增益 $c$ 有多大，而这些**可以逐条检验**；
  - 预登记一个"自检回退"：多算一个节点（$t=3$）估计曲率与斜率之比，超过阈值的样本退回原始提示。这样失败是可检测的，不会默默变差；
  - 主要的 OOD 风险在 HDemucs：它是在音乐上训练的，可能把观众喊声当作人声，导致 $c$ 偏小。由命题 2(a)，这时只是去除比例 $c$ 变小，一阶上不会变差；
  - OOD 评测：外推族只在"音乐 + 噪声"类背景上选，然后在"观众 + 音效"类背景上测（留一类别）；再测真实电竞（从未见过的领域）和 AMI / AliMeeting（安静数据，检验无害）。

**Idea 2（需要训练）：**
- 干净说话人：英文用 LibriSpeech train-clean-100/360（S3，约 460 h、1172 人）；中文用 AISHELL-3 train（下载，174 人）加 Common Voice zh（S3，人数多，录音设备各异）。中英各约 1000 人。
- 背景：MUSAN 音乐约 42 h、噪声约 6 h，FSD50K eval 约 13 h（选类）。按来源文件以 8:1:1 划分 train / dev / test。
- **合成多人流：100 h**（train 80 / dev 10 / test 10，中英各半）。
  - 2 s 窗口、0.5 s 跳步，约 72 万次观测，足够训练一个 1–3 千万参数的模块；
  - 多样性主要来自约 2000 个说话人和数千段背景，继续加时长的边际收益很小；
  - CPU 合成约 1–2 h。
- 真实带噪数据（无监督项）：
  - 中文：WenetSpeech 中 BVR > −20 dB 的视频和播客，取 100–200 h，按来源视频分组。如果音频不在 S3，改为下载 CN-Celeb v1（约 37 分钟）；
  - 英文：S3 上没有现成的真实带噪英文数据（GigaSpeech 音频缺失），先只用合成数据。
- **数据准备总耗时约 1 天：** 下载约 1 h；从 S3 流式解包取子集约 0.5 h；合成约 2 h；分离器、HDemucs 和编码在 4 卡上约 4 h。
- **OOD：**
  - **结构部分与领域无关**：身份静态、背景共享且可以单独观测。只要这些成立，可辨识性定理就成立；
  - **线性映射 $A$、$B$ 和观测编码器是学出来的，存在领域差异。** 缓解办法：
    - 参数少的结构化模型；
    - 背景类型尽量多样，并加入 WenetSpeech 这类真实带噪数据；
    - 在测试流内部做无监督、因果的在线 EM（只更新 $\bar n$ 和噪声尺度，事先登记）；
    - 后验不确定度经过校准：遇到 OOD 观测时似然变得没有信息量，后验自动退回先验加共享背景估计；
  - OOD 评测：只用开源数据训练，然后在真实电竞（未见过的领域）上测，并做留一背景类别实验；
  - 如实预期：Idea 2 的 OOD 稳健性弱于 Idea 1。如果合成集上收益大、电竞上收益小，就说明是领域差异，可以用"ZNE 作为每个观测的前处理"（§4.3）来补。

### 7.3 开源合成数据的构造（两个 idea 共用，第 0 天）

1. **混合模型：** $y=s+\sqrt{t}\,\alpha\,b$。$\alpha$ 让 BVR 服从电竞提示实测的经验分布（中位数约 −15 dB，范围 −25 到 −3 dB）。这个分布是对内部数据的一个统计量，只用来设定混合比例，不是训练数据。
2. **广播式背景轨：**
   - 一条持续的音乐床（MUSAN music 或 DnR music）；
   - 叠加随机出现的观众和欢呼片段（FSD50K）；
   - 叠加稀疏的游戏类音效（FSD50K）；
   - 带有说话时压低背景的侧链处理。
   - 这样就有"持续背景加时变成分"，正好对应定理 2.1 和 2.2 的 $\bar n$ 和 $\tilde n_k$。
3. **多人流（Idea 2）：** 按 AMI 标注统计的轮次长度、换人率和重叠率，把 2–4 个说话人（LibriTTS-R 或 AISHELL-3）拼成 5–10 分钟的流，叠加同一条广播式背景轨。说话人和背景素材都按 train / dev / test 划分，互不重叠。
4. **干净真值：** $u_j^\star$ 是该说话人干净语音的编码，取 6 s 干净提示编码的中心。

---

## 8. 端到端版本（2026-11-19 用户要求：主方案改为端到端）

### 8.0 "端到端"在本文的确切含义

**定义：** 生成损失的梯度从 Talker 的输出，经过 Talker 本体（LoRA 或全参数）、条件模块（ZNE 层或卡尔曼层）、BiCodec 说话人编码器（LoRA），一直传到音频输入；所有可训练部分联合优化。

**不训练的部分：** 流式分离器（Nemotron）和 Thinker（文本翻译）。它们决定"翻译什么、谁在说"，与"背景经音色泄漏"无关。论文里写成"end-to-end trainable speaker-conditioning and Talker"，不笼统声称"整个 S2ST 端到端"，以免夸大。

**规模（已核实）：** Talker（T1c）是 Qwen2 架构的 1.5B 模型（hidden 1536、28 层、3.5 GB）。说话人编码器是 ECAPA + Perceiver + FSQ，规模很小。所以 4 张 H200 上做全参数或 LoRA 训练都足够。

### 8.1 术语：Idea 1 是不是"老师教学生蒸馏"？

**不是。把它叫作师生蒸馏会误导读者。**
- 读者会以为有一个更大、更好的外部老师模型，于是把创新点理解成"又一种蒸馏"，这是一个拥挤的方向。
- 而 ZNE 的核心是一个**有数学定义的算子**（Richardson 外推）加上**由误差界推出来的训练目标**，没有外部老师。

端到端训练里有三种监督信号，必须分别命名：

| 信号 | 是什么 | 准确叫法 | 是否"师生" |
|---|---|---|---|
| (a) 曲率正则 | 让编码器对"加背景"的响应沿放大路径近似线性，直接压低定理 1 余项里的常数 $M_2$ | **界最小化训练**（bound-tightening） | 否，没有老师；在真实带噪数据上自监督 |
| (b) 干净反事实生成监督 | 在开源合成数据上，同一个冻结的 Talker 拿到**干净提示**时的输出，作为目标 | **特权信息学习**（LUPI，Vapnik & Vashist 2009；Lopez-Paz et al. ICLR 2016 统一了蒸馏与特权信息） | 形式上像：老师 = 同一模型 + 只有训练时才有的干净条件。必须明确写成"特权信息"，不能写成"大模型教小模型" |
| (c) logit 外推的摊销（可选） | 把需要两路推理的 logit 外推，训练进单路模型 | **摊销推理**（amortization），类似引导蒸馏和 CCA（ICLR 2025） | 否，目标是同一模型在测试时也能算出的外推分布，只是把计算搬进权重 |

**隐空间 ZNE 本身不需要任何蒸馏：** 它只多做一次编码器前向，Talker 仍然只跑一遍。(c) 只在想把 logit 变体也做成单路时才用。

### 8.2 Idea 1 端到端：ZNE-in-the-loop

**前向：**
$$
\hat u=\mathrm{ZNE}_\theta(y,\hat b)=2\phi_\theta(y)-\phi_\theta\big(y+(\sqrt2-1)\hat b\big),\qquad
\hat g=\mathcal Q_{\rm ST}(\hat u),\qquad x\sim p_\psi(\cdot\mid\text{text},\text{src},\hat g)
$$
- $\phi_\theta$：BiCodec 说话人编码器加 LoRA；
- $\mathcal Q_{\rm ST}$：FSQ 取整，用直通估计反传梯度；
- $p_\psi$：Talker（LoRA 或全参数）；
- 解码器冻结。若改用 E1.0 选出的 Möbius 族，前向公式相应替换。

**损失：**
$$
\mathcal L=\underbrace{\mathrm{KL}\big(p_{\psi_0}(\cdot|\text{text},\text{src},g^\star)\,\big\|\,p_\psi(\cdot|\text{text},\text{src},\hat g)\big)}_{\text{(b) 特权信息：干净反事实，合成数据}}
+\lambda_c\underbrace{\big\|f_\theta(1)-2f_\theta(2)+f_\theta(3)\big\|^2}_{\text{(a) 曲率：压低 }M_2\text{，可用真实带噪数据}}
+\lambda_a\underbrace{\mathbb E_{\rm clean}\|\phi_\theta(s)-\phi_0(s)\|^2}_{\text{锚点：保持编码空间兼容}}
+\lambda_s\underbrace{\mathcal L_{\rm spk}(\hat u)}_{\text{说话人对比：防止塌缩}}
$$
其中 $f_\theta(t)=\phi_\theta(y+(\sqrt t-1)\hat b)$，二阶差分在 $t\in\{1,2,3\}$ 上计算。

**为什么理论在训练后依然成立，而且更强：**
- 定理 1 的偏差界是 $M_2/m!\cdot\prod\lambda_i$。曲率正则直接最小化 $M_2$ 的经验估计。当二阶差分趋于 0 时，ZNE 在放大路径上**精确成立**。
- 所以训练目标是**从误差界推出来的**，提升可以解释为"ZNE 层消去一阶项，训练压低二阶项"。
- 命题 7（只沿干扰方向外推，身份保持）不依赖训练，照样成立；锚点项保证干净输入上不变（对应命题 3 的无害性）。

**关键消融（证明 novelty 来自 ZNE，而不是来自训练本身）：**
1. 只做加噪提示微调，没有 ZNE 层：即 OZSpeech / DINO 式基线，有同样的数据和训练量；
2. ZNE 层，不训练：即原来的免训练版；
3. ZNE 层加 (b)，不加 (a)；
4. 完整版：ZNE 层加 (a)、(b)、锚点。

**预期：** 4 > 1 说明"把外推结构放进模型"带来了额外收益；4 > 3 说明界最小化的曲率正则有效。

**训练数据（全部开源）：**
- 干净反事实对：用 CoVoST 2 的源语音（S3：zh→en 10.4 h，en→zh 取 50 h 子集）和翻译文本，配上同一说话人的另一段录音作提示，再按电竞的 BVR 分布叠加 MUSAN 和 FSD50K 背景。老师目标由冻结的 Talker 在干净提示下生成（teacher forcing 下的语义码分布）。
- 曲率正则：可以用 WenetSpeech 这类真实带噪开源数据（只需要 $\hat b$，不需要干净参考）。

**成本：** 生成老师目标约 0.5 天；1.5B Talker 加编码器的 LoRA 在 4 卡上训练约 0.5–1 天。

### 8.3 Idea 2 端到端：可微卡尔曼层（Structured End-to-End）

**前向：** 窗口编码器 $f_\psi$（BiCodec 编码器加 LoRA，带方差头）和背景编码器 $h_\omega$ 输出观测 → **卡尔曼层**（闭式，可微）给出每个说话人的后验 $m_{j|k}$ → $\mathcal Q_{\rm ST}$ → Talker（LoRA 或全参数）。

**损失：**
$$
\mathcal L=\underbrace{\mathrm{KL}\big(p_{\psi_0}(\cdot|g_j^\star)\,\|\,p_\psi(\cdot|\mathcal Q(m_{j|k}))\big)}_{\text{特权信息生成监督（合成多人流）}}
+\underbrace{\mathcal L_{\rm ML}}_{\text{卡尔曼精确边缘似然（真实带噪，自监督）}}
+\lambda_a\,\text{锚点}+\lambda_{\rm syn}\,\text{校准 NLL}
$$
梯度从 Talker 经过卡尔曼后验，传到两个观测编码器和状态空间参数 $(A,B,\rho,Q,R,S)$。

**为什么还可解释：** 卡尔曼层的结构不变，定理 2.1 和 2.2 照样适用。学出来的 $A$、$B$、$\rho$ 仍有物理意义（背景如何作用于音色编码、背景的持续性），可以直接画出来分析。

**关键消融：**
- 用黑盒注意力池化代替卡尔曼层（同样端到端训练、同样的数据）；
- 卡尔曼层去掉只含背景的观测流；
- 完整版。

**预期：** 结构化层优于黑盒池化，尤其在持续背景、首句和 OOD 条件下。这正是两个定理预测的。

**训练数据：** §7 的 100 h 开源合成多人流（干净反事实对来自同一套合成）加 WenetSpeech 或 CN-Celeb 的真实带噪数据（似然项）。

**成本：** 比非端到端的学习版多约 1–2 天，用于反传 Talker 的联合训练和调试。

### 8.4 对时间表的影响

| | 原方案（模块化） | 端到端主方案 |
|---|---|---|
| Idea 1 | 约 1–1.5 天出核心结果，约 3 天出正式结论 | 先用免训练版做 G1.0 和 G1.2（约 1–1.5 天，作为消融 2），再做端到端训练和评测（+2 天）。**约 3.5 天出核心结果，约 5 天出正式结论** |
| Idea 2 | 线性版约 3 天，学习版约 5–6 天 | 线性版仍约 3 天（作为结构基线），端到端约 **6–7 天** |

两条线仍各用 4 张卡并行。Talker 只有 1.5B，训练不会成为瓶颈。主要时间花在数据构造、生成老师目标和调试上。

---

## 9. 主要参考（完整列表见 solutions 文档 §8、§9）

- 特权信息与摊销（§8 端到端版本用）：
  - Vapnik & Vashist 2009（LUPI）；
  - Lopez-Paz et al.，*Unifying Distillation and Privileged Information*，ICLR 2016；
  - Meng et al.，*On Distillation of Guided Diffusion Models*，CVPR 2023；
  - CCA（ICLR 2025）。

- 零干扰外推和条件外推：
  - Temme, Bravyi, Gambetta（量子 ZNE，arXiv 1612.02058）；
  - Noisier2Noise（CVPR 2020）；
  - Ambient Diffusion（NeurIPS 2023）；
  - Autoguidance（NeurIPS 2024）；
  - Stay on Topic with CFG（ICML 2024）；
  - Contrastive Decoding（ACL 2023）；
  - Instructive Decoding（ICLR 2024）；
  - CCA（ICLR 2025）；
  - ICG（ICLR 2025）。
- 状态空间、可辨识性与聚合：
  - Multi-View Causal Representation Learning（ICLR 2024）；
  - von Kügelgen et al.（NeurIPS 2021）；
  - Bayesian Context Aggregation（ICLR 2021）；
  - MCInfoNCE（ICML 2023）；
  - IFactor（NeurIPS 2023）；
  - SepCLR（ICLR 2024）；
  - 说话人识别中的 JFA / NAP（经典方法）。
- 语音近邻工作：
  - Guided Speaker Embedding（ICASSP 2025）；
  - DAIEN-TTS（arXiv 2509.14684）；
  - NR-LauraTTS（Interspeech 2025）；
  - SEED（Interspeech 2025）；
  - Mega-TTS 2（ICLR 2024）；
  - Spark-TTS / BiCodec（arXiv 2503.01710）。
