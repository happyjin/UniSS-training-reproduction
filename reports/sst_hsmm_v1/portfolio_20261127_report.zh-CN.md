# 组合实验（9 个方法）最终报告 — 2026-11-27 15:00 CET

预注册：prereg.zh-CN.md change 55（含补充）；方案：plan_portfolio_20261127.zh-CN.md；逐方法记录：iterations_20261125.zh-CN.md；
全部指标总表：portfolio_20261127_tables.zh-CN.md。时间线：09:46 第一批开始，14:45 最后一个方法（A2）测试完成。

## 1. 结论

- 9 个方法全部跑完（需要训练的 E1 训练 + rollout + HSMM 拟合 + dev + 测试，其余 rollout + 拟合 + dev + 测试）。全部 dev 选择只用 edev + cvdev，没有在测试集上调参。
- **两个方向都有效：0 个。单方向有效（zh2en）：D1、E1。无效：A1、A2、B1、B2、C1、C2、D2。**
- zh2en 有两个新的工作点：
  - **D1（可提交性 MBR）**：与 p3hk 同质量（ASR +0.10），D2 −0.07 s。增益集中在 OBSI（−0.11 s，CI 不含 0）。dev 筛选和测试在同一方向都通过，是 9 个里唯一的。
  - **E1（P2F-T）**：比 p3hk2 再快 0.14～0.29 s（5/5 cell 显著），ASR 与 p3hk2 的差异 5/5 cell 不显著。
- en2zh：没有任何方法外推前沿。凡是让提交提前的方法（E1、D1、D2、A1、C2），en2zh 的 BLEU 损失都超过前沿斜率。
- COMET（辅助）与 ASR-BLEU 的排序基本一致；唯一的例外是 E1（见第 4 节），不改变判定。

## 2. 每个方法的结果（测试相对 p3hk；"连线" = 比 p3hk–p3hk2 连线在同 D2 处高多少 BLEU，门槛 +0.2）

| 方法 | dev 筛选 en2zh / zh2en | en2zh ΔASR / ΔD2 / 连线 | zh2en ΔASR / ΔD2 / 连线 | 判定 |
|---|---|---|---|---|
| A1 带偏置的投机重写（β 1.0） | 无效 / 无效 | −0.77 / −0.120 s / −0.52 | −1.10 / +0.019 s / −1.16 | 无效 |
| A2 BID 双向草稿选择 | 无效 / 无效 | −0.24 / −0.030 s / −0.17 | −0.24 / +0.017 s / −0.30 | 无效 |
| B1 反事实音频证书 | 无效 / 无效 | −0.08 / −0.016 s / −0.05 | +0.08 / −0.007 s / +0.11 | 无效 |
| B2 边界对齐读 | 无效 / 无效 | −0.28 / +0.019 s / −0.32 | −0.66 / +0.085 s / −0.93 | 无效 |
| C1 分组 conformal 阈值 | 无效 / 无效 | −0.11 / −0.015 s / −0.07 | +0.00 / −0.038 s / +0.12 | 无效 |
| C2 HSMM 发射 MAP 域校准 | 无效 / 无效 | −0.29 / −0.098 s / −0.08 | −1.02 / −0.219 s / −0.33 | 无效 |
| D1 可提交性 MBR（support） | 无效 / **有效** | −0.65 / −0.081 s / −0.49 | **+0.10 / −0.073 s / +0.33** | **zh2en 有效** |
| D2 D1 + 断句候选 | 无效 / **有效** | −0.50 / −0.109 s / −0.27 | −0.29 / −0.139 s / +0.16 | 无效（zh2en 差一点） |
| E1 P2F-T + 时域标签 + 匹配 rollout | 无效 / 无效 | −1.79 / −0.377 s / −1.00 | −0.58 / −0.259 s / **+0.24** | **zh2en 有效** |

所有指标（文本 BLEU、D1、EndOffset、NP_in、PT_in、NC、sim、PESQ、ASR-COMET、文本 COMET，相对 B_b，并列 N、LA-2、p3hk、p3hk2）见 portfolio_20261127_tables.zh-CN.md。

## 3. 单方向有效方法的配对 bootstrap（zh2en，逐 cell，2000 次；顺序 RealSI clean / noisy，OBSI clean / id2 / ood1）

- **D1 相对 p3hk：**
  - ASR +0.58 / −0.01 / **+0.93** / +0.05 / **−1.05**。obsi/clean 显著更好，obsi/ood1 显著更差。
  - D2 +0.01 / −0.04 / **−0.11 / −0.12 / −0.11**，OBSI 三格 CI 都不含 0。
  - → OBSI clean / id2 同质量、提交提前约 0.11 s；RealSI 延迟不变；obsi/ood1 有约 1 BLEU 的显著代价。
- **E1 相对 p3hk2：**
  - D2 −0.17 / −0.29 / −0.14 / −0.16 / −0.19，5/5 显著。
  - ASR +0.84 / +0.36 / −1.08 / −0.66 / −1.27，5/5 CI 含 0。
  - 文本 BLEU 只有 obsi/ood1 显著下降（−1.37）。
  - 相对 p3hk：obsi/ood1 的 ASR 显著更差（−1.28）。

## 4. COMET 辅助指标（wmt22-comet-da，ASR 转写与文本译文分别打分）

- 结果目录：eval_outputs/sst_hsmm_v1/pf_comet2，共 14 个系统，en2zh 2442 条、zh2en 2893 条（各系统共有条目）。
- ASR-COMET 相对 B_b（en2zh / zh2en）：
  - 参照系统：N +0.0147 / +0.0208，p3hk +0.0164 / +0.0222，p3hk2 +0.0112 / +0.0204，LA-2 +0.0089 / +0.0152。
  - D1：+0.0132 / **+0.0227**（zh2en 最高的非 E1 系统）。
  - E1：**+0.0218 / +0.0294**。
- E1 是唯一 COMET 与 BLEU 明显矛盾的方法：en2zh ASR-BLEU 比 p3hk 低 1.79，ASR-COMET 却高 0.0055（OBSI clean / ood1 显著）。核查发现：
  - E1 的 LoRA 用 UniST 数据训练，输出改为全角标点、更意译。官方 BLEU 先去标点，按官方口径 en2zh OBSI clean 的文本 BLEU 比 p3hk 低 6.1；原始 sacrebleu 把与参考一致的全角标点算作匹配，只低 1.8。
  - 抽查到漏译（例："南方四所学校之间存在激烈竞争" → "有激烈的竞争关系"）。COMET 对漏译的惩罚通常偏弱（神经指标的已知问题），所以这里的 COMET 优势可能部分来自更流畅但不完整的输出。
  - → 判定按预注册 BLEU 优先，不变。E1 的 COMET 优势不能当作"质量更好"的证据。

## 5. 为什么大多数方法无效（跨方法的规律）

1. **提交判据一侧已经饱和（B1、C1、C2）。** B1 的反事实证书和跨读一致性高度相关（AUC 0.743 对 0.748），C1 的分组阈值在 0.70～0.85 之间互相抵消，两者都与 p3hk 几乎重合。C2 修好了 CV 上的校准（ECE 0.077 → 0.056），但效果只是整体平移阈值，沿前沿滑动。这与 11-26 "提交证据线饱和"的结论一致。
2. **同一读上的候选选择只改措辞，不改内容何时稳定（A2、D2；A1 是改写偏置）。** 样本之间的差异集中在尾部措辞。D1 是例外：它的效用直接奖励"与 SAP 草稿共享前缀"，等于把提交判据的信息带进了草稿选择，所以在 zh2en 的 OBSI 上让提交提前。
3. **en2zh 的瓶颈是重排序。** 这是第三次看到同样的模式（S3 / S8 / E1，加上 D1 / D2）：让草稿提前稳定的方法在 zh2en 有效，在 en2zh 会提前提交错误语序或字面顺译。
4. **减少输入信息必然有害（B2）。** 每读平均少看 82 ms 音频，质量和延迟双输。

## 6. 自主决策记录（均已提交推送）

- 10:31：E1 训练比预估快（53 min，预估约 4 h），开 lane 4 立即接上 E1，不等 lane 2 排到。
- 11:48：发现旧的 lane 脚本在方法已存在时会先清空该方法日志再拒绝。按 PID 停掉 lane 1 / 2 的外层 bash（正在跑的 B1 / B2 管线不受影响），新开 lane 5–7（pf_lane_lib2.sh：mkdir 原子领取 + noclobber，多 lane 共享剩余方法）。12:08 E1 结束后开 lane 8 接 D1。14:45 全部完成。
- C1 的错误定义（提交单元位于 LCP(final, shadow) 之后）把同义改写也算错误，α≈0.49～0.50 偏高；按预注册原样执行并注明。
- COMET：新建 venvs/comet（unbabel-comet + setuptools<70）；对 14 个系统（含 LA-2）在共同条目上打分，ASR 转写与文本各一次。
- 配对 bootstrap 只做了两个单方向有效方法的 zh2en（相对 p3hk 与 p3hk2），并补了 COMET 的逐 cell 配对 bootstrap（pf_comet_pair.py）。

## 7. 需要你决定（都未启动）

1. 是否把 zh2en 的 D1（同质量、OBSI 快 0.11 s，纯推理，不需要训练）作为 zh2en 的工作点。这样系统会按方向使用不同的解码规则，模型仍是同一个双向模型。
2. E1 的 COMET 与 BLEU 矛盾。如果要追究 E1 是否真的更好，需要人工或独立评测（例如 MQM 抽样）。是否值得做？
3. 论文：你说过无效的不写。之前写进附录的夜间负面结果（leakage、P2F、anticipation、MBR）要不要用 git revert 撤掉？本次 9 个方法都没有写进论文。

## 参考文献（方法来源）

- A1：Arivazhagan et al. 2020. Re-translation versus Streaming for Simultaneous Translation. IWSLT. https://aclanthology.org/2020.iwslt-1.27 ；Zeng et al. 2025. Self-Speculative Biased Decoding for Faster Re-Translation. https://arxiv.org/abs/2509.21740 ；Leviathan, Kalman, Matias 2023. Fast Inference from Transformers via Speculative Decoding. ICML. https://arxiv.org/abs/2211.17192
- A2：Liu et al. 2025. Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling. ICLR. https://arxiv.org/abs/2408.17355 ；Zhao et al. 2023. ACT（temporal ensembling）. RSS. https://arxiv.org/abs/2304.13705
- B1：Leng et al. 2024. Visual Contrastive Decoding. CVPR. https://arxiv.org/abs/2311.16922 ；Shi et al. 2024. Context-aware Decoding. NAACL. https://arxiv.org/abs/2305.14739
- B2：Wang et al. 2024. Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection. Interspeech. https://arxiv.org/abs/2406.10052
- C1：Gibbs, Candès 2021. Adaptive Conformal Inference Under Distribution Shift. NeurIPS. https://arxiv.org/abs/2106.00170 ；Angelopoulos et al. 2024. Conformal Risk Control. ICLR. https://arxiv.org/abs/2208.02814
- C2：Gauvain, Lee 1994. MAP estimation for multivariate Gaussian mixture observations of Markov chains. IEEE TSAP 2(2). https://ieeexplore.ieee.org/document/279278 ；Wang et al. 2021. Tent. ICLR. https://arxiv.org/abs/2006.10726
- D1 / D2：He, Boyd-Graber, Daumé III 2016. Interpretese vs. Translationese. NAACL. https://aclanthology.org/N16-1111/ ；Chen et al. 2021. Pseudo-References with Fewer Reorderings. EMNLP. https://aclanthology.org/2021.emnlp-main.473 ；Zhang, Yang, Nakamura 2026. Redefining Machine Simultaneous Interpretation. IWSLT. https://arxiv.org/abs/2601.11002
- E1：Chen et al. 2024. Diffusion Forcing. NeurIPS. https://arxiv.org/abs/2407.01392 ；Zeghidour, Défossez et al. 2025. Delayed Streams Modeling. https://arxiv.org/abs/2509.08753 ；Ouyang et al. 2025. Anticipating Future with LLM for SiMT (TAF). NAACL. https://arxiv.org/abs/2410.22499
- COMET：Rei et al. 2022. COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task. WMT. https://aclanthology.org/2022.wmt-1.52
