# Idea 7 预登记：失真–感知权衡（第 1 步：机制检验，只用已有测试输出）

2026-11-19 布拉格，写于任何 Idea 7 数字之前。用户："开始第一步验证"。

## 动机
Idea 6 测试里，音色码误差与输出音质反向：C 只用前 8 s（码误差 9.58）PESQ 最好，C 逐窗口 + 闭式后验（7.95）次之，端到端（7.36）更差。失真–感知权衡（Blau & Michaeli 2018；Freirich et al. 2021）预测：后验均值类估计会"回归到平均"，分布被压扁、偏离真实音色分布，从而感知质量变差。

## 性质
只解释已有的测试结果，不做任何选择、拟合或调参；不改变任何方法的结论。

## 数据
- 9 组：OpenBroadcast-SI 的 ID-2 / OOD-1 / clean × 两方向，OOD-3 的 ami / en / ali。
- 臂：C（Idea 1 C，前 8 s 提示）、Z（Idea 1 ZNE）、I2（Idea 2 冻结 shrink+Vfb）、E2E2（Idea 2 端到端）、i6c（C 逐窗口 + 闭式后验）、i6e（Idea 6 端到端）；OOD-3 另有 B（原始 token）。
- 条目：只用 Idea 2 / Idea 6 表里非回退（fallback = False）的条目，所有臂取同一组条目。
- 音质：逐条 PESQ（两个 TTS 种子都有的条目取均值），每臂取中位数。
- token → 32×6 个 4 进制数字（192 维）。干净分布：idea2 refs 最新 ustar 表的 g*（Ledoit–Wolf 高斯）。

## 统计量
- 展宽比 spread = tr(Cov(臂的 token)) / tr(Cov(干净 token))；
- Fréchet 距离 FD（臂的高斯 vs 干净高斯）；
- 逐条马氏距离 dM 的中位数。

## 判据（机制成立 = M1 且 M2）
- M1 压扁：后验均值类臂（I2、i6c、i6e）的展宽比都低于 C，在 ≥ 7/9 组成立。
- M2 解释音质：每组内 6 个估计臂（C、Z、I2、E2E2、i6c、i6e），Spearman(FD, PESQ 中位数) ≤ −0.5，且 Spearman(展宽比, PESQ 中位数) ≥ +0.5，在 ≥ 6/9 组成立。
- M3（附带）：逐条配对（i6c−C、i6e−C、I2−C、i6e−i6c），Δ dM 与 Δ PESQ 的 Spearman 合并 < 0 且 p < 0.01。

成立 → 写第 2 步预登记（最优传输修正，dev 上带 Talker 用真实音质选择）。不成立 → 停止，如实记录。
