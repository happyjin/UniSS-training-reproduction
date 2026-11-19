# E10 预登记：LiveInterpret S5 按 B 的形式训练的多说话人同传 baseline

- 登记时间：2026-11-19（布拉格时间），在正式训练开始之前。
- 此前看过的输出，全部是冒烟用途：
  - E10-0（未经训练的 checkpoint-29773）和一个只训了 5 步的 E10 adapter，各解码了 3 个 AMI Zh→En dev 窗口（ES2011a_on_0026 / 0062 / 0080），只看了文本，打出的分数没有意义（3 个窗口对不上整场会议的参考）；
  - 训练冒烟时 teacher-forced 的 loss 和决策 F1。
  - 任何测试窗口都没有解码过。
- 方案与实现：`reports/mist_w3po_v1/e10_liveinterpret_multispk_plan.zh-CN.md`（§5、§9 用户决定、§11 实现记录）。代码：`third_party/liveinterpret_training_pipelines/e10_multispk/`。

## 1. 系统

- **E10（主行，Table 1）。** 同事的 LiveInterpret S5，结构是 Voxtral 加 Qwen3.5-4B 的 Thinker，加 0.6B 的 Talker（CosyVoice3 FSQ），用 checkpoint-29773 作起点。按 B 的形式在 B 的 AMI 数据上只训 Thinker，Talker 冻结。说话人由模型自己输出的 tag 决定，"own speaker attribution" 一列标 ✓。
- **E10-0（只进附录）。** 不训练的 checkpoint-29773，当作外部系统来跑：说话人用因果重标（E7/E8 的规则），音色用 S5 原生的滚动 2 s 参考。
- **E10-rel（只进附录）。** E10 的同一批输出，说话人改用因果重标，用来区分说话人通道本身的贡献。
- 用户确认：LiveInterpret 的训练数据没有用过 AMI、AliMeeting 或 DiariST。

## 2. 训练（只训一种设置，不扫超参）

- **数据**：`e10_multispk/data/li_s5_records_v1`，由 `build_li_records.py` 用 seed 20261026 生成。
  - 用 B 的 AMI 窗口和截止时间标签：标点切块，块对应的源词结束后再等 1 s 到期；说话人用 lowlat slot。时间粒度为 640 ms。
  - 另加离线行（窗口数的 20%）和 B 的单说话人行（全部池子）。
  - 两个方向数量对齐，各 35,631 条。
  - 只用开源数据（AMI，以及 B 用过的 Emilia 单说话人行）。
- **训练范围**：Thinker LoRA（r64/α128，从 S5 adapter 继续）、12 个 token 行、`e10_sd_proj`。text 权重 1，speech 权重 0，决策权重 1。
- **后验**：第 j 个音频帧加上 `e10_sd_proj(post[j−13])`。训练时按 B 的方式扰动（p=0.2，σ=0.1，平移 ±1 帧）。
- **优化**：
  - lr 1.2e-5，cosine 退到 2e-6，warmup 3%；`e10_sd_proj` 用 1e-4；
  - 卡 0–3，每卡 batch 8，GA 2，GBS 64；1.5 epoch（约 1,670 步）；seed 20261026；
  - 每 250 步存一次 adapter（只存 adapter）。
- **选择**：取最后一步，不做 checkpoint 选择（同 B 和 E9）。训练中的 dev loss 和决策 F1 只用来监控。

## 3. dev 门槛（只做健全性检查，不调参）

在 AMI Zh→En dev（`windows_zh2en_c2g06_v1`）和 AMI En→Zh dev（`windows_en2zh_c2g06_v1`）上，用最后一步、第一个 TTS 种子解码。下面四项任一不满足，就停下来汇报，由用户决定，不自己重训：

1. 塌缩窗口 ≤ 2%（判据同 S5 的 `collapse_ok`）；
2. 有译文的窗口里，带说话人 tag 的 piece ≥ 95%；
3. tcpASR-BLEU 不低于 E10-0 在同一 dev 上的值；
4. 文本 BLEU 与 tcpASR-BLEU 的差距，比 E10-0 的差距扩大不超过 2 BLEU（检查 Talker 是否因输入分布变化而退化）。

## 4. 测试

- **测试集**：AMI Zh→En（`WT_AMI`）、AliMeeting（`WT_ALI`，DiariST 人工英文参考）、AMI En→Zh（`WT_EN`）。每个集合只解码一次，两个 TTS 种子（0 和 20261109）。
- **解码设置**：640 ms 一块；读写决策取 argmax；Thinker 贪心；Talker top-k 25 / top-p 0.8；ode 5 步；`speech_tokenizer_v3.batch.onnx`（用户已确认）。音色用 B 的 `--causal` 规则。
- **指标**：
  - cpASR-BLEU / tcpASR-BLEU（Whisper-large-v3 / Paraformer-zh）、chunk 协议下的 LAAL_block，以及文本 tcpBLEU；
  - EndOffset 只进附录；
  - 计算全部用 E8 的同一条评测链（`eval_e10.py`）。
- **统计**：E10 并入 Table 1 现有的 Holm 族，配对检验与 Table 1 相同。
- MIST 不因 E10 的结果做任何改动。

## 5. 修订

所有修订都带时间戳，写在受影响的结果出来之前，并记进 `reports/mist_w3po_v1/LOG.zh-CN.md`。
