# E2 结果：严格基线 Thinker + 方法的语音层（rl 合并 + PL-EM talker）

完成：2026-11-15 01:16 布拉格时间（E2_DONE）。脚本 experiments/mist_w3po_v1/scripts/e2_baseline_plem.sh。
文本侧不重解码：严格基线 joint v8 SFT s6000 已有文本输出 → read-level merge（`_rl`）→ PL-EM talker（zh→en: talker_plem_20261111T001759Z；En→Zh: talker_plem_en2zh_20261114T045019Z export_i0400），两个 TTS seed → chunk 协议语音延迟。

ASR-BLEU = tcpBLEU_all（seed1 / seed2）；LAAL = 逐单元语音 LAAL（chunk 协议，秒）；End = EndOffset（补充）。

| 集合 | 严格基线 (frozen T1c) ASR | E2 (基线 Thinker + rl + PL-EM) ASR | 基线 LAAL → E2 LAAL | 基线 End → E2 End |
|---|---|---|---|---|
| dev zh→en | 19.22 / 19.25 | 23.27 / 23.29 | 3.968 → 3.528 | 4.128 → 3.099 |
| dev En→Zh | 31.07 / 31.06 | 31.57 / 31.57 | 3.123 → 2.859 | 3.456 → 2.836 |
| test AMI zh→en | 18.09 / 18.04 | 20.34 / 20.36 | 4.012 → 3.611 | 4.079 → 3.128 |
| test Ali zh→en | 16.71 / 16.74 | 18.10 / 18.06 | 4.279 → 3.647 | 4.571 → 3.228 |
| test AMI En→Zh | 31.64 / 31.65 | 32.23 / 32.23 | 3.207 → 2.951 | 3.482 → 2.908 |

## 与完整方法（v2′ s3500 Thinker + rl + PL-EM，δ=0，11-14 上午测试）对照

| 集合 | E2 ASR | 方法 ASR (δ=0) | E2 LAAL | 方法 LAAL (δ=0) |
|---|---|---|---|---|
| AMI zh→en | 20.34 | 20.48 | 3.611 | 3.986 |
| Ali zh→en | 18.10 | 17.37 | 3.647 | 4.848 |
| AMI En→Zh | 32.23 | 32.41 | 2.951 | 3.336 |

## 结论（如实）

1. 语音层（read-level merge + PL-EM talker）单独就带来了大部分 ASR-BLEU 增益和全部延迟下降：AMI zh→en +2.25，Ali +1.39，En→Zh +0.59；LAAL 降 0.26–0.63 s。
2. 在其上换成 v2′ Thinker（δ=0）：AMI 两个方向 ASR 再 +0.14/+0.18，但 LAAL 增加 0.38 s 左右；Ali 上 ASR −0.73、LAAL +1.2 s，比 E2 差。
3. 所以论文里 Thinker（latent write alignment）的贡献主要在文本 tcpBLEU（AMI 25.02→25.34 δ=−1），语音层贡献主要来自 PL-EM talker。δ=−1 的 E1 结果出来后再定最终表述；文中必须把 E2 作为分解行报告，不能把 E2 的增益记在 Thinker 上。
