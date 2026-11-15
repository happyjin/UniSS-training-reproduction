# E7 结果：SeamlessStreaming（开源，只做测试推理）多说话人外部基线

完成：2026-11-15 01:29 布拉格时间（E7_QUEUE_DONE，三个集合均无失败）。预注册 reports/mist_w3po_v1/e7_seamless_multispk_prereg.zh-CN.md；代码 experiments/s2st_baselines_v1/seamless_multispk/；汇总 eval_outputs/mist_w3po_v1/e7_seamless/e7_summary_v1.json（summarize_e7.py）。
默认配置，无任何训练/调参；测试集只用于推理。与我们的系统同一 tcpBLEU / ASR-BLEU / chunk 协议延迟评测。

| 集合 | tcpBLEU all / ov / 1spk | ASR-BLEU all | 文本 LAAL | 语音 LAAL | Start | End |
|---|---|---|---|---|---|---|
| AMI zh→en | 8.69 / 3.07 / 11.19 | 8.48 | 2.63 | 3.29 | 2.89 | 2.53 |
| Ali zh→en | 10.41 / 4.42 / 13.27 | 9.42 | 3.09 | 3.96 | 3.38 | 2.72 |
| AMI En→Zh | 13.28 / 5.86 / 16.71 | 11.31 | 2.31 | 3.19 | 2.39 | 3.04 |

（"1-spk LB" 为把整窗当单说话人时的下界；En→Zh 的"speech LAAL 1ch"=18.02 是单通道拼接口径，不进论文主表。）

## 结论
SeamlessStreaming 延迟更低（语音 LAAL 比我们低 0.0–0.5 s），但质量远低：ASR-BLEU 约为我们的 40%（AMI zh→en 8.48 vs 20.45；En→Zh 11.31 vs 32.33），重叠区 tcpBLEU 尤其低（3–6）。它没有说话人归属，单说话人模型在会议多说话人场景下不可用。论文中作为外部参考行（\ifEseven），注明 zero-shot、无说话人归属、延迟不可直接比较质量-延迟权衡。
