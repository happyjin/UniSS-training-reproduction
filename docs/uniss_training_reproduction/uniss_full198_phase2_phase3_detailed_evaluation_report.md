# UniSS full198 Phase2 / Phase3 详细评估报告

> 生成时间：2026-07-26T17:44:44Z
> 论文参考：[arXiv:2509.21144](https://arxiv.org/pdf/2509.21144)，Table 1

## 1. 结论与比较边界

当前 Phase2/Phase3 全量结果来自 **UniST dev/test**，论文 Table 1 来自 **CVSS-T test 4,897 对**。
因此本报告只做 Phase2 与 Phase3 的同数据内部比较；论文表格作为参考背景展示，**不计算跨数据集差值、胜负或排名**。

> 2026-07-29 修正：原报告 ZH→EN Speech-BLEU 使用的 batched Whisper-large-v3 缺失显式 attention mask，旧 `1.x` 数值无效。21/21 个正式 run 已使用 `whisper-large-v3-attention-mask-v2` 完成重算。EN→ZH Paraformer 和其他指标不受影响。完整根因与审计见 `whisper_attention_mask_v2_correction_report.md`。

## 2. Phase2 与 Phase3 全量指标对比

### unist dev

| 指标 | Mode | 方向 | Phase2 | Phase3 | Δ(Phase3-Phase2) | N2/N3 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| AutoPCP (A.PCP) | Performance (P) | ZH→EN | 2.6711 | 2.8083 | +0.1372 | 6508/6513 |
| AutoPCP (A.PCP) | Performance (P) | EN→ZH | 3.1614 | 3.1667 | +0.0053 | 1434/1434 |
| AutoPCP (A.PCP) | Quality (Q) | ZH→EN | 2.6315 | 2.7701 | +0.1385 | 6511/6518 |
| AutoPCP (A.PCP) | Quality (Q) | EN→ZH | 3.1466 | 3.1598 | +0.0132 | 1434/1434 |
| SLC-0.2 | Performance (P) | ZH→EN | 0.5533 | 0.5953 | +0.0420 | 6508/6513 |
| SLC-0.2 | Performance (P) | EN→ZH | 0.7078 | 0.7225 | +0.0146 | 1434/1434 |
| SLC-0.2 | Quality (Q) | ZH→EN | 0.5543 | 0.5960 | +0.0417 | 6511/6518 |
| SLC-0.2 | Quality (Q) | EN→ZH | 0.6799 | 0.7197 | +0.0397 | 1434/1434 |
| SLC-0.4 | Performance (P) | ZH→EN | 0.8042 | 0.8366 | +0.0324 | 6508/6513 |
| SLC-0.4 | Performance (P) | EN→ZH | 0.9282 | 0.9365 | +0.0084 | 1434/1434 |
| SLC-0.4 | Quality (Q) | ZH→EN | 0.8005 | 0.8326 | +0.0321 | 6511/6518 |
| SLC-0.4 | Quality (Q) | EN→ZH | 0.9296 | 0.9344 | +0.0049 | 1434/1434 |
| Speech-BLEU | Performance (P) | ZH→EN | 16.3810 | 16.5154 | +0.1344 | 6508/6513 |
| Speech-BLEU | Performance (P) | EN→ZH | 35.1379 | 35.9432 | +0.8054 | 1433/1433 |
| Speech-BLEU | Quality (Q) | ZH→EN | 19.3076 | 21.3884 | +2.0808 | 6511/6518 |
| Speech-BLEU | Quality (Q) | EN→ZH | 41.6389 | 42.8128 | +1.1739 | 1434/1433 |
| Text-BLEU | Performance (P) | ZH→EN | 32.7111 | 33.3860 | +0.6749 | 6528/6526 |
| Text-BLEU | Performance (P) | EN→ZH | 37.1654 | 37.8105 | +0.6451 | 1434/1434 |
| Text-BLEU | Quality (Q) | ZH→EN | 39.6015 | 40.4631 | +0.8616 | 6530/6528 |
| Text-BLEU | Quality (Q) | EN→ZH | 44.1074 | 44.8822 | +0.7749 | 1434/1434 |
| UTMOS | Performance (P) | ZH→EN | 3.4923 | 3.4991 | +0.0068 | 6508/6513 |
| UTMOS | Performance (P) | EN→ZH | 3.0829 | 3.0876 | +0.0047 | 1434/1434 |
| UTMOS | Quality (Q) | ZH→EN | 3.5095 | 3.5021 | -0.0074 | 6511/6518 |
| UTMOS | Quality (Q) | EN→ZH | 3.0987 | 3.1044 | +0.0056 | 1434/1434 |

在全部 higher-is-better 指标单元中：Phase3 上升 23 项，下降 1 项，持平 0 项。
该计数用于定位变化，不替代按任务重要性、置信区间和人工试听做模型选择。

### unist test

| 指标 | Mode | 方向 | Phase2 | Phase3 | Δ(Phase3-Phase2) | N2/N3 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| AutoPCP (A.PCP) | Performance (P) | ZH→EN | 2.8311 | 2.9118 | +0.0807 | 14211/14232 |
| AutoPCP (A.PCP) | Performance (P) | EN→ZH | 3.2508 | 3.2765 | +0.0257 | 9112/9112 |
| AutoPCP (A.PCP) | Quality (Q) | ZH→EN | 2.7896 | 2.8850 | +0.0954 | 14223/14235 |
| AutoPCP (A.PCP) | Quality (Q) | EN→ZH | 3.2448 | 3.2752 | +0.0303 | 9112/9112 |
| SLC-0.2 | Performance (P) | ZH→EN | 0.5882 | 0.6324 | +0.0442 | 14211/14232 |
| SLC-0.2 | Performance (P) | EN→ZH | 0.6843 | 0.7184 | +0.0341 | 9112/9112 |
| SLC-0.2 | Quality (Q) | ZH→EN | 0.6046 | 0.6358 | +0.0312 | 14223/14235 |
| SLC-0.2 | Quality (Q) | EN→ZH | 0.6977 | 0.7246 | +0.0270 | 9112/9112 |
| SLC-0.4 | Performance (P) | ZH→EN | 0.8560 | 0.8782 | +0.0221 | 14211/14232 |
| SLC-0.4 | Performance (P) | EN→ZH | 0.9304 | 0.9421 | +0.0116 | 9112/9112 |
| SLC-0.4 | Quality (Q) | ZH→EN | 0.8458 | 0.8712 | +0.0254 | 14223/14235 |
| SLC-0.4 | Quality (Q) | EN→ZH | 0.9267 | 0.9349 | +0.0082 | 9112/9112 |
| Speech-BLEU | Performance (P) | ZH→EN | 16.8206 | 19.3770 | +2.5564 | 14211/14232 |
| Speech-BLEU | Performance (P) | EN→ZH | 37.8378 | 38.9953 | +1.1575 | 9111/9110 |
| Speech-BLEU | Quality (Q) | ZH→EN | 21.3719 | 22.7268 | +1.3549 | 14223/14235 |
| Speech-BLEU | Quality (Q) | EN→ZH | 44.9075 | 46.3063 | +1.3988 | 9110/9111 |
| Text-BLEU | Performance (P) | ZH→EN | 31.7635 | 32.4509 | +0.6875 | 14252/14247 |
| Text-BLEU | Performance (P) | EN→ZH | 39.6430 | 40.5404 | +0.8974 | 9111/9108 |
| Text-BLEU | Quality (Q) | ZH→EN | 38.5321 | 39.3753 | +0.8432 | 14252/14253 |
| Text-BLEU | Quality (Q) | EN→ZH | 47.0460 | 48.1698 | +1.1238 | 9111/9110 |
| UTMOS | Performance (P) | ZH→EN | 3.6798 | 3.6676 | -0.0122 | 14211/14232 |
| UTMOS | Performance (P) | EN→ZH | 3.3526 | 3.3516 | -0.0010 | 9112/9112 |
| UTMOS | Quality (Q) | ZH→EN | 3.6827 | 3.6680 | -0.0147 | 14223/14235 |
| UTMOS | Quality (Q) | EN→ZH | 3.3548 | 3.3586 | +0.0038 | 9112/9112 |

在全部 higher-is-better 指标单元中：Phase3 上升 21 项，下降 3 项，持平 0 项。
该计数用于定位变化，不替代按任务重要性、置信区间和人工试听做模型选择。

## 3. 生成完整性与失败审计

| Run | Stage | Dataset/split | Scope | Decoded/total | Failed | No semantic | Dummy tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| qwen0p5b_phase2_unist198_iter_0015381_unist_dev_full_20260726T065924Z | phase2 | unist/dev | full | 15930 | 43 | 43 | 0 |
| qwen0p5b_phase2_unist198_iter_0015381_unist_dev_listen_20260726T065924Z | phase2 | unist/dev | listen | 100 | 0 | - | - |
| qwen0p5b_phase2_unist198_iter_0015381_unist_dev_smoke_20260726T065924Z | phase2 | unist/dev | smoke | 6 | 0 | - | - |
| qwen0p5b_phase2_unist198_iter_0015381_unist_dev_vllm_smoke_20260726T065924Z | phase2 | unist/dev | vllm_smoke | 6 | 0 | 0 | 0 |
| qwen0p5b_phase2_unist198_iter_0015381_unist_test_full_20260726T065924Z | phase2 | unist/test | full | 46738 | 80 | 80 | 0 |
| qwen0p5b_phase3_unist198_iter_0009075_unist_dev_full_20260726T065924Z | phase3 | unist/dev | full | 15930 | 31 | 31 | 0 |
| qwen0p5b_phase3_unist198_iter_0009075_unist_dev_listen_20260726T065924Z | phase3 | unist/dev | listen | 100 | 0 | - | - |
| qwen0p5b_phase3_unist198_iter_0009075_unist_dev_smoke_20260726T065924Z | phase3 | unist/dev | smoke | 6 | 0 | - | - |
| qwen0p5b_phase3_unist198_iter_0009075_unist_dev_vllm_smoke_20260726T065924Z | phase3 | unist/dev | vllm_smoke | 6 | 0 | 0 | 0 |
| qwen0p5b_phase3_unist198_iter_0009075_unist_test_full_20260726T065924Z | phase3 | unist/test | full | 46738 | 47 | 47 | 0 |

## 4. 原论文 CVSS-T Table 1 基线参考

下表严格抄录论文的 EN→ZH | ZH→EN 口径。只有本地结果同样来自 CVSS-T test 时才可直接比较。

| 类别 | 方法 | 参数量 | Speech-BLEU | Text-BLEU | A.PCP | SLC-0.2 | SLC-0.4 | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cascaded | 3-Stage | 2.6B | 25.0200 | 16.6200 | 25.8000 | 17.0800 | 2.8000 | 2.8500 | 0.5600 | 0.5400 | 0.8200 | 0.8800 | 3.7600 | 3.5000 |
| Cascaded | 2-Stage | 2.8B | 26.9400 | 20.8600 | 27.3800 | 22.2000 | 2.8700 | 2.6400 | 0.6700 | 0.5200 | 0.9300 | 0.7000 | 3.7900 | 3.4800 |
| MLLM | GPT-4o | - | 31.6400 | 19.2700 | - | - | 2.6600 | 2.5800 | 0.4700 | 0.3700 | 0.7100 | 0.6100 | 3.4600 | 4.1800 |
| MLLM | Qwen2.5-O | 7B | 7.1000 | 22.6600 | 34.8500 | 24.3900 | 1.9000 | 1.9200 | 0.3100 | 0.3500 | 0.5700 | 0.6100 | 3.2300 | 4.3000 |
| S2ST | Seamless-M | 1.2B | 14.5300 | 14.3600 | 24.8000 | 18.4400 | 2.3400 | 2.2900 | 0.5400 | 0.2200 | 0.8200 | 0.4500 | 2.7300 | 3.5900 |
| S2ST | Seamless-L | 2.3B | 25.0500 | 17.6700 | 27.6100 | 21.9500 | 2.4100 | 2.1500 | 0.6700 | 0.3600 | 0.9500 | 0.6200 | 2.6900 | 4.0400 |
| S2ST | Seamless-Ex | 1.7B | 24.4500 | 15.8400 | 26.5900 | 16.7400 | 2.8300 | 2.8700 | 0.6800 | 0.5200 | 0.9400 | 0.7700 | 2.4600 | 2.9000 |
| S2ST | UniSS (P) | 1.5B | 30.2800 | 23.6100 | 30.9300 | 24.4500 | 2.7300 | 2.7500 | 0.9800 | 0.8400 | 0.9900 | 0.9700 | 3.7700 | 3.8600 |
| S2ST | UniSS (Q) | 1.5B | 32.2000 | 24.2800 | 32.9500 | 26.2800 | 2.7100 | 2.7400 | 0.9800 | 0.8700 | 0.9900 | 0.9700 | 3.7600 | 3.8600 |

### 论文0.5B模型效率参考

论文 Table 3 使用 CVSS-T, 400 utterances (200 per direction)，硬件/推理条件为 single H800, Transformers, no batching，与本地批量vLLM结果也不能直接比较速度。

| 模型 | 参数量 | 平均Speech-BLEU | 时间(s) |
| --- | ---: | ---: | ---: |
| UniSS-Small (Q) | 0.5B | 28.17 | 1339.24 |
| UniSS-Small (P) | 0.5B | 25.68 | 1212.65 |

## 5. 分析原则

- Text-BLEU 与 Speech-BLEU 分开解读：前者评估模型生成的翻译文本，后者还包含语音解码和ASR误差。
- AutoPCP、SLC 和 UTMOS 分别反映韵律、时长一致性和预测音质，任何单项都不能替代试听。
- UniST source/reference WAV 是BiCodec token重建音频，不是原始数据集波形；论文CVSS-T使用真实配对WAV。
- Phase2/Phase3使用相同manifest、seed、Q/P参数和指标模型，内部差值具有可解释性。
- 随机采样生成仍可能有方差；最终checkpoint选择建议结合多seed子集和人工盲听。

## 6. 结果与试听目录

- `qwen0p5b_phase2_unist198_iter_0015381_unist_dev_full_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase2_unist198_iter_0015381_unist_dev_full_20260726T065924Z`
- `qwen0p5b_phase2_unist198_iter_0015381_unist_dev_listen_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase2_unist198_iter_0015381_unist_dev_listen_20260726T065924Z`
- `qwen0p5b_phase2_unist198_iter_0015381_unist_dev_smoke_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase2_unist198_iter_0015381_unist_dev_smoke_20260726T065924Z`
- `qwen0p5b_phase2_unist198_iter_0015381_unist_dev_vllm_smoke_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase2_unist198_iter_0015381_unist_dev_vllm_smoke_20260726T065924Z`
- `qwen0p5b_phase2_unist198_iter_0015381_unist_test_full_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase2_unist198_iter_0015381_unist_test_full_20260726T065924Z`
- `qwen0p5b_phase3_unist198_iter_0009075_unist_dev_full_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase3_unist198_iter_0009075_unist_dev_full_20260726T065924Z`
- `qwen0p5b_phase3_unist198_iter_0009075_unist_dev_listen_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase3_unist198_iter_0009075_unist_dev_listen_20260726T065924Z`
- `qwen0p5b_phase3_unist198_iter_0009075_unist_dev_smoke_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase3_unist198_iter_0009075_unist_dev_smoke_20260726T065924Z`
- `qwen0p5b_phase3_unist198_iter_0009075_unist_dev_vllm_smoke_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase3_unist198_iter_0009075_unist_dev_vllm_smoke_20260726T065924Z`
- `qwen0p5b_phase3_unist198_iter_0009075_unist_test_full_20260726T065924Z`: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/qwen0p5b_phase3_unist198_iter_0009075_unist_test_full_20260726T065924Z`
