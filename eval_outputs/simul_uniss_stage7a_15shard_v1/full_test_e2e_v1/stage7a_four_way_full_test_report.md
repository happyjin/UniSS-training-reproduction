# Simul-UniSS Stage7A four-way full test report

> Scope: 23,369-sample free-running streaming S2ST test; each experiment uses two fixed H200 GPUs.
> E0/E1/E2/E3 use identical schedules, greedy generation, BiCodec decode, metrics, and batch-one latency audit.

## 1. 实验设计与可归因性

| Experiment | Role | Best step | GPUs | Eval commit | Exported model |
| --- | --- | ---: | --- | --- | --- |
| E0 Stage6 | Stage6 frozen action-policy baseline | 1189 | 0,1 | `3d41b52c33cb` | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/simul_uniss_stage6_streaming_v1_iter_0001189_hf` |
| E1 continued SFT | Matched continued action SFT control | 700 | 2,3 | `4052a7e976b7` | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/simul_uniss_stage7a_15shard_v1/e1_continued_sft_best_hf` |
| E2 GRPO G4 | Action-only GRPO, group size 4 | 600 | 4,5 | `4052a7e976b7` | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/simul_uniss_stage7a_15shard_v1/e2_grpo_g4_best_hf` |
| E3 GRPO G8 | Action-only GRPO, group size 8 | 700 | 6,7 | `4052a7e976b7` | `/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/simul_uniss_stage7a_15shard_v1/e3_grpo_g8_best_hf` |

E0 是原 Stage6 基线；E1 控制额外训练步数和 action-head 继续训练；E2/E3 与 E1 的差异才是 GRPO 与 group size。
四组使用完全相同的 23,369 条 test schedules、greedy decode、BiCodec 和指标实现，因此表内差值是 matched comparison。

## 2. Quality and audio metrics

| Experiment | Text BLEU zh→en | Text BLEU en→zh | Speech BLEU zh→en | Speech BLEU en→zh | UTMOS zh→en | UTMOS en→zh | AutoPCP zh→en | AutoPCP en→zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 Stage6 | 26.378 | 40.560 | 1.155 | 37.251 | 3.558 | 3.362 | 2.521 | 3.134 |
| E1 continued SFT | 26.240 | 40.480 | 1.164 | 37.187 | 3.560 | 3.361 | 2.521 | 3.129 |
| E2 GRPO G4 | 27.587 | 40.224 | 1.127 | 36.990 | 3.556 | 3.361 | 2.519 | 3.132 |
| E3 GRPO G8 | 28.109 | 40.355 | 1.215 | 37.010 | 3.561 | 3.362 | 2.523 | 3.128 |

Text BLEU 衡量文本翻译；Speech BLEU 在实际生成音频经 ASR 后衡量端到端内容；UTMOS/AutoPCP 分别反映感知音质和与参考音频的表示相似度。

## 3. Streaming policy and latency

| Experiment | First WRITE ms | StartOffset NCA ms | ATD ms | LAAL token proxy | WRITE F1 | Premature WRITE | Unnecessary WAIT | Final flush | Source RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 Stage6 | 3986.4 | 3986.5 | 1807.3 | 44.58 | 0.862 | 0.031 | 0.158 | 1.000 | 2.176 |
| E1 continued SFT | 3991.0 | 3991.1 | 1808.4 | 44.54 | 0.862 | 0.030 | 0.157 | 1.000 | 2.182 |
| E2 GRPO G4 | 4055.4 | 4055.2 | 1839.0 | 45.54 | 0.860 | 0.027 | 0.164 | 1.000 | 2.171 |
| E3 GRPO G8 | 4032.8 | 4032.8 | 1827.5 | 45.10 | 0.860 | 0.029 | 0.162 | 1.000 | 2.041 |

带 `_proxy` 的指标来自当前 pseudo-alignment/capacity gate，不能表述为真实 CTC 对齐指标。First WRITE、ATD、LAAL、unnecessary WAIT 越低越好；Final flush 应为 1。

## 4. Batch-one deployable latency

| Experiment | Action TTFT s | WRITE TTFT s | Source RTF | First WRITE ms | ATD ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 Stage6 | 0.015 | 0.015 | 0.165 | 4793.6 | 2178.1 |
| E1 continued SFT | 0.015 | 0.015 | 0.167 | 4819.2 | 2185.8 |
| E2 GRPO G4 | 0.014 | 0.015 | 0.162 | 4883.2 | 2216.1 |
| E3 GRPO G8 | 0.015 | 0.016 | 0.167 | 4857.6 | 2205.1 |

batch-one 审计固定为每组 200 条，用于估计真实部署延迟；大 batch 的全量评估吞吐不能替代这里的 RTF/TTFT。

## 5. GPU utilization and power

| Experiment | Util mean | Util p95 | Power mean | Power p95 |
| --- | ---: | ---: | ---: | ---: |
| E0 Stage6 | 99.3% | 100.0% | 384.7 W | 411.3 W |
| E1 continued SFT | 98.8% | 100.0% | 386.4 W | 402.0 W |
| E2 GRPO G4 | 98.9% | 100.0% | 409.1 W | 441.3 W |
| E3 GRPO G8 | 98.7% | 100.0% | 386.8 W | 401.9 W |

GPU utilization 是吞吐诊断而非模型质量分数。mean 覆盖模型加载、CPU 聚合和阶段切换，p95 更接近 GPU-heavy 稳态。没有使用 dummy computation 或无效 padding 抬高功率。

## 6. GRPO 相对 matched E1 的直接差值

| Metric | Better | E2−E1 | E3−E1 |
| --- | :---: | ---: | ---: |
| Text BLEU zh→en | ↑ | +1.348 | +1.869 |
| Text BLEU en→zh | ↑ | -0.257 | -0.126 |
| Speech BLEU zh→en | ↑ | -0.037 | +0.050 |
| Speech BLEU en→zh | ↑ | -0.197 | -0.176 |
| UTMOS mean | ↑ | -0.002 | +0.001 |
| AutoPCP mean | ↑ | +0.000 | +0.000 |
| First WRITE ms | ↓ | +64.389 | +41.849 |
| ATD ms | ↓ | +30.634 | +19.143 |
| LAAL token proxy | ↓ | +1.000 | +0.564 |
| Premature WRITE | ↓ | -0.003 | -0.002 |
| Unnecessary WAIT | ↓ | +0.007 | +0.005 |
| Final flush | ↑ | +0.000 | +0.000 |
| Batch-one source RTF | ↓ | -0.005 | -0.001 |

这里必须优先比较 E2/E3 与 E1，而不是只比较 E0；如果 E1 与 GRPO 同样改善，收益可能只是继续训练，而不是 GRPO。

## 7. 相对 Stage6 的首轮门槛审计

| Experiment | First WRITE | StartOffset | ATD | LAAL | Text BLEU retention | Premature Δ | Unnecessary WAIT | RTF<1 | Final flush |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 continued SFT | FAIL (-0.1%) | FAIL (-0.1%) | FAIL (-0.1%) | FAIL (+0.1%) | PASS (worst -0.138) | PASS (-0.001) | FAIL | PASS | PASS |
| E2 GRPO G4 | FAIL (-1.7%) | FAIL (-1.7%) | FAIL (-1.8%) | FAIL (-2.1%) | PASS (worst -0.336) | PASS (-0.004) | FAIL | PASS | PASS |
| E3 GRPO G8 | FAIL (-1.2%) | FAIL (-1.2%) | FAIL (-1.1%) | FAIL (-1.2%) | PASS (worst -0.205) | PASS (-0.002) | FAIL | PASS | PASS |

门槛来自 Stage7A plan：first-WRITE 至少 -15% 或 -500 ms，StartOffset/ATD/LAAL 至少 -10%，双向 Text BLEU 最差下降不超过 0.5，premature 增加不超过 0.01，unnecessary WAIT ≤0.12，batch-one RTF<1，final flush≈100%。

## 8. 与 offline Phase3 quality 模式的同指标差值

| Streaming experiment | Direction | Metric | Streaming | Offline Phase3 | Streaming−offline |
| --- | --- | --- | ---: | ---: | ---: |
| E0 Stage6 | cmn->eng | autopcp | 2.521 | 2.885 | -0.364 |
| E0 Stage6 | eng->cmn | autopcp | 3.134 | 3.275 | -0.141 |
| E0 Stage6 | cmn->eng | speech_bleu | 1.155 | 1.750 | -0.595 |
| E0 Stage6 | eng->cmn | speech_bleu | 37.251 | 46.306 | -9.055 |
| E0 Stage6 | cmn->eng | text_bleu | 26.378 | 39.375 | -12.998 |
| E0 Stage6 | eng->cmn | text_bleu | 40.560 | 48.170 | -7.610 |
| E0 Stage6 | cmn->eng | utmos | 3.558 | 3.668 | -0.110 |
| E0 Stage6 | eng->cmn | utmos | 3.362 | 3.359 | 0.003 |
| E1 continued SFT | cmn->eng | autopcp | 2.521 | 2.885 | -0.364 |
| E1 continued SFT | eng->cmn | autopcp | 3.129 | 3.275 | -0.146 |
| E1 continued SFT | cmn->eng | speech_bleu | 1.164 | 1.750 | -0.585 |
| E1 continued SFT | eng->cmn | speech_bleu | 37.187 | 46.306 | -9.120 |
| E1 continued SFT | cmn->eng | text_bleu | 26.240 | 39.375 | -13.136 |
| E1 continued SFT | eng->cmn | text_bleu | 40.480 | 48.170 | -7.689 |
| E1 continued SFT | cmn->eng | utmos | 3.560 | 3.668 | -0.108 |
| E1 continued SFT | eng->cmn | utmos | 3.361 | 3.359 | 0.002 |
| E2 GRPO G4 | cmn->eng | autopcp | 2.519 | 2.885 | -0.366 |
| E2 GRPO G4 | eng->cmn | autopcp | 3.132 | 3.275 | -0.143 |
| E2 GRPO G4 | cmn->eng | speech_bleu | 1.127 | 1.750 | -0.623 |
| E2 GRPO G4 | eng->cmn | speech_bleu | 36.990 | 46.306 | -9.317 |
| E2 GRPO G4 | cmn->eng | text_bleu | 27.587 | 39.375 | -11.788 |
| E2 GRPO G4 | eng->cmn | text_bleu | 40.224 | 48.170 | -7.946 |
| E2 GRPO G4 | cmn->eng | utmos | 3.556 | 3.668 | -0.112 |
| E2 GRPO G4 | eng->cmn | utmos | 3.361 | 3.359 | 0.002 |
| E3 GRPO G8 | cmn->eng | autopcp | 2.523 | 2.885 | -0.361 |
| E3 GRPO G8 | eng->cmn | autopcp | 3.128 | 3.275 | -0.148 |
| E3 GRPO G8 | cmn->eng | speech_bleu | 1.215 | 1.750 | -0.535 |
| E3 GRPO G8 | eng->cmn | speech_bleu | 37.010 | 46.306 | -9.296 |
| E3 GRPO G8 | cmn->eng | text_bleu | 28.109 | 39.375 | -11.267 |
| E3 GRPO G8 | eng->cmn | text_bleu | 40.355 | 48.170 | -7.815 |
| E3 GRPO G8 | cmn->eng | utmos | 3.561 | 3.668 | -0.107 |
| E3 GRPO G8 | eng->cmn | utmos | 3.362 | 3.359 | 0.003 |

该表说明 streaming 相对 offline upper bound 的质量代价；它不能用于 E2/E3 的训练因果判断，但可以判断 simultaneous 延迟收益是否以不可接受的离线质量损失换取。

## 9. 结论与下一步

- E1 是必须超过的 matched-training control；只有 E2/E3 在质量不下降时显著降低延迟，才能支持 GRPO 独立贡献。
- 以下 delta 均为相对 E1；负的延迟 delta 更好，正的 BLEU delta 更好。
- E2 GRPO G4 vs E1: first-WRITE +64.389 ms; ATD +30.634 ms; cmn->eng Text BLEU +1.348; eng->cmn Text BLEU -0.257.
- E3 GRPO G8 vs E1: first-WRITE +41.849 ms; ATD +19.143 ms; cmn->eng Text BLEU +1.869; eng->cmn Text BLEU -0.126.
- 自动判定：本次单种子 full-test 没有 GRPO 实验同时满足：相对 E1 保持双向 Text BLEU、不恶化 premature/final-flush，并同时降低 first-WRITE 与 ATD。当前结果不支持扩大到 full198。
- 正式进入 full198 仍需补齐 fixed wait-k frontier、10,000 次 paired bootstrap、至少 3 个随机种子；本次只有一个训练种子，因此只能给出候选排序，不能给出统计显著性结论。

最终判断必须联合 Text/Speech BLEU、UTMOS/AutoPCP、first-WRITE、ATD/LAAL、premature WRITE、unnecessary WAIT、final flush 和 batch-one RTF。
如果 GRPO 没有超过 E1 的 quality-latency Pareto 点，不应扩大该 reward 到 full198；应先修改 rollout conditioning、真实 alignment 和 reward。

## 10. 结论边界

- 这是 full test 集上的单种子、单 operating-point 比较；没有执行 3-seed mean±std。
- 本轮没有生成 fixed wait-k=1/2/3/5 的完整 test frontier，因此不能声称已超过 fixed wait-k Pareto frontier。
- 本轮保存逐样本输出，但没有在此自动报告中执行 10,000 次 paired bootstrap；数值差异不能直接等同统计显著。
- 当前是 frozen Stage6 backbone 上的 action-only GRPO；它没有训练 text/semantic/BiCodec 权重，不能称为 full-Qwen 或 semantic-token GRPO。
- 本轮报告 Text BLEU、Speech BLEU、UTMOS、AutoPCP 和 streaming 指标；chrF/COMET/ASR-COMET 不在当前可复现链路中，不用缺失指标替代或伪造结论。

## 11. Reproducibility

- Machine-readable comparison: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/simul_uniss_stage7a_15shard_v1/full_test_e2e_v1/comparison.json`
- E0 Stage6: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/simul_uniss_stage7a_15shard_v1/full_test_e2e_v1/e0_stage6/full_test_v1`
- E1 continued SFT: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/simul_uniss_stage7a_15shard_v1/full_test_e2e_v1/e1_continued_sft/full_test_v1`
- E2 GRPO G4: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/simul_uniss_stage7a_15shard_v1/full_test_e2e_v1/e2_grpo_g4/full_test_v1`
- E3 GRPO G8: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/simul_uniss_stage7a_15shard_v1/full_test_e2e_v1/e3_grpo_g8/full_test_v1`
