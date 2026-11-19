# 文档索引

## 入口与维护

- [四条推理链路：Omni offline/streaming、S4 offline、S5 streaming](INFERENCE_ENTRYPOINTS.md)
- [仓库整理记录与保留依据](REPOSITORY_MAINTENANCE.md)
- [Streaming SI 训练管线与代码结构](../pipelines/streaming_si/README.md)
- [SimulS2ST-Omni 训练管线](../pipelines/simuls2st_omni/README.md)
- [qwen_omnispeak 推理 CLI](../inference/README.md)

## 推理跑测

最终结果以对应的四份系统报告为准；过程文档保留失败诊断、对照条件和复现记录。

- [Omni offline](INFER_BENCH_REPRO_OMNI_OFFLINE.md)
- [Omni streaming L1～L4 与优化结果](INFER_BENCH_REPRO_OMNI_STREAMING.md)
- [S4 offline](INFER_BENCH_S4_OFFLINE.md)
- [S5 streaming](INFER_BENCH_S5_STREAMING.md)
- [S5 流式推理验证](S5_STREAMING_VERIFICATION.md)
- [S5 日韩同传 50 句同样本对比](S5_日韩50句同样本对比.md)
- [S5 全量对比与日韩 1596 句首包结果](S5_NO30_FULL_COMPARISON_REPORT.md)
- [FAST_TALKER / 共享模型过程记录](INFER_BENCH_OMNI_FAST_TALKER_RUN.md)
- [静态 cache / CUDA Graph 过程记录](INFER_BENCH_OMNI_STATIC_GRAPH_PROBE.md)
- [长语音 KV 回退与显存边界探针](INFER_BENCH_OMNI_KV_BOUNDARY.md)

## 训练性能与实验

- [训练指标合并报告](TRAINING_METRICS_MERGED.md)
- [SI 优化总体进展](SI_OPTIMIZATION_OVERALL_PROGRESS.md)
- [三模型优化计划](THREE_MODEL_OPTIMIZATION_PLAN.md)
- [SI S1：GBS4096、Liger、ZeRO-1](SI_S1_GBS4096_BASE_LIGER_ZERO1_200STEP_TRAINING_METRICS.md)
- [SI S2：Base / Fused CE](SI_S2_GBS4096_BASE_VS_FUSEDCE_200STEP_TRAINING_METRICS.md)
- [SI S3：Base / Liger](SI_S3_GBS4096_BASE_VS_LIGER_TRAINING_METRICS.md)
- [GBS512 / GBS4096 ABCD 实验](GBS512_GBS4096_BASELINE_ABCD_200STEP_TRAINING_METRICS.md)
- [Lance / WDS 四组对照](LANCE_WDS_GBS4096_ZERO1_4WAY_200STEP_TRAINING_METRICS.md)
- [CE 融合与 ZeRO-0 / ZeRO-1 的 NSYS 对照](CE_FUSION_ZERO0_VS_ZERO1_NSYS_COMPARISON.md)
- [AO 复现记录](AO_REPRODUCTION_LOG.md)

## 架构、数据与环境

- [架构](01_architecture.md)
- [Reply-add 数据流](02_reply_add_dataflow.md)
- [DataLoader](03_dataloader.md)
- [环境与训练](04_env_and_training.md)
- [WDS / EOS / History 配方](05_wds_eos_history_recipe.md)
- [Megatron / DeepSpeed 后端一致性](06_megatron_deepspeed_backend_consistency.md)
- [变更记录](CHANGELOG.md)
