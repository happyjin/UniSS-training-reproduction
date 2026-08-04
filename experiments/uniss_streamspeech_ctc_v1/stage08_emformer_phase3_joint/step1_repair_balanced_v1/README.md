# Stage08 Step1-R: balanced Phase3-gradient repair

This experiment is isolated from the completed Step1 run. It starts a fresh
optimizer from Step1 iteration 800 and changes only the diagnosed failure
modes:

- virtual 50:50 EN→ZH / ZH→EN sampling without copying audio;
- frozen-Phase3 NLL weight 0.5 -> 2.0;
- ZH→EN sample weight 1.0 -> 1.25;
- per-direction CTC, AR and Phase3 TensorBoard metrics;
- 400 iterations at 2e-6 -> 2e-7 with a 40-step warmup.

Qwen, BiCodec and the Stage04 bridge remain frozen. The last four Emformer
layers, CTC heads, AR decoder and B1 residual remain the only trainable parts.
The original Step1 checkpoints, logs, TensorBoard events and reports are never
overwritten.

## Prepare indices

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python \
  experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step1_repair_balanced_v1/build_direction_indices.py \
  --dataset-index data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json \
  --output-dir data/processed/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_v1
```

## Train and evaluate

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step1_repair_balanced_v1/run_megatron_8gpu.sh
bash experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step1_repair_balanced_v1/run_gate_8gpu.sh
```
