# Stage08 Step2 Qwen-LoRA + replay smoke

Date: 2026-08-04 UTC

Status: **PASS for research-only pipeline validation**. Step1-R did not pass
the formal prerequisite gate, so this smoke does not establish model quality.

The isolated Megatron job completed two iterations on eight H200 GPUs with:

- zero skipped iterations and zero NaN iterations;
- rank-8 `q_proj`/`v_proj` LoRA as the only trainable Qwen parameters;
- balanced 50:50 EN→ZH and ZH→EN sampling;
- a 70:30 streaming/offline replay loss mixture;
- a valid iteration-2 torch distributed checkpoint and TensorBoard events.

| Metric | Iteration 1 | Iteration 2 | Validation at iteration 2 |
|---|---:|---:|---:|
| Streaming Phase3 NLL | 4.1493 | 4.1251 | 4.1134 |
| Offline replay NLL | 4.1335 | 4.1100 | 4.0991 |
| Gradient norm | 0.106 | 0.105 | — |
| LoRA B RMS | 0 at pre-update forward | 0 at pre-update forward | 3.6123e-5 |

Selective LoRA checkpoint loading and the eight-process probe path were also
executed successfully. The probe smoke used only four examples per direction,
so its BLEU is intentionally not used as a quality conclusion. The next valid
comparison must use the fixed 32-row probe.

Runtime artifacts:

```text
checkpoints/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_smoke2_v1/iter_0000002
runs/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_smoke2_v1
logs/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_smoke2_v1.log
reports/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_probe_smoke8_v2
```
