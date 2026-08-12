# Generalize15 action/EOS calibration canary

Generalize14 achieves natural sub-second WRITE and PCM but emits too many
false-positive WRITEs and chooses EOS immediately at source end.  Those two
policy errors corrupt the persistent text/semantic history, leave the target
unfinished and push RTF above one.

This isolated Megatron stage starts from Generalize14 iteration 50.  It freezes
the Phase3 base, Qwen LoRA, frontend, text path and semantic microblock head,
then trains only the existing action head and a new learned START_GLM-versus-EOS
continuation head.  WAIT examples receive double action mass, the grouped
first-WRITE deadline remains active, and model-prefix roll-in is capped at one
10% pass after a clean warm-up.

```bash
bash experiments/uniss_phase3_runtime_parity_streaming_v2/generalize15_action_eos_calibration/prepare_data.sh
bash experiments/uniss_phase3_runtime_parity_streaming_v2/generalize15_action_eos_calibration/run_8gpu.sh
```

TensorBoard uses port `6088`.  Promotion still requires natural sub-second
WRITE/PCM, correct translation, non-collapsed playable PCM, natural EOS, zero
revision and RTF below one on both train and held-out records.

