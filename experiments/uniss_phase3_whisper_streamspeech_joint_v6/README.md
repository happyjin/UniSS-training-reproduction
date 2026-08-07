# Phase3 Whisper StreamSpeech Joint V6

V6 is an isolated repair of the delayed semantic-space drift observed in V5.
It never writes into V1/V4/V5 logs, checkpoints, TensorBoard runs, or scripts.

The pilot uses the same immutable first-15-shard manifests as V5 and runs in
two ordered stages:

1. `stage_a_heads_only`: freeze WhisperVQ and Qwen, warm up the randomly
   initialized CTC/unit heads, and measure teacher GLM agreement without moving
   the Phase3 semantic frontend.
2. `stage_b_guarded_joint`: load the Stage A model weights, unfreeze only the
   last pre-VQ Whisper layer, restore the historical Whisper quantization loss,
   supervise the exact stored `source_glm` teacher codes, scale the STE gradient
   to 0.1, and use relative plus absolute commitment guards.

TensorBoard uses port `6033` by default.

Validation balancing is explicit: the bilingual 15-shard runs enable it,
whereas the intentionally single-direction smoke manifest disables it.  This
keeps the production validation contract without making the smoke test require
data that it does not contain.

The validated smoke sequence is:

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/run_stage_a_smoke.sh
bash experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/run_stage_b_smoke.sh
```

Stage B loads the Stage A checkpoint as model-only finetuning state and starts
its own iteration counter at zero; optimizer and RNG state are deliberately not
inherited.
