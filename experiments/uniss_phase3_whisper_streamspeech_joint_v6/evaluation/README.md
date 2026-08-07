# V6 fixed-chunk evaluation

This directory evaluates immutable Stage A and Stage B checkpoints at five
fixed Whisper operating points: `320`, `640`, `960`, `1280` ms and `offline`.
Each run uses 8 GPUs and 1024 balanced validation samples.  Runs are sequential
because every operating point uses all eight GPUs.

The matrix refuses to overwrite an existing log or report namespace:

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v6/evaluation/run_fixed_chunk_matrix.sh
```

The result is written below a unique `RUN_ID` in:

```text
reports/uniss_phase3_whisper_streamspeech_joint_v6/
```

These metrics are a fixed-condition loss gate.  They do not replace end-to-end
BLEU, generated-audio quality, speaker preservation, or wall-clock latency.
