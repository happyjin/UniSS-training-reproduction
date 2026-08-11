# Generalize13 joint runtime canary

This experiment is the direct response to the strict Generalize12 failure.
It initializes from the completed v12 canary checkpoint, preserves the causal
four-unit semantic microblock decoder, and jointly trains the Qwen LoRA,
natural action/support/safe-commit heads and semantic microblock head.

The Phase3 base parameters, embeddings/output matrix and cached causal
WhisperVQ frontend remain frozen.  Phase3 replay remains active.  Promotion to
full15 requires strict real-PCM success on both seen-canary and held-out
records; training loss alone is not a gate.

```bash
bash experiments/uniss_phase3_runtime_parity_streaming_v2/generalize13_joint_runtime/prepare_data.sh
bash experiments/uniss_phase3_runtime_parity_streaming_v2/generalize13_joint_runtime/run_8gpu.sh
```

TensorBoard uses port `6086`.

The first `canary_v1` launch was retained as an OOM diagnostic: it built two
full-vocabulary CE graphs.  `canary_v2` is the non-overwriting corrected run
and shares one CE tensor across text, semantic and boundary objectives.
