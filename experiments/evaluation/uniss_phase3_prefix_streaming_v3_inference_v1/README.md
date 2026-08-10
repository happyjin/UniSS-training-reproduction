# Full198 Phase3 prefix-streaming v3 inference

This directory is independent of all historical training and evaluation paths.
It selects only physically saved checkpoints, exports only the trained LoRA
tensors, and never mutates the Phase3 HF base or Megatron checkpoint.

The runtime is explicitly **source-side prefix/pseudo-streaming**: audio is
revealed in 320/480/640 ms increments and the WhisperVQ/GLM frontend is
cumulatively re-encoded.  It is not a causal Whisper encoder.

Checkpoint selection uses an equal rank-sum across prefix CE, streaming TTS
semantic CE, commit CE, teacher KL, adjacent-prefix consistency and WAIT/WRITE
action CE.  Among saved 500-iteration checkpoints, `iter_0008000` is best.

## Real-audio evaluation

The engine follows the three separately trained heads rather than reusing the
older Stage4 one-shot WRITE format:

```text
cumulative audio prefix -> stable GLM prefix -> WAIT/WRITE
  -> streaming S2TT hypothesis -> stable text commit
  -> streaming TTS continuation blocks -> incremental BiCodec
```

Run all listening granularities without touching historical outputs:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python \
  -m experiments.evaluation.uniss_phase3_prefix_streaming_v3_inference_v1.evaluate_audio \
  --audio /path/to/source.wav --direction zh-en --chunk-ms all
```

Each request writes `source_16k.wav`, continuous target audio, a target timeline
with actual WAIT silence, `stereo_left_source_right_translation.wav`, and a
complete event/latency `result.json` below:

```text
eval_outputs/uniss_phase3_prefix_streaming_v3_iter8000_v1/chunk_{320,480,640}ms/
```
