# Stage02: frozen causal-encoder CTC probe

This stage answers the cheapest decisive question in the plan: do the existing
chunk-causal 1280-dimensional pre-VQ representations contain enough information
for StreamSpeech-style ASR and NAR-S2TT alignments?

Only four linear heads are trained:

- `asr_eng`, `asr_cmn`
- `nar_s2tt_eng`, `nar_s2tt_cmn`

The causal encoder, Qwen and BiCodec remain frozen.  The source latent sidecar is
read in place and never changed.

## Prepare the probe join manifest

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage02_ctc_probe/run_prepare_parallel.sh 16
```

Preparation first creates a versioned SQLite `ID -> current byte offset` lookup.
This is necessary because the historical latent sidecar predates a source
manifest compaction: both its saved byte offsets and, after early records, its
positional indices are stale.  The stable utterance ID is verified on every
join, so a text/latent mismatch cannot be silently trained.

## Train on all eight GPUs

The launcher writes to isolated checkpoint/TensorBoard directories.  Stop the
synthetic `uniss_gpu_load_60` tmux job before launching it.

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage02_ctc_probe/run_8gpu.sh
```

Primary feasibility gates:

- English ASR WER <= 40%
- Chinese ASR CER <= 40%
- NAR-S2TT unigram recall >= 40% in both directions

These are endpoint-task gates; historical WhisperVQ token agreement is not used.
