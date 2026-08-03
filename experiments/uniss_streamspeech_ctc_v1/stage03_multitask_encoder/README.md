# Stage03: causal encoder + endpoint CTC multi-task training

Stage02 proved that the frozen 1280-dimensional causal pre-VQ representation is
not linearly separable for either ASR or NAR speech-to-text translation.  This
stage follows the pre-declared fallback instead of passing a failed probe to the
B2 bridge.

The model starts from the independently trained 121M-parameter Stage-B-v3 causal
Emformer checkpoint.  Historical heads are not reused.  The streaming acoustic
frontend and Emformer are initialized from that checkpoint and jointly trained
with four new 8k CTC heads:

- English and Chinese source ASR
- English and Chinese NAR-S2TT

Architecture properties retained from the causal student:

- 40 ms encoder frame rate
- 160 ms Emformer segment
- 80 ms right context
- explicit streaming cache support in the historical inference implementation

The old checkpoint, old training code, Qwen, BiCodec and Phase1--3 are read-only.

## Eight-GPU training

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage03_multitask_encoder/run_8gpu.sh
```

The encoder learning rate is 10x smaller than the new-head learning rate.  The
StreamSpeech loss weights are retained: ASR CTC `4.0`, NAR-S2TT CTC `4.0`.

