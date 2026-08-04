# Stage09: unified true-chunk online runtime

Stage09 joins three previously disconnected, immutable experiment lines:

```text
Step1-R updated Emformer + CTC heads
  -> Stage05 StreamSpeech CTC-count READ/WRITE policy
  -> Stage04/B1 Phase3 speech embeddings
  -> Step2 iteration-100 Qwen LoRA model provenance
```

Raw audio arrives incrementally. The mel frontend uses `center=False`; only
complete stacked frames are consumed before final flush. Each Emformer call
receives a 160 ms segment plus the configured 80 ms right context and carries
the real `Emformer.infer` state forward. Already committed CTC target tokens are
never revised.

This is faithful to StreamSpeech at the policy level: shared causal encoder,
source ASR CTC, target NAR-S2TT CTC, and stable CTC-count READ/WRITE. It is not a
Fairseq copy: UniSS retains Qwen Phase3, the B1 bridge and BiCodec.

The upstream fixed BLEU gate remains unmet, so every output is research-only.

Run the isolated one-sample GPU smoke:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/uniss_streamspeech_ctc_v1/stage09_online_runtime/run_smoke.sh
```

## Executed bilingual smoke

| Direction | Chunks | WRITEs | First WRITE | Compute RTF | Conflicts |
|---|---:|---:|---:|---:|---:|
| EN→ZH | 38 | 10 | 560 ms | 0.1868 | 0 / 0 |
| ZH→EN | 67 | 3 | 2160 ms | 0.1655 | 0 / 0 |

The result proves the unified online state path and also reproduces the known
direction asymmetry: ZH→EN waits longer and commits less useful text. Stage10
uses these policy events to trigger cached Qwen generation; it does not treat
the CTC text itself as the final translation.
