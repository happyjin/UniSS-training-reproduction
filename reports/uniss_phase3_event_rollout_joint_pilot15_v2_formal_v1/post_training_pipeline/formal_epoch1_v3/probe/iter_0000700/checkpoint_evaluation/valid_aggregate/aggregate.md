# Fixed15 exact-runtime aggregate

- Samples: 8
- Directions: `{"cmn->eng": 4, "eng->cmn": 4}`
- First arbitrary PCM is not first useful audio; prefix ASR remains a hard gate.

| group | samples | natural WRITE | all-WAIT | premature first WRITE | playable PCM | collapse | EOS | first WRITE p50/p95 ms | arbitrary PCM wall p50/p95 ms | RTF p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 8 | 0.3750 | 0.6250 | 0.3750 | 0.3750 | 0.6250 | 0.0000 | 160.0/160.0 | 570.5665718670934/694.374372800812 | 0.4292660824768779/0.8234993145569086 |
| cmn->eng | 4 | 0.2500 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | 0.0000 | 160.0/160.0 | 708.1307951267809/708.1307951267809 | 0.2084883869481035/0.5958707583200129 |
| eng->cmn | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 160.0/160.0 | 565.3482249146327/570.0447371718474 | 0.6760994519749608/0.8589136879146024 |

## Missing hard-gate evidence

- `first_useful_audio_prefix_asr`
- `target_language_asr_and_speech_bleu`
- `autopcp`
- `slc`
- `speaker_similarity`
- `cached_uncached_parity`
- `fused_unfused_parity`
- `phase3_replay_retention`
