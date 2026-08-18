# Fixed15 exact-runtime aggregate

- Samples: 8
- Directions: `{"cmn->eng": 4, "eng->cmn": 4}`
- First arbitrary PCM is not first useful audio; prefix ASR remains a hard gate.

| group | samples | natural WRITE | all-WAIT | premature first WRITE | playable PCM | collapse | EOS | first WRITE p50/p95 ms | arbitrary PCM wall p50/p95 ms | RTF p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 8 | 0.2500 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | 0.0000 | 160.0/160.0 | 719.8995335167274/807.4651654134504 | 0.23313028464288288/0.8030042697817872 |
| cmn->eng | 4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | None/None | None/None | 0.16346451633408787/0.26450656662058925 |
| eng->cmn | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 160.0/160.0 | 719.8995335167274/807.4651654134504 | 0.5502287346735675/0.8393495313606861 |

## Missing hard-gate evidence

- `first_useful_audio_prefix_asr`
- `target_language_asr_and_speech_bleu`
- `autopcp`
- `slc`
- `speaker_similarity`
- `cached_uncached_parity`
- `fused_unfused_parity`
- `phase3_replay_retention`
