# Fixed15 exact-runtime aggregate

- Samples: 8
- Directions: `{"cmn->eng": 4, "eng->cmn": 4}`
- First arbitrary PCM is not first useful audio; prefix ASR remains a hard gate.

| group | samples | natural WRITE | all-WAIT | premature first WRITE | playable PCM | collapse | EOS | first WRITE p50/p95 ms | arbitrary PCM wall p50/p95 ms | RTF p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 8 | 0.2500 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | 0.0000 | 240.0/312.0 | 670.7830213801935/727.7646111096255 | 0.4222861713758973/1.2382382296984105 |
| cmn->eng | 4 | 0.2500 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | 0.0000 | 320.0/320.0 | 734.0958988573402/734.0958988573402 | 0.40726471271807685/0.8481838672491341 |
| eng->cmn | 4 | 0.2500 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | 0.0000 | 160.0/160.0 | 607.4701439030468/607.4701439030468 | 0.493143526617717/1.3033061639258086 |

## Missing hard-gate evidence

- `first_useful_audio_prefix_asr`
- `target_language_asr_and_speech_bleu`
- `autopcp`
- `slc`
- `speaker_similarity`
- `cached_uncached_parity`
- `fused_unfused_parity`
- `phase3_replay_retention`
