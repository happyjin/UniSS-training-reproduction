# Fixed15 exact-runtime aggregate

- Samples: 8
- Directions: `{"cmn->eng": 4, "eng->cmn": 4}`
- First arbitrary PCM is not first useful audio; prefix ASR remains a hard gate.

| group | samples | natural WRITE | all-WAIT | premature first WRITE | playable PCM | collapse | EOS | first WRITE p50/p95 ms | arbitrary PCM wall p50/p95 ms | RTF p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 8 | 0.3750 | 0.6250 | 0.3750 | 0.3750 | 0.6250 | 0.0000 | 160.0/304.0 | 648.6370690073818/712.7377986870706 | 0.576749606732539/1.3562458746843813 |
| cmn->eng | 4 | 0.7500 | 0.2500 | 0.7500 | 0.7500 | 0.2500 | 0.0000 | 160.0/304.0 | 648.6370690073818/712.7377986870706 | 1.0555236969124842/1.3950524440435914 |
| eng->cmn | 4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | None/None | None/None | 0.4130940449928332/0.6500553230935201 |

## Missing hard-gate evidence

- `first_useful_audio_prefix_asr`
- `target_language_asr_and_speech_bleu`
- `autopcp`
- `slc`
- `speaker_similarity`
- `cached_uncached_parity`
- `fused_unfused_parity`
- `phase3_replay_retention`
