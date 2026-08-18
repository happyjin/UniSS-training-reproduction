# Fixed15 exact-runtime aggregate

- Samples: 8
- Directions: `{"cmn->eng": 4, "eng->cmn": 4}`
- First arbitrary PCM is not first useful audio; prefix ASR remains a hard gate.

| group | samples | natural WRITE | all-WAIT | premature first WRITE | playable PCM | collapse | EOS | first WRITE p50/p95 ms | arbitrary PCM wall p50/p95 ms | RTF p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 8 | 0.3750 | 0.6250 | 0.3750 | 0.3750 | 0.6250 | 0.0000 | 320.0/751.9999999999999 | 809.2536518163979/1203.5430769650266 | 0.3813057346017862/1.8576717472475024 |
| cmn->eng | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 240.0/312.0 | 744.2790364241228/802.7561902771704 | 0.6681466229204658/1.352402043034923 |
| eng->cmn | 4 | 0.2500 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | 0.0000 | 800.0/800.0 | 1247.353013092652/1247.353013092652 | 0.3813057346017862/1.8487293589816454 |

## Missing hard-gate evidence

- `first_useful_audio_prefix_asr`
- `target_language_asr_and_speech_bleu`
- `autopcp`
- `slc`
- `speaker_similarity`
- `cached_uncached_parity`
- `fused_unfused_parity`
- `phase3_replay_retention`
