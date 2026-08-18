# Fixed15 exact-runtime aggregate

- Samples: 8
- Directions: `{"cmn->eng": 4, "eng->cmn": 4}`
- First arbitrary PCM is not first useful audio; prefix ASR remains a hard gate.

| group | samples | natural WRITE | all-WAIT | premature first WRITE | playable PCM | collapse | EOS | first WRITE p50/p95 ms | arbitrary PCM wall p50/p95 ms | RTF p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 8 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 160.0/295.99999999999994 | 614.4543290231377/720.0382663668133 | 0.4940451988684041/1.1728409800723711 |
| cmn->eng | 4 | 0.2500 | 0.7500 | 0.2500 | 0.2500 | 0.7500 | 0.0000 | 160.0/160.0 | 582.6075150072575/582.6075150072575 | 0.22862323854984792/0.7386956439586114 |
| eng->cmn | 4 | 0.7500 | 0.2500 | 0.7500 | 0.7500 | 0.2500 | 0.0000 | 160.0/304.0 | 646.3011430390179/724.3757442096248 | 0.7615836222763287/1.2713841180869623 |

## Missing hard-gate evidence

- `first_useful_audio_prefix_asr`
- `target_language_asr_and_speech_bleu`
- `autopcp`
- `slc`
- `speaker_similarity`
- `cached_uncached_parity`
- `fused_unfused_parity`
- `phase3_replay_retention`
