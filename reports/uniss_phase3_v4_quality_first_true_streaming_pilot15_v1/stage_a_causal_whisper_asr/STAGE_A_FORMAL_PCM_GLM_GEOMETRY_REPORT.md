# Stage A formal PCM/GLM geometry gate

## Outcome

The formal Stage A train and validation schedules pass the real-PCM geometry
gate. The audit opens every audio file that the deterministic three-coverage
training schedule selects; it does not infer geometry from rounded duration
metadata alone.

| Split | Packs | Scheduled acoustics | Exact | Terminal +1 at exact hop | Terminal +1 one Whisper frame short | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| train | 16,195 | 97,164 | 75,020 | 22,100 | 44 | 0 |
| valid | 167 | 334 | 251 | 83 | 0 | 0 |

All files are 16 kHz and agree with their sidecar duration within the existing
20 ms dataset tolerance.

## Failure diagnosis

The failed run `stage_a_formal8_20260816T215200Z` reached rank 2 microbatch 14
and stopped on sample `HQ-Conversations_0000007020`:

- reconstructed PCM: 228,800 samples = 14.300 s;
- causal WhisperVQ coverage: 179 tokens;
- released offline GLM coverage: 180 tokens;
- the PCM ends 320 samples, exactly one 20 ms Whisper frame, before the
  nominal end of causal token 179.

The released GLM tokens and the BiCodec-reconstructed PCM are two codec views
of the same utterance. Their only formal-schedule discrepancies are a single
terminal GLM slot with a PCM deficit of exactly 0 or 320 samples. No `+2`,
negative, interior, or wider mismatch appears.

## Authorized repair

Training may repeat the final already-visible causal hidden state exactly once
only when all of the following hold:

1. the causal token count equals `ceil(real_pcm_samples / 1280)`;
2. packed offline GLM coverage is exactly one token longer;
3. the final causal boundary deficit is exactly 0 or 320 samples.

This repair cannot expose future audio. Any other geometry remains a hard
error and now reports the sample ID, audio path, waveform length, and 80 ms
remainder.

## Evidence

- Train audit: `formal_pcm_glm_geometry_20260816T223200Z/train_geometry_audit.json`
- Validation audit: `formal_pcm_glm_geometry_20260816T223200Z/valid_geometry_audit.json`
- Full Stage A CPU suite after repair: `47 passed`
- Focused objective/dataset suite: `7 passed`

The fresh formal launcher and strict-resume launcher both require the immutable
`STAGE_A_FORMAL_PCM_GLM_GEOMETRY_GATE_PASSED.json` gate, whose hash is embedded
in every new run manifest.
