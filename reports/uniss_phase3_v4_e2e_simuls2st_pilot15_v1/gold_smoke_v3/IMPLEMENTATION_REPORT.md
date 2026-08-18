# Gold trajectory smoke v3: decoded PCM gate

Status: **PASSED**

This run upgrades `gold_smoke_v2` by decoding every selected source FLAC in
addition to hashing the encoded file. On 32 real manifest rows, the builder and
independent auditor verified:

- sample rate exactly 16 kHz;
- one source channel;
- decoded frame count exactly equals `source_duration_ms × 16`;
- all decoded PCM samples are finite;
- source audio SHA256 is present;
- all source/target prefix, source GLM, semantic contiguity, and future-leakage
  hard gates remain satisfied.

Counts are unchanged from the lossless smoke: 32 records, 539 unified events,
2,361 source GLM tokens, 8,421 target semantic tokens, and 198 non-empty target
semantic WRITEs before source EOS.
