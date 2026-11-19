# Code Change Log

Public-branch production code changes are listed newest first. Internal
experiment logs and private cache/data paths are intentionally excluded.

## 2026-08-13 · Megatron Lance/mel contract alignment

1. Make Megatron codec-field recovery consume offline
   `target_codec_codes`, using the same reply-add/full-add control-prefix
   builder as the initial template encode.
2. Make mcore audio intake fail fast when audio tokens and precomputed Whisper
   mel are missing or shape-inconsistent, instead of silently substituting a
   zero-audio branch.
3. Make the Lance launcher adapt Shape B to an atomically generated local
   pointer JSONL for current ms-swift compatibility; audio, codec and mel
   payloads remain Lance-backed.
4. Align Megatron speaker prompt with DeepSpeed via shared
   `apply_speaker_prompt_to_codec_emb` after `embed_codec`, including offline
   `speaker_embedding` and online `reference_audio_wav`.
5. Add a Megatron Lance+mel job recipe with speaker prompting enabled when the
   offline ECAPA column is present.

## 2026-07-27 · WDS EOS/history public training recipe

**Public commits**: `054f83c`, `a6e9e05`

**Source WDS commit**: `797b270`

Changes:

1. Add indexed `wds://` S2S entry and mapping-time canonical prompt
   normalization while preserving the public online-ECAPA-only speaker path.
2. Add a path-agnostic launcher and fail-fast preflight for checkpoint
   warm-start, whole Talker + Thinker training, 10× codec EOS loss, and 5%
   train-only codec-history corruption.
3. Pad ultra-short non-empty source waveforms before ECAPA mel extraction and
   return a correctly shaped zero embedding for empty waveforms.
4. Add CPU contracts for WDS byte ranges, prompt normalization, and short
   source audio. No private dataset/checkpoint/cache paths are embedded.

Scope decision:

- Periodic AR validation was briefly ported and then fully reverted by
  `453769a`. The public recipe keeps ordinary validation metrics and leaves AR
  generation to standalone evaluation, avoiding an inference-only dependency
  in the training plugin.
- No offline preprocessing builder or `source_xvector` dataset field is
  restored.

## 2026-07-27 · `15ba9d0` — remove offline x-vector path and unused builder

**Public commit**: `15ba9d0427f225be6eabd1f85457f1e4ce9a4c58`

Changes:

1. Remove the pre-computed `source_xvector` field from dataset preprocessing,
   template encoding/collation, and model forward. Speaker conditioning now
   consistently uses the source/reference waveform and online frozen ECAPA.
2. Remove `dataset_maker/make_common_voice_s2s.py`; it had no public training,
   job, plugin, or test consumer and belonged to the legacy internal dataset
   build flow. Remove its stale `docs/05_data_format.md` instructions and
   handoff index entry at the same time.
3. Keep the generic dataset-maker scripts referenced by Anyscale/KTrain jobs,
   and keep `custom/codec_tokenizer.py` because the training template still
   uses its online target-audio fallback.

Compatibility:

- The current WDS manifests do not contain `source_xvector`, so their training
  behavior is unchanged.
- `QWEN_OMNISPEAK_USE_SPEAKER_PROMPT=true` prefers offline `source_speaker_embedding`
  when present (Lance mel tables); otherwise falls back to online ECAPA from wav.
- EOS loss/history-corruption, prompt normalization, and WDS support remain.

## 2026-07-27 · `772799a` — sync recent master training updates

**Public commit**: `772799a0b7c2400030621adda7b89533f02928c4`

**Source commits**: `1b709be`, `924761a`, `70c79eb`, `c8f6894`

Changes:

1. Support pre-computed `source_xvector` speaker conditioning, retaining the
   online reference-waveform fallback; missing/empty values remain compatible
   with mixed datasets.
2. Add optional main-codec EOS loss weighting and training-only codec-history
   corruption, plus EOS-boundary and premature-EOS metrics on both HF and
   Megatron loss paths.
3. Make S2S task-prompt merging idempotent through the shared target-language
   prompt policy.
4. Add indexed `wds://` FSX byte-range audio resolution while preserving
   standalone WAV/MP3 behavior by default.
5. Remove `custom/qwen_omnispeak/codec_tokenizer.py`. Its codec pre-encoding
   implementation had no training consumer; the only indirect dependency was
   a path helper used by the colocated speaker module, now resolved lazily via
   the root training helper. `custom/codec_tokenizer.py` remains because the
   training template uses it for online target-audio fallback.

Compatibility:

- All new training options default to the previous behavior.
- Legacy local audio paths continue to work with `WDS_AUDIO_MODE=auto`.
- Checkpoint structure and inference behavior are unchanged.
