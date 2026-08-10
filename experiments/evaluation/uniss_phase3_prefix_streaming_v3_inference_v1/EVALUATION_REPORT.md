# Full198 Phase3 prefix-streaming v3 inference evaluation report

## 1. Scope and isolation

This evaluation uses the completed formal run
`uniss_phase3_prefix_streaming_full198_joint_v3`.  All code is under the new
evaluation/demo directories and all generated artifacts are under new output
roots.  No historical checkpoint, training script, evaluation output, or
existing demo file is modified or overwritten.

The acoustic frontend is explicitly **source-side pseudo-streaming**.  Frozen
WhisperVQ/GLM is cumulatively re-encoded as 320/480/640 ms of new audio becomes
available.  This report does not claim a causal Whisper encoder.

## 2. Checkpoint selection

Training validated every 250 iterations but saved checkpoints every 500
iterations.  Selection therefore ranks only checkpoint directories that can
actually be loaded.  Six inference-relevant validation losses receive equal
rank weight:

1. prefix S2TT CE;
2. streaming TTS semantic CE;
3. stable commit suffix CE;
4. full teacher KL;
5. adjacent-prefix consistency;
6. WAIT/WRITE action CE.

`iter_0008000` has rank 1 on five of the six components and rank 8 on semantic
CE, for a total rank-sum of 13.  The second candidate has rank-sum 41.  The
selected checkpoint is therefore:

```text
checkpoints/uniss_phase3_prefix_streaming_full198_joint_v3/iter_0008000
```

The 96 q/v LoRA tensors (1,081,344 parameters) were exported without modifying
the base model:

```text
checkpoints/exported_adapters/
  uniss_phase3_prefix_streaming_full198_joint_v3_iter_0008000_lora_v1/
```

The full machine-readable ranking is in `checkpoint_selection.json`.

## 3. Runtime matching the trained objectives

The old Stage4 one-shot WRITE adapter is not reused.  The v3 checkpoint trained
three distinct capabilities, so inference follows the same three-head order:

```text
cumulative audio prefix
  -> stable GLM prefix
  -> WAIT/WRITE action pair
  -> streaming S2TT hypothesis
  -> irreversible stable text commit
  -> streaming TTS semantic continuation blocks
  -> incremental BiCodec decode
```

At final flush, TTS continuation is called repeatedly with the previously
generated semantic history.  Sampling (`temperature=0.7`, `top_p=0.8`) is used
only for semantic speech tokens.  This fixed the initial greedy failure where
one 25-token block was repeated and sounded like a short/buzzy output.  The
final quality gate rejects invalid semantic IDs, long identical runs and very
low token diversity.

## 4. Three-chunk real-audio comparison

Source sample:

```text
eval_outputs/simul_uniss_stage7a_15shard_v1/full_test_e2e_v1/
  e1_continued_sft/full_test_v1/audio/source_wav/
  13129_stage7a_e1_continued_sft_magicdata_0000030545.wav
```

Direction: Chinese to English. Source duration: 13.90 s.

| Chunk | First WRITE | First audio | AL | AP | RTF | Target duration | WAIT/WRITE | Semantic quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 320 ms | 13,900 ms | 13,900 ms | 7,132.9 ms | 1.000 | 0.797 | 5.02 s | 34 / 1 | unique 0.793, max-run 2 |
| 480 ms | 3,680 ms | **4,160 ms** | **6,348.1 ms** | **0.943** | 0.775 | 6.74 s | 21 / 3 | first block unique 0.760, max-run 1 |
| 640 ms | 13,900 ms | 13,900 ms | 7,137.8 ms | 1.000 | **0.697** | 6.70 s | 17 / 1 | unique 0.839, max-run 1 |

For this sample, 480 ms is the only setting that emits translated speech before
the source ends.  The result demonstrates why chunk size must be evaluated with
the learned policy: a smaller 320 ms observation interval does not by itself
force an earlier WRITE.

Listening/result directories:

```text
320 ms:
eval_outputs/uniss_phase3_prefix_streaming_v3_iter8000_v1/chunk_320ms/
  20260810/86006fb09c4841ed9cb609dbfafdce11/

480 ms:
eval_outputs/uniss_phase3_prefix_streaming_v3_iter8000_v1/chunk_480ms/
  20260810/f51b9a43d59a421da77342bcaf850acb/

640 ms:
eval_outputs/uniss_phase3_prefix_streaming_v3_iter8000_v1/chunk_640ms/
  20260810/0f17205b33be469ebc3dbdc74294bbd6/
```

Each directory contains:

```text
source_16k.wav
translation.wav
translation_timeline.wav
stereo_left_source_right_translation.wav
result.json
```

## 5. Stereo and audio integrity

All three stereo artifacts were decoded back from disk and verified as 16 kHz,
two-channel PCM.  Channel RMS values are non-zero:

| Chunk | Left RMS | Right RMS | Left peak | Right peak | Right peak before First audio |
|---:|---:|---:|---:|---:|---:|
| 320 ms | 0.0617 | 0.0231 | 0.8304 | 0.4493 | 0.0000 |
| 480 ms | 0.0596 | 0.0349 | 0.8304 | 0.7251 | 0.0000 |
| 640 ms | 0.0592 | 0.0424 | 0.8304 | 0.9482 | 0.0000 |

This proves the channel contract: left contains source speech, right contains
translated speech, and right is silent before the measured First-audio point.

## 6. Public Gradio validation

The isolated public demo is:

```text
web_demo/uniss_phase3_prefix_streaming_v3_stereo_v1/
```

It provides upload or browser recording, direction selection, 320/480/640 ms
selection, stable translation text, event timeline, target audio, aligned
stereo playback and downloadable JSON.  It is public without username/password.

An external `gradio_client` request through the public share tunnel was run on a
5.46 s Chinese sample.  It returned all seven expected outputs.  Downloaded
files were verified as:

- continuous target WAV: 92,204 bytes;
- target timeline WAV: 266,924 bytes;
- stereo WAV: 533,804 bytes, 16 kHz, two channels;
- result JSON: 24,818 bytes, `selected_iteration=8000`, `chunk_ms=480`.

The share URL is temporary and is intentionally not committed.  The live URL is
written to `web_demo/uniss_phase3_prefix_streaming_v3_stereo_v1/public_url.txt`,
and full access metadata is in `access_info.json`.

## 7. Interpretation and limits

- The selected model and all generated speech are from the current v3
  checkpoint over the original Phase3 base; no offline fallback model is used.
- 480 ms is the best of the three settings on the audited long sample, but this
  is one listening case, not a corpus-level statistical conclusion.
- The frontend needs two cumulative observations for stable-prefix commit and
  captures speaker tokens after a 3.2 s bootstrap.  This is a major lower bound
  on early output for short utterances.
- AL/LAAL in upload mode use generated text-token emission times and observed
  target length; reference-aware corpus LAAL can be added later without
  changing the JSON schema.
- The model remains conservative on some utterances and may WAIT until final
  flush.  The page reports this honestly instead of presenting offline fallback
  audio as streaming output.

