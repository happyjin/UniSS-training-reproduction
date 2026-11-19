# UniST BiCodec → Qwen3 S2S conversion

This pipeline converts every usable UniST Parquet row and audits rows that
cannot supply target-speech supervision:

```text
source_bicodec + bicodec_global ──Spark-TTS──> source FLAC in indexed WDS tar
target_bicodec + bicodec_global ──Spark-TTS──> target waveform (memory only)
target waveform ──Qwen3-TTS Tokenizer──> target_codec_codes [T,16]
```

Run the resumable eight-GPU build:

```bash
bash dataset_maker/unist/run_unist_qwen3_conversion_8gpu.sh
```

Defaults:

- Input: `/opt/dlami/nvme/zeyangsong/data/unist/raw/*.parquet`
- Output: `/opt/dlami/nvme/zeyangsong/data/unist/converted/unist_qwen3_s2s_v2`
- GPUs: 8 processes, one GPU each
- Batch: up to 64 rows, additionally bounded by 512 padded audio-seconds
- BiCodec padding: zero by default. V2 pools source and target sequences across
  each 8192-row read block, then decodes groups of at most 64 sequences with
  exactly equal semantic length because the official decoder has no padding mask.
  A 256-row A/B found that allowing 16-token padding changed 12.51% of the
  resulting Qwen3 target tokens, so the full build uses the strict setting.
- Qwen3 encoding: direct GPU path. It reproduces the wrapper's codec tokens
  exactly while avoiding the GPU → NumPy/CPU padding → GPU round trip.
- Source audio: PCM16 FLAC packed in indexed tar shards
- Target audio: not duplicated on disk after Qwen3 codec encoding

Output layout:

```text
unist_qwen3_s2s_v2/
  manifests/manifest-*.jsonl
  rejects/reject-*.jsonl
  speech/unist_bicodec_qwen3_v2/rankXX/*.tar
  state/*.json
  logs/rankXX.log
  _SUMMARY.json
  _COMMITTED.json
```

Training uses:

```bash
BRIDGE_ALIGNMENT_DATA='/opt/dlami/nvme/zeyangsong/data/unist/converted/unist_qwen3_s2s_v2/manifests/manifest-*.jsonl' \
WDS_AUDIO_ROOT=/opt/dlami/nvme/zeyangsong/data/unist/converted/unist_qwen3_s2s_v2/speech \
WDS_AUDIO_MODE=required \
bash pipelines/bridge_alignment/train_speech_to_speech_translation_upstream_wds.sh
```

Committed files are skipped on rerun. An input file without a state record is
treated as uncommitted; only that input file's converter-owned partial/orphan
outputs are removed and rebuilt. Each state verifies
`output_rows + rejected_rows == input_rows`. `_COMMITTED.json` is written only
after all input Parquet files and row counts have been verified.

The source audit found 499,920 WenetSpeech4TTS `cmn→eng` rows whose original
`target_bicodec` is empty and another 895 whose target stream has only one
frame (20 ms). Both are skipped as complete samples and written to
`rejects/reject-*.jsonl` with source Parquet, row index, id, language direction,
texts, token lengths, and reason `empty_target_bicodec` or
`single_frame_target_bicodec`. The combined reject count is 500,815; valid
WenetSpeech4TTS rows remain in the training manifest.

## In-place quality filter

After conversion, run the quality audit in dry-run mode first and then apply it
to the same manifest/reject/state set:

```bash
python dataset_maker/unist/filter_unist_quality_in_place.py
python dataset_maker/unist/filter_unist_quality_in_place.py --apply
```

`unist_quality_v1` does not impose a tighter general upper-length cutoff. It
hard-rejects target/source duration ratios outside `[0.1, 10]`, Qwen streams
shorter than two frames, and samples pinned to the known 59 s / 750-frame
generation ceiling. It also detects target BiCodec collapse when either one
token occupies at least 95% of the stream or at least 95% of adjacent tokens
repeat. Every removed row is appended to its existing reject JSONL with all
triggered flags and collapse metrics; no new training subset is created.

The production tar files are intentionally not rewritten. The 98,372 removed
source members (8,676,007,974 encoded bytes) remain physically in tar but are
unreachable from the manifests and are recorded as orphan accounting in
`_COMMITTED.json`. Current committed closure is 19,227,252 manifest rows plus
599,187 rejects = 19,826,439 inputs, with 1,704,324,507 retained Qwen frames and
2,211,796,760,762 reachable source-audio bytes. A second full dry-run must
report zero new removals before the dataset is used for training.

On eight representative train shards (8192 rows per GPU), V2 reduced the
straggler wall time from 293.0 s to 196.6 s (1.49×) and total GPU-seconds by
1.64×. Pooling changes only the numerical batching path: source PCM differed
by less than one int16 unit on average; Qwen main-code changes were 0.146% and
overall 16-codebook changes were 4.73%, concentrated in higher residual codes.
For this reason V2 is written to a new output root and is never mixed with V1.

## Validation

Run the full read-only structural audit, followed by the stratified eight-GPU
codec replay:

```bash
python dataset_maker/unist/validate_unist_qwen3_conversion.py --workers 32
bash dataset_maker/unist/run_unist_validation_8gpu.sh
```

The static pass reads every manifest and reject row, reconciles every original
Parquet row index, validates prompts, codec shapes/ranges/durations and WDS byte
ranges, checks record-id uniqueness and tar ownership, and selects 256 samples
stratified by source dataset, direction and target duration. The GPU pass then:

- reloads the original Parquet row and replays Spark-TTS BiCodec;
- reads source FLAC through the production `custom.wds_audio` resolver;
- compares direct Qwen encoding with the official wrapper on the same waveform;
- decodes stored Qwen codec and checks duration, finiteness, energy and clipping;
- materializes 10 balanced en→zh / zh→en listening samples, each with
  `source_wds.wav`, `target_bicodec.wav`, `target_qwen_roundtrip.wav` and metadata.

The Qwen token stream is sensitive to small CUDA/batch-order floating-point
differences near quantizer boundaries. Therefore a fresh process is expected to
show high token agreement rather than bit identity with the conversion process.
Bit identity is still required between the direct and official wrapper paths
when both consume the exact same waveform. Source replay is accepted only when
RMSE is at most `0.005` and waveform correlation is at least `0.995`.

The initial conversion audit completed on 2026-07-30 and passed all 19,826,439
input rows before the in-place quality filter:

- 19,325,624 manifest rows plus 500,815 rejects exactly account for input;
- 662 tar shards, no missing/orphan tar, no duplicate record-id hash;
- 1,775,547,182 codec frames and 2,220,472,768,736 source-audio bytes;
- 256/256 GPU samples passed across 21 datasets; no source/target near-silence;
- source replay correlation minimum `0.999960`; Qwen duration delta maximum
  `0.060 s`; direct encoder and official wrapper agree for 256/256 samples;
- the 10-sample listening bundle contains 30 readable mono WAV files.

The validator now uses the post-filter expected counts and accepts intentionally
unreferenced WDS members while still rejecting missing/unowned tar files. The
post-filter 32-worker rerun passed all expected checks with zero errors in about
112 seconds.

The report/listening directory is generated under
`/opt/dlami/nvme/zeyangsong/data/unist/validation/unist_qwen3_s2s_v2_20260730`.
The 2026-07-30 result is recorded above; its temporary report and listening
artifacts were removed after acceptance so the data root contains only raw and
directly trainable V2 data. Re-running the two validation commands recreates it.
