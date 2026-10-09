# UniST speech-to-speech subset for the Omni backbone

Direction-balanced, corpus-diverse 10-shard subset of

```
s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup/
    opt/dlami/nvme/zeyangsong/data/unist/converted/unist_qwen3_s2s_v2/
```

Ten shards rather than all 198 because the reference recipe trains on roughly
2,000 hours and reports itself robust at a tenth of that, while the full set is
2.01 TiB against 2.4 TB of free space.  `SELECTED_SHARDS.json` records exactly
which shards and tars were taken and why.

## Layout

```
manifests/manifest-train-NNNNN.jsonl      one per selected shard
speech/unist_bicodec_qwen3_v2/rankNN/     the indexed WebDataset tars
logs/                                     fetch and build logs
```

`WDS_AUDIO_ROOT` must point at `speech/`, not at a `rankNN` directory.

## Reading an utterance

A manifest line's `audios[0]` is a logical URI, not a path:

```
wds://unist/audio/processed/shards/dataset=unist/
     process_version=unist_bicodec_qwen3_v2/rankNN/train-NNNNN-MMMMM.tar
     ?offset=512&length=19216&format=flac
```

Resolve it by taking everything after `process_version=` and joining it under
`WDS_AUDIO_ROOT`, then read `length` bytes at `offset` -- those bytes are a
complete FLAC file.  Verified: offset 512, length 19216 decodes to 1.24 s of
16 kHz mono, matching that record's `source_duration_ms` of 1240.

## Target codes: two options, and why we take the second

Each manifest line carries `target_codec_codes` from the Qwen3-TTS-Tokenizer at
12 Hz -- 16 codebooks per frame.  That is **not** the codec this project uses.

`provenance` carries `source_parquet` and `source_row_index`, which resolve
against the local `data/raw/UniST/*.parquet` and recover `target_bicodec` and
the 32-token `bicodec_global`.  Verified on train-00000 row 249: id,
transcript, translation, `target_bicodec` length 54 and `bicodec_global`
length 32 all agree with the manifest's record.

We take the BiCodec route.  The 32 global tokens bypass the language model
entirely and are where this project's first-place AutoPCP comes from; adopting
the manifest's own codec would discard that and require a different vocoder.

## Licence

The upstream dataset is marked **CC-BY-NC-4.0**.  Non-commercial.
