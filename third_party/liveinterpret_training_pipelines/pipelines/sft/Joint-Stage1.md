# Joint-Stage1 SFT Joint Training

`joint-stage1` is a Stage-3 SFT joint training entry for mixed streaming speech-to-speech translation data. It reads one JSONL file and dispatches rows by each sample's `task` field.

## Supported `task` values

- `normal_translation`
- `speech2speech_translation`
- `bilingual_translation`
- `bilingual_mutual_translation`
- `multi_speaker_translation`

All accepted rows reuse the existing Stage-3 codec-only streaming SFT path: `alignment` validation, `target_audios` normalization, real-timeline codec concatenation, and optional `codec_to_sem_visible` streaming mask.

## Data

Use `SFT_JOINT_STAGE1_DATA` to point to the mixed JSONL:

```bash
export SFT_JOINT_STAGE1_DATA=/abs/path/to/sft_joint_stage1.jsonl
```

Fallback order in the training scripts:

1. `dataset_maker/sft_joint_stage1.jsonl`
2. `dataset_maker/sft_joint_stage1.jsonl`
3. `pipelines/sft/joint_stage1_data_format.jsonl`

Required fields per row are the same as the existing Stage-3 SFT translation tasks:

- `task`
- `messages` with a user `<audio>` turn
- `audios`
- `alignment`
- `target_audios` or per-chunk `alignment[i].target_audio`

The default strict mode is off. Malformed rows are skipped; set `JOINT_STAGE1_STRICT=true` to raise errors.

## DeepSpeed / local

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export SFT_JOINT_STAGE1_DATA=/abs/path/to/sft_joint_stage1.jsonl
bash pipelines/sft/train_joint_stage1.sh
```

Ray mode uses the same entrypoint:

```bash
export SFT_JOINT_STAGE1_DATA=/abs/path/to/sft_joint_stage1.jsonl
USE_RAY=true bash pipelines/sft/train_joint_stage1.sh
```

## Megatron

```bash
export MODEL_PATH=/path/to/project/models/qwen3_omnispeak_d
export MODEL_TYPE=qwen3_omnispeak_d
export SFT_JOINT_STAGE1_DATA=/abs/path/to/sft_joint_stage1.jsonl
export TP=2 PP=1 CP=1 EP=1
export SEQUENCE_PARALLEL=false
bash pipelines/sft/train_joint_stage1_megatron.sh
```

## Key environment variables

| Env | Default | Effect |
| --- | --- | --- |
| `SFT_JOINT_STAGE1_DATA` | `pipelines/sft/joint_stage1_data_format.jsonl` | Mixed JSONL path. |
| `SFT_JOINT_STAGE1_CODEC_CACHE` | `.cache/sft_joint_stage1_codec` | Codec-token cache path, also mapped to `CONTINUE_PRETRAIN_CODEC_CACHE`. |
| `JOINT_STAGE1_STREAMING` | `true` | `false` disables streaming mask and uses offline full-context codec attention. |
| `JOINT_STAGE1_EXTRA_LAG_MS` | `0` | Global wait-k style lag added to the alignment-derived schedule. |
| `JOINT_STAGE1_SOURCE_DURATION_FROM_FILE` | `true` | `false` uses last `source_end_ms` instead of reading the source wav duration. |
| `JOINT_STAGE1_STRICT` | `false` | `true` raises on malformed rows instead of skipping them. |

## Registered names

- Dataset: `sft_joint_stage1`
- Template: `qwen_omnispeak_joint_stage1`
- DeepSpeed runner: `run_joint_stage1 sft_joint_stage1 joint-stage1`
- Megatron runner: `run_joint_stage1_megatron sft_joint_stage1 joint-stage1`
