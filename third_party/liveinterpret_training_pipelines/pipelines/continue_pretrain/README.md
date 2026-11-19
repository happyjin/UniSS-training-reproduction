# Qwen-OmniSpeak Continue Pretrain

Pipelines for continue-pretraining three Thinker families (Qwen3-Omni / Qwen2.5-Omni-7B / Qwen2.5-Omni-3B) × variants A-J with three task families and an optional joint setup.

## Thinker families and model types

| MODEL_TYPE prefix | Thinker | Hidden state strategy | Megatron EP |
|-------------------|---------|-----------------------|-------------|
| `qwen3_omnispeak_{a..j}` | Qwen3-Omni-30B-A3B (MoE) | Layer 24 | `EP>1` recommended (e.g. `EP=8`) |
| `qwen2_5_7b_omnispeak_{a..j}` | Qwen2.5-Omni-7B (dense) | Last/final layer | `EP=1` (default) |
| `qwen2_5_3b_omnispeak_{a..j}` | Qwen2.5-Omni-3B (dense) | Last/final layer | `EP=1` (default) |

Exported model directories live under `models/custom_models/`, named identically to `MODEL_TYPE`.

## Tasks

| Task | Dataset name | Supervision | What is trained |
|------|--------------|-------------|-----------------|
| Speech-to-Text Translation | `continue_pretrain_speech_to_text_translation` | Thinker text CE | Thinker (and bridge/talker through the shared graph when `TUNER_TYPE=full`); Code2Wav frozen |
| Multi-Speaker Segmentation | `continue_pretrain_multi_speaker_segmentation` | Thinker text CE | Same as above |
| Speech-to-Speech Translation | `continue_pretrain_speech_to_speech_translation` | Thinker text CE + Talker codec CE (main + residual) | Thinker + Bridge + Talker end-to-end; Code2Wav frozen |
| Joint | `continue_pretrain_all` | Mixed-batch (per-row gating) | All of the above in a single run |

The codec tokenizer used for speech generation is **Qwen3/Qwen3-TTS-Tokenizer-12Hz**
(override via `QWEN3_TTS_TOKENIZER`).

## Data format

`data_format.jsonl` follows `ms-swift-speech-simul-v1`. Required fields by task:

* `speech_to_text_translation` / `multi_speaker_segmentation`:
  `task`, `messages` (last turn must be `assistant`), `audios`.
* `speech_to_speech_translation`:
  `task`, `messages`, `audios`, `target_audio`. `target_text` is appended
  as the assistant text turn when `CONTINUE_PRETRAIN_APPEND_TARGET_TEXT=true`.

Rows that do not match the requested `task_filter` are dropped automatically.

## Scripts

* `common_args.sh` — shared env and the `run_continue_pretrain` helper (DeepSpeed / Ray).
* `common_args_megatron.sh` — shared env and the `run_continue_pretrain_megatron` helper.
* Per task, three variants are provided (see the matrix below):
  * `train_<task>.sh` — swift sft + DeepSpeed (Variant 1)
  * `train_<task>_ray.sh` — swift sft + Ray + DeepSpeed (Variant 2)
  * `train_<task>_megatron.sh` — megatron sft TP/PP/CP/EP (Variant 3)
  where `<task>` ∈ {`speech_to_text_translation`, `multi_speaker_segmentation`,
  `speech_to_speech_translation`, `joint`}.

All scripts forward extra CLI arguments to the underlying launcher.

## Distributed / parallel training: three variants

Every task (`speech_to_text_translation` / `multi_speaker_segmentation` /
`speech_to_speech_translation` / `joint`) ships three scripts:

| Variant | Parallel strategy | Use case | Script suffix |
| --- | --- | --- | --- |
| swift sft + DeepSpeed ZeRO2/3 | DP + ZeRO | single-node / multi-node | `train_<task>.sh` |
| swift sft + Ray + DeepSpeed | DP + ZeRO | multi-node (Ray scheduled) | `train_<task>_ray.sh` |
| megatron sft | TP/PP/CP/EP + ZeRO-1 | large-scale high-performance | `train_<task>_megatron.sh` |

Task ↔ script matrix:

| Task | DeepSpeed | Ray + DeepSpeed | Megatron |
| --- | --- | --- | --- |
| speech_to_text_translation | `train_speech_to_text_translation.sh` | `train_speech_to_text_translation_ray.sh` | `train_speech_to_text_translation_megatron.sh` |
| multi_speaker_segmentation | `train_multi_speaker_segmentation.sh` | `train_multi_speaker_segmentation_ray.sh` | `train_multi_speaker_segmentation_megatron.sh` |
| speech_to_speech_translation | `train_speech_to_speech_translation.sh` | `train_speech_to_speech_translation_ray.sh` | `train_speech_to_speech_translation_megatron.sh` |
| joint | `train_joint.sh` | `train_joint_ray.sh` | `train_joint_megatron.sh` |

### Variant 1: swift sft + DeepSpeed ZeRO2/3 (DP + ZeRO)

Standard ms-swift / torchrun env vars are honoured.

Single-node:

```bash
NPROC_PER_NODE=8 \
NNODES=1 NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
bash train_joint.sh
```

Multi-node (run once per node, distinguished by `NODE_RANK`):

```bash
NPROC_PER_NODE=8 NNODES=2 NODE_RANK=0 \
MASTER_ADDR=<master-ip> MASTER_PORT=29500 \
bash train_joint.sh

# On worker:
NPROC_PER_NODE=8 NNODES=2 NODE_RANK=1 \
MASTER_ADDR=<master-ip> MASTER_PORT=29500 \
bash train_joint.sh
```

### Variant 2: swift sft + Ray + DeepSpeed (DP + ZeRO, multi-node)

Same parallel strategy as Variant 1 (DP + DeepSpeed ZeRO); only the launch /
placement is handled by a Ray cluster, so you do NOT set `NODE_RANK` per node.

Bring up the Ray cluster first:

```bash
# Head node
ray start --head --port=6379
# Each worker node
ray start --address='<head-ip>:6379'
```

Then run ONCE on the head node; Ray schedules workers across the cluster:

```bash
RAY_WORLD_SIZE=16 \      # total training GPUs across the whole cluster
RAY_NPROC_PER_NODE=8 \   # GPUs per physical node (placement granularity)
bash train_speech_to_speech_translation_ray.sh
```

The wrapper auto-builds the `--device_groups` JSON from `RAY_WORLD_SIZE` /
`RAY_NPROC_PER_NODE`. Override the whole thing with `RAY_DEVICE_GROUPS='{...}'`
if needed, e.g.:

```bash
RAY_DEVICE_GROUPS='{"nproc_per_node":8,"default":{"device":"GPU","ranks":[0,1,2,3,4,5,6,7],"workers":["default"]}}' \
bash train_joint_ray.sh
```

You can also enable Ray on the plain scripts directly via `USE_RAY=true` plus
`RAY_DEVICE_GROUPS`. See
<https://swift.readthedocs.io/zh-cn/latest/Instruction/Ray.html> for the full
`device_groups` schema.

### Variant 3: megatron sft (TP/PP/CP/EP + ZeRO-1, large-scale)

Uses the non-invasive mcore-bridge registration in
`liveinterpret_training_pipelines/custom/qwen_omnispeak/mcore_adaptation.py`,
mapping the thinker to a Megatron `GPTModel` split by TP/EP; the talker + bridge
are loaded as replicated submodules. ZeRO-1 is provided by Megatron's
`DistributedOptimizer`. The talker codec loss (main + residual CE) is computed
natively by `mcore_talker_loss.py` from the captured thinker hidden states --
the SAME objective as the DeepSpeed path (CPU-verified `|Δ|=0` for all variants
in `test_mcore_talker_loss.py`).

> **EP setting per Thinker family**:
> - Qwen3-Omni-30B (MoE thinker) → recommend `EP>1` (e.g. `EP=8`) to shard experts.
> - Qwen2.5-Omni-7B / 3B (dense thinker) → keep default `EP=1`; no MoE args will be added.

```bash
# Qwen2.5-Omni-3B variant B, dense thinker, single node 8 GPUs (EP=1 default)
MODEL_TYPE=qwen2_5_3b_omnispeak_b \
MODEL_PATH=models/custom_models/qwen2_5_3b_omnispeak_b \
TP=2 PP=1 CP=1 \
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train_speech_to_speech_translation_megatron.sh

# Qwen3-Omni variant A, MoE thinker, single node 8 GPUs (EP=2 recommended)
MODEL_TYPE=qwen3_omnispeak_a \
MODEL_PATH=models/custom_models/qwen3_omnispeak_a \
TP=2 PP=1 CP=1 EP=2 \
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train_speech_to_text_translation_megatron.sh

# Multi-node: run once per node
TP=4 EP=4 \
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master-ip> MASTER_PORT=29500 NPROC_PER_NODE=8 \
bash train_joint_megatron.sh
```

Tunable env vars: `TP` / `PP` / `CP` / `EP` / `ETP` / `SEQUENCE_PARALLEL`,
`MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE`, `LR` / `MIN_LR` / `LR_WARMUP_FRACTION`,
`TUNER_TYPE` (`full`/`lora`), `FREEZE_LLM` / `FREEZE_VIT` / `FREEZE_ALIGNER`,
`RESIDUAL_LOSS` / `RESIDUAL_LOSS_WEIGHT`, `RECOMPUTE_*`, `ATTENTION_BACKEND`.

> **Megatron coverage**:
> * Text-supervised tasks (`speech_to_text_translation`,
>   `multi_speaker_segmentation`): thinker text CE under full TP/PP/CP/EP + SP.
> * `speech_to_speech_translation` / `joint`: thinker text CE **plus the native
>   talker codec loss** (per-row gated for mixed batches). The codec loss
>   requires `PP=1`, `CP=1`, `sequence_parallel=false` and no packing/padding-free
>   (the s2s/joint scripts set `SEQUENCE_PARALLEL=false`; the thinker still uses
>   full TP/EP/DP). Unsupported configs fail fast. Set
>   `QWEN_OMNISPEAK_MCORE_CODEC_LOSS=false` to train the thinker text path only.

## Tuner: full vs LoRA

```bash
# Full-parameter (default): freezes only Code2Wav.
TUNER_TYPE=full bash train_speech_to_speech_translation.sh

# LoRA: ms-swift's multimodal target_regex respects FREEZE_LLM/FREEZE_VIT/FREEZE_ALIGNER.
TUNER_TYPE=lora \
LORA_TARGET_MODULES=all-linear \
LORA_RANK=16 LORA_ALPHA=32 \
FREEZE_VIT=true \
bash train_speech_to_speech_translation.sh
```

## Freezing Code2Wav

The user requirement is that **only the Code2Wav (vocoder) sub-module is
frozen**. The current `qwen_omnispeak` package does not include a Code2Wav
module, so the default freeze regex is a best-effort match across common
naming conventions:

```
^(model\.talker\.code2wav|model\.code2wav|model\.talker\.codec_decoder|model\.codec_decoder|model\.talker\.vocoder|model\.vocoder)(\.|$)
```

Override with:

```bash
FREEZE_PARAMETERS='^model\.talker\.code2wav(\.|$)' \
bash train_joint.sh
```

If no parameter name matches the regex, ms-swift simply trains everything
that is not already frozen by the LoRA scope flags.

## Common environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTINUE_PRETRAIN_DATA` | `<this dir>/data_format.jsonl` | Training JSONL path |
| `QWEN3_TTS_TOKENIZER` | `Qwen3/Qwen3-TTS-Tokenizer-12Hz` | Codec tokenizer id |
| `CONTINUE_PRETRAIN_CODEC_CACHE` | `<repo>/.cache/continue_pretrain_codec` | Per-audio codec cache directory |
| `CONTINUE_PRETRAIN_APPEND_TARGET_TEXT` | `true` | Append `target_text` as assistant turn for s2s rows |
| `CONTINUE_PRETRAIN_STRICT` | `false` | Raise on malformed rows instead of dropping them |
| `MODEL_PATH` | `<repo>/models/qwen_omnispeak_b` | Exported variant directory (any of the three families) |
| `MODEL_TYPE` | `qwen_omnispeak_b` | Format: `{qwen3\|qwen2_5_7b\|qwen2_5_3b}_omnispeak_{a..j}` |
| `TEMPLATE` | `qwen_omnispeak_continue_pretrain` | **Continue-pretrain fixed value**, same for all three families; do not change |
| `TUNER_TYPE` | `full` | `full` or `lora` |
| `FREEZE_PARAMETERS` | Code2Wav regex | Regex of parameters to freeze |
| `TRAINABLE_PARAMETERS` | (empty) | Comma-separated extra trainable patterns |
| `DEEPSPEED` | `zero2` | Deepspeed config preset |
| `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` | `1`/`1`/`0`/`127.0.0.1`/`29500` | Distributed launch |
| `USE_RAY` | `false` | Set to `true` to launch via Ray (`--use_ray`) |
| `RAY_EXP_NAME` | (empty) | Forwarded to `--ray_exp_name` |
| `RAY_DEVICE_GROUPS` | (auto) | JSON string forwarded to `--device_groups`; auto-built by `*_ray.sh` |
| `RAY_WORLD_SIZE` | `8` | (`*_ray.sh`) total training GPUs across the cluster |
| `RAY_NPROC_PER_NODE` | `8` | (`*_ray.sh`) GPUs per physical node (placement) |
| `TP` / `PP` / `CP` / `EP` / `ETP` | `2`/`1`/`1`/`1`/`1` | (`*_megatron.sh`) parallelism; keep `EP=1` for dense thinkers, `EP>1` for Qwen3-Omni MoE |
| `SEQUENCE_PARALLEL` | `true` | (`*_megatron.sh`) Megatron sequence parallelism (auto set to `false` for s2s/joint by those scripts) |
| `MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` | `1` / `16` | (`*_megatron.sh`) Megatron batch sizes |
| `LR` / `MIN_LR` / `LR_WARMUP_FRACTION` | `1e-5` / `1e-6` / `0.03` | (`*_megatron.sh`) Megatron LR schedule |
