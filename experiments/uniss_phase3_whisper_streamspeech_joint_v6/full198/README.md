# Phase3 Whisper StreamSpeech Joint V6 — full198

This directory is the isolated full-data continuation of the validated V6
15-shard experiment.  It does not modify or reuse any old output directory.

Data contract:

- 19,286,004 full198 joint training records;
- 7,965 bilingual validation records;
- complete 1,161,587-record Phase3 replay offset index;
- full198-specific CTC vocabularies (CMN 73,054; ENG 71,864).

Because the full198 CTC vocabularies differ from the pilot vocabularies, the
15-shard compound checkpoint must not be loaded.  Stage A initializes from the
original full198 Phase3/Whisper models and trains the new full-vocabulary heads.
Stage B then loads the full198 Stage A model weights only.

Formal schedule:

1. Stage A: 500 iterations, frozen Whisper and Qwen, heads only.
2. Stage B: 9,075 iterations, V6 guarded joint training with 20% Phase3 replay.

Both stages use 8 GPUs, BF16, sequence length 18,000, micro batch 2 and global
batch 128.  MBS=2 is the largest setting already observed to fit safely on the
local 144 GB H200s (about 116 GB peak in the historical full198 smoke run).

Run the non-destructive smoke test first:

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v6/full198/scripts/run_smoke_pipeline.sh
```

Launch the formal automatic Stage A -> Stage B pipeline:

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v6/full198/scripts/launch_pipeline_tmux.sh
```

Inspect it with:

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v6/full198/scripts/status.sh
tmux attach -t uniss_phase3_joint_v6_full198
```

TensorBoard is shared with the parent V6 experiment and defaults to port 6033:

```bash
bash experiments/uniss_phase3_whisper_streamspeech_joint_v6/scripts/start_tensorboard.sh
```
