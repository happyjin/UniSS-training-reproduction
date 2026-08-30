# Content-first joint streaming S2ST v1

An isolated Phase-3-rooted Megatron experiment over the audited fixed 15-shard
event trajectories.  It preserves Phase-3 replay and all streaming objectives,
but coalesces teacher lexical WRITEs smaller than four target tokens into the
next phrase.  This view is process-local and does not edit source packs.

Formal training is one strict globally shuffled coverage epoch (717 updates)
from `uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075`.

After the formal checkpoint, `scripts/run_automatic_pipeline.sh` waits for the
complete fixed-15 free-running evaluation, then executes two coverage-first
GRPO rounds.  Every round uses fresh rollouts over the immutable 64 bilingual
long episodes with four candidates per episode, packs the trajectories with
Phase-3 replay, and trains exactly one fresh-rollout epoch.  It generates a
post-round-2 rollout and a Chinese comparison report before restoring the
eight-GPU 60% utilization/memory holder.  The holder is started only after all
required artifacts exist and no GPU compute process remains.

The pipeline is fail-fast.  If a real stage error stops it, an EXIT finalizer
schedules a separate recovery watcher.  The watcher waits until GPU compute
processes have cleared and then starts the same holder, preserving the failed
stage and logs for diagnosis instead of silently modifying the experiment.
