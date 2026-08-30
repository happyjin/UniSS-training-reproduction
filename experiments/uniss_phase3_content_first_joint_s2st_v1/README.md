# Content-first joint streaming S2ST v1

An isolated Phase-3-rooted Megatron experiment over the audited fixed 15-shard
event trajectories.  It preserves Phase-3 replay and all streaming objectives,
but coalesces teacher lexical WRITEs smaller than four target tokens into the
next phrase.  This view is process-local and does not edit source packs.

Formal training is one strict globally shuffled coverage epoch (717 updates)
from `uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075`.
