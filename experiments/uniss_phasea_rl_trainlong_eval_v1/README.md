# Phase A / long-episode RL train-seen evaluation

This isolated evaluation compares the immutable Phase A checkpoint with the
three long-episode RL checkpoints (`iter15`, `iter30`, and `iter45`) on the
longest episodes that were actually included in the formal RL rollout.

The protocol deliberately selects four `cmn->eng` and four `eng->cmn`
episodes.  These examples are **train-seen/in-domain** and answer whether the
RL objective learned its intended behavior.  They are not a generalization
claim.  The final report keeps this result separate from the four previously
evaluated external Wikimedia recordings.

All inference arms use the same Runtime v2 settings:

- 640 ms decision interval;
- 160 ms physical acoustic blocks;
- 24 s bounded acoustic ring;
- identical fixed Phase A speaker tokens;
- continuous, global-timeline, and left-source/right-translation WAVs.

Run the complete evaluation with:

```bash
bash experiments/uniss_phasea_rl_trainlong_eval_v1/scripts/run_all_8gpu.sh
```

The launcher refuses to run unless eight CUDA devices are visible and never
overwrites a completed historical result.
