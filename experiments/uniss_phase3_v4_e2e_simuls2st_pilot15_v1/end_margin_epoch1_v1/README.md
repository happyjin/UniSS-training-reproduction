# END-margin 15-shard one-coverage-epoch research run

This directory launches one isolated extended research canary over the immutable
15-shard Phase-B task pool.  It does not authorize the formal three-coverage
run and it never overwrites a historical checkpoint, log, report or
TensorBoard directory.

The objective exactly reproduces the best 100-update END-margin canary:

```text
semantic END CE weight       = 0.50
semantic END margin weight   = 0.25
semantic END logit margin    = 2.00
all roll-in/continue/binary  = 0.00
```

The full one-coverage schedule is derived from the immutable task-pool report:

```text
train updates = 1132
warmup        = 34
MBS / GBS     = 2 / 128
sequence      = 18000
GPUs          = 8
```

Launch with a fresh immutable run ID:

```bash
RUN_ID=endmargin_epoch1_$(date -u +%Y%m%dT%H%M%SZ) \
  bash experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/\
end_margin_epoch1_v1/launch_tmux.sh
```

The launcher starts training and a local TensorBoard on port 6044.  The final
report remains explicitly `formal_training_authorized=false`; the fixed-16
free-running gate must be run separately on the final checkpoint.

To monitor that immutable run and automatically perform the frozen Stage-A
audit validation, HF export and identical fixed-16 free-running evaluation at
the 384 semantic-token hard cap, start the isolated post-training waiter:

```bash
TRAIN_RUN_ID=endmargin_epoch1_YYYYMMDDTHHMMSSZ \
  bash experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/\
end_margin_epoch1_v1/launch_post_training_tmux.sh
```

The waiter validates the one-epoch summary and bitwise audit before it claims
the E2E GPU lock.  It accepts an explicit extended-canary checkpoint path; it
does not alias the run into the historical `learning_canaries` namespace.
