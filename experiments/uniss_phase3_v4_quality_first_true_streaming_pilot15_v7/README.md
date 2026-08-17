# UniSS Phase3 v4 quality-first true-streaming pilot15 v7

V7 repairs the optimizer clock exposed by the failed v6 formal run. It does
not modify the v5 objective, frozen WhisperVQ frontend, residual adapter, loss
weights, training data, teacher cache, Megatron topology, or earlier runs.

V6 correctly made curriculum progress independent of total training updates,
but Megatron's cosine learning-rate scheduler still decayed over all 381
formal updates. At update 100 the new-head LR was 4.33 times the successful
canary LR, producing CTC blank collapse despite identical curriculum progress.

V7 adds explicit optimizer and optimizer-warmup horizons. Formal Stage A uses
381 total updates, a 127-update curriculum horizon, a 127-update optimizer
horizon, and a 6-update warmup. The first coverage epoch therefore reproduces
the complete successful canary curriculum and cosine LR curve. Updates
128-381 stay in the 320/160-ms regime at the existing parameter-group minimum
learning rates.

The post-decay hold-canary runs for 191 updates. It advertises the inherited
three strict coverage epochs required by the Stage A data gate, while the
explicit train-iteration cap consumes only 191 updates. It reproduces the
complete 127-update canary clock and then stress-tests 64 additional
short-chunk updates after the optimizer has reached its LR floor. Formal
remains blocked unless the final iteration-191 validation at 160 ms passes the
strict gate.
