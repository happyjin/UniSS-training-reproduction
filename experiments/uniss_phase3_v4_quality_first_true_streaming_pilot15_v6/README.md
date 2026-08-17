# UniSS Phase3 v4 quality-first true-streaming pilot15 v6

V6 repairs only the formal curriculum horizon. It does not modify the v5
objective, the frozen WhisperVQ frontend, the residual adapter, loss weights,
data, Megatron topology, or previous checkpoints.

The failed v5 formal run used `train_iters=381` as the denominator for every
curriculum signal. This stretched the proven 127-update canary schedule across
three epochs and over-trained long chunks before reaching 160 ms. V6 adds an
explicit `--stage-a-curriculum-iters` denominator shared by chunk selection,
CTC seed scheduling, reported curriculum progress, and parameter-group gates.

The hold-canary uses 127 total updates and a deliberately aggressive 42-update
curriculum horizon. It reaches the final short-chunk regime early and then
stress-tests sustained 320/160-ms training with the CTC seed at zero. Formal
training is authorized only if the final 160-ms validation retains code
identity and geometry and has a CTC blank ratio no greater than 0.25.

The authorized formal configuration uses 381 total updates, three strict
globally shuffled coverage epochs, and a 127-update curriculum horizon. The
first epoch reproduces the successful v5 canary schedule; epochs two and three
remain in the target 320/160-ms regime.
