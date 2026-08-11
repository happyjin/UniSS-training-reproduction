# Runtime-parity streaming generalize10

V8/v9 passes every strict runtime gate on its one training trajectory but
fails held-out validation. This isolated experiment starts from the completed
15-shard dense-aligned checkpoint, preserves its Phase3/action/text/frontend
parameters, and trains only the natural-length parallel semantic head on 128
exact deployment-runtime PCM trajectories. Thirty-two disjoint exact-runtime
validation trajectories are evaluated during training.

The first run is deliberately a canary generalization gate. It must pass real
free-running validation before the same objective is scaled to a larger exact
runtime subset. No forced WRITE, oracle length, forced truncation, or relaxed
quality threshold is permitted.
