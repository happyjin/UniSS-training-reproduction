# Phase3 exact-event rollout joint full198 v1

This isolated experiment replaces the historical Overfit/Generalize checkpoint
chain with one continuous Megatron SFT run initialized only from Phase3 v4
`iter_0009075`.

The run mixes three histories inside one optimizer/checkpoint sequence:

1. exact Phase3 replay;
2. clean, complete streaming sessions;
3. model-induced runtime sessions followed by oracle recovery examples.

The third path uses the same append-only grammar as deployment. A model action
may change `WAIT` to `WRITE` (or the reverse), and generated text/semantic
payloads may have different lengths. Recovery examples are rebuilt from that
variable generated transcript; they are not fixed-position token corruption.

Semantic output uses the Phase3-vocabulary-tied four-unit causal microblock
head. Content, natural final length, CONTINUE/END, text, action and EOS are
jointly trained. V9 is retained only as the final fused runtime optimization;
it is not a training checkpoint.

Historical experiment directories and results are imported read-only. New
checkpoints, runs, logs, data and reports use this experiment name and refuse
to overwrite non-empty output directories.

Canary preparation and launch:

```bash
bash experiments/uniss_phase3_event_rollout_joint_full198_v1/prepare_canary.sh
bash experiments/uniss_phase3_event_rollout_joint_full198_v1/run_8gpu.sh --dry-run
bash experiments/uniss_phase3_event_rollout_joint_full198_v1/run_8gpu.sh
```

The canary validates implementation only. Formal training must use full198
complete ordered sessions with the same interface. Independent prefix records
must never be relabeled as exact event-rollout data.
