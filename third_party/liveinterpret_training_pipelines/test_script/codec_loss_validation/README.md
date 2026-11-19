# Codec loss validation

This directory validates two proposed, opt-in training knobs without changing
the production model or trainer:

- `codec_eos_loss_weight`: MiniMind-style stop/EOS numerator weighting.
- `codec_history_corruption_prob`: random corruption of supervised acoustic
  history inputs; targets and control-prefix tokens remain unchanged.

Only two teacher-forced training metrics are reported:

- `eos_boundary_top1_acc`
- `premature_eos_top1_rate`

Run the CPU contract tests:

```bash
cd projects/liveinterpret_training_pipelines/test_script/codec_loss_validation
python test_codec_training_options.py
```

The one-off real-checkpoint runner and its experiment config were removed after
the production implementation and contract tests were validated. This folder
now keeps only the small deterministic regression guard.
