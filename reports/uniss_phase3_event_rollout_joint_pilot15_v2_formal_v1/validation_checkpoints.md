# Fixed15 V2 validation checkpoint summary

- Log: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phase3_event_rollout_joint_pilot15_v2_formal_v1_train_tmux.log`
- Maximum NaN iterations: 0
- Maximum skipped iterations: 0
- Selection status: `exact_runtime_evaluation_required`
- These are teacher-forced validation diagnostics, not proof of useful audio or subsecond latency.

| iteration | checkpoint | interleaved_trajectory | natural_write_fraction | predicted_write_fraction | deadline_forced_fraction | safe_commit_f1 | runtime_action_accuracy | runtime_text_token_accuracy | microblock_token_accuracy | runtime_eos_recall | frontend_residual_rms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | yes | 7.81161 | 0.275563 | 0.179043 | 0 | 0.633159 | 0.688432 | 0.145902 | 0.0176621 | 0 | 0.0148328 |

## Selection rule

Do not select the last checkpoint or the lowest teacher-forced loss alone. Shortlist finite validation checkpoints, then select by natural exact-runtime WRITE, useful-audio latency/quality, EOS, collapse, and Phase3 retention.

A final best checkpoint must remain `not_selected` until exact-runtime train and validation evaluation verifies natural WRITE, no forced WRITE, valid translated PCM, useful-audio latency, EOS, collapse rate, and Phase3 retention.
