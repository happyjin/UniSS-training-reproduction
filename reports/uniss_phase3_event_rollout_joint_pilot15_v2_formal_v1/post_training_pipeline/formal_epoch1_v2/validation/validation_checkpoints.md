# Fixed15 V2 validation checkpoint summary

- Log: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/uniss_phase3_event_rollout_joint_pilot15_v2_formal_v1_train_tmux.log`
- Maximum NaN iterations: 0
- Maximum skipped iterations: 0
- Selection status: `exact_runtime_evaluation_required`
- These are teacher-forced validation diagnostics, not proof of useful audio or subsecond latency.

| iteration | checkpoint | interleaved_trajectory | natural_write_fraction | predicted_write_fraction | deadline_forced_fraction | safe_commit_f1 | runtime_action_accuracy | runtime_text_token_accuracy | microblock_token_accuracy | runtime_eos_recall | frontend_residual_rms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | yes | 7.81161 | 0.275563 | 0.179043 | 0 | 0.633159 | 0.688432 | 0.145902 | 0.0176621 | 0 | 0.0148328 |
| 100 | yes | 7.31688 | 0.275563 | 0.178239 | 0 | 0.603261 | 0.730421 | 0.229742 | 0.0197487 | 0.351193 | 0.0269916 |
| 150 | yes | 7.03003 | 0.275563 | 0.211967 | 0 | 0.584161 | 0.736841 | 0.254375 | 0.0209763 | 0.514297 | 0.0313997 |
| 200 | yes | 6.70361 | 0.275563 | 0.179417 | 0 | 0.606106 | 0.746373 | 0.276282 | 0.0218207 | 0.509983 | 0.0339942 |
| 250 | yes | 6.34267 | 0.275563 | 0.203814 | 0 | 0.593683 | 0.751162 | 0.301634 | 0.0215155 | 0.485701 | 0.0359795 |
| 300 | yes | 6.22693 | 0.275563 | 0.214691 | 0 | 0.606711 | 0.744706 | 0.311391 | 0.0224876 | 0.673844 | 0.0373066 |
| 350 | yes | 6.09427 | 0.275563 | 0.214938 | 0 | 0.62288 | 0.74979 | 0.32786 | 0.0228512 | 0.811237 | 0.0386375 |
| 400 | yes | 5.98235 | 0.275563 | 0.213761 | 0 | 0.599269 | 0.748857 | 0.332612 | 0.0230236 | 0.745569 | 0.0396954 |
| 450 | yes | 5.89132 | 0.275563 | 0.199151 | 0 | 0.597294 | 0.758717 | 0.341431 | 0.0232222 | 0.778537 | 0.0409311 |
| 500 | yes | 5.85237 | 0.275563 | 0.193252 | 0 | 0.59933 | 0.762322 | 0.34449 | 0.023203 | 0.75558 | 0.0416226 |
| 550 | yes | 5.83256 | 0.275563 | 0.209609 | 0 | 0.60132 | 0.75547 | 0.348247 | 0.0233845 | 0.763608 | 0.0421557 |
| 600 | yes | 5.80046 | 0.275563 | 0.199431 | 0 | 0.6004 | 0.757765 | 0.350887 | 0.0234399 | 0.792423 | 0.0426065 |
| 650 | yes | 5.76869 | 0.275563 | 0.199554 | 0 | 0.601483 | 0.759153 | 0.355041 | 0.0234935 | 0.792954 | 0.042761 |
| 700 | yes | 5.76027 | 0.275563 | 0.201311 | 0 | 0.605338 | 0.759314 | 0.355642 | 0.0235117 | 0.785277 | 0.0428822 |

## Selection rule

Do not select the last checkpoint or the lowest teacher-forced loss alone. Shortlist finite validation checkpoints, then select by natural exact-runtime WRITE, useful-audio latency/quality, EOS, collapse, and Phase3 retention.

A final best checkpoint must remain `not_selected` until exact-runtime train and validation evaluation verifies natural WRITE, no forced WRITE, valid translated PCM, useful-audio latency, EOS, collapse rate, and Phase3 retention.
