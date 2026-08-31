# uniss_phase3_content_first_diagnostics_v1

Read-only diagnostics for the `uniss_phase3_content_first_joint_s2st_v1` failure.
This experiment adds **no** training code and mutates **no** existing experiment:
every module here only imports established code and writes into its own report
directory.

Three falsification experiments from
`reports/uniss_phase3_content_first_joint_s2st_v1/root_cause_and_next_plan_v1/ANALYSIS.zh-CN.md`
section 4.1:

| id | module | question |
|---|---|---|
| 0-A | `diagnostics/bridge_parity.py` | Do the block-causal inference codes match the offline codes the SFT was trained on? |
| 0-C | `diagnostics/teacher_forced_ceiling.py` | With a perfect teacher prefix, how much of the target can the SFT checkpoint actually produce? |
| 0-B | `scripts/run_prior_lineage_reeval_8gpu.sh` | Does the prior best lineage still beat content-first under the *current* evaluator? |

0-A decides whether the ASR similarity of 0.048 is a capability gap or an
inference-path bug; 0-C separates capability gap from exposure bias; 0-B fixes
the starting checkpoint for the next training run.
