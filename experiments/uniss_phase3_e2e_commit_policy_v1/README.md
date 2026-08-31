# uniss_phase3_e2e_commit_policy_v1

Isolated fix for the incremental-MT commit policy used by the
`uniss_phase3_v4_e2e_simuls2st_pilot15_v1` free-running gate.  Nothing in that
experiment is modified: the worker wrapper rebinds a single symbol and the gate
runner is a two-hunk copy whose drift is asserted by `tests/test_runner_parity.py`.

## The defect

`evaluation/runtime.py::append_only_commit` commits the first event's full
hypothesis with no stability evidence, then rejects every candidate that does
not extend it.  Measured on the frozen fixed-16 selection at
`endmargin_epoch23 iter_0002264`:

| | value |
|---|---:|
| events | 316 |
| commit conflicts | **260 (82.3%)** |
| `e_mt_target_coverage` (gold source, mean) | 0.211 |

`emilia_zh_0005985930` commits `"That's"` for the whole utterance while the
model's own longest hypothesis is a essentially correct translation:

```
reference : Such a self one who feels that anything is possible and that the future is full of hope
model     : Such a person feels that everything is possible and then everything in the future is full of hope
committed : That's
```

The translation capability is present.  The commit layer discards it.

## The fix

Reuse the audited local-agreement committer the Phase-A cascade already runs,
`uniss_phasea_stateful_longepisode_rl_v1/runtime/commit.py::StablePrefixCommitter`:
commit only what two consecutive hypotheses agree on, minus a holdback, and
flush the remainder at the final event.

`DEFAULT_HOLDBACK = 1` is evidence-based, not arbitrary.  With holdback 0 the
policy commits `"Such a feeling"` at event 5; event 6 genuinely revises to
`"Such a person who thinks ..."`, contradicts it, and the hypothesis freezes
again -- later than the baseline but just as permanently.  Holding one unit back
commits only `"Such a"`, which every later revision still extends, and the
utterance completes.  `tests/test_local_agreement.py` pins both behaviours.

## Run

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python -m pytest -q \
  experiments/uniss_phase3_e2e_commit_policy_v1/tests

RUN_ID=<fresh> CANDIDATE_HF=<hf> CANDIDATE_CHECKPOINT=<ckpt> \
  MAX_S2S_SEMANTIC_TOKENS=384 UNISS_E2E_MT_HOLDBACK=1 \
  bash experiments/uniss_phase3_e2e_commit_policy_v1/scripts/run_gate_local_agreement_8gpu.sh
```

`SELECTION.json` and `CANDIDATE_HF_FINGERPRINT.json` must be placed in the run
root first, exactly as the established gate requires.
