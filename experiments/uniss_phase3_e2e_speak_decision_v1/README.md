# uniss_phase3_e2e_speak_decision_v1

One additional coverage epoch from `endmargin_epoch23_15shard_20260824T190227Z/iter_0002264`,
adding the two loss terms the E2E objective does not have.  Nothing in
`uniss_phase3_v4_e2e_simuls2st_pilot15_v1` is modified: this directory rebinds
four attributes on its trainer and vendors a two-line copy of its Megatron
launcher (whose `ENTRYPOINT` is hardcoded), with a parity test on the diff.

## What is loaded

| | |
|---|---|
| trained from | `extended_canaries/endmargin_epoch23_15shard_20260824T190227Z/iter_0002264` (the E2E student, 3 coverage epochs) |
| frozen teacher / replay anchor | offline phase3 `iter_0009075`, passed as `--phase3-checkpoint` by the established launcher -- **used, not retrained** |
| frozen acoustic front end | Stage-A V1 `iter_0000381`, bitwise audited after the run |
| data | 15-shard `task_pool_formal_p4_20260820T154500Z`, unchanged |
| geometry | 1132 updates (1 coverage epoch), MBS 2 / GBS 128, seq 18000, 8 GPU, seed 20260819 |

## Why these two terms

| measured fact | source |
|---|---|
| The model recognises on 82% of events but translates on 16.8% and speaks on 15.8% | `PHASE0_FINDINGS.zh-CN.md` |
| The three inference policies land at 0.168 (starved), 0.958 and 1.000 (both repetition loops), with nothing between | S0.1 |
| Raising `boundary_ce` cannot help: `WRITE_GENERATE` carries the fragment's own loss kind and is not in that bucket; only `WAIT_READ` is | `task_samples.py:477/496/512/535` |
| Every intervention that raises the speak rate triggers repetition -- `emilia_zh_0004122419` session text length ratio 1.70 -> 15.40 -- and no existing term penalises it | S0.1 |
| `semantic_rollin_continue_decision_margin` is the one existing loss measured to help, once the commit-policy confound is removed: cmn chrF +18.7% | S0.2 |

A fragment's semantic length **is** its END position
(`task_samples.py:526` records `semantic_boundaries[event] = content_start + len(delta)`),
so no separate length loss is written: over-generation is the roll-in END
decision, which the preserved decisionrow configuration already supervises.

## Weights

Preserved exactly as S0.2 measured them, so this run does not perturb the one
configuration with evidence behind it:

| term | weight | note |
|---|---:|---|
| `semantic_end_ce` | 0.50 | parent value |
| `semantic_end_margin` | 0.25 (logit margin 2.0) | parent value |
| `semantic_rollin_end_ce` | 0.25 | decisionrow |
| `semantic_rollin_continue_decision_margin` | 0.25 (margin 1.0) | decisionrow -- the measured winner |
| `semantic_rollin_continue_margin` | 0.025 | decisionrow |
| `semantic_boundary_rollin_rate` | 0.5, ramp 100 | **not optional**: at rate 0 the three roll-in terms select empty masks and stay identically zero |
| `semantic_boundary_binary` | 0.00 | mutually exclusive with the margin family (`pretrain_e2e_megatron.py:546-566`), and S0.2 showed it costs 17% free coverage |

Added, and these two are **starting points rather than measured optima**:

| term | weight | note |
|---|---:|---|
| `speak_decision` | 0.50, logit margin 1.0 | scaled to `semantic_end_ce`; the term normalises WRITE and WAIT separately so each gets half the weight despite the 5:1 imbalance |
| `repetition_penalty` | 0.10, window 8 | small on purpose: new, unvalidated, and a probability in [0,1] so it cannot dominate the cross-entropy |

## Gate

All four together, on the fixed-16 selection with local agreement holdback 2 and
pacing margin 1200 ms:

* `WRITE_MT` per event >= 0.50
* session-own text coverage not below S0's 0.514 (cmn->eng)
* semantic length ratio in [0.9, 1.2]
* audible onset <= 1500 ms

Raising the speak rate while coverage falls or the length ratio explodes is the
failure mode S0.1 already demonstrated, so no single metric decides.

## Run

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python -m pytest -q \
  experiments/uniss_phase3_e2e_speak_decision_v1/tests
RUN_ID=speak_decision_$(date -u +%Y%m%dT%H%M%SZ) \
  bash experiments/uniss_phase3_e2e_speak_decision_v1/scripts/run_8gpu.sh
```
