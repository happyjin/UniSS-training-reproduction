# Stage A v6 formal optimizer-horizon failure

## Decision

The v6 formal run was stopped after semantic collapse. The iteration-50 and
iteration-100 checkpoints are retained only as failure evidence and must not
be used to resume training. Stage A formal remains incomplete and Stage B
remains blocked.

V6 correctly repaired the curriculum horizon, but the formal launcher still
stretched the optimizer learning-rate horizon from 127 to 381 updates. This
made parameter learning rates four to eight times higher than the successful
v5/v6 canary schedule in the late first epoch. CTC argmax predictions became
all blank and Phase3 teacher-code geometry fell below its safety floor.

## Run identity

- run ID: `stage_a_v6_horizon127_formal8_20260817T035800Z`
- initialization: immutable Phase3 v4 iteration 9075
- framework/devices: Megatron, 8 x H200
- sequence length: 18000
- micro/global batch: 1 / 128
- globally shuffled source packs: 16195
- configured coverage: 3 epochs, 381 updates
- independent curriculum horizon: 127 updates
- incorrectly retained LR decay horizon: 381 updates
- configured LR warmup: 19 updates
- interrupted after: iteration 148
- first hard blank stop crossing: iteration 104
- first teacher-geometry stop crossing: iteration 108
- retained evidence checkpoints: iterations 50 and 100
- log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v6/stage_a_formal/stage_a_v6_horizon127_formal8_20260817T035800Z/train.log`
- GPU telemetry: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v6/stage_a_formal/stage_a_v6_horizon127_formal8_20260817T035800Z/train.gpu.csv`
- TensorBoard events: `runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v6/stage_a_formal/stage_a_v6_horizon127_formal8_20260817T035800Z/tensorboard`
- TensorBoard service at failure: `http://10.1.6.203:6118/`

The interrupt reached the process after iteration 148 because the attached
terminal had a large output backlog. The semantic failure had already crossed
the declared hard stop criteria at iterations 104 and 108. No checkpoint
after iteration 100 was written.

## Failure trajectory

| Train iteration | Curriculum progress | Chunk | AR-ASR | Source CTC | CTC blank ratio | Blank posterior | GLM agreement | Teacher cosine | Adapter RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.622 | 320 ms | 0.712295 | 7.391076 | 0.001292 | 0.077698 | 0.121323 | 0.941355 | 0.107392 |
| 90 | 0.701 | 160 ms | 0.640141 | 7.030764 | 0.034570 | 0.147984 | 0.087878 | 0.913673 | 0.157547 |
| 96 | 0.748 | 160 ms | 0.694999 | 6.206858 | 0.247866 | 0.228452 | 0.087877 | 0.895285 | 0.191048 |
| 100 | 0.780 | 640 ms | 0.660574 | 5.698873 | 0.699930 | 0.306263 | 0.155906 | 0.895833 | 0.210962 |
| 104 | 0.811 | 320 ms | 0.645849 | 5.090313 | **0.958572** | 0.403790 | 0.116226 | 0.870396 | 0.232670 |
| 108 | 0.843 | 160 ms | 0.686788 | 4.733317 | **0.992975** | 0.506007 | 0.082400 | **0.845777** | 0.249300 |
| 127 | 0.992 | 320 ms | 0.568358 | 4.043564 | **1.000000** | 0.733250 | 0.119022 | **0.806952** | 0.297158 |
| 148 | 1.000 | 160 ms | 0.520415 | 4.052639 | **1.000000** | 0.723112 | 0.098189 | **0.781766** | 0.313851 |

There were zero NaN and zero skipped iterations. This was a semantic
optimization failure rather than numerical instability. Falling AR-ASR and
source-CTC losses do not make the run healthy: CTC reduced its loss by
concentrating probability on blank while the learned adapter moved away from
the released Phase3 code geometry.

## Controlled comparison with the successful canary clock

The v5 canary and v6 formal run use the same Phase3 initialization, source
pack order, global batch, objective, frozen Whisper frontend, adapter, and
curriculum position at the same update number. Their decisive difference is
the learning-rate clock.

| Iteration | Curriculum progress | Successful canary blank | V6 formal blank | Successful canary cosine | V6 formal cosine | Canary new-head LR | Formal new-head LR | LR ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.386 | 0.000053 | 0.000021 | 0.962460 | 0.965626 | 7.37e-5 | 9.84e-5 | 1.34x |
| 80 | 0.622 | 0.000502 | 0.001292 | 0.943291 | 0.941355 | 3.95e-5 | 9.38e-5 | 2.37x |
| 90 | 0.701 | 0.000998 | 0.034570 | 0.932963 | 0.913673 | 2.92e-5 | 9.17e-5 | 3.14x |
| 100 | 0.780 | 0.001019 | 0.699930 | 0.943506 | 0.895833 | 2.06e-5 | 8.93e-5 | 4.33x |
| 110 | 0.858 | 0.002145 | 0.997573 | 0.928532 | 0.844374 | 1.43e-5 | 8.67e-5 | 6.06x |
| 127 | 0.992 | 0.003397 | 1.000000 | 0.927479 | 0.806952 | 1.00e-5 | 8.16e-5 | 8.16x |

The divergence begins as the LR ratio grows, even though effective curriculum
progress is identical. The independent curriculum repair was therefore
necessary but not sufficient.

## Root cause

The Stage A parameter groups define their own maximum and minimum learning
rates, but all groups still use Megatron's global cosine scheduler. The v6
formal launcher configured:

- `train_iters=381`;
- `lr_decay_iters=381`;
- `lr_warmup_iters=19`.

The successful canary clock instead used:

- `train_iters=127`;
- `lr_decay_iters=127`;
- `lr_warmup_iters=6`.

V6 scaled the curriculum and parameter-unfreeze gates to 127 updates, but did
not scale the base cosine LR returned by `OptimizerParamScheduler`. Thus the
model encountered late-curriculum 320/160-ms batches while the new-head and
adapter learning rates were still close to their maxima. The adapter and CTC
head learned faster than the Phase3-preservation losses could constrain them.

## Required isolated v7 repair

V7 must start again from immutable Phase3 iteration 9075 and must not resume
the failed iteration-100 checkpoint. It must:

1. add an explicit optimizer horizon independent of total train iterations;
2. use `optimizer_iters=127`, `curriculum_iters=127`, and warmup 6 for formal;
3. reproduce the complete successful canary cosine LR curve in epoch one;
4. hold Stage A parameter groups at their existing minimum LR after update
   127 while epochs two and three remain in the 320/160-ms regime;
5. compute parameter-unfreeze progress from the optimizer horizon divided by
   the curriculum horizon, avoiding the v6 double-scaling hazard;
6. add tests for LR-horizon validation, scheduler mutation, gate timing, and
   post-horizon behavior;
7. run an isolated post-decay hold-canary before attempting another formal
   run;
8. stop immediately if hard blank ratio exceeds 0.95 or teacher cosine falls
   below 0.85.

Stage B remains explicitly unauthorized.
