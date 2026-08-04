# Stage08 Step1-R balanced repair result

Date: 2026-08-04 UTC

## Outcome

The 400-iteration balanced repair completed on eight H200 GPUs with zero
skipped iterations and zero NaN iterations. All checkpoints from iteration 50
through 400 were evaluated on the same fixed direction-balanced 32-row probe.

| Model | EN→ZH BLEU | ZH→EN BLEU | Mean BLEU | Step2 gate |
|---|---:|---:|---:|:---:|
| Stage04 B2 | 15.3514 | 13.7876 | 14.5695 | — |
| Stage07 B1 iter600 | 18.8492 | 19.0725 | 18.9609 | FAIL |
| Original Step1 iter800 | 21.9910 | 17.1467 | 19.5688 | FAIL |
| **Step1-R iter350** | **21.2031** | **20.1939** | **20.6985** | **FAIL** |
| Required gate | >22.9500 | >22.4600 | — | — |

Compared with original Step1 iteration 800, the balanced repair changed:

- EN→ZH: -0.7879 BLEU;
- ZH→EN: +3.0472 BLEU;
- bidirectional mean: +1.1296 BLEU.

The repair therefore achieved its direction-balancing objective and produced
the strongest bidirectional mean in this Emformer/B1 line, but it remains
1.7469 BLEU below the EN→ZH gate and 2.2661 below the ZH→EN gate. Qwen LoRA and
offline Phase3 replay were deliberately not started.

## Diagnosis

The exact 50:50 sampler and ZH→EN weight repaired the direction imbalance, but
frozen-Phase3 NLL remained near 4.10 throughout training. Raising its scalar
weight from 0.5 to 2.0 was therefore insufficient to improve speech-to-Phase3
alignment under simultaneous CTC/AR gradients. The remaining bottleneck is more
consistent with gradient conflict or limited frozen bridge capacity than with
direction sampling.

## Recommended next experiment

Before Step2, run one isolated alignment repair initialized from Step1-R
iteration 350:

1. keep 50:50 direction sampling and frozen Qwen/BiCodec;
2. alternate joint CTC/AR steps with Phase3-only alignment steps, or apply
   gradient projection, so Phase3 gradients cannot be cancelled by CTC/AR;
3. optionally add a low-rank adapter to the Stage04 bridge projection rather
   than unfreezing the entire bridge;
4. use a short 200-step, low-learning-rate run and retain the same fixed probe.

Only a checkpoint that exceeds both fixed BLEU thresholds should unlock Step2.

Detailed checkpoint results:

```text
reports/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_gate32_v1/
```
