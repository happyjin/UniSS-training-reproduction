# Stage07 B1 checkpoint-selection gate

| Model | EN→ZH BLEU | ZH→EN BLEU | Δ EN→ZH vs B2 | Δ ZH→EN vs B2 | S3 gate | Mean source RTF |
| --- | ---: | ---: | ---: | ---: | :---: | ---: |
| Stage04 B2 | 15.3514 | 13.7876 | — | — | — | — |
| B1 iter 600 | 22.9013 | 3.9628 | +7.5499 | -9.8248 | FAIL | 0.1309 |

Planned S3 gate: EN→ZH > 22.95 and ZH→EN > 22.46.

Selected fixed-probe candidate: iteration 600 by mean bidirectional BLEU.

This gate uses the same 32 validation rows as the Stage04 probe. Passing it is required before generated-audio and streaming latency evaluation; it does not itself claim online latency.
