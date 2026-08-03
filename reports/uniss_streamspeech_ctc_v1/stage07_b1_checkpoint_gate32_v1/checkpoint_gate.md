# Stage07 B1 checkpoint-selection gate

| Model | EN→ZH BLEU | ZH→EN BLEU | Δ EN→ZH vs B2 | Δ ZH→EN vs B2 | S3 gate | Mean source RTF |
| --- | ---: | ---: | ---: | ---: | :---: | ---: |
| Stage04 B2 | 15.3514 | 13.7876 | — | — | — | — |
| B1 iter 600 | 18.8492 | 19.0725 | +3.4978 | +5.2849 | FAIL | 0.0849 |
| B1 iter 1000 | 19.0636 | 17.0456 | +3.7122 | +3.2580 | FAIL | 0.0872 |

Planned S3 gate: EN→ZH > 22.95 and ZH→EN > 22.46.

Selected fixed-probe candidate: iteration 600 by mean bidirectional BLEU.

This gate uses the same 32 validation rows as the Stage04 probe. Passing it is required before generated-audio and streaming latency evaluation; it does not itself claim online latency.
