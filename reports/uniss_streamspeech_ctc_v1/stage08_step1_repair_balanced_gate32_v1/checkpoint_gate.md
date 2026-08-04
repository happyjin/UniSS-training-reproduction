# Stage08 Step1 checkpoint-selection gate

| Model | EN→ZH BLEU | ZH→EN BLEU | Δ EN→ZH vs B2 | Δ ZH→EN vs B2 | Step2 gate | Mean source RTF |
|---|---:|---:|---:|---:|:---:|---:|
| Stage04 B2 | 15.3514 | 13.7876 | — | — | — | — |
| Step1 iter 50 | 22.3852 | 16.0489 | +7.0339 | +2.2614 | FAIL | 0.0909 |
| Step1 iter 100 | 19.2948 | 20.3198 | +3.9435 | +6.5323 | FAIL | 0.0892 |
| Step1 iter 150 | 21.4048 | 16.1689 | +6.0535 | +2.3813 | FAIL | 0.0892 |
| Step1 iter 200 | 19.0630 | 17.3243 | +3.7116 | +3.5367 | FAIL | 0.0887 |
| Step1 iter 250 | 20.9265 | 16.9047 | +5.5751 | +3.1171 | FAIL | 0.0848 |
| Step1 iter 300 | 20.4462 | 19.6138 | +5.0948 | +5.8263 | FAIL | 0.0874 |
| Step1 iter 350 | 21.2031 | 20.1939 | +5.8517 | +6.4064 | FAIL | 0.0901 |
| Step1 iter 400 | 21.4343 | 19.9459 | +6.0829 | +6.1584 | FAIL | 0.0896 |

Step2 gate: EN→ZH > 22.95 and ZH→EN > 22.46.

Selected checkpoint: iteration 350 by mean bidirectional BLEU.

Qwen LoRA and offline Phase3 replay must not start unless this fixed probe passes both directions.
