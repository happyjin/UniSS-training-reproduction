# Stage08 Step1 checkpoint-selection gate

| Model | EN→ZH BLEU | ZH→EN BLEU | Δ EN→ZH vs B2 | Δ ZH→EN vs B2 | Step2 gate | Mean source RTF |
|---|---:|---:|---:|---:|:---:|---:|
| Stage04 B2 | 15.3514 | 13.7876 | — | — | — | — |
| Step1 iter 100 | 18.6938 | 17.9277 | +3.3425 | +4.1402 | FAIL | 0.0914 |
| Step1 iter 200 | 19.2151 | 17.2577 | +3.8637 | +3.4702 | FAIL | 0.0882 |
| Step1 iter 300 | 18.7601 | 16.8800 | +3.4087 | +3.0924 | FAIL | 0.0827 |
| Step1 iter 400 | 18.5056 | 16.8700 | +3.1542 | +3.0824 | FAIL | 0.0891 |
| Step1 iter 500 | 20.6643 | 14.8042 | +5.3130 | +1.0166 | FAIL | 0.0906 |
| Step1 iter 600 | 19.9454 | 16.6889 | +4.5941 | +2.9014 | FAIL | 0.0867 |
| Step1 iter 700 | 18.9995 | 15.7852 | +3.6482 | +1.9976 | FAIL | 0.0868 |
| Step1 iter 800 | 21.9910 | 17.1467 | +6.6396 | +3.3592 | FAIL | 0.0899 |
| Step1 iter 900 | 21.6516 | 16.3598 | +6.3003 | +2.5723 | FAIL | 0.0890 |
| Step1 iter 1000 | 20.4573 | 15.8197 | +5.1060 | +2.0322 | FAIL | 0.0887 |

Step2 gate: EN→ZH > 22.95 and ZH→EN > 22.46.

Selected checkpoint: iteration 800 by mean bidirectional BLEU.

Qwen LoRA and offline Phase3 replay must not start unless this fixed probe passes both directions.
