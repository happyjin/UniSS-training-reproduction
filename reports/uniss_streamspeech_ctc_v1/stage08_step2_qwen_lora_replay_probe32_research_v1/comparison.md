# Stage08 Step2 research-only comparison

> Step1-R did not pass the formal hard gate. These results validate the pipeline and hypothesis only.

| Model | EN→ZH BLEU | ZH→EN BLEU | Mean | Δ Mean vs Step1-R | RTF |
|---|---:|---:|---:|---:|---:|
| Step1-R iter350 | 21.2031 | 20.1939 | 20.6985 | — | 0.0901 |
| Step2 iter25 | 21.6803 | 19.5990 | 20.6396 | -0.0588 | 0.1068 |
| Step2 iter50 | 20.6664 | 19.9107 | 20.2885 | -0.4099 | 0.1064 |
| Step2 iter75 | 20.7785 | 19.8763 | 20.3274 | -0.3711 | 0.1085 |
| Step2 iter100 | 21.8508 | 20.0660 | 20.9584 | +0.2599 | 0.1048 |

Selected research checkpoint: iter100 by bidirectional mean BLEU.
