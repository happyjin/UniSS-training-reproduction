# Second-round review (v2)

Same strict reviewer as `review_of_v1.md`, against `paper/gstar_si_v2/main.pdf`.

## Resolved

| Point | How it is resolved |
|---|---|
| **W1 Baseline** | A same-data cascade is now the reference. MIST beats it by 6.15 meeting-level cpBLEU (p<0.001 over windows and over meetings) and by 1.76 in speech (p=0.022 over meetings). The comparison isolates joint modelling. |
| **W3 Significance** | Paired bootstrap over windows and over meetings for every claim. The oracle-diarizer gain is correctly reported as not significant. |
| **W4 Segmentation** | Diarizer-defined windows give the same ranking: 42.89 vs 36.52 for the cascade. |
| **W5 Latency** | Real-time factors and listener-side lag are now reported, with Δ = 0.5 / 1 / 2 s points. The failure at 0.5 s is reported honestly. |
| **W6 Policy vs. training** | The offline-decoded MIST-Simul (40.38 < 42.29) shows the gain is the policy. |
| **W7 Voice** | Speaker similarity and voice confusion are measured, and their limits are discussed (WavLM cosine between two different speakers ≈ 0.75). |
| **W8 Novelty** | Contributions are reframed: speaker-stream fusion from ASR, plus what translation, early commitment and speech add. |
| **W9 Presentation** | 8 pages, and Figure 1 is fixed. |

## Remaining weaknesses

1. **References are machine translations.** Mitigated by same-data comparisons and stated clearly, but a human-verified test subset (even 200 windows) would remove the last doubt about absolute numbers.
2. **One corpus, one direction, close-talk audio mixed to one channel.** Generalisation to far-field meetings (AMI-SDM, NOTSOFAR) and to Zh→En is untested.
3. **The read interval is fixed by training.** The policy collapses at Δ = 0.5 s. A Δ-conditioned or multi-Δ training run would turn this into a proper latency–quality curve.
4. **Speech is evaluated automatically only.** There is no MOS or listening test, and the voice prompts are non-causal.
5. **Novelty is moderate.** The architecture is G-STAR-style fusion plus a known read/wait supervision scheme. The paper's strength is the new task, the careful comparisons and the analysis.

## Scores (v2)

| Criterion | Score |
|---|---|
| Soundness | 3.5 / 5 |
| Excitement | 3.5 / 5 |
| Overall assessment | 3.5 (accept for Findings; borderline-to-accept for the main conference) |
| Confidence | 4 |

## To move to a clear main-conference accept

Items 1–3 are the most valuable, in this order:

1. A human-checked test subset.
2. AMI-SDM or a second language direction.
3. Multi-Δ training.
