# Simul-S2ST route — Step 4 decision (after Step2 v3 + Step3 AR smoke)

> Date: 2026-08-09 · Evidence: `step2_trained_nar_decode_v3_blankpen`, `step3_ar_pareto_smoke8_v3`

## Verdict

**Stay on Route A.** Do **not** open Route B (FAST Student retrain) or Route C (Thinker–Talker) yet.

NAR quality gate is still red — but the failure mode moved: v6 **fixed blank collapse** (0% blank frames) while UER stays ~99%. Content under-specification (text+speaker → 8192 BiCodec) is now the blocker; v7 adds source-GLM conditioning. AR+wait-k shows a usable latency–quality slope (n=32: BLEU 2.6→17.3 as k: 0→8) but RTF remains ~2.5–3. Route C data prep (SimAlign + NIR) may continue in parallel; it is not the decision fork.

## Evidence

### Step2 v3 blankpen (15-shard, mbs=64 / gbs=512, blank_penalty=1.0)

| Item | Result |
| --- | --- |
| Train | 3000 iters finished; valid `nar_ctc≈8.92`, `blank_mass≈0.176`, infeasible=0 |
| Greedy decode | **UER 100%**, empty predictions, blank-suppressed still ~100% |
| Conclusion | Soft blank mass alone does not fix argmax blank collapse |

### Step3 AR Pareto smoke (8 samples, Stage11 + lagging-k + Λ-window)

| k | Λ | BLEU | First WRITE ms | Fallback | RTF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1.14 | 680 | 0% | 3.90 |
| 0 | 512 | 1.27 | 680 | 0% | 2.87 |
| 2 | 0 | 3.14 | 2927 | 38% | 3.64 |
| 2 | 512 | 0.42 | 2927 | 38% | 3.14 |
| 4 | 0 | 6.90 | 4760 | 62% | 3.52 |
| 4 | 512 | 0.32 | 4760 | 62% | 2.88 |
| 8 | 0 | 9.52 | 6560 | 75% | 2.54 |
| 8 | 512 | 0.11 | 6560 | 75% | 2.55 |

Reading:

1. **wait-k works as a latency knob** — first WRITE ms and BLEU both rise with k (window=0).
2. **Λ-window=512 without RoPE reindex destroys quality at k≥2** — keep window=0 in shipping path until InfiniSST-style RoPE strip/reapply lands.
3. **RTF still >> 0.5** — confirms the original RTF diagnosis: AR target generation remains the bottleneck.

## Next actions (ordered)

1. **Step2 v4**: duration-guided CE + blank penalty (isolated under `simul_s2st_route_v1`), 15-shard, same mbs geometry; decode probe must leave blank collapse.
2. Keep Step3 harness on **lagging_k sweep with Λ=0** for quality curves; treat Λ>0 as research until RoPE-correct cache lands.
3. Only after NAR offline UER is non-degenerate: wire NAR into Step3 and re-draw LAAL–BLEU on CVSS-T.
4. Revisit B vs C only if (3) still sits far from the 2–3 s LAAL / competitive BLEU band.

## Safety

All Step2/3/4 code and reports remain under `experiments/simul_s2st_route_v1/` and `reports/simul_s2st_route_v1/`. Stage09–11 shipping trees and Phase3 joint trainers were not edited for this decision.
