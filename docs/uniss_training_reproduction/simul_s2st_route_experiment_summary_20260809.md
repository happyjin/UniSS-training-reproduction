# Simul-S2ST Route A — Experiment & Results Summary

> Date: 2026-08-09  
> Plan: `docs/uniss_training_reproduction/simul_s2st_route_decision_and_recommendation.md`  
> Isolation root: `experiments/simul_s2st_route_v1/` · reports: `reports/simul_s2st_route_v1/`  
> Constraint: no edits to shipping Stage09–11 / Phase3 joint trainers; all new work under the route tree.

## Restore status (this session)

- Accidental mass delete of `experiments/simul_s2st_route_v1/` and several route docs was restored via `git restore`.
- Recheck: `git ls-files -d` → **0** tracked deletions; key route code/docs/reports present.
- Unrelated dirty files (Stage09/11, web_demo, other experiments) were **not** touched for restore or this summary.

---

## Bottom line

| Gate | Status | Evidence |
|---|---|---|
| RTF bottleneck = target AR | Confirmed | Step0: Qwen AR ~52.5% of wall; compute RTF ~6.3 |
| V6 agreement as quality proxy | Demoted | Step1: agreement falls while BLEU stays usable |
| Reuse V6 NAR head | Dead | Step2b: all-blank, UER ~100% |
| Duration-anchored NAR quality | **Still red** | v1–v8; best non-blank is v6/v7 (~99% UER, 0% blank) |
| AR + wait-k latency knob | Partial green | n=32 Λ=0: BLEU 2.56→17.30 as k=0→8; RTF still ~2.5–3 |
| Open Route B / C | **No** | Step4: stay Route A |

**Current decision:** Stay Route A. Stop incremental NAR loss-knob chasing; either strengthen alignment/teacher supervision for NAR, or advance AR+wait-k (CVSS-T LAAL–BLEU, RoPE-correct Λ) before wiring NAR into the streaming path.

---

## Step 0 — Streaming wall-clock decomposition

- Run: `reports/simul_s2st_route_v1/step0_rtf_decomposition_v1.md`
- Baseline (16 samples): source 97.7s, wall 615.1s, **compute RTF 6.30**, first-audio NCA 4584 ms / CA 33538 ms.
- Wall shares (instrumented):

| Bucket | Share | Notes |
|---|---:|---|
| `qwen_ar_decode` | **52.5%** | Target-side AR (text + 50 Hz BiCodec) |
| `offline_fallback` | 34.7% | Final-only safety when no WRITE accepted |
| `qwen_prefill_source` | 8.0% | START_GLM + source + END_GLM |
| `source_runtime` | 2.4% | Stage09 frontend |
| Codec stream push | ~0.1% | Negligible |

**Reading:** Sub-second / RTF≪1 must cut target AR cost (NAR or shorter AR), not polish the source frontend alone.

---

## Step 1 — V6 agreement vs Phase3 BLEU (frozen backend)

- Run: `reports/simul_s2st_route_v1/step1_v6_bleu_recheck_v1.md`
- Backend fixed: `qwen0p5b_phase3_unist198_iter_0009075_hf`; 16 samples/direction.

Headline pattern: later Stage-B iters **lower agreement** but offline BLEU can stay high or even rise (e.g. EN→ZH offline BLEU ~40–45 while agreement drops from ~26% → ~13%).

**Reading:** Do not use V6 token agreement as the primary quality gate for streaming/NAR decisions; prefer BLEU / UER / LAAL.

---

## Step 2a / 2b — Duration anchor vs existing V6 NAR head

- **2a** (`step2a_upsample_ratio_v1`): duration / frame-budget anchoring preferred over naive upsample.
- **2b** (`step2b_existing_nar_head_v1`): V6 NAR CTC head under teacher GLM + ref translation → **UER 100%**, empty preds, blank frames ~100%; blank-suppressed UER still ~99.6–99.8%.

**Reading:** Old head is unusable; need a fresh duration-anchored NAR train (Step2).

---

## Step 2 — Duration-anchored NAR CTC (15-shard pilot)

**Data:** `data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/`  
**Geometry:** mbs=64 / gbs=512, typically 3000 iters  
**Ckpts:** `checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v*`  
**Code:** `experiments/simul_s2st_route_v1/step2_nar_ctc_head/`

Decode probe: teacher-forced Phase3 hidden + duration frame budget; 16 samples/direction; 2–10s audio.

### Iteration ledger (decode @ ~iter3000 unless noted)

| Run | Loss / conditioning idea | Blank frames | Empty | UER (approx) | Notes |
|---|---|---:|---:|---:|---|
| v1–v2 | CTC (+ duration) | ~100% | high | 100% | Loss fell; decode all-blank |
| v3 blankpen | `blank_penalty=1.0` | ~100% | high | 100% | Train blank_mass~0.17; argmax still blank |
| v4 guided CE | Duration-guided CE + blankpen | high | high | ~100% | CE ≈ ln(V); still blank |
| v5 CE-dominant | Stronger CE vs CTC | ~45–89% | ~0 | ~99.5–100% | Empty fixed; blank still heavy |
| **v6 speaker+CE-only** | BiCodec global speaker; CE-only | **0%** | **0** | **~98.9–99.9%** | Best blank fix; content still wrong |
| v7 +source GLM | Speaker + source GLM tokens | **0%** | **0** | ~99.2–99.9% | Slightly more pred units; UER still red |
| v8 unit-pooled CE | unit_ce=1.0, ctc=0.25, blankpen=0.5 | **~97–100%** | high | ~99.7–100% | **Regression** vs v6/v7; CE plateau ~8.56 |

### Diagnosis (locked)

1. Optimizer / head capacity OK (overfit / smoke can emit).
2. At 15-shard scale, frame-stretched or unit-pooled CE stays near-uniform over **8192** BiCodec classes → content under-specified / bad supervision, not “optimizer broken”.
3. Re-weighting CTC tends to **reintroduce blank collapse** (v8).
4. Blank collapse and content collapse are separable: v6/v7 killed blank without learning units.

### Best NAR checkpoint so far

- Prefer **v6 or v7** for any further NAR research (non-blank emissions).
- Neither passes a quality gate for streaming replacement of AR.

---

## Step 3 — AR + lagging-k + Λ-KV Pareto

**Code:** `experiments/simul_s2st_route_v1/step3_waitk_pareto/`  
**Fixes landed in harness only:** resolve direction from `tgt_lang`; keep `DynamicCache` type when pruning Λ-KV.

### Smoke8 (`step3_ar_pareto_smoke8_v3`)

| k | Λ | BLEU | First WRITE ms | Fallback | RTF |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.14 | 680 | 0% | 3.90 |
| 2 | 0 | 3.14 | 2927 | 38% | 3.64 |
| 4 | 0 | 6.90 | 4760 | 62% | 3.52 |
| 8 | 0 | 9.52 | 6560 | 75% | 2.54 |
| * | 512 | collapses at k≥2 | same | same | ~2.5–3 |

### n=32, Λ=0 (`step3_ar_pareto_n32_lambda0_v1`)

| k | BLEU | chrF | First WRITE ms | Fallback | RTF | LAAL proxy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.56 | 3.66 | 605 | 6% | 2.83 | 14.44 |
| 2 | 5.99 | 6.15 | 2980 | 38% | 2.83 | 52.79 |
| 4 | 14.64 | 13.56 | 5131 | 78% | 2.96 | 98.28 |
| 8 | 17.30 | 15.95 | 6560 | 94% | 2.54 | 122.50 |

**Reading:**

- wait-k is a real latency–quality knob (BLEU↑ with k; first WRITE↑; fallback↑).
- Λ=512 without RoPE reindex **destroys quality** → keep Λ=0 until InfiniSST-style RoPE strip/reapply.
- RTF still ≫ 0.5 → AR target path remains the compute bottleneck even when wait-k improves text BLEU.

---

## Step 4 — Route decision

Doc: `docs/uniss_training_reproduction/simul_s2st_route_execution_report_step4_decision.md`

**Verdict: Stay Route A. Do not open Route B (FAST student retrain) or Route C (Thinker–Talker) as the main fork.**

Parallel-allowed: Route C data prep (SimAlign + NIR) only as optional prep, not as the decision.

Ordered next (from plan + post-v8 evidence):

1. Stop small NAR loss-weight iterations; need stronger alignment / teacher unit supervision (or accept NAR gate remains red).
2. Advance Step3 on CVSS-T with LAAL–BLEU (Λ=0); treat Λ>0 as research until RoPE-correct cache.
3. Wire NAR into Step3 **only after** offline UER is non-degenerate.
4. Revisit B vs C only if (3) still far from ~2–3 s LAAL / competitive BLEU.

---

## Artefact map

| Kind | Path |
|---|---|
| Plan | `docs/uniss_training_reproduction/simul_s2st_route_decision_and_recommendation.md` |
| Step execution reports | `docs/uniss_training_reproduction/simul_s2st_route_execution_report_*.md` |
| Metric reports | `reports/simul_s2st_route_v1/step*.md` |
| NAR trainer / head | `experiments/simul_s2st_route_v1/step2_nar_ctc_head/` |
| AR Pareto harness | `experiments/simul_s2st_route_v1/step3_waitk_pareto/` |
| NAR checkpoints | `checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v{1..8}/` |
| Pilot data | `data/processed/phase3_whisper_streamspeech_joint_v5/pilot_15shard_joint/` |
| Phase3 backend | `checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf` |

---

## Ops notes (for reproducibility)

- Python: `/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python`
- `PYTHONPATH` must include repo root + `third_party/Megatron-LM`
- Post-train chains: gate on Megatron process exit, not tmux `sleep`
- Push target for route work: `private` remote `HEAD:main` (not `origin` / cmots)
- Speaker bug fixed in v6 path: use padded `batch["bicodec_global"]` tensor, not list fields from `batch_fields`

---

## One-line status for the next session

Route A evidence is complete through Step0–4 and NAR v1–v8: **blank solved in v6/v7, content not; AR wait-k slope exists but RTF and NAR gates still fail shipping criteria.** No tracked files remain deleted after restore; this document is the consolidated snapshot.
