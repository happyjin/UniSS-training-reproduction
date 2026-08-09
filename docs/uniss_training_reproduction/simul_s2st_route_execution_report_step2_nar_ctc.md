# Simul-S2ST route — Step 2 execution report (duration-anchored NAR CTC)

> Isolated under `experiments/simul_s2st_route_v1/`. Phase3 Qwen frozen. Machine: 8× H200.
> Companion decode report: [`step2_trained_nar_decode_v1.md`](../../reports/simul_s2st_route_v1/step2_trained_nar_decode_v1.md).

This completes the “build and train a NAR CTC head” item from
[`simul_s2st_route_decision_and_recommendation.md`](./simul_s2st_route_decision_and_recommendation.md)
§6 Step 2, after Steps 0/1/2a/2b showed that (a) AR decode dominates RTF, (b) the V6
constant-ratio head is not reusable, and (c) the frame budget must be duration-anchored.

---

## 1. What was built

| Piece | Path |
|---|---|
| Causal duration-anchored head | `experiments/simul_s2st_route_v1/step2_nar_ctc_head/duration_anchored_nar_ctc.py` |
| Indexed joint dataset + pad collate | `dataset.py` |
| Teacher-forced Phase3 hidden (batched) | `teacher_forced.py` |
| Megatron entry (Qwen frozen, hard CTC feasibility) | `pretrain_nar_ctc_megatron.py` |
| 15-shard launch | `run_15shard_8gpu.sh` |
| Unit-decode probe | `evaluate_trained_head.py` |

Geometry changes vs V6 `NARBiCodecCTC`:

* frames = `ceil(duration_s × 75)`, raised to the CTC feasibility floor;
* T2U encoder and unit decoder are both causal and padding-aware;
* text → frames via per-sample linear interpolation (no shared integer upsample).

---

## 2. Training configuration

Reused Phase3 joint Megatron knobs where they apply; batch geometry is different because
this job has no Whisper and only a 17M trainable head.

| Knob | Value | Source |
|---|---|---|
| Framework | Megatron-LM (`third_party/Megatron-LM`) | plan / Phase3 |
| GPUs | 8× H200, TP=PP=1 | Phase3 |
| Micro / global batch | **64 / 512** | probe (Phase3 used mbs=1/gbs=128 on a much heavier joint graph) |
| Optim | Adam β=(0.9, 0.98), eps=1e-8, clip=0.5, wd=0.01 | Phase3 joint |
| LR schedule | 2e-4 → 2e-5, warmup 100, inverse-square-root | Step2 head LR + Phase3 schedule style |
| Data | 15-shard joint train **1,319,793** / valid 7,793 | Phase3 V5 pilot |
| Iters | 3000 (≈1 epoch at gbs=512) | matched prior 12000×128 sample budget |
| Workers | 8, `no-data-sharding` | Phase3 |
| Backbone | frozen `qwen0p5b_phase3_unist198_iter_0009075_hf` | Phase3 export |

Run name: `step2_nar_ctc_15shard_v2_mbs64`.
Checkpoint: `checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v2_mbs64/iter_0003000`.
TensorBoard: port **6034**, logdir `runs/simul_s2st_route_v1/step2_nar_ctc_15shard_v2_mbs64`.

### 2.1 Throughput / util (why v1 looked idle)

| Run | mbs/gbs | ms/iter | util p50/max | power p50/max | mem |
|---|---|---:|---:|---:|---:|
| v1 (aborted) | 1/128 | ~710 | ~22% / — | ~140 W | ~10 GB |
| v2 (this run) | 64/512 | ~201 | **66% / 100%** | **396 / 440 W** | ~27 GB |
| Phase3 Stage B (reference) | 1/128 | ~8–9 s | ~100% | ~500 W | ~98 GB |

v1 was launch-overhead bound: 16 tiny frozen-Qwen forwards per step. v2 packs one fat
micro-batch per rank. Phase3’s sustained 500 W still comes from Whisper+joint training;
this head cannot match that without a heavier graph.

---

## 3. Loss curve

| Split | iter 1 / 100 | mid | iter 3000 |
|---|---:|---:|---:|
| train `nar_ctc` | 13.45 | 8.80 @1500 | **8.77** |
| valid `nar_ctc` | 9.49 @100 | — | **8.84** |
| `nar_infeasible` | 0 | 0 | **0** |

CTC path feasibility held for the whole run (hard-fail flag never tripped). Loss moved,
but see §4: the loss decrease is consistent with learning the blank-dominated CTC path,
not a usable unit stream.

---

## 4. Unit-decode probe (gate)

Best case: frozen Phase3, teacher GLM, reference translation, duration frames.
32 valid samples (16 per direction). Full table:
[`step2_trained_nar_decode_v1.md`](../../reports/simul_s2st_route_v1/step2_trained_nar_decode_v1.md).

| Checkpoint | UER (greedy) | Empty preds | Blank frames | Blank-suppressed UER | Distinct (blank-sup) |
|---|---:|---:|---:|---:|---:|
| iter 1000 | 100% | 32/32 | 100% | 99.7% | 4.7 |
| iter 2000 | 100% | 32/32 | 100% | 99.7% | 4.5 |
| iter 3000 | 100% | 32/32 | 100% | 99.7% | 6.2 |
| Step 2b V6 head (prior) | 100% | all | 100% | ~99.6% | ~few |

Control: a **randomly initialised** copy of the same head does *not* argmax-blank on noise
inputs. The trained checkpoint does. So one epoch did not leave the head untrained — it
actively learned the blank collapse that CTC rewards when content is hard.

**Plan gate (offline ASR-BLEU within 2 of Phase3 AR, RTF < 0.3): not met.** There is no
unit stream to decode to audio yet.

---

## 5. Consequences for the route

1. **Duration anchoring + causal padding + Megatron scaffolding are validated** (feasibility
   0, util restored, loss finite). The geometry from 2a is not the failure mode.
2. **One 15-shard epoch is not enough for BiCodec CTC content**, even with a frozen good
   teacher. Next training iterations (not blocking Step 3) should try blank-aware objectives
   (blank penalty / truncated frames / auxiliary unit CE), longer schedule, and/or a smaller
   effective blank prior — not another constant-ratio revive of V6.
3. **Step 3 should proceed on the AR Micro-WRITE path first** (Student v2 stability → wait-k
   → Λ-KV), then swap NAR in once a head emits non-blank units. This matches the plan’s
   “NAR head + wait-k series” while avoiding a Pareto curve that only measures silence.

---

## 6. Artefacts

| Kind | Path |
|---|---|
| Checkpoints | `checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v2_mbs64/` |
| Train log | `logs/simul_s2st_route_v1/step2_nar_ctc_15shard_v2_mbs64.log` |
| GPU util CSV | `logs/simul_s2st_route_v1/step2_nar_ctc_15shard_v2_mbs64_gpu_power_utility.csv` |
| Decode JSON/MD | `reports/simul_s2st_route_v1/step2_trained_nar_decode_v1.{json,md}` |
