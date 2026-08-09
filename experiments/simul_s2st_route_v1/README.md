# simul_s2st_route_v1

Isolated execution tree for the route decided in
[`docs/uniss_training_reproduction/simul_s2st_route_decision_and_recommendation.md`](../../docs/uniss_training_reproduction/simul_s2st_route_decision_and_recommendation.md).

Hard constraint for everything under this directory: **no file outside this tree is modified.**
The existing Stage09/10/11 runtime is consumed as a library and instrumented at runtime through
monkey patching that is installed and removed inside a single process.

## Layout

| Path | Purpose |
| --- | --- |
| `common/instrumentation.py` | Re-entrant call-tree wall-clock profiler with CUDA synchronisation |
| `step0_rtf_decomposition/` | Step 0 — split end-to-end wall clock into source frontend / policy / Qwen prefill / Qwen AR decode / BiCodec decode / IO |
| `step1_v6_bleu_recheck/` | Step 1 (D1) — frozen-Phase3 bidirectional Text-BLEU probe over joint-V6 checkpoints |
| `step2_nar_ctc_head/` | Step 2 — duration-anchored causal NAR CTC head (Megatron, Qwen frozen) |
| `step3_waitk_pareto/` | Step 3 — Student-v2 wait-k + Λ-KV; LAAL–BLEU Pareto (AR first while NAR is blank-collapsed) |

## Conventions

- Reports are written to `reports/simul_s2st_route_v1/` (tracked by git).
- Bulky artefacts (WAV, per-request JSON) go to `eval_outputs/simul_s2st_route_v1/` (git-ignored).
- Every runnable step has a `run.sh` that pins the conda env, `PYTHONPATH` and cache directories
  exactly like the existing `uniss_streamspeech_ctc_v1` stage scripts.
- Steps refuse to overwrite an existing report; pass a new `RUN_NAME`.

## GPU

The instance normally runs a synthetic GPU load holder
(`scripts/gpu_load/target_gpu_util.py`, tmux session `uniss_gpu_load_60`). Stop it before a
measurement run, otherwise every wall-clock number is contaminated:

```bash
tmux send-keys -t uniss_gpu_load_60 C-c
```

Restart it afterwards with the same arguments it was launched with:

```bash
python -u scripts/gpu_load/target_gpu_util.py --devices 0,1,2,3,4,5,6,7 \
  --target-util 60 --target-memory-percent 60 --cycle-seconds 1 \
  --matrix-size 16384 --dtype bfloat16 --sync-every 1 --log-interval 10
```
