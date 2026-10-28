#!/bin/bash
# Final run on the converged E0 backbone. Every stage skips work whose outputs exist, so the driver can be resumed.
#   final_driver.sh <backbone_ckpt> <exp_dir> <cards e.g. 0,1,2,3,4,5,6,7>
set -euo pipefail
CK="${1:?backbone ckpt}"; EXP="${2:?exp dir}"; CARDS="${3:?cards}"
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
CFG=$ROOT/configs/ljs_char_base.json
NW=0.8
cd "$ROOT"
mkdir -p "$EXP"
log() { echo "$(date -u +%H:%M:%S) $*"; }
IFS=, read -ra C <<< "$CARDS"

log "stage 1: d* labels"
[ -f "$EXP/align/durations.npz" ] || CUDA_VISIBLE_DEVICES=${C[0]} NUMBA_NUM_THREADS=16 $PY scripts/align_gt.py \
  --manifest data/manifest_v2.jsonl --ckpt "$CK" --config "$CFG" --out "$EXP/align" > "$EXP/align.log" 2>&1

log "stage 2: fine-tunes"
RUNS=(e1:17 e2:17 s6_future:17 e1:29 e2:29 s6_future:29 s6_rho:17 s6:17)
for n in "${!RUNS[@]}"; do
  v=${RUNS[$n]%%:*}; s=${RUNS[$n]##*:}
  d="$EXP/$v"; [ "$s" = 17 ] || d="$EXP/${v}_seed$s"
  [ -f "$d/best.json" ] || [ -f "$d/train.log" ] || bash scripts/launch_stream.sh "${C[$((n % ${#C[@]}))]}" "$v" "$CK" "$EXP/align" "$d" \
    --steps 8000 --val-every 1000 --seed "$s"
done
for n in "${!RUNS[@]}"; do
  v=${RUNS[$n]%%:*}; s=${RUNS[$n]##*:}
  d="$EXP/$v"; [ "$s" = 17 ] || d="$EXP/${v}_seed$s"
  until [ -f "$d/best.json" ]; do
    if grep -q Traceback "$d/train.log" 2>/dev/null; then log "FAILED $d"; exit 1; fi
    sleep 30
  done
done
log "fine-tunes done"

run_jobs() {  # jobs.json per-card
  $PY scripts/run_jobs.py "$1" --cards "$CARDS" --per-card "$2" --log-dir "$EXP/joblogs" >> "$EXP/jobs.log" 2>&1
  if grep -q FAIL "$EXP/jobs.log"; then log "job failure, see $EXP/jobs.log"; exit 1; fi
}
mk() { local out=$1; shift; [ -f "$out" ] || $PY scripts/make_jobs.py --exp "$EXP" --backbone-ckpt "$CK" --noise-w $NW --out "$out" "$@"; }

log "stage 3: fixed lookahead, test + val"
for split in test val; do
  mk "$EXP/jobs_fixed_$split.json" --stage fixed --models e1,e2,s6_future,s6_rho,s6,e1_seed29,e2_seed29,s6_future_seed29 \
    --ks 0,1,2,3,99 --split $split --shards 4
  run_jobs "$EXP/jobs_fixed_$split.json" 6
done

log "stage 4: T3 dumps and fits (S6-future)"
mk "$EXP/jobs_dump.json" --stage dump --model s6_future --train-limit 1500 --shards 8
run_jobs "$EXP/jobs_dump.json" 2
for lite in "" "--lite"; do
  out="$EXP/t3/risk_s6f${lite:+_lite}.pt"
  [ -f "$out" ] || $PY scripts/fit_t3.py --train-dump "$EXP/t3/dump_s6_future_train" --calib-dump "$EXP/t3/dump_s6_future_calib" \
    --align-dir "$EXP/align" --target full $lite --out "$out" > "${out%.pt}.log" 2>&1
done

log "stage 5: policy sweeps, val + test"
for split in val test; do
  mk "$EXP/jobs_sweep_$split.json" --stage sweep --model s6_future --split $split --shards 4 \
    --entropy 2.5,3.5,4.25,5,7 --spread 0.01,0.02,0.04 --t3 0.4,0.5,0.6,0.8,1.0 --risk "$EXP/t3/risk_s6f.pt"
  mk "$EXP/jobs_sweep_lite_$split.json" --stage sweep --model s6_future --split $split --shards 4 --tag _lite \
    --t3 0.4,0.5,0.6,0.8,1.0 --risk "$EXP/t3/risk_s6f_lite.pt"
  run_jobs "$EXP/jobs_sweep_$split.json" 6
  run_jobs "$EXP/jobs_sweep_lite_$split.json" 6
done
log "ALL_STAGES_DONE"
