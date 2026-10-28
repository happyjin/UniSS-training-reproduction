#!/bin/bash
# Follow-up to final_driver.sh: the converged backbone made the risk head and the spread probe more conservative, so
# the first grid never reached the 0.25 / 0.5 lookahead targets. Extend T3 / T3-lite / spread thresholds on val and
# test, and render waveforms of the fixed-lookahead test systems. Outputs that exist are skipped.
#   final_extend.sh <backbone_ckpt> <exp_dir> <cards>
set -euo pipefail
CK="${1:?backbone ckpt}"; EXP="${2:?exp dir}"; CARDS="${3:?cards}"
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
NW=0.8
cd "$ROOT"
log() { echo "$(date -u +%H:%M:%S) $*"; }
run_jobs() {
  $PY scripts/run_jobs.py "$1" --cards "$CARDS" --per-card "$2" --log-dir "$EXP/joblogs" >> "$EXP/jobs_extend.log" 2>&1
  if grep -q FAIL "$EXP/jobs_extend.log"; then log "job failure, see $EXP/jobs_extend.log"; exit 1; fi
}
mk() { local out=$1; shift; [ -f "$out" ] || $PY scripts/make_jobs.py --exp "$EXP" --backbone-ckpt "$CK" --noise-w $NW --out "$out" "$@"; }

log "extended sweeps, val + test"
for split in val test; do
  mk "$EXP/jobs_sweep2_$split.json" --stage sweep --model s6_future --split $split --shards 4 \
    --spread 0.08,0.16,0.32 --t3 1.5,2,3,4,6 --risk "$EXP/t3/risk_s6f.pt"
  mk "$EXP/jobs_sweep2_lite_$split.json" --stage sweep --model s6_future --split $split --shards 4 --tag _lite \
    --t3 1.5,2,3,4,6 --risk "$EXP/t3/risk_s6f_lite.pt"
  run_jobs "$EXP/jobs_sweep2_$split.json" 6
  run_jobs "$EXP/jobs_sweep2_lite_$split.json" 6
done

log "render fixed-lookahead test systems"
mk "$EXP/jobs_render_fixed.json" --stage fixed --models e1,e2,s6_future --ks 0,1 --split test --shards 4 --render --tag _render
run_jobs "$EXP/jobs_render_fixed.json" 4
log "EXTEND_DONE"
