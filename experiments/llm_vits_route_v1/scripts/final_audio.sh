#!/bin/bash
# Render the val-chosen policy operating points on test (results_v2.json: T3 eps 1.0, T3-lite eps 1.5, entropy 4.25),
# then audio metrics (Whisper WER, UTMOS, word log-F0) for every rendered test system, one card per system.
#   final_audio.sh <backbone_ckpt> <exp_dir> <cards>
set -euo pipefail
CK="${1:?backbone ckpt}"; EXP="${2:?exp dir}"; CARDS="${3:?cards}"
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
NW=0.8
cd "$ROOT"
log() { echo "$(date -u +%H:%M:%S) $*"; }
IFS=, read -ra C <<< "$CARDS"
run_jobs() {
  $PY scripts/run_jobs.py "$1" --cards "$CARDS" --per-card "$2" --log-dir "$EXP/joblogs" >> "$EXP/jobs_audio.log" 2>&1
  if grep -q FAIL "$EXP/jobs_audio.log"; then log "job failure, see $EXP/jobs_audio.log"; exit 1; fi
}
mk() { local out=$1; shift; [ -f "$out" ] || $PY scripts/make_jobs.py --exp "$EXP" --backbone-ckpt "$CK" --noise-w $NW --out "$out" "$@"; }

log "render chosen policies"
mk "$EXP/jobs_render_policy.json" --stage sweep --model s6_future --split test --shards 4 --render --tag _render \
  --entropy 4.25 --t3 1.0 --risk "$EXP/t3/risk_s6f.pt"
mk "$EXP/jobs_render_policy_lite.json" --stage sweep --model s6_future --split test --shards 4 --render --tag _render_lite \
  --t3 1.5 --risk "$EXP/t3/risk_s6f_lite.pt"
run_jobs "$EXP/jobs_render_policy.json" 2
run_jobs "$EXP/jobs_render_policy_lite.json" 4

log "audio metrics"
n=0
for d in "$EXP"/eval_render_test/* "$EXP"/eval_render_lite_test/*; do
  [ -f "$d/audio_metrics.json" ] && continue
  CUDA_VISIBLE_DEVICES=${C[$((n % ${#C[@]}))]} $PY scripts/eval_audio.py "$d" --align-dir "$EXP/align" > "$d/eval_audio.log" 2>&1 &
  n=$((n + 1))
done
wait
for d in "$EXP"/eval_render_test/* "$EXP"/eval_render_lite_test/*; do
  [ -f "$d/audio_metrics.json" ] || { log "no audio metrics in $d"; exit 1; }
done
log "AUDIO_DONE"
