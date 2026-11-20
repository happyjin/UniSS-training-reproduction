#!/usr/bin/env bash
# Robust wave 16 NCWI, group ncwi_erm (change B), detached: CPU first -- wave16e_fit.py (CoVoST ERM endpoints, b-hat and
# bin edges of wave16/noise reused) and wave16_make_lora.py on wave5/M/run_erm with the re-merge check against
# wave5/M/merged_erm_wise05 -- then wait until the UniST group's GPU part is over (wave16 watcher STOP touched, or
# CHAIN16_FAILED / WAIT_FAILED) and no live wave-16 worker of logs/mist_w3po_v1/wave16/w16_workers.pids (argv re-checked)
# -> own watcher (PIDS w16e_workers.pids, STOP w16e_noplaceholder.stop) -> wave16e_orch.sh -> touch STOP ->
# wave16e_score.py -> wave16/erm/DONE (failure: wave16/erm/CHAIN16E_FAILED + STOP).
# Never touches guard 2726951, gpu_watch 52045, sync 1222453; stops nothing itself; cards 6-7 never used.
#   setsid nohup bash wave16e_wait.sh > logs/mist_w3po_v1/wave16/erm/wait.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/wave16/erm
M16=eval_outputs/mist_w3po_v1/wave16
W5M=eval_outputs/mist_w3po_v1/wave5/M
LOGD=logs/mist_w3po_v1/wave16/erm
STOP=$LOGD/w16e_noplaceholder.stop
PIDS=$LOGD/w16e_workers.pids
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
V=experiments/mist_w3po_v1/voicecond
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; mkdir -p "$M"; echo "$*" > "$M/CHAIN16E_FAILED"; touch "$STOP"; exit 1; }
mkdir -p "$LOGD"
[ -e "$STOP" ] && fail "$STOP exists already"
for f in fit lora DONE summary.json NCWI_ROOT; do [ -e "$M/$f" ] && fail "$M/$f exists"; done
mkdir -p "$M"
log "wait $$: ncwi_erm fit (CPU)"
PYTHONPATH=. "$PYQ" $V/wave16e_fit.py --out "$M/fit" > "$LOGD/fit.log" 2>&1 || fail "fit (see $LOGD/fit.log)"
tail -2 "$LOGD/fit.log"
AL=$("$PYQ" -c "import json; r=json.load(open('$M/fit/fit.json')); print(','.join(sorted({f'{x:.4f}' for d in ('en2zh','zh2en') for x in r[d]['distinct_alphas']} | {'0.5000'})))") || fail "alphas"
log "adapters for alphas $AL"
PYTHONPATH=. "$PYQ" $V/wave16_make_lora.py --adapters "$W5M/run_erm/adapters_final.pt" --alphas "$AL" --out "$M/lora" \
  --check-merged "$W5M/merged_erm_wise05" --check-alpha 0.5 > "$LOGD/make_lora.log" 2>&1 || fail "make_lora (see $LOGD/make_lora.log)"
tail -1 "$LOGD/make_lora.log"
log "waiting for the UniST group's GPU part to end"
while :; do
  { [ -e logs/mist_w3po_v1/wave16/w16_noplaceholder.stop ] || [ -e "$M16/CHAIN16_FAILED" ] || [ -e "$M16/WAIT_FAILED" ]; } && break
  sleep 30
done
while :; do
  live=""
  while read -r c p _; do
    [ -n "${p:-}" ] || continue
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$a" in *wave16_*|*si_thinker_stream*|*si_infer*|*asr_transcribe*|*wave5_m_merge*) live="$live $p";; esac
  done < <(cat logs/mist_w3po_v1/wave16/w16_workers.pids 2>/dev/null)
  [ -z "$live" ] && break
  log "live UniST-group workers:$live"
  sleep 30
done
log "UniST group's GPU part over; ncwi_erm on the cards"
PIDS=$PIDS STOP=$STOP setsid nohup bash $V/wave16_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"
bash $V/wave16e_orch.sh > "$LOGD/orch.log" 2>&1 < /dev/null
r=$?
touch "$STOP"
log "orchestrator exit $r; watcher STOP touched, cards back to the guard"
[ "$r" = 0 ] || fail "orchestrator (see $LOGD/orch.log)"
log "scoring"
"$PYQ" $V/wave16e_score.py --out "$M/summary.json" > "$LOGD/score.log" 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/DONE"
log "ncwi_erm done"
