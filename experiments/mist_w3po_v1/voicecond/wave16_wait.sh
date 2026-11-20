#!/usr/bin/env bash
# Robust wave 16 NCWI chain (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md section 5), detached:
#   wait for wave 14's CHAIN_DONE (CHAIN_FAILED, or a missing eval_unist / eval_unist_wise05 EVAL_DONE -> WAIT_FAILED, no
#   card touched) and for no live wave-14 worker in logs/mist_w3po_v1/wave14/w14_workers.pids (argv re-checked in /proc)
#   -> wave16_fit.py (CPU) -> wave16_make_lora.py for every alpha* and 0.5, with the re-merge check against wave 14's
#   merged_unist_wise05 -> own watcher wave16_noplaceholder.sh (own PIDS / STOP) -> wave16_orch.sh on cards 0-3 -> touch
#   STOP (cards back to the guard) -> wave16_score.py -> wave16/DONE (any failure: wave16/CHAIN16_FAILED + STOP).
# Never touches guard 2726951, gpu_watch 52045, sync 1222453; stops nothing itself; cards 4-7 untouched.
#   setsid nohup bash wave16_wait.sh > logs/mist_w3po_v1/wave16/wait.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/wave16
W14=eval_outputs/mist_w3po_v1/wave14
LOGD=logs/mist_w3po_v1/wave16
STOP=$LOGD/w16_noplaceholder.stop
PIDS=$LOGD/w16_workers.pids
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
V=experiments/mist_w3po_v1/voicecond
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN16_FAILED"; touch "$STOP"; exit 1; }
[ -e "$STOP" ] && fail "$STOP exists already"
for f in DONE CHAIN16_FAILED WAIT_FAILED ORCH_FAILED NCWI_ROOT fit lora summary.json; do [ -e "$M/$f" ] && fail "$M/$f exists"; done
log "wait $$: waiting for wave 14 ($W14/CHAIN_DONE)"
while :; do
  if [ -e "$W14/CHAIN_FAILED" ]; then log "wave 14 chain failed: $(cat "$W14/CHAIN_FAILED")"; echo "wave 14 CHAIN_FAILED" > "$M/WAIT_FAILED"; exit 1; fi
  [ -e "$W14/CHAIN_DONE" ] && break
  sleep 60
done
for a in unist unist_wise05; do
  [ -e "$W14/eval_$a/EVAL_DONE" ] || { log "wave 14 eval_$a incomplete"; echo "wave 14 eval_$a incomplete" > "$M/WAIT_FAILED"; exit 1; }
done
log "wave 14 chain done; waiting for its workers to leave"
while :; do
  live=""
  while read -r c p _; do
    [ -n "${p:-}" ] || continue
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$a" in *wave14_*|*torch.distributed.run*|*si_thinker_stream*|*si_infer*|*asr_transcribe*|*wave5_m_merge*) live="$live $p";; esac
  done < logs/mist_w3po_v1/wave14/w14_workers.pids
  [ -z "$live" ] && break
  log "live wave-14 workers:$live"
  sleep 60
done
log "cards 0-3 free of wave 14; fit (CPU)"
PYTHONPATH=. "$PYQ" $V/wave16_fit.py --out "$M/fit" > "$LOGD/fit.log" 2>&1 || fail "fit (see $LOGD/fit.log)"
tail -1 "$LOGD/fit.log"
AL=$("$PYQ" -c "import json; r=json.load(open('$M/fit/fit.json')); print(','.join(sorted({f'{x:.4f}' for d in ('en2zh','zh2en') for x in r[d]['distinct_alphas']} | {'0.5000'})))") || fail "alphas"
log "adapters for alphas $AL"
PYTHONPATH=. "$PYQ" $V/wave16_make_lora.py --adapters "$W14/run_unist/adapters_final.pt" --alphas "$AL" --out "$M/lora" \
  --check-merged "$W14/merged_unist_wise05" --check-alpha 0.5 > "$LOGD/make_lora.log" 2>&1 || fail "make_lora (see $LOGD/make_lora.log)"
tail -1 "$LOGD/make_lora.log"
PIDS=$PIDS STOP=$STOP setsid nohup bash $V/wave16_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"
bash $V/wave16_orch.sh > "$LOGD/orch.log" 2>&1 < /dev/null
r=$?
touch "$STOP"
log "orchestrator exit $r; watcher STOP touched, cards 0-3 back to the guard"
[ "$r" = 0 ] || fail "orchestrator (see $LOGD/orch.log)"
log "scoring"
"$PYQ" $V/wave16_score.py --out "$M/summary.json" > "$LOGD/score.log" 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/DONE"
log "wave 16 done"
