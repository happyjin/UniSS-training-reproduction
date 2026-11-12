#!/usr/bin/env bash
# RC5 training launcher, v2 (start_rc5.sh unchanged; v2 reads crf/rc5_precheck_v2.json, whose unit tests ran with the
# CPU accelerator patch -- in v1 test_train_crf_sft could not import transformers without it -- and has the narrow
# waiver below). reports/mist_w3po_v1/thinker_rc5_prereg_v1.zh-CN.md §1, §3: T5 unchanged (latent_write --lw-mode t5,
# evidence-time labels, NIR filter, DART, 1.0 s reads, 3000 steps, 4 cards) except the granularity pieces4rc: adjacent
# same-label-read pieces4 pieces of an utterance coalesced (sched_t5_rc_w10_v1, manifest t5rc_dart_w10_v1).
#   launch condition (one of):
#     precheck pass                                   -> RC5
#     WAIVE_R2_LATHEUR=1: only R2's lat_heur clause failed (R1, R3, R4 pass, R2 word-level label-read share >= 0.999)
#                         and the waiver note logs/mist_w3po_v1/rc5/waiver_r2_latheur.txt exists (who decided, why)
#                                                     -> RC5 (documented waiver)
#     FALLBACK=1 and the precheck failed              -> LM3 = the same objective + DART at merge3 (sched_t5_merge3_w10_v1,
#                                                        manifest t5_dart_merge3_w10_v1): prereg §6, an ablation of VE1
# Waits until the slot of CARDS is free (no slot owner, only our placeholders on its cards), claims it (noclobber),
# takes it over with takeover_train.sh (our placeholders stopped by PID, launch in the same breath), checks 4 ranks.
#   RUN_ID=rc5_<UTC> CARDS=4,5,6,7|0,1,2,3 [PORT=29991] [WAIVE_R2_LATHEUR=1 | FALLBACK=1] [DRY_RUN=1] \
#     bash experiments/mist_w3po_v1/scripts/start_rc5_v2.sh
# Run detached: setsid nohup bash .../start_rc5_v2.sh > logs/mist_w3po_v1/rc5/start_rc5.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
CARDS=${CARDS:?}
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
C=eval_outputs/mist_w3po_v1/crf
PC=$C/rc5_precheck_v2.json
WV=logs/mist_w3po_v1/rc5/waiver_r2_latheur.txt
SL=logs/mist_w3po_v1/slots
case $CARDS in 0,1,2,3) SLOT=slotA;; 4,5,6,7) SLOT=slotB;; *) echo "CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
log() { echo "$(date -u +%FT%TZ) $*"; }
field() { $PY -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1], {}, {'d': d}))" "$2"; }
busy_on_cards() {   # compute processes on CARDS that are not our placeholders
  local map pid bus idx cmd
  map=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null) || { echo "smi-failed"; return; }
  nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | while IFS=', ' read -r pid bus; do
    idx=$(echo "$map" | awk -F', ' -v b="$bus" '$2 == b {print $1}')
    case ",$CARDS," in *",$idx,"*) ;; *) continue;; esac
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || cmd="(unreadable)"
    case "$cmd" in *scripts/gpu_load/train_mist_workload.py*|*gpu_keeper.py*) ;; *) echo "$idx:$pid";; esac
  done
}
if [ "${FALLBACK:-}" = 1 ]; then
  NAME="LM3 (fallback ablation)"; S=$C/sched_t5_merge3_w10_v1; D=$C/t5_dart_merge3_w10_v1
else
  NAME=RC5; S=$C/sched_t5_rc_w10_v1; D=$C/t5rc_dart_w10_v1
fi
EXTRA="--sched $S --objective latent_write --lw-mode t5 --dart $D"
LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=${PORT:-29991} EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
PASS=$(field $PC "d['pass']" 2>/dev/null || echo missing)
ONLY_LH=$(field $PC "d['R1']['pass'] and d['R3']['pass'] and d['R4']['pass'] and d['R2']['share'] >= 0.999 and not d['R2']['pass']" 2>/dev/null || echo missing)
if [ -n "${DRY_RUN:-}" ]; then
  echo "$NAME; precheck pass: $PASS; only R2's lat_heur clause failed: $ONLY_LH; waiver note: $([ -s $WV ] && echo present || echo absent); WAIVE_R2_LATHEUR=${WAIVE_R2_LATHEUR:-}"
  echo "data: $([ -s $S/summary.json ] && echo ok || echo MISSING); manifest: $([ -s $D/summary.json ] && echo ok || echo MISSING)"
  echo "slot $SLOT owner: $([ -e $SL/$SLOT.owner ] && head -c 150 $SL/$SLOT.owner || echo none)"
  echo "non-placeholder compute processes on $CARDS: $(busy_on_cards | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
  eval "DRY_RUN=1 $LAUNCH"
  exit 0
fi
if [ "${FALLBACK:-}" = 1 ]; then
  [ "$PASS" = False ] || { log "FALLBACK=1 but RC5's precheck did not fail (pass: $PASS): not launched"; exit 1; }
elif [ "$PASS" != True ]; then
  if [ "${WAIVE_R2_LATHEUR:-}" = 1 ] && [ "$ONLY_LH" = True ] && [ -s $WV ]; then
    log "precheck: only R2's lat_heur clause failed; documented waiver $WV: $(head -c 300 $WV | tr '\n' ' ')"
  else
    log "RC5 precheck failed (pass: $PASS; only lat_heur: $ONLY_LH; waiver flag: ${WAIVE_R2_LATHEUR:-} note: $([ -s $WV ] && echo yes || echo no)): not launched"
    exit 1
  fi
fi
[ -s $S/summary.json ] && [ -s $D/summary.json ] || { log "data or manifest missing: not launched"; exit 1; }
log "$NAME: waiting for $SLOT ($CARDS) to hold only placeholders"
while :; do
  B=$(busy_on_cards)
  [ ! -e $SL/$SLOT.owner ] && [ -z "$B" ] && break
  sleep 15
done
( set -o noclobber; printf 'who: method-design executor v2 (mist_w3po_v1 %s)\ntask: %s training %s (%s) on cards %s, ~4-4.5 h; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$NAME" "$NAME" "$RUN_ID" "$EXTRA" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $SL/$SLOT.owner ) && log "$SLOT.owner claimed" \
  || { log "$SLOT.owner claim failed: not launched"; exit 1; }
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "$NAME ranks running: $n"
