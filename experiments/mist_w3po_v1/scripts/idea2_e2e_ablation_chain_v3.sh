#!/usr/bin/env bash
# v3 (2026-11-19 ~09:50 Prague, coordinator / user: start everything not yet training now): the chain on cards 0-3 only
# FINISHES frozen_talker. It adopts v2's slot-A owner file and v2's running frozen_talker torchrun (ADOPT_PID, default
# 2349396; awaited, never relaunched), writes the run's LAUNCHER_RC and the chain marker, then releases slot A.
# constlam is NOT launched here (it runs on cards 4-7 via scripts/idea2_e2e_constlam_run.sh). v2's bash is stopped with
# SIGKILL after v3 is armed (a SIGTERM would fire v2's EXIT trap after its child exits and release slot A early).
#   [ADOPT_PID=<frozen_talker torchrun>] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_e2e_ablation_chain_v3.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/e2e_ablation_chain_v3.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
LOGD=logs/mist_w3po_v1/voicecond/idea2
SL=logs/mist_w3po_v1/slots
E=eval_outputs/mist_w3po_v1/voicecond_idea2/e2e
RUN=idea2_e2e_abl_frozentalker_v1
OUT=$E/$RUN
MARK=$LOGD/E2E_ABLATION_CHAIN_DONE
ADOPT_PID=${ADOPT_PID:-2349396}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
dry() { [ -n "${DRY_RUN:-}" ]; }
OWN=""
release() {
  [ -n "$OWN" ] && [ -e "$OWN" ] && grep -q "Idea 2 e2e ablation chain" "$OWN" && \
    command mv -n "$OWN" $SL/slotA.released_$(date -u +%Y%m%dT%H%M%SZ) && log "slot A released"
  OWN=""
}
if [ -e $SL/slotA.owner ] && grep -q "Idea 2 e2e ablation chain" $SL/slotA.owner; then
  if dry; then log "[dry] would adopt the chain's slot A owner file"
  else echo "adopted: chain v3 pid $$ at $(date -u +%FT%TZ)" >> $SL/slotA.owner; OWN=$SL/slotA.owner; log "adopted slot A"; fi
else log "no chain slot-A owner file to adopt (continuing; nothing to release)"; fi
trap 'release' EXIT
trap 'log "signal received"; exit 1' INT TERM HUP
a=$(tr '\0' ' ' < /proc/$ADOPT_PID/cmdline 2>/dev/null)
if [ -e $OUT/DONE ]; then log "$RUN already DONE"
elif case "$a" in *torchrun*"--out $OUT "*) true;; *) false;; esac; then
  log "$RUN: adopting running torchrun $ADOPT_PID"
  dry || while kill -0 $ADOPT_PID 2>/dev/null; do sleep 30; done
else log "$RUN: pid $ADOPT_PID is not this run's torchrun ('${a:0:80}')"; fi
dry && { log "[dry] would mark $RUN and release slot A"; exit 0; }
st=FAILED; [ -e $OUT/DONE ] && st=DONE
echo "adopted-v3 $(date -u +%FT%TZ)" >> $OUT/LAUNCHER_RC
echo "$RUN - $st" >> $MARK
log "$RUN: $st; chain on cards 0-3 finished (constlam runs on 4-7 separately)"
