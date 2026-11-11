#!/usr/bin/env bash
# T5 on cards 0-3 (prereg §T5, amendment 1, line-4 waiver): waits (event poll every 10 s) until
#   1. the read period is decided (logs/mist_w3po_v1/t4/read_period.txt = 500 or 1000, T1-ctrl rule),
#   2. every T2-era eval orchestrator on 0-3 (t_eval/{t1,t1ctrl,t2,t2plem}) has events/DONE and no live process,
# then claims slotA (noclobber), stops our placeholders on 0-3 by PID (takeover_train.sh: supervisor first) and launches
# T5 at once (run_crf_v2p.sh --objective latent_write --lw-mode t5 --dart <T5 manifest>). bash -n / DRY_RUN only.
#   RUN_ID=<id> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_t5.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
C=eval_outputs/mist_w3po_v1/crf
E=logs/mist_w3po_v1/t_eval
SL=logs/mist_w3po_v1/slots
RP=logs/mist_w3po_v1/t4/read_period.txt
log() { echo "$(date -u +%FT%TZ) $*"; }
evals_done() {
  for r in t1 t1ctrl t2 t2plem; do
    [ -e $E/$r/events/DONE ] || return 1
    pgrep -f "thinker_t_eval_orch.py --run $E/$r " > /dev/null && return 1
  done
  return 0
}
if [ -z "${DRY_RUN:-}" ]; then
  until [ -s $RP ]; do sleep 10; done
fi
G=$(cat $RP 2>/dev/null || echo 500)
case "$G" in 500) S=$C/sched_t5_pieces4_w05_v1; D=$C/t5_dart_pieces4_w05_v1;; 1000) S=$C/sched_t5_pieces4_w10_v1; D=$C/t5_dart_pieces4_w10_v1;;
  *) log "bad read period '$G'"; exit 1;; esac
[ -e $S/summary.json ] && [ -e $D/summary.json ] || { log "missing $S or $D"; exit 1; }
EXTRA="--sched $S --objective latent_write --lw-mode t5 --dart $D"
LAUNCH="CARDS=0,1,2,3 RUN_ID=$RUN_ID PORT=29971 EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
if [ -n "${DRY_RUN:-}" ]; then echo "read period $G; LAUNCH: $LAUNCH"; evals_done && echo "evals done" || echo "evals running"; exit 0; fi
log "read period $G ms: $S, $D; waiting for the T2-era evals on 0-3"
until evals_done; do sleep 10; done
log "T2-era evals on 0-3 done"
( set -o noclobber; printf 'who: thinker-side executor (mist_w3po_v1 T4/T5)\ntask: T5 training %s (evidence-time labels, pieces4, DART, t5 lattice; %s) on cards 0-3, ~5-6 h; released by renaming to slotA.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$S" "$(date -u +%FT%TZ)" > $SL/slotA.owner ) && log "slotA.owner claimed for T5" || log "slotA.owner claim FAILED (someone holds slotA): launching anyway is NOT done"
[ -e $SL/slotA.owner ] && grep -q "$RUN_ID" $SL/slotA.owner || exit 1
CARDS=0,1,2,3 LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "T5 ranks running: $n"
