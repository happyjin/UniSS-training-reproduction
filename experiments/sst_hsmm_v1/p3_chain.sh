#!/usr/bin/env bash
# sst_hsmm_v1 overnight chain (user asleep 2026-11-22 night, best-option autonomy): waits for P2 to end; P2 FAILED or
# NO-GO -> log and stop (no P3, no tuning; prereg change 2). GO -> wait for the report-only P1a diagnostic to end
# (p1/diag/DONE or FAILED; it must not share cards with P3), then start p3_orch.sh detached. P1b skipped (optional,
# T1c recipe not recoverable). Starts nothing else. Checked with bash -n / DRY_RUN only.
#   setsid nohup bash experiments/sst_hsmm_v1/p3_chain.sh > logs/sst_hsmm_v1/p3_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
P1=eval_outputs/sst_hsmm_v1/p1; P2=eval_outputs/sst_hsmm_v1/p2
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ -n "${DRY_RUN:-}" ]; then log "[dry] wait P2 -> GO? -> wait diag -> p3_orch.sh"; exit 0; fi
log "chain $$: waiting for P2"
until [ -e $P2/DONE ] || [ -e $P2/FAILED ]; do [ -e $P1/FAILED ] && { log "P1 FAILED: stop"; exit 1; }; sleep 60; done
[ -e $P2/FAILED ] && { log "P2 FAILED: stop"; exit 1; }
go=$(python3 -c "import json; print(int(json.load(open('$P2/summary.json'))['verdict']['GO']))" 2>/dev/null)
[ "$go" = 1 ] || { log "P2 NO-GO (go=$go): stop, no P3"; exit 0; }
log "P2 GO: waiting for the diagnostic"
until [ -e $P1/diag/DONE ] || [ -e $P1/diag/FAILED ]; do sleep 60; done
log "diag ended ($(ls $P1/diag | grep -E '^(DONE|FAILED)$')): starting P3"
setsid nohup bash experiments/sst_hsmm_v1/p3_orch.sh > logs/sst_hsmm_v1/p3_orch.log 2>&1 < /dev/null 8>&- 9>&- &
log "p3_orch pid $!"
