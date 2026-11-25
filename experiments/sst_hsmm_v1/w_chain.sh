#!/usr/bin/env bash
# prereg change 26 chain: dv2 (SVC-JM dev, Thinker only) -> p3w (SVC-JM test, full pipeline; runs regardless of dev).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/w_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] dv2 -> p3w"; exit 0; }
[ -e $L/w_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
log "chain $$: dv2"
bash experiments/sst_hsmm_v1/dv2_orch.sh > $L/dv2_orch.log 2>&1 < /dev/null; [ -e $D/dv2/DONE ] || fail "dv2"
log "test"
bash experiments/sst_hsmm_v1/p3w_report.sh > $L/p3w_report.log 2>&1 < /dev/null; [ -e $D/p3w/DONE ] || fail "p3w"
log "chain done"
