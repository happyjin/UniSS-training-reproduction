#!/usr/bin/env bash
# prereg change 21 (DLA) chain: q_orch (smoke, train, merge) -> dev p2q (LA2 vs wave17 LA2) -> test p3q (combo vs N, report only).
#   setsid nohup bash experiments/sst_hsmm_v1/q_chain.sh > logs/sst_hsmm_v1/q_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/q_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] q_orch -> p2q -> p3q"; exit 0; }
[ -e $L/q_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until [ -e $D/ceiling/DONE ] || [ -e $D/ceiling/FAILED ]; do sleep 30; done      # cards: the ceiling run first
log "chain $$: train"
bash experiments/sst_hsmm_v1/q_orch.sh > $L/q_orch.log 2>&1 < /dev/null; [ -e $D/q/DONE ] || fail "q train"
log "dev"
bash experiments/sst_hsmm_v1/p2q_orch.sh > $L/p2q_orch.log 2>&1 < /dev/null; [ -e $D/p2q/DONE ] || fail "p2q"
log "test"
bash experiments/sst_hsmm_v1/p3q_report.sh > $L/p3q_report.log 2>&1 < /dev/null; [ -e $D/p3q/DONE ] || fail "p3q"
log "chain done"
