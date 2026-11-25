#!/usr/bin/env bash
# prereg change 22 (CR-LA) chain: t_orch -> dev p2r -> test p3r (combo vs N, report only).
#   setsid nohup bash experiments/sst_hsmm_v1/t_chain.sh > logs/sst_hsmm_v1/t_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/t_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] t_orch -> p2t -> p3t"; exit 0; }
[ -e $L/t_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until grep -q "chain done\|FAILED" $L/s_chain.log; do sleep 30; done      # cards: PC-MBR test first
log "chain $$: train"
bash experiments/sst_hsmm_v1/t_orch.sh > $L/t_orch.log 2>&1 < /dev/null; [ -e $D/t/DONE ] || fail "t train"
log "dev"
bash experiments/sst_hsmm_v1/p2t_orch.sh > $L/p2t_orch.log 2>&1 < /dev/null; [ -e $D/p2t/DONE ] || fail "p2t"
log "test"
bash experiments/sst_hsmm_v1/p3t_report.sh > $L/p3t_report.log 2>&1 < /dev/null; [ -e $D/p3t/DONE ] || fail "p3t"
log "chain done"
