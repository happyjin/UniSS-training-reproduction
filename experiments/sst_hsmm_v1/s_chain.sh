#!/usr/bin/env bash
# prereg change 23 (PC-MBR) chain: s_orch (gen + select + dev/test text) -> test p3s (Talker/ASR/metrics vs N).
#   setsid nohup bash experiments/sst_hsmm_v1/s_chain.sh > logs/sst_hsmm_v1/s_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/s_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] s_orch -> p3s"; exit 0; }
[ -e $L/s_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
log "chain $$: gen"
bash experiments/sst_hsmm_v1/s_orch.sh > $L/s_orch.log 2>&1 < /dev/null; [ -e $D/s/DONE ] || fail "s"
log "test"
bash experiments/sst_hsmm_v1/p3s_report.sh > $L/p3s_report.log 2>&1 < /dev/null; [ -e $D/p3s/DONE ] || fail "p3s"
log "chain done"
