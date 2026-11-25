#!/usr/bin/env bash
# prereg change 27 chain: wait for E0 (p3x) -> E1 rollout + head (e1_orch) -> E2 dev theta (e2_orch) -> E3 test (p3y).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/e_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] p3x wait -> e1 -> e2 -> p3y"; exit 0; }
[ -e $L/e_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until [ -e $D/p3x/DONE ] || [ -e $D/p3x/FAILED ]; do sleep 60; done
log "chain $$: E0 finished ($(ls $D/p3x | grep -c DONE) DONE); e1"
bash experiments/sst_hsmm_v1/e1_orch.sh > $L/e1_orch.log 2>&1 < /dev/null; [ -e $D/e1/run/DONE ] || fail "e1"
log "e2"
bash experiments/sst_hsmm_v1/e2_orch.sh > $L/e2_orch.log 2>&1 < /dev/null; [ -e $D/e2/devrun/DONE ] || fail "e2"
log "e3 theta $(python3 -c "import json;print(json.load(open('$D/e2/theta.json'))['theta'])")"
bash experiments/sst_hsmm_v1/p3y_report.sh > $L/p3y_report.log 2>&1 < /dev/null; [ -e $D/p3y/DONE ] || fail "p3y"
log "chain done"
