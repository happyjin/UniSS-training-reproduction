#!/usr/bin/env bash
# prereg change 28 chain: h1 (HSC rollout + unsupervised EM) -> h2 (dev theta, iso-latency with SVC-JM) -> p3z (test).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/h_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] h1 -> h2 -> p3z"; exit 0; }
[ -e $L/h_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
log "chain $$: h1"
bash experiments/sst_hsmm_v1/h1_orch.sh > $L/h1_orch.log 2>&1 < /dev/null; [ -e $D/h1/run/DONE ] || fail "h1"
log "h2"
bash experiments/sst_hsmm_v1/h2_orch.sh > $L/h2_orch.log 2>&1 < /dev/null; [ -e $D/h2/devrun/DONE ] || fail "h2"
log "test theta $(python3 -c "import json;print(json.load(open('$D/h2/theta.json'))['theta'])")"
bash experiments/sst_hsmm_v1/p3z_report.sh > $L/p3z_report.log 2>&1 < /dev/null; [ -e $D/p3z/DONE ] || fail "p3z"
log "chain done"
