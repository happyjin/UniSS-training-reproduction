#!/usr/bin/env bash
# prereg change 29 chain: wait for the HSC test (p3z) and the HSC2 fit (h3/params.json) -> h4 (dev theta) -> p3z2 (test).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/h2_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] wait p3z + h3 -> h4 -> p3z2"; exit 0; }
[ -e $L/h2_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until { [ -e $D/p3z/DONE ] || [ -e $D/p3z/FAILED ] || [ -e $L/h_chain.FAILED ]; } && [ -s $D/h3/params.json ]; do sleep 60; done
log "chain $$: h4"
bash experiments/sst_hsmm_v1/h4_orch.sh > $L/h4_orch.log 2>&1 < /dev/null; [ -e $D/h4/devrun/DONE ] || fail "h4"
log "test theta $(python3 -c "import json;print(json.load(open('$D/h4/theta.json'))['theta'])")"
bash experiments/sst_hsmm_v1/p3z2_report.sh > $L/p3z2_report.log 2>&1 < /dev/null; [ -e $D/p3z2/DONE ] || fail "p3z2"
log "chain done"
