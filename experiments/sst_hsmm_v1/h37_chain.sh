#!/usr/bin/env bash
# prereg change 37 stage 1: SAP-lag precheck (cards 0-3) || control dev p3h8 config theta 0.65 / 0.70 on edev (cards 4-7, h22a),
# then control dev theta 0.75 / 0.80 (cards 0-3, h22b). SAP2 (37-4) is launched separately after the decision.
#   setsid nohup bash experiments/sst_hsmm_v1/h37_chain.sh > logs/sst_hsmm_v1/h37_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/h37_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] sap_lag || h22a -> h22b"; exit 0; }
[ -e $L/h37_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
for d in en2zh zh2en; do [ -s $D/edev/views/$d/DONE.json ] || fail "edev views $d"; done
EV="DATA=$D/edev/data VIEWS=$D/edev/views VAR=n2 PRM=$D/h14/params.json"
log "chain $$: sap_lag (0-3) || control dev h22a 0.65 0.70 (4-7)"
CARDS="0 1 2 3" bash experiments/sst_hsmm_v1/sap_lag_orch.sh > $L/sap_lag_orch.log 2>&1 < /dev/null &
p1=$!
sleep 200
env $EV OUTN=h22a THS="0.65 0.70" CARDS="4 5 6 7" bash experiments/sst_hsmm_v1/hdev10_orch.sh > $L/h22a_orch.log 2>&1 < /dev/null
wait $p1
[ -e $D/sap_lag/DONE ] || fail "sap_lag"
log "sap_lag: $(cat $D/sap_lag/decision.json | tr -d '\n ')"
[ -e $D/h22a/devrun/DONE ] || fail "h22a"
log "control dev h22b 0.75 0.80"
env $EV OUTN=h22b THS="0.75 0.80" CARDS="0 1 2 3" bash experiments/sst_hsmm_v1/hdev10_orch.sh > $L/h22b_orch.log 2>&1 < /dev/null
[ -e $D/h22b/devrun/DONE ] || fail "h22b"
log "chain done"
