#!/usr/bin/env bash
# P11 (prereg change 19) chain: p11_orch (train) -> dev p2o (lam0 default, control_a) -> p11_pick second -> [dev p2o2 (control_b)]
# -> p11_pick choose -> p11/control_chosen.json -> test p3o (report only). Serial (one vLLM Thinker per card at a time).
#   setsid nohup bash experiments/sst_hsmm_v1/p11_chain.sh > logs/sst_hsmm_v1/p11_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
P1=$D/p11
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/p11_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] p11_orch -> p2o -> pick -> [p2o2] -> choose -> p3o"; exit 0; }
[ -e $L/p11_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
log "chain $$: train"
bash experiments/sst_hsmm_v1/p11_orch.sh > $L/p11_orch.log 2>&1 < /dev/null; [ -e $P1/DONE ] || fail "p11 train"
log "dev a"
P2NAME=p2o P2CTRL=$P1/control_a.json bash experiments/sst_hsmm_v1/p2o_orch.sh > $L/p2o_orch.log 2>&1 < /dev/null
[ -e $D/p2o/DONE ] || fail "p2o"
out=$(PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p11_pick second --runs $D/p2o/summary.json:$P1/control_a.json --out $P1/control_b.json) || fail "pick second"
log "pick second: $out"
RUNS="$D/p2o/summary.json:$P1/control_a.json"
if [ "${out%% *}" = run ]; then
  P2NAME=p2o2 P2CTRL=$P1/control_b.json bash experiments/sst_hsmm_v1/p2o_orch.sh > $L/p2o2_orch.log 2>&1 < /dev/null
  [ -e $D/p2o2/DONE ] || fail "p2o2"
  RUNS="$RUNS $D/p2o2/summary.json:$P1/control_b.json"
fi
out=$(PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p11_pick choose --runs $RUNS --out $P1/control_chosen.json) || fail "pick choose"
log "choose: $out"
bash experiments/sst_hsmm_v1/p3o_report.sh > $L/p3o_report.log 2>&1 < /dev/null; [ -e $D/p3o/DONE ] || fail "p3o"
log "chain done"
