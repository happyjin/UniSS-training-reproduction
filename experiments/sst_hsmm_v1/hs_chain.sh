#!/usr/bin/env bash
# prereg change 30 chain: dev soft (h3) -> [wait h5 fit] -> dev clock (h5) -> dev clockpause (h5) -> pick_s -> p3s1 (best)
# -> p3s2 (best soft family, only if different from best).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/hs_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] soft -> clock -> clockpause -> pick -> p3s1 -> p3s2"; exit 0; }
[ -e $L/hs_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
log "chain $$: dev soft"
VAR=soft PRM=$D/h3/params.json OUTN=h6soft bash experiments/sst_hsmm_v1/hdev_orch.sh > $L/h6soft_orch.log 2>&1 < /dev/null; [ -e $D/h6soft/devrun/DONE ] || fail "h6soft"
until [ -s $D/h5/params.json ]; do grep -q Traceback $L/h5/fit.log && fail "h5 fit"; sleep 30; done
log "dev clock"
VAR=clock PRM=$D/h5/params.json OUTN=h6clock bash experiments/sst_hsmm_v1/hdev_orch.sh > $L/h6clock_orch.log 2>&1 < /dev/null; [ -e $D/h6clock/devrun/DONE ] || fail "h6clock"
log "dev clockpause"
VAR=clockpause PRM=$D/h5/params.json OUTN=h6cp bash experiments/sst_hsmm_v1/hdev_orch.sh > $L/h6cp_orch.log 2>&1 < /dev/null; [ -e $D/h6cp/devrun/DONE ] || fail "h6cp"
PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.sst_hsmm_v1.pick_s > $L/pick_s.log 2>&1 || fail "pick"
log "test p3s1: $(python3 -c "import json;b=json.load(open('$D/s_choice.json'))['best'];print(b['variant'],b['theta'])")"
bash experiments/sst_hsmm_v1/p3s1_report.sh > $L/p3s1_report.log 2>&1 < /dev/null; [ -e $D/p3s1/DONE ] || fail "p3s1"
SAME=$(python3 -c "import json;c=json.load(open('$D/s_choice.json'));print(int((c['best']['variant'],c['best']['theta'])==(c['best_soft']['variant'],c['best_soft']['theta'])))")
if [ "$SAME" = 0 ]; then
  log "test p3s2: $(python3 -c "import json;b=json.load(open('$D/s_choice.json'))['best_soft'];print(b['variant'],b['theta'])")"
  bash experiments/sst_hsmm_v1/p3s2_report.sh > $L/p3s2_report.log 2>&1 < /dev/null; [ -e $D/p3s2/DONE ] || fail "p3s2"
fi
log "chain done"
