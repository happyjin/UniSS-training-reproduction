#!/usr/bin/env bash
# prereg change 36 chain: on-policy rollouts + refit (h20_orch) -> dev n2 theta 0.65/0.70/0.75/0.80 on h20 (h21n2) -> pick_h8 -> p3h9.
#   setsid nohup bash experiments/sst_hsmm_v1/hf_chain.sh > logs/sst_hsmm_v1/hf_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/hf_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] h20 rollouts+refit -> dev h21n2 -> pick_h8 -> p3h9"; exit 0; }
[ -e $L/hf_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
log "chain $$: on-policy rollouts + refit (h20)"
bash experiments/sst_hsmm_v1/h20_orch.sh > $L/h20_orch.log 2>&1 < /dev/null
[ -e $D/h20/run/NOGO ] && { log "precheck NO-GO: $(tail -3 $L/h20run/fit.log | tr '\n' ' ')"; log "chain done"; exit 0; }
[ -e $D/h20/run/DONE ] || fail "h20"
log "precheck GO; dev n2 (h20)"
VAR=n2 PRM=$D/h20/params.json OUTN=h21n2 THS="0.65 0.70 0.75 0.80" bash experiments/sst_hsmm_v1/hdev8_orch.sh > $L/h21n2_orch.log 2>&1 < /dev/null
[ -e $D/h21n2/devrun/DONE ] || fail "h21n2"
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h8 --fam n2:h21n2:0.65/0.70/0.75/0.80 --params h20 --out h21_choice.json > $L/pick_h21.log 2>&1 || fail "pick_h21"
log "test p3h9: $(python3 -c "import json;b=json.load(open('$D/h21_choice.json'))['best'];print(b['fam_dir'],b['variant'],b['theta'],b['deficit'])")"
bash experiments/sst_hsmm_v1/p3h9_report.sh > $L/p3h9_report.log 2>&1 < /dev/null; [ -e $D/p3h9/DONE ] || fail "p3h9"
log "chain done"
