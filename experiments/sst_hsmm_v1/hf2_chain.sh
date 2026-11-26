#!/usr/bin/env bash
# prereg change 36 chain, resumed after the hsmm_onpol commit-count fix: the h20 rollouts (h20_orch) are complete; refit (CPU) -> precheck
# -> dev n2 theta 0.65/0.70/0.75/0.80 on h20 (h21n2) -> pick_h8 -> p3h9. Copy of hf_chain.sh from the fit step.
#   setsid nohup bash experiments/sst_hsmm_v1/hf2_chain.sh > logs/sst_hsmm_v1/hf2_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
H=$D/h20
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/hf2_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] h20 refit -> dev h21n2 -> pick_h8 -> p3h9"; exit 0; }
[ -e $L/hf2_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $H/roll_${d}_$k/blocks_${d}_w1.jsonl ] || fail "roll $d $k"; done; done
log "chain $$: refit h20 (CPU)"
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_onpol fit --roots "$D/h1/roll_*,$H/roll_*" --onpol-root "$H/roll_*" \
  --pre-out $H/precheck.json --out $H/params.json > $L/h20run/fit2.log 2>&1
rc=$?
[ $rc -eq 3 ] && { log "precheck NO-GO: $(grep -h '^h14\|^onpol' $L/h20run/fit2.log | cut -c1-200 | tr '\n' ' ')"; log "chain done"; exit 0; }
[ $rc -eq 0 ] || fail "fit"
log "precheck GO; dev n2 (h20)"
VAR=n2 PRM=$H/params.json OUTN=h21n2 THS="0.65 0.70 0.75 0.80" bash experiments/sst_hsmm_v1/hdev8_orch.sh > $L/h21n2_orch.log 2>&1 < /dev/null
[ -e $D/h21n2/devrun/DONE ] || fail "h21n2"
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h8 --fam n2:h21n2:0.65/0.70/0.75/0.80 --params h20 --out h21_choice.json > $L/pick_h21.log 2>&1 || fail "pick_h21"
log "test p3h9: $(python3 -c "import json;b=json.load(open('$D/h21_choice.json'))['best'];print(b['fam_dir'],b['variant'],b['theta'],b['deficit'])")"
bash experiments/sst_hsmm_v1/p3h9_report.sh > $L/p3h9_report.log 2>&1 < /dev/null; [ -e $D/p3h9/DONE ] || fail "p3h9"
log "chain done"
