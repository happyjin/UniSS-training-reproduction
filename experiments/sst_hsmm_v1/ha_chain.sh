#!/usr/bin/env bash
# prereg change 33-A chain: HSC6 (survival-anchored HSMM) CPU precheck + final fit (h14; NO-GO -> stop) -> wait for hb_chain
# (GPU) -> dev n1 / n2 (h15*) -> pick_h6 -> p3h6. (The prereg's h6_chain is split into hb_chain.sh (B) and this script (A).)
#   setsid nohup bash experiments/sst_hsmm_v1/ha_chain.sh > logs/sst_hsmm_v1/ha_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/ha_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] fit h14 (precheck) -> wait hb_chain -> dev n1 n2 -> pick_h6 -> p3h6"; exit 0; }
[ -e $L/ha_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
[ -e $D/h14 ] && { echo "exists: $D/h14" >&2; exit 1; }
mkdir -p $D/h14 $L/h14
log "chain $$: HSC6 precheck + fit (CPU)"
OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_anchor fit --roots "$D/h1/roll_*" \
  --pre-out $D/h14/precheck.json --out $D/h14/params.json > $L/h14/fit.log 2>&1
rc=$?
if [ $rc -eq 3 ]; then log "precheck NO-GO (see $D/h14/precheck.json); chain stops"; exit 0; fi
[ $rc -eq 0 ] && [ -s $D/h14/params.json ] || fail "h14 fit rc=$rc"
log "precheck GO; h14 fitted; waiting for hb_chain (GPU)"
until grep -q "chain done\|FAILED" $L/hb_chain.log; do sleep 60; done
for v in n1 n2; do
  log "dev $v (h14)"
  VAR=$v PRM=$D/h14/params.json OUTN=h15$v bash experiments/sst_hsmm_v1/hdev6_orch.sh > $L/h15${v}_orch.log 2>&1 < /dev/null
  [ -e $D/h15$v/devrun/DONE ] || fail "h15$v"
done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h6 --fam n1:h15n1,n2:h15n2 --params h14 --out h15_choice.json > $L/pick_h15.log 2>&1 || fail "pick_h15"
log "test p3h6: $(python3 -c "import json;b=json.load(open('$D/h15_choice.json'))['best'];print(b['variant'],b['theta'])")"
bash experiments/sst_hsmm_v1/p3h6_report.sh > $L/p3h6_report.log 2>&1 < /dev/null; [ -e $D/p3h6/DONE ] || fail "p3h6"
log "chain done"
