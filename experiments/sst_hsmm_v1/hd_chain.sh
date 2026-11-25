#!/usr/bin/env bash
# prereg change 34-B chain: HSC7 (tail-distance input) CPU precheck + final fit (h17; NO-GO -> stop) -> wait for hc_chain (GPU)
# -> dev n1 / n2 (h18*) -> pick_h6 -> p3h7.
#   setsid nohup bash experiments/sst_hsmm_v1/hd_chain.sh > logs/sst_hsmm_v1/hd_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/hd_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] fit h17 (precheck) -> wait hc_chain -> dev n1 n2 -> pick_h6 -> p3h7"; exit 0; }
[ -e $L/hd_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
[ -e $D/h17 ] && { echo "exists: $D/h17" >&2; exit 1; }
mkdir -p $D/h17 $L/h17
log "chain $$: HSC7 precheck + fit (CPU)"
OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_tail fit --roots "$D/h1/roll_*" \
  --pre-out $D/h17/precheck.json --out $D/h17/params.json > $L/h17/fit.log 2>&1
rc=$?
if [ $rc -eq 3 ]; then log "precheck NO-GO (see $D/h17/precheck.json); chain stops"; exit 0; fi
[ $rc -eq 0 ] && [ -s $D/h17/params.json ] || fail "h17 fit rc=$rc"
log "precheck GO; h17 fitted; waiting for hc_chain (GPU)"
until grep -q "chain done\|FAILED" $L/hc_chain.log; do sleep 60; done
for v in n1 n2; do
  log "dev $v (h17)"
  VAR=$v PRM=$D/h17/params.json OUTN=h18$v bash experiments/sst_hsmm_v1/hdev7_orch.sh > $L/h18${v}_orch.log 2>&1 < /dev/null
  [ -e $D/h18$v/devrun/DONE ] || fail "h18$v"
done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h6 --fam n1:h18n1,n2:h18n2 --params h17 --out h18_choice.json > $L/pick_h18.log 2>&1 || fail "pick_h18"
log "test p3h7: $(python3 -c "import json;b=json.load(open('$D/h18_choice.json'))['best'];print(b['variant'],b['theta'])")"
bash experiments/sst_hsmm_v1/p3h7_report.sh > $L/p3h7_report.log 2>&1 < /dev/null; [ -e $D/p3h7/DONE ] || fail "p3h7"
log "chain done"
