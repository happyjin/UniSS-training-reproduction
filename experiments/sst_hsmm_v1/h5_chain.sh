#!/usr/bin/env bash
# prereg change 32 chain: HSC5 CPU precheck + final fit (h10; NO-GO -> stop) -> dev n1 / n2 / s1 / s2 (h11*) -> pick_h5 -> p3h5.
#   setsid nohup bash experiments/sst_hsmm_v1/h5_chain.sh > logs/sst_hsmm_v1/h5_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/h5_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] fit h10 (precheck) -> dev n1 n2 s1 s2 -> pick_h5 -> p3h5"; exit 0; }
[ -e $L/h5_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
[ -e $D/h10 ] && { echo "exists: $D/h10" >&2; exit 1; }
mkdir -p $D/h10 $L/h10
log "chain $$: HSC5 precheck + fit (CPU)"
OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.hsmm_dep fit --roots "$D/h1/roll_*" \
  --pre-out $D/h10/precheck.json --out $D/h10/params.json > $L/h10/fit.log 2>&1
rc=$?
if [ $rc -eq 3 ]; then log "precheck NO-GO (see $D/h10/precheck.json); chain stops"; exit 0; fi
[ $rc -eq 0 ] && [ -s $D/h10/params.json ] || fail "h10 fit rc=$rc"
log "precheck GO; h10 fitted"
for v in n1 n2 s1 s2; do
  log "dev $v"
  VAR=$v PRM=$D/h10/params.json OUTN=h11$v bash experiments/sst_hsmm_v1/hdev5_orch.sh > $L/h11${v}_orch.log 2>&1 < /dev/null
  [ -e $D/h11$v/devrun/DONE ] || fail "h11$v"
done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h5 > $L/pick_h5.log 2>&1 || fail "pick_h5"
log "test p3h5: $(python3 -c "import json;b=json.load(open('$D/h5_choice.json'))['best'];print(b['variant'],b['theta'])")"
bash experiments/sst_hsmm_v1/p3h5_report.sh > $L/p3h5_report.log 2>&1 < /dev/null; [ -e $D/p3h5/DONE ] || fail "p3h5"
log "chain done"
