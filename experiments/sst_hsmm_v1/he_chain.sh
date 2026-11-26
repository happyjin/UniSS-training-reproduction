#!/usr/bin/env bash
# prereg change 35 chain (copy of hc_chain.sh): risk-budget r1 + low-theta n2 on HSC6 (h14): dev h19r1 / h19n2 -> pick_h8 (with h15n1 / h15n2)
# -> p3h8 (skipped when an already tested h15 config wins).
#   setsid nohup bash experiments/sst_hsmm_v1/he_chain.sh > logs/sst_hsmm_v1/he_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/he_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] dev r1 n2lo (h14) -> pick_h8 -> p3h8"; exit 0; }
[ -e $L/he_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
[ -s $D/h14/params.json ] || fail "h14 missing"
for spec in "r1:0.30 0.60 1.00 1.50" "n2:0.50 0.60 0.70 0.75"; do
  v=${spec%%:*}; ths=${spec#*:}
  log "chain $$: dev $v ($ths) h14"
  VAR=$v PRM=$D/h14/params.json OUTN=h19$v THS="$ths" bash experiments/sst_hsmm_v1/hdev8_orch.sh > $L/h19${v}_orch.log 2>&1 < /dev/null
  [ -e $D/h19$v/devrun/DONE ] || fail "h19$v"
done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h8 --fam r1:h19r1:0.30/0.60/1.00/1.50,n2:h19n2:0.50/0.60/0.70/0.75,n1:h15n1,n2:h15n2 \
  --params h14 --out h19_choice.json > $L/pick_h19.log 2>&1 || fail "pick_h19"
B=$(python3 -c "import json;b=json.load(open('$D/h19_choice.json'))['best'];print(b['fam_dir'],b['variant'],b['theta'])")
case "$B" in h15*) log "best = $B (already tested as p3h6): no new test"; log "chain done"; exit 0;; esac
log "test p3h8: $B"
bash experiments/sst_hsmm_v1/p3h8_report.sh > $L/p3h8_report.log 2>&1 < /dev/null; [ -e $D/p3h8/DONE ] || fail "p3h8"
log "chain done"
