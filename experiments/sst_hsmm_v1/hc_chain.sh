#!/usr/bin/env bash
# prereg change 34-A chain (copy of hc_chain.sh): slack commit on HSC6 (h14): dev s1 / s2 (h16*) -> pick_h6 -> p3h6s.
#   setsid nohup bash experiments/sst_hsmm_v1/hc_chain.sh > logs/sst_hsmm_v1/hc_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/hc_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] dev s1 s2 (h14) -> pick_h6 -> p3sl"; exit 0; }
[ -e $L/hc_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
[ -s $D/h14/params.json ] || fail "h14 missing"
for v in s1 s2; do
  log "chain $$: dev $v (h14)"
  VAR=$v PRM=$D/h14/params.json OUTN=h16$v bash experiments/sst_hsmm_v1/hdev6_orch.sh > $L/h16${v}_orch.log 2>&1 < /dev/null
  [ -e $D/h16$v/devrun/DONE ] || fail "h16$v"
done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h6 --fam s1:h16s1,s2:h16s2 --params h14 --out h16_choice.json > $L/pick_h16.log 2>&1 || fail "pick_h16"
log "test p3h6s: $(python3 -c "import json;b=json.load(open('$D/h16_choice.json'))['best'];print(b['variant'],b['theta'])")"
bash experiments/sst_hsmm_v1/p3h6s_report.sh > $L/p3h6s_report.log 2>&1 < /dev/null; [ -e $D/p3h6s/DONE ] || fail "p3h6s"
log "chain done"
