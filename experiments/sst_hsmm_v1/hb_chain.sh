#!/usr/bin/env bash
# prereg change 33-B chain: playback-slack commit on the HSC4 model (h9): dev s1 / s2 (h12*) -> pick_h6 -> p3sl.
#   setsid nohup bash experiments/sst_hsmm_v1/hb_chain.sh > logs/sst_hsmm_v1/hb_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/hb_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] dev s1 s2 (h9) -> pick_h6 -> p3sl"; exit 0; }
[ -e $L/hb_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
[ -s $D/h9/params.json ] || fail "h9 missing"
for v in s1 s2; do
  log "chain $$: dev $v (h9)"
  VAR=$v PRM=$D/h9/params.json OUTN=h12$v bash experiments/sst_hsmm_v1/hdev6_orch.sh > $L/h12${v}_orch.log 2>&1 < /dev/null
  [ -e $D/h12$v/devrun/DONE ] || fail "h12$v"
done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h6 --fam s1:h12s1,s2:h12s2 --params h9 --out h12_choice.json > $L/pick_h12.log 2>&1 || fail "pick_h12"
log "test p3sl: $(python3 -c "import json;b=json.load(open('$D/h12_choice.json'))['best'];print(b['variant'],b['theta'])")"
bash experiments/sst_hsmm_v1/p3sl_report.sh > $L/p3sl_report.log 2>&1 < /dev/null; [ -e $D/p3sl/DONE ] || fail "p3sl"
log "chain done"
