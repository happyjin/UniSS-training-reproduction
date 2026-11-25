#!/usr/bin/env bash
# prereg change 31 chain: wait for hs_chain (change 30 tests) and the h7 fit -> dev n1 / n2 (h7) -> pick_n -> p3n1.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/hn_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] wait -> n1 -> n2 -> pick -> p3n1"; exit 0; }
[ -e $L/hn_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until grep -q "chain done\|FAILED" $L/hs_chain.log && [ -s $D/h7/params.json ]; do grep -q Traceback $L/h7/fit.log && fail "h7 fit"; sleep 60; done
log "chain $$: dev n1"
VAR=n1 PRM=$D/h7/params.json OUTN=h8n1 bash experiments/sst_hsmm_v1/hdev4_orch.sh > $L/h8n1_orch.log 2>&1 < /dev/null; [ -e $D/h8n1/devrun/DONE ] || fail "h8n1"
log "dev n2"
VAR=n2 PRM=$D/h7/params.json OUTN=h8n2 bash experiments/sst_hsmm_v1/hdev4_orch.sh > $L/h8n2_orch.log 2>&1 < /dev/null; [ -e $D/h8n2/devrun/DONE ] || fail "h8n2"
PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.sst_hsmm_v1.pick_n > $L/pick_n.log 2>&1 || fail "pick"
log "test p3n1: $(python3 -c "import json;b=json.load(open('$D/n_choice.json'))['best'];print(b['variant'],b['theta'])")"
bash experiments/sst_hsmm_v1/p3n1_report.sh > $L/p3n1_report.log 2>&1 < /dev/null; [ -e $D/p3n1/DONE ] || fail "p3n1"
log "chain done"
