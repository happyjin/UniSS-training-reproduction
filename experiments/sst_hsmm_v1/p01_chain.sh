#!/usr/bin/env bash
# sst_hsmm_v1: rerun of the P0 analysis (1 torch thread per worker; the first run oversubscribed 96 x 192 threads) ->
# GO check (P0a GO and P0b sanity) -> P1a training + merge (p1_orch.sh). No waiting on the session.
#   setsid nohup bash experiments/sst_hsmm_v1/p01_chain.sh > logs/sst_hsmm_v1/p01_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=eval_outputs/sst_hsmm_v1/p0
LOGD=logs/sst_hsmm_v1/p0
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
[ -e $R/summary.json ] || {
  log "analyze start"
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. TOKENIZERS_PARALLELISM=false $PYQ -m experiments.sst_hsmm_v1.p0_analyze \
    > $LOGD/analyze_rerun.log 2>&1 || { log "FAILED analyze"; touch $R/FAILED; exit 1; }
}
touch $R/DONE
log "p0 done"
ok=$($PYQ -c "import json; s=json.load(open('$R/summary.json')); print(int(s['P0a']['GO'] and s['P0b']['sanity_pass'] and s['P0b']['mismatch']==0))")
[ "$ok" = 1 ] || { log "P0 NO-GO: stop before training"; exit 1; }
log "P0 GO -> P1"
bash experiments/sst_hsmm_v1/p1_orch.sh > logs/sst_hsmm_v1/p1_orch.log 2>&1 < /dev/null
log "p1_orch exit $?"
