#!/usr/bin/env bash
# prereg change 38 stage 1: real-time gate (bench_orch b38: K 6 / 24 tokens; if NO-GO once b38b: K 4 / 16) -> recording run on edev
# with the p3h8 policy + consistency (hdev11, h25, theta 0.75, cards 0 1) -> cons_precheck -> h25/precheck.json. 38-2 separately.
#   setsid nohup bash experiments/sst_hsmm_v1/h38_chain.sh > logs/sst_hsmm_v1/h38_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/h38_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] b38 (-> b38b) -> h25 -> cons_precheck"; exit 0; }
[ -e $L/h38_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until [ -e $D/b38/DONE ] || [ -e $D/b38/FAILED ]; do sleep 30; done
[ -e $D/b38/DONE ] || fail "b38"
go() { $PYQ -c "import json;print(int(json.load(open('$1'))['go']))"; }
log "chain $$: b38 $(tr -d '\n ' < $D/b38/decision.json | cut -c1-400)"
if [ "$(go $D/b38/decision.json)" = 1 ]; then KS=6; MT=24
else
  log "b38 NO-GO -> b38b (K 4 / 16 tokens)"
  OUTN=b38b K=4 MT=16 bash experiments/sst_hsmm_v1/bench_orch.sh > $L/b38b_orch.log 2>&1 < /dev/null
  [ -e $D/b38b/DONE ] || fail "b38b"
  log "b38b $(tr -d '\n ' < $D/b38b/decision.json | cut -c1-400)"
  [ "$(go $D/b38b/decision.json)" = 1 ] || { log "real-time gate failed twice: stop (prereg 38-0)"; log "chain done"; exit 0; }
  KS=4; MT=16
fi
log "recording run h25 (K $KS, $MT tokens)"
env DATA=$D/edev/data VIEWS=$D/edev/views VAR=n2 PRM=$D/h14/params.json OUTN=h25 THS="0.75" CARDS="0 1" KS=$KS MT=$MT \
  bash experiments/sst_hsmm_v1/hdev11_orch.sh > $L/h25_orch.log 2>&1 < /dev/null
[ -e $D/h25/devrun/DONE ] || fail "h25"
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.cons_precheck --root $D/h25 --out $D/h25/precheck.json > $L/h25_precheck.log 2>&1 || fail "precheck"
log "precheck: $(tail -n 1 $L/h25_precheck.log)"
log "chain done"
