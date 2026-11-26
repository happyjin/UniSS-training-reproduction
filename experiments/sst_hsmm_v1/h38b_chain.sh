#!/usr/bin/env bash
# prereg change 38 stage 2: waits for h38_chain; if h25/precheck.json go: consistency rollouts + hsmm_cons fit (h26_orch) -> dev of the
# consistency family on edev h28a (0.65 0.70, cards 0-3) || h28b (0.75 0.80, cards 4-7) -> pick_h10 over control (h22a/h22b) and
# consistency (h28a/h28b), ref h22b:0.75 -> p3h11 unless the reference wins.
#   setsid nohup bash experiments/sst_hsmm_v1/h38b_chain.sh > logs/sst_hsmm_v1/h38b_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/h38b_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] wait h38 -> precheck -> h26 -> h28a||h28b -> pick_h10 -> p3h11"; exit 0; }
[ -e $L/h38b_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until grep -q "chain done\|FAILED" $L/h38_chain.log; do sleep 60; done
grep -q "chain done" $L/h38_chain.log || fail "h38_chain"
[ -s $D/h25/precheck.json ] || { log "no consistency precheck (real-time gate failed?)"; log "chain done"; exit 0; }
GO=$($PYQ -c "import json;print(int(json.load(open('$D/h25/precheck.json'))['go']))")
[ "$GO" = 1 ] || { log "chain $$: consistency precheck NO-GO: report to the user"; log "chain done"; exit 0; }
log "chain $$: precheck GO; consistency rollouts + hsmm_cons fit (h26)"
bash experiments/sst_hsmm_v1/h26_orch.sh > $L/h26_orch.log 2>&1 < /dev/null
[ -e $D/h26/run/NOGO ] && { log "h26 precheck NO-GO: $(grep -h '^ref_masked_80\|^cons_80' $L/h26run/fit.log | cut -c1-120 | tr '\n' ' ')"; log "chain done"; exit 0; }
[ -e $D/h26/run/DONE ] || fail "h26"
log "dev consistency family h28a || h28b"
EV="DATA=$D/edev/data VIEWS=$D/edev/views VAR=n2 PRM=$D/h26/params.json KS=6 MT=24"
env $EV OUTN=h28a THS="0.65 0.70" CARDS="0 1 2 3" bash experiments/sst_hsmm_v1/hdev11_orch.sh > $L/h28a_orch.log 2>&1 < /dev/null &
pa=$!
sleep 30
env $EV OUTN=h28b THS="0.75 0.80" CARDS="4 5 6 7" bash experiments/sst_hsmm_v1/hdev11_orch.sh > $L/h28b_orch.log 2>&1 < /dev/null
wait $pa
for x in h28a h28b; do [ -e $D/$x/devrun/DONE ] || fail "$x"; done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h10 --fam "n2:h22a:0.65/0.70:h14:v,n2:h22b:0.75/0.80:h14:v,n2:h28a:0.65/0.70:h26:v,n2:h28b:0.75/0.80:h26:v" \
  --ref h22b:0.75 --out h28_choice.json > $L/pick_h28.log 2>&1 || fail "pick_h28"
B=$(python3 -c "import json;b=json.load(open('$D/h28_choice.json'))['best'];print(b['fam_dir'],b['theta'],b['params'],round(b['deficit'],2),round(b['mean_speech_delay'],3))")
log "best: $B"
case "$B" in "h22b 0.75 "*) log "reference (= p3h8) wins: no new test"; log "chain done"; exit 0;; esac
log "test p3h11"
bash experiments/sst_hsmm_v1/p3h11_report.sh > $L/p3h11_report.log 2>&1 < /dev/null; [ -e $D/p3h11/DONE ] || fail "p3h11"
log "chain done"
