#!/usr/bin/env bash
# prereg change 37 stage 2: waits for h37_chain; if sap_lag/decision.json go_retrain: SAP2 (sap2_orch) -> HSMM refit (h23_orch)
# -> dev SAP2 family on edev h24a (0.65 0.70, cards 0-3) || h24b (0.75 0.80, cards 4-7) -> pick_h10 over control (h22a/h22b, h14, v)
# and SAP2 (h24a/h24b, h23, v2), ref h22b:0.75 -> p3h10 unless the reference itself wins. NO-GO: control-only table, no test.
#   setsid nohup bash experiments/sst_hsmm_v1/h37b_chain.sh > logs/sst_hsmm_v1/h37b_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=eval_outputs/sst_hsmm_v1
L=logs/sst_hsmm_v1
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $L/h37b_chain.FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] && { echo "[dry] wait h37 -> decision -> sap2 -> h23 -> h24a||h24b -> pick_h10 -> p3h10"; exit 0; }
[ -e $L/h37b_chain.FAILED ] && { echo "old FAILED flag exists" >&2; exit 1; }
until grep -q "chain done\|FAILED" $L/h37_chain.log; do sleep 60; done
grep -q "chain done" $L/h37_chain.log || fail "h37_chain"
CTRL="n2:h22a:0.65/0.70:h14:v,n2:h22b:0.75/0.80:h14:v"
GO=$($PYQ -c "import json;print(int(json.load(open('$D/sap_lag/decision.json'))['go_retrain']))")
if [ "$GO" != 1 ]; then
  log "chain $$: sap_lag NO-GO ($(tr -d '\n ' < $D/sap_lag/decision.json)); control-only table"
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h10 --fam $CTRL --ref h22b:0.75 --out h22_control.json > $L/pick_h22.log 2>&1 || fail "pick_h22"
  log "control best: $(python3 -c "import json;b=json.load(open('$D/h22_control.json'))['best'];print(b['fam_dir'],b['theta'],round(b['deficit'],2),round(b['mean_speech_delay'],3))")"
  log "chain done (NO-GO: report to the user)"; exit 0
fi
log "chain $$: sap_lag GO; SAP2 data + training"
bash experiments/sst_hsmm_v1/sap2_orch.sh > $L/sap2_orch.log 2>&1 < /dev/null; [ -e $D/t2/DONE ] || fail "sap2"
log "HSMM refit h23"
bash experiments/sst_hsmm_v1/h23_orch.sh > $L/h23_orch.log 2>&1 < /dev/null
[ -e $D/h23/run/NOGO ] && { log "h23 precheck NO-GO: $(grep -h '^h9\|^hsc6' $L/h23run/fit.log | cut -c1-160 | tr '\n' ' ')"; log "chain done"; exit 0; }
[ -e $D/h23/run/DONE ] || fail "h23"
log "dev SAP2 family h24a || h24b"
EV="DATA=$D/edev/data VIEWS=$D/edev/views VAR=n2 PRM=$D/h23/params.json VER=$D/v2/lora/a1.0000"
env $EV OUTN=h24a THS="0.65 0.70" CARDS="0 1 2 3" bash experiments/sst_hsmm_v1/hdev10_orch.sh > $L/h24a_orch.log 2>&1 < /dev/null &
pa=$!
sleep 30
env $EV OUTN=h24b THS="0.75 0.80" CARDS="4 5 6 7" bash experiments/sst_hsmm_v1/hdev10_orch.sh > $L/h24b_orch.log 2>&1 < /dev/null
wait $pa
for x in h24a h24b; do [ -e $D/$x/devrun/DONE ] || fail "$x"; done
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pick_h10 --fam "$CTRL,n2:h24a:0.65/0.70:h23:v2,n2:h24b:0.75/0.80:h23:v2" --ref h22b:0.75 \
  --out h24_choice.json > $L/pick_h24.log 2>&1 || fail "pick_h24"
B=$(python3 -c "import json;b=json.load(open('$D/h24_choice.json'))['best'];print(b['fam_dir'],b['theta'],b['params'],round(b['deficit'],2))")
log "best: $B"
case "$B" in "h22b 0.75 "*) log "reference (= p3h8) wins: no new test"; log "chain done"; exit 0;; esac
log "test p3h10"
bash experiments/sst_hsmm_v1/p3h10_report.sh > $L/p3h10_report.log 2>&1 < /dev/null; [ -e $D/p3h10/DONE ] || fail "p3h10"
log "chain done"
